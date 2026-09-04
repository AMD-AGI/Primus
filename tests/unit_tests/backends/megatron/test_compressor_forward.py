###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Dedicated Compressor.forward behavioral tests (PRPUNDIT-25).

``test_v4_keep_in_fp32.py`` only checks ape dtype / finiteness, and
``test_compressor_pool.py`` compares the fused Triton kernel against eager
softmax-pool on tensors that are *already* windowed. Neither independently
checks that ``_reshape_into_windows`` + ``_overlap_transform`` stitch the
correct source positions. This file is the missing oracle: CSA overlap
(ratio=4) and HCA non-overlap (ratio=128) against a hand-derived windowing
reference that does not call those helpers, plus fused vs unfused
projections and the sequence-length guard.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from primus.backends.megatron.core.transformer.compressor import Compressor


@pytest.fixture(autouse=True)
def _cpu_eager_pool(monkeypatch):
    monkeypatch.setenv("PRIMUS_COMPRESS_POOL_TRITON", "0")
    monkeypatch.setenv("PRIMUS_RMSNORM_TRITON", "0")


def _project(comp: Compressor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """KV / score projections via the module weights, not the window helpers."""
    if comp._fuse_proj:
        projected = F.linear(hidden, comp.wkv_gate.weight, None)
        return projected.split(comp._proj_out, dim=-1)
    return comp.wkv(hidden), comp.wgate(hidden)


def _manual_windows(
    kv_proj: torch.Tensor,
    score_proj: torch.Tensor,
    *,
    ratio: int,
    overlap: bool,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent overlap / non-overlap window construction.

    Overlap window ``i`` is ``[half_a[i], half_b[i-1]]`` along the window
    axis, with window 0's previous half filled by zeros (causal padding).
    Non-overlap window ``i`` is the contiguous slice ``[i*ratio, (i+1)*ratio)``.
    """
    batch, seq, _ = kv_proj.shape
    n_windows = seq // ratio
    win_len = 2 * ratio if overlap else ratio
    kv_win = kv_proj.new_empty(batch, n_windows, win_len, head_dim)
    score_win = score_proj.new_empty(batch, n_windows, win_len, head_dim)

    for window in range(n_windows):
        cur = slice(window * ratio, (window + 1) * ratio)
        if overlap:
            kv_win[:, window, :ratio] = kv_proj[:, cur, :head_dim]
            score_win[:, window, :ratio] = score_proj[:, cur, :head_dim]
            if window == 0:
                kv_win[:, window, ratio:] = 0
                score_win[:, window, ratio:] = 0
            else:
                prev = slice((window - 1) * ratio, window * ratio)
                kv_win[:, window, ratio:] = kv_proj[:, prev, head_dim : 2 * head_dim]
                score_win[:, window, ratio:] = score_proj[:, prev, head_dim : 2 * head_dim]
        else:
            kv_win[:, window] = kv_proj[:, cur]
            score_win[:, window] = score_proj[:, cur]
    return kv_win, score_win


def _manual_rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Eager LocalRMSNorm reference (fp32 stats, mid-cast, then weight)."""
    hidden_fp32 = hidden.float()
    rstd = torch.rsqrt(hidden_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (hidden_fp32 * rstd).to(hidden.dtype) * weight


def _manual_pool(comp: Compressor, hidden: torch.Tensor) -> torch.Tensor:
    kv_proj, score_proj = _project(comp, hidden)
    kv_win, score_win = _manual_windows(
        kv_proj,
        score_proj,
        ratio=comp.ratio,
        overlap=comp.overlap,
        head_dim=comp.head_dim,
    )
    weights = torch.softmax((score_win + comp.ape).float(), dim=2).to(kv_win.dtype)
    pooled = (kv_win * weights).sum(dim=2)
    return _manual_rms_norm(pooled, comp.kv_norm.weight, comp.kv_norm.eps)


@pytest.mark.parametrize(
    "ratio,overlap",
    [
        (4, True),  # CSA
        (128, False),  # HCA
    ],
)
def test_forward_matches_manual_overlap_and_nonoverlap_oracle(ratio: int, overlap: bool):
    torch.manual_seed(0)
    hidden_size, head_dim, n_windows, batch = 32, 16, 3, 2
    comp = Compressor(hidden_size=hidden_size, head_dim=head_dim, ratio=ratio, overlap=overlap)
    hidden = torch.randn(batch, ratio * n_windows, hidden_size, dtype=torch.float32)

    pooled = comp(hidden)
    expected = _manual_pool(comp, hidden)

    assert pooled.shape == (batch, n_windows, head_dim)
    torch.testing.assert_close(pooled, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "ratio,overlap",
    [
        (4, True),
        (128, False),
    ],
)
def test_manual_windows_match_documented_half_stitch(ratio: int, overlap: bool):
    torch.manual_seed(1)
    hidden_size, head_dim, n_windows = 32, 16, 3
    comp = Compressor(hidden_size=hidden_size, head_dim=head_dim, ratio=ratio, overlap=overlap)
    hidden = torch.arange(2 * ratio * n_windows * hidden_size, dtype=torch.float32).reshape(
        2, ratio * n_windows, hidden_size
    )
    kv_proj, score_proj = _project(comp, hidden)
    kv_win, _ = _manual_windows(
        kv_proj,
        score_proj,
        ratio=ratio,
        overlap=overlap,
        head_dim=head_dim,
    )

    if overlap:
        # Window i first half == current half_a; second half == previous half_b
        # (zeros for window 0).
        torch.testing.assert_close(kv_win[:, 0, ratio:], torch.zeros_like(kv_win[:, 0, ratio:]))
        torch.testing.assert_close(kv_win[:, 0, :ratio], kv_proj[:, :ratio, :head_dim])
        torch.testing.assert_close(kv_win[:, 1, ratio:], kv_proj[:, :ratio, head_dim : 2 * head_dim])
        torch.testing.assert_close(kv_win[:, 1, :ratio], kv_proj[:, ratio : 2 * ratio, :head_dim])
    else:
        torch.testing.assert_close(kv_win[:, 0], kv_proj[:, :ratio])
        torch.testing.assert_close(kv_win[:, 1], kv_proj[:, ratio : 2 * ratio])


def test_overlap_window_zero_ignores_later_tokens():
    """Window 0 has no predecessor; corrupting window 1 must not move pooled[:, 0]."""
    torch.manual_seed(0)
    ratio = 4
    comp = Compressor(hidden_size=32, head_dim=16, ratio=ratio, overlap=True)
    hidden = torch.randn(1, ratio * 3, 32, dtype=torch.float32)
    baseline = comp(hidden)

    perturbed = hidden.clone()
    perturbed[:, ratio : 2 * ratio] += 10.0
    pooled = comp(perturbed)

    torch.testing.assert_close(pooled[:, 0], baseline[:, 0])
    assert not torch.allclose(pooled[:, 1], baseline[:, 1])


@pytest.mark.parametrize(
    "ratio,overlap",
    [
        (4, True),
        (128, False),
    ],
)
def test_fused_and_unfused_projections_agree(ratio: int, overlap: bool, monkeypatch):
    """PRIMUS_COMPRESS_FUSE_PROJ only changes how the KV/gate GEMM is launched."""
    torch.manual_seed(0)
    monkeypatch.setenv("PRIMUS_COMPRESS_FUSE_PROJ", "0")
    unfused = Compressor(hidden_size=32, head_dim=16, ratio=ratio, overlap=overlap)
    monkeypatch.setenv("PRIMUS_COMPRESS_FUSE_PROJ", "1")
    fused = Compressor(hidden_size=32, head_dim=16, ratio=ratio, overlap=overlap)
    with torch.no_grad():
        fused.wkv_gate.weight.copy_(torch.cat([unfused.wkv.weight, unfused.wgate.weight], dim=0))
        fused.ape.copy_(unfused.ape)
        fused.kv_norm.weight.copy_(unfused.kv_norm.weight)

    hidden = torch.randn(2, ratio * 2, 32, dtype=torch.float32)
    torch.testing.assert_close(fused(hidden), unfused(hidden), rtol=1e-5, atol=1e-5)


def test_forward_rejects_sequence_not_divisible_by_ratio():
    comp = Compressor(hidden_size=32, head_dim=16, ratio=4, overlap=True)
    hidden = torch.randn(1, 10, 32, dtype=torch.float32)
    with pytest.raises(AssertionError, match="not divisible by ratio"):
        comp(hidden)
