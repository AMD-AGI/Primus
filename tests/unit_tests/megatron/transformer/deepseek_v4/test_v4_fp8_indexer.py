###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the DeepSeek-V4 FP8 (E4M3) Indexer QK path.

CPU-friendly. Validates that:

* :func:`fake_quantize_fp8_e4m3` is a bounded-error, dtype-preserving,
  zero-safe fake-quantization.
* The FP8 Indexer selects almost the same top-k compressed positions as the
  BF16 reference (high top-k overlap) — the QK precision drop must not change
  which positions the CSA selector picks for the vast majority of queries.
* Output shapes / sentinel (-1) semantics are unchanged by the FP8 path.
"""

from __future__ import annotations

import copy

import pytest
import torch

from primus.backends.megatron.core.transformer.indexer import (
    Indexer,
    fake_quantize_fp8_e4m3,
)

_HAS_FP8 = hasattr(torch, "float8_e4m3fn")


def _fp8_cast_works() -> bool:
    if not _HAS_FP8:
        return False
    try:
        torch.zeros(4).to(torch.float8_e4m3fn).to(torch.float32)
        return True
    except (RuntimeError, TypeError):
        return False


pytestmark = pytest.mark.skipif(
    not _fp8_cast_works(), reason="torch.float8_e4m3fn cast unsupported on this build/device"
)


# ---------------------------------------------------------------------------
# fake_quantize_fp8_e4m3
# ---------------------------------------------------------------------------


def test_fake_quant_preserves_dtype_and_shape():
    x = torch.randn(3, 5, 7, dtype=torch.float32)
    xq = fake_quantize_fp8_e4m3(x)
    assert xq.dtype == x.dtype
    assert xq.shape == x.shape


def test_fake_quant_zero_input_is_safe():
    x = torch.zeros(4, 4)
    xq = fake_quantize_fp8_e4m3(x)
    assert torch.equal(xq, x)


def test_fake_quant_bounded_relative_error():
    # E4M3 has 3 mantissa bits -> ~2^-3 = 12.5% worst-case step; with dynamic
    # per-tensor scaling the typical relative error on sizable values is small.
    torch.manual_seed(0)
    x = torch.randn(2048, dtype=torch.float32) * 2.0
    xq = fake_quantize_fp8_e4m3(x)
    # Compare only on non-tiny values (tiny values have large relative error
    # but negligible absolute impact on the QK dot product).
    big = x.abs() > 0.1 * x.abs().max()
    rel = ((xq[big] - x[big]).abs() / x[big].abs()).max().item()
    assert rel < 0.2, f"FP8 fake-quant relative error too large: {rel}"


# ---------------------------------------------------------------------------
# Indexer FP8 QK path vs BF16 reference
# ---------------------------------------------------------------------------


def _make_indexer(use_fp8_qk: bool) -> Indexer:
    return Indexer(
        hidden_size=64,
        index_head_dim=16,
        index_n_heads=4,
        index_topk=4,
        compress_ratio=4,
        use_fp8_qk=use_fp8_qk,
    )


def _topk_overlap(idx_a: torch.Tensor, idx_b: torch.Tensor) -> float:
    """Mean per-query Jaccard overlap of selected (valid) pool positions."""
    B, S, _K = idx_a.shape
    overlaps = []
    for b in range(B):
        for s in range(S):
            a = {int(i) for i in idx_a[b, s].tolist() if i >= 0}
            c = {int(i) for i in idx_b[b, s].tolist() if i >= 0}
            if not a and not c:
                continue
            union = a | c
            overlaps.append(len(a & c) / max(1, len(union)))
    return sum(overlaps) / max(1, len(overlaps))


def test_fp8_indexer_topk_overlaps_bf16_reference():
    torch.manual_seed(0)
    ref = _make_indexer(use_fp8_qk=False)
    fp8 = _make_indexer(use_fp8_qk=True)
    # Share identical weights so the only difference is the FP8 QK rounding.
    fp8.load_state_dict(copy.deepcopy(ref.state_dict()))

    hidden = torch.randn(2, 32, 64, dtype=torch.float32)
    with torch.no_grad():
        idx_ref, _ = ref(hidden)
        idx_fp8, _ = fp8(hidden)

    assert idx_ref.shape == idx_fp8.shape == (2, 32, 4)
    # Sentinel semantics preserved (same set of masked-out / early queries).
    assert torch.equal(idx_ref < 0, idx_fp8 < 0)

    overlap = _topk_overlap(idx_ref, idx_fp8)
    assert overlap >= 0.6, f"FP8 indexer top-k overlap with BF16 too low: {overlap}"


def test_fp8_flag_off_is_identical_to_reference():
    torch.manual_seed(0)
    a = _make_indexer(use_fp8_qk=False)
    b = _make_indexer(use_fp8_qk=False)
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    hidden = torch.randn(1, 16, 64, dtype=torch.float32)
    with torch.no_grad():
        idx_a, _ = a(hidden)
        idx_b, _ = b(hidden)
    assert torch.equal(idx_a, idx_b)
