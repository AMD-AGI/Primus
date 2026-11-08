###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the attention-residual mixer backends in ``kimi_k3/attn_res_kernels``.

``eager`` is the numerical oracle. ``eager_bcast`` must match it on CPU up to
fp32 rounding; ``flydsl`` (gfx950 / gfx1250 only) must match it in values and
in all four gradients, including when one compiled kernel is reused for a
different token count (the launcher caches per hidden size and block count,
not per token count).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels import (  # noqa: E402
    ATTN_RES_BACKENDS,
    resolve_attn_res_backend,
)

EPS = 1e-6


def _inputs(tokens, hidden, num_blocks, dtype, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ps = torch.randn(tokens, hidden, generator=g).to(device=device, dtype=dtype)
    br = torch.randn(tokens, num_blocks, hidden, generator=g).to(device=device, dtype=dtype)
    nw = (1.0 + 0.1 * torch.randn(hidden, generator=g)).to(device)
    pw = (0.05 * torch.randn(1, hidden, generator=g)).to(device)
    dout = torch.randn(tokens, hidden, generator=g).to(device=device, dtype=dtype)
    return ps, br, nw, pw, dout


def _run(backend, ps, br, nw, pw, dout):
    fn = resolve_attn_res_backend(backend)
    ps, br, nw, pw = (t.detach().clone().requires_grad_() for t in (ps, br, nw, pw))
    out = fn(ps, br, nw, pw, EPS)
    out.backward(dout)
    return [t.detach().float() for t in (out, ps.grad, br.grad, nw.grad, pw.grad)]


def _assert_close(ref, got, rel):
    names = ("out", "d_prefix_sum", "d_block_residual", "d_norm_weight", "d_proj_weight")
    for name, a, b in zip(names, ref, got):
        assert a.shape == b.shape, f"{name}: shape {tuple(b.shape)} != {tuple(a.shape)}"
        if a.numel() == 0:
            continue
        err = ((a - b).abs().max() / a.abs().max().clamp_min(1e-30)).item()
        assert err < rel, f"{name}: relative error {err:.2e} >= {rel:.0e}"


def test_backend_names():
    assert ATTN_RES_BACKENDS == ("eager", "eager_bcast", "flydsl")
    with pytest.raises(ValueError):
        resolve_attn_res_backend("cuda")


@pytest.mark.parametrize("num_blocks", [0, 1, 3])
@pytest.mark.parametrize("dtype, rel", [(torch.float64, 1e-12), (torch.float32, 1e-5)])
def test_eager_bcast_matches_eager(num_blocks, dtype, rel):
    args = _inputs(6, 16, num_blocks, dtype, "cpu", seed=num_blocks)
    _assert_close(_run("eager", *args), _run("eager_bcast", *args), rel)


def _flydsl_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("no ROCm / CUDA device")
    try:
        resolve_attn_res_backend("flydsl")
    except ImportError as e:
        pytest.skip(f"flydsl backend unavailable: {e}")


@pytest.mark.parametrize(
    "tokens, hidden, num_blocks, dtype, rel",
    [
        (512, 1024, 3, torch.float32, 1e-4),
        (256, 768, 2, torch.float32, 1e-4),  # hidden % 1024 != 0: the scalar kernels
        (8192, 7168, 1, torch.bfloat16, 2e-2),  # Kimi K3: seq 4096 x micro-batch 2, hidden 7168
        (8192, 7168, 3, torch.bfloat16, 2e-2),
    ],
)
def test_flydsl_matches_eager(tokens, hidden, num_blocks, dtype, rel):
    _flydsl_or_skip()
    args = _inputs(tokens, hidden, num_blocks, dtype, "cuda", seed=tokens + num_blocks)
    _assert_close(_run("eager", *args), _run("flydsl", *args), rel)


@pytest.mark.parametrize("dtype, rel", [(torch.float32, 1e-4), (torch.bfloat16, 2e-2)])
def test_flydsl_reuses_kernel_across_token_counts(dtype, rel):
    _flydsl_or_skip()
    for tokens in (1024, 3072, 1024):
        args = _inputs(tokens, 2048, 2, dtype, "cuda", seed=tokens)
        _assert_close(_run("eager", *args), _run("flydsl", *args), rel)
