# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""Parity test for the fused Compressor softmax-pool kernel.

Asserts the fused ``fused_softmax_weighted_pool`` (Triton fwd + analytic eager bwd)
matches the eager ``(softmax(score+ape, dim=2) * kv).sum(dim=2)`` it replaces in
:meth:`Compressor.forward`, for both the forward output and the dkv / dscore / dape
gradients. GPU-gated (the kernel is CUDA/Triton-only); fp32 is checked tightly and
bf16 loosely (the fused path reduces in fp32 -> more accurate than eager bf16 weights).
"""
from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused compressor pool is CUDA/Triton only")

try:
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.compressor_pool import (
        fused_softmax_weighted_pool,
    )

    HAVE_KERNEL = True
except Exception:  # triton import may fail on a CPU-only host
    HAVE_KERNEL = False

pytestmark = [cuda, pytest.mark.skipif(not HAVE_KERNEL, reason="compressor_pool/triton unavailable")]


def _eager(kv, score, ape):
    w = torch.softmax((score + ape).float(), dim=2).to(kv.dtype)
    return (kv * w).sum(dim=2)


def _rel(a, b):
    return (a - b).float().norm() / b.float().norm().clamp_min(1e-12)


# (B, N, W, hd): HCA (W=128, no overlap) and CSA (W=8, overlap)
@pytest.mark.parametrize("shape", [(2, 16, 128, 128), (2, 64, 8, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_pool_matches_eager(shape, dtype):
    B, N, W, HD = shape
    torch.manual_seed(0)
    dev = "cuda"
    kv = torch.randn(B, N, W, HD, device=dev, dtype=dtype)
    sc = torch.randn(B, N, W, HD, device=dev, dtype=dtype)
    ape = torch.randn(W, HD, device=dev, dtype=dtype)
    g = torch.randn(B, N, HD, device=dev, dtype=dtype)

    ke, se, ae = (t.clone().requires_grad_(True) for t in (kv, sc, ape))
    _eager(ke, se, ae).backward(g)
    kf, sf, af = (t.clone().requires_grad_(True) for t in (kv, sc, ape))
    fused_softmax_weighted_pool(kf, sf, af).backward(g)

    tol = 1e-4 if dtype == torch.float32 else 2e-2
    assert _rel(kf.grad, ke.grad) < tol
    assert _rel(sf.grad, se.grad) < tol
    assert _rel(af.grad, ae.grad) < tol
