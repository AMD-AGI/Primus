###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``(act * probs).to(act.dtype)`` for the routed experts, bit for bit, without the
FP32 intermediate.

Eager: ``act`` ``[T, F]`` BF16 times ``probs`` ``[T, 1]`` FP32 promotes to an FP32
``[T, F]`` product (written, read back) and a cast. Its backward converts the
BF16 gradient to FP32 (``ToCopyBackward0``), multiplies it by ``probs`` and casts
the result back for ``act``, and multiplies it by ``act`` for ``probs``, whose
row sum is the gradient of ``probs``. Five elementwise launches and ~1 GB of
traffic per call at M3's shape.

Here: one forward kernel (``f32(act) * p`` rounded once to BF16, the same single
rounding as the eager cast), and one backward kernel that writes the BF16
gradient of ``act`` and the FP32 product ``f32(g) * f32(act)``; the row sum of
that product stays on ATen's ``sum`` -- the very reduction autograd's
``sum_to_size`` runs -- so its order, and so every bit, is unchanged.

``PRIMUS_FUSED_PROBS_MUL=0`` falls back to eager.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("PRIMUS_FUSED_PROBS_MUL", "1") != "0"


@triton.jit
def _fwd(A, P, OUT, N, F, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    m = cols < F
    base = row.to(tl.int64) * F
    a = tl.load(A + base + cols, mask=m, other=0.0).to(tl.float32)
    p = tl.load(P + row)
    tl.store(OUT + base + cols, (a * p).to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _bwd(G, A, P, DA, PROD, N, F, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    m = cols < F
    base = row.to(tl.int64) * F
    g = tl.load(G + base + cols, mask=m, other=0.0).to(tl.float32)
    a = tl.load(A + base + cols, mask=m, other=0.0).to(tl.float32)
    p = tl.load(P + row)
    tl.store(DA + base + cols, (g * p).to(DA.dtype.element_ty), mask=m)
    tl.store(PROD + base + cols, g * a, mask=m)


_BLOCK = 1024


class _ProbsMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, act, probs):
        N, F = act.shape
        out = torch.empty_like(act)
        _fwd[(N, triton.cdiv(F, _BLOCK))](act, probs, out, N, F, BLOCK=_BLOCK, num_warps=4)
        ctx.save_for_backward(act, probs)
        return out

    @staticmethod
    def backward(ctx, g):
        act, probs = ctx.saved_tensors
        N, F = act.shape
        g = g.contiguous()
        da = torch.empty_like(act)
        prod = torch.empty((N, F), dtype=torch.float32, device=act.device)
        _bwd[(N, triton.cdiv(F, _BLOCK))](g, act, probs, da, prod, N, F, BLOCK=_BLOCK, num_warps=4)
        # autograd's MulBackward0 + sum_to_size(probs.shape): the same ATen reduction
        dp = prod.sum(dim=1, keepdim=True) if probs.requires_grad else None
        return da, dp


def probs_mul(act: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """``(act * probs).to(act.dtype)`` for BF16 ``act`` ``[T, F]`` and FP32 ``probs`` ``[T, 1]``."""
    if (
        ENABLED
        and act.is_cuda
        and act.dtype == torch.bfloat16
        and probs.dtype == torch.float32
        and act.dim() == 2
        and act.is_contiguous()
        and probs.shape == (act.shape[0], 1)
        and probs.is_contiguous()
    ):
        return _ProbsMul.apply(act, probs)
    out = act * probs
    return out.to(act.dtype)
