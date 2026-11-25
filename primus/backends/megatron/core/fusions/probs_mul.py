###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``(act * probs).to(act.dtype)`` for the routed experts, bit for bit, in one pass each
way.

Eager: ``act`` ``[T, F]`` BF16 times ``probs`` ``[T, 1]`` FP32 promotes to an FP32
``[T, F]`` product (written, read back) and a cast. Its backward converts the
BF16 gradient to FP32 (``ToCopyBackward0``), multiplies it by ``probs`` and casts
the result back for ``act``, and multiplies it by ``act`` for ``probs``, whose
row sum is the gradient of ``probs``: five elementwise launches, and FP32 ``[T, F]``
intermediates both ways.

Here: one forward kernel (``f32(act) * p`` rounded once to BF16, the same single
rounding as the eager cast), and one backward kernel that writes the BF16
gradient of ``act`` and, when ``probs`` needs a gradient, the FP32 product
``f32(g) * f32(act)``; the row sum of that product stays on ATen's ``sum`` -- the very
reduction autograd's ``sum_to_size`` runs -- so its order, and so every bit, is unchanged.
Other dtypes and layouts take the eager ops, as does a run whose first call does not match
them bit for bit (``bitwise_check``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from primus.backends.megatron.core.fusions import bitwise_check


@triton.jit
def _fwd(A, P, OUT, F, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    m = cols < F
    base = row.to(tl.int64) * F
    a = tl.load(A + base + cols, mask=m, other=0.0).to(tl.float32)
    p = tl.load(P + row)
    tl.store(OUT + base + cols, (a * p).to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _bwd(G, A, P, DA, PROD, F, WRITE_PROD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    m = cols < F
    base = row.to(tl.int64) * F
    g = tl.load(G + base + cols, mask=m, other=0.0).to(tl.float32)
    p = tl.load(P + row)
    tl.store(DA + base + cols, (g * p).to(DA.dtype.element_ty), mask=m)
    if WRITE_PROD:
        a = tl.load(A + base + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(PROD + base + cols, g * a, mask=m)


_BLOCK = 1024


class _ProbsMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, act, probs):
        N, F = act.shape
        out = torch.empty_like(act)
        _fwd[(N, triton.cdiv(F, _BLOCK))](act, probs, out, F, BLOCK=_BLOCK, num_warps=4)
        ctx.save_for_backward(act, probs)
        return out

    @staticmethod
    def backward(ctx, g):
        act, probs = ctx.saved_tensors
        need_dp = ctx.needs_input_grad[1]
        if not g.is_contiguous():
            # eager's FP32 product follows g's layout, and so does the order of its row sum
            g32 = g.float()
            dp = (g32 * act).sum(dim=1, keepdim=True) if need_dp else None
            return (g32 * probs).to(act.dtype), dp
        N, F = act.shape
        da = torch.empty_like(act)
        prod = torch.empty((N, F), dtype=torch.float32, device=act.device) if need_dp else da
        _bwd[(N, triton.cdiv(F, _BLOCK))](
            g, act, probs, da, prod, F, WRITE_PROD=need_dp, BLOCK=_BLOCK, num_warps=4
        )
        # autograd's MulBackward0 + sum_to_size(probs.shape): the same ATen reduction
        dp = prod.sum(dim=1, keepdim=True) if need_dp else None
        return da, dp


def _eager(act: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    return (act * probs).to(act.dtype)


def probs_mul(act: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """``(act * probs).to(act.dtype)`` for BF16 ``act`` ``[T, F]`` and FP32 ``probs`` ``[T, 1]``."""
    if (
        act.is_cuda
        and act.dtype == torch.bfloat16
        and probs.dtype == torch.float32
        and act.dim() == 2
        and act.numel() > 0
        and act.is_contiguous()
        and probs.shape == (act.shape[0], 1)
        and probs.is_contiguous()
        and bitwise_check.passed(
            "fused_probs_mul",
            lambda: bitwise_check.outputs_and_grads(_ProbsMul.apply, act, probs),
            lambda: bitwise_check.outputs_and_grads(_eager, act, probs),
            # probs' gradient is ATen's row sum in both paths; its order depends on the width
            key=act.shape[1],
        )
    ):
        return _ProbsMul.apply(act, probs)
    return _eager(act, probs)
