###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Fused FLUX rotary embedding.

Same math as ``apply_rope`` in ``math.py`` (FP32 rotation of interleaved
pairs, result cast back to the input dtype) in one kernel per tensor. The
rotation is linear, so the backward pass is the same kernel with the sine
negated and nothing but the angles is saved.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - triton ships with ROCm/CUDA torch builds
    triton = None


if triton is not None:

    @triton.jit
    def _rope_kernel(
        x_ptr, cos_ptr, sin_ptr, out_ptr, heads, sign, HALF: tl.constexpr, BLOCK_H: tl.constexpr
    ):
        token = tl.program_id(0)
        h = tl.arange(0, BLOCK_H)[:, None]
        d = tl.arange(0, HALF)[None, :]
        mask = h < heads
        base = token.to(tl.int64) * heads * (2 * HALF) + h * (2 * HALF) + 2 * d
        x0 = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
        angle = token.to(tl.int64) * HALF + d
        c = tl.load(cos_ptr + angle)
        s = tl.load(sin_ptr + angle) * sign
        tl.store(out_ptr + base, (c * x0 - s * x1).to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(out_ptr + base + 1, (s * x0 + c * x1).to(out_ptr.dtype.element_ty), mask=mask)


def _launch(x: Tensor, cos: Tensor, sin: Tensor, sign: float) -> Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    tokens = x.shape[0] * x.shape[1]
    heads = x.shape[2]
    half = x.shape[3] // 2
    _rope_kernel[(tokens,)](x, cos, sin, out, heads, sign, HALF=half, BLOCK_H=triton.next_power_of_2(heads))
    return out


class _FusedRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        ctx.save_for_backward(cos, sin)
        return _launch(x, cos, sin, 1.0)

    @staticmethod
    def backward(ctx, grad: Tensor):
        cos, sin = ctx.saved_tensors
        return _launch(grad, cos, sin, -1.0), None, None


def fused_rope_supported(x: Tensor, freqs_cis: Tensor) -> bool:
    if triton is None or not x.is_cuda or freqs_cis.requires_grad:
        return False
    if x.ndim != 4 or freqs_cis.ndim != 6 or freqs_cis.shape[2] != 1 or freqs_cis.shape[-2:] != (2, 2):
        return False
    half = x.shape[3] // 2
    if x.shape[3] % 2 or half & (half - 1) or freqs_cis.shape[3] != half:
        return False
    return freqs_cis.shape[1] == x.shape[1] and freqs_cis.shape[0] in (1, x.shape[0])


def fused_apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """``freqs_cis`` is ``[B, L, 1, D/2, 2, 2]`` holding ``[[cos, -sin], [sin, cos]]``."""
    shape = (xq.shape[0], xq.shape[1], xq.shape[3] // 2)
    cos = freqs_cis[:, :, 0, :, 0, 0].float().expand(shape).contiguous()
    sin = freqs_cis[:, :, 0, :, 1, 0].float().expand(shape).contiguous()
    return _FusedRope.apply(xq, cos, sin), _FusedRope.apply(xk, cos, sin)
