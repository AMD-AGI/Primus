###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Clamped quick-GEGLU (``swigluoai``) as one forward and one backward Triton kernel,
bitwise equal to Megatron's eager GLU branch.

Megatron runs ``swigluoai`` (``quick_geglu`` + ``activation_func_clamp_value``) as::

    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    x_glu = x_glu.clamp(max=L)
    x_linear = x_linear.clamp(-L, L)
    out = quick_gelu(x_glu) * (x_linear + offset)     # quick_gelu is torch.compile'd

That is ~5 forward kernels and ~10 backward kernels (clamp masks, where, muls, the compiled
quick_gelu backward and the ``cat`` that reassembles the two input-gradient halves), each
reading or writing an ffn-wide BF16 tensor.

The kernels here reproduce every rounding of that sequence, so outputs and gradients are
bit-identical:

* forward: ``a = bf16(qg(clamp(xg)))`` with Inductor's FP32 formula for ``quick_gelu``
  (``y * sigmoid(y * 1.702)`` in FP32, rounded once), ``b = bf16(clamp(xl) + offset)``,
  ``out = bf16(a * b)``.
* backward: ``MulBackward0`` gives ``ga = bf16(g_out * b)`` and ``gb = bf16(g_out * a)``;
  the compiled quick_gelu backward gives ``bf16(ga * s + (ga * y) * (s * (1 - s)) * 1.702)``
  in FP32 (``s`` the FP32 sigmoid); the clamp backwards pass the gradient inside the range,
  and at the bounds exactly when the installed torch does (torch 2.11 does, 2.14 does not);
  both halves are written into one ``[.., 2F]`` gradient.

Only the input is saved for backward (the eager path keeps the clamped halves, ``a`` and
``b``). Inputs that eager would not compute this way (``quick_gelu`` not compiled) are not
eligible, nor are clamp bounds or offsets that BF16 does not hold exactly: eager compares and
clamps against the BF16 value, the kernels against the FP32 one.

The kernels give ``quick_gelu(-0.0) = -0.0``, as Inductor's kernel does when it first compiles
``quick_gelu``. Recompiled for dynamic shapes, that kernel returned +0.0 in one case on gfx1250
(2-D, 256 elements, after two 3-D shapes). A process whose ``quick_gelu`` was compiled that way
before the first eligible call fails the self-check's -0.0 probe and keeps the eager branch.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch
import triton
import triton.language as tl

from primus.backends.megatron.core.fusions import bitwise_check

_BLOCK = 1024


@triton.jit
def _quick_gelu(y):
    # Inductor: tmp1 = tmp0 * 1.702; tmp2 = sigmoid(tmp1); tmp3 = tmp0 * tmp2
    return y * tl.sigmoid(y * 1.702)


@triton.jit
def _clamp_max(x, hi):
    # torch.clamp keeps NaN
    return tl.minimum(x, hi, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _clamp(x, lo, hi):
    return _clamp_max(tl.maximum(x, lo, propagate_nan=tl.PropagateNan.ALL), hi)


@triton.jit
def _fwd_kernel(X, Y, F, n_blocks_f, limit, offset, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = (pid // n_blocks_f).to(tl.int64)
    cols = (pid % n_blocks_f) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < F
    xg = tl.load(X + row * 2 * F + cols, mask=mask, other=0.0)
    xl = tl.load(X + row * 2 * F + F + cols, mask=mask, other=0.0)
    g = _clamp_max(xg.to(tl.float32), limit).to(tl.bfloat16)  # clamp(max=L): exact in bf16
    l = _clamp(xl.to(tl.float32), -limit, limit)  # clamp(-L, L): exact
    a = _quick_gelu(g.to(tl.float32)).to(tl.bfloat16)
    b = (l + offset).to(tl.bfloat16)
    y = (a.to(tl.float32) * b.to(tl.float32)).to(tl.bfloat16)
    tl.store(Y + row * F + cols, y, mask=mask)


@triton.jit
def _bwd_kernel(
    X,
    DY,
    DX,
    F,
    n_blocks_f,
    limit,
    offset,
    GLU_AT_BOUND: tl.constexpr,
    LINEAR_AT_BOUND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = (pid // n_blocks_f).to(tl.int64)
    cols = (pid % n_blocks_f) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < F
    xg = tl.load(X + row * 2 * F + cols, mask=mask, other=0.0).to(tl.float32)
    xl = tl.load(X + row * 2 * F + F + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * F + cols, mask=mask, other=0.0).to(tl.float32)
    g = _clamp_max(xg, limit)
    l = _clamp(xl, -limit, limit)
    a = _quick_gelu(g).to(tl.bfloat16).to(tl.float32)
    b = (l + offset).to(tl.bfloat16).to(tl.float32)
    ga = (dy * b).to(tl.bfloat16).to(tl.float32)  # MulBackward0: grad * other
    gb = (dy * a).to(tl.bfloat16)  # MulBackward0: grad * self; AddBackward passes it
    s = tl.sigmoid(g * 1.702)
    gg = (ga * s + (ga * g) * (s * (1.0 - s)) * 1.702).to(tl.bfloat16)
    zero = tl.zeros_like(gg)
    if GLU_AT_BOUND:
        keep_g = xg <= limit
    else:
        keep_g = xg < limit
    if LINEAR_AT_BOUND:
        keep_l = (xl >= -limit) & (xl <= limit)
    else:
        keep_l = (xl > -limit) & (xl < limit)
    dxg = tl.where(keep_g, gg, zero)
    dxl = tl.where(keep_l, gb, zero)
    tl.store(DX + row * 2 * F + cols, dxg, mask=mask)
    tl.store(DX + row * 2 * F + F + cols, dxl, mask=mask)


class FusedClampedQuickGeGLU(torch.autograd.Function):
    """``quick_gelu(clamp(x_glu, max=L)) * (clamp(x_linear, -L, L) + offset)``, bitwise as eager."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, limit: float, offset: float) -> torch.Tensor:
        x = x.contiguous()
        f = x.shape[-1] // 2
        rows = x.numel() // (2 * f)
        y = torch.empty(*x.shape[:-1], f, dtype=x.dtype, device=x.device)
        nbf = triton.cdiv(f, _BLOCK)
        if rows:
            _fwd_kernel[(rows * nbf,)](x, y, f, nbf, float(limit), float(offset), BLOCK=_BLOCK)
        ctx.save_for_backward(x)
        ctx.limit, ctx.offset = float(limit), float(offset)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        f = x.shape[-1] // 2
        rows = x.numel() // (2 * f)
        dx = torch.empty_like(x)
        nbf = triton.cdiv(f, _BLOCK)
        if rows:
            glu_at_bound, linear_at_bound = _clamp_grad_at_bounds()
            _bwd_kernel[(rows * nbf,)](
                x,
                dy,
                dx,
                f,
                nbf,
                ctx.limit,
                ctx.offset,
                GLU_AT_BOUND=glu_at_bound,
                LINEAR_AT_BOUND=linear_at_bound,
                BLOCK=_BLOCK,
            )
        return dx, None, None


@functools.cache
def _clamp_grad_at_bounds() -> tuple:
    """Whether torch's backward of ``clamp(max=L)`` and of ``clamp(-L, L)`` passes the gradient
    at the bounds."""
    with torch.enable_grad():
        x = torch.tensor([1.0, -1.0], requires_grad=True)
        x.clamp(max=1.0).sum().backward()
        glu = bool(x.grad[0] == 1)
        x.grad = None
        x.clamp(min=-1.0, max=1.0).sum().backward()
        linear = bool(x.grad[0] == 1 and x.grad[1] == 1)
    return glu, linear


@functools.cache
def _quick_gelu_is_compiled() -> bool:
    from megatron.core.fusions.fused_bias_geglu import quick_gelu

    return hasattr(quick_gelu, "_torchdynamo_orig_callable")


@functools.cache
def _bf16_exact(value: float) -> bool:
    return float(torch.tensor(value, dtype=torch.bfloat16)) == float(value)


def reference(x: torch.Tensor, limit: float, offset: float) -> torch.Tensor:
    """Megatron MLP's eager clamped quick-GEGLU (swigluoai) branch."""
    from megatron.core.fusions.fused_bias_geglu import quick_gelu

    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    return quick_gelu(x_glu) * (x_linear + offset)


def edge_values(limit: float, offset: float) -> torch.Tensor:
    """BF16 inputs where the clamp masks, roundings and NaN / inf handling are most likely to
    differ: the bounds and their neighbours, the zero of ``x_linear + offset``, signed zeros,
    infinities and NaN."""
    bounds = torch.tensor([limit, -limit], dtype=torch.bfloat16)
    neighbours = torch.cat([bounds.view(torch.int16) + 1, bounds.view(torch.int16) - 1]).view(torch.bfloat16)
    specials = torch.tensor([-offset, 0.0, -0.0, float("inf"), float("-inf"), float("nan")])
    return torch.cat([bounds, neighbours, specials.to(torch.bfloat16)])


def _self_check(shape, device, limit: float, offset: float) -> bool:
    def probe():
        x = (4 * bitwise_check.randn(shape, device)).to(torch.bfloat16)
        rows, half = x.view(-1, shape[-1]), shape[-1] // 2
        if rows.shape[0]:
            edges = edge_values(limit, offset)[:half].to(device)
            rows[0, : edges.numel()] = edges
            rows[0, half : half + edges.numel()] = edges
        return x

    return bitwise_check.passed(
        "fused_clamped_geglu",
        lambda: bitwise_check.outputs_and_grads(
            lambda x: fused_clamped_quick_geglu(x, limit, offset), probe()
        ),
        lambda: bitwise_check.outputs_and_grads(lambda x: reference(x, limit, offset), probe()),
        # elementwise: the shape changes no value
        key=(limit, offset),
    )


def ineligible_reason(shape, dtype, device, limit: float, offset: float) -> Optional[str]:
    """None if :func:`fused_clamped_quick_geglu` gives eager's bits for an input of this shape,
    dtype and device (checked against :func:`reference` on the first call), else why not."""
    if dtype != torch.bfloat16 or torch.device(device).type != "cuda" or shape[-1] % 2:
        return "not an even-width BF16 GPU tensor"
    if not (_bf16_exact(limit) and _bf16_exact(offset)):
        return f"clamp bound {limit} or offset {offset} is not exact in BF16"
    # The kernels follow Inductor's FP32 quick_gelu; eager rounds each of its ops to BF16.
    if not _quick_gelu_is_compiled() or torch._dynamo.config.disable:
        return "Megatron's quick_gelu is not torch.compile'd"
    if not _self_check(shape, device, limit, offset):
        return "not bit-identical to the eager branch with the installed versions"
    return None


def eligible(shape, dtype, device, limit: float, offset: float) -> bool:
    return ineligible_reason(shape, dtype, device, limit, offset) is None


def fused_clamped_quick_geglu(x: torch.Tensor, limit: float, offset: float) -> torch.Tensor:
    return FusedClampedQuickGeGLU.apply(x, limit, offset)
