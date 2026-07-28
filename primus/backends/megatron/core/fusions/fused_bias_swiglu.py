###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus override for clamped weighted SwiGLU MoE activation fusion.

Fuses clamp(gate), clamp(up), SiLU, multiply, and router-prob weighting into
a single ``@jit_fuser`` forward/backward pair instead of separate ``chunk`` /
``clamp`` / ``clamp`` / ``cat`` ATen ops followed by ``WeightedSwiGLUFunction``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from megatron.core.fusions.fused_bias_swiglu import WeightedSwiGLUFunction

# Column tile size for the elementwise kernels / reduction inner loop.
_BLOCK_N = 1024


@triton.jit
def _clamped_weighted_swiglu_fwd_kernel(
    y_ptr,  # [M, 2*half] row-major: gate = [:, :half], up = [:, half:]
    w_ptr,  # [M] per-token weights (only read if HAS_WEIGHTS)
    out_ptr,  # [M, half] row-major output
    half,
    K,  # = 2 * half (row stride of y)
    stride_w,
    clamp_value,
    HAS_WEIGHTS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < half

    y_row = y_ptr + pid_m * K
    gate = tl.load(y_row + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(y_row + half + cols, mask=mask, other=0.0).to(tl.float32)

    gate_c = tl.minimum(gate, clamp_value)
    up_c = tl.minimum(tl.maximum(up, -clamp_value), clamp_value)
    out = (gate_c * tl.sigmoid(gate_c)) * up_c
    if HAS_WEIGHTS:
        out = out * tl.load(w_ptr + pid_m * stride_w).to(tl.float32)

    tl.store(out_ptr + pid_m * half + cols, out, mask=mask)


@triton.jit
def _clamped_swiglu_bwd_kernel(
    g_ptr,  # [M, half] grad w.r.t. output
    y_ptr,  # [M, 2*half] pre-clamp input
    w_ptr,  # [M] per-token weights (only read if HAS_WEIGHTS)
    dy_ptr,  # [M, 2*half] grad w.r.t. input
    half,
    K,
    stride_gm,
    stride_w,
    clamp_value,
    HAS_WEIGHTS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < half

    g = tl.load(g_ptr + pid_m * stride_gm + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_WEIGHTS:
        g = g * tl.load(w_ptr + pid_m * stride_w).to(tl.float32)

    y_row = y_ptr + pid_m * K
    gate = tl.load(y_row + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(y_row + half + cols, mask=mask, other=0.0).to(tl.float32)

    gate_c = tl.minimum(gate, clamp_value)
    up_c = tl.minimum(tl.maximum(up, -clamp_value), clamp_value)
    sig = tl.sigmoid(gate_c)
    silu = gate_c * sig

    # d/d gate_c [SiLU(gate_c)] = sig * (1 + gate_c * (1 - sig))
    dy_glu_c = g * sig * (1.0 + gate_c * (1.0 - sig)) * up_c
    dy_linear_c = g * silu

    # Straight-through the clamp: gradient passes only inside the (un)clamped region.
    gate_keep = (gate <= clamp_value).to(tl.float32)
    up_keep = ((up >= -clamp_value) & (up <= clamp_value)).to(tl.float32)

    dy_row = dy_ptr + pid_m * K
    tl.store(dy_row + cols, dy_glu_c * gate_keep, mask=mask)
    tl.store(dy_row + half + cols, dy_linear_c * up_keep, mask=mask)


@triton.jit
def _clamped_swiglu_weights_grad_kernel(
    g_ptr,  # [M, half] grad w.r.t. output
    y_ptr,  # [M, 2*half] pre-clamp input
    wg_ptr,  # [M] per-token weights grad
    half,
    K,
    stride_gm,
    clamp_value,
    NUM_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    y_row = y_ptr + pid_m * K
    g_row = g_ptr + pid_m * stride_gm

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for t in tl.static_range(NUM_TILES):
        cols = t * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < half
        g = tl.load(g_row + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(y_row + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(y_row + half + cols, mask=mask, other=0.0).to(tl.float32)
        gate_c = tl.minimum(gate, clamp_value)
        up_c = tl.minimum(tl.maximum(up, -clamp_value), clamp_value)
        activation = (gate_c * tl.sigmoid(gate_c)) * up_c
        acc += activation * g  # masked lanes contribute 0 (g == 0)

    tl.store(wg_ptr + pid_m, tl.sum(acc, axis=0))


def clamped_weighted_swiglu(y: torch.Tensor, weights: torch.Tensor, clamp_value: float) -> torch.Tensor:
    """Clamped SwiGLU with per-token weights in one fused Triton kernel.

    Semantics match the eager path:
        gate_c = clamp(gate, max=alpha)
        up_c   = clamp(up, min=-alpha, max=alpha)
        out    = SiLU(gate_c) * up_c * weights
    """
    lead = y.shape[:-1]
    K = y.shape[-1]
    half = K // 2
    y2 = y.reshape(-1, K).contiguous()
    M = y2.shape[0]
    w = weights.reshape(M).contiguous()

    out = torch.empty((M, half), dtype=y.dtype, device=y.device)
    grid = (M, triton.cdiv(half, _BLOCK_N))
    _clamped_weighted_swiglu_fwd_kernel[grid](
        y2,
        w,
        out,
        half,
        K,
        w.stride(0),
        float(clamp_value),
        HAS_WEIGHTS=True,
        BLOCK_N=_BLOCK_N,
    )
    return out.reshape(*lead, half)


def clamped_swiglu(y: torch.Tensor, clamp_value: float) -> torch.Tensor:
    """Clamped SwiGLU without per-token weights in one fused Triton kernel.

    Semantics match the eager path used by ``mlp.MLP`` shared experts:
        gate_c = clamp(gate, max=alpha)
        up_c   = clamp(up, min=-alpha, max=alpha)
        out    = SiLU(gate_c) * up_c
    """
    lead = y.shape[:-1]
    K = y.shape[-1]
    half = K // 2
    y2 = y.reshape(-1, K).contiguous()
    M = y2.shape[0]

    out = torch.empty((M, half), dtype=y.dtype, device=y.device)
    grid = (M, triton.cdiv(half, _BLOCK_N))
    # w_ptr is unused when HAS_WEIGHTS=False; pass y2 as a valid placeholder.
    _clamped_weighted_swiglu_fwd_kernel[grid](
        y2,
        y2,
        out,
        half,
        K,
        0,
        float(clamp_value),
        HAS_WEIGHTS=False,
        BLOCK_N=_BLOCK_N,
    )
    return out.reshape(*lead, half)


def clamped_swiglu_back(g: torch.Tensor, y: torch.Tensor, clamp_value: float) -> torch.Tensor:
    """Backward for clamped SwiGLU w.r.t. the pre-clamp concatenated input."""
    lead = y.shape[:-1]
    K = y.shape[-1]
    half = K // 2
    y2 = y.reshape(-1, K).contiguous()
    g2 = g.reshape(-1, half).contiguous()
    M = y2.shape[0]

    dy = torch.empty((M, K), dtype=g.dtype, device=g.device)
    grid = (M, triton.cdiv(half, _BLOCK_N))
    _clamped_swiglu_bwd_kernel[grid](
        g2,
        y2,
        g2,
        dy,
        half,
        K,
        g2.stride(0),
        0,
        float(clamp_value),
        HAS_WEIGHTS=False,
        BLOCK_N=_BLOCK_N,
    )
    return dy.reshape(*lead, K)


def clamped_weighted_swiglu_back(g: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, clamp_value: float):
    """Backward for :func:`clamped_weighted_swiglu`."""
    input_dtype = y.dtype
    w_dtype = weights.dtype
    lead = y.shape[:-1]
    K = y.shape[-1]
    half = K // 2
    y2 = y.reshape(-1, K).contiguous()
    g2 = g.reshape(-1, half).contiguous()
    M = y2.shape[0]
    w = weights.reshape(M).contiguous()

    # input grad: reuse the clamped-swiglu backward kernel with grad pre-scaled by weights.
    input_grad = torch.empty((M, K), dtype=input_dtype, device=y.device)
    grid = (M, triton.cdiv(half, _BLOCK_N))
    _clamped_swiglu_bwd_kernel[grid](
        g2,
        y2,
        w,
        input_grad,
        half,
        K,
        g2.stride(0),
        w.stride(0),
        float(clamp_value),
        HAS_WEIGHTS=True,
        BLOCK_N=_BLOCK_N,
    )

    # weights grad: sum over the feature dim of SiLU(gate_c) * up_c * g.
    weights_grad = torch.empty((M,), dtype=w_dtype, device=y.device)
    _clamped_swiglu_weights_grad_kernel[(M,)](
        g2,
        y2,
        weights_grad,
        half,
        K,
        g2.stride(0),
        float(clamp_value),
        NUM_TILES=triton.cdiv(half, _BLOCK_N),
        BLOCK_N=_BLOCK_N,
    )

    return input_grad.reshape(*lead, K), weights_grad.reshape(*weights.shape)


class ClampedWeightedSwiGLUFunction(torch.autograd.Function):
    """Autograd wrapper for clamped token-wise weighted SwiGLU."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, weights: torch.Tensor, fp8_input_store: bool, clamp_value: float):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        ctx.clamp_value = float(clamp_value)
        return clamped_weighted_swiglu(input, weights, ctx.clamp_value)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        input_grad, weights_grad = clamped_weighted_swiglu_back(grad_output, input, weights, ctx.clamp_value)
        return input_grad, weights_grad, None, None


def weighted_bias_swiglu_impl(input, bias, weights, fp8_input_store=False, clamp_value=None):
    """Token-wise-weighted SwiGLU fusion with optional clamped gate/up."""
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted swiglu fusion")

    if clamp_value is not None:
        output = ClampedWeightedSwiGLUFunction.apply(input, weights, fp8_input_store, float(clamp_value))
    else:
        output = WeightedSwiGLUFunction.apply(input, weights, fp8_input_store)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


class ClampedSwiGLUFunction(torch.autograd.Function):
    """Autograd wrapper for clamped (non-weighted) SwiGLU."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, fp8_input_store: bool, clamp_value: float):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        ctx.clamp_value = float(clamp_value)
        return clamped_swiglu(input, ctx.clamp_value)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        input_grad = clamped_swiglu_back(grad_output, input, ctx.clamp_value)
        return input_grad, None, None


def swiglu_impl(input, bias, fp8_input_store=False, clamp_value=None):
    """Non-weighted SwiGLU fusion with optional clamped gate/up.

    Mirrors ``megatron.core.fusions.fused_bias_swiglu.bias_swiglu_impl`` but
    adds the DeepSeek-V4 pre-multiplication clamp when ``clamp_value`` is set.
    Used by the shared-expert MLP path, which has no per-token weights.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        raise NotImplementedError("Bias is not supported for clamped swiglu fusion")

    if clamp_value is not None:
        output = ClampedSwiGLUFunction.apply(input, fp8_input_store, float(clamp_value))
    else:
        from megatron.core.fusions.fused_bias_swiglu import SwiGLUFunction

        output = SwiGLUFunction.apply(input, fp8_input_store, False)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
