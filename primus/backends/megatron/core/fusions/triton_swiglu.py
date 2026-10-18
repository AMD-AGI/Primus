###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused (optionally prob-weighted) SwiGLU forward / backward (Triton, one pinned launch config), and drop-ins
for Megatron's and Primus' SwiGLU entry points.

Megatron's SwiGLU functions are ``@jit_fuser``; without torch.compile they run as unfused eager op chains.
Here: one kernel per direction, fp32 math, one rounding to the output dtype. Rows of y are [gate | up] (2H):

  fwd  out[r, :]  = silu(gate) * up * w[r]
  bwd  dy[r, :H]  = g * w[r] * up * sig * (1 + gate * (1 - sig)),  sig = sigmoid(gate)
       dy[r, H:]  = g * w[r] * silu(gate)
       dw[r]      = sum_c silu(gate) * up * g                  (fp32 accumulation over the row)

Without weights w = 1 and there is no dw. The eager chain rounds intermediates (silu, silu * up, sigmoid, ...)
to the activation dtype; the fused kernels do not. One program per row (grid = (M,)); the row's H columns are
processed in NUM_TILES = cdiv(H, BLOCK_N) unrolled tiles, so there is one binary per H and a large H gives a
long kernel. Columns are masked, offsets are int64. This is the unclamped SwiGLU; the clamped Triton variant
of Primus' grouped MLP is in ``fused_bias_swiglu``.

Drop-ins keep the signatures of the functions they replace. Fused contract: bias None, fp8_input_store False,
clamp_value None, cpu_offload_input False, CUDA bf16/fp16 input with 2 or 3 dims and an even last dim; weights
(if any) floating, one value per row, on the same device. Other calls go to the original function unchanged.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from primus.backends.megatron.core.fusions._notices import warn_fallback_once

# Launch config chosen on gfx1250 (MI455X) at the DeepSeek-V3 expert size (H = 2048). Any value is correct
# (columns are masked); only speed depends on it. No in-process autotune.
BLOCK_N = 1024
NUM_WARPS = 4
NUM_STAGES = 1


@triton.jit
def _sigmoid(x):
    # 1 / (1 + exp(-x)) with the fast (rcp-based, ~1 ulp) divide instead of the IEEE div_scale/fmas/fixup
    # sequence: x -> -inf gives 1 / inf = 0, x -> +inf gives 1 / 1 = 1
    return libdevice.fast_dividef(tl.full(x.shape, 1.0, tl.float32), 1.0 + tl.exp(-x))


@triton.jit
def _swiglu_fwd_kernel(
    y_ptr,
    w_ptr,
    out_ptr,
    H,
    stride_y,
    stride_w,
    stride_o,
    HAS_W: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    y_row = y_ptr + row * stride_y
    o_row = out_ptr + row * stride_o
    if HAS_W:
        w = tl.load(w_ptr + row * stride_w).to(tl.float32)
    for t in tl.static_range(NUM_TILES):
        cols = t * BLOCK_N + tl.arange(0, BLOCK_N)
        m = cols < H
        gate = tl.load(y_row + cols, mask=m, other=0.0).to(tl.float32)
        up = tl.load(y_row + H + cols, mask=m, other=0.0).to(tl.float32)
        out = gate * _sigmoid(gate) * up
        if HAS_W:
            out = out * w
        tl.store(o_row + cols, out.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _swiglu_bwd_kernel(
    g_ptr,
    y_ptr,
    w_ptr,
    dy_ptr,
    dw_ptr,
    H,
    stride_g,
    stride_y,
    stride_w,
    stride_dy,
    stride_dw,
    HAS_W: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    g_row = g_ptr + row * stride_g
    y_row = y_ptr + row * stride_y
    dy_row = dy_ptr + row * stride_dy
    if HAS_W:
        w = tl.load(w_ptr + row * stride_w).to(tl.float32)
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for t in tl.static_range(NUM_TILES):
        cols = t * BLOCK_N + tl.arange(0, BLOCK_N)
        m = cols < H
        g = tl.load(g_row + cols, mask=m, other=0.0).to(tl.float32)
        gate = tl.load(y_row + cols, mask=m, other=0.0).to(tl.float32)
        up = tl.load(y_row + H + cols, mask=m, other=0.0).to(tl.float32)
        sig = _sigmoid(gate)
        silu = gate * sig
        if HAS_W:
            acc += silu * up * g  # masked lanes: g = 0
            gw = g * w
        else:
            gw = g
        d_gate = gw * up * (sig * (1.0 + gate * (1.0 - sig)))
        d_up = gw * silu
        tl.store(dy_row + cols, d_gate.to(dy_ptr.dtype.element_ty), mask=m)
        tl.store(dy_row + H + cols, d_up.to(dy_ptr.dtype.element_ty), mask=m)
    if HAS_W:
        tl.store(dw_ptr + row * stride_dw, tl.sum(acc, axis=0).to(dw_ptr.dtype.element_ty))


def _num_tiles(H):
    return triton.cdiv(H, BLOCK_N)


def _launch_fwd(y, w, out):
    """out [M, H] from y [M, 2H] and w ([M] or None); y and out have a contiguous last dim."""
    M, H = out.shape
    if M == 0:
        return
    has_w = w is not None
    _swiglu_fwd_kernel[(M,)](
        y,
        w if has_w else y,
        out,
        H,
        y.stride(0),
        w.stride(0) if has_w else 0,
        out.stride(0),
        HAS_W=has_w,
        NUM_TILES=_num_tiles(H),
        BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )


def _launch_bwd(g, y, w, dy, dw):
    """dy [M, 2H] (and dw [M] when w is given) from g [M, H] and y [M, 2H]; last dims contiguous."""
    M, H = g.shape
    if M == 0:
        return
    has_w = w is not None
    _swiglu_bwd_kernel[(M,)](
        g,
        y,
        w if has_w else y,
        dy,
        dw if has_w else dy,
        H,
        g.stride(0),
        y.stride(0),
        w.stride(0) if has_w else 0,
        dy.stride(0),
        dw.stride(0) if has_w else 0,
        HAS_W=has_w,
        NUM_TILES=_num_tiles(H),
        BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )


class FusedSwiGLU(torch.autograd.Function):
    """out = silu(x2[:, :H]) * x2[:, H:] (* weights); gradients for x2 and, if given, weights."""

    @staticmethod
    def forward(ctx, x2, weights):
        ctx.save_for_backward(x2, weights)
        M, K = x2.shape
        out = torch.empty((M, K // 2), dtype=x2.dtype, device=x2.device)
        _launch_fwd(x2, None if weights is None else weights.reshape(M), out)
        return out

    @staticmethod
    def backward(ctx, g):
        x2, weights = ctx.saved_tensors
        M, K = x2.shape
        g = g if g.stride(-1) == 1 else g.contiguous()
        dx = torch.empty((M, K), dtype=x2.dtype, device=x2.device)
        if weights is None:
            _launch_bwd(g, x2, None, dx, None)
            return dx, None
        dw = torch.empty((M,), dtype=weights.dtype, device=x2.device)
        _launch_bwd(g, x2, weights.reshape(M), dx, dw)
        return dx, dw.view(weights.shape)


def fused_swiglu(x, weights=None):
    """x [..., 2H] (2-D or 3-D), weights None or one value per row -> [..., H] (Megatron's output shape)."""
    x2 = x.reshape(-1, x.shape[-1])
    out = FusedSwiGLU.apply(x2 if x2.stride(-1) == 1 else x2.contiguous(), weights)
    return out if x.dim() == 2 else out.view(*x.shape[:-1], -1)


def _input_reason(x, bias, fp8_input_store, clamp_value=None, cpu_offload_input=False):
    """Why a call cannot take the fused path, from everything but the weights (None = it can)."""
    if bias is not None:
        return "bias"
    if fp8_input_store:
        return "fp8_input_store"
    if clamp_value is not None:
        return "clamp_value"
    if cpu_offload_input:
        return "cpu_offload_input"
    if not x.is_cuda:
        return "input not a CUDA tensor"
    if x.dtype not in (torch.bfloat16, torch.float16):
        return f"input dtype {x.dtype}"
    if x.dim() not in (2, 3) or x.shape[-1] % 2 != 0:
        return "input not 2-D / 3-D with an even last dim"
    return None


def _weights_reason(x, weights):
    if weights is None or not weights.is_floating_point() or weights.device != x.device:
        return "weights missing, not floating point or on another device"
    if (
        weights.numel() != x.numel() // x.shape[-1]
        or weights.dim() not in (1, 2)
        or (weights.dim() == 2 and weights.shape[-1] != 1)
    ):
        return "weights not one value per row"
    return None


def make_primus_weighted(orig, key: str):
    """Drop-in for Primus' ``weighted_bias_swiglu_impl`` (routed experts of ``PrimusGroupedMLP``)."""

    @functools.wraps(orig)
    def weighted_bias_swiglu_impl(input, bias, weights, fp8_input_store=False, clamp_value=None):
        why = _input_reason(input, bias, fp8_input_store, clamp_value=clamp_value) or _weights_reason(
            input, weights
        )
        if why is None:
            return fused_swiglu(input, weights)
        warn_fallback_once(key, why)
        return orig(input, bias, weights, fp8_input_store, clamp_value)

    return weighted_bias_swiglu_impl


def make_megatron_weighted(orig, key: str):
    """Drop-in for Megatron's ``weighted_bias_swiglu_impl``."""

    @functools.wraps(orig)
    def weighted_bias_swiglu_impl(input, bias, weights, fp8_input_store=False):
        why = _input_reason(input, bias, fp8_input_store) or _weights_reason(input, weights)
        if why is None:
            return fused_swiglu(input, weights)
        warn_fallback_once(key, why)
        return orig(input, bias, weights, fp8_input_store)

    return weighted_bias_swiglu_impl


def make_megatron_plain(orig, key: str):
    """Drop-in for Megatron's ``bias_swiglu_impl`` (dense MLP and shared experts)."""

    @functools.wraps(orig)
    def bias_swiglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):
        why = _input_reason(input, bias, fp8_input_store, cpu_offload_input=cpu_offload_input)
        if why is None:
            return fused_swiglu(input)
        warn_fallback_once(key, why)
        return orig(input, bias, fp8_input_store, cpu_offload_input)

    return bias_swiglu_impl
