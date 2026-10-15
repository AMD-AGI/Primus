###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Autograd functions and drop-in replacements for the SwiGLU entry points, backed by
``triton_swiglu_kernels`` (one pinned Triton config, no autotune).

Drop-ins keep the signatures of the functions they replace; any call outside the fused contract goes to the
original function unchanged:

  primus.backends.megatron.core.fusions.fused_bias_swiglu.weighted_bias_swiglu_impl(
      input, bias, weights, fp8_input_store=False, clamp_value=None)          routed experts (PrimusGroupedMLP)
  megatron.core.fusions.fused_bias_swiglu.weighted_bias_swiglu_impl(input, bias, weights, fp8_input_store=False)
  megatron.core.fusions.fused_bias_swiglu.bias_swiglu_impl(
      input, bias, fp8_input_store=False, cpu_offload_input=False)            shared expert / dense MLP

Fused contract: bias None, fp8_input_store False, clamp_value None, cpu_offload_input False, CUDA input of
dtype bf16/fp16 with 2 or 3 dims and an even last dim; weights (if any) floating, one value per row, on the
same device.
"""

from __future__ import annotations

import torch

from primus.backends.megatron.core.fusions import triton_swiglu_kernels as K

ORIGINAL_ATTR = "__primus_triton_swiglu_orig__"


def _ok_input(x):
    return (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.dim() in (2, 3)
        and x.shape[-1] % 2 == 0
    )


def _ok_weights(x, weights):
    return (
        weights is not None
        and weights.is_floating_point()
        and weights.device == x.device
        and weights.numel() == x.numel() // x.shape[-1]
        and weights.dim() in (1, 2)
        and (weights.dim() == 1 or weights.shape[-1] == 1)
    )


def _rows(x):
    x2 = x.reshape(-1, x.shape[-1])
    return x2 if x2.stride(-1) == 1 else x2.contiguous()


def weighted_fwd(x2, weights):
    """x2 [M, 2H] (last dim contiguous), weights with M values -> out [M, H] (x2.dtype)."""
    M, Kd = x2.shape
    out = torch.empty((M, Kd // 2), dtype=x2.dtype, device=x2.device)
    K.fwd(x2, weights.reshape(M), out)
    return out


def weighted_bwd(g, x2, weights):
    """-> (dx [M, 2H] x2.dtype, dw shaped and typed like weights). g [M, H]."""
    M, Kd = x2.shape
    g2 = g if g.stride(-1) == 1 else g.contiguous()
    dy = torch.empty((M, Kd), dtype=x2.dtype, device=x2.device)
    dw = torch.empty((M,), dtype=weights.dtype, device=x2.device)
    K.bwd(g2, x2, weights.reshape(M), dy, dw)
    return dy, dw.view(weights.shape)


def plain_fwd(x2):
    """x2 [M, 2H] (last dim contiguous) -> out [M, H] (x2.dtype)."""
    M, Kd = x2.shape
    out = torch.empty((M, Kd // 2), dtype=x2.dtype, device=x2.device)
    K.fwd(x2, None, out)
    return out


def plain_bwd(g, x2):
    """-> dx [M, 2H] (x2.dtype). g [M, H]."""
    M, Kd = x2.shape
    g2 = g if g.stride(-1) == 1 else g.contiguous()
    dy = torch.empty((M, Kd), dtype=x2.dtype, device=x2.device)
    K.bwd(g2, x2, None, dy, None)
    return dy


class FusedWeightedSwiGLU(torch.autograd.Function):
    """out = silu(x2[:, :H]) * x2[:, H:] * weights, gradients for x2 and weights."""

    @staticmethod
    def forward(ctx, x2, weights):
        ctx.save_for_backward(x2, weights)
        return weighted_fwd(x2, weights)

    @staticmethod
    def backward(ctx, g):
        x2, weights = ctx.saved_tensors
        return weighted_bwd(g, x2, weights)


class FusedSwiGLU(torch.autograd.Function):
    """out = silu(x2[:, :H]) * x2[:, H:]."""

    @staticmethod
    def forward(ctx, x2):
        ctx.save_for_backward(x2)
        return plain_fwd(x2)

    @staticmethod
    def backward(ctx, g):
        (x2,) = ctx.saved_tensors
        return plain_bwd(g, x2)


def fused_weighted_swiglu(x, weights):
    """x [..., 2H] (2-D or 3-D), weights with one value per row -> [..., H] (Megatron's output shape)."""
    shp = x.shape
    out = FusedWeightedSwiGLU.apply(_rows(x), weights)
    return out if len(shp) == 2 else out.view(shp[0], shp[1], -1)


def fused_swiglu(x):
    """x [..., 2H] (2-D or 3-D) -> [..., H]."""
    shp = x.shape
    out = FusedSwiGLU.apply(_rows(x))
    return out if len(shp) == 2 else out.view(shp[0], shp[1], -1)


def make_primus_weighted(orig):
    """Drop-in for Primus' ``weighted_bias_swiglu_impl``; calls outside the fused contract go to ``orig``."""

    def weighted_bias_swiglu_impl(input, bias, weights, fp8_input_store=False, clamp_value=None):
        if (
            bias is None
            and not fp8_input_store
            and clamp_value is None
            and _ok_input(input)
            and _ok_weights(input, weights)
        ):
            return fused_weighted_swiglu(input, weights)
        return orig(input, bias, weights, fp8_input_store, clamp_value)

    setattr(weighted_bias_swiglu_impl, ORIGINAL_ATTR, orig)
    return weighted_bias_swiglu_impl


def make_megatron_weighted(orig):
    """Drop-in for Megatron's ``weighted_bias_swiglu_impl``; calls outside the fused contract go to ``orig``."""

    def weighted_bias_swiglu_impl(input, bias, weights, fp8_input_store=False):
        if bias is None and not fp8_input_store and _ok_input(input) and _ok_weights(input, weights):
            return fused_weighted_swiglu(input, weights)
        return orig(input, bias, weights, fp8_input_store)

    setattr(weighted_bias_swiglu_impl, ORIGINAL_ATTR, orig)
    return weighted_bias_swiglu_impl


def make_megatron_plain(orig):
    """Drop-in for Megatron's ``bias_swiglu_impl``; calls outside the fused contract go to ``orig``."""

    def bias_swiglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):
        if bias is None and not fp8_input_store and not cpu_offload_input and _ok_input(input):
            return fused_swiglu(input)
        return orig(input, bias, fp8_input_store, cpu_offload_input)

    setattr(bias_swiglu_impl, ORIGINAL_ATTR, orig)
    return bias_swiglu_impl
