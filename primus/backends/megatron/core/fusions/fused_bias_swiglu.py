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
import torch.nn.functional as F
from megatron.core.fusions.fused_bias_swiglu import WeightedSwiGLUFunction
from megatron.core.jit import jit_fuser


@jit_fuser
def clamped_weighted_swiglu(y: torch.Tensor, weights: torch.Tensor, clamp_value: float) -> torch.Tensor:
    """Clamped SwiGLU with per-token weights in one fused region.

    Semantics match the eager path:
        gate_c = clamp(gate, max=alpha)
        up_c   = clamp(up, min=-alpha, max=alpha)
        out    = SiLU(gate_c) * up_c * weights
    """
    half = y.shape[-1] // 2
    y_glu = y[..., :half].clamp(max=clamp_value)
    y_linear = y[..., half:].clamp(min=-clamp_value, max=clamp_value)
    dtype = y.dtype
    return (F.silu(y_glu) * y_linear * weights).to(dtype)


@jit_fuser
def clamped_swiglu_back(g: torch.Tensor, y: torch.Tensor, clamp_value: float) -> torch.Tensor:
    """Backward for clamped SwiGLU w.r.t. the pre-clamp concatenated input."""
    half = y.shape[-1] // 2
    y_glu = y[..., :half]
    y_linear = y[..., half:]
    y_glu_c = y_glu.clamp(max=clamp_value)
    y_linear_c = y_linear.clamp(min=-clamp_value, max=clamp_value)

    silu_glu = F.silu(y_glu_c)
    sigmoid_glu = torch.sigmoid(y_glu_c)
    dy_glu_c = g * sigmoid_glu * (1 + y_glu_c * (1 - sigmoid_glu)) * y_linear_c
    dy_linear_c = g * silu_glu

    dy_glu = dy_glu_c * (y_glu <= clamp_value).to(dy_glu_c.dtype)
    dy_linear = dy_linear_c * ((y_linear >= -clamp_value) & (y_linear <= clamp_value)).to(dy_linear_c.dtype)
    return torch.cat((dy_glu, dy_linear), dim=-1)


@jit_fuser
def clamped_weighted_swiglu_back(g: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, clamp_value: float):
    """Backward for :func:`clamped_weighted_swiglu`."""
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = clamped_swiglu_back(g * weights, y, clamp_value)

    half = y.shape[-1] // 2
    y_glu_c = y[..., :half].clamp(max=clamp_value)
    y_linear_c = y[..., half:].clamp(min=-clamp_value, max=clamp_value)
    activation = F.silu(y_glu_c) * y_linear_c
    weights_grad = activation * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


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
