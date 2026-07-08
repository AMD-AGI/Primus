# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Normalization layers for diffusion models.

Provides adaptive normalization layers that condition on timesteps and other
conditioning signals. These are essential for diffusion model architectures
like Flux and DiT.

This module implements:
    - RMSNorm: Root Mean Square Layer Normalization
    - AdaLN: Adaptive Layer Normalization for DiT
    - AdaLNContinuous: Continuous variant of AdaLN for Flux

Reference:
    - DiT Paper: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
    - Flux Paper: "Flux: A Scalable Diffusion Model"
"""

from typing import Tuple

import torch
import torch.nn as nn
import triton
import triton.language as tl
from megatron.core.jit import jit_fuser
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor
from torch.library import triton_op, wrap_triton

# ---------------------------------------------------------------------------
# Opaque LayerNorm custom op — prevents Inductor from decomposing
# native_layer_norm into a Triton Welford reduction (which uses a different
# FP32 accumulation order than the eager CUDA kernel, causing numerical
# divergence amplified by FP8 quantisation).
#
# Unlike @torch.compiler.disable, this stays inside the compiled graph
# (no graph break) while remaining opaque to fusion/decomposition.
# ---------------------------------------------------------------------------

_custom_op = torch.library.custom_op


# ---------------------------------------------------------------------------
# Opaque modulate custom op — prevents Inductor from fusing the
# x * (1 + scale) + shift modulation with surrounding ops.
#
# Uses hand-written Triton kernels that fuse all pointwise ops into single
# kernel launches.  The backward kernel accumulates dscale/dshift in a
# deterministic sequential loop over the sequence dimension (float32
# registers), avoiding the non-deterministic parallel-reduction order that
# Inductor's auto-generated Triton kernels would use.
# ---------------------------------------------------------------------------

_MODULATE_BLOCK_H = 1024


@triton.jit
def _modulate_fwd_kernel(
    X_ptr,
    Scale_ptr,
    Shift_ptr,
    Out_ptr,
    S,
    B,
    H,
    stride_x_s,
    stride_x_b,
    stride_sc_b,
    BLOCK_H: tl.constexpr,
):
    h_block = tl.program_id(0)
    sb_idx = tl.program_id(1)
    s_idx = sb_idx // B
    b_idx = sb_idx % B

    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < H

    x_off = s_idx * stride_x_s + b_idx * stride_x_b + offs_h
    sc_off = b_idx * stride_sc_b + offs_h

    x = tl.load(X_ptr + x_off, mask=mask)
    sc = tl.load(Scale_ptr + sc_off, mask=mask)
    sh = tl.load(Shift_ptr + sc_off, mask=mask)
    out = x * (1.0 + sc) + sh
    tl.store(Out_ptr + x_off, out, mask=mask)


@triton.jit
def _modulate_bwd_kernel(
    Grad_ptr,
    X_ptr,
    Scale_ptr,
    DX_ptr,
    DScale_ptr,
    DShift_ptr,
    S,
    B,
    H,
    stride_g_s,
    stride_g_b,
    stride_x_s,
    stride_x_b,
    stride_sc_b,
    BLOCK_H: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    h_block = tl.program_id(0)
    b_idx = tl.program_id(1)

    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < H

    sc_off = b_idx * stride_sc_b + offs_h
    scale_val = tl.load(Scale_ptr + sc_off, mask=mask).to(tl.float32)
    one_plus_scale = 1.0 + scale_val

    acc_dscale = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_dshift = tl.zeros([BLOCK_H], dtype=tl.float32)

    for s_idx in range(S):
        g_off = s_idx * stride_g_s + b_idx * stride_g_b + offs_h
        x_off = s_idx * stride_x_s + b_idx * stride_x_b + offs_h
        g = tl.load(Grad_ptr + g_off, mask=mask).to(tl.float32)
        x = tl.load(X_ptr + x_off, mask=mask).to(tl.float32)
        dx = g * one_plus_scale
        tl.store(DX_ptr + x_off, dx.to(OUT_DTYPE), mask=mask)
        acc_dscale += g * x
        acc_dshift += g

    tl.store(DScale_ptr + sc_off, acc_dscale.to(OUT_DTYPE), mask=mask)
    tl.store(DShift_ptr + sc_off, acc_dshift.to(OUT_DTYPE), mask=mask)


_TORCH_TO_TRITON_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


@_custom_op("primus::modulate", mutates_args=(), device_types="cuda")
def _opaque_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    if x.dim() != 3 or scale.dim() != 2:
        return x * (1 + scale) + shift
    x, scale, shift = x.contiguous(), scale.contiguous(), shift.contiguous()
    out = torch.empty_like(x)
    S, B, H = x.shape
    BLOCK_H = min(triton.next_power_of_2(H), _MODULATE_BLOCK_H)
    num_h_blocks = triton.cdiv(H, BLOCK_H)
    _modulate_fwd_kernel[(num_h_blocks, S * B)](
        x,
        scale,
        shift,
        out,
        S,
        B,
        H,
        x.stride(0),
        x.stride(1),
        scale.stride(0),
        BLOCK_H=BLOCK_H,
    )
    return out


@_opaque_modulate.register_fake
def _opaque_modulate_fake(x, scale, shift):
    return torch.empty_like(x)


@_custom_op("primus::modulate_backward", mutates_args=(), device_types="cuda")
def _opaque_modulate_backward_op(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    need_dx: bool,
    need_dscale: bool,
    need_dshift: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 3 or scale.dim() != 2:
        grad_output = grad_output.to(x.dtype)
        dx = grad_output * (1 + scale) if need_dx else torch.empty(0, device=x.device, dtype=x.dtype)
        reduce_dims = list(range(grad_output.dim() - scale.dim()))
        dscale = (
            (grad_output * x).sum(dim=reduce_dims)
            if need_dscale
            else torch.empty(0, device=x.device, dtype=x.dtype)
        )
        dshift = (
            grad_output.sum(dim=reduce_dims)
            if need_dshift
            else torch.empty(0, device=x.device, dtype=x.dtype)
        )
        return dx, dscale, dshift

    grad_output = grad_output.to(x.dtype).contiguous()
    x = x.contiguous()
    scale = scale.contiguous()
    S, B, H = x.shape
    dx = torch.empty_like(x)
    dscale = torch.empty_like(scale)
    dshift = torch.empty_like(scale)
    BLOCK_H = min(triton.next_power_of_2(H), _MODULATE_BLOCK_H)
    num_h_blocks = triton.cdiv(H, BLOCK_H)
    out_dtype = _TORCH_TO_TRITON_DTYPE[x.dtype]
    _modulate_bwd_kernel[(num_h_blocks, B)](
        grad_output,
        x,
        scale,
        dx,
        dscale,
        dshift,
        S,
        B,
        H,
        grad_output.stride(0),
        grad_output.stride(1),
        x.stride(0),
        x.stride(1),
        scale.stride(0),
        BLOCK_H=BLOCK_H,
        OUT_DTYPE=out_dtype,
    )
    if not need_dx:
        dx = torch.empty(0, device=x.device, dtype=x.dtype)
    if not need_dscale:
        dscale = torch.empty(0, device=x.device, dtype=x.dtype)
    if not need_dshift:
        dshift = torch.empty(0, device=x.device, dtype=x.dtype)
    return dx, dscale, dshift


@_opaque_modulate_backward_op.register_fake
def _opaque_modulate_backward_fake(grad_output, x, scale, need_dx, need_dscale, need_dshift):
    dx = torch.empty_like(x) if need_dx else torch.empty(0, device=x.device, dtype=x.dtype)
    dscale = torch.empty_like(scale) if need_dscale else torch.empty(0, device=x.device, dtype=x.dtype)
    dshift = torch.empty_like(scale) if need_dshift else torch.empty(0, device=x.device, dtype=x.dtype)
    return dx, dscale, dshift


def _opaque_modulate_setup_context(ctx, inputs, output):
    x, scale, _shift = inputs
    ctx.save_for_backward(x, scale)


def _opaque_modulate_backward(ctx, grad_output):
    x, scale = ctx.saved_tensors
    dx, dscale, dshift = _opaque_modulate_backward_op(
        grad_output,
        x,
        scale,
        True,
        True,
        True,
    )
    return dx, dscale, dshift


_opaque_modulate.register_autograd(
    _opaque_modulate_backward,
    setup_context=_opaque_modulate_setup_context,
)


# ---------------------------------------------------------------------------
# @triton_op versions — transparent to Inductor for cross-op fusion.
# Enabled via config.use_triton_ops=True.
# ---------------------------------------------------------------------------


@triton_op("primus::modulate_v2", mutates_args=())
def _triton_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    if x.dim() != 3 or scale.dim() != 2:
        return x * (1 + scale) + shift
    x, scale, shift = x.contiguous(), scale.contiguous(), shift.contiguous()
    out = torch.empty_like(x)
    S, B, H = x.shape
    BLOCK_H = min(triton.next_power_of_2(H), _MODULATE_BLOCK_H)
    num_h_blocks = triton.cdiv(H, BLOCK_H)
    wrap_triton(_modulate_fwd_kernel)[(num_h_blocks, S * B)](
        x,
        scale,
        shift,
        out,
        S,
        B,
        H,
        x.stride(0),
        x.stride(1),
        scale.stride(0),
        BLOCK_H=BLOCK_H,
    )
    return out


@_triton_modulate.register_fake
def _triton_modulate_fake(x, scale, shift):
    return torch.empty_like(x)


@triton_op("primus::modulate_backward_v2", mutates_args=())
def _triton_modulate_backward_op(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    need_dx: bool,
    need_dscale: bool,
    need_dshift: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 3 or scale.dim() != 2:
        grad_output = grad_output.to(x.dtype)
        dx = grad_output * (1 + scale) if need_dx else torch.empty(0, device=x.device, dtype=x.dtype)
        reduce_dims = list(range(grad_output.dim() - scale.dim()))
        dscale = (
            (grad_output * x).sum(dim=reduce_dims)
            if need_dscale
            else torch.empty(0, device=x.device, dtype=x.dtype)
        )
        dshift = (
            grad_output.sum(dim=reduce_dims)
            if need_dshift
            else torch.empty(0, device=x.device, dtype=x.dtype)
        )
        return dx, dscale, dshift

    grad_output = grad_output.to(x.dtype).contiguous()
    x = x.contiguous()
    scale = scale.contiguous()
    S, B, H = x.shape
    dx = torch.empty_like(x)
    dscale = torch.empty_like(scale)
    dshift = torch.empty_like(scale)
    BLOCK_H = min(triton.next_power_of_2(H), _MODULATE_BLOCK_H)
    num_h_blocks = triton.cdiv(H, BLOCK_H)
    out_dtype = _TORCH_TO_TRITON_DTYPE[x.dtype]
    wrap_triton(_modulate_bwd_kernel)[(num_h_blocks, B)](
        grad_output,
        x,
        scale,
        dx,
        dscale,
        dshift,
        S,
        B,
        H,
        grad_output.stride(0),
        grad_output.stride(1),
        x.stride(0),
        x.stride(1),
        scale.stride(0),
        BLOCK_H=BLOCK_H,
        OUT_DTYPE=out_dtype,
    )
    if not need_dx:
        dx = torch.empty(0, device=x.device, dtype=x.dtype)
    if not need_dscale:
        dscale = torch.empty(0, device=x.device, dtype=x.dtype)
    if not need_dshift:
        dshift = torch.empty(0, device=x.device, dtype=x.dtype)
    return dx, dscale, dshift


@_triton_modulate_backward_op.register_fake
def _triton_modulate_backward_fake(grad_output, x, scale, need_dx, need_dscale, need_dshift):
    dx = torch.empty_like(x) if need_dx else torch.empty(0, device=x.device, dtype=x.dtype)
    dscale = torch.empty_like(scale) if need_dscale else torch.empty(0, device=x.device, dtype=x.dtype)
    dshift = torch.empty_like(scale) if need_dshift else torch.empty(0, device=x.device, dtype=x.dtype)
    return dx, dscale, dshift


def _triton_modulate_setup_context(ctx, inputs, output):
    x, scale, _shift = inputs
    ctx.save_for_backward(x, scale)


def _triton_modulate_backward(ctx, grad_output):
    x, scale = ctx.saved_tensors
    dx, dscale, dshift = _triton_modulate_backward_op(
        grad_output,
        x,
        scale,
        True,
        True,
        True,
    )
    return dx, dscale, dshift


_triton_modulate.register_autograd(
    _triton_modulate_backward,
    setup_context=_triton_modulate_setup_context,
)


# ---------------------------------------------------------------------------
# Fused LayerNorm + Modulate Triton kernel — computes
#   (x - mean) / sqrt(var + eps) * (1 + scale) + shift
# in a single kernel launch, eliminating the intermediate ln_out tensor
# from DRAM.  Only works for elementwise_affine=False LayerNorm (no
# learnable weight/bias), which is the case for all AdaLN variants in Flux.
#
# The backward kernel fuses the LN backward formula with the modulate
# backward, using deterministic sequential accumulation over S for
# d_scale / d_shift (same pattern as _modulate_bwd_kernel).
# ---------------------------------------------------------------------------

_FUSED_LN_MOD_MAX_H = 8192


@triton.jit
def _fused_ln_modulate_fwd_kernel(
    X_ptr,
    Scale_ptr,
    Shift_ptr,
    Out_ptr,
    Mean_ptr,
    Rstd_ptr,
    S,
    B,
    H,
    eps,
    stride_x_sb,
    stride_sc_b,
    BLOCK_H: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    b_idx = row % B
    offs_h = tl.arange(0, BLOCK_H)
    mask = offs_h < H

    x_off = row * stride_x_sb + offs_h
    sc_off = b_idx * stride_sc_b + offs_h

    x = tl.load(X_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / H

    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    x_hat = x_centered * rstd

    sc = tl.load(Scale_ptr + sc_off, mask=mask).to(tl.float32)
    sh = tl.load(Shift_ptr + sc_off, mask=mask).to(tl.float32)
    out = x_hat * (1.0 + sc) + sh

    tl.store(Out_ptr + x_off, out.to(OUT_DTYPE), mask=mask)
    tl.store(Mean_ptr + row, mean)
    tl.store(Rstd_ptr + row, rstd)


@triton.jit
def _fused_ln_modulate_bwd_dscale_dshift_kernel(
    Grad_ptr,
    X_ptr,
    Mean_ptr,
    Rstd_ptr,
    DScale_ptr,
    DShift_ptr,
    S,
    B,
    H,
    stride_g_sb,
    stride_x_sb,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < (B * H)
    x_bh = xindex
    x_b = xindex // H

    r_base = tl.arange(0, RBLOCK)[None, :]

    acc_dscale = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    acc_dshift = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    for r_off in range(0, S, RBLOCK):
        r_idx = r_off + r_base
        r_mask = r_idx < S
        mask = xmask & r_mask

        g_off = x_bh + (r_idx * B * H)
        x_off = x_bh + (r_idx * B * H)
        row = x_b + r_idx * B

        g = tl.load(Grad_ptr + g_off, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
        mean_val = tl.load(Mean_ptr + row, mask=mask, other=0.0)
        rstd_val = tl.load(Rstd_ptr + row, mask=mask, other=0.0)

        x_hat = (x - mean_val) * rstd_val
        acc_dscale = tl.where(mask, acc_dscale + g * x_hat, acc_dscale)
        acc_dshift = tl.where(mask, acc_dshift + g, acc_dshift)

    dscale_val = tl.sum(acc_dscale, 1)[:, None]
    dshift_val = tl.sum(acc_dshift, 1)[:, None]
    tl.store(DScale_ptr + x_bh, dscale_val.to(OUT_DTYPE), mask=xmask)
    tl.store(DShift_ptr + x_bh, dshift_val.to(OUT_DTYPE), mask=xmask)


@triton.jit
def _fused_ln_modulate_bwd_dx_kernel(
    Grad_ptr,
    X_ptr,
    Mean_ptr,
    Rstd_ptr,
    Scale_ptr,
    DX_ptr,
    S,
    B,
    H,
    stride_g_sb,
    stride_x_sb,
    stride_sc_b,
    BLOCK_H: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    sb_idx = tl.program_id(0)
    b_idx = sb_idx % B
    offs_h = tl.arange(0, BLOCK_H)
    mask = offs_h < H

    sc_off = b_idx * stride_sc_b + offs_h
    scale_val = tl.load(Scale_ptr + sc_off, mask=mask).to(tl.float32)
    one_plus_scale = 1.0 + scale_val

    g_off = sb_idx * stride_g_sb + offs_h
    x_off = sb_idx * stride_x_sb + offs_h

    g = tl.load(Grad_ptr + g_off, mask=mask).to(tl.float32)
    x = tl.load(X_ptr + x_off, mask=mask).to(tl.float32)
    mean = tl.load(Mean_ptr + sb_idx)
    rstd = tl.load(Rstd_ptr + sb_idx)

    x_hat = (x - mean) * rstd
    d_x_hat = g * one_plus_scale

    inv_H = 1.0 / H
    d_x_hat_masked = tl.where(mask, d_x_hat, 0.0)
    xhat_masked = tl.where(mask, x_hat, 0.0)
    c1 = tl.sum(xhat_masked * d_x_hat_masked, axis=0) * inv_H
    c2 = tl.sum(d_x_hat_masked, axis=0) * inv_H
    d_x = rstd * (d_x_hat - c2 - x_hat * c1)

    tl.store(DX_ptr + x_off, d_x.to(OUT_DTYPE), mask=mask)


@_custom_op("primus::fused_ln_modulate", mutates_args=(), device_types="cuda")
def _opaque_fused_ln_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 3 or scale.dim() != 2 or x.shape[-1] > _FUSED_LN_MOD_MAX_H:
        out, mean, rstd = torch.ops.aten.native_layer_norm(
            x,
            [x.shape[-1]],
            None,
            None,
            eps,
        )
        out = out * (1 + scale) + shift
        return out, mean.flatten(), rstd.flatten()

    x = x.contiguous()
    scale, shift = scale.contiguous(), shift.contiguous()
    S, B, H = x.shape
    out = torch.empty_like(x)
    M = S * B
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK_H = triton.next_power_of_2(H)
    out_dtype = _TORCH_TO_TRITON_DTYPE[x.dtype]
    _fused_ln_modulate_fwd_kernel[(M,)](
        x,
        scale,
        shift,
        out,
        mean,
        rstd,
        S,
        B,
        H,
        eps,
        x.stride(1),
        scale.stride(0),
        BLOCK_H=BLOCK_H,
        OUT_DTYPE=out_dtype,
    )
    return out, mean, rstd


@_opaque_fused_ln_modulate.register_fake
def _opaque_fused_ln_modulate_fake(x, scale, shift, eps):
    out = torch.empty_like(x)
    M = x.numel() // x.shape[-1]
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    return out, mean, rstd


@_custom_op("primus::fused_ln_modulate_backward", mutates_args=(), device_types="cuda")
def _opaque_fused_ln_modulate_backward_op(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 3 or scale.dim() != 2 or x.shape[-1] > _FUSED_LN_MOD_MAX_H:
        grad_output = grad_output.to(x.dtype)
        mean_r = mean.view(*x.shape[:-1], 1)
        rstd_r = rstd.view(*x.shape[:-1], 1)
        x_hat = (x.float() - mean_r) * rstd_r
        d_x_hat = grad_output.float() * (1 + scale.float())
        c1 = (x_hat * d_x_hat).mean(dim=-1, keepdim=True)
        c2 = d_x_hat.mean(dim=-1, keepdim=True)
        d_x = (rstd_r * (d_x_hat - c2 - x_hat * c1)).to(x.dtype)
        reduce_dims = list(range(grad_output.dim() - scale.dim()))
        dscale = (grad_output.float() * x_hat).sum(dim=reduce_dims).to(scale.dtype)
        dshift = grad_output.sum(dim=reduce_dims).to(scale.dtype)
        return d_x, dscale, dshift

    grad_output = grad_output.to(x.dtype).contiguous()
    x = x.contiguous()
    scale = scale.contiguous()
    S, B, H = x.shape
    dx = torch.empty_like(x)
    dscale = torch.empty_like(scale)
    dshift = torch.empty_like(scale)
    BLOCK_H = triton.next_power_of_2(H)
    out_dtype = _TORCH_TO_TRITON_DTYPE[x.dtype]
    XBLOCK, RBLOCK = 256, 8
    _fused_ln_modulate_bwd_dscale_dshift_kernel[(triton.cdiv(B * H, XBLOCK),)](
        grad_output,
        x,
        mean,
        rstd,
        dscale,
        dshift,
        S,
        B,
        H,
        grad_output.stride(1),
        x.stride(1),
        XBLOCK=XBLOCK,
        RBLOCK=RBLOCK,
        OUT_DTYPE=out_dtype,
    )
    _fused_ln_modulate_bwd_dx_kernel[(S * B,)](
        grad_output,
        x,
        mean,
        rstd,
        scale,
        dx,
        S,
        B,
        H,
        grad_output.stride(1),
        x.stride(1),
        scale.stride(0),
        BLOCK_H=BLOCK_H,
        OUT_DTYPE=out_dtype,
    )
    return dx, dscale, dshift


@_opaque_fused_ln_modulate_backward_op.register_fake
def _opaque_fused_ln_modulate_backward_fake(grad_output, x, mean, rstd, scale):
    dx = torch.empty_like(x)
    dscale = torch.empty_like(scale)
    dshift = torch.empty_like(scale)
    return dx, dscale, dshift


def _opaque_fused_ln_modulate_setup_context(ctx, inputs, output):
    x, scale, _shift, _eps = inputs
    _out, mean, rstd = output
    ctx.save_for_backward(x, mean, rstd, scale)


def _opaque_fused_ln_modulate_backward(ctx, grad_output, _grad_mean, _grad_rstd):
    x, mean, rstd, scale = ctx.saved_tensors
    dx, dscale, dshift = _opaque_fused_ln_modulate_backward_op(
        grad_output,
        x,
        mean,
        rstd,
        scale,
    )
    return dx, dscale, dshift, None


_opaque_fused_ln_modulate.register_autograd(
    _opaque_fused_ln_modulate_backward,
    setup_context=_opaque_fused_ln_modulate_setup_context,
)


# ---------------------------------------------------------------------------
# @triton_op fused LN+modulate — transparent to Inductor.
# ---------------------------------------------------------------------------


@triton_op("primus::fused_ln_modulate_v2", mutates_args=())
def _triton_fused_ln_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 3 or scale.dim() != 2 or x.shape[-1] > _FUSED_LN_MOD_MAX_H:
        out, mean, rstd = torch.ops.aten.native_layer_norm(
            x,
            [x.shape[-1]],
            None,
            None,
            eps,
        )
        out = out * (1 + scale) + shift
        return out, mean.flatten(), rstd.flatten()

    x = x.contiguous()
    scale, shift = scale.contiguous(), shift.contiguous()
    S, B, H = x.shape
    out = torch.empty_like(x)
    M = S * B
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK_H = triton.next_power_of_2(H)
    out_dtype = _TORCH_TO_TRITON_DTYPE[x.dtype]
    wrap_triton(_fused_ln_modulate_fwd_kernel)[(M,)](
        x,
        scale,
        shift,
        out,
        mean,
        rstd,
        S,
        B,
        H,
        eps,
        x.stride(1),
        scale.stride(0),
        BLOCK_H=BLOCK_H,
        OUT_DTYPE=out_dtype,
    )
    return out, mean, rstd


@_triton_fused_ln_modulate.register_fake
def _triton_fused_ln_modulate_fake(x, scale, shift, eps):
    out = torch.empty_like(x)
    M = x.numel() // x.shape[-1]
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    return out, mean, rstd


@triton_op("primus::fused_ln_modulate_backward_v2", mutates_args=())
def _triton_fused_ln_modulate_backward_op(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 3 or scale.dim() != 2 or x.shape[-1] > _FUSED_LN_MOD_MAX_H:
        grad_output = grad_output.to(x.dtype)
        mean_r = mean.view(*x.shape[:-1], 1)
        rstd_r = rstd.view(*x.shape[:-1], 1)
        x_hat = (x.float() - mean_r) * rstd_r
        d_x_hat = grad_output.float() * (1 + scale.float())
        c1 = (x_hat * d_x_hat).mean(dim=-1, keepdim=True)
        c2 = d_x_hat.mean(dim=-1, keepdim=True)
        d_x = (rstd_r * (d_x_hat - c2 - x_hat * c1)).to(x.dtype)
        reduce_dims = list(range(grad_output.dim() - scale.dim()))
        dscale = (grad_output.float() * x_hat).sum(dim=reduce_dims).to(scale.dtype)
        dshift = grad_output.sum(dim=reduce_dims).to(scale.dtype)
        return d_x, dscale, dshift

    grad_output = grad_output.to(x.dtype).contiguous()
    x = x.contiguous()
    scale = scale.contiguous()
    S, B, H = x.shape
    dx = torch.empty_like(x)
    dscale = torch.empty_like(scale)
    dshift = torch.empty_like(scale)
    BLOCK_H = triton.next_power_of_2(H)
    out_dtype = _TORCH_TO_TRITON_DTYPE[x.dtype]
    XBLOCK, RBLOCK = 256, 8
    wrap_triton(_fused_ln_modulate_bwd_dscale_dshift_kernel)[(triton.cdiv(B * H, XBLOCK),)](
        grad_output,
        x,
        mean,
        rstd,
        dscale,
        dshift,
        S,
        B,
        H,
        grad_output.stride(1),
        x.stride(1),
        XBLOCK=XBLOCK,
        RBLOCK=RBLOCK,
        OUT_DTYPE=out_dtype,
    )
    wrap_triton(_fused_ln_modulate_bwd_dx_kernel)[(S * B,)](
        grad_output,
        x,
        mean,
        rstd,
        scale,
        dx,
        S,
        B,
        H,
        grad_output.stride(1),
        x.stride(1),
        scale.stride(0),
        BLOCK_H=BLOCK_H,
        OUT_DTYPE=out_dtype,
    )
    return dx, dscale, dshift


@_triton_fused_ln_modulate_backward_op.register_fake
def _triton_fused_ln_modulate_backward_fake(grad_output, x, mean, rstd, scale):
    dx = torch.empty_like(x)
    dscale = torch.empty_like(scale)
    dshift = torch.empty_like(scale)
    return dx, dscale, dshift


def _triton_fused_ln_modulate_setup_context(ctx, inputs, output):
    x, scale, _shift, _eps = inputs
    _out, mean, rstd = output
    ctx.save_for_backward(x, mean, rstd, scale)


def _triton_fused_ln_modulate_backward(ctx, grad_output, _grad_mean, _grad_rstd):
    x, mean, rstd, scale = ctx.saved_tensors
    dx, dscale, dshift = _triton_fused_ln_modulate_backward_op(
        grad_output,
        x,
        mean,
        rstd,
        scale,
    )
    return dx, dscale, dshift, None


_triton_fused_ln_modulate.register_autograd(
    _triton_fused_ln_modulate_backward,
    setup_context=_triton_fused_ln_modulate_setup_context,
)


def _fused_eager_ln_modulate(
    norm_module: nn.Module,
    x: Tensor,
    scale: Tensor,
    shift: Tensor,
    use_triton_ops: bool = False,
) -> Tensor:
    """Fused LayerNorm + modulate via a single Triton kernel.

    Computes norm(x) * (1 + scale) + shift in one pass, avoiding the
    intermediate ln_out write/read to DRAM.  Only valid for
    elementwise_affine=False norms (no learnable weight/bias).
    """
    _ln_mod_fn = _triton_fused_ln_modulate if use_triton_ops else _opaque_fused_ln_modulate
    out, _mean, _rstd = _ln_mod_fn(
        x,
        scale,
        shift,
        norm_module.eps,
    )
    return out


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Formula: RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight

    Args:
        hidden_size: Size of the normalized dimension
        config: Transformer configuration (for compatibility, not used)
        eps: Small constant for numerical stability (default: 1e-6)

    Reference:
        - "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
        - Adapted from NeMo's DiT implementation
    """

    def __init__(self, hidden_size: int, config=None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: Tensor) -> Tensor:
        """
        Compute RMS normalization.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor (before scaling)
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: Normalize input.

        Args:
            x: Input tensor [..., hidden_size]

        Returns:
            Normalized and scaled tensor [..., hidden_size]
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class AdaLN(MegatronModule):
    """
    Adaptive Layer Normalization for DiT (Diffusion Transformer).

    Conditions layer normalization on timestep embeddings via learned scale,
    shift, and gate parameters. Projects conditioning through SiLU + Linear
    into n_adaln_chunks modulation parameters.

    Args:
        config: Transformer configuration
        n_adaln_chunks: Number of modulation chunks (default: 9, i.e. (shift, scale, gate) x 3)
        norm: Normalization layer class (default: nn.LayerNorm)
        modulation_bias: Whether to use bias in modulation projection (default: False)
        use_second_norm: Whether to use a second normalization layer (default: False)
        init_method: Initialization function for the modulation projection
            weight (default: zero-init via ``nn.init.zeros_``). Pass
            ``nn.init.normal_`` only when matching NeMo's exact RNG draw
            sequence -- Flux's ``init_weights()`` re-zeroes these weights
            after construction, so the normal_ draw only serves to advance
            the CUDA RNG by the same amount as NeMo's init for
            cross-framework convergence comparison. Other consumers
            (test harnesses, downstream DiT models that reuse AdaLN) get
            the same observable zero-init either way and should leave
            this at the default.

    Reference:
        - "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
        - Adapted from NeMo's DiT implementation
    """

    def __init__(
        self,
        config: TransformerConfig,
        n_adaln_chunks: int = 9,
        norm: type = nn.LayerNorm,
        modulation_bias: bool = False,
        use_second_norm: bool = False,
        init_method=nn.init.zeros_,
    ):
        super().__init__(config)

        # Layer normalization (without affine parameters - scale/shift come from conditioning)
        self.ln = norm(config.hidden_size, elementwise_affine=False, eps=config.layernorm_epsilon)

        self.n_adaln_chunks = n_adaln_chunks

        # Modulation network: conditioning -> (scale, shift, gate, ...)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ColumnParallelLinear(
                config.hidden_size,
                n_adaln_chunks * config.hidden_size,
                config=config,
                init_method=init_method,
                bias=modulation_bias,
                gather_output=True,
            ),
        )

        self.use_second_norm = use_second_norm
        if use_second_norm:
            self.ln2 = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)

        # Mark weight as sequence parallel if needed
        setattr(self.adaLN_modulation[-1].weight, "sequence_parallel", config.sequence_parallel)

        self._adaln_plain_ops = getattr(config, "adaln_plain_ops", False)
        self._use_triton_ops = getattr(config, "use_triton_ops", False)

        if self._adaln_plain_ops:
            self.use_fused_ln_modulate = False
        else:
            self.use_fused_ln_modulate = True

        adaln_always_jit = getattr(config, "adaln_always_jit_fuser", False)
        if adaln_always_jit or not getattr(config, "enable_torch_compile", False):
            AdaLN._apply_jit_fuser()

    @classmethod
    def _apply_jit_fuser(cls):
        """Apply @jit_fuser to class methods once (shared across all instances)."""
        if getattr(cls, "_jit_fuser_applied", False):
            return
        cls.forward = jit_fuser(cls.forward)
        cls.modulate = jit_fuser(cls.modulate)
        cls.scale_add = jit_fuser(cls.scale_add)
        cls.modulated_layernorm = jit_fuser(cls.modulated_layernorm)
        cls.scaled_modulated_layernorm = jit_fuser(cls.scaled_modulated_layernorm)
        cls._jit_fuser_applied = True

    def forward(self, timestep_emb: Tensor) -> Tuple[Tensor, ...]:
        """
        Generate modulation parameters from timestep embedding.

        Args:
            timestep_emb: Timestep embeddings [B, hidden_size]

        Returns:
            Tuple of n_adaln_chunks tensors, each [B, hidden_size]
        """
        output, bias = self.adaLN_modulation(timestep_emb)
        if bias is not None:
            output = output + bias
        return output.chunk(self.n_adaln_chunks, dim=-1)

    def modulate(self, x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        """
        Apply adaptive modulation: x * (1 + scale) + shift.

        Args:
            x: Input tensor [B, ..., hidden_size]
            shift: Shift parameter [B, hidden_size]
            scale: Scale parameter [B, hidden_size]

        Returns:
            Modulated tensor [B, ..., hidden_size]
        """
        return x * (1 + scale) + shift

    def scale_add(self, residual: Tensor, x: Tensor, gate: Tensor) -> Tensor:
        """
        Gated residual addition: residual + gate * x.

        Args:
            residual: Residual connection [B, ..., hidden_size]
            x: Input to add [B, ..., hidden_size]
            gate: Gate parameter [B, hidden_size]

        Returns:
            Combined tensor [B, ..., hidden_size]
        """
        return residual + gate * x

    def modulated_layernorm(self, x: Tensor, shift: Tensor, scale: Tensor, layernorm_idx: int = 0) -> Tensor:
        """
        Apply layer normalization followed by adaptive modulation.

        Args:
            x: Input tensor [B, ..., hidden_size]
            shift: Shift parameter [B, hidden_size]
            scale: Scale parameter [B, hidden_size]
            layernorm_idx: Which layer norm to use (0 or 1, if use_second_norm=True)

        Returns:
            Normalized and modulated tensor [B, ..., hidden_size]
        """
        # Select appropriate layer norm
        if self.use_second_norm and layernorm_idx == 1:
            layernorm = self.ln2
        else:
            layernorm = self.ln

        if self._adaln_plain_ops:
            input_layernorm_output = layernorm(x).type_as(x)
            return self.modulate(input_layernorm_output, shift, scale)

        if self.use_fused_ln_modulate:
            return _fused_eager_ln_modulate(layernorm, x, scale, shift, self._use_triton_ops)

        input_layernorm_output = layernorm(x).type_as(x)
        return self.modulate(input_layernorm_output, shift, scale)

    def scaled_modulated_layernorm(
        self,
        residual: Tensor,
        x: Tensor,
        gate: Tensor,
        shift: Tensor,
        scale: Tensor,
        layernorm_idx: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Combined operation: gated residual addition + modulated layer norm.

        This is a common pattern in DiT: add gated residual, then normalize and modulate.

        Args:
            residual: Residual connection [B, ..., hidden_size]
            x: Input to add [B, ..., hidden_size]
            gate: Gate parameter [B, hidden_size]
            shift: Shift parameter [B, hidden_size]
            scale: Scale parameter [B, hidden_size]
            layernorm_idx: Which layer norm to use

        Returns:
            Tuple of (hidden_states, shifted_pre_mlp_layernorm_output)
        """
        # Gated residual addition
        hidden_states = self.scale_add(residual, x, gate)

        # Apply modulated layer normalization
        shifted_pre_mlp_layernorm_output = self.modulated_layernorm(
            hidden_states, shift, scale, layernorm_idx
        )

        return hidden_states, shifted_pre_mlp_layernorm_output


class AdaLNContinuous(MegatronModule):
    """
    Continuous Adaptive Layer Normalization for Flux.

    Simpler variant of AdaLN that only produces scale and shift (no gating).
    Formula: norm(x) * (1 + scale) + shift

    Args:
        config: Transformer configuration
        conditioning_embedding_dim: Dimension of conditioning embedding input
        modulation_bias: Whether to use bias in modulation layers (default: True)
        norm_type: 'layer_norm' (default: 'layer_norm')

    Note:
        Uses Megatron's sequence-first format [S, B, D].

    Reference:
        - Flux Paper: "Flux: A Scalable Diffusion Model"
        - Adapted from NeMo's DiT implementation
    """

    def __init__(
        self,
        config: TransformerConfig,
        conditioning_embedding_dim: int,
        modulation_bias: bool = True,
        norm_type: str = "layer_norm",
    ):
        super().__init__(config)

        # Modulation network: conditioning -> (scale, shift)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(conditioning_embedding_dim, config.hidden_size * 2, bias=modulation_bias),
        )

        # Normalization layer
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=False, eps=1e-6, bias=modulation_bias
            )
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

        self._adaln_plain_ops = getattr(config, "adaln_plain_ops", False)

        if self._adaln_plain_ops:
            self.use_fused_ln_modulate = False
        else:
            self.use_fused_ln_modulate = True

        self._use_triton_ops = getattr(config, "use_triton_ops", False)

    def forward(self, x: Tensor, conditioning_embedding: Tensor) -> Tensor:
        """
        Forward pass: Apply continuous adaptive normalization.

        Args:
            x: Input tensor [seq_len, B, hidden_size] (sequence-first format)
            conditioning_embedding: Conditioning signal [B, conditioning_embedding_dim]

        Returns:
            Normalized and modulated tensor [seq_len, B, hidden_size]
        """
        # Generate scale and shift from conditioning
        emb = self.adaLN_modulation(conditioning_embedding)  # [B, 2 * hidden_size]
        # NeMo convention: first half is scale, second half is shift.
        # NOTE: Checkpoints trained before this fix had (shift, scale) order and
        # need the two halves of norm_out.adaLN_modulation weight/bias swapped.
        scale, shift = torch.chunk(emb, 2, dim=1)

        if self._adaln_plain_ops:
            x = self.norm(x) * (1 + scale) + shift
            return x

        if self.use_fused_ln_modulate:
            return _fused_eager_ln_modulate(self.norm, x, scale, shift, self._use_triton_ops)

        ln_out = self.norm(x)
        _mod_fn = _triton_modulate if self._use_triton_ops else _opaque_modulate
        return _mod_fn(ln_out, scale, shift)
