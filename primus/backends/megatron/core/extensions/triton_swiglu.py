###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton SwiGLU + probs — fused MoE expert activation, no cat/split.

Targets the bwd "elementwise tax" on stream 0 caused by the Inductor-fused
``triton_poi_fused__to_copy_cat_mul_silu_silu_backward_split_1`` kernel
that is generated from the default GroupedMLP composition::

    gate, up = torch.chunk(x, 2, dim=-1)
    out = F.silu(gate) * up * probs
    # bwd materialises ``cat([d_gate, d_up], dim=-1)`` into a fresh buffer
    # and then ``split`` it back, all on stream 0.

We replace this with a pair of Triton kernels that operate on a single
``[N, 2H]`` ``x`` tensor:

  - ``_swiglu_probs_fwd_kernel``: load ``gate = X[:, :H]`` and
    ``up = X[:, H:]`` separately (stride-aware), compute ``silu(gate) * up *
    probs`` in fp32, write ``Y[N, H]`` in the input dtype.
  - ``_swiglu_probs_bwd_kernel``: recompute ``sig``, ``silu``, ``d_silu`` and
    write ``d_gate`` / ``d_up`` directly into the same ``[N, 2H]`` ``DX``
    buffer (no temporary cat). Also accumulates ``d_probs[N]``.

Public API:
  - ``triton_swiglu_with_probs(x, probs)``  -> autograd-aware functional
  - ``SwiGLUWithProbsFn``                   -> raw torch.autograd.Function
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def _pick_config(h: int) -> tuple[int, int, int]:
    """Return ``(BLOCK_H, num_warps, num_stages)`` keyed on hidden dim."""
    block_h = _next_pow2(h)
    if block_h <= 256:
        return block_h, 4, 2
    if block_h <= 1024:
        return block_h, 8, 2
    return block_h, 8, 3


@triton.jit
def _swiglu_probs_fwd_kernel(
    X_ptr,
    P_ptr,
    Y_ptr,
    stride_xb,
    stride_xh,
    stride_pb,
    stride_yb,
    stride_yh,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    gate_ptrs = X_ptr + row * stride_xb + offs * stride_xh
    up_ptrs = X_ptr + row * stride_xb + (H + offs) * stride_xh
    y_ptrs = Y_ptr + row * stride_yb + offs * stride_yh

    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)
    prob = tl.load(P_ptr + row * stride_pb).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-gate))
    silu = gate * sig
    y = silu * up * prob
    tl.store(y_ptrs, y.to(Y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_probs_bwd_kernel(
    X_ptr,
    P_ptr,
    DY_ptr,
    DX_ptr,
    DP_ptr,
    stride_xb,
    stride_xh,
    stride_pb,
    stride_dyb,
    stride_dyh,
    stride_dxb,
    stride_dxh,
    stride_dpb,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    gate_ptrs = X_ptr + row * stride_xb + offs * stride_xh
    up_ptrs = X_ptr + row * stride_xb + (H + offs) * stride_xh
    dy_ptrs = DY_ptr + row * stride_dyb + offs * stride_dyh

    dx_gate_ptrs = DX_ptr + row * stride_dxb + offs * stride_dxh
    dx_up_ptrs = DX_ptr + row * stride_dxb + (H + offs) * stride_dxh

    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
    prob = tl.load(P_ptr + row * stride_pb).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-gate))
    silu = gate * sig
    d_silu = sig * (1.0 + gate * (1.0 - sig))
    grad_main = dy * prob
    d_gate = grad_main * up * d_silu
    d_up = grad_main * silu

    tl.store(dx_gate_ptrs, d_gate.to(DX_ptr.dtype.element_ty), mask=mask)
    tl.store(dx_up_ptrs, d_up.to(DX_ptr.dtype.element_ty), mask=mask)

    d_prob = tl.sum(dy * silu * up, axis=0)
    tl.store(DP_ptr + row * stride_dpb, d_prob.to(DP_ptr.dtype.element_ty))


class SwiGLUWithProbsFn(torch.autograd.Function):
    """Custom autograd that calls Triton fwd/bwd; CPU fallback for tests."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, probs: torch.Tensor):
        if x.ndim != 2:
            raise ValueError(f"expected x.ndim==2, got {x.ndim}")
        if x.shape[-1] % 2 != 0:
            raise ValueError(f"expected even hidden for GLU, got {x.shape[-1]}")

        if not x.is_cuda:
            h = x.shape[-1] // 2
            gate = x[:, :h]
            up = x[:, h:]
            p = probs if probs.ndim == 2 else probs.unsqueeze(-1)
            out = F.silu(gate) * up * p.to(x.dtype)
            ctx.save_for_backward(x, p)
            ctx.h = h
            ctx.use_cpu_fallback = True
            ctx.probs_shape = probs.shape
            ctx.probs_dtype = probs.dtype
            return out

        h = x.shape[-1] // 2
        n = x.shape[0]
        p = probs.reshape(-1).to(x.dtype)
        out = torch.empty((n, h), device=x.device, dtype=x.dtype)

        block_h, num_warps, num_stages = _pick_config(h)
        _swiglu_probs_fwd_kernel[(n,)](
            x,
            p,
            out,
            x.stride(0),
            x.stride(1),
            p.stride(0),
            out.stride(0),
            out.stride(1),
            H=h,
            BLOCK_H=block_h,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        ctx.save_for_backward(x, p)
        ctx.h = h
        ctx.block_h = block_h
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        ctx.probs_shape = probs.shape
        ctx.probs_dtype = probs.dtype
        ctx.use_cpu_fallback = False
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, p = ctx.saved_tensors
        h = ctx.h

        if ctx.use_cpu_fallback:
            gate = x[:, :h]
            up = x[:, h:]
            sig = torch.sigmoid(gate)
            silu = gate * sig
            d_silu = sig * (1 + gate * (1 - sig))
            grad_main = grad_out * p
            d_gate = grad_main * up * d_silu
            d_up = grad_main * silu
            dx = torch.empty_like(x)
            dx[:, :h] = d_gate
            dx[:, h:] = d_up
            d_probs = (grad_out * silu * up).sum(dim=-1, keepdim=True).to(ctx.probs_dtype)
            return dx, d_probs.view(ctx.probs_shape)

        n = x.shape[0]
        dx = torch.empty_like(x)
        dp = torch.empty((n,), device=x.device, dtype=x.dtype)

        _swiglu_probs_bwd_kernel[(n,)](
            x,
            p,
            grad_out,
            dx,
            dp,
            x.stride(0),
            x.stride(1),
            p.stride(0),
            grad_out.stride(0),
            grad_out.stride(1),
            dx.stride(0),
            dx.stride(1),
            dp.stride(0),
            H=h,
            BLOCK_H=ctx.block_h,
            num_warps=ctx.num_warps,
            num_stages=ctx.num_stages,
        )

        d_probs = dp.view(ctx.probs_shape).to(ctx.probs_dtype)
        return dx, d_probs


def triton_swiglu_with_probs(x: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Fused ``F.silu(gate) * up * probs`` for ``x = [gate || up]``.

    ``x`` has shape ``[N, 2H]`` (gate concatenated with up on the last dim).
    Returns ``[N, H]``. ``probs`` may be ``[N]`` or ``[N, 1]`` and is
    broadcast to ``[N, 1]`` internally. No ``torch.cat`` / ``torch.split``
    is materialised in either fwd or bwd.
    """
    return SwiGLUWithProbsFn.apply(x, probs)
