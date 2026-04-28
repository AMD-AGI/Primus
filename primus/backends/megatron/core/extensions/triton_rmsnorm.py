###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton RMSNorm — single-pass kernels tuned for GPT-OSS-20B shapes.

Two flavours, each with single-row + multi-row variants:
  - rmsnorm_fwd_kernel              : y = norm(x)
  - rmsnorm_fwd_residual_kernel     : (y, x_plus_r) = norm(x + residual),
                                      x_plus_r exposed for next add

All kernels are stride-aware on both batch and hidden dims, so callers can
pass non-contiguous views (e.g. ``hidden_states.reshape(-1, H)`` on a
strided fp16/bf16 tensor) without forcing a ``.contiguous()`` copy. This is
critical for the bwd hot path, where ``_to_copy`` / ``direct_copy_kernel_cuda``
were eating ~20 ms / step on stream 0 (see
``slab/notes/2026-04/2026-04-24_gptoss_16_tier1_elementwise_tax_audit.md``).

Bwd: standard formulation
  grad_x = (grad_out * gamma * rstd) - x * rstd^3 * mean(grad_out * gamma * x) / H
  grad_g = sum_over_batch(grad_out * x * rstd)

For the residual variant the bwd additionally folds the gradient flowing
through ``x_plus_r`` (consumed by the next bda) into ``dx``; the autograd
function then returns the same gradient for both ``x`` and ``residual`` since
their sum has Jacobian ``[I, I]``.

We use a 2-stage bwd:
  stage_0 kernel: per-row dx + per-row partial dg (B, H)
  reduction:      .sum(dim=0) in PyTorch (cheap)

Public API (stable, used by ``primus_turbo.py`` and runtime install hooks):
  - ``triton_rmsnorm(x, gamma, eps)``
  - ``triton_rmsnorm_residual(x, residual, gamma, eps)``
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Forward kernel — one row per program. Used when H is large.
# ---------------------------------------------------------------------------


@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr,  # *bf16  [B, H]
    G_ptr,  # *bf16  [H]
    Y_ptr,  # *bf16  [B, H]
    RSTD_ptr,  # *fp32  [B]   — saved for backward
    stride_xb,
    stride_xh,
    stride_yb,
    stride_yh,
    H: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,  # next-pow2 of H
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    x_ptrs = X_ptr + row * stride_xb + offs * stride_xh
    y_ptrs = Y_ptr + row * stride_yb + offs * stride_yh
    g_ptrs = G_ptr + tl.arange(0, BLOCK_H)
    mask = offs < H

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    rstd = tl.rsqrt(var + eps)
    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    y = (x * rstd * g).to(Y_ptr.dtype.element_ty)
    tl.store(y_ptrs, y, mask=mask)
    tl.store(RSTD_ptr + row, rstd)


# ---------------------------------------------------------------------------
# Forward kernel — N rows per program (small H, huge B). Reduces grid size,
# critical for q_norm shape (1M, 128) where 1 row/program → 1M launches.
# ---------------------------------------------------------------------------


@triton.jit
def _rmsnorm_fwd_kernel_multi_row(
    X_ptr,
    G_ptr,
    Y_ptr,
    RSTD_ptr,
    stride_xb,
    stride_xh,
    stride_yb,
    stride_yh,
    B,
    H: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B

    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # 2D pointer arithmetic: [ROWS_PER_BLOCK, BLOCK_H]
    x_ptrs = X_ptr + row_offs[:, None] * stride_xb + h_offs[None, :] * stride_xh
    y_ptrs = Y_ptr + row_offs[:, None] * stride_yb + h_offs[None, :] * stride_yh
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=1) / H  # [ROWS_PER_BLOCK]
    rstd = tl.rsqrt(var + eps)
    y = (x * rstd[:, None] * g[None, :]).to(Y_ptr.dtype.element_ty)
    tl.store(y_ptrs, y, mask=full_mask)
    tl.store(RSTD_ptr + row_offs, rstd, mask=row_mask)


# ---------------------------------------------------------------------------
# Forward kernel — fused residual add. Computes
#     x_plus_r = (x + residual)
#     y        = rmsnorm(x_plus_r) * gamma
# in one pass and exposes ``x_plus_r`` for the next bda. Saves ``x_plus_r``
# (bf16) and ``rstd`` for backward.
# ---------------------------------------------------------------------------


@triton.jit
def _rmsnorm_fwd_residual_kernel(
    X_ptr,  # *bf16  [B, H]
    R_ptr,  # *bf16  [B, H]
    G_ptr,  # *bf16  [H]
    Y_ptr,  # *bf16  [B, H]
    XPR_ptr,  # *bf16  [B, H]   — x + residual, saved for fwd output + bwd
    RSTD_ptr,  # *fp32  [B]
    stride_xb,
    stride_xh,
    stride_rb,
    stride_rh,
    stride_yb,
    stride_yh,
    stride_xprb,
    stride_xprh,
    H: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    x_ptrs = X_ptr + row * stride_xb + offs * stride_xh
    r_ptrs = R_ptr + row * stride_rb + offs * stride_rh
    y_ptrs = Y_ptr + row * stride_yb + offs * stride_yh
    xpr_ptrs = XPR_ptr + row * stride_xprb + offs * stride_xprh
    g_ptrs = G_ptr + offs
    mask = offs < H

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptrs, mask=mask, other=0.0).to(tl.float32)
    xpr = x + r
    tl.store(xpr_ptrs, xpr.to(XPR_ptr.dtype.element_ty), mask=mask)

    var = tl.sum(xpr * xpr, axis=0) / H
    rstd = tl.rsqrt(var + eps)
    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    y = (xpr * rstd * g).to(Y_ptr.dtype.element_ty)
    tl.store(y_ptrs, y, mask=mask)
    tl.store(RSTD_ptr + row, rstd)


@triton.jit
def _rmsnorm_fwd_residual_kernel_multi_row(
    X_ptr,
    R_ptr,
    G_ptr,
    Y_ptr,
    XPR_ptr,
    RSTD_ptr,
    stride_xb,
    stride_xh,
    stride_rb,
    stride_rh,
    stride_yb,
    stride_yh,
    stride_xprb,
    stride_xprh,
    B,
    H: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B

    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    x_ptrs = X_ptr + row_offs[:, None] * stride_xb + h_offs[None, :] * stride_xh
    r_ptrs = R_ptr + row_offs[:, None] * stride_rb + h_offs[None, :] * stride_rh
    y_ptrs = Y_ptr + row_offs[:, None] * stride_yb + h_offs[None, :] * stride_yh
    xpr_ptrs = XPR_ptr + row_offs[:, None] * stride_xprb + h_offs[None, :] * stride_xprh
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)

    xpr = x + r
    tl.store(xpr_ptrs, xpr.to(XPR_ptr.dtype.element_ty), mask=full_mask)

    var = tl.sum(xpr * xpr, axis=1) / H
    rstd = tl.rsqrt(var + eps)
    y = (xpr * rstd[:, None] * g[None, :]).to(Y_ptr.dtype.element_ty)
    tl.store(y_ptrs, y, mask=full_mask)
    tl.store(RSTD_ptr + row_offs, rstd, mask=row_mask)


# ---------------------------------------------------------------------------
# Backward — stage 0: per-row dx + per-row partial dgamma
# ---------------------------------------------------------------------------


@triton.jit
def _rmsnorm_bwd_kernel(
    DY_ptr,
    X_ptr,
    G_ptr,
    RSTD_ptr,
    DX_ptr,
    DG_PART_ptr,
    stride_xb,
    stride_xh,
    stride_dyb,
    stride_dyh,
    stride_dxb,
    stride_dxh,
    stride_dgb,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    x_ptrs = X_ptr + row * stride_xb + offs * stride_xh
    dy_ptrs = DY_ptr + row * stride_dyb + offs * stride_dyh
    dx_ptrs = DX_ptr + row * stride_dxb + offs * stride_dxh
    dgp_ptrs = DG_PART_ptr + row * stride_dgb + offs
    g_ptrs = G_ptr + offs

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row).to(tl.float32)

    x_hat = x * rstd
    dxhat = dy * g
    m = tl.sum(dxhat * x_hat, axis=0) / H
    dx = (dxhat - x_hat * m) * rstd
    dgp = dy * x_hat

    tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=mask)
    tl.store(dgp_ptrs, dgp, mask=mask)


@triton.jit
def _rmsnorm_bwd_kernel_multi_row(
    DY_ptr,
    X_ptr,
    G_ptr,
    RSTD_ptr,
    DX_ptr,
    DG_PART_ptr,
    stride_xb,
    stride_xh,
    stride_dyb,
    stride_dyh,
    stride_dxb,
    stride_dxh,
    stride_dgb,
    B,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    x_ptrs = X_ptr + row_offs[:, None] * stride_xb + h_offs[None, :] * stride_xh
    dy_ptrs = DY_ptr + row_offs[:, None] * stride_dyb + h_offs[None, :] * stride_dyh
    dx_ptrs = DX_ptr + row_offs[:, None] * stride_dxb + h_offs[None, :] * stride_dxh
    dgp_ptrs = DG_PART_ptr + row_offs[:, None] * stride_dgb + h_offs[None, :]
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)

    x_hat = x * rstd[:, None]
    dxhat = dy * g[None, :]
    m = tl.sum(dxhat * x_hat, axis=1) / H  # [ROWS_PER_BLOCK]
    dx = (dxhat - x_hat * m[:, None]) * rstd[:, None]
    dgp = dy * x_hat

    tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=full_mask)
    tl.store(dgp_ptrs, dgp, mask=full_mask)


# ---------------------------------------------------------------------------
# Backward — fused residual variant. Adds the gradient that flows through
# ``x_plus_r`` (consumed by the next bda) to the standard RMSNorm dx.
# ---------------------------------------------------------------------------


@triton.jit
def _rmsnorm_bwd_residual_kernel(
    DY_ptr,  # *bf16  [B, H]   — grad of y
    DXPR_ptr,  # *bf16  [B, H]   — grad of x_plus_r (from next bda)
    XPR_ptr,  # *bf16  [B, H]   — saved x + residual
    G_ptr,  # *bf16  [H]
    RSTD_ptr,  # *fp32  [B]
    DX_ptr,  # *bf16  [B, H]   — dx = dresidual = norm-grad + dxpr
    DG_PART_ptr,  # *fp32  [B, H]
    stride_xprb,
    stride_xprh,
    stride_dyb,
    stride_dyh,
    stride_dxprb,
    stride_dxprh,
    stride_dxb,
    stride_dxh,
    stride_dgb,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    xpr_ptrs = XPR_ptr + row * stride_xprb + offs * stride_xprh
    dy_ptrs = DY_ptr + row * stride_dyb + offs * stride_dyh
    dxpr_ptrs = DXPR_ptr + row * stride_dxprb + offs * stride_dxprh
    dx_ptrs = DX_ptr + row * stride_dxb + offs * stride_dxh
    dgp_ptrs = DG_PART_ptr + row * stride_dgb + offs
    g_ptrs = G_ptr + offs

    xpr = tl.load(xpr_ptrs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
    dxpr = tl.load(dxpr_ptrs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row).to(tl.float32)

    x_hat = xpr * rstd
    dxhat = dy * g
    m = tl.sum(dxhat * x_hat, axis=0) / H
    dx_norm = (dxhat - x_hat * m) * rstd
    dx = dx_norm + dxpr
    dgp = dy * x_hat

    tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=mask)
    tl.store(dgp_ptrs, dgp, mask=mask)


@triton.jit
def _rmsnorm_bwd_residual_kernel_multi_row(
    DY_ptr,
    DXPR_ptr,
    XPR_ptr,
    G_ptr,
    RSTD_ptr,
    DX_ptr,
    DG_PART_ptr,
    stride_xprb,
    stride_xprh,
    stride_dyb,
    stride_dyh,
    stride_dxprb,
    stride_dxprh,
    stride_dxb,
    stride_dxh,
    stride_dgb,
    B,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    xpr_ptrs = XPR_ptr + row_offs[:, None] * stride_xprb + h_offs[None, :] * stride_xprh
    dy_ptrs = DY_ptr + row_offs[:, None] * stride_dyb + h_offs[None, :] * stride_dyh
    dxpr_ptrs = DXPR_ptr + row_offs[:, None] * stride_dxprb + h_offs[None, :] * stride_dxprh
    dx_ptrs = DX_ptr + row_offs[:, None] * stride_dxb + h_offs[None, :] * stride_dxh
    dgp_ptrs = DG_PART_ptr + row_offs[:, None] * stride_dgb + h_offs[None, :]
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    xpr = tl.load(xpr_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    dxpr = tl.load(dxpr_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)

    x_hat = xpr * rstd[:, None]
    dxhat = dy * g[None, :]
    m = tl.sum(dxhat * x_hat, axis=1) / H
    dx_norm = (dxhat - x_hat * m[:, None]) * rstd[:, None]
    dx = dx_norm + dxpr
    dgp = dy * x_hat

    tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=full_mask)
    tl.store(dgp_ptrs, dgp, mask=full_mask)


# ---------------------------------------------------------------------------
# Python autograd wrapper
# ---------------------------------------------------------------------------


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def _reshape_batch_hidden(x: torch.Tensor, H: int) -> torch.Tensor:
    """Flatten to [B, H] without forcing a contiguous copy.

    We keep original strides so Triton kernels can read/write strided rows
    directly, which avoids many explicit ``_to_copy`` / ``direct_copy``
    kernels in backward on stream 0.
    """
    if x.shape[-1] != H:
        raise ValueError(f"last dim mismatch: expected H={H}, got shape={tuple(x.shape)}")
    return x.reshape(-1, H)


def _pick_config(H: int, B: int) -> tuple[int, int, int, int]:
    """Return (BLOCK_H, ROWS_PER_BLOCK, num_warps, num_stages).

    Multi-row mode (ROWS_PER_BLOCK > 1) wins when H is small AND B is huge,
    because grid size B becomes the bottleneck (kernel launch / scheduling).
    """
    BLOCK_H = _next_pow2(H)
    # Multi-row threshold: small H, plenty of rows
    if BLOCK_H <= 256 and B >= 4096:
        # Pack 16 rows per block: total LDS use ≤ 256*16*4B = 16KiB, fits fine
        ROWS = 16 if BLOCK_H <= 128 else 8
        return BLOCK_H, ROWS, 4, 2
    # Single-row mode
    if BLOCK_H <= 256:
        return BLOCK_H, 1, 1, 1
    if BLOCK_H <= 1024:
        return BLOCK_H, 1, 4, 2
    if BLOCK_H <= 4096:
        return BLOCK_H, 1, 8, 2
    return BLOCK_H, 1, 16, 2


class TritonRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, eps: float):
        assert x.is_cuda and gamma.is_cuda
        orig_shape = x.shape
        H = gamma.shape[0]
        assert orig_shape[-1] == H
        x2 = _reshape_batch_hidden(x, H)
        B = x2.shape[0]
        y = torch.empty_like(x2)
        rstd = torch.empty(B, device=x.device, dtype=torch.float32)
        BLOCK_H, ROWS, num_warps, num_stages = _pick_config(H, B)
        if ROWS == 1:
            _rmsnorm_fwd_kernel[(B,)](
                x2,
                gamma,
                y,
                rstd,
                x2.stride(0),
                x2.stride(1),
                y.stride(0),
                y.stride(1),
                H=H,
                eps=eps,
                BLOCK_H=BLOCK_H,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            grid = ((B + ROWS - 1) // ROWS,)
            _rmsnorm_fwd_kernel_multi_row[grid](
                x2,
                gamma,
                y,
                rstd,
                x2.stride(0),
                x2.stride(1),
                y.stride(0),
                y.stride(1),
                B=B,
                H=H,
                eps=eps,
                BLOCK_H=BLOCK_H,
                ROWS_PER_BLOCK=ROWS,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        ctx.save_for_backward(x2, gamma, rstd)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        ctx.BLOCK_H = BLOCK_H
        ctx.ROWS = ROWS
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x2, gamma, rstd = ctx.saved_tensors
        H = gamma.shape[0]
        B = x2.shape[0]
        dy2 = _reshape_batch_hidden(dy, H)
        dx = torch.empty_like(x2)
        dg_partial = torch.empty(B, H, device=x2.device, dtype=torch.float32)
        if ctx.ROWS == 1:
            _rmsnorm_bwd_kernel[(B,)](
                dy2,
                x2,
                gamma,
                rstd,
                dx,
                dg_partial,
                x2.stride(0),
                x2.stride(1),
                dy2.stride(0),
                dy2.stride(1),
                dx.stride(0),
                dx.stride(1),
                dg_partial.stride(0),
                H=H,
                BLOCK_H=ctx.BLOCK_H,
                num_warps=ctx.num_warps,
                num_stages=ctx.num_stages,
            )
        else:
            grid = ((B + ctx.ROWS - 1) // ctx.ROWS,)
            _rmsnorm_bwd_kernel_multi_row[grid](
                dy2,
                x2,
                gamma,
                rstd,
                dx,
                dg_partial,
                x2.stride(0),
                x2.stride(1),
                dy2.stride(0),
                dy2.stride(1),
                dx.stride(0),
                dx.stride(1),
                dg_partial.stride(0),
                B=B,
                H=H,
                BLOCK_H=ctx.BLOCK_H,
                ROWS_PER_BLOCK=ctx.ROWS,
                num_warps=ctx.num_warps,
                num_stages=ctx.num_stages,
            )
        dg = dg_partial.sum(dim=0).to(gamma.dtype)
        return dx.reshape(ctx.orig_shape), dg, None


def triton_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return TritonRMSNormFn.apply(x, gamma, eps)


class TritonRMSNormResidualFn(torch.autograd.Function):
    """Fused (x + residual) → rmsnorm → (y, x_plus_r).

    Replaces the pattern
        h = bda(x, residual)            # ~42-ms vectorized_elementwise_kernel<add>
        y = rmsnorm(h)                  # PrimusTurboRMSNorm fwd
    with a single Triton kernel that computes both ``h`` and ``y`` from one
    load of ``x`` and one load of ``residual``. ``h`` is exposed as a second
    return so the caller can feed it to the *next* bda (the mlp_bda).

    Backward returns ``(dx, dresidual, dgamma, None)`` where
    ``dx == dresidual`` because the upstream ``+`` has Jacobian ``[I, I]``.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, eps: float):
        assert x.is_cuda and residual.is_cuda and gamma.is_cuda
        assert x.shape == residual.shape, (x.shape, residual.shape)
        orig_shape = x.shape
        H = gamma.shape[0]
        assert orig_shape[-1] == H

        x2 = _reshape_batch_hidden(x, H)
        r2 = _reshape_batch_hidden(residual, H)
        B = x2.shape[0]
        y = torch.empty_like(x2)
        x_plus_r = torch.empty_like(x2)
        rstd = torch.empty(B, device=x.device, dtype=torch.float32)

        BLOCK_H, ROWS, num_warps, num_stages = _pick_config(H, B)
        if ROWS == 1:
            _rmsnorm_fwd_residual_kernel[(B,)](
                x2,
                r2,
                gamma,
                y,
                x_plus_r,
                rstd,
                x2.stride(0),
                x2.stride(1),
                r2.stride(0),
                r2.stride(1),
                y.stride(0),
                y.stride(1),
                x_plus_r.stride(0),
                x_plus_r.stride(1),
                H=H,
                eps=eps,
                BLOCK_H=BLOCK_H,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            grid = ((B + ROWS - 1) // ROWS,)
            _rmsnorm_fwd_residual_kernel_multi_row[grid](
                x2,
                r2,
                gamma,
                y,
                x_plus_r,
                rstd,
                x2.stride(0),
                x2.stride(1),
                r2.stride(0),
                r2.stride(1),
                y.stride(0),
                y.stride(1),
                x_plus_r.stride(0),
                x_plus_r.stride(1),
                B=B,
                H=H,
                eps=eps,
                BLOCK_H=BLOCK_H,
                ROWS_PER_BLOCK=ROWS,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        ctx.save_for_backward(x_plus_r, gamma, rstd)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        ctx.BLOCK_H = BLOCK_H
        ctx.ROWS = ROWS
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return y.reshape(orig_shape), x_plus_r.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dxpr: torch.Tensor):
        x_plus_r, gamma, rstd = ctx.saved_tensors
        H = gamma.shape[0]
        B = x_plus_r.shape[0]

        dy2 = _reshape_batch_hidden(dy, H)
        # When the second output (x_plus_r) is consumed only by the *next*
        # bda's add and that add is folded into the next layer's RMSNorm via
        # the V2 carry, autograd hands us None for ``dxpr``. Guard with zeros
        # so the kernel always sees a valid pointer.
        if dxpr is None:
            dxpr2 = torch.zeros_like(x_plus_r)
        else:
            dxpr2 = _reshape_batch_hidden(dxpr, H)
        dx = torch.empty_like(x_plus_r)
        dg_partial = torch.empty(B, H, device=x_plus_r.device, dtype=torch.float32)

        if ctx.ROWS == 1:
            _rmsnorm_bwd_residual_kernel[(B,)](
                dy2,
                dxpr2,
                x_plus_r,
                gamma,
                rstd,
                dx,
                dg_partial,
                x_plus_r.stride(0),
                x_plus_r.stride(1),
                dy2.stride(0),
                dy2.stride(1),
                dxpr2.stride(0),
                dxpr2.stride(1),
                dx.stride(0),
                dx.stride(1),
                dg_partial.stride(0),
                H=H,
                BLOCK_H=ctx.BLOCK_H,
                num_warps=ctx.num_warps,
                num_stages=ctx.num_stages,
            )
        else:
            grid = ((B + ctx.ROWS - 1) // ctx.ROWS,)
            _rmsnorm_bwd_residual_kernel_multi_row[grid](
                dy2,
                dxpr2,
                x_plus_r,
                gamma,
                rstd,
                dx,
                dg_partial,
                x_plus_r.stride(0),
                x_plus_r.stride(1),
                dy2.stride(0),
                dy2.stride(1),
                dxpr2.stride(0),
                dxpr2.stride(1),
                dx.stride(0),
                dx.stride(1),
                dg_partial.stride(0),
                B=B,
                H=H,
                BLOCK_H=ctx.BLOCK_H,
                ROWS_PER_BLOCK=ctx.ROWS,
                num_warps=ctx.num_warps,
                num_stages=ctx.num_stages,
            )

        dg = dg_partial.sum(dim=0).to(gamma.dtype)
        # add() Jacobian [I, I] → both x and residual receive the same gradient
        dx_out = dx.reshape(ctx.orig_shape)
        return dx_out, dx_out, dg, None


def triton_rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
):
    """Compute ``(rmsnorm(x + residual) * gamma, x + residual)`` in one kernel.

    Returns a 2-tuple. Caller should pass the second element as the residual
    of the *next* bda, replacing the standalone bf16 add.
    """
    return TritonRMSNormResidualFn.apply(x, residual, gamma, eps)
