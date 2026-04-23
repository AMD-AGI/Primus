"""Triton RMSNorm — single-pass kernel tuned for GPT-OSS-20B shapes.

Target sites (rank-0, mbs=2, S=8192):
  main_norm  (16384, 2880)   — TE: 57us, turbo: 78us  → target <50us
  q_norm     (1048576, 128)  — TE: 166us, turbo: 4255us → target <100us
  k_norm     (131072, 128)   — TE: 27us,  turbo: 504us  → target <30us

Two kernels:
  - rmsnorm_fwd_kernel_small_h: H<=2048, one warp per row (vectorized load)
  - rmsnorm_fwd_kernel_large_h: H>2048, one block per row (no two-scan; one-shot reduce)

Bwd: standard formulation
  grad_x  = (grad_out * gamma * rstd) - x * rstd^3 * mean(grad_out * gamma * x) / H
  grad_g  = sum_over_batch(grad_out * x * rstd)

We use a 2-stage bwd:
  stage_0 kernel: per-row dx + per-row partial dg (B, H)
  reduction:      .sum(dim=0) in PyTorch (cheap)
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
    X_ptr,        # *bf16  [B, H]
    G_ptr,        # *bf16  [H]
    Y_ptr,        # *bf16  [B, H]
    RSTD_ptr,     # *fp32  [B]   — saved for backward
    stride_xb,
    stride_yb,
    H: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,   # next-pow2 of H
):
    row = tl.program_id(0)
    x_ptrs = X_ptr + row * stride_xb + tl.arange(0, BLOCK_H)
    y_ptrs = Y_ptr + row * stride_yb + tl.arange(0, BLOCK_H)
    g_ptrs = G_ptr + tl.arange(0, BLOCK_H)
    mask = tl.arange(0, BLOCK_H) < H

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
    stride_yb,
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
    x_ptrs = X_ptr + row_offs[:, None] * stride_xb + h_offs[None, :]
    y_ptrs = Y_ptr + row_offs[:, None] * stride_yb + h_offs[None, :]
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=1) / H        # [ROWS_PER_BLOCK]
    rstd = tl.rsqrt(var + eps)
    y = (x * rstd[:, None] * g[None, :]).to(Y_ptr.dtype.element_ty)
    tl.store(y_ptrs, y, mask=full_mask)
    tl.store(RSTD_ptr + row_offs, rstd, mask=row_mask)


# ---------------------------------------------------------------------------
# Backward — stage 0: per-row dx + per-row partial dgamma
# ---------------------------------------------------------------------------

@triton.jit
def _rmsnorm_bwd_kernel(
    DY_ptr, X_ptr, G_ptr, RSTD_ptr, DX_ptr, DG_PART_ptr,
    stride_xb, stride_dyb, stride_dxb, stride_dgb,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    x_ptrs   = X_ptr   + row * stride_xb  + offs
    dy_ptrs  = DY_ptr  + row * stride_dyb + offs
    dx_ptrs  = DX_ptr  + row * stride_dxb + offs
    dgp_ptrs = DG_PART_ptr + row * stride_dgb + offs
    g_ptrs   = G_ptr + offs

    x    = tl.load(x_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy   = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
    g    = tl.load(g_ptrs,  mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row).to(tl.float32)

    x_hat = x * rstd
    dxhat = dy * g
    m = tl.sum(dxhat * x_hat, axis=0) / H
    dx = (dxhat - x_hat * m) * rstd
    dgp = dy * x_hat

    tl.store(dx_ptrs,  dx.to(DX_ptr.dtype.element_ty),  mask=mask)
    tl.store(dgp_ptrs, dgp, mask=mask)


@triton.jit
def _rmsnorm_bwd_kernel_multi_row(
    DY_ptr, X_ptr, G_ptr, RSTD_ptr, DX_ptr, DG_PART_ptr,
    stride_xb, stride_dyb, stride_dxb, stride_dgb,
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

    x_ptrs   = X_ptr   + row_offs[:, None] * stride_xb  + h_offs[None, :]
    dy_ptrs  = DY_ptr  + row_offs[:, None] * stride_dyb + h_offs[None, :]
    dx_ptrs  = DX_ptr  + row_offs[:, None] * stride_dxb + h_offs[None, :]
    dgp_ptrs = DG_PART_ptr + row_offs[:, None] * stride_dgb + h_offs[None, :]
    g_ptrs   = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    x   = tl.load(x_ptrs,  mask=full_mask, other=0.0).to(tl.float32)
    dy  = tl.load(dy_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g   = tl.load(g_ptrs,  mask=h_mask,    other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)

    x_hat = x * rstd[:, None]
    dxhat = dy * g[None, :]
    m = tl.sum(dxhat * x_hat, axis=1) / H        # [ROWS_PER_BLOCK]
    dx = (dxhat - x_hat * m[:, None]) * rstd[:, None]
    dgp = dy * x_hat

    tl.store(dx_ptrs,  dx.to(DX_ptr.dtype.element_ty),  mask=full_mask)
    tl.store(dgp_ptrs, dgp, mask=full_mask)


# ---------------------------------------------------------------------------
# Python autograd wrapper
# ---------------------------------------------------------------------------

def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


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
        x2 = x.reshape(-1, H).contiguous()
        B = x2.shape[0]
        y = torch.empty_like(x2)
        rstd = torch.empty(B, device=x.device, dtype=torch.float32)
        BLOCK_H, ROWS, num_warps, num_stages = _pick_config(H, B)
        if ROWS == 1:
            _rmsnorm_fwd_kernel[(B,)](
                x2, gamma, y, rstd,
                x2.stride(0), y.stride(0),
                H=H, eps=eps, BLOCK_H=BLOCK_H,
                num_warps=num_warps, num_stages=num_stages,
            )
        else:
            grid = ((B + ROWS - 1) // ROWS,)
            _rmsnorm_fwd_kernel_multi_row[grid](
                x2, gamma, y, rstd,
                x2.stride(0), y.stride(0),
                B=B, H=H, eps=eps,
                BLOCK_H=BLOCK_H, ROWS_PER_BLOCK=ROWS,
                num_warps=num_warps, num_stages=num_stages,
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
        dy2 = dy.reshape(-1, H).contiguous()
        dx = torch.empty_like(x2)
        dg_partial = torch.empty(B, H, device=x2.device, dtype=torch.float32)
        if ctx.ROWS == 1:
            _rmsnorm_bwd_kernel[(B,)](
                dy2, x2, gamma, rstd, dx, dg_partial,
                x2.stride(0), dy2.stride(0), dx.stride(0), dg_partial.stride(0),
                H=H, BLOCK_H=ctx.BLOCK_H,
                num_warps=ctx.num_warps, num_stages=ctx.num_stages,
            )
        else:
            grid = ((B + ctx.ROWS - 1) // ctx.ROWS,)
            _rmsnorm_bwd_kernel_multi_row[grid](
                dy2, x2, gamma, rstd, dx, dg_partial,
                x2.stride(0), dy2.stride(0), dx.stride(0), dg_partial.stride(0),
                B=B, H=H, BLOCK_H=ctx.BLOCK_H, ROWS_PER_BLOCK=ctx.ROWS,
                num_warps=ctx.num_warps, num_stages=ctx.num_stages,
            )
        dg = dg_partial.sum(dim=0).to(gamma.dtype)
        return dx.reshape(ctx.orig_shape), dg, None


def triton_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return TritonRMSNormFn.apply(x, gamma, eps)
