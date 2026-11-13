###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""q/k RMSNorm + RoPE straight from the fused ``linear_qkv`` output, bit for bit.

Upstream splits ``mixed_qkv`` ``[sq, b, ng, (np/ng + 2) * hn]`` into strided q/k/v
views, copies q to ``[sq, b, np, hn]`` (its heads are not uniformly strided),
runs the Turbo RMSNorm on q and k, then RoPE on each; the backward runs RoPE
backward and RMSNorm backward, and ``SplitWithSizesBackward`` concatenates
dq/dk/dv back into a fresh ``dmixed_qkv``. Per MSA layer that is a 64 MB copy in
the forward, a 75 MB cat in the backward and eight launches more than needed.

Here the norm kernels address the rows of ``mixed_qkv`` in place (row ``r`` of
q is head ``r % (np/ng)`` of token-group ``r // (np/ng)``), so the q copy goes,
and the backward writes dx straight into its slice of ``dmixed_qkv``, so the
cat goes. Everything else is the Turbo / Primus code unchanged:

* the RMSNorm kernels are Turbo's ``rmsnorm_{fwd,bwd}_kernel_multi_row``
  verbatim except for the row addressing, launched with Turbo's own configs
  (``_pick_config`` / ``_pick_bwd_config``) and dgamma reduced by Turbo's
  ``_finalize_dgamma`` -- same tiles, same reductions, same partials;
* RoPE uses the Primus ``_FusedRope`` kernels (rope.py) on the normed q/k.

``PRIMUS_FUSED_QK_NORM_ROPE=0`` falls back to the upstream path.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("PRIMUS_FUSED_QK_NORM_ROPE", "1") != "0"
# rope inside the norm kernels (one pass each way) instead of separate rope launches
FUSE_ROPE = os.environ.get("PRIMUS_QK_NORM_ROPE_ONE_PASS", "1") != "0"


@triton.jit
def _norm_fwd_rows(
    X_ptr, G_ptr, Y_ptr, RSTD_ptr,
    x_off, HPG, stride_grp, stride_xh, stride_yb, stride_yh,
    B, H: tl.constexpr, eps, BLOCK_H: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr,
    D: tl.constexpr,
):
    # rmsnorm_fwd_kernel_multi_row with x row r at x_off + (r // HPG) * stride_grp + (r % HPG) * D
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B

    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    x_rows = x_off + (row_offs // HPG).to(tl.int64) * stride_grp + (row_offs % HPG) * D
    x_ptrs = X_ptr + x_rows[:, None] + h_offs[None, :] * stride_xh
    y_ptrs = Y_ptr + row_offs[:, None].to(tl.int64) * stride_yb + h_offs[None, :] * stride_yh
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=1) / H
    rstd = tl.rsqrt(var + eps)
    y = (x * rstd[:, None] * g[None, :]).to(Y_ptr.dtype.element_ty)
    tl.store(y_ptrs, y, mask=full_mask)
    tl.store(RSTD_ptr + row_offs, rstd, mask=row_mask)


@triton.jit
def _norm_rope_fwd_rows(
    X_ptr, G_ptr, Y_ptr, RSTD_ptr, COS, SIN,
    x_off, HPG, stride_grp, ROWS_PER_S,
    B, H: tl.constexpr, eps, BLOCK_H: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr,
    D: tl.constexpr, ROT: tl.constexpr,
):
    # _norm_fwd_rows, then rope.py's _fwd on the normed rows (y = norm output in bf16)
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B

    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    x_rows = x_off + (row_offs // HPG).to(tl.int64) * stride_grp + (row_offs % HPG) * D
    x_ptrs = X_ptr + x_rows[:, None] + h_offs[None, :]
    y_ptrs = Y_ptr + row_offs[:, None].to(tl.int64) * D + h_offs[None, :]
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=1) / H
    rstd = tl.rsqrt(var + eps)
    y = (x * rstd[:, None] * g[None, :]).to(tl.bfloat16)
    tl.store(RSTD_ptr + row_offs, rstd, mask=row_mask)

    HALF: tl.constexpr = ROT // 2
    d = h_offs
    in_rot = d < ROT
    partner = tl.where(d < HALF, d + HALF, d - HALF)
    partner = tl.where(in_rot, partner, d)
    xp = tl.load(X_ptr + x_rows[:, None] + partner[None, :], mask=full_mask, other=0.0).to(tl.float32)
    gp = tl.load(G_ptr + partner, mask=h_mask, other=0.0).to(tl.float32)
    yp = (xp * rstd[:, None] * gp[None, :]).to(tl.bfloat16).to(tl.float32)
    s = (row_offs // ROWS_PER_S).to(tl.int64)
    c = tl.load(COS + s[:, None] * ROT + d[None, :], mask=row_mask[:, None] & in_rot[None, :], other=0.0).to(tl.float32)
    sn = tl.load(SIN + s[:, None] * ROT + d[None, :], mask=row_mask[:, None] & in_rot[None, :], other=0.0).to(tl.float32)
    sign = tl.where(d < HALF, -1.0, 1.0)
    a = (y.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
    r = ((sign[None, :] * yp) * sn).to(tl.bfloat16).to(tl.float32)
    o = (a + r).to(tl.bfloat16)
    o = tl.where(in_rot[None, :], o, y)
    tl.store(y_ptrs, o.to(Y_ptr.dtype.element_ty), mask=full_mask)


@triton.jit
def _rope_norm_bwd_rows(
    DQ_ptr, X_ptr, G_ptr, RSTD_ptr, DX_ptr, DG_PART_ptr, COS, SIN,
    x_off, HPG, stride_grp, ROWS_PER_S, stride_dgp,
    B, H: tl.constexpr, BLOCK_H: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr, D: tl.constexpr,
    ROT: tl.constexpr,
):
    # rope.py's _bwd on the incoming gradient rows, then _norm_bwd_rows
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    x_rows = x_off + (row_offs // HPG).to(tl.int64) * stride_grp + (row_offs % HPG) * D
    x_ptrs = X_ptr + x_rows[:, None] + h_offs[None, :]
    dq_base = DQ_ptr + row_offs[:, None].to(tl.int64) * D
    dx_ptrs = DX_ptr + x_rows[:, None] + h_offs[None, :]
    dgp_ptrs = DG_PART_ptr + pid * stride_dgp + h_offs
    g_ptrs = G_ptr + h_offs
    full_mask = row_mask[:, None] & h_mask[None, :]

    HALF: tl.constexpr = ROT // 2
    d = h_offs
    in_rot = d < ROT
    partner = tl.where(d < HALF, d + HALF, d - HALF)
    partner = tl.where(in_rot, partner, d)
    gq = tl.load(dq_base + d[None, :], mask=full_mask, other=0.0)
    gqp = tl.load(dq_base + partner[None, :], mask=full_mask, other=0.0).to(tl.float32)
    s = (row_offs // ROWS_PER_S).to(tl.int64)
    rm2 = row_mask[:, None] & in_rot[None, :]
    c = tl.load(COS + s[:, None] * ROT + d[None, :], mask=rm2, other=0.0).to(tl.float32)
    sp = tl.load(SIN + s[:, None] * ROT + partner[None, :], mask=rm2, other=0.0).to(tl.float32)
    sign = tl.where(d < HALF, 1.0, -1.0)
    a = (gq.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
    r = sign[None, :] * (gqp * sp).to(tl.bfloat16).to(tl.float32)
    dyb = (a + r).to(tl.bfloat16)
    dyb = tl.where(in_rot[None, :], dyb, gq.to(tl.bfloat16))

    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    dy = dyb.to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)

    x_hat = x * rstd[:, None]
    dxhat = dy * g[None, :]
    m = tl.sum(dxhat * x_hat, axis=1) / H
    dx = (dxhat - x_hat * m[:, None]) * rstd[:, None]

    tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=full_mask)

    dgp_block = (dy * x_hat) * row_mask[:, None].to(tl.float32)
    dgp_row = tl.sum(dgp_block, axis=0)
    tl.store(dgp_ptrs, dgp_row, mask=h_mask)


@triton.jit
def _norm_bwd_rows(
    DY_ptr, X_ptr, G_ptr, RSTD_ptr, DX_ptr, DG_PART_ptr,
    x_off, HPG, stride_grp, stride_xh, stride_dyb, stride_dyh, stride_dxh, stride_dgp,
    B, H: tl.constexpr, BLOCK_H: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr, D: tl.constexpr,
):
    # rmsnorm_bwd_kernel_multi_row with x and dx rows addressed as in _norm_fwd_rows
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    x_rows = x_off + (row_offs // HPG).to(tl.int64) * stride_grp + (row_offs % HPG) * D
    x_ptrs = X_ptr + x_rows[:, None] + h_offs[None, :] * stride_xh
    dy_ptrs = DY_ptr + row_offs[:, None].to(tl.int64) * stride_dyb + h_offs[None, :] * stride_dyh
    dx_ptrs = DX_ptr + x_rows[:, None] + h_offs[None, :] * stride_dxh
    dgp_ptrs = DG_PART_ptr + pid * stride_dgp + h_offs
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)

    x_hat = x * rstd[:, None]
    dxhat = dy * g[None, :]
    m = tl.sum(dxhat * x_hat, axis=1) / H
    dx = (dxhat - x_hat * m[:, None]) * rstd[:, None]

    tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=full_mask)

    dgp_block = (dy * x_hat) * row_mask[:, None].to(tl.float32)
    dgp_row = tl.sum(dgp_block, axis=0)
    tl.store(dgp_ptrs, dgp_row, mask=h_mask)


@triton.jit
def _copy_rows(SRC_ptr, DST_ptr, src_off, stride_src, stride_dst, dst_off, R, D: tl.constexpr,
               BR: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    d = tl.arange(0, D)
    m = (r < R)[:, None]
    v = tl.load(SRC_ptr + src_off + r[:, None].to(tl.int64) * stride_src + d[None, :], mask=m)
    tl.store(DST_ptr + dst_off + r[:, None].to(tl.int64) * stride_dst + d[None, :], v, mask=m)


def _turbo_cfg():
    from primus_turbo.pytorch.kernels.normalization import rmsnorm_impl as R

    return R


def _norm_fwd(mixed, x_off, hpg, n_rows, gamma, eps):
    """Normed rows ``[n_rows, D]`` and rstd, rows addressed inside ``mixed``."""
    R = _turbo_cfg()
    D = gamma.shape[0]
    y = torch.empty((n_rows, D), dtype=mixed.dtype, device=mixed.device)
    rstd = torch.empty(n_rows, device=mixed.device, dtype=torch.float32)
    BLOCK_H, ROWS, num_warps, num_stages = R._pick_config(D, n_rows)
    assert ROWS > 1 and BLOCK_H == D
    _norm_fwd_rows[((n_rows + ROWS - 1) // ROWS,)](
        mixed, gamma, y, rstd,
        x_off, hpg, mixed.shape[-1], 1, y.stride(0), y.stride(1),
        B=n_rows, H=D, eps=eps, BLOCK_H=BLOCK_H, ROWS_PER_BLOCK=ROWS, D=D,
        num_warps=num_warps, num_stages=num_stages,
    )
    return y, rstd


def _norm_bwd(dy, mixed, dmixed, x_off, hpg, n_rows, gamma, rstd):
    R = _turbo_cfg()
    D = gamma.shape[0]
    dy2 = dy.reshape(n_rows, D)
    mode, BLOCK_H, ROWS, num_warps, num_stages = R._pick_bwd_config(D, n_rows)
    assert mode == "multi" and BLOCK_H == D
    num_programs = (n_rows + ROWS - 1) // ROWS
    dg_partial = torch.empty(num_programs, D, device=mixed.device, dtype=torch.float32)
    _norm_bwd_rows[(num_programs,)](
        dy2, mixed, gamma, rstd, dmixed, dg_partial,
        x_off, hpg, mixed.shape[-1], 1, dy2.stride(0), dy2.stride(1), 1, dg_partial.stride(0),
        B=n_rows, H=D, BLOCK_H=BLOCK_H, ROWS_PER_BLOCK=ROWS, D=D,
        num_warps=num_warps, num_stages=num_stages,
    )
    return R._finalize_dgamma(dg_partial, gamma.dtype)


def _norm_rope_fwd(mixed, x_off, hpg, stride_grp, n_rows, rows_per_s, gamma, eps, cos_, sin_):
    """Normed + roped rows ``[n_rows, D]`` and rstd, in one pass."""
    R = _turbo_cfg()
    D = gamma.shape[0]
    out = torch.empty((n_rows, D), dtype=mixed.dtype, device=mixed.device)
    rstd = torch.empty(n_rows, device=mixed.device, dtype=torch.float32)
    BLOCK_H, ROWS, num_warps, num_stages = R._pick_config(D, n_rows)
    assert ROWS > 1 and BLOCK_H == D
    _norm_rope_fwd_rows[((n_rows + ROWS - 1) // ROWS,)](
        mixed, gamma, out, rstd, cos_, sin_,
        x_off, hpg, stride_grp, rows_per_s,
        B=n_rows, H=D, eps=eps, BLOCK_H=BLOCK_H, ROWS_PER_BLOCK=ROWS, D=D, ROT=cos_.shape[-1],
        num_warps=num_warps, num_stages=num_stages,
    )
    return out, rstd


def _rope_norm_bwd(dq, mixed, dmixed, x_off, hpg, stride_grp, n_rows, rows_per_s, gamma, rstd, cos_, sin_):
    R = _turbo_cfg()
    D = gamma.shape[0]
    mode, BLOCK_H, ROWS, num_warps, num_stages = R._pick_bwd_config(D, n_rows)
    assert mode == "multi" and BLOCK_H == D
    num_programs = (n_rows + ROWS - 1) // ROWS
    dg_partial = torch.empty(num_programs, D, device=mixed.device, dtype=torch.float32)
    _rope_norm_bwd_rows[(num_programs,)](
        dq, mixed, gamma, rstd, dmixed, dg_partial, cos_, sin_,
        x_off, hpg, stride_grp, rows_per_s, dg_partial.stride(0),
        B=n_rows, H=D, BLOCK_H=BLOCK_H, ROWS_PER_BLOCK=ROWS, D=D, ROT=cos_.shape[-1],
        num_warps=num_warps, num_stages=num_stages,
    )
    return R._finalize_dgamma(dg_partial, gamma.dtype)


# ---- q, k and v of one _QKNormRope in a single launch each way (PRIMUS_QK_NORM_ROPE_ONE_LAUNCH).
# Programs [0, NPQ) run the q rows, [NPQ, NPQ + NPK) the k rows, the rest copy v / dv; every
# program runs the same body (verbatim copies of the kernels above, pid passed in) with the
# same rows per program and config as the separate launches, so the outputs and the
# per-program dgamma partials are unchanged. The k / v launches alone are latency-bound
# (256 programs for 4 MB).
ONE_LAUNCH = os.environ.get("PRIMUS_QK_NORM_ROPE_ONE_LAUNCH", "1") != "0"


@triton.jit
def _norm_rope_fwd_body(pid, 
    X_ptr, G_ptr, Y_ptr, RSTD_ptr, COS, SIN,
    x_off, HPG, stride_grp, ROWS_PER_S,
    B, H: tl.constexpr, eps, BLOCK_H: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr,
    D: tl.constexpr, ROT: tl.constexpr,
):
    # _norm_fwd_rows, then rope.py's _fwd on the normed rows (y = norm output in bf16)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B

    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    x_rows = x_off + (row_offs // HPG).to(tl.int64) * stride_grp + (row_offs % HPG) * D
    x_ptrs = X_ptr + x_rows[:, None] + h_offs[None, :]
    y_ptrs = Y_ptr + row_offs[:, None].to(tl.int64) * D + h_offs[None, :]
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=1) / H
    rstd = tl.rsqrt(var + eps)
    y = (x * rstd[:, None] * g[None, :]).to(tl.bfloat16)
    tl.store(RSTD_ptr + row_offs, rstd, mask=row_mask)

    HALF: tl.constexpr = ROT // 2
    d = h_offs
    in_rot = d < ROT
    partner = tl.where(d < HALF, d + HALF, d - HALF)
    partner = tl.where(in_rot, partner, d)
    xp = tl.load(X_ptr + x_rows[:, None] + partner[None, :], mask=full_mask, other=0.0).to(tl.float32)
    gp = tl.load(G_ptr + partner, mask=h_mask, other=0.0).to(tl.float32)
    yp = (xp * rstd[:, None] * gp[None, :]).to(tl.bfloat16).to(tl.float32)
    s = (row_offs // ROWS_PER_S).to(tl.int64)
    c = tl.load(COS + s[:, None] * ROT + d[None, :], mask=row_mask[:, None] & in_rot[None, :], other=0.0).to(tl.float32)
    sn = tl.load(SIN + s[:, None] * ROT + d[None, :], mask=row_mask[:, None] & in_rot[None, :], other=0.0).to(tl.float32)
    sign = tl.where(d < HALF, -1.0, 1.0)
    a = (y.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
    r = ((sign[None, :] * yp) * sn).to(tl.bfloat16).to(tl.float32)
    o = (a + r).to(tl.bfloat16)
    o = tl.where(in_rot[None, :], o, y)
    tl.store(y_ptrs, o.to(Y_ptr.dtype.element_ty), mask=full_mask)


@triton.jit
def _rope_norm_bwd_body(pid, 
    DQ_ptr, X_ptr, G_ptr, RSTD_ptr, DX_ptr, DG_PART_ptr, COS, SIN,
    x_off, HPG, stride_grp, ROWS_PER_S, stride_dgp,
    B, H: tl.constexpr, BLOCK_H: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr, D: tl.constexpr,
    ROT: tl.constexpr,
):
    # rope.py's _bwd on the incoming gradient rows, then _norm_bwd_rows
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    x_rows = x_off + (row_offs // HPG).to(tl.int64) * stride_grp + (row_offs % HPG) * D
    x_ptrs = X_ptr + x_rows[:, None] + h_offs[None, :]
    dq_base = DQ_ptr + row_offs[:, None].to(tl.int64) * D
    dx_ptrs = DX_ptr + x_rows[:, None] + h_offs[None, :]
    dgp_ptrs = DG_PART_ptr + pid * stride_dgp + h_offs
    g_ptrs = G_ptr + h_offs
    full_mask = row_mask[:, None] & h_mask[None, :]

    HALF: tl.constexpr = ROT // 2
    d = h_offs
    in_rot = d < ROT
    partner = tl.where(d < HALF, d + HALF, d - HALF)
    partner = tl.where(in_rot, partner, d)
    gq = tl.load(dq_base + d[None, :], mask=full_mask, other=0.0)
    gqp = tl.load(dq_base + partner[None, :], mask=full_mask, other=0.0).to(tl.float32)
    s = (row_offs // ROWS_PER_S).to(tl.int64)
    rm2 = row_mask[:, None] & in_rot[None, :]
    c = tl.load(COS + s[:, None] * ROT + d[None, :], mask=rm2, other=0.0).to(tl.float32)
    sp = tl.load(SIN + s[:, None] * ROT + partner[None, :], mask=rm2, other=0.0).to(tl.float32)
    sign = tl.where(d < HALF, 1.0, -1.0)
    a = (gq.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
    r = sign[None, :] * (gqp * sp).to(tl.bfloat16).to(tl.float32)
    dyb = (a + r).to(tl.bfloat16)
    dyb = tl.where(in_rot[None, :], dyb, gq.to(tl.bfloat16))

    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    dy = dyb.to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)

    x_hat = x * rstd[:, None]
    dxhat = dy * g[None, :]
    m = tl.sum(dxhat * x_hat, axis=1) / H
    dx = (dxhat - x_hat * m[:, None]) * rstd[:, None]

    tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=full_mask)

    dgp_block = (dy * x_hat) * row_mask[:, None].to(tl.float32)
    dgp_row = tl.sum(dgp_block, axis=0)
    tl.store(dgp_ptrs, dgp_row, mask=h_mask)


@triton.jit
def _copy_rows_body(pid, SRC_ptr, DST_ptr, src_off, stride_src, stride_dst, dst_off, R, D: tl.constexpr,
               BR: tl.constexpr):
    r = pid * BR + tl.arange(0, BR)
    d = tl.arange(0, D)
    m = (r < R)[:, None]
    v = tl.load(SRC_ptr + src_off + r[:, None].to(tl.int64) * stride_src + d[None, :], mask=m)
    tl.store(DST_ptr + dst_off + r[:, None].to(tl.int64) * stride_dst + d[None, :], v, mask=m)


@triton.jit
def _qkv_norm_rope_fwd(
    X_ptr, GQ_ptr, GK_ptr, YQ_ptr, YK_ptr, RQ_ptr, RK_ptr, COS, SIN, V_ptr,
    k_off, HPG, stride_grp, RPS_Q, RPS_K, BQ, BK, v_off, NPQ, NPK,
    H: tl.constexpr, eps, BLOCK_H: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr,
    D: tl.constexpr, ROT: tl.constexpr, BR: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < NPQ:
        _norm_rope_fwd_body(pid, X_ptr, GQ_ptr, YQ_ptr, RQ_ptr, COS, SIN, 0, HPG, stride_grp, RPS_Q,
                            BQ, H, eps, BLOCK_H, ROWS_PER_BLOCK, D, ROT)
    elif pid < NPQ + NPK:
        _norm_rope_fwd_body(pid - NPQ, X_ptr, GK_ptr, YK_ptr, RK_ptr, COS, SIN, k_off, 1, stride_grp, RPS_K,
                            BK, H, eps, BLOCK_H, ROWS_PER_BLOCK, D, ROT)
    else:
        _copy_rows_body(pid - NPQ - NPK, X_ptr, V_ptr, v_off, stride_grp, D, 0, BK, D, BR)


@triton.jit
def _qkv_rope_norm_bwd(
    DQ_ptr, DK_ptr, X_ptr, GQ_ptr, GK_ptr, RQ_ptr, RK_ptr, DX_ptr, DGQ_ptr, DGK_ptr, COS, SIN, DV_ptr,
    k_off, HPG, stride_grp, RPS_Q, RPS_K, stride_dgq, stride_dgk, BQ, BK, v_off, NPQ, NPK,
    H: tl.constexpr, BLOCK_H: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr, D: tl.constexpr,
    ROT: tl.constexpr, BR: tl.constexpr, HAS_DV: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < NPQ:
        _rope_norm_bwd_body(pid, DQ_ptr, X_ptr, GQ_ptr, RQ_ptr, DX_ptr, DGQ_ptr, COS, SIN, 0, HPG, stride_grp,
                            RPS_Q, stride_dgq, BQ, H, BLOCK_H, ROWS_PER_BLOCK, D, ROT)
    elif pid < NPQ + NPK:
        _rope_norm_bwd_body(pid - NPQ, DK_ptr, X_ptr, GK_ptr, RK_ptr, DX_ptr, DGK_ptr, COS, SIN, k_off, 1,
                            stride_grp, RPS_K, stride_dgk, BK, H, BLOCK_H, ROWS_PER_BLOCK, D, ROT)
    else:
        if HAS_DV:
            _copy_rows_body(pid - NPQ - NPK, DV_ptr, DX_ptr, 0, D, stride_grp, v_off, BK, D, BR)
        else:
            r = (pid - NPQ - NPK) * BR + tl.arange(0, BR)
            d = tl.arange(0, D)
            m = (r < BK)[:, None]
            z = tl.zeros([BR, D], dtype=DX_ptr.dtype.element_ty)
            tl.store(DX_ptr + v_off + r[:, None].to(tl.int64) * stride_grp + d[None, :], z, mask=m)


def _one_launch_fwd(mixed, hpg, n_q, n_k, rps_q, rps_k, gq, gk, eps, cos_, sin_):
    R = _turbo_cfg()
    D = gq.shape[0]
    W = mixed.shape[-1]
    BLOCK_H, ROWS, nw, ns = R._pick_config(D, n_q)
    if R._pick_config(D, n_k) != (BLOCK_H, ROWS, nw, ns) or not (ROWS > 1 and BLOCK_H == D):
        return None
    q = torch.empty((n_q, D), dtype=mixed.dtype, device=mixed.device)
    k = torch.empty((n_k, D), dtype=mixed.dtype, device=mixed.device)
    v = torch.empty((n_k, D), dtype=mixed.dtype, device=mixed.device)
    rq = torch.empty(n_q, device=mixed.device, dtype=torch.float32)
    rk = torch.empty(n_k, device=mixed.device, dtype=torch.float32)
    npq, npk = triton.cdiv(n_q, ROWS), triton.cdiv(n_k, ROWS)
    BR = 64
    _qkv_norm_rope_fwd[(npq + npk + triton.cdiv(n_k, BR),)](
        mixed, gq, gk, q, k, rq, rk, cos_, sin_, v,
        hpg * D, hpg, W, rps_q, rps_k, n_q, n_k, (hpg + 1) * D, npq, npk,
        H=D, eps=eps, BLOCK_H=BLOCK_H, ROWS_PER_BLOCK=ROWS, D=D, ROT=cos_.shape[-1], BR=BR,
        num_warps=nw, num_stages=ns,
    )
    return q, k, v, rq, rk


def _one_launch_bwd(dq, dk, dv, mixed, dmixed, hpg, n_q, n_k, rps_q, rps_k, gq, gk, rq, rk, cos_, sin_):
    R = _turbo_cfg()
    D = gq.shape[0]
    W = mixed.shape[-1]
    mode, BLOCK_H, ROWS, nw, ns = R._pick_bwd_config(D, n_q)
    if R._pick_bwd_config(D, n_k) != (mode, BLOCK_H, ROWS, nw, ns) or not (mode == "multi" and BLOCK_H == D):
        return None
    npq, npk = triton.cdiv(n_q, ROWS), triton.cdiv(n_k, ROWS)
    dgq_p = torch.empty(npq, D, device=mixed.device, dtype=torch.float32)
    dgk_p = torch.empty(npk, D, device=mixed.device, dtype=torch.float32)
    BR = 64
    _qkv_rope_norm_bwd[(npq + npk + triton.cdiv(n_k, BR),)](
        dq, dk, mixed, gq, gk, rq, rk, dmixed, dgq_p, dgk_p, cos_, sin_, dq if dv is None else dv,
        hpg * D, hpg, W, rps_q, rps_k, dgq_p.stride(0), dgk_p.stride(0), n_q, n_k, (hpg + 1) * D, npq, npk,
        H=D, BLOCK_H=BLOCK_H, ROWS_PER_BLOCK=ROWS, D=D, ROT=cos_.shape[-1], BR=BR, HAS_DV=dv is not None,
        num_warps=nw, num_stages=ns,
    )
    return R._finalize_dgamma(dgq_p, gq.dtype), R._finalize_dgamma(dgk_p, gk.dtype)


class _QKNormRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixed, gq, gk, cos_, sin_, hpg, D, eps):
        from primus.backends.megatron.core.transformer.minimax_m3.rope import _get_rope_kernels

        S, Bt, NG, W = mixed.shape
        assert mixed.is_contiguous() and W == (hpg + 2) * D
        nq, nk = S * Bt * NG * hpg, S * Bt * NG
        fwd, _ = _get_rope_kernels()
        rot = cos_.shape[-1]
        ctx.one_pass = FUSE_ROPE and cos_.is_contiguous() and sin_.is_contiguous() and cos_.numel() == S * rot
        res = None
        if ctx.one_pass and ONE_LAUNCH:
            res = _one_launch_fwd(mixed, hpg, nq, nk, Bt * NG * hpg, Bt * NG, gq, gk, eps, cos_, sin_)
        if res is not None:
            q, k, v, rq, rk = res
            ctx.save_for_backward(mixed, gq, gk, rq, rk, cos_, sin_)
            ctx.hpg, ctx.D = hpg, D
            return q.view(S, Bt, NG * hpg, D), k.view(S, Bt, NG, D), v.view(S, Bt, NG, D)
        if ctx.one_pass:
            q, rq = _norm_rope_fwd(mixed, 0, hpg, W, nq, Bt * NG * hpg, gq, eps, cos_, sin_)
            k, rk = _norm_rope_fwd(mixed, hpg * D, 1, W, nk, Bt * NG, gk, eps, cos_, sin_)
            q = q.view(S, Bt, NG * hpg, D)
            k = k.view(S, Bt, NG, D)
            v = torch.empty((S, Bt, NG, D), dtype=mixed.dtype, device=mixed.device)
            BR = 64
            _copy_rows[(triton.cdiv(nk, BR),)](mixed, v, (hpg + 1) * D, W, D, 0, nk, D=D, BR=BR)
            ctx.save_for_backward(mixed, gq, gk, rq, rk, cos_, sin_)
            ctx.hpg, ctx.D = hpg, D
            return q, k, v
        yq, rq = _norm_fwd(mixed, 0, hpg, nq, gq, eps)
        yk, rk = _norm_fwd(mixed, hpg * D, 1, nk, gk, eps)
        q = torch.empty((S, Bt, NG * hpg, D), dtype=mixed.dtype, device=mixed.device)
        k = torch.empty((S, Bt, NG, D), dtype=mixed.dtype, device=mixed.device)
        yq4 = yq.view(S, Bt, NG * hpg, D)
        yk4 = yk.view(S, Bt, NG, D)
        fwd[(S * Bt,)](yq4, cos_, sin_, q, Bt, NG * hpg, yq4.stride(0), yq4.stride(1), yq4.stride(2),
                       ROT=rot, D=D, BH=triton.next_power_of_2(NG * hpg))
        fwd[(S * Bt,)](yk4, cos_, sin_, k, Bt, NG, yk4.stride(0), yk4.stride(1), yk4.stride(2),
                       ROT=rot, D=D, BH=triton.next_power_of_2(NG))
        v = torch.empty((S, Bt, NG, D), dtype=mixed.dtype, device=mixed.device)
        BR = 64
        _copy_rows[(triton.cdiv(nk, BR),)](mixed, v, (hpg + 1) * D, W, D, 0, nk, D=D, BR=BR)
        ctx.save_for_backward(mixed, gq, gk, rq, rk, cos_, sin_)
        ctx.hpg, ctx.D = hpg, D
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        from primus.backends.megatron.core.transformer.minimax_m3.rope import _get_rope_kernels

        mixed, gq, gk, rq, rk, cos_, sin_ = ctx.saved_tensors
        hpg, D = ctx.hpg, ctx.D
        S, Bt, NG, W = mixed.shape
        nq, nk = S * Bt * NG * hpg, S * Bt * NG
        _, bwd = _get_rope_kernels()
        rot = cos_.shape[-1]
        dmixed = torch.empty_like(mixed)
        dq = torch.zeros((S, Bt, NG * hpg, D), dtype=mixed.dtype, device=mixed.device) if dq is None else dq.contiguous()
        dk = torch.zeros((S, Bt, NG, D), dtype=mixed.dtype, device=mixed.device) if dk is None else dk.contiguous()
        if ctx.one_pass and ONE_LAUNCH:
            res = _one_launch_bwd(dq, dk, None if dv is None else dv.contiguous(), mixed, dmixed, hpg, nq, nk,
                                  Bt * NG * hpg, Bt * NG, gq, gk, rq, rk, cos_, sin_)
            if res is not None:
                return dmixed, res[0], res[1], None, None, None, None, None
        if ctx.one_pass:
            dgq = _rope_norm_bwd(dq, mixed, dmixed, 0, hpg, W, nq, Bt * NG * hpg, gq, rq, cos_, sin_)
            dgk = _rope_norm_bwd(dk, mixed, dmixed, hpg * D, 1, W, nk, Bt * NG, gk, rk, cos_, sin_)
        else:
            dgq, dgk = _QKNormRope._two_pass_bwd(dq, dk, mixed, dmixed, gq, gk, rq, rk, cos_, sin_, hpg, D)
        if dv is None:
            dmixed.view(S, Bt, NG, W)[..., (hpg + 1) * D:].zero_()
        else:
            dv = dv.contiguous()
            BR = 64
            _copy_rows[(triton.cdiv(nk, BR),)](dv, dmixed, 0, D, W, (hpg + 1) * D, nk, D=D, BR=BR)
        return dmixed, dgq, dgk, None, None, None, None, None

    @staticmethod
    def _two_pass_bwd(dq, dk, mixed, dmixed, gq, gk, rq, rk, cos_, sin_, hpg, D):
        from primus.backends.megatron.core.transformer.minimax_m3.rope import _get_rope_kernels

        S, Bt, NG, W = mixed.shape
        nq, nk = S * Bt * NG * hpg, S * Bt * NG
        _, bwd = _get_rope_kernels()
        rot = cos_.shape[-1]
        dyq = torch.empty_like(dq)
        bwd[(S * Bt,)](dq, cos_, sin_, dyq, Bt, NG * hpg, ROT=rot, D=D, BH=triton.next_power_of_2(NG * hpg))
        dgq = _norm_bwd(dyq, mixed, dmixed, 0, hpg, nq, gq, rq)
        dyk = torch.empty_like(dk)
        bwd[(S * Bt,)](dk, cos_, sin_, dyk, Bt, NG, ROT=rot, D=D, BH=triton.next_power_of_2(NG))
        dgk = _norm_bwd(dyk, mixed, dmixed, hpg * D, 1, nk, gk, rk)
        return dgq, dgk


def eligible(attn, mixed_qkv, rotary_interleaved) -> bool:
    if not ENABLED or rotary_interleaved:
        return False
    qn, kn = getattr(attn, "q_layernorm", None), getattr(attn, "k_layernorm", None)
    from primus.backends.megatron.core.extensions.primus_turbo import PrimusTurboRMSNorm

    if type(qn) is not PrimusTurboRMSNorm or type(kn) is not PrimusTurboRMSNorm:
        return False
    if qn.eps != kn.eps or qn.zero_centered_gamma != kn.zero_centered_gamma:
        return False
    D = attn.hidden_size_per_attention_head
    hpg = attn.num_attention_heads_per_partition // attn.num_query_groups_per_partition
    return (
        mixed_qkv.is_cuda and mixed_qkv.dtype == torch.bfloat16 and mixed_qkv.dim() == 4
        and mixed_qkv.is_contiguous() and mixed_qkv.shape[-1] == (hpg + 2) * D and D == 128
        and qn.weight.shape[0] == D and kn.weight.shape[0] == D
        and qn.weight.dtype == torch.bfloat16 and kn.weight.dtype == torch.bfloat16
    )


def qk_norm_rope(attn, mixed_qkv, rope_q, rope_k):
    """``(query, key, value)`` exactly as upstream's split + q/k norm + ``RopeTables.apply``:
    q ``[sq, b, np, hn]`` and k/v ``[sq, b, ng, hn]``, contiguous."""
    from primus.backends.megatron.core.extensions.step_weight_cache import plus_one

    qn, kn = attn.q_layernorm, attn.k_layernorm
    gq = plus_one(qn.weight) if qn.zero_centered_gamma else qn.weight
    gk = plus_one(kn.weight) if kn.zero_centered_gamma else kn.weight
    cq, sq_ = rope_q.cos_sin(mixed_qkv.dtype)
    ck, sk = rope_k.cos_sin(mixed_qkv.dtype)
    assert cq is ck and sq_ is sk, "q and k share the rope tables"
    hpg = attn.num_attention_heads_per_partition // attn.num_query_groups_per_partition
    return _QKNormRope.apply(mixed_qkv, gq, gk, cq, sq_, hpg, attn.hidden_size_per_attention_head, qn.eps)


class _NormRope(torch.autograd.Function):
    """Turbo RMSNorm then ``RopeTables.apply`` on a contiguous ``[sq, b, n, hn]`` tensor, one pass
    each way (the indexer's q/k)."""

    @staticmethod
    def forward(ctx, x, gamma, cos_, sin_, eps):
        S, Bt, N, D = x.shape
        n = S * Bt * N
        out, rstd = _norm_rope_fwd(x, 0, 1, D, n, Bt * N, gamma, eps, cos_, sin_)
        ctx.save_for_backward(x, gamma, rstd, cos_, sin_)
        return out.view(S, Bt, N, D)

    @staticmethod
    def backward(ctx, dout):
        x, gamma, rstd, cos_, sin_ = ctx.saved_tensors
        S, Bt, N, D = x.shape
        dx = torch.empty_like(x)
        dg = _rope_norm_bwd(dout.contiguous(), x, dx, 0, 1, D, S * Bt * N, Bt * N, gamma, rstd, cos_, sin_)
        return dx, dg, None, None, None


def norm_rope_ok(x, norm, rope, rotary_interleaved) -> bool:
    if not (ENABLED and FUSE_ROPE) or rotary_interleaved or rope is None:
        return False
    from primus.backends.megatron.core.extensions.primus_turbo import PrimusTurboRMSNorm

    if type(norm) is not PrimusTurboRMSNorm:
        return False
    S, D = x.shape[0], x.shape[-1]
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 4 and x.is_contiguous() and D == 128):
        return False
    if norm.weight.shape[0] != D or norm.weight.dtype != torch.bfloat16:
        return False
    cos_, sin_ = rope.cos_sin(x.dtype)
    rot = cos_.shape[-1]
    from primus_turbo.pytorch.kernels.normalization.rmsnorm_impl import _pick_config

    return (
        cos_.is_contiguous() and sin_.is_contiguous() and cos_.numel() == S * rot
        and rot % 2 == 0 and rot <= D and _pick_config(D, x.numel() // D)[1] > 1
    )


def norm_rope(x, norm, rope):
    """``rope.apply(norm(x))`` bitwise, for :func:`norm_rope_ok` inputs."""
    from primus.backends.megatron.core.extensions.step_weight_cache import plus_one

    gamma = plus_one(norm.weight) if norm.zero_centered_gamma else norm.weight
    cos_, sin_ = rope.cos_sin(x.dtype)
    return _NormRope.apply(x, gamma, cos_, sin_, norm.eps)
