###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The V4 compressor tail (window pool -> kv_norm -> compress-base RoPE) as one autograd op.

Unfused (glue window pool with the d(ape) backward, glue RMSNorm, Triton RoPE), per compressor:
  fwd  _window_pool_fwd, _rmsnorm_fwd, _apply_rope_fwd                                               3 launches
  bwd  _apply_rope_bwd, _rmsnorm_bwd_part, _rmsnorm_dw_reduce, _window_pool_bwd_dape, _dape_reduce    5 launches
Here:
  fwd  _window_pool_fwd (unchanged), ``_comp_norm_rope_fwd_kernel`` (RMSNorm + RoPE of the rounded norm output)  2
  bwd  ``_comp_bwd_kernel``: per (window group, 64-column chunk) the RoPE adjoint of the incoming gradient, the
       RMSNorm backward of each row (row reduction recomputed per chunk program on the full row) feeding the pool
       backward directly, plus the RMSNorm dw partials of the chunk; ``_comp_reduce_kernel``: dw reduce and d(ape)
       reduce as one launch (disjoint program ranges)                                                           2

Bit-identical to the unfused chain: every reduction keeps the original kernel's tile shape / warp count (row sums on
a [D] row with 4 warps, pool reductions on [BLOCK_R, 64] tiles, dw partials in row order over the same row groups,
reduces on the original tiles), every elementwise expression is the original's (bf16 RoPE products, mid-cast of the
norm output, dx rounding before the pool backward). Only taken when the d(ape) window grouping equals the RMSNorm
backward's row partition (CSA at the V4 shapes: 4 rows / 256 groups) and the pool window has at most
``_FUSED_MAX_R`` slots; larger windows (HCA, 128 slots) keep the unfused ops, where a fused backward on [128, 64]
tiles is ~33 KB of code.
"""

import os

import torch
import triton
import triton.language as tl

from .compressor_pool import (
    _ape_main_grad_ok,
    _window_pool_fwd_kernel,
    _window_src,
    fuse2_wpool_dape_enabled,
    glue_window_pool_ok,
)
from .rmsnorm import _main_grad_target, glue_rmsnorm_enabled, is_triton_path_enabled

# pool windows up to this many slots (CSA overlap: 8 -> 16-row tiles) take the fused kernels
_FUSED_MAX_R = 16


@triton.jit
def _comp_norm_rope_fwd_kernel(
    X_PTR,
    W_PTR,
    OUT_PTR,
    RSTD_PTR,
    COS_PTR,
    SIN_PTR,
    N,
    D,
    SO,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NOPE: tl.constexpr,
    RD_HALF: tl.constexpr,
):
    """_rmsnorm_fwd_kernel (weight, mid-cast, bf16 out; D == BLOCK_D) then _apply_rope_fwd_kernel (H = 1) on the
    rounded output, in registers."""
    row = tl.program_id(0)
    if row >= N:
        return
    x_row = X_PTR + row * D
    acc = tl.zeros((), dtype=tl.float32)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    acc += tl.sum(x * x, axis=0)
    rstd = 1.0 / tl.sqrt(acc / D + EPS)
    tl.store(RSTD_PTR + row, rstd)
    y = x * rstd
    y = y.to(X_PTR.dtype.element_ty).to(tl.float32)
    w = tl.load(W_PTR + cols, mask=mask, other=0.0).to(tl.float32)
    y = y * w
    yb = y.to(OUT_PTR.dtype.element_ty)
    # RoPE (interleaved pairs on the last 2 * RD_HALF channels), bf16 expressions of _apply_rope_fwd_kernel
    HALF: tl.constexpr = BLOCK_D // 2
    fi = tl.arange(0, HALF) - NOPE // 2
    is_rope = fi >= 0
    c = tl.load(COS_PTR + row * RD_HALF + fi, mask=is_rope, other=1.0)
    s = tl.load(SIN_PTR + row * RD_HALF + fi, mask=is_rope, other=0.0)
    even, odd = tl.split(tl.reshape(yb, (HALF, 2)))
    rot_even = even * c - odd * s
    rot_odd = even * s + odd * c
    e = tl.where(is_rope, rot_even, even)
    o = tl.where(is_rope, rot_odd, odd)
    tl.store(OUT_PTR + row.to(tl.int64) * SO + cols, tl.reshape(tl.join(e, o), (BLOCK_D,)))


@triton.jit
def _rope_adj(DO_PTR, row, cols, sdo, COS_PTR, SIN_PTR, NOPE: tl.constexpr, RD_HALF: tl.constexpr):
    """dx of _apply_rope_bwd_kernel at columns ``cols`` of one row (partner-column loads keep the load layout):
    even = dout_e * cos + dout_o * sin, odd = -dout_e * sin + dout_o * cos (bf16), nope columns copied."""
    p = DO_PTR + row.to(tl.int64) * sdo
    g = tl.load(p + cols)
    gp = tl.load(p + (cols ^ 1))
    fi = cols // 2 - NOPE // 2
    rm = fi >= 0
    c = tl.load(COS_PTR + row * RD_HALF + fi, mask=rm, other=1.0)
    s = tl.load(SIN_PTR + row * RD_HALF + fi, mask=rm, other=0.0)
    ev = (cols % 2) == 0
    de = g * c + gp * s
    do = -gp * s + g * c
    return tl.where(rm, tl.where(ev, de, do), g)


@triton.jit
def _comp_bwd_kernel(
    P_PTR,
    APE_PTR,
    DO_PTR,
    X_PTR,
    W_PTR,
    RSTD_PTR,
    COS_PTR,
    SIN_PTR,
    DP_PTR,
    PART_PTR,
    DWP_PTR,
    NBN,
    NW,
    R,
    stride_p,
    GW,
    D,
    sdo,
    RATIO: tl.constexpr,
    HD: tl.constexpr,
    OVERLAP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NOPE: tl.constexpr,
    RD_HALF: tl.constexpr,
):
    """_window_pool_bwd_dape_kernel whose dout is computed in place: RoPE adjoint + _rmsnorm_bwd_part_kernel's dx
    for this chunk (row reduction on the full [HD] row, as there); the chunk's dw partial over the GW rows."""
    pid_g = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_r = tl.arange(0, BLOCK_R)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    r_mask = offs_r < R
    d_mask = offs_d < HD
    mask = r_mask[:, None] & d_mask[None, :]
    dt = DP_PTR.dtype.element_ty
    colf = tl.arange(0, HD)
    mf = colf < D
    wf = tl.load(W_PTR + colf, mask=mf, other=0.0).to(tl.float32)
    wc = tl.load(W_PTR + offs_d).to(tl.float32)
    acc = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
    dw = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for g in range(GW):
        pid_bn = pid_g * GW + g
        inb = pid_bn < NBN
        rr = tl.minimum(pid_bn, NBN - 1)
        # ---- RMSNorm backward of row rr (dy = RoPE adjoint, rounded like the RoPE kernel's store) ----
        xf = tl.load(X_PTR + rr * D + colf, mask=mf, other=0.0).to(tl.float32)
        dyf = _rope_adj(DO_PTR, rr, colf, sdo, COS_PTR, SIN_PTR, NOPE, RD_HALF).to(tl.float32)
        rstd = tl.load(RSTD_PTR + rr)
        c = tl.zeros((), dtype=tl.float32)
        c += tl.sum(dyf * wf * xf, axis=0)
        coef = rstd * rstd * rstd * c / D
        xc = tl.load(X_PTR + rr * D + offs_d).to(tl.float32)
        dyc = _rope_adj(DO_PTR, rr, offs_d, sdo, COS_PTR, SIN_PTR, NOPE, RD_HALF).to(tl.float32)
        dxc = rstd * dyc * wc - xc * coef
        dw = tl.where(inb, dw + dyc * (xc * rstd), dw)
        dout = dxc.to(X_PTR.dtype.element_ty).to(tl.float32)
        # ---- pool backward (body of _window_pool_bwd_dape_kernel) ----
        row, kv_c, sc_c, valid = _window_src(pid_bn, offs_r, NW, RATIO, HD, OVERLAP)
        lmask = mask & valid[:, None] & inb
        rbase = row.to(tl.int64)[:, None] * stride_p + offs_d[None, :]
        s = tl.load(P_PTR + rbase + sc_c[:, None], mask=lmask, other=0.0).to(tl.float32)
        a = tl.load(APE_PTR + offs_r[:, None] * HD + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        s = s + a
        s = tl.where(r_mask[:, None], s, float("-inf"))
        s_max = tl.max(s, axis=0)
        e = tl.exp(s - s_max[None, :])
        e = tl.where(r_mask[:, None], e, 0.0)
        denom = tl.sum(e, axis=0)
        w = e / denom[None, :]
        kv = tl.load(P_PTR + rbase + kv_c[:, None], mask=lmask, other=0.0).to(tl.float32)
        dout = tl.where(d_mask & inb, dout, 0.0)
        dkv = dout[None, :] * w
        dwt = dout[None, :] * kv
        sum_dw_w = tl.sum(dwt * w, axis=0)
        dscore = w * (dwt - sum_dw_w[None, :])
        tl.store(DP_PTR + rbase + kv_c[:, None], dkv.to(dt), mask=lmask)
        tl.store(DP_PTR + rbase + sc_c[:, None], dscore.to(dt), mask=lmask)
        acc += tl.where(mask & inb, dscore.to(dt).to(tl.float32), 0.0)
        if OVERLAP:
            zr = (pid_bn * RATIO + offs_r).to(tl.int64)
            zm = (offs_r < RATIO)[:, None] & d_mask[None, :] & inb & (pid_bn % NW == NW - 1)
            z = tl.zeros((BLOCK_R, BLOCK_D), dtype=dt)
            tl.store(DP_PTR + zr[:, None] * stride_p + HD + offs_d[None, :], z, mask=zm)
            tl.store(DP_PTR + zr[:, None] * stride_p + 3 * HD + offs_d[None, :], z, mask=zm)
    tl.store(
        PART_PTR + pid_g.to(tl.int64) * (R * HD) + offs_r[:, None] * HD + offs_d[None, :], acc, mask=mask
    )
    tl.store(DWP_PTR + pid_g * D + offs_d, dw)


@triton.jit
def _comp_reduce_kernel(
    DWP_PTR,
    DW_OUT,
    NP,
    D,
    PART_PTR,
    DAPE_OUT,
    NG,
    NUMEL,
    NDW,
    ACCUM: tl.constexpr,
    W_DTYPE: tl.constexpr,
    BETA: tl.constexpr,
    NG_P: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """_rmsnorm_dw_reduce_kernel (BH = BP = 64) on programs [0, NDW), _dape_reduce_kernel on the rest (same bodies)."""
    pid = tl.program_id(0)
    if pid < NDW:
        cols = pid * 64 + tl.arange(0, 64)
        cm = cols < D
        acc = tl.zeros((64, 64), dtype=tl.float32)
        for p0 in range(0, NP, 64):
            pr = p0 + tl.arange(0, 64)
            acc += tl.load(
                DWP_PTR + pr[:, None] * D + cols[None, :], mask=(pr[:, None] < NP) & cm[None, :], other=0.0
            )
        dw = tl.sum(acc, axis=0).to(W_DTYPE).to(tl.float32)
        if ACCUM:
            dw = tl.load(DW_OUT + cols, mask=cm, other=0.0).to(tl.float32) + dw
        tl.store(DW_OUT + cols, dw.to(DW_OUT.dtype.element_ty), mask=cm)
    else:
        offs = (pid - NDW) * BLOCK + tl.arange(0, BLOCK)
        gs = tl.arange(0, NG_P)
        m = offs < NUMEL
        p = tl.load(
            PART_PTR + gs[:, None].to(tl.int64) * NUMEL + offs[None, :],
            mask=(gs[:, None] < NG) & m[None, :],
            other=0.0,
        )
        v = tl.sum(p, axis=0).to(tl.bfloat16).to(tl.float32)
        if BETA:
            v = v + tl.load(DAPE_OUT + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(DAPE_OUT + offs, v.to(DAPE_OUT.dtype.element_ty), mask=m)


def _groups(n, nd):
    """(GW, NG) of the d(ape) pool backward and (rows per program, P) of the glue RMSNorm backward."""
    gw = max(1, min(8, (n * nd) // 2048))
    ng = triton.cdiv(n, gw)
    p = max(1, min(256, n))
    rpp = max(1, triton.cdiv(n, p))
    p = max(1, triton.cdiv(n, rpp))
    return gw, ng, rpp, p


_TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


def comp_fused_ok(proj, ape, wn, ratio, hd, overlap, rotary_dim) -> bool:
    """The configuration whose unfused chain the fused kernels reproduce bit for bit."""
    if not (fuse2_wpool_dape_enabled() and glue_rmsnorm_enabled() and is_triton_path_enabled()):
        return False
    if os.environ.get("PRIMUS_ROPE_TRITON", "1") == "0":
        return False
    if not glue_window_pool_ok(proj, ape, ratio, hd, overlap):
        return False
    if not (
        ape.requires_grad
        and proj.dtype == torch.bfloat16
        and wn.dtype == torch.bfloat16
        and wn.is_contiguous()
        and wn.numel() == hd
        and hd % 64 == 0
        and (hd & (hd - 1)) == 0
        and hd <= 2048
    ):
        return False
    if not (rotary_dim >= 2 and rotary_dim % 2 == 0 and rotary_dim <= hd):
        return False
    if (2 * ratio if overlap else ratio) > _FUSED_MAX_R:
        return False
    n = proj.shape[0] * (proj.shape[1] // ratio)
    gw, ng, rpp, p = _groups(n, hd // 64)
    return gw == rpp and ng == p


class CompPoolNormRopeFn(torch.autograd.Function):
    """proj [B, S, 2*coff*hd] -> rope(kv_norm(window_pool(proj))) [B, NW, hd]; see the module doc."""

    @staticmethod
    def forward(ctx, proj, ape, wn, cos_flat, sin_flat, ratio, hd, overlap, eps, rotary_dim):  # type: ignore[override]
        B, S, C = proj.shape
        NW = S // ratio
        R = 2 * ratio if overlap else ratio
        N = B * NW
        p2 = proj.reshape(B * S, C)
        pooled = torch.empty((N, hd), dtype=proj.dtype, device=proj.device)
        BLOCK_R = max(16, triton.next_power_of_2(R))
        BLOCK_D = min(64, max(16, triton.next_power_of_2(hd)))
        _window_pool_fwd_kernel[(N, triton.cdiv(hd, BLOCK_D))](
            p2,
            ape,
            pooled,
            NW,
            R,
            p2.stride(0),
            RATIO=ratio,
            HD=hd,
            OVERLAP=bool(overlap),
            BLOCK_R=BLOCK_R,
            BLOCK_D=BLOCK_D,
        )
        out = torch.empty((N, hd), dtype=proj.dtype, device=proj.device)
        rstd = torch.empty((N,), dtype=torch.float32, device=proj.device)
        wd = wn.detach().reshape(hd)
        _comp_norm_rope_fwd_kernel[(N,)](
            pooled,
            wd,
            out,
            rstd,
            cos_flat,
            sin_flat,
            N,
            hd,
            out.stride(0),
            EPS=float(eps),
            BLOCK_D=hd,
            NOPE=hd - rotary_dim,
            RD_HALF=rotary_dim // 2,
        )
        ctx.save_for_backward(p2, ape, pooled, wd, rstd, cos_flat, sin_flat)
        ctx.meta = (B, S, C, NW, R, ratio, hd, bool(overlap), BLOCK_R, BLOCK_D, rotary_dim)
        # main_grad claims exactly as _WindowPool (d(ape)) and FusedRMSNormFn (glue) make them
        ctx.ape_param = None
        if _ape_main_grad_ok(ape):
            ctx.ape_param = ape
            ape.grad_added_to_main_grad = True
            ape._primus_fwd_claim = True
        ctx.wparam = None
        ctx.wn_dtype, ctx.wn_shape = wn.dtype, tuple(wn.shape)
        if _main_grad_target(wn) is not None:
            wn.grad_added_to_main_grad = True
            wn._primus_fwd_claim = True
            ctx.wparam = wn
        return out.view(B, NW, hd)

    @staticmethod
    def backward(ctx, dout):  # type: ignore[override]
        from primus_turbo.pytorch.ops.utils import (
            _get_dummy_wgrad,
            main_grad_first_write,
        )

        p2, ape, pooled, wd, rstd, cos_flat, sin_flat = ctx.saved_tensors
        B, S, C, NW, R, ratio, hd, overlap, BLOCK_R, BLOCK_D, rd = ctx.meta
        N = B * NW
        do2 = dout.reshape(N, hd)
        if do2.stride(1) != 1:
            do2 = do2.contiguous()
        dev = p2.device
        dp = torch.empty_like(p2)
        nd = triton.cdiv(hd, BLOCK_D)
        GW, NG, _, P = _groups(N, nd)
        part = torch.empty((NG, R, hd), dtype=torch.float32, device=dev)
        dwp = torch.empty((P, hd), dtype=torch.float32, device=dev)
        _comp_bwd_kernel[(NG, nd)](
            p2,
            ape,
            do2,
            pooled,
            wd,
            rstd,
            cos_flat,
            sin_flat,
            dp,
            part,
            dwp,
            N,
            NW,
            R,
            p2.stride(0),
            GW,
            hd,
            do2.stride(0),
            RATIO=ratio,
            HD=hd,
            OVERLAP=overlap,
            BLOCK_R=BLOCK_R,
            BLOCK_D=BLOCK_D,
            NOPE=hd - rd,
            RD_HALF=rd // 2,
        )
        # RMSNorm dw target (as FusedRMSNormFn's glue backward)
        wparam, ctx.wparam = ctx.wparam, None
        mg = _main_grad_target(wparam) if wparam is not None else None
        accum = False
        if mg is not None:
            accum = not main_grad_first_write(wparam)
            dw_out = mg.view(-1)
        else:
            dw_out = torch.empty((hd,), dtype=ctx.wn_dtype, device=dev)
        # d(ape) target (as _WindowPool)
        ap, ctx.ape_param = ctx.ape_param, None
        beta = False
        if ap is not None:
            dape_out = ap.main_grad
            beta = not main_grad_first_write(ap)
        else:
            dape_out = torch.empty((R, hd), dtype=ape.dtype, device=dev)
        numel = R * hd
        ng_p = triton.next_power_of_2(NG)
        blk = max(16, min(1024, 8192 // ng_p))
        ndw = triton.cdiv(hd, 64)
        _comp_reduce_kernel[(ndw + triton.cdiv(numel, blk),)](
            dwp,
            dw_out,
            P,
            hd,
            part,
            dape_out,
            NG,
            numel,
            ndw,
            ACCUM=accum,
            W_DTYPE=_TL[ctx.wn_dtype],
            BETA=beta,
            NG_P=ng_p,
            BLOCK=blk,
            num_warps=4,
        )
        dwn = (
            _get_dummy_wgrad(list(ctx.wn_shape), ctx.wn_dtype)
            if mg is not None
            else dw_out.reshape(ctx.wn_shape)
        )
        dape = _get_dummy_wgrad(list(ape.shape), ape.dtype) if ap is not None else dape_out
        return dp.view(B, S, C), dape, dwn, None, None, None, None, None, None, None


def comp_pool_norm_rope(proj, ape, wn, cos_flat, sin_flat, *, ratio, hd, overlap, eps, rotary_dim):
    """-> rope(kv_norm(window_pool(proj))) [B, S // ratio, hd]; ``cos_flat`` / ``sin_flat`` are the
    [B * S // ratio, rotary_dim // 2] RoPE tables in the activation dtype."""
    return CompPoolNormRopeFn.apply(proj, ape, wn, cos_flat, sin_flat, ratio, hd, overlap, eps, rotary_dim)
