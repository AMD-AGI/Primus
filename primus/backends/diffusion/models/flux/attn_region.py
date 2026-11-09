###############################################################################
# Flux attention region: fused QK-RMSNorm + RoPE around backend_attention, with
# the copies the patch path pays around attention removed. Same kernels and
# math as fused_norm_rope.py; forward outputs and every gradient are bitwise
# equal to it (campaign flux_primus_attn_copy_elision, ruler
# bench_flux_attn_copies).
#
# - Rows are addressed as (batch, token, head), so a sequence slice of a joint
#   [B, L, H, D] tensor is read and written in place. The double block's txt
#   and img q / k / v land in one buffer (no torch.cat) and its backward reads
#   the joint gradient's slices in place (no .contiguous()).
# - Single block: v is a view of qkv's V column instead of a copy.
# - Double block: the dv -> d_qkv V-column write rides the q-backward launch.
# - q and k of the same rows share one forward launch and one cos/sin load.
#
# Attention itself is whatever backend_attention dispatches to (AITER or the
# FlyDSL FLUX_ATTN_FLYDSL hook); nothing here calls an attention library.
###############################################################################

import os
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.library import custom_op, wrap_triton

# Distinct from fused_norm_rope.py's primus_flux namespace: both modules load in
# one process.
NS = "primus_flux_region"


@triton.jit
def _materialize(x):
    return tl.inline_asm_elementwise("v_mov_b32 $0, $1", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _base(rows, STOK, SH, H, SBX, LH, LVL3: tl.constexpr):
    """Element offset of each row: batch term, token stride, head stride.

    ``SBX`` is ``s_batch - L * s_tok``, which is zero exactly when the tokens of
    every batch are uniformly strided. fused_norm_rope.py's two-level expression
    is the LVL3=False specialisation, textually unchanged, so a launch that does
    not need the batch term compiles that kernel.
    """
    o = (rows // H) * STOK + (rows % H) * SH
    if LVL3:
        o = o + (rows // LH) * SBX
    return o


@triton.jit
def _pair(
    P,
    rows,
    mask,
    STOK,
    SH,
    H,
    SBX,
    LH,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    LVL3: tl.constexpr,
):
    half: tl.constexpr = D // 2
    b = _base(rows, STOK, SH, H, SBX, LH, LVL3)[:, None]
    if INTERLEAVED:
        v = tl.load(P + b + tl.arange(0, D)[None, :], mask=mask, other=0.0).to(tl.float32)
        return tl.split(tl.reshape(v, (BLOCK_M, half, 2)))
    lo = tl.arange(0, half)[None, :]
    a = tl.load(P + b + lo, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(P + b + lo + half, mask=mask, other=0.0).to(tl.float32)
    return a, c


@triton.jit
def _unpair(
    P,
    rows,
    mask,
    v_lo,
    v_hi,
    STOK,
    SH,
    H,
    SBX,
    LH,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    LVL3: tl.constexpr,
):
    half: tl.constexpr = D // 2
    b = _base(rows, STOK, SH, H, SBX, LH, LVL3)[:, None]
    if INTERLEAVED:
        v = tl.reshape(tl.join(v_lo, v_hi), (BLOCK_M, D))
        tl.store(P + b + tl.arange(0, D)[None, :], v.to(P.dtype.element_ty), mask=mask)
    else:
        lo = tl.arange(0, half)[None, :]
        tl.store(P + b + lo, v_lo.to(P.dtype.element_ty), mask=mask)
        tl.store(P + b + lo + half, v_hi.to(P.dtype.element_ty), mask=mask)


@triton.jit
def _pair_w(W, D: tl.constexpr, INTERLEAVED: tl.constexpr):
    half: tl.constexpr = D // 2
    if INTERLEAVED:
        v = tl.load(W + tl.arange(0, D)).to(tl.float32)
        a, b = tl.split(tl.reshape(v, (half, 2)))
        return a[None, :], b[None, :]
    lo = tl.arange(0, half)
    return (tl.load(W + lo).to(tl.float32)[None, :], tl.load(W + lo + half).to(tl.float32)[None, :])


def _fwd_configs():
    return [triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=2)]


@triton.autotune(configs=_fwd_configs(), key=["M", "D"])
@triton.jit
def _fwd_kernel(
    X,
    W,
    COS,
    SIN,
    OUT,
    RSTD,
    COPY_X,
    COPY_OUT,
    M,
    D: tl.constexpr,
    H,
    EPS,
    SX_TOK,
    SX_H,
    SO_TOK,
    SO_H,
    SCOPY_TOK,
    SCOPY_H,
    SC,
    SX_BX,
    SO_BX,
    SCX_BX,
    SCO_BX,
    LH,
    BLOCK_M: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    COPY_ROW: tl.constexpr,
    LVL3: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M
    mask = mask_m[:, None]

    b = _base(rows, SX_TOK, SX_H, H, SX_BX, LH, LVL3)[:, None]
    x = tl.load(X + b + tl.arange(0, D)[None, :], mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1)[:, None] / D + EPS)
    if INTERLEAVED:
        x_lo, x_hi = tl.split(tl.reshape(x, (BLOCK_M, D // 2, 2)))
    else:
        x_lo, x_hi = _pair(X, rows, mask, SX_TOK, SX_H, H, SX_BX, LH, D, BLOCK_M, INTERLEAVED, LVL3)

    w_lo, w_hi = _pair_w(W, D, INTERLEAVED)
    n_lo, n_hi = x_lo * rstd * w_lo, x_hi * rstd * w_hi

    tok = rows // H
    c_lo, c_hi = _pair(COS, tok, mask, SC, SC, 1, 0, LH, D, BLOCK_M, INTERLEAVED, False)
    s_lo, s_hi = _pair(SIN, tok, mask, SC, SC, 1, 0, LH, D, BLOCK_M, INTERLEAVED, False)

    _unpair(
        OUT,
        rows,
        mask,
        tl.fma(c_lo, n_lo, (-s_lo) * n_hi),
        tl.fma(s_hi, n_lo, c_hi * n_hi),
        SO_TOK,
        SO_H,
        H,
        SO_BX,
        LH,
        D,
        BLOCK_M,
        INTERLEAVED,
        LVL3,
    )
    tl.store(RSTD + rows, tl.reshape(rstd, (BLOCK_M,)), mask=mask_m)

    if COPY_ROW:
        src = _base(rows, SCOPY_TOK, SCOPY_H, H, SCX_BX, LH, LVL3)[:, None] + tl.arange(0, D)[None, :]
        dst = _base(rows, SO_TOK, SO_H, H, SCO_BX, LH, LVL3)[:, None] + tl.arange(0, D)[None, :]
        tl.store(COPY_OUT + dst, tl.load(COPY_X + src, mask=mask, other=0.0), mask=mask)


@triton.autotune(configs=_fwd_configs(), key=["M", "D"])
@triton.jit
def _fwd_qk_kernel(
    XQ,
    XK,
    WQ,
    WK,
    COS,
    SIN,
    OUTQ,
    OUTK,
    RSTDQ,
    RSTDK,
    COPY_X,
    COPY_OUT,
    M,
    D: tl.constexpr,
    H,
    EPS,
    SXQ_TOK,
    SXQ_H,
    SXK_TOK,
    SXK_H,
    SOQ_TOK,
    SOQ_H,
    SOK_TOK,
    SOK_H,
    SCOPY_TOK,
    SCOPY_H,
    SC,
    SXQ_BX,
    SXK_BX,
    SOQ_BX,
    SOK_BX,
    SCX_BX,
    SCO_BX,
    LH,
    BLOCK_M: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    COPY_ROW: tl.constexpr,
    LVL3: tl.constexpr,
):
    """q and k of the SAME rows in one program, sharing one cos/sin load.

    Every expression below is the verbatim per-channel body of ``_fwd_kernel``,
    duplicated once for Q and once for K with no new arithmetic -- the only
    shared computation is ``tok``/``c_lo``/``c_hi``/``s_lo``/``s_hi``, which
    ``_fwd_kernel`` itself already derives solely from ``rows`` (same rows for
    both channels), so reusing one load for both stores changes nothing about
    the floating-point values either channel's RoPE multiply sees.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M
    mask = mask_m[:, None]

    # ---- Q channel ----
    bq = _base(rows, SXQ_TOK, SXQ_H, H, SXQ_BX, LH, LVL3)[:, None]
    xq = tl.load(XQ + bq + tl.arange(0, D)[None, :], mask=mask, other=0.0).to(tl.float32)
    rstdq = tl.rsqrt(tl.sum(xq * xq, axis=1)[:, None] / D + EPS)
    if INTERLEAVED:
        xq_lo, xq_hi = tl.split(tl.reshape(xq, (BLOCK_M, D // 2, 2)))
    else:
        xq_lo, xq_hi = _pair(XQ, rows, mask, SXQ_TOK, SXQ_H, H, SXQ_BX, LH, D, BLOCK_M, INTERLEAVED, LVL3)
    wq_lo, wq_hi = _pair_w(WQ, D, INTERLEAVED)
    nq_lo, nq_hi = xq_lo * rstdq * wq_lo, xq_hi * rstdq * wq_hi

    # ---- K channel ----
    bk = _base(rows, SXK_TOK, SXK_H, H, SXK_BX, LH, LVL3)[:, None]
    xk = tl.load(XK + bk + tl.arange(0, D)[None, :], mask=mask, other=0.0).to(tl.float32)
    rstdk = tl.rsqrt(tl.sum(xk * xk, axis=1)[:, None] / D + EPS)
    if INTERLEAVED:
        xk_lo, xk_hi = tl.split(tl.reshape(xk, (BLOCK_M, D // 2, 2)))
    else:
        xk_lo, xk_hi = _pair(XK, rows, mask, SXK_TOK, SXK_H, H, SXK_BX, LH, D, BLOCK_M, INTERLEAVED, LVL3)
    wk_lo, wk_hi = _pair_w(WK, D, INTERLEAVED)
    nk_lo, nk_hi = xk_lo * rstdk * wk_lo, xk_hi * rstdk * wk_hi

    # ---- shared cos/sin: the one load _fwd_kernel would otherwise pay twice ----
    tok = rows // H
    c_lo, c_hi = _pair(COS, tok, mask, SC, SC, 1, 0, LH, D, BLOCK_M, INTERLEAVED, False)
    s_lo, s_hi = _pair(SIN, tok, mask, SC, SC, 1, 0, LH, D, BLOCK_M, INTERLEAVED, False)

    _unpair(
        OUTQ,
        rows,
        mask,
        tl.fma(c_lo, nq_lo, (-s_lo) * nq_hi),
        tl.fma(s_hi, nq_lo, c_hi * nq_hi),
        SOQ_TOK,
        SOQ_H,
        H,
        SOQ_BX,
        LH,
        D,
        BLOCK_M,
        INTERLEAVED,
        LVL3,
    )
    tl.store(RSTDQ + rows, tl.reshape(rstdq, (BLOCK_M,)), mask=mask_m)

    _unpair(
        OUTK,
        rows,
        mask,
        tl.fma(c_lo, nk_lo, (-s_lo) * nk_hi),
        tl.fma(s_hi, nk_lo, c_hi * nk_hi),
        SOK_TOK,
        SOK_H,
        H,
        SOK_BX,
        LH,
        D,
        BLOCK_M,
        INTERLEAVED,
        LVL3,
    )
    tl.store(RSTDK + rows, tl.reshape(rstdk, (BLOCK_M,)), mask=mask_m)

    if COPY_ROW:
        src = _base(rows, SCOPY_TOK, SCOPY_H, H, SCX_BX, LH, LVL3)[:, None] + tl.arange(0, D)[None, :]
        dst = _base(rows, SOQ_TOK, SOQ_H, H, SCO_BX, LH, LVL3)[:, None] + tl.arange(0, D)[None, :]
        tl.store(COPY_OUT + dst, tl.load(COPY_X + src, mask=mask, other=0.0), mask=mask)


def _mix_order_split(M: int, device: torch.device) -> int:
    per = M // (torch.cuda.get_device_properties(device).multi_processor_count * 8)
    split = 1 << (per.bit_length() - 1) if per > 0 else 1
    return min(max(split, 16), 128)


def _bwd_configs():
    bm = int(os.getenv("FLUX_FUSED_NORM_ROPE_BWD_BLOCK_M", "8"))
    return [triton.Config({"BLOCK_M": bm}, num_warps=1, num_stages=1)]


@triton.autotune(configs=_bwd_configs(), key=["M", "D"])
@triton.jit
def _bwd_kernel(
    G,
    X,
    W,
    COS,
    SIN,
    RSTD,
    DX,
    DWP,
    COPY_X,
    COPY_OUT,
    M,
    D: tl.constexpr,
    H,
    SG_TOK,
    SG_H,
    SX_TOK,
    SX_H,
    SDX_TOK,
    SDX_H,
    SC,
    SG_BX,
    SX_BX,
    SDX_BX,
    SCOPY_TOK,
    SCOPY_H,
    SCX_BX,
    SCO_BX,
    LH,
    BLOCK_M: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    RSPLIT: tl.constexpr,
    ROUND_GRAD: tl.constexpr,
    LVL3: tl.constexpr,
    COPY_ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    half: tl.constexpr = D // 2
    w_lo, w_hi = _pair_w(W, D, INTERLEAVED)
    dw_lo = tl.zeros((half,), dtype=tl.float32)
    dw_hi = tl.zeros((half,), dtype=tl.float32)
    dw_nat = tl.zeros((D,), dtype=tl.float32)

    for start in tl.range(pid * RSPLIT, pid * RSPLIT + RSPLIT, BLOCK_M):
        rows = start + tl.arange(0, BLOCK_M)
        mask = (rows < M)[:, None]
        rstd = tl.load(RSTD + rows, mask=rows < M, other=0.0)[:, None]

        if INTERLEAVED:
            cols = tl.arange(0, D)[None, :]
            first = (cols % 2) == 0
            g = tl.load(
                G + _base(rows, SG_TOK, SG_H, H, SG_BX, LH, LVL3)[:, None] + cols, mask=mask, other=0.0
            ).to(tl.float32)
            g0, g1 = tl.split(tl.reshape(g, (BLOCK_M, D // 2, 2)))
            g_a = tl.reshape(tl.join(g0, g0), (BLOCK_M, D))
            g_b = tl.reshape(tl.join(g1, g1), (BLOCK_M, D))
            x = tl.load(
                X + _base(rows, SX_TOK, SX_H, H, SX_BX, LH, LVL3)[:, None] + cols, mask=mask, other=0.0
            ).to(tl.float32)
            cos = tl.load(COS + (rows // H)[:, None] * SC + cols, mask=mask, other=0.0)
            sin = tl.load(SIN + (rows // H)[:, None] * SC + cols, mask=mask, other=0.0)
            w = tl.load(W + cols).to(tl.float32)
            dn = (g_a * tl.where(first, cos, -sin) + g_b * tl.where(first, sin, cos)) + 0.0
            if ROUND_GRAD:
                dn = dn.to(tl.bfloat16).to(tl.float32)
            else:
                dn = _materialize(dn)
            u = x * rstd
            du = dn * w
            s = tl.sum(u * du, axis=1)[:, None]
            inv_d: tl.constexpr = 1.0 / D
            dx = (du - (u * inv_d) * s) * rstd
            tl.store(
                DX + _base(rows, SDX_TOK, SDX_H, H, SDX_BX, LH, LVL3)[:, None] + cols,
                dx.to(DX.dtype.element_ty),
                mask=mask,
            )
            dw_nat += tl.sum(tl.where(mask, dn * u, 0.0), axis=0)
        else:
            g_lo, g_hi = _pair(G, rows, mask, SG_TOK, SG_H, H, SG_BX, LH, D, BLOCK_M, INTERLEAVED, LVL3)
            x_lo, x_hi = _pair(X, rows, mask, SX_TOK, SX_H, H, SX_BX, LH, D, BLOCK_M, INTERLEAVED, LVL3)
            tok = rows // H
            c_lo, c_hi = _pair(COS, tok, mask, SC, SC, 1, 0, LH, D, BLOCK_M, INTERLEAVED, False)
            s_lo, s_hi = _pair(SIN, tok, mask, SC, SC, 1, 0, LH, D, BLOCK_M, INTERLEAVED, False)
            dn_lo = g_lo * c_lo + g_hi * s_hi
            dn_hi = g_hi * c_hi - g_lo * s_lo
            u_lo, u_hi = x_lo * rstd, x_hi * rstd
            du_lo, du_hi = dn_lo * w_lo, dn_hi * w_hi
            m = (tl.sum(du_lo * u_lo, axis=1) + tl.sum(du_hi * u_hi, axis=1))[:, None] / D
            _unpair(
                DX,
                rows,
                mask,
                rstd * (du_lo - u_lo * m),
                rstd * (du_hi - u_hi * m),
                SDX_TOK,
                SDX_H,
                H,
                SDX_BX,
                LH,
                D,
                BLOCK_M,
                INTERLEAVED,
                LVL3,
            )
            dw_lo += tl.sum(tl.where(mask, dn_lo * u_lo, 0.0), axis=0)
            dw_hi += tl.sum(tl.where(mask, dn_hi * u_hi, 0.0), axis=0)

        if COPY_ROW:
            src = _base(rows, SCOPY_TOK, SCOPY_H, H, SCX_BX, LH, LVL3)[:, None] + tl.arange(0, D)[None, :]
            dst = _base(rows, SDX_TOK, SDX_H, H, SCO_BX, LH, LVL3)[:, None] + tl.arange(0, D)[None, :]
            tl.store(COPY_OUT + dst, tl.load(COPY_X + src, mask=mask, other=0.0), mask=mask)

    if INTERLEAVED:
        tl.store(DWP + pid * D + tl.arange(0, D), dw_nat)
    else:
        lo = tl.arange(0, half)
        tl.store(DWP + pid * D + lo, dw_lo)
        tl.store(DWP + pid * D + lo + half, dw_hi)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def default_eps(dtype: torch.dtype) -> float:
    return torch.finfo(torch.float64 if dtype == torch.float64 else torch.float32).eps


def _strides3(t: torch.Tensor, H: int, D: int) -> Tuple[int, int, int, int]:
    """(token stride, head stride, batch excess, L) for a [B, L, H, D] view.

    ``batch excess`` is ``s_batch - L * s_tok``: zero exactly when two-level
    addressing already covers the tensor, non-zero for a sequence slice of a
    joint tensor.
    """
    if t.dim() != 4 or t.shape[-2:] != (H, D):
        raise ValueError(f"expected [B, L, {H}, {D}], got {tuple(t.shape)}")
    s_b, s_tok, s_h, s_d = t.stride()
    if s_d != 1 or s_h < D:
        raise ValueError(f"rows must be contiguous and non-overlapping, strides {t.stride()}")
    L = t.shape[1]
    return s_tok, s_h, (s_b - L * s_tok if t.shape[0] > 1 else 0), L


def _launch_fwd(x, w, cos, sin, eps, interleaved, traceable, copy_x=None, copy_out=None, out=None):
    H, D = x.shape[-2], x.shape[-1]
    M = x.numel() // D
    if cos.shape != (M // H, D) or sin.shape != cos.shape:
        raise ValueError(f"cos/sin must be ({M // H}, {D}), got {tuple(cos.shape)}")
    sx_tok, sx_h, sx_bx, L = _strides3(x, H, D)
    if out is None:
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    so_tok, so_h, so_bx, _ = _strides3(out, H, D)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)

    if copy_x is not None:
        scopy_tok, scopy_h, scx_bx, _ = _strides3(copy_x, H, D)
        cto, cho, sco_bx, _ = _strides3(copy_out, H, D)
        if (cto, cho) != (so_tok, so_h):
            raise ValueError("copy_out must share out's token/head strides")
        has_copy = True
    else:
        copy_x, copy_out = x, out
        scopy_tok, scopy_h, scx_bx, sco_bx = sx_tok, sx_h, sx_bx, so_bx
        has_copy = False

    lvl3 = any(b for b in (sx_bx, so_bx, scx_bx, sco_bx))
    launch = wrap_triton(_fwd_kernel) if traceable else _fwd_kernel
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    launch[grid](
        x,
        w,
        cos,
        sin,
        out,
        rstd,
        copy_x,
        copy_out,
        M,
        D,
        H,
        eps,
        sx_tok,
        sx_h,
        so_tok,
        so_h,
        scopy_tok,
        scopy_h,
        cos.stride(0),
        sx_bx,
        so_bx,
        scx_bx,
        sco_bx,
        L * H,
        INTERLEAVED=interleaved,
        COPY_ROW=has_copy,
        LVL3=lvl3,
    )
    return out, rstd


def _launch_fwd_qk(
    xq, xk, wq, wk, cos, sin, eps, interleaved, traceable, copy_x=None, copy_out=None, outq=None, outk=None
):
    """One launch of _fwd_qk_kernel instead of two _launch_fwd calls.

    ``xq``/``xk`` must be the Q/K channel views of the SAME qkv tensor (so
    they share token/head strides, differing only in base pointer) -- exactly
    what every caller already has. Mirrors _launch_fwd's own structure.
    """
    H, D = xq.shape[-2], xq.shape[-1]
    M = xq.numel() // D
    if cos.shape != (M // H, D) or sin.shape != cos.shape:
        raise ValueError(f"cos/sin must be ({M // H}, {D}), got {tuple(cos.shape)}")
    sxq_tok, sxq_h, sxq_bx, L = _strides3(xq, H, D)
    sxk_tok, sxk_h, sxk_bx, _ = _strides3(xk, H, D)
    if outq is None:
        outq = torch.empty(xq.shape, device=xq.device, dtype=xq.dtype)
    if outk is None:
        outk = torch.empty(xk.shape, device=xk.device, dtype=xk.dtype)
    soq_tok, soq_h, soq_bx, _ = _strides3(outq, H, D)
    sok_tok, sok_h, sok_bx, _ = _strides3(outk, H, D)
    rstdq = torch.empty(M, device=xq.device, dtype=torch.float32)
    rstdk = torch.empty(M, device=xq.device, dtype=torch.float32)

    if copy_x is not None:
        scopy_tok, scopy_h, scx_bx, _ = _strides3(copy_x, H, D)
        cto, cho, sco_bx, _ = _strides3(copy_out, H, D)
        if (cto, cho) != (soq_tok, soq_h):
            raise ValueError("copy_out must share outq's token/head strides")
        has_copy = True
    else:
        copy_x, copy_out = xq, outq
        scopy_tok, scopy_h, scx_bx, sco_bx = sxq_tok, sxq_h, sxq_bx, soq_bx
        has_copy = False

    lvl3 = any(b for b in (sxq_bx, sxk_bx, soq_bx, sok_bx, scx_bx, sco_bx))
    launch = wrap_triton(_fwd_qk_kernel) if traceable else _fwd_qk_kernel
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    launch[grid](
        xq,
        xk,
        wq,
        wk,
        cos,
        sin,
        outq,
        outk,
        rstdq,
        rstdk,
        copy_x,
        copy_out,
        M,
        D,
        H,
        eps,
        sxq_tok,
        sxq_h,
        sxk_tok,
        sxk_h,
        soq_tok,
        soq_h,
        sok_tok,
        sok_h,
        scopy_tok,
        scopy_h,
        cos.stride(0),
        sxq_bx,
        sxk_bx,
        soq_bx,
        sok_bx,
        scx_bx,
        sco_bx,
        L * H,
        INTERLEAVED=interleaved,
        COPY_ROW=has_copy,
        LVL3=lvl3,
    )
    return outq, outk, rstdq, rstdk


def _launch_bwd(
    g, x, w, cos, sin, rstd, dx, interleaved, traceable, round_grad=False, copy_x=None, copy_out=None
):
    H, D = x.shape[-2], x.shape[-1]
    M = x.numel() // D
    sg_tok, sg_h, sg_bx, L = _strides3(g, H, D)
    sx_tok, sx_h, sx_bx, _ = _strides3(x, H, D)
    sdx_tok, sdx_h, sdx_bx, _ = _strides3(dx, H, D)
    rsplit = _mix_order_split(M, x.device)
    nprog = triton.cdiv(M, rsplit)
    dwp = torch.empty(nprog, D, device=x.device, dtype=torch.float32)

    if copy_x is not None:
        # An extra row-copy rides this launch's grid -- same mechanism as
        # the forward's COPY_ROW, mirrored onto the backward.
        scopy_tok, scopy_h, scx_bx, _ = _strides3(copy_x, H, D)
        cto, cho, sco_bx, _ = _strides3(copy_out, H, D)
        if (cto, cho) != (sdx_tok, sdx_h):
            raise ValueError("copy_out must share dx's token/head strides")
        has_copy = True
    else:
        copy_x, copy_out = x, dx
        scopy_tok, scopy_h, scx_bx, sco_bx = sx_tok, sx_h, sx_bx, sdx_bx
        has_copy = False

    lvl3 = any(b for b in (sg_bx, sx_bx, sdx_bx, scx_bx, sco_bx))
    launch = wrap_triton(_bwd_kernel) if traceable else _bwd_kernel
    launch[(nprog,)](
        g,
        x,
        w,
        cos,
        sin,
        rstd,
        dx,
        dwp,
        copy_x,
        copy_out,
        M,
        D,
        H,
        sg_tok,
        sg_h,
        sx_tok,
        sx_h,
        sdx_tok,
        sdx_h,
        cos.stride(0),
        sg_bx,
        sx_bx,
        sdx_bx,
        scopy_tok,
        scopy_h,
        scx_bx,
        sco_bx,
        L * H,
        INTERLEAVED=interleaved,
        RSPLIT=rsplit,
        ROUND_GRAD=round_grad,
        LVL3=lvl3,
        COPY_ROW=has_copy,
    )
    return dwp


# ---------------------------------------------------------------------------
# Single-block q/k + v-direct: the kernel produces q and k; v is the qkv
# buffer's own V column, handed to attention as a view (no COPY_ROW, no v
# allocation at all).
#
# A custom_op may not return an alias of one of its inputs, so the alias is
# produced by a thin autograd.Function instead. Its forward calls the q/k-only
# kernel op below and then slices v out of qkv directly; its backward owns dq,
# dk and dv together through qkv_norm_rope_backward, which writes dq / dk into
# d_qkv and copies dv into its V column. q_rstd / k_rstd live in the Function's ctx rather than as declared
# custom-op outputs, so autograd never has to synthesize a gradient for them.
# ---------------------------------------------------------------------------


@custom_op(f"{NS}::qk_norm_rope_pair", mutates_args=())
def _qk_fwd(
    qkv: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    eps: float,
    interleaved: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, L, T = qkv.shape
    D = T // (3 * heads)
    view = qkv.view(B, L, 3, heads, D)
    q, k, q_rstd, k_rstd = _launch_fwd_qk(
        view[:, :, 0], view[:, :, 1], wq, wk, cos, sin, eps, interleaved, traceable=False
    )
    return q, k, q_rstd, k_rstd


@_qk_fwd.register_fake
def _(qkv, wq, wk, cos, sin, heads, eps, interleaved):
    B, L, T = qkv.shape
    D = T // (3 * heads)
    head = torch.empty((B, L, heads, D), device=qkv.device, dtype=qkv.dtype)
    rstd = torch.empty(B * L * heads, device=qkv.device, dtype=torch.float32)
    return (head, torch.empty_like(head), rstd, torch.empty_like(rstd))


# FlyDSL attention takes contiguous q / k / v only (K and V share one token
# stride), so a v view is copied by its .contiguous() anyway, as a standalone
# kernel. FLUX_ATTN_SINGLE_V_COPYROW=1 writes that copy from the q/k launch's
# COPY_ROW instead, as the double block does; v is then already contiguous.
_SINGLE_V_COPYROW = os.getenv("FLUX_ATTN_SINGLE_V_COPYROW", "0") == "1"
# Double block: the FlyDSL attention writes its txt and img rows to two contiguous outputs (needs
# FLUX_ATTN_FLYDSL=1), so neither the consumers' reshape copies nor the backward's join run.
_DOUBLE_SPLIT = os.getenv("FLUX_ATTN_DOUBLE_SPLIT", "0") == "1" and os.getenv("FLUX_ATTN_FLYDSL", "0") == "1"


@custom_op(f"{NS}::qk_norm_rope_pair_v", mutates_args=())
def _qk_fwd_v(
    qkv: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    eps: float,
    interleaved: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, L, T = qkv.shape
    D = T // (3 * heads)
    view = qkv.view(B, L, 3, heads, D)
    v = torch.empty((B, L, heads, D), device=qkv.device, dtype=qkv.dtype)
    q, k, q_rstd, k_rstd = _launch_fwd_qk(
        view[:, :, 0],
        view[:, :, 1],
        wq,
        wk,
        cos,
        sin,
        eps,
        interleaved,
        traceable=False,
        copy_x=view[:, :, 2],
        copy_out=v,
    )
    return q, k, v, q_rstd, k_rstd


@_qk_fwd_v.register_fake
def _(qkv, wq, wk, cos, sin, heads, eps, interleaved):
    B, L, T = qkv.shape
    D = T // (3 * heads)
    head = torch.empty((B, L, heads, D), device=qkv.device, dtype=qkv.dtype)
    rstd = torch.empty(B * L * heads, device=qkv.device, dtype=torch.float32)
    return (head, torch.empty_like(head), torch.empty_like(head), rstd, torch.empty_like(rstd))


@custom_op(f"{NS}::qkv_norm_rope_backward", mutates_args=())
def _qkv_bwd(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    qkv: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    heads: int,
    interleaved: bool,
    round_grad: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, L, T = qkv.shape
    D = T // (3 * heads)
    d_qkv = torch.empty_like(qkv, memory_format=torch.contiguous_format)
    src = qkv.view(B, L, 3, heads, D)
    dst = d_qkv.view(B, L, 3, heads, D)
    dwqp = _launch_bwd(
        dq,
        src[:, :, 0],
        wq,
        cos,
        sin,
        q_rstd,
        dst[:, :, 0],
        interleaved,
        traceable=False,
        round_grad=round_grad,
    )
    dwkp = _launch_bwd(
        dk,
        src[:, :, 1],
        wk,
        cos,
        sin,
        k_rstd,
        dst[:, :, 1],
        interleaved,
        traceable=False,
        round_grad=round_grad,
    )
    dst[:, :, 2].copy_(dv)
    return d_qkv, dwqp, dwkp


@_qkv_bwd.register_fake
def _(dq, dk, dv, qkv, wq, wk, cos, sin, q_rstd, k_rstd, heads, interleaved, round_grad):
    D = qkv.shape[-1] // (3 * heads)
    M = qkv.numel() // (3 * D)
    p = torch.empty(
        triton.cdiv(M, _mix_order_split(M, qkv.device)), D, device=qkv.device, dtype=torch.float32
    )
    return (torch.empty_like(qkv, memory_format=torch.contiguous_format), p, torch.empty_like(p))


class _SingleQKVFn(torch.autograd.Function):
    """q, k from the kernel; v is qkv's own V column, returned as a view
    (or copied by the same launch under FLUX_ATTN_SINGLE_V_COPYROW)."""

    @staticmethod
    def forward(ctx, qkv, wq, wk, cos, sin, heads, eps, interleaved):
        if _SINGLE_V_COPYROW:
            q, k, v, q_rstd, k_rstd = _qk_fwd_v(qkv, wq, wk, cos, sin, heads, eps, interleaved)
        else:
            q, k, q_rstd, k_rstd = _qk_fwd(qkv, wq, wk, cos, sin, heads, eps, interleaved)
            B, L, T = qkv.shape
            D = T // (3 * heads)
            v = qkv.view(B, L, 3, heads, D)[:, :, 2]
        ctx.save_for_backward(qkv, wq, wk, cos, sin, q_rstd, k_rstd)
        ctx.heads = heads
        ctx.interleaved = interleaved
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        qkv, wq, wk, cos, sin, q_rstd, k_rstd = ctx.saved_tensors
        d_qkv, dwqp, dwkp = _qkv_bwd(
            dq.contiguous(),
            dk.contiguous(),
            dv.contiguous(),
            qkv,
            wq,
            wk,
            cos,
            sin,
            q_rstd,
            k_rstd,
            ctx.heads,
            ctx.interleaved,
            False,
        )
        return (d_qkv, dwqp.sum(0).to(wq.dtype), dwkp.sum(0).to(wk.dtype), None, None, None, None, None)


# ---------------------------------------------------------------------------
# Joint double-block op: both sides write into one q / k / v, no cat; the
# backward reads the joint gradient's slices in place, no .contiguous().
# ---------------------------------------------------------------------------


@custom_op(f"{NS}::qkv_norm_rope_double", mutates_args=())
def _dbl_fwd(
    txt_qkv: torch.Tensor,
    img_qkv: torch.Tensor,
    txt_wq: torch.Tensor,
    txt_wk: torch.Tensor,
    img_wq: torch.Tensor,
    img_wk: torch.Tensor,
    txt_cos: torch.Tensor,
    txt_sin: torch.Tensor,
    img_cos: torch.Tensor,
    img_sin: torch.Tensor,
    heads: int,
    eps: float,
    interleaved: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, Lt, T = txt_qkv.shape
    Li = img_qkv.shape[1]
    D = T // (3 * heads)
    L = Lt + Li
    q = torch.empty((B, L, heads, D), device=txt_qkv.device, dtype=txt_qkv.dtype)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    tv = txt_qkv.view(B, Lt, 3, heads, D)
    iv = img_qkv.view(B, Li, 3, heads, D)
    # One launch per side, shared by q and k.
    _, _, iq_rstd, ik_rstd = _launch_fwd_qk(
        iv[:, :, 0],
        iv[:, :, 1],
        img_wq,
        img_wk,
        img_cos,
        img_sin,
        eps,
        interleaved,
        False,
        copy_x=iv[:, :, 2],
        copy_out=v[:, Lt:],
        outq=q[:, Lt:],
        outk=k[:, Lt:],
    )
    _, _, tq_rstd, tk_rstd = _launch_fwd_qk(
        tv[:, :, 0],
        tv[:, :, 1],
        txt_wq,
        txt_wk,
        txt_cos,
        txt_sin,
        eps,
        interleaved,
        False,
        copy_x=tv[:, :, 2],
        copy_out=v[:, :Lt],
        outq=q[:, :Lt],
        outk=k[:, :Lt],
    )
    return q, k, v, tq_rstd, tk_rstd, iq_rstd, ik_rstd


@_dbl_fwd.register_fake
def _(
    txt_qkv,
    img_qkv,
    txt_wq,
    txt_wk,
    img_wq,
    img_wk,
    txt_cos,
    txt_sin,
    img_cos,
    img_sin,
    heads,
    eps,
    interleaved,
):
    B, Lt, T = txt_qkv.shape
    Li = img_qkv.shape[1]
    D = T // (3 * heads)
    q = torch.empty((B, Lt + Li, heads, D), device=txt_qkv.device, dtype=txt_qkv.dtype)
    rt = torch.empty(B * Lt * heads, device=txt_qkv.device, dtype=torch.float32)
    ri = torch.empty(B * Li * heads, device=txt_qkv.device, dtype=torch.float32)
    return (q, torch.empty_like(q), torch.empty_like(q), rt, torch.empty_like(rt), ri, torch.empty_like(ri))


@custom_op(f"{NS}::qkv_norm_rope_double_backward", mutates_args=())
def _dbl_bwd(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    txt_qkv: torch.Tensor,
    img_qkv: torch.Tensor,
    txt_wq: torch.Tensor,
    txt_wk: torch.Tensor,
    img_wq: torch.Tensor,
    img_wk: torch.Tensor,
    txt_cos: torch.Tensor,
    txt_sin: torch.Tensor,
    img_cos: torch.Tensor,
    img_sin: torch.Tensor,
    tq_rstd: torch.Tensor,
    tk_rstd: torch.Tensor,
    iq_rstd: torch.Tensor,
    ik_rstd: torch.Tensor,
    heads: int,
    interleaved: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, Lt, T = txt_qkv.shape
    Li = img_qkv.shape[1]
    D = T // (3 * heads)
    d_txt = torch.empty_like(txt_qkv, memory_format=torch.contiguous_format)
    d_img = torch.empty_like(img_qkv, memory_format=torch.contiguous_format)
    tsrc, tdst = txt_qkv.view(B, Lt, 3, heads, D), d_txt.view(B, Lt, 3, heads, D)
    isrc, idst = img_qkv.view(B, Li, 3, heads, D), d_img.view(B, Li, 3, heads, D)

    def side(dqs, dks, dvs, src, dst, wq, wk, cos, sin, qr, kr):
        # Fold dv's write into the q-backward launch's own
        # grid instead of a separate copy_() kernel, mirroring the forward's
        # COPY_ROW. dst[:, :, 0] (dx target) and dst[:, :, 2] (copy target) are
        # slices of the same contiguous d_txt/d_img tensor, so they share
        # token/head strides; only the base pointer differs.
        cx = dict(copy_x=dvs, copy_out=dst[:, :, 2])
        dwqp = _launch_bwd(
            dqs,
            src[:, :, 0],
            wq,
            cos,
            sin,
            qr,
            dst[:, :, 0],
            interleaved,
            traceable=False,
            round_grad=True,
            **cx,
        )
        dwkp = _launch_bwd(
            dks, src[:, :, 1], wk, cos, sin, kr, dst[:, :, 1], interleaved, traceable=False, round_grad=True
        )
        return dwqp, dwkp

    dwqp_i, dwkp_i = side(
        dq[:, Lt:], dk[:, Lt:], dv[:, Lt:], isrc, idst, img_wq, img_wk, img_cos, img_sin, iq_rstd, ik_rstd
    )
    dwqp_t, dwkp_t = side(
        dq[:, :Lt], dk[:, :Lt], dv[:, :Lt], tsrc, tdst, txt_wq, txt_wk, txt_cos, txt_sin, tq_rstd, tk_rstd
    )
    return d_txt, d_img, dwqp_t, dwkp_t, dwqp_i, dwkp_i


@_dbl_bwd.register_fake
def _(
    dq,
    dk,
    dv,
    txt_qkv,
    img_qkv,
    txt_wq,
    txt_wk,
    img_wq,
    img_wk,
    txt_cos,
    txt_sin,
    img_cos,
    img_sin,
    tq_rstd,
    tk_rstd,
    iq_rstd,
    ik_rstd,
    heads,
    interleaved,
):
    D = txt_qkv.shape[-1] // (3 * heads)
    Mt = txt_qkv.numel() // (3 * D)
    Mi = img_qkv.numel() // (3 * D)
    pt = torch.empty(
        triton.cdiv(Mt, _mix_order_split(Mt, txt_qkv.device)), D, device=txt_qkv.device, dtype=torch.float32
    )
    pi = torch.empty(
        triton.cdiv(Mi, _mix_order_split(Mi, img_qkv.device)), D, device=img_qkv.device, dtype=torch.float32
    )
    return (
        torch.empty_like(txt_qkv, memory_format=torch.contiguous_format),
        torch.empty_like(img_qkv, memory_format=torch.contiguous_format),
        pt,
        torch.empty_like(pt),
        pi,
        torch.empty_like(pi),
    )


def _dbl_setup(ctx, inputs, output):
    (
        txt_qkv,
        img_qkv,
        txt_wq,
        txt_wk,
        img_wq,
        img_wk,
        txt_cos,
        txt_sin,
        img_cos,
        img_sin,
        heads,
        eps,
        interleaved,
    ) = inputs
    ctx.save_for_backward(
        txt_qkv, img_qkv, txt_wq, txt_wk, img_wq, img_wk, txt_cos, txt_sin, img_cos, img_sin, *output[3:]
    )
    ctx.heads = heads
    ctx.interleaved = interleaved


def _dbl_backward(ctx, dq, dk, dv, *_r):
    (
        txt_qkv,
        img_qkv,
        txt_wq,
        txt_wk,
        img_wq,
        img_wk,
        txt_cos,
        txt_sin,
        img_cos,
        img_sin,
        tq_rstd,
        tk_rstd,
        iq_rstd,
        ik_rstd,
    ) = ctx.saved_tensors
    d_txt, d_img, pqt, pkt, pqi, pki = _dbl_bwd(
        dq,
        dk,
        dv,
        txt_qkv,
        img_qkv,
        txt_wq,
        txt_wk,
        img_wq,
        img_wk,
        txt_cos,
        txt_sin,
        img_cos,
        img_sin,
        tq_rstd,
        tk_rstd,
        iq_rstd,
        ik_rstd,
        ctx.heads,
        ctx.interleaved,
    )
    return (
        d_txt,
        d_img,
        pqt.sum(0).to(txt_wq.dtype),
        pkt.sum(0).to(txt_wk.dtype),
        pqi.sum(0).to(img_wq.dtype),
        pki.sum(0).to(img_wk.dtype),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_dbl_fwd.register_autograd(_dbl_backward, setup_context=_dbl_setup)


def _backend_attention(q, k, v):
    from primus.backends.diffusion.models.flux.math import backend_attention

    return backend_attention(q=q, k=k, v=v, dtype=q.dtype)


def double_attention(txt_qkv, img_qkv, txt_wq, txt_wk, img_wq, img_wk, tables, heads):
    """[B, Lt, 3*H*D] and [B, Li, 3*H*D] qkv -> (txt_attn, img_attn), each [B, L, H*D]."""
    L_txt = txt_qkv.shape[1]
    q, k, v = _dbl_fwd(
        txt_qkv,
        img_qkv,
        txt_wq,
        txt_wk,
        img_wq,
        img_wk,
        tables.txt_cos,
        tables.txt_sin,
        tables.img_cos,
        tables.img_sin,
        heads,
        default_eps(txt_qkv.dtype),
        True,
    )[:3]
    if _DOUBLE_SPLIT and L_txt % 128 == 0:
        from primus.backends.diffusion.attention import flydsl_flux

        if flydsl_flux.eligible(q, k, v, None, None, 0.0, None, None, False, (-1, -1)):
            txt, img = flydsl_flux.flydsl_flux_attention_split(q, k, v, L_txt)
            return txt.reshape(txt.shape[0], L_txt, -1), img.reshape(img.shape[0], img.shape[1], -1)
    attn = _backend_attention(q, k, v)
    attn = attn.reshape(attn.shape[0], attn.shape[1], -1)
    return attn[:, :L_txt], attn[:, L_txt:]


def single_attention(qkv, wq, wk, cos, sin, heads):
    """[B, L, 3*H*D] qkv -> [B, L, H*D]."""
    q, k, v = _SingleQKVFn.apply(qkv, wq, wk, cos, sin, heads, default_eps(qkv.dtype), True)
    x = _backend_attention(q, k, v)
    return x.reshape(x.shape[0], x.shape[1], -1)
