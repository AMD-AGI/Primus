###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""hcfuse (2026-11-22): HC launches merged now that HC fusions may change numerics within the loss gate.

PRIMUS_HCFUSE_BWD=1  ``_hcf_norm_pre_bwd_kernel``: the folded-norm backward (``_hc_norm_dpre_bwd_kernel``) and the
    HC row math (``_hc_pre_bwd_kernel``) in ONE launch with two kinds of programs:
      * Sinkhorn programs (first, RS rows each): sum the post-side partials (dpost / dcomb^T, the same sequential order
        as _hc_pre_bwd), Sinkhorn + softmax/sigmoid VJP of post / comb, write their 20 g columns (bf16 hi/lo, both
        layouts), coef_b = rstd^3 <dl_pc, raw_pc> / KD, and the per-row dscale / dbase partials of post / comb.
        None of this depends on the norm backward, so it runs beside it instead of after it.
      * norm programs (persistent rows): dh = rmsnorm_bwd(dy) as before, dpre_k = <x_k, dh> reduced over the whole row
        in-program (was 16 column-tile partials), then the pre-sigmoid VJP: the 4 pre g columns, coef_a, and the
        pre dscale / dbase partials.
    B3X adds coef_a + coef_b. Numerics: dpre summation order (row tree instead of 16 sequential 256-column partials),
    coef = rstd^3 (c_a + c_b) / KD split in two, dgamma partial grouping (ppc); everything else as before.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .hc_norm_fold import _collapse_bf16, _ld_pre4, _ld_rows4


def _flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def _mode(name: str) -> int:
    return int(os.environ.get(name, "0") or "0")


def hcfuse_bwd_enabled() -> bool:
    """PRIMUS_HCFUSE_BWD: 1 = one launch (Sinkhorn programs + norm programs); 2 = two kernels, the Sinkhorn-VJP kernel
    on a side stream concurrent with the norm kernel (joined before B3X); 3 = two kernels on the main stream."""
    return _mode("PRIMUS_HCFUSE_BWD") > 0


def hcfuse_fwd_enabled() -> bool:
    """PRIMUS_HCFUSE_FWD: 1 = one launch (Sinkhorn programs + collapse rows); 2 = two kernels, Sinkhorn on a side
    stream concurrent with the collapse kernel (joined before returning); 3 = two kernels on the main stream."""
    return _mode("PRIMUS_HCFUSE_FWD") > 0


def hcfuse_tail_enabled() -> bool:
    """PRIMUS_HCFUSE_TAIL=1: HCPreFn(L).backward also runs HCPostFn(L-1)'s expand backward (post_bwd2d) on the main stream,
    with dnew = its own dx, while the tail (dW / dscale / dbase / dgamma -> main_grad) runs on a side stream; both are
    joined before returning (main_grad is written before the DDP hook). HCPostFn(L-1).backward then returns the
    precomputed dout if it receives that same dx tensor (else it recomputes). Bit-identical."""
    return _mode("PRIMUS_HCFUSE_TAIL") > 0


_SIDE = {}


def _side_stream(dev):
    key = torch.device(dev).index
    if key not in _SIDE:
        _SIDE[key] = torch.cuda.Stream(device=dev)
    return _SIDE[key]


def _cfg(name, default):
    """PRIMUS_HCFUSE_<name>_CFG="k=v,k=v" over a default dict of ints."""
    c = dict(default)
    v = os.environ.get(f"PRIMUS_HCFUSE_{name}_CFG", "").strip()
    if v:
        for kv in v.split(","):
            k, x = kv.split("=")
            c[k.strip()] = int(x)
    return c


BWD_CFG = dict(ppc=2, rs=32, nw=8, rb=1, srs=1, snw=1)  # srs/snw: Sinkhorn kernel of modes 2/3
FWD_CFG = dict(rs=32, nw=8, srs=1, snw=1)


@triton.jit
def _st_g2(GH, GL, GTH, GTL, off_rm, off_t, v, m):
    h = v.to(tl.bfloat16)
    lo = (v - h.to(tl.float32)).to(tl.bfloat16)
    tl.store(GH + off_rm, h, mask=m)
    tl.store(GL + off_rm, lo, mask=m)
    tl.store(GTH + off_t, h, mask=m)
    tl.store(GTL + off_t, lo, mask=m)


@triton.jit
def _hcf_sk_bwd(pid_s, DPOSTP, DCOMBTP, RAW, RSTD, STATES, SCALE, BASE, GH, GL, GTH, GTL, COEF_B, PART, M, D,
                NC: tl.constexpr, EPS: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, N_ITERS: tl.constexpr,
                RS: tl.constexpr):
    rows = pid_s * RS + tl.arange(0, RS)
    rmask = rows < M
    rows64 = rows.to(tl.int64)
    kk = tl.arange(0, K)
    KK: tl.constexpr = K * K
    KD = K * D
    r2 = rows64[:, None]
    m2 = rmask[:, None]
    r3 = rows64[:, None, None]
    m3 = rmask[:, None, None]
    i3 = kk[None, :, None]
    j3 = kk[None, None, :]
    dpost = tl.zeros((RS, K), dtype=tl.float32)
    dm_ = tl.zeros((RS, K, K), dtype=tl.float32)
    for c in tl.static_range(NC):  # unrolled: every partial load in flight, summed in _hc_pre_bwd's order
        dpost += tl.load(DPOSTP + c * M * K + r2 * K + kk[None, :], mask=m2, other=0.0)
        dm_ += tl.load(DCOMBTP + c * M * KK + r3 * KK + j3 * K + i3, mask=m3, other=0.0)
    raw_post = tl.load(RAW + r2 * BN + K + kk[None, :], mask=m2, other=0.0)
    raw_comb = tl.load(RAW + r3 * BN + 2 * K + i3 * K + j3, mask=m3, other=0.0)
    rstd = tl.load(RSTD + rows64, mask=rmask, other=0.0)
    s1 = tl.load(SCALE + 1).to(tl.float32)
    s2 = tl.load(SCALE + 2).to(tl.float32)
    b_post = tl.load(BASE + K + kk).to(tl.float32)
    b_comb = tl.load(BASE + 2 * K + kk[:, None] * K + kk[None, :]).to(tl.float32)
    l_post = raw_post * rstd[:, None]
    l_comb = raw_comb * rstd[:, None, None]
    post_sig = tl.sigmoid(l_post * s1 + b_post[None, :])
    comb_logit = l_comb * s2 + b_comb[None, :, :]
    cmax = tl.max(comb_logit, axis=2, keep_dims=True)
    cexp = tl.exp(comb_logit - cmax)
    comb_sm = cexp / tl.sum(cexp, axis=2, keep_dims=True)

    SROW: tl.constexpr = 2 * N_ITERS * KK
    sb = STATES + r3 * SROW + i3 * K + j3
    for i in tl.static_range(N_ITERS - 1):  # unrolled (as _hc_pre_bwd): the state loads do not depend on dm_
        col_step = 2 * N_ITERS - 1 - 2 * i
        mb = tl.load(sb + (col_step - 1) * KK, mask=m3, other=0.0)
        ma = tl.load(sb + col_step * KK, mask=m3, other=0.0)
        sc = tl.sum(mb, axis=1, keep_dims=True) + EPS
        dot = tl.sum(dm_ * ma, axis=1, keep_dims=True)
        dm_ = (dm_ - dot) / sc
        row_step = col_step - 1
        mb = tl.load(sb + (row_step - 1) * KK, mask=m3, other=0.0)
        ma = tl.load(sb + row_step * KK, mask=m3, other=0.0)
        sr = tl.sum(mb, axis=2, keep_dims=True) + EPS
        dot = tl.sum(dm_ * ma, axis=2, keep_dims=True)
        dm_ = (dm_ - dot) / sr
    mb = tl.load(sb, mask=m3, other=0.0)
    ma = tl.load(sb + KK, mask=m3, other=0.0)
    sp = tl.sum(mb, axis=1, keep_dims=True) + EPS
    dot = tl.sum(dm_ * ma, axis=1, keep_dims=True)
    dm_ = (dm_ - dot) / sp

    d_post_l = dpost * 2.0 * post_sig * (1.0 - post_sig)
    d_comb_l = comb_sm * (dm_ - tl.sum(dm_ * comb_sm, axis=2, keep_dims=True))
    dl_post = d_post_l * s1
    dl_comb = d_comb_l * s2
    _st_g2(GH, GL, GTH, GTL, r2 * BN + K + kk[None, :], (K + kk[None, :]) * M + r2, dl_post * rstd[:, None], m2)
    _st_g2(GH, GL, GTH, GTL, r3 * BN + 2 * K + i3 * K + j3, (2 * K + i3 * K + j3) * M + r3,
           dl_comb * rstd[:, None, None], m3)
    NPAD: tl.constexpr = BN - (2 + K) * K
    if NPAD > 0:
        pc = tl.arange(0, NPAD)
        _st_g2(GH, GL, GTH, GTL, r2 * BN + (2 + K) * K + pc[None, :], ((2 + K) * K + pc[None, :]) * M + r2,
               tl.zeros((RS, NPAD), dtype=tl.float32), m2)
    cb = tl.sum(dl_post * raw_post, axis=1) + tl.sum(tl.sum(dl_comb * raw_comb, axis=2), axis=1)
    tl.store(COEF_B + rows64, rstd * rstd * rstd * cb / KD, mask=rmask)
    # per-row dscale / dbase partials, PART[j, row] (the tail sums each row of PART)
    tl.store(PART + 1 * M + rows64, tl.sum(l_post * d_post_l, axis=1), mask=rmask)
    tl.store(PART + 2 * M + rows64, tl.sum(tl.sum(l_comb * d_comb_l, axis=2), axis=1), mask=rmask)
    tl.store(PART + (3 + K + kk[None, :]) * M + r2, d_post_l, mask=m2)
    tl.store(PART + (3 + 2 * K + i3 * K + j3) * M + r3, d_comb_l, mask=m3)


@triton.jit
def _hcf_norm_bwd(pid, NPROG, X, PRE, GAMMA, RSTDN, DY, DH, DGP, RAW, RSTD, SCALE, BASE, GH, GL, GTH, GTL, COEF_A,
                  PART, M, D, BD: tl.constexpr, RB: tl.constexpr, BN: tl.constexpr):
    """Persistent rows: dh = rmsnorm_bwd(dy) (as _hc_norm_dpre_bwd_kernel), dpre over the whole row, pre VJP."""
    K: tl.constexpr = 4
    cols = tl.arange(0, BD)
    cm = cols < D
    kk = tl.arange(0, K)
    g = tl.load(GAMMA + cols, mask=cm, other=0.0).to(tl.float32)
    s0 = tl.load(SCALE + 0).to(tl.float32)
    b_pre = tl.load(BASE + kk).to(tl.float32)
    KD = K * D
    dg = tl.zeros((RB, BD), dtype=tl.float32)
    STEP = NPROG * RB
    rows = pid * RB + tl.arange(0, RB)
    rm = rows < M
    t64 = rows.to(tl.int64)
    m2 = rm[:, None] & cm[None, :]
    a0, a1, a2, a3 = _ld_rows4(X, t64, cols, m2, D)
    p0, p1, p2, p3 = _ld_pre4(PRE, t64, rm)
    dyb = tl.load(DY + t64[:, None] * D + cols[None, :], mask=m2, other=0.0)
    rs = tl.load(RSTDN + rows, mask=rm, other=0.0)
    rw = tl.load(RAW + t64[:, None] * BN + kk[None, :], mask=rm[:, None], other=0.0)
    rh = tl.load(RSTD + t64, mask=rm, other=0.0)
    for r0 in range(pid * RB, M, STEP):
        x0, x1, x2, x3 = a0, a1, a2, a3
        q0, q1, q2, q3 = p0, p1, p2, p3
        dy = dyb.to(tl.float32)
        rstd = rs
        raw_pre = rw
        rstd_h = rh
        tc, rmc, m2c = t64, rm, m2
        # prefetch the next row block
        rows = r0 + STEP + tl.arange(0, RB)
        rm = rows < M
        t64 = rows.to(tl.int64)
        m2 = rm[:, None] & cm[None, :]
        a0, a1, a2, a3 = _ld_rows4(X, t64, cols, m2, D)
        p0, p1, p2, p3 = _ld_pre4(PRE, t64, rm)
        dyb = tl.load(DY + t64[:, None] * D + cols[None, :], mask=m2, other=0.0)
        rs = tl.load(RSTDN + rows, mask=rm, other=0.0)
        rw = tl.load(RAW + t64[:, None] * BN + kk[None, :], mask=rm[:, None], other=0.0)
        rh = tl.load(RSTD + t64, mask=rm, other=0.0)

        h = _collapse_bf16(x0, x1, x2, x3, q0, q1, q2, q3)
        xh = h * rstd[:, None]
        dxh = dy * g[None, :]
        mm = tl.sum(dxh * xh, axis=1) / D
        dhb = ((dxh - xh * mm[:, None]) * rstd[:, None]).to(DH.dtype.element_ty)
        tl.store(DH + tc[:, None] * D + cols[None, :], dhb, mask=m2c)
        dhf = dhb.to(tl.float32)
        d0 = tl.sum(x0.to(tl.float32) * dhf, axis=1)
        d1 = tl.sum(x1.to(tl.float32) * dhf, axis=1)
        d2 = tl.sum(x2.to(tl.float32) * dhf, axis=1)
        d3 = tl.sum(x3.to(tl.float32) * dhf, axis=1)
        dg += dy * xh
        k2 = kk[None, :]
        dpre = tl.where(k2 == 0, d0[:, None], tl.where(k2 == 1, d1[:, None], tl.where(k2 == 2, d2[:, None],
                                                                                         d3[:, None])))
        l_pre = raw_pre * rstd_h[:, None]
        pre_sig = tl.sigmoid(l_pre * s0 + b_pre[None, :])
        d_pre_l = dpre * pre_sig * (1.0 - pre_sig)
        dl_pre = d_pre_l * s0
        rmk = rmc[:, None]
        _st_g2(GH, GL, GTH, GTL, tc[:, None] * BN + k2, k2 * M + tc[:, None], dl_pre * rstd_h[:, None], rmk)
        ca = tl.sum(dl_pre * raw_pre, axis=1)
        tl.store(COEF_A + tc, rstd_h * rstd_h * rstd_h * ca / KD, mask=rmc)
        tl.store(PART + tc, tl.sum(l_pre * d_pre_l, axis=1), mask=rmc)
        tl.store(PART + (3 + k2) * M + tc[:, None], d_pre_l, mask=rmk)
    tl.store(DGP + pid * D + cols, tl.sum(dg, axis=0), mask=cm)


@triton.jit
def _hcf_norm_pre_bwd_kernel(X, PRE, GAMMA, RSTDN, DY, DH, DGP, RAW, RSTD, STATES, SCALE, BASE, DPOSTP, DCOMBTP,
                             GH, GL, GTH, GTL, COEF_A, COEF_B, PART, M, D, NPROG_N, NPROG_S,
                             NC: tl.constexpr, EPS: tl.constexpr, BN: tl.constexpr, N_ITERS: tl.constexpr,
                             RS: tl.constexpr, BD: tl.constexpr, RB: tl.constexpr):
    pid = tl.program_id(0)
    if pid < NPROG_S:
        _hcf_sk_bwd(pid, DPOSTP, DCOMBTP, RAW, RSTD, STATES, SCALE, BASE, GH, GL, GTH, GTL, COEF_B, PART, M, D,
                    NC, EPS, 4, BN, N_ITERS, RS)
    else:
        _hcf_norm_bwd(pid - NPROG_S, NPROG_N, X, PRE, GAMMA, RSTDN, DY, DH, DGP, RAW, RSTD, SCALE, BASE,
                      GH, GL, GTH, GTL, COEF_A, PART, M, D, BD, RB, BN)


# ------------------------------------------------------------------------------------------------
# PRIMUS_HCFUSE_FWD: _hc_pre_fwd (finalize + Sinkhorn) and the folded-norm collapse in one launch
# ------------------------------------------------------------------------------------------------
@triton.jit
def _hcf_sk_fwd(pid_s, P_RAW, P_SS, SCALE, BASE, RAW, RSTD, STATES, POST, COMBT, M, D, eps_rms,
                EPS: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, NS: tl.constexpr, N_ITERS: tl.constexpr,
                RS: tl.constexpr):
    """_hc_pre_fwd_kernel for RS rows without the collapse and the pre store (the collapse programs do both)."""
    rows = pid_s * RS + tl.arange(0, RS)
    rmask = rows < M
    rows64 = rows.to(tl.int64)
    kk = tl.arange(0, K)
    KK: tl.constexpr = K * K
    r2 = rows64[:, None]
    m2 = rmask[:, None]
    r3 = rows64[:, None, None]
    m3 = rmask[:, None, None]
    i3 = kk[None, :, None]
    j3 = kk[None, None, :]
    raw_pre = tl.zeros((RS, K), dtype=tl.float32)
    raw_post = tl.zeros((RS, K), dtype=tl.float32)
    raw_comb = tl.zeros((RS, K, K), dtype=tl.float32)
    ss = tl.zeros((RS,), dtype=tl.float32)
    for s in tl.static_range(NS):  # unrolled; same sequential order as _hc_pre_fwd_kernel
        pb = P_RAW + s * M * BN
        raw_pre += tl.load(pb + r2 * BN + kk[None, :], mask=m2, other=0.0)
        raw_post += tl.load(pb + r2 * BN + K + kk[None, :], mask=m2, other=0.0)
        raw_comb += tl.load(pb + r3 * BN + 2 * K + i3 * K + j3, mask=m3, other=0.0)
        ss += tl.load(P_SS + s * M + rows64, mask=rmask, other=0.0)
    KD = K * D
    rstd = 1.0 / tl.sqrt(ss / KD + eps_rms)
    tl.store(RSTD + rows64, rstd, mask=rmask)
    tl.store(RAW + r2 * BN + kk[None, :], raw_pre, mask=m2)
    tl.store(RAW + r2 * BN + K + kk[None, :], raw_post, mask=m2)
    tl.store(RAW + r3 * BN + 2 * K + i3 * K + j3, raw_comb, mask=m3)
    s1 = tl.load(SCALE + 1).to(tl.float32)
    s2 = tl.load(SCALE + 2).to(tl.float32)
    b_post = tl.load(BASE + K + kk).to(tl.float32)
    b_comb = tl.load(BASE + 2 * K + kk[:, None] * K + kk[None, :]).to(tl.float32)
    post_logit = (raw_post * rstd[:, None]) * s1 + b_post[None, :]
    comb_logit = (raw_comb * rstd[:, None, None]) * s2 + b_comb[None, :, :]
    post = 2.0 * tl.sigmoid(post_logit)
    cmax = tl.max(comb_logit, axis=2, keep_dims=True)
    cexp = tl.exp(comb_logit - cmax)
    m = cexp / tl.sum(cexp, axis=2, keep_dims=True) + EPS
    SROW: tl.constexpr = 2 * N_ITERS * KK
    sb = STATES + r3 * SROW + i3 * K + j3
    tl.store(sb, m, mask=m3)
    m = m / (tl.sum(m, axis=1, keep_dims=True) + EPS)
    tl.store(sb + KK, m, mask=m3)
    for it in range(N_ITERS - 1):  # runtime loop: stores only, nothing to hoist
        m = m / (tl.sum(m, axis=2, keep_dims=True) + EPS)
        tl.store(sb + (2 + 2 * it) * KK, m, mask=m3)
        m = m / (tl.sum(m, axis=1, keep_dims=True) + EPS)
        tl.store(sb + (3 + 2 * it) * KK, m, mask=m3)
    tl.store(POST + r2 * K + kk[None, :], post.to(tl.bfloat16), mask=m2)
    tl.store(COMBT + r3 * KK + j3 * K + i3, m.to(tl.bfloat16), mask=m3)


@triton.jit
def _hcf_collapse_fwd(row0, X, P_RAW, P_SS, SCALE, BASE, GAMMA, PRE, Y, RSTDN, M, D, eps_rms, norm_eps,
                      EPS: tl.constexpr, BN: tl.constexpr, NS: tl.constexpr, BD: tl.constexpr):
    """One row: pre from the rms-linear partials (as _hc_pre_fwd_kernel), then _hc_collapse_norm_fwd_kernel."""
    K: tl.constexpr = 4
    rows = row0 + tl.arange(0, 1)
    rm = rows < M
    t64 = rows.to(tl.int64)
    cols = tl.arange(0, BD)
    cm = cols < D
    m2 = rm[:, None] & cm[None, :]
    x0, x1, x2, x3 = _ld_rows4(X, t64, cols, m2, D)  # streaming loads first; the partial loads overlap them
    kk = tl.arange(0, K)
    raw_pre = tl.zeros((1, K), dtype=tl.float32)
    ss = tl.zeros((1,), dtype=tl.float32)
    for s in tl.static_range(NS):
        raw_pre += tl.load(P_RAW + s * M * BN + t64[:, None] * BN + kk[None, :], mask=rm[:, None], other=0.0)
        ss += tl.load(P_SS + s * M + t64, mask=rm, other=0.0)
    g = tl.load(GAMMA + cols, mask=cm, other=0.0).to(tl.float32)
    s0 = tl.load(SCALE + 0).to(tl.float32)
    b_pre = tl.load(BASE + kk).to(tl.float32)
    KD = K * D
    rstd = 1.0 / tl.sqrt(ss / KD + eps_rms)
    pre = tl.sigmoid((raw_pre * rstd[:, None]) * s0 + b_pre[None, :]) + EPS
    pre_b = pre.to(tl.bfloat16)
    tl.store(PRE + t64[:, None] * K + kk[None, :], pre_b, mask=rm[:, None])
    pf = pre_b.to(tl.float32)
    k2 = kk[None, :]
    p0 = tl.sum(tl.where(k2 == 0, pf, 0.0), axis=1)
    p1 = tl.sum(tl.where(k2 == 1, pf, 0.0), axis=1)
    p2 = tl.sum(tl.where(k2 == 2, pf, 0.0), axis=1)
    p3 = tl.sum(tl.where(k2 == 3, pf, 0.0), axis=1)
    h = _collapse_bf16(x0, x1, x2, x3, p0, p1, p2, p3)
    rn = 1.0 / tl.sqrt(tl.sum(h * h, axis=1) / D + norm_eps)
    y = (h * rn[:, None] * g[None, :]).to(Y.dtype.element_ty)
    tl.store(Y + t64[:, None] * D + cols[None, :], y, mask=m2)
    tl.store(RSTDN + rows, rn, mask=rm)


@triton.jit
def _hcf_pre_collapse_fwd_kernel(X, P_RAW, P_SS, SCALE, BASE, GAMMA, RAW, RSTD, STATES, PRE, POST, COMBT, Y, RSTDN,
                                 M, D, NPROG_S, eps_rms, norm_eps, EPS: tl.constexpr, BN: tl.constexpr,
                                 NS: tl.constexpr, N_ITERS: tl.constexpr, RS: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    if pid < NPROG_S:
        _hcf_sk_fwd(pid, P_RAW, P_SS, SCALE, BASE, RAW, RSTD, STATES, POST, COMBT, M, D, eps_rms,
                    EPS, 4, BN, NS, N_ITERS, RS)
    else:
        _hcf_collapse_fwd(pid - NPROG_S, X, P_RAW, P_SS, SCALE, BASE, GAMMA, PRE, Y, RSTDN, M, D, eps_rms, norm_eps,
                          EPS, BN, NS, BD)


def pre_collapse_fwd(x3, p_raw, p_ss, n_split, scale_c, base_c, gamma, eps, n_iters, norm_eps):
    """-> raw [M, BN], rstd [M], states, pre, post [M, K] bf16, combt [M, K, K] bf16, y [M, D], rstd_n [M]."""
    M, K, D = x3.shape
    assert K == 4
    BN = p_raw.shape[2]
    BD = triton.next_power_of_2(D)
    assert BD == D
    c = _cfg("FWD", FWD_CFG)
    dev = x3.device
    raw = torch.empty((M, BN), dtype=torch.float32, device=dev)
    rstd = torch.empty((M,), dtype=torch.float32, device=dev)
    states = torch.empty((M, 2 * n_iters, K, K), dtype=torch.float32, device=dev)
    pre = torch.empty((M, K), dtype=x3.dtype, device=dev)
    post = torch.empty((M, K), dtype=x3.dtype, device=dev)
    combt = torch.empty((M, K, K), dtype=x3.dtype, device=dev)
    y = torch.empty((M, D), dtype=x3.dtype, device=dev)
    rstd_n = torch.empty((M,), dtype=torch.float32, device=dev)
    mode = _mode("PRIMUS_HCFUSE_FWD")
    if mode == 1:
        nprog_s = triton.cdiv(M, c["rs"])
        _hcf_pre_collapse_fwd_kernel[(nprog_s + M,)](
            x3, p_raw, p_ss, scale_c, base_c, gamma, raw, rstd, states, pre, post, combt, y, rstd_n,
            M, D, nprog_s, float(eps), float(norm_eps),
            EPS=float(eps), BN=BN, NS=int(n_split), N_ITERS=int(n_iters), RS=c["rs"], BD=BD, num_warps=c["nw"],
        )
        return raw, rstd, states, pre, post, combt, y, rstd_n
    main = torch.cuda.current_stream(dev)
    side = _side_stream(dev) if mode == 2 else main
    if mode == 2:
        side.wait_stream(main)
    with torch.cuda.stream(side):
        _hcf_sk_fwd_kernel[(triton.cdiv(M, c["srs"]),)](
            p_raw, p_ss, scale_c, base_c, raw, rstd, states, post, combt, M, D, float(eps),
            EPS=float(eps), BN=BN, NS=int(n_split), N_ITERS=int(n_iters), RS=c["srs"], num_warps=c["snw"],
        )
    _hcf_collapse_fwd_kernel[(M,)](
        x3, p_raw, p_ss, scale_c, base_c, gamma, pre, y, rstd_n, M, D, float(eps), float(norm_eps),
        EPS=float(eps), BN=BN, NS=int(n_split), BD=BD, num_warps=c["nw"],
    )
    if mode == 2:
        main.wait_stream(side)
    return raw, rstd, states, pre, post, combt, y, rstd_n


@triton.jit
def _hcf_sk_bwd_kernel(DPOSTP, DCOMBTP, RAW, RSTD, STATES, SCALE, BASE, GH, GL, GTH, GTL, COEF_B, PART, M, D,
                       NC: tl.constexpr, EPS: tl.constexpr, BN: tl.constexpr, N_ITERS: tl.constexpr, RS: tl.constexpr):
    _hcf_sk_bwd(tl.program_id(0), DPOSTP, DCOMBTP, RAW, RSTD, STATES, SCALE, BASE, GH, GL, GTH, GTL, COEF_B, PART,
                M, D, NC, EPS, 4, BN, N_ITERS, RS)


@triton.jit
def _hcf_norm_bwd_kernel(X, PRE, GAMMA, RSTDN, DY, DH, DGP, RAW, RSTD, SCALE, BASE, GH, GL, GTH, GTL, COEF_A, PART,
                         M, D, NPROG_N, BD: tl.constexpr, RB: tl.constexpr, BN: tl.constexpr):
    _hcf_norm_bwd(tl.program_id(0), NPROG_N, X, PRE, GAMMA, RSTDN, DY, DH, DGP, RAW, RSTD, SCALE, BASE,
                  GH, GL, GTH, GTL, COEF_A, PART, M, D, BD, RB, BN)


@triton.jit
def _hcf_sk_fwd_kernel(P_RAW, P_SS, SCALE, BASE, RAW, RSTD, STATES, POST, COMBT, M, D, eps_rms,
                       EPS: tl.constexpr, BN: tl.constexpr, NS: tl.constexpr, N_ITERS: tl.constexpr, RS: tl.constexpr):
    _hcf_sk_fwd(tl.program_id(0), P_RAW, P_SS, SCALE, BASE, RAW, RSTD, STATES, POST, COMBT, M, D, eps_rms,
                EPS, 4, BN, NS, N_ITERS, RS)


@triton.jit
def _hcf_collapse_fwd_kernel(X, P_RAW, P_SS, SCALE, BASE, GAMMA, PRE, Y, RSTDN, M, D, eps_rms, norm_eps,
                             EPS: tl.constexpr, BN: tl.constexpr, NS: tl.constexpr, BD: tl.constexpr):
    _hcf_collapse_fwd(tl.program_id(0), X, P_RAW, P_SS, SCALE, BASE, GAMMA, PRE, Y, RSTDN, M, D, eps_rms, norm_eps,
                      EPS, BN, NS, BD)


def _n_cu(dev) -> int:
    try:
        return torch.cuda.get_device_properties(dev).multi_processor_count
    except Exception:
        return 256


def norm_pre_bwd(x3, pre, gamma, rstd_n, dy, raw, rstd, states, scale_c, base_c, dpost_p, dcombt_p, nc, eps, n_iters):
    """-> dh [M, D] bf16, gh, gl [M, BN] / gth, gtl [BN, M] bf16, coef_a, coef_b [M] fp32, part [3+N, M] fp32,
    dgp [P, D] fp32 dgamma partials."""
    M, K, D = x3.shape
    assert K == 4
    BN = raw.shape[1]
    N = (2 + K) * K
    BD = triton.next_power_of_2(D)
    assert BD == D
    c = _cfg("BWD", BWD_CFG)
    dev = x3.device
    dh = torch.empty((M, D), dtype=x3.dtype, device=dev)
    gh = torch.empty((M, BN), dtype=torch.bfloat16, device=dev)
    glo = torch.empty((M, BN), dtype=torch.bfloat16, device=dev)
    gth = torch.empty((BN, M), dtype=torch.bfloat16, device=dev)
    gtl = torch.empty((BN, M), dtype=torch.bfloat16, device=dev)
    coef_a = torch.empty((M,), dtype=torch.float32, device=dev)
    coef_b = torch.empty((M,), dtype=torch.float32, device=dev)
    part = torch.empty((3 + N, M), dtype=torch.float32, device=dev)
    rb = c["rb"]
    nprog_n = min(triton.cdiv(M, rb), c["ppc"] * _n_cu(dev))
    nprog_s = triton.cdiv(M, c["rs"])
    dgp = torch.empty((nprog_n, D), dtype=torch.float32, device=dev)
    mode = _mode("PRIMUS_HCFUSE_BWD")
    if mode == 1:
        _hcf_norm_pre_bwd_kernel[(nprog_s + nprog_n,)](
            x3, pre, gamma, rstd_n, dy, dh, dgp, raw, rstd, states, scale_c, base_c, dpost_p, dcombt_p,
            gh, glo, gth, gtl, coef_a, coef_b, part, M, D, nprog_n, nprog_s,
            NC=nc, EPS=float(eps), BN=BN, N_ITERS=int(n_iters), RS=c["rs"], BD=BD, RB=rb, num_warps=c["nw"],
        )
        return dh, gh, glo, gth, gtl, coef_a, coef_b, part, dgp
    main = torch.cuda.current_stream(dev)
    side = _side_stream(dev) if mode == 2 else main
    if mode == 2:
        side.wait_stream(main)
    with torch.cuda.stream(side):
        _hcf_sk_bwd_kernel[(triton.cdiv(M, c["srs"]),)](
            dpost_p, dcombt_p, raw, rstd, states, scale_c, base_c, gh, glo, gth, gtl, coef_b, part, M, D,
            NC=nc, EPS=float(eps), BN=BN, N_ITERS=int(n_iters), RS=c["srs"], num_warps=c["snw"],
        )
    _hcf_norm_bwd_kernel[(nprog_n,)](
        x3, pre, gamma, rstd_n, dy, dh, dgp, raw, rstd, scale_c, base_c, gh, glo, gth, gtl, coef_a, part,
        M, D, nprog_n, BD=BD, RB=rb, BN=BN, num_warps=c["nw"],
    )
    if mode == 2:
        main.wait_stream(side)
    return dh, gh, glo, gth, gtl, coef_a, coef_b, part, dgp


__all__ = ["hcfuse_bwd_enabled", "hcfuse_fwd_enabled", "hcfuse_tail_enabled", "side_stream", "norm_pre_bwd", "pre_collapse_fwd"]
side_stream = _side_stream
