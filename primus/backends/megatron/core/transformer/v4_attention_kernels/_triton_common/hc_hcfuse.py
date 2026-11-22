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


def tail_mode() -> int:
    """1 = side-stream tail || post_bwd2d (measured +5 us per boundary in-step: the fork / join costs more than the
    overlap); 2 = one launch with post_bwd2d programs + tail programs."""
    return _mode("PRIMUS_HCFUSE_TAIL")


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
FWD_CFG = dict(rs=32, nw=8, srs=1, snw=1, ppc=4)


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
    if mode == 6:
        nprog_s = triton.cdiv(M, c["rs"])
        nprog_c = min(M, c["ppc"] * _n_cu(dev))
        _hcf_pre_collapse2_kernel[(nprog_s + nprog_c,)](
            x3, p_raw, p_ss, scale_c, base_c, gamma, raw, rstd, states, pre, post, combt, y, rstd_n,
            M, D, nprog_s, nprog_c, float(eps), float(norm_eps),
            EPS=float(eps), BN=BN, NS=int(n_split), N_ITERS=int(n_iters), RS=c["rs"], BD=BD, num_warps=c["nw"],
        )
        return raw, rstd, states, pre, post, combt, y, rstd_n
    if mode == 1:
        nprog_s = triton.cdiv(M, c["rs"])
        _hcf_pre_collapse_fwd_kernel[(nprog_s + M,)](
            x3, p_raw, p_ss, scale_c, base_c, gamma, raw, rstd, states, pre, post, combt, y, rstd_n,
            M, D, nprog_s, float(eps), float(norm_eps),
            EPS=float(eps), BN=BN, NS=int(n_split), N_ITERS=int(n_iters), RS=c["rs"], BD=BD, num_warps=c["nw"],
        )
        return raw, rstd, states, pre, post, combt, y, rstd_n
    main = torch.cuda.current_stream(dev)
    side = _side_stream(dev) if mode in (2, 4) else main
    if mode in (2, 4):
        side.wait_stream(main)
    with torch.cuda.stream(side):
        _hcf_sk_fwd_kernel[(triton.cdiv(M, c["srs"]),)](
            p_raw, p_ss, scale_c, base_c, raw, rstd, states, post, combt, M, D, float(eps),
            EPS=float(eps), BN=BN, NS=int(n_split), N_ITERS=int(n_iters), RS=c["srs"], num_warps=c["snw"],
        )
    if mode in (4, 5):  # persistent collapse rows, partials prefetched one row block ahead
        nprog = min(M, c["ppc"] * _n_cu(dev))
        _hcf_collapse_persist_kernel[(nprog,)](
            x3, p_raw, p_ss, scale_c, base_c, gamma, pre, y, rstd_n, M, D, float(eps), float(norm_eps), nprog,
            EPS=float(eps), BN=BN, NS=int(n_split), BD=BD, RB=1, num_warps=c["nw"],
        )
    else:
        _hcf_collapse_fwd_kernel[(M,)](
            x3, p_raw, p_ss, scale_c, base_c, gamma, pre, y, rstd_n, M, D, float(eps), float(norm_eps),
            EPS=float(eps), BN=BN, NS=int(n_split), BD=BD, num_warps=c["nw"],
        )
    if mode in (2, 4):
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


@triton.jit
def _ld_parts(P_RAW, P_SS, t64, rm, M, BN: tl.constexpr, NS: tl.constexpr):
    """Raw (unsummed) rms-linear partials of the pre logits for rows t64: [NS, RB, 4] and [NS, RB]."""
    sidx = tl.arange(0, NS)
    kk = tl.arange(0, 4)
    pr = tl.load(P_RAW + sidx[:, None, None] * (M * BN) + t64[None, :, None] * BN + kk[None, None, :],
                 mask=rm[None, :, None], other=0.0)
    ps = tl.load(P_SS + sidx[:, None] * M + t64[None, :], mask=rm[None, :], other=0.0)
    return pr, ps


@triton.jit
def _hcf_collapse_persist_kernel(X, P_RAW, P_SS, SCALE, BASE, GAMMA, PRE, Y, RSTDN, M, D, eps_rms, norm_eps, NPROG,
                                 EPS: tl.constexpr, BN: tl.constexpr, NS: tl.constexpr, BD: tl.constexpr,
                                 RB: tl.constexpr):
    _hcf_collapse_persist(tl.program_id(0), X, P_RAW, P_SS, SCALE, BASE, GAMMA, PRE, Y, RSTDN, M, D, eps_rms, norm_eps,
                          NPROG, EPS, BN, NS, BD, RB)


@triton.jit
def _hcf_pre_collapse2_kernel(X, P_RAW, P_SS, SCALE, BASE, GAMMA, RAW, RSTD, STATES, PRE, POST, COMBT, Y, RSTDN,
                              M, D, NPROG_S, NPROG_C, eps_rms, norm_eps, EPS: tl.constexpr, BN: tl.constexpr,
                              NS: tl.constexpr, N_ITERS: tl.constexpr, RS: tl.constexpr, BD: tl.constexpr):
    """FWD mode 6: one launch = Sinkhorn programs (first) + persistent collapse programs (pre computed per row from
    prefetched partials)."""
    pid = tl.program_id(0)
    if pid < NPROG_S:
        _hcf_sk_fwd(pid, P_RAW, P_SS, SCALE, BASE, RAW, RSTD, STATES, POST, COMBT, M, D, eps_rms,
                    EPS, 4, BN, NS, N_ITERS, RS)
    else:
        _hcf_collapse_persist(pid - NPROG_S, X, P_RAW, P_SS, SCALE, BASE, GAMMA, PRE, Y, RSTDN, M, D, eps_rms,
                              norm_eps, NPROG_C, EPS, BN, NS, BD, 1)


@triton.jit
def _hcf_collapse_persist(pid, X, P_RAW, P_SS, SCALE, BASE, GAMMA, PRE, Y, RSTDN, M, D, eps_rms, norm_eps, NPROG,
                          EPS: tl.constexpr, BN: tl.constexpr, NS: tl.constexpr, BD: tl.constexpr, RB: tl.constexpr):
    """Persistent rows (as _hc_norm_dpre_bwd_kernel): pre from the rms-linear partials (prefetched one row block ahead
    with the x rows, summed at use), then the folded-norm collapse (as _hc_collapse_norm_fwd_kernel)."""
    K: tl.constexpr = 4
    cols = tl.arange(0, BD)
    cm = cols < D
    kk = tl.arange(0, K)
    g = tl.load(GAMMA + cols, mask=cm, other=0.0).to(tl.float32)
    s0 = tl.load(SCALE + 0).to(tl.float32)
    b_pre = tl.load(BASE + kk).to(tl.float32)
    KD = K * D
    STEP = NPROG * RB
    rows = pid * RB + tl.arange(0, RB)
    rm = rows < M
    t64 = rows.to(tl.int64)
    m2 = rm[:, None] & cm[None, :]
    a0, a1, a2, a3 = _ld_rows4(X, t64, cols, m2, D)
    pr, ps = _ld_parts(P_RAW, P_SS, t64, rm, M, BN, NS)
    for r0 in range(pid * RB, M, STEP):
        x0, x1, x2, x3 = a0, a1, a2, a3
        cpr, cps = pr, ps
        tc, rmc, m2c = t64, rm, m2
        rows = r0 + STEP + tl.arange(0, RB)
        rm = rows < M
        t64 = rows.to(tl.int64)
        m2 = rm[:, None] & cm[None, :]
        a0, a1, a2, a3 = _ld_rows4(X, t64, cols, m2, D)
        pr, ps = _ld_parts(P_RAW, P_SS, t64, rm, M, BN, NS)

        raw_pre = tl.sum(cpr, axis=0)  # [RB, 4]
        ss = tl.sum(cps, axis=0)  # [RB]
        rstd = 1.0 / tl.sqrt(ss / KD + eps_rms)
        pre = tl.sigmoid((raw_pre * rstd[:, None]) * s0 + b_pre[None, :]) + EPS
        pre_b = pre.to(tl.bfloat16)
        k2 = kk[None, :]
        tl.store(PRE + tc[:, None] * K + k2, pre_b, mask=rmc[:, None])
        pf = pre_b.to(tl.float32)
        p0 = tl.sum(tl.where(k2 == 0, pf, 0.0), axis=1)
        p1 = tl.sum(tl.where(k2 == 1, pf, 0.0), axis=1)
        p2 = tl.sum(tl.where(k2 == 2, pf, 0.0), axis=1)
        p3 = tl.sum(tl.where(k2 == 3, pf, 0.0), axis=1)
        h = _collapse_bf16(x0, x1, x2, x3, p0, p1, p2, p3)
        rn = 1.0 / tl.sqrt(tl.sum(h * h, axis=1) / D + norm_eps)
        y = (h * rn[:, None] * g[None, :]).to(Y.dtype.element_ty)
        tl.store(Y + tc[:, None] * D + cols[None, :], y, mask=m2c)
        tl.store(RSTDN + tc, rn, mask=rmc)



# ------------------------------------------------------------------------------------------------
# PRIMUS_HCFUSE_TAIL=2: one launch = post_bwd2d(L-1) programs + the GRAD_DIRECT tail(L) programs
# ------------------------------------------------------------------------------------------------
@triton.jit
def _post_body(pid_t, pid_c, G, X, OUT, POST, DOUT, DPOSTP, DCOMBTP, M, D, K: tl.constexpr, BT: tl.constexpr,
               BC: tl.constexpr):
    """_hc_post_bwd2d_kernel for one (row tile, column tile)."""
    t = pid_t * BT + tl.arange(0, BT)
    cc = pid_c * BC + tl.arange(0, BC)
    kk = tl.arange(0, K)
    KK: tl.constexpr = K * K
    tm = t < M
    t64 = t.to(tl.int64)
    KD = K * D
    x3 = tl.load(X + t64[:, None, None] * KD + kk[None, :, None] * D + cc[None, None, :],
                 mask=tm[:, None, None], other=0.0).to(tl.float32)
    o = tl.load(OUT + t64[:, None] * D + cc[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
    post = tl.load(POST + t64[:, None] * K + kk[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
    dout = tl.zeros((BT, BC), dtype=tl.float32)
    pbase = pid_c * M
    for h in tl.static_range(K):
        g_h = tl.load(G + t64[:, None] * KD + h * D + cc[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        post_h = tl.sum(tl.where(kk[None, :] == h, post, 0.0), axis=1)
        dout += post_h[:, None] * g_h
        tl.store(DPOSTP + (pbase + t64) * K + h, tl.sum(o * g_h, axis=1), mask=tm)
        dch = tl.sum(x3 * g_h[:, None, :], axis=2)
        tl.store(DCOMBTP + (pbase + t64[:, None]) * KK + h * K + kk[None, :], dch, mask=tm[:, None])
    tl.store(DOUT + t64[:, None] * D + cc[None, :], dout.to(DOUT.dtype.element_ty), mask=tm[:, None])


@triton.jit
def _mg_write2(OUT, offs, v, m, PDT: tl.constexpr, ACC: tl.constexpr):
    gp = v.to(PDT).to(tl.float32)
    if ACC:
        gp = tl.load(OUT + offs, mask=m, other=0.0).to(tl.float32) + gp
    tl.store(OUT + offs, gp.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _tail_body(pid, DWP, PART, DGP, DW_OUT, S_OUT, B_OUT, DG_OUT, NS, NTOT, P, N_DW, P2, D,
               NP: tl.constexpr, BLOCK: tl.constexpr, BH: tl.constexpr, BP: tl.constexpr,
               W_PDT: tl.constexpr, S_PDT: tl.constexpr, B_PDT: tl.constexpr, G_PDT: tl.constexpr,
               W_ACC: tl.constexpr, S_ACC: tl.constexpr, B_ACC: tl.constexpr, G_ACC: tl.constexpr,
               HAS_DG: tl.constexpr, BW: tl.constexpr):
    """hc_boundary._hc_tail_kernel for one program (dW: same per-element sequential sum over the S partials)."""
    n_dw = tl.cdiv(N_DW, BW)
    if pid < n_dw:
        o = pid * BW + tl.arange(0, BW)
        m = o < N_DW
        acc = tl.zeros((BW,), dtype=tl.float32)
        for s_ in range(0, NS):
            acc += tl.load(DWP + s_ * NTOT + o, mask=m, other=0.0)
        _mg_write2(DW_OUT, o, acc, m, W_PDT, W_ACC)
    elif pid < n_dw + NP:
        j = pid - n_dw
        acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
        for p0 in range(0, P, BLOCK):
            r = p0 + tl.arange(0, BLOCK)
            acc2 += tl.load(PART + j * P + r, mask=r < P, other=0.0)
        v = tl.sum(acc2, axis=0)
        if j < 3:
            _mg_write2(S_OUT, j, v, True, S_PDT, S_ACC)
        else:
            _mg_write2(B_OUT, j - 3, v, True, B_PDT, B_ACC)
    elif HAS_DG:
        cols = (pid - n_dw - NP) * BH + tl.arange(0, BH)
        cm = cols < D
        acc3 = tl.zeros((BP, BH), dtype=tl.float32)
        for p0 in range(0, P2, BP):
            pr = p0 + tl.arange(0, BP)
            acc3 += tl.load(DGP + pr[:, None] * D + cols[None, :], mask=(pr[:, None] < P2) & cm[None, :], other=0.0)
        _mg_write2(DG_OUT, cols, tl.sum(acc3, axis=0), cm, G_PDT, G_ACC)


@triton.jit
def _hcf_post_tail_kernel(G, X, OUT, POST, DOUT, DPOSTP, DCOMBTP, M, D, NT, NTAIL,
                          DWP, PART, DGP, DW_OUT, S_OUT, B_OUT, DG_OUT, NS, NTOT, P, N_DW, P2,
                          K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
                          NP: tl.constexpr, BLOCK: tl.constexpr, BH: tl.constexpr, BP: tl.constexpr,
                          W_PDT: tl.constexpr, S_PDT: tl.constexpr, B_PDT: tl.constexpr, G_PDT: tl.constexpr,
                          W_ACC: tl.constexpr, S_ACC: tl.constexpr, B_ACC: tl.constexpr, G_ACC: tl.constexpr,
                          HAS_DG: tl.constexpr, BW: tl.constexpr):
    pid = tl.program_id(0)
    if pid < NTAIL:  # tail programs first: dispatched at the start, they overlap the post programs
        _tail_body(pid, DWP, PART, DGP, DW_OUT, S_OUT, B_OUT, DG_OUT, NS, NTOT, P, N_DW, P2, D,
                   NP, BLOCK, BH, BP, W_PDT, S_PDT, B_PDT, G_PDT, W_ACC, S_ACC, B_ACC, G_ACC, HAS_DG, BW)
    else:
        q = pid - NTAIL
        _post_body(q % NT, q // NT, G, X, OUT, POST, DOUT, DPOSTP, DCOMBTP, M, D, K, BT, BC)


TAIL2_CFG = dict(nw=1, block=512, bw=512, bh=32, bp=32)


def post_tail(dx, xp, outp, postp, doutp, dpostp, dcombtp, pbt, pbc, dw_p, n, part, dgp, D, params, dtypes, claimed):
    """One launch: post_bwd2d(L-1) (dnew = dx) + the GRAD_DIRECT tail(L). Returns the 4 gradients for autograd."""
    from .hc_boundary import _GradSlot, _TL_DT

    M = xp.shape[0]
    K = xp.shape[1]
    S_, BN_, KD_ = dw_p.shape
    dev = dw_p.device
    NP_, P_ = part.shape
    w, s_, b, g = params
    slots = [_GradSlot(w, dtypes[0], (n, KD_), dev, claimed[0]), _GradSlot(s_, dtypes[1], (3,), dev, claimed[1]),
             _GradSlot(b, dtypes[2], (n,), dev, claimed[2])]
    has_dg = dgp is not None
    slots.append(_GradSlot(g, dtypes[3], (D,), dev, claimed[3]) if has_dg else None)
    c = _cfg("TAIL2", TAIL2_CFG)
    n_dw = n * KD_
    nt = triton.cdiv(M, pbt)
    npost = nt * (D // pbc)
    ntail = triton.cdiv(n_dw, c["bw"]) + NP_ + (triton.cdiv(D, c["bh"]) if has_dg else 0)
    dg_out = slots[3].out if has_dg else slots[0].out
    _hcf_post_tail_kernel[(npost + ntail,)](
        dx, xp, outp, postp, doutp, dpostp, dcombtp, M, D, nt, ntail,
        dw_p, part, dgp if has_dg else part, slots[0].out, slots[1].out, slots[2].out, dg_out,
        S_, BN_ * KD_, P_, n_dw, dgp.shape[0] if has_dg else 0,
        K=K, BT=pbt, BC=pbc, NP=NP_, BLOCK=c["block"], BH=c["bh"], BP=c["bp"],
        W_PDT=_TL_DT[dtypes[0]], S_PDT=_TL_DT[dtypes[1]], B_PDT=_TL_DT[dtypes[2]],
        G_PDT=_TL_DT[dtypes[3]] if has_dg else tl.float32,
        W_ACC=slots[0].acc, S_ACC=slots[1].acc, B_ACC=slots[2].acc, G_ACC=slots[3].acc if has_dg else False,
        HAS_DG=has_dg, BW=c["bw"], num_warps=c["nw"],
    )
    return tuple(sl.grad(dt) if sl is not None else None for sl, dt in zip(slots, dtypes))


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


__all__ = ["hcfuse_bwd_enabled", "hcfuse_fwd_enabled", "hcfuse_tail_enabled", "side_stream", "norm_pre_bwd",
           "pre_collapse_fwd", "post_tail", "tail_mode"]
side_stream = _side_stream


# ------------------------------------------------------------------------------------------------
# PRIMUS_HCFUSE_POST=1: expand backward (post_bwd2d) with the 20 per-row reductions on WMMA (Gluon)
# ------------------------------------------------------------------------------------------------
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    _HAVE_GLUON = True
except Exception:  # pragma: no cover
    _HAVE_GLUON = False


def hcfuse_post_enabled() -> bool:
    return _mode("PRIMUS_HCFUSE_POST") > 0


if _HAVE_GLUON:

    @gluon.jit
    def _hcf_post_wmma_kernel(G, X, OUT, POST, DOUT, DPOSTP, DCOMBTP, M, D,
                              WM: gl.constexpr, OPA: gl.constexpr, OPB: gl.constexpr, BLK: gl.constexpr,
                              BC: gl.constexpr):
        """One 1-warp program = 4 rows x BC columns. dcomb^T[t, h, k] = <dnew[t, h], x[t, k]> and
        dpost[t, h] = <out[t], dnew[t, h]> come from 16x16x32 WMMAs on A = dnew [(t, h) = 16 rows, 32 cols] and
        B = x / out [32 cols, (t', k) = 16]: only the t == t' diagonal 4x4 blocks are used. dout = sum_h post_h dnew_h
        on a blocked copy of dnew. Partials per BC column tile, same layout as _hc_post_bwd2d_kernel."""
        pid_t = gl.program_id(0)
        pid_c = gl.program_id(1)
        r0 = pid_t * 4
        c0 = pid_c * BC
        KD = 4 * D
        # WMMA operands
        a_r = gl.arange(0, 16, layout=gl.SliceLayout(1, OPA))
        a_k = gl.arange(0, 32, layout=gl.SliceLayout(0, OPA))
        b_k = gl.arange(0, 32, layout=gl.SliceLayout(1, OPB))
        b_c = gl.arange(0, 16, layout=gl.SliceLayout(0, OPB))
        a_off = gl.expand_dims((r0 + a_r // 4).to(gl.int64) * KD + (a_r % 4) * D, 1) + gl.expand_dims(c0 + a_k, 0)
        b_off = gl.expand_dims(c0 + b_k, 1) + gl.expand_dims((r0 + b_c // 4).to(gl.int64) * KD + (b_c % 4) * D, 0)
        o_off = gl.expand_dims(c0 + b_k, 1) + gl.expand_dims((r0 + b_c // 4).to(gl.int64) * D, 0)
        acc = gl.zeros([16, 16], gl.float32, layout=WM)
        acc2 = gl.zeros([16, 16], gl.float32, layout=WM)
        for kc in gl.static_range(0, BC, 32):
            a = gl.load(G + a_off + kc)
            b = gl.load(X + b_off + kc)
            bo = gl.load(OUT + o_off + kc)
            acc = gl.amd.gfx1250.wmma(a, b, acc)
            acc2 = gl.amd.gfx1250.wmma(a, bo, acc2)
        # diagonal 4x4 blocks -> partials
        c_r = gl.expand_dims(gl.arange(0, 16, layout=gl.SliceLayout(1, WM)), 1)
        c_c = gl.expand_dims(gl.arange(0, 16, layout=gl.SliceLayout(0, WM)), 0)
        diag = (c_r // 4) == (c_c // 4)
        row = (r0 + c_r // 4).to(gl.int64)
        pb = pid_c * M
        gl.store(DCOMBTP + (pb + row) * 16 + (c_r % 4) * 4 + (c_c % 4), acc, mask=diag)
        gl.store(DPOSTP + (pb + row) * 4 + (c_r % 4) + c_c * 0, acc2, mask=diag & ((c_c % 4) == 0))
        # dout on a blocked copy (L1/L2 hit)
        t_r = gl.arange(0, 4, layout=gl.SliceLayout(1, BLK))
        t_c = c0 + gl.arange(0, BC, layout=gl.SliceLayout(0, BLK))
        r64 = (r0 + t_r).to(gl.int64)
        base = gl.expand_dims(r64 * KD, 1) + gl.expand_dims(t_c, 0)
        p0 = gl.expand_dims(gl.load(POST + r64 * 4 + 0).to(gl.float32), 1)
        p1 = gl.expand_dims(gl.load(POST + r64 * 4 + 1).to(gl.float32), 1)
        p2 = gl.expand_dims(gl.load(POST + r64 * 4 + 2).to(gl.float32), 1)
        p3 = gl.expand_dims(gl.load(POST + r64 * 4 + 3).to(gl.float32), 1)
        d = (p0 * gl.load(G + base).to(gl.float32) + p1 * gl.load(G + base + D).to(gl.float32)
             + p2 * gl.load(G + base + 2 * D).to(gl.float32) + p3 * gl.load(G + base + 3 * D).to(gl.float32))
        gl.store(DOUT + gl.expand_dims(r64 * D, 1) + gl.expand_dims(t_c, 0), d.to(gl.bfloat16))

    _PL = {}

    def _post_layouts(bc):
        if bc not in _PL:
            wm = gl.amd.AMDWMMALayout(version=3, transposed=True, warp_bases=[], instr_shape=[16, 16, 32])
            opa = gl.DotOperandLayout(operand_index=0, parent=wm, k_width=8)
            opb = gl.DotOperandLayout(operand_index=1, parent=wm, k_width=8)
            blk = gl.BlockedLayout([1, 8], [1, 32], [1, 1], [1, 0])
            _PL[bc] = (wm, opa, opb, blk)
        return _PL[bc]


def post_bwd_wmma(dnew, x3, out_c, post_c, dout, dpost, dcombt, bc=256):
    """dnew / x3 [M, 4, D] bf16, out_c [M, D] bf16, post_c [M, 4] bf16 -> dout, dpost [D/bc, M, 4], dcombt
    [D/bc, M, 4, 4] fp32 partials (the _hc_post_bwd2d_kernel layout). Needs M % 4 == 0, D % bc == 0, bc % 32 == 0."""
    M, K, D = x3.shape
    wm, opa, opb, blk = _post_layouts(bc)
    _hcf_post_wmma_kernel[(M // 4, D // bc)](dnew, x3, out_c, post_c, dout, dpost, dcombt, M, D,
                                             WM=wm, OPA=opa, OPB=opb, BLK=blk, BC=bc, num_warps=1)


def post_wmma_ok(M, K, D, bc=256) -> bool:
    return _HAVE_GLUON and K == 4 and M % 4 == 0 and D % bc == 0 and bc % 32 == 0
