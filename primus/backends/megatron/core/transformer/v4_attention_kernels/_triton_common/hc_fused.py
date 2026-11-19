###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Fused HyperConnection mixer (g90mem, 2026-11-19): one HC application = 3 fwd + 4 bwd launches.

``DeepseekV4HybridLayer._hc_apply`` ran, per HyperMixer application (x ``[M, K, D]`` bf16):

  fwd: rms-linear partials, finalize, compute-tail, sinkhorn, 3 casts, comb transpose copy, collapse, expand
  bwd: expand bwd (+2 partial sums, casts), collapse bwd (+sum, cast), tail bwd (+~10 torch reductions / muls),
       sinkhorn bwd, rms prep, rms bwd (+dW sum), and two 128 MB bf16 adds merging the three ``dx`` streams.

Here (bf16 x, K = 4; anything else falls back to the old path):

  fwd  F1 ``_hc_rms_linear_fwd_kernel`` (unchanged): split-K partials of ``x @ W^T`` and ``sum(x^2)``.
       F2 ``_hc_pre_fwd_kernel`` (rows): finalize + scale/base + sigmoid/softmax + Sinkhorn + bf16 rounding of
          pre/post/comb + collapse ``sum_k pre_k x_k``. Writes collapsed, post, comb^T, and the tiny bwd state.
       F3 ``_hc_expand_fwd_kernel`` (unchanged).
  bwd  B1 ``_hc_post_bwd_kernel`` (rows): ``dout``, ``dpost``, ``dcomb`` from ``dnew``; does NOT form ``dx``;
          ``dnew`` is handed to the pre-side backward through a link object (autograd orders the two:
          the pre Function's outputs post/comb feed only the post Function).
       B2 ``_hc_pre_bwd_kernel`` (rows): ``dpre = <dcollapsed, x_k>``, Sinkhorn + tail VJP, ``g = dlogits*rstd``,
          ``coef``, per-program dscale/dbase partials.
       B3 ``_hc_bwd_main_kernel`` (stream x column tile x row split): the single ``dx`` write
          ``dx_k = g @ W_k - x_k*coef + pre_k*dcollapsed + sum_h comb[k,h]*dnew_h`` plus dW partials.
       B4 partial reductions (torch sums).

Numerics: pre/post/comb are rounded to bf16 exactly where the eager path casts them (``compute_weights``
returns ``x.dtype``); the three ``dx`` contributions are summed in fp32 and rounded once (the old path
rounded each to bf16 and added in bf16). Opt-in: ``PRIMUS_HC_FUSED=1`` (default off until an e2e A/B is run).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .hc_expand import _hc_expand_fwd_kernel
from .hc_rms_linear import HC_CFG, _block_n, _dot_bf16_x_f32, _dot_f32_x_f32, _hc_rms_linear_fwd_kernel

HCF_CFG = dict(
    row_bm=4,  # rows per program for the row kernels (F2, B1, B2)
    row_bd=512,
    row_warps=4,
    exp_bm=16,  # expand fwd
    exp_bd=128,
    exp_warps=4,
    main_bt=64,  # B3 (per-stream kernel; sweep 2026-11-19: 197 us, the all-K variant 196-300 us)
    main_bd=128,
    main_tsplit=4,
    main_warps=8,
    main_stages=1,
    main_prec="ieee",
)


def hc_fused_enabled() -> bool:
    return os.environ.get("PRIMUS_HC_FUSED", "0") == "1"


def hc_fused_supported(x: torch.Tensor, hc_mult: int) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and hc_mult == 4
        and x.dim() >= 2
        and x.shape[-2] == hc_mult
        and x.is_contiguous()
    )


# ------------------------------------------------------------------------------------------------
# F2: finalize + tail + Sinkhorn + collapse (rows)
# ------------------------------------------------------------------------------------------------


@triton.jit
def _hc_pre_fwd_kernel(
    X,
    P_RAW,
    P_SS,
    SCALE,
    BASE,
    RAW,
    RSTD,
    STATES,
    PRE,
    POST,
    COMBT,
    COLL,
    M,
    D,
    n_split,
    eps_rms,
    EPS: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    N_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
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

    raw_pre = tl.zeros((BLOCK_M, K), dtype=tl.float32)
    raw_post = tl.zeros((BLOCK_M, K), dtype=tl.float32)
    raw_comb = tl.zeros((BLOCK_M, K, K), dtype=tl.float32)
    ss = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for s in range(0, n_split):
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

    s0 = tl.load(SCALE + 0)
    s1 = tl.load(SCALE + 1)
    s2 = tl.load(SCALE + 2)
    b_pre = tl.load(BASE + kk)
    b_post = tl.load(BASE + K + kk)
    b_comb = tl.load(BASE + 2 * K + kk[:, None] * K + kk[None, :])
    pre_logit = (raw_pre * rstd[:, None]) * s0 + b_pre[None, :]
    post_logit = (raw_post * rstd[:, None]) * s1 + b_post[None, :]
    comb_logit = (raw_comb * rstd[:, None, None]) * s2 + b_comb[None, :, :]
    pre = tl.sigmoid(pre_logit) + EPS
    post = 2.0 * tl.sigmoid(post_logit)
    cmax = tl.max(comb_logit, axis=2, keep_dims=True)
    cexp = tl.exp(comb_logit - cmax)
    m = cexp / tl.sum(cexp, axis=2, keep_dims=True) + EPS

    # Sinkhorn (same trajectory + state layout as sinkhorn.py)
    SROW: tl.constexpr = 2 * N_ITERS * KK
    sb = STATES + r3 * SROW + i3 * K + j3
    tl.store(sb, m, mask=m3)
    m = m / (tl.sum(m, axis=1, keep_dims=True) + EPS)
    tl.store(sb + KK, m, mask=m3)
    for it in tl.static_range(N_ITERS - 1):
        m = m / (tl.sum(m, axis=2, keep_dims=True) + EPS)
        tl.store(sb + (2 + 2 * it) * KK, m, mask=m3)
        m = m / (tl.sum(m, axis=1, keep_dims=True) + EPS)
        tl.store(sb + (3 + 2 * it) * KK, m, mask=m3)

    pre_b = pre.to(tl.bfloat16)
    tl.store(PRE + r2 * K + kk[None, :], pre_b, mask=m2)
    tl.store(POST + r2 * K + kk[None, :], post.to(tl.bfloat16), mask=m2)
    tl.store(COMBT + r3 * KK + j3 * K + i3, m.to(tl.bfloat16), mask=m3)  # comb^T for expand

    pre_f = pre_b.to(tl.float32)
    dd = tl.arange(0, BLOCK_D)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + dd
        dm = offs < D
        x3 = tl.load(X + r3 * KD + i3 * D + offs[None, None, :], mask=m3 & dm[None, None, :], other=0.0).to(
            tl.float32
        )
        c = tl.sum(pre_f[:, :, None] * x3, axis=1)
        tl.store(COLL + r2 * D + offs[None, :], c.to(COLL.dtype.element_ty), mask=m2 & dm[None, :])


# ------------------------------------------------------------------------------------------------
# B1: expand backward without dx (rows)
# ------------------------------------------------------------------------------------------------


@triton.jit
def _hc_post_bwd_kernel(
    G,
    X,
    OUT,
    POST,
    DOUT,
    DPOST,
    DCOMBT,
    M,
    D,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
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
    post = tl.load(POST + r2 * K + kk[None, :], mask=m2, other=0.0).to(tl.float32)
    dpost = tl.zeros((BLOCK_M, K), dtype=tl.float32)
    dcombt = tl.zeros((BLOCK_M, K, K), dtype=tl.float32)
    dd = tl.arange(0, BLOCK_D)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + dd
        dm = offs < D
        md2 = m2 & dm[None, :]
        x3 = tl.load(X + r3 * KD + i3 * D + offs[None, None, :], mask=m3 & dm[None, None, :], other=0.0).to(
            tl.float32
        )
        o = tl.load(OUT + r2 * D + offs[None, :], mask=md2, other=0.0).to(tl.float32)
        dout = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        for h in tl.static_range(K):
            g_h = tl.load(G + r2 * KD + h * D + offs[None, :], mask=md2, other=0.0).to(tl.float32)
            post_h = tl.sum(tl.where(kk[None, :] == h, post, 0.0), axis=1)
            dout += post_h[:, None] * g_h
            dpost += tl.where(kk[None, :] == h, tl.sum(o * g_h, axis=1)[:, None], 0.0)
            dcombt += tl.where(i3 == h, tl.sum(x3 * g_h[:, None, :], axis=2)[:, None, :], 0.0)
        tl.store(DOUT + r2 * D + offs[None, :], dout.to(DOUT.dtype.element_ty), mask=md2)
    tl.store(DPOST + r2 * K + kk[None, :], dpost, mask=m2)
    tl.store(DCOMBT + r3 * KK + i3 * K + j3, dcombt, mask=m3)


# ------------------------------------------------------------------------------------------------
# B2: dpre + Sinkhorn/tail VJP + rms prep (rows)
# ------------------------------------------------------------------------------------------------


@triton.jit
def _hc_pre_bwd_kernel(
    X,
    DC,
    RAW,
    RSTD,
    STATES,
    SCALE,
    BASE,
    DPOST,
    DCOMBT,
    G,
    COEF,
    PART,
    M,
    D,
    HAS_POST: tl.constexpr,
    EPS: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    NP: tl.constexpr,
    N_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
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

    # dpre_k = <dcollapsed, x_k>
    dpre = tl.zeros((BLOCK_M, K), dtype=tl.float32)
    dd = tl.arange(0, BLOCK_D)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + dd
        dm = offs < D
        x3 = tl.load(X + r3 * KD + i3 * D + offs[None, None, :], mask=m3 & dm[None, None, :], other=0.0).to(
            tl.float32
        )
        dc = tl.load(DC + r2 * D + offs[None, :], mask=m2 & dm[None, :], other=0.0).to(tl.float32)
        dpre += tl.sum(x3 * dc[:, None, :], axis=2)

    # recompute the tail
    raw_pre = tl.load(RAW + r2 * BN + kk[None, :], mask=m2, other=0.0)
    raw_post = tl.load(RAW + r2 * BN + K + kk[None, :], mask=m2, other=0.0)
    raw_comb = tl.load(RAW + r3 * BN + 2 * K + i3 * K + j3, mask=m3, other=0.0)
    rstd = tl.load(RSTD + rows64, mask=rmask, other=0.0)
    s0 = tl.load(SCALE + 0)
    s1 = tl.load(SCALE + 1)
    s2 = tl.load(SCALE + 2)
    b_pre = tl.load(BASE + kk)
    b_post = tl.load(BASE + K + kk)
    b_comb = tl.load(BASE + 2 * K + kk[:, None] * K + kk[None, :])
    l_pre = raw_pre * rstd[:, None]
    l_post = raw_post * rstd[:, None]
    l_comb = raw_comb * rstd[:, None, None]
    pre_sig = tl.sigmoid(l_pre * s0 + b_pre[None, :])
    post_sig = tl.sigmoid(l_post * s1 + b_post[None, :])
    comb_logit = l_comb * s2 + b_comb[None, :, :]
    cmax = tl.max(comb_logit, axis=2, keep_dims=True)
    cexp = tl.exp(comb_logit - cmax)
    comb_sm = cexp / tl.sum(cexp, axis=2, keep_dims=True)

    if HAS_POST:
        dpost = tl.load(DPOST + r2 * K + kk[None, :], mask=m2, other=0.0)
        dm_ = tl.load(DCOMBT + r3 * KK + j3 * K + i3, mask=m3, other=0.0)  # dcomb[i, j] = dcomb^T[j, i]
    else:
        dpost = tl.zeros((BLOCK_M, K), dtype=tl.float32)
        dm_ = tl.zeros((BLOCK_M, K, K), dtype=tl.float32)

    # Sinkhorn VJP (as sinkhorn.py)
    SROW: tl.constexpr = 2 * N_ITERS * KK
    sb = STATES + r3 * SROW + i3 * K + j3
    for i in range(N_ITERS - 1):
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

    # tail VJP
    d_pre_l = dpre * pre_sig * (1.0 - pre_sig)
    d_post_l = dpost * 2.0 * post_sig * (1.0 - post_sig)
    d_comb_l = comb_sm * (dm_ - tl.sum(dm_ * comb_sm, axis=2, keep_dims=True))
    dl_pre = d_pre_l * s0
    dl_post = d_post_l * s1
    dl_comb = d_comb_l * s2

    # rms prep: g = dlogits * rstd (padded to BN), coef = rstd^3 * <dlogits, raw> / KD
    tl.store(G + r2 * BN + kk[None, :], dl_pre * rstd[:, None], mask=m2)
    tl.store(G + r2 * BN + K + kk[None, :], dl_post * rstd[:, None], mask=m2)
    tl.store(G + r3 * BN + 2 * K + i3 * K + j3, dl_comb * rstd[:, None, None], mask=m3)
    NPAD: tl.constexpr = BN - (2 + K) * K
    if NPAD > 0:
        pc = tl.arange(0, NPAD)
        tl.store(G + r2 * BN + (2 + K) * K + pc[None, :], tl.zeros((BLOCK_M, NPAD), dtype=tl.float32), mask=m2)
    c = tl.sum(dl_pre * raw_pre, axis=1) + tl.sum(dl_post * raw_post, axis=1) + tl.sum(tl.sum(dl_comb * raw_comb, axis=2), axis=1)
    tl.store(COEF + rows64, rstd * rstd * rstd * c / KD, mask=rmask)

    # per-program partials: [ds0, ds1, ds2, dbase(N)]
    pp = PART + pid * NP
    tl.store(pp + 0, tl.sum(tl.sum(l_pre * d_pre_l, axis=1), axis=0))
    tl.store(pp + 1, tl.sum(tl.sum(l_post * d_post_l, axis=1), axis=0))
    tl.store(pp + 2, tl.sum(tl.sum(tl.sum(l_comb * d_comb_l, axis=2), axis=1), axis=0))
    tl.store(pp + 3 + kk, tl.sum(d_pre_l, axis=0))
    tl.store(pp + 3 + K + kk, tl.sum(d_post_l, axis=0))
    tl.store(pp + 3 + 2 * K + kk[:, None] * K + kk[None, :], tl.sum(d_comb_l, axis=0))


# ------------------------------------------------------------------------------------------------
# B3: single dx write + dW partials (one program per stream x D tile x row split)
# ------------------------------------------------------------------------------------------------


@triton.jit
def _hc_bwd_main_kernel(
    X,
    W,
    G,
    COEF,
    PRE,
    COMBT,
    DC,
    DNEW,
    DX,
    DW,
    M,
    D,
    N,
    T_PER_SPLIT,
    N_D,
    N_TILES,
    HAS_DNEW: tl.constexpr,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BN: tl.constexpr,
    PREC: tl.constexpr,
):
    """One program = one stream k x one D tile x one row split. The K programs sharing a (D tile, split) get
    pids p, p+8, p+16, ... (same XCD, adjacent dispatch) so dcollapsed / dnew tiles are re-read from L2."""
    pid = tl.program_id(0)
    u = pid % 8
    v = pid // 8
    k = v % K
    tile = (v // K) * 8 + u
    if tile >= N_TILES:
        return
    pid_d = tile % N_D
    pid_s = tile // N_D
    KK: tl.constexpr = K * K
    KD = K * D
    ds = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dmask = ds < D
    ncol = tl.arange(0, BN)
    nmask = ncol < N
    w = tl.load(W + ncol[:, None].to(tl.int64) * KD + k * D + ds[None, :], mask=nmask[:, None] & dmask[None, :], other=0.0)
    dw = tl.zeros((BN, BLOCK_D), dtype=tl.float32)
    t_lo = pid_s * T_PER_SPLIT
    for t0 in range(t_lo, t_lo + T_PER_SPLIT, BLOCK_T):
        rows = t0 + tl.arange(0, BLOCK_T)
        rmask = rows < M
        rows64 = rows.to(tl.int64)
        r2 = rows64[:, None]
        md2 = rmask[:, None] & dmask[None, :]
        g = tl.load(G + r2 * BN + ncol[None, :], mask=rmask[:, None], other=0.0)
        coef = tl.load(COEF + rows64, mask=rmask, other=0.0)
        pk = tl.load(PRE + rows64 * K + k, mask=rmask, other=0.0).to(tl.float32)
        dc = tl.load(DC + r2 * D + ds[None, :], mask=md2, other=0.0).to(tl.float32)
        xk = tl.load(X + r2 * KD + k * D + ds[None, :], mask=md2, other=0.0)
        acc = _dot_f32_x_f32(g, w, PREC) - xk.to(tl.float32) * coef[:, None] + pk[:, None] * dc
        if HAS_DNEW:
            for h in tl.static_range(K):
                ch = tl.load(COMBT + rows64 * KK + h * K + k, mask=rmask, other=0.0).to(tl.float32)
                nh = tl.load(DNEW + r2 * KD + h * D + ds[None, :], mask=md2, other=0.0).to(tl.float32)
                acc += ch[:, None] * nh
        tl.store(DX + r2 * KD + k * D + ds[None, :], acc.to(DX.dtype.element_ty), mask=md2)
        dw += tl.trans(_dot_bf16_x_f32(tl.trans(xk), g, PREC))
    tl.store(
        DW + pid_s * BN * KD + ncol[:, None].to(tl.int64) * KD + k * D + ds[None, :],
        dw,
        mask=dmask[None, :] & (ncol[:, None] < BN),
    )


# ------------------------------------------------------------------------------------------------
# autograd
# ------------------------------------------------------------------------------------------------


class HCLink:
    """Carries the post-side backward results (dnew, dpost, dcomb^T) to the pre-side backward."""

    __slots__ = ("dnew", "dpost", "dcombt")

    def __init__(self):
        self.dnew = None
        self.dpost = None
        self.dcombt = None


class HCPreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scale, base, eps, n_iters, link):  # type: ignore[override]
        K, D = x.shape[-2], x.shape[-1]
        x3 = x.reshape(-1, K, D)
        M = x3.shape[0]
        KD = K * D
        x2d = x3.view(M, KD)
        w = weight.float().contiguous()
        N = w.shape[0]
        BN = _block_n(N)
        assert N == (2 + K) * K
        c = HC_CFG
        BLOCK_T, BLOCK_K, chunk_cfg = c["fwd_bt"], c["fwd_bk"], c["fwd_chunk"]
        n_split = max(1, triton.cdiv(KD, chunk_cfg))
        chunk = triton.cdiv(triton.cdiv(KD, n_split), BLOCK_K) * BLOCK_K
        n_split = triton.cdiv(KD, chunk)
        dev = x.device
        p_raw = torch.empty((n_split, M, BN), dtype=torch.float32, device=dev)
        p_ss = torch.empty((n_split, M), dtype=torch.float32, device=dev)
        _hc_rms_linear_fwd_kernel[(triton.cdiv(M, BLOCK_T), n_split)](
            x2d, w, p_raw, p_ss, M, KD, N, KD, chunk, 1, KD,
            BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K, BLOCK_N=BN, PREC="ieee",
            num_warps=c["fwd_warps"], num_stages=c["fwd_stages"],
        )
        f = HCF_CFG
        raw = torch.empty((M, BN), dtype=torch.float32, device=dev)
        rstd = torch.empty((M,), dtype=torch.float32, device=dev)
        states = torch.empty((M, 2 * n_iters, K, K), dtype=torch.float32, device=dev)
        pre = torch.empty((M, K), dtype=x.dtype, device=dev)
        post = torch.empty((M, K), dtype=x.dtype, device=dev)
        combt = torch.empty((M, K, K), dtype=x.dtype, device=dev)
        coll = torch.empty(x.shape[:-2] + (D,), dtype=x.dtype, device=dev)
        scale_c = scale.float().contiguous()
        base_c = base.float().contiguous()
        _hc_pre_fwd_kernel[(triton.cdiv(M, f["row_bm"]),)](
            x3, p_raw, p_ss, scale_c, base_c, raw, rstd, states, pre, post, combt, coll,
            M, D, n_split, float(eps),
            EPS=float(eps), K=K, BN=BN, N_ITERS=int(n_iters), BLOCK_M=f["row_bm"], BLOCK_D=f["row_bd"],
            num_warps=f["row_warps"],
        )
        ctx.save_for_backward(x3, w, scale_c, base_c, raw, rstd, states, pre, combt)
        ctx.link = link
        ctx.eps = float(eps)
        ctx.n_iters = int(n_iters)
        ctx.weight_dtype = weight.dtype
        ctx.scale_dtype = scale.dtype
        ctx.base_dtype = base.dtype
        ctx.x_shape = x.shape
        ctx.set_materialize_grads(False)
        lead = x.shape[:-2]
        return coll, post.view(*lead, K), combt.view(*lead, K, K)

    @staticmethod
    def backward(ctx, dcoll, _dpost_unused, _dcombt_unused):  # type: ignore[override]
        x3, w, scale_c, base_c, raw, rstd, states, pre, combt = ctx.saved_tensors
        link = ctx.link
        M, K, D = x3.shape
        KD = K * D
        N = w.shape[0]
        BN = raw.shape[1]
        dev = x3.device
        f = HCF_CFG
        if dcoll is None:
            dcoll = torch.zeros((M, D), dtype=x3.dtype, device=dev)
        dcoll = dcoll.contiguous()
        has_post = link is not None and link.dpost is not None
        dnew = link.dnew if has_post else None
        g = torch.empty((M, BN), dtype=torch.float32, device=dev)
        coef = torch.empty((M,), dtype=torch.float32, device=dev)
        n_prog = triton.cdiv(M, f["row_bm"])
        NP = 3 + N
        part = torch.empty((n_prog, NP), dtype=torch.float32, device=dev)
        _hc_pre_bwd_kernel[(n_prog,)](
            x3, dcoll, raw, rstd, states, scale_c, base_c,
            link.dpost if has_post else g, link.dcombt if has_post else g,
            g, coef, part, M, D,
            HAS_POST=has_post, EPS=ctx.eps, K=K, BN=BN, NP=NP, N_ITERS=ctx.n_iters,
            BLOCK_M=f["row_bm"], BLOCK_D=f["row_bd"], num_warps=f["row_warps"],
        )
        BLOCK_T, BLOCK_D, tsplit = f["main_bt"], f["main_bd"], f["main_tsplit"]
        t_per = triton.cdiv(triton.cdiv(M, tsplit), BLOCK_T) * BLOCK_T
        tsplit = triton.cdiv(M, t_per)
        dx = torch.empty_like(x3)
        dw_p = torch.empty((tsplit, BN, KD), dtype=torch.float32, device=dev)
        n_d = triton.cdiv(D, BLOCK_D)
        n_tiles = n_d * tsplit
        _hc_bwd_main_kernel[(triton.cdiv(n_tiles, 8) * 8 * K,)](
            x3, w, g, coef, pre, combt, dcoll, dnew if dnew is not None else x3, dx, dw_p,
            M, D, N, t_per, n_d, n_tiles,
            HAS_DNEW=dnew is not None, K=K, BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D, BN=BN, PREC=f["main_prec"],
            num_warps=f["main_warps"], num_stages=f["main_stages"],
        )
        if link is not None:
            link.dnew = link.dpost = link.dcombt = None
        dw = dw_p[0, :N] if tsplit == 1 else dw_p.sum(0)[:N]
        psum = part.sum(0)
        dscale = psum[:3].to(ctx.scale_dtype)
        dbase = psum[3:].to(ctx.base_dtype)
        return dx.view(ctx.x_shape), dw.to(ctx.weight_dtype), dscale, dbase, None, None, None


class HCPostFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out, post, combt, link):  # type: ignore[override]
        K, D = x.shape[-2], x.shape[-1]
        x3 = x.reshape(-1, K, D)
        M = x3.shape[0]
        out_c = out.contiguous()
        post_c = post.reshape(M, K)
        combt_c = combt.reshape(M, K, K)
        new = torch.empty_like(x3)
        f = HCF_CFG
        _hc_expand_fwd_kernel[(triton.cdiv(M, f["exp_bm"]), triton.cdiv(D, f["exp_bd"]))](
            x3, out_c, post_c, combt_c, new, M, D,
            K=K, BLOCK_M=f["exp_bm"], BLOCK_D=f["exp_bd"], num_warps=f["exp_warps"],
        )
        ctx.save_for_backward(x3, out_c, post_c)
        ctx.link = link
        ctx.out_shape = out.shape
        return new.view(x.shape)

    @staticmethod
    def backward(ctx, dnew):  # type: ignore[override]
        x3, out_c, post_c = ctx.saved_tensors
        M, K, D = x3.shape
        dnew = dnew.contiguous().view(M, K, D)
        dev = x3.device
        dout = torch.empty((M, D), dtype=out_c.dtype, device=dev)
        dpost = torch.empty((M, K), dtype=torch.float32, device=dev)
        dcombt = torch.empty((M, K, K), dtype=torch.float32, device=dev)
        f = HCF_CFG
        _hc_post_bwd_kernel[(triton.cdiv(M, f["row_bm"]),)](
            dnew, x3, out_c, post_c, dout, dpost, dcombt, M, D,
            K=K, BLOCK_M=f["row_bm"], BLOCK_D=f["row_bd"], num_warps=f["row_warps"],
        )
        link = ctx.link
        link.dnew, link.dpost, link.dcombt = dnew, dpost, dcombt
        return None, dout.view(ctx.out_shape), None, None, None


def hc_fused_pre(x, weight, scale, base, eps, n_iters):
    """-> (collapsed [..., D], post [..., K], comb^T [..., K, K], link)."""
    link = HCLink()
    coll, post, combt = HCPreFn.apply(x, weight, scale, base, eps, n_iters, link)
    return coll, post, combt, link


def hc_fused_post(x, out, post, combt, link):
    """new[..., h, :] = post_h * out + sum_k comb[k, h] x_k  (comb^T passed)."""
    return HCPostFn.apply(x, out, post, combt, link)


__all__ = ["hc_fused_enabled", "hc_fused_supported", "hc_fused_pre", "hc_fused_post", "HCF_CFG"]
