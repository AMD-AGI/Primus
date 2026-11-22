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

try:  # explicit-layout backward kernel (gfx1250 WMMA); falls back to the Triton kernel if unavailable
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    _HAVE_GLUON = True
except Exception:  # pragma: no cover
    _HAVE_GLUON = False

if _HAVE_GLUON:  # fuse-hc all-stream B3 (import errors must surface, not silently drop the gluon B3)
    from .hc_b3x import _hc_b3x_kernel, b3x_cfg, b3x_enabled, b3x_layouts, b3x_splitk_kw, b3x_splitk_ts
else:  # pragma: no cover

    def b3x_splitk_ts() -> int:
        return 0

    def b3x_splitk_kw():
        return {}

    def b3x_enabled() -> bool:
        return False

    def b3x_cfg():
        return dict(bc=64, w=8, bt=32, pf=0, ewm=0, buf=0)

from .hc_bif import (
    _hc_collapse_bif_kernel,
    _hc_dpre_bif_kernel,
    _hc_expand_bif_kernel,
    _hc_post_bwd_bif_kernel,
    bif_grid,
)
from .hc_boundary import (
    claim_main_grad,
    expand_rms,
    grad_direct_enabled,
    hc_tail,
    xrms_enabled,
    xrms_supported,
)
from .hc_expand import _hc_expand_fwd_kernel
from .hc_norm_fold import collapse_norm_fwd, norm_dpre_bwd
from .hc_rms_linear import (
    HC_CFG,
    _block_n,
    _dot_bf16_x_f32,
    _dot_f32_x_f32,
    _hc_fwd_wmma3_kernel,
    _hc_rms_linear_fwd_kernel,
)

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
    # gluon B3 (dx + dW, explicit WMMA layouts); 2026-11-19 lab: 85 us vs 178 us for the Triton kernel
    gl_bt=32,
    gl_bc=256,
    gl_tsplit=8,
    gl_warps=8,
    # fold the next HC's rms-linear partials (F1) into this HC's expand (g90mem round 3)
    # F1 on bf16 operands: x fed to WMMA as bf16, W pre-split into 3 bf16 parts (g90mem round 5: ATT showed the
    # fp32-ieee kernel LDS/VALU bound; 37.6 -> 23.5 us, same accuracy 5.4e-6)
    f1_wmma3=True,
    f1_bt=128,
    f1_bk=128,
    f1_chunk=1024,
    f1_warps=4,
    f1_stages=3,
    fuse_next_rms=False,  # untested on GPU (TDM hang stopped the GPU before the chain test ran)
    xr_bm=16,
    xr_bd=256,
    xr_warps=4,
    # 2-D streaming tiles for collapse / expand-bwd / dpre (partials over column tiles) + row-math kernels
    two_d=True,
    t2_bt=8,
    t2_bc=256,
    t2_warps=4,
    rm_bm=1,  # rows per program in the row-math-only kernels (sweep: bm1/1 warp 11.4 + 15.1 us, bm16/4 19.8 + 28.4)
    rm_warps=1,
    # opt3-mem: bytes-in-flight variants of collapse / dpre / post-bwd / expand (hc_bif.py): persistent grid-stride
    # programs, next tile's loads issued before the current tile is used, K=4 planes with in-wave reductions.
    # Same outputs and partial layouts. Opt-in until timed on the GPU (PRIMUS_HC_BIF=1); sweep bt/bc/warps/ppc.
    bif=os.environ.get("PRIMUS_HC_BIF", "0") == "1",
    # (bt, bc, warps) per kernel; gfx1250 compile (opt3-mem isa/): collapse 150, dpre 168, post_bwd bt4 142,
    # expand bt4 158 VGPRs (bt8 would be 262 / 304 > 256 -> 1 wave/SIMD). bc must equal t2_bc for the partials.
    bif_collapse=(8, 256, 4),
    bif_dpre=(8, 256, 4),
    bif_post=(4, 256, 4),
    bif_expand=(4, 256, 4),
    bif_ppc=8,  # programs per CU (persistent grid = ppc * #CU)
    # opt3-mem: RMSNorm folded into F2 (collapse) / B2 (dpre) when hc_fused_pre gets the norm weight
    # (PRIMUS_HC_FOLD_NORM=1, deepseek_v4_block._hc_apply); fold_fp8 also writes the 1x128 FP8 row quant of y
    fold_fp8=False,
)


def _exp_tile():
    """(BLOCK_M, BLOCK_D, num_warps) of F3 ``_hc_expand_fwd_kernel``. ``PRIMUS_HC_EXPAND_TILE=bm,bd,warps`` overrides
    the default (16, 128, 4). hc track: under Triton main (3.9, the e2e compiler) bm16/bd128 runs 74 us vs 26.6 us
    under the image Triton 3.8; bm1/bd2048/8 warps is 26.5 us and byte-identical (bench_expand2.py)."""
    v = os.environ.get("PRIMUS_HC_EXPAND_TILE", "").strip()
    if v:
        bm, bd, nw = (int(s) for s in v.split(","))
        return bm, bd, nw
    return HCF_CFG["exp_bm"], HCF_CFG["exp_bd"], HCF_CFG["exp_warps"]


def _post_tile():
    """(BT, num_warps) of B1 ``_hc_post_bwd2d_kernel`` (BC stays t2_bc: B2 consumes D / t2_bc partials).
    ``PRIMUS_HC_POST_TILE=bt,warps`` overrides the default (t2_bt, t2_warps) = (8, 4). hc track: bt1/1 warp is
    35.8 vs 50.5 us under Triton main, byte-identical dout / dpost / dcomb^T partials (bench_post.py)."""
    v = os.environ.get("PRIMUS_HC_POST_TILE", "").strip()
    if v:
        bt, nw = (int(s) for s in v.split(","))
        return bt, nw
    return HCF_CFG["t2_bt"], HCF_CFG["t2_warps"]


def _hc_nocast() -> bool:
    return os.environ.get("PRIMUS_HC_NOCAST", "0") == "1"


def _bif_nprog(dev) -> int:
    try:
        n_cu = torch.cuda.get_device_properties(dev).multi_processor_count
    except Exception:  # CPU / interpreter
        n_cu = 64
    return int(HCF_CFG["bif_ppc"]) * int(n_cu)


def _bif_ok(D, K, kind="bif_collapse") -> bool:
    f = HCF_CFG
    bc = f[kind][1]
    return bool(f["bif"]) and K == 4 and D % bc == 0 and (kind in ("bif_collapse", "bif_expand") or bc == f["t2_bc"])


def _gluon_b3_enabled() -> bool:
    return _HAVE_GLUON and os.environ.get("PRIMUS_HC_FUSED_GLUON", "1") != "0"



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
    DO_COLL: tl.constexpr = True,
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

    s0 = tl.load(SCALE + 0).to(tl.float32)
    s1 = tl.load(SCALE + 1).to(tl.float32)
    s2 = tl.load(SCALE + 2).to(tl.float32)
    b_pre = tl.load(BASE + kk).to(tl.float32)
    b_post = tl.load(BASE + K + kk).to(tl.float32)
    b_comb = tl.load(BASE + 2 * K + kk[:, None] * K + kk[None, :]).to(tl.float32)
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

    if DO_COLL:
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
# 2-D streaming kernels: one [BT rows x 4 streams x BC cols] tile per program, no loop; the row reductions
# go to per-column-tile partials [NC, M, ...] that the row-math kernel B2 sums (NC = D / BC).
# ------------------------------------------------------------------------------------------------


@triton.jit
def _hc_collapse2d_kernel(X, PRE, COLL, M, D, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    t = pid_t * BT + tl.arange(0, BT)
    cc = pid_c * BC + tl.arange(0, BC)
    kk = tl.arange(0, K)
    tm = t < M
    t64 = t.to(tl.int64)
    x3 = tl.load(X + t64[:, None, None] * (K * D) + kk[None, :, None] * D + cc[None, None, :],
                 mask=tm[:, None, None], other=0.0).to(tl.float32)
    p = tl.load(PRE + t64[:, None] * K + kk[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
    c = tl.sum(p[:, :, None] * x3, axis=1)
    tl.store(COLL + t64[:, None] * D + cc[None, :], c.to(COLL.dtype.element_ty), mask=tm[:, None])


@triton.jit
def _hc_dpre2d_kernel(X, DC, DPREP, M, D, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    t = pid_t * BT + tl.arange(0, BT)
    cc = pid_c * BC + tl.arange(0, BC)
    kk = tl.arange(0, K)
    tm = t < M
    t64 = t.to(tl.int64)
    x3 = tl.load(X + t64[:, None, None] * (K * D) + kk[None, :, None] * D + cc[None, None, :],
                 mask=tm[:, None, None], other=0.0).to(tl.float32)
    dc = tl.load(DC + t64[:, None] * D + cc[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
    dp = tl.sum(x3 * dc[:, None, :], axis=2)
    tl.store(DPREP + pid_c * M * K + t64[:, None] * K + kk[None, :], dp, mask=tm[:, None])


@triton.jit
def _hc_post_bwd2d_kernel(G, X, OUT, POST, DOUT, DPOSTP, DCOMBTP, M, D,
                          K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    """dout = sum_h post_h dnew_h; partials dpost_h = <out, dnew_h>, dcomb^T[h, k] = <dnew_h, x_k> per column tile."""
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
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
        dch = tl.sum(x3 * g_h[:, None, :], axis=2)  # [BT, K(k)]
        tl.store(DCOMBTP + (pbase + t64[:, None]) * KK + h * K + kk[None, :], dch, mask=tm[:, None])
    tl.store(DOUT + t64[:, None] * D + cc[None, :], dout.to(DOUT.dtype.element_ty), mask=tm[:, None])


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
def _st_g(GH, GL, GTH, GTL, off_rm, off_t, v, m):
    h = v.to(tl.bfloat16)
    lo = (v - h.to(tl.float32)).to(tl.bfloat16)
    tl.store(GH + off_rm, h, mask=m)
    tl.store(GL + off_rm, lo, mask=m)
    tl.store(GTH + off_t, h, mask=m)
    tl.store(GTL + off_t, lo, mask=m)


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
    GH,
    GL,
    GTH,
    GTL,
    DPREP,
    NC,
    M,
    D,
    HAS_POST: tl.constexpr,
    SPLIT_G: tl.constexpr,
    EPS: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    NP: tl.constexpr,
    N_ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PARTS: tl.constexpr = False,
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
    if PARTS:  # per-column-tile partials from _hc_dpre2d_kernel
        for c in range(0, NC):
            dpre += tl.load(DPREP + c * M * K + r2 * K + kk[None, :], mask=m2, other=0.0)
    else:
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
    s0 = tl.load(SCALE + 0).to(tl.float32)
    s1 = tl.load(SCALE + 1).to(tl.float32)
    s2 = tl.load(SCALE + 2).to(tl.float32)
    b_pre = tl.load(BASE + kk).to(tl.float32)
    b_post = tl.load(BASE + K + kk).to(tl.float32)
    b_comb = tl.load(BASE + 2 * K + kk[:, None] * K + kk[None, :]).to(tl.float32)
    l_pre = raw_pre * rstd[:, None]
    l_post = raw_post * rstd[:, None]
    l_comb = raw_comb * rstd[:, None, None]
    pre_sig = tl.sigmoid(l_pre * s0 + b_pre[None, :])
    post_sig = tl.sigmoid(l_post * s1 + b_post[None, :])
    comb_logit = l_comb * s2 + b_comb[None, :, :]
    cmax = tl.max(comb_logit, axis=2, keep_dims=True)
    cexp = tl.exp(comb_logit - cmax)
    comb_sm = cexp / tl.sum(cexp, axis=2, keep_dims=True)

    if HAS_POST and PARTS:
        dpost = tl.zeros((BLOCK_M, K), dtype=tl.float32)
        dm_ = tl.zeros((BLOCK_M, K, K), dtype=tl.float32)
        for c in range(0, NC):
            dpost += tl.load(DPOST + c * M * K + r2 * K + kk[None, :], mask=m2, other=0.0)
            dm_ += tl.load(DCOMBT + c * M * KK + r3 * KK + j3 * K + i3, mask=m3, other=0.0)
    elif HAS_POST:
        dpost = tl.load(DPOST + r2 * K + kk[None, :], mask=m2, other=0.0)
        dm_ = tl.load(DCOMBT + r3 * KK + j3 * K + i3, mask=m3, other=0.0)  # dcomb[i, j] = dcomb^T[j, i]
    else:
        dpost = tl.zeros((BLOCK_M, K), dtype=tl.float32)
        dm_ = tl.zeros((BLOCK_M, K, K), dtype=tl.float32)

    # Sinkhorn VJP (as sinkhorn.py)
    SROW: tl.constexpr = 2 * N_ITERS * KK
    sb = STATES + r3 * SROW + i3 * K + j3
    for i in tl.static_range(N_ITERS - 1):  # unrolled: the state loads do not depend on dm_
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
    if SPLIT_G:  # bf16 hi/lo of g, row-major [M, BN] and transposed [BN, M], for the gluon B3
        _st_g(GH, GL, GTH, GTL, r2 * BN + kk[None, :], kk[None, :] * M + r2, dl_pre * rstd[:, None], m2)
        _st_g(GH, GL, GTH, GTL, r2 * BN + K + kk[None, :], (K + kk[None, :]) * M + r2, dl_post * rstd[:, None], m2)
        _st_g(GH, GL, GTH, GTL, r3 * BN + 2 * K + i3 * K + j3, (2 * K + i3 * K + j3) * M + r3,
              dl_comb * rstd[:, None, None], m3)
        if NPAD > 0:
            pc2 = tl.arange(0, NPAD)
            _st_g(GH, GL, GTH, GTL, r2 * BN + (2 + K) * K + pc2[None, :], ((2 + K) * K + pc2[None, :]) * M + r2,
                  tl.zeros((BLOCK_M, NPAD), dtype=tl.float32), m2)
    c = tl.sum(dl_pre * raw_pre, axis=1) + tl.sum(dl_post * raw_post, axis=1) + tl.sum(tl.sum(dl_comb * raw_comb, axis=2), axis=1)
    tl.store(COEF + rows64, rstd * rstd * rstd * c / KD, mask=rmask)

    # per-program partials: [ds0, ds1, ds2, dbase(N)]
    NPR = tl.num_programs(0)  # partials are stored transposed: PART[j, pid]
    pp = PART + pid
    tl.store(pp + 0 * NPR, tl.sum(tl.sum(l_pre * d_pre_l, axis=1), axis=0))
    tl.store(pp + 1 * NPR, tl.sum(tl.sum(l_post * d_post_l, axis=1), axis=0))
    tl.store(pp + 2 * NPR, tl.sum(tl.sum(tl.sum(l_comb * d_comb_l, axis=2), axis=1), axis=0))
    tl.store(pp + (3 + kk) * NPR, tl.sum(d_pre_l, axis=0))
    tl.store(pp + (3 + K + kk) * NPR, tl.sum(d_post_l, axis=0))
    tl.store(pp + (3 + 2 * K + kk[:, None] * K + kk[None, :]) * NPR, tl.sum(d_comb_l, axis=0))


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


@triton.jit
def _hc_wt_split_kernel(W, WTH, WTL, N, KD, BN: tl.constexpr, BLOCK: tl.constexpr, W3=None,
                        WRITE_W3: tl.constexpr = False):
    """W [N, KD] fp32 -> W^T hi/lo [KD, BN] bf16 (zero for n >= N); optionally W3 [3, BN, KD] bf16 hi/mid/lo (F1)."""
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = tl.arange(0, BN)
    m = (c[:, None] < KD) & (n[None, :] < N)
    v = tl.load(W + n[None, :].to(tl.int64) * KD + c[:, None], mask=m, other=0.0).to(tl.float32)  # W may be bf16 (PRIMUS_HC_NOCAST)
    h = v.to(tl.bfloat16)
    o = c[:, None].to(tl.int64) * BN + n[None, :]
    om = c[:, None] < KD
    r = v - h.to(tl.float32)
    tl.store(WTH + o, h, mask=om)
    tl.store(WTL + o, r.to(tl.bfloat16), mask=om)
    if WRITE_W3:
        mid = r.to(tl.bfloat16)
        lo = (r - mid.to(tl.float32)).to(tl.bfloat16)
        o3 = n[None, :].to(tl.int64) * KD + c[:, None]
        PART = BN * KD
        tl.store(W3 + o3, h, mask=om)
        tl.store(W3 + PART + o3, mid, mask=om)
        tl.store(W3 + 2 * PART + o3, lo, mask=om)


if _HAVE_GLUON:

    @gluon.jit
    def _hc_b3_gl_kernel(X, WTH, WTL, GH, GL, GTH, GTL, COEF, PRE, COMBT, DC, DNEW, DX, DWP, M, D, T_PER, N_D, N_TILES,
                         BT: gl.constexpr, BC: gl.constexpr, BN: gl.constexpr, HAS_DNEW: gl.constexpr,
                         WMMA: gl.constexpr, OPA: gl.constexpr, OPB: gl.constexpr, BLK: gl.constexpr,
                         SMEM: gl.constexpr, SMEM_W: gl.constexpr = None, WLDS: gl.constexpr = False,
                         LATE: gl.constexpr = False):
        """B3 with explicit layouts: one program = one stream k x BC columns x T_PER rows (K = 4).

        dx_k = g @ W_k (bf16 hi/lo WMMA, 3 products) - x_k*coef + pre_k*dcoll + sum_h comb^T[h,k]*dnew_h,
        dW_k += g^T @ x_k (x_k staged through LDS, read back with ds_load_tr as the B operand).
        The 4 stream programs of a tile get pids p, p+8, p+16, p+24 (same XCD) so dnew/dcoll hit L2.
        hc track (PRIMUS_HC_B3_VAR): WLDS keeps the loop-invariant W^T hi/lo B operands in LDS instead of VGPRs, LATE loads
        the 4 dnew planes after the WMMA instead of hoisting them; both only cut VGPRs (326 -> 190 under Triton main),
        the arithmetic is unchanged (dx / dW partials byte-identical).
        """
        pid = gl.program_id(0)
        u = pid % 8
        v = pid // 8
        k = v % 4
        tile = (v // 4) * 8 + u
        if tile < N_TILES:
            pid_d = tile % N_D
            pid_s = tile // N_D
            c0 = pid_d * BC
            KD = 4 * D
            nb = gl.arange(0, BN, layout=gl.SliceLayout(1, OPB))
            cb = c0 + gl.arange(0, BC, layout=gl.SliceLayout(0, OPB))
            woff = gl.expand_dims((k * D + cb) * BN, 0) + gl.expand_dims(nb, 1)
            wh = gl.load(WTH + woff)
            wl = gl.load(WTL + woff)
            if WLDS:
                wsh = gl.allocate_shared_memory(gl.bfloat16, [BN, BC], SMEM_W)
                wsl = gl.allocate_shared_memory(gl.bfloat16, [BN, BC], SMEM_W)
                wsh.store(wh)
                wsl.store(wl)
            na_r = gl.arange(0, BN, layout=gl.SliceLayout(0, OPA))
            na_c = gl.arange(0, BN, layout=gl.SliceLayout(1, OPA))
            tb = gl.arange(0, BT, layout=gl.SliceLayout(1, BLK))
            cr = c0 + gl.arange(0, BC, layout=gl.SliceLayout(0, BLK))
            accw = gl.zeros([BN, BC], gl.float32, layout=WMMA)
            xs = gl.allocate_shared_memory(gl.bfloat16, [BT, BC], SMEM)
            t_lo = pid_s * T_PER
            for t0 in range(t_lo, t_lo + T_PER, BT):
                tr = t0 + tb
                tr64 = tr.to(gl.int64)
                rowoff = gl.expand_dims(tr64 * KD, 1) + gl.expand_dims(cr, 0)
                xk = gl.load(X + rowoff + k * D)
                dcv = gl.load(DC + gl.expand_dims(tr64 * D, 1) + gl.expand_dims(cr, 0)).to(gl.float32)
                cf = gl.expand_dims(gl.load(COEF + tr), 1)
                pk = gl.expand_dims(gl.load(PRE + tr * 4 + k).to(gl.float32), 1)
                if HAS_DNEW:  # issue every streaming load before the WMMA / layout conversion
                    if not LATE:
                        n0 = gl.load(DNEW + rowoff)
                        n1 = gl.load(DNEW + rowoff + D)
                        n2 = gl.load(DNEW + rowoff + 2 * D)
                        n3 = gl.load(DNEW + rowoff + 3 * D)
                    q0 = gl.expand_dims(gl.load(COMBT + tr * 16 + 0 + k).to(gl.float32), 1)
                    q1 = gl.expand_dims(gl.load(COMBT + tr * 16 + 4 + k).to(gl.float32), 1)
                    q2 = gl.expand_dims(gl.load(COMBT + tr * 16 + 8 + k).to(gl.float32), 1)
                    q3 = gl.expand_dims(gl.load(COMBT + tr * 16 + 12 + k).to(gl.float32), 1)
                ta = t0 + gl.arange(0, BT, layout=gl.SliceLayout(1, OPA))
                goff = gl.expand_dims(ta, 1) * BN + gl.expand_dims(na_r, 0)
                gh = gl.load(GH + goff)
                gq = gl.load(GL + goff)
                if WLDS:
                    wh = wsh.load(OPB)
                    wl = wsl.load(OPB)
                acc = gl.zeros([BT, BC], gl.float32, layout=WMMA)
                acc = gl.amd.gfx1250.wmma(gh, wh, acc)
                acc = gl.amd.gfx1250.wmma(gh, wl, acc)
                acc = gl.amd.gfx1250.wmma(gq, wh, acc)
                out = gl.convert_layout(acc, BLK) - xk.to(gl.float32) * cf + pk * dcv
                if HAS_DNEW:
                    if LATE:
                        n0 = gl.load(DNEW + rowoff)
                        n1 = gl.load(DNEW + rowoff + D)
                        n2 = gl.load(DNEW + rowoff + 2 * D)
                        n3 = gl.load(DNEW + rowoff + 3 * D)
                    out = (out + q0 * n0.to(gl.float32) + q1 * n1.to(gl.float32) + q2 * n2.to(gl.float32)
                           + q3 * n3.to(gl.float32))
                gl.store(DX + rowoff + k * D, out.to(gl.bfloat16))
                tc = t0 + gl.arange(0, BT, layout=gl.SliceLayout(0, OPA))
                gto = gl.expand_dims(na_c, 1) * M + gl.expand_dims(tc, 0)
                gth = gl.load(GTH + gto)
                gtl = gl.load(GTL + gto)
                gl.barrier()
                xs.store(xk)
                gl.barrier()
                xb = xs.load(OPB)
                accw = gl.amd.gfx1250.wmma(gth, xb, accw)
                accw = gl.amd.gfx1250.wmma(gtl, xb, accw)
            nr = gl.arange(0, BN, layout=gl.SliceLayout(1, WMMA))
            cw = c0 + gl.arange(0, BC, layout=gl.SliceLayout(0, WMMA))
            gl.store(DWP + pid_s * BN * KD + gl.expand_dims(nr.to(gl.int64) * KD, 1) + k * D + gl.expand_dims(cw, 0),
                     accw)

    _GL_LAYOUTS = {}

    def _gl_layouts(bt, bc, nw):
        key = (bt, bc, nw)
        if key not in _GL_LAYOUTS:
            wb = [[0, 1 << i] for i in range(nw.bit_length() - 1)]
            wm = gl.amd.AMDWMMALayout(version=3, transposed=True, warp_bases=wb, instr_shape=[16, 16, 32])
            opa = gl.DotOperandLayout(operand_index=0, parent=wm, k_width=8)
            opb = gl.DotOperandLayout(operand_index=1, parent=wm, k_width=8)
            tpw = [256 // bc, bc // 8] if bc < 256 else [1, 32]
            blk = gl.BlockedLayout([1, 8], tpw, [nw, 1], [1, 0])
            smem = gl.PaddedSharedLayout.with_identity_for([[bc, 16]], [bt, bc], [1, 0])
            _GL_LAYOUTS[key] = (wm, opa, opb, blk, smem)
        return _GL_LAYOUTS[key]

    def _gl_smem_w(bn, bc):
        key = ("w", bn, bc)
        if key not in _GL_LAYOUTS:
            _GL_LAYOUTS[key] = gl.PaddedSharedLayout.with_identity_for([[bc, 16]], [bn, bc], [1, 0])
        return _GL_LAYOUTS[key]


def _b3_var():
    """PRIMUS_HC_B3_VAR=wlds,late (each 0/1); default 0,0 = the original register schedule."""
    v = os.environ.get("PRIMUS_HC_B3_VAR", "").strip()
    if not v:
        return False, False
    a, b = (int(t) for t in v.split(","))
    return bool(a), bool(b)


def _gl_b3_ok(M, D, K):
    f = HCF_CFG
    return _gluon_b3_enabled() and K == 4 and D % f["gl_bc"] == 0 and M % (f["gl_tsplit"] * f["gl_bt"]) == 0


@triton.jit
def _hc_reduce_kernel(DWP, DW, PART, PSUM, NS, NTOT, P, N_DW, NP: tl.constexpr, NPP: tl.constexpr,
                      BLOCK: tl.constexpr, BR: tl.constexpr):
    """Deterministic reductions in one launch: DW[i] = sum_s DWP[s, i] (first N_DW entries of the [BN, KD] partials)
    and, in the last program, PSUM[j] = sum_p PART[p, j] (dscale / dbase)."""
    pid = tl.program_id(0)
    if pid * BLOCK < N_DW:
        o = pid * BLOCK + tl.arange(0, BLOCK)
        m = o < N_DW
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for s_ in range(0, NS):
            acc += tl.load(DWP + s_ * NTOT + o, mask=m, other=0.0)
        tl.store(DW + o, acc, mask=m)
    else:  # one program per dscale/dbase entry j: PART is [NP, P]
        j = pid - tl.cdiv(N_DW, BLOCK)
        acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
        for p0 in range(0, P, BLOCK):
            r = p0 + tl.arange(0, BLOCK)
            acc2 += tl.load(PART + j * P + r, mask=r < P, other=0.0)
        tl.store(PSUM + j, tl.sum(acc2, axis=0))


def _hc_reduce(dw_p, n, part):
    """-> (dW [n, KD] fp32, psum [NP] fp32) from dw_p [S, BN, KD] and part [P, NP]."""
    S_, BN_, KD_ = dw_p.shape
    n_dw = n * KD_
    dw = torch.empty((n, KD_), dtype=torch.float32, device=dw_p.device)
    NP_, P_ = part.shape
    psum = torch.empty((NP_,), dtype=torch.float32, device=dw_p.device)
    BLOCK = 2048
    _hc_reduce_kernel[(triton.cdiv(n_dw, BLOCK) + NP_,)](
        dw_p, dw, part, psum, S_, BN_ * KD_, P_, n_dw, NP=NP_, NPP=triton.next_power_of_2(NP_), BLOCK=BLOCK, BR=64,
        num_warps=4,
    )
    return dw, psum


@triton.jit
def _hc_expand_rms_fwd_kernel(X, OUT, POST, COMBT, NEW, WTH, WTL, P_RAW, P_SS, M, D,
                              K: tl.constexpr, BN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    """expand (new_h = post_h*out + sum_k comb^T[h,k] x_k) + the NEXT HyperMixer's rms-linear partials on the
    bf16 new tile: P_RAW[pid_d] = new @ W_next^T (bf16 hi/lo WMMA), P_SS[pid_d] = sum(new^2).
    Removes the next HC's separate 128 MB read of x (g90mem round 3)."""
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    k_ax = tl.arange(0, K)
    ncol = tl.arange(0, BN)
    mask_m = rm < M
    rm64 = rm.to(tl.int64)
    KD = K * D
    KK: tl.constexpr = K * K
    x3 = tl.load(X + rm64[:, None, None] * KD + k_ax[None, :, None] * D + rd[None, None, :],
                 mask=mask_m[:, None, None], other=0.0).to(tl.float32)
    out_tile = tl.load(OUT + rm64[:, None] * D + rd[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BN), dtype=tl.float32)
    ss = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for h in tl.static_range(K):
        comb_h = tl.load(COMBT + rm64[:, None] * KK + h * K + k_ax[None, :], mask=mask_m[:, None], other=0.0).to(
            tl.float32)
        post_h = tl.load(POST + rm64 * K + h, mask=mask_m, other=0.0).to(tl.float32)
        new_h = (post_h[:, None] * out_tile + tl.sum(comb_h[:, :, None] * x3, axis=1)).to(NEW.dtype.element_ty)
        tl.store(NEW + rm64[:, None] * KD + h * D + rd[None, :], new_h, mask=mask_m[:, None])
        wo = (h * D + rd)[:, None].to(tl.int64) * BN + ncol[None, :]
        acc = tl.dot(new_h, tl.load(WTH + wo), acc)
        acc = tl.dot(new_h, tl.load(WTL + wo), acc)
        nf = new_h.to(tl.float32)
        ss += tl.sum(nf * nf, axis=1)
    tl.store(P_RAW + (pid_d * M + rm64[:, None]) * BN + ncol[None, :], acc, mask=mask_m[:, None])
    tl.store(P_SS + pid_d * M + rm64, ss, mask=mask_m)


# ------------------------------------------------------------------------------------------------
# autograd
# ------------------------------------------------------------------------------------------------


class HCLink:
    """Carries the post-side backward results (dnew, dpost, dcomb^T) to the pre-side backward."""

    __slots__ = ("dnew", "dpost", "dcombt", "nc")

    def __init__(self):
        self.nc = 0
        self.dnew = None
        self.dpost = None
        self.dcombt = None


class HCPreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scale, base, eps, n_iters, link, gamma=None, norm_eps=1e-6):  # type: ignore[override]
        K, D = x.shape[-2], x.shape[-1]
        x3 = x.reshape(-1, K, D)
        M = x3.shape[0]
        KD = K * D
        x2d = x3.view(M, KD)
        # PRIMUS_HC_NOCAST=1: the kernels read bf16 fn.weight / scale / base directly (exact upcast in-kernel) instead
        # of 3 torch casts per HC application; only on the wmma3 + gluon path, where w feeds _hc_wt_split_kernel only.
        stash = getattr(x, "_hc_stash", None)  # rms-linear partials computed by the previous HC's expand
        xrms_stash = stash is not None and stash[2] is None  # fuse-hc PRIMUS_HC_XRMS (bf16 W, no W^T split yet)
        nocast = (_hc_nocast() and weight.is_contiguous() and weight.dtype in (torch.float32, torch.bfloat16)
                  and HCF_CFG["f1_wmma3"] and x.dtype == torch.bfloat16 and _gl_b3_ok(M, D, K)
                  and (stash is None or xrms_stash))
        w = weight if nocast else weight.float().contiguous()
        N = w.shape[0]
        BN = _block_n(N)
        assert N == (2 + K) * K
        dev = x.device
        if stash is not None:
            x._hc_stash = None
            if not (stash[0] is weight and stash[1] == weight._version):
                stash = None
        # fuse-hc PRIMUS_HC_B3X: the all-stream B3 reads bf16 fn.weight directly (no W^T hi/lo split needed)
        direct_w = (b3x_enabled() and _gl_b3_ok(M, D, K) and w.dtype == torch.bfloat16 and w.is_contiguous()
                    and x.dtype == torch.bfloat16 and D % b3x_cfg()["bc"] == 0)
        if stash is not None:
            _, _, wth_s, wtl_s, p_raw, p_ss, n_split = stash
            if wth_s is None and _gl_b3_ok(M, D, K) and not direct_w:  # W^T hi/lo for B3 (same kernel / bits as the wmma3 branch)
                wth_s = torch.empty((KD, BN), dtype=torch.bfloat16, device=dev)
                wtl_s = torch.empty((KD, BN), dtype=torch.bfloat16, device=dev)
                _hc_wt_split_kernel[(triton.cdiv(KD, 256),)](w, wth_s, wtl_s, N, KD, BN=BN, BLOCK=256, num_warps=4)
        elif HCF_CFG["f1_wmma3"] and x.dtype == torch.bfloat16 and _gl_b3_ok(M, D, K):
            f1 = HCF_CFG
            wth_s = torch.empty((KD, BN), dtype=torch.bfloat16, device=dev)
            wtl_s = torch.empty((KD, BN), dtype=torch.bfloat16, device=dev)
            w3 = torch.empty((3, BN, KD), dtype=torch.bfloat16, device=dev)
            _hc_wt_split_kernel[(triton.cdiv(KD, 256),)](w, wth_s, wtl_s, N, KD, BN=BN, BLOCK=256, W3=w3, WRITE_W3=True,
                                                         num_warps=4)
            chunk = f1["f1_chunk"]
            n_split = triton.cdiv(KD, chunk)
            p_raw = torch.empty((n_split, M, BN), dtype=torch.float32, device=dev)
            p_ss = torch.empty((n_split, M), dtype=torch.float32, device=dev)
            _hc_fwd_wmma3_kernel[(triton.cdiv(M, f1["f1_bt"]), n_split)](
                x2d, w3, p_raw, p_ss, M, KD, KD, chunk,
                BLOCK_T=f1["f1_bt"], BLOCK_K=f1["f1_bk"], BLOCK_N=BN, num_warps=f1["f1_warps"], num_stages=f1["f1_stages"],
            )
            stash = (weight, weight._version, wth_s, wtl_s, p_raw, p_ss, n_split)  # reuse the W^T split below
        else:
            c = HC_CFG
            BLOCK_T, BLOCK_K, chunk_cfg = c["fwd_bt"], c["fwd_bk"], c["fwd_chunk"]
            n_split = max(1, triton.cdiv(KD, chunk_cfg))
            chunk = triton.cdiv(triton.cdiv(KD, n_split), BLOCK_K) * BLOCK_K
            n_split = triton.cdiv(KD, chunk)
            p_raw = torch.empty((n_split, M, BN), dtype=torch.float32, device=dev)
            p_ss = torch.empty((n_split, M), dtype=torch.float32, device=dev)
            _hc_rms_linear_fwd_kernel[(triton.cdiv(M, BLOCK_T), n_split)](
                x2d, w, p_raw, p_ss, M, KD, N, KD, chunk, 1, KD,
                BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K, BLOCK_N=BN, PREC="ieee",
                num_warps=c["fwd_warps"], num_stages=c["fwd_stages"],
            )
        f = HCF_CFG
        scale_c = scale.contiguous() if nocast else scale.float().contiguous()
        base_c = base.contiguous() if nocast else base.float().contiguous()
        two_d = f["two_d"] and D % f["t2_bc"] == 0
        fold = gamma is not None and two_d and K == 4 and triton.next_power_of_2(D) == D
        rstd_n = None
        raw = torch.empty((M, BN), dtype=torch.float32, device=dev)
        rstd = torch.empty((M,), dtype=torch.float32, device=dev)
        states = torch.empty((M, 2 * n_iters, K, K), dtype=torch.float32, device=dev)
        pre = torch.empty((M, K), dtype=x.dtype, device=dev)
        post = torch.empty((M, K), dtype=x.dtype, device=dev)
        combt = torch.empty((M, K, K), dtype=x.dtype, device=dev)
        coll = torch.empty(x.shape[:-2] + (D,), dtype=x.dtype, device=dev)
        bm = f["rm_bm"] if two_d else f["row_bm"]
        _hc_pre_fwd_kernel[(triton.cdiv(M, bm),)](
            x3, p_raw, p_ss, scale_c, base_c, raw, rstd, states, pre, post, combt, coll,
            M, D, n_split, float(eps),
            EPS=float(eps), K=K, BN=BN, N_ITERS=int(n_iters), BLOCK_M=bm, BLOCK_D=f["row_bd"],
            DO_COLL=not two_d, num_warps=f["rm_warps"] if two_d else f["row_warps"],
        )
        if fold:
            # y = rmsnorm(collapse) * gamma replaces the collapsed output (the sub-block skips its norm)
            y2, rstd_n, fq = collapse_norm_fwd(x3, pre, gamma.contiguous(), norm_eps, fp8=f["fold_fp8"])
            coll = y2.view(x.shape[:-2] + (D,))
            if fq is not None:
                coll._fp8_row = fq
        elif two_d and _bif_ok(D, K, "bif_collapse"):
            bt, bc, nw = f["bif_collapse"]
            g_ = bif_grid(M, D, bt, bc, _bif_nprog(dev))
            _hc_collapse_bif_kernel[g_](x3, pre, coll, M, D, BT=bt, BC=bc, NPROG=g_[0], num_warps=nw)
        elif two_d:
            _hc_collapse2d_kernel[(triton.cdiv(M, f["t2_bt"]), D // f["t2_bc"])](
                x3, pre, coll, M, D, K=K, BT=f["t2_bt"], BC=f["t2_bc"], num_warps=f["t2_warps"],
            )
        if direct_w:
            wth = wtl = w
        elif stash is not None and _gl_b3_ok(M, D, K):
            wth, wtl = wth_s, wtl_s
        elif _gl_b3_ok(M, D, K):
            wth = torch.empty((KD, BN), dtype=torch.bfloat16, device=dev)
            wtl = torch.empty((KD, BN), dtype=torch.bfloat16, device=dev)
            _hc_wt_split_kernel[(triton.cdiv(KD, 256),)](w, wth, wtl, N, KD, BN=BN, BLOCK=256, num_warps=4)
        else:
            wth = wtl = w
        ctx.save_for_backward(x3, w, scale_c, base_c, raw, rstd, states, pre, combt, wth, wtl,
                              gamma if fold else None, rstd_n)
        ctx.fold = fold
        ctx.direct_w = direct_w
        ctx.gamma_needs_grad = gamma is not None and gamma.requires_grad
        # fuse-hc PRIMUS_HC_GRAD_DIRECT: the backward tail writes main_grad itself (claimed here, as the other fused
        # writers do); only on the gluon-B3 path, which is the only backward branch that runs the tail.
        ctx.direct = None
        if grad_direct_enabled() and wth.dtype == torch.bfloat16 and (not fold or ctx.gamma_needs_grad):
            ctx.direct = (weight, scale, base, gamma if fold else None)
            ctx.claimed = tuple(claim_main_grad(p) if p is not None else False for p in ctx.direct)
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
        x3, w, scale_c, base_c, raw, rstd, states, pre, combt, wth, wtl, gamma, rstd_n = ctx.saved_tensors
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
        part = torch.empty((NP, n_prog), dtype=torch.float32, device=dev)
        use_gl = wth.dtype == torch.bfloat16
        if use_gl:
            gh = torch.empty((M, BN), dtype=torch.bfloat16, device=dev)
            glo = torch.empty((M, BN), dtype=torch.bfloat16, device=dev)
            gth = torch.empty((BN, M), dtype=torch.bfloat16, device=dev)
            gtl = torch.empty((BN, M), dtype=torch.bfloat16, device=dev)
        else:
            gh = glo = gth = gtl = g
        two_d = f["two_d"] and D % f["t2_bc"] == 0 and (not has_post or link.nc > 0)
        dgamma = None
        dgp = None
        tail = use_gl and ctx.direct is not None
        if ctx.fold and tail:
            nc = D // f["t2_bc"]
            dcoll, dprep, dgp = norm_dpre_bwd(x3, pre, gamma, rstd_n, dcoll, f["t2_bc"], reduce=False)
            n_prog = triton.cdiv(M, f["rm_bm"])
            part = torch.empty((NP, n_prog), dtype=torch.float32, device=dev)
        elif ctx.fold:
            assert two_d, "folded RMSNorm needs the 2-D HC backward (D % t2_bc == 0 and the 2-D post backward)"
            # dcoll arrives as dy (grad of the norm output): dh = rmsnorm bwd, written as the HC dcoll
            nc = D // f["t2_bc"]
            dcoll, dprep, dgamma = norm_dpre_bwd(x3, pre, gamma, rstd_n, dcoll, f["t2_bc"])
            if not ctx.gamma_needs_grad:
                dgamma = None
            n_prog = triton.cdiv(M, f["rm_bm"])
            part = torch.empty((NP, n_prog), dtype=torch.float32, device=dev)
        elif two_d:
            nc = D // f["t2_bc"]
            dprep = torch.empty((nc, M, K), dtype=torch.float32, device=dev)
            if _bif_ok(D, K, "bif_dpre"):
                bt, bc, nw = f["bif_dpre"]
                g_ = bif_grid(M, D, bt, bc, _bif_nprog(dev))
                _hc_dpre_bif_kernel[g_](x3, dcoll, dprep, M, D, BT=bt, BC=bc, NPROG=g_[0], num_warps=nw)
            else:
                _hc_dpre2d_kernel[(triton.cdiv(M, f["t2_bt"]), nc)](
                    x3, dcoll, dprep, M, D, K=K, BT=f["t2_bt"], BC=f["t2_bc"], num_warps=f["t2_warps"],
                )
            n_prog = triton.cdiv(M, f["rm_bm"])
            part = torch.empty((NP, n_prog), dtype=torch.float32, device=dev)
        else:
            nc = 0
            dprep = g
        _hc_pre_bwd_kernel[(n_prog,)](
            x3, dcoll, raw, rstd, states, scale_c, base_c,
            link.dpost if has_post else g, link.dcombt if has_post else g,
            g, coef, part, gh, glo, gth, gtl, dprep, nc, M, D,
            HAS_POST=has_post, SPLIT_G=use_gl, PARTS=two_d, EPS=ctx.eps, K=K, BN=BN, NP=NP, N_ITERS=ctx.n_iters,
            BLOCK_M=f["rm_bm"] if two_d else f["row_bm"], BLOCK_D=f["row_bd"],
            num_warps=f["rm_warps"] if two_d else f["row_warps"],
        )
        if use_gl:
            bt, bc, ts, nw = f["gl_bt"], f["gl_bc"], f["gl_tsplit"], f["gl_warps"]
            wm, opa, opb, blk, smem = _gl_layouts(bt, bc, nw)
            n_d = D // bc
            if b3x_enabled() and D % b3x_cfg()["bc"] == 0:
                # splitk PRIMUS_SPLITK_HC_TS=<ts>: more, shorter B3X programs (row splits); the dW partials are
                # summed by the tail / _hc_reduce in one fixed order (deterministic; dW bits differ from ts = 8)
                ts_x = b3x_splitk_ts()
                if ts_x and M % b3x_cfg()["bt"] == 0 and M // b3x_cfg()["bt"] >= ts_x and not b3x_cfg()["pf"]:
                    ts = ts_x
            n_tiles = n_d * ts
            dx = torch.empty_like(x3)
            dw_p = torch.empty((ts, BN, KD), dtype=torch.float32, device=dev)
            if b3x_enabled() and D % b3x_cfg()["bc"] == 0:
                # fuse-hc: all 4 streams per program (x / dnew / dcollapsed loaded once per tile), same bits
                xc = b3x_cfg()
                ekw = {}
                xbt = xc["bt"] if (M // ts) % xc["bt"] == 0 else bt
                if b3x_splitk_ts() and ts == b3x_splitk_ts() and M % (ts * xc["bt"]) != 0:
                    xbt = xc["bt"]
                    ekw["UNEVEN"] = True  # splitk: ts does not divide M / bt (uneven fixed row splits)
                wm, opa, opb, blk, smem, smem_w = b3x_layouts(xbt, xc["bc"], xc["w"])
                if xc["ewm"]:
                    blk = wm
                n_dx = D // xc["bc"]
                kern = _hc_b3x_kernel
                ekw.update(PF=bool(xc["pf"]), BUF=bool(xc["buf"]) and M * KD * 2 < 2 ** 31)
                kern[(n_dx * ts,)](
                    x3, w, wth, wtl, gh, glo, gth, gtl, coef, pre, combt, dcoll, dnew if dnew is not None else x3, dx,
                    dw_p, M, D, N, M // ts, n_dx, n_dx * ts,
                    BT=xbt, BC=xc["bc"], BN=BN, HAS_DNEW=dnew is not None, WMMA=wm, OPA=opa, OPB=opb, BLK=blk,
                    SMEM=smem, SMEM_W=smem_w, DIRECT_W=ctx.direct_w, num_warps=xc["w"], **ekw, **b3x_splitk_kw(),
                )
            else:
                assert not ctx.direct_w
                wlds, late = _b3_var()
                vkw = dict(SMEM_W=_gl_smem_w(BN, bc), WLDS=True) if wlds else {}
                if late:
                    vkw["LATE"] = True
                _hc_b3_gl_kernel[(triton.cdiv(n_tiles, 8) * 8 * K,)](
                    x3, wth, wtl, gh, glo, gth, gtl, coef, pre, combt, dcoll, dnew if dnew is not None else x3, dx,
                    dw_p, M, D, M // ts, n_d, n_tiles,
                    BT=bt, BC=bc, BN=BN, HAS_DNEW=dnew is not None, WMMA=wm, OPA=opa, OPB=opb, BLK=blk, SMEM=smem,
                    num_warps=nw, **vkw,
                )
            if link is not None:
                link.dnew = link.dpost = link.dcombt = None
            if tail:
                # fuse-hc: dW / dscale / dbase / dgamma reductions in one launch, straight into main_grad
                gd = (ctx.weight_dtype, ctx.scale_dtype, ctx.base_dtype, gamma.dtype if ctx.fold else torch.float32)
                gw, gs, gb, gg = hc_tail(dw_p, N, part, dgp if ctx.fold else None, D, ctx.direct, gd, ctx.claimed)
                ctx.direct = None
                return dx.view(ctx.x_shape), gw, gs, gb, None, None, None, gg, None
            dw, psum = _hc_reduce(dw_p, N, part)
            return (dx.view(ctx.x_shape), dw.to(ctx.weight_dtype), psum[:3].to(ctx.scale_dtype),
                    psum[3:].to(ctx.base_dtype), None, None, None, dgamma, None)
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
        psum = part.sum(1)
        dscale = psum[:3].to(ctx.scale_dtype)
        dbase = psum[3:].to(ctx.base_dtype)
        return dx.view(ctx.x_shape), dw.to(ctx.weight_dtype), dscale, dbase, None, None, None, dgamma, None


class HCPostFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out, post, combt, link, next_w=None):  # type: ignore[override]
        K, D = x.shape[-2], x.shape[-1]
        x3 = x.reshape(-1, K, D)
        M = x3.shape[0]
        out_c = out.contiguous()
        post_c = post.reshape(M, K)
        combt_c = combt.reshape(M, K, K)
        new = torch.empty_like(x3)
        f = HCF_CFG
        stash = None
        if xrms_enabled() and xrms_supported(x3, next_w) and _gl_b3_ok(M, D, K) and f["f1_wmma3"]:
            # fuse-hc: expand + the next HC's rms-linear partials (its _hc_wt_split + _hc_fwd_wmma3 are skipped)
            new, p_raw, p_ss, n_split = expand_rms(x3, out_c, post_c, combt_c, next_w)
            stash = (next_w, next_w._version, None, None, p_raw, p_ss, n_split)
        elif (next_w is not None and f["fuse_next_rms"] and K == 4 and D % f["xr_bd"] == 0
                and next_w.shape[0] == (2 + K) * K and _gl_b3_ok(M, D, K)):
            KD = K * D
            N = next_w.shape[0]
            BN = _block_n(N)
            wn = next_w.float().contiguous()
            wth = torch.empty((KD, BN), dtype=torch.bfloat16, device=x.device)
            wtl = torch.empty((KD, BN), dtype=torch.bfloat16, device=x.device)
            _hc_wt_split_kernel[(triton.cdiv(KD, 256),)](wn, wth, wtl, N, KD, BN=BN, BLOCK=256, num_warps=4)
            n_d = D // f["xr_bd"]
            p_raw = torch.empty((n_d, M, BN), dtype=torch.float32, device=x.device)
            p_ss = torch.empty((n_d, M), dtype=torch.float32, device=x.device)
            _hc_expand_rms_fwd_kernel[(triton.cdiv(M, f["xr_bm"]), n_d)](
                x3, out_c, post_c, combt_c, new, wth, wtl, p_raw, p_ss, M, D,
                K=K, BN=BN, BLOCK_M=f["xr_bm"], BLOCK_D=f["xr_bd"], num_warps=f["xr_warps"],
            )
            stash = (next_w, next_w._version, wth, wtl, p_raw, p_ss, n_d)
        elif _bif_ok(D, K, "bif_expand"):
            bt, bc, nw = f["bif_expand"]
            g_ = bif_grid(M, D, bt, bc, _bif_nprog(x.device))
            _hc_expand_bif_kernel[g_](x3, out_c, post_c, combt_c, new, M, D, BT=bt, BC=bc, NPROG=g_[0], num_warps=nw)
        else:
            ebm, ebd, ew = _exp_tile()
            _hc_expand_fwd_kernel[(triton.cdiv(M, ebm), triton.cdiv(D, ebd))](
                x3, out_c, post_c, combt_c, new, M, D,
                K=K, BLOCK_M=ebm, BLOCK_D=ebd, num_warps=ew,
            )
        ctx.save_for_backward(x3, out_c, post_c)
        ctx.link = link
        ctx.out_shape = out.shape
        res = new.view(x.shape)
        if stash is not None:
            res._hc_stash = stash
        return res

    @staticmethod
    def backward(ctx, dnew):  # type: ignore[override]
        x3, out_c, post_c = ctx.saved_tensors
        M, K, D = x3.shape
        dnew = dnew.contiguous().view(M, K, D)
        dev = x3.device
        dout = torch.empty((M, D), dtype=out_c.dtype, device=dev)
        f = HCF_CFG
        link = ctx.link
        if f["two_d"] and D % f["t2_bc"] == 0:
            nc = D // f["t2_bc"]
            dpost = torch.empty((nc, M, K), dtype=torch.float32, device=dev)
            dcombt = torch.empty((nc, M, K, K), dtype=torch.float32, device=dev)
            if _bif_ok(D, K, "bif_post"):
                bt, bc, nw = f["bif_post"]
                g_ = bif_grid(M, D, bt, bc, _bif_nprog(dev))
                _hc_post_bwd_bif_kernel[g_](dnew, x3, out_c, post_c, dout, dpost, dcombt, M, D,
                                            BT=bt, BC=bc, NPROG=g_[0], num_warps=nw)
            else:
                pbt, pw = _post_tile()
                _hc_post_bwd2d_kernel[(triton.cdiv(M, pbt), nc)](
                    dnew, x3, out_c, post_c, dout, dpost, dcombt, M, D,
                    K=K, BT=pbt, BC=f["t2_bc"], num_warps=pw,
                )
            link.nc = nc
        else:
            dpost = torch.empty((M, K), dtype=torch.float32, device=dev)
            dcombt = torch.empty((M, K, K), dtype=torch.float32, device=dev)
            _hc_post_bwd_kernel[(triton.cdiv(M, f["row_bm"]),)](
                dnew, x3, out_c, post_c, dout, dpost, dcombt, M, D,
                K=K, BLOCK_M=f["row_bm"], BLOCK_D=f["row_bd"], num_warps=f["row_warps"],
            )
            link.nc = 0
        link.dnew, link.dpost, link.dcombt = dnew, dpost, dcombt
        return None, dout.view(ctx.out_shape), None, None, None, None


def hc_fused_pre(x, weight, scale, base, eps, n_iters, norm_weight=None, norm_eps=1e-6):
    """-> (collapsed [..., D], post [..., K], comb^T [..., K, K], link). With ``norm_weight`` the first output is
    ``rmsnorm(collapsed) * norm_weight`` (folded into the collapse / dpre kernels, opt3-mem)."""
    link = HCLink()
    coll, post, combt = HCPreFn.apply(x, weight, scale, base, eps, n_iters, link, norm_weight, norm_eps)
    return coll, post, combt, link


def hc_fused_post(x, out, post, combt, link, next_w=None):
    """new[..., h, :] = post_h * out + sum_k comb[k, h] x_k  (comb^T passed). With ``next_w`` (the next
    HyperMixer's fn.weight) the expand also produces that mixer's rms-linear partials, picked up by its
    ``hc_fused_pre`` through an attribute on the returned tensor."""
    return HCPostFn.apply(x, out, post, combt, link, next_w)


__all__ = ["hc_fused_enabled", "hc_fused_supported", "hc_fused_pre", "hc_fused_post", "hc_fold_supported", "HCF_CFG"]


def hc_fold_supported(D: int) -> bool:
    """The folded norm path needs the 2-D HC kernels and a power-of-two hidden (row kept in registers)."""
    f = HCF_CFG
    return bool(f["two_d"]) and D % f["t2_bc"] == 0 and triton.next_power_of_2(D) == D
