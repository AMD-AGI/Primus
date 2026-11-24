###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""RMSNorm folded into the HC collapse (F2) and dpre (B2) kernels, plus q-norm fused with the wq_b input quant
Enabled with PRIMUS_HC_FOLD_NORM=1 (HC folds).

Per V4 layer the two big norms (input_layernorm on the attention HC collapse, pre_mlp_layernorm on the FFN HC
collapse, [4096, 4096] bf16) read the collapsed hidden right after the HC wrote it, and in backward write a dx that
the HC dpre kernel immediately reads back:

  unfused fwd : collapse (read x 128 MB, write h 32 MB) -> norm (read h 32, write y 32)              4 launches/HC*
  fused fwd   : collapse+norm (read x 128 MB, write y 32 MB [+ FP8 y 16 MB + scales])              1 launch
  unfused bwd : norm bwd (read h, dy; write dh) + dg reduce + dpre2d (read x 128, dh 32)          3 launches
  fused bwd   : dpre+norm bwd (read x 128, dy 32; write dh 32 = HC dcoll) + dg reduce              2 launches
  (*rmsnorm_g90 fwd 1 + bwd 2 launches + HC collapse/dpre)

Saved per HC application ~96 MB of traffic and 2 launches (32 per step at 16 HC applications with a norm);
estimated 0.9 ms/step with the q-norm + quant fusion below.

Numerics are those of the unfused path: the collapsed hidden is rounded to bf16 before the norm (the unfused path
stores it in bf16), rstd / y / dh use the rmsnorm_g90 formulas, dh is rounded to bf16 before the dpre dot (as the
unfused dcoll), dpre partials are per t2_bc column tile (the layout _hc_pre_bwd_kernel reads).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def hc_fold_norm_enabled() -> bool:
    return os.environ.get("PRIMUS_HC_FOLD_NORM", "0") == "1"


@triton.jit
def _ld_rows4(X, t64, cols, m2, D):
    base = X + t64[:, None] * (4 * D) + cols[None, :]
    x0 = tl.load(base, mask=m2, other=0.0)
    x1 = tl.load(base + D, mask=m2, other=0.0)
    x2 = tl.load(base + 2 * D, mask=m2, other=0.0)
    x3 = tl.load(base + 3 * D, mask=m2, other=0.0)
    return x0, x1, x2, x3


@triton.jit
def _ld_pre4(P, t64, rm):
    p0 = tl.load(P + t64 * 4, mask=rm, other=0.0).to(tl.float32)
    p1 = tl.load(P + t64 * 4 + 1, mask=rm, other=0.0).to(tl.float32)
    p2 = tl.load(P + t64 * 4 + 2, mask=rm, other=0.0).to(tl.float32)
    p3 = tl.load(P + t64 * 4 + 3, mask=rm, other=0.0).to(tl.float32)
    return p0, p1, p2, p3


@triton.jit
def _collapse_bf16(x0, x1, x2, x3, p0, p1, p2, p3):
    h = p0[:, None] * x0.to(tl.float32) + p1[:, None] * x1.to(tl.float32)
    h += p2[:, None] * x2.to(tl.float32) + p3[:, None] * x3.to(tl.float32)
    return h.to(tl.bfloat16).to(tl.float32)  # the unfused path stores the collapsed hidden in bf16


# ------------------------------------------------------------------------------------------------
# F2': y = rmsnorm(sum_k pre_k x_k) * gamma
# ------------------------------------------------------------------------------------------------
@triton.jit
def _hc_collapse_norm_fwd_kernel(
    X, PRE, GAMMA, Y, RSTD, YQ, YS, M, D, eps, BD: tl.constexpr, RB: tl.constexpr
):
    rows = tl.program_id(0) * RB + tl.arange(0, RB)
    rm = rows < M
    cols = tl.arange(0, BD)
    cm = cols < D
    t64 = rows.to(tl.int64)
    m2 = rm[:, None] & cm[None, :]
    x0, x1, x2, x3 = _ld_rows4(X, t64, cols, m2, D)
    p0, p1, p2, p3 = _ld_pre4(PRE, t64, rm)
    g = tl.load(GAMMA + cols, mask=cm, other=0.0).to(tl.float32)
    h = _collapse_bf16(x0, x1, x2, x3, p0, p1, p2, p3)
    rstd = 1.0 / tl.sqrt(tl.sum(h * h, axis=1) / D + eps)
    y = (h * rstd[:, None] * g[None, :]).to(Y.dtype.element_ty)
    off = t64[:, None] * D + cols[None, :]
    tl.store(Y + off, y, mask=m2)
    tl.store(RSTD + rows, rstd, mask=rm)


# ------------------------------------------------------------------------------------------------
# B2': dh = rmsnorm_bwd(dy) (written as the HC dcoll), dpre partials per BC column tile, dgamma partials.
# Persistent row programs; the next row block's loads are issued before the current one is consumed.
# ------------------------------------------------------------------------------------------------
@triton.jit
def _dpre_store(
    DPREP, xk, dhf, c_t, t64, rm, M, k: tl.constexpr, RB: tl.constexpr, NCT: tl.constexpr, BC: tl.constexpr
):
    pk = tl.sum(tl.reshape(xk.to(tl.float32) * dhf, (RB, NCT, BC)), axis=2)  # [RB, NCT]
    tl.store(DPREP + c_t[None, :].to(tl.int64) * M * 4 + t64[:, None] * 4 + k, pk, mask=rm[:, None])


@triton.jit
def _hc_norm_dpre_bwd_kernel(
    X,
    PRE,
    GAMMA,
    RSTD,
    DY,
    DH,
    DPREP,
    DGP,
    M,
    D,
    BD: tl.constexpr,
    RB: tl.constexpr,
    BC: tl.constexpr,
    NPROG: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BD)
    cm = cols < D
    g = tl.load(GAMMA + cols, mask=cm, other=0.0).to(tl.float32)
    NCT: tl.constexpr = BD // BC
    c_t = tl.arange(0, NCT)
    dg = tl.zeros((RB, BD), dtype=tl.float32)
    STEP: tl.constexpr = NPROG * RB
    rows = pid * RB + tl.arange(0, RB)
    rm = rows < M
    t64 = rows.to(tl.int64)
    m2 = rm[:, None] & cm[None, :]
    a0, a1, a2, a3 = _ld_rows4(X, t64, cols, m2, D)
    p0, p1, p2, p3 = _ld_pre4(PRE, t64, rm)
    dyb = tl.load(DY + t64[:, None] * D + cols[None, :], mask=m2, other=0.0)
    rs = tl.load(RSTD + rows, mask=rm, other=0.0)
    for r0 in range(pid * RB, M, STEP):
        x0, x1, x2, x3 = a0, a1, a2, a3
        q0, q1, q2, q3 = p0, p1, p2, p3
        dy = dyb.to(tl.float32)
        rstd = rs
        tc, rmc, m2c = t64, rm, m2
        # prefetch the next row block
        rows = r0 + STEP + tl.arange(0, RB)
        rm = rows < M
        t64 = rows.to(tl.int64)
        m2 = rm[:, None] & cm[None, :]
        a0, a1, a2, a3 = _ld_rows4(X, t64, cols, m2, D)
        p0, p1, p2, p3 = _ld_pre4(PRE, t64, rm)
        dyb = tl.load(DY + t64[:, None] * D + cols[None, :], mask=m2, other=0.0)
        rs = tl.load(RSTD + rows, mask=rm, other=0.0)

        h = _collapse_bf16(x0, x1, x2, x3, q0, q1, q2, q3)
        xh = h * rstd[:, None]
        dxh = dy * g[None, :]
        mm = tl.sum(dxh * xh, axis=1) / D
        dhb = ((dxh - xh * mm[:, None]) * rstd[:, None]).to(DH.dtype.element_ty)
        tl.store(DH + tc[:, None] * D + cols[None, :], dhb, mask=m2c)
        dhf = dhb.to(tl.float32)
        _dpre_store(DPREP, x0, dhf, c_t, tc, rmc, M, 0, RB, NCT, BC)
        _dpre_store(DPREP, x1, dhf, c_t, tc, rmc, M, 1, RB, NCT, BC)
        _dpre_store(DPREP, x2, dhf, c_t, tc, rmc, M, 2, RB, NCT, BC)
        _dpre_store(DPREP, x3, dhf, c_t, tc, rmc, M, 3, RB, NCT, BC)
        dg += dy * xh
    tl.store(DGP + pid * D + cols, tl.sum(dg, axis=0), mask=cm)


FOLD_CFG = dict(fwd_rb=1, fwd_warps=8, bwd_rb=1, bwd_warps=8, bwd_ppc=2)


def _n_cu(dev) -> int:
    try:
        return torch.cuda.get_device_properties(dev).multi_processor_count
    except Exception:
        return 64


def collapse_norm_fwd(x3, pre, gamma, eps):
    """x3 [M, 4, D] bf16, pre [M, 4] -> y [M, D], rstd [M]."""
    M, K, D = x3.shape
    assert K == 4
    BD = triton.next_power_of_2(D)
    f = FOLD_CFG
    y = torch.empty((M, D), dtype=x3.dtype, device=x3.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x3.device)
    _hc_collapse_norm_fwd_kernel[(triton.cdiv(M, f["fwd_rb"]),)](
        x3,
        pre,
        gamma,
        y,
        rstd,
        rstd,
        rstd,
        M,
        D,
        float(eps),
        BD=BD,
        RB=f["fwd_rb"],
        num_warps=f["fwd_warps"],
    )
    return y, rstd


def norm_dpre_bwd(x3, pre, gamma, rstd, dy, bc, reduce=True):
    """-> dh [M, D] (bf16, = HC dcoll), dprep [D/bc, M, 4] fp32 partials, dgamma (gamma dtype); with
    ``reduce=False`` the [P, D] fp32 dgamma partials instead of dgamma (fuse-hc reduces them in its tail launch).
    """
    from .rmsnorm_g90 import _rms_dg_reduce_kernel

    M, K, D = x3.shape
    BD = triton.next_power_of_2(D)
    assert BD == D and D % bc == 0
    f = FOLD_CFG
    dev = x3.device
    dh = torch.empty((M, D), dtype=x3.dtype, device=dev)
    nc = D // bc
    dprep = torch.empty((nc, M, K), dtype=torch.float32, device=dev)
    rb = f["bwd_rb"]
    nprog = min(triton.cdiv(M, rb), f["bwd_ppc"] * _n_cu(dev))
    dgp = torch.empty((nprog, D), dtype=torch.float32, device=dev)
    _hc_norm_dpre_bwd_kernel[(nprog,)](
        x3,
        pre,
        gamma,
        rstd,
        dy,
        dh,
        dprep,
        dgp,
        M,
        D,
        BD=BD,
        RB=rb,
        BC=bc,
        NPROG=nprog,
        num_warps=f["bwd_warps"],
    )
    if not reduce:
        return dh, dprep, dgp
    dg = torch.empty_like(gamma)
    _rms_dg_reduce_kernel[(triton.cdiv(D, 64),)](dgp, dg, nprog, D, BH=64, BP=64, num_warps=4)
    return dh, dprep, dg


__all__ = ["hc_fold_norm_enabled", "collapse_norm_fwd", "norm_dpre_bwd", "FOLD_CFG"]
