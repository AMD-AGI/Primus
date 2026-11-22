###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""fuse-hc (2026-11-22): fewer launches / HBM passes at the HC boundary between sublayers.

All of these reproduce the bits of the kernels they replace (same arithmetic, same summation order); each one is a
separate switch, default off, and falls back to the old kernels whenever its preconditions do not hold.

PRIMUS_HC_GRAD_DIRECT=1  backward tail in ONE launch (``_hc_tail_kernel``): the dW split-K reduce, the dscale/dbase
    partial sums and the folded-norm dgamma reduce (was ``_hc_reduce`` + ``_rms_dg_reduce`` + 3 ``.to(bf16)`` casts +
    the DDP hook copies into main_grad), written straight into ``param.main_grad`` (beta 0 on the first write of the
    step, else accumulate, exactly what the DDP hook would have done; flagged in forward as the other fused writers).

PRIMUS_HC_XRMS=1  expand(L) also produces HC(L+1)'s rms-linear split-K partials (``_hc_expand_rms_kernel``): the
    next HCPreFn skips ``_hc_wt_split`` + ``_hc_fwd_wmma3`` (one fewer 134 MB read of x per boundary). Needs bf16
    fn.weight (then the wmma3 "mid"/"lo" parts are exact zeros) and the wmma3 chunking (CHUNK | D, BK = 128).

PRIMUS_HC_PRENORM=1  ``_hc_pre_fwd`` (finalize + tail + Sinkhorn) folded into the folded-norm collapse kernel
    (``_hc_pre_collapse_norm_kernel``): one row per program, the x row loads are issued before the row math.

PRIMUS_HC_DPRE_BWD=1  ``_hc_pre_bwd`` (dpre sum + Sinkhorn/tail VJP + g/coef + dscale/dbase partials) folded into the
    folded-norm backward (``_hc_norm_dpre_prebwd_kernel``); the [D/256, M, 4] dpre partials are never written.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .hc_norm_fold import _collapse_bf16, _ld_pre4, _ld_rows4, _pow2_floor


def _flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def grad_direct_enabled() -> bool:
    return _flag("PRIMUS_HC_GRAD_DIRECT")


def xrms_enabled() -> bool:
    return _flag("PRIMUS_HC_XRMS")


def prenorm_enabled() -> bool:
    return _flag("PRIMUS_HC_PRENORM")


def dpre_bwd_enabled() -> bool:
    return _flag("PRIMUS_HC_DPRE_BWD")


# ------------------------------------------------------------------------------------------------
# main_grad direct write (same eligibility as rmsnorm_g90's PRIMUS_GLUE_RMSNORM path, any rank)
# ------------------------------------------------------------------------------------------------
def direct_main_grad(p):
    """``p.main_grad`` when the gradient of parameter ``p`` can be written into it directly, else None."""
    mg = getattr(p, "main_grad", None)
    if (
        isinstance(p, torch.nn.Parameter)
        and hasattr(p, "grad_added_to_main_grad")
        and isinstance(mg, torch.Tensor)
        and mg.is_cuda
        and mg.shape == p.shape
        and mg.is_contiguous()
        and mg.dtype in (torch.bfloat16, torch.float32)
        and p.dtype in (torch.bfloat16, torch.float32)
        and not getattr(p, "shared", False)
    ):
        return mg
    return None


def claim_main_grad(p) -> bool:
    """Forward-time claim (as rmsnorm_g90 / Turbo fused wgrad): the backward writes p.main_grad itself."""
    if p is None or direct_main_grad(p) is None:
        return False
    p.grad_added_to_main_grad = True
    p._primus_fwd_claim = True
    return True


# ------------------------------------------------------------------------------------------------
# PRIMUS_HC_GRAD_DIRECT: one tail launch
# ------------------------------------------------------------------------------------------------
@triton.jit
def _mg_write(OUT, offs, v, m, PDT: tl.constexpr, ACC: tl.constexpr):
    """Store fp32 ``v`` as the parameter-dtype gradient (PDT) into OUT, overwriting (beta 0) or accumulating like
    ``main_grad.add_(grad)`` (fp32 add of the param-dtype grad, rounded to OUT's dtype)."""
    gp = v.to(PDT).to(tl.float32)
    if ACC:
        gp = tl.load(OUT + offs, mask=m, other=0.0).to(tl.float32) + gp
    tl.store(OUT + offs, gp.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _hc_tail_kernel(DWP, PART, DGP, DW_OUT, S_OUT, B_OUT, DG_OUT,
                    NS, NTOT, P, N_DW, P2, D,
                    NP: tl.constexpr, BLOCK: tl.constexpr, BH: tl.constexpr, BP: tl.constexpr,
                    W_PDT: tl.constexpr, S_PDT: tl.constexpr, B_PDT: tl.constexpr, G_PDT: tl.constexpr,
                    W_ACC: tl.constexpr, S_ACC: tl.constexpr, B_ACC: tl.constexpr, G_ACC: tl.constexpr,
                    HAS_DG: tl.constexpr):
    """Programs [0, n_dw): dW[i] = sum_s DWP[s, i] (as _hc_reduce_kernel);
    [n_dw, n_dw + NP): PART row j summed (as _hc_reduce_kernel), j < 3 -> dscale[j], else dbase[j - 3];
    [n_dw + NP, ...): dgamma column block (as rmsnorm_g90._rms_dg_reduce_kernel, BH x BP tile)."""
    pid = tl.program_id(0)
    n_dw = tl.cdiv(N_DW, BLOCK)
    if pid < n_dw:
        o = pid * BLOCK + tl.arange(0, BLOCK)
        m = o < N_DW
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for s_ in range(0, NS):
            acc += tl.load(DWP + s_ * NTOT + o, mask=m, other=0.0)
        _mg_write(DW_OUT, o, acc, m, W_PDT, W_ACC)
    elif pid < n_dw + NP:
        j = pid - n_dw
        acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
        for p0 in range(0, P, BLOCK):
            r = p0 + tl.arange(0, BLOCK)
            acc2 += tl.load(PART + j * P + r, mask=r < P, other=0.0)
        v = tl.sum(acc2, axis=0)
        if j < 3:
            _mg_write(S_OUT, j, v, True, S_PDT, S_ACC)
        else:
            _mg_write(B_OUT, j - 3, v, True, B_PDT, B_ACC)
    elif HAS_DG:
        cols = (pid - n_dw - NP) * BH + tl.arange(0, BH)
        cm = cols < D
        acc3 = tl.zeros((BP, BH), dtype=tl.float32)
        for p0 in range(0, P2, BP):
            pr = p0 + tl.arange(0, BP)
            acc3 += tl.load(DGP + pr[:, None] * D + cols[None, :], mask=(pr[:, None] < P2) & cm[None, :], other=0.0)
        _mg_write(DG_OUT, cols, tl.sum(acc3, axis=0), cm, G_PDT, G_ACC)


_TL_DT = {torch.bfloat16: tl.bfloat16, torch.float32: tl.float32, torch.float16: tl.float16}


class _GradSlot:
    """Where one parameter's gradient goes: its main_grad (direct) or a fresh tensor returned to autograd."""

    __slots__ = ("param", "out", "acc", "direct")

    def __init__(self, param, like_dtype, shape, device, claimed):
        self.param = param
        mg = direct_main_grad(param) if (claimed and param is not None) else None
        if mg is not None:
            from primus_turbo.pytorch.ops.utils import main_grad_first_write

            self.out = mg
            self.acc = not main_grad_first_write(param)
            self.direct = True
        else:
            self.out = torch.empty(shape, dtype=like_dtype, device=device)
            self.acc = False
            self.direct = False

    def grad(self, pdtype):
        if self.direct:
            from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad

            return _get_dummy_wgrad(list(self.param.shape), self.param.dtype)
        return self.out


def hc_tail(dw_p, n, part, dgp, D, params, dtypes, claimed):
    """One launch for the HC backward tail.

    dw_p [S, BN, KD] fp32 split-K dW partials (first n rows of each are real), part [NP, P] fp32 (NP = 3 + n),
    dgp [P2, D] fp32 dgamma partials or None. params = (weight, scale, base, gamma) parameters (gamma may be None),
    dtypes = their gradient dtypes. Returns the 4 gradients to hand to autograd (dummies for main_grad writes).
    """
    S_, BN_, KD_ = dw_p.shape
    dev = dw_p.device
    NP_, P_ = part.shape
    assert NP_ == 3 + n
    w, s, b, g = params
    slots = [
        _GradSlot(w, dtypes[0], (n, KD_), dev, claimed[0]),
        _GradSlot(s, dtypes[1], (3,), dev, claimed[1]),
        _GradSlot(b, dtypes[2], (n,), dev, claimed[2]),
    ]
    has_dg = dgp is not None
    slots.append(_GradSlot(g, dtypes[3], (D,), dev, claimed[3]) if has_dg else None)
    BLOCK, BH, BP = 2048, 64, 64
    n_dw = n * KD_
    grid = triton.cdiv(n_dw, BLOCK) + NP_ + (triton.cdiv(D, BH) if has_dg else 0)
    dg_out = slots[3].out if has_dg else slots[0].out
    _hc_tail_kernel[(grid,)](
        dw_p, part, dgp if has_dg else part, slots[0].out, slots[1].out, slots[2].out, dg_out,
        S_, BN_ * KD_, P_, n_dw, dgp.shape[0] if has_dg else 0, D,
        NP=NP_, BLOCK=BLOCK, BH=BH, BP=BP,
        W_PDT=_TL_DT[dtypes[0]], S_PDT=_TL_DT[dtypes[1]], B_PDT=_TL_DT[dtypes[2]],
        G_PDT=_TL_DT[dtypes[3]] if has_dg else tl.float32,
        W_ACC=slots[0].acc, S_ACC=slots[1].acc, B_ACC=slots[2].acc, G_ACC=slots[3].acc if has_dg else False,
        HAS_DG=has_dg, num_warps=4,
    )
    return tuple(sl.grad(dt) if sl is not None else None for sl, dt in zip(slots, dtypes))


# ------------------------------------------------------------------------------------------------
# PRIMUS_HC_XRMS: expand(L) + rms-linear partials of HC(L+1)
# ------------------------------------------------------------------------------------------------
@triton.jit
def _new_h(o, x0, x1, x2, x3, c0, c1, c2, c3, p, ORDER: tl.constexpr):
    """new_h = post_h * out + sum_k comb^T[h, k] x_k in the order of _hc_expand_fwd_kernel's in-thread reduce."""
    if ORDER == 0:
        m = c0[:, None] * x0 + c1[:, None] * x1
        m = m + c2[:, None] * x2
        m = m + c3[:, None] * x3
    else:
        m = (c0[:, None] * x0 + c1[:, None] * x1) + (c2[:, None] * x2 + c3[:, None] * x3)
    return p[:, None] * o + m


@triton.jit
def _xrms_one(NEW, W, off, rm2, ncol, nm, wcol, x0, x1, x2, x3, o, cb, r64, rm, POST, h: tl.constexpr,
              KD, ORDER: tl.constexpr, acc, ss):
    c0 = tl.load(cb + h * 4 + 0, mask=rm, other=0.0).to(tl.float32)
    c1 = tl.load(cb + h * 4 + 1, mask=rm, other=0.0).to(tl.float32)
    c2 = tl.load(cb + h * 4 + 2, mask=rm, other=0.0).to(tl.float32)
    c3 = tl.load(cb + h * 4 + 3, mask=rm, other=0.0).to(tl.float32)
    ph = tl.load(POST + r64 * 4 + h, mask=rm, other=0.0).to(tl.float32)
    nb = _new_h(o, x0, x1, x2, x3, c0, c1, c2, c3, ph, ORDER).to(NEW.dtype.element_ty)
    tl.store(NEW + off, nb, mask=rm2)
    w = tl.load(W + ncol[None, :].to(tl.int64) * KD + wcol[:, None], mask=nm[None, :], other=0.0)  # [BK, BN]
    acc += tl.dot(nb, w)
    nf = nb.to(tl.float32)
    ss += tl.sum(nf * nf, axis=1)
    return acc, ss


@triton.jit
def _hc_expand_rms_kernel(X, OUT, POST, COMBT, NEW, W, P_RAW, P_SS, M, D, N,
                          BT: tl.constexpr, BK: tl.constexpr, CHUNK: tl.constexpr, BN: tl.constexpr,
                          ORDER: tl.constexpr):
    """One program = BT rows x CHUNK columns x all 4 output streams. new_h is written (bf16) and, as the next HC's
    _hc_fwd_wmma3_kernel would, P_RAW[s] = new_h[:, chunk] @ W^T[chunk] and P_SS[s] = sum(new_h[:, chunk]^2) with
    s = h * (D / CHUNK) + pid_c (the wmma3 chunk index of KD columns h * D + pid_c * CHUNK ...)."""
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_t * BT + tl.arange(0, BT)
    rm = rows < M
    r64 = rows.to(tl.int64)
    rm2 = rm[:, None]
    KD = 4 * D
    ncol = tl.arange(0, BN)
    nm = ncol < N
    cb = COMBT + r64 * 16
    a0 = tl.zeros((BT, BN), dtype=tl.float32)
    a1 = tl.zeros((BT, BN), dtype=tl.float32)
    a2 = tl.zeros((BT, BN), dtype=tl.float32)
    a3 = tl.zeros((BT, BN), dtype=tl.float32)
    s0 = tl.zeros((BT,), dtype=tl.float32)
    s1 = tl.zeros((BT,), dtype=tl.float32)
    s2 = tl.zeros((BT,), dtype=tl.float32)
    s3 = tl.zeros((BT,), dtype=tl.float32)
    for kk in range(0, CHUNK, BK):
        cols = pid_c * CHUNK + kk + tl.arange(0, BK)
        off = r64[:, None] * KD + cols[None, :]
        x0 = tl.load(X + off, mask=rm2, other=0.0).to(tl.float32)
        x1 = tl.load(X + off + D, mask=rm2, other=0.0).to(tl.float32)
        x2 = tl.load(X + off + 2 * D, mask=rm2, other=0.0).to(tl.float32)
        x3 = tl.load(X + off + 3 * D, mask=rm2, other=0.0).to(tl.float32)
        o = tl.load(OUT + r64[:, None] * D + cols[None, :], mask=rm2, other=0.0).to(tl.float32)
        a0, s0 = _xrms_one(NEW, W, off, rm2, ncol, nm, cols, x0, x1, x2, x3, o, cb, r64, rm, POST, 0, KD, ORDER, a0, s0)
        a1, s1 = _xrms_one(NEW, W, off + D, rm2, ncol, nm, D + cols, x0, x1, x2, x3, o, cb, r64, rm, POST, 1, KD, ORDER,
                           a1, s1)
        a2, s2 = _xrms_one(NEW, W, off + 2 * D, rm2, ncol, nm, 2 * D + cols, x0, x1, x2, x3, o, cb, r64, rm, POST, 2,
                           KD, ORDER, a2, s2)
        a3, s3 = _xrms_one(NEW, W, off + 3 * D, rm2, ncol, nm, 3 * D + cols, x0, x1, x2, x3, o, cb, r64, rm, POST, 3,
                           KD, ORDER, a3, s3)
    NCH = D // CHUNK
    for h in tl.static_range(4):
        if h == 0:
            a, s = a0, s0
        elif h == 1:
            a, s = a1, s1
        elif h == 2:
            a, s = a2, s2
        else:
            a, s = a3, s3
        orow = (h * NCH + pid_c) * M + r64
        tl.store(P_RAW + orow[:, None] * BN + ncol[None, :], a, mask=rm2)
        tl.store(P_SS + orow, s, mask=rm)


XRMS_CFG = dict(bt=32, bk=128, chunk=1024, warps=8, order=1)  # order 1 = (c0x0 + c1x1) + (c2x2 + c3x3): the production expand tree (calib_expand.py)


def _xrms_cfg():
    v = os.environ.get("PRIMUS_HC_XRMS_CFG", "").strip()  # bt,warps
    c = dict(XRMS_CFG)
    if v:
        bt, nw = (int(t) for t in v.split(","))
        c.update(bt=bt, warps=nw)
    return c


def xrms_supported(x3, next_w) -> bool:
    M, K, D = x3.shape
    c = XRMS_CFG
    return (next_w is not None and next_w.dtype == torch.bfloat16 and next_w.is_contiguous() and K == 4
            and next_w.dim() == 2 and next_w.shape[1] == K * D and next_w.shape[0] <= 32 and D % c["chunk"] == 0
            and next_w.is_cuda and next_w.data_ptr() != 0)


def expand_rms(x3, out_c, post_c, combt_c, next_w):
    """-> new [M, 4, D], p_raw [KD/CHUNK, M, BN] fp32, p_ss [KD/CHUNK, M] fp32 (the wmma3 partials of next_w)."""
    M, K, D = x3.shape
    c = _xrms_cfg()
    N = next_w.shape[0]
    BN = max(16, triton.next_power_of_2(N))
    n_split = K * D // c["chunk"]
    new = torch.empty_like(x3)
    p_raw = torch.empty((n_split, M, BN), dtype=torch.float32, device=x3.device)
    p_ss = torch.empty((n_split, M), dtype=torch.float32, device=x3.device)
    _hc_expand_rms_kernel[(triton.cdiv(M, c["bt"]), D // c["chunk"])](
        x3, out_c, post_c, combt_c, new, next_w, p_raw, p_ss, M, D, N,
        BT=c["bt"], BK=c["bk"], CHUNK=c["chunk"], BN=BN, ORDER=c["order"], num_warps=c["warps"],
    )
    return new, p_raw, p_ss, n_split


# ------------------------------------------------------------------------------------------------
# PRIMUS_HC_PRENORM: _hc_pre_fwd row math + folded-norm collapse, one row per program
# ------------------------------------------------------------------------------------------------
@triton.jit
def _pick(v, kk, k: tl.constexpr):
    """Element k of a [1, K] row vector as a [1] tensor (exact: one value plus zeros)."""
    return tl.sum(tl.where(kk[None, :] == k, v, 0.0), axis=1)


@triton.jit
def _hc_pre_collapse_norm_kernel(X, P_RAW, P_SS, SCALE, BASE, RAW, RSTD, STATES, PRE, POST, COMBT,
                                 GAMMA, Y, RSTD_N, M, D, n_split, eps_rms, norm_eps,
                                 EPS: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, N_ITERS: tl.constexpr,
                                 BD: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid + tl.arange(0, 1)
    rmask = rows < M
    rows64 = rows.to(tl.int64)
    # x row loads first: they do not depend on the row math
    cols = tl.arange(0, BD)
    cm = cols < D
    m2c = rmask[:, None] & cm[None, :]
    x0, x1, x2, x3 = _ld_rows4(X, rows64, cols, m2c, D)
    gam = tl.load(GAMMA + cols, mask=cm, other=0.0).to(tl.float32)

    # ---- _hc_pre_fwd_kernel body, BLOCK_M = 1 ----
    kk = tl.arange(0, K)
    KK: tl.constexpr = K * K
    r2 = rows64[:, None]
    m2 = rmask[:, None]
    r3 = rows64[:, None, None]
    m3 = rmask[:, None, None]
    i3 = kk[None, :, None]
    j3 = kk[None, None, :]
    raw_pre = tl.zeros((1, K), dtype=tl.float32)
    raw_post = tl.zeros((1, K), dtype=tl.float32)
    raw_comb = tl.zeros((1, K, K), dtype=tl.float32)
    ss = tl.zeros((1,), dtype=tl.float32)
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
    tl.store(COMBT + r3 * KK + j3 * K + i3, m.to(tl.bfloat16), mask=m3)

    # ---- _hc_collapse_norm_fwd_kernel body (RB = 1), pre from registers (bf16-rounded, as stored) ----
    pf = pre_b.to(tl.float32)
    p0 = _pick(pf, kk, 0)
    p1 = _pick(pf, kk, 1)
    p2 = _pick(pf, kk, 2)
    p3 = _pick(pf, kk, 3)
    h = _collapse_bf16(x0, x1, x2, x3, p0, p1, p2, p3)
    rstd_n = 1.0 / tl.sqrt(tl.sum(h * h, axis=1) / D + norm_eps)
    y = (h * rstd_n[:, None] * gam[None, :]).to(Y.dtype.element_ty)
    tl.store(Y + rows64[:, None] * D + cols[None, :], y, mask=m2c)
    tl.store(RSTD_N + rows, rstd_n, mask=rmask)


PRENORM_CFG = dict(warps=8)


def pre_collapse_norm(x3, p_raw, p_ss, n_split, scale_c, base_c, eps, n_iters, gamma, norm_eps, BN):
    """-> raw, rstd, states, pre, post, combt, y [M, D], rstd_n (the outputs of _hc_pre_fwd + collapse_norm_fwd)."""
    M, K, D = x3.shape
    dev = x3.device
    raw = torch.empty((M, BN), dtype=torch.float32, device=dev)
    rstd = torch.empty((M,), dtype=torch.float32, device=dev)
    states = torch.empty((M, 2 * n_iters, K, K), dtype=torch.float32, device=dev)
    pre = torch.empty((M, K), dtype=x3.dtype, device=dev)
    post = torch.empty((M, K), dtype=x3.dtype, device=dev)
    combt = torch.empty((M, K, K), dtype=x3.dtype, device=dev)
    y = torch.empty((M, D), dtype=x3.dtype, device=dev)
    rstd_n = torch.empty((M,), dtype=torch.float32, device=dev)
    _hc_pre_collapse_norm_kernel[(M,)](
        x3, p_raw, p_ss, scale_c, base_c, raw, rstd, states, pre, post, combt, gamma, y, rstd_n,
        M, D, n_split, float(eps), float(norm_eps),
        EPS=float(eps), K=K, BN=BN, N_ITERS=int(n_iters), BD=triton.next_power_of_2(D), num_warps=PRENORM_CFG["warps"],
    )
    return raw, rstd, states, pre, post, combt, y, rstd_n


# ------------------------------------------------------------------------------------------------
# PRIMUS_HC_DPRE_BWD: folded-norm backward + _hc_pre_bwd row math, persistent row programs
# ------------------------------------------------------------------------------------------------
@triton.jit
def _st_g(GH, GL, GTH, GTL, off_rm, off_t, v, m):
    hi = v.to(tl.bfloat16)
    lo = (v - hi.to(tl.float32)).to(tl.bfloat16)
    tl.store(GH + off_rm, hi, mask=m)
    tl.store(GL + off_rm, lo, mask=m)
    tl.store(GTH + off_t, hi, mask=m)
    tl.store(GTL + off_t, lo, mask=m)


@triton.jit
def _pick_c(pk, c_t, c: tl.constexpr):
    return tl.sum(tl.where(c_t[None, :] == c, pk, 0.0), axis=1)  # [RB]


@triton.jit
def _row_prebwd(rows64, rmask, dpre, RAW, RSTD, STATES, SCALE, BASE, DPOST, DCOMBT, G, COEF, PART, GH, GL, GTH, GTL,
                NC, M, D, HAS_POST: tl.constexpr, SPLIT_G: tl.constexpr, EPS: tl.constexpr, K: tl.constexpr,
                BN: tl.constexpr, N_ITERS: tl.constexpr):
    """_hc_pre_bwd_kernel (PARTS=True, BLOCK_M=1) after the dpre sum, for one row; PART column = the row."""
    kk = tl.arange(0, K)
    KK: tl.constexpr = K * K
    KD = K * D
    r2 = rows64[:, None]
    m2 = rmask[:, None]
    r3 = rows64[:, None, None]
    m3 = rmask[:, None, None]
    i3 = kk[None, :, None]
    j3 = kk[None, None, :]
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
    if HAS_POST:
        dpost = tl.zeros((1, K), dtype=tl.float32)
        dm_ = tl.zeros((1, K, K), dtype=tl.float32)
        for c in range(0, NC):
            dpost += tl.load(DPOST + c * M * K + r2 * K + kk[None, :], mask=m2, other=0.0)
            dm_ += tl.load(DCOMBT + c * M * KK + r3 * KK + j3 * K + i3, mask=m3, other=0.0)
    else:
        dpost = tl.zeros((1, K), dtype=tl.float32)
        dm_ = tl.zeros((1, K, K), dtype=tl.float32)
    SROW: tl.constexpr = 2 * N_ITERS * KK
    sb = STATES + r3 * SROW + i3 * K + j3
    for i in tl.static_range(N_ITERS - 1):
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
    d_pre_l = dpre * pre_sig * (1.0 - pre_sig)
    d_post_l = dpost * 2.0 * post_sig * (1.0 - post_sig)
    d_comb_l = comb_sm * (dm_ - tl.sum(dm_ * comb_sm, axis=2, keep_dims=True))
    dl_pre = d_pre_l * s0
    dl_post = d_post_l * s1
    dl_comb = d_comb_l * s2
    tl.store(G + r2 * BN + kk[None, :], dl_pre * rstd[:, None], mask=m2)
    tl.store(G + r2 * BN + K + kk[None, :], dl_post * rstd[:, None], mask=m2)
    tl.store(G + r3 * BN + 2 * K + i3 * K + j3, dl_comb * rstd[:, None, None], mask=m3)
    NPAD: tl.constexpr = BN - (2 + K) * K
    if NPAD > 0:
        pc = tl.arange(0, NPAD)
        tl.store(G + r2 * BN + (2 + K) * K + pc[None, :], tl.zeros((1, NPAD), dtype=tl.float32), mask=m2)
    if SPLIT_G:
        _st_g(GH, GL, GTH, GTL, r2 * BN + kk[None, :], kk[None, :] * M + r2, dl_pre * rstd[:, None], m2)
        _st_g(GH, GL, GTH, GTL, r2 * BN + K + kk[None, :], (K + kk[None, :]) * M + r2, dl_post * rstd[:, None], m2)
        _st_g(GH, GL, GTH, GTL, r3 * BN + 2 * K + i3 * K + j3, (2 * K + i3 * K + j3) * M + r3,
              dl_comb * rstd[:, None, None], m3)
        if NPAD > 0:
            pc2 = tl.arange(0, NPAD)
            _st_g(GH, GL, GTH, GTL, r2 * BN + (2 + K) * K + pc2[None, :], ((2 + K) * K + pc2[None, :]) * M + r2,
                  tl.zeros((1, NPAD), dtype=tl.float32), m2)
    c = tl.sum(dl_pre * raw_pre, axis=1) + tl.sum(dl_post * raw_post, axis=1) + tl.sum(tl.sum(dl_comb * raw_comb, axis=2), axis=1)
    tl.store(COEF + rows64, rstd * rstd * rstd * c / KD, mask=rmask)
    pp = PART + rows64  # PART[j, row], NPR = M
    tl.store(pp + 0 * M, tl.sum(tl.sum(l_pre * d_pre_l, axis=1), axis=0), mask=rmask)
    tl.store(pp + 1 * M, tl.sum(tl.sum(l_post * d_post_l, axis=1), axis=0), mask=rmask)
    tl.store(pp + 2 * M, tl.sum(tl.sum(tl.sum(l_comb * d_comb_l, axis=2), axis=1), axis=0), mask=rmask)
    tl.store(pp + (3 + kk[None, :]) * M, d_pre_l, mask=m2)
    tl.store(pp + (3 + K + kk[None, :]) * M, d_post_l, mask=m2)
    tl.store(pp + (3 + 2 * K + i3 * K + j3) * M, d_comb_l, mask=m3)


@triton.jit
def _hc_norm_dpre_prebwd_kernel(X, PRE, GAMMA, RSTD_N, DY, DH, DGP,
                                RAW, RSTD, STATES, SCALE, BASE, DPOST, DCOMBT, G, COEF, PART, GH, GL, GTH, GTL,
                                NC, M, D,
                                BD: tl.constexpr, BC: tl.constexpr, NPROG: tl.constexpr,
                                HAS_POST: tl.constexpr, SPLIT_G: tl.constexpr, EPS: tl.constexpr, K: tl.constexpr,
                                BN: tl.constexpr, N_ITERS: tl.constexpr):
    """hc_norm_fold._hc_norm_dpre_bwd_kernel (RB = 1) with the dpre partials summed in registers (column-tile order,
    as _hc_pre_bwd_kernel sums the stored partials) and _hc_pre_bwd_kernel's row math run per row."""
    RB: tl.constexpr = 1
    pid = tl.program_id(0)
    cols = tl.arange(0, BD)
    cm = cols < D
    g = tl.load(GAMMA + cols, mask=cm, other=0.0).to(tl.float32)
    NCT: tl.constexpr = BD // BC
    c_t = tl.arange(0, NCT)
    kk = tl.arange(0, K)
    dg = tl.zeros((RB, BD), dtype=tl.float32)
    STEP: tl.constexpr = NPROG * RB
    rows = pid * RB + tl.arange(0, RB)
    rm = rows < M
    t64 = rows.to(tl.int64)
    m2 = rm[:, None] & cm[None, :]
    a0, a1, a2, a3 = _ld_rows4(X, t64, cols, m2, D)
    p0, p1, p2, p3 = _ld_pre4(PRE, t64, rm)
    dyb = tl.load(DY + t64[:, None] * D + cols[None, :], mask=m2, other=0.0)
    rs = tl.load(RSTD_N + rows, mask=rm, other=0.0)
    for r0 in range(pid * RB, M, STEP):
        x0, x1, x2, x3 = a0, a1, a2, a3
        q0, q1, q2, q3 = p0, p1, p2, p3
        dy = dyb.to(tl.float32)
        rstd = rs
        tc, rmc, m2c = t64, rm, m2
        rows = r0 + STEP + tl.arange(0, RB)
        rm = rows < M
        t64 = rows.to(tl.int64)
        m2 = rm[:, None] & cm[None, :]
        a0, a1, a2, a3 = _ld_rows4(X, t64, cols, m2, D)
        p0, p1, p2, p3 = _ld_pre4(PRE, t64, rm)
        dyb = tl.load(DY + t64[:, None] * D + cols[None, :], mask=m2, other=0.0)
        rs = tl.load(RSTD_N + rows, mask=rm, other=0.0)

        h = _collapse_bf16(x0, x1, x2, x3, q0, q1, q2, q3)
        xh = h * rstd[:, None]
        dxh = dy * g[None, :]
        mm = tl.sum(dxh * xh, axis=1) / D
        dhb = ((dxh - xh * mm[:, None]) * rstd[:, None]).to(DH.dtype.element_ty)
        tl.store(DH + tc[:, None] * D + cols[None, :], dhb, mask=m2c)
        dhf = dhb.to(tl.float32)
        pk0 = tl.sum(tl.reshape(x0.to(tl.float32) * dhf, (RB, NCT, BC)), axis=2)  # [RB, NCT]
        pk1 = tl.sum(tl.reshape(x1.to(tl.float32) * dhf, (RB, NCT, BC)), axis=2)
        pk2 = tl.sum(tl.reshape(x2.to(tl.float32) * dhf, (RB, NCT, BC)), axis=2)
        pk3 = tl.sum(tl.reshape(x3.to(tl.float32) * dhf, (RB, NCT, BC)), axis=2)
        dg += dy * xh
        # dpre_k = sum_c partial[c, k], c ascending from 0.0 (as _hc_pre_bwd_kernel's PARTS loop)
        dpre = tl.zeros((1, K), dtype=tl.float32)
        for c in tl.static_range(NCT):
            e0 = _pick_c(pk0, c_t, c)
            e1 = _pick_c(pk1, c_t, c)
            e2 = _pick_c(pk2, c_t, c)
            e3 = _pick_c(pk3, c_t, c)
            ev = tl.where(kk[None, :] == 0, e0[:, None],
                          tl.where(kk[None, :] == 1, e1[:, None], tl.where(kk[None, :] == 2, e2[:, None], e3[:, None])))
            dpre += ev
        _row_prebwd(tc, rmc, dpre, RAW, RSTD, STATES, SCALE, BASE, DPOST, DCOMBT, G, COEF, PART, GH, GL, GTH, GTL,
                    NC, M, D, HAS_POST, SPLIT_G, EPS, K, BN, N_ITERS)
    tl.store(DGP + pid * D + cols, tl.sum(dg, axis=0), mask=cm)


def _n_cu(dev) -> int:
    try:
        return torch.cuda.get_device_properties(dev).multi_processor_count
    except Exception:
        return 64


DPRE_CFG = dict(warps=8, ppc=4)


def norm_dpre_prebwd(x3, pre, gamma, rstd_n, dy, raw, rstd, states, scale_c, base_c, dpost, dcombt, nc, eps, n_iters,
                     BN, N, split_g):
    """-> dh (= HC dcoll), dgp [P, D] dgamma partials, g, coef, part [3 + N, M], gh, glo, gth, gtl."""
    M, K, D = x3.shape
    BD = triton.next_power_of_2(D)
    assert BD == D and D % 256 == 0
    dev = x3.device
    dh = torch.empty((M, D), dtype=x3.dtype, device=dev)
    nprog = min(M, DPRE_CFG["ppc"] * _n_cu(dev))
    dgp = torch.empty((nprog, D), dtype=torch.float32, device=dev)
    g = torch.empty((M, BN), dtype=torch.float32, device=dev)
    coef = torch.empty((M,), dtype=torch.float32, device=dev)
    part = torch.empty((3 + N, M), dtype=torch.float32, device=dev)
    if split_g:
        gh = torch.empty((M, BN), dtype=torch.bfloat16, device=dev)
        glo = torch.empty((M, BN), dtype=torch.bfloat16, device=dev)
        gth = torch.empty((BN, M), dtype=torch.bfloat16, device=dev)
        gtl = torch.empty((BN, M), dtype=torch.bfloat16, device=dev)
    else:
        gh = glo = gth = gtl = g
    has_post = dpost is not None
    _hc_norm_dpre_prebwd_kernel[(nprog,)](
        x3, pre, gamma, rstd_n, dy, dh, dgp,
        raw, rstd, states, scale_c, base_c, dpost if has_post else g, dcombt if has_post else g,
        g, coef, part, gh, glo, gth, gtl, nc, M, D,
        BD=BD, BC=256, NPROG=nprog, HAS_POST=has_post, SPLIT_G=split_g, EPS=float(eps), K=K, BN=BN,
        N_ITERS=int(n_iters), num_warps=DPRE_CFG["warps"],
    )
    return dh, dgp, g, coef, part, gh, glo, gth, gtl
