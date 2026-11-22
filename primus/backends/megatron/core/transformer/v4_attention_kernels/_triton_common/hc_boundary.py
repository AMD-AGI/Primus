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

Tried and dropped (byte-equal but slower; git 19f60d8c7, ledger #3/#4/#7): folding _hc_pre_fwd's row math into the
folded-norm collapse kernel, and _hc_pre_bwd's into the folded-norm backward.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl



def _flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def grad_direct_enabled() -> bool:
    return _flag("PRIMUS_HC_GRAD_DIRECT")


def xrms_enabled() -> bool:
    return _flag("PRIMUS_HC_XRMS")


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
                    HAS_DG: tl.constexpr, UNROLL4: tl.constexpr = False, BLOCK_W: tl.constexpr = 0):
    """Programs [0, n_dw): dW[i] = sum_s DWP[s, i] (as _hc_reduce_kernel);
    [n_dw, n_dw + NP): PART row j summed (as _hc_reduce_kernel), j < 3 -> dscale[j], else dbase[j - 3];
    [n_dw + NP, ...): dgamma column block (as rmsnorm_g90._rms_dg_reduce_kernel, BH x BP tile)."""
    pid = tl.program_id(0)
    BW: tl.constexpr = BLOCK_W if BLOCK_W > 0 else BLOCK  # splitk: dW section block (same per-element order)
    n_dw = tl.cdiv(N_DW, BW)
    if pid < n_dw:
        o = pid * BW + tl.arange(0, BW)
        m = o < N_DW
        acc = tl.zeros((BW,), dtype=tl.float32)
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
        if UNROLL4:  # P2 % (4 BP) == 0: four tiles in flight per step, added in the same order
            for p0 in range(0, P2, 4 * BP):
                pr = p0 + tl.arange(0, BP)
                ptr = DGP + pr[:, None] * D + cols[None, :]
                m3 = cm[None, :] & (pr[:, None] >= 0)
                t0 = tl.load(ptr, mask=m3, other=0.0)
                t1 = tl.load(ptr + BP * D, mask=m3, other=0.0)
                t2 = tl.load(ptr + 2 * BP * D, mask=m3, other=0.0)
                t3 = tl.load(ptr + 3 * BP * D, mask=m3, other=0.0)
                acc3 += t0
                acc3 += t1
                acc3 += t2
                acc3 += t3
        else:
            for p0 in range(0, P2, BP):
                pr = p0 + tl.arange(0, BP)
                acc3 += tl.load(DGP + pr[:, None] * D + cols[None, :], mask=(pr[:, None] < P2) & cm[None, :],
                                other=0.0)
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
    # splitk PRIMUS_SPLITK_HC_TAILBW=<block> (default 0 = BLOCK): more, smaller dW-reduce programs; each dW element is
    # still the same sequential sum over the S partials (bit-identical for a given S)
    bw = int(os.environ.get("PRIMUS_SPLITK_HC_TAILBW", "0") or "0")
    tkw = dict(BLOCK_W=bw) if bw else {}
    grid = triton.cdiv(n_dw, bw or BLOCK) + NP_ + (triton.cdiv(D, BH) if has_dg else 0)
    dg_out = slots[3].out if has_dg else slots[0].out
    _hc_tail_kernel[(grid,)](
        dw_p, part, dgp if has_dg else part, slots[0].out, slots[1].out, slots[2].out, dg_out,
        S_, BN_ * KD_, P_, n_dw, dgp.shape[0] if has_dg else 0, D,
        NP=NP_, BLOCK=BLOCK, BH=BH, BP=BP,
        W_PDT=_TL_DT[dtypes[0]], S_PDT=_TL_DT[dtypes[1]], B_PDT=_TL_DT[dtypes[2]],
        G_PDT=_TL_DT[dtypes[3]] if has_dg else tl.float32,
        W_ACC=slots[0].acc, S_ACC=slots[1].acc, B_ACC=slots[2].acc, G_ACC=slots[3].acc if has_dg else False,
        HAS_DG=has_dg, UNROLL4=has_dg and dgp.shape[0] % (4 * BP) == 0, num_warps=4, **tkw,
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
                          ORDER: tl.constexpr, PF: tl.constexpr = False):
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
    if PF:
        c_n = pid_c * CHUNK + tl.arange(0, BK)
        off_n = r64[:, None] * KD + c_n[None, :]
        y0 = tl.load(X + off_n, mask=rm2, other=0.0)
        y1 = tl.load(X + off_n + D, mask=rm2, other=0.0)
        y2 = tl.load(X + off_n + 2 * D, mask=rm2, other=0.0)
        y3 = tl.load(X + off_n + 3 * D, mask=rm2, other=0.0)
        yo = tl.load(OUT + r64[:, None] * D + c_n[None, :], mask=rm2, other=0.0)
    for kk in range(0, CHUNK, BK):
        cols = pid_c * CHUNK + kk + tl.arange(0, BK)
        off = r64[:, None] * KD + cols[None, :]
        if PF:
            x0 = y0.to(tl.float32)
            x1 = y1.to(tl.float32)
            x2 = y2.to(tl.float32)
            x3 = y3.to(tl.float32)
            o = yo.to(tl.float32)
            c_n = pid_c * CHUNK + (kk + BK) % CHUNK + tl.arange(0, BK)  # wraps to the first block (harmless reload)
            off_n = r64[:, None] * KD + c_n[None, :]
            y0 = tl.load(X + off_n, mask=rm2, other=0.0)
            y1 = tl.load(X + off_n + D, mask=rm2, other=0.0)
            y2 = tl.load(X + off_n + 2 * D, mask=rm2, other=0.0)
            y3 = tl.load(X + off_n + 3 * D, mask=rm2, other=0.0)
            yo = tl.load(OUT + r64[:, None] * D + c_n[None, :], mask=rm2, other=0.0)
        else:
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


XRMS_CFG = dict(bt=32, bk=128, chunk=1024, warps=8, order=1, pf=False)  # order 1 = (c0x0 + c1x1) + (c2x2 + c3x3): the production expand tree (calib_expand.py)


def _xrms_cfg():
    v = os.environ.get("PRIMUS_HC_XRMS_CFG", "").strip()  # bt,warps[,pf]
    c = dict(XRMS_CFG)
    if v:
        t = [int(a) for a in v.split(",")]
        c.update(bt=t[0], warps=t[1], pf=bool(t[2]) if len(t) > 2 else False)
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
        BT=c["bt"], BK=c["bk"], CHUNK=c["chunk"], BN=BN, ORDER=c["order"], PF=c["pf"], num_warps=c["warps"],
    )
    return new, p_raw, p_ss, n_split
