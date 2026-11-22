###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused skinny router kernels for the V4 routers (``[N, H] x [E, H]^T``, E = 32).

Replaces ``F.linear(x.float(), w.float())`` + score fn + bias + ``topk`` (hipBLASLt MT32x32x16 fp32
GEMMs plus fp32 casts of x and dx) with three Triton kernels that only use bf16 WMMA:

* forward: 16 tokens per program, x (bf16) read once. An fp32 weight is split in registers into three
  bf16 parts ``w = w0 + w1 + w2`` (exact for normal values), so ``x . w`` is three bf16 products whose
  partial products are exact in fp32: fp32-GEMM accuracy. The epilogue applies the score fn, adds the
  selection bias and runs a TOPK-step argmax, writing logits (fp32), indices and the gathered weights.
* dgrad: ``dx = dl . w`` written straight in x's dtype (no fp32 dx + cast); ``BETA=1`` adds into an
  existing dx. dl and w are split in three bf16 parts, products with i + j <= 2 (6 WMMAs, K = E = 32).
* wgrad: ``dw = dl^T . x``: 128 columns x 8 token splits, dl split in 3 bf16 parts, fp32 partials
  ``[S, E, H]`` summed in a fixed order by a second kernel (deterministic, no atomics).

Gate: ``PRIMUS_V4_ROUTER_FUSED=1`` (default off).
"""

import os

import torch
import triton
import triton.language as tl

_SCORE_FN = {"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2}


@triton.jit
def _split2(v):
    v0 = v.to(tl.bfloat16)
    v1 = (v - v0.to(tl.float32)).to(tl.bfloat16)
    return v0, v1


@triton.jit
def _split3(v):
    v0 = v.to(tl.bfloat16)
    r = v - v0.to(tl.float32)
    v1 = r.to(tl.bfloat16)
    v2 = (r - v1.to(tl.float32)).to(tl.bfloat16)
    return v0, v1, v2


@triton.jit
def _mma(a, b, acc, EMU: tl.constexpr):
    # EMU: Triton-interpreter CPU check only. The interpreter's bf16 tl.dot is broken (it multiplies the raw
    # 16-bit patterns), so upcast first; bf16 -> fp32 is exact and bf16 x bf16 products are exact in fp32.
    if EMU:
        return tl.dot(a.to(tl.float32), b.to(tl.float32), acc)
    else:
        return tl.dot(a, b, acc)


@triton.jit
def _cast_out(v, dtype: tl.constexpr, EMU: tl.constexpr):
    # EMU: the interpreter's fp32 -> bf16 cast truncates; round to nearest-even by hand (the GPU cast is RNE).
    if EMU and dtype == tl.bfloat16:
        u = v.to(tl.uint32, bitcast=True)
        u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
        v = u.to(tl.float32, bitcast=True)
    return v.to(dtype)


@triton.jit
def _softplus(x):
    # torch.nn.functional.softplus (beta=1, threshold=20) with an accurate log1p(exp(x)).
    e = tl.exp(tl.minimum(x, 20.0))
    u = 1.0 + e
    lp = tl.where(u == 1.0, e, tl.log(u) * (e / (u - 1.0)))
    return tl.where(x > 20.0, x, lp)


@triton.jit
def _router_fwd_kernel(
    X, W, BIAS, LOGITS, IDX, WTS, N, H,
    E: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
    TOPK: tl.constexpr, KP: tl.constexpr, W_SPLIT: tl.constexpr,
    SCORE_FN: tl.constexpr, HAS_BIAS: tl.constexpr, RENORM: tl.constexpr, SCALE: tl.constexpr, EMU: tl.constexpr,
    IDXS: tl.constexpr = 0,
):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    mrow = rm < N
    re = tl.arange(0, E)
    rk = tl.arange(0, BK)
    xp = X + rm[:, None].to(tl.int64) * H + rk[None, :]
    wp = W + re[None, :] * H + rk[:, None]  # [BK, E] view of w[E, H]
    acc_hi = tl.zeros([BM, E], dtype=tl.float32)
    acc_lo = tl.zeros([BM, E], dtype=tl.float32)
    for _ in range(0, H, BK):
        a = tl.load(xp, mask=mrow[:, None], other=0.0)
        w = tl.load(wp)
        if W_SPLIT == 3:
            w0, w1, w2 = _split3(w.to(tl.float32))
            acc_lo = _mma(a, w2, acc_lo, EMU)
            acc_lo = _mma(a, w1, acc_lo, EMU)
            acc_hi = _mma(a, w0, acc_hi, EMU)
        else:
            acc_hi = _mma(a, w.to(tl.bfloat16), acc_hi, EMU)
        xp += BK
        wp += BK
    logits = acc_hi + acc_lo
    tl.store(LOGITS + rm[:, None] * E + re[None, :], logits, mask=mrow[:, None])

    if TOPK > 0:
        if SCORE_FN == 0:
            mx = tl.max(logits, 1)
            ex = tl.exp(logits - mx[:, None])
            s = ex / tl.sum(ex, 1)[:, None]
        elif SCORE_FN == 1:
            s = tl.sigmoid(logits)
        else:
            s = tl.sqrt(_softplus(logits))
        if HAS_BIAS:
            sel = s + tl.load(BIAS + re).to(tl.float32)[None, :]
        else:
            sel = s
        kk = tl.arange(0, KP)
        idx = tl.zeros([BM, KP], dtype=tl.int32)
        wts = tl.zeros([BM, KP], dtype=tl.float32)
        for j in tl.static_range(TOPK):
            m = tl.max(sel, 1)
            ij = tl.min(tl.where(sel == m[:, None], re[None, :], E), 1)  # lowest index among ties
            hit = re[None, :] == ij[:, None]
            wj = tl.sum(tl.where(hit, s, 0.0), 1)
            sel = tl.where(hit, float("-inf"), sel)
            idx = tl.where(kk[None, :] == j, ij[:, None], idx)
            wts = tl.where(kk[None, :] == j, wj[:, None], wts)
        if RENORM:
            wts = wts / tl.maximum(tl.sum(wts, 1), 1.0e-12)[:, None]
        if SCALE != 1.0:
            wts = wts * SCALE
        km = mrow[:, None] & (kk[None, :] < TOPK)
        # IDXS > 0: index rows padded to IDXS int64 (fuse track compact route reads them in V4RouterPostFn's layout)
        tl.store(IDX + rm[:, None] * (IDXS if IDXS > 0 else TOPK) + kk[None, :], idx.to(tl.int64), mask=km)
        tl.store(WTS + rm[:, None] * TOPK + kk[None, :], wts, mask=km)


@triton.jit
def _router_dgrad_kernel(
    DL, W, DX, N, H,
    E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, W_SPLIT: tl.constexpr, BETA: tl.constexpr, EMU: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    re = tl.arange(0, E)
    mrow = rm < N
    dl = tl.load(DL + rm[:, None] * E + re[None, :], mask=mrow[:, None], other=0.0)  # [BM, E] fp32
    d0, d1, d2 = _split3(dl)
    w = tl.load(W + re[:, None] * H + rn[None, :]).to(tl.float32)  # [E, BN]
    # products d_i w_j with i + j <= 2, smallest first: ~fp32 dgrad, then one rounding to x's dtype
    lo = tl.zeros([BM, BN], dtype=tl.float32)
    if W_SPLIT == 3:
        w0, w1, w2 = _split3(w)
        lo = _mma(d0, w2, lo, EMU)
        lo = _mma(d1, w1, lo, EMU)
        lo = _mma(d2, w0, lo, EMU)
        lo = _mma(d0, w1, lo, EMU)
    else:
        w0 = w.to(tl.bfloat16)
        lo = _mma(d2, w0, lo, EMU)
    lo = _mma(d1, w0, lo, EMU)
    acc = _mma(d0, w0, tl.zeros([BM, BN], dtype=tl.float32), EMU) + lo
    dxp = DX + rm[:, None].to(tl.int64) * H + rn[None, :]
    if BETA:
        acc += tl.load(dxp, mask=mrow[:, None], other=0.0).to(tl.float32)
    tl.store(dxp, _cast_out(acc, DX.dtype.element_ty, EMU), mask=mrow[:, None])


@triton.jit
def _router_wgrad_kernel(
    DL, X, PART, N, H, ROWS,
    E: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, EMU: tl.constexpr,
):
    # PART[s, :, cols] = sum over the ROWS tokens of split s of dl[t, :]^T x[t, cols]
    pn = tl.program_id(0)
    s = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    re = tl.arange(0, E)
    rk = tl.arange(0, BK)
    acc_hi = tl.zeros([E, BN], dtype=tl.float32)
    acc_lo = tl.zeros([E, BN], dtype=tl.float32)
    t0 = s * ROWS
    for k0 in range(t0, t0 + ROWS, BK):
        tk = k0 + rk
        m = tk < N
        a = tl.load(DL + tk[None, :] * E + re[:, None], mask=m[None, :], other=0.0)  # [E, BK] fp32
        b = tl.load(X + tk[:, None].to(tl.int64) * H + rn[None, :], mask=m[:, None], other=0.0)  # [BK, BN]
        a0, a1, a2 = _split3(a)
        acc_lo = _mma(a2, b, acc_lo, EMU)
        acc_lo = _mma(a1, b, acc_lo, EMU)
        acc_hi = _mma(a0, b, acc_hi, EMU)
    tl.store(PART + (s * E + re[:, None]).to(tl.int64) * H + rn[None, :], acc_hi + acc_lo)


@triton.jit
def _split_reduce_kernel(PART, OUT, NUMEL, S: tl.constexpr, BETA: tl.constexpr, BLOCK: tl.constexpr, EMU: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < NUMEL
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in tl.static_range(S):  # fixed order: deterministic
        acc += tl.load(PART + s * NUMEL + offs, mask=m, other=0.0)
    if BETA:
        acc += tl.load(OUT + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + offs, _cast_out(acc, OUT.dtype.element_ty, EMU), mask=m)


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------

FWD_BM, FWD_BK = 16, 128
DG_BM, DG_BN = 64, 128
WG_BN, WG_BK, WG_SPLIT = 128, 64, 8


def _w_split(w):
    return 3 if w.dtype == torch.float32 else 1


def router_fwd(x, w, bias=None, topk=0, score_function="sqrtsoftplus", scale=1.0, idx_pad=False):
    """``logits = x . w^T`` (fp32) and, when ``topk > 0``, the top-k ``(indices int64, weights fp32)``
    of ``score_fn(logits) + bias`` (weights gathered from the unbiased score, renormalized for
    non-softmax, times ``scale``). ``idx_pad``: indices returned as the whole [N, next_pow2(topk)] row buffer
    (columns >= topk unwritten)."""
    N, H = x.shape
    E = w.shape[0]
    logits = torch.empty(N, E, dtype=torch.float32, device=x.device)
    k = max(int(topk), 0)
    kp = triton.next_power_of_2(max(k, 1))
    idx = torch.empty(N, kp if idx_pad else max(k, 1), dtype=torch.int64, device=x.device)
    wts = torch.empty(N, max(k, 1), dtype=torch.float32, device=x.device)
    bk, extra = FWD_BK, {}
    cfg = os.environ.get("PRIMUS_MB_ROUTER_FWD", "")
    if cfg and _w_split(w) == 1 and not _interpreting():
        # membound: "BK,NUM_STAGES" for a bf16 router weight (the per-output sums run over the same 32-deep k-steps
        # in the same order for any BK / pipelining depth: bit-identical, bench_router.py); default off
        b, ns = (int(v) for v in cfg.split(","))
        if H % b == 0:
            bk, extra = b, {"num_stages": ns}
    _router_fwd_kernel[(triton.cdiv(N, FWD_BM),)](
        x, w, bias if bias is not None else w, logits, idx, wts, N, H,
        E=E, BM=FWD_BM, BK=bk, TOPK=k, KP=triton.next_power_of_2(max(k, 1)), W_SPLIT=_w_split(w),
        SCORE_FN=_SCORE_FN[score_function], HAS_BIAS=bias is not None,
        RENORM=score_function != "softmax", SCALE=float(scale), EMU=_interpreting(), IDXS=kp if idx_pad else 0,
        num_warps=4, **extra,
    )
    return logits, (idx if idx_pad else idx[:, :k]), wts[:, :k]


def router_dgrad(dl, w, dx=None, dtype=torch.bfloat16):
    """``dx = dl . w`` in ``dtype``; with ``dx`` given, ``dx += dl . w`` in place (beta = 1)."""
    N, E = dl.shape
    H = w.shape[1]
    beta = dx is not None
    if dx is None:
        dx = torch.empty(N, H, dtype=dtype, device=dl.device)
    _router_dgrad_kernel[(triton.cdiv(N, DG_BM), H // DG_BN)](
        dl, w, dx, N, H, E=E, BM=DG_BM, BN=DG_BN, W_SPLIT=_w_split(w), BETA=beta, EMU=_interpreting(), num_warps=4,
    )
    return dx


def router_wgrad(dl, x, out=None, dtype=torch.float32, split=WG_SPLIT):
    """``dw = dl^T . x`` ([E, H]); with ``out`` given, ``out += dw`` (beta = 1, e.g. main_grad)."""
    N, H = x.shape
    E = dl.shape[1]
    rows = triton.cdiv(triton.cdiv(N, split), WG_BK) * WG_BK
    split = triton.cdiv(N, rows)
    part = torch.empty(split, E, H, dtype=torch.float32, device=x.device)
    _router_wgrad_kernel[(H // WG_BN, split)](dl, x, part, N, H, rows, E=E, BN=WG_BN, BK=WG_BK, EMU=_interpreting(), num_warps=4)
    beta = out is not None
    if out is None:
        out = torch.empty(E, H, dtype=dtype, device=x.device)
    numel = E * H
    _split_reduce_kernel[(triton.cdiv(numel, 1024),)](part, out, numel, S=split, BETA=beta, BLOCK=1024, EMU=_interpreting(), num_warps=4)
    return out


class _RouterFusedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias, topk, score_function, scale, idx_pad=False):
        logits, idx, wts = router_fwd(x, w, bias, topk, score_function, scale, idx_pad)
        ctx.save_for_backward(x, w)
        ctx.mark_non_differentiable(idx, wts)
        if os.environ.get("PRIMUS_GLUE_SMALLS", "0") == "1":
            # the index / weight outputs never carry a gradient: don't let autograd build two zero tensors for them
            ctx.set_materialize_grads(False)
        return logits, idx, wts

    @staticmethod
    def backward(ctx, dl, _didx, _dwts):
        x, w = ctx.saved_tensors
        if dl is None:  # only possible with set_materialize_grads(False): logits unused
            dl = torch.zeros(x.shape[0], w.shape[0], dtype=torch.float32, device=x.device)
        dl = dl.contiguous().float()
        dx = router_dgrad(dl, w, dtype=x.dtype) if ctx.needs_input_grad[0] else None
        dw = router_wgrad(dl, x, dtype=w.dtype) if ctx.needs_input_grad[1] else None
        return dx, dw, None, None, None, None, None


def _interpreting():
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def is_fused_router_enabled() -> bool:
    return os.environ.get("PRIMUS_V4_ROUTER_FUSED", "0") == "1"


def router_fused_supported(x, w, topk=0):
    E = w.shape[0]
    return (
        (x.is_cuda or _interpreting())
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and w.dtype in (torch.float32, torch.bfloat16)
        and x.is_contiguous()
        and w.is_contiguous()
        and E in (16, 32, 64)
        and x.shape[1] % 128 == 0
        and 0 <= topk <= E
    )


def v4_router_fused(x, w, bias=None, topk=0, score_function="sqrtsoftplus", scale=1.0, idx_pad=False):
    """Autograd entry: returns ``(logits fp32 [N, E], indices int64 [N, topk], weights fp32 [N, topk])``.
    Gradients flow through ``logits`` only (to x and w); ``bias`` only steers the selection.
    ``idx_pad``: indices as the padded [N, next_pow2(topk)] row buffer (see router_fwd)."""
    return _RouterFusedFn.apply(x, w, bias, int(topk), score_function, float(scale), bool(idx_pad))


__all__ = [
    "router_fwd",
    "router_dgrad",
    "router_wgrad",
    "v4_router_fused",
    "router_fused_supported",
    "is_fused_router_enabled",
]
