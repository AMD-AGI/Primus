###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""fuse track: compact V4 routing for the MegaMoE fp8_block experts (``PRIMUS_FUSE_ROUTE=1``, default off).

The dense route chain between the router GEMM and the MegaMoE dispatch is ~28 forward and ~30 backward kernels per
layer: ``V4RouterPostFn`` writes dense ``probs`` / ``routing_map`` [N, E], the sequence-balance loss runs as ~16 torch
ops (+ ~16 in backward), force-load-balancing does ``topk`` + ``RowsFromSlots``, and ``_mega_moe_forward`` runs
``probs.topk`` again to get back to the compact ``(topk_idx, topk_weights)`` MegaMoE consumes; the backward scatters
and gathers its way back through all of them.

Here the router GEMM kernel (``v4_router_fused``) is followed by

* forward, one kernel per token block: score function, gather + renormalise + scale of the top-k weights (the
  ``V4RouterPostFn`` expressions), the descending-weight order that ``probs.topk`` produces, the final expert ids
  (force-load-balancing ``even`` slots, or the routed experts), and per-block partials of the sequence-balance loss
  (expert counts from the routed set, sums of the normalised unbiased affinities);
* forward, one single-program kernel: the loss from the partials in a fixed order (deterministic), and the per
  sequence expert counts the backward needs;
* backward, one kernel per token block: d logits = the ``V4RouterPostFn`` VJP of d topk_weights (mapped back through
  the order) + the sequence-balance-loss VJP scaled by ``MoEAuxLossAutoScaler.main_loss_backward_scale`` (what the
  aux-loss scaler feeds the loss), then the router GEMM's own dgrad / wgrad as before.

Supported: score function ``sqrtsoftplus`` (V4), force-load-balancing off or ``even``, hash and learned routers on
the fused router kernel, a sequence-balance loss whose reduce group has one rank. Anything else keeps the dense path.
Numerics: the routing weights, expert ids and their order are the dense path's (same expressions; ties between equal
weights are ordered by position); the loss and d logits are fp32 sums in a different order (not bit-identical).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_EPS_DENOM = 1.0e-12  # V4RouterPostFn renormalisation clamp
_EPS_AFF = 1.0e-20  # normalise_affinities


def fuse_route_enabled() -> bool:
    return os.environ.get("PRIMUS_FUSE_ROUTE", "0") == "1"


@triton.jit
def _score_post(l):
    # V4RouterPostFn sqrtsoftplus (the expression whose weights feed the experts)
    return tl.sqrt(tl.log(1.0 + tl.exp(l)))


@triton.jit
def _softplus_acc(x):
    # torch F.softplus (beta 1, threshold 20) with an accurate log1p(exp(x)) (the affinities of the balance loss)
    e = tl.exp(tl.minimum(x, 20.0))
    u = 1.0 + e
    lp = tl.where(u == 1.0, e, tl.log(u) * (e / (u - 1.0)))
    return tl.where(x > 20.0, x, lp)


@triton.jit
def _gather_k(tile, idx, e_idx, k_idx, KP: tl.constexpr):
    """tile [BN, E], idx [BN, KP] -> tile[n, idx[n, k]] as [BN, KP] (register compare-select)."""
    out = tl.zeros(idx.shape, dtype=tl.float32)
    for k in tl.static_range(KP):
        col = tl.sum(tl.where(k_idx[None, :] == k, idx, 0), axis=1)
        v = tl.sum(tl.where(e_idx[None, :] == col[:, None], tile, 0.0), axis=1)
        out = tl.where(k_idx[None, :] == k, v[:, None], out)
    return out


@triton.jit
def _rank_desc(w, k_idx, K, KP: tl.constexpr):
    """rank of each valid column in descending-weight order (ties by column), [BN, KP] int32."""
    rank = tl.zeros(w.shape, dtype=tl.int32)
    for i in tl.static_range(KP):
        wi = tl.sum(tl.where(k_idx[None, :] == i, w, 0.0), axis=1)[:, None]
        before = (wi > w) | ((wi == w) & (i < k_idx[None, :]))
        rank += (before & (i < K)).to(tl.int32)
    return rank


@triton.jit
def _load_route(LOGITS, IDX, IDS, n_offs, n_mask, e_idx, k_idx, E: tl.constexpr, K, KP: tl.constexpr,
                HASH: tl.constexpr):
    """logits [BN, E] and the routed experts [BN, KP] read exactly as V4RouterPostFn reads them (row stride KP int64
    index rows: the same register layout, so the same reduction order -> bit-identical routing weights).
    HASH: IDX is the padded tid2eid table [V, KP] int64 and IDS the token ids."""
    km = n_mask[:, None] & (k_idx[None, :] < K)
    logits = tl.load(LOGITS + n_offs[:, None] * E + e_idx[None, :], mask=n_mask[:, None] & (e_idx[None, :] < E),
                     other=0.0).to(tl.float32)
    if HASH:
        rows = tl.load(IDS + n_offs, mask=n_mask, other=0).to(tl.int64)
        idx = tl.load(IDX + rows[:, None] * KP + k_idx[None, :], mask=km, other=0)
    else:
        idx = tl.load(IDX + n_offs[:, None] * KP + k_idx[None, :], mask=km, other=0)
    return logits, idx.to(tl.int32), km


@triton.jit
def _route_fwd_kernel(LOGITS, IDX, IDS, TOPW, TOPI, WSAVE, DENOM, CNT_P, SUM_P, N, K,
                      E: tl.constexpr, KP: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr, HASH: tl.constexpr,
                      SLOT_EVEN: tl.constexpr, LOSS: tl.constexpr,
                      RINV=None, PADROWS=None, RWP=None, NPAD=None, RW_OUT: tl.constexpr = 0, PADB: tl.constexpr = 1,
                      CNTB=None, TBL_CNT: tl.constexpr = 0):
    pid = tl.program_id(0)
    n_offs = pid * BN + tl.arange(0, BN)
    n_mask = n_offs < N
    e_idx = tl.arange(0, E)
    k_idx = tl.arange(0, KP)
    logits, idx, km = _load_route(LOGITS, IDX, IDS, n_offs, n_mask, e_idx, k_idx, E, K, KP, HASH)
    if RW_OUT:
        # route3: MegaMoE pool route weights for the static force-LB-even tables (Turbo even_routing_tables):
        # route (n, k) lives at pool row RINV[n*K + k] (stored below); the padding rows PADROWS get 0 (this
        # program's share). Issued first so the table loads overlap the routing math.
        prow = tl.load(RINV + n_offs[:, None].to(tl.int64) * K + k_idx[None, :], mask=km, other=0)
        pr = pid * PADB + tl.arange(0, PADB)
        pm = pr < NPAD
        zrow = tl.load(PADROWS + pr, mask=pm, other=0)
        tl.store(RWP + zrow, tl.zeros((PADB,), dtype=tl.float32), mask=pm)
    scores = _score_post(logits)
    w = _gather_k(scores, idx, e_idx, k_idx, KP)
    w = tl.where(km, w, 0.0)
    s = tl.sum(w, axis=1, keep_dims=True)
    denom = tl.maximum(s, 1.0e-12)
    w = w / denom
    tl.store(DENOM + n_offs, tl.reshape(denom, (BN,)), mask=n_mask)
    w = w * SCALE
    # router-order weights (V4RouterPostFn's weights_saved): the backward's input, and the [N, KP] row store keeps
    # this tile in V4RouterPostFn's register layout (same denominator reduction order -> bit-identical weights)
    tl.store(WSAVE + n_offs[:, None] * KP + k_idx[None, :], w, mask=km)
    rank = _rank_desc(w, k_idx, K, KP)
    # scatter column j to its rank position (registers: select per output position)
    topw = tl.zeros((BN, KP), dtype=tl.float32)
    topi = tl.zeros((BN, KP), dtype=tl.int32)
    for j in tl.static_range(KP):
        sel = k_idx[None, :] == j
        rj = tl.sum(tl.where(sel, rank, 0), axis=1)
        wj = tl.sum(tl.where(sel, w, 0.0), axis=1)
        ij = tl.sum(tl.where(sel, idx, 0), axis=1)
        hit = (k_idx[None, :] == rj[:, None]) & (j < K)
        topw = tl.where(hit, wj[:, None], topw)
        topi = tl.where(hit, ij[:, None], topi)
    if SLOT_EVEN:
        topi = ((n_offs[:, None].to(tl.int64) * K + k_idx[None, :]) % E).to(tl.int32)
    tl.store(TOPW + n_offs[:, None].to(tl.int64) * K + k_idx[None, :], topw, mask=km)
    tl.store(TOPI + n_offs[:, None].to(tl.int64) * K + k_idx[None, :], topi.to(tl.int64), mask=km)
    if RW_OUT:
        tl.store(RWP + prow, topw, mask=km)
    if TBL_CNT:
        # route3 EARLY=2: this block's expert histogram of the final expert ids (for _route_tables_blk_kernel)
        oh = tl.zeros((BN, E), dtype=tl.int32)
        for k in tl.static_range(KP):
            ck = tl.sum(tl.where(k_idx[None, :] == k, topi, 0), axis=1)
            oh += ((e_idx[None, :] == ck[:, None]) & n_mask[:, None] & (k < K)).to(tl.int32)
        tl.store(CNTB + pid * E + e_idx, tl.sum(oh, axis=0))
    if LOSS:
        sa = tl.sqrt(_softplus_acc(logits))
        sa = tl.where(n_mask[:, None], sa, 0.0)
        aff = sa / (tl.sum(sa, axis=1, keep_dims=True) + 1.0e-20)
        cnt = tl.zeros((BN, E), dtype=tl.float32)
        for k in tl.static_range(KP):
            col = tl.sum(tl.where(k_idx[None, :] == k, idx, 0), axis=1)
            cnt += ((e_idx[None, :] == col[:, None]) & n_mask[:, None] & (k < K)).to(tl.float32)
        tl.store(CNT_P + pid * E + e_idx, tl.sum(cnt, axis=0))
        tl.store(SUM_P + pid * E + e_idx, tl.sum(aff, axis=0))


@triton.jit
def _route_tables_blk_kernel(TOPI, TOPW, CNTB, ROWMAP, RW, INV, POFFS, R, P, NB, K: tl.constexpr, E: tl.constexpr,
                             RB: tl.constexpr, RBP: tl.constexpr, NBP: tl.constexpr, ALIGN: tl.constexpr,
                             TAILB: tl.constexpr):
    """route3 EARLY=2: Turbo build_routing_blk's tables (count > offs > scatter) in ONE launch, no cross-program
    hand-off: program b owns the routes of route-kernel block b ([b*RB, (b+1)*RB)) and recomputes the expert offsets
    from all blocks' histograms CNTB [NB, E] (written by _route_fwd_kernel, the previous launch). Same pool rows (token
    order within an expert = route order), POFFS, padding (ROWMAP -1, RW 0) and tail as the three Turbo kernels."""
    pid = tl.program_id(0)
    e_ar = tl.arange(0, E)
    cb = tl.arange(0, NBP)
    c = tl.load(CNTB + cb[:, None] * E + e_ar[None, :], mask=(cb < NB)[:, None], other=0)
    cnt = tl.sum(c, axis=0)
    before = tl.sum(tl.where((cb < pid)[:, None], c, 0), axis=0)
    plen = tl.maximum((cnt + ALIGN - 1) // ALIGN, 1) * ALIGN
    pend = tl.cumsum(plen, 0)
    pstart = pend - plen
    if pid == 0:
        tl.store(POFFS + 1 + e_ar, pend)
        tl.store(POFFS + e_ar, pstart, mask=e_ar == 0)
    j = tl.arange(0, RBP)
    r = pid * RB + j
    m = (j < RB) & (r < R)
    idx = tl.load(TOPI + r, mask=m, other=-1).to(tl.int32)
    oh = (idx[:, None] == e_ar[None, :]).to(tl.int32)
    pos = tl.sum(oh * ((pstart + before)[None, :] + tl.cumsum(oh, 0) - 1), axis=1)
    tl.store(ROWMAP + pos, r // K, mask=m)
    tl.store(RW + pos, tl.load(TOPW + r, mask=m, other=0.0).to(tl.float32), mask=m)
    tl.store(INV + r, pos, mask=m)
    pa = tl.arange(0, ALIGN)
    for e in range(pid, E, NB):  # padding rows of expert e
        lo = tl.sum(tl.where(e_ar == e, pstart + cnt, 0), axis=0)
        hi = tl.sum(tl.where(e_ar == e, pend, 0), axis=0)
        rows = lo + pa
        tl.store(ROWMAP + rows, tl.full((ALIGN,), -1, tl.int32), mask=rows < hi)
        tl.store(RW + rows, tl.zeros((ALIGN,), dtype=tl.float32), mask=rows < hi)
    last = tl.sum(tl.where(e_ar == E - 1, pend, 0), axis=0)
    tr = last + pid * TAILB + tl.arange(0, TAILB)  # this program's share of the unused tail [last, P)
    tl.store(ROWMAP + tr, tl.full((TAILB,), -1, tl.int32), mask=tr < P)
    tl.store(RW + tr, tl.zeros((TAILB,), dtype=tl.float32), mask=tr < P)


@triton.jit
def _route_loss_kernel(CNT_P, SUM_P, CNT, LOSS, NB_SEQ, B, COEF, E: tl.constexpr, BP: tl.constexpr):
    """one program: per sequence b, C_e and S_e summed over its blocks in block order; loss = mean_b sum_e C_e S_e *
    COEF (COEF = alpha * E / (K * S^2))."""
    e_idx = tl.arange(0, E)
    r = tl.arange(0, BP)
    tot = 0.0
    for b in range(0, B):
        c = tl.zeros((E,), dtype=tl.float32)
        sm = tl.zeros((E,), dtype=tl.float32)
        for p0 in range(0, NB_SEQ, BP):
            rows = b * NB_SEQ + p0 + r
            m = (p0 + r) < NB_SEQ
            c += tl.sum(tl.load(CNT_P + rows[:, None] * E + e_idx[None, :], mask=m[:, None], other=0.0), axis=0)
            sm += tl.sum(tl.load(SUM_P + rows[:, None] * E + e_idx[None, :], mask=m[:, None], other=0.0), axis=0)
        tl.store(CNT + b * E + e_idx, c)
        tot += tl.sum(c * sm, axis=0) * COEF
    tl.store(LOSS, tot / B)


@triton.jit
def _route_bwd_kernel(LOGITS, IDX, IDS, WSAVE, DENOM, DTOPW, CNT, DSCALE, DLOGITS, N, K, S_LEN, LCOEF,
                      E: tl.constexpr, KP: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr, HASH: tl.constexpr,
                      LOSS: tl.constexpr):
    pid = tl.program_id(0)
    n_offs = pid * BN + tl.arange(0, BN)
    n_mask = n_offs < N
    e_idx = tl.arange(0, E)
    k_idx = tl.arange(0, KP)
    logits, idx, km = _load_route(LOGITS, IDX, IDS, n_offs, n_mask, e_idx, k_idx, E, K, KP, HASH)
    # ---- V4RouterPostFn VJP of the top-k weights, mapped back through the descending order (same expressions and
    # saved state as _v4_router_post_bwd_kernel, so this part of d logits is the dense path's)
    scores = _score_post(logits)
    denom = tl.load(DENOM + n_offs, mask=n_mask, other=1.0)
    weights_saved = tl.load(WSAVE + n_offs[:, None] * KP + k_idx[None, :], mask=km, other=0.0)
    rank = _rank_desc(weights_saved, k_idx, K, KP)
    dt = tl.load(DTOPW + n_offs[:, None].to(tl.int64) * K + k_idx[None, :], mask=km, other=0.0).to(tl.float32)
    dprobs_at = tl.zeros((BN, KP), dtype=tl.float32)
    for k in tl.static_range(KP):  # dprobs_at[j] = dt[rank[j]]
        dk = tl.sum(tl.where(k_idx[None, :] == k, dt, 0.0), axis=1)
        dprobs_at = tl.where(rank == k, dk[:, None], dprobs_at)
    dprobs_at = tl.where(km, dprobs_at, 0.0)
    dweights_pre_scale = dprobs_at * SCALE
    gathered_k = weights_saved * (denom[:, None] / SCALE)
    d_denom_per_n = -tl.sum(dweights_pre_scale * gathered_k, axis=1) / (denom * denom)
    d_gathered = dweights_pre_scale / denom[:, None] + d_denom_per_n[:, None]
    d_gathered = tl.where(k_idx[None, :] < K, d_gathered, 0.0)
    d_scores_full = tl.zeros((BN, E), dtype=tl.float32)
    for k_off in tl.static_range(KP):
        idx_col = tl.sum(tl.where(k_idx[None, :] == k_off, idx, 0), axis=1)
        grad_k = tl.sum(d_gathered * (k_idx[None, :] == k_off).to(tl.float32), axis=1)
        d_scores_full += tl.where(e_idx[None, :] == idx_col[:, None], grad_k[:, None], 0.0)
    sig_x = 1.0 - tl.exp(-scores * scores)
    y_safe = tl.maximum(scores, 1.0e-12)
    dl = d_scores_full * sig_x / (2.0 * y_safe)
    if LOSS:
        # sequence-balance loss: L = LCOEF * sum_b sum_e C_be sum_{t in b} a_te (LCOEF = alpha E / (K S^2 B))
        sp = _softplus_acc(logits)
        sa = tl.sqrt(sp)
        z = tl.sum(sa, axis=1, keep_dims=True) + 1.0e-20
        a = sa / z
        b = n_offs // S_LEN
        c = tl.load(CNT + b[:, None] * E + e_idx[None, :], mask=n_mask[:, None], other=0.0)
        c = c * (tl.load(DSCALE).to(tl.float32) * LCOEF)
        dsa = (c - tl.sum(c * a, axis=1, keep_dims=True)) / z
        ex = tl.exp(tl.minimum(logits, 20.0))
        dsp = dsa / (2.0 * sa)
        dl += tl.where(logits > 20.0, dsp, dsp * ex / (ex + 1.0))
    tl.store(DLOGITS + n_offs[:, None].to(tl.int64) * E + e_idx[None, :], dl, mask=n_mask[:, None])


_BN = 64


class RouteCompactFn(torch.autograd.Function):
    """(logits [N, E] fp32, idx_rows [R, KP] int64, ids [N] | None) -> (topk_w [N, K] fp32, topk_idx [N, K] int64,
    loss scalar). Learned routers: ``idx_rows`` holds the routed experts of each token (R = N; columns >= K unread),
    ``ids`` None. Hash routers: ``idx_rows`` is the padded tid2eid table and ``ids`` the token ids.
    ``loss_cfg``: None or (coeff, batch, seq_len) of the sequence-balance loss."""

    @staticmethod
    def forward(ctx, logits, idx_rows, ids, topk, scale, slot_even, loss_cfg, tables=None):
        N, E = logits.shape
        K = int(topk)
        KP = triton.next_power_of_2(K)
        dev = logits.device
        assert logits.dtype == torch.float32 and logits.is_contiguous()
        assert idx_rows.dtype == torch.int64 and idx_rows.dim() == 2 and idx_rows.shape[1] == KP
        assert idx_rows.is_contiguous() and idx_rows.data_ptr() != 0
        hash_ = ids is not None
        if hash_:
            ids = ids.contiguous()
            assert ids.numel() == N and ids.data_ptr() != 0
        else:
            assert idx_rows.shape[0] == N
            ids = idx_rows  # unused
        nb = triton.cdiv(N, _BN)
        topw = torch.empty((N, K), dtype=torch.float32, device=dev)
        topi = torch.empty((N, K), dtype=torch.int64, device=dev)
        wsave = torch.empty((N, KP), dtype=torch.float32, device=dev)
        denom = torch.empty((N,), dtype=torch.float32, device=dev)
        has_loss = loss_cfg is not None
        if has_loss:
            coeff, B, S = loss_cfg
            assert S % _BN == 0 and B * S == N
            cnt_p = torch.empty((nb, E), dtype=torch.float32, device=dev)
            sum_p = torch.empty((nb, E), dtype=torch.float32, device=dev)
        else:
            cnt_p = sum_p = topw  # unused
        rw_kw = {}
        if isinstance(tables, Route3Dyn):
            tables.cntb = torch.empty((nb, E), dtype=torch.int32, device=dev)
            rw_kw = dict(CNTB=tables.cntb, TBL_CNT=1)
        elif tables is not None:
            # route3: also write the MegaMoE pool route weights (tables.rw) of the static even tables
            assert slot_even and tables.inv.numel() == N * K and tables.padrows.dtype == torch.int32
            tables.rw = torch.empty((tables.P,), dtype=torch.float32, device=dev)
            npad = tables.padrows.numel()
            rw_kw = dict(RINV=tables.inv, PADROWS=tables.padrows if npad else tables.inv, RWP=tables.rw, NPAD=npad,
                         RW_OUT=1, PADB=triton.next_power_of_2(max(triton.cdiv(npad, nb), 1)))
        _route_fwd_kernel[(nb,)](logits, idx_rows, ids, topw, topi, wsave, denom, cnt_p, sum_p, N, K, E=E, KP=KP, BN=_BN,
                                 SCALE=float(scale), HASH=int(hash_), SLOT_EVEN=int(slot_even), LOSS=int(has_loss),
                                 num_warps=4, **rw_kw)
        if isinstance(tables, Route3Dyn):
            tables.build(topi, topw, nb, _BN)
        cnt = None
        if has_loss:
            cnt = torch.empty((B, E), dtype=torch.float32, device=dev)
            loss = torch.empty((), dtype=torch.float32, device=dev)
            lcoef = float(coeff) * E / (float(K) * float(S) ** 2)
            _route_loss_kernel[(1,)](cnt_p, sum_p, cnt, loss, S // _BN, B, lcoef, E=E, BP=32, num_warps=4)
            ctx.lcoef = lcoef / B
            ctx.S = S
        else:
            loss = torch.zeros((), dtype=torch.float32, device=dev)
        ctx.save_for_backward(logits, idx_rows, ids, wsave, denom, cnt if has_loss else logits)
        ctx.has_loss, ctx.hash, ctx.K = has_loss, hash_, K
        ctx.scale = float(scale)
        ctx.mark_non_differentiable(topi, loss)
        ctx.set_materialize_grads(False)
        return topw, topi, loss

    @staticmethod
    def backward(ctx, dtopw, _dtopi, _dloss):
        logits, idx_rows, ids, wsave, denom, cnt = ctx.saved_tensors
        N, E = logits.shape
        K = ctx.K
        if dtopw is None and not ctx.has_loss:
            return None, None, None, None, None, None, None, None
        if dtopw is None:
            dtopw = torch.zeros((N, K), dtype=torch.float32, device=logits.device)
        dtopw = dtopw.contiguous().float()
        dscale = logits  # unused without the loss
        if ctx.has_loss:
            from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

            if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
                MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0, device=logits.device)
            dscale = MoEAuxLossAutoScaler.main_loss_backward_scale
            assert dscale.is_cuda and dscale.numel() == 1
        dl = torch.empty((N, E), dtype=torch.float32, device=logits.device)
        _route_bwd_kernel[(triton.cdiv(N, _BN),)](
            logits, idx_rows, ids, wsave, denom, dtopw, cnt, dscale, dl, N, K, ctx.S if ctx.has_loss else 1,
            ctx.lcoef if ctx.has_loss else 0.0, E=E, KP=triton.next_power_of_2(K), BN=_BN, SCALE=ctx.scale,
            HASH=int(ctx.hash), LOSS=int(ctx.has_loss), num_warps=4)
        return dl, None, None, None, None, None, None, None


class Route3Tables:
    """route3 (PRIMUS_ROUTE3_TABLES=1): the static MegaMoE pool tables of force-LB-even routing
    (Turbo ``even_routing_tables``) plus this call's route weights ``rw`` (written by _route_fwd_kernel)."""

    def __init__(self, rowmap, inv, poffs, perm, padrows, P):
        self.rowmap, self.inv, self.poffs, self.perm, self.padrows, self.P = rowmap, inv, poffs, perm, padrows, int(P)
        self.rw = None

    def routing(self):
        assert self.rw is not None
        return (self.rowmap, self.rw, self.inv, self.poffs, self.perm)


class Route3Dyn:
    """route3 EARLY=2 (any routing): MegaMoE pool tables from the route kernel's per-block histograms + one
    _route_tables_blk_kernel launch (replaces Turbo's count > offs > scatter; same bytes)."""

    def __init__(self, E, P, align):
        self.E, self.P, self.align = int(E), int(P), int(align)
        self.cntb = None
        self.tabs = None

    def build(self, topi, topw, nb, bn):
        N, K = topi.shape
        R, E, dev = N * K, self.E, topi.device
        rowmap = torch.empty((self.P,), dtype=torch.int32, device=dev)
        rw = torch.empty((self.P,), dtype=torch.float32, device=dev)
        inv = torch.empty((R,), dtype=torch.int32, device=dev)
        poffs = torch.empty((E + 1,), dtype=torch.int32, device=dev)
        rb = bn * K
        _route_tables_blk_kernel[(nb,)](
            topi, topw, self.cntb, rowmap, rw, inv, poffs, R, self.P, nb, K=K, E=E, RB=rb,
            RBP=triton.next_power_of_2(rb), NBP=triton.next_power_of_2(nb), ALIGN=self.align,
            TAILB=triton.next_power_of_2(max(triton.cdiv(max(self.P - R, 1), nb), 1)), num_warps=8)
        self.tabs = (rowmap, rw, inv, poffs, None)

    def routing(self):
        assert self.tabs is not None
        return self.tabs


def route3_tables_enabled() -> bool:
    return os.environ.get("PRIMUS_ROUTE3_TABLES", "0") == "1"


def route3_static_tables(force_lb_mode, N, K, E, device):
    """route3 guard: the static force-LB-even pool tables, or None (-> the dynamic int path, untouched) unless
    PRIMUS_ROUTE3_TABLES=1 AND force-load-balancing "even" is on (the only mode where the expert of route r is
    r % E whatever the router picks; real training has no forced routing)."""
    if force_lb_mode != "even" or not route3_tables_enabled():
        return None
    from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8blk import even_routing_tables

    t = even_routing_tables(int(N), int(K), int(E), device)
    return Route3Tables(*t) if t is not None else None


def route3_early_mode() -> int:
    """PRIMUS_ROUTE3_EARLY (any routing): 1 = build the MegaMoE pool tables (same Turbo kernels, same inputs) right
    after the route kernel, i.e. before the shared-expert join, instead of inside the MegaMoE forward after it;
    2 = the same tables from the route kernel's block histograms + ONE merged launch (Route3Dyn). 0 = off."""
    return int(os.environ.get("PRIMUS_ROUTE3_EARLY", "0"))


def route3_dyn_tables(N, K, E):
    """Route3Dyn for EARLY=2, else None."""
    if route3_early_mode() != 2:
        return None
    from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8blk import _ALIGN, routing_pool_rows

    return Route3Dyn(E, routing_pool_rows(int(N), int(K), int(E)), _ALIGN)


def pad_rows(idx: torch.Tensor) -> torch.Tensor:
    """[R, K] integer -> [R, next_pow2(K)] int64 (columns >= K zero)."""
    R, K = idx.shape
    KP = triton.next_power_of_2(K)
    out = torch.zeros((R, KP), dtype=torch.int64, device=idx.device)
    out[:, :K] = idx
    return out


def route_compact(logits, idx_rows, ids, topk, scale, slot_even, loss_cfg, tables=None):
    return RouteCompactFn.apply(logits, idx_rows, ids, int(topk), float(scale), bool(slot_even), loss_cfg, tables)


__all__ = ["fuse_route_enabled", "pad_rows", "route_compact", "RouteCompactFn", "Route3Tables",
           "route3_tables_enabled", "route3_static_tables", "route3_early_mode", "route3_dyn_tables", "Route3Dyn"]
