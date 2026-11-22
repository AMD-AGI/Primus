###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""persist track (PRIMUS_PERSIST_ROUTER_BWD=1): the MoE router backward chain in TWO launches instead of seven.

The int step runs, per MoE layer, 7 short serial kernels after the shared/routed expert backward:
    add (x grad: shared + routed)  > _route_bwd > _router_dgrad > _router_wgrad > _split_reduce
    > copy (beta-0 main_grad hook) > add_ (x grad += router dgrad)
(~52 us of kernels + ~18 us of launch gaps per layer). ``PersistRouterFn`` owns the fused router GEMM, the compact
route kernel and the fork of the MoE input into its shared / routed-expert consumers, so its backward receives all
three gradients and runs:

  1. _route_bwd_kernel (unchanged): d logits.
  2. _router_bwd_mega_kernel:
     programs [0, NWG)          router wgrad split-K tile (= _router_wgrad_kernel); the last of the S split tiles of a
                                column block (atomic counter) sums the S partials in _split_reduce_kernel's fixed
                                order and writes main_grad (= split_reduce + the DDP hook's beta-0 copy / accumulate)
     programs [NWG, NWG + NDG)  router dgrad tiles (= _router_dgrad_kernel), rounded to bf16 and added to the shared +
                                routed x gradients with the two bf16 roundings of the autograd accumulation
                                ((g_shared + g_routed) + g_router)

Every program runs the separate kernels' code on the same tile shapes and warps: all outputs are byte-identical.
No program ever waits (last-arriver counters only), so there is nothing that can deadlock, with or without other
streams or graph replay; the last arriver resets its counter, so the sync buffer needs no memset per call.
Cross-XCD hand-off of the split-K partials: device-scope (.cg) stores, s_wait_storecnt 0 in every wave (barrier), a
RELAXED counter atomic, device-scope loads in the reducer. An acq_rel counter (global_wb + global_inv of the per-XCD
L2) cost +25 us per call; d logits come from the previous launch and are read through the cache as in the int kernels.
Microbench (graph replay after a 4k GEMM, L2 flushed): int chain 56 us -> 34.5 us per layer (PRIMUS_PERSIST_RB_MODE=
plain, 4 sync-free launches: 41 us).
Measured dead ends (2026-11-22, ~/dsv41/work/persist/ledger.md): the route backward inside the same launch behind
per-block flags plus an atomic work ticket (+90..+250 us: same-address atomics serialize at ~65 ns, acquire polls
invalidate the L2), a fused router + route forward (acq_rel per program, +40..+65 us), a one-launch MegaMoE route-table
build with a flag wait (+22..+40 us); last-arriver tails on the forward routing chain (route fwd + balance loss: +14 us;
route count + offsets: +1 us) -- a single-program tail behind device-scope loads costs as much as the launch it saves.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .v4_route_compact import RouteCompactFn
from .v4_route_compact import _BN as _RBN
from .v4_router_fused import (
    DG_BM,
    DG_BN,
    WG_BK,
    WG_BN,
    WG_SPLIT,
    _mma,
    _split3,
    _w_split,
    router_fwd,
)


def persist_router_bwd_enabled() -> bool:
    return os.environ.get("PRIMUS_PERSIST_ROUTER_BWD", "0") == "1"


@triton.jit
def _rb_wgrad(pn, s, DL, X, PART, N, H, ROWS, E: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # == _router_wgrad_kernel (v4_router_fused.py), program (pn, s); DL produced in this launch: device-scope loads
    rn = pn * BN + tl.arange(0, BN)
    re = tl.arange(0, E)
    rk = tl.arange(0, BK)
    acc_hi = tl.zeros([E, BN], dtype=tl.float32)
    acc_lo = tl.zeros([E, BN], dtype=tl.float32)
    t0 = s * ROWS
    for k0 in range(t0, t0 + ROWS, BK):
        tk = k0 + rk
        m = tk < N
        a = tl.load(DL + tk[None, :] * E + re[:, None], mask=m[None, :], other=0.0)
        b = tl.load(X + tk[:, None].to(tl.int64) * H + rn[None, :], mask=m[:, None], other=0.0)
        a0, a1, a2 = _split3(a)
        acc_lo = _mma(a2, b, acc_lo, False)
        acc_lo = _mma(a1, b, acc_lo, False)
        acc_hi = _mma(a0, b, acc_hi, False)
    # device-scope (.cg) store: complete past this XCD's L2 once the wave's store counter drains, so the last-arriver
    # counter can be a relaxed atomic (no L2 write-back) and the reducer on another XCD reads it with .cg loads
    tl.store(PART + (s * E + re[:, None]).to(tl.int64) * H + rn[None, :], acc_hi + acc_lo, cache_modifier=".cg")


@triton.jit
def _rb_reduce(pn, PART, OUT, H, NUMEL, E: tl.constexpr, BN: tl.constexpr, S: tl.constexpr,
               PDT: tl.constexpr, ACC: tl.constexpr):
    # == _split_reduce_kernel (fixed split order, beta 0) on the [E, BN] column block, then the main_grad write of the
    # DDP hook: copy (first write of the step) or fp32 add of the param-dtype gradient
    rn = pn * BN + tl.arange(0, BN)
    re = tl.arange(0, E)
    offs = re[:, None] * H + rn[None, :]
    acc = tl.zeros([E, BN], dtype=tl.float32)
    for s in tl.static_range(S):
        acc += tl.load(PART + s * NUMEL + offs, cache_modifier=".cg")
    gp = acc.to(PDT).to(tl.float32)
    if ACC:
        gp = tl.load(OUT + offs).to(tl.float32) + gp
    tl.store(OUT + offs, gp.to(OUT.dtype.element_ty))


@triton.jit
def _rb_split_reduce_mg_kernel(PART, OUT, NUMEL, S: tl.constexpr, BLOCK: tl.constexpr, PDT: tl.constexpr,
                               ACC: tl.constexpr):
    # == _split_reduce_kernel (beta 0, fixed split order, cast to the parameter dtype) + the DDP hook's main_grad write
    # (copy on the first write of the step, else fp32 add) in one launch
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < NUMEL
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in tl.static_range(S):
        acc += tl.load(PART + s * NUMEL + offs, mask=m, other=0.0)
    gp = acc.to(PDT).to(tl.float32)
    if ACC:
        gp = tl.load(OUT + offs, mask=m, other=0.0).to(tl.float32) + gp
    tl.store(OUT + offs, gp.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _rb_dgrad(pm, pn, DL, W, GA, GB, DX, N, H, E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
              W_SPLIT: tl.constexpr, NADD: tl.constexpr):
    # == _router_dgrad_kernel (beta 0, bf16 out), then the autograd accumulation of the MoE input gradient
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    re = tl.arange(0, E)
    mrow = rm < N
    dl = tl.load(DL + rm[:, None] * E + re[None, :], mask=mrow[:, None], other=0.0)
    d0, d1, d2 = _split3(dl)
    w = tl.load(W + re[:, None] * H + rn[None, :]).to(tl.float32)  # [E, BN]
    lo = tl.zeros([BM, BN], dtype=tl.float32)
    if W_SPLIT == 3:
        w0, w1, w2 = _split3(w)
        lo = _mma(d0, w2, lo, False)
        lo = _mma(d1, w1, lo, False)
        lo = _mma(d2, w0, lo, False)
        lo = _mma(d0, w1, lo, False)
    else:
        w0 = w.to(tl.bfloat16)
        lo = _mma(d2, w0, lo, False)
    lo = _mma(d1, w0, lo, False)
    acc = _mma(d0, w0, tl.zeros([BM, BN], dtype=tl.float32), False) + lo
    r = acc.to(DX.dtype.element_ty)
    offs = rm[:, None].to(tl.int64) * H + rn[None, :]
    if NADD == 2:
        ga = tl.load(GA + offs, mask=mrow[:, None], other=0.0).to(tl.float32)
        gb = tl.load(GB + offs, mask=mrow[:, None], other=0.0).to(tl.float32)
        t = (ga + gb).to(DX.dtype.element_ty)
        r = (t.to(tl.float32) + r.to(tl.float32)).to(DX.dtype.element_ty)
    elif NADD == 1:
        ga = tl.load(GA + offs, mask=mrow[:, None], other=0.0).to(tl.float32)
        r = (ga + r.to(tl.float32)).to(DX.dtype.element_ty)
    tl.store(DX + offs, r, mask=mrow[:, None])


@triton.jit
def _router_bwd_mega_kernel(
    DLOGITS, W, X, GA, GB, DX, PART, OUT_W, N, H, ROWS, SYNC,
    NWG: tl.constexpr, NDG: tl.constexpr, NPN: tl.constexpr, NPD: tl.constexpr,
    E: tl.constexpr, S: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, WBN: tl.constexpr,
    WBK: tl.constexpr, W_SPLIT: tl.constexpr, NADD: tl.constexpr, PDT: tl.constexpr, ACC: tl.constexpr,
    DGT: tl.constexpr, SEM: tl.constexpr,
):
    """programs [0, NWG): router wgrad split tile (s, pn); the last of the S splits of column block pn (atomic
    counter, which it then resets) runs the fixed-order split reduce + main_grad write. programs [NWG, NWG + NDG):
    DGT router dgrad tiles + the x-gradient adds. Nothing waits (last-arriver only): no deadlock possible."""
    t = tl.program_id(0)
    if t < NWG:
        s = t // NPN
        pn = t % NPN
        _rb_wgrad(pn, s, DLOGITS, X, PART, N, H, ROWS, E, WBN, WBK)
        tl.debug_barrier()  # s_wait_storecnt 0 in every wave: the device-scope partial stores are complete
        c = tl.atomic_add(SYNC + pn, 1, sem=SEM, scope="gpu")
        if c == S - 1:  # last split of this column block: fixed-order reduce (device-scope loads of the partials)
            _rb_reduce(pn, PART, OUT_W, H, E * H, E, WBN, S, PDT, ACC)
            tl.atomic_xchg(SYNC + pn, 0, sem="relaxed", scope="gpu")  # ready for the next launch
    else:
        j = t - NWG
        NPQ: tl.constexpr = NPD // DGT
        pm = j // NPQ
        pq = j % NPQ
        for q in tl.static_range(DGT):  # DGT column tiles per program (same per-tile code)
            _rb_dgrad(pm, pq * DGT + q, DLOGITS, W, GA, GB, DX, N, H, E, BM, BN, W_SPLIT, NADD)


_TL_DT = {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16}
_sync_bufs = {}
_last_kernel = None


def _sync_buffer(key, n, device):
    buf = _sync_bufs.get(key)
    if buf is None or buf.numel() < n or buf.device != device:
        buf = torch.zeros(n, dtype=torch.int32, device=device)
        _sync_bufs[key] = buf
    return buf


def check_error() -> bool:
    """True when a column counter was left non-zero (it is reset by its last arriver every launch; host sync)."""
    return any(int(b.abs().sum().item()) != 0 for b in _sync_bufs.values())


_DGT = int(os.environ.get("PRIMUS_PERSIST_RB_DGT", "1"))
# counter atomic: "relaxed" (default; the partials are device-scope .cg stores, complete before the atomic) or
# "acq_rel" (measured +25 us per call: the release is a write-back of the XCD's L2)
_SEM = os.environ.get("PRIMUS_PERSIST_RB_SEM", "relaxed")
# "mega": route bwd + one no-wait launch (dgrad + adds | wgrad + last-arriver reduce), 2 launches;
# "plain": 4 launches without any cross-workgroup sync (dgrad + adds, int wgrad, split reduce into main_grad)
_MODE = os.environ.get("PRIMUS_PERSIST_RB_MODE", "mega")


def router_bwd_mega(route, x, w, dtopw, ga, gb, out_w, pdt, acc, need_dx=True, sync_key=None):
    """Two launches (was 7): the route backward (_route_bwd_kernel, d logits), then ``_router_bwd_mega_kernel``:
    dx = [(ga + gb) +] router dgrad (bf16) and dw split-K + fixed-order reduce written into ``out_w`` (``pdt``
    rounding, accumulate when ``acc``). ``route`` = (logits, idx_rows, ids, wsave, denom, cnt, dscale, K, S_LEN,
    LCOEF, SCALE, HASH, LOSS). Returns dx (or None)."""
    from .v4_route_compact import _route_bwd_kernel

    logits, idx_rows, ids, wsave, denom, cnt, dscale, K, s_len, lcoef, scale, hash_, loss = route
    N, E = logits.shape
    H = x.shape[1]
    KP = triton.next_power_of_2(K)
    dev = x.device
    assert H % DG_BN == 0 and H % WG_BN == 0
    rows = triton.cdiv(triton.cdiv(N, WG_SPLIT), WG_BK) * WG_BK
    split = triton.cdiv(N, rows)
    NPN = H // WG_BN
    NPD = H // DG_BN
    NWG = NPN * split if out_w is not None else 0
    dgt = _DGT if NPD % _DGT == 0 else 1
    NDG = triton.cdiv(N, DG_BM) * (NPD // dgt) if need_dx else 0
    dl = torch.empty((N, E), dtype=torch.float32, device=dev)
    _route_bwd_kernel[(triton.cdiv(N, _RBN),)](logits, idx_rows, ids, wsave, denom, dtopw, cnt, dscale, dl, N, K,
                                              s_len, lcoef, E=E, KP=KP, BN=_RBN, SCALE=float(scale), HASH=int(hash_),
                                              LOSS=int(loss), num_warps=4)
    if NWG + NDG == 0:
        return None
    if _MODE == "plain":
        # no cross-workgroup sync at all: dgrad + x-grad adds (one launch), wgrad (int kernel), split reduce straight
        # into main_grad (one launch) -> 4 launches instead of 7
        dx = None
        if need_dx:
            nadd = (ga is not None) + (gb is not None)
            if ga is None and gb is not None:
                ga, gb = gb, None
            dx = torch.empty((N, H), dtype=x.dtype, device=dev)
            _router_bwd_mega_kernel[(NDG,)](
                dl, w, x, ga if ga is not None else x, gb if gb is not None else x, dx, dl, dl, N, H, rows, dl,
                NWG=0, NDG=NDG, NPN=NPN, NPD=NPD, E=E, S=split, BM=DG_BM, BN=DG_BN, WBN=WG_BN, WBK=WG_BK,
                W_SPLIT=_w_split(w), NADD=nadd, PDT=_TL_DT[pdt], ACC=bool(acc), DGT=dgt, SEM=_SEM, num_warps=4,
            )
        if out_w is not None:
            from .v4_router_fused import _router_wgrad_kernel

            part = torch.empty((split, E, H), dtype=torch.float32, device=dev)
            _router_wgrad_kernel[(NPN, split)](dl, x, part, N, H, rows, E=E, BN=WG_BN, BK=WG_BK, EMU=False,
                                               num_warps=4)
            numel = E * H
            _rb_split_reduce_mg_kernel[(triton.cdiv(numel, 1024),)](part, out_w, numel, S=split, BLOCK=1024,
                                                                    PDT=_TL_DT[pdt], ACC=bool(acc), num_warps=4)
        return dx
    part = torch.empty((split, E, H), dtype=torch.float32, device=dev) if out_w is not None else dl
    dx = torch.empty((N, H), dtype=x.dtype, device=dev) if need_dx else None
    nadd = (ga is not None) + (gb is not None)
    if ga is None and gb is not None:
        ga, gb = gb, None
    sync = _sync_buffer((sync_key, dev), NPN, dev)
    global _last_kernel
    _last_kernel = _router_bwd_mega_kernel[(NWG + NDG,)](
        dl, w, x, ga if ga is not None else x, gb if gb is not None else x, dx if dx is not None else x, part,
        out_w if out_w is not None else dl, N, H, rows, sync,
        NWG=NWG, NDG=NDG, NPN=NPN, NPD=NPD, E=E, S=split, BM=DG_BM, BN=DG_BN, WBN=WG_BN, WBK=WG_BK,
        W_SPLIT=_w_split(w), NADD=nadd, PDT=_TL_DT[pdt], ACC=bool(acc), DGT=dgt, SEM=_SEM, num_warps=4,
    )
    return dx


class _Ctx:
    """Minimal ctx for RouteCompactFn.forward (it only saves tensors and sets attributes)."""

    def save_for_backward(self, *t):
        self.saved = t

    def mark_non_differentiable(self, *a):
        pass

    def set_materialize_grads(self, v):
        pass


def _ddp_grad_checks_on() -> bool:
    """Megatron's DDP grad checks (--check_for_nan_in_loss_and_grad, the eager default; --check_for_large_grads) read
    each bucket's main_grad from the backward hooks (PRIMUS_GRAD_NORM_FROM_NAN_CHECK takes the grad norm from that
    read): the router weight's main_grad must then be written before its hook, so the stash hand-off is not used."""
    try:
        from megatron.training import get_args

        a = get_args()
    except Exception:  # noqa: BLE001 -- no Megatron args (unit tests)
        return False
    return bool(getattr(a, "check_for_nan_in_loss_and_grad", False) or getattr(a, "check_for_large_grads", False))


class RouterStash:
    """Hand-off from PersistRouterFn's backward (no kernels: d top-k weights + route state) to PersistForkFn's backward
    (the two launches). Lives for one forward/backward of one MoE layer."""

    __slots__ = ("job",)

    def __init__(self):
        self.job = None


def _ordered_sum(gs):
    acc = None
    for g in gs:
        if g is not None:
            acc = g if acc is None else acc + g
    return acc


class PersistForkFn(torch.autograd.Function):
    """hidden -> (router alias, routed-expert alias, shared-expert alias) [no kernel]. The backward receives the three
    consumers' gradients; when PersistRouterFn left its job in the stash (router input gradient None), it runs the
    route backward + ONE _router_bwd_mega_kernel launch: dx = (g_shared + g_routed) + router dgrad (ForkGrad's /
    the serial engine's order and bf16 roundings) and the router wgrad into main_grad. Without a job it is ForkGrad."""

    @staticmethod
    def forward(ctx, hidden, stash):
        ctx.stash = stash
        ctx.hshape = hidden.shape
        ctx.set_materialize_grads(False)
        return hidden.view_as(hidden), hidden.view_as(hidden), hidden.view_as(hidden)

    @staticmethod
    def backward(ctx, g_r, g_e, g_s):
        job, ctx.stash.job = ctx.stash.job, None
        for g in (g_r, g_e, g_s):  # a branch-stream gradient (PRIMUS_OVERLAP_SHARED) is read on this stream too
            if g is not None and g.is_cuda:
                g.record_stream(torch.cuda.current_stream(g.device))
        if job is None:
            return _ordered_sum((g_s, g_e, g_r)), None
        assert g_r is None, "router input gradient must come through the stash"
        route, flat, w, dtopw, out_w, pdt, acc = job
        N, H = flat.shape
        gs = [g.reshape(N, H) if g is not None else None for g in (g_s, g_e)]
        if any(g is not None and (not g.is_contiguous() or g.dtype != flat.dtype) for g in gs):
            gs = [g.contiguous().to(flat.dtype) if g is not None else None for g in gs]
        dx = router_bwd_mega(route, flat, w, dtopw, gs[0], gs[1], out_w, pdt, acc, need_dx=True, sync_key=id(w))
        return dx.view(ctx.hshape), None


class PersistRouterFn(torch.autograd.Function):
    """(router alias of the MoE input [.., H] bf16, router weight [E, H], ...) -> (topk_w, topk_idx, loss). Forward =
    the int kernels (_router_fwd + _route_fwd [+ _route_loss]). Backward: with a claimed main_grad and a stash it only
    records the job (d top-k weights, route state, main_grad slot; the beta-0 / accumulate decision is taken now, as
    the DDP hook would) and returns no input gradient -- PersistForkFn's backward runs the kernels. Otherwise it runs
    the route backward + megakernel itself (dgrad without adds) and returns dx / dW."""

    @staticmethod
    def forward(ctx, hidden, w, bias, idx_tab, ids, topk, score_function, scale, slot_even, loss_cfg, hash_, claimed,
                stash, tables=None):
        H = hidden.shape[-1]
        flat = hidden.reshape(-1, H)
        if hash_:
            logits, _, _ = router_fwd(flat, w)
            idx_rows = idx_tab
        else:
            logits, idx_rows, _ = router_fwd(flat, w, bias, topk, score_function, scale, idx_pad=True)
        rc = _Ctx()
        topw, topi, loss = RouteCompactFn.forward(rc, logits, idx_rows, ids, topk, scale, slot_even, loss_cfg, tables)
        logits_, idx_rows_, ids_, wsave, denom, cnt = rc.saved
        ctx.save_for_backward(flat, w, logits_, idx_rows_, ids_, wsave, denom, cnt)
        ctx.has_loss, ctx.hash, ctx.K, ctx.scale = rc.has_loss, rc.hash, rc.K, rc.scale
        ctx.lcoef = getattr(rc, "lcoef", 0.0)
        ctx.S = getattr(rc, "S", 1)
        ctx.claimed = claimed
        ctx.stash = stash
        ctx.hshape = hidden.shape
        ctx.mark_non_differentiable(topi, loss)
        ctx.set_materialize_grads(False)
        return topw, topi, loss

    @staticmethod
    def backward(ctx, dtopw, _dtopi, _dloss):
        flat, w, logits, idx_rows, ids, wsave, denom, cnt = ctx.saved_tensors
        N, E = logits.shape
        K = ctx.K
        dev = logits.device
        if dtopw is None:
            dtopw = torch.zeros((N, K), dtype=torch.float32, device=dev)
        dtopw = dtopw.contiguous().float()
        dscale = logits
        if ctx.has_loss:
            from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

            if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
                MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0, device=dev)
            dscale = MoEAuxLossAutoScaler.main_loss_backward_scale
            assert dscale.is_cuda and dscale.numel() == 1
        route = (logits, idx_rows, ids, wsave, denom, cnt, dscale, K, ctx.S if ctx.has_loss else 1,
                 ctx.lcoef if ctx.has_loss else 0.0, ctx.scale, ctx.hash, ctx.has_loss)
        from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.hc_boundary import (
            _GradSlot,
            direct_main_grad,
        )

        stash, ctx.stash = ctx.stash, None
        nones = (None,) * 12
        if (stash is not None and ctx.claimed and ctx.needs_input_grad[0] and ctx.needs_input_grad[1]
                and direct_main_grad(w) is not None and not _ddp_grad_checks_on()):
            slot = _GradSlot(w, w.dtype, tuple(w.shape), dev, ctx.claimed)
            stash.job = (route, flat, w, dtopw, slot.out, w.dtype, slot.acc)
            return (None, slot.grad(w.dtype)) + nones
        dw_ret = out_w = None
        acc = False
        if ctx.needs_input_grad[1]:
            slot = _GradSlot(w, w.dtype, tuple(w.shape), dev, ctx.claimed)
            out_w, acc = slot.out, slot.acc
            dw_ret = slot.grad(w.dtype)
        dx = router_bwd_mega(route, flat, w, dtopw, None, None, out_w, w.dtype, acc,
                             need_dx=ctx.needs_input_grad[0], sync_key=id(w))
        if dx is not None:
            dx = dx.view(ctx.hshape)
        return (dx, dw_ret) + nones


__all__ = ["persist_router_bwd_enabled", "PersistForkFn", "PersistRouterFn", "RouterStash", "router_bwd_mega", "check_error"]
