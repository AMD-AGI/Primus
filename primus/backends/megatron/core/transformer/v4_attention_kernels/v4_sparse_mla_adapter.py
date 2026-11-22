###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel-agnostic V4 attention adapters over a sparse-MLA fwd/bwd kernel pair.

Every "fused single-latent" V4 backend (gluon, triton-v2, flydsl-v2) speaks the
same sparse-MLA contract:

* ``fwd(q[T,H,Dqk], kv[T,1,Dqk], topk[T,TOPK], attn_sink, kv_lora_rank, scale)``
  -> ``(o[T,H,Dv], lse[T,H])``
* ``bwd(q, kv, o, do, topk, lse, attn_sink, kv_lora_rank, scale)``
  -> ``(dq, dkv, d_sink)``

This module maps Primus's V4 attention representations (per-head q, single MQA
latent K = V with RoPE baked in-place over ``head_dim = 512``, compressed pool,
per-query top-K, joint local-SWA + sparse softmax with sink) onto that contract
and maps gradients back — once — so each backend is just a kernel pair:

* :func:`make_csa_from_pool(fwd, bwd)`  -> CSA (cr=4) wrapper
* :func:`make_attention(fwd, bwd)`      -> dense (cr=0) / HCA (cr=128) wrapper

The fwd/bwd kernels are passed to the autograd Function as non-tensor args, so
the same Function serves all backends (backward returns ``None`` for them).
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import weakref
from typing import Callable, Optional

import torch

# Set while a fwd / bwd kernel runs on a "canonical band" top-k map: B == 1, no CP, no
# packing, the 128-row sliding window [t-127, t] in slot order, plus (HCA) the causal pool
# kv row Skv' + p visible iff t >= cr * (p + 1) - 1 with cr = S / P. A backend may then
# use closed-form band kernels that do not read the indices (Turbo gfx1250 use_swa=True).
_BAND_HINT = contextvars.ContextVar("v4_sparse_mla_band_hint", default=False)
_SWA_BAND_W = 128
# id(mask) -> (weakref to mask, mask._version, ok). Not a WeakKeyDictionary: tensor keys
# compare with the elementwise __eq__.
_mask_ok_cache = {}


_BAND_TOPK_CACHE = {}

# PRIMUS_V4_OROPE_INPLACE: (data_ptr, numel) of the output the LAST sparse-MLA forward saved for its backward. The
# attention module's single exit consumes it right after that forward (take_saved_output), so a stale entry can never
# match a later, unrelated tensor at a reused address.
_PENDING_OUT = [None]


def take_saved_output(t: torch.Tensor) -> bool:
    """True if ``t`` is (the start and size of) the output the last sparse-MLA forward saved; clears the record."""
    rec, _PENDING_OUT[0] = _PENDING_OUT[0], None
    return rec is not None and rec == (t.data_ptr(), t.numel())


def _fused_glue_on(t: torch.Tensor) -> bool:
    """Fused Triton glue (one kernel for the CSA top-k map / the rope-padded kv) on GPU tensors."""
    return t.is_cuda and os.environ.get("PRIMUS_V4_FUSED_GLUE", "1") == "1"


def _glue():
    from ._triton_common import sparse_mla_glue

    return sparse_mla_glue


def band_hint():
    """True inside a fwd / bwd call whose top-k map is a canonical band (see above)."""
    return _BAND_HINT.get()


@contextlib.contextmanager
def _band(on: bool):
    tok = _BAND_HINT.set(on if isinstance(on, tuple) else bool(on))
    try:
        yield
    finally:
        _BAND_HINT.reset(tok)


def _hca_mask_is_closed_form(mask: torch.Tensor, S: int, P: int) -> bool:
    """``mask`` ([S, P] additive) == the causal pool mask with ratio S // P. Checked once per
    mask tensor (one device sync; the model caches its mask, so once per run and layer)."""
    ent = _mask_ok_cache.get(id(mask))
    if ent is not None and ent[0]() is mask and ent[1] == mask._version:
        return ent[2]
    cr = S // P
    t = torch.arange(S, device=mask.device).view(S, 1)
    vis = (torch.arange(P, device=mask.device).view(1, P) + 1) * cr - 1 <= t
    ok = bool(torch.equal(mask.reshape(S, P) == 0, vis))
    if len(_mask_ok_cache) > 64:
        _mask_ok_cache.clear()
    _mask_ok_cache[id(mask)] = (weakref.ref(mask), mask._version, ok)
    return ok


def _csa_band_pool(B, S, Skv, P, W, cp_dwindow, cp_global_start, seq_starts):
    """``(W, Skv, P)`` when a CSA map is "window + pool" in its simplest form: B == 1, no CP,
    no packing, so slot s < W of token t is raw row t - W + 1 + s (or -1) and the rest are
    pool rows Skv + p (indexer top-k: each at most once per token) or -1. The backward may
    then reduce dKV without a CSR inverse (Turbo ``band_pool``). Else ``False``."""
    if os.environ.get("PRIMUS_V4_BAND_KERNELS", "1") != "1":
        return False
    if B != 1 or W <= 0 or int(cp_dwindow) != 0 or int(cp_global_start) != 0 or seq_starts is not None:
        return False
    if Skv != S or P <= 0:
        return False
    return (int(W), int(Skv), int(P))


def _canonical_band(B, S, Skv, W, cp_dwindow, cp_global_start, seq_starts, hca_local_seqlen, mask) -> bool:
    if os.environ.get("PRIMUS_V4_BAND_KERNELS", "1") != "1":
        return False
    if B != 1 or W != _SWA_BAND_W or int(cp_dwindow) != 0 or int(cp_global_start) != 0 or seq_starts is not None:
        return False
    if hca_local_seqlen > 0 and mask is not None:
        P = Skv - int(hca_local_seqlen)
        if int(hca_local_seqlen) != S or P <= 0 or S % P or mask.numel() != S * P:
            return False
        return _hca_mask_is_closed_form(mask, S, P)
    return Skv == S

_ROPE_PAD = 64  # dummy separate-rope block (zeros); the kernels need D_ROPE > 0


def _rope_pad_q(q_bh: torch.Tensor) -> torch.Tensor:
    """``[B, H, S, D]`` -> ``[B*S, H, D + _ROPE_PAD]``, pad block zeroed.

    Allocates the padded buffer and strided-copies the query into its first ``D``
    columns, rather than ``cat([q_bh.permute(...).reshape(...), zeros], -1)``. The
    permuted view is non-contiguous, so that ``.reshape`` materialises a FULL extra
    copy before the cat allocates the result on top of it: the cat form peaks at
    ~3.2x the query size where this one peaks at ~2.1x. At 1M with CP=8 the query is
    ``[1, 64, 131072, 512]`` = 8 GiB, and that difference (~9 GiB) is exactly what
    put the run over a 192 GiB card -- it OOM'd here with 176.11 GiB already live.
    """
    B, H, S, D = q_bh.shape
    Dp = D + _ROPE_PAD
    # The fused query norm+RoPE (``q_norm_rope.QNormRopePadFn``) already writes the query
    # into a zero-padded [B, S, H, D + _ROPE_PAD] buffer and hands out its [..., :D] view:
    # use that buffer as-is instead of copying the query into a fresh one.
    base = q_bh._base
    if (
        base is not None
        and getattr(base, "_v4_rope_pad", 0) == _ROPE_PAD
        and tuple(base.shape) == (B, S, H, Dp)
        and base.is_contiguous()
        and q_bh.data_ptr() == base.data_ptr()
        and q_bh.stride() == (S * H * Dp, Dp, H * Dp, 1)
    ):
        return base.view(B * S, H, Dp)
    out = q_bh.new_empty(B, S, H, Dp)
    out[..., :D] = q_bh.permute(0, 2, 1, 3)  # strided copy, no contiguous temp
    out[..., D:].zero_()  # zero only the pad block, not the whole buffer first
    return out.reshape(B * S, H, D + _ROPE_PAD)  # contiguous -> free view


def _bshd_slice_to_bhsd(dq_g: torch.Tensor, B: int, S: int, H: int, D: int) -> torch.Tensor:
    """Drop the rope pad off ``[B*S, H, D+_ROPE_PAD]`` and return ``[B, H, S, D]``.

    The inverse of :func:`_rope_pad_q`, and it has to dodge the same trap. Written the
    obvious way -- ``dq_g[:, :, :D].reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()``
    -- it costs TWO full copies: the padded slice is non-contiguous so ``.reshape``
    materialises one, and ``.contiguous()`` after the permute materialises another. At 1M
    with CP=8 that is 8.00 GiB each, in the backward pass, which is where the run died.
    Allocating the destination and doing one strided copy through a permuted view of it
    leaves a single 8.00 GiB allocation.

    Returning a permuted VIEW of the padded buffer costs no copy at all: the only consumer
    is ``q_bh = q.transpose(1, 2)``, whose backward turns this straight back into a
    row-strided BSHD view, and the fused RoPE backward reads rows through their strides.
    Materialising BHSD here only made the RoPE backward materialise BSHD again (two
    transposing copies of the full query gradient that cancel out).
    """
    return dq_g[:, :, :D].unflatten(0, (B, S)).permute(0, 2, 1, 3)


def _rope_pad_kv(kv512: torch.Tensor) -> torch.Tensor:
    """``[N, 1, D]`` -> ``[N, 1, D + _ROPE_PAD]``, pad block zeroed. See :func:`_rope_pad_q`."""
    N, G, D = kv512.shape
    out = kv512.new_empty(N, G, D + _ROPE_PAD)
    out[..., :D] = kv512
    out[..., D:].zero_()
    return out


def _window_indices(S, W, base, cp_dwindow, cp_global_start, seq_starts, idx_dtype, neg1, device):
    """Sliding-window column indices ``[1, S, W]``, ``-1`` where the slot is invalid.

    The window is validated against a per-query ORIGIN and indexed in local buffer
    coordinates. Without packing the origin is the scalar ``cp_global_start`` (this
    rank's first global row), so "position >= 0" means "not before the sequence start".
    Under packing (``seq_starts`` given, an ``[S]`` tensor) each row has its own origin --
    the first row of its packed sequence -- so the same test additionally stops a query
    from reaching back into the previous sample.

    The kernel is purely index-driven and treats ``-1`` as "this column does not exist"
    (it clamps the pointer, zeroes the value and forces the score to -inf, in both the
    forward and the backward), so masking cross-sequence columns this way is exact.
    """
    q = torch.arange(S, device=device, dtype=idx_dtype).view(S, 1)
    off = torch.arange(W, device=device, dtype=idx_dtype).view(1, W)
    if seq_starts is None:
        gpos = q + int(cp_global_start) - W + 1 + off
        win_valid = gpos >= 0
        win_pos = gpos - int(cp_global_start) + int(cp_dwindow)
    else:
        # Local coordinates throughout: positions are row indices in this rank's buffer,
        # and the origin is the row where this query's own sequence starts.
        starts = seq_starts.to(device=device, dtype=idx_dtype).view(S, 1)
        pos = q - W + 1 + off  # [S, W] candidate rows
        win_valid = pos >= starts
        win_pos = pos + int(cp_dwindow)
    win_idx = base + win_pos.view(1, S, W)
    return torch.where(win_valid.view(1, S, W), win_idx, neg1)


def _pad_topk_64(topk: torch.Tensor) -> torch.Tensor:
    """Pad the topk width to a multiple of 64 with -1 so a backend whose dKV
    tiling is 64-wide (e.g. gluon) stays valid (HCA 128+32=160 -> 192)."""
    tk = topk.shape[1]
    pad = ((tk + 63) // 64) * 64 - tk
    if pad > 0:
        topk = torch.cat(
            [topk, torch.full((topk.shape[0], pad), -1, device=topk.device, dtype=topk.dtype)], dim=1
        )
    return topk.contiguous()


def _build_csa_topk(
    topk_idxs: torch.Tensor,
    S: int,
    Skv: int,
    P: int,
    W: int,
    cp_dwindow: int = 0,
    cp_global_start: int = 0,
    seq_starts=None,
) -> torch.Tensor:
    """Flat topk [B*S, W+K] over the per-batch [raw ++ pool] buffer.

    ``topk_idxs`` [B, S, K] holds pool indices in [0, P) (or -1). Batch ``b``
    occupies rows ``[b*(Skv+P) : (b+1)*(Skv+P))`` (raw 0..Skv-1, pool Skv..Skv+P-1).

    ``S`` is the QUERY count, ``Skv`` the raw-token KV count. They differ only under
    context parallelism, where the raw buffer is ``[boundary ++ local]`` and so
    ``Skv == cp_dwindow + S``: the sliding window is then validated against GLOBAL
    positions (a token must not attend before the sequence start) but indexed in LOCAL
    buffer coordinates. The pool is already global, so ``topk_idxs`` needs no shift.
    With ``cp_dwindow == cp_global_start == 0`` and ``Skv == S`` this is byte-identical
    to the non-CP form.
    """
    B, _, K = topk_idxs.shape
    device = topk_idxs.device
    # int32 throughout -- see the note in _V4SparseMLAAttnFn.forward. The result is cast
    # to int32 regardless, so int64 intermediates only double the transient peak.
    idx_dtype = torch.int32
    neg1 = -1  # Python scalar: torch.tensor(-1, device=cuda) was a pageable H2D copy (host sync, not graph-capturable)
    base = (torch.arange(B, device=device, dtype=idx_dtype) * (Skv + P)).view(B, 1, 1)

    win_idx = _window_indices(S, W, base, cp_dwindow, cp_global_start, seq_starts, idx_dtype, neg1, device)

    pool_valid = topk_idxs >= 0
    pool_idx = torch.where(pool_valid, base + Skv + topk_idxs.to(idx_dtype), neg1)

    return torch.cat([win_idx, pool_idx], dim=2).reshape(B * S, W + K).contiguous()


class _V4SparseMLACSAFn(torch.autograd.Function):
    """Autograd wrapper: sparse-MLA FWD/BWD for the V4 CSA (cr=4) layer."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q_bh: torch.Tensor,  # [B, H, S, D]
        k_local_bh: torch.Tensor,  # [B, H, S, D] broadcast, or [B, Skv, 1, D] if k_is_latent
        v_local_bh: Optional[torch.Tensor],  # == k_local in V4; None when k_is_latent
        pool: torch.Tensor,  # [B, P, D]
        topk_idxs: torch.Tensor,  # [B, S, K] pool indices, -1 = invalid
        sink: Optional[torch.Tensor],  # [H] fp32 or None
        swa_window: int,
        scale: float,
        cp_dwindow: int,
        cp_global_start: int,
        k_is_latent: bool,
        seq_starts,  # [S] per-row sequence origin under THD packing, else None
        fwd_fn: Callable,
        bwd_fn: Callable,
    ) -> torch.Tensor:
        B, H, S, D = q_bh.shape
        # Under CP the raw KV buffer is [boundary ++ local], so it is LONGER than the
        # query count; every KV-side extent below is Skv, not S.
        # `k_is_latent` means the caller handed us the un-broadcast [B, Skv, 1, D] latent
        # rather than its head-broadcast view -- see _V4SparseMLAAttnFn.forward for why
        # (the broadcast's gradient would be a [B, H, Skv, D] buffer that is zero except
        # at head 0: 8.01 GiB at 1M with CP=8, which is what the backward died on).
        ctx.k_is_latent = bool(k_is_latent)
        Skv = k_local_bh.shape[1] if k_is_latent else k_local_bh.shape[2]
        P = pool.shape[1]
        W = int(swa_window)
        assert q_bh.dtype == torch.bfloat16, "sparse-MLA adapter requires bf16"
        assert W > 0, "sparse-MLA adapter requires swa_window > 0"

        latent = k_local_bh[:, :, 0, :] if k_is_latent else k_local_bh[:, 0, :, :]  # [B, Skv, D]

        q_g = _rope_pad_q(q_bh)
        if _fused_glue_on(q_bh) and latent.stride(-1) == 1 and pool.stride(-1) == 1 and pool.dtype == latent.dtype:
            kv_g = _glue().rope_pad_kv(latent, pool, _ROPE_PAD)
        else:
            kv_g = _rope_pad_kv(torch.cat([latent, pool], dim=1).reshape(B * (Skv + P), 1, D))

        if seq_starts is None and _fused_glue_on(q_bh) and topk_idxs.stride(-1) == 1:
            topk_g = _glue().csa_topk(topk_idxs, S, Skv, P, W, int(cp_dwindow), int(cp_global_start))
        else:
            topk_g = _pad_topk_64(
                _build_csa_topk(topk_idxs, S, Skv, P, W, int(cp_dwindow), int(cp_global_start), seq_starts)
            )

        sink_arg = sink.float().contiguous() if sink is not None else None
        o_g, lse = fwd_fn(q_g, kv_g, topk_g, attn_sink=sink_arg, kv_lora_rank=D, scale=float(scale))

        ctx.save_for_backward(q_g, kv_g, o_g, lse, topk_g, sink_arg if sink is not None else q_g.new_empty(0))
        _PENDING_OUT[0] = (o_g.data_ptr(), o_g.numel())
        ctx.shapes = (B, H, S, Skv, D, P, W)
        ctx.band = _csa_band_pool(B, S, Skv, P, W, cp_dwindow, cp_global_start, seq_starts)
        ctx.scale = float(scale)
        ctx.sink_was_none = sink is None
        ctx.bwd_fn = bwd_fn
        # Return a VIEW, not a copy. The kernel's `o_g` is contiguous [B*S, H, D], i.e.
        # already BSHD; every caller immediately does `out_bh.transpose(1, 2).contiguous()`
        # to get back to BSHD, so a `.contiguous()` here would materialise BHSD only for
        # the caller to materialise BSHD again -- two full copies of the output that
        # cancel out. Without it, the caller's transpose restores the original strides and
        # its `.contiguous()` becomes a no-op. At 1M with CP=8 each copy is 8.00 GiB, and
        # this is the allocation the run actually died on.
        return o_g.reshape(B, S, H, D).permute(0, 2, 1, 3)

    @staticmethod
    def backward(ctx, grad_o_bh: torch.Tensor):  # type: ignore[override]
        q_g, kv_g, o_g, lse, topk_g, sink_saved = ctx.saved_tensors
        B, H, S, Skv, D, P, W = ctx.shapes
        sink_arg = None if ctx.sink_was_none else sink_saved

        grad_o_g = grad_o_bh.permute(0, 2, 1, 3).reshape(B * S, H, D).contiguous()
        with _band(ctx.band):
            dq_g, dkv_g, dsink = ctx.bwd_fn(
                q_g, kv_g, o_g, grad_o_g, topk_g, lse, attn_sink=sink_arg, kv_lora_rank=D, scale=ctx.scale
            )

        dq_bh = _bshd_slice_to_bhsd(dq_g, B, S, H, D)
        dkv512 = dkv_g[:, 0, :D].reshape(B, Skv + P, D)
        dlatent = dkv512[:, :Skv, :]
        dpool = dkv512[:, Skv:, :].contiguous()

        if ctx.k_is_latent:
            # Gradient has the latent's own [B, Skv, 1, D] shape -- 128 MiB at 1M/CP=8
            # instead of an 8.01 GiB buffer that is 63/64 zeros. Nothing to memset.
            dk_local = dlatent.to(dq_bh.dtype).unsqueeze(2)
        else:
            dk_local = torch.zeros(B, H, Skv, D, device=dq_bh.device, dtype=dq_bh.dtype)
            dk_local[:, 0, :, :] = dlatent.to(dq_bh.dtype)
        # V4 is single-latent (K = V = kv): the kernel returns one combined
        # ``dkv`` which we route entirely through ``dk_local``. The V branch
        # gradient is structurally zero, so we return ``None`` for it instead
        # of allocating (and zeroing) a full [B, H, S, D] tensor — this removes
        # the largest ``Memset (Device)`` bucket in the trace (~268 MB / call).
        # ``k_local_bh`` and ``v_local_bh`` are two ``kv.expand`` views of the
        # same latent, so autograd accumulates ``dk_local + 0`` into ``kv`` —
        # identical to before.
        dv_local = None

        dsink_out = None
        if not ctx.sink_was_none and dsink is not None:
            dsink_out = dsink.to(sink_saved.dtype)

        # forward args: (q, k_local, v_local, pool, topk_idxs, sink, swa_window, scale,
        #                cp_dwindow, cp_global_start, k_is_latent, seq_starts, fwd_fn, bwd_fn)
        return (
            dq_bh,
            dk_local,
            dv_local,
            dpool.to(dq_bh.dtype),
            None,
            dsink_out,
            None,
            None,
            None,
            None,
            None,  # seq_starts
            None,
            None,
            None,
        )


class _V4SparseMLAAttnFn(torch.autograd.Function):
    """Sparse-MLA FWD/BWD for the V4 dense (cr=0) and HCA (cr=128) layers."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q_bh: torch.Tensor,  # [B, H, S, D]
        k_bh: torch.Tensor,  # [B, H, Skv, D], or the [B, Skv, 1, D] latent if k_is_latent
        v_bh: Optional[torch.Tensor],  # == k_bh in V4; None when k_is_latent
        sink: Optional[torch.Tensor],
        swa_window: int,
        additive_mask: Optional[torch.Tensor],  # [S, P] pool-only mask (HCA) or None
        scale: float,
        hca_local_seqlen: int,
        cp_dwindow: int,
        cp_global_start: int,
        k_is_latent: bool,
        seq_starts,  # [S] per-row sequence origin under THD packing, else None
        fwd_fn: Callable,
        bwd_fn: Callable,
    ) -> torch.Tensor:
        B, H, S, D = q_bh.shape
        # V4 is single-latent MQA: this kernel reads ONE key row per position, and the H
        # query heads all use it. Callers may therefore hand us the raw [B, Skv, 1, D]
        # latent (k_is_latent) instead of its [B, H, Skv, D] head-broadcast view. That is
        # strictly better: the broadcast view is free going forward, but autograd would
        # make us return a [B, H, Skv, D] gradient of which 63/64 is zeros -- 8.5 GiB at
        # 1M with CP=8, which is what the backward pass died on.
        ctx.k_is_latent = bool(k_is_latent)
        Skv = k_bh.shape[1] if k_is_latent else k_bh.shape[2]
        W = int(swa_window)
        assert q_bh.dtype == torch.bfloat16, "sparse-MLA adapter requires bf16"
        assert W > 0, "sparse-MLA adapter requires swa_window > 0"

        band = _canonical_band(
            B, S, Skv, W, cp_dwindow, cp_global_start, seq_starts, int(hca_local_seqlen), additive_mask
        )
        # A certified band map depends only on (S, Skv, W, pool) -- the same tensor for every layer
        # and step: build it once. (Kernels only read it.)
        band_key = (B, S, Skv, W, int(hca_local_seqlen), str(q_bh.device)) if band else None
        topk_g = _BAND_TOPK_CACHE.get(band_key) if band else None
        if topk_g is None:
            device = q_bh.device
            # int32 throughout: this index matrix is cast to int32 by _pad_topk_64 anyway, and
            # the largest value is B*(Skv+P) -- 139264 at 1M with CP=8, nowhere near the int32
            # range. Building it in torch's default int64 doubled every intermediate below,
            # which at 1M is ~8.6 GiB apiece.
            idx_dtype = torch.int32
            neg1 = -1  # Python scalar: torch.tensor(-1, device=cuda) was a pageable H2D copy (host sync, not graph-capturable)
            base = (torch.arange(B, device=device, dtype=idx_dtype) * Skv).view(B, 1, 1)
            # Context parallel: this rank owns global rows [cp_global_start, +S), and its KV
            # buffer is [boundary ++ local] with `cp_dwindow` boundary rows received from the
            # left neighbour. So the window is validated against GLOBAL positions (a token must
            # not attend before the sequence start) but indexed in LOCAL buffer coordinates.
            # With cp_dwindow == cp_global_start == 0 this is byte-identical to the non-CP form.
            win_idx = _window_indices(
                S, W, base, cp_dwindow, cp_global_start, seq_starts, idx_dtype, neg1, device
            )

            if hca_local_seqlen > 0 and additive_mask is not None:
                P = Skv - int(hca_local_seqlen)
                vis = (additive_mask == 0).view(1, S, P)
                ps = torch.arange(P, device=device, dtype=idx_dtype).view(1, 1, P)
                # `neg1` is a 0-dim tensor, not torch.full((B, S, P), -1): the full form
                # materialised an entire [B, S, P] constant just to be the `where` else-branch
                # -- 8.6 GiB at 1M, thrown away immediately.
                pool_idx = torch.where(vis, base + hca_local_seqlen + ps, neg1)
                topk = torch.cat([win_idx, pool_idx], dim=2)
            else:
                topk = win_idx
            topk_g = _pad_topk_64(topk.reshape(B * S, -1))
            if band:
                if len(_BAND_TOPK_CACHE) > 16:
                    _BAND_TOPK_CACHE.clear()
                _BAND_TOPK_CACHE[band_key] = topk_g

        q_g = _rope_pad_q(q_bh)
        latent = k_bh[:, :, 0, :] if k_is_latent else k_bh[:, 0, :, :]  # [B, Skv, D]
        if _fused_glue_on(q_bh) and latent.stride(-1) == 1:
            kv_g = _glue().rope_pad_kv(latent, None, _ROPE_PAD)
        else:
            kv_g = _rope_pad_kv(latent.reshape(B * Skv, 1, D))

        sink_arg = sink.float().contiguous() if sink is not None else None
        ctx.band = band
        with _band(band):
            o_g, lse = fwd_fn(q_g, kv_g, topk_g, attn_sink=sink_arg, kv_lora_rank=D, scale=float(scale))

        ctx.save_for_backward(q_g, kv_g, o_g, lse, topk_g, sink_arg if sink is not None else q_g.new_empty(0))
        _PENDING_OUT[0] = (o_g.data_ptr(), o_g.numel())
        ctx.shapes = (B, H, S, D, Skv)
        ctx.scale = float(scale)
        ctx.sink_was_none = sink is None
        ctx.bwd_fn = bwd_fn
        # Return a VIEW, not a copy. The kernel's `o_g` is contiguous [B*S, H, D], i.e.
        # already BSHD; every caller immediately does `out_bh.transpose(1, 2).contiguous()`
        # to get back to BSHD, so a `.contiguous()` here would materialise BHSD only for
        # the caller to materialise BSHD again -- two full copies of the output that
        # cancel out. Without it, the caller's transpose restores the original strides and
        # its `.contiguous()` becomes a no-op. At 1M with CP=8 each copy is 8.00 GiB, and
        # this is the allocation the run actually died on.
        return o_g.reshape(B, S, H, D).permute(0, 2, 1, 3)

    @staticmethod
    def backward(ctx, grad_o_bh: torch.Tensor):  # type: ignore[override]
        q_g, kv_g, o_g, lse, topk_g, sink_saved = ctx.saved_tensors
        B, H, S, D, Skv = ctx.shapes
        sink_arg = None if ctx.sink_was_none else sink_saved

        grad_o_g = grad_o_bh.permute(0, 2, 1, 3).reshape(B * S, H, D).contiguous()
        with _band(ctx.band):
            dq_g, dkv_g, dsink = ctx.bwd_fn(
                q_g, kv_g, o_g, grad_o_g, topk_g, lse, attn_sink=sink_arg, kv_lora_rank=D, scale=ctx.scale
            )

        dq_bh = _bshd_slice_to_bhsd(dq_g, B, S, H, D)
        dkv = dkv_g[:, 0, :D].reshape(B, Skv, D)
        # Single-latent (K = V): route the combined ``dkv`` through the K slot; the V
        # branch gradient is structurally zero, so return ``None`` for it.
        if ctx.k_is_latent:
            # The caller passed the [B, Skv, 1, D] latent, so the gradient has that shape
            # too -- 136 MiB at 1M/CP=8 instead of the 8.5 GiB [B, H, Skv, D] buffer below,
            # of which only head 0 was ever nonzero. No zeros to allocate or memset.
            dk_bh = dkv.to(dq_bh.dtype).unsqueeze(2)
        else:
            dk_bh = torch.zeros(B, H, Skv, D, device=dq_bh.device, dtype=dq_bh.dtype)
            dk_bh[:, 0, :, :] = dkv.to(dq_bh.dtype)
        dv_bh = None

        dsink_out = None
        if not ctx.sink_was_none and dsink is not None:
            dsink_out = dsink.to(sink_saved.dtype)

        # forward args: (q, k, v, sink, swa_window, additive_mask, scale, hca_local_seqlen,
        #                cp_dwindow, cp_global_start, k_is_latent, seq_starts, fwd_fn, bwd_fn)
        return (
            dq_bh,
            dk_bh,
            dv_bh,
            dsink_out,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # seq_starts
            None,
        )


def make_csa_from_pool(fwd_fn: Callable, bwd_fn: Callable) -> Callable:
    """Build a ``v4_csa_attention_v1``-style wrapper for a kernel pair."""

    def _csa_from_pool(
        q_bh,
        k_local_bh,
        v_local_bh,
        pool,
        *,
        topk_idxs,
        sink,
        swa_window,
        attn_dropout,
        training,
        scale,
        cp_dwindow=0,
        cp_global_start=0,
        k_is_latent=False,
        seq_starts=None,
    ):
        if attn_dropout > 0.0 and training:
            raise NotImplementedError(
                "sparse-MLA CSA adapter does not implement in-kernel attention dropout "
                f"(V4 trains with attn_dropout=0). Got attn_dropout={attn_dropout}, training={training}."
            )
        return _V4SparseMLACSAFn.apply(
            q_bh,
            k_local_bh,
            v_local_bh,
            pool,
            topk_idxs,
            sink,
            int(swa_window),
            float(scale),
            int(cp_dwindow),
            int(cp_global_start),
            bool(k_is_latent),
            seq_starts,
            fwd_fn,
            bwd_fn,
        )

    return _csa_from_pool


def make_attention(fwd_fn: Callable, bwd_fn: Callable) -> Callable:
    """Build a dense (cr=0) / HCA (cr=128) attention wrapper for a kernel pair."""

    def _attention(
        q,
        k,
        v,
        *,
        sink,
        swa_window,
        additive_mask,
        attn_dropout,
        training,
        scale,
        hca_local_seqlen=0,
        cp_dwindow=0,
        cp_global_start=0,
        k_is_latent=False,
        seq_starts=None,
    ):
        if attn_dropout > 0.0 and training:
            raise NotImplementedError(
                "sparse-MLA attention adapter does not implement in-kernel attention dropout "
                f"(V4 trains with attn_dropout=0). Got attn_dropout={attn_dropout}, training={training}."
            )
        return _V4SparseMLAAttnFn.apply(
            q,
            k,
            v,
            sink,
            int(swa_window),
            additive_mask,
            float(scale),
            int(hca_local_seqlen),
            int(cp_dwindow),
            int(cp_global_start),
            bool(k_is_latent),
            seq_starts,
            fwd_fn,
            bwd_fn,
        )

    return _attention


__all__ = ["make_csa_from_pool", "make_attention"]
