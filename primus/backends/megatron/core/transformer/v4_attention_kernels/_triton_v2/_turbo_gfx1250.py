###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in routing of the triton_v2 sparse-MLA kernel pair to Primus-Turbo's gfx1250
FlyDSL sparse-MLA kernels (``primus_turbo.flydsl.attention.gfx1250_sparse_mla``).

Enabled only on gfx1250 with ``PRIMUS_DSA_GFX1250_TURBO=1``; gfx950 and the default
gfx1250 path are untouched. Same contract as ``sparse_mla_{fwd,bwd}_v4_triton``
(natural-log, sink-inclusive lse; zero rope columns in dq / dkv; d_sink returned).

Two traps handled here:
* Turbo's default scale is 1/sqrt(576); V4 uses 1/sqrt(kv_lora_rank) -> always pass it.
* ``use_swa=True`` (Turbo's default, also what its public op uses) detects the cr0 / cr128
  bands from tensor SHAPES and ignores ``topk_indices``; that is wrong for V4's flattened
  B>1 batches, CP boundary rows and THD packing. So it is used only when the adapter
  certified the map as the canonical band (band_hint); otherwise index-driven mode.
"""

import functools
import os

from ._amd_knobs import is_gfx1250


@functools.lru_cache(maxsize=1)
def _kernels():
    from primus_turbo.flydsl.attention.gfx1250_sparse_mla.bwd_kernel import (
        sparse_mla_bwd_gfx1250,
    )
    from primus_turbo.flydsl.attention.gfx1250_sparse_mla.fwd_kernel import (
        sparse_mla_fwd_gfx1250,
    )

    return sparse_mla_fwd_gfx1250, sparse_mla_bwd_gfx1250


def turbo_gfx1250_enabled(q, num_heads, topk) -> bool:
    if os.environ.get("PRIMUS_DSA_GFX1250_TURBO", "0") != "1" or not is_gfx1250():
        return False
    # Turbo kernel constraints: d_qk == 576, H % 32 == 0 and 128 % H == 0, topk % 32 == 0.
    return q.shape[-1] == 576 and num_heads in (32, 64, 128) and topk % 32 == 0


def _use_band_kernels() -> bool:
    """Closed-form band kernels only when the V4 adapter certified the top-k map as the
    canonical cr0 / cr128 band (B == 1, no CP / packing, causal pool); see
    ``v4_sparse_mla_adapter.band_hint``. Turbo re-checks the shapes and falls back to the
    index-driven kernels when they do not fit."""
    from ..v4_sparse_mla_adapter import band_hint

    return band_hint() is True


def _band_pool():
    """``(window, n_raw, pool)`` when the adapter certified a CSA window + pool map, else None."""
    from ..v4_sparse_mla_adapter import band_hint

    h = band_hint()
    return h if isinstance(h, tuple) else None


def sparse_mla_fwd_turbo_gfx1250(q, kv, topk_indices, attn_sink, kv_lora_rank, scale):
    fwd, _ = _kernels()
    return fwd(
        q,
        kv,
        topk_indices,
        attn_sink=attn_sink,
        kv_lora_rank=kv_lora_rank,
        scale=float(scale),
        use_swa=_use_band_kernels(),
    )


@functools.lru_cache(maxsize=1)
def _bwd_takes_dkv_stream() -> bool:
    import inspect

    _, bwd = _kernels()
    return "dkv_stream" in inspect.signature(bwd).parameters


@functools.lru_cache(maxsize=1)
def _bwd_takes_dpool_out() -> bool:
    import inspect

    _, bwd = _kernels()
    return "dpool_out" in inspect.signature(bwd).parameters


@functools.lru_cache(maxsize=1)
def _bwd_takes_dsink_main_grad() -> bool:
    import inspect

    _, bwd = _kernels()
    return "dsink_main_grad" in inspect.signature(bwd).parameters


@functools.lru_cache(maxsize=1)
def _bwd_takes_dkv_defer() -> bool:
    import inspect

    _, bwd = _kernels()
    return "dkv_defer" in inspect.signature(bwd).parameters


def _dsink_kwargs():
    """PRIMUS_LAUNCH_DSINK: the sink's ``(main_grad, accumulate)`` set by the adapter's backward is passed as
    ``dsink_main_grad`` (Turbo writes the sink gradient there and returns None for it). Taking the entry tells the
    adapter it was consumed; a Turbo build without the kwarg leaves it, and the adapter writes main_grad itself.
    """
    from ..v4_sparse_mla_adapter import dsink_mg_hint

    slot = dsink_mg_hint()
    if not slot or slot[0] is None or not _bwd_takes_dsink_main_grad():
        return {}
    t, slot[0] = slot[0], None
    return {"dsink_main_grad": t}


def _dkv_side_kwargs():
    """PRIMUS_OVERLAP (CSA_BWD): (stream, keep list) set by the CSA adapter's backward -> Turbo leaves the dKV gathers
    on that stream (no join); only for the CSA band map and a Turbo build that has the kwargs."""
    from ..v4_sparse_mla_adapter import dkv_side_hint

    h = dkv_side_hint()
    if h is None or _band_pool() is None or not _bwd_takes_dkv_stream():
        return {}
    kw = {"dkv_stream": h[0], "dkv_keep": h[1]}
    if _bwd_takes_dpool_out():
        kw["dpool_out"] = h[2]
    return kw


def sparse_mla_bwd_turbo_gfx1250(q, kv, o, do, topk_indices, lse, attn_sink, kv_lora_rank, scale):
    _, bwd = _kernels()
    return bwd(
        q,
        kv,
        o,
        do.contiguous(),
        topk_indices,
        lse,
        attn_sink=attn_sink,
        kv_lora_rank=kv_lora_rank,
        scale=float(scale),
        use_swa=_use_band_kernels(),
        **({"band_pool": _band_pool()} if _band_pool() is not None else {}),
        **_dkv_side_kwargs(),
        **_dsink_kwargs(),
    )


# ---------------------------------------------------------------------------------------------------------------------
# PRIMUS_V4_BAND_PER_SAMPLE: closed-form band kernels for a flattened B > 1 batch, one call per sample.
#
# The band kernels (SWA / HCA forward and backward, CSA "window + pool" dKV gather without a CSR inverse) are certified
# only for B == 1: they derive the window from the token position, so in a flattened [B*S] call the first 127 tokens of
# sample b > 0 would see sample b-1. With MBS > 1 every layer therefore fell back to the index-driven kernels: an
# intermediate [T, topk, 512] bf16 written and re-read, a torch.sort + searchsorted CSR inverse per chunk and the CSR
# gather (MBS4 trace: ~11 ms per microbatch of sparse attention, ~2.8 ms of it sort / gather glue in the CSA layer).
# Here each sample runs exactly the B == 1 (MBS1) kernels on its own rows: q / o / dO / lse / dq / delta are row slices
# of the batch tensors (contiguous), kv / dkv the sample's [Skv (+ P)] block (the adapter's kv layout is per-sample
# blocks already), outputs are written in place through Turbo's out= / *_out= kwargs (no concatenation). dsink is taken
# ONCE over the batch-wide lse / delta (same function and inputs as the flattened call).
# Numerics: per sample identical to an MBS1 step (the configuration the band kernels were validated in); vs the
# flattened index-driven MBS4 path the dKV / dsink sums differ in fp32 order (bf16-rounding level), o / dq per token
# may also differ in the last bit (different kernel); see test_sparse_mla_band_per_sample.py. Default on for B > 1
# (PRIMUS_V4_BAND_PER_SAMPLE=0 restores the index-driven path); B == 1 never takes it.
# ---------------------------------------------------------------------------------------------------------------------


def per_sample_ok(d_qk, num_heads, topk) -> bool:
    """Per-sample band calls available for a [.., num_heads, d_qk] query and a top-k width: the Turbo gfx1250 kernels
    are selected (same test as ``turbo_gfx1250_enabled``) and the Turbo build has the out= kwargs."""
    if os.environ.get("PRIMUS_DSA_GFX1250_TURBO", "0") != "1" or not is_gfx1250():
        return False
    if not (d_qk == 576 and num_heads in (32, 64, 128) and topk % 32 == 0):
        return False
    return _takes_out_kwargs()


@functools.lru_cache(maxsize=1)
def _takes_out_kwargs() -> bool:
    import inspect

    fwd, bwd = _kernels()
    f = inspect.signature(fwd).parameters
    b = inspect.signature(bwd).parameters
    return all(k in f for k in ("out", "lse_out")) and all(k in b for k in ("dq_out", "dkv_out", "delta_out"))


@functools.lru_cache(maxsize=1)
def _takes_batch_kwarg() -> bool:
    """Turbo build whose fwd / bwd take ``batch=`` (all samples' band kernels in one launch, same bytes)."""
    import inspect

    fwd, bwd = _kernels()
    return "batch" in inspect.signature(fwd).parameters and "batch" in inspect.signature(bwd).parameters


def _batched_band() -> bool:
    # PRIMUS_V4_BAND_BATCHED (default 1): one launch for all samples of a band layer instead of one per sample.
    return os.environ.get("PRIMUS_V4_BAND_BATCHED", "1") == "1" and _takes_batch_kwarg()


def sparse_mla_fwd_turbo_gfx1250_ps(q, kv, topk1, attn_sink, kv_lora_rank, scale, batch):
    """Forward of a band map (SWA / HCA) for ``batch`` samples: ``q`` [B*S, H, 576], ``kv`` [B*nkv, 1, 576] (sample b =
    rows [b*nkv, (b+1)*nkv)), ``topk1`` the B == 1 map [S, topk] (read only if Turbo falls back to the index-driven
    kernel). Returns (o [B*S, H, 512], lse [B*S, H])."""
    import torch

    fwd, _ = _kernels()
    T, H, _ = q.shape
    S, nkv = T // batch, kv.shape[0] // batch
    assert S * batch == T and nkv * batch == kv.shape[0] and topk1.shape[0] == S
    o = q.new_empty(T, H, kv_lora_rank)
    lse = torch.empty(T, H, dtype=torch.float32, device=q.device)
    if _batched_band():
        fwd(q, kv, topk1, attn_sink=attn_sink, kv_lora_rank=kv_lora_rank, scale=float(scale), use_swa=True, out=o,
            lse_out=lse, batch=batch)
        return o, lse
    for b in range(batch):
        r = slice(b * S, (b + 1) * S)
        fwd(
            q[r],
            kv[b * nkv : (b + 1) * nkv],
            topk1,
            attn_sink=attn_sink,
            kv_lora_rank=kv_lora_rank,
            scale=float(scale),
            use_swa=True,
            out=o[r],
            lse_out=lse[r],
        )
    return o, lse


def sparse_mla_bwd_turbo_gfx1250_ps(q, kv, o, do, topk_local, lse, attn_sink, kv_lora_rank, scale, batch):
    """Backward counterpart: ``topk_local`` [B*S, topk] holds each sample's map in sample-local kv rows (or one
    [S, topk] map shared by all samples). The band hint (``band_hint()``) selects the SWA / HCA band kernels (True)
    or the CSA window + pool dKV gather (tuple); the CSA_BWD dKV stream hint is honoured per call (no dpool_out).
    Returns (dq, dkv, dsink) like ``sparse_mla_bwd_turbo_gfx1250``."""
    import torch

    from primus_turbo.flydsl.attention.gfx1250_sparse_mla.bwd_kernel import _dsink

    _, bwd = _kernels()
    T, H, _ = q.shape
    S, nkv = T // batch, kv.shape[0] // batch
    assert S * batch == T and nkv * batch == kv.shape[0]
    shared = topk_local.shape[0] == S
    dq = torch.empty_like(q)
    dkv = torch.empty_like(kv)
    delta = torch.empty(T, H, dtype=torch.float32, device=q.device)
    band = _band_pool()
    side = _dkv_side_kwargs()
    side.pop("dpool_out", None)
    if os.environ.get("PRIMUS_V4_BAND_PER_SAMPLE_DKV_SIDE", "1") == "0":
        # Joined per call: the B per-sample [S, topk, 512] intermediates (2.7 GB each for the V4-Flash CSA layer at
        # S = 4096) are reused instead of all staying alive on the dKV stream until the CSADkvJoin. Leaving the
        # holder's event unset makes the adapter take the plain (joined) path.
        side = {}
    if band is None and _use_band_kernels() and _batched_band():
        # SWA / HCA band: all samples in one launch (dsink below, as in the loop)
        bwd(q, kv, o, do, topk_local, lse, attn_sink=None, kv_lora_rank=kv_lora_rank, scale=float(scale),
            use_swa=True, dq_out=dq, dkv_out=dkv, delta_out=delta, batch=batch)
        b_iter = ()
    else:
        b_iter = range(batch)
    # PRIMUS_V4_CSA_GATHER_LATE=1: the CSA dKV entry lists + gathers of all samples are issued on the dKV stream
    # after the last sample's pp kernel instead of next to the following samples' pp kernels (which they slow down
    # by more than they take). Same kernels and inputs -> same bits.
    late = None
    if (
        band is not None
        and "dkv_stream" in side
        and os.environ.get("PRIMUS_V4_CSA_GATHER_LATE", "0") == "1"
        and _bwd_takes_dkv_defer()
    ):
        late = []
        side["dkv_defer"] = late
    for b in b_iter:
        r = slice(b * S, (b + 1) * S)
        k = slice(b * nkv, (b + 1) * nkv)
        bwd(
            q[r],
            kv[k],
            o[r],
            do[r],
            topk_local if shared else topk_local[r],
            lse[r],
            attn_sink=None,  # dsink below, once over the batch
            kv_lora_rank=kv_lora_rank,
            scale=float(scale),
            use_swa=_use_band_kernels(),
            **({"band_pool": band} if band is not None else {}),
            **side,
            dq_out=dq[r],
            dkv_out=dkv[k],
            delta_out=delta[r],
        )
    if late:
        st = side["dkv_stream"]
        st.wait_stream(torch.cuda.current_stream(q.device))
        with torch.cuda.stream(st):
            for fn in late:
                side["dkv_keep"].extend(fn())
    dsink = None
    if attn_sink is not None:
        tgt = _dsink_kwargs().get("dsink_main_grad")
        dsink = _dsink(attn_sink, lse, delta, tgt)
    return dq, dkv, dsink
