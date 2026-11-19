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
    )
