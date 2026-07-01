###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL-v2 sparse-MLA forward (native FlyDSL MFMA) — kernel in progress.

Target API (mirrors :func:`sparse_mla_fwd_v4_gluon`):

    sparse_mla_fwd_v4_flydsl(q, kv, topk_indices, attn_sink=None,
                             kv_lora_rank=512, scale=None) -> (o, lse)

Design (the per-token [H, TOPK] attention, MFMA tiled):

  * Grid: one workgroup per query token ``t`` (program_id 0). The "M" axis of
    the MFMA is the HEAD axis (H rows), the "N" axis is the gathered key axis
    (TOPK), and the contraction "K" axis is ``d_qk`` (tiled by 16 for the gfx950
    ``mfma_f32_32x32x16_bf16`` instruction).
  * For each TOPK tile: gather the ``TILE_K`` latent rows ``kv[topk[t, j]]``
    (``-1`` -> masked), MFMA ``S[H, TILE_K] = Q[H, d_qk] @ K[TILE_K, d_qk]^T``,
    online softmax over the key axis, MFMA ``acc[H, D_V] += P[H, TILE_K] @
    V[TILE_K, D_V]`` (``V = K[:, :kv_lora_rank]``; single MQA latent, K == V).
  * Epilogue: fold the per-head ``attn_sink`` into the denominator (V4), divide,
    write ``O[t, H, D_V]`` and sink-inclusive ``LSE[t, H]``.

This reuses the proven MFMA lane-layout machinery of the in-tree FlyDSL SWA
forward (``_flydsl/kernels/v4_sla_fwd_kernel.py``, ``mfma_f32_32x32x16_bf16``,
the wave/lane decomposition and the v16f32 accumulator fragment), re-targeted
from "contiguous window load" to "top-k gather" and from BHLD-per-seq to
token-major-per-head, and written with FlyDSL 0.2.2 ``const_expr`` guards from
the start. It is a substantial low-level MLIR kernel and is being implemented +
validated against :func:`eager_v4_csa_attention` incrementally.
"""

from __future__ import annotations


def sparse_mla_fwd_v4_flydsl(q, kv, topk_indices, attn_sink=None, kv_lora_rank=512, scale=None):
    raise NotImplementedError(
        "FlyDSL-v2 native MFMA forward kernel is under construction. The package "
        "scaffold, public API and backward (shared MFMA chunked-gather) are in "
        "place; the FlyDSL MLIR forward kernel (top-k gather + MFMA QK/PV + sink) "
        "is the remaining work. See this module's docstring for the design."
    )


__all__ = ["sparse_mla_fwd_v4_flydsl"]
