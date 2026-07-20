###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 sparse-MLA kernel-pair via the **Primus-Turbo** public API.

Unlike the other in-tree fused single-latent backends (``_gluon_v2`` /
``_triton_v2`` / ``_flydsl_v1``), which vendor their kernels inside Primus, this
backend calls straight into the installed **``primus_turbo``** package — the
native-FlyDSL sparse-MLA v2 attention
(``primus_turbo.flydsl.attention.kernels.sparse_mla_v2``) from the Primus-Turbo
``dev/kyle/flydsl_attn_deepseekv4`` line. This is the "turbo API" integration:
Primus owns only the thin V4 adapter binding; the kernels live in Primus-Turbo.

Public kernel-pair API (identical to the other sparse-MLA backends):

* ``sparse_mla_fwd_v4_turbo_flydsl(q, kv, topk, attn_sink=None,
  kv_lora_rank=512, scale=None) -> (o, lse)``
* ``sparse_mla_bwd_v4_turbo_flydsl(q, kv, o, do, topk, lse, attn_sink=None,
  kv_lora_rank=512, scale=None) -> (dq, dkv, d_sink)``

``primus_turbo`` (with the flydsl sparse-MLA attention) and the ``flydsl`` pip
package are required; the import fails with a clear message otherwise (handled by
the lazy loader in :mod:`..` / :func:`load_turbo_attention_backends`).
"""

from __future__ import annotations

from primus_turbo.flydsl.attention.kernels.sparse_mla_v2 import (
    sparse_mla_bwd_v4_flydsl,
    sparse_mla_fwd_v4_flydsl,
)

# Turbo-suffixed re-exports so the V4 wrapper / benchmark can bind them without
# clashing with the in-tree ``_flydsl_v1`` (which has identically-named kernels).
sparse_mla_fwd_v4_turbo_flydsl = sparse_mla_fwd_v4_flydsl
sparse_mla_bwd_v4_turbo_flydsl = sparse_mla_bwd_v4_flydsl

__all__ = ["sparse_mla_fwd_v4_turbo_flydsl", "sparse_mla_bwd_v4_turbo_flydsl"]
