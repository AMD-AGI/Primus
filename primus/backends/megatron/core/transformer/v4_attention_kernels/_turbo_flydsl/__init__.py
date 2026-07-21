###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 sparse-MLA kernel-pair via the **Primus-Turbo** public API.

Unlike the other in-tree fused single-latent backends (``_gluon_v2`` /
``_triton_v2`` / ``_flydsl_v1``), which vendor their kernels inside Primus, this
backend calls straight into the installed **``primus_turbo``** package — the
native-FlyDSL sparse-MLA attention under ``primus_turbo.flydsl.attention``. This
is the "turbo API" integration: Primus owns only the thin V4 adapter binding; the
kernels live in Primus-Turbo.

Public kernel-pair API (identical to the other sparse-MLA backends):

* ``sparse_mla_fwd_v4_turbo_flydsl(q, kv, topk, attn_sink=None,
  kv_lora_rank=512, scale=None) -> (o, lse)``
* ``sparse_mla_bwd_v4_turbo_flydsl(q, kv, o, do, topk, lse, attn_sink=None,
  kv_lora_rank=512, scale=None) -> (dq, dkv, d_sink)``

The upstream module layout / function names changed between Primus-Turbo
releases, so we bind to whichever is present (the call signatures are identical):

* **new** (>= the ``flydsl_attn`` refactor, e.g. commit ``edc8d2c``):
  ``primus_turbo.flydsl.attention.sparse_mla_fwd.sparse_mla_fwd_flydsl`` /
  ``...sparse_mla_bwd.sparse_mla_bwd_flydsl`` (flat modules, no ``_v4`` infix).
* **old** (<= ``0.3.2.dev*``):
  ``primus_turbo.flydsl.attention.kernels.sparse_mla_v2.{sparse_mla_fwd_v4_flydsl,
  sparse_mla_bwd_v4_flydsl}``.

``primus_turbo`` (with the flydsl sparse-MLA attention) and the ``flydsl`` pip
package are required; the import fails with a clear message otherwise (handled by
the lazy loader in :mod:`..` / :func:`load_turbo_attention_backends`).
"""

from __future__ import annotations

try:
    # New Primus-Turbo layout: flat modules, function names without the _v4 infix.
    from primus_turbo.flydsl.attention.sparse_mla_bwd import (
        sparse_mla_bwd_flydsl as _sparse_mla_bwd,
    )
    from primus_turbo.flydsl.attention.sparse_mla_fwd import (
        sparse_mla_fwd_flydsl as _sparse_mla_fwd,
    )
except ImportError:
    # Older Primus-Turbo layout: kernels.sparse_mla_v2 with the _v4 infix.
    from primus_turbo.flydsl.attention.kernels.sparse_mla_v2 import (
        sparse_mla_bwd_v4_flydsl as _sparse_mla_bwd,
    )
    from primus_turbo.flydsl.attention.kernels.sparse_mla_v2 import (
        sparse_mla_fwd_v4_flydsl as _sparse_mla_fwd,
    )

# Turbo-suffixed re-exports so the V4 wrapper / benchmark can bind them without
# clashing with the in-tree ``_flydsl_v1`` (which has identically-named kernels).
sparse_mla_fwd_v4_turbo_flydsl = _sparse_mla_fwd
sparse_mla_bwd_v4_turbo_flydsl = _sparse_mla_bwd

__all__ = ["sparse_mla_fwd_v4_turbo_flydsl", "sparse_mla_bwd_v4_turbo_flydsl"]
