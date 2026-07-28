###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""V4 attention via the Gluon sparse-MLA backend (gfx950).

Thin binding of the kernel-agnostic V4 adapters (:mod:`v4_sparse_mla_adapter`)
to the gluon ``sparse_mla_{fwd,bwd}_v4_gluon`` kernel pair (``_gluon_dsa``).
See the adapter module for the full V4 <-> sparse-MLA mapping (zero rope pad,
``[local ++ pool]`` kv buffer, ``[SWA window ++ pool]`` topk, grad mapping).
Numerically equivalent to :func:`eager_v4_csa_attention` / :func:`eager_v4_attention`.
"""

from __future__ import annotations

from primus.backends.megatron.core.transformer.v4_attention_kernels._gluon_dsa import (
    sparse_mla_bwd_v4_gluon,
    sparse_mla_fwd_v4_gluon,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels.v4_sparse_mla_adapter import (
    make_attention,
    make_csa_from_pool,
)

v4_csa_attention_gluon = make_csa_from_pool(sparse_mla_fwd_v4_gluon, sparse_mla_bwd_v4_gluon)
v4_attention_gluon = make_attention(sparse_mla_fwd_v4_gluon, sparse_mla_bwd_v4_gluon)

__all__ = [
    "v4_csa_attention_gluon",
    "v4_attention_gluon",
]
