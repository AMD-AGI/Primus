###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""V4 attention via the native FlyDSL sparse-MLA backend ("flydsl_v1").

Thin binding of the kernel-agnostic V4 adapters (:mod:`v4_sparse_mla_adapter`)
to the ``sparse_mla_{fwd,bwd}_v4_flydsl`` kernel pair (``_flydsl_v1``): the fused
single-latent (K == V) sparse-MLA path implemented in native FlyDSL MFMA
(``rocdl.mfma_*``) over a per-token top-k gather. The forward is fully native
FlyDSL; the backward uses a native FlyDSL dQ kernel plus the shared Triton dKV
intermediate/scatter-gather. Numerically equivalent to the eager V4 references.

Depends only on the installed ``flydsl`` pip package (gfx950 / CDNA4); it is
therefore loaded LAZILY (see ``load_flydsl_attention_backends``), never at
package import time.
"""

from __future__ import annotations

from primus.backends.megatron.core.transformer.v4_attention_kernels._flydsl_v1 import (
    sparse_mla_bwd_v4_flydsl,
    sparse_mla_fwd_v4_flydsl,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels.v4_sparse_mla_adapter import (
    make_attention,
    make_csa_from_pool,
)

v4_csa_attention_flydsl = make_csa_from_pool(sparse_mla_fwd_v4_flydsl, sparse_mla_bwd_v4_flydsl)
v4_attention_flydsl = make_attention(sparse_mla_fwd_v4_flydsl, sparse_mla_bwd_v4_flydsl)

__all__ = [
    "v4_csa_attention_flydsl",
    "v4_attention_flydsl",
]
