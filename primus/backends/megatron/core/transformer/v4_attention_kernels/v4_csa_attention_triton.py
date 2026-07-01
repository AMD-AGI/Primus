###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""V4 attention via the plain-Triton sparse-MLA backend ("triton_v2").

The ``_v2`` denotes the second *kernel* implementation of the V4 sparse-MLA
attention (NOT a new Triton release). Thin binding of the kernel-agnostic V4
adapters (:mod:`v4_sparse_mla_adapter`) to the ``sparse_mla_{fwd,bwd}_v4_triton``
kernel pair (``_triton_dsa``) — the fused single-latent (K == V) sparse-MLA path
whose GEMMs lower to MFMA via ``tl.dot``. Unlike the gluon backend (gfx950 /
CDNA4 hardware-controlled layouts), this is vanilla Triton and runs on any
MFMA-capable arch (gfx942 / gfx950). Distinct from the in-tree separate-KV CSA
Triton backend (``v4_csa_attention``). Numerically equivalent to the eager V4
references.
"""

from __future__ import annotations

from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_dsa import (
    sparse_mla_bwd_v4_triton,
    sparse_mla_fwd_v4_triton,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels.v4_sparse_mla_adapter import (
    make_attention,
    make_csa_from_pool,
)

v4_csa_attention_triton_v2_from_pool = make_csa_from_pool(sparse_mla_fwd_v4_triton, sparse_mla_bwd_v4_triton)
v4_attention_triton_v2 = make_attention(sparse_mla_fwd_v4_triton, sparse_mla_bwd_v4_triton)

__all__ = [
    "v4_csa_attention_triton_v2_from_pool",
    "v4_attention_triton_v2",
]
