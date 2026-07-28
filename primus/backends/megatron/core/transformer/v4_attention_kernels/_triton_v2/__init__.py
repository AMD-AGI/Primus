###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plain-Triton DeepSeek-V4 sparse-MLA attention backend ("triton v2").

Same fused single-latent (K == V) sparse-MLA representation and public API as
the gluon backend, but written in vanilla Triton so the QK / PV GEMMs lower to
MFMA via ``tl.dot``. Distinct from the separate-KV Triton CSA/dense backends
(``_triton_v0_deprecated`` gathered / ``_triton_v1`` pool), which keep K and V separate.

* :func:`sparse_mla_fwd_v4_triton` -> ``(o, lse)``
* :func:`sparse_mla_bwd_v4_triton` -> ``(dq, dkv, d_sink)``
"""

from .dsa_bwd_v4_triton import sparse_mla_bwd_v4_triton
from .dsa_fwd_v4_triton import sparse_mla_fwd_v4_triton

__all__ = [
    "sparse_mla_fwd_v4_triton",
    "sparse_mla_bwd_v4_triton",
]
