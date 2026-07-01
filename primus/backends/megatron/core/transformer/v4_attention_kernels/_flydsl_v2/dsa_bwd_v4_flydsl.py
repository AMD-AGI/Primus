###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL-v2 sparse-MLA backward.

For now this reuses the shared, proven pure-Triton non-atomic chunked-gather
backward (``sparse_mla_bwd_v4_triton``: ``tl.dot`` / MFMA dQ + dKV-intermediate
+ CSR inverted-topk gather). It already operates on exactly the sparse-MLA
contract (q[T,H,d_qk], single latent kv[T,1,d_qk], topk, sink) and returns
``(dq, dkv, d_sink)``, so it is backend-neutral. A FlyDSL-native MFMA backward
can replace this later without touching the adapter or the forward.
"""

from .._triton_dsa.dsa_bwd_v4_triton import (
    sparse_mla_bwd_v4_triton as sparse_mla_bwd_v4_flydsl,
)

__all__ = ["sparse_mla_bwd_v4_flydsl"]
