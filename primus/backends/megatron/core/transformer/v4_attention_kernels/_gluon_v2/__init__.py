###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gluon DeepSeek-V4 sparse-MLA attention backend ("gluon v2").

Second-generation Gluon (gfx950 / CDNA4) sparse-MLA backend, using the same fused
single-latent (K == V) sparse-MLA representation and public API as the other V4
backends. Both directions are explicit-layout Gluon kernels:

* forward (``dsa_fwd_v4_gluon``): MFMA layouts, padded/swizzled shared, async
  double-buffered pipeline, rope-skip, exp2 softmax, MFMA K=32.
* backward (``dsa_bwd_v4_gluon``): Gluon dQ + dKV-intermediate kernels (rope-skip,
  MFMA K=32, single-chunk dQ RMW) + Triton Delta preprocess + CSR inverted-topk gather.

The Gluon kernels need a Gluon-capable (recompiled) triton whose CDNA4 async_copy
accepts general offset layouts, and raise a clear install hint otherwise.

* :func:`sparse_mla_fwd_v4_gluon_v2` -> ``(o, lse)``
* :func:`sparse_mla_bwd_v4_gluon_v2` -> ``(dq, dkv, d_sink)``
"""

from .dsa_bwd_v4_gluon import sparse_mla_bwd_v4_gluon_v2
from .dsa_fwd_v4_gluon import sparse_mla_fwd_v4_gluon_v2

__all__ = [
    "sparse_mla_fwd_v4_gluon_v2",
    "sparse_mla_bwd_v4_gluon_v2",
]
