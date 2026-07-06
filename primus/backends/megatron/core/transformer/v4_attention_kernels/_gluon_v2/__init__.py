###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gluon DeepSeek-V4 sparse-MLA attention backend ("gluon v2").

Second-generation Gluon (gfx950 / CDNA4) sparse-MLA backend, using the same fused
single-latent (K == V) sparse-MLA representation and public API as the other V4
backends. The forward is an explicit-layout Gluon kernel (MFMA layouts, padded/
swizzled shared, async double-buffered pipeline, rope-skip, exp2 softmax, MFMA K=32);
see ``dsa_fwd_v4_gluon``. The backward is currently the plain-Triton chunked-gather
kernel (``dsa_bwd_v4_triton``) and is being migrated to Gluon (``dsa_bwd_v4_gluon``).

The Gluon forward needs a Gluon-capable (recompiled) triton whose CDNA4 async_copy
accepts general offset layouts, and raises a clear install hint otherwise.

* :func:`sparse_mla_fwd_v4_gluon_v2` -> ``(o, lse)``
* :func:`sparse_mla_bwd_v4_gluon_v2` -> ``(dq, dkv, d_sink)``
"""

import os

from .dsa_bwd_v4_gluon import sparse_mla_bwd_v4_gluon_v2_gluon as _bwd_v2_gluon
from .dsa_bwd_v4_triton import sparse_mla_bwd_v4_gluon_v2 as _bwd_v2_triton
from .dsa_fwd_v4_gluon import sparse_mla_fwd_v4_gluon_v2


def sparse_mla_bwd_v4_gluon_v2(*args, **kwargs):
    """gluon_v2 backward dispatcher.

    Default = the **Gluon** chunked-gather backward (``dsa_bwd_v4_gluon``), which beats the
    plain-Triton backward at all 6 flash/pro x cr{0,4,128} shapes (bwd geomean +11.3%, pro cr4
    +25.4%; eager-UT 9/9). ``PRIMUS_DSA_V2_BWD=triton`` selects the plain-Triton backward
    (``dsa_bwd_v4_triton``) as an escape hatch.
    """
    if os.environ.get("PRIMUS_DSA_V2_BWD", "gluon").lower() == "triton":
        return _bwd_v2_triton(*args, **kwargs)
    return _bwd_v2_gluon(*args, **kwargs)


__all__ = [
    "sparse_mla_fwd_v4_gluon_v2",
    "sparse_mla_bwd_v4_gluon_v2",
]
