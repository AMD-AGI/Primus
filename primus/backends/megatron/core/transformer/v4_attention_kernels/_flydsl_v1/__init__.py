###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL-v2 DeepSeek-V4 sparse-MLA attention backend (MFMA, gfx950 / CDNA4).

A fresh re-implementation that mirrors the gluon backend (``_gluon_dsa``): same
fused single-latent (K == V) sparse-MLA representation and the **same public
kernel-pair API**, so it plugs straight into the kernel-agnostic V4 adapter
(:mod:`v4_sparse_mla_adapter`) with zero adapter changes:

* ``q``  : ``[T, H, d_qk]``     bf16  (``d_qk = kv_lora_rank + rope_rank``)
* ``kv`` : ``[T, 1, d_qk]``     bf16  (single MQA latent; ``V = K[:kv_lora_rank]``)
* ``topk_indices`` : ``[T, TOPK]`` int32 (SWA window ++ sparse pool, -1 = invalid)
* ``attn_sink`` : ``[H]`` fp32 optional per-head softmax sink

Unlike the legacy wired FlyDSL CSA path (scalarized GEMV, no MFMA), the v2
**forward** uses FlyDSL MFMA (``rocdl.mfma_*`` matrix cores) over a top-k gather.

* :func:`sparse_mla_fwd_v4_flydsl` -> ``(o, lse)``   (native FlyDSL MFMA)
* :func:`sparse_mla_bwd_v4_flydsl` -> ``(dq, dkv, d_sink)``

NOTE: the v2 backward currently reuses the shared, proven pure-Triton non-atomic
chunked-gather backward (also MFMA via ``tl.dot``); a FlyDSL-native MFMA backward
is a follow-up. This keeps the autograd path correct end-to-end today.
"""

from .dsa_bwd_v4_flydsl import sparse_mla_bwd_v4_flydsl
from .dsa_fwd_v4_flydsl import sparse_mla_fwd_v4_flydsl

__all__ = [
    "sparse_mla_fwd_v4_flydsl",
    "sparse_mla_bwd_v4_flydsl",
]
