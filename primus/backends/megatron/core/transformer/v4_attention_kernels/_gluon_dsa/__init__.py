###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gluon (hardware-controlled Triton) DeepSeek-V4 sparse-MLA attention backend.

Ported from ROCm/aiter PR #2922 (``aiter/ops/triton/_gluon_kernels/gfx950``)
for gfx950 / CDNA4 (MI350/MI355X). These kernels operate on the **sparse-MLA
latent** representation used by the DeepSeek V4 paper / FlashMLA:

* ``q``  : ``[T, H, d_qk]``  with ``d_qk = kv_lora_rank (512) + rope_rank (64)``
* ``kv`` : ``[T, 1, d_qk]``  single MQA latent (K and V share it; ``V_lora`` is
           the first ``kv_lora_rank`` channels of ``K_lora``)
* ``topk_indices`` : ``[T, TOPK]`` int32 absolute KV-token indices (SWA window +
           sparse top-k already concatenated by the caller; ``-1`` = invalid)
* ``attn_sink`` : ``[H]`` fp32 optional per-head learnable softmax sink

This is a different (latent + per-token-topk) representation than the in-tree
CSA path (``v4_csa_attention_v0``: ``q / k_local / v_local / gathered /
sparse_mask``); it is exposed here as a standalone ``gluon`` backend.

Public API mirrors aiter's ``sparse_mla_fwd_v4`` / ``sparse_mla_bwd_v4`` with
``backend="gluon"``:

* :func:`sparse_mla_fwd_v4_gluon` -> ``(o, lse)``
* :func:`sparse_mla_bwd_v4_gluon` -> ``(dq, dkv, d_sink)``
"""

from .dsa_bwd_v4_gluon import sparse_mla_bwd_v4_gluon
from .dsa_fwd_v4_gluon import sparse_mla_fwd_v4_gluon

__all__ = [
    "sparse_mla_fwd_v4_gluon",
    "sparse_mla_bwd_v4_gluon",
]
