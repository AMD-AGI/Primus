###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 attention kernels — single entry point for every backend.

``DeepseekV4Attention`` imports all attention entries from here, so this module
is the one place that maps a backend to its functional entry. Naming:

* dense (cr=0) / HCA (cr=128) entry: ``v4_attention_<backend>``
* CSA (cr=4) entry:                  ``v4_csa_attention_<backend>``

Backends:

* ``eager``  — pure-Python reference (:mod:`_eager`): ``eager_v4_attention`` /
  ``eager_v4_csa_attention``.
* ``v0``     — Triton, DEPRECATED gathered CSA (:mod:`_triton_v0_deprecated`):
  ``v4_csa_attention_v0`` (cr=4 only, ~30-260x slower; not used in production).
* ``v1``     — Triton production, separate K/V (:mod:`_triton_v1`):
  ``v4_attention_v1`` (dense/HCA) / ``v4_csa_attention_v1`` (pool CSA).
* ``v2``     — Triton fused single-latent sparse-MLA (:mod:`_triton_v2`,
  ``tl.dot`` / MFMA): ``v4_attention_v2`` / ``v4_csa_attention_v2``.
* ``gluon``  — hand-tuned gfx950 fused single-latent sparse-MLA (:mod:`_gluon_dsa`):
  ``v4_attention_gluon`` / ``v4_csa_attention_gluon``.

The eager references share exactly one definition with the kernels + unit tests
and keep the checkpoint-reproduction baseline bit-identical at the call sites.
"""

from primus.backends.megatron.core.transformer.v4_attention_kernels._eager import (
    eager_v4_attention,
    eager_v4_csa_attention,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v0_deprecated import (
    V4CSAAttentionFn,
    v4_csa_attention_v0,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v1 import (
    V4AttentionFn,
    V4CSAPoolAttentionFn,
    v4_attention_v1,
    v4_csa_attention_v1,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels.v4_csa_attention_gluon import (
    v4_attention_gluon,
    v4_csa_attention_gluon,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels.v4_csa_attention_triton import (
    v4_attention_v2,
    v4_csa_attention_v2,
)

__all__ = [
    # eager reference
    "eager_v4_attention",
    "eager_v4_csa_attention",
    # triton v0 (deprecated gathered CSA)
    "v4_csa_attention_v0",
    "V4CSAAttentionFn",
    # triton v1 (production, separate K/V)
    "v4_attention_v1",
    "v4_csa_attention_v1",
    "V4AttentionFn",
    "V4CSAPoolAttentionFn",
    # triton v2 (fused single-latent sparse-MLA)
    "v4_attention_v2",
    "v4_csa_attention_v2",
    # gluon (fused single-latent sparse-MLA, gfx950)
    "v4_attention_gluon",
    "v4_csa_attention_gluon",
]
