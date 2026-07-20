###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton **v1** attention backend (production, separate K/V).

The current production Triton attention kernels + autograd Functions:

* dense (cr=0) / HCA (cr=128): :func:`v4_attention_v1` / :class:`V4AttentionFn`
  (``v4_attention_fwd`` / ``v4_attention_bwd`` launchers).
* CSA (cr=4), in-kernel pool gather + scatter-add:
  :func:`v4_csa_attention_v1` / :class:`V4CSAPoolAttentionFn`
  (``v4_csa_attention_fwd`` / ``v4_csa_attention_bwd`` pool launchers).

See ``_triton_v0_deprecated`` for the deprecated gathered CSA path and ``_triton_v2`` for
the fused single-latent sparse-MLA path.
"""

from .v4_attention import V4AttentionFn, v4_attention_v1
from .v4_csa_attention import V4CSAPoolAttentionFn, v4_csa_attention_v1

__all__ = [
    "v4_attention_v1",
    "V4AttentionFn",
    "v4_csa_attention_v1",
    "V4CSAPoolAttentionFn",
]
