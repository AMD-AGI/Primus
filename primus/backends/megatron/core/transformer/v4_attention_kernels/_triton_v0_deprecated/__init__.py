###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton **v0** attention backend — DEPRECATED gathered CSA.

The original ``compress_ratio == 4`` CSA path that consumes a pre-gathered
``[B, Sq, K_topk, head_dim]`` tensor. It is ~30-260x slower than the v1 pool
path (see ``deepseek-v4/develop/perf/attention_perf.md``) and is NOT used by the
production dispatch. Kept for reference / correctness tests only.

Prefer :mod:`.._triton_v1` (pool CSA + dense/HCA) or :mod:`.._triton_v2`
(fused single-latent sparse-MLA).
"""

from .v4_csa_attention import V4CSAAttentionFn, v4_csa_attention_v0

__all__ = [
    "v4_csa_attention_v0",
    "V4CSAAttentionFn",
]
