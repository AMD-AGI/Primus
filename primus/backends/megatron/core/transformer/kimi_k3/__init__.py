###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 transformer modules.

Public surface for the new mixers the Kimi K3 text backbone introduces.
Kimi Delta Attention (KDA) is the linear-attention mixer used by 69 of
the 93 decoder layers; the remaining 24 use NoPE MLA with an output gate.

The compute kernels behind KDA — the eager reference, the ``fla`` fused
Triton path and (later) a FlyDSL kernel — are dispatched from
:mod:`.kda_kernels`.
"""

from primus.backends.megatron.core.transformer.kimi_k3.kimi_delta_attention import (
    KimiDeltaAttention,
    KimiDeltaAttentionSubmodules,
    KimiGatedRMSNorm,
)

__all__ = [
    "KimiDeltaAttention",
    "KimiDeltaAttentionSubmodules",
    "KimiGatedRMSNorm",
]
