###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron MoE Patches

This module contains patches for Megatron's Mixture-of-Experts (MoE) components:
    - Deprecated MoE layer implementations
    - Primus TopKRouter
    - MoE permutation fusion with Transformer Engine
"""

from primus.backends.megatron.patches.moe_patches import (
    deprecated_layer_patches,
    permute_fusion_patches,
    topk_router_patches,
)

__all__ = [
    "deprecated_layer_patches",
    "topk_router_patches",
    "permute_fusion_patches",
]
