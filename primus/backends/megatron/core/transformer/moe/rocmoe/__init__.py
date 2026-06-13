###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ROCMoE-backed Megatron MoE layer (drop-in replacement for MoELayer)."""

from primus.backends.megatron.core.transformer.moe.rocmoe.moe_layer import ROCMoELayer

__all__ = ["ROCMoELayer"]
