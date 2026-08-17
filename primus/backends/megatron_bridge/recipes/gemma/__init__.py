###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gemma 4 (26B MoE and 31B Dense) recipe extensions for Megatron-Bridge."""

from .gemma4 import (
    gemma4_26b_finetune_config,
    gemma4_26b_pretrain_config,
    gemma4_31b_finetune_config,
    gemma4_31b_pretrain_config,
)

__all__ = [
    "gemma4_26b_pretrain_config",
    "gemma4_31b_pretrain_config",
    "gemma4_26b_finetune_config",
    "gemma4_31b_finetune_config",
]
