###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
PEFT (Parameter-Efficient Fine-Tuning) module for Megatron-LM.

This module provides LoRA and other PEFT methods for efficient fine-tuning
of large language models with Megatron-LM.

Ported from Megatron-Bridge with modifications for Primus compatibility.
"""

from primus.backends.megatron.peft.base import PEFT
from primus.backends.megatron.peft.lora import LoRA, LoRAMerge, VLMLoRA

__all__ = [
    "PEFT",
    "LoRA",
    "LoRAMerge",
    "VLMLoRA",
]
