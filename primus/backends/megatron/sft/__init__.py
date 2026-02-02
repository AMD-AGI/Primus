###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
SFT (Supervised Fine-Tuning) module for Megatron-LM.

This module contains SFT-specific components:
- forward_step: Forward step function with loss masking
- gpt_sft_chat_dataset: Chat dataset for instruction tuning
"""

from primus.backends.megatron.sft.forward_step import create_sft_forward_step

__all__ = [
    "create_sft_forward_step",
]
