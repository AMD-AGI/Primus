###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron SFT (Supervised Fine-Tuning) utilities.

This module provides SFT-specific components for native Megatron-LM:
    - SFTDataset: Dataset class for SFT data (JSON/JSONL/HuggingFace)
    - train_valid_test_sft_datasets_provider: Dataset provider for pretrain()
    - forward_step: Forward step with answer-only loss masking
"""

from primus.backends.megatron.sft.sft_utils import (
    SFTDataset,
    forward_step,
    get_batch,
    train_valid_test_sft_datasets_provider,
)

__all__ = [
    "SFTDataset",
    "train_valid_test_sft_datasets_provider",
    "get_batch",
    "forward_step",
]
