###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
SFT (Supervised Fine-Tuning) module for Megatron-LM.

This module contains SFT-specific components:
- forward_step: Forward step function with loss masking
- schema / formatters / preprocessing: Megatron-local SFT abstraction helpers
"""

from primus.backends.megatron.sft.formatters import (
    AlpacaFormatter,
    ChatMLFormatter,
    ConversationFormatter,
    OpenAIMessagesFormatter,
    create_formatter,
)
from primus.backends.megatron.sft.dataset import (
    SFTDataset,
    build_train_valid_test_datasets,
)
from primus.backends.megatron.sft.forward_step import create_sft_forward_step
from primus.backends.megatron.sft.preprocessing import (
    load_jsonl_file,
    normalize_sft_sample,
    tokenize_formatted_sft_sample,
    tokenize_text,
)
from primus.backends.megatron.sft.runtime import (
    create_sft_datasets_provider,
    run_sft_pretrain,
)
from primus.backends.megatron.sft.schema import (
    CharSpan,
    FormattedSFTSample,
    SFTMessage,
    SFTSample,
    TextSegment,
    collapse_messages_to_single_turn,
)

__all__ = [
    "AlpacaFormatter",
    "CharSpan",
    "ChatMLFormatter",
    "ConversationFormatter",
    "FormattedSFTSample",
    "OpenAIMessagesFormatter",
    "SFTDataset",
    "SFTMessage",
    "SFTSample",
    "TextSegment",
    "build_train_valid_test_datasets",
    "collapse_messages_to_single_turn",
    "create_sft_datasets_provider",
    "create_formatter",
    "create_sft_forward_step",
    "load_jsonl_file",
    "normalize_sft_sample",
    "run_sft_pretrain",
    "tokenize_formatted_sft_sample",
    "tokenize_text",
]
