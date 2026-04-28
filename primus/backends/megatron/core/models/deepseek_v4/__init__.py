###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 model package.

Phase 5 surface:

    DeepseekV4Model               # top-level model
    DeepseekV4TransformerBlock    # decoder block (HC + hybrid attention + V4 MoE)
    DeepseekV4MTPBlock            # MTP head (separate HyperHead per MTP layer)
    deepseek_v4_builder           # builder used by model_provider
    model_provider                # Megatron pretrain() entry point

Phase 6 (deferred) will: wire MTP loss path, PP / TP / EP integration,
and skip the stock-TransformerBlock allocation in ``GPTModel.__init__``.
"""

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_block import (
    DeepseekV4TransformerBlock,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_builders import (
    deepseek_v4_builder,
    model_provider,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_model import (
    DeepseekV4Model,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_mtp import (
    DeepseekV4MTPBlock,
)

__all__ = [
    "DeepseekV4Model",
    "DeepseekV4TransformerBlock",
    "DeepseekV4MTPBlock",
    "deepseek_v4_builder",
    "model_provider",
]
