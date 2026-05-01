###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 model package.

Plan-2 P16 surface:

    DeepseekV4Model               # top-level model (LanguageModule)
    DeepseekV4TransformerBlock    # decoder block (TransformerBlock subclass)
    DeepseekV4HybridLayer         # decoder layer (TransformerLayer subclass)
    DeepseekV4MTPBlock            # legacy V4 MTP head (deprecated, P21 removal)
    get_v4_mtp_block_spec         # spec helper for upstream MultiTokenPredictionBlock
    deepseek_v4_builder           # builder used by model_provider
    model_provider                # Megatron pretrain() entry point
"""

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_block import (
    DeepseekV4HybridLayer,
    DeepseekV4HybridLayerSubmodules,
    DeepseekV4TransformerBlock,
    DeepseekV4TransformerBlockSubmodules,
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
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_mtp_specs import (
    get_v4_mtp_block_spec,
)

__all__ = [
    "DeepseekV4Model",
    "DeepseekV4TransformerBlock",
    "DeepseekV4TransformerBlockSubmodules",
    "DeepseekV4HybridLayer",
    "DeepseekV4HybridLayerSubmodules",
    "DeepseekV4MTPBlock",
    "get_v4_mtp_block_spec",
    "deepseek_v4_builder",
    "model_provider",
]
