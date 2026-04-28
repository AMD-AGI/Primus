###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 model package (Phase 2 stub).

Phase 2 surface:

    DeepseekV4Model               # thin subclass of GPTModel
    deepseek_v4_builder           # builder used by model_provider
    model_provider                # Megatron pretrain() entry point

Phase 3 will add ``DeepseekV4TransformerBlock`` and the V4 layer-spec
helpers; Phase 4 will plug HC + hybrid attention into the block.
"""

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_builders import (
    deepseek_v4_builder,
    model_provider,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_model import (
    DeepseekV4Model,
)

__all__ = [
    "DeepseekV4Model",
    "deepseek_v4_builder",
    "model_provider",
]
