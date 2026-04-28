###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 model package.

Phase 3 surface (current):

    DeepseekV4Model               # top-level model
    DeepseekV4TransformerBlock    # decoder block (P4 will plug in HC)
    deepseek_v4_builder           # builder used by model_provider
    model_provider                # Megatron pretrain() entry point

Phase 4+ will add: HyperConnection / Compressor / Indexer / SWA / AttnSink /
DualRoPE / CSA / HCA modules under ``core/transformer`` siblings, and a
V4-specific MoE submodule under this package.
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

__all__ = [
    "DeepseekV4Model",
    "DeepseekV4TransformerBlock",
    "deepseek_v4_builder",
    "model_provider",
]
