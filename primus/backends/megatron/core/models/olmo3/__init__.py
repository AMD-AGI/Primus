###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 (``model_type: olmo3``) model package.

OLMo-3 (allenai/Olmo-3-1025-7B) is a dense decoder that differs from Llama in four places:

* post-sublayer norms: ``h = x + norm(attn(x))``, ``out = h + norm(mlp(h))`` (no input / pre-MLP norm);
* QK RMSNorm over the full projection width (all heads together, 4096), not per head;
* YaRN RoPE (factor 8, original 8192, mscale 0.1*ln(8)+1) on every layer;
* sliding-window attention (4096 keys) on three of every four layers; every fourth layer is full.
"""

from primus.backends.megatron.core.models.olmo3.olmo3_builders import (
    model_provider,
    olmo3_builder,
)
from primus.backends.megatron.core.models.olmo3.olmo3_transformer_config import (
    Olmo3TransformerConfig,
)

__all__ = ["Olmo3TransformerConfig", "model_provider", "olmo3_builder"]
