###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 model package (text backbone, a.k.a. Kimi-Linear).

Public surface:

    KimiK3TransformerConfig          # config dataclass (MLATransformerConfig subclass)
    normalize_linear_attention_freq  # per-layer KDA / full-attention pattern normalizer
    KimiK3Model                      # top-level model (LanguageModule)
    kimi_k3_builder                  # builder used by model_provider
    model_provider                   # Megatron pretrain() entry point

The layer / block classes, the spec tree and the MTP spec
(``kimi_k3_block``, ``kimi_k3_layer_specs``, ``kimi_k3_mtp_specs``) are
**not** re-exported here. Importing them pulls in the NoPE MLA module and
therefore transformer_engine, which the KDA tests deliberately avoid; import
them by module path instead.
"""

from primus.backends.megatron.core.models.kimi_k3.kimi_k3_builders import (
    kimi_k3_builder,
    model_provider,
)
from primus.backends.megatron.core.models.kimi_k3.kimi_k3_model import KimiK3Model
from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
    KimiK3TransformerConfig,
    normalize_linear_attention_freq,
)

__all__ = [
    "KimiK3TransformerConfig",
    "normalize_linear_attention_freq",
    "KimiK3Model",
    "kimi_k3_builder",
    "model_provider",
]
