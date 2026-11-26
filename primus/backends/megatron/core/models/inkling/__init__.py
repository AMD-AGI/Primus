###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling (Thinking Machines) text backbone on Megatron.

Public surface:

    InklingTransformerConfig   # config dataclass (TransformerConfig subclass)
    InklingModel               # top-level model (LanguageModule)
    build_inkling_model        # args-free builder (``meta=True`` for shape-only builds)
    inkling_builder            # builder used by model_provider
    model_provider             # Megatron pretrain() entry point

The released-checkpoint weight mapping lives in ``inkling_hf_mapping``.
"""

from primus.backends.megatron.core.models.inkling.inkling_builders import (
    build_inkling_model,
    inkling_builder,
    model_provider,
)
from primus.backends.megatron.core.models.inkling.inkling_model import InklingModel
from primus.backends.megatron.core.models.inkling.inkling_transformer_config import (
    InklingTransformerConfig,
)

__all__ = [
    "InklingTransformerConfig",
    "InklingModel",
    "build_inkling_model",
    "inkling_builder",
    "model_provider",
]
