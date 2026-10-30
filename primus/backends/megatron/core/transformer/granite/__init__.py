###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""IBM Granite (dense, ``GraniteForCausalLM``) transformer pieces."""

from primus.backends.megatron.core.transformer.granite.granite_transformer_layer import (
    GraniteTransformerLayer,
    scaled_bias_dropout_add,
)

__all__ = ["GraniteTransformerLayer", "scaled_bias_dropout_add"]
