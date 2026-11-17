###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native converter family: SmolLM3 (HuggingFaceTB/SmolLM3-3B and -3B-Base).

Llama layout with tied embeddings and NoPE on every ``no_rope_layer_interval``-th
layer; converted by ``native_loader_smollm3`` and built as Megatron's GPTModel.
"""

from primus.backends.megatron.checkpoint.native_registry import (
    NativeFamily,
    register_native_family,
)

register_native_family(
    NativeFamily(
        name="smollm3",
        label="SmolLM3",
        converter="primus.backends.megatron.checkpoint.native_loader_smollm3",
        architectures=("SmolLM3ForCausalLM",),
        model_types=("smollm3",),
    )
)
