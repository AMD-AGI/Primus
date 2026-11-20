###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native converter family: Mistral Large 4 (``mistral_large4_text`` text decoder).

The release's ``config.json`` names ``MistralLarge4ForConditionalGeneration`` /
``mistral_large4`` at the top level (the text decoder's ``mistral_large4_text`` is
its ``text_config.model_type``) and gives its dtype as a top-level ``dtype``, not
``torch_dtype`` (read by the
``dtype_from_config`` hook). The converter dequantises the FP8 consolidated weights and builds
the model with the ``mistral_large4`` model provider.
"""

from primus.backends.megatron.checkpoint.native_registry import (
    NativeFamily,
    register_native_family,
)

register_native_family(
    NativeFamily(
        name="mistral_large4",
        label="Mistral Large 4 (text)",
        converter="primus.backends.megatron.checkpoint.native_loader_mistral_large4_text",
        architectures=("MistralLarge4ForConditionalGeneration",),
        model_types=("mistral_large4", "mistral_large4_text"),
        model_type="mistral_large4",
        dtype_from_config="primus.backends.megatron.checkpoint.native_loader_mistral_large4_text:hf_dtype",
    )
)
