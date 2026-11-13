###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native converter family: OLMo-3 (HF ``Olmo3ForCausalLM``, ``model_type: olmo3``).

The model is built with Primus' ``olmo3`` model_type (post-sublayer norms, full-width QK-norm,
per-layer sliding window / YaRN RoPE). The ``prepare`` step passes the fine-tune's
``seq_length``, ``olmo3_rope_layout`` and vocabulary padding (``make_vocab_size_divisible_by``
times TP) to the converter and keys the converted checkpoint on the padded vocabulary size.
"""

from primus.backends.megatron.checkpoint.native_registry import (
    NativeFamily,
    register_native_family,
)

register_native_family(
    NativeFamily(
        name="olmo3",
        label="OLMo-3",
        converter="primus.backends.megatron.checkpoint.native_loader_olmo3",
        architectures=("Olmo3ForCausalLM",),
        model_types=("olmo3",),
        model_type="olmo3",
        dtype_from_config="primus.backends.megatron.checkpoint.olmo3_convert_hooks:dtype_from_config",
        prepare="primus.backends.megatron.checkpoint.olmo3_convert_hooks:hook_prepare",
    )
)
