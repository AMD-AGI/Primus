###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.megatron.checkpoint.native_loader_llama import (
    derive_megatron_fields,
)


def test_llama3_rope_scaling_is_preserved_for_teacher_conversion():
    fields = derive_megatron_fields(
        {
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 16,
            "max_position_embeddings": 131072,
            "intermediate_size": 8192,
            "rope_theta": 500000.0,
            "rope_scaling": {"rope_type": "llama3", "factor": 32.0},
        }
    )
    assert fields["use_rope_scaling"] is True
    assert fields["rope_scaling_factor"] == 32.0


def test_unscaled_llama_keeps_megatron_rope_scaling_disabled():
    fields = derive_megatron_fields(
        {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "max_position_embeddings": 4096,
            "intermediate_size": 11008,
        }
    )
    assert fields["use_rope_scaling"] is False
