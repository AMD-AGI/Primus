###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest

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


@pytest.mark.parametrize(
    "rope_scaling, match",
    [
        ({"rope_type": "linear", "factor": 2.0}, "'linear' has no Megatron conversion"),
        ({"type": "yarn", "factor": 4.0}, "'yarn' has no Megatron conversion"),
        ({"rope_type": "llama3", "factor": 8.0, "high_freq_factor": 2.0}, "'high_freq_factor': 2.0"),
        ({"rope_type": "llama3", "low_freq_factor": 1.0}, "no factor"),
        ({"factor": 8.0}, "no rope_type"),
    ],
)
def test_rope_scaling_megatron_cannot_reproduce_is_rejected(rope_scaling, match):
    cfg = {
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_hidden_layers": 16,
        "max_position_embeddings": 131072,
        "intermediate_size": 8192,
        "rope_scaling": rope_scaling,
    }
    with pytest.raises(ValueError, match=match):
        derive_megatron_fields(cfg)
    standard = {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0}
    cfg["rope_scaling"] = {**standard, "original_max_position_embeddings": 8192}
    assert derive_megatron_fields(cfg)["rope_scaling_factor"] == 32.0
