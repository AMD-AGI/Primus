###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""CPU tests: the native converters read the RoPE base also when transformers v5
re-saved the checkpoint (base only under ``rope_parameters``), and read every
config with a top-level ``rope_theta`` exactly as before."""

import pytest

pytest.importorskip("torch")

from primus.backends.megatron.checkpoint import (  # noqa: E402
    native_convert_common as common,
)
from primus.backends.megatron.checkpoint import (  # noqa: E402
    native_loader_llama as llama,
)
from primus.backends.megatron.checkpoint import (  # noqa: E402
    native_loader_qwen3 as qwen3,
)

LLAMA31 = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 32,
    "max_position_embeddings": 131072,
    "intermediate_size": 14336,
    "rms_norm_eps": 1e-5,
}
QWEN3 = dict(LLAMA31, head_dim=128, max_position_embeddings=40960)
V5_ROPE = {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0}


@pytest.mark.parametrize(
    "cfg,default,expected",
    [
        ({"rope_theta": 500000.0}, 10000.0, 500000.0),  # top level: as before
        ({"rope_theta": 500000.0, "rope_parameters": {"rope_theta": 1.0}}, 10000.0, 500000.0),
        ({"rope_parameters": V5_ROPE}, 10000.0, 500000.0),  # transformers v5 re-save
        ({"rope_theta": None, "rope_parameters": V5_ROPE}, 10000.0, 500000.0),
        ({"rope_scaling": {"rope_type": "llama3", "rope_theta": 5e5}}, 10000.0, 5e5),
        ({}, 10000.0, 10000.0),  # none given: the family default, as before
        ({"rope_parameters": None, "rope_scaling": {"factor": 8.0}}, 1e6, 1e6),
    ],
)
def test_hf_rope_theta(cfg, default, expected):
    assert common.hf_rope_theta(cfg, default) == expected


def test_llama_v5_resave_keeps_its_base():
    """Llama 3.1 re-saved by transformers v5: base 500000 only under rope_parameters."""
    assert llama.derive_megatron_fields(dict(LLAMA31, rope_parameters=V5_ROPE))["rotary_base"] == 500000.0
    assert llama.derive_megatron_fields(dict(LLAMA31, rope_theta=500000.0))["rotary_base"] == 500000.0
    assert llama.derive_megatron_fields(LLAMA31)["rotary_base"] == 10000.0


def test_qwen3_v5_resave_keeps_its_base():
    rope = {"rope_type": "default", "rope_theta": 5000000.0}
    assert qwen3.derive_megatron_fields(dict(QWEN3, rope_parameters=rope))["rotary_base"] == 5000000.0
    assert qwen3.derive_megatron_fields(dict(QWEN3, rope_theta=1e6))["rotary_base"] == 1e6
    assert qwen3.derive_megatron_fields(QWEN3)["rotary_base"] == 1000000.0
