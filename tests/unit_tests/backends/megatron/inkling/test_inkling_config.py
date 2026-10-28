###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling presets and :class:`InklingTransformerConfig`."""

from __future__ import annotations

import pytest

from primus.backends.megatron.core.models.inkling.inkling_builders import (
    inkling_config_from_model_yaml,
)
from primus.backends.megatron.core.models.inkling.inkling_transformer_config import (
    InklingTransformerConfig,
    default_local_layer_ids,
)
from tests.unit_tests.backends.megatron.inkling.inkling_test_utils import (
    load_converter,
    preset,
    release_config,
)


def test_small_preset_matches_released_config():
    config = inkling_config_from_model_yaml(preset("inkling_small"))
    assert isinstance(config, InklingTransformerConfig)
    load_converter().check_preset(config, release_config()["text_config"])


def test_small_layer_patterns():
    config = inkling_config_from_model_yaml(preset("inkling_small"))
    assert config.local_layer_ids == release_config()["text_config"]["local_layer_ids"]
    assert [i for i in range(42) if not config.is_sliding_layer(i)] == [5, 11, 17, 23, 29, 35, 41]
    assert config.moe_layer_pattern() == [0, 0] + [1] * 40
    assert config.moe_shared_expert_intermediate_size == 2 * 2048
    assert config.layernorm_epsilon == 1e-6
    assert config.gated_linear_unit


def test_large_preset_matches_released_config():
    config = inkling_config_from_model_yaml(preset("inkling_large"))
    text_config = release_config("inkling_large")["text_config"]
    load_converter().check_preset(config, text_config)
    assert config.local_layer_ids == text_config["local_layer_ids"]
    assert [i for i in range(66) if not config.is_sliding_layer(i)] == list(range(5, 66, 6))
    assert config.moe_layer_pattern() == [0, 0] + [1] * 64
    sliding, full = config.attention_config(0), config.attention_config(5)
    assert (sliding.num_attention_heads, sliding.num_query_groups) == (64, 16)
    assert (full.num_attention_heads, full.num_query_groups) == (64, 8)

    proxy = inkling_config_from_model_yaml(preset("inkling_large_6L"))
    assert proxy.num_layers == 6 and proxy.hidden_size == 6144
    assert [i for i in range(6) if not proxy.is_sliding_layer(i)] == [5]
    assert proxy.moe_layer_pattern() == [0, 0, 1, 1, 1, 1]


def test_proxy_and_tiny_patterns():
    proxy = inkling_config_from_model_yaml(preset("inkling_small_12L"))
    assert proxy.num_layers == 12 and proxy.hidden_size == 4096
    assert [i for i in range(12) if not proxy.is_sliding_layer(i)] == [5, 11]
    assert proxy.moe_layer_pattern() == [0, 0] + [1] * 10

    tiny = inkling_config_from_model_yaml(preset("inkling_small_tiny"))
    assert [tiny.is_sliding_layer(i) for i in range(2)] == [True, False]
    assert tiny.moe_layer_pattern() == [0, 1]
    assert tiny.moe_shared_expert_intermediate_size == 2 * 128


def test_attention_config_swaps_geometry():
    tiny = inkling_config_from_model_yaml(preset("inkling_small_tiny"))
    sliding, full = tiny.attention_config(0), tiny.attention_config(1)
    assert (sliding.num_attention_heads, sliding.num_query_groups) == (4, 4)
    assert (full.num_attention_heads, full.num_query_groups) == (4, 2)
    assert sliding.softmax_scale == full.softmax_scale == 1 / 64
    assert tiny.attention_geometry(0)["rel_extent"] == tiny.sliding_window_size == 32
    assert tiny.attention_geometry(1)["rel_extent"] == tiny.rel_extent == 64
    assert tiny.num_query_groups == 2 and tiny.softmax_scale is None


def test_default_local_layer_ids():
    assert default_local_layer_ids(12) == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"moe_router_score_function": "softmax"}, "sigmoid"),
        ({"moe_shared_expert_intermediate_size": 100}, "moe_shared_expert_intermediate_size"),
        ({"local_layer_ids": [0, 7]}, "out of range"),
        ({"swa_num_query_groups": 3}, "not divisible"),
        ({"inkling_attention_backend": "sdpa"}, "inkling_attention_backend"),
        ({"moe_num_shared_experts": 0, "moe_shared_expert_intermediate_size": None}, "shared expert"),
    ],
)
def test_invalid_configs(overrides, message):
    with pytest.raises(ValueError, match=message):
        inkling_config_from_model_yaml(preset("inkling_small_tiny"), **overrides)
