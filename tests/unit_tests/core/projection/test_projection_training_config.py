###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for projection TrainingConfig conversion helpers.

Covers ``update_config_from_args`` and ``megatron_derive_default_args`` —
in particular the ``moe_pattern`` derivation (dense / int / list / str forms)
and derived defaults (kv_channels, data_parallel_size, vpp). Pure CPU, no deps.
"""

import argparse

import pytest

from primus.core.projection.training_config import (
    ModelConfig,
    ModelParallelConfig,
    RuntimeConfig,
    megatron_derive_default_args,
    update_config_from_args,
)


def test_dataclass_defaults():
    assert ModelConfig().num_layers == 0
    # Dense default: num_experts is 0 (falsy) so MoE math is guarded.
    assert not ModelConfig().num_experts
    assert RuntimeConfig().micro_batch_size == 1
    assert ModelParallelConfig().tensor_model_parallel_size == 1


def test_update_config_from_args_copies_matching_fields_only():
    args = argparse.Namespace(
        global_batch_size=16,
        micro_batch_size=2,
        sequence_length=2048,
        data_parallel_size=4,
        unrelated_field="ignored",
    )
    rc = update_config_from_args(RuntimeConfig(), args)
    assert rc.global_batch_size == 16
    assert rc.micro_batch_size == 2
    assert rc.sequence_length == 2048
    assert rc.data_parallel_size == 4
    assert not hasattr(rc, "unrelated_field")


def _make_megatron_args(**overrides):
    base = dict(
        hidden_size=512,
        num_attention_heads=8,
        kv_channels=None,
        group_query_attention=False,
        num_query_groups=0,
        data_parallel_size=None,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        num_layers_per_virtual_pipeline_stage=None,
        untie_embeddings_and_output_weights=False,
        num_experts=0,
        moe_layer_freq=1,
        seq_length=2048,
        padded_vocab_size=None,
        num_layers=4,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _single_node_env(monkeypatch):
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("GPUS_PER_NODE", "8")


def test_derive_kv_channels_and_dp_size():
    args = megatron_derive_default_args(_make_megatron_args())
    assert args.kv_channels == 512 // 8  # derived from hidden / heads
    # world_size = 1 * 8; dp = 8 / (tp1 * pp1 * cp1)
    assert args.data_parallel_size == 8
    assert args.virtual_pipeline_model_parallel_size == 1


def test_moe_pattern_dense_is_all_zero():
    args = megatron_derive_default_args(_make_megatron_args(num_experts=None, num_layers=4))
    assert args.moe_pattern == [0, 0, 0, 0]


def test_moe_pattern_int_freq_one_is_all_moe():
    args = megatron_derive_default_args(_make_megatron_args(num_experts=8, num_layers=4, moe_layer_freq=1))
    assert args.moe_pattern == [1, 1, 1, 1]


def test_moe_pattern_list_used_as_is():
    args = megatron_derive_default_args(
        _make_megatron_args(num_experts=8, num_layers=4, moe_layer_freq=[0, 1, 1, 1])
    )
    assert args.moe_pattern == [0, 1, 1, 1]


def test_moe_pattern_str_expression_is_evaluated():
    args = megatron_derive_default_args(
        _make_megatron_args(num_experts=8, num_layers=4, moe_layer_freq="[0]*1+[1]*3")
    )
    assert args.moe_pattern == [0, 1, 1, 1]


def test_padded_vocab_size_defaulted_when_missing():
    args = megatron_derive_default_args(_make_megatron_args(padded_vocab_size=None))
    assert args.padded_vocab_size == 100352
