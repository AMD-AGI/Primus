###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.projection import param_counts
from primus.core.projection.training_config import (
    ModelConfig,
    ModelParallelConfig,
    RuntimeConfig,
    TrainingConfig,
)

H, FFN, VOCAB, LAYERS, EXPERTS = 256, 512, 1000, 4, 8


def _config(**mp):
    model = ModelConfig(
        num_layers=LAYERS,
        hidden_size=H,
        ffn_hidden_size=FFN,
        moe_ffn_hidden_size=FFN,
        padded_vocab_size=VOCAB,
        num_attention_heads=8,
        kv_channels=32,
        num_query_groups=8,
        swiglu=True,
        num_experts=EXPERTS,
        moe_router_topk=2,
        moe_pattern=[1] * LAYERS,
    )
    return TrainingConfig(
        model_config=model,
        runtime_config=RuntimeConfig(global_batch_size=8, micro_batch_size=1, sequence_length=128),
        model_parallel_config=ModelParallelConfig(**mp),
    )


def test_expert_dp_is_one_when_ep_spans_dp():
    assert param_counts.expert_dp_size(_config(expert_model_parallel_size=8), 8) == 1
    assert param_counts.expert_dp_size(_config(expert_model_parallel_size=4), 8) == 2


def test_local_params_shard_experts_by_ep_and_count_both_vocab_matrices():
    tc = _config(expert_model_parallel_size=4)
    local = param_counts.local_params(tc)
    attn = 2 * H * 8 * 32 + 2 * H * 8 * 32
    assert local.dense == LAYERS * (attn + H * EXPERTS) + 2 * VOCAB * H
    assert local.expert == LAYERS * (EXPERTS // 4) * 3 * H * FFN


def test_distributed_optimizer_shards_experts_only_by_expert_dp():
    tc = _config(expert_model_parallel_size=8, use_distributed_optimizer=True)
    local = param_counts.local_params(tc)
    assert param_counts.optimizer_params(tc, 8) == local.dense // 8 + local.expert


def test_grads_are_bf16_unless_accumulated_in_fp32():
    assert param_counts.grad_bytes_per_param(_config()) == 2
    assert param_counts.grad_bytes_per_param(_config(accumulate_allreduce_grads_in_fp32=True)) == 4
