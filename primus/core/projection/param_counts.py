###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Parameters one rank holds, split the way Megatron's data-parallel groups split them.

Dense (non-expert) parameters are replicated over the full data-parallel group; expert
parameters only over the expert data-parallel group, ``dp * cp * tp / (etp * ep)``. With
expert parallelism filling the data-parallel group that group is one rank, and expert
parameters are neither reduced nor sharded by the distributed optimizer.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalParams:
    dense: int
    expert: int

    @property
    def total(self) -> int:
        return self.dense + self.expert


def _mp(training_config, name, default=1):
    return getattr(training_config.model_parallel_config, name, None) or default


def attention_params(model_config) -> int:
    """Q, K, V and O weights of one layer (grouped-query aware)."""
    h = model_config.hidden_size
    heads = model_config.num_attention_heads
    kv_ch = getattr(model_config, "kv_channels", None) or h // max(1, heads)
    groups = getattr(model_config, "num_query_groups", None) or heads
    return 2 * h * heads * kv_ch + 2 * h * groups * kv_ch


def expert_dp_size(training_config, dp: int) -> int:
    tp = _mp(training_config, "tensor_model_parallel_size")
    cp = _mp(training_config, "context_model_parallel_size")
    ep = _mp(training_config, "expert_model_parallel_size")
    etp = _mp(training_config, "expert_tensor_parallel_size", tp)
    return max(1, (dp * cp * tp) // max(1, etp * ep))


def local_params(training_config) -> LocalParams:
    """Parameters on the most loaded pipeline stage of one rank, after TP/PP/EP sharding."""
    m = training_config.model_config
    tp = _mp(training_config, "tensor_model_parallel_size")
    pp = _mp(training_config, "pipeline_model_parallel_size")
    ep = _mp(training_config, "expert_model_parallel_size")
    etp = _mp(training_config, "expert_tensor_parallel_size", tp)
    h = m.hidden_size
    ffn = m.ffn_hidden_size or 4 * h
    moe_ffn = m.moe_ffn_hidden_size or ffn
    mlp_mats = 3 if getattr(m, "swiglu", True) else 2
    pattern = list(m.moe_pattern or [0] * m.num_layers)
    layers = (m.num_layers + pp - 1) // pp
    moe_layers = round(sum(1 for p in pattern if p == 1) * layers / max(1, m.num_layers))
    dense_layers = layers - moe_layers
    shared = getattr(m, "moe_shared_expert_intermediate_size", 0) or 0
    experts = m.num_experts or 0

    dense = layers * attention_params(m) // tp + dense_layers * mlp_mats * h * ffn // tp
    dense += moe_layers * (h * experts + mlp_mats * h * shared // tp)
    vocab = getattr(m, "padded_vocab_size", 0) or 0
    tied = getattr(m, "share_embeddings_and_output_weights", False)
    # The first stage holds the embedding and the last the output layer; at PP 1 one rank holds both.
    dense += vocab * h // tp * (2 if (pp == 1 and not tied) else 1)
    expert = moe_layers * (experts // max(1, ep)) * mlp_mats * h * moe_ffn // max(1, etp)
    return LocalParams(int(dense), int(expert))


def optimizer_params(training_config, dp: int) -> int:
    """Parameters one rank's optimizer steps: its shard of each group under the distributed optimizer."""
    p = local_params(training_config)
    mp = training_config.model_parallel_config
    if not (getattr(mp, "use_distributed_optimizer", False) or getattr(mp, "use_torch_fsdp2", False)):
        return p.total
    return p.dense // max(1, dp) + p.expert // expert_dp_size(training_config, dp)


def grad_bytes_per_param(training_config) -> int:
    return (
        4
        if getattr(training_config.model_parallel_config, "accumulate_allreduce_grads_in_fp32", False)
        else 2
    )
