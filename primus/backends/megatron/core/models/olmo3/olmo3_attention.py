###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 self-attention: QK RMSNorm over the full projection width.

HF ``Olmo3Attention`` applies ``q_norm = RMSNorm(num_heads * head_dim)`` to the
whole ``q_proj`` output (and ``k_norm`` to ``k_proj``) before the head split and
RoPE. The RMS statistic therefore spans every head. Under tensor parallelism
each rank holds ``num_heads / tp`` heads, so the per-rank sum of squares is
all-reduced over the TP group (autograd-aware) before normalising; the norm
weight is sharded with the heads (``partition_dim=0``), so its gradient needs no
TP reduction (unlike Megatron's per-head ``q_layernorm``, which is replicated).
"""

from __future__ import annotations

import torch
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint


class Olmo3FullWidthRMSNorm(MegatronModule):
    """RMSNorm over ``[..., heads, head_dim]`` jointly (all heads), TP-sharded along heads."""

    def __init__(self, config, local_heads: int, head_dim: int, total_width: int, eps: float, tp_group):
        super().__init__(config=config)
        self.local_heads, self.head_dim, self.total_width, self.eps = local_heads, head_dim, total_width, eps
        self.tp_group = tp_group
        self.weight = torch.nn.Parameter(
            torch.ones(
                local_heads * head_dim,
                dtype=config.params_dtype,
                device="cpu" if config.use_cpu_initialization else torch.cuda.current_device(),
            )
        )
        set_tensor_model_parallel_attributes(self.weight, is_parallel=True, dim=0, stride=1)
        setattr(self.weight, "sequence_parallel", False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [s, b, local_heads, head_dim]
        x32 = x.float()
        sumsq = x32.pow(2).sum(dim=(-2, -1), keepdim=True)
        if self.tp_group is not None and self.tp_group.size() > 1:
            sumsq = torch.distributed.nn.functional.all_reduce(sumsq, group=self.tp_group)
        y = x32 * torch.rsqrt(sumsq / self.total_width + self.eps)
        y = self.weight.float().view(self.local_heads, self.head_dim) * y
        return y.to(x.dtype).contiguous()

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(state_dict, prefix, {"weight": 0}, sharded_offsets)


class Olmo3SelfAttention(SelfAttention):
    """``SelfAttention`` with OLMo-3's full-width ``q_norm`` / ``k_norm``."""

    def __init__(self, config, submodules, layer_number, **kwargs):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, **kwargs)
        tp_group = self.pg_collection.tp
        hd = self.hidden_size_per_attention_head
        self.q_norm = Olmo3FullWidthRMSNorm(
            config,
            self.num_attention_heads_per_partition,
            hd,
            config.num_attention_heads * hd,
            config.layernorm_epsilon,
            tp_group,
        )
        self.k_norm = Olmo3FullWidthRMSNorm(
            config,
            self.num_query_groups_per_partition,
            hd,
            config.num_query_groups * hd,
            config.layernorm_epsilon,
            tp_group,
        )

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True
    ):
        assert (
            split_qkv and not output_gate
        ), "OLMo-3 attention needs split q/k/v (no fused qkv-rope, no gate)"
        assert (
            self.config.num_query_groups >= self.world_size
        ), "OLMo-3 full-width QK norm needs tp <= kv heads"
        query, key, value = super().get_query_key_value_tensors(
            hidden_states, key_value_states, output_gate=False, split_qkv=True
        )
        return self.q_norm(query), self.k_norm(key), value.contiguous()
