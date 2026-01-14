from typing import Optional
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.profiler_spec import ModuleProfilerSpec
from primus.core.projection.training_config import TrainingConfig

from .utils import benchmark_layer


class MoEMLPProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        self.module = None  # Will be set during benchmarking
        self._cached_results = None  # Cache for (forward_time, backward_time, activation_memory)
        self._cache_key = None  # Cache key (batch_size, seq_len)

    def set_module(self, module):
        """Set the actual MoE MLP module for benchmarking."""
        self.module = module
        # Invalidate cache when module changes
        self._cached_results = None
        self._cache_key = None

    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        if self.config.model_config.moe_ffn_hidden_size is not None:
            moe_ffn = self.config.model_config.moe_ffn_hidden_size
        else:
            moe_ffn = self.config.model_config.ffn_hidden_size

        # For SwiGLU: 3 projections per expert (gate, up, down)
        # For standard FFN: 2 projections per expert (up, down)
        num_ffn_projections = 3 if self.config.model_config.swiglu else 2
        per_expert_params = num_ffn_projections * self.config.model_config.hidden_size * moe_ffn
        ep = 1 if rank is None else self.config.model_parallel_config.expert_model_parallel_size

        all_experts_params = self.config.model_config.num_experts * per_expert_params // ep

        # Shared experts (if any)
        shared_sz = 0
        if self.config.model_config.moe_shared_expert_intermediate_size is not None:
            shared_sz = self.config.model_config.moe_shared_expert_intermediate_size
        shared_params = num_ffn_projections * self.config.model_config.hidden_size * shared_sz

        return all_experts_params + shared_params

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        num_tokens = (
            batch_size
            * seq_len
            // self.config.model_parallel_config.tensor_model_parallel_size
            // self.config.model_parallel_config.context_model_parallel_size
        )
        topk_tokens = num_tokens * self.config.model_config.moe_router_topk

        if self.config.model_config.moe_ffn_hidden_size is not None:
            moe_ffn = self.config.model_config.moe_ffn_hidden_size
        else:
            moe_ffn = self.config.model_config.ffn_hidden_size

        if self.config.model_config.swiglu:
            # Need to store both gate and up projections for backward
            intermediate_memory = 2 * topk_tokens * moe_ffn * 2  # bf16
        else:
            intermediate_memory = topk_tokens * moe_ffn * 2  # bf16

        # After activation
        activation_memory = topk_tokens * moe_ffn * 2  # bf16
        output_memory = topk_tokens * self.config.model_config.hidden_size * 2  # bf16
        total = intermediate_memory + activation_memory + output_memory
        if self.config.model_config.moe_shared_expert_intermediate_size is not None:
            if self.config.model_config.swiglu:
                # Need to store both gate and up projections for backward
                intermediate_memory = 2 * num_tokens * moe_ffn * 2  # bf16
            else:
                intermediate_memory = num_tokens * moe_ffn * 2  # bf16

            # After activation
            activation_memory = num_tokens * moe_ffn * 2  # bf16
            output_memory = num_tokens * self.config.model_config.hidden_size * 2  # bf16
            total += intermediate_memory + activation_memory + output_memory

        return total

    def _get_benchmark_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        """Get or compute benchmark results (cached)."""
        cache_key = (batch_size, seq_len)
        if self._cached_results is None or self._cache_key != cache_key:
            self._cached_results = benchmark_layer(
                self.module,
                [(seq_len, batch_size, self.config.model_config.hidden_size)],
            )
            self._cache_key = cache_key
        return self._cached_results

    def measured_forward_time(self, batch_size: int, seq_len: int) -> float:
        forward_time, _, _ = self._get_benchmark_results(batch_size, seq_len)
        return forward_time

    def measured_backward_time(self, batch_size: int, seq_len: int) -> float:
        _, backward_time, _ = self._get_benchmark_results(batch_size, seq_len)
        return backward_time

    def measured_activation_memory(self, batch_size: int, seq_len: int) -> int:
        _, _, activation_memory = self._get_benchmark_results(batch_size, seq_len)
        return activation_memory


def get_moe_mlp_profiler_spec(config: TrainingConfig) -> ModuleProfilerSpec:
    return ModuleProfilerSpec(
        profiler=MoEMLPProfiler,
        config=config,
        sub_profiler_specs=None,
    )
