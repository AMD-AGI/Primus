###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple

from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.profiler_spec import ModuleProfilerSpec
from primus.core.projection.training_config import TrainingConfig

from .utils import benchmark_layer


class DenseMLPProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        self.module = None  # Will be set during benchmarking
        self._cached_results = None  # Cache for (forward_time, backward_time, activation_memory)
        self._cache_key = None  # Cache key (batch_size, seq_len)

    def set_module(self, module):
        """Set the actual Dense MLP module for benchmarking."""
        self.module = module
        # Invalidate cache when module changes
        self._cached_results = None
        self._cache_key = None

    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        # For SwiGLU: 3 projections (gate, up, down)
        # For standard FFN: 2 projections (up, down)
        num_ffn_projections = 3 if self.config.model_config.swiglu else 2
        return (
            self.config.model_config.hidden_size
            * self.config.model_config.ffn_hidden_size
            * num_ffn_projections
        )

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        num_tokens = (
            batch_size
            * seq_len
            // self.config.model_parallel_config.tensor_model_parallel_size
            // self.config.model_parallel_config.context_model_parallel_size
        )
        # Memory after first projection(s)
        if self.config.model_config.swiglu:
            # Need to store both gate and up projections for backward
            intermediate_memory = 2 * num_tokens * self.config.model_config.ffn_hidden_size * 2  # bf16
        else:
            intermediate_memory = num_tokens * self.config.model_config.ffn_hidden_size * 2  # bf16

        # After activation
        activation_memory = num_tokens * self.config.model_config.ffn_hidden_size * 2  # bf16
        output_memory = num_tokens * self.config.model_config.hidden_size * 2  # bf16

        # Peak memory is input + intermediate (both needed for backward)
        return intermediate_memory + activation_memory + output_memory

    def _get_benchmark_results(self, batch_size: int, seq_len: int) -> Tuple[float, float, int]:
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


def get_dense_mlp_profiler_spec(config: TrainingConfig) -> ModuleProfilerSpec:
    return ModuleProfilerSpec(
        profiler=DenseMLPProfiler,
        config=config,
        sub_profiler_specs=None,
    )
