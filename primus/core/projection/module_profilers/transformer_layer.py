from typing import Optional

from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.profiler_spec import ModuleProfilerSpec
from primus.core.projection.training_config import TrainingConfig

from .attention import AttentionProfiler
from .dense_mlp import DenseMLPProfiler
from .layer_norm import LayerNormProfiler
from .moe_mlp import MoEMLPProfiler
from .residual_add import ResidualAddProfiler
from .router import RouterProfiler
from .utils import benchmark_layer

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Transformer Layer Data Flow
#
#             +----------------+
#             |     Input      |
#             +----------------+
#                     | ----------------------
#        +-------------------------+         |
#        |     Input LayerNorm     |         |
#        +-------------------------+         |
#                     |                      |
#        +------------------------+          |
#        |     Self-Attention     |          |
#        +------------------------+          |
#                     |                      |
#            +-----------------+             |
#            |     Dropout     |             |
#            +-----------------+             |
#                     |                      |
#                     o ---------------------|
#         +----------------------+
#         |     Residual Add     |
#         +----------------------+
#                     | ----------------------
#        +-------------------------+         |
#        |    Pre-mlp LayerNorm    |         |
#        +-------------------------+         |
#                     |                      |
#              +-------------+               |
#              |     MLP     |               |
#              +-------------+               |
#                     |                      |
#            +-----------------+             |
#            |     Dropout     |             |
#            +-----------------+             |
#                     |                      |
#                     o ---------------------|
#         +----------------------+
#         |     Residual Add     |
#         +----------------------+
#                     |
#             +----------------+
#             |     Output      |
#             +----------------+


class DenseTransformerLayerProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        self.layer_module = None  # Will be set during benchmarking
        self._cached_results = None  # Cache for (forward_time, backward_time, activation_memory)
        self._cache_key = None  # Cache key (batch_size, seq_len)

    def get_sub_profiler(self, name: str):
        return self.sub_profilers.get(name)

    def set_layer_module(self, layer_module):
        """Set the actual transformer layer module for benchmarking."""
        self.layer_module = layer_module
        self.sub_profilers["self_attention"].set_module(layer_module.self_attention)
        self.sub_profilers["mlp"].set_module(layer_module.mlp)

        # Invalidate cache when layer changes
        self._cached_results = None
        self._cache_key = None

    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        return (
            self.sub_profilers["layer_norm"].estimated_num_params(rank) * 3
            + self.sub_profilers["self_attention"].estimated_num_params(rank)
            + self.sub_profilers["mlp"].estimated_num_params(rank)
            + self.sub_profilers["residual_add"].estimated_num_params(rank) * 2
        )

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        return (
            self.sub_profilers["layer_norm"].estimated_activation_memory(batch_size, seq_len) * 3
            + self.sub_profilers["self_attention"].estimated_activation_memory(batch_size, seq_len)
            + self.sub_profilers["mlp"].estimated_activation_memory(batch_size, seq_len)
            + self.sub_profilers["residual_add"].estimated_activation_memory(batch_size, seq_len) * 2
        )

    def _get_benchmark_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        """Get or compute benchmark results (cached)."""
        cache_key = (batch_size, seq_len)
        if self._cached_results is None or self._cache_key != cache_key:
            # Get TransformerConfig from the layer module itself (has fp8 setting)
            transformer_config = getattr(self.layer_module, 'config', None)
            self._cached_results = benchmark_layer(
                self.layer_module,
                [(seq_len, batch_size, self.config.model_config.hidden_size)],
                transformer_config=transformer_config,
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


class MoETransformerLayerProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        self.layer_module = None  # Will be set during benchmarking
        self._cached_results = None  # Cache for (forward_time, backward_time, activation_memory)
        self._cache_key = None  # Cache key (batch_size, seq_len)

    def get_sub_profiler(self, name: str):
        return self.sub_profilers.get(name)

    def set_layer_module(self, layer_module):
        """Set the actual transformer layer module for benchmarking."""
        self.layer_module = layer_module
        self.sub_profilers["self_attention"].set_module(layer_module.self_attention)
        self.sub_profilers["mlp"].set_module(layer_module.mlp)

        # Invalidate cache when layer changes
        self._cached_results = None
        self._cache_key = None

    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        return (
            self.sub_profilers["layer_norm"].estimated_num_params(rank) * 3
            + self.sub_profilers["self_attention"].estimated_num_params(rank)
            + self.sub_profilers["mlp"].estimated_num_params(rank)
            + self.sub_profilers["router"].estimated_num_params(rank)
            + self.sub_profilers["residual_add"].estimated_num_params(rank) * 2
        )

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        return (
            self.sub_profilers["layer_norm"].estimated_activation_memory(batch_size, seq_len) * 3
            + self.sub_profilers["self_attention"].estimated_activation_memory(batch_size, seq_len)
            + self.sub_profilers["mlp"].estimated_activation_memory(batch_size, seq_len)
            + self.sub_profilers["router"].estimated_activation_memory(batch_size, seq_len)
            + self.sub_profilers["residual_add"].estimated_activation_memory(batch_size, seq_len) * 2
        )

    def _get_benchmark_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        """Get or compute benchmark results (cached)."""
        cache_key = (batch_size, seq_len)
        if self._cached_results is None or self._cache_key != cache_key:
            # Get TransformerConfig from the layer module itself (has fp8 setting)
            transformer_config = getattr(self.layer_module, 'config', None)
            self._cached_results = benchmark_layer(
                self.layer_module,
                [(seq_len, batch_size, self.config.model_config.hidden_size)],
                transformer_config=transformer_config,
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


def get_dense_transformer_layer_profiler_spec(config: TrainingConfig) -> "ModuleProfilerSpec":
    return ModuleProfilerSpec(
        profiler=DenseTransformerLayerProfiler,
        config=config,
        sub_profiler_specs={
            "layer_norm": LayerNormProfiler,
            "self_attention": AttentionProfiler,
            "residual_add": ResidualAddProfiler,
            "mlp": DenseMLPProfiler,
        },
    )


def get_moe_transformer_layer_profiler_spec(config: TrainingConfig) -> "ModuleProfilerSpec":
    return ModuleProfilerSpec(
        profiler=MoETransformerLayerProfiler,
        config=config,
        sub_profiler_specs={
            "layer_norm": LayerNormProfiler,
            "self_attention": AttentionProfiler,
            "residual_add": ResidualAddProfiler,
            "router": RouterProfiler,
            "mlp": MoEMLPProfiler,
        },
    )
