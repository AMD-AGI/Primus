###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

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
        self._gemm_backend = None  # Optional: GEMM simulation backend
        self._sdpa_backend = None  # Optional: SDPA simulation backend

    def get_sub_profiler(self, name: str):
        return self.sub_profilers.get(name)

    def set_simulation_backends(self, gemm_backend=None, sdpa_backend=None):
        """Set simulation backends and propagate to sub-profilers."""
        self._gemm_backend = gemm_backend
        self._sdpa_backend = sdpa_backend
        # Propagate to sub-profilers
        if "self_attention" in self.sub_profilers:
            attn = self.sub_profilers["self_attention"]
            if gemm_backend is not None and hasattr(attn, "set_gemm_backend"):
                attn.set_gemm_backend(gemm_backend)
            if sdpa_backend is not None and hasattr(attn, "set_sdpa_backend"):
                attn.set_sdpa_backend(sdpa_backend)
        if "mlp" in self.sub_profilers:
            mlp = self.sub_profilers["mlp"]
            if gemm_backend is not None and hasattr(mlp, "set_gemm_backend"):
                mlp.set_gemm_backend(gemm_backend)
        # Invalidate cache
        self._cached_results = None
        self._cache_key = None

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

    def _get_simulated_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        """Aggregate simulated results from sub-profilers."""
        attn_fwd = self.sub_profilers["self_attention"].measured_forward_time(batch_size, seq_len)
        attn_bwd = self.sub_profilers["self_attention"].measured_backward_time(batch_size, seq_len)
        mlp_fwd = self.sub_profilers["mlp"].measured_forward_time(batch_size, seq_len)
        mlp_bwd = self.sub_profilers["mlp"].measured_backward_time(batch_size, seq_len)
        fwd_time = attn_fwd + mlp_fwd
        bwd_time = attn_bwd + mlp_bwd
        activation_memory = self.estimated_activation_memory(batch_size, seq_len)
        return (fwd_time, bwd_time, activation_memory)

    def _get_benchmark_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        """Get or compute benchmark results (cached)."""
        cache_key = (batch_size, seq_len)
        if self._cached_results is None or self._cache_key != cache_key:
            if self._gemm_backend is not None or self._sdpa_backend is not None:
                # Use simulation mode
                self._cached_results = self._get_simulated_results(batch_size, seq_len)
            else:
                # Get TransformerConfig from the layer module itself (has fp8 setting)
                transformer_config = getattr(self.layer_module, "config", None)
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
        self._gemm_backend = None  # Optional: GEMM simulation backend
        self._sdpa_backend = None  # Optional: SDPA simulation backend

    def get_sub_profiler(self, name: str):
        return self.sub_profilers.get(name)

    def set_simulation_backends(self, gemm_backend=None, sdpa_backend=None):
        """Set simulation backends and propagate to sub-profilers."""
        self._gemm_backend = gemm_backend
        self._sdpa_backend = sdpa_backend
        # Propagate to sub-profilers
        if "self_attention" in self.sub_profilers:
            attn = self.sub_profilers["self_attention"]
            if gemm_backend is not None and hasattr(attn, "set_gemm_backend"):
                attn.set_gemm_backend(gemm_backend)
            if sdpa_backend is not None and hasattr(attn, "set_sdpa_backend"):
                attn.set_sdpa_backend(sdpa_backend)
        if "mlp" in self.sub_profilers:
            mlp = self.sub_profilers["mlp"]
            if gemm_backend is not None and hasattr(mlp, "set_gemm_backend"):
                mlp.set_gemm_backend(gemm_backend)
        # Invalidate cache
        self._cached_results = None
        self._cache_key = None

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

    def _get_simulated_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        """Aggregate simulated results from sub-profilers."""
        attn_fwd = self.sub_profilers["self_attention"].measured_forward_time(batch_size, seq_len)
        attn_bwd = self.sub_profilers["self_attention"].measured_backward_time(batch_size, seq_len)
        mlp_fwd = self.sub_profilers["mlp"].measured_forward_time(batch_size, seq_len)
        mlp_bwd = self.sub_profilers["mlp"].measured_backward_time(batch_size, seq_len)
        fwd_time = attn_fwd + mlp_fwd
        bwd_time = attn_bwd + mlp_bwd
        activation_memory = self.estimated_activation_memory(batch_size, seq_len)
        return (fwd_time, bwd_time, activation_memory)

    def _get_benchmark_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        """Get or compute benchmark results (cached)."""
        cache_key = (batch_size, seq_len)
        if self._cached_results is None or self._cache_key != cache_key:
            if self._gemm_backend is not None or self._sdpa_backend is not None:
                # Use simulation mode
                self._cached_results = self._get_simulated_results(batch_size, seq_len)
            else:
                # Get TransformerConfig from the layer module itself (has fp8 setting)
                transformer_config = getattr(self.layer_module, "config", None)
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


def get_dense_transformer_layer_profiler_spec(
    config: TrainingConfig,
) -> "ModuleProfilerSpec":
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


def get_moe_transformer_layer_profiler_spec(
    config: TrainingConfig,
) -> "ModuleProfilerSpec":
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
