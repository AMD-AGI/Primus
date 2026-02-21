###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from typing import Optional

from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.profiler_spec import ModuleProfilerSpec
from primus.core.projection.training_config import TrainingConfig

from . import collective_model as cm
from .collective_args import get_default_args
from .attention import AttentionProfiler
from .dense_mlp import DenseMLPProfiler
from .layer_norm import LayerNormProfiler
from .moe_mlp import MoEMLPProfiler
from .residual_add import ResidualAddProfiler
from .router import RouterProfiler
from .utils import benchmark_layer


def _estimate_tp_allreduce_time_ms(config, batch_size: int, seq_len: int) -> float:
    """
    Estimate TP AllReduce time for a single AllReduce operation (in ms).

    In Megatron-style tensor parallelism each transformer layer performs
    2 AllReduces in forward (after attention row-parallel output projection,
    after MLP row-parallel down projection) and 2 in backward.
    With sequence parallelism the AllReduce is replaced by
    ReduceScatter + AllGather pairs, but the total data volume is equivalent.

    Returns 0.0 when TP <= 1 (no communication needed).
    """
    tp = config.model_parallel_config.tensor_model_parallel_size
    if tp <= 1:
        return 0.0

    cp = getattr(config.model_parallel_config, "context_parallel_size", 1) or 1
    pp = config.model_parallel_config.pipeline_model_parallel_size
    ep = getattr(config.model_parallel_config, "expert_model_parallel_size", 1) or 1
    hidden_size = config.model_config.hidden_size

    # Message size: activations after row-parallel projection
    # Shape: [batch_size * seq_len / CP, hidden_size], BF16 (2 bytes)
    message_size_bytes = batch_size * seq_len * hidden_size * 2 // cp

    # Setup collective communication args
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", "8"))
    num_nodes = int(os.environ.get("NNODES", "1"))

    coll_args = get_default_args(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        tp=tp,
        pp=pp,
        ep=ep,
        cp=cp,
    )

    # TP AllReduce is across tp ranks (typically intra-node)
    ar_time_us = cm.allreduce(coll_args, message_size_bytes, tp, groups=["tp"])
    return ar_time_us / 1000.0  # Convert microseconds → milliseconds


def _estimate_moe_a2a_time_ms(config, batch_size: int, seq_len: int) -> float:
    """
    Estimate MoE All-to-All time (dispatch + combine) per layer per direction (in ms).

    Each MoE layer performs two A2A operations per direction:
      1. **Dispatch**: scatter tokens to the EP ranks that own the assigned experts.
      2. **Combine**: gather expert outputs back to the originating ranks.

    In benchmark mode this cost is captured inside the measured layer time.
    In simulation mode we must add it explicitly because the layer profiler
    only simulates GEMM / SDPA compute and TP AllReduce.

    Returns 0.0 when EP <= 1 (all experts are local, no A2A needed).
    """
    ep = getattr(config.model_parallel_config, "expert_model_parallel_size", 1) or 1
    if ep <= 1:
        return 0.0

    tp = config.model_parallel_config.tensor_model_parallel_size
    pp = config.model_parallel_config.pipeline_model_parallel_size
    cp = getattr(config.model_parallel_config, "context_parallel_size", 1) or 1
    hidden_size = config.model_config.hidden_size
    moe_router_topk = getattr(config.model_config, "moe_router_topk", 2)

    # A2A message size: each rank sends/receives all routed tokens
    # Shape: [batch_size * seq_len * topk, hidden_size], BF16 (2 bytes)
    tokens_per_batch = batch_size * seq_len
    dispatch_size_bytes = tokens_per_batch * hidden_size * moe_router_topk * 2

    # Setup collective communication args
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", "8"))
    num_nodes = int(os.environ.get("NNODES", "1"))

    coll_args = get_default_args(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        tp=tp,
        pp=pp,
        ep=ep,
        cp=cp,
    )

    # Propagate DeepEP setting if present (affects A2A algorithm selection)
    moe_enable_deepep = getattr(
        config.model_parallel_config, "moe_enable_deepep", False
    )
    use_turbo_deepep = getattr(config.model_parallel_config, "use_turbo_deepep", False)
    coll_args.moe_enable_deepep = moe_enable_deepep
    coll_args.use_turbo_deepep = use_turbo_deepep

    # Dispatch A2A + Combine A2A (same message size, same time)
    a2a_dispatch_us = cm.alltoall(coll_args, dispatch_size_bytes, ep, groups=["ep"])
    a2a_combine_us = cm.alltoall(coll_args, dispatch_size_bytes, ep, groups=["ep"])

    return (a2a_dispatch_us + a2a_combine_us) / 1000.0  # Convert us → ms


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
        self._cached_results = (
            None  # Cache for (forward_time, backward_time, activation_memory)
        )
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
            self.sub_profilers["layer_norm"].estimated_activation_memory(
                batch_size, seq_len
            )
            * 3
            + self.sub_profilers["self_attention"].estimated_activation_memory(
                batch_size, seq_len
            )
            + self.sub_profilers["mlp"].estimated_activation_memory(batch_size, seq_len)
            + self.sub_profilers["residual_add"].estimated_activation_memory(
                batch_size, seq_len
            )
            * 2
        )

    def _get_simulated_results(
        self, batch_size: int, seq_len: int
    ) -> tuple[float, float, int]:
        """Aggregate simulated results from sub-profilers, including TP AllReduce."""
        attn_fwd = self.sub_profilers["self_attention"].measured_forward_time(
            batch_size, seq_len
        )
        attn_bwd = self.sub_profilers["self_attention"].measured_backward_time(
            batch_size, seq_len
        )
        mlp_fwd = self.sub_profilers["mlp"].measured_forward_time(batch_size, seq_len)
        mlp_bwd = self.sub_profilers["mlp"].measured_backward_time(batch_size, seq_len)

        # Add TP AllReduce communication overhead (simulation only).
        # Each transformer layer has 2 AllReduces per direction:
        #   - After attention row-parallel output projection
        #   - After MLP row-parallel down projection
        # (With sequence parallelism these become RS+AG pairs with equal volume.)
        tp_ar_ms = _estimate_tp_allreduce_time_ms(self.config, batch_size, seq_len)

        fwd_time = attn_fwd + mlp_fwd + 2 * tp_ar_ms
        bwd_time = attn_bwd + mlp_bwd + 2 * tp_ar_ms
        activation_memory = self.estimated_activation_memory(batch_size, seq_len)
        return (fwd_time, bwd_time, activation_memory)

    def _get_benchmark_results(
        self, batch_size: int, seq_len: int
    ) -> tuple[float, float, int]:
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
        self._cached_results = (
            None  # Cache for (forward_time, backward_time, activation_memory)
        )
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
            self.sub_profilers["layer_norm"].estimated_activation_memory(
                batch_size, seq_len
            )
            * 3
            + self.sub_profilers["self_attention"].estimated_activation_memory(
                batch_size, seq_len
            )
            + self.sub_profilers["mlp"].estimated_activation_memory(batch_size, seq_len)
            + self.sub_profilers["router"].estimated_activation_memory(
                batch_size, seq_len
            )
            + self.sub_profilers["residual_add"].estimated_activation_memory(
                batch_size, seq_len
            )
            * 2
        )

    def _get_simulated_results(
        self, batch_size: int, seq_len: int
    ) -> tuple[float, float, int]:
        """Aggregate simulated results from sub-profilers.

        Includes TP AllReduce and MoE All-to-All communication overhead that
        would be captured in the measured layer time during benchmark mode but
        must be added explicitly in simulation mode.
        """
        attn_fwd = self.sub_profilers["self_attention"].measured_forward_time(
            batch_size, seq_len
        )
        attn_bwd = self.sub_profilers["self_attention"].measured_backward_time(
            batch_size, seq_len
        )
        mlp_fwd = self.sub_profilers["mlp"].measured_forward_time(batch_size, seq_len)
        mlp_bwd = self.sub_profilers["mlp"].measured_backward_time(batch_size, seq_len)

        # Add TP AllReduce communication overhead (simulation only).
        # Each transformer layer has 2 AllReduces per direction:
        #   - After attention row-parallel output projection
        #   - After MLP row-parallel down projection
        # (With sequence parallelism these become RS+AG pairs with equal volume.)
        tp_ar_ms = _estimate_tp_allreduce_time_ms(self.config, batch_size, seq_len)

        # Add MoE All-to-All communication overhead (simulation only).
        # Each MoE layer performs dispatch A2A + combine A2A per direction.
        # In benchmark mode this is captured in the measured layer time;
        # in simulation mode the layer profiler only computes GEMM / SDPA
        # so we must add it here.
        moe_a2a_ms = _estimate_moe_a2a_time_ms(self.config, batch_size, seq_len)

        fwd_time = attn_fwd + mlp_fwd + 2 * tp_ar_ms + moe_a2a_ms
        bwd_time = attn_bwd + mlp_bwd + 2 * tp_ar_ms + moe_a2a_ms
        activation_memory = self.estimated_activation_memory(batch_size, seq_len)
        return (fwd_time, bwd_time, activation_memory)

    def _get_benchmark_results(
        self, batch_size: int, seq_len: int
    ) -> tuple[float, float, int]:
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
