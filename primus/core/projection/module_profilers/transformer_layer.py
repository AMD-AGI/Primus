###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.profiler_spec import ModuleProfilerSpec
from primus.core.projection.training_config import TrainingConfig

from .attention import AttentionProfiler
from .dense_mlp import DenseMLPProfiler
from .layer_norm import LayerNormProfiler
from .moe_mlp import MoEMLPProfiler
from .router import RouterProfiler

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


def benchmark_layer(
    layer_module, hidden_size: int, batch_size: int, seq_len: int, num_iterations: int = 10
) -> tuple[float, float, int]:
    """
    Benchmark both forward and backward passes of a transformer layer using CUDA events.
    Also measures activation memory used by the forward pass.

    Args:
        layer_module: The transformer layer module
        hidden_size: Hidden size of the model
        batch_size: Micro batch size
        seq_len: Sequence length
        num_iterations: Number of iterations to average over

    Returns:
        Tuple of (average forward time in ms, average backward time in ms, activation memory in bytes)
    """
    import torch

    device = next(layer_module.parameters()).device

    # Create dummy input
    hidden_states = torch.randn(
        seq_len, batch_size, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True
    )

    # Warm-up: forward and backward passes
    for _ in range(3):
        output = layer_module(hidden_states)
        output[0].backward(torch.randn_like(output[0]))
        layer_module.zero_grad()
        hidden_states.grad = None

    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Measure forward and backward passes using CUDA events
    forward_times = []
    backward_times = []
    activation_memories = []

    for _ in range(num_iterations):
        # Clear cache and reset memory stats before measuring
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        # Get baseline memory
        mem_before = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0

        # Measure forward pass
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)

        forward_start.record()
        output = layer_module(hidden_states)
        forward_end.record()

        # Measure activation memory (peak memory during forward - baseline)
        torch.cuda.synchronize(device)
        mem_after_forward = torch.cuda.max_memory_allocated(device)
        activation_memory = mem_after_forward - mem_before
        activation_memories.append(activation_memory)

        # Measure backward pass
        grad_output = torch.randn_like(output[0])
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)

        backward_start.record()
        output[0].backward(grad_output)
        backward_end.record()

        # Wait for all events to complete
        torch.cuda.synchronize(device)

        # Record times
        forward_times.append(forward_start.elapsed_time(forward_end))
        backward_times.append(backward_start.elapsed_time(backward_end))

        # Clear gradients for next iteration
        layer_module.zero_grad()
        hidden_states.grad = None
        del output, grad_output

    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)
    avg_activation_memory = (
        int(sum(activation_memories) / len(activation_memories)) if activation_memories else 0
    )

    return avg_forward_time, avg_backward_time, avg_activation_memory


class DenseTransformerLayerProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        self.layer_module = None  # Will be set during benchmarking
        self._cached_results = None  # Cache for (forward_time, backward_time, activation_memory)
        self._cache_key = None  # Cache key (batch_size, seq_len)

    def set_layer_module(self, layer_module):
        """Set the actual transformer layer module for benchmarking."""
        self.layer_module = layer_module
        # Invalidate cache when layer changes
        self._cached_results = None
        self._cache_key = None

    def estimated_num_params(self, rank: int | None = None) -> int:
        return (
            self.sub_profilers["layer_norm"].estimated_num_params(rank) * 2
            + self.sub_profilers["self_attention"].estimated_num_params(rank)
            + self.sub_profilers["mlp"].estimated_num_params(rank)
        )

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        return (
            self.sub_profilers["layer_norm"].estimated_activation_memory(batch_size, seq_len) * 2
            + self.sub_profilers["self_attention"].estimated_activation_memory(batch_size, seq_len)
            + self.sub_profilers["mlp"].estimated_activation_memory(batch_size, seq_len)
        )

    def _get_benchmark_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        """Get or compute benchmark results (cached)."""
        cache_key = (batch_size, seq_len)
        if self._cached_results is None or self._cache_key != cache_key:
            self._cached_results = benchmark_layer(
                self.layer_module,
                self.config.model_config.hidden_size,
                batch_size,
                seq_len,
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

    def set_layer_module(self, layer_module):
        """Set the actual transformer layer module for benchmarking."""
        self.layer_module = layer_module
        # Invalidate cache when layer changes
        self._cached_results = None
        self._cache_key = None

    def estimated_num_params(self, rank: int | None = None) -> int:
        return (
            self.sub_profilers["layer_norm"].estimated_num_params(rank) * 2
            + self.sub_profilers["self_attention"].estimated_num_params(rank)
            + self.sub_profilers["mlp"].estimated_num_params(rank)
            + self.sub_profilers["router"].estimated_num_params(rank)
        )

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        return (
            self.sub_profilers["layer_norm"].estimated_activation_memory(batch_size, seq_len) * 2
            + self.sub_profilers["self_attention"].estimated_activation_memory(batch_size, seq_len)
            + self.sub_profilers["mlp"].estimated_activation_memory(batch_size, seq_len)
            + self.sub_profilers["router"].estimated_activation_memory(batch_size, seq_len)
        )

    def _get_benchmark_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        """Get or compute benchmark results (cached)."""
        cache_key = (batch_size, seq_len)
        if self._cached_results is None or self._cache_key != cache_key:
            self._cached_results = benchmark_layer(
                self.layer_module, self.config.model_config.hidden_size, batch_size, seq_len
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
            "router": RouterProfiler,
            "mlp": MoEMLPProfiler,
        },
    )
