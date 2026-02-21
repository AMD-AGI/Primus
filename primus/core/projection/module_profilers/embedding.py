###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

from primus.core.projection.base_module_profiler import BaseModuleProfiler

from .utils import benchmark_layer


class EmbeddingProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        self.module = None  # Will be set during benchmarking
        self._cached_results = (
            None  # Cache for (forward_time, backward_time, activation_memory)
        )
        self._cache_key = None  # Cache key (batch_size, seq_len)
        self._simulation_mode = False  # Set to True when simulation backends are active

    def set_module(self, module):
        """Set the actual module for benchmarking."""
        self.module = module
        # Invalidate cache when module changes
        self._cached_results = None
        self._cache_key = None

    def set_simulation_mode(self, enabled: bool = True):
        """Enable simulation mode (embedding lookup is estimated analytically)."""
        self._simulation_mode = enabled
        self._cached_results = None
        self._cache_key = None

    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        return (
            self.config.model_config.padded_vocab_size
            * self.config.model_config.hidden_size
        )

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        return (
            batch_size
            * seq_len
            // self.config.model_parallel_config.tensor_model_parallel_size
            // self.config.model_parallel_config.context_model_parallel_size
            * self.config.model_config.hidden_size
            * 2
        )  # bf16

    def _get_simulated_results(
        self, batch_size: int, seq_len: int
    ) -> tuple[float, float, int]:
        """Estimate embedding time analytically (lookup is memory-bound, very fast)."""
        tp_size = self.config.model_parallel_config.tensor_model_parallel_size
        cp_size = self.config.model_parallel_config.context_model_parallel_size
        tokens = batch_size * seq_len // tp_size // cp_size
        hidden = self.config.model_config.hidden_size
        # Embedding lookup: read tokens indices + write output embeddings
        # Very fast relative to GEMM layers – use small fixed estimate
        output_bytes = tokens * hidden * 2  # bf16
        # Assume ~4 TB/s effective bandwidth (MI300X), 1 read + 1 write pass
        bw_bytes_per_ms = 4e9  # ~4 TB/s → bytes/ms
        fwd_time = max(0.01, output_bytes / bw_bytes_per_ms)
        bwd_time = fwd_time  # Gradient scatter is similar cost
        activation_memory = self.estimated_activation_memory(batch_size, seq_len)
        return (fwd_time, bwd_time, activation_memory)

    def _get_benchmark_results(
        self, batch_size: int, seq_len: int
    ) -> tuple[float, float, int]:
        """Get or compute benchmark results (cached)."""
        cache_key = (batch_size, seq_len)

        if self._cached_results is None or self._cache_key != cache_key:
            if self._simulation_mode:
                self._cached_results = self._get_simulated_results(batch_size, seq_len)
            else:
                # Context parallel / Sequence parallel adjustment
                cp_size = self.config.model_parallel_config.context_model_parallel_size
                # Effective sequence length per rank if CP is used
                slen_per_cp = seq_len // cp_size

                self._cached_results = benchmark_layer(
                    self.module,
                    [
                        ((batch_size, slen_per_cp), torch.int64),
                    ],
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
