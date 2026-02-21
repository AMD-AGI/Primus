###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

from primus.core.projection.base_module_profiler import BaseModuleProfiler

from .utils import benchmark_layer


class OutputLayerProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        self.module = None  # Will be set during benchmarking
        self._cached_results = (
            None  # Cache for (forward_time, backward_time, activation_memory)
        )
        self._cache_key = None  # Cache key (batch_size, seq_len)
        self._gemm_backend = None  # Optional: GEMM simulation backend

    def set_module(self, module):
        """Set the actual module for benchmarking."""
        self.module = module
        # Invalidate cache when module changes
        self._cached_results = None
        self._cache_key = None

    def set_gemm_backend(self, backend):
        """Set a GEMM simulation backend for simulated profiling."""
        self._gemm_backend = backend
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
            * self.config.model_config.padded_vocab_size
            * 2
        )  # bf16

    def _get_simulated_results(
        self, batch_size: int, seq_len: int
    ) -> tuple[float, float, int]:
        """Simulate output layer using GEMM backend (vocab projection GEMM)."""
        tp_size = self.config.model_parallel_config.tensor_model_parallel_size
        cp_size = self.config.model_parallel_config.context_model_parallel_size
        batch_tokens = batch_size * seq_len // tp_size // cp_size
        hidden_size = self.config.model_config.hidden_size
        vocab_size = self.config.model_config.padded_vocab_size

        # Output projection GEMM fwd: [batch_tokens, hidden_size] x [hidden_size, vocab_size]
        fwd_result = self._gemm_backend.simulate_gemm(
            m=batch_tokens,
            n=vocab_size,
            k=hidden_size,
            dtype="bf16",
        )
        fwd_time = fwd_result.forward_time_ms

        # Backward: simulate actual dgrad + wgrad GEMMs
        # dgrad: [batch_tokens, vocab_size] x [vocab_size, hidden_size] -> [batch_tokens, hidden_size]
        dgrad_result = self._gemm_backend.simulate_gemm(
            m=batch_tokens,
            n=hidden_size,
            k=vocab_size,
            dtype="bf16",
        )
        # wgrad: [hidden_size, batch_tokens] x [batch_tokens, vocab_size] -> [hidden_size, vocab_size]
        wgrad_result = self._gemm_backend.simulate_gemm(
            m=hidden_size,
            n=vocab_size,
            k=batch_tokens,
            dtype="bf16",
        )
        bwd_time = dgrad_result.forward_time_ms + wgrad_result.forward_time_ms

        activation_memory = self.estimated_activation_memory(batch_size, seq_len)
        return (fwd_time, bwd_time, activation_memory)

    def _get_benchmark_results(
        self, batch_size: int, seq_len: int
    ) -> tuple[float, float, int]:
        """Get or compute benchmark results (cached)."""
        cache_key = (batch_size, seq_len)

        if self._cached_results is None or self._cache_key != cache_key:
            if self._gemm_backend is not None:
                self._cached_results = self._get_simulated_results(batch_size, seq_len)
            else:
                # Context parallel / Sequence parallel adjustment
                cp_size = self.config.model_parallel_config.context_model_parallel_size
                # Effective sequence length per rank if CP is used
                slen_per_cp = seq_len // cp_size

                self._cached_results = benchmark_layer(
                    self.module,
                    [
                        (slen_per_cp, batch_size, self.config.model_config.hidden_size),
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
