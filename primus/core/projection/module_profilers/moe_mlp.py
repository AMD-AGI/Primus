###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import math
import os
from typing import Optional

from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.profiler_spec import ModuleProfilerSpec
from primus.core.projection.training_config import TrainingConfig

from .utils import benchmark_layer

# Efficiency fractions for non-GEMM MoE overhead estimation.
# These express achievable bandwidth as a fraction of peak HBM bandwidth.
# The actual BW is ``fraction × peak_hbm_bw`` for the target architecture,
# so the model scales automatically across MI300X (5.3 TB/s), MI325X (6.0
# TB/s), MI355X (8.0 TB/s), etc.
#
# PERMUTE (scatter/gather) — random-access token dispatch/combine.  Irregular
# access patterns achieve only ~5-7 % of peak HBM bandwidth.
_PERMUTE_BW_FRACTION = 0.057
#
# ACTIVATION (SwiGLU / GELU) — sequential element-wise ops that stream over
# contiguous buffers.  Typically ~55-60 % of peak HBM bandwidth.
_ACTIVATION_BW_FRACTION = 0.566
#
# Fallback absolute values used when the backend cannot report HBM bandwidth.
_FALLBACK_HBM_BW_GBPS = 5300.0  # MI300X default


class MoEMLPProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        self.module = None  # Will be set during benchmarking
        self._cached_results = (
            None  # Cache for (forward_time, backward_time, activation_memory)
        )
        self._cache_key = None  # Cache key (batch_size, seq_len)
        self._gemm_backend = None  # Optional: GEMM simulation backend

    def set_module(self, module):
        """Set the actual MoE MLP module for benchmarking."""
        self.module = module
        # Invalidate cache when module changes
        self._cached_results = None
        self._cache_key = None

    def set_gemm_backend(self, backend):
        """Set a GEMM simulation backend for simulated profiling."""
        self._gemm_backend = backend
        # Invalidate cache when backend changes
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
        per_expert_params = (
            num_ffn_projections * self.config.model_config.hidden_size * moe_ffn
        )
        ep = (
            1
            if rank is None
            else self.config.model_parallel_config.expert_model_parallel_size
        )

        all_experts_params = (
            self.config.model_config.num_experts * per_expert_params // ep
        )

        # Shared experts (if any)
        shared_sz = 0
        if self.config.model_config.moe_shared_expert_intermediate_size is not None:
            shared_sz = self.config.model_config.moe_shared_expert_intermediate_size
        shared_params = (
            num_ffn_projections * self.config.model_config.hidden_size * shared_sz
        )

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
            output_memory = (
                num_tokens * self.config.model_config.hidden_size * 2
            )  # bf16
            total += intermediate_memory + activation_memory + output_memory

        return total

    def _get_simulated_results(
        self, batch_size: int, seq_len: int
    ) -> tuple[float, float, int]:
        """Get simulated results from the GEMM simulation backend for MoE MLP.

        In addition to expert GEMM time, this method estimates several
        components of MoE execution that the GEMM simulation alone misses:

        1. **Router overhead** — gate linear projection + softmax/top-K.
        2. **Token permutation** — dispatch (scatter) and combine (gather)
           memory traffic with random-access patterns.
        3. **Activation function** — SwiGLU / GELU element-wise overhead.

        **Grouped GEMM performance model selection**:
        When ``enable_primus_turbo`` and ``use_turbo_grouped_mlp`` are both
        ``True`` in the training config, the expert GEMMs are modelled using
        Origami's *batched* GEMM path (``batch=num_local_experts``).  Primus
        Turbo's grouped-GEMM kernel achieves near-ideal batched execution,
        so the batched model is an accurate proxy.

        Otherwise (legacy ``grouped_gemm`` package), each expert is simulated
        independently (``batch=1``) and the result is scaled by the number of
        local experts.  This more closely reflects the sequential per-expert
        execution of the legacy kernel.
        """
        tp_size = self.config.model_parallel_config.tensor_model_parallel_size
        cp_size = self.config.model_parallel_config.context_model_parallel_size
        ep_size = self.config.model_parallel_config.expert_model_parallel_size

        hidden_size = self.config.model_config.hidden_size
        batch_tokens = batch_size * seq_len // tp_size // cp_size
        topk = self.config.model_config.moe_router_topk
        topk_tokens = batch_tokens * topk

        if self.config.model_config.moe_ffn_hidden_size is not None:
            moe_ffn = self.config.model_config.moe_ffn_hidden_size
        else:
            moe_ffn = self.config.model_config.ffn_hidden_size

        num_experts = self.config.model_config.num_experts or 1
        num_local_experts = num_experts // ep_size
        tokens_per_expert = topk_tokens // max(num_local_experts, 1)

        # FP8-hybrid: MoE expert MLP projections run in FP8
        gemm_dtype = "fp8" if getattr(self.config.model_config, "fp8", None) else "bf16"
        bytes_per_el = 1 if gemm_dtype == "fp8" else 2

        # ── 1. Routed expert GEMMs ──
        M = tokens_per_expert
        H = hidden_size
        F = moe_ffn

        # Determine grouped-GEMM performance model.
        # Primus Turbo's grouped-GEMM kernel achieves near-ideal batched
        # execution → model as Origami batched GEMM (batch=num_local_experts).
        # Legacy grouped_gemm executes experts more sequentially → model as
        # individual GEMM (batch=1) × num_local_experts.
        use_turbo = getattr(
            self.config.model_config, "enable_primus_turbo", False
        ) and getattr(self.config.model_config, "use_turbo_grouped_mlp", False)

        is_rank_0 = int(os.getenv("RANK", "0")) == 0
        if is_rank_0 and num_local_experts > 1:
            mode = "Turbo (batched)" if use_turbo else "Legacy (sequential)"
            print(
                f"  [MoE MLP] Grouped-GEMM model: {mode}"
                f"  ({num_local_experts} local experts, M={M}, H={H}, F={F})"
            )

        expert_fwd_ms = 0.0
        expert_bwd_ms = 0.0

        if use_turbo:
            # ── Turbo model: batched GEMM (all experts in parallel) ──
            B = num_local_experts
            if self.config.model_config.swiglu:
                gate_fwd = self._gemm_backend.simulate_gemm(
                    M, F, H, gemm_dtype, batch=B
                )
                up_fwd = self._gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=B)
                down_fwd = self._gemm_backend.simulate_gemm(
                    M, H, F, gemm_dtype, batch=B
                )
                expert_fwd_ms = (
                    gate_fwd.forward_time_ms
                    + up_fwd.forward_time_ms
                    + down_fwd.forward_time_ms
                )
                gate_dg = self._gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=B)
                gate_wg = self._gemm_backend.simulate_gemm(H, F, M, gemm_dtype, batch=B)
                up_dg = self._gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=B)
                up_wg = self._gemm_backend.simulate_gemm(H, F, M, gemm_dtype, batch=B)
                down_dg = self._gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=B)
                down_wg = self._gemm_backend.simulate_gemm(F, H, M, gemm_dtype, batch=B)
                expert_bwd_ms = (
                    gate_dg.forward_time_ms
                    + gate_wg.forward_time_ms
                    + up_dg.forward_time_ms
                    + up_wg.forward_time_ms
                    + down_dg.forward_time_ms
                    + down_wg.forward_time_ms
                )
            else:
                up_fwd = self._gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=B)
                down_fwd = self._gemm_backend.simulate_gemm(
                    M, H, F, gemm_dtype, batch=B
                )
                expert_fwd_ms = up_fwd.forward_time_ms + down_fwd.forward_time_ms
                up_dg = self._gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=B)
                up_wg = self._gemm_backend.simulate_gemm(H, F, M, gemm_dtype, batch=B)
                down_dg = self._gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=B)
                down_wg = self._gemm_backend.simulate_gemm(F, H, M, gemm_dtype, batch=B)
                expert_bwd_ms = (
                    up_dg.forward_time_ms
                    + up_wg.forward_time_ms
                    + down_dg.forward_time_ms
                    + down_wg.forward_time_ms
                )

            expert_fwd = expert_fwd_ms
            expert_bwd = expert_bwd_ms
        else:
            # ── Legacy model: individual GEMM × num_local_experts ──
            if self.config.model_config.swiglu:
                gate_fwd = self._gemm_backend.simulate_gemm(
                    M, F, H, gemm_dtype, batch=1
                )
                up_fwd = self._gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=1)
                down_fwd = self._gemm_backend.simulate_gemm(
                    M, H, F, gemm_dtype, batch=1
                )
                expert_fwd_ms = (
                    gate_fwd.forward_time_ms
                    + up_fwd.forward_time_ms
                    + down_fwd.forward_time_ms
                )
                gate_dg = self._gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=1)
                gate_wg = self._gemm_backend.simulate_gemm(H, F, M, gemm_dtype, batch=1)
                up_dg = self._gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=1)
                up_wg = self._gemm_backend.simulate_gemm(H, F, M, gemm_dtype, batch=1)
                down_dg = self._gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=1)
                down_wg = self._gemm_backend.simulate_gemm(F, H, M, gemm_dtype, batch=1)
                expert_bwd_ms = (
                    gate_dg.forward_time_ms
                    + gate_wg.forward_time_ms
                    + up_dg.forward_time_ms
                    + up_wg.forward_time_ms
                    + down_dg.forward_time_ms
                    + down_wg.forward_time_ms
                )
            else:
                up_fwd = self._gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=1)
                down_fwd = self._gemm_backend.simulate_gemm(
                    M, H, F, gemm_dtype, batch=1
                )
                expert_fwd_ms = up_fwd.forward_time_ms + down_fwd.forward_time_ms
                up_dg = self._gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=1)
                up_wg = self._gemm_backend.simulate_gemm(H, F, M, gemm_dtype, batch=1)
                down_dg = self._gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=1)
                down_wg = self._gemm_backend.simulate_gemm(F, H, M, gemm_dtype, batch=1)
                expert_bwd_ms = (
                    up_dg.forward_time_ms
                    + up_wg.forward_time_ms
                    + down_dg.forward_time_ms
                    + down_wg.forward_time_ms
                )

            expert_fwd = expert_fwd_ms * num_local_experts
            expert_bwd = expert_bwd_ms * num_local_experts

        fwd_time = expert_fwd
        bwd_time = expert_bwd

        # ── 2. Router overhead ──
        # Gate linear: [batch_tokens, num_experts, hidden_size]
        router_gemm = self._gemm_backend.simulate_gemm(
            batch_tokens, num_experts, hidden_size, gemm_dtype
        )
        router_fwd_ms = router_gemm.forward_time_ms
        # Softmax + top-K selection + auxiliary loss overhead (empirical)
        topk_overhead_ms = 0.1 + 0.002 * num_experts
        router_fwd_ms += topk_overhead_ms
        # Backward: dgrad + wgrad for gate linear
        router_bwd_ms = 2.0 * router_gemm.forward_time_ms + topk_overhead_ms

        fwd_time += router_fwd_ms
        bwd_time += router_bwd_ms

        # ── 3. Token permutation overhead (dispatch + combine) ──
        # Dispatch: gather tokens by expert assignment → irregular memory access
        # Combine: scatter expert outputs back → weighted reduce
        #
        # Derive effective BW from the target GPU's peak HBM bandwidth so the
        # model adapts automatically to different architectures.
        peak_hbm = (
            self._gemm_backend.hbm_bandwidth_gbps
            if self._gemm_backend is not None
            and self._gemm_backend.hbm_bandwidth_gbps is not None
            else _FALLBACK_HBM_BW_GBPS
        )
        permute_eff_bw_gbps = peak_hbm * _PERMUTE_BW_FRACTION
        activation_bw_gbps = peak_hbm * _ACTIVATION_BW_FRACTION

        dispatch_bytes = (batch_tokens + topk_tokens) * hidden_size * bytes_per_el
        combine_bytes = (topk_tokens + batch_tokens) * hidden_size * bytes_per_el
        permute_fwd_ms = dispatch_bytes / (permute_eff_bw_gbps * 1e6)
        permute_bwd_ms = combine_bytes / (permute_eff_bw_gbps * 1e6)

        fwd_time += permute_fwd_ms
        bwd_time += permute_bwd_ms

        # ── 4. Activation function overhead (SwiGLU / GELU) ──
        if self.config.model_config.swiglu:
            act_bytes = (
                3 * topk_tokens * moe_ffn * bytes_per_el
            )  # gate+up read, result write
        else:
            act_bytes = 2 * topk_tokens * moe_ffn * bytes_per_el  # read + write
        activation_ms = act_bytes / (activation_bw_gbps * 1e6)

        fwd_time += activation_ms
        bwd_time += activation_ms

        # ── 5. Shared experts (if any) ──
        shared_sz = self.config.model_config.moe_shared_expert_intermediate_size
        if shared_sz:
            shared_result = self._gemm_backend.simulate_mlp_gemms(
                batch_tokens=batch_tokens,
                hidden_size=hidden_size,
                ffn_hidden_size=shared_sz,
                dtype=gemm_dtype,
                swiglu=self.config.model_config.swiglu,
            )
            fwd_time += shared_result.forward_time_ms
            bwd_time += shared_result.backward_time_ms

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
