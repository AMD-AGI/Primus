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

from .utils import benchmark_moe_layer_decomposed

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
        self._cached_results = None  # Cache for (forward_time, backward_time, activation_memory)
        self._cache_key = None  # Cache key (batch_size, seq_len)
        self._gemm_backend = None  # Optional: GEMM simulation backend
        # Decomposed A2A timings (populated during benchmarking)
        self._a2a_fwd_ms = 0.0  # Measured A2A dispatch+combine forward time
        self._a2a_bwd_ms = 0.0  # Measured A2A dispatch+combine backward time (estimated)

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

    def _get_simulated_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        """Get simulated results from the GEMM simulation backend for MoE MLP.

        In addition to expert GEMM time, this method estimates several
        components of MoE execution that the GEMM simulation alone misses:

        1. **Router overhead** — gate linear projection + softmax/top-K.
        2. **Token permutation** — dispatch (scatter) and combine (gather)
           memory traffic with random-access patterns.
        3. **Activation function** — SwiGLU / GELU element-wise overhead.

        **Grouped GEMM performance model selection**:
        When ``enable_primus_turbo`` and ``use_turbo_grouped_gemm`` are both
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
        batch_tokens = max(1, batch_size * seq_len // tp_size // cp_size)
        topk = self.config.model_config.moe_router_topk
        topk_tokens = batch_tokens * topk

        if self.config.model_config.moe_ffn_hidden_size is not None:
            moe_ffn = self.config.model_config.moe_ffn_hidden_size
        else:
            moe_ffn = self.config.model_config.ffn_hidden_size

        num_experts = self.config.model_config.num_experts or 1
        num_local_experts = num_experts // ep_size

        # Expert tensor-parallel degree: within a TP×EP domain, EP distributes
        # whole experts across ranks and the remaining ``tp_size // ep_size`` ranks
        # tensor-shard each expert's FFN intermediate (gate/up column-parallel,
        # down row-parallel — all shard the ``moe_ffn`` dimension). Without this,
        # the weight-bandwidth-bound decode GEMM reads the *full* expert weights on
        # every rank and shows no TP speedup (e.g. EP=1: experts never shard).
        #   EP=1, TP=8 -> etp=8 (full TP sharding)   EP=8, TP=8 -> etp=1 (EP-only)
        expert_tp = max(1, tp_size // max(1, ep_size))

        # Decode weight-bandwidth relief cap for TP-sharded experts (``etp>1``,
        # i.e. TP>EP): measured pure-TP MoE decode barely beats the single-GPU
        # step (~0.87x at TP2, non-monotonic at higher TP), NOT the ~1/etp the
        # weight roofline predicts. At decode (M~1) the expert grouped-GEMM is
        # latency/launch-bound rather than weight-streaming-bound, and any
        # compute relief from TP-sharding the FFN is offset by the post-expert
        # all-reduce that reassembles the row-parallel output — a cost the
        # comm-free decode ratio drops. So the FFN TP-shard yields ~no net decode
        # relief: read each active expert's weights as if replicated across the
        # ``etp`` ranks (attention/token/EP sharding still apply). Prefill (large
        # M, compute-bound, seq_len>1) keeps the full ``etp`` speedup. Only the
        # extra TP-over-EP factor is capped; EP-only sharding (etp==1) is a no-op.
        # Toggle: PRIMUS_DECODE_ETP_CAP=0 restores the uncapped weight relief.
        weight_expert_tp = expert_tp
        if (
            int(seq_len) <= 1
            and expert_tp > 1
            and os.getenv("PRIMUS_DECODE_ETP_CAP", "1").strip().lower()
            not in ("0", "false", "no")
        ):
            weight_expert_tp = 1

        # Expected number of *distinct* experts whose weights are actually read
        # this step. MoE decode is weight-bandwidth-bound, so its cost tracks the
        # number of experts touched, not the full local set: at small batch only
        # ``topk × tokens`` routings land, covering a fraction of experts, while
        # at prefill / high batch nearly every expert is hit. Uniform-routing
        # approximation for the hit probability of a given expert over
        # ``routing_tokens`` tokens each selecting ``topk`` of ``num_experts``:
        #   P(hit) = 1 - (1 - topk/E)^tokens
        # This is a no-op for prefill/training (tokens large -> P(hit) -> 1) and
        # only reshapes the small-batch decode curve.
        routing_tokens = max(1, int(batch_size) * int(seq_len))
        if num_experts > 1 and num_local_experts > 1:
            hit_frac = 1.0 - (1.0 - min(1.0, topk / num_experts)) ** routing_tokens
            active_local_experts = max(
                1, min(num_local_experts, int(round(num_local_experts * hit_frac)))
            )
        else:
            active_local_experts = max(1, num_local_experts)

        # Tokens concentrate on the experts actually hit. Floor at 1: small decode
        # batches yield < 1 token per expert, which would size a 0-row grouped
        # GEMM and divide-by-zero in the simulator.
        tokens_per_expert = max(1, topk_tokens // max(active_local_experts, 1))

        # MoE routing imbalance (busiest EP rank), applied *inside* the roofline:
        # the hottest rank sees ``imbalance``x the mean token load, so scale its
        # per-expert row count ``M``. Because the expert GEMM is
        # ``max(compute, weight_read)``, this is a near no-op at weight-bandwidth-
        # bound decode (small M) and grows to the full factor at compute-bound
        # prefill / high batch — matching measured EP=8 decode (~flat vs EP=1).
        # EP-gated (no cross-rank imbalance when EP<=1) and per-view (each view's
        # own ``ep_size``). Toggle: PRIMUS_MOE_IMB_ROOFLINE=0 disables (imbalance
        # then handled by the legacy outer multiplier in performance.py).
        imbalance = 1.0
        if (
            os.getenv("PRIMUS_MOE_IMB_ROOFLINE", "1").strip().lower() not in ("0", "false", "no")
            and ep_size > 1
            and num_experts > 1
        ):
            elb = float(getattr(self.config.model_config, "ep_load_balance", 1.0) or 1.0)
            if elb > 1.0:
                red = max(0, int(getattr(self.config.model_config, "redundant_experts", 0) or 0))
                imbalance = 1.0 + (elb - 1.0) * num_experts / (num_experts + red) if red else elb
                imbalance = max(1.0, imbalance)
        tokens_per_expert = max(1, int(round(tokens_per_expert * imbalance)))

        # FP8-hybrid: MoE expert MLP projections run in FP8
        gemm_dtype = "fp8" if getattr(self.config.model_config, "fp8", None) else "bf16"
        bytes_per_el = 1 if gemm_dtype == "fp8" else 2

        # ── 1. Routed expert GEMMs ──
        # ``F`` is the per-rank FFN intermediate after expert tensor-sharding, so
        # the weight-bound decode step scales ~1/etp (and total per-rank expert
        # weight ∝ num_experts·H·F/tp, independent of the EP/TP split — physically
        # correct: total expert params spread over all GPUs).
        M = tokens_per_expert
        H = hidden_size
        F = max(1, moe_ffn // weight_expert_tp)

        # Determine grouped-GEMM performance model.
        # Primus Turbo's grouped-GEMM kernel achieves near-ideal batched
        # execution → model as Origami batched GEMM (batch=num_local_experts).
        # Legacy grouped_gemm executes experts more sequentially → model as
        # individual GEMM (batch=1) × num_local_experts.
        use_turbo = getattr(self.config.model_config, "enable_primus_turbo", False) and getattr(
            self.config.model_config, "use_turbo_grouped_gemm", False
        )

        # Kernel selection. ``vllm_fused`` models the vLLM fused-MoE decode kernel:
        # a single batched op over the *active* experts (no per-expert launch
        # overhead like the Megatron ``legacy`` grouped_gemm) that is weight-
        # bandwidth-bound at decode — the backend roofline over ``batch=active``
        # captures reading each touched expert's (sharded) weights once. Resolution
        # order: explicit env / model_config override, else turbo flag, else legacy.
        # vLLM serving should use ``vllm_fused``; Megatron training keeps turbo/legacy.
        kernel = (
            os.getenv("PRIMUS_MOE_SIM_KERNEL")
            or getattr(self.config.model_config, "moe_sim_kernel", None)
            or ("turbo" if use_turbo else "legacy")
        ).lower()
        batched = kernel in ("turbo", "vllm_fused")

        is_rank_0 = int(os.getenv("RANK", "0")) == 0
        if is_rank_0 and num_local_experts > 1:
            label = {"vllm_fused": "vLLM fused (batched)", "turbo": "Turbo (batched)"}.get(
                kernel, "Legacy (sequential)"
            )
            print(
                f"  [MoE MLP] Grouped-GEMM model: {label}"
                f"  ({active_local_experts}/{num_local_experts} active local experts, "
                f"M={M}, H={H}, F={F})"
            )

        if batched:
            # ── Batched model (turbo / vLLM-fused): active experts in parallel ──
            B = active_local_experts
            if self.config.model_config.swiglu:
                gate_fwd = self._gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=B)
                up_fwd = self._gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=B)
                down_fwd = self._gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=B)
                expert_fwd_ms = gate_fwd.forward_time_ms + up_fwd.forward_time_ms + down_fwd.forward_time_ms
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
                down_fwd = self._gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=B)
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
                gate_fwd = self._gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=1)
                up_fwd = self._gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=1)
                down_fwd = self._gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=1)
                expert_fwd_ms = gate_fwd.forward_time_ms + up_fwd.forward_time_ms + down_fwd.forward_time_ms
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
                down_fwd = self._gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=1)
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

            expert_fwd = expert_fwd_ms * active_local_experts
            expert_bwd = expert_bwd_ms * active_local_experts

            # NOTE: Legacy grouped GEMM is not properly modelled. Origami
            # simulates ideal single-kernel execution
            if is_rank_0:
                print(
                    "  [MoE MLP] WARNING: Legacy grouped GEMM not properly modelled. "
                    "Estimates may be inaccurate."
                )

        fwd_time = expert_fwd
        bwd_time = expert_bwd

        # Effective HBM bandwidths for the memory-bound (non-GEMM) MoE terms.
        # Derived from the target GPU's peak HBM bandwidth so the model adapts
        # automatically across architectures (MI300X/MI325X/MI355X).
        peak_hbm = (
            self._gemm_backend.hbm_bandwidth_gbps
            if self._gemm_backend is not None and self._gemm_backend.hbm_bandwidth_gbps is not None
            else _FALLBACK_HBM_BW_GBPS
        )
        permute_eff_bw_gbps = peak_hbm * _PERMUTE_BW_FRACTION
        activation_bw_gbps = peak_hbm * _ACTIVATION_BW_FRACTION

        # ── 2. Router overhead ──
        # Gate linear: [batch_tokens, num_experts, hidden_size]
        router_gemm = self._gemm_backend.simulate_gemm(batch_tokens, num_experts, hidden_size, gemm_dtype)
        router_fwd_ms = router_gemm.forward_time_ms
        # Softmax + top-K selection over the router logits [batch_tokens,
        # num_experts]. Modelled as a memory-bound streaming op (read logits,
        # write gates/indices) rather than a fixed per-layer constant. This
        # scales with the tokens actually routed on this rank — so it shards
        # with TP (via ``batch_tokens``) and collapses to ~0 at decode (few
        # tokens). A batch-independent constant instead stays equal across the
        # bench/target views and pulls the EP/TP restore ratio toward 1
        # (under-relief). Softmax streams contiguous buffers, so it uses the
        # same sequential-streaming BW fraction as the activation term.
        router_logits_bytes = 2 * batch_tokens * num_experts * 4  # fp32 logits read + write
        topk_overhead_ms = router_logits_bytes / (activation_bw_gbps * 1e6)
        router_fwd_ms += topk_overhead_ms
        # Backward: dgrad + wgrad for gate linear
        router_bwd_ms = 2.0 * router_gemm.forward_time_ms + topk_overhead_ms

        fwd_time += router_fwd_ms
        bwd_time += router_bwd_ms

        # ── 3. Token permutation overhead (dispatch + combine) ──
        # Dispatch: gather tokens by expert assignment → irregular memory access
        # Combine: scatter expert outputs back → weighted reduce

        dispatch_bytes = (batch_tokens + topk_tokens) * hidden_size * bytes_per_el
        combine_bytes = (topk_tokens + batch_tokens) * hidden_size * bytes_per_el
        permute_fwd_ms = dispatch_bytes / (permute_eff_bw_gbps * 1e6)
        permute_bwd_ms = combine_bytes / (permute_eff_bw_gbps * 1e6)

        fwd_time += permute_fwd_ms
        bwd_time += permute_bwd_ms

        # ── 4. Activation function overhead (SwiGLU / GELU) ──
        # Uses the per-rank (tensor-sharded) intermediate ``F``, not full moe_ffn.
        if self.config.model_config.swiglu:
            act_bytes = 3 * topk_tokens * F * bytes_per_el  # gate+up read, result write
        else:
            act_bytes = 2 * topk_tokens * F * bytes_per_el  # read + write
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

    def _get_benchmark_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        """Get or compute benchmark results (cached).

        When benchmarking (not simulating), uses decomposed MoE benchmarking
        to separately measure A2A communication time.  The A2A times are
        stored in ``self._a2a_fwd_ms`` / ``self._a2a_bwd_ms`` and can be
        retrieved via :meth:`measured_a2a_forward_time` /
        :meth:`measured_a2a_backward_time`.
        """
        cache_key = (batch_size, seq_len)
        if self._cached_results is None or self._cache_key != cache_key:
            if self._gemm_backend is not None:
                self._cached_results = self._get_simulated_results(batch_size, seq_len)
                self._a2a_fwd_ms = 0.0
                self._a2a_bwd_ms = 0.0
            else:
                fwd, bwd, act_mem, a2a_fwd, a2a_bwd = benchmark_moe_layer_decomposed(
                    self.module,
                    [(seq_len, batch_size, self.config.model_config.hidden_size)],
                )
                self._cached_results = (fwd, bwd, act_mem)
                self._a2a_fwd_ms = a2a_fwd
                self._a2a_bwd_ms = a2a_bwd
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

    def measured_a2a_forward_time(self, batch_size: int, seq_len: int) -> float:
        """Return the measured A2A (dispatch+combine) forward time in ms.

        Must be called after :meth:`measured_forward_time` so that the cache
        is populated.  Returns 0.0 in simulation mode.
        """
        self._get_benchmark_results(batch_size, seq_len)  # ensure cache
        return self._a2a_fwd_ms

    def measured_a2a_backward_time(self, batch_size: int, seq_len: int) -> float:
        """Return the estimated A2A backward time in ms (≈ forward A2A).

        Must be called after :meth:`measured_backward_time` so that the cache
        is populated.  Returns 0.0 in simulation mode.
        """
        self._get_benchmark_results(batch_size, seq_len)  # ensure cache
        return self._a2a_bwd_ms


def get_moe_mlp_profiler_spec(config: TrainingConfig) -> ModuleProfilerSpec:
    return ModuleProfilerSpec(
        profiler=MoEMLPProfiler,
        config=config,
        sub_profiler_specs=None,
    )
