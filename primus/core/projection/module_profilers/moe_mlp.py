###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.profiler_spec import ModuleProfilerSpec
from primus.core.projection.training_config import TrainingConfig, training_config_debug_one_line

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

# Origami simulate_gemm(m,n,k): C=A@B with A [m,k], B [k,n], C [m,n].


def _projection_rank0() -> bool:
    """True for global rank 0.

    Prefer ``RANK`` when set (correct on multi-node). If only ``LOCAL_RANK`` is
    set (some single-process launchers), use that. Otherwise treat as rank 0.
    """
    if os.getenv("RANK") is not None:
        return int(os.getenv("RANK", "0")) == 0
    if os.getenv("LOCAL_RANK") is not None:
        return int(os.getenv("LOCAL_RANK", "0")) == 0
    return True


@dataclass
class ExpertGemmOrigamiBreakdown:
    """Per-op Origami expert routed GEMM (grouped-GEMM model) — simulation only."""

    expert_fwd: float
    expert_bwd: float
    use_turbo: bool
    M: int
    H: int
    F: int
    num_local_experts: int
    gemm_dtype: str
    forward_ops: List[Tuple[str, float, int, int, int, int]] = field(default_factory=list)
    backward_ops: List[Tuple[str, float, int, int, int, int]] = field(default_factory=list)
    # Each tuple: (name, ms, m, n, k, batch) — ms from SimulationResult.forward_time_ms


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

    def _expert_gemm_origami_breakdown(
        self, batch_size: int, seq_len: int, gemm_backend
    ) -> ExpertGemmOrigamiBreakdown:
        """Routed expert GEMM: Origami times plus per-op (m,n,k,batch) for forward and backward."""
        tp_size = self.config.model_parallel_config.tensor_model_parallel_size or 1
        cp_size = self.config.model_parallel_config.context_model_parallel_size or 1
        ep_size = self.config.model_parallel_config.expert_model_parallel_size or 1

        hidden_size = self.config.model_config.hidden_size
        batch_tokens = batch_size * seq_len // tp_size // cp_size
        topk_tokens = batch_tokens * self.config.model_config.moe_router_topk

        if self.config.model_config.moe_ffn_hidden_size is not None:
            moe_ffn = self.config.model_config.moe_ffn_hidden_size
        else:
            moe_ffn = self.config.model_config.ffn_hidden_size

        num_experts = self.config.model_config.num_experts or 1
        num_local_experts = max(1, num_experts // max(ep_size, 1))
        tokens_per_expert = topk_tokens // max(num_local_experts, 1)

        gemm_dtype = "fp8" if getattr(self.config.model_config, "fp8", None) else "bf16"

        M = tokens_per_expert
        H = hidden_size
        F = moe_ffn

        use_turbo = getattr(self.config.model_config, "enable_primus_turbo", False) and getattr(
            self.config.model_config, "use_turbo_grouped_mlp", False
        )

        fwd_ops: List[Tuple[str, float, int, int, int, int]] = []
        bwd_ops: List[Tuple[str, float, int, int, int, int]] = []

        def _add_fwd(name: str, r, m: int, n: int, k: int, batch: int) -> float:
            ms = float(r.forward_time_ms)
            fwd_ops.append((name, ms, m, n, k, batch))
            return ms

        def _add_bwd(name: str, r, m: int, n: int, k: int, batch: int) -> float:
            ms = float(r.forward_time_ms)
            bwd_ops.append((name, ms, m, n, k, batch))
            return ms

        if use_turbo:
            B = num_local_experts
            if self.config.model_config.swiglu:
                gate_fwd = gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=B)
                up_fwd = gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=B)
                down_fwd = gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=B)
                expert_fwd_ms = _add_fwd("gate_fwd", gate_fwd, M, F, H, B)
                expert_fwd_ms += _add_fwd("up_fwd", up_fwd, M, F, H, B)
                expert_fwd_ms += _add_fwd("down_fwd", down_fwd, M, H, F, B)
                gate_dg = gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=B)
                gate_wg = gemm_backend.simulate_gemm(H, F, M, gemm_dtype, batch=B)
                up_dg = gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=B)
                up_wg = gemm_backend.simulate_gemm(H, F, M, gemm_dtype, batch=B)
                down_dg = gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=B)
                down_wg = gemm_backend.simulate_gemm(F, H, M, gemm_dtype, batch=B)
                expert_bwd_ms = _add_bwd("gate_dgrad", gate_dg, M, H, F, B)
                expert_bwd_ms += _add_bwd("gate_wgrad", gate_wg, H, F, M, B)
                expert_bwd_ms += _add_bwd("up_dgrad", up_dg, M, H, F, B)
                expert_bwd_ms += _add_bwd("up_wgrad", up_wg, H, F, M, B)
                expert_bwd_ms += _add_bwd("down_dgrad", down_dg, M, F, H, B)
                expert_bwd_ms += _add_bwd("down_wgrad", down_wg, F, H, M, B)
            else:
                up_fwd = gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=B)
                down_fwd = gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=B)
                expert_fwd_ms = _add_fwd("up_fwd", up_fwd, M, F, H, B)
                expert_fwd_ms += _add_fwd("down_fwd", down_fwd, M, H, F, B)
                up_dg = gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=B)
                up_wg = gemm_backend.simulate_gemm(H, F, M, gemm_dtype, batch=B)
                down_dg = gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=B)
                down_wg = gemm_backend.simulate_gemm(F, H, M, gemm_dtype, batch=B)
                expert_bwd_ms = _add_bwd("up_dgrad", up_dg, M, H, F, B)
                expert_bwd_ms += _add_bwd("up_wgrad", up_wg, H, F, M, B)
                expert_bwd_ms += _add_bwd("down_dgrad", down_dg, M, F, H, B)
                expert_bwd_ms += _add_bwd("down_wgrad", down_wg, F, H, M, B)

            expert_fwd = expert_fwd_ms
            expert_bwd = expert_bwd_ms
        else:
            B1 = 1
            if self.config.model_config.swiglu:
                gate_fwd = gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=B1)
                up_fwd = gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=B1)
                down_fwd = gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=B1)
                expert_fwd_ms = _add_fwd("gate_fwd", gate_fwd, M, F, H, B1)
                expert_fwd_ms += _add_fwd("up_fwd", up_fwd, M, F, H, B1)
                expert_fwd_ms += _add_fwd("down_fwd", down_fwd, M, H, F, B1)
                gate_dg = gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=B1)
                gate_wg = gemm_backend.simulate_gemm(H, F, M, gemm_dtype, batch=B1)
                up_dg = gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=B1)
                up_wg = gemm_backend.simulate_gemm(H, F, M, gemm_dtype, batch=B1)
                down_dg = gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=B1)
                down_wg = gemm_backend.simulate_gemm(F, H, M, gemm_dtype, batch=B1)
                expert_bwd_ms = _add_bwd("gate_dgrad", gate_dg, M, H, F, B1)
                expert_bwd_ms += _add_bwd("gate_wgrad", gate_wg, H, F, M, B1)
                expert_bwd_ms += _add_bwd("up_dgrad", up_dg, M, H, F, B1)
                expert_bwd_ms += _add_bwd("up_wgrad", up_wg, H, F, M, B1)
                expert_bwd_ms += _add_bwd("down_dgrad", down_dg, M, F, H, B1)
                expert_bwd_ms += _add_bwd("down_wgrad", down_wg, F, H, M, B1)
            else:
                up_fwd = gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=B1)
                down_fwd = gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=B1)
                expert_fwd_ms = _add_fwd("up_fwd", up_fwd, M, F, H, B1)
                expert_fwd_ms += _add_fwd("down_fwd", down_fwd, M, H, F, B1)
                up_dg = gemm_backend.simulate_gemm(M, H, F, gemm_dtype, batch=B1)
                up_wg = gemm_backend.simulate_gemm(H, F, M, gemm_dtype, batch=B1)
                down_dg = gemm_backend.simulate_gemm(M, F, H, gemm_dtype, batch=B1)
                down_wg = gemm_backend.simulate_gemm(F, H, M, gemm_dtype, batch=B1)
                expert_bwd_ms = _add_bwd("up_dgrad", up_dg, M, H, F, B1)
                expert_bwd_ms += _add_bwd("up_wgrad", up_wg, H, F, M, B1)
                expert_bwd_ms += _add_bwd("down_dgrad", down_dg, M, F, H, B1)
                expert_bwd_ms += _add_bwd("down_wgrad", down_wg, F, H, M, B1)

            expert_fwd = expert_fwd_ms * num_local_experts
            expert_bwd = expert_bwd_ms * num_local_experts

        return ExpertGemmOrigamiBreakdown(
            expert_fwd=expert_fwd,
            expert_bwd=expert_bwd,
            use_turbo=use_turbo,
            M=M,
            H=H,
            F=F,
            num_local_experts=num_local_experts,
            gemm_dtype=gemm_dtype,
            forward_ops=fwd_ops,
            backward_ops=bwd_ops,
        )

    def _print_origami_expert_grouped_gemm_simulation(
        self, bd: ExpertGemmOrigamiBreakdown, batch_size: int, seq_len: int
    ) -> None:
        """Simulation-only: full config + per-op forward/backward Origami GEMM lines (rank 0)."""
        if not _projection_rank0():
            return
        cfg_line = training_config_debug_one_line(self.config)
        gg_mode = "Turbo_batched" if bd.use_turbo else "Legacy_sequential"
        shape_line = self._grouped_gemm_shape_args_line(batch_size, seq_len)

        print(
            "  ========== [MoE MLP] Origami expert grouped-GEMM FORWARD =========="
        )
        print(
            f"    (simulate_gemm m,n,k per op; {gg_mode}; "
            f"legacy scales x{bd.num_local_experts} local experts)"
        )
        print(shape_line)
        print(f"    training_config: {cfg_line}")
        for name, ms, m, n, k, batch in bd.forward_ops:
            print(
                f"    [fwd] {name}: time_ms={ms:.4f}  simulate_gemm(m,n,k)=({m},{n},{k})  "
                f"batch={batch}  dtype={bd.gemm_dtype}"
            )
        print(f"    [fwd] expert_routed_GEMM_forward_total_ms: {bd.expert_fwd:.4f}")
        sys.stdout.flush()

        print(
            "  ========== [MoE MLP] Origami expert grouped-GEMM BACKWARD =========="
        )
        print(
            f"    (dgrad+wgrad as separate simulate_gemm calls; {gg_mode}; same scaling as forward)"
        )
        print(shape_line)
        print(f"    training_config: {cfg_line}")
        if not bd.backward_ops:
            print(
                "    [bwd] ERROR: backward_ops is empty (unexpected); "
                "expert_routed_GEMM_backward_total_ms may be wrong."
            )
        for name, ms, m, n, k, batch in bd.backward_ops:
            print(
                f"    [bwd] {name}: time_ms={ms:.4f}  simulate_gemm(m,n,k)=({m},{n},{k})  "
                f"batch={batch}  dtype={bd.gemm_dtype}"
            )
        print(f"    [bwd] expert_routed_GEMM_backward_total_ms: {bd.expert_bwd:.4f}")
        print(
            "  [MoE MLP] Origami note: legacy path multiplies per-expert ms by num_local_experts; "
            "totals above include that scaling where applicable."
        )
        sys.stdout.flush()

    def _grouped_gemm_shape_args_line(self, batch_size: int, seq_len: int) -> str:
        """Single log line: M/H/F, grouped_batch, token counts, dtype, layer GEMM pattern."""
        mc = self.config.model_config
        mp = self.config.model_parallel_config
        num_experts = mc.num_experts or 1
        ep = mp.expert_model_parallel_size or 1
        tp = mp.tensor_model_parallel_size or 1
        cp = mp.context_model_parallel_size or 1
        num_local = max(1, num_experts // max(ep, 1))
        use_turbo = getattr(mc, "enable_primus_turbo", False) and getattr(mc, "use_turbo_grouped_mlp", False)
        batch_tokens = batch_size * seq_len // tp // cp
        topk_tokens = batch_tokens * mc.moe_router_topk
        if mc.moe_ffn_hidden_size is not None:
            moe_ffn = mc.moe_ffn_hidden_size
        else:
            moe_ffn = mc.ffn_hidden_size
        tokens_per_expert = topk_tokens // max(num_local, 1)
        m_tokens = tokens_per_expert
        h = mc.hidden_size
        f_ffn = moe_ffn
        grouped_batch = num_local if use_turbo else 1
        gemm_dtype = "fp8" if getattr(mc, "fp8", None) else "bf16"
        if mc.swiglu:
            layer_pat = "SwiGLU 3xGEMM: gate+up (MxFxH), down (MxHxF)"
        else:
            layer_pat = "2xGEMM: up (MxFxH), down (MxHxF)"
        return (
            "  [MoE MLP] Grouped GEMM args from MoE block (config; whole-module GPU time is not split): "
            f"M={m_tokens}, H={h}, F={f_ffn}, grouped_batch={grouped_batch}, TP={tp}, CP={cp}, "
            f"batch_tokens={batch_tokens}, topk_tokens={topk_tokens}, router_topk={mc.moe_router_topk}, "
            f"dtype={gemm_dtype}; {layer_pat}"
        )

    def _log_grouped_gemm_context(self, batch_size: int, seq_len: int, source: str) -> None:
        """Rank-0: grouped-GEMM / MoE kernel flags (benchmark and simulation)."""
        if not _projection_rank0():
            return
        mc = self.config.model_config
        mp = self.config.model_parallel_config
        num_experts = mc.num_experts or 0
        if num_experts <= 0:
            return
        ep = mp.expert_model_parallel_size or 1
        num_local = max(1, num_experts // max(ep, 1))
        use_turbo = getattr(mc, "enable_primus_turbo", False) and getattr(mc, "use_turbo_grouped_mlp", False)
        sim_label = "Turbo_batched_origami" if use_turbo else "Legacy_sequential_origami"
        mgg = getattr(mc, "moe_grouped_gemm", False)
        legacy_gg = getattr(mc, "moe_use_legacy_grouped_gemm", False)
        print(
            f"  [MoE MLP] Grouped GEMM ({source}): Megatron moe_grouped_gemm={mgg}, "
            f"moe_use_legacy_grouped_gemm={legacy_gg}; "
            f"simulation expert-GEMM model={sim_label}; "
            f"local_experts={num_local}, EP={ep}, experts_total={num_experts} "
            f"(batch={batch_size}, seq_len={seq_len})"
        )
        print(self._grouped_gemm_shape_args_line(batch_size, seq_len))

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
        ep_size = self.config.model_parallel_config.expert_model_parallel_size or 1

        hidden_size = self.config.model_config.hidden_size
        batch_tokens = batch_size * seq_len // tp_size // cp_size
        topk = self.config.model_config.moe_router_topk
        topk_tokens = batch_tokens * topk

        if self.config.model_config.moe_ffn_hidden_size is not None:
            moe_ffn = self.config.model_config.moe_ffn_hidden_size
        else:
            moe_ffn = self.config.model_config.ffn_hidden_size

        num_experts = self.config.model_config.num_experts or 1
        num_local_experts = max(1, num_experts // ep_size)
        tokens_per_expert = topk_tokens // max(num_local_experts, 1)

        # FP8-hybrid: MoE expert MLP projections run in FP8
        gemm_dtype = "fp8" if getattr(self.config.model_config, "fp8", None) else "bf16"
        bytes_per_el = 1 if gemm_dtype == "fp8" else 2

        # ── 1. Routed expert GEMMs ──
        M = tokens_per_expert
        H = hidden_size
        F = moe_ffn

        is_rank_0 = _projection_rank0()
        if is_rank_0:
            self._log_grouped_gemm_context(batch_size, seq_len, "simulation")

        bd = self._expert_gemm_origami_breakdown(batch_size, seq_len, self._gemm_backend)
        expert_fwd, expert_bwd, use_turbo = bd.expert_fwd, bd.expert_bwd, bd.use_turbo

        if is_rank_0:
            mode = "Turbo (batched)" if use_turbo else "Legacy (sequential)"
            print(
                f"  [MoE MLP] Origami expert-GEMM layout: {mode} "
                f"(local_experts={num_local_experts}, M={M}, H={H}, F={F})"
            )
            sys.stdout.flush()

        if not use_turbo and is_rank_0:
            print(
                "  [MoE MLP] WARNING: Legacy grouped GEMM not properly modelled. "
                "Estimates may be inaccurate."
            )
            sys.stdout.flush()

        self._print_origami_expert_grouped_gemm_simulation(bd, batch_size, seq_len)

        fwd_time = expert_fwd
        bwd_time = expert_bwd

        # ── 2. Router overhead ──
        # Gate linear: [batch_tokens, num_experts, hidden_size]
        router_gemm = self._gemm_backend.simulate_gemm(batch_tokens, num_experts, hidden_size, gemm_dtype)
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
            if self._gemm_backend is not None and self._gemm_backend.hbm_bandwidth_gbps is not None
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
            act_bytes = 3 * topk_tokens * moe_ffn * bytes_per_el  # gate+up read, result write
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

    def _get_benchmark_results(self, batch_size: int, seq_len: int) -> tuple[float, float, int]:
        """Get or compute benchmark results (cached).

        When benchmarking (not simulating), uses decomposed MoE benchmarking
        to separately measure A2A communication time.  The A2A times are
        stored in ``self._a2a_fwd_ms`` / ``self._a2a_bwd_ms`` and can be
        retrieved via :meth:`measured_a2a_forward_time` /
        :meth:`measured_a2a_backward_time`.

        GPU benchmark mode does not emit MoE grouped-GEMM / Origami diagnostic
        prints (those are simulation-only).
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
                    backward_autograd_label=self.__class__.__name__,
                    backward_autograd_args=training_config_debug_one_line(self.config),
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
