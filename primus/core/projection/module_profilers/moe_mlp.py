###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from typing import Optional

from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.profiler_spec import ModuleProfilerSpec
from primus.core.projection.training_config import (
    TrainingConfig,
    gemm_dtype_from_config,
)

from .utils import benchmark_layer, benchmark_moe_layer_decomposed, v4_module_inputs

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
        # v4 correction: the routed expert outputs are weight-summed back to a
        # single [num_tokens, hidden] tensor (the combine step), so the stored
        # per-layer output activation is num_tokens*hidden, NOT topk_tokens*hidden
        # (the latter over-counted the combined output by moe_router_topk = 6x).
        output_memory = num_tokens * self.config.model_config.hidden_size * 2  # bf16
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

    def _permute_first_principles(
        self,
        *,
        batch_tokens: int,
        topk_tokens: int,
        hidden_size: int,
        bytes_per_el: int,
        ep_size: int,
        num_local_experts: int,
        peak_hbm: float,
    ) -> tuple[float, float, float, float]:
        """First-principles byte-traffic model of the MoE token permutation.

        Replaces the calibrated ``_PERMUTE_BW_FRACTION = 0.057`` fudge with an
        explicit enumeration of the real permutation kernels of the alltoall
        dispatcher (TE ``permutation.py``).  Each kernel is a ROW-WISE
        contiguous copy -- ``grid = (num_tokens, cdiv(hidden, BLOCK))`` copies a
        token's whole ``hidden``-wide row (7168 el = 7 KiB fp8 / 14 KiB bf16) to
        its (scattered) destination -- so the byte traffic is contiguous
        streaming, priced at ``_ACTIVATION_BW_FRACTION`` (the same sustained-BW
        fraction the SwiGLU activation uses), NOT the 5.7%-of-peak of a
        fine-grained random-access gather.

        Real kernels per MoE layer (alltoall dispatcher, ``num_local_experts>1``):

          dispatch (forward side):
            * ``permute``        : read input tokens once, write ``topk`` copies
                                   -> ``(batch + topk_tokens) * hidden`` (fp8 in).
            * ``sort_chunks``    : regroup post-A2A tokens by local expert
                                   (EP>1 only) -> ``2 * topk_tokens * hidden``.
          combine (backward side):
            * ``sort_chunks``    : reverse regroup -> ``2 * topk_tokens * hidden``
                                   on the bf16 expert outputs.
            * ``unpermute``      : weighted reduce ``topk`` copies -> 1
                                   -> ``(topk_tokens + batch) * hidden`` (bf16).

        Dispatch moves fp8 tokens (the quantized expert inputs); combine moves
        bf16 (expert GEMM outputs, reduced in bf16) -- so the two directions use
        different element sizes.  The A2A itself is communication and is modeled
        separately (transformer-layer A2A term), NOT here.

        A small fixed launch term covers the index / sort-map kernels
        (``make_row_id_map`` = 3 passes, ``make_chunk_sort_map``), which are
        latency-bound, not BW-bound.

        Scatter derate ``PRIMUS_MOE_PERMUTE_SCATTER`` (default 1.0) optionally
        knocks the streaming efficiency down for the random row-start addressing
        (TLB / L2), i.e. eff = ``_ACTIVATION_BW_FRACTION * scatter``.
        """
        eta = _ACTIVATION_BW_FRACTION * float(os.getenv("PRIMUS_MOE_PERMUTE_SCATTER", "1.0"))
        eff_bw = peak_hbm * eta  # GB/s (== 1e9 B/s units)
        bpe_disp = bytes_per_el  # fp8 (quantized expert inputs)
        bpe_comb = 2  # bf16 (expert outputs / grads reduced in bf16)
        has_sort = 1.0 if (ep_size > 1 and num_local_experts > 1) else 0.0
        launch_us = float(os.getenv("PRIMUS_MOE_PERMUTE_LAUNCH_US", "2.8"))

        # #10: dispatch (permute) AND combine (unpermute) each run in BOTH the
        # forward pass (dispatch -> experts -> combine) and the backward pass
        # (their VJPs).  The old model booked dispatch as fwd-only and combine as
        # bwd-only, ~2x undercounting the total permute traffic and mislabeling
        # the fwd/bwd attribution.  Element counts (local permute + EP sort_chunks):
        disp_elems = (batch_tokens + topk_tokens) * hidden_size + has_sort * (2 * topk_tokens) * hidden_size
        comb_elems = (topk_tokens + batch_tokens) * hidden_size + has_sort * (2 * topk_tokens) * hidden_size

        # Dispatch fwd moves fp8 tokens; every gradient pass moves bf16.
        dispatch_fwd = disp_elems * bpe_disp / (eff_bw * 1e6) + (4.0 * launch_us) / 1e3
        dispatch_bwd = disp_elems * bpe_comb / (eff_bw * 1e6) + (2.0 * launch_us) / 1e3
        combine_fwd = comb_elems * bpe_comb / (eff_bw * 1e6) + (2.0 * launch_us) / 1e3
        combine_bwd = comb_elems * bpe_comb / (eff_bw * 1e6) + (2.0 * launch_us) / 1e3

        return dispatch_fwd, dispatch_bwd, combine_fwd, combine_bwd

    def _expert_overhead_first_principles(
        self,
        *,
        topk_tokens: int,
        hidden_size: int,
        moe_ffn: int,
        num_local_experts: int,
        swiglu: bool,
        peak_hbm: float,
        batched: bool = False,
    ) -> tuple[float, float]:
        """First-principles model of the routed-expert *non-GEMM* overhead.

        Replaces the calibrated ``PRIMUS_MOE_EXPERT_OVH_FWD_MS/_BWD_MS`` fudge
        (0.19 / 0.66 ms per local expert) with an explicit accounting of the
        two costs the ideal GEMM sim leaves out for fp8 grouped experts:

          1. **JIT activation quant/dequant traffic.**  the GEMM cost model prices the
             expert GEMMs assuming their operands are *already* fp8; it does not
             charge the element-wise kernels that cast bf16 activations to fp8
             (and produce the transposed fp8 copy needed for wgrad) each step.
             These are contiguous streaming kernels, priced at the same
             sustained-BW fraction as the SwiGLU activation
             (``_ACTIVATION_BW_FRACTION``).

             Forward casts the GEMM *inputs* (bf16 read -> fp8 write = 3 B/el):
               * gate/up input  X  [topk_tokens, H]
               * down     input  h  [topk_tokens, F]
             (Summed over local experts ``sum(M)=topk_tokens``, so the traffic
             is independent of how many experts a rank holds.)

             Backward casts the grad-outputs of each projection, and fp8 wgrad
             needs a transposed fp8 copy too (bf16 read + fp8 write + fp8-T
             write = 4 B/el):
               * down grad-out [topk_tokens, H]
               * gate grad-out [topk_tokens, F]  (swiglu only)
               * up   grad-out [topk_tokens, F]

             Weights are quantized once per step and amortized across all tokens
             (not per-token traffic), so they are excluded here.

          2. **Per-group launch latency.**  The batched-GEMM sim charges a
             single kernel launch for all local experts, but the quant/dequant
             kernels and the grouped-GEMM group boundaries add a handful of
             latency-bound launches that do not scale with token count.  Priced
             at ``PRIMUS_MOE_EXPERT_LAUNCH_US`` (default = the GEMM fixed-startup
             0.75 us).

        Tile/wave quantization of the small per-expert ``M`` is already modeled
        by the GEMM cost model (it sweeps tiles and charges granularity), so it is NOT
        re-added here.
        """
        eta = _ACTIVATION_BW_FRACTION * float(os.getenv("PRIMUS_MOE_EXPERT_QUANT_EFF", "1.0"))
        eff_bw = peak_hbm * eta  # GB/s

        H = hidden_size
        F = moe_ffn

        # Forward: only the intermediate h (down-proj input, F-wide) needs a
        # bf16->fp8 cast (read 2 B + write 1 B = 3 B/el).  #8: the expert INPUT X
        # (H-wide) is ALREADY fp8 -- tokens are dispatched in fp8 by the permute
        # (bpe_disp=fp8) -- so re-quantizing ``topk_tokens*H`` was a double-count.
        fwd_quant_elems = topk_tokens * F
        fwd_bytes = fwd_quant_elems * 3

        # Backward: cast grad-outputs bf16->fp8 with transposed copy for wgrad
        # (read 2 B + write fp8 1 B + write fp8-T 1 B = 4 B/el).
        if swiglu:
            bwd_quant_elems = topk_tokens * (H + 2 * F)  # down:H, gate:F, up:F
        else:
            bwd_quant_elems = topk_tokens * (H + F)  # down:H, up:F
        bwd_bytes = bwd_quant_elems * 4

        fwd_ms = fwd_bytes / (eff_bw * 1e6)
        bwd_ms = bwd_bytes / (eff_bw * 1e6)

        # Per-group launch latency (BW-independent).  Forward casts ~2 buffers
        # (X, h); backward casts ~3 (swiglu) grad-outputs, each a small launch.
        launch_us = float(os.getenv("PRIMUS_MOE_EXPERT_LAUNCH_US", "0.75"))
        n_fwd_launch = 2.0
        n_bwd_launch = 3.0 if swiglu else 2.0
        # #8: the batched grouped-GEMM quant/dequant kernels launch a CONSTANT
        # number of times (one batched kernel spanning all local experts), NOT
        # once per expert.  Only the legacy sequential path pays per-expert
        # launches, so the num_local_experts (=48 at EP8) multiplier applies
        # only when NOT batched.
        n_groups = 1 if batched else num_local_experts
        fwd_ms += (n_groups * n_fwd_launch * launch_us) / 1e3
        bwd_ms += (n_groups * n_bwd_launch * launch_us) / 1e3

        return fwd_ms, bwd_ms

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

        # FP8-hybrid: MoE expert MLP projections run in FP8 (MX8 for mxfp8 recipe)
        gemm_dtype = gemm_dtype_from_config(self.config.model_config)
        bytes_per_el = 1 if gemm_dtype in ("fp8", "mx8") else 2

        # ── 1. Routed expert GEMMs ──
        M = tokens_per_expert
        H = hidden_size
        F = moe_ffn

        # Determine grouped-GEMM performance model.
        # Primus Turbo's grouped-GEMM kernel achieves near-ideal batched
        # execution → model as Origami batched GEMM (batch=num_local_experts).
        # Legacy grouped_gemm executes experts more sequentially → model as
        # individual GEMM (batch=1) × num_local_experts.
        #
        # The turbo grouped-MLP flag is surfaced under two names across the
        # Megatron/Primus stack (``use_turbo_grouped_mlp`` in YAML,
        # ``use_turbo_grouped_gemm`` in some configs); honor either.  The batched
        # (near-ideal) model is used whenever turbo is on and the config has not
        # explicitly opted into the legacy grouped_gemm kernel.
        _mc = self.config.model_config
        _turbo_on = getattr(_mc, "enable_primus_turbo", False)
        _turbo_grouped = getattr(_mc, "use_turbo_grouped_gemm", False) or getattr(
            _mc, "use_turbo_grouped_mlp", False
        )
        _wants_legacy = getattr(_mc, "moe_use_legacy_grouped_gemm", False)
        use_turbo = _turbo_on and _turbo_grouped and not _wants_legacy

        is_rank_0 = int(os.getenv("RANK", "0")) == 0
        if is_rank_0 and num_local_experts > 1:
            mode = "Turbo (batched)" if use_turbo else "Legacy (sequential)"
            print(
                f"  [MoE MLP] Grouped-GEMM model: {mode}"
                f"  ({num_local_experts} local experts, M={M}, H={H}, F={F})"
            )

        if use_turbo:
            # ── Turbo model: batched GEMM (all experts in parallel) ──
            B = num_local_experts
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

            expert_fwd = expert_fwd_ms * num_local_experts
            expert_bwd = expert_bwd_ms * num_local_experts

            # NOTE: Legacy grouped GEMM is not properly modelled. Origami
            # simulates ideal single-kernel execution
            if is_rank_0:
                print(
                    "  [MoE MLP] WARNING: Legacy grouped GEMM not properly modelled. "
                    "Estimates may be inaccurate."
                )

        # ── Grouped-GEMM non-GEMM overhead (fp8 quant/dequant + launch) ──
        # The ideal-FLOP GEMM sim prices the expert GEMMs assuming fp8 operands
        # and (in the batched path) a single launch; it misses the JIT
        # activation quant/dequant traffic and per-group launch latency.  These
        # are now derived from first principles (byte traffic + launch), which
        # replaces the old calibrated per-expert constants (0.19 / 0.66 ms).
        #
        # Opt back into the legacy calibrated constants with
        # ``PRIMUS_MOE_EXPERT_OVH_CALIBRATED=1``.
        _ovh_ref_hbm = 8000.0
        _tgt_hbm = (
            self._gemm_backend.hbm_bandwidth_gbps
            if self._gemm_backend is not None and self._gemm_backend.hbm_bandwidth_gbps is not None
            else _ovh_ref_hbm
        )
        _brk_gemm_f, _brk_gemm_b = expert_fwd, expert_bwd  # pure grouped-GEMM (pre small-problem overhead)
        if os.getenv("PRIMUS_MOE_EXPERT_OVH_CALIBRATED", "0") == "1":
            # Legacy: fixed per-local-expert ms, anchored at mi355x HBM (8 TB/s)
            # and scaled inversely with the target arch's HBM.
            moe_ovh_fwd = float(os.getenv("PRIMUS_MOE_EXPERT_OVH_FWD_MS", "0.19"))
            moe_ovh_bwd = float(os.getenv("PRIMUS_MOE_EXPERT_OVH_BWD_MS", "0.66"))
            _ovh_arch_scale = _ovh_ref_hbm / _tgt_hbm
            expert_fwd += num_local_experts * moe_ovh_fwd * _ovh_arch_scale
            expert_bwd += num_local_experts * moe_ovh_bwd * _ovh_arch_scale
        else:
            ovh_fwd_ms, ovh_bwd_ms = self._expert_overhead_first_principles(
                topk_tokens=topk_tokens,
                hidden_size=hidden_size,
                moe_ffn=moe_ffn,
                num_local_experts=num_local_experts,
                swiglu=bool(self.config.model_config.swiglu),
                peak_hbm=_tgt_hbm,
                batched=use_turbo,
            )
            expert_fwd += ovh_fwd_ms
            expert_bwd += ovh_bwd_ms

        fwd_time = expert_fwd
        bwd_time = expert_bwd

        # ── 2. Router overhead ──
        # Gate linear: [batch_tokens, num_experts, hidden_size]
        router_gemm = self._gemm_backend.simulate_gemm(batch_tokens, num_experts, hidden_size, gemm_dtype)
        router_fwd_ms = router_gemm.forward_time_ms
        # Softmax + top-K over the gate logits [batch_tokens, num_experts]: a
        # memory-bound reduction (read logits, softmax, write top-K weights),
        # priced first-principles at ~3 passes over the fp32 logits.  Replaces
        # the empirical ``0.1 + 0.002*num_experts`` constant (~0.87 ms/layer for
        # 384 experts) which was ~100x the real cost and layer-agnostic.
        _peak_hbm = getattr(self._gemm_backend, "hbm_bandwidth_gbps", None) or _FALLBACK_HBM_BW_GBPS
        _topk_eff_bw = _peak_hbm * _ACTIVATION_BW_FRACTION  # GB/s
        topk_overhead_ms = 3.0 * batch_tokens * num_experts * 4 / (_topk_eff_bw * 1e6)
        router_fwd_ms += topk_overhead_ms
        # Backward: dgrad + wgrad for gate linear (+ the same top-K traffic).
        # NOTE: the first ``num_hash_layers`` MoE layers STILL run the learned
        # gate GEMM (``v4_hash_router.py:214`` F.linear over all experts); only
        # the top-K *argmax* is replaced by a static bucket lookup.  So for hash
        # layers, zero ONLY ``topk_overhead_ms`` (keep ``router_gemm``) at the
        # model roll-up (per-layer index is not available here).  ~0.01 ms/layer,
        # negligible; do NOT zero the whole router term (that drops a live GEMM).
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
        activation_bw_gbps = peak_hbm * _ACTIVATION_BW_FRACTION

        if os.getenv("PRIMUS_MOE_PERMUTE_FIRST_PRINCIPLES", "1") == "1":
            # Default: first-principles byte-traffic model of the real dispatch/
            # combine kernels (local permute + EP sort_chunks + weighted
            # unpermute), each split into fwd + bwd (#10).
            dispatch_fwd, dispatch_bwd, combine_fwd, combine_bwd = self._permute_first_principles(
                batch_tokens=batch_tokens,
                topk_tokens=topk_tokens,
                hidden_size=hidden_size,
                bytes_per_el=bytes_per_el,
                ep_size=ep_size,
                num_local_experts=num_local_experts,
                peak_hbm=peak_hbm,
            )
        else:
            # Calibrated opt-out (PRIMUS_MOE_PERMUTE_FIRST_PRINCIPLES=0): single
            # local permute/unpermute at _PERMUTE_BW_FRACTION (5.7% of peak HBM).
            permute_eff_bw_gbps = peak_hbm * _PERMUTE_BW_FRACTION
            dispatch_bytes = (batch_tokens + topk_tokens) * hidden_size * bytes_per_el
            combine_bytes = (topk_tokens + batch_tokens) * hidden_size * bytes_per_el
            dispatch_fwd = dispatch_bytes / (permute_eff_bw_gbps * 1e6)
            combine_bwd = combine_bytes / (permute_eff_bw_gbps * 1e6)
            dispatch_bwd = combine_fwd = 0.0

        permute_fwd_ms = dispatch_fwd + combine_fwd
        permute_bwd_ms = dispatch_bwd + combine_bwd
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
        _brk_shared_f = _brk_shared_b = 0.0
        shared_sz = self.config.model_config.moe_shared_expert_intermediate_size
        if shared_sz:
            shared_result = self._gemm_backend.simulate_mlp_gemms(
                batch_tokens=batch_tokens,
                hidden_size=hidden_size,
                ffn_hidden_size=shared_sz,
                dtype=gemm_dtype,
                swiglu=self.config.model_config.swiglu,
            )
            _brk_shared_f = shared_result.forward_time_ms
            _brk_shared_b = shared_result.backward_time_ms
            fwd_time += _brk_shared_f
            bwd_time += _brk_shared_b

        # ── 6. ffn_hc mHC (block-level HyperMixer around the FFN sub-block) ──
        # #4: every V4 layer wraps BOTH attention (attn_hc, counted in the
        # attention profiler) AND the FFN (ffn_hc) in a HyperMixer.  ffn_hc was
        # entirely uncounted here -> the whole-layer mHC was only ~half.  Mirror
        # the attention mHC: expand/collapse bandwidth (bf16 residual) +
        # compute_weights (mapping_proj GEMM + RMS(K*D) + Sinkhorn).
        mhc_fwd = mhc_bwd = 0.0
        hc = getattr(self.config.model_config, "hc_mult", 1) or 1
        if hc > 1:
            eta_mhc = float(os.getenv("PRIMUS_ATTN_MISC_ETA", os.getenv("PRIMUS_ATTN_FP_ETA", "0.9")))
            M2 = batch_tokens
            mbpe = 2  # bf16 residual stream
            expand_f = (2 * hc + 1) * M2 * hidden_size * mbpe
            expand_b = (3 * hc + 2) * M2 * hidden_size * mbpe
            collapse_f = (hc + 1) * M2 * hidden_size * mbpe
            collapse_b = (hc + 1) * M2 * hidden_size * mbpe
            mhc_fwd = (expand_f + collapse_f) / (eta_mhc * peak_hbm * 1e6)
            mhc_bwd = (expand_b + collapse_b) / (eta_mhc * peak_hbm * 1e6)
            dt_mhc = gemm_dtype_from_config(self.config.model_config)
            cw = self._gemm_backend.simulate_gemm(
                M2, hc * hc + 2 * hc, hc * hidden_size, dt_mhc
            ).forward_time_ms
            cw_mem = 2.0 * M2 * hc * hidden_size * mbpe  # RMS over K*D-wide input
            mhc_fwd += cw + cw_mem / (eta_mhc * peak_hbm * 1e6)
            mhc_bwd += 2.0 * cw + 1.5 * cw_mem / (eta_mhc * peak_hbm * 1e6)
            n_iters = int(os.getenv("PRIMUS_HC_SINKHORN_ITERS", "20"))
            sink_ms = (1.0 + 2.0 * max(0, n_iters - 1)) * 0.75 / 1e3
            mhc_fwd += sink_ms
            mhc_bwd += sink_ms
            fwd_time += mhc_fwd
            bwd_time += mhc_bwd

        if os.getenv("PRIMUS_PRINT_LAYER_BREAKDOWN") and int(os.getenv("RANK", "0")) == 0:
            print(
                "[BRKDN] moe M=%d gemm_f=%.4f gemm_b=%.4f ovh_f=%.4f ovh_b=%.4f router_f=%.4f router_b=%.4f "
                "dispatch_f=%.4f dispatch_b=%.4f combine_f=%.4f combine_b=%.4f act_f=%.4f act_b=%.4f "
                "shared_f=%.4f shared_b=%.4f mhc_f=%.4f mhc_b=%.4f"
                % (
                    M,
                    _brk_gemm_f,
                    _brk_gemm_b,
                    expert_fwd - _brk_gemm_f,
                    expert_bwd - _brk_gemm_b,
                    router_fwd_ms,
                    router_bwd_ms,
                    dispatch_fwd,
                    dispatch_bwd,
                    combine_fwd,
                    combine_bwd,
                    activation_ms,
                    activation_ms,
                    _brk_shared_f,
                    _brk_shared_b,
                    mhc_fwd,
                    mhc_bwd,
                )
            )
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
                hidden = self.config.model_config.hidden_size
                tcfg = getattr(self.module, "config", None)
                # DeepSeek-V4 MoE: forward(hidden[B,S,D], *, token_ids[B,S]).
                # Feed V4-aware inputs (right layout + token_ids for hash routing).
                v4 = v4_module_inputs(self.module, batch_size, seq_len, hidden, 1, "moe")
                if v4 is not None:
                    # DeepseekV4MoE has no stock .dispatch/.combine to decompose
                    # A2A; benchmark the whole MoE forward. At EP=1 (single-GPU
                    # benchmark) A2A is ~0 and is restored analytically later.
                    ishapes, fkwargs = v4
                    fwd, bwd, act_mem = benchmark_layer(
                        self.module, ishapes, transformer_config=tcfg, forward_kwargs=fkwargs
                    )
                    a2a_fwd = a2a_bwd = 0.0
                else:
                    fwd, bwd, act_mem, a2a_fwd, a2a_bwd = benchmark_moe_layer_decomposed(
                        self.module,
                        [(seq_len, batch_size, hidden)],
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
