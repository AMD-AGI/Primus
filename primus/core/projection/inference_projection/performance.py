###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Inference performance projection: prefill + autoregressive decode.

This reuses the existing analytical profiler tree (in *simulation* mode, so
no GPU is required) to estimate **forward-only** per-component latency, then
composes those into the two serving phases:

  * **Prefill** — process the whole prompt (optionally in chunks) to produce
    the first token.  Drives **TTFT** (time-to-first-token).
  * **Decode** — generate ``output_seq_len`` tokens autoregressively, each
    step attending to a growing KV cache.  Drives **ITL / TPOT** and decode
    throughput.

Serving features modelled here: chunked prefill, KV-cache quantization
(via the SDPA/KV dtype), batching / concurrency, and speculative decoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from primus.core.projection.module_profilers.language_model import (
    build_profiler,
    get_language_model_profiler_spec,
)
from primus.core.projection.module_profilers.transformer_layer import (
    _estimate_moe_a2a_time_ms,
    _estimate_tp_allreduce_time_ms,
)
from primus.core.projection.simulation_backends.factory import (
    get_gemm_simulation_backend,
    get_sdpa_simulation_backend,
)
from primus.core.projection.training_config import InferenceConfig

from .collectives import (
    CommBreakdown,
    InferenceCollectiveModel,
    deepep_overlap_efficiency,
)


def _safe_forward(profiler, batch: int, seq_len: int) -> float:
    """Forward time of a sub-profiler, or 0 if it does not implement timing.

    Some element-wise profilers (LayerNorm, residual) only model memory and
    raise ``NotImplementedError`` for timing; those contributions are
    negligible for serving latency.
    """
    if profiler is None:
        return 0.0
    try:
        return float(profiler.measured_forward_time(batch, seq_len))
    except NotImplementedError:
        return 0.0


def _layers_on_rank(inference_config: InferenceConfig) -> int:
    mc = inference_config.model_config
    pp = max(1, inference_config.model_parallel_config.pipeline_model_parallel_size)
    return max(1, (mc.num_layers + pp - 1) // pp)


def _replica_gpus(inference_config: InferenceConfig) -> int:
    """GPUs in one model replica that serves a request.

    Latency-wise a request traverses TP×PP GPUs; for MoE the EP ranks live
    within that mesh, so we lower-bound the replica by EP.
    """
    mp = inference_config.model_parallel_config
    tp = max(1, mp.tensor_model_parallel_size)
    pp = max(1, mp.pipeline_model_parallel_size)
    ep = max(1, mp.expert_model_parallel_size)
    return max(tp * pp, ep)


@dataclass
class PhaseForwardTimes:
    """Forward latency (ms) of each component for one forward pass."""

    layers_ms: float
    embedding_ms: float
    final_norm_ms: float
    output_ms: float
    dense_layer_ms: float
    moe_layer_ms: float
    # Explicit communication (exposed, i.e. after overlap) for this forward.
    comm: CommBreakdown = field(default_factory=CommBreakdown)

    @property
    def total_ms(self) -> float:
        return (
            self.layers_ms
            + self.embedding_ms
            + self.final_norm_ms
            + self.output_ms
            + self.comm.pp_p2p_ms
        )


@dataclass
class InferencePerfResult:
    ttft_ms: float
    decode_total_ms: float
    itl_ms: float                 # inter-token latency per sequence (= TPOT)
    request_latency_ms: float     # TTFT + full decode for one sequence
    per_request_decode_tps: float
    decode_throughput_tps: float          # aggregate, whole batch
    decode_throughput_tps_per_gpu: float
    prefill_throughput_tps: float
    decode_step_latency_ms: float          # one decode forward (whole batch)
    replica_gpus: int
    # Disaggregation (feature A). ``is_disaggregated`` toggles the extra report.
    is_disaggregated: bool = False
    kv_transfer_ms: float = 0.0
    prefill_replica_gpus: int = 0
    decode_replica_gpus: int = 0
    extras: Dict[str, float] = field(default_factory=dict)


class InferencePerformanceProjector:
    """Builds the profiler once and answers prefill / decode timing queries."""

    def __init__(self, inference_config: InferenceConfig, args=None, benchmark_layer_times=None):
        self.cfg = inference_config
        self._args_ref = args
        gpu_arch = getattr(args, "gpu_arch", None) if args else None
        gpu_clock = getattr(args, "gpu_clock_mhz", None) if args else None
        gemm_name = getattr(args, "gemm_backend", None) if args else None

        self._gemm = get_gemm_simulation_backend(
            backend_name=gemm_name, gpu_arch=gpu_arch, gpu_clock_mhz=gpu_clock
        )
        self._sdpa = get_sdpa_simulation_backend(gpu_arch=gpu_arch, gpu_clock_mhz=gpu_clock)

        # Build profiler tree against a representative TrainingConfig view.
        view = inference_config.as_training_config(
            batch_size=inference_config.request_config.batch_size,
            seq_len=inference_config.request_config.input_seq_len,
        )
        self._view = view
        self._lm = build_profiler(get_language_model_profiler_spec(view))
        self._lm.set_simulation_backends(self._gemm, self._sdpa)

        mc = inference_config.model_config
        self._moe_pattern = mc.moe_pattern or [0] * mc.num_layers
        self._n_moe = sum(1 for x in self._moe_pattern if x)
        self._n_dense = mc.num_layers - self._n_moe

        # DeepEP / SyncFree EP-A2A compute-overlap fraction (0 = disabled).
        # Applied to the *builtin* comm path here; the explicit comm model
        # (``InferenceCollectiveModel``) applies the same factor internally.
        self._deepep_overlap = deepep_overlap_efficiency(mc)

        # MoE expert-routing imbalance multiplier (>= 1.0).  Real routing is
        # skewed, so the MoE step is gated by the busiest EP rank rather than
        # the perfectly-balanced average.  Only meaningful for an EP-sharded MoE
        # model; a no-op (1.0) otherwise.
        self._moe_imbalance = self._moe_imbalance_factor()

        # Kernel-backend (AITER/Triton/CK/HIP) attention multiplier, native
        # sparse-attention selection, and MoE expert-dtype (mxfp4/fp8/bf16)
        # compute speedup.  All affect the *simulation* path only (the measured
        # path bundles these into the whole-model step).  Defaults are no-ops.
        self._attn_backend_mult = inference_config.request_config.resolved_attention_backend_multiplier()
        self._moe_expert_speedup = inference_config.request_config.resolved_moe_expert_dtype_speedup()

        # Feature B: explicit, knob-driven communication model. When enabled we
        # replace the layer profiler's *implicit* TP-AllReduce / EP-AllToAll
        # cost with this model (delta applied per layer), enabling algorithm
        # selection, comm/compute overlap, fused-op speedups and a reportable
        # per-phase breakdown.
        self._cc = inference_config.collective_config
        self._comm = (
            InferenceCollectiveModel(mc, inference_config.model_parallel_config, self._cc)
            if (self._cc and self._cc.enabled)
            else None
        )

        # Benchmark mode (BENCHMARK-BASED PROJECTION — no calibration factors).
        # We use the *measured* silicon times directly as the projection. Two
        # measurement schemas are supported:
        #
        #   * whole-model (vLLM/SGLang): measured prefill / decode *step* latency
        #     (ms) for the full model, optionally swept over batch. Stored as
        #     per-phase (batch -> ms) curves; interpolated by concurrency.
        #   * per-layer (Megatron worker): measured forward time of one dense and
        #     one MoE layer per phase. Composed directly by layer counts.
        #
        # Empty => pure simulation.
        self._meas_whole: Dict[str, list] = {}   # {"prefill": [(batch, ms)], "decode": [...]}
        self._meas_prefill_rate_ms_per_tok: float = 0.0  # for sub-prompt prefill pieces
        self._meas_layer: Dict[tuple, float] = {}        # {(phase, ltype): ms}
        self._meas_ref_input: int = 0
        self._bench_backend: str = "megatron"
        self._bench_measured = benchmark_layer_times
        if benchmark_layer_times:
            self.set_benchmark_calibration(benchmark_layer_times)

    @property
    def is_benchmark_calibrated(self) -> bool:
        return bool(self._meas_whole or self._meas_layer)

    @property
    def _measured_mode(self) -> bool:
        return bool(self._meas_whole or self._meas_layer)

    @staticmethod
    def _interp(batch: int, pts: list) -> float:
        """Piecewise-linear interpolation of a sorted (batch, value) curve.

        Clamps to the endpoints outside the measured range so concurrencies
        below/above the swept batches reuse the nearest measured anchor.
        """
        if not pts:
            return 0.0
        if batch <= pts[0][0]:
            return pts[0][1]
        if batch >= pts[-1][0]:
            return pts[-1][1]
        for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
            if b0 <= batch <= b1:
                w = (batch - b0) / (b1 - b0) if b1 > b0 else 0.0
                return v0 + w * (v1 - v0)
        return pts[-1][1]

    # -- measured-time accessors (benchmark-based projection) ------------------

    def _measured_decode_step_ms(self, batch: int) -> float:
        """Measured whole-model / composed decode *step* latency at ``batch``."""
        if self._meas_whole.get("decode"):
            return self._interp(batch, self._meas_whole["decode"])
        # Per-layer schema: sum measured layer times by layer count.
        d = self._meas_layer.get(("decode", "dense"), 0.0)
        m = self._meas_layer.get(("decode", "moe"), 0.0)
        return self._n_dense * d + self._n_moe * m

    def _measured_full_prefill_ms(self, batch: int) -> float:
        """Measured whole-model / composed prefill latency for the full prompt."""
        if self._meas_whole.get("prefill"):
            return self._interp(batch, self._meas_whole["prefill"])
        d = self._meas_layer.get(("prefill", "dense"), 0.0)
        m = self._meas_layer.get(("prefill", "moe"), 0.0)
        return self._n_dense * d + self._n_moe * m

    def _measured_prefill_tokens_ms(self, total_tokens: int) -> float:
        """Measured prefill time for an arbitrary token count (chunk pieces).

        Prefill is compute-bound and ~linear in total processed tokens, so we
        scale by a measured per-token rate rather than re-simulating.
        """
        rate = self._meas_prefill_rate_ms_per_tok
        if rate <= 0:
            return 0.0
        return rate * max(1, total_tokens)

    # -- benchmark ingestion ---------------------------------------------------

    def _builtin_comm_ms(self, ltype: str, batch: int, q_len: int) -> float:
        """Implicit comm baked into the layer profiler's forward time."""
        tp_ar_one = _estimate_tp_allreduce_time_ms(self._view, batch, q_len)
        if ltype == "moe":
            return 2.0 * tp_ar_one + _estimate_moe_a2a_time_ms(self._view, batch, q_len, self._gemm)
        return 2.0 * tp_ar_one

    def set_benchmark_calibration(self, benchmark_layer_times: dict) -> None:
        """Ingest measured silicon times for a **benchmark-based** projection.

        No calibration factors are applied to the analytical model: the measured
        times are used *directly* as the projection. Two schemas are accepted:

        * whole-model (vLLM/SGLang)::

              {"backend": "vllm",
               "measured": {"model": {"prefill_ms", "decode_ms"}},
               "sweep": [{"batch", "prefill_ms", "decode_ms"}, ...],
               "meta": {"batch", "input_len", ...}}

          ``prefill_ms``/``decode_ms`` are full-model step latencies; ``sweep``
          gives the per-concurrency curve (preferred), interpolated by batch.

        * per-layer (Megatron worker)::

              {"measured": {"dense"|"moe": {"prefill_ms", "decode_ms"}},
               "meta": {"batch", "input_len"}}

          composed by layer counts.
        """
        if not benchmark_layer_times:
            return
        measured = benchmark_layer_times.get("measured", benchmark_layer_times)
        meta = benchmark_layer_times.get("meta", {})
        self._bench_backend = str(benchmark_layer_times.get("backend", "megatron"))
        ref_batch = int(meta.get("batch") or self.cfg.request_config.batch_size or 1)
        ref_input = int(meta.get("input_len") or self.cfg.request_config.input_seq_len or 1)

        self._meas_ref_input = ref_input

        # Whole-model schema (vLLM/SGLang): measured step latencies, used
        # DIRECTLY (no factor, no simulator). ``sweep`` gives the per-batch
        # curve; fall back to the single ``model`` anchor at ``ref_batch``.
        model_step = measured.get("model")
        if model_step:
            sweep = benchmark_layer_times.get("sweep") or []
            pre_pts, dec_pts = [], []
            for e in sweep:
                try:
                    b = int(e["batch"])
                except (KeyError, TypeError, ValueError):
                    continue
                if e.get("prefill_ms"):
                    pre_pts.append((b, float(e["prefill_ms"])))
                if e.get("decode_ms"):
                    dec_pts.append((b, float(e["decode_ms"])))
            if not pre_pts and model_step.get("prefill_ms"):
                pre_pts = [(ref_batch, float(model_step["prefill_ms"]))]
            if not dec_pts and model_step.get("decode_ms"):
                dec_pts = [(ref_batch, float(model_step["decode_ms"]))]
            self._meas_whole = {
                k: v for k, v in (("prefill", sorted(pre_pts)), ("decode", sorted(dec_pts))) if v
            }
            # Per-token prefill rate (for sub-prompt chunk pieces): full-prompt
            # prefill of ``b`` seqs processes ``b * ref_input`` tokens.
            if pre_pts and ref_input > 0:
                rates = [ms / (b * ref_input) for b, ms in pre_pts if b > 0]
                self._meas_prefill_rate_ms_per_tok = sum(rates) / len(rates) if rates else 0.0
            return

        # Per-layer schema (Megatron worker): measured forward time of one dense
        # and one MoE layer per phase. Used directly, composed by layer counts.
        layer: Dict[tuple, float] = {}
        for ltype in ("dense", "moe"):
            entry = measured.get(ltype)
            if not entry:
                continue
            if entry.get("prefill_ms"):
                layer[("prefill", ltype)] = float(entry["prefill_ms"])
            if entry.get("decode_ms"):
                layer[("decode", ltype)] = float(entry["decode_ms"])
        self._meas_layer = layer
        # Prefill rate from the dominant (per-layer * count) prefill total.
        if ref_input > 0:
            full_pre = self._measured_full_prefill_ms(ref_batch)
            self._meas_prefill_rate_ms_per_tok = full_pre / (ref_batch * ref_input)

    # -- per-pass forward time -------------------------------------------------

    def _moe_imbalance_factor(self) -> float:
        """MoE expert-compute imbalance multiplier (>= 1.0).

        Only EP-sharded MoE models (``num_experts > 0`` and ``EP > 1``) see
        routing imbalance; for everything else this is a no-op (1.0).  The
        magnitude (and the ``redundant_experts`` mitigation) is resolved on the
        request config, given the model's expert count.
        """
        mc = self.cfg.model_config
        num_experts = int(getattr(mc, "num_experts", 0) or 0)
        ep = max(1, self.cfg.model_parallel_config.expert_model_parallel_size)
        if num_experts <= 0 or ep <= 1:
            return 1.0
        return self.cfg.request_config.resolved_ep_imbalance(num_experts)

    def _forward_times(self, batch: int, q_len: int, phase: str, kv_len: int) -> PhaseForwardTimes:
        lm = self._lm
        lm.set_inference_phase(phase, kv_len)

        dense_p = lm.sub_profilers.get("dense_transformer_layer")
        moe_p = lm.sub_profilers.get("moe_transformer_layer")

        has_dense = bool(self._n_dense and dense_p)
        has_moe = bool(self._n_moe and moe_p)

        dense_raw = dense_p.measured_forward_time(batch, q_len) if has_dense else 0.0
        moe_raw = moe_p.measured_forward_time(batch, q_len) if has_moe else 0.0

        # Split implicit per-layer comm out of the raw forward time so it can be
        # handled explicitly below.  Doing this unconditionally (not only when
        # the explicit comm model is active) lets the benchmark calibration
        # scale the *compute* part without disturbing communication cost.
        builtin_dense_comm = self._builtin_comm_ms("dense", batch, q_len) if has_dense else 0.0
        builtin_moe_comm = self._builtin_comm_ms("moe", batch, q_len) if has_moe else 0.0
        dense_compute = max(0.0, dense_raw - builtin_dense_comm) if has_dense else 0.0
        moe_compute = max(0.0, moe_raw - builtin_moe_comm) if has_moe else 0.0

        # MoE expert-MLP (grouped-GEMM) adjustments — applied only to the
        # expert-MLP portion of the layer (attention, router and comm are
        # unaffected):
        #   * routing imbalance (>= 1.0): the MoE step is gated by the busiest
        #     EP rank, which does ``imbalance``x the average expert work;
        #   * expert dtype speedup (<= 1.0): low-precision expert kernels
        #     (mxfp4 / fp8) run the grouped-GEMM faster.
        # These compose multiplicatively. No-op when balanced + bf16 / non-MoE.
        if (
            has_moe
            and (self._moe_imbalance > 1.0 or self._moe_expert_speedup != 1.0)
            and hasattr(moe_p, "get_sub_profiler")
        ):
            mlp_p = moe_p.get_sub_profiler("mlp")
            if mlp_p is not None:
                expert_mlp_ms = mlp_p.measured_forward_time(batch, q_len)
                new_expert = expert_mlp_ms * self._moe_imbalance * self._moe_expert_speedup
                moe_compute += new_expert - expert_mlp_ms

        # Kernel-backend (attention library) + native-sparse-attention: adjust
        # only the attention sub-profiler's compute.  ``attn_mult`` scales the
        # whole attention forward (Triton baseline = 1.0); ``sparse_scale``
        # shrinks attention toward ``topk/context`` for long contexts (NSA).
        sparse_scale = self.cfg.request_config.resolved_sparse_attention_scale(kv_len)
        if self._attn_backend_mult != 1.0 or sparse_scale != 1.0:
            factor = self._attn_backend_mult * sparse_scale
            if has_dense:
                ad = dense_p.get_sub_profiler("self_attention") if hasattr(dense_p, "get_sub_profiler") else None
                if ad is not None:
                    a = ad.measured_forward_time(batch, q_len)
                    dense_compute = max(0.0, dense_compute + a * (factor - 1.0))
            if has_moe:
                am = moe_p.get_sub_profiler("self_attention") if hasattr(moe_p, "get_sub_profiler") else None
                if am is not None:
                    a = am.measured_forward_time(batch, q_len)
                    moe_compute = max(0.0, moe_compute + a * (factor - 1.0))

        comm = CommBreakdown()
        if self._comm is not None:
            # Feature B: explicit, knob-driven communication model.
            overlap = (
                float(self._cc.prefill_overlap)
                if phase == "prefill"
                else float(self._cc.decode_overlap)
            )
            overlap = min(max(overlap, 0.0), 1.0)
            keep = 1.0 - overlap

            new_tp_ar = self._comm.tp_allreduce_ms(batch, q_len)
            new_ep_a2a = self._comm.ep_a2a_ms(batch, q_len)

            dense_fwd = dense_compute + (new_tp_ar * keep if has_dense else 0.0)
            moe_fwd = moe_compute + ((new_tp_ar + new_ep_a2a) * keep if has_moe else 0.0)

            comm.tp_allreduce_ms = (self._n_dense + self._n_moe) * new_tp_ar * keep
            comm.ep_a2a_ms = self._n_moe * new_ep_a2a * keep
            comm.pp_p2p_ms = self._comm.pp_p2p_ms(batch, q_len) * keep
        else:
            # Implicit comm: add the built-in cost back onto (calibrated)
            # compute. When DeepEP/SyncFree is enabled, the EP A2A overlaps
            # expert compute, so charge only the exposed (non-overlapped)
            # fraction of the raw A2A baked into the layer time.
            eff_moe_comm = builtin_moe_comm
            if self._deepep_overlap > 0 and has_moe:
                a2a_raw = _estimate_moe_a2a_time_ms(self._view, batch, q_len, self._gemm)
                eff_moe_comm = builtin_moe_comm - a2a_raw * self._deepep_overlap
            dense_fwd = dense_compute + builtin_dense_comm
            moe_fwd = moe_compute + eff_moe_comm

        layers = self._n_dense * dense_fwd + self._n_moe * moe_fwd

        emb = _safe_forward(lm.sub_profilers.get("embedding"), batch, q_len)
        # The final LayerNorm is element-wise and not separately timed by the
        # profiler (training does not measure it either) — treat as ~0.
        fnorm = _safe_forward(lm.sub_profilers.get("final_layernorm"), batch, q_len)
        # LM head only materialises logits for the token(s) being sampled.
        # Prefill samples 1 token; decode samples 1 per step.  Speculative
        # decode verifies q_len tokens, so size the head by q_len there.
        head_tokens = q_len if phase == "decode" else 1
        out = _safe_forward(lm.sub_profilers.get("output_layer"), batch, head_tokens)

        return PhaseForwardTimes(
            layers_ms=layers,
            embedding_ms=emb,
            final_norm_ms=fnorm,
            output_ms=out,
            dense_layer_ms=dense_fwd,
            moe_layer_ms=moe_fwd,
            comm=comm,
        )

    # -- prefill ---------------------------------------------------------------

    def prefill_latency_ms(self, batch: int, input_len: int) -> float:
        """Time to process the prompt (→ first token).  Honors chunked prefill."""
        # Benchmark-based: use the measured full-prompt prefill step directly.
        if self._measured_mode:
            if self._meas_whole.get("prefill") or self._meas_layer:
                # Measured anchor is at ``ref_input``; scale by the per-token
                # prefill rate when the requested prompt differs in length.
                if self._meas_ref_input and input_len != self._meas_ref_input:
                    return self._measured_prefill_tokens_ms(batch * input_len)
                return self._measured_full_prefill_ms(batch)

        chunk = int(self.cfg.request_config.chunked_prefill_size or 0)
        if chunk <= 0 or chunk >= input_len:
            ft = self._forward_times(batch, input_len, "prefill", input_len)
            return ft.total_ms

        # Chunked prefill: each chunk attends to all preceding context.
        total = 0.0
        processed = 0
        while processed < input_len:
            this = min(chunk, input_len - processed)
            kv_len = processed + this
            ft = self._forward_times(batch, this, "prefill", kv_len)
            total += ft.total_ms
            processed += this
        return total

    # -- decode ----------------------------------------------------------------

    def _decode_step_overhead_ms(self) -> float:
        """Fixed per-step host/launch overhead (CUDA-graph-reducible)."""
        return max(0.0, self.cfg.request_config.resolved_decode_step_overhead_us()) / 1000.0

    def _draft_overhead_ms(self, per_token_step_ms: float) -> float:
        """Speculative draft-model forward cost added to a verify step.

        The draft runs ``speculative_num_tokens`` times per verify step; each
        draft pass costs ``speculative_draft_cost_factor`` of one target decode
        token.  ``0`` for either knob is a no-op (legacy behaviour that only
        credited the accepted-token speedup).
        """
        req = self.cfg.request_config
        spec_k = int(req.speculative_num_tokens or 0)
        dcf = float(req.speculative_draft_cost_factor or 0.0)
        if spec_k > 0 and dcf > 0.0:
            return dcf * spec_k * max(0.0, per_token_step_ms)
        return 0.0

    def _decode_step_latency_ms(self, batch: int, kv_len: int, q_len: int = 1) -> float:
        # Benchmark-based: use the measured decode step directly (memory-bound,
        # ~flat in context over a generation, so no simulator context-scaling).
        if self._measured_mode:
            per_token = self._measured_decode_step_ms(batch)
            step = per_token * q_len if q_len > 1 else per_token  # verify q_len tokens/step
            return step + self._draft_overhead_ms(per_token) + self._decode_step_overhead_ms()
        ft = self._forward_times(batch, q_len, "decode", kv_len)
        per_token = ft.total_ms / max(1, q_len)
        return ft.total_ms + self._draft_overhead_ms(per_token) + self._decode_step_overhead_ms()

    # -- DES event-duration kernel --------------------------------------------
    # Public wrappers used by the discrete-event simulator (``des.py``) so that
    # each simulated step's duration is drawn from this (possibly
    # benchmark-calibrated) cost model — i.e. "benchmark calibration inside a
    # DES". They mirror the pure/mixed step costs the steady-state
    # ``_continuous_decode_metrics`` blends analytically.

    def decode_step_latency_ms(self, batch: int, kv_len: int, q_len: int = 1) -> float:
        """One pure-decode step over ``batch`` resident sequences."""
        return self._decode_step_latency_ms(max(1, batch), max(1, kv_len), q_len)

    def mixed_step_latency_ms(
        self,
        num_decode: int,
        chunk_tokens: int,
        decode_ctx: int,
        prefill_kv_len: int,
        q_len: int = 1,
    ) -> float:
        """One scheduler step carrying a prefill chunk plus ``num_decode``
        concurrent decodes (``num_decode == 0`` → a pure prefill-chunk step)."""
        penalty = max(0.0, self.cfg.request_config.resolved_mixed_batch_penalty())
        ov = self._decode_step_overhead_ms()
        chunk_tokens = max(1, int(chunk_tokens))
        num_decode = max(0, int(num_decode))
        if self._measured_mode:
            spec = q_len if q_len > 1 else 1
            prefill_piece = self._measured_prefill_tokens_ms(chunk_tokens)
            dec_piece = self._measured_decode_step_ms(num_decode) * spec if num_decode > 0 else 0.0
            return (prefill_piece + dec_piece) * (1.0 + penalty) + ov
        prefill_piece = self._forward_times(1, chunk_tokens, "prefill", max(1, prefill_kv_len)).total_ms
        dec_piece = (
            self._forward_times(num_decode, q_len, "decode", max(1, decode_ctx)).total_ms
            if num_decode > 0
            else 0.0
        )
        return (prefill_piece + dec_piece) * (1.0 + penalty) + ov

    def decode_total_ms(self, batch: int, input_len: int, output_len: int) -> float:
        """Integrate per-token decode latency over the growing KV cache.

        Per-step latency grows slowly with context, so we sample a handful of
        context lengths and trapezoid-integrate rather than simulating every
        one of ``output_len`` steps.
        """
        if output_len <= 0:
            return 0.0

        spec_k = int(self.cfg.request_config.speculative_num_tokens or 0)
        accept = float(self.cfg.request_config.speculative_acceptance_rate or 0.0)
        q_len = (spec_k + 1) if spec_k > 0 else 1
        # Expected accepted tokens per verify step (geometric series).
        if spec_k > 0 and 0.0 < accept < 1.0:
            tokens_per_step = (1.0 - accept ** (spec_k + 1)) / (1.0 - accept)
        elif spec_k > 0 and accept >= 1.0:
            tokens_per_step = spec_k + 1
        else:
            tokens_per_step = 1.0

        num_steps = max(1.0, output_len / tokens_per_step)

        # Sample step latency across [input_len, input_len + output_len].
        n_samples = min(8, max(2, int(output_len)))
        ctx_lo, ctx_hi = input_len, input_len + output_len
        samples = []
        for i in range(n_samples):
            frac = i / (n_samples - 1) if n_samples > 1 else 0.0
            ctx = int(ctx_lo + frac * (ctx_hi - ctx_lo))
            samples.append(self._decode_step_latency_ms(batch, ctx, q_len=q_len))
        avg_step = sum(samples) / len(samples)
        return avg_step * num_steps

    # -- continuous batching (steady-state TPOT) -------------------------------

    def _continuous_decode_metrics(self, input_len: int, output_len: int, concurrency: int) -> Dict[str, float]:
        """Steady-state decode under *continuous batching*.

        Real servers (vLLM, SGLang, ...) keep ``concurrency`` sequences resident
        and admit a new request's prefill the moment one finishes.  That makes a
        fraction of scheduler steps **mixed** (1 prefill chunk + ``C-1`` decode)
        which are far slower per token than a uniform **pure-decode** step — the
        "TPOT pollution" effect.  This models the blended steady state.

        Accounting (per admitted request, the ``R`` factor cancels):
          * a request's prefill is processed in ``n_chunks`` mixed steps;
          * pure steps emit ``C * tok/step`` decode tokens, mixed steps emit
            ``(C-1) * tok/step``;
          * total decode tokens per request = ``OSL``.
        From the per-request window time ``T`` we get
        ``TPOT = C * T / OSL`` and system ``throughput = 1000 * OSL / T``.
        """
        req = self.cfg.request_config
        ISL = max(1, input_len)
        OSL = max(1, output_len)
        C = max(1, int(concurrency))

        spec_k = int(req.speculative_num_tokens or 0)
        q_len = (spec_k + 1) if spec_k > 0 else 1
        tok_per_step = max(1e-6, self._spec_tokens_per_step())

        # Prefill of a newly-admitted request is split into chunks; with chunked
        # prefill each mixed step carries only one chunk (less pollution/step).
        chunk = int(req.chunked_prefill_size or 0)
        if chunk <= 0 or chunk >= ISL:
            n_chunks = 1
            chunk_tokens = ISL
        else:
            n_chunks = max(1, math.ceil(ISL / chunk))
            chunk_tokens = chunk

        # Scheduler per-step token budget (vLLM ``max_num_batched_tokens``). A
        # mixed step processes the prefill chunk PLUS the decode tokens of the
        # other ``C-1`` running sequences; that sum cannot exceed the cap. When
        # it would, the prefill admitted per step is bounded by the leftover
        # budget, so the prompt is split into more (smaller) prefill chunks →
        # more mixed steps → higher TPOT / lower throughput. First-order model:
        # clamp the effective prefill chunk to ``cap - decode_tokens`` and
        # recompute the chunk count. ``0`` = unlimited (path unchanged).
        cap = int(req.max_num_batched_tokens or 0)
        if cap > 0:
            decode_tokens_mixed = max(0, C - 1) * int(q_len)
            # Always make at least one token of prefill progress per step so the
            # model stays finite even if decode tokens alone saturate the cap.
            eff_chunk = min(chunk_tokens, max(1, cap - decode_tokens_mixed))
            if eff_chunk < chunk_tokens:
                chunk_tokens = eff_chunk
                n_chunks = max(1, math.ceil(ISL / eff_chunk))

        penalty = max(0.0, req.resolved_mixed_batch_penalty())
        ov = self._decode_step_overhead_ms()

        if self._measured_mode:
            # Benchmark-based: measured decode is ~flat in context, so the pure
            # and mixed steps are constant across the generation window.
            spec = q_len if q_len > 1 else 1
            draft = self._draft_overhead_ms(self._measured_decode_step_ms(C))
            t_pure = self._measured_decode_step_ms(C) * spec + draft + ov
            prefill_piece = self._measured_prefill_tokens_ms(chunk_tokens)
            dec_piece = self._measured_decode_step_ms(max(1, C - 1)) * spec
            t_mixed = (prefill_piece + dec_piece) * (1.0 + penalty) + ov
        else:
            # Simulation: average pure/mixed step latency over the (uniform)
            # context distribution [ISL, ISL+OSL].
            n_samples = min(8, max(2, int(OSL)))
            ctx_lo, ctx_hi = ISL, ISL + OSL
            pure, mixed = [], []
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                ctx = int(ctx_lo + frac * (ctx_hi - ctx_lo))
                pure_fwd = self._forward_times(C, q_len, "decode", ctx).total_ms
                t_pure = pure_fwd + self._draft_overhead_ms(pure_fwd / max(1, q_len)) + ov
                prefill_piece = self._forward_times(1, chunk_tokens, "prefill", min(ctx, ISL)).total_ms
                dec_piece = self._forward_times(max(1, C - 1), q_len, "decode", ctx).total_ms
                t_mixed = (prefill_piece + dec_piece) * (1.0 + penalty) + ov
                pure.append(t_pure)
                mixed.append(t_mixed)
            t_pure = sum(pure) / len(pure)
            t_mixed = sum(mixed) / len(mixed)

        # Pure steps per request needed to make up the decode tokens the mixed
        # steps did not cover.
        mixed_tokens = n_chunks * (C - 1) * tok_per_step
        n_pure = max(0.0, (OSL - mixed_tokens) / (C * tok_per_step))
        window_ms = n_pure * t_pure + n_chunks * t_mixed
        if window_ms <= 0:
            window_ms = t_pure

        tpot_ms = C * window_ms / OSL
        system_tps = 1000.0 * OSL / window_ms
        decode_total_ms = tpot_ms * OSL
        total_steps = n_pure + n_chunks
        mixed_fraction = (n_chunks / total_steps) if total_steps > 0 else 0.0
        pollution_pct = (n_chunks * t_mixed / window_ms * 100.0) if window_ms > 0 else 0.0

        return {
            "tpot_ms": tpot_ms,
            "decode_total_ms": decode_total_ms,
            "system_tps": system_tps,
            "pure_step_ms": t_pure,
            "mixed_step_ms": t_mixed,
            "mixed_step_fraction": mixed_fraction,
            "tpot_pollution_pct": pollution_pct,
            "concurrency": float(C),
        }

    def _request_rate_queueing(
        self, system_decode_tps: float, output_len: int, ttft_ms: float, request_latency_ms: float
    ) -> Dict[str, float]:
        """First-order open-loop queueing delay for a given offered load.

        Closed-loop (``request_rate == 0`` or ``arrival_model == "closed"``) is
        the legacy behaviour and returns ``{}`` (no adjustment).  Otherwise the
        engine sustains a finite request-completion rate ``mu`` (decode-bound:
        ``system_decode_tps / OSL``); the offered rate ``lambda`` gives a
        utilization ``rho = lambda / mu`` and a queue-wait that is added to TTFT
        and end-to-end latency:

          * poisson      → M/M/1: ``Wq = rho/(1-rho) * (1/mu)``
          * deterministic→ ~D/M/1: roughly half the M/M/1 wait

        At/above saturation (``rho >= 1``) the queue is unbounded; we report a
        large finite penalty + a ``saturated`` flag so the agent ranks it last.
        """
        req = self.cfg.request_config
        rate = float(req.request_rate or 0.0)
        model = (req.arrival_model or "closed").lower()
        osl = max(1, output_len)
        # "none" is an alias of "closed"; "trace" has no closed-form rate and is
        # handled by the DES, so the analytical queue is a no-op for both.
        if rate <= 0.0 or model in ("closed", "none", "trace"):
            return {}
        mu = system_decode_tps / osl if system_decode_tps > 0 else 0.0
        if mu <= 0.0:
            return {}
        rho = rate / mu
        ts_ms = 1000.0 / mu  # mean service time per request
        out: Dict[str, float] = {
            "offered_request_rate": rate,
            "max_sustainable_request_rate": mu,
            "utilization": rho,
        }
        if rho >= 1.0:
            out["saturated"] = 1.0
            wq_ms = ts_ms * 1000.0  # large but finite penalty
        else:
            out["saturated"] = 0.0
            wq_ms = rho / (1.0 - rho) * ts_ms
            if model == "deterministic":
                wq_ms *= 0.5
            wq_ms = min(wq_ms, ts_ms * 1000.0)
        out["queue_wait_ms"] = wq_ms
        out["ttft_with_queue_ms"] = ttft_ms + wq_ms
        out["request_latency_with_queue_ms"] = request_latency_ms + wq_ms
        return out

    def _use_continuous_batching(self, concurrency: int, output_len: int) -> bool:
        model = (self.cfg.request_config.serving_model or "continuous").lower()
        # With a single resident sequence there are no concurrent mixed batches,
        # so continuous batching degenerates to the static (pure-decode) case.
        return model == "continuous" and concurrency > 1 and output_len > 0

    # -- comm reporting --------------------------------------------------------

    def _spec_tokens_per_step(self) -> float:
        req = self.cfg.request_config
        spec_k = int(req.speculative_num_tokens or 0)
        accept = float(req.speculative_acceptance_rate or 0.0)
        if spec_k > 0 and 0.0 < accept < 1.0:
            return (1.0 - accept ** (spec_k + 1)) / (1.0 - accept)
        return float(spec_k + 1 if spec_k > 0 else 1)

    def _comm_extras(self, batch: int, input_len: int, output_len: int) -> Dict[str, float]:
        """Representative per-phase comm breakdown (ms) for reporting."""
        if self._comm is None:
            return {}
        pre = self._forward_times(batch, input_len, "prefill", input_len)
        spec_k = int(self.cfg.request_config.speculative_num_tokens or 0)
        q_len = (spec_k + 1) if spec_k > 0 else 1
        dec = self._forward_times(batch, q_len, "decode", input_len + output_len // 2)
        return {
            "comm_prefill_tp_allreduce_ms": pre.comm.tp_allreduce_ms,
            "comm_prefill_ep_a2a_ms": pre.comm.ep_a2a_ms,
            "comm_prefill_pp_p2p_ms": pre.comm.pp_p2p_ms,
            "comm_prefill_total_ms": pre.comm.total_ms,
            "comm_decode_tp_allreduce_ms": dec.comm.tp_allreduce_ms,
            "comm_decode_ep_a2a_ms": dec.comm.ep_a2a_ms,
            "comm_decode_pp_p2p_ms": dec.comm.pp_p2p_ms,
            "comm_decode_total_ms": dec.comm.total_ms,
        }

    # -- top level -------------------------------------------------------------

    def project(self) -> InferencePerfResult:
        if self.cfg.disaggregation_config and self.cfg.disaggregation_config.enabled:
            return self._project_disaggregated()
        return self._project_colocated()

    def _project_colocated(self) -> InferencePerfResult:
        req = self.cfg.request_config
        batch = max(1, req.batch_size)
        input_len = max(1, req.input_seq_len)
        output_len = max(0, req.output_seq_len)

        concurrency = max(1, req.resolved_max_concurrency())
        spec_k = int(req.speculative_num_tokens or 0)
        q_len = (spec_k + 1) if spec_k > 0 else 1
        replica_gpus = _replica_gpus(self.cfg)

        ttft = self.prefill_latency_ms(batch, input_len)
        extras = {"speculative_tokens_per_step": self._spec_tokens_per_step()}

        if self._use_continuous_batching(concurrency, output_len):
            # Continuous batching: TPOT is the blended pure/mixed steady state.
            m = self._continuous_decode_metrics(input_len, output_len, concurrency)
            decode_total = m["decode_total_ms"]
            itl = m["tpot_ms"]
            step_latency = m["pure_step_ms"]
            decode_tps = m["system_tps"]
            per_req_decode_tps = (1000.0 / itl) if itl > 0 else 0.0
            extras.update(
                {
                    "serving_continuous_batching": 1.0,
                    "concurrency": m["concurrency"],
                    "pure_step_latency_ms": m["pure_step_ms"],
                    "mixed_step_latency_ms": m["mixed_step_ms"],
                    "mixed_step_fraction": m["mixed_step_fraction"],
                    "tpot_pollution_pct": m["tpot_pollution_pct"],
                }
            )
        else:
            decode_total = self.decode_total_ms(batch, input_len, output_len)
            mid_ctx = input_len + output_len // 2
            step_latency = self._decode_step_latency_ms(batch, mid_ctx, q_len=q_len)
            itl = (decode_total / output_len) if output_len > 0 else step_latency
            per_req_decode_tps = (1000.0 / itl) if itl > 0 else 0.0
            decode_tps = (batch * 1000.0 / step_latency) if step_latency > 0 else 0.0

        request_latency = ttft + decode_total
        decode_tps_per_gpu = decode_tps / replica_gpus if replica_gpus else 0.0
        prefill_tps = (batch * input_len * 1000.0 / ttft) if ttft > 0 else 0.0

        # Offered-load queueing (open-loop). Adjusts TTFT + e2e latency only;
        # decode throughput / TPOT are steady-state and unaffected. No-op unless
        # a request rate is set. Computed from compute-TTFT (prefill_tps above
        # uses the pre-queue TTFT, which is correct).
        q = self._request_rate_queueing(decode_tps, output_len, ttft, request_latency)
        if q:
            ttft = q["ttft_with_queue_ms"]
            request_latency = q["request_latency_with_queue_ms"]
            extras.update(q)

        extras.update(self._comm_extras(batch, input_len, output_len))
        if self.is_benchmark_calibrated:
            extras["benchmark_calibrated"] = 1.0

        return InferencePerfResult(
            ttft_ms=ttft,
            decode_total_ms=decode_total,
            itl_ms=itl,
            request_latency_ms=request_latency,
            per_request_decode_tps=per_req_decode_tps,
            decode_throughput_tps=decode_tps,
            decode_throughput_tps_per_gpu=decode_tps_per_gpu,
            prefill_throughput_tps=prefill_tps,
            decode_step_latency_ms=step_latency,
            replica_gpus=replica_gpus,
            extras=extras,
        )

    def _kv_transfer_ms(self, decode_proj: "InferencePerformanceProjector", batch: int, input_len: int) -> float:
        """KV-cache transfer time prefill→decode worker (per matching rank)."""
        from .kv_cache import estimate_kv_cache

        disagg = self.cfg.disaggregation_config
        layers_on_rank = _layers_on_rank(decode_proj.cfg)
        kv = estimate_kv_cache(
            decode_proj.cfg, layers_on_rank, concurrency=batch, context_len=input_len
        )
        comm = decode_proj._comm or self._comm
        if comm is None:
            # No collective model available; fall back to a direct bytes/bw calc.
            from .collectives import InferenceCollectiveModel

            comm = InferenceCollectiveModel(
                decode_proj.cfg.model_config,
                decode_proj.cfg.model_parallel_config,
                decode_proj.cfg.collective_config,
            )
        return comm.kv_transfer_ms(
            kv.bytes_total,
            bw_gbps=disagg.resolved_kv_transfer_bw_gbps(),
            latency_us=disagg.resolved_kv_transfer_latency_us(),
        )

    def _project_disaggregated(self) -> InferencePerfResult:
        from dataclasses import replace

        req = self.cfg.request_config
        batch = max(1, req.batch_size)
        input_len = max(1, req.input_seq_len)
        output_len = max(0, req.output_seq_len)
        disagg = self.cfg.disaggregation_config
        mp = self.cfg.model_parallel_config

        # Build dedicated prefill / decode projectors with per-pool parallelism.
        # Disable disaggregation on the sub-configs to avoid recursion.
        prefill_cfg = replace(
            self.cfg,
            model_parallel_config=disagg.prefill_parallel(mp),
            disaggregation_config=replace(disagg, enabled=False),
        )
        decode_cfg = replace(
            self.cfg,
            model_parallel_config=disagg.decode_parallel(mp),
            disaggregation_config=replace(disagg, enabled=False),
        )
        prefill_proj = InferencePerformanceProjector(
            prefill_cfg, args=self._args_ref, benchmark_layer_times=self._bench_measured
        )
        decode_proj = InferencePerformanceProjector(
            decode_cfg, args=self._args_ref, benchmark_layer_times=self._bench_measured
        )

        # Prefill phase on the prefill pool (drives TTFT + prefill throughput).
        ttft_compute = prefill_proj.prefill_latency_ms(batch, input_len)
        kv_transfer = self._kv_transfer_ms(decode_proj, batch, input_len)
        ttft = ttft_compute + kv_transfer

        # Decode phase on the decode pool (drives ITL + decode throughput).
        decode_total = decode_proj.decode_total_ms(batch, input_len, output_len)
        mid_ctx = input_len + output_len // 2
        spec_k = int(req.speculative_num_tokens or 0)
        q_len = (spec_k + 1) if spec_k > 0 else 1
        step_latency = decode_proj._decode_step_latency_ms(batch, mid_ctx, q_len=q_len)

        itl = (decode_total / output_len) if output_len > 0 else step_latency
        request_latency = ttft + decode_total
        per_req_decode_tps = (1000.0 / itl) if itl > 0 else 0.0

        # Per-replica decode throughput, scaled by the decode-pool replica count.
        decode_tps_replica = (batch * 1000.0 / step_latency) if step_latency > 0 else 0.0
        decode_tps = decode_tps_replica * max(1, disagg.decode_replicas)
        decode_replica_gpus = _replica_gpus(decode_cfg)
        prefill_replica_gpus = _replica_gpus(prefill_cfg)
        total_decode_gpus = decode_replica_gpus * max(1, disagg.decode_replicas)
        decode_tps_per_gpu = decode_tps / total_decode_gpus if total_decode_gpus else 0.0

        prefill_tps_replica = (batch * input_len * 1000.0 / ttft_compute) if ttft_compute > 0 else 0.0
        prefill_tps = prefill_tps_replica * max(1, disagg.prefill_replicas)

        extras = {"speculative_tokens_per_step": self._spec_tokens_per_step()}
        extras.update(decode_proj._comm_extras(batch, input_len, output_len))
        if self.is_benchmark_calibrated:
            extras["benchmark_calibrated"] = 1.0
        extras["prefill_compute_ttft_ms"] = ttft_compute
        extras["prefill_replicas"] = float(disagg.prefill_replicas)
        extras["decode_replicas"] = float(disagg.decode_replicas)

        return InferencePerfResult(
            ttft_ms=ttft,
            decode_total_ms=decode_total,
            itl_ms=itl,
            request_latency_ms=request_latency,
            per_request_decode_tps=per_req_decode_tps,
            decode_throughput_tps=decode_tps,
            decode_throughput_tps_per_gpu=decode_tps_per_gpu,
            prefill_throughput_tps=prefill_tps,
            decode_step_latency_ms=step_latency,
            replica_gpus=decode_replica_gpus,
            is_disaggregated=True,
            kv_transfer_ms=kv_transfer,
            prefill_replica_gpus=prefill_replica_gpus,
            decode_replica_gpus=decode_replica_gpus,
            extras=extras,
        )


def project_inference_performance(
    inference_config: InferenceConfig, args=None, benchmark_layer_times=None
) -> InferencePerfResult:
    return InferencePerformanceProjector(
        inference_config, args=args, benchmark_layer_times=benchmark_layer_times
    ).project()
