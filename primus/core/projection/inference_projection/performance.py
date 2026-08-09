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
import os
from dataclasses import dataclass, field, replace
from typing import Dict, Optional

from primus.core.projection.module_profilers.language_model import (
    build_profiler,
    get_language_model_profiler_spec,
)
from primus.core.projection.module_profilers.transformer_layer import (
    _estimate_moe_a2a_time_ms,
    _estimate_tp_allreduce_time_ms,
    _moe_tp_allreduce_count,
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

    def __init__(self, inference_config: InferenceConfig, args=None, benchmark_layer_times=None,
                 scaling_benchmarks=None, decode_floor=None):
        self.cfg = inference_config
        self._args_ref = args
        # Optional measured decode latency floor {batch: ms} from a sharded
        # probe. Applied as decode = max(restored, floor(batch)) — see
        # ``_decode_floor_ms``.
        self._decode_floor = {int(b): float(v) for b, v in (decode_floor or {}).items() if v}
        gpu_arch = getattr(args, "gpu_arch", None) if args else None
        gpu_clock = getattr(args, "gpu_clock_mhz", None) if args else None
        gemm_name = getattr(args, "gemm_backend", None) if args else None
        # Kept for the origami-ratio restore, which builds its own guaranteed-
        # simulating profilers at the bench/target views (see _setup_restoration).
        self._gpu_arch, self._gpu_clock, self._gemm_name = gpu_arch, gpu_clock, gemm_name
        # TP/EP-restore scaling: how a measured anchor is extrapolated to another
        # TP. "origami" (default) scales the measured step by the simulator's
        # (vLLM-fused MoE) TP-scaling ratio — validated to beat a 2-point measured
        # fit at high TP; "fit" forces the measured shardable/invariant fit;
        # "blind" the naive TP^-1. Env override for A/B testing.
        self._scaling_mode = (os.getenv("PRIMUS_RESTORE_SCALING") or "origami").strip().lower()
        self._lm_ratio_bench = None

        # In benchmark mode the projection is driven *entirely* by measured
        # layer times, so the analytical GEMM/SDPA simulators (origami) are not
        # exercised for a dense model.  Don't hard-require origami there — fall
        # back to a metadata-only backend and skip SDPA if unavailable.
        benchmark_mode = benchmark_layer_times is not None
        self._gemm = get_gemm_simulation_backend(
            backend_name=gemm_name,
            gpu_arch=gpu_arch,
            gpu_clock_mhz=gpu_clock,
            require_simulation=not benchmark_mode,
        )
        try:
            self._sdpa = get_sdpa_simulation_backend(gpu_arch=gpu_arch, gpu_clock_mhz=gpu_clock)
        except RuntimeError:
            if not benchmark_mode:
                raise
            self._sdpa = None

        # Serving engines (vLLM/SGLang with AITER) run the MoE experts through a
        # *batched* grouped GEMM, not the training-time legacy sequential kernel.
        # The MoE profiler selects the batched Origami model via
        # ``use_turbo_grouped_gemm``, but the ModelConfig dataclass only carries
        # ``use_turbo_grouped_mlp`` — so the profiler-facing flag is never set on
        # the inference path and decode is otherwise mis-modelled as N sequential
        # per-expert GEMMs (grossly inflating small-batch decode). Set it here for
        # MoE serving, respecting an explicit legacy request.
        _mc = inference_config.model_config
        if getattr(_mc, "num_experts", 0) and not getattr(
            _mc, "moe_use_legacy_grouped_gemm", False
        ):
            _mc.use_turbo_grouped_gemm = True

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
        # Per-view imbalance for the origami ratio (populated in
        # _setup_restoration when a bench<->target restore is active).
        self._imb_tgt = self._moe_imbalance
        self._imb_bench = self._moe_imbalance
        # Mirror the routing-imbalance knobs onto the (shared) model_config so the
        # expert-GEMM simulator can apply imbalance *inside* the roofline, per
        # view (EP lives on each view's parallel config). Default ("roofline")
        # mode; PRIMUS_MOE_IMB_ROOFLINE=0 falls back to the outer multiplier.
        self._imb_roofline = os.getenv(
            "PRIMUS_MOE_IMB_ROOFLINE", "1"
        ).strip().lower() not in ("0", "false", "no")
        try:
            mc.ep_load_balance = float(self.cfg.request_config.ep_load_balance or 1.0)
            mc.redundant_experts = int(self.cfg.request_config.redundant_experts or 0)
        except Exception:
            pass

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
        # When the benchmark swept the decode curve at the engine's CUDA-graph
        # capture sizes, runtime pads the decode batch UP to the nearest captured
        # size — so decode latency is a staircase and we look it up by bucket
        # rather than interpolating. Set from meta in set_benchmark_calibration.
        self._decode_pad_to_capture: bool = False
        # Attention KV term for the FULL decode step: the step grows with context
        # by an ~batch-independent additive per-token amount (measured: the rise
        # over context is nearly the same at low and high batch, so it is NOT
        # proportional to batch). Fit from the benchmark's decode-vs-context grid;
        # 0 => flat (no grid), preserving prior behaviour.
        self._decode_kv_slope_ms: float = 0.0     # ms per KV token (batch-independent)
        self._decode_ctx_ref: float = 0.0         # context the batch curve was measured at
        self._decode_ctx_max: float = 0.0         # largest measured context (guard)
        self._meas_prefill_rate_ms_per_tok: float = 0.0  # for sub-prompt prefill pieces
        self._meas_layer: Dict[tuple, float] = {}        # {(phase, ltype): ms}
        self._meas_ref_input: int = 0
        self._bench_backend: str = "megatron"
        self._bench_measured = benchmark_layer_times
        # Training-style TP/EP restoration state (populated for the per-layer
        # schema in set_benchmark_calibration). Off unless the benchmark ran at
        # a reduced parallelism vs the target.
        self._restore = False
        self._bench_tp = 1
        self._bench_ep = 1
        self._bench_pp = 1
        # phase -> batch -> tp -> (ms, ep, pp), and the split fitted from it.
        self._bench_scaling_raw: dict = {}
        self._bench_scaling_fit: dict = {}
        for _blob in (scaling_benchmarks or []):
            self.add_scaling_benchmark(_blob)
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

    # -- batch transport (measured curve only) --------------------------------

    @staticmethod
    def _nearest_anchor(batch: int, pts: list):
        """Nearest measured point to ``batch`` in log-space (batch grids are
        geometric, e.g. 1/4/16/64)."""
        lb = math.log(max(1, batch))
        return min(pts, key=lambda p: abs(lb - math.log(max(1, p[0]))))

    def _fit_decode_kv_slope(self, decode_ctx: list, ref_ctx: float) -> None:
        """Fit the FULL-decode attention KV term from the benchmark's
        decode-vs-context grid: ``step(b, c) = bucket(b) + slope * (c - ref)``.
        ``slope`` is the median per-token increment (``(step - bucket) / (c-ref)``)
        over the measured points — batch-independent, because the measured rise
        with context is ~the same at low and high batch. 0 when no grid (→ decode
        stays flat in context, prior behaviour). Skipped under parallelism restore
        (the grid is only emitted un-restored)."""
        self._decode_ctx_ref = ref_ctx
        self._decode_ctx_max = ref_ctx
        dec = self._meas_whole.get("decode")
        if not decode_ctx or not dec or self._restore or ref_ctx <= 0:
            return
        slopes = []
        for e in decode_ctx:
            try:
                b, c, ms = int(e["batch"]), float(e["context"]), float(e["decode_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            self._decode_ctx_max = max(self._decode_ctx_max, c)
            if b > 0 and c > ref_ctx:
                s = (ms - self._bucket_up(b, dec)) / (c - ref_ctx)
                if s > 0:
                    slopes.append(s)
        if slopes:
            slopes.sort()
            self._decode_kv_slope_ms = slopes[len(slopes) // 2]

    @staticmethod
    def _bucket_up(batch: int, pts: list) -> float:
        """Decode value with the runtime CUDA-graph padding applied: the batch is
        padded UP to the nearest captured size, so return the measured value at
        the smallest measured (capture-aligned) batch >= ``batch``. Clamp to the
        largest measured point above the top capture size. ``pts`` sorted asc."""
        for b0, v0 in pts:
            if b0 >= batch:
                return v0
        return pts[-1][1]

    @staticmethod
    def _loglog_transport(batch: int, pts: list) -> float:
        """Piecewise power-law (log-log linear) interpolation of a measured
        ``(batch -> ms)`` curve; extrapolate with the nearest end segment's
        slope. ``pts`` must be sorted with >= 2 points.

        Interpolating the *measured* curve directly is more accurate than
        modulating the analytical simulator, which can carry a spurious knee
        (e.g. an MoE expert-coverage bump) that the real silicon does not show.
        Latency-vs-batch is close to a local power law between adjacent measured
        points, so a straight line in (log batch, log ms) tracks it tightly and
        extrapolates monotonically."""
        lb = math.log(max(1, batch))
        xs = [(math.log(max(1, b)), math.log(max(1e-9, v))) for b, v in pts]
        if lb <= xs[0][0]:
            (x0, y0), (x1, y1) = xs[0], xs[1]
        elif lb >= xs[-1][0]:
            (x0, y0), (x1, y1) = xs[-2], xs[-1]
        else:
            (x0, y0), (x1, y1) = xs[0], xs[1]
            for i in range(len(xs) - 1):
                if xs[i][0] <= lb <= xs[i + 1][0]:
                    (x0, y0), (x1, y1) = xs[i], xs[i + 1]
                    break
        slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
        return math.exp(y0 + slope * (lb - x0))

    def _transport_batch(self, batch: int, pts: list) -> float:
        """Transport a measured ``(batch -> ms)`` curve to an arbitrary
        ``batch`` — MEASUREMENT-ONLY.

        The benchmark protocol always sweeps batch within a single run, so
        ``pts`` carries >= 2 measured points and we interpolate/extrapolate the
        real curve in log-log space (:meth:`_loglog_transport`) — the analytical
        (origami) simulator is never consulted for the batch shape. A lone
        anchor (a degenerate, non-swept artifact) holds its measured value
        rather than falling back to the simulator, so a benchmark-calibrated
        projection stays free of simulator bias by construction.

        Returns the exact measured value when ``batch`` is itself measured."""
        if not pts:
            return 0.0
        P = sorted(pts)
        for b0, v0 in P:
            if b0 == batch:
                return v0
        if len(P) >= 2:
            return self._loglog_transport(batch, P)
        return P[0][1]

    # -- measured-time accessors (benchmark-based projection) ------------------

    def _measured_decode_step_ms(self, batch: int, context: Optional[float] = None) -> float:
        """Measured whole-model / composed decode *step* latency at ``batch``.

        ``context`` (the resident KV length) adds the fitted attention KV term on
        top of the batch-bucket value; omitting it (or a zero slope / no grid)
        reproduces the flat-in-context behaviour."""
        if self._meas_whole.get("decode"):
            pts = self._meas_whole["decode"]
            if self._decode_pad_to_capture:
                base = self._bucket_up(batch, pts)
            else:
                base = self._transport_batch(batch, pts)
            if context is not None and self._decode_kv_slope_ms > 0.0 and self._decode_ctx_ref > 0.0:
                base += self._decode_kv_slope_ms * max(0.0, float(context) - self._decode_ctx_ref)
            return base
        # Per-layer schema: restore each layer to the target TP/EP, then sum by
        # layer count. Decode processes 1 token/step.
        d = self._restore_per_layer("dense", self._meas_layer.get(("decode", "dense"), 0.0), batch, 1)
        m = self._restore_per_layer("moe", self._meas_layer.get(("decode", "moe"), 0.0), batch, 1)
        return self._n_dense * d + self._n_moe * m + self._restore_pp_ms(batch, 1)

    def _measured_full_prefill_ms(self, batch: int) -> float:
        """Measured whole-model / composed prefill latency for the full prompt."""
        if self._meas_whole.get("prefill"):
            return self._transport_batch(batch, self._meas_whole["prefill"])
        tok = self._meas_ref_input or 1
        d = self._restore_per_layer("dense", self._meas_layer.get(("prefill", "dense"), 0.0), batch, tok)
        m = self._restore_per_layer("moe", self._meas_layer.get(("prefill", "moe"), 0.0), batch, tok)
        return self._n_dense * d + self._n_moe * m + self._restore_pp_ms(batch, tok)

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
            # EP>1 (expert_tp==1) drops the post-expert TP-AR (combined by A2A).
            n_ar = _moe_tp_allreduce_count(self._view)
            return n_ar * tp_ar_one + _estimate_moe_a2a_time_ms(self._view, batch, q_len, self._gemm)
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
        # Parallelism the benchmark ran at (for training-style restoration of a
        # reduced-parallelism per-layer bench to the target TP/EP).
        self._bench_tp = int(meta.get("benchmark_tp") or meta.get("tp") or 1)
        self._bench_ep = int(meta.get("benchmark_ep") or meta.get("ep") or 1)
        self._bench_pp = int(meta.get("benchmark_pp") or meta.get("pp") or 1)
        self._decode_pad_to_capture = bool(meta.get("decode_pad_to_capture"))

        self._meas_ref_input = ref_input

        # Whole-model schema (vLLM/SGLang): measured step latencies, used
        # DIRECTLY (no factor, no simulator). ``sweep`` gives the per-batch
        # curve; fall back to the single ``model`` anchor at ``ref_batch``.
        model_step = measured.get("model")
        if model_step:
            # Restore a reduced-parallelism (benchmark) whole-model measurement to
            # the target TP/EP/PP, in the same pp -> ep -> tp order as the Megatron
            # per-layer path. Builds the bench/target collective models; a no-op
            # when the benchmark already ran at the target parallelism.
            self._setup_restoration()
            # The primary artifact is itself a point on the scaling curve.
            self.add_scaling_benchmark(benchmark_layer_times)
            if self._restore:
                for _phase in ("decode", "prefill"):
                    self._fit_tp_scaling(_phase)
                self._report_tp_scaling()
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
            if self._restore:
                # Prefill processes ``ref_input`` tokens/seq; decode 1 token/step.
                pre_pts = [(b, self._restore_whole(ms, b, ref_input, "prefill")) for b, ms in pre_pts]
                dec_pts = [(b, self._restore_whole(ms, b, 1, "decode")) for b, ms in dec_pts]
            self._meas_whole = {
                k: v for k, v in (("prefill", sorted(pre_pts)), ("decode", sorted(dec_pts))) if v
            }
            # Batch transport interpolates the MEASURED curve and never falls
            # back to the simulator. That requires a batch sweep (>= 2 points);
            # a single-batch artifact would force a flat hold. Warn so the
            # benchmark is (re)run with a sweep rather than silently degrading.
            for _ph in ("prefill", "decode"):
                if len(self._meas_whole.get(_ph, [])) < 2:
                    print(
                        f"[Primus:Inference] WARNING: {_ph} benchmark has a single "
                        f"batch point — batch transport will hold it flat. Re-run the "
                        f"benchmark with a batch sweep for an accurate {_ph} batch curve."
                    )
            # Per-token prefill rate (for sub-prompt chunk pieces): full-prompt
            # prefill of ``b`` seqs processes ``b * ref_input`` tokens.
            if pre_pts and ref_input > 0:
                rates = [ms / (b * ref_input) for b, ms in pre_pts if b > 0]
                self._meas_prefill_rate_ms_per_tok = sum(rates) / len(rates) if rates else 0.0
            self._fit_decode_kv_slope(
                benchmark_layer_times.get("decode_ctx") or [], float(ref_input)
            )
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
        self._setup_restoration()
        # Prefill rate from the dominant (per-layer * count) prefill total.
        if ref_input > 0:
            full_pre = self._measured_full_prefill_ms(ref_batch)
            self._meas_prefill_rate_ms_per_tok = full_pre / (ref_batch * ref_input)

    def _setup_restoration(self) -> None:
        """Prepare TP/EP restoration when a per-layer benchmark was captured at a
        reduced parallelism (mirrors training's benchmark-at-fewer-GPUs → target
        extrapolation). Builds analytical collective models at the benchmark and
        target TP/EP so ``_restore_per_layer`` can strip the benchmark's comm,
        scale the sharded compute, and add the target comm back."""
        mp = self.cfg.model_parallel_config
        self._tgt_tp = max(1, mp.tensor_model_parallel_size)
        tgt_ep = max(1, getattr(mp, "expert_model_parallel_size", 1) or 1)
        tgt_pp = max(1, mp.pipeline_model_parallel_size)
        self._restore = (
            self._bench_tp != self._tgt_tp
            or self._bench_ep != tgt_ep
            or self._bench_pp != tgt_pp
        )
        if not self._restore:
            return
        mc = self.cfg.model_config
        bench_mp = replace(
            mp,
            tensor_model_parallel_size=self._bench_tp,
            expert_model_parallel_size=self._bench_ep,
            pipeline_model_parallel_size=self._bench_pp,
        )
        self._comm_bench = InferenceCollectiveModel(mc, bench_mp, self._cc)
        self._comm_tgt = InferenceCollectiveModel(mc, mp, self._cc)

        # Per-view MoE routing imbalance for the origami ratio.  The target view
        # already uses ``self._moe_imbalance`` (constructor).  The bench view is
        # evaluated at the *bench* EP so that, e.g., an EP=1 bench (experts
        # TP-sharded, balanced -> 1.0) restored to an EP=8 target (all-to-all,
        # busiest-rank gated -> ep_load_balance) keeps the EP sharding penalty
        # in ``sim(target)/sim(bench)`` instead of cancelling it.
        self._imb_tgt = self._moe_imbalance
        self._imb_bench = self._moe_imbalance_for_ep(self._bench_ep)
        # Diagnostic escape hatch: PRIMUS_ORIGAMI_IMB_PERVIEW=0 restores the old
        # behaviour (single target imbalance on both views, which cancels in the
        # ratio) for before/after validation of the per-view fix.
        if os.getenv("PRIMUS_ORIGAMI_IMB_PERVIEW", "1").strip().lower() in ("0", "false", "no"):
            self._imb_bench = self._moe_imbalance

        # Origami-ratio setup: build guaranteed-simulating profiler trees at the
        # bench and target views so ``_restore_whole`` can scale the measured
        # anchor by the simulator's TP-scaling ratio (the absolute origami bias
        # cancels in the ratio). Benchmark mode may hold a metadata-only GEMM
        # backend, so build dedicated simulating backends here. On any failure
        # (e.g. no SDPA simulator for the arch) origami is disabled and the
        # restore falls back to the measured fit / blind TP^-1.
        self._view_tgt = self._view
        self._lm_ratio_bench = None
        if self._scaling_mode == "origami":
            try:
                self._gemm_sim = get_gemm_simulation_backend(
                    backend_name=self._gemm_name, gpu_arch=self._gpu_arch,
                    gpu_clock_mhz=self._gpu_clock, require_simulation=True,
                )
                self._sdpa_sim = get_sdpa_simulation_backend(
                    gpu_arch=self._gpu_arch, gpu_clock_mhz=self._gpu_clock,
                )
                rc = self.cfg.request_config
                saved_mp = self.cfg.model_parallel_config
                self.cfg.model_parallel_config = bench_mp
                self._view_bench = self.cfg.as_training_config(
                    batch_size=rc.batch_size, seq_len=rc.input_seq_len,
                )
                self.cfg.model_parallel_config = saved_mp
                self._lm_ratio_tgt = build_profiler(
                    get_language_model_profiler_spec(self._view_tgt))
                self._lm_ratio_tgt.set_simulation_backends(self._gemm_sim, self._sdpa_sim)
                self._lm_ratio_bench = build_profiler(
                    get_language_model_profiler_spec(self._view_bench))
                self._lm_ratio_bench.set_simulation_backends(self._gemm_sim, self._sdpa_sim)
            except Exception as e:  # pragma: no cover - arch-dependent
                self._lm_ratio_bench = None
                print(f"[Primus:Inference] origami-ratio unavailable ({e}); "
                      "falling back to measured fit / blind TP^-1.")

    def _comm_model_at_tp(self, tp: int, ep: int, pp: int) -> "InferenceCollectiveModel":
        """Collective model at an arbitrary parallelism, for the scaling fit."""
        mp = self.cfg.model_parallel_config
        return InferenceCollectiveModel(
            self.cfg.model_config,
            replace(
                mp,
                tensor_model_parallel_size=max(1, tp),
                expert_model_parallel_size=max(1, ep),
                pipeline_model_parallel_size=max(1, pp),
            ),
            self._cc,
        )

    def _fit_tp_scaling(self, phase: str) -> None:
        """Fit ``compute(tp) = shardable / tp + invariant`` per batch size.

        Least squares in ``1/tp`` over the points registered by
        ``add_scaling_benchmark``; needs at least two parallelisms.
        """
        pts = self._bench_scaling_raw.get(phase) or {}
        fits = {}
        for batch, per_tp in pts.items():
            if len(per_tp) < 2:
                continue
            xs, ys = [], []
            for tp, (ms, ep, pp) in sorted(per_tp.items()):
                tokens = 1 if phase == "decode" else self._meas_ref_input
                cm = self._comm_model_at_tp(tp, ep, pp)
                dense = (cm.layer_comm_ms(batch, tokens, is_moe=False).total_ms
                         if self._n_dense else 0.0)
                moe = (cm.layer_comm_ms(batch, tokens, is_moe=True).total_ms
                       if self._n_moe else 0.0)
                comm = self._n_dense * dense + self._n_moe * moe
                xs.append(1.0 / tp)
                ys.append(max(0.0, ms - comm))
            n = len(xs)
            sx, sy = sum(xs), sum(ys)
            sxx = sum(x * x for x in xs)
            sxy = sum(x * y for x, y in zip(xs, ys))
            det = n * sxx - sx * sx
            if abs(det) < 1e-12:
                continue
            shardable = (n * sxy - sx * sy) / det
            invariant = (sy - shardable * sx) / n
            if shardable <= 0.0:
                # No positive TP-shardable component — not a usable fit.
                continue
            if invariant < 0.0:
                # Near-/super-linear scaling. This is expected for a close pair
                # such as TP=1 + TP=2, where the TP-invariant remainder is still
                # masked by compute at low TP and the fit's intercept dips
                # slightly negative. Rather than discard the two measured anchors
                # (and fall back to a blind TP^-1), clamp the invariant to 0 and
                # refit ``shardable`` as least-squares through the origin. The law
                # then reproduces both measured anchors and scales as ~TP^-1 —
                # exact where measured, optimistic for TP well above the range.
                invariant = 0.0
                shardable = sxy / sxx if sxx > 0.0 else shardable
            fits[batch] = (shardable, invariant)
        if fits:
            self._bench_scaling_fit[phase] = fits

    def _report_tp_scaling(self) -> None:
        """Print the TP-scaling law the restore will use."""
        if (self._scaling_mode == "origami" and self._restore
                and getattr(self, "_lm_ratio_bench", None) is not None):
            print(
                f"[Primus:Inference] TP scaling: origami-ratio (simulate vLLM-fused "
                f"MoE) — scaling measured TP={self._bench_tp} anchor to TP="
                f"{self._tgt_tp} by sim(target)/sim(bench)."
            )
            return
        fits = self._bench_scaling_fit.get("decode") or {}
        if fits:
            # Largest batch: the concurrency the step is usually judged at.
            batch = max(fits)
            shardable, invariant = fits[batch]
            tps = sorted((self._bench_scaling_raw.get("decode") or {}).get(batch, {}))
            total = shardable + invariant
            print(
                f"[Primus:Inference] TP scaling fitted from benchmark TP="
                f"{','.join(str(t) for t in tps)} at batch {batch}: "
                f"{shardable:.2f} ms shardable + {invariant:.2f} ms TP-invariant"
                f" ({invariant / total * 100:.0f}% does not shrink with TP)"
            )
            # Interpolation between measured anchors is exact; extrapolation
            # ABOVE the measured range is only as good as the fitted invariant,
            # which two low-TP anchors cannot fully resolve (the non-shrinking
            # floor is still masked at low TP). Flag it so an out-of-range TP is
            # treated as lower-confidence rather than trusted like a measurement.
            max_tp = max(tps) if tps else self._bench_tp
            if self._tgt_tp > max_tp:
                print(
                    f"[Primus:Inference] WARNING: target TP={self._tgt_tp} is ABOVE the "
                    f"measured range (TP<={max_tp}); this decode step is EXTRAPOLATED and "
                    f"tends to under-predict latency / over-predict throughput. Add a "
                    f"benchmark at TP>={self._tgt_tp} to make it exact."
                )
            return
        if self._bench_tp != self._tgt_tp:
            print(
                f"[Primus:Inference] WARNING: restoring benchmark TP="
                f"{self._bench_tp} to TP={self._tgt_tp} assuming the whole step is"
                " TP-shardable and scales as TP^-1. Real decode steps keep a"
                " TP-invariant remainder, so this under-predicts step latency."
                " Pass --load-benchmark-scaling with a run at another"
                " --benchmark-gpus to fit the split instead."
            )

    def add_scaling_benchmark(self, blob: dict) -> None:
        """Register a benchmark artifact taken at a different TP, for the fit only."""
        measured = blob.get("measured", blob)
        meta = blob.get("meta", {})
        tp = int(meta.get("benchmark_tp") or meta.get("tp") or 1)
        ep = int(meta.get("benchmark_ep") or meta.get("ep") or 1)
        pp = int(meta.get("benchmark_pp") or meta.get("pp") or 1)
        sweep = blob.get("sweep") or []
        model_step = measured.get("model") or {}
        ref_batch = int(meta.get("batch") or self.cfg.request_config.batch_size or 1)
        for phase, key in (("decode", "decode_ms"), ("prefill", "prefill_ms")):
            rows = [(int(e["batch"]), float(e[key])) for e in sweep
                    if e.get("batch") is not None and e.get(key)]
            if not rows and model_step.get(key):
                rows = [(ref_batch, float(model_step[key]))]
            for batch, ms in rows:
                self._bench_scaling_raw.setdefault(phase, {}).setdefault(
                    batch, {}
                )[tp] = (ms, ep, pp)

    def _restore_per_layer(self, ltype: str, ms_bench: float, batch: int, tokens: int) -> float:
        """Restore a per-layer time measured at the benchmark's (reduced) TP/EP to
        the target parallelism, training-style: the shardable compute scales by
        ``bench_tp / target_tp`` and the blocking collective at the target TP/EP
        is added back analytically. No-op when bench and target parallelism match
        or for the whole-model (vLLM) schema, which is non-decomposable."""
        if not self._restore or ms_bench <= 0.0:
            return ms_bench
        is_moe = ltype == "moe"
        comm_bench = self._comm_bench.layer_comm_ms(batch, tokens, is_moe=is_moe).total_ms
        comm_tgt = self._comm_tgt.layer_comm_ms(batch, tokens, is_moe=is_moe).total_ms
        compute = max(0.0, ms_bench - comm_bench) * (self._bench_tp / self._tgt_tp)
        return compute + comm_tgt

    def _restore_pp_ms(self, batch: int, tokens: int) -> float:
        """Per-*forward* pipeline P2P delta added when restoring to a target PP.
        PP distributes whole layers across stages, so it adds ``(pp-1)``
        send/recv hops to a forward pass without sharding compute — hence a
        single additive term per step, not a per-layer one. Reuses the shared
        ``cm.sendrecv`` primitive via ``InferenceCollectiveModel.pp_p2p_ms``."""
        if not self._restore:
            return 0.0
        return self._comm_tgt.pp_p2p_ms(batch, tokens) - self._comm_bench.pp_p2p_ms(batch, tokens)

    def _origami_ratio(self, batch: int, tokens: int, phase: str) -> Optional[float]:
        """Simulator TP-scaling ratio sim(target)/sim(bench) for the whole step.

        Reuses the analytical ``_forward_times`` at the bench and target views by
        temporarily swapping the profiler tree / comm model / view / sim backends
        (the analytical path is otherwise unused in benchmark mode). Returns
        ``None`` when origami is unavailable so the caller falls back to the fit.
        """
        if getattr(self, "_lm_ratio_bench", None) is None:
            return None
        if phase == "prefill":
            q_len, kv = max(1, tokens), max(1, tokens)
        else:
            q_len = 1
            kv = max(1, self._meas_ref_input or self.cfg.request_config.input_seq_len or 1024)
        # Explicit comm model per view when active; else builtin (comm=None) so
        # ``_forward_times`` derives it from the (swapped) view.
        comm_tgt = self._comm if self._comm is not None else None
        comm_bench = self._comm_bench if self._comm is not None else None

        # For decode, exclude communication from the scaling ratio: decode
        # collectives (TP all-reduce, EP all-to-all) are small-message,
        # latency-bound, and either overlapped with compute at high batch or
        # pipelined into the fixed per-step overhead at low batch — they are NOT
        # additive on top of a comm-free anchor. Charging them in the ratio
        # double-counts and, when restoring from a comm-free 1-GPU anchor,
        # explodes the ratio at low batch (the target grows a full A2A+AR the
        # anchor never had). The resident decode comm is instead captured by the
        # measured latency floor (``_decode_floor_ms``). Prefill comm is large-
        # message and genuinely exposed, so it stays in the ratio.
        comm_free = phase == "decode"

        def _step(lm, comm, view, imb) -> float:
            saved = (self._lm, self._comm, self._view, self._gemm, self._sdpa,
                     self._moe_imbalance)
            self._lm, self._comm, self._view = lm, comm, view
            self._gemm, self._sdpa = self._gemm_sim, self._sdpa_sim
            # Per-view imbalance: bench-EP vs target-EP (see _setup_restoration).
            self._moe_imbalance = imb
            try:
                ft = self._forward_times(batch, q_len, phase, kv)
                if comm_free:
                    return max(0.0, ft.total_ms - ft.comm.tp_allreduce_ms
                               - ft.comm.ep_a2a_ms - ft.comm.pp_p2p_ms)
                return ft.total_ms
            finally:
                (self._lm, self._comm, self._view, self._gemm, self._sdpa,
                 self._moe_imbalance) = saved

        try:
            s_tgt = _step(self._lm_ratio_tgt, comm_tgt, self._view_tgt, self._imb_tgt)
            s_bench = _step(self._lm_ratio_bench, comm_bench, self._view_bench, self._imb_bench)
        except Exception:
            return None
        if s_bench <= 0.0 or s_tgt <= 0.0:
            return None
        return s_tgt / s_bench

    def _restore_whole(self, ms_bench: float, batch: int, tokens: int, phase: str = "decode") -> float:
        """Restore a whole-model (vLLM) step latency measured at the benchmark's
        reduced parallelism to the target TP/EP/PP, training-style and in the same
        ``pp -> ep -> tp`` order as the Megatron per-layer path:

          * strip the benchmark's per-layer communication (summed over layers) so
            only shardable compute remains,
          * scale that compute by ``bench_tp / target_tp`` (TP),
          * add the target communication back — which includes the target EP
            all-to-all and TP all-reduce (EP + TP), and
          * add the ``(pp-1)`` P2P delta for the target PP (PP).

        No-op when bench and target parallelism match, mirroring
        ``_restore_per_layer`` but applied to the full-model total (the whole-model
        vLLM step is not separable per layer, so comm is composed by layer count)."""
        if not self._restore or ms_bench <= 0.0:
            return ms_bench

        # Origami-ratio (default): scale the measured anchor by the simulator's
        # whole-step TP-scaling ratio sim(target)/sim(bench). The vLLM-fused MoE
        # cost model captures the saturating decode curve (compute sharding +
        # comm growth) better than a 2-point linear fit; the ~5x absolute origami
        # bias cancels in the ratio. Falls through to fit/blind if unavailable.
        if self._scaling_mode == "origami":
            r = self._origami_ratio(batch, tokens, phase)
            if r is not None and r > 0.0:
                return ms_bench * r

        def _comm_total(cm) -> float:
            dense = cm.layer_comm_ms(batch, tokens, is_moe=False).total_ms if self._n_dense else 0.0
            moe = cm.layer_comm_ms(batch, tokens, is_moe=True).total_ms if self._n_moe else 0.0
            return self._n_dense * dense + self._n_moe * moe

        comm_bench = _comm_total(self._comm_bench)
        comm_tgt = _comm_total(self._comm_tgt)
        compute = max(0.0, ms_bench - comm_bench)
        fit = (self._bench_scaling_fit.get(phase) or {}).get(batch)
        if fit:
            # Measured scaling: only the shardable part shrinks with TP.
            shardable, invariant = fit
            compute = shardable / self._tgt_tp + invariant
        else:
            compute *= self._bench_tp / self._tgt_tp
        # Apply the same compute-limited comm/compute overlap used on the
        # analytical path (``_overlap_keep``): the configured prefill/decode
        # overlap is a ceiling, but you can hide at most ``compute`` worth of
        # comm behind compute. No-op when the overlap knob is 0 (default).
        keep = self._overlap_keep(phase, comm_tgt, compute)
        return compute + comm_tgt * keep + self._restore_pp_ms(batch, tokens)

    # -- per-pass forward time -------------------------------------------------

    def _moe_imbalance_factor(self) -> float:
        """MoE expert-compute imbalance multiplier (>= 1.0) at the target EP."""
        return self._moe_imbalance_for_ep(
            max(1, self.cfg.model_parallel_config.expert_model_parallel_size)
        )

    def _moe_imbalance_for_ep(self, ep: int) -> float:
        """MoE expert-compute imbalance multiplier (>= 1.0) at an arbitrary EP.

        Only EP-sharded MoE models (``num_experts > 0`` and ``EP > 1``) see
        routing imbalance; for everything else this is a no-op (1.0).  The
        magnitude (and the ``redundant_experts`` mitigation) is resolved on the
        request config, given the model's expert count.  Evaluating this per-EP
        is what lets the origami ratio keep the EP sharding penalty instead of
        cancelling a single (target) imbalance value on both bench and target
        sides.
        """
        mc = self.cfg.model_config
        num_experts = int(getattr(mc, "num_experts", 0) or 0)
        if num_experts <= 0 or max(1, ep) <= 1:
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
        # In roofline mode the imbalance is applied inside the expert GEMM (M
        # scaling in moe_mlp), so the outer multiplier only carries the dtype
        # speedup here to avoid double-counting. When roofline mode is disabled
        # the outer multiplier applies the imbalance (legacy behaviour).
        outer_imb = 1.0 if self._imb_roofline else self._moe_imbalance
        if (
            has_moe
            and (outer_imb > 1.0 or self._moe_expert_speedup != 1.0)
            and hasattr(moe_p, "get_sub_profiler")
        ):
            mlp_p = moe_p.get_sub_profiler("mlp")
            if mlp_p is not None:
                expert_mlp_ms = mlp_p.measured_forward_time(batch, q_len)
                new_expert = expert_mlp_ms * outer_imb * self._moe_expert_speedup
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
            # Feature B: explicit, knob-driven communication model with a
            # batch-dependent compute/comm overlap. The exposed comm is hidden
            # behind the SAME layer-type compute up to the configured ceiling
            # (see _overlap_keep), so dense and MoE layers get different exposure
            # and the residual at high batch is captured analytically.
            new_tp_ar = self._comm.tp_allreduce_ms(batch, q_len)
            new_ep_a2a = self._comm.ep_a2a_ms(batch, q_len)
            # Dense: 2 TP-AR (attention + MLP). MoE: the post-expert TP-AR is
            # only present while experts stay TP-sharded (EP=1); at EP==TP the
            # expert output is combined by the A2A, so MoE carries just the
            # attention AR (half of the dense 2-AR) plus the A2A. Charging the
            # full dense 2-AR + A2A on MoE at EP>1 double-counts (see
            # _moe_tp_allreduce_count).
            moe_tp_ar = new_tp_ar * (_moe_tp_allreduce_count(self._view) / 2.0)

            # EP A2A (dispatch/combine) is overlapped with expert compute by the
            # fused MoE kernel (aiter/DeepEP), so its exposed cost is limited by
            # available compute, NOT the configured comm-overlap ceiling (which
            # defaults to 0). A direct fused-MoE microbench + the EP validation
            # sweep show the exposed decode A2A is near-zero, not the ~30us
            # isolated collective latency; charging it fully makes EP>1 look
            # slower than silicon. The attention TP-AR stays on the configured
            # ceiling (serial reduction). Toggle: PRIMUS_MOE_A2A_OVERLAP=0.
            a2a_overlap = os.getenv("PRIMUS_MOE_A2A_OVERLAP", "1") != "0"
            if a2a_overlap and has_moe and moe_compute > 0.0 and new_ep_a2a > 0.0:
                keep_a2a = 1.0 - min(1.0, moe_compute / new_ep_a2a)
            else:
                keep_a2a = self._overlap_keep(phase, new_ep_a2a, moe_compute) if has_moe else 1.0
            ep_a2a_exposed = new_ep_a2a * keep_a2a

            dense_comm = new_tp_ar
            keep_dense = self._overlap_keep(phase, dense_comm, dense_compute) if has_dense else 1.0
            keep_moe_ar = self._overlap_keep(phase, moe_tp_ar, moe_compute) if has_moe else 1.0

            dense_fwd = dense_compute + (dense_comm * keep_dense if has_dense else 0.0)
            moe_fwd = moe_compute + (
                moe_tp_ar * keep_moe_ar + ep_a2a_exposed if has_moe else 0.0
            )

            # TP-AR appears in both layer types; charge each at its own exposure.
            comm.tp_allreduce_ms = (
                self._n_dense * new_tp_ar * keep_dense + self._n_moe * moe_tp_ar * keep_moe_ar
            )
            comm.ep_a2a_ms = self._n_moe * ep_a2a_exposed
            pp_keep = keep_moe_ar if has_moe else keep_dense
            comm.pp_p2p_ms = self._comm.pp_p2p_ms(batch, q_len) * pp_keep
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

    def _decode_floor_ms(self, batch: int) -> float:
        """Hardware decode latency floor at ``batch`` from a sharded probe.

        Above the roofline knee the decode step is set by fixed per-step
        launch/dispatch overhead (parallelism-invariant), so a sharded probe's
        measured decode curve is the floor for any more-sharded target. Clamps
        to the probe's batch range and linearly interpolates between measured
        points. Returns 0.0 (no floor) when no probe was provided.
        """
        floor = self._decode_floor
        if not floor:
            return 0.0
        if batch in floor:
            return floor[batch]
        bs = sorted(floor)
        if batch <= bs[0]:
            return floor[bs[0]]
        if batch >= bs[-1]:
            return floor[bs[-1]]
        lo = max(b for b in bs if b <= batch)
        hi = min(b for b in bs if b >= batch)
        if hi == lo:
            return floor[lo]
        w = (batch - lo) / (hi - lo)
        return floor[lo] * (1.0 - w) + floor[hi] * w

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
            per_token = self._measured_decode_step_ms(batch, kv_len)
            step = per_token * q_len if q_len > 1 else per_token  # verify q_len tokens/step
            step = step + self._draft_overhead_ms(per_token) + self._decode_step_overhead_ms()
            return max(step, self._decode_floor_ms(batch))
        ft = self._forward_times(batch, q_len, "decode", kv_len)
        per_token = ft.total_ms / max(1, q_len)
        step = ft.total_ms + self._draft_overhead_ms(per_token) + self._decode_step_overhead_ms()
        return max(step, self._decode_floor_ms(batch))

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
            dec_piece = self._measured_decode_step_ms(num_decode, decode_ctx) * spec if num_decode > 0 else 0.0
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
            # Benchmark-based: average the pure/mixed step over the context window
            # [ISL, ISL+OSL]. The measured decode step carries its fitted KV term,
            # so this is flat only when that slope is ~0 (prior behaviour).
            spec = q_len if q_len > 1 else 1
            n_samples = min(8, max(2, int(OSL)))
            pure, mixed = [], []
            for i in range(n_samples):
                frac = i / (n_samples - 1) if n_samples > 1 else 0.0
                ctx = int(ISL + frac * OSL)
                d_pure = self._measured_decode_step_ms(C, ctx)
                pure.append(d_pure * spec + self._draft_overhead_ms(d_pure) + ov)
                prefill_piece = self._measured_prefill_tokens_ms(chunk_tokens)
                dec_piece = self._measured_decode_step_ms(max(1, C - 1), ctx) * spec
                mixed.append((prefill_piece + dec_piece) * (1.0 + penalty) + ov)
            t_pure = sum(pure) / len(pure)
            t_mixed = sum(mixed) / len(mixed)
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

        # Hardware decode latency floor (from a sharded probe): above the
        # roofline knee the pure-decode step can't drop below the fixed
        # per-step launch/dispatch overhead. A mixed step carries this decode
        # work plus a prefill chunk, so the same floor is a valid lower bound.
        floor = self._decode_floor_ms(C)
        if floor > 0.0:
            t_pure = max(t_pure, floor)
            t_mixed = max(t_mixed, floor)

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

    def _overlap_keep(self, phase: str, comm_ms: float, compute_ms: Optional[float]) -> float:
        """Exposed-comm fraction (1 - hidden) after compute/comm overlap.

        The configured ``prefill_overlap`` / ``decode_overlap`` is the *ceiling*
        (max fraction of comm hideable behind compute). Physically you cannot
        hide more comm than there is compute to overlap it with, so the
        achievable overlap is ``min(ceiling, compute/comm)``. This makes the
        exposed comm naturally **batch-dependent** and fixes the residual seen at
        high batch: at small batch compute is small relative to the (partly
        fixed) comm, so more comm is exposed; as batch grows compute dominates
        and the overlap saturates at the ceiling.

        ``compute_ms=None`` (e.g. the benchmark-mode reporting path, where layer
        compute is not separately modelled) falls back to the constant ceiling,
        preserving the previous behaviour there.
        """
        ceiling = float(
            self._cc.prefill_overlap if phase == "prefill" else self._cc.decode_overlap
        )
        ceiling = min(max(ceiling, 0.0), 1.0)
        if ceiling <= 0.0:
            return 1.0
        if compute_ms is None or comm_ms <= 0.0:
            return 1.0 - ceiling
        hideable = min(ceiling, max(0.0, compute_ms) / comm_ms)
        return 1.0 - min(1.0, max(0.0, hideable))

    def _comm_breakdown(self, batch: int, q_len: int, phase: str) -> CommBreakdown:
        """Explicit per-phase communication breakdown (ms).

        Derived directly from the knob-driven communication model, without the
        analytical *compute* path (``_forward_times``).  This keeps the comm
        report available in **benchmark mode**, where the layer compute comes
        from measured silicon times and the GEMM/SDPA simulators are not built.
        """
        comm = CommBreakdown()
        if self._comm is None:
            return comm
        # Reporting path (no separated compute → constant ceiling, matching the
        # historical breakdown). The projection path applies the batch-dependent
        # overlap in _phase_forward_times.
        keep = self._overlap_keep(phase, 1.0, None)
        tp_ar = self._comm.tp_allreduce_ms(batch, q_len)
        ep_a2a = self._comm.ep_a2a_ms(batch, q_len)
        # MoE keeps only the attention AR when experts are fully EP-distributed
        # (see _moe_tp_allreduce_count); avoids double-counting with the A2A.
        moe_tp_ar = tp_ar * (_moe_tp_allreduce_count(self._view) / 2.0)
        comm.tp_allreduce_ms = (self._n_dense * tp_ar + self._n_moe * moe_tp_ar) * keep
        comm.ep_a2a_ms = self._n_moe * ep_a2a * keep
        comm.pp_p2p_ms = self._comm.pp_p2p_ms(batch, q_len) * keep
        return comm

    def _comm_extras(
        self, batch: int, input_len: int, output_len: int, prefill_batch: Optional[int] = None
    ) -> Dict[str, float]:
        """Representative per-phase comm breakdown (ms) for reporting.

        ``prefill_batch`` sizes the prefill breakdown; it defaults to ``batch``
        but is set to the per-request prefill batch (1 under continuous batching)
        so the reported prefill comm matches the per-request TTFT basis.
        """
        if self._comm is None:
            return {}
        pre = self._comm_breakdown(prefill_batch if prefill_batch else batch, input_len, "prefill")
        spec_k = int(self.cfg.request_config.speculative_num_tokens or 0)
        q_len = (spec_k + 1) if spec_k > 0 else 1
        dec = self._comm_breakdown(batch, q_len, "decode")
        return {
            "comm_prefill_tp_allreduce_ms": pre.tp_allreduce_ms,
            "comm_prefill_ep_a2a_ms": pre.ep_a2a_ms,
            "comm_prefill_pp_p2p_ms": pre.pp_p2p_ms,
            "comm_prefill_total_ms": pre.total_ms,
            "comm_decode_tp_allreduce_ms": dec.tp_allreduce_ms,
            "comm_decode_ep_a2a_ms": dec.ep_a2a_ms,
            "comm_decode_pp_p2p_ms": dec.pp_p2p_ms,
            "comm_decode_total_ms": dec.total_ms,
        }

    # -- top level -------------------------------------------------------------

    def _resolve_hbm_gb(self) -> tuple[float, str]:
        """Per-GPU HBM capacity + where it came from, resolved in order:
        explicit ``--hbm-capacity-gb`` → live device query (GPU node). No default
        is assumed — if neither is available this raises, so the sustainable-
        concurrency number is never computed against a guessed memory size. The
        source string is surfaced so it always states the size it used."""
        args = self._args_ref
        hbm = getattr(args, "hbm_capacity_gb", None) if args else None
        if hbm:
            return float(hbm), "--hbm-capacity-gb"
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return props.total_memory / (1024.0 ** 3), f"device({props.name})"
        except Exception:
            pass
        raise ValueError(
            "Per-GPU HBM capacity is required but was not provided: pass "
            "--hbm-capacity-gb (e.g. 192 for MI300X, 256 for MI325X, 288 for "
            "MI355X) or run on a GPU node where the device can be queried. "
            "No default is assumed."
        )

    def _sustainable_concurrency(self) -> tuple[Optional[int], float, str]:
        """KV-feasible max concurrent sequences at the target context length,
        i.e. how many sequences fit in the HBM left after weights + activations.
        Reuses the memory projection so there is a single sizing formula. Returns
        ``(max_conc_or_None, hbm_gb, hbm_source)``."""
        hbm_gb, source = self._resolve_hbm_gb()
        try:
            from .memory import project_inference_memory

            mem = project_inference_memory(
                self.cfg, hbm_capacity_gb=hbm_gb, verbose=False
            )
            return mem.max_concurrent_sequences, hbm_gb, source
        except Exception:
            return None, hbm_gb, source

    def _effective_concurrency(self) -> dict:
        """Concurrency that drives throughput, reconciled against the KV-feasible
        ceiling (cap + report):

          * no explicit ``max_concurrency`` → use the KV-derived sustainable max
            (instead of ``batch_size``), so a config that frees HBM is scored at
            the load it can actually serve;
          * explicit ``max_concurrency`` above the ceiling → clamp to it and flag;
          * HBM unknown / KV sizing unavailable → fall back to the prior
            ``resolved_max_concurrency()`` behaviour.

        Always records the sustainable max, the concurrency actually used, and
        the HBM capacity + source in ``extras``."""
        req = self.cfg.request_config
        explicit = req.max_concurrency
        sustainable, hbm_gb, hbm_source = self._sustainable_concurrency()
        capped = False
        if sustainable and sustainable > 0:
            if explicit is None:
                concurrency = sustainable
            else:
                concurrency = min(int(explicit), sustainable)
                capped = int(explicit) > sustainable
        else:
            concurrency = req.resolved_max_concurrency()
        concurrency = max(1, int(concurrency))
        extras = {
            "sustainable_concurrency": int(sustainable) if sustainable else 0,
            "concurrency_used": concurrency,
            "hbm_capacity_gb": float(hbm_gb),
            "hbm_capacity_source": hbm_source,
            "concurrency_capped": 1.0 if capped else 0.0,
        }
        return {"concurrency": concurrency, "extras": extras}

    def project(self) -> InferencePerfResult:
        if self.cfg.disaggregation_config and self.cfg.disaggregation_config.enabled:
            return self._project_disaggregated()
        return self._project_colocated()

    def _project_colocated(self) -> InferencePerfResult:
        req = self.cfg.request_config
        batch = max(1, req.batch_size)
        input_len = max(1, req.input_seq_len)
        output_len = max(0, req.output_seq_len)

        conc = self._effective_concurrency()
        concurrency = conc["concurrency"]
        spec_k = int(req.speculative_num_tokens or 0)
        q_len = (spec_k + 1) if spec_k > 0 else 1
        replica_gpus = _replica_gpus(self.cfg)

        # TTFT is a per-request quantity: a request's first token follows the
        # prefill of its OWN prompt. Under continuous batching, pricing the whole
        # concurrent batch here would scale every prefill collective message by
        # the batch size (e.g. a batch-x larger EP All-to-All) and massively
        # over-state TTFT. Price TTFT at a single request; the batched prefill
        # still drives aggregate prefill throughput below.
        continuous = self._use_continuous_batching(concurrency, output_len)
        ttft = self.prefill_latency_ms(1 if continuous else batch, input_len)
        prefill_full_ms = self.prefill_latency_ms(batch, input_len) if continuous else ttft
        # Host prompt-tokenization cost (client sends text; server tokenizes it
        # after the TTFT clock starts). Latency-only, TTFT side -- symmetric with
        # the decode-side detokenization term below. Applied after prefill_full_ms
        # so it never leaks into prefill throughput.
        ttft += max(0.0, self.cfg.request_config.tokenize_overhead_us) / 1000.0 * max(0, input_len)
        extras = {"speculative_tokens_per_step": self._spec_tokens_per_step()}
        extras.update(conc["extras"])

        if continuous:
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

        # Per-token detokenization + streaming (client-side host cost). Serving
        # harnesses measure ITL client-side, so it carries this; the GPU decode
        # step does not. Latency-only: it overlaps the next server step, so
        # aggregate decode throughput is unchanged.
        detok_ms = max(0.0, self.cfg.request_config.detokenize_overhead_us) / 1000.0
        if detok_ms:
            itl += detok_ms
            decode_total += detok_ms * max(0, output_len)
            per_req_decode_tps = (1000.0 / itl) if itl > 0 else 0.0

        request_latency = ttft + decode_total
        decode_tps_per_gpu = decode_tps / replica_gpus if replica_gpus else 0.0
        prefill_tps = (batch * input_len * 1000.0 / prefill_full_ms) if prefill_full_ms > 0 else 0.0

        # Offered-load queueing (open-loop). The offered-load queue wait is the
        # client-side wait for a serving slot; like the vLLM / InferenceX harness
        # (whose TTFT clock starts after the request is admitted), we keep it OUT
        # of the primary TTFT and end-to-end latency and expose it separately
        # (queue_wait_ms, ttft_with_queue_ms, request_latency_with_queue_ms).
        # No-op unless a request rate is set. TPOT / throughput are steady-state
        # and unaffected either way.
        q = self._request_rate_queueing(decode_tps, output_len, ttft, request_latency)
        if q:
            extras.update(q)

        extras.update(self._comm_extras(batch, input_len, output_len, prefill_batch=(1 if continuous else batch)))
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
        # TTFT is per-request (a single prompt's prefill); the batched prefill
        # drives aggregate prefill throughput below.
        ttft_compute = prefill_proj.prefill_latency_ms(1, input_len)
        prefill_full_ms = prefill_proj.prefill_latency_ms(batch, input_len)
        kv_transfer = self._kv_transfer_ms(decode_proj, batch, input_len)
        # Host prompt-tokenization cost (latency-only, TTFT side).
        tok_ms = max(0.0, req.tokenize_overhead_us) / 1000.0 * max(0, input_len)
        ttft = ttft_compute + kv_transfer + tok_ms

        # Decode phase on the decode pool (drives ITL + decode throughput).
        decode_total = decode_proj.decode_total_ms(batch, input_len, output_len)
        mid_ctx = input_len + output_len // 2
        spec_k = int(req.speculative_num_tokens or 0)
        q_len = (spec_k + 1) if spec_k > 0 else 1
        step_latency = decode_proj._decode_step_latency_ms(batch, mid_ctx, q_len=q_len)

        itl = (decode_total / output_len) if output_len > 0 else step_latency
        # Per-token detokenization + streaming (latency-only; see the co-located
        # projection path for the rationale).
        detok_ms = max(0.0, req.detokenize_overhead_us) / 1000.0
        if detok_ms:
            itl += detok_ms
            decode_total += detok_ms * max(0, output_len)
        request_latency = ttft + decode_total
        per_req_decode_tps = (1000.0 / itl) if itl > 0 else 0.0

        # Per-replica decode throughput, scaled by the decode-pool replica count.
        decode_tps_replica = (batch * 1000.0 / step_latency) if step_latency > 0 else 0.0
        decode_tps = decode_tps_replica * max(1, disagg.decode_replicas)
        decode_replica_gpus = _replica_gpus(decode_cfg)
        prefill_replica_gpus = _replica_gpus(prefill_cfg)
        total_decode_gpus = decode_replica_gpus * max(1, disagg.decode_replicas)
        decode_tps_per_gpu = decode_tps / total_decode_gpus if total_decode_gpus else 0.0

        prefill_tps_replica = (batch * input_len * 1000.0 / prefill_full_ms) if prefill_full_ms > 0 else 0.0
        prefill_tps = prefill_tps_replica * max(1, disagg.prefill_replicas)

        extras = {"speculative_tokens_per_step": self._spec_tokens_per_step()}
        extras.update(decode_proj._comm_extras(batch, input_len, output_len, prefill_batch=1))
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
