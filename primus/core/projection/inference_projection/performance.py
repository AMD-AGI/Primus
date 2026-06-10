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

from .collectives import CommBreakdown, InferenceCollectiveModel


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

    def __init__(self, inference_config: InferenceConfig, args=None):
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

    # -- per-pass forward time -------------------------------------------------

    def _forward_times(self, batch: int, q_len: int, phase: str, kv_len: int) -> PhaseForwardTimes:
        lm = self._lm
        lm.set_inference_phase(phase, kv_len)

        dense_p = lm.sub_profilers.get("dense_transformer_layer")
        moe_p = lm.sub_profilers.get("moe_transformer_layer")

        dense_fwd = (
            dense_p.measured_forward_time(batch, q_len) if (self._n_dense and dense_p) else 0.0
        )
        moe_fwd = moe_p.measured_forward_time(batch, q_len) if (self._n_moe and moe_p) else 0.0

        # Feature B: replace the implicit per-layer comm baked into the layer
        # forward time with the explicit, knob-driven model (delta approach so
        # default knobs reproduce the implicit cost exactly).
        comm = CommBreakdown()
        if self._comm is not None:
            overlap = (
                float(self._cc.prefill_overlap)
                if phase == "prefill"
                else float(self._cc.decode_overlap)
            )
            overlap = min(max(overlap, 0.0), 1.0)
            keep = 1.0 - overlap

            # Implicit comm currently inside the layer forward time.
            tp_ar_one = _estimate_tp_allreduce_time_ms(self._view, batch, q_len)
            builtin_dense_comm = 2.0 * tp_ar_one
            builtin_moe_a2a = _estimate_moe_a2a_time_ms(self._view, batch, q_len, self._gemm)
            builtin_moe_comm = 2.0 * tp_ar_one + builtin_moe_a2a

            # New explicit comm (already exposed; overlap applied below).
            new_tp_ar = self._comm.tp_allreduce_ms(batch, q_len)
            new_ep_a2a = self._comm.ep_a2a_ms(batch, q_len)

            if self._n_dense and dense_p:
                dense_fwd = dense_fwd - builtin_dense_comm + new_tp_ar * keep
            if self._n_moe and moe_p:
                moe_fwd = moe_fwd - builtin_moe_comm + (new_tp_ar + new_ep_a2a) * keep

            # Reportable breakdown (exposed times that contribute to latency).
            comm.tp_allreduce_ms = (self._n_dense + self._n_moe) * new_tp_ar * keep
            comm.ep_a2a_ms = self._n_moe * new_ep_a2a * keep
            comm.pp_p2p_ms = self._comm.pp_p2p_ms(batch, q_len) * keep

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

    def _decode_step_latency_ms(self, batch: int, kv_len: int, q_len: int = 1) -> float:
        ft = self._forward_times(batch, q_len, "decode", kv_len)
        return ft.total_ms

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

        ttft = self.prefill_latency_ms(batch, input_len)
        decode_total = self.decode_total_ms(batch, input_len, output_len)

        # Representative decode-step latency (mid context) for ITL/throughput.
        mid_ctx = input_len + output_len // 2
        spec_k = int(req.speculative_num_tokens or 0)
        q_len = (spec_k + 1) if spec_k > 0 else 1
        step_latency = self._decode_step_latency_ms(batch, mid_ctx, q_len=q_len)

        itl = (decode_total / output_len) if output_len > 0 else step_latency
        request_latency = ttft + decode_total
        per_req_decode_tps = (1000.0 / itl) if itl > 0 else 0.0
        decode_tps = (batch * 1000.0 / step_latency) if step_latency > 0 else 0.0
        replica_gpus = _replica_gpus(self.cfg)
        decode_tps_per_gpu = decode_tps / replica_gpus if replica_gpus else 0.0
        prefill_tps = (batch * input_len * 1000.0 / ttft) if ttft > 0 else 0.0

        extras = {"speculative_tokens_per_step": self._spec_tokens_per_step()}
        extras.update(self._comm_extras(batch, input_len, output_len))

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
            bw_gbps=disagg.kv_transfer_bw_gbps,
            latency_us=disagg.kv_transfer_latency_us,
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
        prefill_proj = InferencePerformanceProjector(prefill_cfg, args=self._args_ref)
        decode_proj = InferencePerformanceProjector(decode_cfg, args=self._args_ref)

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
    inference_config: InferenceConfig, args=None
) -> InferencePerfResult:
    return InferencePerformanceProjector(inference_config, args=args).project()
