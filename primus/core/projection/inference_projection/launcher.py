###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""CLI launcher for ``primus projection inference``."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from primus.core.launcher.parser import load_primus_config
from primus.core.projection.training_config import (
    convert_primus_config_to_inference_config,
)

from .memory import project_inference_memory
from .performance import InferencePerformanceProjector, project_inference_performance

# Map CLI arg attribute names → InferenceRequestConfig field names.
_ARG_TO_FIELD = {
    "input_len": "input_seq_len",
    "output_len": "output_seq_len",
    "inference_batch_size": "batch_size",
    "max_concurrency": "max_concurrency",
    "max_context_len": "max_context_len",
    "weight_dtype": "weight_dtype",
    "kv_cache_dtype": "kv_cache_dtype",
    "chunked_prefill_size": "chunked_prefill_size",
    "speculative_num_tokens": "speculative_num_tokens",
    "speculative_acceptance_rate": "speculative_acceptance_rate",
    "serving_model": "serving_model",
    "decode_step_overhead_us": "decode_step_overhead_us",
    "mixed_batch_penalty": "mixed_batch_penalty",
    "cudagraph_mode": "cudagraph_mode",
    "kv_cache_memory_fraction": "kv_cache_memory_fraction",
    "kv_block_size": "kv_block_size",
    "max_num_batched_tokens": "max_num_batched_tokens",
    "ep_load_balance": "ep_load_balance",
    "redundant_experts": "redundant_experts",
    "request_rate": "request_rate",
    "arrival_model": "arrival_model",
    "attention_backend": "attention_backend",
    "sparse_attention_topk": "sparse_attention_topk",
    "moe_expert_dtype": "moe_expert_dtype",
    "speculative_draft_cost_factor": "speculative_draft_cost_factor",
}

# Feature B (custom collective ops): CLI arg → ``collective_<field>`` key.
_COLL_ARG_TO_FIELD = {
    "tp_allreduce_algo": "collective_tp_allreduce_algo",
    "ep_a2a_algo": "collective_ep_a2a_algo",
    "prefill_comm_overlap": "collective_prefill_overlap",
    "decode_comm_overlap": "collective_decode_overlap",
    "tp_allreduce_efficiency": "collective_tp_allreduce_efficiency",
    "ep_a2a_efficiency": "collective_ep_a2a_efficiency",
}

# Feature A (prefill/decode disaggregation): CLI arg → ``disagg_<field>`` key.
_DISAGG_ARG_TO_FIELD = {
    "prefill_tp": "disagg_prefill_tp",
    "prefill_pp": "disagg_prefill_pp",
    "prefill_ep": "disagg_prefill_ep",
    "decode_tp": "disagg_decode_tp",
    "decode_pp": "disagg_decode_pp",
    "decode_ep": "disagg_decode_ep",
    "prefill_replicas": "disagg_prefill_replicas",
    "decode_replicas": "disagg_decode_replicas",
    "kv_transfer_bw_gbps": "disagg_kv_transfer_bw_gbps",
    "kv_transfer_latency_us": "disagg_kv_transfer_latency_us",
    "transfer_backend": "disagg_transfer_backend",
}

_GB = 1024.0 ** 3


def _collect_inference_overrides(args) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    for mapping in (_ARG_TO_FIELD, _COLL_ARG_TO_FIELD, _DISAGG_ARG_TO_FIELD):
        for arg_name, field_name in mapping.items():
            if hasattr(args, arg_name):
                val = getattr(args, arg_name)
                if val is not None:
                    overrides[field_name] = val

    # --comm-model {explicit,builtin} → collective_enabled.
    comm_model = getattr(args, "comm_model", None)
    if comm_model is not None:
        overrides["collective_enabled"] = comm_model != "builtin"
    # --disaggregate flag → disagg_enabled.
    if getattr(args, "disaggregate", False):
        overrides["disagg_enabled"] = True
    # store_true serving flags: only override when actually set, so they never
    # clobber a value carried by a YAML ``inference:`` block.
    if getattr(args, "fused_kernels", False):
        overrides["fused_kernels"] = True
    if getattr(args, "quick_reduce", False):
        overrides["collective_quick_reduce"] = True
    if getattr(args, "fuse_rmsnorm_allreduce", False):
        overrides["collective_fuse_rmsnorm_allreduce"] = True
    return overrides


def _print_performance(inference_config, perf) -> None:
    req = inference_config.request_config
    mc = inference_config.model_config
    print("\n" + "=" * 100)
    print("[Primus:Inference] Performance Projection")
    print("=" * 100)
    print(
        f"  Workload: input={req.input_seq_len} tok, output={req.output_seq_len} tok, "
        f"batch={req.batch_size}"
    )
    feats = []
    if req.chunked_prefill_size:
        feats.append(f"chunked_prefill={req.chunked_prefill_size}")
    if req.speculative_num_tokens:
        feats.append(
            f"speculative(k={req.speculative_num_tokens}, "
            f"accept={req.speculative_acceptance_rate})"
        )
    if req.kv_cache_dtype != "bf16":
        feats.append(f"kv_dtype={req.kv_cache_dtype}")
    if getattr(mc, "use_turbo_deepep", False):
        sync_free = getattr(mc, "turbo_sync_free_moe_stage", 0) or 0
        feats.append("deepep" + (f"(sync_free={sync_free})" if sync_free else ""))
    if req.cudagraph_mode:
        feats.append(f"cudagraph={req.cudagraph_mode}")
    if req.kv_cache_memory_fraction:
        feats.append(f"kv_mem_frac={req.kv_cache_memory_fraction:.2f}")
    if req.kv_block_size:
        feats.append(f"kv_block={req.kv_block_size}")
    if req.max_num_batched_tokens:
        feats.append(f"max_batched_tokens={req.max_num_batched_tokens}")
    if req.ep_load_balance and req.ep_load_balance != 1.0:
        feats.append(f"ep_load_balance={req.ep_load_balance:.2f}")
    if req.redundant_experts:
        feats.append(f"redundant_experts={req.redundant_experts}")
    if getattr(req, "attention_backend", None):
        feats.append(f"attn_backend={req.attention_backend}")
    if getattr(req, "sparse_attention_topk", 0):
        feats.append(f"sparse_attn_topk={req.sparse_attention_topk}")
    if getattr(req, "moe_expert_dtype", None):
        feats.append(f"moe_expert_dtype={req.moe_expert_dtype}")
    if getattr(req, "fused_kernels", False):
        feats.append("fused_kernels")
    if getattr(req, "speculative_draft_cost_factor", 0.0):
        feats.append(f"draft_cost={req.speculative_draft_cost_factor:.2f}")
    if getattr(req, "request_rate", 0.0) and (req.arrival_model or "closed") != "closed":
        feats.append(f"request_rate={req.request_rate:g}/s({req.arrival_model})")
    cc = inference_config.collective_config
    if getattr(cc, "quick_reduce", False):
        feats.append("quick_reduce")
    if getattr(cc, "fuse_rmsnorm_allreduce", False):
        feats.append("fused_rmsnorm_ar")
    if feats:
        print(f"  Features: {', '.join(feats)}")
    if perf.is_disaggregated:
        print("  Mode: prefill/decode DISAGGREGATED")
    if perf.extras.get("serving_continuous_batching"):
        print(
            f"  Serving model: CONTINUOUS BATCHING "
            f"(concurrency={int(perf.extras.get('concurrency', req.batch_size))})"
        )
    elif not perf.is_disaggregated:
        print("  Serving model: STATIC (pure-decode batch)")
    src = "BENCHMARK (GPU-calibrated)" if perf.extras.get("benchmark_calibrated") else "SIMULATION"
    print(f"  Profiling source: {src}")
    print("-" * 100)
    print(f"  TTFT (time to first token):      {perf.ttft_ms:.2f} ms")
    if perf.is_disaggregated:
        print(f"    prefill compute:               {perf.extras.get('prefill_compute_ttft_ms', 0.0):.2f} ms")
        print(f"    KV-cache transfer:             {perf.kv_transfer_ms:.2f} ms")
    print(f"  ITL / TPOT (per token):          {perf.itl_ms:.2f} ms")
    print(f"  Interactivity (per user):        {perf.per_request_decode_tps:.1f} tok/s/user")
    if perf.extras.get("serving_continuous_batching"):
        print(
            f"  Decode step latency (pure):      {perf.decode_step_latency_ms:.2f} ms"
            f"  | mixed: {perf.extras.get('mixed_step_latency_ms', 0.0):.2f} ms"
        )
        print(
            f"    Mixed-step fraction:           {perf.extras.get('mixed_step_fraction', 0.0) * 100:.2f}%"
            f"  → TPOT pollution: {perf.extras.get('tpot_pollution_pct', 0.0):.1f}%"
        )
    else:
        print(f"  Decode step latency (batch):     {perf.decode_step_latency_ms:.2f} ms")
    print(f"  End-to-end request latency:      {perf.request_latency_ms:.2f} ms")
    if "offered_request_rate" in perf.extras:
        sat = " [SATURATED]" if perf.extras.get("saturated") else ""
        print(
            f"  Offered load:                    {perf.extras['offered_request_rate']:g} req/s "
            f"(max sustainable {perf.extras.get('max_sustainable_request_rate', 0.0):.2f} req/s, "
            f"utilization {perf.extras.get('utilization', 0.0) * 100:.0f}%){sat}"
        )
        print(f"  Queue wait (in TTFT):            {perf.extras.get('queue_wait_ms', 0.0):.2f} ms")
    print("-" * 100)
    print(f"  Per-request decode throughput:   {perf.per_request_decode_tps:.1f} tok/s")
    print(f"  Aggregate decode throughput:     {perf.decode_throughput_tps:.1f} tok/s")
    print(f"  Decode throughput / GPU:         {perf.decode_throughput_tps_per_gpu:.1f} tok/s/gpu")
    print(f"  Prefill throughput:              {perf.prefill_throughput_tps:.1f} tok/s")
    if perf.is_disaggregated:
        print(
            f"  Prefill pool:                    {int(perf.extras.get('prefill_replicas', 1))} "
            f"replica(s) x {perf.prefill_replica_gpus} GPU"
        )
        print(
            f"  Decode pool:                     {int(perf.extras.get('decode_replicas', 1))} "
            f"replica(s) x {perf.decode_replica_gpus} GPU"
        )
    else:
        print(f"  Replica GPUs (TP×PP):            {perf.replica_gpus}")
    if perf.extras.get("speculative_tokens_per_step", 1.0) > 1.0:
        print(
            f"  Speculative tokens / step:       "
            f"{perf.extras['speculative_tokens_per_step']:.2f}"
        )
    # Feature B: explicit communication breakdown (exposed, post-overlap).
    if "comm_prefill_total_ms" in perf.extras:
        print("-" * 100)
        print("  Communication breakdown (exposed ms/forward):")
        print(
            f"    prefill:  TP-AR {perf.extras['comm_prefill_tp_allreduce_ms']:.2f} | "
            f"EP-A2A {perf.extras['comm_prefill_ep_a2a_ms']:.2f} | "
            f"PP-P2P {perf.extras['comm_prefill_pp_p2p_ms']:.2f} | "
            f"total {perf.extras['comm_prefill_total_ms']:.2f}"
        )
        print(
            f"    decode:   TP-AR {perf.extras['comm_decode_tp_allreduce_ms']:.2f} | "
            f"EP-A2A {perf.extras['comm_decode_ep_a2a_ms']:.2f} | "
            f"PP-P2P {perf.extras['comm_decode_pp_p2p_ms']:.2f} | "
            f"total {perf.extras['comm_decode_total_ms']:.2f}"
        )
    print("=" * 100)


def _print_des(des: Dict[str, object]) -> None:
    point = des["point"]
    print("\n" + "=" * 100)
    print("[Primus:Inference] Discrete-Event Simulation (arrival-driven)")
    print("=" * 100)
    sat = " [SATURATED]" if point.saturated else ""
    print(
        f"  Arrivals: {point.arrival_model} @ {point.offered_rate:g} req/s offered  "
        f"(achieved {point.achieved_rate:.2f} req/s, "
        f"utilization {point.utilization * 100:.0f}%){sat}"
    )
    print(
        f"  Simulated: {point.num_requests} requests over {point.makespan_ms / 1000.0:.2f} s  "
        f"→ system throughput {point.system_throughput_tps:.0f} tok/s"
    )
    print("-" * 100)
    print(f"  {'metric':<22}{'mean':>12}{'p50':>12}{'p90':>12}{'p99':>12}")

    def _row(label: str, d: Dict[str, float], unit: str = "ms") -> None:
        print(
            f"  {label:<22}"
            f"{d.get('mean', 0.0):>10.2f} {unit:<1}"
            f"{d.get('p50', 0.0):>10.2f} {unit:<1}"
            f"{d.get('p90', 0.0):>10.2f} {unit:<1}"
            f"{d.get('p99', 0.0):>10.2f} {unit:<1}"
        )

    _row("TTFT", point.ttft)
    _row("TPOT (per token)", point.tpot)
    _row("ITL (inter-token)", point.itl)
    _row("End-to-end latency", point.e2e)

    curve = des.get("curve")
    if curve:
        mu = des.get("max_sustainable_rate", 0.0)
        print("-" * 100)
        print(f"  Throughput–latency curve (max sustainable ≈ {mu:.2f} req/s):")
        print(
            f"  {'load(req/s)':>12}{'util%':>8}{'tok/s':>10}"
            f"{'TTFT p50':>11}{'TTFT p99':>11}{'TPOT p50':>11}{'TPOT p99':>11}"
        )
        for r in curve:
            flag = "*" if r.saturated else " "
            print(
                f"  {r.offered_rate:>11.2f}{flag}{r.utilization * 100:>7.0f}"
                f"{r.system_throughput_tps:>10.0f}"
                f"{r.ttft.get('p50', 0.0):>11.1f}{r.ttft.get('p99', 0.0):>11.1f}"
                f"{r.tpot.get('p50', 0.0):>11.2f}{r.tpot.get('p99', 0.0):>11.2f}"
            )
    print("=" * 100)


def launch_projection_from_cli(args, overrides):
    """Entry point for ``primus projection inference``."""
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"[Primus:Inference] Config file '{cfg_path}' not found.")

    primus_config, _unknown = load_primus_config(args, overrides or [])

    # Internal: this process *is* the GPU benchmark worker (spawned under
    # torchrun by the parent). Build the real model, time forward-only layers,
    # write the result JSON, and exit — no projection happens here.
    if getattr(args, "inference_bench_worker", False):
        from .benchmark import run_inference_benchmark_worker

        run_inference_benchmark_worker(primus_config, _unknown, args)
        return {}

    inf_overrides = _collect_inference_overrides(args)
    inference_config = convert_primus_config_to_inference_config(
        primus_config, inference_overrides=inf_overrides
    )

    # DeepEP / SyncFree (shared perf flags) — enable async EP All-to-All overlap
    # for the serving projection, mirroring the training projection override.
    if getattr(args, "enable_deepep", False):
        inference_config.model_config.use_turbo_deepep = True
    sync_free_stage = getattr(args, "sync_free_stage", 0) or 0
    if sync_free_stage > 0:
        inference_config.model_config.turbo_sync_free_moe_stage = sync_free_stage
        inference_config.model_config.use_turbo_deepep = True

    mode = getattr(args, "inference_mode", "both") or "both"
    hbm_gb = getattr(args, "hbm_capacity_gb", None)
    profiling_mode = getattr(args, "profiling_mode", "simulate") or "simulate"

    # Benchmark mode: spawn a torchrun worker to measure forward-only layer
    # times on real GPUs, then calibrate the analytical projection to them.
    benchmark_layer_times = None
    load_bench = getattr(args, "load_benchmark", None)
    if load_bench and mode in ("performance", "both"):
        # Reuse a previously-saved GPU layer benchmark (skips the spawn). Lets a
        # concurrency sweep calibrate against one bench run.
        import json as _json

        with open(load_bench) as _f:
            benchmark_layer_times = _json.load(_f)
        print(f"[Primus:Inference] loaded GPU benchmark from {load_bench}")
    elif profiling_mode == "benchmark" and mode in ("performance", "both"):
        from .benchmark import spawn_inference_benchmark

        benchmark_layer_times = spawn_inference_benchmark(args, overrides)
        if benchmark_layer_times is None:
            print(
                "[Primus:Inference] benchmark unavailable — falling back to "
                "simulation for the performance projection."
            )

    results = {}
    if mode in ("memory", "both"):
        results["memory"] = project_inference_memory(
            inference_config, hbm_capacity_gb=hbm_gb, verbose=True
        )
    if mode in ("performance", "both"):
        projector = InferencePerformanceProjector(
            inference_config, args=args, benchmark_layer_times=benchmark_layer_times
        )
        perf = projector.project()
        _print_performance(inference_config, perf)
        results["performance"] = perf

        # Phase 3: opt-in discrete-event simulation for arrival-driven
        # percentiles. Reuses ``projector`` as the (possibly benchmark-
        # calibrated) cost kernel for each step's duration.
        req = inference_config.request_config
        arrival_model = (getattr(req, "arrival_model", "closed") or "closed").lower()
        if arrival_model in ("poisson", "deterministic") and (req.request_rate or 0) > 0:
            from .des import run_des

            des = run_des(
                inference_config,
                projector,
                arrival_model=arrival_model,
                rate_per_s=float(req.request_rate),
                num_requests=int(getattr(args, "des_num_requests", 400) or 400),
                seed=int(getattr(args, "des_seed", 0) or 0),
                sweep=bool(getattr(args, "des_sweep", False)),
            )
            _print_des(des)
            results["des"] = des

    return results
