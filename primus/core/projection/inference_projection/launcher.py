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
from .performance import project_inference_performance

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
    return overrides


def _print_performance(inference_config, perf) -> None:
    req = inference_config.request_config
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
        perf = project_inference_performance(
            inference_config, args=args, benchmark_layer_times=benchmark_layer_times
        )
        _print_performance(inference_config, perf)
        results["performance"] = perf

    return results
