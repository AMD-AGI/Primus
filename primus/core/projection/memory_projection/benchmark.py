###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Memory projection — *benchmark* mode orchestration.

Workflow::

    primus projection memory --memory-mode benchmark --config exp.yaml \
        [--benchmark-gpus N] [--target-nodes K] \
        [--save-benchmark out.json | --load-benchmark in.json]

The orchestrator reuses the perf-projection bench (``_run_layer_benchmark``)
because the same artifact serves both projections — see slice 5 of the
memory model design.  When ``--load-benchmark`` is provided, the bench
step is skipped and the projection runs purely from the saved JSON.
"""

from __future__ import annotations

import copy
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from primus.core.launcher.parser import load_primus_config
from primus.core.projection.memory_projection.extrapolation import (
    BenchMeasurement,
    extract_bench_measurement,
    extrapolate_per_rank_peak,
)
from primus.core.projection.memory_projection.reports import print_per_rank_breakdown
from primus.core.projection.module_profilers.language_model import (
    build_profiler,
    get_language_model_profiler_spec,
)
from primus.core.projection.config_validation import assert_recompute_pipeline_compat
from primus.core.projection.training_config import (
    convert_primus_config_to_projection_config,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_load_path(args) -> Optional[str]:
    """Pick up ``--load-benchmark`` (preferred) or the deprecated memory-side alias.

    Emits a deprecation warning on rank 0 when only the old name is set.
    """
    new = getattr(args, "load_benchmark", None)
    old = getattr(args, "compute_baseline", None)
    if old and not new and int(os.getenv("RANK", "0")) == 0:
        print(
            "[Primus:Memory Projection] WARNING: --compute-baseline is "
            "deprecated as a memory-side load flag; use --load-benchmark instead."
        )
    return new or old


def _resolve_save_path(args) -> Optional[str]:
    """Pick up ``--save-benchmark`` (preferred) or the deprecated alias.

    Emits a deprecation warning on rank 0 when only the old name is set.
    """
    new = getattr(args, "save_benchmark", None)
    old = getattr(args, "save_profiling", None)
    if old and not new and int(os.getenv("RANK", "0")) == 0:
        print(
            "[Primus:Memory Projection] WARNING: --save-profiling is "
            "deprecated; use --save-benchmark instead."
        )
    return new or old


def _build_bench_training_config(
    target_training_config,
    bench_summary: Dict[str, Any],
):
    """Clone ``target_training_config`` and override the bench-shaped fields.

    Used for the ``--load-benchmark`` path, where we don't have the
    original bench primus_config in memory but have the saved summary.
    """
    bench = copy.deepcopy(target_training_config)
    mc = bench.model_config
    mp = bench.model_parallel_config
    rt = bench.runtime_config

    if "num_layers" in bench_summary:
        mc.num_layers = int(bench_summary["num_layers"])
    if "moe_pattern" in bench_summary and bench_summary["moe_pattern"]:
        mc.moe_pattern = list(bench_summary["moe_pattern"])
    if "num_experts" in bench_summary:
        mc.num_experts = int(bench_summary["num_experts"] or 0)

    if "tensor_model_parallel_size" in bench_summary:
        mp.tensor_model_parallel_size = int(bench_summary["tensor_model_parallel_size"])
    if "pipeline_model_parallel_size" in bench_summary:
        mp.pipeline_model_parallel_size = int(bench_summary["pipeline_model_parallel_size"])
    if "virtual_pipeline_model_parallel_size" in bench_summary:
        mp.virtual_pipeline_model_parallel_size = int(
            bench_summary["virtual_pipeline_model_parallel_size"] or 1
        )
    # Layer bench runs at PP=1; keep the target's 16-stage layout string only when
    # PP×VPP matches, or LanguageModelProfiler raises during memory extrapolation.
    bench_pp = int(getattr(mp, "pipeline_model_parallel_size", 1) or 1)
    bench_vpp = int(getattr(mp, "virtual_pipeline_model_parallel_size", 1) or 1)
    if bench_pp <= 1:
        mp.virtual_pipeline_model_parallel_size = 1
        mp.pipeline_model_parallel_layout = None
    if "context_model_parallel_size" in bench_summary:
        mp.context_model_parallel_size = int(bench_summary["context_model_parallel_size"] or 1)
    if "expert_model_parallel_size" in bench_summary:
        mp.expert_model_parallel_size = int(bench_summary["expert_model_parallel_size"] or 1)
    if "recompute_granularity" in bench_summary:
        mp.recompute_granularity = bench_summary["recompute_granularity"]
    if "recompute_num_layers" in bench_summary:
        mp.recompute_num_layers = int(bench_summary["recompute_num_layers"] or 0)

    if "micro_batch_size" in bench_summary:
        rt.micro_batch_size = int(bench_summary["micro_batch_size"])
    if "sequence_length" in bench_summary:
        rt.sequence_length = int(bench_summary["sequence_length"])
    if "global_batch_size" in bench_summary:
        rt.global_batch_size = int(bench_summary["global_batch_size"])

    return bench


def _run_bench(
    args,
    overrides,
    primus_config_original,
    primus_config_bench,
    benchmark_gpus: int,
    gpus_per_node: int,
) -> Tuple[Dict[Any, Any], Dict[str, Any]]:
    """Run the perf-projection bench, returning (profiling_results, metadata).

    ``metadata`` here is the same dict that ``_save_profiling_results``
    would persist (parallelism + bench training-config summary).
    """
    # Lazy import — only needed when actually running a bench (which
    # itself needs the megatron backend on the import path).
    from primus.core.projection.performance_projection.projection import (
        _calculate_single_node_config,
        _run_layer_benchmark,
        _save_profiling_results,
    )

    module_config_for_calc = copy.deepcopy(primus_config_bench.get_module_config("pre_trainer"))
    reduction_info = _calculate_single_node_config(
        module_config_for_calc, gpus_per_node, benchmark_gpus=benchmark_gpus
    )

    if reduction_info["adjusted"]:
        # Apply the reduction to the bench primus_config (matches the perf
        # launcher's logic).
        bench_module = primus_config_bench.get_module_config("pre_trainer")
        bench_module.pipeline_model_parallel_size = reduction_info["benchmark_pp"]
        if reduction_info["benchmark_pp"] <= 1:
            if hasattr(bench_module, "virtual_pipeline_model_parallel_size"):
                bench_module.virtual_pipeline_model_parallel_size = 1
            if hasattr(bench_module, "pipeline_model_parallel_layout"):
                bench_module.pipeline_model_parallel_layout = None
        bench_module.expert_model_parallel_size = reduction_info["benchmark_ep"]
        if reduction_info["benchmark_tp"] != reduction_info["original_tp"]:
            bench_module.tensor_model_parallel_size = reduction_info["benchmark_tp"]
        if reduction_info.get("benchmark_num_experts") is not None:
            bench_module.num_experts = reduction_info["benchmark_num_experts"]

    profiling_results = _run_layer_benchmark(primus_config_bench, overrides, reduction_info)

    # Optionally persist the artifact.  We piggy-back on the perf saver so
    # that the JSON is identical-shape to the perf-side artifact (one
    # bench run, two projections).
    save_path = _resolve_save_path(args)
    if save_path and int(os.getenv("RANK", "0")) == 0:
        _save_profiling_results(profiling_results, reduction_info, save_path)
        print(f"[Primus:Memory Projection] Bench artifact saved to: {save_path}")

    metadata = {
        "benchmark_ep": reduction_info.get("benchmark_ep", 1),
        "benchmark_tp": reduction_info.get("benchmark_tp", 1),
        "benchmark_pp": reduction_info.get("benchmark_pp", 1),
        "benchmark_gpus": reduction_info.get("benchmark_gpus", benchmark_gpus),
        "original_ep": reduction_info.get("original_ep", 1),
        "original_tp": reduction_info.get("original_tp", 1),
        "original_pp": reduction_info.get("original_pp", 1),
        "original_cp": reduction_info.get("original_cp", 1),
        "original_num_experts": reduction_info.get("original_num_experts"),
        "benchmark_num_experts": reduction_info.get("benchmark_num_experts"),
        "bench_training_config_summary": reduction_info.get("bench_training_config_summary"),
    }
    return profiling_results, metadata


def _load_bench(load_path: str) -> Tuple[Dict[Any, Any], Dict[str, Any]]:
    """Load a previously saved bench artifact."""
    from primus.core.projection.performance_projection.projection import (
        _load_artifact,
    )

    payload = _load_artifact(load_path)
    profiling_results = payload.get("profiling_results", {}) or {}
    # Re-key integer layer indices.
    keyed: Dict[Any, Any] = {}
    for k, v in profiling_results.items():
        try:
            keyed[int(k)] = v
        except (TypeError, ValueError):
            keyed[k] = v

    mem = payload.get("memory_results")
    if mem is not None:
        keyed["_memory_benchmark"] = mem

    return keyed, payload.get("metadata", {}) or {}


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def launch_projection_from_cli(args, overrides):
    is_rank_0 = int(os.getenv("RANK", "0")) == 0

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"[Primus:Memory Projection] Config file '{cfg_path}' not found."
        )

    primus_config, _unknown = load_primus_config(args, overrides or [])
    primus_config_original = copy.deepcopy(primus_config)
    primus_config_bench = copy.deepcopy(primus_config)

    gpus_per_node = int(os.getenv("GPUS_PER_NODE", "8"))
    benchmark_gpus = getattr(args, "benchmark_gpus", None) or gpus_per_node

    # Target topology: derived from the original parallelism config unless
    # the user explicitly sets --target-nodes.
    module_cfg = primus_config_original.get_module_config("pre_trainer")
    target_tp = getattr(module_cfg, "tensor_model_parallel_size", 1) or 1
    target_pp = getattr(module_cfg, "pipeline_model_parallel_size", 1) or 1
    target_ep = getattr(module_cfg, "expert_model_parallel_size", 1) or 1
    target_cp = getattr(module_cfg, "context_parallel_size", 1) or 1
    min_target_gpus = max(1, target_tp * target_pp * target_cp)
    target_nodes = getattr(args, "target_nodes", None)
    if target_nodes is None:
        target_nodes = max(1, (min_target_gpus + gpus_per_node - 1) // gpus_per_node)

    safety_margin = float(getattr(args, "memory_safety_margin", 0.05) or 0.05)

    # ── Source the bench data: load or run ──
    load_path = _resolve_load_path(args)
    if load_path:
        if is_rank_0:
            print(
                f"[Primus:Memory Projection] Loading bench artifact: {load_path}"
            )
        profiling_results, metadata = _load_bench(load_path)
    else:
        if is_rank_0:
            print(
                f"[Primus:Memory Projection] Running bench on {benchmark_gpus} GPU(s)..."
            )
        profiling_results, metadata = _run_bench(
            args,
            overrides or [],
            primus_config_original,
            primus_config_bench,
            benchmark_gpus,
            gpus_per_node,
        )

    # ── Build target & bench training_configs and profilers ──
    target_training_config = convert_primus_config_to_projection_config(primus_config_original)
    assert_recompute_pipeline_compat(
        target_training_config,
        primus_config=primus_config_original,
        pipeline_schedule_algorithm=getattr(args, "pipeline_schedule_algorithm", None),
    )

    bench_summary = metadata.get("bench_training_config_summary")
    if bench_summary:
        bench_training_config = _build_bench_training_config(
            target_training_config, bench_summary
        )
    else:
        # Best-effort fallback: assume bench config equals target with the
        # parallelism dims overridden.  Activation correction will still
        # work because per-layer measurements are independent of layout.
        if is_rank_0:
            print(
                "[Primus:Memory Projection] No bench training-config summary in artifact; "
                "falling back to target config with bench parallelism overrides."
            )
        bench_training_config = copy.deepcopy(target_training_config)
        mp = bench_training_config.model_parallel_config
        mp.tensor_model_parallel_size = int(metadata.get("benchmark_tp", 1) or 1)
        mp.pipeline_model_parallel_size = int(metadata.get("benchmark_pp", 1) or 1)
        mp.expert_model_parallel_size = int(metadata.get("benchmark_ep", 1) or 1)
        if metadata.get("benchmark_num_experts") is not None:
            bench_training_config.model_config.num_experts = int(
                metadata["benchmark_num_experts"]
            )

    target_profiler = build_profiler(get_language_model_profiler_spec(target_training_config))
    bench_profiler = build_profiler(get_language_model_profiler_spec(bench_training_config))

    # ── Extract bench measurement ──
    bm: BenchMeasurement = extract_bench_measurement(profiling_results)
    if bm.global_peak_allocated_bytes == 0 and bm.global_peak_reserved_bytes == 0:
        if is_rank_0:
            print(
                "[Primus:Memory Projection] WARNING: bench artifact contains no "
                "memory_results.  Falling back to analytical-only projection at target."
            )

    # ── Bench cluster shape (for analytical-at-bench evaluation) ──
    bench_gpus = int(metadata.get("benchmark_gpus", benchmark_gpus) or benchmark_gpus)
    if bench_gpus <= gpus_per_node:
        bench_nnodes = 1
        bench_gpus_per_node_eff = bench_gpus
    else:
        bench_nnodes = (bench_gpus + gpus_per_node - 1) // gpus_per_node
        bench_gpus_per_node_eff = gpus_per_node

    target_seq_len = target_training_config.runtime_config.sequence_length
    target_micro_batch = target_training_config.runtime_config.micro_batch_size

    projection = extrapolate_per_rank_peak(
        bench_profiler=bench_profiler,
        target_profiler=target_profiler,
        bench=bm,
        bench_training_config=bench_training_config,
        target_training_config=target_training_config,
        bench_nnodes=bench_nnodes,
        bench_gpus_per_node=bench_gpus_per_node_eff,
        target_nnodes=int(target_nodes),
        target_gpus_per_node=gpus_per_node,
        batch_size=target_micro_batch,
        seq_len=target_seq_len,
        safety_margin=safety_margin,
    )

    # ── Total VRAM probe (best-effort, only if running on GPU) ──
    total_vram_bytes = 0
    try:
        import torch

        if torch.cuda.is_available():
            _, total_vram_bytes = torch.cuda.mem_get_info()
    except Exception:
        pass

    if is_rank_0:
        target_label = (
            f"{target_nodes} nodes × {gpus_per_node} GPUs "
            f"(TP={target_tp}, PP={target_pp}, EP={target_ep}, CP={target_cp})"
        )
        print_per_rank_breakdown(
            projection,
            target_label=target_label,
            total_vram_bytes=int(total_vram_bytes),
        )

    return projection


# Keep ``replace`` referenced so unused-import linters don't whine — the
# helper is exported for callers who want to clone training configs in
# tests.
__all__ = ["launch_projection_from_cli", "replace"]
