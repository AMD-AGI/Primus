###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse as _argparse
import os
import tempfile


def run(args, overrides):
    """
    Entry point for the 'projection' subcommand.
    """
    framework = "megatron"

    if args.suite == "memory":
        # Benchmark / both modes need the backend on the import path
        # because they actually run the trainer.  Simulate stays import-free.
        # Default to benchmark: real-GPU anchored projection is what users
        # almost always want; simulate is an explicit opt-in for no-GPU hosts.
        memory_mode = getattr(args, "memory_mode", "benchmark") or "benchmark"
        if memory_mode in ("benchmark", "both"):
            # If only loading a previously saved artifact, no backend needed.
            load_path = getattr(args, "load_benchmark", None) or getattr(args, "compute_baseline", None)
            if not load_path:
                from primus.pretrain import setup_backend_path

                setup_backend_path(framework=framework, verbose=True)

        from primus.core.projection.memory_projection import launch_projection_from_cli

        launch_projection_from_cli(args, overrides)
    elif args.suite == "performance":
        profiling_mode = getattr(args, "profiling_mode", "benchmark")
        # When running purely from a saved artifact (--load-benchmark), the
        # bench is skipped entirely, so we don't need to import the backend.
        # Same for pure-simulation mode.
        load_benchmark_path = getattr(args, "load_benchmark", None)
        needs_backend = profiling_mode != "simulate" and not load_benchmark_path

        if needs_backend:
            from primus.pretrain import setup_backend_path

            setup_backend_path(framework=framework, verbose=True)

        from primus.core.projection.performance_projection import (
            launch_projection_from_cli,
        )

        launch_projection_from_cli(args, overrides)
    elif args.suite == "inference":
        # Inference / serving projection.  Simulation mode is a forward-only
        # analytical model (no training backend needed), but benchmark mode —
        # and the spawned benchmark worker — build the real model and therefore
        # require the Megatron backend on the path.
        inf_profiling_mode = getattr(args, "profiling_mode", "simulate")
        is_bench_worker = getattr(args, "inference_bench_worker", False)
        if is_bench_worker or inf_profiling_mode == "benchmark":
            from primus.pretrain import setup_backend_path

            setup_backend_path(framework=framework, verbose=True)

        from primus.core.projection.inference_projection import (
            launch_projection_from_cli,
        )

        launch_projection_from_cli(args, overrides)
    elif args.suite == "both":
        # Run the perf bench once, save the artifact, then run memory
        # projection from the loaded artifact (no second bench).
        from primus.core.projection.memory_projection.benchmark import (
            launch_projection_from_cli as memory_benchmark_launch,
        )
        from primus.core.projection.performance_projection import (
            launch_projection_from_cli as performance_launch,
        )
        from primus.pretrain import setup_backend_path

        setup_backend_path(framework=framework, verbose=True)

        # If the user did not pre-set --save-benchmark / --save-profiling
        # we allocate a temp file so the perf bench's artifact can be
        # consumed by the memory side.
        save_path = getattr(args, "save_benchmark", None) or getattr(args, "save_profiling", None)
        cleanup_save = False
        if not save_path:
            fd, save_path = tempfile.mkstemp(prefix="primus_projection_", suffix=".json")
            os.close(fd)
            cleanup_save = True
        # The perf launcher reads `save_profiling` (the historical name);
        # set both so deprecated and current callers pick it up.
        args.save_profiling = save_path
        args.save_benchmark = save_path

        try:
            performance_launch(args, overrides)

            # Now run the memory projection from the artifact.  Switch
            # args to "load" mode so the memory benchmark skips the bench.
            args.load_benchmark = save_path
            args.compute_baseline = save_path  # deprecated alias mirror
            memory_benchmark_launch(args, overrides)
        finally:
            if cleanup_save:
                try:
                    os.unlink(save_path)
                except OSError:
                    # Best-effort cleanup: ignore temp-file deletion failures
                    # to avoid masking projection/benchmark errors.
                    pass
    else:
        raise NotImplementedError(f"Unsupported projection suite: {args.suite}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-suite arg registration (factored out so the `both` subparser can
# reuse the exact same surface).
# ─────────────────────────────────────────────────────────────────────────────


def _add_memory_safety_margin_arg(parser):
    parser.add_argument(
        "--memory-safety-margin",
        type=float,
        required=False,
        default=0.05,
        help=(
            "Safety margin applied on top of the residual-overhead term when "
            "computing the upper-bound peak (used for OOM-fits decisions). "
            "Default: 0.05 (5%%)."
        ),
    )


def _add_pipeline_schedule_algorithm_arg(parser):
    parser.add_argument(
        "--pipeline-schedule-algorithm",
        type=str,
        required=False,
        default="auto",
        choices=[
            "auto",
            "zerobubble",
            "zerobubble-heuristic",
            "zbv-formatted",
            "zbv-greedy-half",
            "zbv-greedy-min",
            "seaailab-ilp",
            "all",
        ],
        help=(
            "Pipeline schedule for validation and (perf) simulation. "
            "Must not be combined with activation recompute — split-wgrad "
            "schedules pin inputs that recompute cannot free."
        ),
    )


def _add_memory_args(parser):
    _add_pipeline_schedule_algorithm_arg(parser)
    parser.add_argument(
        "--memory-mode",
        type=str,
        required=False,
        default="benchmark",
        choices=["simulate", "benchmark", "both"],
        help=(
            "Memory projection mode:\n"
            "  benchmark - (default) Run sub-node layer benchmark, capture per-rank\n"
            "              memory, and analytically extrapolate to target cluster.\n"
            "              OOM-accurate. Requires a ROCm GPU on the local host.\n"
            "  simulate  - Analytical only (no GPU). Opt in only when the host has\n"
            "              no GPU; cluster numbers will be analytical, not anchored.\n"
            "  both      - Run both and print a side-by-side comparison.\n"
        ),
    )
    _add_memory_safety_margin_arg(parser)


def _add_save_benchmark_arg(parser):
    """``--save-benchmark`` (with deprecated ``--save-profiling`` alias).

    The bench artifact (timing + memory) is shared between perf and
    memory projections; both subparsers accept the same save flag.
    """
    parser.add_argument(
        "--save-benchmark",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to write the bench artifact JSON (timing + memory). "
            "The artifact is shareable: a single bench run can feed both "
            "perf and memory projections via --load-benchmark."
        ),
    )
    parser.add_argument(
        "--save-profiling",
        type=str,
        required=False,
        default=None,
        help=_argparse.SUPPRESS,  # deprecated alias for --save-benchmark
    )


def _add_load_benchmark_arg(parser, *, include_compute_baseline_alias: bool):
    """``--load-benchmark`` (skip bench, project from saved artifact).

    Memory subparser: accept ``--compute-baseline`` as a deprecated alias
    (same semantic — "load a saved bench artifact").

    Perf subparser: do NOT alias ``--compute-baseline`` — that flag has a
    different meaning on the perf side (the bg=1 hybrid baseline), and
    it is registered separately.
    """
    parser.add_argument(
        "--load-benchmark",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to a previously saved bench artifact JSON. When provided, "
            "the bench is skipped and the projection runs directly from the "
            "loaded measurements."
        ),
    )
    if include_compute_baseline_alias:
        parser.add_argument(
            "--compute-baseline",
            type=str,
            required=False,
            default=None,
            help=_argparse.SUPPRESS,  # deprecated memory-side alias for --load-benchmark
        )


def _add_perf_compute_baseline_arg(parser):
    """Perf-side ``--compute-baseline`` (bg=1 hybrid baseline).

    Distinct semantic from ``--load-benchmark``: this is a *secondary*
    artifact used to source clean compute timings during EP-reduced
    benches.  Kept hidden from --help (internal/subprocess use).
    """
    parser.add_argument(
        "--compute-baseline",
        type=str,
        required=False,
        default=None,
        help=_argparse.SUPPRESS,
    )


def _add_topology_args(parser):
    """``--target-nodes`` / ``--benchmark-gpus`` — shared between memory and perf."""
    parser.add_argument(
        "--target-nodes",
        type=int,
        required=False,
        default=None,
        help=(
            "Target number of nodes for projection. "
            "If not specified, defaults to the minimum nodes required by "
            "the parallelism config (TP × PP × CP / GPUs_per_node)."
        ),
    )
    parser.add_argument(
        "--benchmark-gpus",
        type=int,
        required=False,
        default=None,
        help=(
            "Number of GPUs to use for the underlying bench. When set lower "
            "than GPUS_PER_NODE, enables sub-node benchmarking with "
            "analytical upscaling. Defaults to GPUS_PER_NODE."
        ),
    )


def _add_performance_args(parser):
    """All the perf-specific knobs (profiling-mode, gpu-arch, schedules, etc.)."""
    parser.add_argument(
        "--hardware-config",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to YAML file with hardware configuration for collective communication modeling. "
            "If not provided, uses default cluster parameters.\n\n"
        ),
    )
    parser.add_argument(
        "--profiling-mode",
        type=str,
        required=False,
        default="benchmark",
        choices=["benchmark", "simulate", "both"],
        help=(
            "Profiling mode for layer timing:\n"
            "  benchmark  - Run actual GPU benchmarks (default, requires GPU)\n"
            "  simulate   - Use simulation backends (origami for GEMM,\n"
            "               analytical model for SDPA). No GPU required.\n"
            "  both       - Run both benchmark and simulation, report side-by-side\n"
        ),
    )
    parser.add_argument(
        "--gemm-backend",
        type=str,
        required=False,
        default=None,
        choices=["origami"],
        help=(
            "GEMM simulation backend (only used when --profiling-mode is 'simulate' or 'both').\n"
            "  origami  - Open-source GEMM performance model (default)\n"
        ),
    )
    parser.add_argument(
        "--gpu-arch",
        type=str,
        required=False,
        default=None,
        help=(
            "Target GPU architecture for simulation (e.g. 'mi300x', 'gfx942', 'mi355x', 'gfx950').\n"
            "If not specified, auto-detected or uses PRIMUS_GPU_ARCH env var.\n"
        ),
    )
    parser.add_argument(
        "--gpu-clock-mhz",
        type=int,
        required=False,
        default=None,
        help=(
            "Override the GPU compute clock frequency in MHz for simulation.\n"
            "If not specified, uses the default from the hardware profile for the\n"
            "given --gpu-arch (e.g. 2100 MHz for MI300X/MI325X).\n"
            "Can also be set via the PRIMUS_GPU_CLOCK_MHZ env var.\n"
            "Example: --gpu-clock-mhz 1500\n"
        ),
    )
    _add_pipeline_schedule_algorithm_arg(parser)
    # Projection-specific overrides.
    parser.add_argument(
        "--target-num-nodes",
        type=int,
        required=False,
        default=None,
        help="Target number of nodes for multinode projection (alias for --target-nodes).",
    )
    parser.add_argument(
        "--target-ep-size",
        type=int,
        required=False,
        default=None,
        help="Override expert_model_parallel_size for projection target.",
    )
    parser.add_argument(
        "--enable-zero-bubble",
        action="store_true",
        default=False,
        help="Enable zero-bubble pipeline scheduling.",
    )
    parser.add_argument(
        "--enable-deepep",
        action="store_true",
        default=False,
        help="Enable DeepEP (async All-to-All overlap with compute).",
    )
    parser.add_argument(
        "--sync-free-stage",
        type=int,
        required=False,
        default=0,
        help="SyncFree MoE stage (0=off, 1=fused router, 2=+DeepEP+grouped, 3=+fused act). Auto-enables DeepEP.",
    )
    parser.add_argument(
        "--num-virtual-stages-per-pipeline-rank",
        type=int,
        required=False,
        default=None,
        help="Override virtual_pipeline_model_parallel_size (VPP) for projection.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        required=False,
        default=None,
        help="Override micro_batch_size for projection.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        required=False,
        default=None,
        help="Override global_batch_size for projection.",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        default=False,
        help=_argparse.SUPPRESS,
    )


def _add_inference_args(parser):
    """Inference / serving projection knobs.

    Reuses ``--gpu-arch`` / ``--gpu-clock-mhz`` / ``--gemm-backend`` from the
    perf arg group (added separately) for the simulation backends.
    """
    parser.add_argument(
        "--inference-mode",
        type=str,
        required=False,
        default="both",
        choices=["performance", "memory", "both"],
        help="Which inference projection to run (default: both).",
    )
    # ---- Request / serving workload ----
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Prompt length in tokens (prefill). Defaults to the config seq_length.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Number of tokens to generate (decode steps).",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=None,
        help="Number of sequences processed together per decode forward.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Max resident sequences for KV-cache sizing (default: batch size).",
    )
    parser.add_argument(
        "--max-context-len",
        type=int,
        default=None,
        help="Largest context (prompt+generated) for KV sizing (default: input+output).",
    )
    # ---- Precision ----
    parser.add_argument(
        "--weight-dtype",
        type=str,
        default=None,
        help="Resident weight precision (bf16 | fp8 | ...). Default: bf16.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default=None,
        help="KV-cache precision (bf16 | fp8 | int8 | ...). Default: bf16.",
    )
    # ---- Serving features ----
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=None,
        help="Chunked-prefill chunk size in tokens (0 disables).",
    )
    parser.add_argument(
        "--speculative-num-tokens",
        type=int,
        default=None,
        help="Draft tokens proposed per speculative verify step (0 disables).",
    )
    parser.add_argument(
        "--speculative-acceptance-rate",
        type=float,
        default=None,
        help="Expected per-token acceptance rate for speculative decoding [0,1].",
    )
    # ---- Capacity ----
    parser.add_argument(
        "--hbm-capacity-gb",
        type=float,
        default=None,
        help="Per-GPU HBM capacity (GB). When set, reports fit + max concurrency.",
    )
    # ---- Feature B: custom collective ops ----
    coll = parser.add_argument_group("inference collectives (feature B)")
    coll.add_argument(
        "--comm-model",
        type=str,
        default=None,
        choices=["explicit", "builtin"],
        help="Communication model: 'explicit' (knob-driven breakdown, default) "
        "or 'builtin' (folded into layer time, no breakdown).",
    )
    coll.add_argument(
        "--tp-allreduce-algo",
        type=str,
        default=None,
        choices=["auto", "ring", "one_shot", "two_shot", "hierarchical"],
        help="Force the TP AllReduce algorithm (default: auto = fastest).",
    )
    coll.add_argument(
        "--ep-a2a-algo",
        type=str,
        default=None,
        choices=["auto", "direct", "single_shot", "hierarchical"],
        help="Force the EP AllToAll algorithm (default: auto = fastest).",
    )
    coll.add_argument(
        "--prefill-comm-overlap",
        type=float,
        default=None,
        help="Fraction of prefill comm hidden behind compute [0,1] (default 0).",
    )
    coll.add_argument(
        "--decode-comm-overlap",
        type=float,
        default=None,
        help="Fraction of decode comm hidden behind compute [0,1] (default 0).",
    )
    coll.add_argument(
        "--tp-allreduce-efficiency",
        type=float,
        default=None,
        help="TP AllReduce time multiplier (<1 = fused-op speedup, default 1.0).",
    )
    coll.add_argument(
        "--ep-a2a-efficiency",
        type=float,
        default=None,
        help="EP AllToAll time multiplier (<1 = fused/overlapped speedup, default 1.0).",
    )
    # ---- Feature A: prefill/decode disaggregation ----
    dis = parser.add_argument_group("inference disaggregation (feature A)")
    dis.add_argument(
        "--disaggregate",
        action="store_true",
        help="Enable prefill/decode disaggregation (separate worker pools).",
    )
    dis.add_argument("--prefill-tp", type=int, default=None, help="Prefill-pool tensor parallelism.")
    dis.add_argument("--prefill-pp", type=int, default=None, help="Prefill-pool pipeline parallelism.")
    dis.add_argument("--prefill-ep", type=int, default=None, help="Prefill-pool expert parallelism.")
    dis.add_argument("--decode-tp", type=int, default=None, help="Decode-pool tensor parallelism.")
    dis.add_argument("--decode-pp", type=int, default=None, help="Decode-pool pipeline parallelism.")
    dis.add_argument("--decode-ep", type=int, default=None, help="Decode-pool expert parallelism.")
    dis.add_argument(
        "--prefill-replicas", type=int, default=None, help="Number of prefill-pool replicas."
    )
    dis.add_argument(
        "--decode-replicas", type=int, default=None, help="Number of decode-pool replicas."
    )
    dis.add_argument(
        "--kv-transfer-bw-gbps",
        type=float,
        default=None,
        help="KV-cache transfer bandwidth GB/s (default: inter-node pod BW).",
    )
    dis.add_argument(
        "--kv-transfer-latency-us",
        type=float,
        default=None,
        help="Fixed KV-cache transfer latency overhead (us).",
    )
    # ---- Serving / continuous-batching dynamics ----
    serv = parser.add_argument_group("inference serving dynamics")
    serv.add_argument(
        "--serving-model",
        type=str,
        default=None,
        choices=["continuous", "static"],
        help=(
            "Decode latency model: 'continuous' (continuous batching with mixed "
            "prefill+decode steps → models TPOT pollution; default) or 'static' "
            "(idealized pure-decode batch; prefill charged once as TTFT)."
        ),
    )
    serv.add_argument(
        "--decode-step-overhead-us",
        type=float,
        default=None,
        help="Fixed per-decode-step host/launch overhead (us). CUDA graphs reduce this. Default 0.",
    )
    serv.add_argument(
        "--mixed-batch-penalty",
        type=float,
        default=None,
        help="Extra cost fraction for mixed prefill+decode steps (PIECEWISE vs FULL CUDA graph). Default 0.",
    )
    # Internal: marks this process as the spawned GPU benchmark worker.
    parser.add_argument(
        "--inference-bench-worker",
        action="store_true",
        default=False,
        help=_argparse.SUPPRESS,
    )


def register_subcommand(subparsers):
    """
    Register the 'projection' subcommand to the main CLI parser.

    Examples::

        # Memory projection (analytical, default)
        primus projection memory --config exp.yaml

        # Memory projection (benchmark-anchored, OOM-accurate)
        primus projection memory --config exp.yaml --memory-mode benchmark \\
            --benchmark-gpus 8 --target-nodes 32

        # Performance projection (single-node benchmarking only)
        primus projection performance --config exp.yaml

        # Performance projection with multinode scaling to 4 nodes
        primus projection performance --config exp.yaml --target-nodes 4

        # One bench, both projections (perf + OOM-accurate memory)
        primus projection both --config exp.yaml --benchmark-gpus 8 \\
            --target-nodes 32

    Args:
        subparsers: argparse subparsers object from main.py

    Returns:
        parser: The parser for this subcommand
    """

    parser = subparsers.add_parser(
        "projection",
        help="Pre-training performance / memory projection tool",
        description="Primus projection entry point.",
    )
    suite_parsers = parser.add_subparsers(dest="suite", required=True)
    from primus.core.launcher.parser import add_pretrain_parser

    # ---------- memory ----------
    memory = suite_parsers.add_parser("memory", help="Memory projection (per-GPU memory analysis).")
    add_pretrain_parser(memory)
    _add_memory_args(memory)
    _add_topology_args(memory)
    _add_save_benchmark_arg(memory)
    _add_load_benchmark_arg(memory, include_compute_baseline_alias=True)

    # ---------- performance ----------
    performance = suite_parsers.add_parser(
        "performance", help="Performance projection with optional multinode scaling."
    )
    add_pretrain_parser(performance)
    _add_topology_args(performance)
    _add_performance_args(performance)
    _add_save_benchmark_arg(performance)
    # ``--load-benchmark`` skips the bench entirely and runs the perf
    # projection from a saved artifact (timing + reduction info come from
    # the artifact's metadata).  Memory's deprecated ``--compute-baseline``
    # alias is NOT registered here, since perf has its own (bg=1 hybrid)
    # ``--compute-baseline`` flag with different semantics.
    _add_load_benchmark_arg(performance, include_compute_baseline_alias=False)
    _add_perf_compute_baseline_arg(performance)

    # ---------- both ----------
    both = suite_parsers.add_parser(
        "both",
        help=(
            "Run a single bench and produce both performance and memory "
            "projections from it. Recommended for cluster-sizing workflows."
        ),
    )
    add_pretrain_parser(both)
    _add_topology_args(both)
    _add_performance_args(both)
    # `both` always runs perf-bench + benchmark-anchored memory projection;
    # --memory-mode is meaningless here, so we only expose the safety
    # margin from the memory side.
    _add_memory_safety_margin_arg(both)
    _add_save_benchmark_arg(both)
    # `both` registers --compute-baseline for the perf bg=1 baseline only;
    # it should NOT be a memory-load alias here because the orchestrator
    # always runs a fresh bench.
    _add_perf_compute_baseline_arg(both)

    # ---------- inference ----------
    inference = suite_parsers.add_parser(
        "inference",
        help="Inference / serving projection (TTFT, ITL, throughput, KV cache).",
    )
    add_pretrain_parser(inference)
    _add_topology_args(inference)
    # Reuse the perf knobs for the simulation backend selection
    # (--gpu-arch, --gpu-clock-mhz, --gemm-backend, etc.) plus --profiling-mode.
    _add_performance_args(inference)
    _add_save_benchmark_arg(inference)  # provides --save-profiling for the bench worker
    _add_load_benchmark_arg(inference, include_compute_baseline_alias=False)  # --load-benchmark reuse
    _add_inference_args(inference)
    # Inference defaults to pure simulation (no GPU). Opt into real-GPU layer
    # timing with ``--profiling-mode benchmark``.
    inference.set_defaults(profiling_mode="simulate")

    parser.set_defaults(func=run)
    return parser
