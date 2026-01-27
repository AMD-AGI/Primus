###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
UTBP (Unified Training Benchmark & Preflight) CLI subcommand.

UTBP is a unified entrypoint for:
- Preflight: host/GPU/network/storage/container/env sanity checks
- Benchmark: micro (e.g., GEMM/RCCL/IO) + training-level metrics
- Debug/Diag: hang / slow node / comm diagnostics (extensible)

This module is auto-discovered by `primus.cli.main` and must expose:
- register_subcommand(subparsers) -> argparse.ArgumentParser
- run(args, extra_args) -> None
"""

from __future__ import annotations

from typing import Any, Dict, List


def _write_run_context(ctx, suite: str) -> None:
    """
    Persist a minimal environment fingerprint alongside UTBP artifacts.
    """
    import json
    import os
    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    git_sha = None
    try:
        git_sha = (
            subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True)
            .strip()
            .splitlines()[-1]
        )
    except Exception:
        git_sha = os.environ.get("GIT_SHA") or os.environ.get("PRIMUS_GIT_SHA")

    # Keep this compact and stable for CI consumers.
    env_allowlist = [
        "ROCM_VERSION",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG",
        "RCCL_DEBUG",
        "HSA_FORCE_FINE_GRAIN_PCIE",
        "NCCL_SOCKET_IFNAME",
        "RCCL_SOCKET_IFNAME",
        "SLURM_JOB_ID",
        "SLURM_NNODES",
        "SLURM_NODEID",
        "HOSTNAME",
        "CONTAINER_RUNTIME",
    ]
    env: Dict[str, str] = {k: v for k, v in ctx.env.items() if k in env_allowlist}

    payload = {
        "run_id": ctx.run_id,
        "suite": suite,
        "hostname": ctx.hostname,
        "runtime": ctx.runtime,
        "node_rank": ctx.node_rank,
        "world_size": ctx.world_size,
        "git_sha": git_sha,
        "env": env,
    }
    path = os.path.join(ctx.artifact_dir, "context.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def run(args: Any, extra_args: List[str]) -> None:
    """
    Entry point for the 'utbp' subcommand.
    """
    if extra_args:
        # Keep behavior consistent with other subcommands: don't fail on unknown args.
        print(f"[Primus:UTBP] Ignoring extra CLI args: {extra_args}")

    from primus.utbp.context import ValidationContext

    ctx = ValidationContext.from_env(output_dir=args.output_dir)
    _write_run_context(ctx, suite=args.suite)

    if args.suite == "validate":
        from primus.utbp.context import ValidationContext
        from primus.utbp.executor import run_validation
        from primus.utbp.result import summarize_results, write_results

        results = run_validation(ctx, scope=args.scope)
        write_results(ctx, results)
        raise SystemExit(summarize_results(results))

    if args.suite == "preflight":
        from primus.tools.preflight.preflight_perf_test import run_preflight

        rc = run_preflight(args)
        raise SystemExit(rc)

    if args.suite == "benchmark":
        # Mirror `primus.cli.subcommands.benchmark.run`, but keep UTBP context/outputs.
        from primus.tools.utils import finalize_distributed, init_distributed

        init_distributed()
        try:
            if args.bench_suite == "gemm":
                from primus.tools.benchmark.gemm_bench import run_gemm_benchmark

                run_gemm_benchmark(args)
            elif args.bench_suite == "attention":
                from primus.tools.benchmark.attention_bench import run_attention_benchmark

                run_attention_benchmark(args)
            elif args.bench_suite == "gemm-dense":
                from primus.tools.benchmark.dense_gemm_bench import run_gemm_benchmark

                run_gemm_benchmark(args)
            elif args.bench_suite == "gemm-deepseek":
                from primus.tools.benchmark.deepseek_dense_gemm_bench import run_gemm_benchmark

                run_gemm_benchmark(args)
            elif args.bench_suite == "strided-allgather":
                from primus.tools.benchmark.strided_allgather_bench import (
                    run_strided_allgather_benchmark,
                )

                run_strided_allgather_benchmark(args)
            elif args.bench_suite == "rccl":
                from primus.tools.benchmark.rccl_bench import run_rccl_benchmark

                run_rccl_benchmark(args)
            else:
                raise NotImplementedError(f"Unsupported benchmark suite: {args.bench_suite}")
        finally:
            finalize_distributed()

        return

    if args.suite == "diag":
        raise NotImplementedError(
            "UTBP diag suites are not implemented yet. "
            "Planned: hang/slow-node/comm diagnostics via a plugin-style registry."
        )

    raise NotImplementedError(f"Unsupported utbp suite: {args.suite}")


def register_subcommand(subparsers):
    """
    Register the 'utbp' subcommand to the main CLI parser.

    Usage:
        primus utbp validate <cluster|node|container> [--output-dir DIR]
        primus utbp preflight [--host|--gpu|--network|--perf-test] [--output-dir DIR]
        primus utbp benchmark <suite> [suite-args...] [--output-dir DIR]
    """
    parser = subparsers.add_parser(
        "utbp",
        help="Unified Training Benchmark & Preflight (UTBP)",
        description="Primus Unified Training Benchmark & Preflight (UTBP).",
    )
    parser.add_argument(
        "--output-dir",
        default="utbp-artifacts",
        help="Directory to store UTBP results and artifacts",
    )
    suite_parsers = parser.add_subparsers(dest="suite", required=True)

    # ---------- validate ----------
    validate = suite_parsers.add_parser("validate", help="Validate cluster/node/container readiness.")
    validate.add_argument(
        "scope",
        choices=["cluster", "node", "container"],
        help="Validation scope",
    )

    # ---------- preflight ----------
    preflight = suite_parsers.add_parser("preflight", help="Preflight: host/GPU/network/env sanity checks.")
    from primus.tools.preflight.preflight_args import add_preflight_parser

    add_preflight_parser(preflight)

    # ---------- benchmark ----------
    benchmark = suite_parsers.add_parser("benchmark", help="Benchmarks: GEMM/Attention/RCCL/etc.")
    bench_parsers = benchmark.add_subparsers(dest="bench_suite", required=True)

    gemm = bench_parsers.add_parser("gemm", help="GEMM microbench.")
    from primus.tools.benchmark.gemm_bench_args import add_gemm_parser

    add_gemm_parser(gemm)

    attention = bench_parsers.add_parser("attention", help="Attention microbench.")
    from primus.tools.benchmark.attention_bench_args import add_attention_parser

    add_attention_parser(attention)

    dense_gemm = bench_parsers.add_parser("gemm-dense", help="GEMM-DENSE microbench.")
    from primus.tools.benchmark.dense_gemm_bench_args import add_gemm_parser as add_dense_gemm_parser

    add_dense_gemm_parser(dense_gemm)

    deepseek_gemm = bench_parsers.add_parser("gemm-deepseek", help="DEEPSEEK-GEMM microbench.")
    from primus.tools.benchmark.deepseek_dense_gemm_bench_args import (
        add_gemm_parser as add_deepseek_gemm_parser,
    )

    add_deepseek_gemm_parser(deepseek_gemm)

    strided_allgather = bench_parsers.add_parser("strided-allgather", help="Strided allgather microbench.")
    from primus.tools.benchmark.strided_allgather_bench_args import add_arguments

    add_arguments(strided_allgather)

    rccl = bench_parsers.add_parser("rccl", help="RCCL microbench.")
    from primus.tools.benchmark.rccl_bench_args import add_rccl_parser

    add_rccl_parser(rccl)

    # ---------- diag (placeholder) ----------
    suite_parsers.add_parser(
        "diag",
        help="Debug/diagnostics (planned): hang/slow-node/comm troubleshooting.",
    )

    parser.set_defaults(func=run)
    return parser
