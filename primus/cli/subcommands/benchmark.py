###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
from typing import List, Optional

from primus.cli.base import CommandBase


class BenchmarkCommand(CommandBase):
    """Command for running benchmarks."""

    @classmethod
    def name(cls) -> str:
        return "benchmark"

    @classmethod
    def help(cls) -> str:
        return "Run performance benchmarks (GEMM / Attention / RCCL)"

    @classmethod
    def register_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Register benchmark-specific arguments.

        Supported suites:
            - gemm: Basic GEMM microbenchmarks
            - gemm-dense: Dense GEMM benchmarks
            - gemm-deepseek: DeepSeek GEMM benchmarks
        """
        suite_parsers = parser.add_subparsers(dest="suite", required=True)

        # ---------- GEMM ----------
        gemm = suite_parsers.add_parser("gemm", help="GEMM microbench.")
        from primus.tools.benchmark import gemm_bench

        gemm_bench.add_gemm_parser(gemm)

        # ---------- Dense GEMM ----------
        gemm_dense = suite_parsers.add_parser("gemm-dense", help="Dense GEMM bench.")
        gemm_bench.add_gemm_parser(gemm_dense)

        # ---------- DeepSeek GEMM ----------
        gemm_deepseek = suite_parsers.add_parser("gemm-deepseek", help="DeepSeek GEMM bench.")
        gemm_bench.add_gemm_parser(gemm_deepseek)

    @classmethod
    def run(cls, args: argparse.Namespace, unknown_args: Optional[List[str]] = None) -> None:
        """Execute the benchmark command."""
        suite = args.suite
        print(f"[Primus:Benchmark] suite={suite} args={args}")

        from primus.tools.utils import finalize_distributed, init_distributed

        init_distributed()

        if suite == "gemm":
            from primus.tools.benchmark.gemm_bench import run_gemm_benchmark

            run_gemm_benchmark(args)
        elif suite == "gemm-dense":
            from primus.tools.benchmark.dense_gemm_bench import run_gemm_benchmark

            run_gemm_benchmark(args)
        elif suite == "gemm-deepseek":
            from primus.tools.benchmark.deepseek_dense_gemm_bench import (
                run_gemm_benchmark,
            )

            run_gemm_benchmark(args)

        finalize_distributed()
