###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import logging
from typing import List, Optional

from primus.cli.base import CommandBase

logger = logging.getLogger(__name__)


class BenchmarkCommand(CommandBase):
    """Command for running performance benchmarks.

    Provides microbenchmarks for various GPU operations including:
    - GEMM (General Matrix Multiply) operations
    - Attention mechanisms
    - Communication primitives (RCCL)
    """

    @classmethod
    def name(cls) -> str:
        return "benchmark"

    @classmethod
    def help(cls) -> str:
        return "Run performance benchmarks (GEMM / Attention / RCCL)"

    @classmethod
    def description(cls) -> str:
        return """
Performance benchmarking suite for GPU operations.

Supported benchmark suites:
  gemm          - Basic GEMM microbenchmarks
  gemm-dense    - Dense GEMM benchmarks
  gemm-deepseek - DeepSeek-specific GEMM patterns

Examples:
  # Run GEMM benchmark
  primus benchmark gemm --batch-size 32 --seq-len 2048

  # Run dense GEMM benchmark
  primus benchmark gemm-dense --warmup 10 --iterations 100

  # Run DeepSeek GEMM benchmark
  primus benchmark gemm-deepseek --model-size 7B
        """

    @classmethod
    def register_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Register benchmark-specific arguments.

        Supported suites:
            - gemm: Basic GEMM microbenchmarks
            - gemm-dense: Dense GEMM benchmarks
            - gemm-deepseek: DeepSeek GEMM benchmarks
        """
        parser.description = cls.description()
        suite_parsers = parser.add_subparsers(dest="suite", required=True)

        # ---------- GEMM ----------
        gemm = suite_parsers.add_parser(
            "gemm",
            help="GEMM microbenchmarks",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        from primus.tools.benchmark import gemm_bench

        gemm_bench.add_gemm_parser(gemm)

        # ---------- Dense GEMM ----------
        gemm_dense = suite_parsers.add_parser(
            "gemm-dense",
            help="Dense GEMM benchmarks",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        gemm_bench.add_gemm_parser(gemm_dense)

        # ---------- DeepSeek GEMM ----------
        gemm_deepseek = suite_parsers.add_parser(
            "gemm-deepseek",
            help="DeepSeek GEMM benchmarks",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        gemm_bench.add_gemm_parser(gemm_deepseek)

    @classmethod
    def validate_args(cls, args: argparse.Namespace, unknown_args: Optional[List[str]] = None) -> bool:
        """Validate benchmark arguments.

        Args:
            args: Parsed arguments.
            unknown_args: Unknown arguments.

        Returns:
            True if validation passes, False otherwise.
        """
        if not hasattr(args, "suite"):
            logger.error("No benchmark suite specified")
            return False

        valid_suites = ["gemm", "gemm-dense", "gemm-deepseek"]
        if args.suite not in valid_suites:
            logger.error(f"Invalid benchmark suite: {args.suite}")
            logger.error(f"Valid suites: {', '.join(valid_suites)}")
            return False

        return True

    @classmethod
    def run(cls, args: argparse.Namespace, unknown_args: Optional[List[str]] = None) -> None:
        """Execute the benchmark command.

        Args:
            args: Parsed arguments.
            unknown_args: Unknown arguments (typically empty for benchmarks).
        """
        suite = args.suite
        logger.info(f"Starting benchmark suite: {suite}")
        logger.debug(f"Benchmark arguments: {args}")

        # Initialize distributed environment
        from primus.tools.utils import finalize_distributed, init_distributed

        logger.debug("Initializing distributed environment")
        init_distributed()

        try:
            if suite == "gemm":
                logger.info("Running GEMM microbenchmarks")
                from primus.tools.benchmark.gemm_bench import run_gemm_benchmark

                run_gemm_benchmark(args)
            elif suite == "gemm-dense":
                logger.info("Running dense GEMM benchmarks")
                from primus.tools.benchmark.dense_gemm_bench import run_gemm_benchmark

                run_gemm_benchmark(args)
            elif suite == "gemm-deepseek":
                logger.info("Running DeepSeek GEMM benchmarks")
                from primus.tools.benchmark.deepseek_dense_gemm_bench import (
                    run_gemm_benchmark,
                )

                run_gemm_benchmark(args)
            else:
                # This should not happen due to validation
                raise ValueError(f"Unsupported benchmark suite: {suite}")

            logger.info(f"Benchmark suite '{suite}' completed successfully")

        finally:
            logger.debug("Finalizing distributed environment")
            finalize_distributed()
