###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


def run(args, extra_args):
    """
    Execute the benchmark command.
    This can internally call Megatron / TorchTitan hooks, or profile.py scripts.
    """

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
        from primus.tools.benchmark.deepseek_dense_gemm_bench import run_gemm_benchmark

        run_gemm_benchmark(args)

    finalize_distributed()


def register_subcommand(subparsers):
    """
    primus-cli benchmark <suite> [suite-specific-args]
    suites: gemm | attention | rccl
    """
    parser = subparsers.add_parser("benchmark", help="Run performance benchmarks (GEMM / Attention / RCCL).")
    suite_parsers = parser.add_subparsers(dest="suite", required=True)

    # ---------- GEMM ----------
    gemm = suite_parsers.add_parser("gemm", help="GEMM microbench.")
    from primus.tools.benchmark import gemm_bench

    gemm_bench.add_gemm_parser(gemm)

    # ---------- DENSE-GEMM ----------
    dense_gemm = suite_parsers.add_parser("gemm-dense", help="GEMM-DENSE microbench.")
    from primus.tools.benchmark import dense_gemm_bench

    dense_gemm_bench.add_gemm_parser(dense_gemm)

    # ---------- DEEPSEEK-GEMM ----------
    deepseek_gemm = suite_parsers.add_parser("gemm-deepseek", help="DEEPSEEK-GEMM microbench.")
    from primus.tools.benchmark import deepseek_dense_gemm_bench

    deepseek_dense_gemm_bench.add_gemm_parser(deepseek_gemm)

    parser.set_defaults(func=run)

    return parser
