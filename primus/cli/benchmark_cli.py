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


def load_and_register_benchmark_suites(suite_parsers):
    """
    Dynamically load and register benchmark suites.
    """
    import importlib

    benchmark_modules = [
        "gemm_bench",
        "dense_gemm_bench",
        "deepseek_dense_gemm_bench",
    ]

    for module_name in benchmark_modules:
        module = importlib.import_module(f"primus.tools.benchmark.{module_name}")
        if hasattr(module, "add_gemm_parser"):
            module.add_gemm_parser(
                suite_parsers.add_parser(module_name, help=f"{module_name.replace('_', '-')} microbench.")
            )


def register_subcommand(subparsers):
    """
    primus-cli benchmark <suite> [suite-specific-args]
    suites: gemm | attention | rccl
    """
    parser = subparsers.add_parser("benchmark", help="Run performance benchmarks (GEMM / Attention / RCCL).")
    suite_parsers = parser.add_subparsers(dest="suite", required=True)

    # Dynamically load and register benchmark suites
    load_and_register_benchmark_suites(suite_parsers)

    parser.set_defaults(func=run)

    return parser
