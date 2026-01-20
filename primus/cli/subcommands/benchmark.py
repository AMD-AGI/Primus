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
    elif suite == "attention":
        from primus.tools.benchmark.attention_bench import run_attention_benchmark

        run_attention_benchmark(args)
    elif suite == "gemm-dense":
        from primus.tools.benchmark.dense_gemm_bench import run_gemm_benchmark

        run_gemm_benchmark(args)
    elif suite == "gemm-deepseek":
        from primus.tools.benchmark.deepseek_dense_gemm_bench import run_gemm_benchmark

        run_gemm_benchmark(args)
    elif suite == "strided-allgather":
        from primus.tools.benchmark.strided_allgather_bench import (
            run_strided_allgather_benchmark,
        )

        run_strided_allgather_benchmark(args)
    elif suite == "rccl":
        from primus.tools.benchmark.rccl_bench import run_rccl_benchmark

        run_rccl_benchmark(args)
    else:
        raise ValueError(f"Unknown benchmark suite: {suite}")

    finalize_distributed()


def register_subcommand(subparsers):
    """
    primus-cli benchmark <suite> [suite-specific-args]
    suites: gemm | attention | gemm-dense | gemm-deepseek | strided-allgather | rccl
    """
    parser = subparsers.add_parser("benchmark", help="Run performance benchmarks (GEMM / Attention / RCCL).")
    suite_parsers = parser.add_subparsers(dest="suite", required=True)

    # ---------- GEMM ----------
    gemm = suite_parsers.add_parser("gemm", help="GEMM microbench.")
    from primus.tools.benchmark.gemm_bench_args import add_gemm_parser

    add_gemm_parser(gemm)

    # ---------- ATTENTION ----------
    attention = suite_parsers.add_parser("attention", help="Attention microbench.")
    from primus.tools.benchmark.attention_bench_args import add_attention_parser

    add_attention_parser(attention)

    # ---------- DENSE-GEMM ----------
    dense_gemm = suite_parsers.add_parser("gemm-dense", help="GEMM-DENSE microbench.")
    from primus.tools.benchmark.dense_gemm_bench_args import (
        add_gemm_parser as add_dense_gemm_parser,
    )

    add_dense_gemm_parser(dense_gemm)

    # ---------- DEEPSEEK-GEMM ----------
    deepseek_gemm = suite_parsers.add_parser("gemm-deepseek", help="DEEPSEEK-GEMM microbench.")
    from primus.tools.benchmark.deepseek_dense_gemm_bench_args import (
        add_gemm_parser as add_deepseek_gemm_parser,
    )

    add_deepseek_gemm_parser(deepseek_gemm)

    # ---------- INTER-NODE-ALLGATHER-BY-LOCAL-RANK ----------
    strided_allgather_parser = suite_parsers.add_parser(
        "strided-allgather", help="Strided Allgather microbench."
    )
    from primus.tools.benchmark.strided_allgather_bench_args import add_arguments

    add_arguments(strided_allgather_parser)

    # ---------- RCCL ----------
    rccl = suite_parsers.add_parser("rccl", help="RCCL microbench.")
    from primus.tools.benchmark.rccl_bench_args import add_rccl_parser

    add_rccl_parser(rccl)

    parser.set_defaults(func=run)

    return parser
