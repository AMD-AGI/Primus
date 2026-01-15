###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Argument helpers for the Primus preflight tool.

This mirrors the pattern used by `primus.tools.benchmark.*_bench_args`.
"""

import argparse


def add_preflight_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register arguments for `primus-cli preflight`.

    Usage:
        primus-cli preflight                          # Show all info (Host + GPU + Network)
        primus-cli preflight --host                   # Host only
        primus-cli preflight --gpu                    # GPU only
        primus-cli preflight --network                # Network only
        primus-cli preflight --gpu --network          # GPU + Network
        primus-cli preflight --perf-test              # Run perf tests ONLY (skip info)
    """
    # Check selection flags
    # Keep --check-* as compatibility aliases.
    parser.add_argument(
        "--host",
        "--check-host",
        dest="check_host",
        action="store_true",
        help="Show host info (CPU, memory, PCIe)",
    )
    parser.add_argument(
        "--gpu",
        "--check-gpu",
        dest="check_gpu",
        action="store_true",
        help="Show GPU info",
    )
    parser.add_argument(
        "--network",
        "--check-network",
        dest="check_network",
        action="store_true",
        help="Show network info",
    )

    # Performance test mode (full GEMM, intra/inter node comm tests)
    parser.add_argument(
        "--perf-test",
        action="store_true",
        help="Run perf tests ONLY (GEMM, intra/inter node communication). "
        "This is slower and skips the host/gpu/network info report.",
    )

    # Performance test specific options (only used with --perf-test)
    parser.add_argument("--plot", action="store_true", help="Generate plots (only with --perf-test)")

    # Distributed init timeout (prevents hangs when network/rendezvous is misconfigured)
    parser.add_argument(
        "--dist-timeout-sec",
        type=int,
        default=120,
        help="Timeout (seconds) for torch.distributed process group init. "
        "If init times out, preflight will write the info report and exit with failure.",
    )

    # Report output options
    parser.add_argument(
        "--dump-path",
        type=str,
        default="output/preflight",
        help="Directory to store preflight reports (default: output/preflight).",
    )
    parser.add_argument(
        "--report-file-name",
        type=str,
        default="preflight_report",
        help="Base name for report files (default: preflight_report).",
    )
    parser.add_argument(
        "--disable-pdf",
        dest="save_pdf",
        action="store_false",
        help="Disable PDF report generation.",
    )
    return parser
