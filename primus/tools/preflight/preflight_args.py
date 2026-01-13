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
        primus-cli preflight                          # Run all checks (GPU + Network)
        primus-cli preflight --check-gpu              # GPU only
        primus-cli preflight --check-network          # Network only
        primus-cli preflight --check-gpu --check-network  # Both
    """
    # Check selection flags
    parser.add_argument("--check-gpu", action="store_true", help="Run GPU checks only")
    parser.add_argument("--check-network", action="store_true", help="Run network checks only")

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
