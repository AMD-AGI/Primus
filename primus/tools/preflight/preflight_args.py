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


def add_preflight_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add common reporting arguments shared by preflight and preflight check.

    These flags control output directory, report file name, and PDF generation.
    """
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


def add_preflight_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register arguments for the top-level `primus-cli preflight` command.

    Usage:
        primus-cli preflight [--dump-path ...] [--report-file-name ...]
    """
    add_preflight_common_args(parser)
    return parser


def add_preflight_check_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register arguments for `primus-cli preflight check`.

    Usage:
        primus-cli preflight check --gpu
        primus-cli preflight check --network
        primus-cli preflight check --gpu --network
    """
    parser.add_argument("--gpu", action="store_true", help="Run GPU checks")
    parser.add_argument("--network", action="store_true", help="Run network checks")
    add_preflight_common_args(parser)
    return parser
