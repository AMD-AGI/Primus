###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
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
    Register preflight arguments to the given CLI parser.
    """
    parser.add_argument(
        "--dump-path",
        type=str,
        default="output/preflight",
        help="Directory to store preflight reports and intermediate artifacts "
        "(default: output/preflight).",
    )
    parser.add_argument(
        "--report-file-name",
        type=str,
        default="preflight_report",
        help="Base name for the generated report files (default: preflight_report).",
    )
    parser.add_argument(
        "--disable-pdf",
        dest="save_pdf",
        action="store_false",
        help="Disable generation of the PDF report (only Markdown will be produced).",
    )
    parser.add_argument(
        "--disable-plot",
        dest="plot",
        action="store_false",
        help="Disable generation of plots in the report.",
    )
    return parser
