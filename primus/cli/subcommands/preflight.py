###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Preflight CLI subcommand.

This subcommand is a thin wrapper around
`primus.tools.preflight.preflight_perf_test.run_preflight`.

Example:
    primus-cli preflight --dump-path output/preflight --report-file-name my_report
"""

from __future__ import annotations

from typing import Any, List


def run(args: Any, extra_args: List[str]) -> None:
    """
    Entry point for the 'preflight' subcommand.

    The subcommand parses arguments via the main CLI and then calls
    `run_preflight`. Any extra_args (unknown CLI tokens) are currently ignored.
    """

    if extra_args:
        # For now we ignore unknown args; could be extended later if needed.
        print(f"[Primus:Preflight] Ignoring extra CLI args: {extra_args}")

    from primus.tools.preflight.preflight_perf_test import run_preflight

    run_preflight(args)


def register_subcommand(subparsers):
    """
    Register the 'preflight' subcommand to the main CLI parser.

    Example:
        primus preflight --dump-path output/preflight --report-file-name preflight_report

    Args:
        subparsers: argparse subparsers object from main.py
    """

    parser = subparsers.add_parser(
        "preflight",
        help="Run cluster preflight checks (GPU compute & interconnect sanity).",
        description=(
            "Run Primus preflight diagnostics (compute, intra-node and inter-node communication) "
            "via primus.tools.preflight.preflight_perf_test.run_preflight."
        ),
    )
    from primus.tools.preflight.preflight_args import add_preflight_parser

    from primus.tools.preflight.preflight_args import add_preflight_parser

    # Reuse the shared preflight argument builder
    add_preflight_parser(parser)

    parser.set_defaults(func=run)

    return parser
