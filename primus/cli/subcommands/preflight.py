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

    suite = getattr(args, "suite", None)

    # Lightweight checks (benchmark-like suites):
    #   primus-cli preflight gpu|network|all ...
    if suite in ("gpu", "network", "all"):
        from primus.tools.preflight.preflight_check import run_preflight_check
        from primus.tools.utils import finalize_distributed, init_distributed

        init_distributed()
        try:
            rc = run_preflight_check(args)
        finally:
            finalize_distributed()
        raise SystemExit(rc)

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
    # Legacy perf flags on the parent parser (keeps `primus-cli preflight --dump-path ...` working).
    from primus.tools.preflight.preflight_args import (
        add_preflight_all_check_parser,
        add_preflight_gpu_check_parser,
        add_preflight_network_check_parser,
        add_preflight_perf_parser,
    )

    add_preflight_perf_parser(parser)

    # Benchmark-like suites:
    suite_parsers = parser.add_subparsers(dest="suite", required=False)

    gpu = suite_parsers.add_parser("gpu", help="Run GPU preflight checks")
    add_preflight_gpu_check_parser(gpu)

    network = suite_parsers.add_parser("network", help="Run network preflight checks")
    add_preflight_network_check_parser(network)

    all_ = suite_parsers.add_parser("all", help="Run all preflight checks (GPU + network)")
    add_preflight_all_check_parser(all_)

    parser.set_defaults(func=run)

    return parser
