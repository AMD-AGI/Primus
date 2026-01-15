###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Preflight CLI subcommand.

Usage:
    primus-cli preflight                              # Run ALL: info report + perf tests
    primus-cli preflight --host                       # Host info only
    primus-cli preflight --gpu                        # GPU info only
    primus-cli preflight --network                    # Network info only
    primus-cli preflight --perf-test                  # Perf tests only (skip info)
"""

from __future__ import annotations

from typing import Any, List


def run(args: Any, extra_args: List[str]) -> None:
    """
    Entry point for the 'preflight' subcommand.
    """
    from primus.tools.preflight.preflight_perf_test import run_preflight

    if extra_args:
        print(f"[Primus:Preflight] Ignoring extra CLI args: {extra_args}")

    rc = run_preflight(args)
    raise SystemExit(rc)


def register_subcommand(subparsers):
    """
    Register the 'preflight' subcommand.

    Usage:
        primus-cli preflight                              # Info + perf tests
        primus-cli preflight --host                       # Host info only
        primus-cli preflight --gpu                        # GPU info only
        primus-cli preflight --network                    # Network info only
        primus-cli preflight --perf-test                  # Perf only
    """
    from primus.tools.preflight.preflight_args import add_preflight_parser

    parser = subparsers.add_parser("preflight", help="Run cluster preflight (info + perf).")
    add_preflight_parser(parser)

    parser.set_defaults(func=run)
    return parser
