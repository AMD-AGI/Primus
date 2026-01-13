###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Preflight CLI subcommand.

Simplified interface:
    primus-cli preflight                              # Run all checks (GPU + Network)
    primus-cli preflight --check-gpu                  # Run GPU checks only
    primus-cli preflight --check-network              # Run network checks only
    primus-cli preflight --check-gpu --check-network  # Run both
"""

from __future__ import annotations

from typing import Any, List


def run(args: Any, extra_args: List[str]) -> None:
    """
    Entry point for the 'preflight' subcommand.
    """
    from primus.tools.preflight.preflight_check import run_preflight_check
    from primus.tools.utils import finalize_distributed, init_distributed

    if extra_args:
        print(f"[Primus:Preflight] Ignoring extra CLI args: {extra_args}")

    init_distributed()
    try:
        rc = run_preflight_check(args)
    finally:
        finalize_distributed()
    raise SystemExit(rc)


def register_subcommand(subparsers):
    """
    Register the 'preflight' subcommand.

    Usage:
        primus-cli preflight                              # Run all checks
        primus-cli preflight --check-gpu                  # GPU only
        primus-cli preflight --check-network              # Network only
        primus-cli preflight --check-gpu --check-network  # Both
    """
    from primus.tools.preflight.preflight_args import add_preflight_parser

    parser = subparsers.add_parser(
        "preflight",
        help="Run cluster preflight checks (GPU + Network by default).",
    )
    add_preflight_parser(parser)

    parser.set_defaults(func=run)
    return parser
