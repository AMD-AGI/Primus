###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
UTBP (Unified Training Benchmark & Preflight) CLI subcommand.

This module is auto-discovered by `primus.cli.main` and must expose:
- register_subcommand(subparsers) -> argparse.ArgumentParser
- run(args, extra_args) -> None
"""

from __future__ import annotations

from typing import Any, List


def run(args: Any, extra_args: List[str]) -> None:
    """
    Entry point for the 'utbp' subcommand.
    """
    if extra_args:
        # Keep behavior consistent with other subcommands: don't fail on unknown args.
        print(f"[Primus:UTBP] Ignoring extra CLI args: {extra_args}")

    if args.suite == "validate":
        # NOTE: UTBP implementation may live outside the core package in some
        # environments; keep these imports lazy and silence static analyzers.
        from primus.utbp.context import ValidationContext  # type: ignore[import-not-found]
        from primus.utbp.executor import run_validation  # type: ignore[import-not-found]
        from primus.utbp.result import summarize_results, write_results  # type: ignore[import-not-found]

        ctx = ValidationContext.from_env(output_dir=args.output_dir)
        results = run_validation(ctx, scope=args.scope)
        write_results(ctx, results)
        raise SystemExit(summarize_results(results))

    raise NotImplementedError(f"Unsupported utbp suite: {args.suite}")


def register_subcommand(subparsers):
    """
    Register the 'utbp' subcommand to the main CLI parser.

    Usage:
        primus utbp validate <cluster|node|container> [--output-dir DIR]
    """
    parser = subparsers.add_parser(
        "utbp",
        help="Unified Training Benchmark & Preflight (UTBP)",
        description="Primus Unified Training Benchmark & Preflight (UTBP).",
    )
    suite_parsers = parser.add_subparsers(dest="suite", required=True)

    # ---------- validate ----------
    validate = suite_parsers.add_parser("validate", help="Validate cluster/node/container readiness.")
    validate.add_argument(
        "scope",
        choices=["cluster", "node", "container"],
        help="Validation scope",
    )
    validate.add_argument(
        "--output-dir",
        default="utbp-artifacts",
        help="Directory to store UTBP results and artifacts",
    )

    parser.set_defaults(func=run)
    return parser
