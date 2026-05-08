###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Helper shared by every ``pilot.cli.subcommands.<x>`` wrapper.

Each wrapper is a passthrough: it builds a one-line help entry for
``pilot --help`` and re-emits ``sys.argv`` to the targeted
``pilot.tools.<x>._cli()`` so that flags, exit codes, and stdout/stderr
behaviour match the legacy ``python -m pilot.tools.<x>`` path verbatim.

This file is imported only through ``register_for(...)`` from each
subcommand module; it is not a CLI subcommand itself (filename starts with
``_`` so ``main.py`` skips it during discovery).
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from typing import Any, Callable


def _load_tool_cli(tool_module: str) -> Callable[[], int]:
    """Import ``pilot.tools.<tool_module>`` lazily and return its ``_cli``."""
    mod = import_module(f"pilot.tools.{tool_module}")
    cli = getattr(mod, "_cli", None)
    if cli is None:
        raise RuntimeError(
            f"pilot.tools.{tool_module} is missing the required `_cli` entry"
            " function; the unified CLI cannot delegate to it."
        )
    return cli


def register_for(
    subparsers: argparse._SubParsersAction,
    *,
    name: str,
    tool_module: str,
    help: str,
    description: str | None = None,
) -> argparse.ArgumentParser:
    """Register a ``pilot <name>`` subcommand that re-dispatches to ``pilot.tools.<tool_module>``.

    Args:
        subparsers: parent parser produced by ``pilot.cli.main``.
        name: subcommand name as exposed under the ``pilot`` CLI.
        tool_module: leaf module under ``pilot.tools`` whose ``_cli()``
            handles the real argument parsing and dispatch.
        help: one-line summary shown by ``pilot --help``.
        description: optional longer description for ``pilot <name> --help``
            (rarely visible because ``-h`` is forwarded to the tool itself).
    """
    parser = subparsers.add_parser(
        name,
        help=help,
        description=description or help,
        # Suppress the default --help: we want -h/--help to fall through to
        # the underlying tool so users see exactly the same usage message
        # `python -m pilot.tools.<x> --help` would print.
        add_help=False,
    )

    def run(args: Any, extra_args: list[str]) -> None:
        # Forward unknown args verbatim to the tool. argparse-on-the-tool-side
        # then handles its own subcommand routing (`run`, `cancel`, ...).
        sys.argv = [f"pilot.tools.{tool_module}", *extra_args]
        cli = _load_tool_cli(tool_module)
        rc = cli()
        raise SystemExit(rc if isinstance(rc, int) else 0)

    parser.set_defaults(func=run)
    return parser
