###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pilot Unified CLI entry point.

Dispatches to the per-tool CLIs under ``pilot/tools/*``. The two invocation
forms are equivalent:

    python -m pilot           <tool> <subcommand> [args...]
    python -m pilot.tools.<tool>      <subcommand> [args...]

The unified entry exists for discoverability (`python -m pilot --help` lists
every tool); the per-module entry is kept for backward compatibility and so
Cursor / MCP wrappers that already wire to ``pilot.tools.<x>`` keep working.

Layout mirrors ``primus/cli/main.py`` (dynamic subcommand discovery via
``pkgutil.walk_packages``). Each submodule under
``pilot.cli.subcommands.*`` must expose ``register_subcommand(subparsers)``
returning a configured ``argparse.ArgumentParser`` with
``parser.set_defaults(func=run)``.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Set


SUBCOMMAND_PACKAGE = "pilot.cli.subcommands"


def _ensure_project_root_on_path() -> None:
    """Allow ``python pilot/cli/main.py`` (no install) from the repo root."""
    if __package__:
        return
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _iter_subcommand_modules() -> Iterable[str]:
    """Yield every importable module path inside ``SUBCOMMAND_PACKAGE``."""
    package = importlib.import_module(SUBCOMMAND_PACKAGE)
    prefix = package.__name__ + "."
    module_paths: list[str] = []
    for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, prefix):
        leaf = module_name.split(".")[-1]
        if leaf.startswith("_"):
            continue
        if is_pkg:
            continue
        module_paths.append(module_name)
    yield from sorted(module_paths, key=lambda name: name.split(".")[-1])


def _discover_subcommands() -> Dict[str, str]:
    """Return mapping of CLI subcommand name -> fully-qualified module path."""
    commands: Dict[str, str] = {}
    for module_path in _iter_subcommand_modules():
        commands[module_path.split(".")[-1]] = module_path
    return commands


def _register_subcommand(subparsers: argparse._SubParsersAction, module_path: str) -> None:
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import CLI subcommand module '{module_path}'."
        ) from exc

    register: Callable[[argparse._SubParsersAction], argparse.ArgumentParser] = getattr(
        module, "register_subcommand", None
    )
    if register is None:
        raise AttributeError(
            f"Module '{module_path}' must expose register_subcommand(subparsers)."
        )
    parser = register(subparsers)
    if parser is None:
        raise RuntimeError(
            f"register_subcommand() in '{module_path}' must return the parser it configured."
        )
    if not hasattr(parser, "get_default") or parser.get_default("func") is None:
        raise RuntimeError(
            f"Subcommand registered by '{module_path}' must call parser.set_defaults(func=...)."
        )


def _load_subcommands(subparsers: argparse._SubParsersAction, module_paths: Iterable[str]) -> None:
    for module_path in module_paths:
        _register_subcommand(subparsers, module_path)


def _extract_command(argv: Iterable[str], available: Set[str]) -> Optional[str]:
    """Best-effort extraction of the subcommand from argv (mirrors primus CLI)."""
    argv_list = list(argv)
    for i, token in enumerate(argv_list):
        if token == "--":
            return argv_list[i + 1] if i + 1 < len(argv_list) else None
        if token.startswith("-"):
            continue
        if token in available:
            return token
        return token
    return None


def main() -> None:
    """
    Pilot Unified CLI Entry.

    Currently registered subcommands map 1:1 with ``pilot/tools/*.py``:

      preflight   profiler   submit    observe   constraint
      state       subagent   knowledge report    tune_single

    Examples:

        # equivalent to: python -m pilot.tools.preflight run --cluster-config cluster.yaml
        python -m pilot preflight run --cluster-config cluster.yaml

        # equivalent to: python -m pilot.tools.state checkpoint --input -
        python -m pilot state checkpoint --input -

    Each subcommand re-emits ``sys.argv`` to the underlying tool's own
    argparse, so all flags/help messages match ``python -m pilot.tools.<x>``
    exactly. There is no behavioural divergence between the two paths.
    """
    _ensure_project_root_on_path()

    parser = argparse.ArgumentParser(
        prog="pilot",
        description=(
            "Pilot Unified CLI — dispatches to pilot/tools/* "
            "(preflight / submit / observe / state / constraint / ...). "
            "Equivalent to `python -m pilot.tools.<tool>`."
        ),
        epilog="Use `pilot <tool> --help` for tool-specific options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full Python traceback on tool errors.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    available_subcommands = _discover_subcommands()
    command = _extract_command(sys.argv[1:], set(available_subcommands.keys()))

    if command and command not in available_subcommands:
        print(f"[Pilot] Unknown command '{command}'.", file=sys.stderr)
        print(
            f"[Pilot] Available commands: {', '.join(sorted(available_subcommands))}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # When no subcommand resolves yet (e.g. `pilot --help`) we register all of
    # them so the help text enumerates every tool. Otherwise we lazy-load only
    # the targeted module to avoid importing every tool's transitive deps.
    if command is None:
        _load_subcommands(subparsers, available_subcommands.values())
    else:
        _register_subcommand(subparsers, available_subcommands[command])

    args, unknown_args = parser.parse_known_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    try:
        args.func(args, unknown_args)
    except SystemExit:
        raise
    except Exception:
        if getattr(args, "debug", False):
            traceback.print_exc()
        exc_type, exc_value, exc_tb = sys.exc_info()
        err_msg = traceback.format_exc().splitlines()[-1]
        loc = ""
        if exc_tb is not None:
            frames = traceback.extract_tb(exc_tb)
            if frames:
                last = frames[-1]
                loc = f" ({last.filename}:{last.lineno})"
        print(f"[Pilot] Error: {err_msg}{loc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--":
        sys.argv.pop(1)
    main()
