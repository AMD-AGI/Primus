###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import importlib
import pkgutil
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Set

SUBCOMMAND_PACKAGE = "primus.cli.subcommands"


def _ensure_project_root_on_path() -> None:
    """
    Allow running `python primus/cli/main.py` from the repo root without
    requiring an installed package.
    """
    if __package__:
        return
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _iter_subcommand_modules() -> Iterable[str]:
    """
    Discover every module inside `primus.cli.subcommands` (excluding those that
    start with `_`) and yield its full import path.
    """

    package = importlib.import_module(SUBCOMMAND_PACKAGE)
    prefix = package.__name__ + "."
    module_paths = []
    for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, prefix):
        leaf = module_name.split(".")[-1]
        if leaf.startswith("_"):
            continue
        if is_pkg:
            # Subpackages can contain nested commands; include their modules.
            # walk_packages already recurses, so we just skip the placeholder.
            continue
        module_paths.append(module_name)
    yield from sorted(module_paths, key=lambda name: name.split(".")[-1])


def _discover_subcommands() -> Dict[str, str]:
    """
    Return a mapping of CLI subcommand name -> module path.
    """
    commands: Dict[str, str] = {}
    for module_path in _iter_subcommand_modules():
        command = module_path.split(".")[-1]
        commands[command] = module_path
    return commands


def _register_subcommand(subparsers: argparse._SubParsersAction, module_path: str) -> None:
    """
    Import a single subcommand module and invoke its register hook.
    """
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to import CLI subcommand module " f"'{module_path}'. Ensure the module is importable."
        ) from exc

    register: Callable[[argparse._SubParsersAction], argparse.ArgumentParser] = getattr(
        module, "register_subcommand", None
    )
    if register is None:
        raise AttributeError(
            f"Module '{module_path}' must expose register_subcommand(subparsers). "
            "Please add a register_subcommand() function."
        )
    parser = register(subparsers)
    if parser is None:
        raise RuntimeError(f"register_subcommand() in '{module_path}' must return the parser it configured.")
    if not hasattr(parser, "get_default") or parser.get_default("func") is None:
        raise RuntimeError(
            f"Subcommand registered by '{module_path}' must call parser.set_defaults(func=...)."
        )


def _load_subcommands(subparsers: argparse._SubParsersAction, module_paths: Iterable[str]) -> None:
    """
    Dynamically import each discovered module and invoke its
    `register_subcommand(subparsers)` hook.
    """

    for module_path in module_paths:
        _register_subcommand(subparsers, module_path)


def _extract_command(argv: Iterable[str], available: Set[str]) -> Optional[str]:
    """
    Best-effort extraction of the subcommand from argv.
    """
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


def main():
    """
    Primus Unified CLI Entry

    Currently supported:
    - train: Launch Megatron / TorchTitan / Jax training.

    Reserved for future expansion:
    - benchmark: Run benchmarking tools for performance evaluation.
    - preflight: Environment and configuration checks.
      ...
    """
    _ensure_project_root_on_path()
    parser = argparse.ArgumentParser(
        prog="primus",
        description="Primus Unified CLI for Training & Utilities",
        epilog="Use `primus <command> --help` for command-specific options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose error output (stack traces).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    available_subcommands = _discover_subcommands()
    command = _extract_command(sys.argv[1:], set(available_subcommands.keys()))
    if command and command not in available_subcommands:
        print(f"[Primus] Unknown command '{command}'.", file=sys.stderr)
        print(
            f"[Primus] Available commands: {', '.join(sorted(available_subcommands))}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if command is None:
        _load_subcommands(subparsers, available_subcommands.values())
    else:
        _register_subcommand(subparsers, available_subcommands[command])

    args, unknown_args = parser.parse_known_args()

    if hasattr(args, "func"):
        try:
            extra_args_allowed = {"train", "projection", "preflight"}
            if unknown_args and args.command not in extra_args_allowed:
                print(
                    f"[Primus] Unknown arguments for '{args.command}': {unknown_args}",
                    file=sys.stderr,
                )
                print(
                    f"[Primus] Run `primus {args.command} --help` for valid options.",
                    file=sys.stderr,
                )
                raise SystemExit(2)
            args.func(args, unknown_args)
        except SystemExit:
            raise
        except Exception:
            # Torchrun/elastic can sometimes suppress worker tracebacks.
            # Print here so users can see the real root cause.
            if getattr(args, "debug", False):
                traceback.print_exc()
            else:
                print(f"[Primus] Error: {traceback.format_exc().splitlines()[-1]}", file=sys.stderr)
                print("[Primus] Re-run with --debug for full stack trace.", file=sys.stderr)
            raise SystemExit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--":
        sys.argv.pop(1)
    main()
