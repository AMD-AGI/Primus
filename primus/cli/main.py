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
from typing import Callable, Iterable

SUBCOMMAND_PACKAGE = "primus.cli.subcommands"


def _iter_subcommand_modules() -> Iterable[str]:
    """
    Discover every module inside `primus.cli.subcommands` (excluding those that
    start with `_`) and yield its full import path.
    """

    package = importlib.import_module(SUBCOMMAND_PACKAGE)
    prefix = package.__name__ + "."
    for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, prefix):
        leaf = module_name.split(".")[-1]
        if leaf.startswith("_"):
            continue
        if is_pkg:
            # Subpackages can contain nested commands; include their modules.
            # walk_packages already recurses, so we just skip the placeholder.
            continue
        yield module_name


def _load_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """
    Dynamically import each discovered module and invoke its
    `register_subcommand(subparsers)` hook.
    """

    for module_path in _iter_subcommand_modules():
        module = importlib.import_module(module_path)
        register: Callable[[argparse._SubParsersAction], argparse.ArgumentParser] = getattr(
            module, "register_subcommand", None
        )
        assert register is not None, f"Module '{module_path}' must expose register_subcommand()"
        parser = register(subparsers)
        assert (
            parser is not None
        ), f"register_subcommand() in '{module_path}' must return the parser it configured"
        assert hasattr(parser, "get_default") and parser.get_default("func") is not None, (
            f"Subcommand registered by '{module_path}' must call parser.set_defaults(func=...)"
        )


def main():
    """
    Primus Unified CLI Entry

    Currently supported:
    - train: Launch training workflows (e.g., Megatron / TorchTitan / MaxText).
    - benchmark: Run performance benchmarks.
    - preflight: Run environment and configuration checks.
    - projection: Performance projection utilities.
    """
    parser = argparse.ArgumentParser(prog="primus", description="Primus Unified CLI for Training & Utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _load_subcommands(subparsers)

    args, unknown_args = parser.parse_known_args()

    if hasattr(args, "func"):
        try:
            args.func(args, unknown_args)
        except SystemExit:
            raise
        except Exception:
            # Torchrun/elastic can sometimes suppress worker tracebacks.
            # Print here so users can see the real root cause.
            traceback.print_exc()
            raise
    else:
        parser.print_help()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--":
        sys.argv.pop(1)
    main()
