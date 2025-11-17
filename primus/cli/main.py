###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import importlib
import pkgutil
import sys
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
        if register is None:
            continue
        parser = register(subparsers)
        if parser is None:
            continue
        if not hasattr(parser, "get_default") or parser.get_default("func") is None:
            raise RuntimeError(
                f"Subcommand registered by '{module_path}' must call parser.set_defaults(func=...)"
            )


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
    parser = argparse.ArgumentParser(prog="primus", description="Primus Unified CLI for Training & Utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _load_subcommands(subparsers)

    args, unknown_args = parser.parse_known_args()

    if hasattr(args, "func"):
        args.func(args, unknown_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--":
        sys.argv.pop(1)
    main()
