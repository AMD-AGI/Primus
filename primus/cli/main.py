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
        if register is None:
            raise AttributeError(f"Module '{module_path}' must expose register_subcommand()")
        parser = register(subparsers)
        if parser is None:
            raise RuntimeError(
                f"register_subcommand() in '{module_path}' must return the parser it configured"
            )
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
    print("[PRIMUS-CLI] main() entered", flush=True)
    parser = argparse.ArgumentParser(prog="primus", description="Primus Unified CLI for Training & Utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    print("[PRIMUS-CLI] loading subcommands...", flush=True)
    _load_subcommands(subparsers)
    print("[PRIMUS-CLI] subcommands loaded", flush=True)

    args, unknown_args = parser.parse_known_args()
    print(f"[PRIMUS-CLI] parsed command={getattr(args, 'command', None)}, unknown_args={unknown_args}", flush=True)

    if hasattr(args, "func"):
        func_name = getattr(args.func, "__name__", str(args.func))
        func_module = getattr(args.func, "__module__", "unknown")
        print(f"[PRIMUS-CLI] dispatching to {func_module}.{func_name}()", flush=True)
        try:
            args.func(args, unknown_args)
        except SystemExit:
            raise
        except Exception:
            # Torchrun/elastic can sometimes suppress worker tracebacks.
            # Print here so users can see the real root cause.
            traceback.print_exc()
            raise
        print(f"[PRIMUS-CLI] {func_module}.{func_name}() completed", flush=True)
    else:
        parser.print_help()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--":
        sys.argv.pop(1)
    main()
