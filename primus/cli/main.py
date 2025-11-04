###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import importlib
import sys


def load_and_register_subcommands(subparsers):
    """
    Dynamically load and register subcommands from CLI modules.
    """
    cli_modules = ["train_cli", "benchmark_cli"]  # Add more CLI modules here as needed

    for module_name in cli_modules:
        module = importlib.import_module(f"primus.cli.{module_name}")
        if hasattr(module, "register_subcommand"):
            module.register_subcommand(subparsers)


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

    # Dynamically load and register subcommands
    load_and_register_subcommands(subparsers)

    args, unknown_args = parser.parse_known_args()

    if hasattr(args, "func"):
        args.func(args, unknown_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--":
        sys.argv.pop(1)
    main()
