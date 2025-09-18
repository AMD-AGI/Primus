###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import sys


def main():
    """
    Primus Unified CLI Entry

    Currently supported:
    - train: Launch Megatron / TorchTitan / Jax training.
    - benchmark: Run benchmarking tools for performance evaluation.

    Reserved for future expansion:
    - preflight: Environment and configuration checks.
      ...
    """
    parser = argparse.ArgumentParser(
        prog="primus-cli", description="Primus Unified CLI for Training & Utilities"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    from primus.cli import benchmark_cli, train_cli

    # Register train subcommand (only implemented one for now)
    train_cli.register_subcommand(subparsers)
    benchmark_cli.register_subcommand(subparsers)

    args, unknown_args = parser.parse_known_args()

    # Dispatch to the implemented subcommands
    if args.command == "train":
        train_cli.run(args, unknown_args)
    elif args.command == "benchmark":
        benchmark_cli.run(args, unknown_args)
    else:
        # Future subcommands can be dispatched here
        parser.print_help()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--":
        sys.argv.pop(1)
    main()
