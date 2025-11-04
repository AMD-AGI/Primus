###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import sys

from primus.cli.registry import CommandRegistry


def main():
    """
    Primus Unified CLI Entry

    Currently supported:
    - train: Launch Megatron / TorchTitan / Jax training.
    - benchmark: Run benchmarking tools for performance evaluation.

    Reserved for future expansion:
    - preflight: Environment and configuration checks.
    """
    parser = argparse.ArgumentParser(prog="primus", description="Primus Unified CLI for Training & Utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Auto-discover and register all commands
    CommandRegistry.discover_commands()
    for command_cls in CommandRegistry.get_all_commands().values():
        command_parser = subparsers.add_parser(command_cls.name(), help=command_cls.help())
        command_cls.register_arguments(command_parser)
        command_parser.set_defaults(command_cls=command_cls)

    args, unknown_args = parser.parse_known_args()

    if hasattr(args, "command_cls"):
        args.command_cls.run(args, unknown_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--":
        sys.argv.pop(1)
    main()
