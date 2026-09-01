###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compatibility entry point for the existing Megatron data commands."""

import argparse

from primus.backends.megatron.data.cli import register_data_subcommands


def register_subcommand(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register ``primus data`` and delegate to the Megatron implementation."""
    parser = subparsers.add_parser(
        "data",
        help="Dataset preparation tools (Megatron diffusion)",
        description=(
            "Data preparation utilities for Megatron diffusion models.\n"
            "The public command names are retained for compatibility."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    data_subparsers = parser.add_subparsers(
        dest="data_command",
        required=True,
        help="Data preparation command",
    )

    register_data_subcommands(data_subparsers)

    # Required by the CLI dispatch contract. The nested subparser always
    # replaces this with the selected Megatron command handler.
    parser.set_defaults(func=lambda args, unknown: None)
    return parser
