###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Entry point for the backend data commands under ``primus data``."""

import argparse

from primus.backends.automodel.data.cli import (
    register_data_subcommands as register_automodel_data_subcommands,
)
from primus.backends.megatron.data.cli import (
    register_data_subcommands as register_megatron_data_subcommands,
)


def register_subcommand(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register ``primus data`` with each backend's commands under it."""
    parser = subparsers.add_parser(
        "data",
        help="Dataset preparation tools (diffusion)",
        description=(
            "Data preparation utilities for diffusion models.\n"
            "The Megatron commands keep their existing names; automodel-cache\n"
            "builds the cache the AutoModel diffusion recipe reads."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    data_subparsers = parser.add_subparsers(
        dest="data_command",
        required=True,
        help="Data preparation command",
    )

    register_megatron_data_subcommands(data_subparsers)
    register_automodel_data_subcommands(data_subparsers)

    # Required by the CLI dispatch contract. The nested subparser always
    # replaces this with the selected command's handler.
    parser.set_defaults(func=lambda args, unknown: None)
    return parser
