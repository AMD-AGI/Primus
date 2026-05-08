###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``pilot observe`` — wraps ``pilot.tools.observe`` (snapshot / watch / compare_loss)."""

from __future__ import annotations

from pilot.cli.subcommands._base import register_for


def register_subcommand(subparsers):
    return register_for(
        subparsers,
        name="observe",
        tool_module="observe",
        help="Snapshot or watch a training run; CORRECTNESS gate.",
    )
