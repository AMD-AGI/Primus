###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``pilot learn`` — wraps ``pilot.tools.learn`` (between-session learn loop)."""

from __future__ import annotations

from pilot.cli.subcommands._base import register_for


def register_subcommand(subparsers):
    return register_for(
        subparsers,
        name="learn",
        tool_module="learn",
        help="LEARN: derive cross-session findings + emit knowledge drafts.",
    )
