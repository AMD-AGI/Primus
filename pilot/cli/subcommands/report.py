###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``pilot report`` — wraps ``pilot.tools.report`` (build / show TuningReport)."""

from __future__ import annotations

from pilot.cli.subcommands._base import register_for


def register_subcommand(subparsers):
    return register_for(
        subparsers,
        name="report",
        tool_module="report",
        help="Aggregate stage artifacts into a TuningReport (yaml + markdown).",
    )
