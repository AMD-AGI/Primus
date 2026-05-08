###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``pilot state`` — wraps ``pilot.tools.state`` (checkpoint / resume / trim / handoff)."""

from __future__ import annotations

from pilot.cli.subcommands._base import register_for


def register_subcommand(subparsers):
    return register_for(
        subparsers,
        name="state",
        tool_module="state",
        help="TuningState persistence: checkpoint / resume / trim / handoff.",
    )
