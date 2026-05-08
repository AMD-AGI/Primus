###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``pilot profiler`` — wraps ``pilot.tools.profiler`` (single-node profiling)."""

from __future__ import annotations

from pilot.cli.subcommands._base import register_for


def register_subcommand(subparsers):
    return register_for(
        subparsers,
        name="profiler",
        tool_module="profiler",
        help="Single-node profiling estimates for the PROJECTION stage.",
    )
