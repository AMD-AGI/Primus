###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``pilot submit`` — wraps ``pilot.tools.submit`` (run / cancel / status)."""

from __future__ import annotations

from pilot.cli.subcommands._base import register_for


def register_subcommand(subparsers):
    return register_for(
        subparsers,
        name="submit",
        tool_module="submit",
        help="Launch / cancel / inspect Primus training jobs (cluster.yaml driven).",
    )
