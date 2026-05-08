###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``pilot knowledge`` — wraps ``pilot.tools.knowledge`` (LEARN draft writer)."""

from __future__ import annotations

from pilot.cli.subcommands._base import register_for


def register_subcommand(subparsers):
    return register_for(
        subparsers,
        name="knowledge",
        tool_module="knowledge",
        help="LEARN: write best/failure drafts under state/knowledge_drafts/.",
    )
