###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``pilot tune_single`` — wraps ``pilot.tools.tune_single``.

Subcommands: diagnose, replan, settle, run (single-node end-to-end loop).
"""

from __future__ import annotations

from pilot.cli.subcommands._base import register_for


def register_subcommand(subparsers):
    return register_for(
        subparsers,
        name="tune_single",
        tool_module="tune_single",
        help="Single-node end-to-end tuning runner (SMOKE → BASELINE → loop).",
    )
