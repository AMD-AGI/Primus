###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``pilot constraint`` — wraps ``pilot.tools.constraint``.

Subcommands: check, check_env, estimate_mem, diagnose_failure.
"""

from __future__ import annotations

from pilot.cli.subcommands._base import register_for


def register_subcommand(subparsers):
    return register_for(
        subparsers,
        name="constraint",
        tool_module="constraint",
        help="Plan / env validity, memory estimates, failure attribution.",
    )
