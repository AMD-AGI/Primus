###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``pilot session`` — wraps ``pilot.tools.session`` (init / ...)."""

from __future__ import annotations

from pilot.cli.subcommands._base import register_for


def register_subcommand(subparsers):
    return register_for(
        subparsers,
        name="session",
        tool_module="session",
        help="Tuning-session bootstrap: create state/<session_id>/{tuning,target_vector,tuning_state}.yaml + r0 checkpoint.",
    )
