###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``pilot subagent`` — wraps ``pilot.tools.subagent``.

Protocol abstraction: concrete spawn impl is registered by
``pilot/integrations/<framework>/`` adapters (Cursor Task / Claude Code
Task / etc.). Direct invocation currently raises NotImplementedError.
"""

from __future__ import annotations

from pilot.cli.subcommands._base import register_for


def register_subcommand(subparsers):
    return register_for(
        subparsers,
        name="subagent",
        tool_module="subagent",
        help="Stage Worker spawn protocol (delegated to integrations).",
    )
