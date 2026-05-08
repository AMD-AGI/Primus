###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``pilot preflight`` — wraps ``pilot.tools.preflight``.

Subcommands (run ``pilot preflight -h`` for full options):
    run        Collect / refresh ClusterProfile (5-step protocol)
    env_probe  Validate env_baseline candidate (3-tier safe probe)
    env_sweep  Inner-loop env diff sweep (≤ 8 combos, ≤ 50 step)
"""

from __future__ import annotations

from pilot.cli.subcommands._base import register_for


def register_subcommand(subparsers):
    return register_for(
        subparsers,
        name="preflight",
        tool_module="preflight",
        help="Cluster preflight: ClusterProfile + env_baseline + EnvSweep.",
    )
