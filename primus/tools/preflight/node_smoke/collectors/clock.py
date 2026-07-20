###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tier 1 -- E: clock state (wall time + time-daemon active states)."""

from __future__ import annotations

import time
from typing import Any, Dict

from ..shell_utils import _systemctl_is_active


def _collect_clock_state() -> Dict[str, Any]:
    """Capture this node's wall time and time-daemon health.

    Wall time is captured early so the aggregator can compute a
    cluster-wide spread. Note this includes srun launch jitter, so the
    spread is an *upper bound* on the real clock skew. The aggregator
    uses loose thresholds (warn at 30 s, no hard fail) for the spread,
    and reserves the hard fail for "no time-sync daemon active".
    """
    out: Dict[str, Any] = {
        "wall_time_unix": time.time(),
        "monotonic": time.monotonic(),
        "daemons": {},
    }
    for unit in ("chronyd", "ntp", "ntpd", "systemd-timesyncd"):
        out["daemons"][unit] = _systemctl_is_active(unit)
    active = [u for u, s in out["daemons"].items() if s == "active"]
    out["any_active"] = bool(active)
    out["active_units"] = active
    return out
