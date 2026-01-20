###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List

from .network_probe import probe_network
from .utils import Finding


def run_network_basic_checks() -> Dict[str, Any]:
    """
    Level: basic

    - Determine distributed intent
    - Check presence of essential env variables (presence-only)
    """
    probe = probe_network()
    findings: List[Finding] = []

    summary = {
        "level": "basic",
        "has_fail": False,
        "has_warn": False,
        "is_distributed": bool(probe.intent.get("is_distributed")),
        "network_mode": probe.intent.get("network_mode", "single-node"),
    }

    intent = {
        "is_distributed": bool(probe.intent.get("is_distributed")),
        "WORLD_SIZE": probe.intent.get("WORLD_SIZE", 1),
        "SLURM_NTASKS": probe.intent.get("SLURM_NTASKS", None),
        "OMPI_COMM_WORLD_SIZE": probe.intent.get("OMPI_COMM_WORLD_SIZE", None),
    }

    env_presence = {
        "MASTER_ADDR": probe.env.get("MASTER_ADDR"),
        "MASTER_PORT": probe.env.get("MASTER_PORT"),
        "WORLD_SIZE": probe.env.get("WORLD_SIZE"),
        "RANK": probe.env.get("RANK"),
        "LOCAL_RANK": probe.env.get("LOCAL_RANK"),
    }

    # Presence-only checks; warn if distributed intent but missing key vars.
    if intent["is_distributed"]:
        missing = [k for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK") if not env_presence.get(k)]
        if missing:
            findings.append(
                Finding(
                    "warn",
                    "Distributed intent detected but some env vars are missing",
                    {"missing": missing, "env": env_presence},
                )
            )

    summary["has_fail"] = any(f.level == "fail" for f in findings)
    summary["has_warn"] = any(f.level == "warn" for f in findings)

    # Emit as info findings so rank0 report can extract structured details.
    findings.append(Finding("info", "Network summary", {"summary": summary}))
    findings.append(Finding("info", "Distributed intent", {"intent": intent}))
    findings.append(Finding("info", "Distributed env presence", {"env": env_presence}))

    return {"probe": probe, "findings": findings}
