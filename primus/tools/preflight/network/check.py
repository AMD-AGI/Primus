###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Network check orchestration and summary functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .network_basic import run_network_basic_checks
from .network_full import run_network_full_checks
from .network_standard import run_network_standard_checks


@dataclass
class Finding:
    level: str  # "info" | "warn" | "fail"
    message: str
    details: Dict[str, Any]


def run_network_checks(expect_ib: bool = False, comm_sanity: bool = False) -> List[Finding]:
    """Run all network checks (basic + standard + full)."""
    out: List[Finding] = []

    # basic
    nb = run_network_basic_checks()
    for f in nb["findings"]:
        out.append(Finding(level=f.level, message=f.message, details=f.details))

    # standard
    ns = run_network_standard_checks(expect_ib=True if expect_ib else None)
    for f in ns["findings"]:
        out.append(Finding(level=f.level, message=f.message, details=f.details))

    # full
    nf = run_network_full_checks(comm_sanity=comm_sanity)
    for f in nf["findings"]:
        out.append(Finding(level=f.level, message=f.message, details=f.details))

    return out


def _status_from_counts(fail_count: int, warn_count: int) -> str:
    if fail_count > 0:
        return "FAIL"
    if warn_count > 0:
        return "WARN"
    return "OK"


def _find_first_finding_details(findings: List[Dict[str, Any]], message: str) -> Optional[Dict[str, Any]]:
    for x in findings:
        if x.get("message") == message:
            d = x.get("details")
            return d if isinstance(d, dict) else None
    return None


def host_network_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize network-related fields for a host from per-rank records.
    """
    host_fail = 0
    host_warn = 0
    ranks: List[int] = []
    env: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    intent: Dict[str, Any] = {}
    nics: Dict[str, Any] = {}
    ib: Dict[str, Any] = {}
    rccl: Dict[str, Any] = {}
    runtime: Dict[str, Any] = {}
    runtime_comm: Dict[str, Any] = {}
    for r in records:
        host_fail += int(r.get("fail_count", 0) or 0)
        host_warn += int(r.get("warn_count", 0) or 0)
        if r.get("rank") is not None:
            try:
                ranks.append(int(r["rank"]))
            except Exception:
                pass
        rf = r.get("findings", [])
        if isinstance(rf, list):
            if not summary:
                d = _find_first_finding_details(rf, "Network summary")
                if d and isinstance(d.get("summary"), dict):
                    summary = d["summary"]
            if not intent:
                d = _find_first_finding_details(rf, "Distributed intent")
                if d and isinstance(d.get("intent"), dict):
                    intent = d["intent"]
            if not env:
                d = _find_first_finding_details(rf, "Distributed env presence")
                if d and isinstance(d.get("env"), dict):
                    env = d["env"]
            if not nics:
                d = _find_first_finding_details(rf, "NIC and network path")
                if d and isinstance(d.get("nics"), dict):
                    nics = d["nics"]
            if not ib:
                d = _find_first_finding_details(rf, "InfiniBand / RDMA")
                if d and isinstance(d.get("ib"), dict):
                    ib = d["ib"]
            if not rccl:
                d = _find_first_finding_details(rf, "RCCL/NCCL snapshot")
                if d and isinstance(d.get("rccl"), dict):
                    rccl = d["rccl"]
            if not runtime:
                d = _find_first_finding_details(rf, "Runtime process group sanity")
                if d and isinstance(d.get("runtime"), dict):
                    runtime = d["runtime"]
            if not runtime_comm:
                d = _find_first_finding_details(rf, "Minimal communication sanity")
                if d and isinstance(d.get("runtime_comm"), dict):
                    runtime_comm = d["runtime_comm"]
    return {
        "ranks": sorted(set(ranks)),
        "status": _status_from_counts(host_fail, host_warn),
        "summary": summary,
        "intent": intent,
        "env": env,
        "nics": nics,
        "ib": ib,
        "rccl": rccl,
        "runtime": runtime,
        "runtime_comm": runtime_comm,
    }
