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


def write_network_report(f: Any, by_host: Dict[str, List[Dict[str, Any]]]) -> None:
    """Write network report sections to file handle."""
    # Network Status table
    f.write("## Network Status\n\n")
    f.write("| host | ranks | status | is_distributed | network_mode | has_fail | has_warn |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for h in sorted(by_host.keys()):
        s = host_network_summary(by_host[h])
        ranks_str = ",".join(str(x) for x in s["ranks"]) if s["ranks"] else ""
        summ = s.get("summary", {}) or {}
        f.write(
            f"| {h} | {ranks_str} | {s['status']} | {summ.get('is_distributed','')} | "
            f"{summ.get('network_mode','')} | {summ.get('has_fail','')} | {summ.get('has_warn','')} |\n"
        )
    f.write("\n")

    # Distributed Environment table
    f.write("## Distributed Environment\n\n")
    f.write(
        "| host | WORLD_SIZE | SLURM_NTASKS | OMPI_COMM_WORLD_SIZE | MASTER_ADDR | MASTER_PORT | RANK | LOCAL_RANK |\n"
    )
    f.write("|---|---:|---:|---:|---|---|---:|---:|\n")
    for h in sorted(by_host.keys()):
        s = host_network_summary(by_host[h])
        intent = s.get("intent", {}) or {}
        envp = s.get("env", {}) or {}
        f.write(
            f"| {h} | {intent.get('WORLD_SIZE','')} | {intent.get('SLURM_NTASKS','')} | "
            f"{intent.get('OMPI_COMM_WORLD_SIZE','')} | {envp.get('MASTER_ADDR','')} | {envp.get('MASTER_PORT','')} | "
            f"{envp.get('RANK','')} | {envp.get('LOCAL_RANK','')} |\n"
        )
    f.write("\n")

    # Network Path table
    f.write("## Network Path\n\n")
    f.write("| host | NCCL_SOCKET_IFNAME | GLOO_SOCKET_IFNAME | ifname_valid | ifname_suspect |\n")
    f.write("|---|---|---|---|---|\n")
    for h in sorted(by_host.keys()):
        s = host_network_summary(by_host[h])
        nics = s.get("nics", {}) or {}
        f.write(
            f"| {h} | {nics.get('NCCL_SOCKET_IFNAME','')} | {nics.get('GLOO_SOCKET_IFNAME','')} | "
            f"{nics.get('ifname_valid','')} | {nics.get('ifname_suspect','')} |\n"
        )
    f.write("\n")

    # InfiniBand / RDMA table
    f.write("## InfiniBand / RDMA\n\n")
    f.write("| host | expected_ib | has_ib | ib_status | NCCL_IB_DISABLE | ib_devices |\n")
    f.write("|---|---|---|---|---|---|\n")
    for h in sorted(by_host.keys()):
        s = host_network_summary(by_host[h])
        ib = s.get("ib", {}) or {}
        devs = ib.get("ib_devices", [])
        devs_s = ",".join(devs) if isinstance(devs, list) else str(devs)
        f.write(
            f"| {h} | {ib.get('expected_ib','')} | {ib.get('has_ib','')} | {ib.get('ib_status','')} | "
            f"{ib.get('NCCL_IB_DISABLE','')} | {devs_s} |\n"
        )
    f.write("\n")

    # RCCL / NCCL Configuration table
    f.write("## RCCL / NCCL Configuration\n\n")
    f.write("| host | NCCL_IB_HCA | NCCL_NET_GDR_LEVEL | NCCL_DEBUG |\n")
    f.write("|---|---|---|---|\n")
    for h in sorted(by_host.keys()):
        s = host_network_summary(by_host[h])
        rccl = s.get("rccl", {}) or {}
        f.write(
            f"| {h} | {rccl.get('NCCL_IB_HCA','')} | {rccl.get('NCCL_NET_GDR_LEVEL','')} | {rccl.get('NCCL_DEBUG','')} |\n"
        )
    f.write("\n")

    # Runtime Process Group table
    f.write("## Runtime Process Group\n\n")
    f.write("| host | pg_backend | pg_init_ok | pg_error |\n")
    f.write("|---|---|---|---|\n")
    for h in sorted(by_host.keys()):
        s = host_network_summary(by_host[h])
        rt = s.get("runtime", {}) or {}
        f.write(f"| {h} | {rt.get('pg_backend','')} | {rt.get('pg_init_ok','')} | {rt.get('pg_error','')} |\n")
    f.write("\n")

    # (Optional) Minimal Communication Sanity table
    f.write("## (Optional) Minimal Communication Sanity\n\n")
    f.write("| host | allreduce_tested | allreduce_ok | allreduce_error |\n")
    f.write("|---|---|---|---|\n")
    for h in sorted(by_host.keys()):
        s = host_network_summary(by_host[h])
        runtime_comm_row = s.get("runtime_comm", {}) or {}
        f.write(
            f"| {h} | {runtime_comm_row.get('allreduce_tested','')} | {runtime_comm_row.get('allreduce_ok','')} | {runtime_comm_row.get('allreduce_error','')} |\n"
        )
    f.write("\n")
