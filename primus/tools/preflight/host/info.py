###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Host info collection + report writing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .host_probe import (
    get_cpu_info,
    get_gpu_count_rocm,
    get_hostname,
    get_kernel_version,
    get_memory_info,
    get_numa_info,
    get_os_info,
    get_pcie_info,
    get_pcie_link_info,
    is_container,
    is_slurm_job,
)


@dataclass
class Finding:
    level: str  # "info" | "warn" | "fail"
    message: str
    details: Dict[str, Any]


def collect_host_info() -> List[Finding]:
    """Collect host info (CPU, memory, PCIe, etc.)."""
    findings: List[Finding] = []

    # Basic host info
    hostname = get_hostname()
    kernel = get_kernel_version()
    os_info = get_os_info()
    container = is_container()
    slurm = is_slurm_job()

    findings.append(
        Finding(
            level="info",
            message="Host identity",
            details={
                "hostname": hostname,
                "kernel": kernel,
                "os": os_info,
                "is_container": container,
                "is_slurm_job": slurm,
            },
        )
    )

    # CPU info
    cpu = get_cpu_info()
    findings.append(Finding(level="info", message="CPU info", details={"cpu": cpu}))

    # Memory info
    memory = get_memory_info()
    findings.append(Finding(level="info", message="Memory info", details={"memory": memory}))

    # Check for low memory
    if memory.get("available_gb", 0) > 0:
        available_pct = (memory["available_gb"] / memory["total_gb"]) * 100 if memory["total_gb"] > 0 else 0
        if available_pct < 10:
            findings.append(
                Finding(
                    level="warn",
                    message="Low available memory",
                    details={
                        "available_gb": memory["available_gb"],
                        "total_gb": memory["total_gb"],
                        "available_pct": round(available_pct, 1),
                    },
                )
            )

    # NUMA info
    numa = get_numa_info()
    if numa.get("nodes", 0) > 0:
        findings.append(Finding(level="info", message="NUMA topology", details={"numa": numa}))

    # PCIe devices (coarse inventory)
    pcie_devices = get_pcie_info()
    gpu_count_pcie = sum(1 for d in pcie_devices if d["type"] == "GPU")
    ib_count = sum(1 for d in pcie_devices if d["type"] == "Infiniband")
    eth_count = sum(1 for d in pcie_devices if d["type"] == "Ethernet")

    # Use rocm-smi as fallback for GPU count (more reliable in containers)
    gpu_count_rocm = get_gpu_count_rocm()
    gpu_count = max(gpu_count_pcie, gpu_count_rocm)

    findings.append(
        Finding(
            level="info",
            message="PCIe devices",
            details={
                "gpu_count": gpu_count,
                "infiniband_count": ib_count,
                "ethernet_count": eth_count,
                "devices": pcie_devices,
            },
        )
    )

    # PCIe link status for GPUs
    pcie_link = get_pcie_link_info()
    if pcie_link.get("gpu_links"):
        findings.append(Finding(level="info", message="PCIe link status", details={"pcie_link": pcie_link}))

        # Check for suboptimal PCIe config
        summary = pcie_link.get("summary", {})
        min_width = summary.get("min_width", "")
        if min_width and min_width != "x16":
            findings.append(
                Finding(
                    level="warn",
                    message=f"Some GPUs not running at x16 PCIe width (found {min_width})",
                    details={"pcie_summary": summary},
                )
            )

    return findings


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


def host_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize host-related fields for a host from per-rank records."""
    host_fail = 0
    host_warn = 0
    ranks: List[int] = []
    host_info: Dict[str, Any] = {}
    cpu_info: Dict[str, Any] = {}
    memory_info: Dict[str, Any] = {}
    numa_info: Dict[str, Any] = {}
    pcie_info: Dict[str, Any] = {}
    pcie_link_info: Dict[str, Any] = {}

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
            if not host_info:
                d = _find_first_finding_details(rf, "Host identity")
                if d:
                    host_info = d
            if not cpu_info:
                d = _find_first_finding_details(rf, "CPU info")
                if d and isinstance(d.get("cpu"), dict):
                    cpu_info = d["cpu"]
            if not memory_info:
                d = _find_first_finding_details(rf, "Memory info")
                if d and isinstance(d.get("memory"), dict):
                    memory_info = d["memory"]
            if not numa_info:
                d = _find_first_finding_details(rf, "NUMA topology")
                if d and isinstance(d.get("numa"), dict):
                    numa_info = d["numa"]
            if not pcie_info:
                d = _find_first_finding_details(rf, "PCIe devices")
                if d:
                    pcie_info = d
            if not pcie_link_info:
                d = _find_first_finding_details(rf, "PCIe link status")
                if d and isinstance(d.get("pcie_link"), dict):
                    pcie_link_info = d["pcie_link"]

    return {
        "ranks": sorted(set(ranks)),
        "status": _status_from_counts(host_fail, host_warn),
        "host_info": host_info,
        "cpu": cpu_info,
        "memory": memory_info,
        "numa": numa_info,
        "pcie": pcie_info,
        "pcie_link": pcie_link_info,
    }


def write_host_report(f: Any, by_host: Dict[str, List[Dict[str, Any]]]) -> None:
    """Write host report sections to file handle."""
    # Section header
    f.write("---\n\n")
    f.write("# Host Info\n\n")

    # Host System Info table
    f.write("## Host System\n\n")
    f.write("| host | kernel | is_container | is_slurm |\n")
    f.write("|---|---|---|---|\n")
    for h in sorted(by_host.keys()):
        s = host_summary(by_host[h])
        hi = s.get("host_info", {}) or {}
        kernel = hi.get("kernel", "")
        container = "yes" if hi.get("is_container") else "no"
        slurm = "yes" if hi.get("is_slurm_job") else "no"
        f.write(f"| {h} | {kernel} | {container} | {slurm} |\n")
    f.write("\n")

    # CPU Info table
    f.write("## CPU\n\n")
    f.write("| host | model | sockets | cores/socket | threads/core | logical_cores | numa_nodes |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|\n")
    for h in sorted(by_host.keys()):
        s = host_summary(by_host[h])
        cpu = s.get("cpu", {}) or {}
        model = cpu.get("model_name", "")
        if len(model) > 40:
            model = model[:37] + "..."
        sockets = cpu.get("sockets", "")
        cores_per_socket = cpu.get("cores_per_socket", "")
        threads_per_core = cpu.get("threads_per_core", "")
        logical_cores = cpu.get("logical_cores", "")
        numa_nodes = cpu.get("numa_nodes", "")
        f.write(
            f"| {h} | {model} | {sockets} | {cores_per_socket} | {threads_per_core} | {logical_cores} | {numa_nodes} |\n"
        )
    f.write("\n")

    # Memory Info table
    f.write("## Memory\n\n")
    f.write("| host | total_gb | available_gb | free_gb | cached_gb | swap_total_gb |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    for h in sorted(by_host.keys()):
        s = host_summary(by_host[h])
        mem = s.get("memory", {}) or {}
        f.write(
            f"| {h} | {mem.get('total_gb', '')} | {mem.get('available_gb', '')} | "
            f"{mem.get('free_gb', '')} | {mem.get('cached_gb', '')} | {mem.get('swap_total_gb', '')} |\n"
        )
    f.write("\n")

    # PCIe Link table (GPU bandwidth)
    f.write("## PCIe Link Status\n\n")
    f.write("| host | gpu_links | speed | width | per_gpu_bw (GB/s) | total_bw (GB/s) |\n")
    f.write("|---|---:|---|---|---:|---:|\n")
    for h in sorted(by_host.keys()):
        s = host_summary(by_host[h])
        pcie_link = s.get("pcie_link", {}) or {}
        summary = pcie_link.get("summary", {}) or {}
        gpu_links = len(pcie_link.get("gpu_links", []))
        min_speed = summary.get("min_speed", "")
        max_speed = summary.get("max_speed", "")
        speed = min_speed if min_speed == max_speed else f"{min_speed}-{max_speed}" if min_speed else ""
        min_width = summary.get("min_width", "")
        max_width = summary.get("max_width", "")
        width = min_width if min_width == max_width else f"{min_width}-{max_width}" if min_width else ""
        per_gpu_bw = summary.get("per_gpu_bandwidth_gbps", "")
        total_bw = summary.get("total_bandwidth_gbps", "")
        f.write(f"| {h} | {gpu_links} | {speed} | {width} | {per_gpu_bw} | {total_bw} |\n")
    f.write("\n")
