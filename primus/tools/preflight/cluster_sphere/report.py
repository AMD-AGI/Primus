###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import io
from typing import Any, Dict, List

from primus.tools.preflight.network.info import Finding


def wants_cluster_sphere_env(args: Any) -> bool:
    return bool(getattr(args, "cluster_sphere", False) or getattr(args, "cluster_sphere_env", False))


def wants_cluster_sphere_rdma_bw(args: Any) -> bool:
    return bool(getattr(args, "cluster_sphere", False) or getattr(args, "cluster_sphere_rdma_bw", False))


def emit_cluster_sphere_env_markdown(hostname: str, findings: List[Finding]) -> str:
    """Serialize Cluster Sphere env findings to Markdown (standalone CLI / Slurm)."""
    fd: List[Dict[str, Any]] = []
    for fin in findings:
        fd.append({"message": fin.message, "level": fin.level, "details": fin.details})
    gathered = [{"host": hostname, "findings": fd}]
    buf = io.StringIO()
    write_cluster_sphere_env_section(buf, gathered)
    return buf.getvalue()


def write_cluster_sphere_env_section(f, gathered: List[Dict[str, Any]]) -> None:
    """Append Cluster Sphere env recommender markdown (rank0)."""
    f.write("## Cluster Sphere — RDMA environment recommendations\n\n")

    blocks = 0
    for r in gathered:
        host = str(r.get("host", "unknown"))
        for fin in r.get("findings", []):
            if not isinstance(fin, dict):
                continue
            msg = fin.get("message", "")
            if msg == "Cluster Sphere RDMA environment recommendations":
                blocks += 1
                details = fin.get("details") or {}
                _write_one_host_block(f, host, details)
            elif msg == "Cluster Sphere env recommender: no RDMA devices found":
                blocks += 1
                f.write(f"### Host `{host}`\n\n")
                f.write("_No RDMA devices detected for Cluster Sphere env recommender._\n\n")
                w = (fin.get("details") or {}).get("warnings") or []
                if w:
                    f.write("Warnings:\n\n")
                    for line in w:
                        f.write(f"- {line}\n")
                    f.write("\n")

    if blocks == 0:
        f.write("_No Cluster Sphere environment data was collected._\n\n")


def _write_one_host_block(f, host: str, details: Dict[str, Any]) -> None:
    f.write(f"### Host `{host}`\n\n")

    warns = details.get("warnings") or []
    if warns:
        f.write("**Warnings:**\n\n")
        for w in warns:
            f.write(f"- {w}\n")
        f.write("\n")

    devices = details.get("devices") or []
    if devices:
        f.write("| RDMA | PCI | NETDEV | Firmware | GID idx | GID | Vendor |\n")
        f.write("|------|-----|--------|----------|---------|-----|--------|\n")
        for d in devices:
            f.write(
                f"| {d.get('rdma','')} | {d.get('pci','')} | {d.get('netdev','')} | "
                f"{d.get('firmware','')} | {d.get('gid_index','')} | {d.get('gid_value','')} | "
                f"{d.get('vendor','')} |\n"
            )
        f.write("\n")

    fw_by = details.get("firmware_by_version") or {}
    if fw_by:
        f.write("**Firmware report:**\n\n")
        f.write("| Firmware | RDMA devices |\n")
        f.write("|----------|---------------|\n")
        for fw in sorted(fw_by.keys()):
            devs = fw_by[fw]
            devs_s = ", ".join(devs) if isinstance(devs, list) else str(devs)
            f.write(f"| {fw} | {devs_s} |\n")
        f.write("\n")

    nccl = details.get("nccl_exports") or []
    if nccl:
        f.write("**Suggested NCCL / socket exports:**\n\n```bash\n")
        for line in nccl:
            f.write(f"{line}\n")
        f.write("```\n\n")

    roc = details.get("rocshmem_exports") or []
    if roc:
        f.write("**Suggested rocSHMEM exports:**\n\n```bash\n")
        for line in roc:
            f.write(f"{line}\n")
        f.write("```\n\n")

    docker_cmd = (details.get("docker_cmd") or "").strip()
    if docker_cmd:
        f.write("**Example Docker launch (vendor-specific template):**\n\n```bash\n")
        f.write(docker_cmd)
        f.write("\n```\n\n")
