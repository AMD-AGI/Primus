###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Direct sysfs GPU probing — reads KFD topology and DRM sysfs to enumerate AMD
GPUs, PCI BDFs, NUMA mapping, and inter-GPU link topology (XGMI vs PCIe)
without invoking amd-smi or rocm-smi subprocesses.

Approach adapted from RCCL's alt_rsmi.cc
(https://github.com/ROCm/rocm-systems/blob/develop/projects/rccl/src/misc/alt_rsmi.cc)
which reads the same KFD sysfs paths to avoid rocm_smi_lib's /dev/shm mutex
contention (rocm_smi_lib#88).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

KFD_NODES_ROOT = "/sys/class/kfd/kfd/topology/nodes"
AMD_VENDOR_ID = 0x1002

LINK_TYPE_XGMI = 11
LINK_TYPE_PCIE = 2


@dataclass
class GpuNode:
    node_id: int = 0
    gpu_id: int = 0
    unique_id: int = 0
    location_id: int = 0
    domain: int = 0
    bdf: int = 0
    bus: int = 0
    device: int = 0
    function: int = 0
    partition_id: int = 0
    pci_bdf_str: str = ""
    numa_node: Optional[int] = None
    properties: Dict[str, int] = field(default_factory=dict)


@dataclass
class LinkInfo:
    src_idx: int = 0
    dst_idx: int = 0
    link_type: str = "unknown"
    weight: int = 0
    min_bandwidth: int = 0
    max_bandwidth: int = 0
    hops: int = 0


@dataclass
class SysfsProbeResult:
    ok: bool = False
    gpu_count: int = 0
    gpus: List[GpuNode] = field(default_factory=list)
    links: List[LinkInfo] = field(default_factory=list)
    error: Optional[str] = None


def _read_sysfs_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def _read_sysfs_int(path: str) -> Optional[int]:
    val = _read_sysfs_file(path)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _read_kfd_properties(node_id: int) -> Dict[str, int]:
    """Parse /sys/class/kfd/kfd/topology/nodes/{node_id}/properties → dict."""
    props: Dict[str, int] = {}
    path = f"{KFD_NODES_ROOT}/{node_id}/properties"
    content = _read_sysfs_file(path)
    if content is None:
        return props
    for line in content.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                props[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return props


def _read_kfd_gpu_id(node_id: int) -> Optional[int]:
    """Read /sys/class/kfd/kfd/topology/nodes/{node_id}/gpu_id."""
    return _read_sysfs_int(f"{KFD_NODES_ROOT}/{node_id}/gpu_id")


def _read_link_properties(node_id: int, link_id: int) -> Dict[str, int]:
    """Parse /sys/class/kfd/kfd/topology/nodes/{node_id}/io_links/{link_id}/properties."""
    path = f"{KFD_NODES_ROOT}/{node_id}/io_links/{link_id}/properties"
    props: Dict[str, int] = {}
    content = _read_sysfs_file(path)
    if content is None:
        return props
    for line in content.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                props[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return props


def _count_io_links(node_id: int) -> int:
    """Count subdirectories in io_links/ for a given node."""
    links_dir = f"{KFD_NODES_ROOT}/{node_id}/io_links"
    if not os.path.isdir(links_dir):
        return 0
    count = 0
    try:
        for entry in os.listdir(links_dir):
            if os.path.isdir(os.path.join(links_dir, entry)) and entry not in (".", ".."):
                count += 1
    except OSError:
        pass
    return count


def _bdf_to_string(domain: int, bus: int, device: int, function: int) -> str:
    """Format PCI BDF as string like '0000:c1:00.0'."""
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function:x}"


_GPU_NODES_CACHE: Optional[List["GpuNode"]] = None


def _enumerate_gpu_nodes() -> List[GpuNode]:
    """
    Scan KFD topology nodes, filter AMD GPUs (vendor_id == 0x1002),
    compute PCI BDF from location_id + domain, sort by BDF.

    Results are cached at module level since GPU topology is static
    during a single preflight run and this is called from multiple paths.

    Logic mirrors ARSMI_init() from alt_rsmi.cc.
    """
    global _GPU_NODES_CACHE
    if _GPU_NODES_CACHE is not None:
        return _GPU_NODES_CACHE

    if not os.path.isdir(KFD_NODES_ROOT):
        return []

    raw_nodes: List[Tuple[int, GpuNode]] = []

    try:
        entries = os.listdir(KFD_NODES_ROOT)
    except OSError:
        return []

    for entry in entries:
        if not entry.isdigit():
            continue

        node_id = int(entry)
        gpu_id = _read_kfd_gpu_id(node_id)
        if gpu_id is None or gpu_id == 0:
            continue

        props = _read_kfd_properties(node_id)
        vendor_id = props.get("vendor_id", 0)
        if vendor_id != AMD_VENDOR_ID:
            continue

        unique_id = props.get("unique_id", 0)
        location_id = props.get("location_id", 0)
        domain = props.get("domain", 0) & 0xFFFFFFFF

        bdf_raw = (domain << 32) | location_id
        bus = (location_id >> 8) & 0xFF
        device = (location_id >> 3) & 0x1F
        function = location_id & 0x7
        partition_id = (location_id >> 28) & 0xF

        pci_bdf_str = _bdf_to_string(domain, bus, device, function)

        numa = _read_sysfs_int(f"/sys/bus/pci/devices/{pci_bdf_str}/numa_node")

        node = GpuNode(
            node_id=node_id,
            gpu_id=gpu_id,
            unique_id=unique_id,
            location_id=location_id,
            domain=domain,
            bdf=bdf_raw,
            bus=bus,
            device=device,
            function=function,
            partition_id=partition_id,
            pci_bdf_str=pci_bdf_str,
            numa_node=numa,
            properties=props,
        )
        raw_nodes.append((bdf_raw, node))

    raw_nodes.sort(key=lambda x: x[0])
    _GPU_NODES_CACHE = [n for _, n in raw_nodes]
    return _GPU_NODES_CACHE


def _build_link_matrix(gpu_nodes: List[GpuNode]) -> List[LinkInfo]:
    """
    Read io_links for each GPU node to determine XGMI vs PCIe connectivity.
    Returns a flat list of LinkInfo for GPU-to-GPU links only.

    Logic mirrors the link matrix construction in ARSMI_init().
    """
    if not gpu_nodes:
        return []

    node_id_to_idx = {g.node_id: i for i, g in enumerate(gpu_nodes)}
    links: List[LinkInfo] = []

    for src_idx, src_node in enumerate(gpu_nodes):
        n_links = _count_io_links(src_node.node_id)
        for link_id in range(n_links):
            props = _read_link_properties(src_node.node_id, link_id)
            if not props:
                continue

            dst_node_id = props.get("node_to")
            if dst_node_id is None:
                continue

            dst_idx = node_id_to_idx.get(dst_node_id)
            if dst_idx is None:
                continue

            link_type_raw = props.get("type", 0)
            weight = props.get("weight", 0)
            min_bw = props.get("min_bandwidth", 0)
            max_bw = props.get("max_bandwidth", 0)

            if link_type_raw == LINK_TYPE_XGMI:
                link_type = "XGMI"
                hops = 1
            elif link_type_raw == LINK_TYPE_PCIE:
                link_type = "PCIe"
                hops = 2
            else:
                link_type = "unknown"
                hops = 0

            links.append(LinkInfo(
                src_idx=src_idx,
                dst_idx=dst_idx,
                link_type=link_type,
                weight=weight,
                min_bandwidth=min_bw,
                max_bandwidth=max_bw,
                hops=hops,
            ))

    return links


def sysfs_probe() -> SysfsProbeResult:
    """
    Main entry point: enumerate AMD GPUs and topology via sysfs.

    No subprocess calls, no /dev/shm mutex, safe to call from any rank.
    """
    try:
        gpu_nodes = _enumerate_gpu_nodes()
        if not gpu_nodes:
            return SysfsProbeResult(ok=False, error="No AMD GPUs found via KFD sysfs")

        links = _build_link_matrix(gpu_nodes)
        return SysfsProbeResult(
            ok=True,
            gpu_count=len(gpu_nodes),
            gpus=gpu_nodes,
            links=links,
        )
    except Exception as e:
        return SysfsProbeResult(ok=False, error=f"sysfs probe failed: {e}")


# ── Convenience helpers for integration with existing preflight code ──


def sysfs_gpu_count() -> int:
    """GPU count via KFD sysfs. Returns 0 on failure."""
    try:
        nodes = _enumerate_gpu_nodes()
        return len(nodes)
    except Exception:
        return 0


def sysfs_gpu_bdfs() -> List[Dict[str, Any]]:
    """
    Return per-GPU BDF + NUMA mapping, compatible with the format
    _numa_mapping_best_effort() currently returns.
    """
    try:
        nodes = _enumerate_gpu_nodes()
        return [
            {"gpu": i, "pci_bdf": n.pci_bdf_str, "numa_node": n.numa_node}
            for i, n in enumerate(nodes)
        ]
    except Exception:
        return []


def sysfs_has_xgmi() -> Optional[bool]:
    """
    Check if any GPU-to-GPU link is XGMI. Returns None if probe fails.
    """
    try:
        nodes = _enumerate_gpu_nodes()
        if not nodes:
            return None
        links = _build_link_matrix(nodes)
        return any(lk.link_type == "XGMI" for lk in links)
    except Exception:
        return None


def sysfs_topology_summary() -> Optional[Dict[str, Any]]:
    """
    Build a topology summary similar to what `amd-smi topo` provides,
    formatted for preflight reporting.
    """
    try:
        nodes = _enumerate_gpu_nodes()
        if not nodes:
            return None

        links = _build_link_matrix(nodes)
        n = len(nodes)

        matrix: List[List[str]] = [["" for _ in range(n)] for _ in range(n)]
        for i in range(n):
            matrix[i][i] = "self"

        for lk in links:
            matrix[lk.src_idx][lk.dst_idx] = lk.link_type

        header = [f"GPU{i}" for i in range(n)]
        lines = ["       " + "  ".join(f"{h:>6}" for h in header)]
        for i in range(n):
            row = f"GPU{i:<3} " + "  ".join(f"{matrix[i][j]:>6}" for j in range(n))
            lines.append(row)

        has_xgmi = any(lk.link_type == "XGMI" for lk in links)

        return {
            "rc": 0,
            "source": "sysfs",
            "gpu_count": n,
            "has_xgmi": has_xgmi,
            "matrix": "\n".join(lines),
            "links": [
                {
                    "src": lk.src_idx,
                    "dst": lk.dst_idx,
                    "type": lk.link_type,
                    "weight": lk.weight,
                }
                for lk in links
            ],
        }
    except Exception as e:
        return {"rc": 1, "error": str(e)}
