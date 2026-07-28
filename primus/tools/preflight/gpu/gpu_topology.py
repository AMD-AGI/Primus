###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Any, Dict, List, Optional

from .gpu_probe import probe_gpus
from .sysfs_probe import sysfs_gpu_bdfs, sysfs_topology_summary
from .utils import Finding, ProbeResult, run_cmd, which

logger = logging.getLogger(__name__)


def _device_consistency(devices: List[Dict[str, Any]]) -> List[Finding]:
    findings: List[Finding] = []
    names = [d.get("name") for d in devices if d.get("name")]
    totals = [d.get("total_memory") for d in devices if d.get("total_memory") is not None]
    if names and len(set(names)) > 1:
        findings.append(Finding("warn", "GPU names differ across devices", {"names": names}))
    if totals and len(set(totals)) > 1:
        findings.append(
            Finding("warn", "GPU total memory differs across devices", {"total_memory_bytes": totals})
        )
    return findings


def _numa_mapping_best_effort() -> Optional[Dict[str, Any]]:
    """
    Best-effort GPU<->NUMA mapping via sysfs (primary) or amd-smi (fallback).

    Sysfs reads are safe from any rank (no subprocesses, no mutex).
    amd-smi fallback only attempted on LOCAL_RANK == 0.
    """
    bdfs = sysfs_gpu_bdfs()
    if bdfs:
        return {"rc": 0, "gpus": bdfs, "source": "sysfs"}

    logger.debug("sysfs NUMA mapping unavailable; trying amd-smi fallback")

    from primus.tools.preflight.global_vars import LOCAL_RANK

    if LOCAL_RANK != 0:
        return None

    if which("amd-smi") is None:
        return None

    rc, out, err = run_cmd(["amd-smi", "list", "--csv"], timeout_s=10)
    if rc != 0 or not out:
        return {"rc": rc, "err": err, "note": "amd-smi list --csv failed"}

    lines = [l for l in out.splitlines() if l.strip()]
    bdf_list: List[str] = []
    for ln in lines[1:]:
        cols = [c.strip() for c in ln.split(",")]
        if len(cols) >= 2 and ":" in cols[1] and "." in cols[1]:
            bdf_list.append(cols[1])

    mapping: List[Dict[str, Any]] = []
    for i, bdf in enumerate(bdf_list):
        node_path = f"/sys/bus/pci/devices/{bdf}/numa_node"
        numa = None
        if os.path.exists(node_path):
            try:
                numa = int(open(node_path, "r", encoding="utf-8").read().strip())
            except Exception:
                numa = None
        mapping.append({"gpu": i, "pci_bdf": bdf, "numa_node": numa})

    return {"rc": rc, "gpus": mapping, "source": "amd-smi"}


def _xgmi_presence_best_effort() -> Optional[Dict[str, Any]]:
    """
    Best-effort topology (XGMI vs PCIe) via sysfs (primary) or amd-smi (fallback).

    Sysfs reads are safe from any rank. amd-smi fallback only on LOCAL_RANK == 0.
    """
    topo = sysfs_topology_summary()
    if topo is not None and topo.get("rc") == 0:
        return topo

    logger.debug("sysfs topology unavailable; trying amd-smi fallback")

    from primus.tools.preflight.global_vars import LOCAL_RANK

    if LOCAL_RANK != 0:
        return None

    if which("amd-smi") is None:
        return None
    rc, out, err = run_cmd(["amd-smi", "topology"], timeout_s=10)
    if rc != 0:
        return {"rc": rc, "err": err}
    return {"rc": rc, "out": out}


def _rccl_nccl_env_checks() -> List[Finding]:
    findings: List[Finding] = []
    nccl_if = os.environ.get("NCCL_SOCKET_IFNAME", "")
    ib_hca = os.environ.get("NCCL_IB_HCA", "")

    details = {"NCCL_SOCKET_IFNAME": nccl_if, "NCCL_IB_HCA": ib_hca}
    if not nccl_if:
        findings.append(Finding("warn", "NCCL_SOCKET_IFNAME not set (may fall back to socket)", details))
    if not ib_hca:
        findings.append(Finding("warn", "NCCL_IB_HCA not set (may fall back to socket)", details))
    if ib_hca and not (os.path.exists("/dev/infiniband") or os.path.exists("/sys/class/infiniband")):
        findings.append(Finding("warn", "NCCL_IB_HCA set but RDMA devices not present", details))
    return findings


def run_gpu_standard_checks(*, force_topology: bool = False) -> Dict[str, Any]:
    """
    Level: standard

    Purpose: Verify multi-GPU consistency and topology correctness.

    Most findings should be WARN by default (not FAIL).
    """
    findings: List[Finding] = []
    probe: ProbeResult = probe_gpus()
    if not probe.ok:
        # Standard checks are not meaningful without GPUs; keep this a WARN.
        findings.append(Finding("warn", "No GPUs detected; skipping standard checks", {}))
        return {"ok": True, "probe": probe, "findings": findings}

    findings.extend(_device_consistency(probe.devices))

    # NUMA mapping / imbalance detection (best-effort).
    # Prefers sysfs (safe for all ranks); falls back to amd-smi on LOCAL_RANK 0.
    from primus.tools.preflight.global_vars import LOCAL_RANK

    numa = _numa_mapping_best_effort()
    if numa is None and LOCAL_RANK != 0:
        findings.append(Finding("info", "NUMA mapping skipped (non-zero LOCAL_RANK, no sysfs)", {}))
    elif numa is None:
        findings.append(Finding("warn", "NUMA mapping unavailable (sysfs/amd-smi not found); skipped", {}))
    else:
        nodes = [x.get("numa_node") for x in numa.get("gpus", []) if x.get("numa_node") is not None]
        imbalance = False
        if nodes:
            counts = Counter(nodes).values()
            imbalance = len(set(counts)) > 1
        findings.append(
            Finding("info", "GPU↔NUMA mapping", {"mapping": numa.get("gpus", []), "imbalance": imbalance})
        )
        if imbalance:
            findings.append(Finding("warn", "GPU↔NUMA mapping imbalance detected", {"nodes": nodes}))

    # Topology (XGMI vs PCIe) best-effort.
    if force_topology or probe.backend == "rocm":
        topo = _xgmi_presence_best_effort()
        if topo is None and LOCAL_RANK != 0:
            findings.append(Finding("info", "Topology check skipped (non-zero LOCAL_RANK, no sysfs)", {}))
        elif topo is None:
            findings.append(Finding("warn", "Topology check skipped (sysfs/amd-smi not found)", {}))
        else:
            source = topo.get("source", "amd-smi")
            findings.append(Finding("info", f"GPU topology ({source})", topo))
            out = str(topo.get("out", "") or topo.get("matrix", ""))
            has_xgmi = topo.get("has_xgmi")
            if has_xgmi is False or (out and "PCIE" in out.upper() and "XGMI" not in out.upper()):
                findings.append(Finding("warn", "Topology indicates PCIe paths; XGMI may be absent", {}))

    # RCCL/NCCL env sanity (WARN-only).
    findings.extend(_rccl_nccl_env_checks())

    # Never hard-fail standard checks by default.
    return {"ok": True, "probe": probe, "findings": findings}
