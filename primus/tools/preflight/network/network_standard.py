###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .network_probe import probe_network
from .utils import Finding


def _ifname_suspect(ifname: Optional[str]) -> bool:
    if not ifname:
        return False
    bad = {"lo", "docker0"}
    if ifname in bad:
        return True
    if ifname.startswith("veth"):
        return True
    return False


def _expected_ib(env: Dict[str, Any], available_nics: List[str], override: Optional[bool]) -> bool:
    if override is not None:
        return bool(override)

    # Heuristics:
    # - If user set NCCL_IB_HCA, they expect IB.
    # - If NCCL_SOCKET_IFNAME points to ib*, they expect IB.
    # - If there are ib* nics, assume IB might be expected in cluster env.
    if env.get("NCCL_IB_HCA"):
        return True
    if str(env.get("NCCL_SOCKET_IFNAME") or "").startswith("ib"):
        return True
    if any(x.startswith("ib") for x in available_nics):
        return True
    if os.environ.get("PRIMUS_EXPECT_IB", "") == "1":
        return True
    return False


def run_network_standard_checks() -> Dict[str, Any]:
    """
    Level: standard

    Validate likely network path issues (NIC/IB/RCCL snapshot).
    Mostly WARN by default.
    """
    probe = probe_network()
    findings: List[Finding] = []

    available_nics = probe.available_nics
    env = probe.env

    nccl_if = env.get("NCCL_SOCKET_IFNAME")
    gloo_if = env.get("GLOO_SOCKET_IFNAME")

    ifname_valid = True
    if nccl_if and nccl_if not in available_nics:
        ifname_valid = False
        findings.append(
            Finding(
                "warn",
                "NCCL_SOCKET_IFNAME does not match any visible NIC",
                {"ifname": nccl_if, "available_nics": available_nics},
            )
        )
    if gloo_if and gloo_if not in available_nics:
        ifname_valid = False
        findings.append(
            Finding(
                "warn",
                "GLOO_SOCKET_IFNAME does not match any visible NIC",
                {"ifname": gloo_if, "available_nics": available_nics},
            )
        )

    ifname_suspect = _ifname_suspect(nccl_if) or _ifname_suspect(gloo_if)
    if ifname_suspect:
        findings.append(
            Finding(
                "warn",
                "IFNAME looks suspect (lo/docker0/veth)",
                {"NCCL_SOCKET_IFNAME": nccl_if, "GLOO_SOCKET_IFNAME": gloo_if},
            )
        )

    nics_metrics = {
        "available_nics": available_nics,
        "NCCL_SOCKET_IFNAME": nccl_if,
        "GLOO_SOCKET_IFNAME": gloo_if,
        "ifname_valid": ifname_valid,
        "ifname_suspect": ifname_suspect,
    }
    findings.append(Finding("info", "NIC and network path", {"nics": nics_metrics}))

    expected = _expected_ib(env, available_nics, None)
    ib_devices = probe.ib_devices
    has_ib = len(ib_devices) > 0

    ib_disable = str(env.get("NCCL_IB_DISABLE") or "0")
    if ib_disable not in ("0", "1"):
        # keep as string; unusual but don't fail
        findings.append(
            Finding("warn", "NCCL_IB_DISABLE has an unexpected value", {"NCCL_IB_DISABLE": ib_disable})
        )

    if ib_disable == "1":
        ib_status = "disabled"
    elif not has_ib:
        ib_status = "absent"
    else:
        ib_status = "enabled"

    ib_metrics = {
        "expected_ib": expected,
        "has_ib": has_ib,
        "ib_devices": ib_devices,
        "NCCL_IB_DISABLE": ib_disable,
        "ib_status": ib_status,
    }
    findings.append(Finding("info", "InfiniBand / RDMA", {"ib": ib_metrics}))

    if expected and ib_status != "enabled":
        findings.append(Finding("warn", "IB expected but not enabled/available", {"ib": ib_metrics}))

    # Snapshot (no evaluation)
    rccl = {
        "NCCL_SOCKET_IFNAME": nccl_if,
        "NCCL_IB_HCA": env.get("NCCL_IB_HCA"),
        "NCCL_NET_GDR_LEVEL": env.get("NCCL_NET_GDR_LEVEL"),
        "NCCL_DEBUG": env.get("NCCL_DEBUG"),
    }
    findings.append(Finding("info", "RCCL/NCCL snapshot", {"rccl": rccl}))

    # In standard report we often want to highlight missing IB HCA selection.
    if expected and not env.get("NCCL_IB_HCA"):
        findings.append(
            Finding(
                "warn",
                "NCCL_IB_HCA not set (may fall back to socket)",
                {"NCCL_IB_HCA": env.get("NCCL_IB_HCA") or ""},
            )
        )

    return {"probe": probe, "findings": findings}
