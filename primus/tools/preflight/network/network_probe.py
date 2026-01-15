###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import os
import re
import socket
import subprocess
from typing import Any, Dict, List, Optional

from .utils import NetworkProbe


def _env_get(name: str) -> Optional[str]:
    v = os.environ.get(name)
    return None if v is None or v == "" else v


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def detect_distributed_intent() -> Dict[str, Any]:
    world_size = _env_int("WORLD_SIZE", 1)
    slurm_ntasks = _env_get("SLURM_NTASKS")
    ompi_size = _env_get("OMPI_COMM_WORLD_SIZE")

    slurm_ntasks_i = int(slurm_ntasks) if slurm_ntasks and slurm_ntasks.isdigit() else None
    ompi_size_i = int(ompi_size) if ompi_size and ompi_size.isdigit() else None

    is_distributed = (
        bool(world_size and world_size > 1)
        or bool(slurm_ntasks_i and slurm_ntasks_i > 1)
        or bool(ompi_size_i and ompi_size_i > 1)
    )
    network_mode = "multi-node" if is_distributed else "single-node"

    return {
        "is_distributed": is_distributed,
        "WORLD_SIZE": world_size,
        "SLURM_NTASKS": slurm_ntasks_i,
        "OMPI_COMM_WORLD_SIZE": ompi_size_i,
        "network_mode": network_mode,
    }


def list_nics() -> List[str]:
    try:
        return sorted([x for x in os.listdir("/sys/class/net") if x])
    except Exception:
        return []


def list_ib_devices() -> List[str]:
    try:
        return sorted([x for x in os.listdir("/sys/class/infiniband") if x])
    except Exception:
        return []


def list_ipv4_addrs() -> Dict[str, List[str]]:
    """
    Return IPv4 addresses per interface, best-effort, using `ip -o -4 addr show`.
    Example: {"eth0": ["10.0.0.2"], "ib0": ["172.16.0.2"]}.
    """
    out: Dict[str, List[str]] = {}
    try:
        r = subprocess.run(
            ["ip", "-o", "-4", "addr", "show"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
        if r.returncode != 0:
            return out
        for line in r.stdout.splitlines():
            # e.g. "2: eth0    inet 10.0.0.2/24 brd 10.0.0.255 scope global eth0"
            m = re.search(r"^\s*\d+:\s+(\S+)\s+inet\s+(\d+\.\d+\.\d+\.\d+)/", line)
            if not m:
                continue
            ifname, ip = m.group(1), m.group(2)
            out.setdefault(ifname, []).append(ip)
    except Exception:
        return out
    return out


def route_to_master() -> Dict[str, Any]:
    """
    Best-effort: determine which interface would be used to reach MASTER_ADDR.
    Uses `ip -o route get <master_ip>` and extracts `dev` and `src`.
    """
    master_addr = os.environ.get("MASTER_ADDR") or ""
    if not master_addr:
        return {"ok": False, "error": "MASTER_ADDR not set"}
    try:
        master_ip = socket.gethostbyname(master_addr)
    except Exception as e:
        return {"ok": False, "master_addr": master_addr, "error": f"resolve MASTER_ADDR failed: {e}"}

    try:
        r = subprocess.run(
            ["ip", "-o", "route", "get", master_ip],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
        if r.returncode != 0:
            return {
                "ok": False,
                "master_addr": master_addr,
                "master_ip": master_ip,
                "error": (r.stderr or r.stdout).strip() or f"ip route get failed (rc={r.returncode})",
            }
        s = r.stdout.strip()
        # Typical: "<ip> via <gw> dev <if> src <srcip> ..."
        m_dev = re.search(r"\bdev\s+(\S+)", s)
        m_src = re.search(r"\bsrc\s+(\d+\.\d+\.\d+\.\d+)", s)
        return {
            "ok": True,
            "master_addr": master_addr,
            "master_ip": master_ip,
            "dev": m_dev.group(1) if m_dev else None,
            "src_ip": m_src.group(1) if m_src else None,
            "raw": s,
        }
    except Exception as e:
        return {"ok": False, "master_addr": master_addr, "master_ip": master_ip, "error": str(e)}


def probe_network_env() -> Dict[str, Any]:
    # Presence-only snapshot (as in spec).
    route = route_to_master()
    ipv4 = list_ipv4_addrs()
    return {
        "MASTER_ADDR": _env_get("MASTER_ADDR"),
        "MASTER_PORT": _env_get("MASTER_PORT"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE", "1"),
        "RANK": _env_get("RANK"),
        "LOCAL_RANK": _env_get("LOCAL_RANK"),
        "NCCL_SOCKET_IFNAME": _env_get("NCCL_SOCKET_IFNAME"),
        "GLOO_SOCKET_IFNAME": _env_get("GLOO_SOCKET_IFNAME"),
        "NCCL_IB_HCA": _env_get("NCCL_IB_HCA"),
        "NCCL_IB_DISABLE": _env_get("NCCL_IB_DISABLE") or "0",
        "NCCL_DEBUG": _env_get("NCCL_DEBUG"),
        "NCCL_NET_GDR_LEVEL": _env_get("NCCL_NET_GDR_LEVEL"),
        "NCCL_IB_GID_INDEX": _env_get("NCCL_IB_GID_INDEX"),
        # Extra diagnostics for auto-selecting correct socket IFNAME
        "ROUTE_TO_MASTER": route,
        "IPV4_ADDRS": ipv4,
    }


def probe_network() -> NetworkProbe:
    intent = detect_distributed_intent()
    env = probe_network_env()
    return NetworkProbe(
        available_nics=list_nics(),
        ib_devices=list_ib_devices(),
        env=env,
        intent=intent,
    )
