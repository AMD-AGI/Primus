###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import os
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


def probe_network_env() -> Dict[str, Any]:
    # Presence-only snapshot (as in spec).
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
