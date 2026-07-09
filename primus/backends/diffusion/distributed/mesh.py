###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Device mesh creation and distributed setup.

Supports:
  - Pure FSDP2 (1D mesh: dp_shard)
  - FSDP2 + HSDP (2D mesh: dp_replicate × dp_shard)
  - FSDP2 + Ulysses SP (mesh: dp_shard × ulysses, flattened into dp_shard_sp)
  - FSDP2 + HSDP + Ulysses SP (mesh: dp_replicate × dp_shard × ulysses)

SP ranks are included in the FSDP sharding mesh so that parameters are
sharded across both DP and SP ranks — this reduces per-rank memory.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from primus.backends.diffusion.utils.log import logger


def _ensure_process_group(*, backend: str) -> None:
    """Best-effort init_process_group (lazy, avoids single-GPU hangs)."""
    if dist.is_initialized():
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12345")
    dist.init_process_group(backend, timeout=timedelta(minutes=60))


def setup_distributed() -> tuple[int, int, int]:
    """
    Initialize distributed process group.

    Returns: (rank, world_size, local_rank)
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist.is_initialized():
        _ensure_process_group(backend="nccl")

    return rank, world_size, local_rank


def create_device_mesh(
    world_size: int,
    sp_size: int = 1,
    dp_replicate: int = 1,
) -> Optional[DeviceMesh]:
    """
    Create a DeviceMesh for FSDP2 + optional Ulysses Sequence Parallel.

    Mesh layout (innermost → outermost):
        [dp_replicate?] × [dp_shard] × [ulysses?]

    Flattened sub-meshes created automatically:
        - "dp_shard_sp":  dp_shard × ulysses  (used for FSDP2 fully_shard)
        - "dp":           dp_replicate × dp_shard  (used for DistributedSampler)

    Args:
        world_size: total number of ranks
        sp_size:    Ulysses sequence parallel size (must divide world_size)
        dp_replicate: HSDP replicate dimension (1 = no HSDP)

    Returns:
        DeviceMesh, or None if world_size <= 1
    """
    if world_size <= 1:
        return None

    _ensure_process_group(backend="nccl")

    dp_shard = world_size // (sp_size * dp_replicate)
    if dp_shard * sp_size * dp_replicate != world_size:
        raise ValueError(
            f"world_size={world_size} is not divisible by " f"sp_size={sp_size} * dp_replicate={dp_replicate}"
        )

    # Build mesh dimensions
    dims = []
    names = []
    if dp_replicate > 1:
        dims.append(dp_replicate)
        names.append("dp_replicate")
    dims.append(dp_shard)
    names.append("dp_shard")
    if sp_size > 1:
        dims.append(sp_size)
        names.append("ulysses")

    mesh = init_device_mesh("cuda", tuple(dims), mesh_dim_names=tuple(names))

    # Flatten composite sub-meshes for convenient access
    if sp_size > 1:
        mesh["dp_shard", "ulysses"]._flatten("dp_shard_sp")

    if dp_replicate > 1:
        mesh["dp_replicate", "dp_shard"]._flatten("dp")

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        logger.info(
            f"DeviceMesh created: dims={dict(zip(names, dims))}, "
            f"sp_size={sp_size}, dp_replicate={dp_replicate}"
        )

    return mesh
