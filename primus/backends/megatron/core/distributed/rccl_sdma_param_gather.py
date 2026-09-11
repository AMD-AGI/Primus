###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Direct RCCL copy-engine parameter AllGather support for Megatron.

Megatron's parameter buffer is allocated from PyTorch NCCL symmetric memory
and gathered in place through a dedicated zero-CTA ProcessGroupNCCL
communicator. Gradient ReduceScatter and all other collectives retain their
original process groups.
"""

from __future__ import annotations

import math
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

LARGE_SEGMENT_BYTES = 2 * 1024 * 1024

_DIRECT_POOLS: dict[tuple[str, int], torch.cuda.MemPool] = {}
_DIRECT_EAGER_PARAM_BUFFERS: dict[tuple[str, int, int], torch.Tensor] = {}
_SDMA_GROUP: dist.ProcessGroup | None = None
DIRECT_BUFFER_ATTR = "_primus_rccl_sdma_direct_buffer"


def recommended_eager_param_bytes(size_bytes: int) -> int:
    """Round a parameter buffer size up to the symmetric allocator granule."""
    if size_bytes <= 0:
        raise ValueError("parameter buffer size must be positive")
    return (
        (size_bytes + LARGE_SEGMENT_BYTES - 1)
        // LARGE_SEGMENT_BYTES
        * LARGE_SEGMENT_BYTES
    )


def get_sdma_process_group(
    original_group: dist.ProcessGroup,
) -> dist.ProcessGroup:
    """Create one full-rank communicator used only for zero-CTA AllGather."""
    global _SDMA_GROUP

    if original_group.size() != dist.get_world_size():
        raise RuntimeError(
            "RCCL-SDMA direct parameter gather requires the distributed-optimizer "
            "group to contain every rank"
        )
    if _SDMA_GROUP is None:
        options = dist.ProcessGroupNCCL.Options()
        cta_policy = int(os.getenv("MEGATRON_RCCL_SDMA_CTA_POLICY", "2"))
        if cta_policy == 2:
            options.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
        elif cta_policy == 0:
            options.config.cta_policy = getattr(
                dist.ProcessGroupNCCL,
                "NCCL_CTA_POLICY_DEFAULT",
                0,
            )
        else:
            raise ValueError(
                f"MEGATRON_RCCL_SDMA_CTA_POLICY must be 0 or 2, got {cta_policy}"
            )
        options.config.split_share = 0
        _SDMA_GROUP = dist.new_group(
            ranks=list(range(dist.get_world_size())),
            backend="nccl",
            pg_options=options,
            group_desc=f"MEGATRON_RCCL_SDMA_PARAM_GATHER_POLICY_{cta_policy}",
        )
        if dist.get_rank() == 0:
            print(
                "[RCCL-SDMA:Megatron] created dedicated zero-CTA group "
                f"name={_SDMA_GROUP.group_name}",
                flush=True,
            )
    return _SDMA_GROUP


def prepare_direct_param_buffer_pool(
    original_group: dist.ProcessGroup,
    device: torch.device,
) -> tuple[dist.ProcessGroup, torch.cuda.MemPool]:
    """Enable and return the symmetric pool used by direct parameter buffers."""
    group = get_sdma_process_group(original_group)
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (group.group_name, device_index)
    pool = _DIRECT_POOLS.get(key)
    if pool is None:
        symm_mem.set_backend("NCCL")
        symm_mem.enable_symm_mem_for_group(group.group_name)
        dist.barrier(group=group)
        pool = symm_mem.get_mem_pool(device)
        _DIRECT_POOLS[key] = pool
    return group, pool


def reserve_direct_param_buffer(
    group: dist.ProcessGroup | None,
    pool: torch.cuda.MemPool,
    device: torch.device,
    size_bytes: int,
) -> None:
    """Allocate a direct parameter buffer before model allocations fragment HBM."""
    if size_bytes <= 0:
        raise ValueError("eager direct parameter buffer size must be positive")
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    group_name = group.group_name if group is not None else ""
    key = (group_name, device_index, size_bytes)
    if key in _DIRECT_EAGER_PARAM_BUFFERS:
        return
    with torch.cuda.use_mem_pool(pool):
        storage = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    _DIRECT_EAGER_PARAM_BUFFERS[key] = storage
    rank = group.rank() if group is not None else int(os.getenv("RANK", "0"))
    if rank == 0:
        print(
            "[RCCL-SDMA:Megatron] eagerly reserved direct parameter buffer "
            f"bytes={size_bytes} group={group_name or '<pending>'}",
            flush=True,
        )


def take_direct_param_buffer(
    group: dist.ProcessGroup,
    device: torch.device,
    shape,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Transfer a size-compatible eager reservation to Megatron's param_data."""
    numel = int(shape) if isinstance(shape, int) else math.prod(shape)
    size_bytes = numel * torch.empty((), dtype=dtype).element_size()
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    matching_keys = [
        key
        for key in _DIRECT_EAGER_PARAM_BUFFERS
        if key[0] in ("", group.group_name)
        and key[1] == device_index
        and key[2] >= size_bytes
        and key[2] - size_bytes < LARGE_SEGMENT_BYTES
    ]
    if not matching_keys:
        if group.rank() == 0:
            recommended_bytes = recommended_eager_param_bytes(size_bytes)
            print(
                "[RCCL-SDMA:Megatron] no eager direct parameter buffer match "
                f"requested_bytes={size_bytes} "
                f"recommended_eager_bytes={recommended_bytes} "
                f"reserved={list(_DIRECT_EAGER_PARAM_BUFFERS)}; "
                "if direct allocation fails, rerun with "
                f"MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES={recommended_bytes}",
                flush=True,
            )
        return None
    key = min(matching_keys, key=lambda candidate: candidate[2])
    storage = _DIRECT_EAGER_PARAM_BUFFERS.pop(key)
    tensor = storage[:size_bytes].view(dtype).view(shape)
    tensor.zero_()
    if group.rank() == 0:
        print(
            "[RCCL-SDMA:Megatron] consumed eager direct parameter buffer "
            f"reserved_bytes={storage.nbytes} requested_bytes={size_bytes}",
            flush=True,
        )
    return tensor


def rendezvous_direct_param_buffer(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
) -> object:
    """Register a pool-backed Megatron parameter buffer for direct CE gather."""
    symmetric_memory = symm_mem.rendezvous(tensor, group=group.group_name)
    setattr(tensor, DIRECT_BUFFER_ATTR, True)
    if group.rank() == 0 and os.getenv("MEGATRON_RCCL_SDMA_LOG", "0") == "1":
        print(
            "[RCCL-SDMA:Megatron] rendezvoused direct parameter buffer "
            f"bytes={tensor.nbytes} group={group.group_name}",
            flush=True,
        )
    return symmetric_memory


def mark_direct_param_buffer(tensor: torch.Tensor) -> None:
    """Mark a view whose storage was rendezvoused for direct gather."""
    setattr(tensor, DIRECT_BUFFER_ATTR, True)


def is_direct_param_buffer(tensor: torch.Tensor) -> bool:
    """Return whether a tensor view belongs to a direct symmetric buffer."""
    return bool(getattr(tensor, DIRECT_BUFFER_ATTR, False))


def reset_runtime_state_for_tests() -> None:
    """Clear process-global caches used by isolated unit tests."""
    global _SDMA_GROUP
    _DIRECT_POOLS.clear()
    _DIRECT_EAGER_PARAM_BUFFERS.clear()
    _SDMA_GROUP = None
