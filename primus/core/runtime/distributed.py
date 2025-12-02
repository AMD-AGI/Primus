###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Distributed training environment initialization.

This module handles one-time initialization of distributed training
environment variables and configuration.
"""

import os

from primus.core.utils import checker
from primus.core.utils.global_vars import get_target_platform

from .context import RuntimeContext


def init_distributed_env(
    rank: int = None,
    world_size: int = None,
    master_addr: str = None,
    master_port: int = None,
) -> RuntimeContext:
    """
    Initialize distributed training environment.

    This function should be called once at the start of training to set up
    the distributed environment. It:
        1. Auto-detects distributed parameters from environment variables
        2. Validates the parameters
        3. Sets environment variables for downstream frameworks
        4. Stores parameters in RuntimeContext

    Args:
        rank: Global rank (auto-detected from RANK if None)
        world_size: Total number of processes (auto-detected from WORLD_SIZE if None)
        master_addr: Master node address (auto-detected from MASTER_ADDR if None)
        master_port: Master node port (auto-detected from MASTER_PORT if None)

    Returns:
        RuntimeContext instance with initialized distributed parameters
    """
    context = RuntimeContext.get_instance()

    # Skip if already initialized
    if context.distributed_initialized:
        print("[Primus:Runtime] Distributed environment already initialized")
        return context

    # Auto-detect from environment if not provided
    rank = rank if rank is not None else int(os.environ.get("RANK", 0))
    world_size = world_size if world_size is not None else int(os.environ.get("WORLD_SIZE", 1))
    master_addr = master_addr if master_addr is not None else os.environ.get("MASTER_ADDR", "localhost")
    master_port = master_port if master_port is not None else int(os.environ.get("MASTER_PORT", 29500))

    # Get platform information
    platform = get_target_platform()
    context.set_platform(platform)

    # Calculate local rank
    num_gpus_per_node = platform.get_gpus_per_node()
    local_rank = rank % num_gpus_per_node

    # Validate distributed parameters
    if rank > 0:
        checker.check_true(master_addr is not None, msg=f"Must provide master addr for workers with rank > 0")

    # For rank 0, use actual platform address
    if rank == 0 and master_addr is None:
        master_addr = platform.get_addr()
    else:
        master_addr = master_addr

    # Set environment variables for downstream frameworks (Megatron, PyTorch, etc.)
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)

    # Store in context
    context.set_distributed_params(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
    )

    print(f"[Primus:Runtime] Distributed environment initialized: {context}")

    return context


def get_distributed_info() -> dict:
    """
    Get current distributed training information.

    Returns:
        Dictionary with rank, world_size, local_rank, master_addr, master_port
    """
    context = RuntimeContext.get_instance()
    return {
        "rank": context.rank,
        "world_size": context.world_size,
        "local_rank": context.local_rank,
        "master_addr": context.master_addr,
        "master_port": context.master_port,
    }
