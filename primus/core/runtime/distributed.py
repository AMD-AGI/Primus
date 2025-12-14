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

from primus.core.utils import constant_vars
from primus.core.utils.env import get_torchrun_env
from primus.core.utils.global_vars import get_target_platform

from .context import RuntimeContext


def init_distributed_env() -> RuntimeContext:
    context = RuntimeContext.get_instance()

    # Skip if already initialized
    if context.distributed_initialized:
        print("[Primus:Runtime] Distributed environment already initialized")
        return context

    torchrun_env = get_torchrun_env()
    print(f"[Primus:Runtime] Distributed environment initialized: {torchrun_env}")

    # Store in context
    context.set_distributed_params(
        rank=torchrun_env["rank"],
        world_size=torchrun_env["world_size"],
        local_rank=torchrun_env["local_rank"],
        master_addr=torchrun_env["master_addr"],
        master_port=torchrun_env["master_port"],
        local_world_size=torchrun_env["local_world_size"],
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
