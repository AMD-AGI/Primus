###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Environment Variable Patches

Sets environment variables for optimal Megatron performance and compatibility.
"""

import os

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_kv_rank_0

# ============================================================================
# CUDA Device Configuration
# ============================================================================


@register_patch(
    "megatron.env.cuda_device_max_connections",
    backend="megatron",
    phase="setup",
    description="Set CUDA_DEVICE_MAX_CONNECTIONS based on FSDP configuration",
)
def set_cuda_device_max_connections(ctx: PatchContext):
    """
    Set CUDA_DEVICE_MAX_CONNECTIONS environment variable.

    This controls the number of CUDA streams for device-to-device communication.

    Strategy:
        - FSDP (Fully Sharded Data Parallel): Use 8 connections for better overlap
        - Non-FSDP (standard DDP/TP/PP): Use 1 connection to avoid contention

    The value is determined by checking the config for FSDP usage.
    """
    # Get config from context
    module_config = ctx.extra.get("module_config", {})

    # Determine CUDA_DEVICE_MAX_CONNECTIONS based on FSDP usage
    use_fsdp = (
        getattr(module_config.params, "use_torch_fsdp2", False)
        or getattr(module_config.params, "use_custom_fsdp", False)
        or getattr(module_config.params, "use_megatron_fsdp", False)
    )

    if use_fsdp:
        cuda_connections = "8"
        log_kv_rank_0(f"[Patch:megatron.env.cuda_device_max_connections]   -use_fsdp", f"True")
    else:
        cuda_connections = "1"
        log_kv_rank_0(f"[Patch:megatron.env.cuda_device_max_connections]   -use_fsdp", f"False")

    # Set environment variable
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = cuda_connections
    log_kv_rank_0(
        "[Patch:megatron.env.cuda_device_max_connections]   -env[CUDA_DEVICE_MAX_CONNECTIONS]",
        f"{cuda_connections}",
    )
