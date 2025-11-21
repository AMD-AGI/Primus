###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Version Compatibility Patches

Handles breaking changes between Megatron versions.
"""

import os

from primus.core.patches import PatchContext, register_patch, version_matches

# ============================================================================
# Megatron 0.7.x Patches
# ============================================================================


@register_patch(
    "megatron.v07.parse_args_fix",
    backend="megatron",
    phase="before_build_args",
    description="Fix argument parsing issue in Megatron 0.7.x",
    condition=lambda ctx: ctx.backend_version and version_matches(ctx.backend_version, "0.7.*"),
)
def _fix_megatron_v07_parse_args(ctx: PatchContext):
    """
    Fix for Megatron 0.7.x argument parsing issue.

    Issue: --no-async-tensor-model-parallel-allreduce not handled correctly
    Fix: Remove from sys.argv and set via environment variable
    """
    import sys

    if "--no-async-tensor-model-parallel-allreduce" in sys.argv:
        sys.argv.remove("--no-async-tensor-model-parallel-allreduce")
        os.environ["MEGATRON_ASYNC_TP_ALLREDUCE"] = "0"
        print("[Patch] Fixed --no-async-tensor-model-parallel-allreduce for Megatron 0.7.x")


# ============================================================================
# Megatron 0.8.x+ Patches
# ============================================================================


@register_patch(
    "megatron.v08.init_order_fix",
    backend="megatron",
    phase="before_train",
    description="Fix initialization order in Megatron 0.8.0+",
    condition=lambda ctx: ctx.backend_version
    and (version_matches(ctx.backend_version, "0.8.*") or version_matches(ctx.backend_version, "0.9.*")),
)
def _fix_megatron_v08_init_order(ctx: PatchContext):
    """
    Fix for Megatron 0.8.0+ initialization order.

    Issue: Distributed backend must be initialized before model parallel
    Fix: Ensure torch.distributed.init_process_group is called first
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        backend = os.getenv("DISTRIBUTED_BACKEND", "nccl")
        dist.init_process_group(backend=backend)
        print(f"[Patch] Initialized distributed backend ({backend}) for Megatron 0.8.0+")


# ============================================================================
# General Compatibility Patches
# ============================================================================


@register_patch(
    "megatron.cuda_device_max_connections",
    backend="megatron",
    phase="before_import_backend",
    description="Set CUDA_DEVICE_MAX_CONNECTIONS for better performance",
)
def _set_cuda_device_max_connections(ctx: PatchContext):
    """Set CUDA_DEVICE_MAX_CONNECTIONS if not already set."""
    if "CUDA_DEVICE_MAX_CONNECTIONS" not in os.environ:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        print("[Patch] Set CUDA_DEVICE_MAX_CONNECTIONS=1")
