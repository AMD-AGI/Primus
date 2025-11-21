###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Version Compatibility Patches

Handles breaking changes between Megatron versions.
"""

from primus.core.patches import FunctionPatch, PatchPriority, PatchRegistry


# Example: Fix for Megatron 0.7.0 argument parsing issue
def patched_parse_args_v07(original_func, *args, **kwargs):
    """
    Wrapper for Megatron 0.7.0 parse_args to fix missing argument issue.

    Issue: Megatron 0.7.0 doesn't handle --no-async-tensor-model-parallel-allreduce correctly
    Fix: Add default value before calling original function
    """
    import sys

    # Check if problematic arg is present
    if "--no-async-tensor-model-parallel-allreduce" in sys.argv:
        sys.argv.remove("--no-async-tensor-model-parallel-allreduce")
        # Set via environment variable instead
        import os

        os.environ["MEGATRON_ASYNC_TP_ALLREDUCE"] = "0"

    return original_func(*args, **kwargs)


# Register the patch
PatchRegistry.register(
    FunctionPatch(
        name="megatron_v07_parse_args_fix",
        description="Fix argument parsing issue in Megatron 0.7.0",
        target_module="megatron.training.arguments",
        target_function="parse_args",
        patch_function=patched_parse_args_v07,
        wrap=True,
        framework="megatron",
        version_range=">=0.7.0,<0.8.0",
        priority=PatchPriority.HIGH,
    )
)


# Example: Fix for Megatron 0.8.0+ initialization order
def patched_initialize_megatron_v08(original_func, *args, **kwargs):
    """
    Wrapper for Megatron 0.8.0+ initialize_megatron to fix initialization order.

    Issue: In 0.8.0+, distributed backend must be initialized before model parallel
    Fix: Ensure torch.distributed.init_process_group is called first
    """
    import torch.distributed as dist

    # Check if already initialized
    if not dist.is_initialized():
        # Initialize with proper backend
        import os

        backend = os.getenv("DISTRIBUTED_BACKEND", "nccl")
        dist.init_process_group(backend=backend)

    return original_func(*args, **kwargs)


PatchRegistry.register(
    FunctionPatch(
        name="megatron_v08_init_order_fix",
        description="Fix initialization order in Megatron 0.8.0+",
        target_module="megatron.training.initialize",
        target_function="initialize_megatron",
        patch_function=patched_initialize_megatron_v08,
        wrap=True,
        framework="megatron",
        version_range=">=0.8.0",
        priority=PatchPriority.CRITICAL,
    )
)


# Example: Backward compatibility shim for renamed functions
def get_args_shim():
    """Shim for get_args that works across versions."""
    try:
        from megatron.training import get_args

        return get_args()
    except ImportError:
        # Fallback for older versions
        from megatron import get_args

        return get_args()


PatchRegistry.register(
    FunctionPatch(
        name="megatron_get_args_compat",
        description="Compatibility shim for get_args across versions",
        target_module="megatron",
        target_function="get_args",
        patch_function=get_args_shim,
        wrap=False,
        framework="megatron",
        version_range="<0.7.0",
        priority=PatchPriority.HIGH,
    )
)
