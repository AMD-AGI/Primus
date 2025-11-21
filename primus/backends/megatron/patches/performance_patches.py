###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Performance Optimization Patches

General performance improvements for Megatron training.
"""

import os

from primus.core.patches import PatchContext, register_patch

# ============================================================================
# GPU Optimization Patches
# ============================================================================


@register_patch(
    "megatron.perf.enable_tf32",
    backend="megatron",
    phase="before_train",
    description="Enable TF32 for Ampere+ GPUs",
)
def _enable_tf32(ctx: PatchContext):
    """
    Enable TF32 for better performance on Ampere+ GPUs.

    Benefit: ~2x speedup for matmul operations on A100/H100
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability()
            if device_capability[0] >= 8:  # Ampere or newer
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("[Patch] Enabled TF32 for Ampere+ GPU")
    except Exception:
        pass


@register_patch(
    "megatron.perf.optimize_nccl",
    backend="megatron",
    phase="before_import_backend",
    description="Set optimal NCCL environment variables",
)
def _optimize_nccl(ctx: PatchContext):
    """
    Set optimal NCCL environment variables for better communication performance.

    Benefit: Improved collective communication performance
    """
    nccl_vars = {
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_DEBUG": "WARN",
    }

    for key, value in nccl_vars.items():
        if key not in os.environ:
            os.environ[key] = value

    print("[Patch] Optimized NCCL environment variables")


# ============================================================================
# Memory Optimization Patches
# ============================================================================


@register_patch(
    "megatron.perf.gradient_accumulation_memory",
    backend="megatron",
    phase="before_train",
    description="Optimize gradient accumulation for memory efficiency",
)
def _optimize_gradient_accumulation(ctx: PatchContext):
    """
    Optimize gradient accumulation to reduce memory usage.

    Benefit: Reduced memory footprint during gradient accumulation
    """
    args = ctx.extra.get("args")
    if args is None:
        return

    grad_accum_steps = getattr(args, "gradient_accumulation_steps", 1)
    if grad_accum_steps > 1:
        # Enable memory-efficient gradient accumulation
        if hasattr(args, "no_sync_on_grad_accum_boundary"):
            if not getattr(args, "no_sync_on_grad_accum_boundary", False):
                print(
                    f"[Patch] Enabling memory-efficient gradient accumulation " f"({grad_accum_steps} steps)"
                )
                setattr(args, "no_sync_on_grad_accum_boundary", True)


# ============================================================================
# Compilation Optimization Patches
# ============================================================================


@register_patch(
    "megatron.perf.torch_compile",
    backend="megatron",
    phase="before_train",
    description="Enable torch.compile for supported models",
    condition=lambda ctx: ctx.backend_version and ctx.backend_version >= "0.9.0",
)
def _enable_torch_compile(ctx: PatchContext):
    """
    Enable torch.compile for supported Megatron versions.

    Benefit: Up to 30% speedup with PyTorch 2.0+
    Note: Only enabled for Megatron 0.9.0+ which has compile support
    """
    args = ctx.extra.get("args")
    if args is None:
        return

    try:
        import torch

        if hasattr(torch, "compile") and hasattr(args, "use_torch_compile"):
            if not getattr(args, "use_torch_compile", False):
                print("[Patch] Enabling torch.compile for performance")
                setattr(args, "use_torch_compile", True)
    except Exception:
        pass
