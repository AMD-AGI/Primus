###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Patch System

A lightweight, phase-aware patching system for backend frameworks.

Design Goals:
    1. Handle version compatibility issues across Megatron/TorchTitan/etc
    2. Apply hotfixes without modifying upstream framework code
    3. Support model-specific patches (DeepSeek, Llama, Mixtral, etc)
    4. Phase-based execution (before_import, after_build_args, before_train, etc)
    5. Environment variable control (PRIMUS_PATCHES)

Module Structure:
    - context: PatchContext and phase management
    - patch: FunctionPatch implementation
    - patch_registry: PatchRegistry and @register_patch decorator
    - patch_runner: Patch execution logic (run_patches)
    - utils: Utility functions (version_matches)

Usage:
    # 1. Define a patch
    @register_patch(
        "megatron.deepseek_v3.fix_moe",
        backend="megatron",
        phase="before_train",
        description="Fix MoE load balancing for DeepSeek V3",
        condition=lambda ctx: ctx.model_name == "deepseek_v3",
    )
    def fix_deepseek_moe(ctx: PatchContext):
        args = ctx.extra.get("args")
        if args and hasattr(args, "moe_aux_loss_coeff"):
            args.moe_aux_loss_coeff = 0.001

    # 2. Execute patches
    run_patches(
        backend="megatron",
        phase="before_train",  # Can also use legacy names like "before_import_backend"
        backend_version="0.8.0",
        model_name="deepseek_v3",
        extra={"args": megatron_args},
    )
"""

# Core components
from primus.core.patches.context import PatchContext, get_args, get_param
from primus.core.patches.patch import FunctionPatch
from primus.core.patches.patch_registry import PatchRegistry, register_patch
from primus.core.patches.patch_runner import run_patches
from primus.core.patches.utils import version_matches

__all__ = [
    "PatchContext",
    "get_args",
    "get_param",
    "FunctionPatch",
    "PatchRegistry",
    "register_patch",
    "run_patches",
    "version_matches",
]
