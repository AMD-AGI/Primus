###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Patch Collection

This module registers all Megatron-specific patches with the PatchRegistry.

Patches are organized by category:
    - env_patches: Environment variable configuration
    - mlflow_patches: MLflow logging integration
    - args_patches: Argument configuration and path setup
    - te_patches: Transformer Engine integration patches
    - flops_patches: FLOPs calculation and performance profiling

All patches are automatically registered on import via the @register_patch decorator.
"""

# Import all patch modules to trigger registration
# Patches are registered via @register_patch decorator in each module
from primus.backends.megatron.patches import (  # noqa: F401
    args_patches,
    env_patches,
    flops_patches,
    mlflow_patches,
    te_patches,
)
from primus.core.patches import run_patches


def apply_megatron_patches(
    *,
    backend_version: str = None,
    primus_version: str = None,
    model_name: str = None,
    phase: str = "before_train",
    extra: dict = None,
) -> int:
    """
    Apply all applicable Megatron patches for the given context.

    Args:
        backend_version: Megatron version (e.g., "0.8.0")
        primus_version: Primus version
        model_name: Model name (e.g., "llama3_70B", "deepseek_v3")
        phase: Execution phase (default: "before_train")
        extra: Additional context data (e.g., {"args": megatron_args})

    Returns:
        Number of patches applied

    Example:
        apply_megatron_patches(
            backend_version="0.8.0",
            model_name="deepseek_v3",
            phase="before_train",
            extra={"args": megatron_args},
        )
    """
    return run_patches(
        backend="megatron",
        phase=phase,
        backend_version=backend_version,
        primus_version=primus_version,
        model_name=model_name,
        extra=extra,
    )


__all__ = ["apply_megatron_patches"]
