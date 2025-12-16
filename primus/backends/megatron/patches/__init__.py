###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Patch Collection

This module defines the public entrypoint for applying Megatron-specific patches.
"""

# Import patch modules for side-effect registration via @register_patch.
# NOTE: These imports are intentionally unused; they register patches with
# the core patch registry when this package is imported.
from primus.backends.megatron.patches import args_patches as _args_patches  # noqa: F401
from primus.backends.megatron.patches import checkpoint_patches as _checkpoint_patches
from primus.backends.megatron.patches import env_patches as _env_patches
from primus.backends.megatron.patches import flops_patches as _flops_patches
from primus.backends.megatron.patches import fsdp_patches as _fsdp_patches
from primus.backends.megatron.patches import (
    training_log_patches as _training_log_patches,
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
