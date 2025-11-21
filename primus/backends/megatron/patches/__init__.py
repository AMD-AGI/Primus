###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Patch Collection

This module registers all Megatron-specific patches with the PatchRegistry.

Patches are organized by category:
    - Version compatibility patches
    - Model-specific patches (DeepSeek, Llama, etc)
    - Bug fixes
    - Performance optimizations
"""

# Import all patch modules to trigger registration
from primus.backends.megatron.patches import (
    compatibility_patches,
    deepseek_patches,
    llama_patches,
    performance_patches,
)
from primus.core.patches import PatchContext, PatchRegistry


def register_all_patches():
    """
    Register all Megatron patches.

    This is called automatically when the module is imported,
    but can also be called explicitly if needed.
    """
    # Patches are registered via decorators in their respective modules


def apply_megatron_patches(context: PatchContext):
    """
    Apply all applicable Megatron patches for the given context.

    Args:
        context: Runtime context with framework version, model info, etc

    Returns:
        Tuple of (applied_count, failed_count)
    """
    return PatchRegistry.apply_patches(context)


# Auto-register patches on import
register_all_patches()
