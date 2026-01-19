###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Patch registry and initialization for Megatron-Bridge backend.

This module serves as the central registry for all Megatron-Bridge patches,
including setup patches, build_args patches, and runtime patches.

Patches are automatically registered when this module is imported.
"""

from primus.core.backend.backend_registry import BackendRegistry


# Register setup patches
def megatron_bridge_setup_patch():
    """
    Setup patch for Megatron-Bridge initialization.

    This runs during backend preparation and handles:
        - Environment variable setup
        - Import path configuration
        - Any necessary pre-initialization steps
    """
    from primus.modules.module_utils import log_rank_0

    log_rank_0("[Patch] Megatron-Bridge setup patch applied")

    # TODO: Add actual setup logic here
    # Examples:
    # - Set CUDA/ROCm environment variables
    # - Configure Python import paths for Megatron-Bridge
    # - Initialize any global state


# Register build_args patches
def megatron_bridge_build_args_patch(args):
    """
    Build args patch for Megatron-Bridge configuration.

    This runs after argument building and can modify/validate arguments.

    Args:
        args: SimpleNamespace containing Megatron-Bridge arguments

    Returns:
        Modified args namespace
    """
    from primus.modules.module_utils import log_rank_0

    log_rank_0("[Patch] Megatron-Bridge build_args patch applied")

    # TODO: Add argument validation/modification logic here
    # Examples:
    # - Validate parallelism settings
    # - Set derived configuration values
    # - Apply platform-specific adjustments

    return args


# Register all patches with the BackendRegistry
BackendRegistry.register_setup_hook("megatron_bridge", megatron_bridge_setup_patch)

# Note: build_args patches are registered through the patch management system
# For now, we'll handle them in the adapter if needed
