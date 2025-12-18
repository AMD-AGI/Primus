###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Transformer Engine Patches Module

This module contains all Transformer Engine related patches for Megatron.
Each patch is organized in its own file for better maintainability.
"""

# Import all patch modules to trigger registration
# NOTE: These imports are intentionally unused; they register patches with
# the core patch registry when this package is imported.
from primus.backends.megatron.patches.te_patches import (
    delayed_scaling_patches as _delayed_scaling_patches,
)
from primus.backends.megatron.patches.te_patches import (
    layernorm_linear_fp8_cache_patches as _layernorm_linear_fp8_cache_patches,
)
from primus.backends.megatron.patches.te_patches import (
    linear_fp8_cache_patches as _linear_fp8_cache_patches,
)
from primus.backends.megatron.patches.te_patches import (
    tp_overlap_patches as _tp_overlap_patches,
)

__all__ = [
    "_delayed_scaling_patches",
    "_layernorm_linear_fp8_cache_patches",
    "_linear_fp8_cache_patches",
    "_tp_overlap_patches",
]
