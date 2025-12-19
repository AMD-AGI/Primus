###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Turbo Patches Module

This module contains all PrimusTurbo backend related patches for Megatron.
Each patch is organized in its own file for better maintainability.

Patches included:
  - te_spec_provider_patches: Replace TESpecProvider with PrimusTurboSpecProvider
  - gpt_output_layer_patches: Replace GPT ColumnParallelLinear with PrimusTurbo implementation
  - moe_dispatcher_patches: Replace MoE token dispatcher with PrimusTurbo DeepEP implementation
  - rms_norm_patches: Replace RMSNorm with PrimusTurbo implementation
"""

# Import all patch modules to trigger registration
# NOTE: These imports are intentionally unused; they register patches with
# the core patch registry when this package is imported.
from primus.backends.megatron.patches.turbo import (
    gpt_output_layer_patches as _gpt_output_layer_patches,
)
from primus.backends.megatron.patches.turbo import (
    moe_dispatcher_patches as _moe_dispatcher_patches,
)
from primus.backends.megatron.patches.turbo import rms_norm_patches as _rms_norm_patches
from primus.backends.megatron.patches.turbo import (
    te_spec_provider_patches as _te_spec_provider_patches,
)

__all__ = [
    "_te_spec_provider_patches",
    "_gpt_output_layer_patches",
    "_moe_dispatcher_patches",
    "_rms_norm_patches",
]
