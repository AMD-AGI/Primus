###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron-Bridge Gemma 4 patches package.
"""

from primus.backends.megatron_bridge.patches.gemma4 import (  # noqa: F401
    gemma4_bridge_patches,
    gemma4_config_overrides,
    gemma4_cpu_offload,
    gemma4_diagnostics,
    gemma4_fused_norms,
    gemma4_local_spec,
)

__all__ = [
    "gemma4_bridge_patches",
    "gemma4_config_overrides",
    "gemma4_cpu_offload",
    "gemma4_diagnostics",
    "gemma4_fused_norms",
    "gemma4_local_spec",
]
