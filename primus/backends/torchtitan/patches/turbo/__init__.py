###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Primus-Turbo patches.

This subpackage contains TorchTitan patches that extend or integrate with
Primus-Turbo. Importing ``primus.backends.torchtitan.patches`` is sufficient
to register all patches in this subpackage.
"""

from primus.backends.torchtitan.patches.turbo import (  # noqa: F401
    async_tp_patches,
    attention_patches,
    deepseek_v3_classic_attention_patches,
    fp8_linear_patches,
    moe_grouped_mm_patches,
    mx_linear_patches,
)
