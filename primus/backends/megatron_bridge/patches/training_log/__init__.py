###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron-Bridge training_log patches package.
"""

from primus.backends.megatron_bridge.patches.training_log import (  # noqa: F401
    bridge_training_log_patches,
)

__all__ = ["bridge_training_log_patches"]
