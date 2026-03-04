###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron adapter layer for backend-agnostic MoE core."""

from .config_bridge import build_dispatch_runtime_config, build_router_runtime_config
from .patching import patch_megatron_moe_dispatcher, patch_megatron_topk_router

__all__ = [
    "build_dispatch_runtime_config",
    "build_router_runtime_config",
    "patch_megatron_moe_dispatcher",
    "patch_megatron_topk_router",
]
