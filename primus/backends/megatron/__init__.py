###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.megatron.megatron_adapter import MegatronAdapter
from primus.core.backend.backend_registry import BackendRegistry

BackendRegistry.register_adapter("megatron", MegatronAdapter)

try:
    from primus.backends.megatron.moe_umco_patch import apply_umco_patches

    apply_umco_patches()
except Exception:
    # Keep backend import resilient when Megatron packages are not installed.
    pass
