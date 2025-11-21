###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.megatron.adapters.megatron_adapter import MegatronAdapter
from primus.core.backend.backend_registry import BackendRegistry

BackendRegistry.register_path_name("megatron", "Megatron-LM")

# Register adapter
BackendRegistry.register_adapter("megatron", MegatronAdapter)
