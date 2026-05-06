###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.torchtitan.torchtitan_adapter import TorchTitanAdapter
from primus.core.backend.backend_registry import BackendRegistry

# Register adapter
BackendRegistry.register_adapter("torchtitan", TorchTitanAdapter)
