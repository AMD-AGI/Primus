###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.hummingbirdxt.hummingbirdxt_adapter import HummingbirdXTAdapter
from primus.core.backend.backend_registry import BackendRegistry

# Register adapter
BackendRegistry.register_adapter("hummingbirdxt", HummingbirdXTAdapter)
