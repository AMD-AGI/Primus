###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText Backend Registration

Register MaxText backend adapter and trainer classes.
"""

from primus.backends.maxtext.maxtext_adapter import MaxTextAdapter
from primus.core.backend.backend_registry import BackendRegistry

# Register MaxText backend adapter
BackendRegistry.register_adapter("maxtext", MaxTextAdapter)
