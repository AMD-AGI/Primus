###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
SpecForge Backend Registration

Register the SpecForge backend adapter so that experiments declaring
``framework: specforge`` resolve through ``BackendRegistry``.
"""

from primus.backends.specforge.specforge_adapter import SpecForgeAdapter
from primus.core.backend.backend_registry import BackendRegistry

BackendRegistry.register_adapter("specforge", SpecForgeAdapter)
