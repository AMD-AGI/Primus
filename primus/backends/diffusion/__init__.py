###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Primus-owned diffusion backend components for t2i/t2v training."""

from primus.backends.diffusion.diffusion_adapter import DiffusionAdapter
from primus.core.backend.backend_registry import BackendRegistry

BackendRegistry.register_adapter("diffusion", DiffusionAdapter)
