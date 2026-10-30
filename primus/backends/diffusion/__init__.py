###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Primus-owned diffusion backend components for t2i/t2v training."""

from primus.backends.diffusion.diffusion_adapter import DiffusionAdapter
from primus.backends.diffusion.utils.hf_load import disable_safetensors_auto_conversion
from primus.core.backend.backend_registry import BackendRegistry

disable_safetensors_auto_conversion()

BackendRegistry.register_adapter("diffusion", DiffusionAdapter)
