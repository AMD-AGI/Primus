###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Primus-owned diffusion backend components for t2i/t2v training."""

from primus.backends.diffusion.diffusion_adapter import DiffusionAdapter
from primus.core.backend.backend_registry import BackendRegistry

# Safetensors auto-conversion is patched in SDXLTextEncoder, immediately before
# the CLIP load that needs it. Doing it here would change Transformers behavior
# for every diffusion run, including Wan and FLUX.
BackendRegistry.register_adapter("diffusion", DiffusionAdapter)
