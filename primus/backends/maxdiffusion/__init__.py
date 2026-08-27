###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
MaxDiffusion Backend Registration.

Registers the MaxDiffusion (JAX) backend adapter, mirroring the MaxText
backend. MaxDiffusion is Google's JAX diffusion library
(https://github.com/AI-Hypercomputer/maxdiffusion); this backend runs its
WAN video and FLUX text-to-image trainers through Primus so that JAX
diffusion models follow the same launch/discovery pattern as MaxText.

Note: this is distinct from the in-tree PyTorch ``diffusion`` backend
(``primus/backends/diffusion``); ``maxdiffusion`` is JAX and is launched
without torchrun (like ``maxtext``).
"""

from primus.backends.maxdiffusion.maxdiffusion_adapter import MaxDiffusionAdapter
from primus.core.backend.backend_registry import BackendRegistry

BackendRegistry.register_adapter("maxdiffusion", MaxDiffusionAdapter)
