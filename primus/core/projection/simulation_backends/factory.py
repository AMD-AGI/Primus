###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Factory functions for creating simulation backends.

Backend selection for GEMM:
  1. If ``PRIMUS_GEMM_BACKEND`` is set, use that backend explicitly.
  2. Otherwise, use **origami** (the default, open-source backend).

SDPA always uses the built-in analytical simulator.
"""

import os
from typing import Optional

from primus.core.projection.simulation_backends.base import (
    GEMMSimulationBackend,
    SDPASimulationBackend,
)


def get_gemm_simulation_backend(
    backend_name: Optional[str] = None,
    gpu_arch: Optional[str] = None,
    gpu_clock_mhz: Optional[int] = None,
) -> GEMMSimulationBackend:
    """
    Create and return the GEMM simulation backend (origami).

    Args:
        backend_name: Explicit backend name. Currently only "origami" is supported.
                      If None, defaults to origami.
        gpu_arch: GPU architecture override (e.g. "gfx942", "mi300x", "mi325x").
        gpu_clock_mhz: Override the GPU compute clock frequency in MHz.

    Returns:
        A GEMMSimulationBackend instance.

    Raises:
        RuntimeError: If the backend is not available.
    """
    name = backend_name or os.getenv("PRIMUS_GEMM_BACKEND", None)

    if name is not None:
        name = name.lower().strip()

    is_rank_0 = int(os.getenv("RANK", "0")) == 0

    if name is not None and name != "origami":
        raise ValueError(
            f"Unknown GEMM simulation backend: '{name}'. "
            f"Supported backend: 'origami'"
        )

    from primus.core.projection.simulation_backends.origami_backend import (
        OrigamiGEMMBackend,
    )

    backend = OrigamiGEMMBackend(gpu_arch=gpu_arch, gpu_clock_mhz=gpu_clock_mhz)
    if not backend.is_available():
        raise RuntimeError(
            "Origami GEMM simulation backend is not available.\n"
            "Install it with: pip install origami"
        )

    if is_rank_0:
        print("[Primus:Simulation] Using GEMM backend: origami")
    return backend


def get_sdpa_simulation_backend(
    gpu_arch: Optional[str] = None,
    gpu_clock_mhz: Optional[int] = None,
) -> SDPASimulationBackend:
    """
    Create and return the SDPA simulation backend.

    Uses the Origami 1-CU tile-level model of the FAv3 (Flash Attention v3)
    kernels.  Origami must be installed.

    Args:
        gpu_arch: GPU architecture override (e.g. "mi300x", "mi355x").
        gpu_clock_mhz: Override the GPU compute clock frequency in MHz.

    Returns:
        An SDPASimulationBackend instance.

    Raises:
        RuntimeError: If the Origami backend is not available.
    """
    from primus.core.projection.simulation_backends.sdpa_simulator import (
        SDPASimulator,
    )

    is_rank_0 = int(os.getenv("RANK", "0")) == 0
    if is_rank_0:
        print(
            "[Primus:Simulation] Using SDPA backend: sdpa_simulator (FAv3 Origami 1-CU)"
        )

    return SDPASimulator(
        gpu_arch=gpu_arch,
        gpu_clock_mhz=gpu_clock_mhz,
    )
