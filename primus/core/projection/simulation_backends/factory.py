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
    compute_efficiency: float = 0.51,
    memory_efficiency: float = 0.85,
    gpu_clock_mhz: Optional[int] = None,
) -> SDPASimulationBackend:
    """
    Create and return the SDPA simulation backend.

    The default backend is an analytical model of the FAv3 (Flash Attention v3)
    kernels, with tile sizes, wavefront counts, and efficiency factors
    derived from the kernel configurations.

    Args:
        gpu_arch: GPU architecture override (e.g. "mi300x", "mi355x").
        compute_efficiency: Fraction of peak compute achieved (0-1).
            Defaults to 0.51 — calibrated against measured FAv3 traces on
            MI300X (B=3, H_Q=64, S=8192, D=128, H_KV=8, GQA, causal, BF16).
            The lower-than-theoretical efficiency accounts for GQA head
            broadcasting overhead, LDS bank conflicts, barrier synchronisation,
            and register pressure.
        memory_efficiency: Fraction of peak HBM bandwidth achieved (0-1).
            Defaults to 0.85 — FAv3 streaming pattern typically achieves 0.80-0.90.
        gpu_clock_mhz: Override the GPU compute clock frequency in MHz.

    Returns:
        An SDPASimulationBackend instance.
    """
    from primus.core.projection.simulation_backends.sdpa_simulator import (
        SDPASimulator,
    )

    is_rank_0 = int(os.getenv("RANK", "0")) == 0
    if is_rank_0:
        print(
            "[Primus:Simulation] Using SDPA backend: sdpa_simulator (FAv3 analytical model)"
        )

    return SDPASimulator(
        gpu_arch=gpu_arch,
        compute_efficiency=compute_efficiency,
        memory_efficiency=memory_efficiency,
        gpu_clock_mhz=gpu_clock_mhz,
    )
