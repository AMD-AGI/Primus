###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Factory functions for creating simulation backends.

Backend selection for GEMM is **registry-driven** — no backend name is
hardcoded here.  Backends register themselves with the shared registry (see
``base.register_gemm_backend``) when their module is imported:

  * The open-source tree ships only **origami** (``origami_backend``), which is
    the default.
  * Additional backends may be added without editing any file in this package.
    They are discovered via, in order:
      1. ``primus.gemm_backends`` entry points of installed packages;
      2. module paths listed in the ``PRIMUS_GEMM_BACKEND_PLUGINS`` env var
         (comma/semicolon separated).

Selection order at call time: explicit ``backend_name`` → ``PRIMUS_GEMM_BACKEND``
→ ``"origami"``.

SDPA always uses the built-in analytical simulator (which prices its sub-GEMMs
with whichever GEMM backend is selected).
"""

import importlib
import os
from typing import Optional

from primus.core.projection.simulation_backends.base import (
    GEMMSimulationBackend,
    SDPASimulationBackend,
    available_gemm_backends,
    get_gemm_backend_factory,
)

_DISCOVERY_DONE = False


def _ensure_backends_discovered() -> None:
    """Import backend modules so they self-register (idempotent).

    This is the single discovery seam.  It never raises: a backend that fails to
    import simply is not registered, and selecting it later yields a clear error.
    """
    global _DISCOVERY_DONE
    if _DISCOVERY_DONE:
        return

    # 1. Built-in open-source backend(s).
    for mod in ("primus.core.projection.simulation_backends.origami_backend",):
        try:
            importlib.import_module(mod)
        except Exception:
            # Best-effort discovery: a backend that fails to import simply stays
            # unregistered; selecting it later raises a clear error.
            pass

    # 2. External backends declared as ``primus.gemm_backends`` entry points.
    #    Convention: each entry point is a zero-arg ``register`` callable.
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        if hasattr(eps, "select"):  # Python >= 3.10
            group = eps.select(group="primus.gemm_backends")
        else:  # pragma: no cover - legacy
            group = eps.get("primus.gemm_backends", [])
        for ep in group:
            try:
                loaded = ep.load()
                if callable(loaded):
                    loaded()
            except Exception:
                # One misbehaving plugin must not block the others; skip it.
                pass
    except Exception:
        # Entry-point enumeration is optional/best-effort; never abort setup.
        pass

    # 3. External backends declared via env var (module paths to import).
    raw = os.getenv("PRIMUS_GEMM_BACKEND_PLUGINS", "")
    for mod in (m.strip() for m in raw.replace(";", ",").split(",")):
        if mod:
            try:
                importlib.import_module(mod)
            except Exception:
                # Ignore a misconfigured plugin path so it can't break selection.
                pass

    _DISCOVERY_DONE = True


def list_available_gemm_backends() -> tuple:
    """Trigger discovery and return the sorted names of registered GEMM backends."""
    _ensure_backends_discovered()
    names = available_gemm_backends()
    return names or ("origami",)


def get_gemm_simulation_backend(
    backend_name: Optional[str] = None,
    gpu_arch: Optional[str] = None,
    gpu_clock_mhz: Optional[int] = None,
    require_simulation: bool = True,
) -> GEMMSimulationBackend:
    """
    Create and return the GEMM simulation backend.

    Args:
        backend_name: Explicit backend name (any registered backend, e.g.
                      "origami").  If None, falls back to ``PRIMUS_GEMM_BACKEND``
                      and finally defaults to "origami".
        gpu_arch: GPU architecture override (e.g. "gfx942", "mi300x", "mi325x").
        gpu_clock_mhz: Override the GPU compute clock frequency in MHz.
        require_simulation: If True (default), raise RuntimeError when the
            selected backend is not available.  Set to False when only
            hardware-profile metadata (e.g. ``hbm_bandwidth_gbps``) is
            needed — this avoids a hard dependency on the backend library in
            benchmark mode.

    Returns:
        A GEMMSimulationBackend instance.

    Raises:
        ValueError: If the requested backend name is not registered.
        RuntimeError: If require_simulation is True and the backend is not available.
    """
    _ensure_backends_discovered()

    name = backend_name or os.getenv("PRIMUS_GEMM_BACKEND", None)
    if name is not None:
        name = name.lower().strip()
    if not name:
        name = "origami"

    is_rank_0 = int(os.getenv("RANK", "0")) == 0

    factory = get_gemm_backend_factory(name)
    if factory is None:
        supported = ", ".join(available_gemm_backends()) or "origami"
        raise ValueError(
            f"Unknown GEMM simulation backend: '{name}'. Supported backends: {supported}. "
            "Extra backends can be added via the entry-point group 'primus.gemm_backends' "
            "or the PRIMUS_GEMM_BACKEND_PLUGINS env var."
        )

    backend = factory(gpu_arch=gpu_arch, gpu_clock_mhz=gpu_clock_mhz)
    if require_simulation and not backend.is_available():
        raise RuntimeError(
            f"GEMM simulation backend '{name}' is registered but not available in "
            "this environment. Check that its underlying library is installed "
            "(e.g. 'pip install origami')."
        )

    if is_rank_0 and require_simulation:
        print(f"[Primus:Simulation] Using GEMM backend: {name}")
    return backend


def get_sdpa_simulation_backend(
    gpu_arch: Optional[str] = None,
    gpu_clock_mhz: Optional[int] = None,
    backend_name: Optional[str] = None,
) -> SDPASimulationBackend:
    """
    Create and return the SDPA simulation backend.

    Models the FAv3 (Flash Attention v3) kernels, pricing the per-tile sub-GEMMs
    with the selected GEMM backend (default origami).  The GEMM engine follows
    the same registry-driven selection as ``get_gemm_simulation_backend``.

    Args:
        gpu_arch: GPU architecture override (e.g. "mi300x", "mi355x").
        gpu_clock_mhz: Override the GPU compute clock frequency in MHz.
        backend_name: Explicit GEMM engine name.  If None, falls back to
            ``PRIMUS_GEMM_BACKEND`` and finally "origami".

    Returns:
        An SDPASimulationBackend instance.

    Raises:
        RuntimeError: If the selected GEMM backend is not available.
    """
    from primus.core.projection.simulation_backends.sdpa_simulator import SDPASimulator

    _ensure_backends_discovered()

    name = (backend_name or os.getenv("PRIMUS_GEMM_BACKEND") or "origami").lower().strip()
    # Fall back to the built-in origami engine if the requested GEMM backend is
    # not registered in this environment.
    if get_gemm_backend_factory(name) is None:
        name = "origami"

    is_rank_0 = int(os.getenv("RANK", "0")) == 0
    if is_rank_0:
        print(f"[Primus:Simulation] Using SDPA backend: sdpa_simulator (FAv3, {name} 1-CU)")

    return SDPASimulator(
        gpu_arch=gpu_arch,
        gpu_clock_mhz=gpu_clock_mhz,
        gemm_backend=name,
    )
