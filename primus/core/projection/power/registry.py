###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Power-model registry.

The built-in ``parametric`` model is the default.  Other models plug in
without changes to this tree, either by calling :func:`register_power_model`
or through the ``primus.power_models`` entry-point group; a factory takes
``(profile, config)`` and returns an object with
``solve(phases, cap_w, cap_mode, base_clock_fraction) -> PowerSolution`` and
``phase_power_w(phase, s, s0) -> float``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from primus.core.projection.power.model import ParametricPowerModel
from primus.core.projection.power.specs import GPUPowerProfile

PowerModelFactory = Callable[[GPUPowerProfile, Dict[str, Any]], Any]

_REGISTRY: Dict[str, PowerModelFactory] = {
    "parametric": lambda profile, cfg: ParametricPowerModel(profile),
}
_ENTRY_POINTS_LOADED = False


def register_power_model(name: str, factory: PowerModelFactory) -> None:
    _REGISTRY[name.lower()] = factory


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        group = (
            eps.select(group="primus.power_models")
            if hasattr(eps, "select")
            else eps.get("primus.power_models", [])
        )
        for ep in group:
            try:
                register_power_model(ep.name, ep.load())
            except Exception as exc:
                print(f"[Primus:Power] WARNING: failed to load power model plugin '{ep.name}': {exc}")
    except Exception:
        pass  # no plugin metadata available; built-ins only


def available_power_models() -> Tuple[str, ...]:
    _load_entry_points()
    return tuple(sorted(_REGISTRY))


def get_power_model(name: Optional[str], profile: GPUPowerProfile, config: Optional[Dict[str, Any]] = None):
    _load_entry_points()
    key = (name or "parametric").lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown power model '{name}'; available: {', '.join(available_power_models())}")
    return _REGISTRY[key](profile, config or {})
