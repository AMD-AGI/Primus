###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU power profiles: public board-power specs plus an optional private overlay.

A :class:`GPUPowerProfile` describes one GPU with a small parametric power
surrogate (see :mod:`primus.core.projection.power.model`)::

    P(f, activity) = P_idle
                   + (f / f_peak) ** alpha * (C_busy * u_busy + C_matrix * u_matrix)
                   + C_hbm * u_hbm + C_link * u_link

The built-in table holds only what AMD publishes in product datasheets: total
board power (TBP), peak engine clock and peak HBM bandwidth.  Everything else
(idle power, the activity coefficients and the frequency exponent ``alpha``)
starts from generic, architecture-agnostic defaults that are *anchored* to the
public TBP and is meant to be replaced by a fit to measured telemetry
(``primus-cli projection power-fit``).

Fitted or otherwise private profiles are never needed in this tree: point
``PRIMUS_POWER_PROFILE_DIR`` at one or more directories of YAML files, or pass
``--power-config``.  Schema::

    gpu_power_profiles:
      <arch>:
        tbp_w: 1400              # board power limit at the default cap
        peak_clock_mhz: 2400
        min_clock_mhz: 500       # optional
        hbm_bandwidth_gbps: 8000 # optional
        idle_w: 180              # optional; else idle_fraction * tbp_w
        alpha: 2.5               # optional frequency exponent
        coefficients:            # optional, watts at full activity / peak clock
          busy_w: ...
          matrix_w: ...
          hbm_w: ...
          link_w: ...
        aliases: [<other names>]
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional

import yaml

POWER_PROFILE_DIR_ENV = "PRIMUS_POWER_PROFILE_DIR"

# Generic surrogate defaults.  None of these are measured on any AMD part; they
# only fix the shape of the model until telemetry replaces them.
DEFAULT_IDLE_FRACTION = 0.15
DEFAULT_ALPHA = 2.5
DEFAULT_MIN_CLOCK_FRACTION = 0.25
# Share of the dynamic headroom (TBP - idle) each term may consume at the
# reference activity below.
DEFAULT_DYNAMIC_SHARES = {"busy": 0.20, "matrix": 0.55, "hbm": 0.20, "link": 0.05}
# Activity at which a GPU at its peak clock is assumed to sit exactly at TBP: a
# large, well-tuned GEMM stream with moderate memory traffic, which is the load
# board power limits are sized for.
REFERENCE_ACTIVITY = {"busy": 1.0, "matrix": 0.75, "hbm": 0.5, "link": 0.5}


@dataclass(frozen=True)
class PublicGPUSpec:
    """Datasheet numbers only."""

    tbp_w: float
    peak_clock_mhz: float
    hbm_bandwidth_gbps: float
    hbm_capacity_gb: float


# AMD Instinct product datasheets (total board power, peak engine clock, peak
# memory bandwidth, memory capacity).
_PUBLIC_SPECS: Dict[str, PublicGPUSpec] = {
    "mi300x": PublicGPUSpec(
        tbp_w=750.0, peak_clock_mhz=2100.0, hbm_bandwidth_gbps=5300.0, hbm_capacity_gb=192.0
    ),
    "mi325x": PublicGPUSpec(
        tbp_w=1000.0, peak_clock_mhz=2100.0, hbm_bandwidth_gbps=6000.0, hbm_capacity_gb=256.0
    ),
    "mi350x": PublicGPUSpec(
        tbp_w=1000.0, peak_clock_mhz=2200.0, hbm_bandwidth_gbps=8000.0, hbm_capacity_gb=288.0
    ),
    "mi355x": PublicGPUSpec(
        tbp_w=1400.0, peak_clock_mhz=2400.0, hbm_bandwidth_gbps=8000.0, hbm_capacity_gb=288.0
    ),
}
_PUBLIC_ALIASES = {"gfx942": "mi300x", "gfx950": "mi355x"}


@dataclass(frozen=True)
class PowerCoefficients:
    """Watts each activity term draws at full activity and peak clock."""

    busy_w: float
    matrix_w: float
    hbm_w: float
    link_w: float

    def as_dict(self) -> Dict[str, float]:
        return {"busy_w": self.busy_w, "matrix_w": self.matrix_w, "hbm_w": self.hbm_w, "link_w": self.link_w}


@dataclass(frozen=True)
class GPUPowerProfile:
    name: str
    tbp_w: float
    peak_clock_mhz: float
    idle_w: float
    alpha: float
    coefficients: PowerCoefficients
    min_clock_mhz: float
    hbm_bandwidth_gbps: Optional[float] = None
    source: str = "public-spec"
    notes: List[str] = field(default_factory=list)

    @property
    def min_clock_fraction(self) -> float:
        return max(1e-3, min(1.0, self.min_clock_mhz / self.peak_clock_mhz))

    def with_overrides(self, **kw: Any) -> "GPUPowerProfile":
        return replace(self, **kw)

    def to_yaml_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "tbp_w": round(self.tbp_w, 3),
            "peak_clock_mhz": round(self.peak_clock_mhz, 3),
            "min_clock_mhz": round(self.min_clock_mhz, 3),
            "idle_w": round(self.idle_w, 3),
            "alpha": round(self.alpha, 4),
            "coefficients": {k: round(v, 3) for k, v in self.coefficients.as_dict().items()},
        }
        if self.hbm_bandwidth_gbps:
            out["hbm_bandwidth_gbps"] = self.hbm_bandwidth_gbps
        return out


def anchored_coefficients(
    tbp_w: float,
    idle_w: float,
    shares: Optional[Dict[str, float]] = None,
    reference: Optional[Dict[str, float]] = None,
) -> PowerCoefficients:
    """Split ``tbp_w - idle_w`` so the reference activity lands exactly on TBP."""
    shares = shares or DEFAULT_DYNAMIC_SHARES
    reference = reference or REFERENCE_ACTIVITY
    total = sum(shares.values())
    headroom = max(0.0, tbp_w - idle_w)

    def coef(key: str) -> float:
        return headroom * shares[key] / total / max(reference[key], 1e-6)

    return PowerCoefficients(
        busy_w=coef("busy"), matrix_w=coef("matrix"), hbm_w=coef("hbm"), link_w=coef("link")
    )


def _normalize(arch: Optional[str]) -> str:
    return (arch or "").lower().strip()


def public_spec(arch: Optional[str]) -> Optional[PublicGPUSpec]:
    key = _normalize(arch)
    return _PUBLIC_SPECS.get(_PUBLIC_ALIASES.get(key, key))


def public_archs() -> List[str]:
    return sorted(_PUBLIC_SPECS)


def profile_from_public_spec(arch: str, spec: PublicGPUSpec) -> GPUPowerProfile:
    idle_w = DEFAULT_IDLE_FRACTION * spec.tbp_w
    return GPUPowerProfile(
        name=_normalize(arch),
        tbp_w=spec.tbp_w,
        peak_clock_mhz=spec.peak_clock_mhz,
        idle_w=idle_w,
        alpha=DEFAULT_ALPHA,
        coefficients=anchored_coefficients(spec.tbp_w, idle_w),
        min_clock_mhz=DEFAULT_MIN_CLOCK_FRACTION * spec.peak_clock_mhz,
        hbm_bandwidth_gbps=spec.hbm_bandwidth_gbps,
        source="public-spec",
        notes=["idle power, coefficients and alpha are generic defaults; fit them with power-fit"],
    )


def profile_from_mapping(
    name: str, data: Dict[str, Any], base: Optional[GPUPowerProfile] = None
) -> GPUPowerProfile:
    """Build a profile from a YAML mapping, filling gaps from *base*."""
    tbp_w = float(data.get("tbp_w", base.tbp_w if base else 0.0))
    peak = float(data.get("peak_clock_mhz", base.peak_clock_mhz if base else 0.0))
    if tbp_w <= 0 or peak <= 0:
        raise ValueError(f"power profile '{name}' needs tbp_w and peak_clock_mhz")
    if "idle_w" in data:
        idle_w = float(data["idle_w"])
    elif "idle_fraction" in data:
        idle_w = float(data["idle_fraction"]) * tbp_w
    elif base is not None and base.tbp_w == tbp_w:
        idle_w = base.idle_w
    else:
        idle_w = DEFAULT_IDLE_FRACTION * tbp_w
    coefs = data.get("coefficients")
    if coefs:
        default = anchored_coefficients(tbp_w, idle_w)
        coefficients = PowerCoefficients(
            busy_w=float(coefs.get("busy_w", default.busy_w)),
            matrix_w=float(coefs.get("matrix_w", default.matrix_w)),
            hbm_w=float(coefs.get("hbm_w", default.hbm_w)),
            link_w=float(coefs.get("link_w", default.link_w)),
        )
    elif base is not None and base.tbp_w == tbp_w and base.idle_w == idle_w:
        coefficients = base.coefficients
    else:
        coefficients = anchored_coefficients(tbp_w, idle_w)
    min_clock = data.get("min_clock_mhz")
    if min_clock is None:
        min_clock = base.min_clock_mhz if base else DEFAULT_MIN_CLOCK_FRACTION * peak
    hbm = data.get("hbm_bandwidth_gbps", base.hbm_bandwidth_gbps if base else None)
    return GPUPowerProfile(
        name=_normalize(name),
        tbp_w=tbp_w,
        peak_clock_mhz=peak,
        idle_w=idle_w,
        alpha=float(data.get("alpha", base.alpha if base else DEFAULT_ALPHA)),
        coefficients=coefficients,
        min_clock_mhz=float(min_clock),
        hbm_bandwidth_gbps=float(hbm) if hbm else None,
        source=str(data.get("source", "profile")),
        notes=list(data.get("notes", []) or []),
    )


def _iter_profile_files(dirs: Iterable[str]) -> Iterable[str]:
    for d in dirs:
        d = d.strip()
        if not d or not os.path.isdir(d):
            continue
        for pattern in ("*.yaml", "*.yml"):
            yield from sorted(glob.glob(os.path.join(d, pattern)))


def load_profile_file(path: str) -> Dict[str, Dict[str, Any]]:
    """``{arch: mapping}`` from one YAML file (aliases expanded)."""
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    table = doc.get("gpu_power_profiles") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for arch, data in table.items():
        if not isinstance(data, dict):
            continue
        data = dict(data)
        data.setdefault("source", os.path.basename(path))
        out[_normalize(arch)] = data
        for alias in data.get("aliases", []) or []:
            out[_normalize(alias)] = data
    return out


def _external_profiles(extra_files: Iterable[str] = ()) -> Dict[str, Dict[str, Any]]:
    dirs = os.environ.get(POWER_PROFILE_DIR_ENV, "").split(os.pathsep)
    table: Dict[str, Dict[str, Any]] = {}
    for path in list(_iter_profile_files(dirs)) + [p for p in extra_files if p]:
        try:
            table.update(load_profile_file(path))
        except Exception as exc:  # a broken private overlay must not break projection
            print(f"[Primus:Power] WARNING: ignoring power profile {path}: {exc}")
    return table


def resolve_power_profile(
    arch: Optional[str],
    profile_files: Iterable[str] = (),
    overrides: Optional[Dict[str, Any]] = None,
) -> Optional[GPUPowerProfile]:
    """Public spec, then ``PRIMUS_POWER_PROFILE_DIR`` / *profile_files*, then *overrides*.

    Returns *None* only when the arch has neither a public spec nor a profile
    and *overrides* does not give ``tbp_w`` and ``peak_clock_mhz``.
    """
    key = _normalize(arch) or _normalize(os.getenv("PRIMUS_GPU_ARCH", "mi300x"))
    spec = public_spec(key)
    profile = profile_from_public_spec(key, spec) if spec else None
    external = _external_profiles(profile_files).get(key)
    if external is not None:
        profile = profile_from_mapping(key, external, base=profile)
    if overrides:
        try:
            profile = profile_from_mapping(key, overrides, base=profile)
        except ValueError:
            if profile is None:
                return None
            raise
    return profile
