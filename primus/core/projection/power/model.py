###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Phase-based GPU power model and power-cap solver.

An iteration is a list of :class:`Phase` objects (matrix compute, vector /
memory-bound compute, exposed communication, optimizer, pipeline bubble).  Each
phase carries its projected duration and four activities in [0, 1]:

``busy``    fraction of the phase the compute units are executing kernels
``matrix``  achieved matrix FLOP/s over peak matrix FLOP/s *at the running clock*
``hbm``     achieved HBM bandwidth over peak HBM bandwidth
``link``    fraction of the phase a scale-up / scale-out link is moving data

and ``clock_sensitivity`` -- the fraction of the phase whose duration scales
with ``1 / f`` (compute-bound work); the rest (memory, network) does not.

Board power for a phase at relative clock ``s = f / f_peak`` is::

    P(s) = P_idle + s**alpha * (C_busy*busy + C_matrix*matrix) + C_hbm*hbm' + C_link*link'

where ``hbm'`` and ``link'`` are the memory and link activities after the phase
has been stretched by the lower clock (same bytes over a longer time).

Phase durations are taken to be at the relative clock ``s0`` the performance
projection ran at (1.0 = peak clock); the cap can only lower the clock from
there.  Under a power cap the firmware lowers the clock until board power fits.
Two control assumptions are supported:

``instantaneous`` (default) each phase is capped on its own -- power limits act
                  on millisecond windows, shorter than most training phases, so
                  a GEMM-heavy phase throttles even if the iteration average is
                  under the cap.
``average``       one clock for the whole iteration, chosen so the
                  time-averaged power fits (long averaging window).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from primus.core.projection.power.specs import GPUPowerProfile


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass
class Phase:
    name: str
    time_ms: float
    busy: float = 1.0
    matrix: float = 0.0
    hbm: float = 0.0
    link: float = 0.0
    clock_sensitivity: float = 1.0

    def stretched_time_ms(self, s: float, s0: float = 1.0) -> float:
        """Duration at relative clock *s* of a phase timed at relative clock *s0*."""
        b = _clamp01(self.clock_sensitivity)
        return self.time_ms * (b * s0 / s + (1.0 - b))


@dataclass
class PhaseResult:
    name: str
    base_time_ms: float
    time_ms: float
    clock_fraction: float
    power_w: float
    energy_j: float
    capped: bool

    def as_dict(self) -> Dict[str, float]:
        return {
            "name": self.name,
            "base_time_ms": self.base_time_ms,
            "time_ms": self.time_ms,
            "clock_fraction": self.clock_fraction,
            "power_w": self.power_w,
            "energy_j": self.energy_j,
            "capped": self.capped,
        }


@dataclass
class PowerSolution:
    """Per-GPU power over one iteration, with and without the cap."""

    cap_w: float
    cap_mode: str
    phases: List[PhaseResult]
    base_iteration_ms: float
    iteration_ms: float
    avg_power_w: float
    peak_phase_power_w: float
    uncapped_avg_power_w: float
    uncapped_peak_phase_power_w: float
    base_clock_fraction: float = 1.0
    feasible: bool = True
    warnings: List[str] = field(default_factory=list)

    @property
    def slowdown(self) -> float:
        return self.iteration_ms / self.base_iteration_ms if self.base_iteration_ms > 0 else 1.0

    @property
    def energy_per_iteration_j(self) -> float:
        return sum(p.energy_j for p in self.phases)

    @property
    def effective_clock_fraction(self) -> float:
        """Time-weighted clock over the phases that execute kernels."""
        num = sum(p.clock_fraction * p.time_ms for p in self.phases if p.name != "bubble")
        den = sum(p.time_ms for p in self.phases if p.name != "bubble")
        return num / den if den > 0 else self.base_clock_fraction

    def as_dict(self) -> Dict[str, object]:
        return {
            "cap_w": self.cap_w,
            "cap_mode": self.cap_mode,
            "base_clock_fraction": self.base_clock_fraction,
            "base_iteration_ms": self.base_iteration_ms,
            "iteration_ms": self.iteration_ms,
            "slowdown": self.slowdown,
            "avg_power_w": self.avg_power_w,
            "peak_phase_power_w": self.peak_phase_power_w,
            "uncapped_avg_power_w": self.uncapped_avg_power_w,
            "uncapped_peak_phase_power_w": self.uncapped_peak_phase_power_w,
            "energy_per_iteration_j": self.energy_per_iteration_j,
            "effective_clock_fraction": self.effective_clock_fraction,
            "feasible": self.feasible,
            "warnings": list(self.warnings),
            "phases": [p.as_dict() for p in self.phases],
        }


def _bisect_clock(fits: Callable[[float], bool], s_min: float, s_max: float) -> float:
    """Highest clock in [s_min, s_max] for which *fits* holds (monotone)."""
    if fits(s_max):
        return s_max
    if not fits(s_min):
        return s_min
    lo, hi = s_min, s_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


class ParametricPowerModel:
    """Board-power surrogate for one GPU; see the module docstring."""

    name = "parametric"

    def __init__(self, profile: GPUPowerProfile):
        self.profile = profile

    def phase_power_w(self, phase: Phase, s: float = 1.0, s0: float = 1.0) -> float:
        c = self.profile.coefficients
        s = max(s, 1e-6)
        stretch = phase.stretched_time_ms(s, s0) / phase.time_ms if phase.time_ms > 0 else 1.0
        clocked = c.busy_w * _clamp01(phase.busy) + c.matrix_w * _clamp01(phase.matrix)
        unclocked = (c.hbm_w * _clamp01(phase.hbm) + c.link_w * _clamp01(phase.link)) / stretch
        return self.profile.idle_w + (s**self.profile.alpha) * clocked + unclocked

    def _evaluate(self, phases: List[Phase], clocks: List[float], s0: float) -> List[PhaseResult]:
        out = []
        for ph, s in zip(phases, clocks):
            t = ph.stretched_time_ms(s, s0)
            p = self.phase_power_w(ph, s, s0)
            out.append(
                PhaseResult(
                    name=ph.name,
                    base_time_ms=ph.time_ms,
                    time_ms=t,
                    clock_fraction=s,
                    power_w=p,
                    energy_j=p * t / 1000.0,
                    capped=s < s0 - 1e-9,
                )
            )
        return out

    def _average_power(self, phases: List[Phase], s: float, s0: float) -> float:
        rs = self._evaluate(phases, [s] * len(phases), s0)
        t = sum(r.time_ms for r in rs)
        return sum(r.energy_j for r in rs) * 1000.0 / t if t > 0 else self.profile.idle_w

    def solve(
        self,
        phases: List[Phase],
        cap_w: Optional[float] = None,
        cap_mode: str = "instantaneous",
        base_clock_fraction: float = 1.0,
    ) -> PowerSolution:
        """Clock per phase that keeps board power under *cap_w* (default: TBP).

        *base_clock_fraction* is the relative clock the phase durations were
        projected at; it is also the highest clock the solve will pick.
        """
        if cap_mode not in ("instantaneous", "average"):
            raise ValueError(f"unknown cap_mode '{cap_mode}'")
        phases = [p for p in phases if p.time_ms > 0]
        cap = float(cap_w) if cap_w else self.profile.tbp_w
        s_min = self.profile.min_clock_fraction
        s0 = min(1.0, max(base_clock_fraction or 1.0, s_min))

        base = self._evaluate(phases, [s0] * len(phases), s0)
        base_t = sum(r.time_ms for r in base)
        base_avg = sum(r.energy_j for r in base) * 1000.0 / base_t if base_t > 0 else self.profile.idle_w
        base_peak = max((r.power_w for r in base), default=self.profile.idle_w)

        if cap_mode == "instantaneous":
            clocks = [
                _bisect_clock(lambda s, ph=ph: self.phase_power_w(ph, s, s0) <= cap, s_min, s0)
                for ph in phases
            ]
        else:
            s = _bisect_clock(lambda s: self._average_power(phases, s, s0) <= cap, s_min, s0)
            clocks = [s] * len(phases)
        capped = self._evaluate(phases, clocks, s0)
        t = sum(r.time_ms for r in capped)
        avg = sum(r.energy_j for r in capped) * 1000.0 / t if t > 0 else self.profile.idle_w

        warnings: List[str] = []
        feasible = True
        if cap < self.profile.idle_w:
            feasible = False
            warnings.append(f"cap {cap:.0f} W is below idle power {self.profile.idle_w:.0f} W")
        if cap_mode == "instantaneous":
            for r in capped:
                if r.power_w > cap * 1.001:
                    feasible = False
                    warnings.append(
                        f"phase '{r.name}' exceeds the cap even at the minimum clock ({s_min:.2f})"
                    )
        elif avg > cap * 1.001:
            feasible = False
            warnings.append("iteration-average power exceeds the cap even at the minimum clock")
        return PowerSolution(
            cap_w=cap,
            cap_mode=cap_mode,
            phases=capped,
            base_iteration_ms=base_t,
            iteration_ms=t,
            avg_power_w=avg,
            peak_phase_power_w=max((r.power_w for r in capped), default=self.profile.idle_w),
            uncapped_avg_power_w=base_avg,
            uncapped_peak_phase_power_w=base_peak,
            base_clock_fraction=s0,
            feasible=feasible,
            warnings=warnings,
        )
