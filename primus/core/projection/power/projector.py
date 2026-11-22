###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Power projection on top of a performance projection result.

:func:`project_power` takes the performance projection's iteration breakdown
and workload footprint, solves the per-GPU power (and the clock the power cap
forces), and rolls it up to node / cluster / facility power.  The printed
lines are stable so the tuning agent can parse them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from primus.core.projection.power.footprint import (
    IterationBreakdown,
    WorkloadFootprint,
    build_iteration_phases,
)
from primus.core.projection.power.infrastructure import (
    InfraConfig,
    capacity_plan,
    cluster_power,
)
from primus.core.projection.power.model import PowerSolution
from primus.core.projection.power.registry import get_power_model
from primus.core.projection.power.specs import GPUPowerProfile, resolve_power_profile

POWER_CAP_ENV = "PRIMUS_POWER_CAP_W"


@dataclass
class PowerConfig:
    gpu_arch: Optional[str] = None
    cap_w: Optional[float] = None
    cap_mode: str = "instantaneous"
    model: str = "parametric"
    profile_overrides: Dict[str, Any] = field(default_factory=dict)
    profile_files: List[str] = field(default_factory=list)
    footprint_overrides: Dict[str, Any] = field(default_factory=dict)
    infrastructure: Dict[str, Any] = field(default_factory=dict)
    capacity: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Optional[str]) -> "PowerConfig":
        if not path:
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        return cls.from_dict(doc, base_dir=os.path.dirname(os.path.abspath(path)))

    @classmethod
    def from_dict(cls, doc: Dict[str, Any], base_dir: str = "") -> "PowerConfig":
        power = doc.get("power", {}) or {}
        files = [
            os.path.join(base_dir, p) if base_dir and not os.path.isabs(p) else p
            for p in power.get("profile_files", []) or []
        ]
        cfg = cls(
            gpu_arch=power.get("gpu_arch"),
            cap_w=power.get("cap_w"),
            cap_mode=power.get("cap_mode", "instantaneous"),
            model=power.get("model", "parametric"),
            profile_overrides=dict(power.get("profile", {}) or {}),
            profile_files=files,
            footprint_overrides=dict(power.get("footprint", {}) or {}),
            infrastructure=dict(doc.get("infrastructure", {}) or {}),
            capacity=dict(doc.get("capacity", {}) or {}),
        )
        if doc.get("gpu_power_profiles"):
            # A fitted profile file passed directly as --power-config.
            arch, data = next(iter(doc["gpu_power_profiles"].items()))
            cfg.gpu_arch = cfg.gpu_arch or arch
            cfg.profile_overrides = {**data, **cfg.profile_overrides}
        return cfg


@dataclass
class PowerReport:
    profile: GPUPowerProfile
    footprint: WorkloadFootprint
    breakdown: IterationBreakdown
    solution: PowerSolution
    tokens_per_iteration: float
    num_gpus: int
    num_nodes: int
    gpus_per_node: int
    infra: InfraConfig
    capacity: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def capped_tokens_per_s_per_gpu(self) -> float:
        t = self.solution.iteration_ms
        return self.tokens_per_iteration * 1000.0 / t / self.num_gpus if t > 0 and self.num_gpus else 0.0

    @property
    def energy_per_token_j(self) -> float:
        """GPU energy per token (all GPUs of the job)."""
        if self.tokens_per_iteration <= 0:
            return 0.0
        return self.solution.energy_per_iteration_j * self.num_gpus / self.tokens_per_iteration

    @property
    def tokens_per_joule_per_gpu(self) -> float:
        p = self.solution.avg_power_w
        return self.capped_tokens_per_s_per_gpu / p if p > 0 else 0.0

    def cluster(self):
        return cluster_power(self.num_nodes, self.solution.avg_power_w, self.infra)

    def as_dict(self) -> Dict[str, Any]:
        cp = self.cluster()
        return {
            "gpu_arch": self.profile.name,
            "profile_source": self.profile.source,
            "tbp_w": self.profile.tbp_w,
            "footprint": self.footprint.as_dict(),
            "breakdown": self.breakdown.as_dict(),
            "solution": self.solution.as_dict(),
            "capped_tokens_per_s_per_gpu": self.capped_tokens_per_s_per_gpu,
            "energy_per_token_j": self.energy_per_token_j,
            "tokens_per_joule_per_gpu": self.tokens_per_joule_per_gpu,
            "cluster": cp.as_dict(),
            "capacity": self.capacity,
        }

    def print(self) -> None:
        s = self.solution
        cp = self.cluster()
        print("=" * 100)
        print("Power Projection")
        print("=" * 100)
        print(
            f"   GPU: {self.profile.name} (TBP {self.profile.tbp_w:.0f} W, peak clock "
            f"{self.profile.peak_clock_mhz:.0f} MHz, profile: {self.profile.source})"
        )
        fp = self.footprint
        print(
            f"   Workload footprint ({fp.source}): matrix util {fp.matrix_utilization:.2f}, "
            f"HBM util {fp.matrix_hbm_utilization:.2f}, clock-bound share {fp.matrix_clock_sensitivity:.2f}, "
            f"non-matrix share {fp.non_matrix_fraction:.2f}"
        )
        print("   Phases (per GPU, per iteration):")
        for p in s.phases:
            flag = " [capped]" if p.capped else ""
            print(
                f"     {p.name:<10s} {p.base_time_ms:10.2f} ms -> {p.time_ms:10.2f} ms  "
                f"clock {p.clock_fraction * 100:5.1f}%  {p.power_w:7.1f} W{flag}"
            )
        print(
            f"   Uncapped Avg Power/GPU: {s.uncapped_avg_power_w:.1f} W (peak phase {s.uncapped_peak_phase_power_w:.1f} W)"
        )
        print(f"   Power Cap/GPU: {s.cap_w:.0f} W ({s.cap_mode})")
        print(f"   Avg Power/GPU: {s.avg_power_w:.1f} W (peak phase {s.peak_phase_power_w:.1f} W)")
        print(f"   Effective Clock: {s.effective_clock_fraction * self.profile.peak_clock_mhz:.0f} MHz")
        print(f"   Power-Capped Step Time: {s.iteration_ms:.3f} ms (slowdown {s.slowdown:.3f}x)")
        print(f"   Power-Capped Throughput: {self.capped_tokens_per_s_per_gpu:,.0f} tokens/s per GPU")
        print(f"   Energy/Token (GPU): {self.energy_per_token_j:.4f} J")
        print(f"   Tokens/J/GPU: {self.tokens_per_joule_per_gpu:.3f}")
        print(
            f"   Node Power: {cp.node['node_w'] / 1000:.2f} kW | Cluster IT Power: {cp.it_w / 1000:.1f} kW "
            f"({cp.num_nodes} nodes, {cp.switches} switches) | Facility Power: {cp.facility_w / 1000:.1f} kW "
            f"(PUE {self.infra.pue:.2f})"
        )
        if cp.racks is not None:
            print(f"   Racks @ {self.infra.rack_power_kw:.0f} kW: {cp.racks}")
        print(f"   Power Feasible: {'yes' if s.feasible else 'no'}")
        for w in s.warnings:
            print(f"   WARNING: {w}")
        if self.capacity:
            print("   Capacity plan (fixed power budget):")
            print(
                f"     {'cap W':>7s} {'tok/s/GPU':>11s} {'avg W':>7s} {'nodes':>6s} {'GPUs':>6s} "
                f"{'cluster tok/s':>15s} {'run MW':>7s} {'tok/kWh':>12s}"
            )
            for o in self.capacity:
                print(
                    f"     {o['gpu_cap_w']:7.0f} {o['tokens_per_s_per_gpu']:11,.0f} {o['avg_gpu_power_w']:7.0f} "
                    f"{o['num_nodes']:6d} {o['num_gpus']:6d} {o['cluster_tokens_per_s']:15,.0f} "
                    f"{o['facility_w'] / 1e6:7.3f} {o['tokens_per_kwh']:12,.0f}"
                )
            best = max(self.capacity, key=lambda o: o["cluster_tokens_per_s"])
            print(
                f"   Best cap under budget: {best['gpu_cap_w']:.0f} W/GPU -> {best['cluster_tokens_per_s']:,.0f} tokens/s"
            )
        print("=" * 100)


def resolve_cap_w(config: PowerConfig, cli_cap_w: Optional[float], profile: GPUPowerProfile) -> float:
    if cli_cap_w:
        return float(cli_cap_w)
    if config.cap_w:
        return float(config.cap_w)
    env = os.getenv(POWER_CAP_ENV)
    if env:
        return float(env)
    return profile.tbp_w


def project_power(
    breakdown: IterationBreakdown,
    tokens_per_iteration: float,
    num_gpus: int,
    num_nodes: int,
    gpus_per_node: int,
    gpu_arch: Optional[str] = None,
    config: Optional[PowerConfig] = None,
    footprint: Optional[WorkloadFootprint] = None,
    cap_w: Optional[float] = None,
    cap_mode: Optional[str] = None,
    hardware_config: Optional[Dict[str, Any]] = None,
    base_clock_mhz: Optional[float] = None,
) -> Optional[PowerReport]:
    """Per-GPU power for one projected iteration, rolled up to the cluster.

    *base_clock_mhz* is the compute clock the performance projection's times
    correspond to (``--gpu-clock-mhz``, or the bench GPU's measured effective
    clock); the profile's peak clock when omitted.
    """
    config = config or PowerConfig()
    arch = config.gpu_arch or gpu_arch
    profile = resolve_power_profile(arch, config.profile_files, config.profile_overrides)
    if profile is None:
        print(
            f"[Primus:Power] No power profile for '{arch}'. Give tbp_w and peak_clock_mhz under power.profile "
            f"in --power-config, or a profile in ${{PRIMUS_POWER_PROFILE_DIR}}."
        )
        return None
    fp = footprint or WorkloadFootprint()
    if config.footprint_overrides:
        fp = WorkloadFootprint.from_dict({**fp.as_dict(), **config.footprint_overrides, "source": fp.source})
    model = get_power_model(config.model, profile, {"power": config})
    cap = resolve_cap_w(config, cap_w, profile)
    mode = cap_mode or config.cap_mode
    phases = build_iteration_phases(breakdown, fp)
    s0 = min(1.0, float(base_clock_mhz) / profile.peak_clock_mhz) if base_clock_mhz else 1.0
    solution = model.solve(phases, cap_w=cap, cap_mode=mode, base_clock_fraction=s0)

    hw = hardware_config or {}
    infra_defaults = {"gpus_per_node": gpus_per_node}
    if hw.get("nics_per_node") is not None:
        infra_defaults["nics_per_node"] = int(hw["nics_per_node"])
    infra = InfraConfig.from_dict(config.infrastructure, **infra_defaults)

    report = PowerReport(
        profile=profile,
        footprint=fp,
        breakdown=breakdown,
        solution=solution,
        tokens_per_iteration=tokens_per_iteration,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        infra=infra,
    )

    cap_cfg = config.capacity
    budget_kw = cap_cfg.get("facility_power_kw") or cap_cfg.get("it_power_kw")
    if budget_kw:
        caps = cap_cfg.get("caps_w") or [profile.tbp_w * f for f in (1.0, 0.9, 0.8, 0.7, 0.6)]

        def at_cap(c: float) -> Dict[str, float]:
            sol = model.solve(phases, cap_w=c, cap_mode=mode, base_clock_fraction=s0)
            t = sol.iteration_ms
            tps = tokens_per_iteration * 1000.0 / t / num_gpus if t > 0 and num_gpus else 0.0
            return {"tokens_per_s_per_gpu": tps, "avg_power_w": sol.avg_power_w}

        options = capacity_plan(
            budget_w=float(budget_kw) * 1000.0,
            caps_w=[float(c) for c in caps],
            throughput_at_cap=at_cap,
            cfg=infra,
            budget_is_facility=bool(cap_cfg.get("facility_power_kw")),
            max_nodes=cap_cfg.get("max_nodes"),
            node_multiple=int(cap_cfg.get("node_multiple", 1) or 1),
        )
        report.capacity = [o.as_dict() for o in options]
    return report
