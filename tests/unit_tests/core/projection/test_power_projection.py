###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the power projection package (``primus.core.projection.power``)."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from primus.core.projection.power import (
    InfraConfig,
    IterationBreakdown,
    OpRecorder,
    ParametricPowerModel,
    Phase,
    PowerConfig,
    TelemetrySample,
    WorkloadFootprint,
    build_iteration_phases,
    capacity_plan,
    cluster_power,
    fit_power_profile,
    get_power_model,
    load_telemetry,
    project_power,
    public_archs,
    register_power_model,
    resolve_power_profile,
)
from primus.core.projection.power.infrastructure import (
    node_power_w,
    scaleout_switch_count,
)
from primus.core.projection.power.specs import (
    POWER_PROFILE_DIR_ENV,
    REFERENCE_ACTIVITY,
    GPUPowerProfile,
    PowerCoefficients,
)
from primus.core.projection.power.telemetry import write_profile_yaml


@pytest.fixture(autouse=True)
def _no_private_profiles(monkeypatch):
    monkeypatch.delenv(POWER_PROFILE_DIR_ENV, raising=False)
    monkeypatch.delenv("PRIMUS_POWER_CAP_W", raising=False)


def _reference_phase(time_ms: float = 10.0, clock_sensitivity: float = 1.0) -> Phase:
    return Phase(name="matrix", time_ms=time_ms, clock_sensitivity=clock_sensitivity, **REFERENCE_ACTIVITY)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["mi300x", "mi325x", "mi350x", "mi355x"])
def test_public_profile_lands_on_tbp_at_reference_activity(arch):
    profile = resolve_power_profile(arch)
    assert profile is not None and profile.source == "public-spec"
    model = ParametricPowerModel(profile)
    assert model.phase_power_w(_reference_phase(), s=1.0) == pytest.approx(profile.tbp_w)
    assert model.phase_power_w(Phase("idle", 1.0, busy=0.0), s=1.0) == pytest.approx(profile.idle_w)


def test_public_archs_and_aliases():
    assert set(public_archs()) == {"mi300x", "mi325x", "mi350x", "mi355x"}
    assert resolve_power_profile("gfx942").tbp_w == resolve_power_profile("mi300x").tbp_w
    assert resolve_power_profile("MI355X").name == "mi355x"


def test_unknown_arch_needs_tbp_and_clock():
    assert resolve_power_profile("future-gpu") is None
    profile = resolve_power_profile("future-gpu", overrides={"tbp_w": 900, "peak_clock_mhz": 2000})
    assert profile.tbp_w == 900 and profile.peak_clock_mhz == 2000
    model = ParametricPowerModel(profile)
    assert model.phase_power_w(_reference_phase()) == pytest.approx(900)


def test_private_profile_dir_overrides_public(tmp_path, monkeypatch):
    (tmp_path / "lab.yaml").write_text(
        "gpu_power_profiles:\n"
        "  lab-gpu:\n"
        "    tbp_w: 500\n"
        "    peak_clock_mhz: 1500\n"
        "    idle_w: 60\n"
        "    alpha: 2.0\n"
        "    coefficients: {busy_w: 100, matrix_w: 300, hbm_w: 80, link_w: 20}\n"
        "    aliases: [gfx999]\n"
    )
    monkeypatch.setenv(POWER_PROFILE_DIR_ENV, str(tmp_path))
    profile = resolve_power_profile("gfx999")
    assert profile.source == "lab.yaml"
    assert (profile.idle_w, profile.alpha, profile.coefficients.matrix_w) == (60, 2.0, 300)


def test_overrides_keep_base_coefficients_unless_tbp_changes():
    base = resolve_power_profile("mi355x")
    same = resolve_power_profile("mi355x", overrides={"alpha": 2.0})
    assert same.coefficients == base.coefficients and same.alpha == 2.0
    lower = resolve_power_profile("mi355x", overrides={"tbp_w": 1000})
    assert ParametricPowerModel(lower).phase_power_w(_reference_phase()) == pytest.approx(1000)


# ---------------------------------------------------------------------------
# Power model and cap solve
# ---------------------------------------------------------------------------


def test_phase_power_increases_with_clock():
    model = ParametricPowerModel(resolve_power_profile("mi355x"))
    phase = _reference_phase(clock_sensitivity=0.7)
    powers = [model.phase_power_w(phase, s) for s in (0.3, 0.5, 0.7, 0.9, 1.0)]
    assert powers == sorted(powers)


def test_cap_at_tbp_does_not_throttle_reference_load():
    model = ParametricPowerModel(resolve_power_profile("mi355x"))
    sol = model.solve([_reference_phase()])
    assert sol.cap_w == 1400
    assert sol.slowdown == pytest.approx(1.0)
    assert sol.feasible


def test_lower_cap_throttles_and_respects_cap():
    model = ParametricPowerModel(resolve_power_profile("mi355x"))
    phases = [_reference_phase(10.0), Phase("comm", 5.0, busy=0.3, hbm=0.2, link=1.0, clock_sensitivity=0.0)]
    prev = 1.0
    for cap in (1200, 1000, 800):
        sol = model.solve(phases, cap_w=cap)
        assert sol.feasible
        assert sol.peak_phase_power_w <= cap * 1.001
        assert sol.slowdown > prev
        prev = sol.slowdown
    # The comm phase is under every cap, so it keeps the full clock.
    assert sol.phases[1].clock_fraction == pytest.approx(1.0)


def test_average_mode_throttles_no_more_than_instantaneous():
    model = ParametricPowerModel(resolve_power_profile("mi355x"))
    phases = [_reference_phase(10.0), Phase("bubble", 10.0, busy=0.0, clock_sensitivity=0.0)]
    inst = model.solve(phases, cap_w=1000, cap_mode="instantaneous")
    avg = model.solve(phases, cap_w=1000, cap_mode="average")
    assert avg.slowdown <= inst.slowdown
    assert avg.avg_power_w <= 1000 * 1.001
    with pytest.raises(ValueError):
        model.solve(phases, cap_mode="bogus")


def test_cap_below_idle_is_infeasible():
    profile = resolve_power_profile("mi355x")
    sol = ParametricPowerModel(profile).solve([_reference_phase()], cap_w=profile.idle_w - 1)
    assert not sol.feasible and sol.warnings


def test_base_clock_lowers_power_and_bounds_the_solve():
    model = ParametricPowerModel(resolve_power_profile("mi355x"))
    full = model.solve([_reference_phase()], cap_w=5000)
    slow = model.solve([_reference_phase()], cap_w=5000, base_clock_fraction=0.8)
    assert slow.uncapped_avg_power_w < full.uncapped_avg_power_w
    assert slow.base_clock_fraction == pytest.approx(0.8)
    assert all(p.clock_fraction <= 0.8 + 1e-9 for p in slow.phases)
    assert slow.slowdown == pytest.approx(1.0)


def test_stretched_time():
    p = Phase("x", 10.0, clock_sensitivity=0.5)
    assert p.stretched_time_ms(0.5) == pytest.approx(15.0)
    assert p.stretched_time_ms(0.8, s0=0.8) == pytest.approx(10.0)


def test_registry_default_and_plugin():
    profile = resolve_power_profile("mi300x")
    assert isinstance(get_power_model("parametric", profile), ParametricPowerModel)

    class Flat(ParametricPowerModel):
        name = "flat-test"

    register_power_model("flat-test", lambda prof, cfg: Flat(prof))
    assert isinstance(get_power_model("flat-test", profile), Flat)
    with pytest.raises(Exception):
        get_power_model("does-not-exist", profile)


# ---------------------------------------------------------------------------
# Workload footprint
# ---------------------------------------------------------------------------


class _StubBackend:
    def simulate_gemm(self, m, n, k, dtype="bf16", trans_a=False, trans_b=False, batch=1):
        return SimpleNamespace(forward_time_ms=1.0, backward_time_ms=0.0)

    def simulate_sdpa(self, batch_size, num_heads, seq_len, head_dim, **kw):
        # Composite helper calling back into the (wrapped) GEMM model.
        self.simulate_gemm(seq_len, seq_len, head_dim)
        return SimpleNamespace(
            forward_time_ms=2.0,
            backward_time_ms=2.0,
            metadata={"fwd_flops": 1e12, "bwd_flops": 2e12, "fwd_bytes": 1e9, "bwd_bytes": 1e9},
        )


def test_op_recorder_footprint_and_uninstall():
    backend = _StubBackend()
    # 1000 TFLOP/s, 1000 GB/s: 1e12 flop/ms over 1 GB/ms.
    rec = OpRecorder(peak_tflops=1000.0, hbm_gbps=1000.0).install(backend, backend)
    backend.simulate_gemm(8192, 8192, 8192)  # 1.1e12 flops in 1 ms -> compute bound
    backend.simulate_sdpa(1, 1, 128, 64)  # outer call only: 3e12 flops in 4 ms
    rec.uninstall()
    assert "simulate_gemm" not in vars(backend) and "simulate_sdpa" not in vars(backend)

    assert rec.num_ops == 2
    fp = rec.footprint()
    assert fp.source == "simulation" and fp.num_ops == 2
    expected_util = (2 * 8192**3 + 3e12) / (5.0 * 1e12)
    assert fp.matrix_utilization == pytest.approx(expected_util)
    assert 0.0 < fp.matrix_hbm_utilization < 1.0
    assert fp.matrix_clock_sensitivity == pytest.approx(1.0)


def test_op_recorder_restores_instance_attribute():
    backend = _StubBackend()
    own = backend.simulate_gemm
    backend.simulate_gemm = own  # instance attribute, as some backends bind them
    rec = OpRecorder(1000.0, 1000.0).install(backend)
    assert backend.simulate_gemm is not own
    rec.uninstall()
    assert backend.simulate_gemm is own


def test_op_recorder_without_ops_uses_defaults():
    assert OpRecorder(1000.0, 1000.0).footprint() == WorkloadFootprint()


def test_iteration_phases_cover_the_iteration():
    bd = IterationBreakdown(
        iteration_ms=100.0, compute_span_ms=80.0, bubble_ms=10.0, exposed_comm_ms=15.0, optimizer_ms=5.0
    )
    phases = {p.name: p for p in build_iteration_phases(bd, WorkloadFootprint(non_matrix_fraction=0.2))}
    assert sum(p.time_ms for p in phases.values()) == pytest.approx(100.0)
    assert phases["matrix"].time_ms == pytest.approx(56.0)
    assert phases["vector"].time_ms == pytest.approx(14.0)
    assert phases["bubble"].busy == 0.0


def test_footprint_round_trip():
    fp = WorkloadFootprint(matrix_utilization=0.4, num_ops=3, source="simulation")
    assert WorkloadFootprint.from_dict(fp.as_dict()) == fp
    assert WorkloadFootprint.from_dict(None) == WorkloadFootprint()


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def test_load_amd_smi_style_csv(tmp_path):
    log = tmp_path / "amdsmi.csv"
    log.write_text(
        "amd-smi metric banner\n"
        "timestamp,gpu,power_cap,current_socket_power,gfx_clk,gfx_activity,umc_activity,vram_used\n"
        "1,0,1400,900 W,2100 MHz,95 %,40 %,100\n"
        "2,0,1400,600 W,1800 MHz,50 %,20 %,100\n"
        "3,0,1400,0,1800,0,0,100\n"
    )
    samples = load_telemetry(str(log))
    assert len(samples) == 2  # zero-power row dropped
    assert samples[0].power_w == 900 and samples[0].sclk_mhz == 2100
    assert samples[0].busy == pytest.approx(0.95) and samples[0].hbm == pytest.approx(0.40)


def test_load_jsonl(tmp_path):
    log = tmp_path / "t.jsonl"
    log.write_text("\n".join(json.dumps({"power_w": 500 + i, "busy": 0.5}) for i in range(3)))
    samples = load_telemetry(str(log))
    assert [s.power_w for s in samples] == [500, 501, 502]


def test_fit_recovers_synthetic_profile(tmp_path):
    truth = GPUPowerProfile(
        name="synthetic",
        tbp_w=1000.0,
        peak_clock_mhz=2000.0,
        idle_w=120.0,
        alpha=2.0,
        coefficients=PowerCoefficients(busy_w=150.0, matrix_w=500.0, hbm_w=180.0, link_w=40.0),
        min_clock_mhz=500.0,
    )
    samples = []
    for i in range(80):
        s = 0.5 + 0.5 * ((i * 7) % 11) / 10
        busy = 0.3 + 0.7 * ((i * 3) % 7) / 6
        matrix = busy * ((i * 5) % 9) / 8
        hbm = ((i * 11) % 13) / 12
        link = ((i * 13) % 5) / 4
        power = (
            truth.idle_w
            + s**truth.alpha * (truth.coefficients.busy_w * busy + truth.coefficients.matrix_w * matrix)
            + truth.coefficients.hbm_w * hbm
            + truth.coefficients.link_w * link
        )
        samples.append(
            TelemetrySample(power_w=power, sclk_mhz=s * 2000.0, busy=busy, matrix=matrix, hbm=hbm, link=link)
        )
    base = resolve_power_profile("synthetic", overrides={"tbp_w": 1000, "peak_clock_mhz": 2000})
    report = fit_power_profile(samples, base)
    p = report.profile
    assert p.alpha == pytest.approx(2.0)
    assert p.idle_w == pytest.approx(120.0, rel=1e-3)
    assert p.coefficients.matrix_w == pytest.approx(500.0, rel=1e-3)
    assert p.coefficients.hbm_w == pytest.approx(180.0, rel=1e-3)
    assert report.rmse_w < 1e-3
    assert "alpha" in report.fitted

    out = tmp_path / "fitted.yaml"
    write_profile_yaml(p, str(out), arch="synthetic")
    reloaded = resolve_power_profile("synthetic", profile_files=[str(out)])
    assert reloaded.alpha == pytest.approx(2.0)
    assert reloaded.coefficients.matrix_w == pytest.approx(500.0, rel=1e-3)


def test_fit_without_matrix_column_scales_jointly():
    base = resolve_power_profile("mi300x")
    samples = [TelemetrySample(power_w=base.idle_w + 300.0 * b, busy=b) for b in (0.2, 0.4, 0.6, 0.8, 1.0)]
    report = fit_power_profile(samples, base)
    c0, c = base.coefficients, report.profile.coefficients
    assert c.matrix_w / c.busy_w == pytest.approx(c0.matrix_w / c0.busy_w)
    assert any("matrix" in w for w in report.warnings)


def test_fit_needs_samples():
    with pytest.raises(ValueError):
        fit_power_profile([TelemetrySample(power_w=100.0)], resolve_power_profile("mi300x"))


# ---------------------------------------------------------------------------
# Infrastructure and capacity planning
# ---------------------------------------------------------------------------


def test_node_power_arithmetic():
    cfg = InfraConfig(psu_efficiency=1.0)
    node = node_power_w(1000.0, cfg)
    assert node["node_w"] == pytest.approx(8 * 1000 + 2 * 360 + 800 + 9 * 25)


def test_switch_count():
    cfg = InfraConfig(switch_radix=64)
    assert scaleout_switch_count(0, cfg) == 0
    assert scaleout_switch_count(8, cfg) == 1  # 64 ports fit one switch
    assert scaleout_switch_count(64, cfg) == 16 + 8  # 512 ports: 16 leaves, 8 spines


def test_cluster_power_racks_and_pue():
    cfg = InfraConfig(rack_power_kw=40.0, pue=1.5)
    cp = cluster_power(10, 1000.0, cfg)
    assert cp.facility_w == pytest.approx(cp.it_w * 1.5)
    per_rack = int(40_000 // cp.node["node_w"])
    assert cp.racks == -(-10 // per_rack)


def test_capacity_plan_lower_cap_buys_more_nodes():
    cfg = InfraConfig()

    def at_cap(cap):
        return {"tokens_per_s_per_gpu": 1000.0 * (cap / 1400.0) ** 0.5, "avg_power_w": cap * 0.9}

    opts = capacity_plan(1_000_000.0, [1400.0, 1000.0, 700.0], at_cap, cfg, node_multiple=2)
    nodes = [o.num_nodes for o in opts]
    assert nodes == sorted(nodes) and nodes[0] < nodes[-1]
    assert all(o.num_nodes % 2 == 0 for o in opts)
    assert all(o.facility_w <= 1_000_000.0 for o in opts)
    capped = capacity_plan(1_000_000.0, [700.0], at_cap, cfg, max_nodes=3)
    assert capped[0].num_nodes == 3


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _breakdown():
    return IterationBreakdown(
        iteration_ms=1000.0,
        compute_span_ms=900.0,
        bubble_ms=50.0,
        exposed_comm_ms=60.0,
        optimizer_ms=40.0,
        overlapped_comm_ms=200.0,
    )


def test_project_power_report(capsys):
    fp = WorkloadFootprint(matrix_utilization=0.75, matrix_hbm_utilization=0.5, source="simulation")
    uncapped = project_power(
        _breakdown(), 8 * 4096 * 4, 16, 2, 8, gpu_arch="mi355x", footprint=fp, cap_w=5000
    )
    capped = project_power(_breakdown(), 8 * 4096 * 4, 16, 2, 8, gpu_arch="mi355x", footprint=fp, cap_w=900)
    assert uncapped.solution.slowdown == pytest.approx(1.0)
    assert capped.solution.slowdown > 1.0
    assert capped.capped_tokens_per_s_per_gpu < uncapped.capped_tokens_per_s_per_gpu
    assert capped.solution.peak_phase_power_w <= 900 * 1.001
    d = capped.as_dict()
    assert d["gpu_arch"] == "mi355x" and d["cluster"]["num_nodes"] == 2
    json.dumps(d)  # result dict is serializable

    capped.print()
    out = capsys.readouterr().out
    for line in (
        "Uncapped Avg Power/GPU:",
        "Power Cap/GPU: 900 W",
        "Power-Capped Step Time:",
        "Power-Capped Throughput:",
        "Energy/Token (GPU):",
        "Facility Power:",
        "Power Feasible: yes",
    ):
        assert line in out


def test_project_power_default_cap_is_tbp_and_env_override(monkeypatch):
    rep = project_power(_breakdown(), 1000, 8, 1, 8, gpu_arch="mi300x")
    assert rep.solution.cap_w == 750
    monkeypatch.setenv("PRIMUS_POWER_CAP_W", "600")
    assert project_power(_breakdown(), 1000, 8, 1, 8, gpu_arch="mi300x").solution.cap_w == 600


def test_project_power_base_clock_from_projection():
    full = project_power(_breakdown(), 1000, 8, 1, 8, gpu_arch="mi355x", cap_w=5000)
    lower = project_power(_breakdown(), 1000, 8, 1, 8, gpu_arch="mi355x", cap_w=5000, base_clock_mhz=1920)
    assert lower.solution.base_clock_fraction == pytest.approx(0.8)
    assert lower.solution.uncapped_avg_power_w < full.solution.uncapped_avg_power_w


def test_power_config_yaml_with_capacity(tmp_path, capsys):
    cfg_path = tmp_path / "power.yaml"
    cfg_path.write_text(
        "power:\n"
        "  gpu_arch: mi355x\n"
        "  cap_w: 1100\n"
        "  cap_mode: average\n"
        "  footprint: {matrix_utilization: 0.6}\n"
        "infrastructure:\n"
        "  pue: 1.2\n"
        "  rack_power_kw: 130\n"
        "capacity:\n"
        "  facility_power_kw: 2000\n"
        "  caps_w: [1400, 1100, 900]\n"
    )
    cfg = PowerConfig.from_yaml(str(cfg_path))
    assert (cfg.gpu_arch, cfg.cap_w, cfg.cap_mode) == ("mi355x", 1100, "average")
    rep = project_power(_breakdown(), 1000, 8, 1, 8, config=cfg)
    assert rep.solution.cap_mode == "average" and rep.solution.cap_w == 1100
    assert rep.footprint.matrix_utilization == pytest.approx(0.6)
    assert rep.infra.pue == pytest.approx(1.2)
    assert [o["gpu_cap_w"] for o in rep.capacity] == [1400, 1100, 900]
    rep.print()
    assert "Best cap under budget:" in capsys.readouterr().out


def test_fitted_profile_accepted_as_power_config(tmp_path):
    path = tmp_path / "fit.yaml"
    path.write_text(
        "gpu_power_profiles:\n  mi300x:\n    tbp_w: 750\n    peak_clock_mhz: 2100\n    idle_w: 90\n"
    )
    cfg = PowerConfig.from_yaml(str(path))
    rep = project_power(_breakdown(), 1000, 8, 1, 8, config=cfg)
    assert rep.profile.idle_w == 90 and rep.profile.name == "mi300x"


def test_project_power_unknown_arch_returns_none():
    assert project_power(_breakdown(), 1000, 8, 1, 8, gpu_arch="future-gpu") is None


def test_projection_falls_back_to_calibrated_device(monkeypatch):
    from primus.core.projection.performance_projection import projection as proj

    monkeypatch.delenv("PRIMUS_GPU_CLOCK_MHZ", raising=False)
    results = {
        "_hardware_calibration": {
            "device": {"profile_key": "mi355x", "gcn_arch": "gfx950"},
            "compute": {"effective_clock_mhz": 2100.0},
        }
    }
    bare = SimpleNamespace(gpu_arch=None, gpu_clock_mhz=None)
    assert proj._projected_gpu_arch(bare, results) == "mi355x"
    assert proj._projected_base_clock_mhz(bare, results) == 2100.0
    explicit = SimpleNamespace(gpu_arch="mi300x", gpu_clock_mhz=1800)
    assert proj._projected_gpu_arch(explicit, results) == "mi300x"
    assert proj._projected_base_clock_mhz(explicit, results) == 1800.0
    assert proj._projected_gpu_arch(bare, {}) is None
    assert proj._projected_base_clock_mhz(bare, {}) is None


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def _projection_parser():
    from primus.cli.subcommands import projection

    parser = argparse.ArgumentParser(prog="primus")
    projection.register_subcommand(parser.add_subparsers(dest="cmd"))
    return parser


def test_performance_power_flags():
    parser = _projection_parser()
    args = parser.parse_args(["projection", "performance", "--config", "x"])
    assert args.power_report is False and args.power_cap_w is None
    args = parser.parse_args(
        ["projection", "performance", "--config", "x", "--power-cap-w", "900", "--power-cap-mode", "average"]
    )
    assert args.power_cap_w == 900.0 and args.power_cap_mode == "average"
    with pytest.raises(SystemExit):
        parser.parse_args(["projection", "performance", "--config", "x", "--power-cap-mode", "bogus"])


def test_power_fit_suite_runs(tmp_path, capsys):
    log = tmp_path / "t.csv"
    log.write_text(
        "power_w,busy,hbm\n" + "\n".join(f"{200 + 400 * b},{b},{b / 2}" for b in (0.2, 0.5, 0.8, 1.0))
    )
    out = tmp_path / "fit.yaml"
    parser = _projection_parser()
    args = parser.parse_args(
        ["projection", "power-fit", "--telemetry", str(log), "--gpu-arch", "mi300x", "--out", str(out)]
    )
    from primus.core.projection.power.cli import run_power_fit

    assert run_power_fit(args) == 0
    assert out.is_file() and "Fitted 4 samples" in capsys.readouterr().out
