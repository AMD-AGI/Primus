###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tuning agent under a customer power cap: config, perf cmd, parsing, policy, ranking."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("primus.agents.tuning_agent.evaluator")

from primus.agents.tuning_agent.config import (  # noqa: E402
    load_config,
    power_prompt_dict,
)
from primus.agents.tuning_agent.evaluator import (  # noqa: E402
    EvalResult,
    _apply_power_policy,
    _build_perf_cmd,
    _parse_metrics,
)
from primus.agents.tuning_agent.history import History  # noqa: E402
from primus.agents.tuning_agent.legality import TrialConfig  # noqa: E402
from primus.core.projection.power import (  # noqa: E402
    IterationBreakdown,
    WorkloadFootprint,
    project_power,
)


def _trial() -> TrialConfig:
    return TrialConfig(
        tp=1,
        pp=1,
        ep=8,
        cp=1,
        mbs=2,
        gbs=32,
        vpp=None,
        pp_schedule="auto",
        enable_zero_bubble=None,
        recompute_granularity=None,
        recompute_num_layers=0,
        cross_entropy_loss_fusion=None,
        use_torch_fsdp2=None,
        use_distributed_optimizer=None,
        use_turbo_deepep=None,
        sync_free_stage=None,
        target_ep_size=None,
        fp8=None,
        overlap_grad_reduce=True,
    )


def _agent_cfg(power_cap_w=None, power_config=None, power_cap_mode=None, policy="throttle"):
    return SimpleNamespace(
        target_cluster=SimpleNamespace(
            num_nodes=2,
            gpus_per_node=8,
            gpu_arch="mi355x",
            hardware_config=None,
            gpu_clock_mhz=None,
            power_cap_w=power_cap_w,
            power_config=power_config,
            power_cap_mode=power_cap_mode,
        ),
        benchmark_host=SimpleNamespace(has_gpu=False, benchmark_gpus=0),
        optimization=SimpleNamespace(objective="capped_tokens_per_s_per_gpu", power_policy=policy),
    )


def _report_stdout(capsys, cap_w: float):
    bd = IterationBreakdown(
        iteration_ms=1000.0, compute_span_ms=900.0, bubble_ms=50.0, exposed_comm_ms=60.0, optimizer_ms=40.0
    )
    fp = WorkloadFootprint(matrix_utilization=0.75, matrix_hbm_utilization=0.5)
    rep = project_power(bd, 8 * 4096 * 4, 16, 2, 8, gpu_arch="mi355x", footprint=fp, cap_w=cap_w)
    capsys.readouterr()
    # Surrounding perf lines the existing regexes also read.
    print("Iteration Time: 1000.000 ms")
    print("Tokens/s/GPU: 8,192")
    rep.print()
    return capsys.readouterr().out, rep


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def _write_cfg(tmp_path: Path, extra_tc: str = "", extra_opt: str = "") -> Path:
    tc = tmp_path / "target_cluster.yaml"
    tc.write_text(
        "target_cluster:\n  name: c\n  num_nodes: 2\n  gpus_per_node: 8\n  gpu_arch: mi355x\n"
        + extra_tc
        + "optimization:\n  hbm_capacity_gb: 288\n"
        + extra_opt
    )
    return tc


def test_power_cap_switches_default_objective(tmp_path):
    cfg = load_config(
        _write_cfg(tmp_path, "  power_cap_w: 1000\n  power_cap_mode: average\n"), tmp_path / "w.yaml"
    )
    assert cfg.target_cluster.power_cap_w == 1000.0
    assert cfg.target_cluster.power_cap_mode == "average"
    assert cfg.target_cluster.power_enabled
    assert cfg.optimization.objective == "capped_tokens_per_s_per_gpu"
    assert cfg.optimization.power_policy == "throttle"
    blob = power_prompt_dict(cfg)
    assert blob["power_cap_w_per_gpu"] == 1000.0 and blob["objective"] == "capped_tokens_per_s_per_gpu"


def test_no_power_keeps_defaults(tmp_path):
    cfg = load_config(_write_cfg(tmp_path), tmp_path / "w.yaml")
    assert not cfg.target_cluster.power_enabled
    assert cfg.optimization.objective == "tokens_per_s_per_gpu"
    assert power_prompt_dict(cfg) == {}


def test_explicit_objective_wins_and_policy_validated(tmp_path):
    cfg = load_config(
        _write_cfg(
            tmp_path, "  power_cap_w: 1000\n", "  objective: energy_per_token_j\n  power_policy: reject\n"
        ),
        tmp_path / "w.yaml",
    )
    assert cfg.optimization.objective == "energy_per_token_j"
    assert cfg.optimization.power_policy == "reject"
    with pytest.raises(ValueError, match="power_policy"):
        load_config(
            _write_cfg(tmp_path, "  power_cap_w: 1000\n", "  power_policy: ignore\n"), tmp_path / "w.yaml"
        )


# ---------------------------------------------------------------------------
# perf command
# ---------------------------------------------------------------------------


def test_perf_cmd_carries_power_flags(tmp_path):
    cmd = _build_perf_cmd(
        tmp_path / "t.yaml",
        _trial(),
        _agent_cfg(power_cap_w=1000.0, power_config="power.yaml", power_cap_mode="average"),
        "simulate",
        Path("/fake/primus"),
    )
    assert "--power-report" in cmd
    assert cmd[cmd.index("--power-cap-w") + 1] == "1000.0"
    assert cmd[cmd.index("--power-config") + 1] == "power.yaml"
    assert cmd[cmd.index("--power-cap-mode") + 1] == "average"


def test_perf_cmd_without_power(tmp_path):
    cmd = _build_perf_cmd(tmp_path / "t.yaml", _trial(), _agent_cfg(), "simulate", Path("/fake/primus"))
    assert not any(a.startswith("--power") for a in cmd)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_power_report_lines(capsys):
    out, rep = _report_stdout(capsys, cap_w=900)
    m = _parse_metrics(out)
    # Power lines must not clobber the uncapped perf metrics.
    assert m["iteration_ms"] == pytest.approx(1000.0)
    assert m["tokens_per_s_per_gpu"] == pytest.approx(8192)
    s = rep.solution
    assert m["avg_power_w"] == pytest.approx(s.avg_power_w, abs=0.1)
    assert m["uncapped_avg_power_w"] == pytest.approx(s.uncapped_avg_power_w, abs=0.1)
    assert m["uncapped_peak_power_w"] == pytest.approx(s.uncapped_peak_phase_power_w, abs=0.1)
    assert m["capped_iteration_ms"] == pytest.approx(s.iteration_ms, abs=1e-3)
    assert m["power_slowdown"] == pytest.approx(s.slowdown, abs=1e-3)
    assert m["capped_tokens_per_s_per_gpu"] == pytest.approx(rep.capped_tokens_per_s_per_gpu, abs=1.0)
    assert m["energy_per_token_j"] == pytest.approx(rep.energy_per_token_j, abs=1e-4)
    assert m["tokens_per_joule_per_gpu"] == pytest.approx(rep.tokens_per_joule_per_gpu, abs=1e-3)
    assert m["facility_power_kw"] > 0
    assert m["power_feasible"] is True


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


def _result(**kw) -> EvalResult:
    r = EvalResult(legal=True, reason="", source="simulate")
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def test_throttle_policy_keeps_throttled_configs():
    r = _apply_power_policy(
        _result(capped_tokens_per_s_per_gpu=900.0, uncapped_peak_power_w=1300.0, power_feasible=True),
        _agent_cfg(power_cap_w=1000.0),
    )
    assert r.legal


def test_reject_policy_drops_configs_over_the_cap():
    r = _apply_power_policy(
        _result(capped_tokens_per_s_per_gpu=900.0, uncapped_peak_power_w=1300.0, power_feasible=True),
        _agent_cfg(power_cap_w=1000.0, policy="reject"),
    )
    assert not r.legal and "power_policy=reject" in r.reason


def test_infeasible_cap_is_always_illegal():
    r = _apply_power_policy(
        _result(capped_tokens_per_s_per_gpu=100.0, power_feasible=False), _agent_cfg(power_cap_w=150.0)
    )
    assert not r.legal and "infeasible" in r.reason


def test_policy_is_noop_without_power():
    r = _apply_power_policy(_result(uncapped_peak_power_w=5000.0), _agent_cfg(policy="reject"))
    assert r.legal


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------


def test_best_ranks_by_capped_throughput_and_minimizes_energy(tmp_path):
    h = History(path=tmp_path / "h.jsonl")
    # Config A: faster uncapped, but throttled harder under the cap.
    h.add(
        {"tp": 1},
        _result(tokens_per_s_per_gpu=1100.0, capped_tokens_per_s_per_gpu=900.0, energy_per_token_j=1.2),
    )
    h.add(
        {"tp": 2},
        _result(tokens_per_s_per_gpu=1000.0, capped_tokens_per_s_per_gpu=950.0, energy_per_token_j=1.0),
    )
    assert h.best("tokens_per_s_per_gpu").config == {"tp": 1}
    assert h.best("capped_tokens_per_s_per_gpu").config == {"tp": 2}
    assert h.best("energy_per_token_j").config == {"tp": 2}
    assert "capped_tps=" in h.summary_for_llm()
