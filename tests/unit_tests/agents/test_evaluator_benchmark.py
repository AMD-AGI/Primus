###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Unit tests for the tuning-agent evaluator's *benchmark* path.

The benchmark path was rewritten to share a single bench artifact between
``projection performance --save-benchmark`` and
``projection memory --memory-mode benchmark --load-benchmark`` (one bench,
two projections).  These tests verify:

  1. ``--save-benchmark`` is added to the perf cmd.
  2. ``--memory-mode benchmark`` and ``--load-benchmark <same path>`` are
     added to the memory cmd.
  3. The artifact path passed to perf is the same one consumed by memory.
  4. Memory's ``Point estimate (per rank): X GB`` is parsed into
     ``memory_per_gpu_gb`` and ``memory_source='benchmark_point'``.
  5. The OOM-fits decision uses the upper bound when available.
  6. The temp artifact is cleaned up (best-effort) after the path returns.
  7. ``_build_env`` strips the tuning-agent subtree from PYTHONPATH so the
     subprocess can't pick up a nested ``primus/`` package that shadows
     the outer one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


# Skip the whole module if the tuning-agent isn't importable in this env
# (e.g. running unit_tests against the outer Primus alone without the
# tuning-agent subtree on PYTHONPATH).
pytest.importorskip("primus.agents.tuning_agent.evaluator")

from primus.agents.tuning_agent import evaluator as ev_mod  # noqa: E402
from primus.agents.tuning_agent.evaluator import (  # noqa: E402
    EvalResult,
    Evaluator,
    _build_env,
    _build_memory_cmd,
    _build_perf_cmd,
    _parse_metrics,
)
from primus.agents.tuning_agent.legality import TrialConfig  # noqa: E402


def _stub_legality(monkeypatch):
    """Bypass derive_legality / validate / derived_dp for these tests.

    ``derive_legality`` is imported *lazily* inside ``_evaluate`` (so we
    have to patch the source module), while ``validate`` and
    ``derived_dp`` are imported at module top into ``evaluator`` (so we
    patch the evaluator module's namespace).
    """
    monkeypatch.setattr(
        "primus.agents.tuning_agent.legality.derive_legality",
        lambda arch, cluster: SimpleNamespace(),
    )
    monkeypatch.setattr(ev_mod, "validate", lambda *a, **kw: (True, None))
    monkeypatch.setattr(ev_mod, "derived_dp", lambda *a, **kw: 1)


# ─────────────────────────────────────────────────────────────────────────────
# Tiny stand-ins for AgentConfig + ArchitectureRecord
# ─────────────────────────────────────────────────────────────────────────────


def _make_agent_cfg(out_dir: Path):
    return SimpleNamespace(
        target_cluster=SimpleNamespace(
            num_nodes=1,
            gpus_per_node=8,
            gpu_arch="mi355x",
            hardware_config="examples/hardware_configs/mi355x.yaml",
            gpu_clock_mhz=None,
        ),
        benchmark_host=SimpleNamespace(
            has_gpu=True,
            benchmark_gpus=8,
        ),
        optimization=SimpleNamespace(
            objective="tokens_per_s_per_gpu",
            hbm_capacity_gb=288.0,
            memory_safety_margin=0.10,
        ),
        out_dir=out_dir,
    )


def _make_arch():
    return SimpleNamespace(
        model_name="mixtral_8x7B",
        num_layers=32,
        hidden_size=4096,
        is_moe=True,
        num_experts=8,
        moe_router_topk=2,
        workload_path="/dev/null",
        attention_type="standard",
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_layout=None,
        fp8=None,
    )


def _make_trial_cfg() -> TrialConfig:
    return TrialConfig(
        tp=1, pp=1, ep=8, cp=1, mbs=2, gbs=32, vpp=None,
        pp_schedule="auto", enable_zero_bubble=None,
        recompute_granularity=None, recompute_num_layers=0,
        cross_entropy_loss_fusion=None, use_torch_fsdp2=None,
        use_distributed_optimizer=None, use_turbo_deepep=None,
        sync_free_stage=None, target_ep_size=None, fp8=None,
        overlap_grad_reduce=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _build_*_cmd unit tests
# ─────────────────────────────────────────────────────────────────────────────


def test_build_perf_cmd_appends_save_benchmark(tmp_path):
    agent_cfg = _make_agent_cfg(tmp_path)
    cfg = _make_trial_cfg()
    primus_root = Path("/fake/primus")
    save = tmp_path / "art.json"
    cmd = _build_perf_cmd(
        tmp_path / "trial.yaml", cfg, agent_cfg, "benchmark", primus_root,
        save_benchmark=save,
    )
    assert "--save-benchmark" in cmd
    idx = cmd.index("--save-benchmark")
    assert cmd[idx + 1] == str(save)
    # ``--benchmark-gpus`` must remain present too.
    assert "--benchmark-gpus" in cmd


def test_build_perf_cmd_skips_save_benchmark_in_simulate(tmp_path):
    agent_cfg = _make_agent_cfg(tmp_path)
    cfg = _make_trial_cfg()
    cmd = _build_perf_cmd(
        tmp_path / "trial.yaml", cfg, agent_cfg, "simulate", Path("/fake/primus"),
        save_benchmark=tmp_path / "ignored.json",
    )
    # save-benchmark only makes sense in benchmark mode; the helper must
    # not append it for simulate.
    assert "--save-benchmark" not in cmd
    assert "--benchmark-gpus" not in cmd


def test_build_memory_cmd_with_load_benchmark(tmp_path):
    agent_cfg = _make_agent_cfg(tmp_path)
    art = tmp_path / "art.json"
    cmd = _build_memory_cmd(
        tmp_path / "trial.yaml", _make_trial_cfg(), agent_cfg, Path("/fake/primus"),
        memory_mode="benchmark",
        load_benchmark=art,
        safety_margin=0.10,
    )
    assert "projection" in cmd and "memory" in cmd
    assert cmd[cmd.index("--memory-mode") + 1] == "benchmark"
    assert cmd[cmd.index("--load-benchmark") + 1] == str(art)
    assert cmd[cmd.index("--memory-safety-margin") + 1] == "0.1000"
    assert cmd[cmd.index("--target-nodes") + 1] == "1"


def test_build_memory_cmd_default_simulate_is_clean(tmp_path):
    agent_cfg = _make_agent_cfg(tmp_path)
    cmd = _build_memory_cmd(
        tmp_path / "trial.yaml", _make_trial_cfg(), agent_cfg, Path("/fake/primus"),
    )
    # Default invocation: no --memory-mode (=> simulate), no
    # --load-benchmark, no --memory-safety-margin.
    assert "--memory-mode" not in cmd
    assert "--load-benchmark" not in cmd
    assert "--memory-safety-margin" not in cmd
    # --target-nodes is always passed so simulate-mode's get_dp_size sees
    # the real target shape (existing behaviour).
    assert "--target-nodes" in cmd


# ─────────────────────────────────────────────────────────────────────────────
# _build_env PYTHONPATH discipline
# ─────────────────────────────────────────────────────────────────────────────


def test_build_env_promotes_primus_root_and_strips_tuning_agent(tmp_path):
    """primus_root must come first; nested tuning-agent path must be removed.

    Without this, the subprocess's ``primus.cli.subcommands.projection``
    can resolve to the tuning-agent's nested copy (which lacks the new
    ``--save-benchmark`` flag) instead of the outer Primus.
    """
    primus_root = Path("/outer/primus")
    nested = "/outer/primus/primus/agents/tuning-agent"
    saved = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        f"{nested}{os.pathsep}/some/other/lib{os.pathsep}{primus_root}"
    )
    try:
        env = _build_env(_make_agent_cfg(tmp_path), primus_root)
    finally:
        if saved is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = saved

    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(primus_root), f"primus_root must be first; got {parts}"
    assert nested not in parts, "tuning-agent subtree must be stripped"
    assert "/some/other/lib" in parts, "non-primus paths are preserved"


# ─────────────────────────────────────────────────────────────────────────────
# _parse_metrics on bench-mode memory output
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_metrics_recognises_bench_point_and_upper():
    sample = """
[Primus:Memory Projection] Per-rank peak memory at 1 nodes × 8 GPUs
============================================================
  Bench measurements (rank 0, world_size=8):
    Global peak allocated:  82.31 GB
  Analytical at target (world_size=8):
    Params (bf16):          5.00 GB
  ─────────────────────────────────────────────────────────────────
  Point estimate (per rank): 84.4 GB
  Upper bound    (per rank): 88.6 GB
  ─────────────────────────────────────────────────────────────────
"""
    m = _parse_metrics(sample)
    assert m["memory_source"] == "benchmark_point"
    assert m["memory_per_gpu_gb"] == 84.4
    assert m["memory_per_gpu_gb_upper"] == 88.6


def test_parse_metrics_falls_back_to_simulate_when_bench_absent():
    sample = """
[Primus:Projection] Memory Projection Summary on Rank 0:
  Params: 9.0 Billion
  Param+Optimizer Memory: 117.86 GB
  Activation Memory (per batch size 2, seq len 4096): 10.06 GB
  Projected Total Memory: 127.92 GB
"""
    m = _parse_metrics(sample)
    assert m["memory_source"] == "simulate"
    assert m["memory_per_gpu_gb"] == 127.92
    assert "memory_per_gpu_gb_upper" not in m


def test_parse_metrics_prefers_bench_over_simulate_when_both_present():
    """``--memory-mode both`` produces both blocks; bench wins."""
    sample = """
... simulate block ...
  Projected Total Memory: 127.92 GB
... benchmark block ...
  Point estimate (per rank): 84.4 GB
  Upper bound    (per rank): 88.6 GB
"""
    m = _parse_metrics(sample)
    assert m["memory_per_gpu_gb"] == 84.4
    assert m["memory_source"] == "benchmark_point"


# ─────────────────────────────────────────────────────────────────────────────
# _evaluate_benchmark integration (mocked _run)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def evaluator(tmp_path):
    e = Evaluator(
        agent_cfg=_make_agent_cfg(tmp_path),
        arch=_make_arch(),
        primus_root=tmp_path,  # avoid touching real filesystem
        mode="full",
    )
    return e


def _make_perf_stdout() -> str:
    return (
        "[Primus:Performance Projection] Initializing...\n"
        "Iteration Time: 350.0 ms\n"
        "Tokens/s/GPU: 12000.0\n"
        "TFLOPs/s/GPU: 800.0\n"
    )


def _make_mem_stdout() -> str:
    return (
        "[Primus:Memory Projection] Per-rank peak memory at 1 nodes × 8 GPUs\n"
        "  Point estimate (per rank): 100.0 GB\n"
        "  Upper bound    (per rank): 110.0 GB\n"
    )


def test_evaluate_benchmark_shares_artifact_and_parses_both(evaluator, tmp_path, monkeypatch):
    """Perf saves an artifact; memory loads from the same path; both metrics parsed."""
    captured = {"perf_cmd": None, "mem_cmd": None, "artifact_at_perf": None}

    def fake_run(cmd, cwd, env, timeout):
        # Identify which subprocess this is by the arg list
        if "performance" in cmd:
            captured["perf_cmd"] = list(cmd)
            # The save-benchmark path should be the one we'll later read.
            idx = cmd.index("--save-benchmark")
            artifact = Path(cmd[idx + 1])
            captured["artifact_at_perf"] = artifact
            # Simulate the perf bench writing the artifact.
            artifact.write_text(json.dumps({"schema_version": 2}))
            return 0, _make_perf_stdout(), 1.0
        else:
            captured["mem_cmd"] = list(cmd)
            return 0, _make_mem_stdout(), 0.5

    monkeypatch.setattr(ev_mod, "_run", fake_run)
    # Bypass write_trial_yaml: just create a placeholder.
    monkeypatch.setattr(
        ev_mod, "write_trial_yaml",
        lambda arch, cfg, out, tag: out / f"trial_{tag}.yaml",
    )
    # Stub legality helpers (imported into evaluator's namespace) — the
    # arch stub here only carries fields the evaluator's bench path
    # actually reads.
    _stub_legality(monkeypatch)

    cfg = _make_trial_cfg()
    result = evaluator.evaluate_benchmark(cfg, "test_tag")

    # Both subprocesses ran.
    assert captured["perf_cmd"] is not None and captured["mem_cmd"] is not None
    # Perf cmd carries --save-benchmark.
    assert "--save-benchmark" in captured["perf_cmd"]
    # Memory cmd carries --memory-mode benchmark + --load-benchmark <same path>.
    mem = captured["mem_cmd"]
    assert mem[mem.index("--memory-mode") + 1] == "benchmark"
    artifact = captured["artifact_at_perf"]
    assert mem[mem.index("--load-benchmark") + 1] == str(artifact)
    assert mem[mem.index("--memory-safety-margin") + 1] == "0.1000"

    # Both projections' metrics landed on the result.
    assert result.legal is True
    assert result.tokens_per_s_per_gpu == 12000.0
    assert result.iteration_ms == 350.0
    assert result.memory_per_gpu_gb == 100.0
    assert result.memory_per_gpu_gb_upper == 110.0
    assert result.memory_source == "benchmark_point"
    # OOM decision uses the upper bound and 100 GB upper < 288*0.9=259 cap → legal.
    assert result.memory_per_gpu_gb_adjusted == 110.0
    # Temp artifact cleaned up.
    assert not artifact.exists()


def test_evaluate_benchmark_rejects_when_upper_exceeds_cap(evaluator, tmp_path, monkeypatch):
    """Upper bound > HBM*0.9 → trial is illegal with a clear cap-exceeded reason."""

    def fake_run(cmd, cwd, env, timeout):
        if "performance" in cmd:
            idx = cmd.index("--save-benchmark")
            Path(cmd[idx + 1]).write_text("{}")
            return 0, _make_perf_stdout(), 1.0
        # Memory returns an upper bound > 259 GB cap (288 × 0.9).
        return 0, (
            "  Point estimate (per rank): 270.0 GB\n"
            "  Upper bound    (per rank): 285.0 GB\n"
        ), 0.5

    monkeypatch.setattr(ev_mod, "_run", fake_run)
    monkeypatch.setattr(
        ev_mod, "write_trial_yaml",
        lambda arch, cfg, out, tag: out / f"trial_{tag}.yaml",
    )
    _stub_legality(monkeypatch)

    result = evaluator.evaluate_benchmark(_make_trial_cfg(), "cap_test")
    assert result.legal is False
    assert "cap" in (result.reason or "")
    # The decision number is the upper bound, not the point estimate.
    assert result.memory_per_gpu_gb_adjusted == 285.0


def test_evaluate_benchmark_handles_perf_failure(evaluator, tmp_path, monkeypatch):
    """Perf bench failing must mark trial illegal and not call memory."""
    n_calls = {"perf": 0, "mem": 0}

    def fake_run(cmd, cwd, env, timeout):
        if "performance" in cmd:
            n_calls["perf"] += 1
            return 1, "boom\n", 0.1
        n_calls["mem"] += 1
        return 0, _make_mem_stdout(), 0.1

    monkeypatch.setattr(ev_mod, "_run", fake_run)
    monkeypatch.setattr(
        ev_mod, "write_trial_yaml",
        lambda arch, cfg, out, tag: out / f"trial_{tag}.yaml",
    )
    _stub_legality(monkeypatch)

    result = evaluator.evaluate_benchmark(_make_trial_cfg(), "fail_perf")
    assert result.legal is False
    assert n_calls["perf"] == 1 and n_calls["mem"] == 0
    assert "performance --benchmark failed" in (result.reason or "")
