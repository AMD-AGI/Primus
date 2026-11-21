###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Expected vs. ceiling bound and the per-lever headroom report."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from primus.core.projection import ceiling
from primus.core.projection.ceiling import (
    LEVERS,
    at_ceiling,
    ceiling_levers,
    check_simulate_only,
    format_headroom,
    headroom,
    parse_levers,
)

REPO = Path(__file__).resolve().parents[4]


def test_parse_levers_defaults_dedupes_and_validates():
    assert parse_levers(None) == LEVERS
    assert parse_levers("") == LEVERS
    assert parse_levers("fusion, overlap,fusion") == ("fusion", "overlap")
    assert parse_levers(["launch"]) == ("launch",)
    with pytest.raises(ValueError, match="unknown ceiling lever"):
        parse_levers("overlap,kernel")


def test_ceiling_levers_scope_nests_and_restores():
    assert not any(at_ceiling(lv) for lv in LEVERS)
    with ceiling_levers(("overlap",)):
        assert at_ceiling("overlap") and not at_ceiling("fusion")
        with ceiling_levers(()):
            assert not at_ceiling("overlap")
        assert at_ceiling("overlap")
    assert ceiling.active_levers() == ()


@pytest.mark.parametrize(
    "args",
    [
        SimpleNamespace(profiling_mode="benchmark"),
        SimpleNamespace(profiling_mode="both"),
        SimpleNamespace(profiling_mode="simulate", load_benchmark="bench.json"),
    ],
)
def test_ceiling_refuses_measured_layer_times(args):
    with pytest.raises(ValueError, match="--profiling-mode simulate"):
        check_simulate_only(args)


def test_headroom_sorts_levers_and_waterfall_ends_at_ceiling():
    # Iteration ms saved by each lever; levers combine additively in this fake.
    saved = {"overlap": 100.0, "fusion": 250.0, "launch": 10.0}
    calls = []

    def project(active):
        calls.append(tuple(active))
        it = 1000.0 - sum(saved[lv] for lv in active)
        return {"iteration_time_ms": it, "tokens_per_sec_per_gpu": 1e6 / it}

    report = headroom(project, LEVERS, measured_tokens_per_sec_per_gpu=1200.0)

    assert [r["lever"] for r in report["lever_gains"]] == ["fusion", "overlap", "launch"]
    assert [r["levers"] for r in report["waterfall"]] == [
        ["fusion"],
        ["fusion", "overlap"],
        ["fusion", "overlap", "launch"],
    ]
    assert report["ceiling"]["iteration_time_ms"] == pytest.approx(640.0)
    assert report["headroom_pct"] == pytest.approx(100.0 * (1000.0 / 640.0 - 1.0))
    assert report["measured"]["of_expected"] == pytest.approx(1.2)
    assert not report["measured"]["above_ceiling"]
    assert calls[0] == ()
    assert "Waterfall" in format_headroom(report) and "within ceiling" in format_headroom(report)


def test_headroom_flags_measurement_above_ceiling():
    def project(active):
        return {"iteration_time_ms": 100.0, "tokens_per_sec_per_gpu": 50.0}

    report = headroom(project, ("overlap",), measured_tokens_per_sec_per_gpu=60.0)
    assert report["measured"]["above_ceiling"]
    assert "ABOVE CEILING" in format_headroom(report)


def test_headroom_needs_a_multinode_result():
    with pytest.raises(RuntimeError, match="no multinode result"):
        headroom(lambda active: None, ("overlap",))


def test_layer_comm_hidden_behind_compute_only_at_overlap_ceiling():
    from primus.core.projection.module_profilers.transformer_layer import _layer_comm_ms

    assert _layer_comm_ms(5.0, 3.0, expected_ms=4.0) == 4.0
    with ceiling_levers(("overlap",)):
        assert _layer_comm_ms(5.0, 3.0, expected_ms=4.0) == pytest.approx(2.0)
        assert _layer_comm_ms(2.0, 3.0, expected_ms=4.0) == 0.0


def test_fusion_removes_expert_cast_traffic_and_launch_removes_launches():
    from primus.core.projection.module_profilers.moe_mlp import MoEMLPProfiler

    prof = MoEMLPProfiler.__new__(MoEMLPProfiler)
    kw = dict(
        topk_tokens=8192,
        hidden_size=2048,
        moe_ffn=768,
        num_local_experts=16,
        swiglu=True,
        peak_hbm=6000.0,
        batched=False,
    )
    fwd, bwd = prof._expert_overhead_first_principles(**kw)
    with ceiling_levers(("fusion",)):
        assert prof._expert_overhead_first_principles(**kw) == (0.0, 0.0)
    with ceiling_levers(("launch",)):
        lfwd, lbwd = prof._expert_overhead_first_principles(**kw)
    assert lfwd == pytest.approx(fwd - 16 * 2 * 0.75e-3)
    assert lbwd == pytest.approx(bwd - 16 * 3 * 0.75e-3)


def test_bubble_free_pipeline_is_busiest_stage_times_microbatches(monkeypatch):
    from primus.core.projection.performance_projection import projection

    matrix = [
        [{"fwd": 1.0, "bwd": 2.0, "wgrad": 0.0}, {"fwd": 1.0, "bwd": 2.0, "wgrad": 0.0}],
        [{"fwd": 1.5, "bwd": 3.0, "wgrad": 0.5}, {"fwd": 1.0, "bwd": 2.0, "wgrad": 0.0}],
    ]
    monkeypatch.setattr(projection, "_build_chunk_time_matrix", lambda cfg, res: matrix)
    monkeypatch.setattr(projection, "_compute_micro_batches", lambda rt, mp: 8)
    cfg = SimpleNamespace(runtime_config=None, model_parallel_config=None)
    with ceiling_levers(("launch",)):
        assert projection._run_pipeline_simulation(cfg, {}) == pytest.approx(8.0 * 8)


def test_overlap_ceiling_hides_deepep_and_pp_p2p():
    from primus.core.projection.performance_projection import projection

    with ceiling_levers(("overlap",)):
        assert projection._get_deepep_overlap_efficiency(SimpleNamespace()) == 1.0
        assert projection._estimate_pp_communication_overhead(SimpleNamespace(), 4) == 0.0


def _cli_args(*extra):
    from primus.cli.subcommands.projection import register_subcommand

    parser = argparse.ArgumentParser()
    register_subcommand(parser.add_subparsers(dest="command"))
    return parser.parse_args(["projection", "performance", *extra])


def test_cli_registers_ceiling_flags():
    args = _cli_args("--config", "x.yaml", "--bound", "ceiling", "--ceiling-levers", "fusion", "--headroom")
    assert args.projection_bound == "ceiling"
    assert args.ceiling_levers == "fusion"
    assert args.headroom and args.headroom_measured is None
    assert _cli_args("--config", "x.yaml").projection_bound == "expected"


def test_end_to_end_ceiling_never_below_expected(monkeypatch, capsys):
    pytest.importorskip("origami")
    from primus.core.projection.performance_projection import projection

    config = REPO / "examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-pretrain.yaml"
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.chdir(REPO)
    args = _cli_args(
        "--config",
        str(config),
        "--target-nodes",
        "1",
        "--profiling-mode",
        "simulate",
        "--gpu-arch",
        "mi355x",
        "--headroom",
    )
    expected = projection.launch_projection_from_cli(args, [])
    report = expected["headroom"]
    assert report["expected"]["tokens_per_sec_per_gpu"] == pytest.approx(expected["tokens_per_sec_per_gpu"])
    assert report["ceiling"]["tokens_per_sec_per_gpu"] > report["expected"]["tokens_per_sec_per_gpu"]
    assert all(r["gain_pct"] >= 0.0 for r in report["lever_gains"])

    args.projection_bound, args.headroom = "ceiling", False
    at_ceiling_result = projection.launch_projection_from_cli(args, [])
    assert at_ceiling_result["tokens_per_sec_per_gpu"] == pytest.approx(
        report["ceiling"]["tokens_per_sec_per_gpu"]
    )
    assert "Headroom (expected vs ceiling)" in capsys.readouterr().out
