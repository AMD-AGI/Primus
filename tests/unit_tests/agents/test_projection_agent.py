###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Projection agent: op bring-up (context pack, coder retry on validator feedback,
acknowledge path), the agent's preflight / runtime handling, and its callers --
``primus projection --op-bringup`` and the tuning agent's evaluator hook."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("yaml")

from primus.agents.projection_agent import ProjectionAgent  # noqa: E402
from primus.agents.projection_agent import bringup as ob  # noqa: E402
from primus.agents.projection_agent import cli as pa_cli  # noqa: E402
from primus.agents.tuning_agent.evaluator import EvalResult, Evaluator  # noqa: E402
from primus.core.projection.op_models.coverage import OpGap  # noqa: E402

_ROOT = Path(__file__).resolve().parents[3]
_WORKLOAD = _ROOT / "examples/megatron/configs/MI355X/qwen3_5_35B_A3B-BF16-pretrain.yaml"
_GDN = _ROOT / "primus/core/projection/op_models/generated/gated_delta_net.py"

pytestmark = pytest.mark.skipif(not _WORKLOAD.is_file(), reason="Qwen3.5 workload not in this checkout")


def _gap(**keys):
    return OpGap(op_id="test_op", severity="gap", slot="attention", reason="unit test", keys=keys)


def _variant(miscount: bool) -> str:
    src = _GDN.read_text().replace('name = "gated_delta_net"', 'name = "gdn_bringup_test"')
    src = src.replace("class GatedDeltaNet(", "class GdnBringupTest(")
    if miscount:
        src = src.replace("return fwd, bwd\n", "return 3 * fwd, 3 * bwd\n")
    return "Here is the module.\n```python\n" + src + "```\n"


def test_reply_parsing():
    assert ob.extract_module("```python\nclass A(OpModel):\n    pass\n```").startswith("class A")
    assert ob.extract_module("no code here") is None
    assert ob.parse_acknowledgements("ACKNOWLEDGE a_key: tiny\nnoise\nACKNOWLEDGE b: 1% of layer") == {
        "a_key": "tiny",
        "b": "1% of layer",
    }


def test_context_pack_has_the_pieces(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "fetch_hf_config", lambda *a, **k: '{"model_type": "qwen3_5_moe"}')
    from primus.agents.tuning_agent.workload import resolve_workload

    arch = resolve_workload(_WORKLOAD, primus_root=_ROOT)
    text = ob.build_context(
        _gap(experimental_attention_variant="gated_delta_net"),
        primus_root=_ROOT,
        model_path=Path(arch.model_path),
        raw_model=arch.raw_model,
        papers=["S_t = a_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T"],
    )
    for needle in (
        "## The gap",
        "qwen3_5_35B_A3B.yaml",
        "qwen3_5_moe",
        "S_t = a_t",
        "class OpContext",
        "generated/gated_delta_net.py",
        "_simulate_tile_level",
        "ACKNOWLEDGE",
    ):
        assert needle in text, needle


def test_failed_validation_is_fed_back_then_the_fix_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "fetch_hf_config", lambda *a, **k: None)
    prompts = []

    def coder(prompt, target):
        prompts.append(prompt)
        return _variant(miscount=len(prompts) == 1)

    res = ob.bring_up_op(
        _gap(experimental_attention_variant="gated_delta_net", linear_num_value_heads=32),
        workload=_WORKLOAD,
        primus_root=_ROOT,
        out_dir=tmp_path,
        coder=coder,
        gpu_arch="mi355x",
        register=False,
        log=lambda s: None,
    )
    assert res.status == "registered" and res.attempts == 2, res.message
    assert res.op_name == "gdn_bringup_test"
    assert "Feedback on your previous attempt" in prompts[1] and "reference_flops" in prompts[1]
    assert (tmp_path / "op_bringup/test_op/summary.json").is_file()
    assert not (_ROOT / "primus/core/projection/op_models/generated/gdn_bringup_test.py").exists()


def test_acknowledge_must_cover_exactly_the_gap_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "fetch_hf_config", lambda *a, **k: None)
    replies = iter(
        ["ACKNOWLEDGE unrelated: x", "ACKNOWLEDGE moe_shared_expert_gate: hidden->1 GEMV, <1% of layer"]
    )
    res = ob.bring_up_op(
        _gap(moe_shared_expert_gate=True),
        workload=_WORKLOAD,
        primus_root=_ROOT,
        out_dir=tmp_path,
        coder=lambda p, t: next(replies),
        register=False,
        log=lambda s: None,
    )
    assert res.status == "acknowledged" and res.attempts == 2
    assert res.acknowledged == {"moe_shared_expert_gate": "hidden->1 GEMV, <1% of layer"}


def test_no_coder_leaves_a_context_pack(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "fetch_hf_config", lambda *a, **k: None)
    res = ob.bring_up_op(
        _gap(x_gate=True),
        workload=_WORKLOAD,
        primus_root=_ROOT,
        out_dir=tmp_path,
        coder=None,
        log=lambda s: None,
    )
    assert res.status == "context_only" and Path(res.context_path).is_file()


def test_evaluator_reruns_once_after_new_op_models():
    ev = Evaluator.__new__(Evaluator)
    ev.op_gap_handler, ev.handled_op_gaps, ev.op_bringup = None, set(), None
    gap = {"op_id": "new_attn", "severity": "gap", "slot": "attention", "reason": "r", "keys": {"k": 1}}
    calls = []

    def fake_eval(cfg, tag, mode):
        calls.append(tag)
        return EvalResult(op_gaps=[gap] if len(calls) == 1 else [], tokens_per_s_per_gpu=float(len(calls)))

    ev._evaluate = fake_eval
    handled = []
    ev.op_gap_handler = lambda gaps: handled.append([g["op_id"] for g in gaps]) or True
    r = ev._with_op_bringup(None, "t0", "simulate")
    assert calls == ["t0", "t0_ops"] and handled == [["new_attn"]] and r.tokens_per_s_per_gpu == 2.0
    calls.clear()
    ev._with_op_bringup(None, "t1", "simulate")  # same gap again: no second bring-up
    assert calls == ["t1"] and len(handled) == 1


# ---------------------------------------------------------------- the agent


def test_agent_preflight_finds_the_workload_gaps_once(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "fetch_hf_config", lambda *a, **k: None)
    monkeypatch.setenv("PRIMUS_OP_MODELS", "0")  # hide the shipped Qwen3.5 ops so they are gaps again
    agent = ProjectionAgent(primus_root=_ROOT, out_dir=tmp_path, policy="report", log=lambda s: None)
    res = agent.ensure_ops(_WORKLOAD)
    assert {"gated_delta_net", "attention_output"} <= {r.op_id for r in res}
    assert all(r.status == "context_only" and Path(r.context_path).is_file() for r in res)
    assert agent.ensure_ops(_WORKLOAD) == []  # already brought up: nothing repeats
    assert set(agent.summary()["gaps_seen"]) >= {"gated_delta_net", "attention_output"}


def test_agent_runtime_gaps_from_markers(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "fetch_hf_config", lambda *a, **k: None)
    agent = ProjectionAgent(
        primus_root=_ROOT,
        out_dir=tmp_path,
        coder=lambda p, t: "ACKNOWLEDGE fancy_gate: hidden->1 GEMV, <1% of layer",
        register=False,
        log=lambda s: None,
    )
    gap = {"op_id": "fancy", "severity": "gap", "slot": "mlp", "reason": "r", "keys": {"fancy_gate": True}}
    assert agent.handle_gaps([gap], _WORKLOAD) is True
    assert agent.handle_gaps([gap], _WORKLOAD) is False
    assert [r.status for r in agent.results] == ["acknowledged"]


def test_auto_without_a_coder_reports(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("PRIMUS_OP_LLM_MODEL", raising=False)
    monkeypatch.setattr(pa_cli, "_llm_available", lambda args: False)
    p = argparse.ArgumentParser()
    pa_cli.add_op_bringup_args(p)
    agent = pa_cli.agent_from_args(p.parse_args([]))
    assert agent.policy == "report" and agent.coder is None
    assert pa_cli.agent_from_args(p.parse_args(["--op-bringup", "off"])) is None
    agent = pa_cli.agent_from_args(p.parse_args(["--op-coder-cmd", "cp x {target_file}"]))
    assert agent.policy == "auto" and agent.coder is not None


def test_primus_projection_brings_ops_up_before_projecting(monkeypatch):
    from primus.cli.subcommands import projection as proj_cli
    from primus.core.projection import performance_projection

    order = []
    monkeypatch.setattr(
        ProjectionAgent, "ensure_ops", lambda self, workload, **kw: order.append(("ensure", str(workload)))
    )
    monkeypatch.setattr(
        performance_projection, "launch_projection_from_cli", lambda args, ov: order.append(("project",))
    )
    parser = argparse.ArgumentParser()
    proj_cli.register_subcommand(parser.add_subparsers(dest="command"))
    argv = ["projection", "performance", "--config", str(_WORKLOAD), "--profiling-mode", "simulate"]
    args, extra = parser.parse_known_args(argv + ["--op-bringup", "report"])
    proj_cli.run(args, extra)
    assert order == [("ensure", str(_WORKLOAD)), ("project",)]

    order.clear()
    args, extra = parser.parse_known_args(argv)  # default: off
    proj_cli.run(args, extra)
    assert order == [("project",)]


def test_command_coder_surfaces_a_failing_command(tmp_path):
    coder = ob.command_coder("echo 'auth failed' && exit 3")
    with pytest.raises(RuntimeError, match="exited with status 3: auth failed"):
        coder("prompt", tmp_path / "op.py")
    # A command that writes the module is fine whatever its exit status.
    ok = ob.command_coder("echo 'class X(OpModel): pass' > {target_file}; exit 1")
    assert "class X" in ok("prompt", tmp_path / "op2.py")


@pytest.mark.parametrize("argv", [["--op-coder", "command"], ["--op-coder", "dspy"]])
def test_explicit_coder_misconfiguration_is_an_error(monkeypatch, argv):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("PRIMUS_OP_LLM_MODEL", raising=False)
    try:
        import dspy

        monkeypatch.setattr(dspy.settings, "lm", None, raising=False)
    except ImportError:
        pass  # without dspy the dspy coder is just as unavailable
    p = argparse.ArgumentParser()
    pa_cli.add_op_bringup_args(p)
    with pytest.raises(ValueError, match=f"--op-coder {argv[1]}"):
        pa_cli.coder_from_args(p.parse_args(argv))


def test_acknowledging_a_key_twice_keeps_one_entry(tmp_path):
    gen = tmp_path / "primus/core/projection/op_models/generated"
    gen.mkdir(parents=True)
    (gen / "acknowledged.yaml").write_text("old_key: >-\n  negligible\n")
    ob.acknowledge_keys(gen, {"old_key": "again", "new_key": "tiny"})
    ob.acknowledge_keys(gen, {"new_key": "tiny again"})
    text = (gen / "acknowledged.yaml").read_text()
    assert text.count("old_key:") == 1 and text.count("new_key:") == 1
    assert yaml.safe_load(text) == {"old_key": "negligible", "new_key": "tiny"}


def test_context_is_fetched_once_for_all_gaps(tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setattr(ob, "fetch_hf_config", lambda *a, **k: fetched.append(a) or None)
    gaps = [
        OpGap(op_id=f"op{i}", severity="gap", slot="mlp", reason="r", keys={f"k{i}": True}) for i in range(3)
    ]
    ob.bring_up_gaps(
        gaps, workload=_WORKLOAD, primus_root=_ROOT, out_dir=tmp_path, coder=None, log=lambda s: None
    )
    assert len(fetched) == 1


def test_unexpected_bring_up_errors_do_not_abort_the_projection(tmp_path, monkeypatch):
    from primus.agents.projection_agent import agent as agent_mod

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(agent_mod, "bring_up_gaps", boom)
    logs = []
    agent = ProjectionAgent(primus_root=_ROOT, out_dir=tmp_path, policy="report", log=logs.append)
    gap = {"op_id": "fancy", "severity": "gap", "slot": "mlp", "reason": "r", "keys": {"fancy_gate": True}}
    assert agent.handle_gaps([gap], _WORKLOAD) is False
    assert any("op bring-up failed" in s and "disk full" in s for s in logs)


def test_standalone_cli_loads_the_env_file(monkeypatch):
    import primus.agents.tuning_agent.config as ta_config

    loaded = []
    monkeypatch.setattr(ta_config, "load_env", lambda *a, **k: loaded.append(True))
    assert pa_cli.main(["--workload", str(_WORKLOAD), "--op-bringup", "off"]) == 0
    assert loaded == [True]


def _named(name: str) -> str:
    return _variant(miscount=False).replace('name = "gdn_bringup_test"', f'name = "{name}"')


def _tree_status() -> str:
    gen = "primus/core/projection/op_models/generated"
    tests = "tests/unit_tests/core/projection/op_models"
    return subprocess.run(
        ["git", "-C", str(_ROOT), "status", "--porcelain", "--", gen, tests], capture_output=True, text=True
    ).stdout


def test_registration_stays_out_of_the_source_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "fetch_hf_config", lambda *a, **k: None)
    monkeypatch.setenv("PRIMUS_OP_MODEL_DIRS", "")
    before = _tree_status()
    res = ob.bring_up_op(
        _gap(experimental_attention_variant="gated_delta_net", linear_num_value_heads=32),
        workload=_WORKLOAD,
        primus_root=_ROOT,
        out_dir=tmp_path,
        coder=lambda p, t: _variant(miscount=False),
        gpu_arch="mi355x",
        log=lambda s: None,
    )
    assert res.status == "registered", res.message
    reg = tmp_path.resolve() / "op_models"
    assert res.module_path == str(reg / "gdn_bringup_test.py")
    assert res.test_path == str(reg / "tests" / "test_generated_gdn_bringup_test.py")
    assert _tree_status() == before
    assert str(reg) in os.environ["PRIMUS_OP_MODEL_DIRS"].split(os.pathsep)
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", res.test_path],
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": str(_ROOT)},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stdout[-3000:]


def test_an_existing_op_model_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "fetch_hf_config", lambda *a, **k: None)
    monkeypatch.setenv("PRIMUS_OP_MODEL_DIRS", "")
    reg = tmp_path.resolve() / "op_models"
    reg.mkdir()
    (reg / "gdn_mine.py").write_text("# someone else's op\n")
    replies = iter([_named("gated_delta_net"), _named("gdn_mine"), _named("gdn_fresh")])
    prompts = []

    def coder(prompt, target):
        prompts.append(prompt)
        return next(replies)

    res = ob.bring_up_op(
        _gap(experimental_attention_variant="gated_delta_net", linear_num_value_heads=32),
        workload=_WORKLOAD,
        primus_root=_ROOT,
        out_dir=tmp_path,
        coder=coder,
        gpu_arch="mi355x",
        log=lambda s: None,
    )
    assert res.status == "registered" and res.attempts == 3 and res.op_name == "gdn_fresh", res.message
    assert "An op model named `gated_delta_net` already exists" in prompts[1]
    assert "An op model named `gdn_mine` already exists" in prompts[2]
    assert (reg / "gdn_mine.py").read_text() == "# someone else's op\n"


def test_out_of_tree_acknowledgements_reach_coverage(tmp_path, monkeypatch):
    from primus.core.projection.op_models import coverage

    reg = tmp_path / "op_models"
    ob.acknowledge_keys(reg, {"fancy_gate": "hidden->1 GEMV"})
    assert "fancy_gate" not in coverage._acknowledged()
    monkeypatch.setenv("PRIMUS_OP_MODEL_DIRS", str(reg))
    assert coverage._acknowledged()["fancy_gate"] == "hidden->1 GEMV"


def test_agent_reuses_its_out_dir_registry_and_honours_in_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIMUS_OP_MODEL_DIRS", "")
    (tmp_path / "op_models").mkdir()
    agent = ProjectionAgent(primus_root=_ROOT, out_dir=tmp_path, policy="report", log=lambda s: None)
    assert agent.registry_dir == tmp_path.resolve() / "op_models"
    assert os.environ["PRIMUS_OP_MODEL_DIRS"] == str(agent.registry_dir)
    p = argparse.ArgumentParser()
    pa_cli.add_op_bringup_args(p)
    agent = pa_cli.agent_from_args(p.parse_args(["--op-bringup", "report", "--op-register-in-tree"]))
    assert agent.in_tree and agent.registry_dir == _ROOT / "primus/core/projection/op_models/generated"
