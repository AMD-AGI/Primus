###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Op bring-up: context pack, coder retry on validator feedback, acknowledge path, evaluator hook."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from primus.agents.tuning_agent import op_bringup as ob  # noqa: E402
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
