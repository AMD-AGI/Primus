"""Regression test for the generated op model ``lfm2_short_conv`` (written by op bring-up)."""

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from primus.core.projection.op_models.registry import registered_op_models
from primus.core.projection.op_models.validate import load_workload, validate_op_model

_ROOT = Path(__file__).resolve().parents[5]
_WORKLOAD = _ROOT / "examples/megatron/configs/MI355X/lfm2_8B_A1B-BF16-pretrain.yaml"
_GAP_KEYS = ["lfm_layer_types"]


@pytest.mark.skipif(not _WORKLOAD.is_file(), reason="workload YAML not in this checkout")
def test_lfm2_short_conv_validates(monkeypatch):
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    monkeypatch.setenv("PRIMUS_OP_COVERAGE", "0")
    ops = {c.name: c for c in registered_op_models()}
    assert "lfm2_short_conv" in ops
    wl = load_workload(str(_WORKLOAD), simulate_backends=False)
    rep = validate_op_model(ops["lfm2_short_conv"], wl, gap_keys=_GAP_KEYS)
    assert rep.ok, rep.feedback()
