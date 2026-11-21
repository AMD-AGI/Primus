###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Routing replay on CPU: reading a Megatron router trace and handing a router its recorded routes."""

import json
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("importing megatron.training needs a GPU", allow_module_level=True)

from primus.backends.megatron.core.transformer.moe import (  # noqa: E402
    router as router_mod,
)


def _record(layer, routes, block="decoder", step=0):
    return {
        "step": step,
        "stage": "pre_dispatch",
        "block": block,
        "layer": layer,
        "rank": 0,
        "num_tokens": len(routes),
        "topk": len(routes[0]),
        "top_indices": routes,
    }


@pytest.fixture
def trace(tmp_path, monkeypatch):
    monkeypatch.setattr(router_mod, "_REPLAY_ROUTERS", [])
    monkeypatch.setattr(router_mod, "_REPLAY_RECORDS", {})
    records = [
        _record(0, [[0, 1], [2, 3]]),
        _record(1, [[4, 5], [6, 7], [0, 2]]),
        _record(1, [[1, 3]], block="mtp"),
        _record(1, [[3, 5]], step=1),
    ]
    with open(tmp_path / "router_trace_rank0.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return tmp_path


def _use_args(monkeypatch, path, layers=None):
    args = SimpleNamespace(
        moe_routing_replay_path=str(path) if path else None, moe_routing_replay_layers=layers
    )
    monkeypatch.setattr(router_mod, "get_args", lambda: args)


def _router(layer_number, topk=2, num_experts=8):
    return SimpleNamespace(
        layer_number=layer_number,
        topk=topk,
        config=SimpleNamespace(num_moe_experts=num_experts),
        weight=torch.zeros(1),
        router_replay=None,
    )


def test_replay_off(monkeypatch):
    _use_args(monkeypatch, None)
    assert router_mod._replayed_topk(_router(1), 4) is None


def test_replays_the_layer_with_the_same_number_and_wraps(trace, monkeypatch):
    _use_args(monkeypatch, trace)
    r = _router(layer_number=2)  # decoder layer 1: its decoder records in file order, MTP skipped
    assert router_mod._replayed_topk(r, 3).tolist() == [[4, 5], [6, 7], [0, 2]]
    assert router_mod._replayed_topk(r, 3).tolist() == [[3, 5], [4, 5], [6, 7]]
    assert r.router_replay is not None


def test_replay_layers_pick_the_recorded_layer(trace, monkeypatch):
    _use_args(monkeypatch, trace, layers=[0, 1])
    first, second = _router(layer_number=7), _router(layer_number=9)
    assert router_mod._replayed_topk(first, 2).tolist() == [[0, 1], [2, 3]]
    assert router_mod._replayed_topk(second, 1).tolist() == [[4, 5]]
    with pytest.raises(ValueError, match="has 2 entries"):
        router_mod._replayed_topk(_router(layer_number=11), 1)


@pytest.mark.parametrize(
    ("router", "match"),
    [
        (_router(layer_number=5), "no routes of decoder layer 4"),
        (_router(layer_number=1, topk=4), "do not fit top-4"),
        (_router(layer_number=1, num_experts=2), "over 2 experts"),
    ],
)
def test_replay_rejects_a_trace_that_does_not_fit(trace, monkeypatch, router, match):
    _use_args(monkeypatch, trace)
    with pytest.raises(ValueError, match=match):
        router_mod._replayed_topk(router, 1)
