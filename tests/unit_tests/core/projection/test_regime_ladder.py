###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Unit tests for the confidence-ladder anchor policy
(``core/projection/inference_projection/regime.py``).

The ladder decides whether a TP/EP restore from a low-GPU anchor is trustworthy
or whether the run should benchmark at a higher GPU count first. Trust requires
the per-GPU decode throughput to be FLAT (within +/-tol, two-sided) across an
adjacent measured pair whose top rung sits within one doubling of the target.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("primus.core.projection.inference_projection.regime")

from primus.core.projection.inference_projection.regime import (  # noqa: E402
    climb_anchor_ladder,
    confidence_ladder,
    relief_between,
)


def _anchor(tmp_path, name, tp, per_gpu_tput_at_32):
    """Write a minimal anchor file whose batch-32 decode_ms yields the target
    per-GPU throughput ``per_gpu_tput_at_32`` (tok/s/GPU)."""
    decode_ms = 32 * 1000.0 / per_gpu_tput_at_32 / tp
    blob = {
        "meta": {"tp": tp, "pp": 1, "ep": tp},
        "sweep": [{"batch": 32, "decode_ms": decode_ms}],
    }
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(blob))
    return str(p)


def test_relief_two_sided(tmp_path):
    a1 = _anchor(tmp_path, "g1", 1, 1000.0)
    a2_flat = _anchor(tmp_path, "g2f", 2, 1000.0)
    a2_super = _anchor(tmp_path, "g2s", 2, 1300.0)
    a2_sub = _anchor(tmp_path, "g2b", 2, 440.0)
    assert relief_between(a1, a2_flat) == pytest.approx(1.0)
    assert relief_between(a1, a2_super) == pytest.approx(1.3)
    assert relief_between(a1, a2_sub) == pytest.approx(0.44)


def test_exact_when_rung_at_or_above_target(tmp_path):
    a2 = _anchor(tmp_path, "g2", 2, 1000.0)
    v = confidence_ladder(2, {2: a2})
    assert v["confidence"] == "high" and v["converged"] and v["gpus"] == 2


def test_flat_within_one_doubling_certifies(tmp_path):
    # flat 1->2 (relief 1.0); target 4 is within one doubling of the 2-GPU rung.
    a1 = _anchor(tmp_path, "g1", 1, 1000.0)
    a2 = _anchor(tmp_path, "g2", 2, 1000.0)
    v = confidence_ladder(4, {1: a1, 2: a2})
    assert v["confidence"] == "high" and v["gpus"] == 2 and v["next_gpus"] is None


def test_flat_far_below_target_is_not_trusted(tmp_path):
    # flat 1->2 but target 8 is >1 doubling away (e.g. high-EP collapse unprobed).
    a1 = _anchor(tmp_path, "g1", 1, 1000.0)
    a2 = _anchor(tmp_path, "g2", 2, 1000.0)
    v = confidence_ladder(8, {1: a1, 2: a2})
    assert v["confidence"] == "low" and v["next_gpus"] == 4


def test_sublinear_pair_is_not_flat(tmp_path):
    # sub-linear 2->4 (sharding penalty) must NOT be accepted as flat.
    a2 = _anchor(tmp_path, "g2", 2, 2000.0)
    a4 = _anchor(tmp_path, "g4", 4, 880.0)  # relief 0.44
    v = confidence_ladder(8, {2: a2, 4: a4})
    assert v["confidence"] == "low" and v["next_gpus"] == 8


def test_superlinear_single_rung_climbs(tmp_path):
    a1 = _anchor(tmp_path, "g1", 1, 1000.0)
    v = confidence_ladder(2, {1: a1})
    assert v["confidence"] == "low" and v["next_gpus"] == 2


def test_climb_loop_stops_at_target(tmp_path):
    rungs = {
        2: _anchor(tmp_path, "g2", 2, 2000.0),
        4: _anchor(tmp_path, "g4", 4, 880.0),
        8: _anchor(tmp_path, "g8", 8, 760.0),
    }
    calls = []

    def bench(g):
        calls.append(g)
        return rungs.get(g)

    v = climb_anchor_ladder(8, bench, max_gpus=8, start_gpus=2)
    assert calls == [2, 4, 8]
    assert v["converged"] and v["gpus"] == 8


def test_climb_loop_stops_early_when_flat(tmp_path):
    # flat all the way: 1->2 flat, so target 4 is certified from the 2-GPU rung
    # without benchmarking at 4.
    rungs = {
        1: _anchor(tmp_path, "g1", 1, 1000.0),
        2: _anchor(tmp_path, "g2", 2, 1000.0),
    }
    calls = []

    def bench(g):
        calls.append(g)
        return rungs.get(g)

    v = climb_anchor_ladder(4, bench, max_gpus=4, start_gpus=1)
    assert v["converged"] and v["confidence"] == "high" and v["gpus"] == 2
    assert 4 not in calls  # never had to benchmark the 4-GPU rung
