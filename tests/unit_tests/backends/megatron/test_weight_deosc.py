###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU unit tests for MXFP4 weight de-oscillation logic.

These tests avoid any GPU / Primus-Turbo dependency by monkeypatching the
MXFP4 quantize-dequantize with a simple integer-rounding fake, and by driving a
minimal fake ``DistributedOptimizer``. They exercise the parts that are easy to
get wrong: DistRatio snap masking, write-back into the local fp32 shard, period
reset, and checkpoint state round-trip.
"""

import types

import pytest

torch = pytest.importorskip("torch")

from primus.backends.megatron.core.optimizer import weight_deosc
from primus.backends.megatron.core.optimizer.weight_deosc import (
    WeightDeOscConfig,
    WeightDeOscRunner,
    _ParamDeOscState,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeModule:
    def __init__(self, weight):
        self.quantized_weight_buffer = object()  # signal: fp4 forward ran
        self._parameters = {"weight": weight}

    def modules(self):
        return [self]


class _FakeChunk:
    def __init__(self, module):
        self._module = module

    def modules(self):
        return [self._module]


class _FakeDistOpt:
    """Minimal stand-in for DistributedOptimizer used by WeightDeOscRunner."""

    def __init__(self, model_param, shard_main_param, start, end, overlap=False):
        self.model_float16_groups = [[model_param]]
        self.shard_fp32_from_float16_groups = [[shard_main_param]]
        self._range = types.SimpleNamespace(start=start, end=end)
        self.ddp_config = types.SimpleNamespace(overlap_param_gather=overlap)
        self.model_chunks = [_FakeChunk(_FakeModule(model_param))]

    def _get_model_param_range_map(self, model_param):
        return {"param": self._range}

    def _param_name(self, model_param):
        return "decoder.layers.0.mlp.linear.weight"


def _fake_qdq_round(weight):
    """Quantize-dequantize fake: snap each element to the nearest integer."""
    return torch.round(weight)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_snap_writes_back_to_fp32_shard(monkeypatch):
    monkeypatch.setattr(weight_deosc, "qdq_mxfp4", _fake_qdq_round)

    n = 4
    model_param = torch.zeros(1, n)  # full 2D weight
    shard_main_param = torch.zeros(n)  # fp32 master, whole param on this rank
    opt = _FakeDistOpt(model_param, shard_main_param, start=0, end=n)

    runner = WeightDeOscRunner(WeightDeOscConfig(enable=True, period=2, ratio_threshold=2.0))

    # element 0: oscillates across the 0/1 bin boundary (Q flips, w barely moves)
    # element 1: constant 5.0 (no movement -> never snapped)
    seqs = [
        [0.49, 5.0, 0.0, 0.0],  # t0 (seed)
        [0.51, 5.0, 0.0, 0.0],  # t1 (track step 1)
        [0.49, 5.0, 0.0, 0.0],  # t2 (track step 2 -> period end -> snap)
    ]
    for vals in seqs:
        v = torch.tensor(vals)
        model_param.copy_(v.view(1, n))
        shard_main_param.copy_(v)
        runner.run(opt)

    # element 0 oscillated -> snapped to Q(0.49) = 0.0
    assert shard_main_param[0].item() == pytest.approx(0.0)
    # element 1 was constant -> untouched
    assert shard_main_param[1].item() == pytest.approx(5.0)


def test_period_resets_after_snap(monkeypatch):
    monkeypatch.setattr(weight_deosc, "qdq_mxfp4", _fake_qdq_round)

    n = 2
    model_param = torch.zeros(1, n)
    shard_main_param = torch.zeros(n)
    opt = _FakeDistOpt(model_param, shard_main_param, start=0, end=n)
    runner = WeightDeOscRunner(WeightDeOscConfig(enable=True, period=2, ratio_threshold=2.0))

    for vals in ([0.49, 1.0], [0.51, 1.0], [0.49, 1.0]):
        v = torch.tensor(vals)
        model_param.copy_(v.view(1, n))
        shard_main_param.copy_(v)
        runner.run(opt)

    key = next(iter(runner._state))
    state = runner._state[key]
    assert state.step == 0  # period was reset
    assert torch.all(state.dist_w == 0)
    assert torch.all(state.dist_w_qdq == 0)


def test_eligibility_excludes_non_fp4_modules(monkeypatch):
    monkeypatch.setattr(weight_deosc, "qdq_mxfp4", _fake_qdq_round)
    n = 4
    model_param = torch.zeros(1, n)
    shard_main_param = torch.zeros(n)
    opt = _FakeDistOpt(model_param, shard_main_param, start=0, end=n)
    # Drop the fp4 signal: module never quantized -> not eligible.
    opt.model_chunks[0]._module.quantized_weight_buffer = None

    runner = WeightDeOscRunner(WeightDeOscConfig(enable=True, period=2, ratio_threshold=2.0))
    runner.run(opt)
    assert runner._eligible_ids == set()
    assert len(runner._state) == 0


def test_state_dict_round_trip(monkeypatch):
    monkeypatch.setattr(weight_deosc, "qdq_mxfp4", _fake_qdq_round)
    n = 4
    model_param = torch.zeros(1, n)
    shard_main_param = torch.zeros(n)
    opt = _FakeDistOpt(model_param, shard_main_param, start=0, end=n)
    runner = WeightDeOscRunner(WeightDeOscConfig(enable=True, period=10, ratio_threshold=2.0))

    for vals in ([0.49, 5.0, 0.0, 0.0], [0.51, 5.0, 0.0, 0.0]):
        v = torch.tensor(vals)
        model_param.copy_(v.view(1, n))
        shard_main_param.copy_(v)
        runner.run(opt)

    sd = runner.state_dict()
    assert sd["global_step"] == 2
    assert len(sd["params"]) == 1

    # New runner restores and continues from the same window.
    runner2 = WeightDeOscRunner(WeightDeOscConfig(enable=True, period=10, ratio_threshold=2.0))
    runner2.load_state_dict(sd)
    assert runner2._global_step == 2

    key = next(iter(sd["params"]))
    blob = sd["params"][key]
    restored = _ParamDeOscState.from_serializable(blob, torch.device("cpu"), shard_main_param)
    assert restored is not None
    assert restored.step == 1  # one tracked step accumulated before save

    # Shape mismatch (resharding) is rejected -> caller re-seeds.
    mismatched = _ParamDeOscState.from_serializable(blob, torch.device("cpu"), torch.zeros(n + 1))
    assert mismatched is None


def test_disabled_runner_is_noop(monkeypatch):
    monkeypatch.setattr(weight_deosc, "qdq_mxfp4", _fake_qdq_round)
    n = 2
    model_param = torch.zeros(1, n)
    shard_main_param = torch.zeros(n)
    opt = _FakeDistOpt(model_param, shard_main_param, start=0, end=n)
    runner = WeightDeOscRunner(WeightDeOscConfig(enable=False, period=2, ratio_threshold=2.0))
    runner.run(opt)
    assert len(runner._state) == 0
