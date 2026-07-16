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
    _uses_precision_aware_main_params,
    qdq_mxfp4_local_shard,
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


class _FakeMultiChunk:
    def __init__(self, *modules):
        self._modules = modules

    def modules(self):
        return self._modules


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


def test_local_shard_qdq_preserves_2d_tile_coordinates_and_bf16_input(monkeypatch):
    calls = []

    def _identity_qdq(tile):
        calls.append(tile.clone())
        return tile

    monkeypatch.setattr(weight_deosc, "qdq_mxfp4", _identity_qdq)
    shape = (128, 64)
    start = 35 * shape[1] + 5
    end = 80 * shape[1] + 7
    shard = torch.linspace(1.0001, 2.0001, end - start, dtype=torch.float32)

    q_local = qdq_mxfp4_local_shard(shard, shape, start, end, torch.bfloat16)

    assert q_local.dtype == torch.bfloat16
    assert torch.equal(q_local, shard.to(torch.bfloat16))
    assert len(calls) == 1
    tile = calls[0]
    assert tile.shape == (64, 64)  # original rows 32:96
    tile_begin = (35 - 32) * shape[1] + 5
    tile_end = (80 - 32) * shape[1] + 7
    tile_flat = tile.reshape(-1)
    assert torch.count_nonzero(tile_flat[:tile_begin]) == 0
    assert torch.count_nonzero(tile_flat[tile_end:]) == 0
    assert torch.equal(tile_flat[tile_begin:tile_end], shard.to(torch.bfloat16))


def test_local_shard_qdq_treats_grouped_experts_as_independent_2d_matrices(monkeypatch):
    calls = []

    def _identity_qdq(tile):
        calls.append(tile.clone())
        return tile

    monkeypatch.setattr(weight_deosc, "qdq_mxfp4", _identity_qdq)
    shape = (3, 64, 64)
    matrix_numel = shape[1] * shape[2]
    start = 40 * shape[2] + 3
    end = 2 * matrix_numel + 10 * shape[2] + 5
    shard = torch.arange(end - start, dtype=torch.float32)

    q_local = qdq_mxfp4_local_shard(shard, shape, start, end, torch.bfloat16)

    assert torch.equal(q_local, shard.to(torch.bfloat16))
    assert [tuple(tile.shape) for tile in calls] == [(32, 64), (64, 64), (32, 64)]


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


def test_eligibility_collects_dense_and_grouped_fp4_weights():
    dense_weight = torch.zeros(4, 4)
    grouped_weights = torch.zeros(2, 4, 4)
    dense_module = _FakeModule(dense_weight)
    grouped_module = types.SimpleNamespace(
        quantized_weight_buffer=object(),
        _parameters={},
        weights=grouped_weights,
    )
    non_fp4_module = types.SimpleNamespace(
        quantized_weight_buffer=None,
        _parameters={"weight": torch.zeros(4, 4)},
    )
    opt = types.SimpleNamespace(
        model_chunks=[_FakeMultiChunk(dense_module, grouped_module, non_fp4_module)]
    )

    runner = WeightDeOscRunner(WeightDeOscConfig(enable=True))
    eligible_ids = runner._build_eligible_ids(opt)

    assert eligible_ids == {id(dense_weight), id(grouped_weights)}


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


def test_precision_aware_detected_by_config():
    opt = types.SimpleNamespace(
        config=types.SimpleNamespace(use_precision_aware_optimizer=True),
        shard_fp32_from_float16_groups=[],
    )
    assert _uses_precision_aware_main_params(opt) is True


def test_precision_aware_detected_structurally():
    # float16 slots present but every main shard is None -> bf16 main params.
    opt = types.SimpleNamespace(
        config=types.SimpleNamespace(),
        shard_fp32_from_float16_groups=[[None, None]],
    )
    assert _uses_precision_aware_main_params(opt) is True


def test_standard_fp32_main_is_not_precision_aware():
    opt = types.SimpleNamespace(
        config=types.SimpleNamespace(),
        shard_fp32_from_float16_groups=[[torch.zeros(4)]],
    )
    assert _uses_precision_aware_main_params(opt) is False


def test_disabled_runner_is_noop(monkeypatch):
    monkeypatch.setattr(weight_deosc, "qdq_mxfp4", _fake_qdq_round)
    n = 2
    model_param = torch.zeros(1, n)
    shard_main_param = torch.zeros(n)
    opt = _FakeDistOpt(model_param, shard_main_param, start=0, end=n)
    runner = WeightDeOscRunner(WeightDeOscConfig(enable=False, period=2, ratio_threshold=2.0))
    runner.run(opt)
    assert len(runner._state) == 0
