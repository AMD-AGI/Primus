###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from primus.backends.megatron.patches import hylo_order_data_patches as patches
from primus.backends.megatron.training.hylo_order_dataset import HyLoOrderDataset


def _args(**overrides):
    args = SimpleNamespace(
        model_type="mamba",
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        world_size=8,
        micro_batch_size=12,
        global_batch_size=96,
        dataloader_type="cyclic",
        hylo_order_data_dir="/dump",
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def test_replay_accepts_one_micro_batch_per_rank_per_step():
    patches.check_replay_args(_args())
    patches.check_replay_args(_args(tensor_model_parallel_size=2, global_batch_size=48))


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"global_batch_size": 192}, "global_batch_size=192"),
        ({"world_size": 4}, r"12 x 4"),
        ({"tensor_model_parallel_size": 2}, r"12 x 4"),
        ({"context_parallel_size": 2}, "context_parallel_size"),
        ({"model_type": "gpt"}, "hybrid"),
    ],
)
def test_replay_rejects_runs_that_cannot_follow_the_recording(overrides, match):
    with pytest.raises(ValueError, match=match):
        patches.check_replay_args(_args(**overrides))


def test_patch_switches_to_the_sequential_sampler(monkeypatch):
    args = _args()
    installed = []
    monkeypatch.setattr(patches, "get_args", lambda ctx: args)
    monkeypatch.setattr(patches, "_install_hylo_order_data_patch", lambda: installed.append(True))
    patches.patch_hylo_order_data(None)
    assert args.dataloader_type == "single"
    assert installed == [True]


def test_patch_checks_the_layout_before_installing(monkeypatch):
    monkeypatch.setattr(patches, "get_args", lambda ctx: _args(global_batch_size=24))
    monkeypatch.setattr(patches, "_install_hylo_order_data_patch", lambda: pytest.fail("installed"))
    with pytest.raises(ValueError, match="global_batch_size"):
        patches.patch_hylo_order_data(None)


def _module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def megatron_0_19(monkeypatch, tmp_path):
    """Megatron-core 0.19: ``pretrain_mamba`` is a shim and the provider lives in ``pretrain_hybrid``."""
    steps, mbs, seq, dp = 3, 2, 4, 1
    np.arange(steps * mbs * seq, dtype=np.int32).tofile(tmp_path / "batches_rank0.bin")
    (tmp_path / "meta.json").write_text(
        json.dumps({"micro_batch": mbs, "seq_len": seq, "world_size": dp, "steps": steps})
    )
    args = SimpleNamespace(
        hylo_order_data_dir=str(tmp_path),
        seq_length=seq,
        micro_batch_size=mbs,
        global_batch_size=mbs,
        train_iters=3,
    )
    original = []
    hybrid = _module("pretrain_hybrid", train_valid_test_datasets_provider=lambda *a, **k: original.append(a))
    monkeypatch.setitem(sys.modules, "pretrain_mamba", _module("pretrain_mamba"))
    monkeypatch.setitem(sys.modules, "pretrain_hybrid", hybrid)
    parallel_state = _module("megatron.core.parallel_state", get_data_parallel_world_size=lambda: dp)
    monkeypatch.setitem(sys.modules, "megatron", _module("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", _module("megatron.core", parallel_state=parallel_state))
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    monkeypatch.setitem(
        sys.modules,
        "megatron.training",
        _module("megatron.training", get_args=lambda: args, print_rank_0=lambda *a: None),
    )
    return SimpleNamespace(args=args, hybrid=hybrid, original=original)


def test_provider_patch_lands_in_pretrain_hybrid_and_serves_the_recording(megatron_0_19):
    patches._install_hylo_order_data_patch()
    train, valid, test = megatron_0_19.hybrid.train_valid_test_datasets_provider([6, 0, 0])
    assert isinstance(train, HyLoOrderDataset) and train.steps == 3
    assert (valid, test) == (None, None)

    megatron_0_19.args.hylo_order_data_dir = None
    megatron_0_19.hybrid.train_valid_test_datasets_provider([6, 0, 0])
    assert megatron_0_19.original == [([6, 0, 0],)]


def test_provider_rejects_a_recording_shorter_than_the_run(megatron_0_19):
    megatron_0_19.args.train_iters = 4
    patches._install_hylo_order_data_patch()
    with pytest.raises(ValueError, match="fewer than train_iters=4"):
        megatron_0_19.hybrid.train_valid_test_datasets_provider([8, 0, 0])
