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
import torch

from primus.backends.megatron.core.models.registry import get_model_family
from primus.backends.megatron.patches import hylo_patches
from primus.backends.megatron.sft import forward_step as sft_forward_step
from primus.backends.megatron.training import hylo_distillation
from primus.backends.megatron.training.hylo_order_dataset import HyLoOrderDataset


def _args(**overrides):
    args = SimpleNamespace(
        model_type="hylo",
        hylo_distill_stage="ild",
        hylo_teacher_checkpoint="/teacher",
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


def _module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_model_type_hylo_builds_through_megatrons_model_provider(monkeypatch):
    from primus.core.utils import import_utils

    monkeypatch.setattr(import_utils, "log_rank_0", lambda *a, **k: None)
    upstream = _module("model_provider", model_provider=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "model_provider", upstream)
    provider = import_utils.get_model_provider(model_type="hylo")
    assert provider.func is upstream.model_provider
    assert provider.args == (hylo_distillation.hylo_builder,)


def test_model_type_hylo_trains_with_the_hylo_forward_step_and_datasets():
    assert get_model_family("hylo").training_entrypoint() is hylo_distillation


def test_each_build_gets_the_mamba_student_and_a_new_teacher(monkeypatch):
    students = []

    def mamba_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
        students.append((pre_process, post_process, vp_stage, config, pg_collection))
        return f"student{len(students)}"

    monkeypatch.setitem(sys.modules, "mamba_builders", _module("mamba_builders", mamba_builder=mamba_builder))
    monkeypatch.setattr(hylo_distillation, "_build_teacher", lambda *a: torch.nn.Identity())
    monkeypatch.setattr(hylo_distillation, "_RUNTIME", None)
    removed = []
    monkeypatch.setattr(hylo_distillation.HyLoRuntime, "remove_hooks", lambda self: removed.append(self))

    args = SimpleNamespace(hylo_distill_stage="SFT")
    assert (
        hylo_distillation.hylo_builder(args, True, False, None, config="cfg", pg_collection="pg")
        == "student1"
    )
    first = hylo_distillation._RUNTIME
    assert hylo_distillation.hylo_builder(args, True, True) == "student2"
    assert students == [(True, False, None, "cfg", "pg"), (True, True, None, None, None)]
    assert first.stage == "sft" and hylo_distillation._RUNTIME is not first and removed == [first]


def test_replay_accepts_one_micro_batch_per_rank_per_step():
    hylo_distillation._check_replay_args(_args())
    hylo_distillation._check_replay_args(_args(tensor_model_parallel_size=2, global_batch_size=48))


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"global_batch_size": 192}, "global_batch_size=192"),
        ({"world_size": 4}, r"12 x 4"),
        ({"tensor_model_parallel_size": 2}, r"12 x 4"),
        ({"context_parallel_size": 2}, "context_parallel_size"),
    ],
)
def test_replay_rejects_runs_that_cannot_follow_the_recording(overrides, match):
    with pytest.raises(ValueError, match=match):
        hylo_distillation._check_replay_args(_args(**overrides))


@pytest.fixture
def run_patch(monkeypatch):
    # The SFT trainer imports create_sft_forward_step from this module when it trains.
    monkeypatch.setattr(sft_forward_step, "create_sft_forward_step", sft_forward_step.create_sft_forward_step)

    def run(args):
        monkeypatch.setattr(hylo_patches, "get_args", lambda ctx: args)
        hylo_patches.patch_hylo(None)

    return run


def test_patch_switches_replay_to_the_sequential_sampler(run_patch):
    args = _args()
    original = sft_forward_step.create_sft_forward_step
    run_patch(args)
    assert args.dataloader_type == "single"
    assert sft_forward_step.create_sft_forward_step is original


def test_patch_gives_the_sft_trainer_the_hylo_forward_step(run_patch):
    args = _args(stage="sft", hylo_distill_stage="sft", hylo_order_data_dir=None)
    run_patch(args)
    assert sft_forward_step.create_sft_forward_step is hylo_distillation.create_sft_forward_step
    assert args.dataloader_type == "cyclic"


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"global_batch_size": 24}, "global_batch_size"),
        ({"hylo_distill_stage": "kd"}, "hylo_distill_stage"),
        ({"lora": SimpleNamespace(enabled=True)}, "LoRA"),
        ({"hylo_ce_weight": 0.5}, "hylo_ce_weight"),
    ],
)
def test_patch_stops_the_run_on_settings_hylo_does_not_support(run_patch, overrides, match):
    args = _args(stage="sft", **overrides)
    original = sft_forward_step.create_sft_forward_step
    with pytest.raises(SystemExit, match=match):
        run_patch(args)
    assert sft_forward_step.create_sft_forward_step is original
    assert args.dataloader_type == "cyclic"


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
    hybrid = _module(
        "pretrain_hybrid",
        train_valid_test_datasets_provider=lambda *a, **k: original.append((a, k)),
    )
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
    return SimpleNamespace(args=args, original=original)


def test_provider_serves_the_recording_or_the_hybrid_scripts_datasets(megatron_0_19):
    train, valid, test = hylo_distillation.train_valid_test_datasets_provider([6, 0, 0])
    assert isinstance(train, HyLoOrderDataset) and train.steps == 3
    assert (valid, test) == (None, None)

    megatron_0_19.args.hylo_order_data_dir = None
    hylo_distillation.train_valid_test_datasets_provider([6, 0, 0], vp_stage=1)
    assert megatron_0_19.original == [(([6, 0, 0],), {"vp_stage": 1})]


def test_provider_rejects_a_recording_shorter_than_the_run(megatron_0_19):
    megatron_0_19.args.train_iters = 4
    with pytest.raises(ValueError, match="fewer than train_iters=4"):
        hylo_distillation.train_valid_test_datasets_provider([8, 0, 0])
