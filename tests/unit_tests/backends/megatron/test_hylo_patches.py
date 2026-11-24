###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from primus.backends.megatron.core.models.registry import get_model_family
from primus.backends.megatron.patches import hylo_patches
from primus.backends.megatron.sft import forward_step as sft_forward_step
from primus.backends.megatron.training import hylo_distillation


def _args(**overrides):
    args = SimpleNamespace(
        model_type="hylo",
        hylo_distill_stage="ild",
        hylo_teacher_checkpoint="/teacher",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
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
    monkeypatch.setattr(hylo_distillation.HyLoRuntime, "_install_teacher_hooks", lambda self: None)

    args = SimpleNamespace(hylo_distill_stage="ILD")
    assert (
        hylo_distillation.hylo_builder(args, True, False, None, config="cfg", pg_collection="pg")
        == "student1"
    )
    first = hylo_distillation._RUNTIME
    assert hylo_distillation.hylo_builder(args, True, True) == "student2"
    assert students == [(True, False, None, "cfg", "pg"), (True, True, None, None, None)]
    assert first.stage == "ild" and hylo_distillation._RUNTIME is not first and removed == [first]


@pytest.fixture
def run_patch(monkeypatch):
    # The SFT trainer imports create_sft_forward_step from this module when it trains.
    monkeypatch.setattr(sft_forward_step, "create_sft_forward_step", sft_forward_step.create_sft_forward_step)

    def run(args):
        monkeypatch.setattr(hylo_patches, "get_args", lambda ctx: args)
        hylo_patches.patch_hylo(None)

    return run


def test_patch_gives_the_sft_trainer_the_hylo_forward_step(run_patch):
    args = _args(stage="sft")
    run_patch(args)
    assert sft_forward_step.create_sft_forward_step is hylo_distillation.create_sft_forward_step


@pytest.mark.parametrize(
    "overrides, match",
    [
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


def test_provider_serves_the_hybrid_scripts_datasets(monkeypatch):
    # Megatron-core 0.19: pretrain_mamba is a shim and the provider lives in pretrain_hybrid.
    original = []
    hybrid = _module(
        "pretrain_hybrid",
        train_valid_test_datasets_provider=lambda *a, **k: original.append((a, k)),
    )
    monkeypatch.setitem(sys.modules, "pretrain_mamba", _module("pretrain_mamba"))
    monkeypatch.setitem(sys.modules, "pretrain_hybrid", hybrid)
    hylo_distillation.train_valid_test_datasets_provider([6, 0, 0], vp_stage=1)
    assert original == [(([6, 0, 0],), {"vp_stage": 1})]
