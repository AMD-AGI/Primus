###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from primus.backends.megatron.training.hylo_distillation import (
    HyLoRuntime,
    enhanced_ild_loss,
    masked_chunked_kl,
    validate_hylo_args,
)
from primus.core.config.primus_config import load_primus_config
from primus.core.utils import file_utils

ROOT = Path(__file__).resolve().parents[4]


def test_enhanced_ild_matches_hylo_equation():
    teacher_hidden = [torch.zeros(2, 3, 4), torch.ones(2, 3, 4)]
    student_hidden = [torch.ones(2, 3, 4), torch.full((2, 3, 4), 3.0)]
    teacher_mixer = [torch.full((2, 3, 4), 2.0), torch.full((2, 3, 4), 4.0)]
    student_mixer = [torch.full((2, 3, 4), 5.0), torch.full((2, 3, 4), 8.0)]
    loss, parts = enhanced_ild_loss(
        student_hidden, teacher_hidden, student_mixer, teacher_mixer
    )
    expected_hidden = sum(
        torch.norm(s - t, p=2, dim=-1).mean()
        for s, t in zip(student_hidden, teacher_hidden)
    )
    expected_mixer = sum(
        torch.norm(s - t, p=2, dim=-1).mean()
        for s, t in zip(student_mixer, teacher_mixer)
    )
    assert torch.equal(loss, expected_hidden + expected_mixer)
    assert torch.equal(parts["hylo hidden ild"], expected_hidden)
    assert torch.equal(parts["hylo mixer ild"], expected_mixer)


def test_masked_chunked_kl_matches_dense_reference_and_has_grad():
    generator = torch.Generator().manual_seed(7)
    student = torch.randn(2, 5, 11, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 5, 11, generator=generator)
    mask = torch.tensor([[1, 1, 0, 1, 0], [0, 1, 1, 1, 1]], dtype=torch.float32)
    actual, count = masked_chunked_kl(student, teacher, mask, chunk_size=3)
    per_token = F.kl_div(
        F.log_softmax(student.float(), dim=-1),
        F.softmax(teacher.float(), dim=-1),
        reduction="none",
    ).sum(dim=-1)
    expected = (per_token * mask).sum()
    torch.testing.assert_close(actual, expected)
    assert count.item() == int(mask.sum())
    actual.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad[mask == 0]) == 0


def test_validate_hylo_args_rejects_unsupported_parallelism():
    base = dict(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        hylo_distill_stage="sft",
        hylo_teacher_checkpoint="/tmp/teacher",
        hylo_ce_weight=0.0,
    )
    validate_hylo_args(Namespace(**base))
    with pytest.raises(ValueError, match="tensor_model_parallel_size=1"):
        validate_hylo_args(Namespace(**{**base, "tensor_model_parallel_size": 2}))
    with pytest.raises(ValueError, match="pipeline_model_parallel_size=1"):
        validate_hylo_args(Namespace(**{**base, "pipeline_model_parallel_size": 2}))
    with pytest.raises(ValueError, match="hylo_ce_weight=0"):
        validate_hylo_args(Namespace(**{**base, "hylo_ce_weight": 1.0}))
    with pytest.raises(ValueError, match="hylo_distill_stage"):
        validate_hylo_args(Namespace(**{**base, "hylo_distill_stage": "pretrain"}))
    with pytest.raises(ValueError, match="hylo_teacher_checkpoint"):
        validate_hylo_args(Namespace(**{**base, "hylo_teacher_checkpoint": None}))


def test_enhanced_ild_rejects_mismatched_captures():
    hidden = [torch.zeros(1, 2)]
    with pytest.raises(RuntimeError, match="capture mismatch"):
        enhanced_ild_loss(hidden, [], hidden, hidden)


def test_masked_chunked_kl_rejects_shape_mismatch():
    student = torch.zeros(2, 3)
    teacher = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="student logits"):
        masked_chunked_kl(student, teacher)


class _Symbol:
    def __init__(self, value):
        self.value = value


class _Attention(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class _TeacherLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = _Attention()

    def forward(self, hidden_states):
        return hidden_states


class _Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([_TeacherLayer()])


class _Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = nn.Identity()
        self.seen = None

    def forward(self, hidden_states=None):
        self.seen = hidden_states.detach().clone()
        return hidden_states


class _MLP(nn.Module):
    def forward(self, hidden_states=None):
        return hidden_states


class _Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.mixer = _Mixer()
        self.decoder.mlp = _MLP()
        self.decoder.layer_type_list = [_Symbol("M"), _Symbol("-")]
        self.decoder.layers = nn.ModuleList([self.decoder.mixer, self.decoder.mlp])


def test_teacher_forcing_replaces_only_ild_mixer_inputs():
    runtime = HyLoRuntime(_Teacher(), "ild")
    forced = torch.arange(4, dtype=torch.float32).view(1, 1, 4)
    runtime.teacher_capture.layer_inputs.append(forced)
    student = _Student()
    runtime.install_student_hooks(student)
    original = torch.zeros(1, 1, 4)

    student.decoder.layers[0](hidden_states=original)
    assert torch.equal(student.decoder.mixer.seen, forced)

    runtime.stage = "sft"
    student.decoder.layers[0](hidden_states=original)
    assert torch.equal(student.decoder.mixer.seen, original)


def test_hylo_recipes_match_stage_contract(monkeypatch):
    monkeypatch.setattr(file_utils, "create_path_if_not_exists", lambda *args, **kwargs: None)
    recipes = {
        "examples/megatron/configs/MI300X/hylo_gdn_1B-enhanced-ild.yaml": ("ild", 4096, 0.0),
        "examples/megatron/configs/MI300X/hylo_mla_1B-enhanced-ild.yaml": ("ild", 4096, 1.0),
        "examples/megatron/configs/MI300X/hylo_4MLA12GDN_1B-8K-sft.yaml": ("sft", 8192, 0.25),
    }
    for relative, (stage, seq_length, attention_ratio) in recipes.items():
        module = load_primus_config(ROOT / relative, None).modules[0]
        params = module.params
        assert params.hylo_distill_stage == stage
        assert params.hylo_kl_weight == 1.0
        assert params.hylo_ce_weight == 0.0
        assert params.seq_length == seq_length
        assert params.tensor_model_parallel_size == 1
        assert params.pipeline_model_parallel_size == 1
        assert params.hybrid_attention_ratio == attention_ratio
        assert params.finetune is True
        assert params.no_load_optim is True
        assert params.no_load_rng is True
