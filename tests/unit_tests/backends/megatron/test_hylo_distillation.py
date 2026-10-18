###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F

from primus.backends.megatron.training.hylo_distillation import (
    enhanced_ild_loss,
    masked_chunked_kl,
    validate_hylo_args,
)


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
