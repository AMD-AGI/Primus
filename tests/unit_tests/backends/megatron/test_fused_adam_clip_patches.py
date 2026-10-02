###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus.backends.megatron.patches.te_patches.fused_adam_clip_patches import (
    _clip_coeff,
    _get_grad_norm_tensor,
)


@pytest.mark.parametrize(
    ("total_norm", "max_norm", "expected"),
    [
        (0.25, 1.0, 1.0),
        (1.0, 1.0, 1.0 / 1.000001),
        (2.0, 1.0, 1.0 / 2.000001),
        (8.0, 2.0, 2.0 / 8.000001),
    ],
)
def test_clip_coeff(total_norm, max_norm, expected):
    assert _clip_coeff(total_norm, max_norm) == pytest.approx(expected)


def test_grad_norm_stays_on_device(monkeypatch):
    from megatron.core.optimizer import clip_grads

    class FakeOptimizer:
        def __init__(self):
            self.grad = torch.ones(4, dtype=torch.float32)
            self.group = object()

        def get_main_grads_for_grad_norm(self):
            return [self.grad]

        def get_grad_stats_parallel_group(self):
            return self.group

    optimizer = FakeOptimizer()
    monkeypatch.setattr(
        clip_grads,
        "multi_tensor_applier",
        lambda impl, overflow, tensor_lists, per_parameter: (
            torch.tensor([3.0]),
            None,
        ),
    )

    def fake_all_reduce(value, op, group):
        assert op == torch.distributed.ReduceOp.SUM
        assert group is optimizer.group
        value.mul_(4.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    norm = _get_grad_norm_tensor(optimizer)

    assert isinstance(norm, torch.Tensor)
    assert norm.device == optimizer.grad.device
    assert torch.equal(norm, torch.tensor([6.0]))
    assert optimizer._primus_norm_overflow_buf.dtype == torch.int32
