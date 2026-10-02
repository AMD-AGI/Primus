###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus.backends.megatron.patches.te_patches.fused_adam_clip_patches import (
    _chained_step_with_device_clip,
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


def test_chained_step_forwards_device_norm_to_each_adam(monkeypatch):
    class FakeAdam:
        pass

    class FakeChildOptimizer:
        def __init__(self, clip_grad):
            self.optimizer = FakeAdam()
            self.config = type("Config", (), {"clip_grad": clip_grad})()

        def get_parameters(self):
            return [object()]

    class FakeChainedOptimizer:
        def __init__(self):
            self.chained_optimizers = [FakeChildOptimizer(1.0), FakeChildOptimizer(2.0)]
            self.config = type("Config", (), {"log_num_zeros_in_grad": True})()
            self.stepped = False

        def prepare_grads(self):
            return False

        def grads_states_parallel_group_is_shared(self):
            return True

        def count_zeros(self):
            return 7.0

        def step_with_ready_grads(self):
            self.stepped = True
            return True

    norm = torch.tensor([3.0])
    monkeypatch.setattr(
        "primus.backends.megatron.patches.te_patches.fused_adam_clip_patches."
        "_get_grad_norm_tensor",
        lambda optimizer: norm,
    )
    optimizer = FakeChainedOptimizer()

    success, returned_norm, num_zeros = _chained_step_with_device_clip(
        optimizer, FakeAdam
    )

    assert success
    assert returned_norm is None
    assert num_zeros == 7.0
    assert optimizer.stepped
    assert [
        child.optimizer._primus_clip_norm for child in optimizer.chained_optimizers
    ] == [norm, norm]
    assert [
        child.optimizer._primus_clip_max_norm
        for child in optimizer.chained_optimizers
    ] == [1.0, 2.0]
