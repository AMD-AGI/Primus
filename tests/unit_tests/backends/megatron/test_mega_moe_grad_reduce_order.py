###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MegaMoE backward stages wait for the overlapped grad reduce-scatters issued before them.

No GPU and no Turbo kernels: the stages are stand-in autograd functions and the reduce-scatters
are stand-in bucket groups, so this checks the order and the bookkeeping only.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from primus.backends.megatron.patches.turbo import mega_moe_patches
from primus.backends.megatron.patches.turbo.mega_moe_patches import (
    _is_turbo_mega_moe_grad_reduce_ordered,
    finish_issued_grad_reduce,
    patch_mega_moe_grad_reduce_order,
)
from primus.core.patches import PatchContext

_MEGA_MOE = Path(__file__).resolve().parents[4] / "primus/backends/megatron/core/extensions/mega_moe.py"


def _stage(events, name):
    """An autograd function standing in for a MegaMoE stage; its backward logs ``name``."""

    class Stage(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            return x * 2.0, x.detach().clone()

        @staticmethod
        def backward(ctx, grad_out, _grad_aux):
            events.append(name)
            return grad_out * 2.0, None

    return Stage.apply


@pytest.fixture
def mega_moe(monkeypatch):
    """``mega_moe.py`` under a private name with stand-in stages; ``events`` logs their backwards."""
    events = []
    moe = "primus_turbo.pytorch.ops.moe"
    for name in ("primus_turbo", "primus_turbo.pytorch", "primus_turbo.pytorch.ops", moe):
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, name, package)
    stage1, stage2 = _stage(events, "stage1"), _stage(events, "stage2")
    ops = ModuleType("primus_turbo.pytorch.ops.moe.fused_mega_moe")
    ops.GLUActivation = object
    ops.fused_mega_moe_stage1 = lambda x, idx, weights, w1, group, **kw: (*stage1(x, w1), "handle")
    ops.fused_mega_moe_stage2 = lambda l1, dwib, handle, idx, weights, w2, group, act, **kw: stage2(l1, w2)[0]
    fp8_ops = ModuleType("primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8")
    fp8_ops.fused_mega_moe_fp8_stage1 = lambda x, idx, weights, w1, group: (*stage1(x, w1), "handle", "state")
    fp8_ops.fused_mega_moe_fp8_stage2 = (
        lambda l1, dwib, handle, state, idx, weights, w2, group, act, **kw: stage2(l1, w2)[0]
    )
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    monkeypatch.setitem(sys.modules, fp8_ops.__name__, fp8_ops)

    spec = importlib.util.spec_from_file_location("mega_moe_under_test", _MEGA_MOE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.events = events
    return module


def _experts():
    """What ``MegaMoEExperts.forward`` reads off ``self``."""
    weight = torch.ones(1, requires_grad=True)
    return SimpleNamespace(
        fc1_weight=lambda: (weight, None),
        fc2_weight=lambda: (weight, None),
        wgrad_accum_pattern=None,
        ep_group=None,
        activation=None,
    )


@pytest.mark.parametrize("experts_cls", ["MegaMoEExperts", "MegaMoEFP8Experts"])
def test_callback_runs_right_before_each_backward_stage(mega_moe, experts_cls):
    mega_moe.before_backward_stage = lambda: mega_moe.events.append("callback")
    x = torch.ones(4, requires_grad=True)
    out = getattr(mega_moe, experts_cls).forward(_experts(), x, None, None)
    # autograd work that is not a MegaMoE stage, on both sides of the layer
    (out.exp().sum() + x.sum()).backward()
    assert mega_moe.events == ["callback", "stage2", "callback", "stage1"]
    # out = 4 x through the two stand-in stages: the hooks leave the gradient alone
    assert torch.allclose(x.grad, 4.0 * torch.full((4,), 4.0).exp() + 1.0)


def test_callback_does_not_change_the_gradient(mega_moe):
    def grad(callback):
        mega_moe.before_backward_stage = callback
        x = torch.arange(4.0, requires_grad=True)
        mega_moe.MegaMoEExperts.forward(_experts(), x, None, None).square().sum().backward()
        return x.grad

    assert torch.equal(grad(lambda: None), grad(None))


def test_no_hook_without_a_callback_or_without_grad(mega_moe):
    x = torch.ones(4, requires_grad=True)
    out = mega_moe.MegaMoEExperts.forward(_experts(), x, None, None)
    assert not out._backward_hooks  # no callback set

    mega_moe.before_backward_stage = lambda: mega_moe.events.append("callback")
    with torch.no_grad():
        mega_moe.MegaMoEExperts.forward(_experts(), x, None, None)
    assert mega_moe.events == []


def test_callback_dropped_after_the_forward_is_not_called(mega_moe):
    mega_moe.before_backward_stage = lambda: mega_moe.events.append("callback")
    x = torch.ones(4, requires_grad=True)
    out = mega_moe.MegaMoEExperts.forward(_experts(), x, None, None)
    mega_moe.before_backward_stage = None
    out.sum().backward()
    assert mega_moe.events == ["stage2", "stage1"]


class _BucketGroup:
    def __init__(self, issued):
        self.grad_reduce_handle = object() if issued else None
        self.finished = 0

    def finish_grad_sync(self):
        # like Megatron's: waits, then drops the handle
        assert self.grad_reduce_handle is not None
        self.grad_reduce_handle = None
        self.finished += 1


def test_only_issued_reduce_scatters_are_finished():
    groups = [_BucketGroup(issued) for issued in (True, False, True)]
    expert_groups = [_BucketGroup(issued) for issued in (False, True)]
    model = SimpleNamespace(bucket_groups=groups, expert_parallel_bucket_groups=expert_groups)

    assert finish_issued_grad_reduce([model]) == 3
    assert [g.finished for g in groups + expert_groups] == [1, 0, 1, 0, 1]
    # nothing left to finish: the next stage of the same backward waits for nothing
    assert finish_issued_grad_reduce([model]) == 0
    assert [g.finished for g in groups + expert_groups] == [1, 0, 1, 0, 1]


def _ctx(**params):
    params = dict(dict(overlap_grad_reduce=True, use_turbo_mega_moe=True), **params)
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**params))},
    )


@pytest.mark.parametrize(
    "params, expected",
    [
        ({}, True),
        ({"turbo_mega_moe_ordered_grad_reduce": None}, True),
        ({"turbo_mega_moe_ordered_grad_reduce": True}, True),
        ({"turbo_mega_moe_ordered_grad_reduce": False}, False),
        ({"overlap_grad_reduce": False}, False),
        ({"overlap_grad_reduce": False, "turbo_mega_moe_ordered_grad_reduce": True}, False),
        ({"use_megatron_fsdp": True}, False),
        ({"use_torch_fsdp2": True}, False),
    ],
)
def test_condition(monkeypatch, params, expected):
    monkeypatch.setattr(mega_moe_patches, "_is_turbo_mega_moe_can_patch", lambda _ctx: True)
    assert _is_turbo_mega_moe_grad_reduce_ordered(_ctx(**params)) is expected


def test_condition_needs_mega_moe(monkeypatch):
    monkeypatch.setattr(mega_moe_patches, "_is_turbo_mega_moe_can_patch", lambda _ctx: False)
    assert _is_turbo_mega_moe_grad_reduce_ordered(_ctx()) is False


@pytest.fixture
def megatron_ddp(monkeypatch):
    """Stand-ins for the two Megatron classes the patch touches, and for Primus' MegaMoE module."""

    class _ParamAndGradBucketGroup:
        def finish_grad_sync(self):
            if self.grad_reduce_finished:
                return
            self.grad_reduce_handle = None
            self.grad_reduce_finished = True

    class DistributedDataParallel:
        def __init__(self, issued):
            self.bucket_groups = [_group(_ParamAndGradBucketGroup, i) for i in issued]
            self.expert_parallel_bucket_groups = []

    def _group(cls, issued):
        group = cls()
        group.grad_reduce_handle = object() if issued else None
        group.grad_reduce_finished = False
        return group

    for name in ("megatron", "megatron.core", "megatron.core.distributed"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    ddp = ModuleType("megatron.core.distributed.distributed_data_parallel")
    ddp.DistributedDataParallel = DistributedDataParallel
    buffers = ModuleType("megatron.core.distributed.param_and_grad_buffer")
    buffers._ParamAndGradBucketGroup = _ParamAndGradBucketGroup
    mega_moe = ModuleType("primus.backends.megatron.core.extensions.mega_moe")
    mega_moe.before_backward_stage = None
    for module in (ddp, buffers, mega_moe):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    # `from primus.backends.megatron.core.extensions import mega_moe` reads the package attribute
    import primus.backends.megatron.core.extensions as extensions

    monkeypatch.setattr(extensions, "mega_moe", mega_moe, raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return SimpleNamespace(ddp=ddp, buffers=buffers, mega_moe=mega_moe)


def _pending(model):
    return [group.grad_reduce_handle is not None for group in model.bucket_groups]


@pytest.mark.parametrize(
    "option, gfx1250, ordered", [(None, True, True), (None, False, False), (True, False, True)]
)
def test_patch_finishes_issued_reduce_scatters_of_models_built_after_it(
    monkeypatch, megatron_ddp, option, gfx1250, ordered
):
    import primus.core.utils.gpu_arch as gpu_arch

    monkeypatch.setattr(gpu_arch, "is_gfx1250", lambda: gfx1250)
    patch_mega_moe_grad_reduce_order(_ctx(turbo_mega_moe_ordered_grad_reduce=option))
    callback = megatron_ddp.mega_moe.before_backward_stage
    assert callback is not None

    model = megatron_ddp.ddp.DistributedDataParallel([True, False, True])
    callback()
    assert _pending(model) == ([False, False, False] if ordered else [True, False, True])
    # null on another GPU: the callback takes itself out, so later forwards register no hook
    assert (megatron_ddp.mega_moe.before_backward_stage is None) == (not ordered)


def test_patch_leaves_a_captured_backward_alone(monkeypatch, megatron_ddp):
    patch_mega_moe_grad_reduce_order(_ctx(turbo_mega_moe_ordered_grad_reduce=True))
    model = megatron_ddp.ddp.DistributedDataParallel([True])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    megatron_ddp.mega_moe.before_backward_stage()
    assert _pending(model) == [True]


def test_patch_turns_itself_off_when_megatron_cannot_finish_early(megatron_ddp):
    patch_mega_moe_grad_reduce_order(_ctx(turbo_mega_moe_ordered_grad_reduce=True))
    model = megatron_ddp.ddp.DistributedDataParallel([True, True])
    # a Megatron whose finish_grad_sync asserts on a second call keeps no such flag
    del model.bucket_groups[1].grad_reduce_finished
    megatron_ddp.mega_moe.before_backward_stage()
    assert _pending(model) == [True, True]
    assert megatron_ddp.mega_moe.before_backward_stage is None


def test_patch_does_not_depend_on_who_else_wrapped_finish_grad_sync(megatron_ddp):
    cls = megatron_ddp.buffers._ParamAndGradBucketGroup
    original = cls.finish_grad_sync
    cls.finish_grad_sync = lambda self: original(self)  # another patch's wrapper, no functools.wraps
    patch_mega_moe_grad_reduce_order(_ctx(turbo_mega_moe_ordered_grad_reduce=True))
    model = megatron_ddp.ddp.DistributedDataParallel([True, False])
    megatron_ddp.mega_moe.before_backward_stage()
    assert _pending(model) == [False, False]
