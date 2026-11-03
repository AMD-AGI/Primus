###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The Inkling padded-label patch checks every queued count after each forward-backward pass
(before the optimizer step), summed over ranks, and drops the counts of a pass that raised."""

import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")

from primus.backends.megatron.core.models.inkling import inkling_model  # noqa: E402
from primus.backends.megatron.patches import (  # noqa: E402
    inkling_padded_label_patches as pl,
)
from primus.core.patches.context import PatchContext  # noqa: E402
from primus.core.patches.patch_registry import PatchRegistry  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_checks(monkeypatch):
    monkeypatch.setattr(inkling_model, "_LIVE_CHECKS", inkling_model.weakref.WeakSet())


@pytest.fixture
def fake_training(monkeypatch):
    """Fake ``megatron.training.training`` / ``megatron.core.pipeline_parallel`` whose schedule
    runs a given micro-batch function (the model's forward, queuing counts) and records calls."""
    calls = []

    def schedule(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs.get("forward_step_func", lambda: None)()
        return "LOSSES"

    mod = types.ModuleType("megatron.training.training")
    mod.get_forward_backward_func = lambda: schedule
    pp = types.ModuleType("megatron.core.pipeline_parallel")
    pp.get_forward_backward_func = lambda: schedule
    pkg = types.ModuleType("megatron.training")
    pkg.training = mod
    core = types.ModuleType("megatron.core")
    core.pipeline_parallel = pp
    root = types.ModuleType("megatron")
    root.training, root.core = pkg, core
    monkeypatch.setitem(sys.modules, "megatron", root)
    monkeypatch.setitem(sys.modules, "megatron.training", pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.pipeline_parallel", pp)
    return SimpleNamespace(mod=mod, pp=pp, calls=calls)


def _ctx(model_type="inkling", unpadded_vocab_size=90):
    params = SimpleNamespace(model_type=model_type, inkling_unpadded_vocab_size=unpadded_vocab_size)
    return PatchContext(
        backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=params)}
    )


class _QueuedCheck(inkling_model._PaddedLabelCheck):
    """A check whose counts are queued (the GPU path) on any device."""

    def __call__(self, scored, n):
        self.pending.append(scored.sum())


@pytest.mark.parametrize("model_type,applies", [("inkling", True), ("gpt", False), (None, False)])
def test_condition_only_for_inkling(model_type, applies):
    assert bool(PatchRegistry.get(pl._PATCH_KEY).condition(_ctx(model_type))) is applies


def test_both_getters_are_wrapped_once_and_forward_everything(fake_training):
    pl.patch_inkling_padded_label_check(_ctx())
    getters = (fake_training.mod.get_forward_backward_func, fake_training.pp.get_forward_backward_func)
    pl.patch_inkling_padded_label_check(_ctx())  # re-apply: no second wrap
    assert (
        fake_training.mod.get_forward_backward_func,
        fake_training.pp.get_forward_backward_func,
    ) == getters
    for get in getters:
        assert get()("A", 1, forward_step_func=lambda: None, k="v") == "LOSSES"
    assert [
        (a, {k: v for k, v in kw.items() if k != "forward_step_func"}) for a, kw in fake_training.calls
    ] == [(("A", 1), {"k": "v"})] * 2


def test_a_bad_count_queued_during_the_pass_fails_right_after_it(fake_training):
    """The count is queued by the pass itself (the last micro-batch), and the error comes when the
    pass returns: before train_step's optimizer step."""
    pl.patch_inkling_padded_label_check(_ctx())
    check = _QueuedCheck()
    fbf = fake_training.mod.get_forward_backward_func()
    fbf(forward_step_func=lambda: check(torch.tensor([False, False]), 90))  # clean
    with pytest.raises(ValueError, match=r"^Inkling: 3 label\(s\) >= inkling_unpadded_vocab_size=90 "):
        fbf(forward_step_func=lambda: check(torch.tensor([True, True, True]), 90))
    assert check.pending == []


def test_counts_of_a_pass_that_raised_are_dropped(fake_training):
    pl.patch_inkling_padded_label_check(_ctx())
    check = _QueuedCheck()
    fbf = fake_training.mod.get_forward_backward_func()

    def boom():
        check(torch.tensor([True]), 90)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        fbf(forward_step_func=boom)
    fbf(forward_step_func=lambda: None)  # the earlier bad count does not leak into this step


def test_the_pass_error_survives_the_cleanup(fake_training, monkeypatch):
    """Dropping the queued counts after a failed pass never reads them (no device access), so
    the pass's own exception is the one that propagates, even when take() would fail."""
    pl.patch_inkling_padded_label_check(_ctx())
    check = _QueuedCheck()

    def broken_take():
        raise RuntimeError("HIP error: device fault")

    monkeypatch.setattr(check, "take", broken_take)
    fbf = fake_training.mod.get_forward_backward_func()

    def boom():
        check(torch.tensor([True]), 90)
        raise RuntimeError("the real cause")

    with pytest.raises(RuntimeError, match="the real cause"):
        fbf(forward_step_func=boom)
    assert check.pending == []


def test_a_cpu_rank_queues_inside_the_pass_and_raises_on_every_rank(fake_training, monkeypatch):
    """CPU / gloo too: inside the wrapped pass the real check queues (no rank raises alone, which
    would leave the others in the flush's all-reduce); the flush raises on every rank, also on a
    rank that saw no bad label."""
    pl.patch_inkling_padded_label_check(_ctx())
    check = inkling_model._PaddedLabelCheck()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda *a: "gloo")
    reduced = []

    def all_reduce(t):
        reduced.append(int(t))
        t += 2  # the other rank's bad labels

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    queued = []
    fbf = fake_training.mod.get_forward_backward_func()

    def step():
        check(torch.tensor([True, False]), 90)  # this rank's bad label: queued, not raised
        queued.append(len(check.pending))

    with pytest.raises(ValueError, match=r"^Inkling: 3 label\(s\)"):
        fbf(forward_step_func=step)
    assert queued == [1] and reduced == [1] and check.pending == []


def test_outside_the_pass_a_check_raises_at_once_and_queues_nothing():
    """A forward outside the patched schedule (eval script, converter, teacher model) is checked
    on its own rank at once: nothing accumulates for a later flush to misattribute."""
    check = inkling_model._PaddedLabelCheck()
    check(torch.tensor([False, False]), 90)
    with pytest.raises(ValueError, match=r"^Inkling: 2 label\(s\)"):
        check(torch.tensor([True, True]), 90)
    assert check.pending == []


def test_the_count_is_summed_over_ranks(monkeypatch):
    """With torch.distributed up, the count is all-reduced: a rank that saw no bad label raises too."""
    check = _QueuedCheck()
    reduced = []

    def all_reduce(t):
        reduced.append(int(t))
        t += 4  # another rank's bad labels

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    check(torch.tensor([False]), 90)
    with pytest.raises(ValueError, match=r"^Inkling: 4 label\(s\)"):
        inkling_model.flush_padded_label_checks(90)
    assert reduced == [0]


def test_a_gloo_group_flushes_on_the_cpu_without_a_gpu(monkeypatch):
    """A CPU run (gloo group) with nothing queued all-reduces a CPU count: no CUDA device is asked for."""
    check = _QueuedCheck()  # noqa: F841 (kept alive: _LIVE_CHECKS holds weak references)
    reduced = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda *a: "gloo")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda t: reduced.append(t.device.type))

    def no_gpu(*_a, **_k):
        raise RuntimeError("No HIP GPUs are available")

    monkeypatch.setattr(torch.cuda, "current_device", no_gpu)
    inkling_model.flush_padded_label_checks(90)
    assert reduced == ["cpu"]


def test_no_collective_when_the_run_checks_no_padded_labels(monkeypatch):
    """inkling_unpadded_vocab_size unset in the run's config: no collective and no host read, even
    if a stray check with queued counts is alive on this rank (dropped)."""
    check = _QueuedCheck()
    check(torch.tensor([True]), 90)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fail(*_a, **_k):
        raise AssertionError("no collective expected")

    monkeypatch.setattr(torch.distributed, "all_reduce", fail)
    inkling_model.flush_padded_label_checks(None)
    assert check.pending == []


def test_every_rank_joins_the_collective_even_with_no_live_model(monkeypatch):
    """Participation follows the run's config, not the models alive on the rank: a rank whose
    model is gone (or was never built) still joins the all-reduce, and raises with the others."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda *a: "gloo")
    reduced = []

    def all_reduce(t):
        reduced.append(int(t))
        t += 1

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    assert len(inkling_model._LIVE_CHECKS) == 0
    with pytest.raises(ValueError, match=r"^Inkling: 1 label\(s\)"):
        inkling_model.flush_padded_label_checks(90)
    assert reduced == [0]


def test_the_patch_passes_the_run_config_to_the_flush(fake_training, monkeypatch):
    pl.patch_inkling_padded_label_check(_ctx(unpadded_vocab_size=None))
    check = _QueuedCheck()
    fbf = fake_training.mod.get_forward_backward_func()
    fbf(forward_step_func=lambda: check(torch.tensor([True]), 90))  # unchecked run: dropped
    assert check.pending == []
