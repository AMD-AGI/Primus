###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the deferred NaN / Inf validation patch (``defer_nan_check``).

A deferred check must still stop the step: the NaN / Inf has to raise the original ``RuntimeError`` once
``forward_backward_func`` returns, before ``train_step`` reaches the optimizer, and every check the patch
does not defer has to behave exactly as before.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="defers CUDA tensors only")


@pytest.fixture
def patched(monkeypatch):
    import megatron.training.training as megatron_training
    from megatron.core.rerun_state_machine import RerunMode, RerunStateMachine

    from primus.backends.megatron.patches.defer_nan_check_patches import (
        patch_defer_nan_check,
    )

    calls = []

    def fake_train_step(fwd_step, data_it, model, optimizer, sched, config, forward_backward_func, **kw):
        forward_backward_func()
        calls.append("optimizer.step")
        return "done"

    monkeypatch.setattr(megatron_training, "train_step", fake_train_step)
    monkeypatch.setattr(megatron_training, "evaluate", lambda: "evaluated")
    monkeypatch.setattr(megatron_training, "evaluate_and_print_results", lambda: megatron_training.evaluate())
    monkeypatch.setattr(RerunStateMachine, "validate_result", RerunStateMachine.validate_result)
    monkeypatch.setattr(RerunStateMachine, "_log_validation_error_to_file", lambda self, **kw: None)
    patch_defer_nan_check(None)
    machine = RerunStateMachine(mode=RerunMode.DISABLED)
    return megatron_training, machine, calls


def test_enabled_by_config_field():
    from primus.backends.megatron.patches.defer_nan_check_patches import _enabled
    from primus.core.patches import PatchContext

    def ctx(**params):
        return PatchContext(
            backend="megatron",
            phase="before_train",
            extra={"module_config": SimpleNamespace(params=SimpleNamespace(**params))},
        )

    assert _enabled(ctx(defer_nan_check=True))
    assert not _enabled(ctx(defer_nan_check=False))
    assert not _enabled(ctx())


def _validate(machine, value, rejection_func=torch.isnan, fatal=True):
    machine.validate_result(
        result=torch.tensor(value, device="cuda"),
        rejection_func=rejection_func,
        message="found NaN in local forward loss calculation",
        tolerance=0.0,
        fatal=fatal,
    )


def test_clean_step_reaches_optimizer(patched):
    megatron_training, machine, calls = patched

    def fbf():
        for _ in range(8):
            _validate(machine, 1.5)
            _validate(machine, 1.5, torch.isinf)

    assert megatron_training.train_step(None, None, None, None, None, None, fbf) == "done"
    assert calls == ["optimizer.step"]


@pytest.mark.parametrize("value, rejection_func", [(float("nan"), torch.isnan), (float("inf"), torch.isinf)])
def test_bad_loss_raises_before_optimizer(patched, value, rejection_func):
    megatron_training, machine, calls = patched

    def fbf():
        _validate(machine, 1.5)
        _validate(machine, value, rejection_func)
        _validate(machine, 1.5)

    with pytest.raises(RuntimeError, match="Unexpected result"):
        megatron_training.train_step(None, None, None, None, None, None, fbf)
    assert calls == []


def test_bad_loss_does_not_raise_until_flush(patched):
    _, machine, _ = patched
    _validate(machine, float("nan"))


def test_forward_backward_func_as_keyword(patched):
    megatron_training, machine, calls = patched

    def fbf():
        _validate(machine, float("nan"))

    with pytest.raises(RuntimeError, match="Unexpected result"):
        megatron_training.train_step(None, None, None, None, None, None, forward_backward_func=fbf)
    assert calls == []


def test_evaluate_flushes(patched, monkeypatch):
    megatron_training, machine, _ = patched
    # the megatron.training.evaluate patch replaces evaluate after this patch is installed
    monkeypatch.setattr(megatron_training, "evaluate", lambda: "primus evaluate")
    _validate(machine, float("nan"))
    with pytest.raises(RuntimeError, match="Unexpected result"):
        megatron_training.evaluate_and_print_results()


def test_needs_forward_backward_func_parameter(monkeypatch):
    import megatron.training.training as megatron_training
    from megatron.core.rerun_state_machine import RerunStateMachine

    from primus.backends.megatron.patches.defer_nan_check_patches import (
        patch_defer_nan_check,
    )

    monkeypatch.setattr(megatron_training, "train_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(RerunStateMachine, "validate_result", RerunStateMachine.validate_result)
    with pytest.raises(RuntimeError, match="no forward_backward_func parameter"):
        patch_defer_nan_check(None)


def _check_grad_buckets(monkeypatch, machine, *bucket_grads):
    """Megatron's per-bucket check (``_ParamAndGradBucketGroup.check_grads``) on these grad buffers."""
    from megatron.core.distributed import param_and_grad_buffer

    monkeypatch.setattr(param_and_grad_buffer, "get_rerun_state_machine", lambda: machine)
    group = SimpleNamespace(buckets=[SimpleNamespace(grad_data=g) for g in bucket_grads])
    param_and_grad_buffer._ParamAndGradBucketGroup.check_grads(
        group, check_for_nan_or_inf=True, check_for_large=False
    )


def test_clean_grads_reach_optimizer(patched, monkeypatch):
    megatron_training, machine, calls = patched
    grads = [torch.ones(1024, device="cuda", dtype=torch.bfloat16) for _ in range(3)]

    def fbf():
        _validate(machine, 1.5)
        _check_grad_buckets(monkeypatch, machine, *grads)

    assert megatron_training.train_step(None, None, None, None, None, None, fbf) == "done"
    assert calls == ["optimizer.step"]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_bad_grad_bucket_raises_before_optimizer(patched, monkeypatch, value):
    megatron_training, machine, calls = patched
    bad = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
    bad[7] = value

    def fbf():
        _validate(machine, 1.5)
        _check_grad_buckets(monkeypatch, machine, torch.ones_like(bad), bad)

    with pytest.raises(RuntimeError, match="Unexpected result"):
        megatron_training.train_step(None, None, None, None, None, None, fbf)
    assert calls == []


def test_non_fatal_and_other_checks_run_immediately(patched):
    _, machine, _ = patched
    _validate(machine, float("nan"), fatal=False)
    with pytest.raises(RuntimeError, match="Unexpected result"):
        _validate(machine, 5.0, rejection_func=lambda x: bool(x > 1))
