###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Deferred NaN / Inf validations raise upstream's error, in order, at the flush (GPU tests); the
patch's switches (CPU tests)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from primus.core.patches import PatchContext

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
rsm = pytest.importorskip("megatron.core.rerun_state_machine")
ddp_mod = pytest.importorskip("megatron.core.distributed.distributed_data_parallel")
pgb = pytest.importorskip("megatron.core.distributed.param_and_grad_buffer")
mcore_optimizer = pytest.importorskip("megatron.core.optimizer.optimizer")

_ROOT = Path(__file__).resolve().parents[4]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


deferred = _load(
    "deferred_nan_check_under_test", "primus/backends/megatron/patches/deferred_nan_check_patches.py"
)
fused = _load(
    "fused_grad_check_under_test", "primus/backends/megatron/patches/ddp_grad_check_fused_scaling_patches.py"
)
CTX = PatchContext(backend="megatron", phase="before_train")
_OWNERS = [
    (rsm.RerunStateMachine, "validate_result"),
    (ddp_mod.DistributedDataParallel, "finish_grad_sync"),
    (mcore_optimizer.MixedPrecisionOptimizer, "step"),
    (mcore_optimizer.FP32Optimizer, "step"),
    (mcore_optimizer.ChainedOptimizer, "step"),
    (pgb._ParamAndGradBucketGroup, "check_grads"),
]


@pytest.fixture
def originals():
    """Everything the two patches replace, put back afterwards, with the patches marked unapplied."""
    from primus.backends.megatron.patches import _patch_guard

    saved = {(o, n): o.__dict__[n] for o, n in _OWNERS}
    try:
        from megatron.training import training

        saved_eval = training.evaluate
    except ImportError:
        training, saved_eval = None, None
    yield saved
    deferred._PENDING.clear()
    deferred._GRADS_DEFERRABLE = None
    for (o, n), fn in saved.items():
        setattr(o, n, fn)
    if training is not None:
        training.evaluate = saved_eval
    for owner, key in (
        (rsm.RerunStateMachine, deferred._PATCH_KEY),
        (pgb._ParamAndGradBucketGroup, fused._PATCH_KEY),
    ):
        getattr(owner, _patch_guard._SENTINEL_ATTR, set()).discard(key)


@pytest.fixture
def patched(originals):
    deferred.patch_deferred_nan_check(CTX)
    return originals


@pytest.fixture
def patched_in_order(originals):
    """The fused check, then the deferral on top of it, as their priorities order them in a run."""
    fused.patch_fused_grad_check_scaling(CTX)
    deferred.patch_deferred_nan_check(CTX)
    return originals


def _machine():
    return rsm.get_rerun_state_machine()


def _message(fn):
    with pytest.raises(RuntimeError) as err:
        fn()
    return str(err.value).split(": ", 1)[1]


def _check(value, func=torch.isnan, message="found NaN in local forward loss calculation", fatal=True):
    _machine().validate_result(
        result=torch.tensor(value, device="cuda"),
        rejection_func=func,
        message=message,
        tolerance=0.0,
        fatal=fatal,
    )


@needs_gpu
def test_upstream_reference_raises_at_once():
    assert _message(lambda: _check(float("nan")))


@needs_gpu
def test_nan_raises_upstream_error_at_flush(patched):
    _check(1.0)
    _check(float("nan"))  # queued, not raised
    assert len(deferred._PENDING) == 2
    got = _message(deferred.flush)
    assert not deferred._PENDING
    want = _message(
        lambda: deferred._ORIGINAL(
            _machine(),
            torch.tensor(float("nan"), device="cuda"),
            torch.isnan,
            "found NaN in local forward loss calculation",
            None,
            0.0,
            True,
        )
    )
    assert got == want


@needs_gpu
def test_first_queued_failure_wins(patched):
    _check(float("inf"), torch.isinf, "found Inf in local forward loss calculation")
    _check(float("nan"), torch.isnan, "found NaN in local grad norm for bucket #0")
    assert "found Inf in local forward loss calculation" in _message(deferred.flush)


@needs_gpu
def test_finite_flush_is_quiet_and_clears(patched):
    for v in (1.0, 2.0, 0.0):
        _check(v)
        _check(v, torch.isinf)
    deferred.flush()
    assert not deferred._PENDING


@needs_gpu
def test_other_checks_run_at_once(patched):
    with pytest.raises(RuntimeError):  # a host result
        _machine().validate_result(
            result=torch.tensor(float("nan")), rejection_func=torch.isnan, message="cpu", fatal=True
        )
    _check(float("nan"), torch.isnan, "non-fatal", fatal=False)  # warns, never queued
    with pytest.raises(RuntimeError):  # another rejection function
        _machine().validate_result(
            result=torch.tensor(5.0, device="cuda"),
            rejection_func=lambda r: bool(r > 1),
            message="custom",
            fatal=True,
        )
    assert not deferred._PENDING


@needs_gpu
def test_flush_points_are_wrapped(patched):
    for owner, name in _OWNERS[1:5]:
        assert getattr(owner, name).__wrapped__ is patched[(owner, name)]
    calls = []
    wrapped = deferred._flushing(lambda: calls.append(len(deferred._PENDING)))
    _check(1.0)
    wrapped()
    assert calls == [0] and not deferred._PENDING


@needs_gpu
def test_patching_twice_wraps_once(patched):
    deferred.patch_deferred_nan_check(CTX)
    assert (
        rsm.RerunStateMachine.validate_result.__wrapped__
        is patched[(rsm.RerunStateMachine, "validate_result")]
    )
    owner = mcore_optimizer.ChainedOptimizer
    assert owner.step.__wrapped__ is patched[(owner, "step")]


@needs_gpu
def test_fused_grad_check_defers(patched):
    fused.patch_fused_grad_check_scaling(CTX)
    g = torch.full((1000,), 0.5, dtype=torch.bfloat16, device="cuda")
    g[7] = float("nan")
    group = SimpleNamespace(buckets=[SimpleNamespace(grad_data=g, gradient_scaling_factor=0.25)])
    pgb._ParamAndGradBucketGroup.check_grads(group, True, False)  # no host sync, nothing raised
    assert len(deferred._PENDING) == 2
    assert "found NaN in local grad norm for bucket #0" in _message(deferred.flush)


def _nan_group():
    g = torch.full((1000,), 0.5, dtype=torch.bfloat16, device="cuda")
    g[7] = float("nan")
    return SimpleNamespace(buckets=[SimpleNamespace(grad_data=g, gradient_scaling_factor=0.25)])


@needs_gpu
def test_grad_checks_sync_per_group_when_not_deferrable(patched_in_order, monkeypatch):
    monkeypatch.setattr(deferred, "_grad_checks_deferrable", lambda: (False, "test"))
    monkeypatch.setattr(deferred, "_GRADS_DEFERRABLE", None)
    check_grads = pgb._ParamAndGradBucketGroup.check_grads
    # the guard, over the fused check, over upstream's
    assert (
        check_grads.__wrapped__.__wrapped__ is patched_in_order[(pgb._ParamAndGradBucketGroup, "check_grads")]
    )
    message = _message(lambda: check_grads(_nan_group(), True, False))
    assert "found NaN in local grad norm for bucket #0" in message
    assert not deferred._PENDING


@needs_gpu
def test_grad_checks_deferred_when_deferrable(patched_in_order, monkeypatch):
    monkeypatch.setattr(deferred, "_grad_checks_deferrable", lambda: (True, "test"))
    monkeypatch.setattr(deferred, "_GRADS_DEFERRABLE", None)
    pgb._ParamAndGradBucketGroup.check_grads(_nan_group(), True, False)  # nothing raised yet
    assert len(deferred._PENDING) == 2
    assert "found NaN in local grad norm for bucket #0" in _message(deferred.flush)


@needs_gpu  # importing megatron.training initializes aiter, which needs a GPU in ROCm builds
@pytest.mark.parametrize(
    "mega, ep, overlap, queues, expected",
    [
        (False, 4, True, "2", True),  # no MegaMoE
        (True, 1, True, "2", True),  # no expert parallelism
        (True, 4, False, "2", True),  # reduce-scatters after the backward
        (True, 4, True, "2", False),
        (True, 4, True, "4", True),
        (True, 4, True, "8", True),
        (True, 4, True, None, True),  # ROCm's default, 4
    ],
)
def test_grad_checks_deferrable(monkeypatch, mega, ep, overlap, queues, expected):
    from megatron.training import global_vars

    args = SimpleNamespace(
        use_turbo_mega_moe=mega, expert_model_parallel_size=ep, overlap_grad_reduce=overlap
    )
    monkeypatch.setattr(global_vars, "get_args", lambda: args)
    if queues is None:
        monkeypatch.delenv("GPU_MAX_HW_QUEUES", raising=False)
    else:
        monkeypatch.setenv("GPU_MAX_HW_QUEUES", queues)
    assert deferred._grad_checks_deferrable()[0] is expected


@pytest.mark.parametrize(
    "setting, gfx1250, expected",
    [(None, True, True), (None, False, False), (True, False, True), (False, True, False)],
)
def test_on_by_default_on_gfx1250_only(monkeypatch, setting, gfx1250, expected):
    monkeypatch.setattr(deferred, "is_gfx1250", lambda: gfx1250)
    params = SimpleNamespace(defer_nan_check=setting)
    ctx = PatchContext(
        backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=params)}
    )
    assert deferred._enabled(ctx) is expected


def test_trainer_config_defaults_to_gfx1250_only():
    with open(_ROOT / "primus/configs/modules/megatron/trainer_base.yaml") as f:
        assert yaml.safe_load(f)["defer_nan_check"] is None
