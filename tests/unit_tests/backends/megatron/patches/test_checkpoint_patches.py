# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for the load_checkpoint wrapper in checkpoint_patches.

Megatron is replaced by fake modules, so the tests run on CPU without
Megatron installed. They check that the wrapper forwards every argument to
the original load_checkpoint (old positional form and the keyword arguments
added by newer Megatron, e.g. tp_group/pp_group), and that the existing
KeyError('model') handling is unchanged.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from primus.core.patches.context import PatchContext


def _build_patch_context():
    backend_args = SimpleNamespace()
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"backend_args": backend_args, "module_config": SimpleNamespace(params=backend_args)},
    )


def _install_fake_megatron(monkeypatch, original_load_checkpoint):
    checkpointing_mod = types.ModuleType("megatron.training.checkpointing")
    checkpointing_mod.load_checkpoint = original_load_checkpoint
    training_mod = types.ModuleType("megatron.training.training")
    training_mod.load_checkpoint = original_load_checkpoint

    training_pkg = types.ModuleType("megatron.training")
    training_pkg.checkpointing = checkpointing_mod
    training_pkg.training = training_mod

    megatron_pkg = types.ModuleType("megatron")
    megatron_pkg.training = training_pkg

    monkeypatch.setitem(sys.modules, "megatron", megatron_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training", training_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.checkpointing", checkpointing_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.training", training_mod)
    return checkpointing_mod, training_mod


def _apply_patch(monkeypatch, original_load_checkpoint):
    from primus.backends.megatron.patches.checkpoint_patches import (
        patch_load_checkpoint_for_distributed,
    )

    checkpointing_mod, training_mod = _install_fake_megatron(monkeypatch, original_load_checkpoint)
    patch_load_checkpoint_for_distributed(_build_patch_context())
    assert checkpointing_mod.load_checkpoint is training_mod.load_checkpoint
    assert checkpointing_mod.load_checkpoint is not original_load_checkpoint
    return training_mod.load_checkpoint


class _Recorder:
    """Stands in for Megatron's load_checkpoint and records each call."""

    def __init__(self, result=(7, 11.0), exc=None):
        self.calls = []
        self.result = result
        self.exc = exc

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.exc is not None:
            raise self.exc
        return self.result


class TestLoadCheckpointDistributedFix:
    def test_forwards_new_keyword_arguments(self, monkeypatch):
        """setup_model_and_optimizer on core_v0.19 passes process groups and rng_state_key_prefix."""
        original = _Recorder()
        patched = _apply_patch(monkeypatch, original)

        model, optimizer, scheduler, ctx = object(), object(), object(), {}
        tp, pp, dp_cp, dp, expt_dp = object(), object(), object(), object(), object()
        result = patched(
            model,
            optimizer,
            scheduler,
            checkpointing_context=ctx,
            skip_load_to_model_and_opt=False,
            tp_group=tp,
            pp_group=pp,
            dp_cp_group=dp_cp,
            dp_group=dp,
            expt_dp_group=expt_dp,
            rng_state_key_prefix="prefix",
        )

        assert result == (7, 11.0)
        assert len(original.calls) == 1
        args, kwargs = original.calls[0]
        assert args == (model, optimizer, scheduler)
        assert kwargs == {
            "checkpointing_context": ctx,
            "skip_load_to_model_and_opt": False,
            "tp_group": tp,
            "pp_group": pp,
            "dp_cp_group": dp_cp,
            "dp_group": dp,
            "expt_dp_group": expt_dp,
            "rng_state_key_prefix": "prefix",
        }

    def test_forwards_old_positional_form(self, monkeypatch):
        original = _Recorder()
        patched = _apply_patch(monkeypatch, original)

        positional = ("model", "optimizer", "scheduler", "load", False, {"k": 1}, True)
        assert patched(*positional) == (7, 11.0)
        assert original.calls == [(positional, {})]

    def test_forwards_keyword_only_minimal_call(self, monkeypatch):
        """Call shape used by post-training: load_checkpoint(model, None, None, load_arg=...)."""
        original = _Recorder()
        patched = _apply_patch(monkeypatch, original)

        patched(["teacher"], None, None, load_arg="export_kd_teacher_load")
        assert original.calls == [((["teacher"], None, None), {"load_arg": "export_kd_teacher_load"})]

    def test_missing_model_key_returns_zero_iteration(self, monkeypatch):
        original = _Recorder(exc=KeyError("model"))
        patched = _apply_patch(monkeypatch, original)

        assert patched("model", None, None, tp_group=object()) == (0, 0)
        assert len(original.calls) == 1

    def test_other_key_errors_propagate(self, monkeypatch):
        original = _Recorder(exc=KeyError("optimizer"))
        patched = _apply_patch(monkeypatch, original)

        with pytest.raises(KeyError):
            patched("model", None, None, tp_group=object())
