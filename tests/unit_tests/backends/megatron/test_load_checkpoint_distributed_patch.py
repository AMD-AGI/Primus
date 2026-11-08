###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The load_checkpoint wrapper forwards every argument, including Megatron-core 0.19's groups."""

import sys
import types

import pytest

from primus.backends.megatron.patches.checkpoint_patches import (
    patch_load_checkpoint_for_distributed,
)


def _install(monkeypatch, original):
    checkpointing = types.ModuleType("megatron.training.checkpointing")
    training = types.ModuleType("megatron.training.training")
    checkpointing.load_checkpoint = original
    training.load_checkpoint = original
    pkg = types.ModuleType("megatron.training")
    pkg.__path__ = []
    pkg.checkpointing, pkg.training = checkpointing, training
    top = types.ModuleType("megatron")
    top.__path__ = []
    for name, mod in (
        ("megatron", top),
        ("megatron.training", pkg),
        ("megatron.training.checkpointing", checkpointing),
        ("megatron.training.training", training),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return checkpointing, training


def test_forwards_positional_and_keyword_arguments(monkeypatch):
    calls = []

    def original(ddp_model, optimizer, opt_param_scheduler, load_arg="load", strict=True, **kwargs):
        calls.append((ddp_model, optimizer, opt_param_scheduler, load_arg, strict, kwargs))
        return 7, 11

    checkpointing, training = _install(monkeypatch, original)
    patch_load_checkpoint_for_distributed(None)

    assert training.load_checkpoint is checkpointing.load_checkpoint
    out = training.load_checkpoint("m", "o", "s", strict=False, tp_group="tp", pp_group="pp")
    assert out == (7, 11)
    assert calls == [("m", "o", "s", "load", False, {"tp_group": "tp", "pp_group": "pp"})]


def test_missing_model_key_returns_zero_iteration(monkeypatch):
    def original(*args, **kwargs):
        raise KeyError("model")

    checkpointing, _ = _install(monkeypatch, original)
    patch_load_checkpoint_for_distributed(None)
    assert checkpointing.load_checkpoint("m", "o", "s", tp_group="tp") == (0, 0)


def test_other_key_errors_propagate(monkeypatch):
    def original(*args, **kwargs):
        raise KeyError("optimizer")

    checkpointing, _ = _install(monkeypatch, original)
    patch_load_checkpoint_for_distributed(None)
    with pytest.raises(KeyError):
        checkpointing.load_checkpoint("m", "o", "s")
