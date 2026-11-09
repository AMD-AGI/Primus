###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import sys
from types import ModuleType

import pytest

from primus.backends.megatron.patches import checkpoint_patches


@pytest.fixture
def megatron_training(monkeypatch):
    calls = []

    def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, load_arg="load", **kwargs):
        calls.append((ddp_model, load_arg, kwargs))
        if kwargs.get("raise_model_key"):
            raise KeyError("model")
        return 7, 11

    checkpointing = ModuleType("megatron.training.checkpointing")
    checkpointing.load_checkpoint = load_checkpoint
    training = ModuleType("megatron.training.training")
    package = ModuleType("megatron.training")
    package.checkpointing, package.training = checkpointing, training
    monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.training", package)
    monkeypatch.setitem(sys.modules, "megatron.training.checkpointing", checkpointing)
    monkeypatch.setitem(sys.modules, "megatron.training.training", training)
    checkpoint_patches.patch_load_checkpoint_for_distributed(None)
    return checkpointing, training, calls


def test_forwards_megatron_0_19_process_group_keywords(megatron_training):
    checkpointing, training, calls = megatron_training
    assert training.load_checkpoint is checkpointing.load_checkpoint
    result = checkpointing.load_checkpoint(
        "model", None, None, checkpointing_context={}, tp_group="tp", dp_group="dp"
    )
    assert result == (7, 11)
    assert calls == [("model", "load", {"checkpointing_context": {}, "tp_group": "tp", "dp_group": "dp"})]


def test_missing_model_key_is_a_fresh_start(megatron_training):
    checkpointing, _training, _calls = megatron_training
    assert checkpointing.load_checkpoint("model", None, None, raise_model_key=True) == (0, 0)
