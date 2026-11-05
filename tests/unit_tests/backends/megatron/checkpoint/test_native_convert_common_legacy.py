###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""init_megatron_single_process loads the legacy fused kernels only when Megatron has
megatron.legacy (Megatron-core 0.19+ has none: before, the import failed in every native
converter). Megatron's modules are stand-ins, so this runs on CPU without Megatron."""

import importlib.machinery
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from primus.backends.megatron.checkpoint import (  # noqa: E402
    native_convert_common as common,
)


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    mod.__path__ = []
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


@pytest.fixture
def megatron(monkeypatch):
    """Stand-in megatron.core.mpu and megatron.training.global_vars, without megatron.legacy."""
    margs = SimpleNamespace(
        padded_vocab_size=256,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
    mpu = SimpleNamespace(initialize_model_parallel=lambda **k: None)
    global_vars = _module(
        "megatron.training.global_vars",
        set_global_variables=lambda m, build_tokenizer: None,
        get_args=lambda: margs,
    )
    mods = {
        "megatron": _module("megatron"),
        "megatron.core": _module("megatron.core", mpu=mpu),
        "megatron.training": _module("megatron.training", global_vars=global_vars),
        "megatron.training.global_vars": global_vars,
    }
    for name in ("megatron.legacy", "megatron.legacy.fused_kernels"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(common, "neutralize_nvte_attention_env", lambda: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    return margs


def test_without_megatron_legacy_the_kernels_are_skipped(megatron):
    logs = []
    assert common.init_megatron_single_process(megatron, 29999, log=logs.append) is megatron
    assert any("megatron.legacy not present" in line for line in logs)


def test_with_megatron_legacy_the_kernels_load(megatron, monkeypatch):
    calls = []
    fused = _module("megatron.legacy.fused_kernels", load=calls.append)
    monkeypatch.setitem(sys.modules, "megatron.legacy", _module("megatron.legacy", fused_kernels=fused))
    monkeypatch.setitem(sys.modules, "megatron.legacy.fused_kernels", fused)
    logs = []
    common.init_megatron_single_process(megatron, 29999, log=logs.append)
    assert calls == [megatron]
    assert not any("megatron.legacy not present" in line for line in logs)
