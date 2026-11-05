###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
CPU tests for ``native_convert_common.init_megatron_single_process`` with and
without ``megatron.legacy`` (removed in Megatron-LM core v0.19).

Megatron is replaced by stand-in modules in ``sys.modules``, so the tests need
neither the Megatron-LM submodule nor a GPU.
"""

import importlib.abc
import importlib.machinery
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.checkpoint import native_convert_common as common


def _package(name, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    mod.__path__ = []
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


@pytest.fixture
def fake_megatron(monkeypatch):
    """Stand-in ``megatron.core`` and ``megatron.training``; no ``megatron.legacy``."""
    margs = SimpleNamespace(
        padded_vocab_size=256,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
    calls = {}
    mpu = SimpleNamespace(initialize_model_parallel=lambda **kw: calls.setdefault("mpu", kw))
    global_vars = _package(
        "megatron.training.global_vars",
        set_global_variables=lambda args, build_tokenizer: calls.setdefault("globals", build_tokenizer),
        get_args=lambda: margs,
    )
    modules = {
        "megatron": _package("megatron"),
        "megatron.core": _package("megatron.core", mpu=mpu),
        "megatron.training": _package("megatron.training", global_vars=global_vars),
        "megatron.training.global_vars": global_vars,
    }
    for name in ("megatron.legacy", "megatron.legacy.fused_kernels"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(common, "neutralize_nvte_attention_env", lambda: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    return SimpleNamespace(margs=margs, calls=calls)


def test_without_megatron_legacy_skips_fused_kernels(fake_megatron):
    logs = []
    out = common.init_megatron_single_process(fake_megatron.margs, 29999, log=logs.append)
    assert out is fake_megatron.margs
    assert fake_megatron.calls["globals"] is True
    assert fake_megatron.calls["mpu"]["tensor_model_parallel_size"] == 1
    assert any("megatron.legacy not present" in line for line in logs)


def test_with_megatron_legacy_loads_fused_kernels(fake_megatron, monkeypatch):
    loaded = []
    fused = _package("megatron.legacy.fused_kernels", load=loaded.append)
    monkeypatch.setitem(sys.modules, "megatron.legacy", _package("megatron.legacy", fused_kernels=fused))
    monkeypatch.setitem(sys.modules, "megatron.legacy.fused_kernels", fused)
    logs = []
    common.init_megatron_single_process(fake_megatron.margs, 29999, log=logs.append)
    assert loaded == [fake_megatron.margs]
    assert not any("megatron.legacy not present" in line for line in logs)


def test_missing_fused_kernels_inside_megatron_legacy_raises(fake_megatron, monkeypatch):
    monkeypatch.setitem(sys.modules, "megatron.legacy", _package("megatron.legacy"))
    with pytest.raises(ImportError, match="fused_kernels"):
        common.init_megatron_single_process(fake_megatron.margs, 29999, log=lambda *_: None)


class _LegacyWithMissingDependency(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Finds ``megatron.legacy``, whose import then fails on a missing dependency."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "megatron.legacy":
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ModuleNotFoundError("No module named 'some_missing_dependency'", name="some_missing_dependency")


def test_unrelated_missing_module_raises(fake_megatron, monkeypatch):
    monkeypatch.setattr(sys, "meta_path", [_LegacyWithMissingDependency(), *sys.meta_path])
    with pytest.raises(ModuleNotFoundError) as excinfo:
        common.init_megatron_single_process(fake_megatron.margs, 29999, log=lambda *_: None)
    assert excinfo.value.name == "some_missing_dependency"
