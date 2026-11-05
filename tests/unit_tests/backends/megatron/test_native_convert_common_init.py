###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""CPU tests: native_convert_common.init_megatron_single_process with and without
megatron.legacy (absent in Primus's Megatron-LM pin, core_v0.19.2)."""

import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from primus.backends.megatron.checkpoint import (  # noqa: E402
    native_convert_common as common,
)


@pytest.fixture
def fake_megatron(monkeypatch):
    calls = {}
    margs = SimpleNamespace(
        padded_vocab_size=32768,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
    mpu = types.ModuleType("megatron.core.mpu")
    mpu.initialize_model_parallel = lambda **kw: calls.setdefault("mpu", kw)
    core = types.ModuleType("megatron.core")
    core.mpu = mpu
    global_vars = types.ModuleType("megatron.training.global_vars")
    global_vars.set_global_variables = lambda a, build_tokenizer: calls.setdefault("globals", build_tokenizer)
    global_vars.get_args = lambda: margs
    training = types.ModuleType("megatron.training")
    training.global_vars = global_vars
    megatron = types.ModuleType("megatron")
    megatron.core, megatron.training = core, training
    for name, mod in {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.mpu": mpu,
        "megatron.training": training,
        "megatron.training.global_vars": global_vars,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)
    return SimpleNamespace(calls=calls, margs=margs, megatron=megatron)


def test_without_megatron_legacy(fake_megatron, monkeypatch):
    """Megatron-core 0.19+: no megatron.legacy -> no fused-kernel load, init completes."""
    monkeypatch.setitem(sys.modules, "megatron.legacy", None)  # import raises ModuleNotFoundError
    logs = []
    out = common.init_megatron_single_process(fake_megatron.margs, 29999, log=logs.append)
    assert out is fake_megatron.margs
    assert (
        fake_megatron.calls["globals"] is True
        and fake_megatron.calls["mpu"]["tensor_model_parallel_size"] == 1
    )
    assert any("megatron.legacy is not present" in x for x in logs)


def test_missing_fused_kernels_in_existing_legacy_is_not_swallowed(fake_megatron, monkeypatch):
    """A megatron.legacy that exists but lacks fused_kernels is an error, not a skip."""
    legacy = types.ModuleType("megatron.legacy")
    legacy.__path__ = []  # a package without fused_kernels
    monkeypatch.setitem(sys.modules, "megatron.legacy", legacy)
    monkeypatch.setitem(sys.modules, "megatron.legacy.fused_kernels", None)
    with pytest.raises(ModuleNotFoundError) as e:
        common.init_megatron_single_process(fake_megatron.margs, 29999, log=lambda *a: None)
    assert e.value.name == "megatron.legacy.fused_kernels"


def test_with_megatron_legacy_loads_fused_kernels(fake_megatron, monkeypatch):
    """Older Megatron: megatron.legacy.fused_kernels.load(margs) still runs."""
    loaded = []
    fused = types.ModuleType("megatron.legacy.fused_kernels")
    fused.load = loaded.append
    legacy = types.ModuleType("megatron.legacy")
    legacy.fused_kernels = fused
    monkeypatch.setitem(sys.modules, "megatron.legacy", legacy)
    monkeypatch.setitem(sys.modules, "megatron.legacy.fused_kernels", fused)
    common.init_megatron_single_process(fake_megatron.margs, 29999, log=lambda *a: None)
    assert loaded == [fake_megatron.margs]


def test_other_missing_module_is_not_swallowed(fake_megatron, monkeypatch):
    """A module that megatron.legacy itself fails to import is raised, not skipped."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "megatron.legacy":
            raise ModuleNotFoundError("No module named 'apex'", name="apex")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ModuleNotFoundError, match="apex"):
        common.init_megatron_single_process(fake_megatron.margs, 29999, log=lambda *a: None)
