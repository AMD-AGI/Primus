###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron-core 0.19+ compatibility (no ``megatron.legacy``):
``native_convert_common.init_megatron_single_process`` loads the legacy fused kernels
only when ``megatron.legacy`` exists (all native converters).
"""

import sys
import types

import pytest


def _install_fake_megatron(monkeypatch, with_legacy, calls, broken_legacy=False):
    """Fake ``megatron.core.mpu`` / ``megatron.training.global_vars`` (and optionally
    ``megatron.legacy.fused_kernels``) so ``init_megatron_single_process`` runs without
    Megatron or a process group."""
    margs = types.SimpleNamespace(
        padded_vocab_size=128,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
    mpu = types.SimpleNamespace(initialize_model_parallel=lambda **kw: calls.append("mpu"))
    core = types.ModuleType("megatron.core")
    core.mpu = mpu
    gv = types.ModuleType("megatron.training.global_vars")
    gv.get_args = lambda: margs
    gv.set_global_variables = lambda m, build_tokenizer=True: calls.append("globals")
    modules = {
        "megatron": types.ModuleType("megatron"),
        "megatron.core": core,
        "megatron.training": types.ModuleType("megatron.training"),
        "megatron.training.global_vars": gv,
    }
    if with_legacy:
        fk = types.ModuleType("megatron.legacy.fused_kernels")
        fk.load = lambda m: calls.append("fused_kernels")
        legacy = types.ModuleType("megatron.legacy")
        legacy.fused_kernels = fk
        modules["megatron.legacy"] = legacy
        modules["megatron.legacy.fused_kernels"] = fk
    elif broken_legacy:
        # megatron.legacy exists, but one of its own imports is missing.
        class _Broken(types.ModuleType):
            def __getattr__(self, name):
                raise ModuleNotFoundError("No module named 'apex'", name="apex")

        modules["megatron.legacy"] = _Broken("megatron.legacy")
    else:
        modules["megatron.legacy"] = None  # import raises ModuleNotFoundError(name="megatron.legacy")
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return margs


@pytest.mark.parametrize("with_legacy", [True, False])
def test_init_single_process_with_and_without_megatron_legacy(monkeypatch, with_legacy):
    import torch

    from primus.backends.megatron.checkpoint import native_convert_common as common

    calls = []
    margs = _install_fake_megatron(monkeypatch, with_legacy, calls)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    # The function sets these; monkeypatch restores them after the test.
    for name in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setattr(common, "neutralize_nvte_attention_env", lambda: None)
    logged = []
    out = common.init_megatron_single_process(margs, 29999, log=logged.append)
    assert out is margs
    assert ("fused_kernels" in calls) == with_legacy
    assert calls[:2] == ["globals", "mpu"]
    if not with_legacy:
        assert any("megatron.legacy not present" in m for m in logged)


def test_init_single_process_raises_on_a_broken_megatron_legacy(monkeypatch):
    # An existing megatron.legacy whose own import fails is a broken install, not "absent".
    import torch

    from primus.backends.megatron.checkpoint import native_convert_common as common

    calls = []
    margs = _install_fake_megatron(monkeypatch, False, calls, broken_legacy=True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    for name in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setattr(common, "neutralize_nvte_attention_env", lambda: None)
    with pytest.raises(ImportError):
        common.init_megatron_single_process(margs, 29999, log=lambda *a: None)
