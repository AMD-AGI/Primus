###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the opt-in Turbo FP8 GPT output-layer constructor patch."""

import sys
import types
from types import SimpleNamespace

import primus.backends.megatron.patches.turbo.gpt_output_layer_patches as patch_mod


def _condition(monkeypatch, **overrides):
    values = {
        "use_turbo_fp8_output_layer": True,
        "use_turbo_gemm": True,
        "tensor_model_parallel_size": 1,
    }
    values.update(overrides)
    monkeypatch.setattr(patch_mod, "get_args", lambda ctx: SimpleNamespace(**values))
    monkeypatch.setattr(patch_mod, "is_primus_turbo_can_patch", lambda ctx: True)
    return object()


def test_condition_accepts_supported_configuration(monkeypatch):
    assert patch_mod._can_route_fp8_output_layer(_condition(monkeypatch))


def test_condition_requires_explicit_opt_in(monkeypatch):
    ctx = _condition(monkeypatch, use_turbo_fp8_output_layer=False)
    assert not patch_mod._can_route_fp8_output_layer(ctx)


def test_condition_requires_turbo_gemm(monkeypatch):
    ctx = _condition(monkeypatch, use_turbo_gemm=False)
    assert not patch_mod._can_route_fp8_output_layer(ctx)


def test_condition_rejects_tensor_parallelism(monkeypatch):
    ctx = _condition(monkeypatch, tensor_model_parallel_size=2)
    assert not patch_mod._can_route_fp8_output_layer(ctx)


def test_constructor_override_is_scoped_and_idempotent(monkeypatch):
    native_column = type("NativeColumn", (), {})
    turbo_column = type("TurboFP8OutputColumn", (), {})
    observations = []

    gpt_model = types.ModuleType("megatron.core.models.gpt.gpt_model")
    gpt_model.tensor_parallel = SimpleNamespace(ColumnParallelLinear=native_column)

    class GPTModel:
        def __init__(self):
            observations.append(gpt_model.tensor_parallel.ColumnParallelLinear)

    gpt_model.GPTModel = GPTModel
    gpt_pkg = types.ModuleType("megatron.core.models.gpt")
    gpt_pkg.gpt_model = gpt_model
    turbo_ext = types.ModuleType(
        "primus.backends.megatron.core.extensions.primus_turbo"
    )
    turbo_ext.PrimusTurboFP8OutputColumnParallelLinear = turbo_column

    monkeypatch.setitem(sys.modules, "megatron.core.models.gpt", gpt_pkg)
    monkeypatch.setitem(sys.modules, "megatron.core.models.gpt.gpt_model", gpt_model)
    monkeypatch.setitem(
        sys.modules,
        "primus.backends.megatron.core.extensions.primus_turbo",
        turbo_ext,
    )
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *args, **kwargs: None)

    patch_mod.patch_fp8_output_layer(object())
    patched_init = gpt_model.GPTModel.__init__
    patch_mod.patch_fp8_output_layer(object())

    assert gpt_model.GPTModel.__init__ is patched_init
    gpt_model.GPTModel()
    assert observations == [turbo_column]
    assert gpt_model.tensor_parallel.ColumnParallelLinear is native_column
