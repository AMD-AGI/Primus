###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark's TE GEMM-workspace patch reuses GLM-5.3's patch, gated on Muse Spark."""

from types import SimpleNamespace

import pytest

from primus.backends.megatron.patches import muse_spark_te_gemm_workspace_patches as mod
from primus.core.patches import PatchContext, PatchRegistry


def test_the_patch_calls_the_glm5_patch(monkeypatch):
    import torch

    monkeypatch.setattr(torch.version, "hip", "6.4", raising=False)  # ROCm: the GLM body acts
    monkeypatch.delenv("PRIMUS_TE_GEMM_WORKSPACE_MB", raising=False)  # restored after the test
    calls = []
    monkeypatch.setattr(mod, "patch_glm5_next_te_gemm_workspace", calls.append)
    ctx = object()
    mod.patch_muse_spark_te_gemm_workspace(ctx)
    assert calls == [ctx]


@pytest.mark.parametrize("model_type, expected", [("muse_spark", True), ("glm5_next", False), ("gpt", False)])
def test_the_patch_is_registered_for_muse_spark_only(model_type, expected):
    patch = PatchRegistry.get("megatron.muse_spark.te_gemm_workspace")
    assert patch is not None and patch.phase == "before_train"
    ctx = PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(model_type=model_type))},
    )
    assert bool(patch.condition(ctx)) is expected


def test_the_real_body_enlarges_the_workspace_for_a_muse_spark_context(monkeypatch):
    # GLM-5.3's body (not a spy) on a stubbed TE gemm module, with a Muse Spark context
    import sys
    import types

    import torch

    monkeypatch.setattr(torch.version, "hip", "6.4", raising=False)
    cleared = []
    gemm = types.ModuleType("transformer_engine.pytorch.cpp_extensions.gemm")
    gemm.get_cublas_workspace_size_bytes = lambda: 64 << 20
    gemm.get_cublas_workspace = SimpleNamespace(cache_clear=lambda: cleared.append(True))
    for name in (
        "transformer_engine",
        "transformer_engine.pytorch",
        "transformer_engine.pytorch.cpp_extensions",
    ):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, gemm.__name__, gemm)
    monkeypatch.setattr(sys.modules["transformer_engine.pytorch.cpp_extensions"], "gemm", gemm, raising=False)
    monkeypatch.setenv("PRIMUS_TE_GEMM_WORKSPACE_MB", "256")
    ctx = PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(model_type="muse_spark"))},
    )
    mod.patch_muse_spark_te_gemm_workspace(ctx)
    assert gemm.get_cublas_workspace_size_bytes() == 256 << 20 and cleared == [True]


@pytest.fixture
def rocm(monkeypatch):
    import torch

    monkeypatch.setattr(torch.version, "hip", "6.4", raising=False)


@pytest.mark.parametrize(
    "value",
    [
        "abc",
        "1.5",
        "1024.0",
        "1e3",
        "+5",
        "0",
        "-4",
        "",
        "  ",
        "\u0661\u0662",
        "\u00b2",
        "99999999999",
        "65537",
    ],
)
def test_a_bad_workspace_size_raises_a_clear_error(monkeypatch, rocm, value):
    # blank, malformed, non-positive, non-integer spellings and non-ASCII digits
    calls = []
    monkeypatch.setattr(mod, "patch_glm5_next_te_gemm_workspace", calls.append)
    monkeypatch.setenv("PRIMUS_TE_GEMM_WORKSPACE_MB", value)
    with pytest.raises(ValueError, match="PRIMUS_TE_GEMM_WORKSPACE_MB must be an integer number of MiB"):
        mod.patch_muse_spark_te_gemm_workspace(object())
    assert calls == []


def test_the_workspace_size_value():
    for good in (" 256 ", "1", "000512", str(mod.MAX_WORKSPACE_MB)):
        mod.validate_workspace_mb(good)  # no error


@pytest.mark.parametrize("value", [None, " 256 "])
def test_the_environment_is_not_changed(monkeypatch, rocm, value):
    # GLM-5.3's body reads the user's own value (or its default when unset)
    import os

    seen = []
    monkeypatch.setattr(
        mod, "patch_glm5_next_te_gemm_workspace", lambda ctx: seen.append(os.environ.get(mod.WORKSPACE_ENV))
    )
    if value is None:
        monkeypatch.delenv(mod.WORKSPACE_ENV, raising=False)
    else:
        monkeypatch.setenv(mod.WORKSPACE_ENV, value)
    mod.patch_muse_spark_te_gemm_workspace(object())
    assert seen == [value]


def test_off_rocm_the_value_is_not_checked(monkeypatch):
    # GLM-5.3's body does nothing off ROCm: a blank value does not stop a CUDA run
    import torch

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    calls = []
    monkeypatch.setattr(mod, "patch_glm5_next_te_gemm_workspace", calls.append)
    monkeypatch.setenv(mod.WORKSPACE_ENV, "")
    mod.patch_muse_spark_te_gemm_workspace("ctx")
    assert calls == []  # GLM-5.3's body would do nothing off ROCm: not called, nothing logged
