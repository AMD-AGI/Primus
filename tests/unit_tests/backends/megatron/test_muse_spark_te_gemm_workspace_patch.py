###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The shared TE GEMM-workspace helper and the two patches that call it (Muse Spark, GLM-5.3)."""

import sys
import types
from types import SimpleNamespace

import pytest

from primus.backends.megatron.core.extensions import te_gemm_workspace as ws
from primus.backends.megatron.patches import glm5_next_patches as glm
from primus.backends.megatron.patches import muse_spark_te_gemm_workspace_patches as mod
from primus.core.patches import PatchContext, PatchRegistry


@pytest.fixture
def te_gemm(monkeypatch):
    """ROCm with a stubbed TE gemm module (64 MiB default workspace)."""
    import torch

    monkeypatch.setattr(torch.version, "hip", "6.4", raising=False)
    monkeypatch.delenv(ws.WORKSPACE_ENV, raising=False)
    gemm = types.ModuleType("transformer_engine.pytorch.cpp_extensions.gemm")
    gemm.get_cublas_workspace_size_bytes = lambda: 64 << 20
    gemm.cleared = []
    gemm.get_cublas_workspace = SimpleNamespace(cache_clear=lambda: gemm.cleared.append(True))
    for name in (
        "transformer_engine",
        "transformer_engine.pytorch",
        "transformer_engine.pytorch.cpp_extensions",
    ):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, gemm.__name__, gemm)
    monkeypatch.setattr(sys.modules["transformer_engine.pytorch.cpp_extensions"], "gemm", gemm, raising=False)
    return gemm


def _ctx(model_type):
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(model_type=model_type))},
    )


@pytest.mark.parametrize("model_type, expected", [("muse_spark", True), ("glm5_next", False), ("gpt", False)])
def test_the_patch_is_registered_for_muse_spark_only(model_type, expected):
    patch = PatchRegistry.get("megatron.muse_spark.te_gemm_workspace")
    assert patch is not None and patch.phase == "before_train"
    assert bool(patch.condition(_ctx(model_type))) is expected


@pytest.mark.parametrize(
    "patch_fn, tag",
    [
        (mod.patch_muse_spark_te_gemm_workspace, "megatron.muse_spark.te_gemm_workspace"),
        (glm.patch_glm5_next_te_gemm_workspace, "megatron.glm5_next.te_gemm_workspace"),
    ],
)
@pytest.mark.parametrize("value, mb", [(None, 512), ("256", 256), (" 1024 ", 1024)])
def test_both_patches_enlarge_the_workspace_by_one_rule(monkeypatch, te_gemm, patch_fn, tag, value, mb):
    logs = []
    monkeypatch.setattr(ws, "log_rank_0", logs.append)
    if value is not None:
        monkeypatch.setenv(ws.WORKSPACE_ENV, value)
    patch_fn(_ctx("x"))
    assert te_gemm.get_cublas_workspace_size_bytes() == mb << 20 and te_gemm.cleared == [True]
    assert len(logs) == 1 and logs[0].startswith(f"[Patch:{tag}] TE GEMM workspace -> {mb} MiB (")


def test_a_larger_te_workspace_is_kept(monkeypatch, te_gemm):
    te_gemm.get_cublas_workspace_size_bytes = lambda: 1 << 30
    monkeypatch.setenv(ws.WORKSPACE_ENV, "256")
    mod.patch_muse_spark_te_gemm_workspace(None)
    assert te_gemm.get_cublas_workspace_size_bytes() == 1 << 30 and te_gemm.cleared == []


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
        "4097",  # above MAX_WORKSPACE_MB (4 GiB per workspace)
        "9" * 5000,  # over Python's 4300-digit int() limit: the digit-count guard names the variable
    ],
)
def test_a_bad_workspace_size_raises_a_clear_error_on_muse_spark(monkeypatch, te_gemm, value):
    monkeypatch.setenv(ws.WORKSPACE_ENV, value)
    with pytest.raises(ValueError, match="PRIMUS_TE_GEMM_WORKSPACE_MB must be an integer number of MiB"):
        mod.patch_muse_spark_te_gemm_workspace.__wrapped__(None)
    assert te_gemm.cleared == []


@pytest.mark.parametrize("value", ["0", "-1", "+5", "32"])
def test_glm5_keeps_its_int_reading(monkeypatch, te_gemm, value):
    # GLM-5.3's contract is unchanged (strict=False): int() accepts these, and a size below
    # TE's 64 MiB changes nothing
    monkeypatch.setenv(ws.WORKSPACE_ENV, value)
    glm.patch_glm5_next_te_gemm_workspace(None)
    assert te_gemm.get_cublas_workspace_size_bytes() == 64 << 20 and te_gemm.cleared == []


@pytest.mark.parametrize("value", ["abc", "1.5", ""])
def test_glm5_malformed_value_fails_naming_the_variable(monkeypatch, te_gemm, value):
    # still a ValueError, as int() raised before, now naming the variable
    monkeypatch.setenv(ws.WORKSPACE_ENV, value)
    with pytest.raises(ValueError, match="PRIMUS_TE_GEMM_WORKSPACE_MB must be an integer number of MiB, got"):
        glm.patch_glm5_next_te_gemm_workspace(None)


def test_the_workspace_size_value():
    assert ws.parse_workspace_mb(None) == ws.DEFAULT_WORKSPACE_MB
    for good, mb in (
        (" 256 ", 256),
        ("1", 1),
        ("000512", 512),
        (str(ws.MAX_WORKSPACE_MB), ws.MAX_WORKSPACE_MB),
    ):
        assert ws.parse_workspace_mb(good) == mb


def test_off_rocm_nothing_is_done(monkeypatch):
    import torch

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setenv(ws.WORKSPACE_ENV, "")  # not even checked: a CUDA run is not stopped
    logs = []
    monkeypatch.setattr(ws, "log_rank_0", logs.append)
    mod.patch_muse_spark_te_gemm_workspace(None)
    glm.patch_glm5_next_te_gemm_workspace(None)
    assert logs == []


def test_real_te_reads_the_size_through_the_module(monkeypatch):
    """With the real TE (when installed): the helper finds the module that owns the size
    getter (``cpp_extensions.gemm`` in TE >= 2.1x, ``module.base`` in older TE). In the gemm
    layout the plain, grouped-GEMM (multi-stream) and UB workspaces all read the getter
    through the module's globals, so the rebinding reaches the MoE grouped expert GEMMs; and
    no other loaded TE module holds a by-name copy of the getter (it would keep the old size)."""
    pytest.importorskip("transformer_engine.pytorch.cpp_extensions.gemm")
    import transformer_engine.pytorch  # noqa: F401  (loads TE's linear / grouped-linear modules)

    owner, _ = ws._te_workspace_owner()
    if owner.__name__.endswith("cpp_extensions.gemm"):
        wrapped = owner.get_cublas_workspace.__wrapped__
        assert "get_cublas_workspace_size_bytes" in wrapped.__code__.co_names
        assert wrapped.__globals__ is vars(owner)
    holders = [
        name
        for name, mod_ in list(sys.modules.items())
        if name.startswith("transformer_engine")
        and mod_ is not owner
        and getattr(mod_, "get_cublas_workspace_size_bytes", None) is owner.get_cublas_workspace_size_bytes
    ]
    assert holders == [], f"TE modules hold a by-name copy of the size function: {holders}"


def _stub_module_base(monkeypatch):
    """An empty ``transformer_engine.pytorch.module.base`` (reached by import and attribute)."""
    package = types.ModuleType("transformer_engine.pytorch.module")
    base = types.ModuleType("transformer_engine.pytorch.module.base")
    package.base = base
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setattr(sys.modules["transformer_engine.pytorch"], "module", package, raising=False)
    return base


def test_older_te_layout_module_base(monkeypatch, te_gemm):
    """TE 2.8-style layout: the getter and the cached workspaces live in module.base."""
    del te_gemm.get_cublas_workspace_size_bytes
    base = _stub_module_base(monkeypatch)
    base.get_cublas_workspace_size_bytes = lambda: 64 << 20
    base._cublas_workspace = object()
    base._multi_stream_cublas_workspace = [object()]
    mod.patch_muse_spark_te_gemm_workspace(None)
    assert base.get_cublas_workspace_size_bytes() == 512 << 20
    assert base._cublas_workspace is None and base._multi_stream_cublas_workspace == []


def test_no_known_te_layout_fails_loudly(monkeypatch, te_gemm):
    del te_gemm.get_cublas_workspace_size_bytes
    _stub_module_base(monkeypatch)
    with pytest.raises(RuntimeError, match="cannot be enlarged"):
        ws.enlarge_te_gemm_workspace("probe")
    with pytest.raises(SystemExit, match="cannot be enlarged"):  # the Muse patch stops the run
        mod.patch_muse_spark_te_gemm_workspace(None)


def test_after_train_restores_tes_own_size(monkeypatch, te_gemm):
    # a later model in the same process gets TE's default workspace back
    monkeypatch.setattr(ws, "log_rank_0", lambda *_a, **_k: None)
    te_default = te_gemm.get_cublas_workspace_size_bytes
    restore = PatchRegistry.get(mod.RESTORE_PATCH_NAME)
    assert restore is not None and restore.phase == "after_train"
    assert not restore.condition(None)  # nothing enlarged yet: the restore does not run
    mod.patch_muse_spark_te_gemm_workspace(None)
    assert te_gemm.get_cublas_workspace_size_bytes() == 512 << 20
    assert ws.enlarged_by() == mod.PATCH_NAME and restore.condition(None)
    mod.patch_muse_spark_te_gemm_workspace_restore(None)
    assert te_gemm.get_cublas_workspace_size_bytes is te_default
    assert te_gemm.cleared == [True, True]  # the cached workspaces are dropped both times
    assert ws.enlarged_by() is None and not restore.condition(None)
    assert ws.restore_te_gemm_workspace("again") is False  # a second restore is a no-op


def test_glm5_enlargement_is_not_restored_by_the_muse_patch(monkeypatch, te_gemm):
    monkeypatch.setattr(ws, "log_rank_0", lambda *_a, **_k: None)
    glm.patch_glm5_next_te_gemm_workspace(None)
    assert ws.enlarged_by() == "megatron.glm5_next.te_gemm_workspace"
    assert not PatchRegistry.get(mod.RESTORE_PATCH_NAME).condition(None)


def test_a_second_enlarge_keeps_tes_own_getter_for_the_restore(monkeypatch, te_gemm):
    monkeypatch.setattr(ws, "log_rank_0", lambda *_a, **_k: None)
    te_default = te_gemm.get_cublas_workspace_size_bytes
    monkeypatch.setenv(ws.WORKSPACE_ENV, "256")
    mod.patch_muse_spark_te_gemm_workspace(None)
    monkeypatch.setenv(ws.WORKSPACE_ENV, "1024")
    mod.patch_muse_spark_te_gemm_workspace(None)
    assert te_gemm.get_cublas_workspace_size_bytes() == 1024 << 20
    assert ws.restore_te_gemm_workspace("t") is True
    assert te_gemm.get_cublas_workspace_size_bytes is te_default
