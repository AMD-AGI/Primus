# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
A malformed PRIMUS_TE_GEMM_WORKSPACE_MIB has to be reported, not dropped.

The patch leaves TE's workspace alone for an invalid value, and a run that
needed the larger workspace then fails later inside hipBLASLt, far from the
setting that caused it.
"""

import functools
import sys
import types

import pytest

from primus.backends.megatron.patches.te_patches import (
    hipblaslt_workspace_patches as patches,
)


@pytest.fixture
def warnings(monkeypatch):
    seen = []
    monkeypatch.setattr(patches, "warning_rank_0", lambda msg, *a, **k: seen.append(msg))
    return seen


def test_unset_is_silent(monkeypatch, warnings):
    monkeypatch.delenv(patches.ENV_VAR, raising=False)

    assert patches._requested_workspace_mib() is None
    assert warnings == []


def test_a_whole_number_of_mib_is_used(monkeypatch, warnings):
    monkeypatch.setenv(patches.ENV_VAR, " 128 ")

    assert patches._requested_workspace_mib() == 128
    assert warnings == []


@pytest.mark.parametrize("raw", ["128MiB", "1.5", "0", "-64"])
def test_a_malformed_value_warns_and_is_ignored(monkeypatch, warnings, raw):
    monkeypatch.setenv(patches.ENV_VAR, raw)

    assert patches._requested_workspace_mib() is None
    assert len(warnings) == 1
    assert repr(raw) in warnings[0]


MIB = 1024 * 1024


@pytest.fixture
def te_gemm(monkeypatch):
    """A stand-in for TE's gemm module: the size getter and the memoized workspace."""
    module = types.ModuleType("transformer_engine.pytorch.cpp_extensions.gemm")
    module.get_cublas_workspace_size_bytes = lambda: 64 * MIB

    @functools.lru_cache(maxsize=None)
    def get_cublas_workspace():
        return module.get_cublas_workspace_size_bytes()

    module.get_cublas_workspace = get_cublas_workspace
    package = types.ModuleType("transformer_engine.pytorch.cpp_extensions")
    package.gemm = module
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.cpp_extensions", package)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.cpp_extensions.gemm", module)
    monkeypatch.setattr(patches, "log_rank_0", lambda *a, **k: None)
    return module


def test_the_requested_size_reaches_every_gemm(monkeypatch, te_gemm):
    monkeypatch.setenv(patches.ENV_VAR, "128")
    assert te_gemm.get_cublas_workspace() == 64 * MIB

    assert patches._needs_te_gemm_workspace_patch(None)
    patches.patch_te_gemm_workspace_size(None)

    assert te_gemm.get_cublas_workspace_size_bytes() == 128 * MIB
    assert te_gemm.get_cublas_workspace() == 128 * MIB, "the memoized 64 MiB workspace was kept"


def test_a_larger_workspace_is_not_shrunk(monkeypatch, te_gemm):
    """GLM-5's own patch enlarges it to 512 MiB through a different variable."""
    te_gemm.get_cublas_workspace_size_bytes = lambda: 512 * MIB
    monkeypatch.setenv(patches.ENV_VAR, "128")

    patches.patch_te_gemm_workspace_size(None)

    assert te_gemm.get_cublas_workspace_size_bytes() == 512 * MIB


def test_unset_leaves_te_alone(monkeypatch, te_gemm):
    monkeypatch.delenv(patches.ENV_VAR, raising=False)

    assert not patches._needs_te_gemm_workspace_patch(None)


def _fail_te_gemm_import(monkeypatch, exc):
    """Make ``from transformer_engine.pytorch.cpp_extensions import gemm`` raise ``exc``."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformer_engine.pytorch.cpp_extensions" and fromlist and "gemm" in fromlist:
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


@pytest.mark.parametrize("name", patches._TE_ABSENT_MODULES)
def test_no_te_gemm_module_warns_and_skips(monkeypatch, warnings, name):
    """No TE GEMM module to resize (TE not installed, a JAX- or core-only build, no
    cpp_extensions.gemm): an explicit request is skipped with a warning, never silently
    (owner decision dec-fd41a477), and the patch is not selected."""
    monkeypatch.setenv(patches.ENV_VAR, "128")
    _fail_te_gemm_import(monkeypatch, ModuleNotFoundError(f"No module named '{name}'", name=name))

    assert not patches._needs_te_gemm_workspace_patch(None)
    assert len(warnings) == 1 and "no Transformer Engine PyTorch GEMM module" in warnings[0]


def test_te_without_the_size_getter_warns(monkeypatch, te_gemm, warnings):
    del te_gemm.get_cublas_workspace_size_bytes
    monkeypatch.setenv(patches.ENV_VAR, "128")

    assert not patches._needs_te_gemm_workspace_patch(None)
    assert len(warnings) == 1 and "get_cublas_workspace_size_bytes" in warnings[0]


@pytest.mark.parametrize(
    "exc,match",
    [
        # TE's Python package is there but its compiled extension is not (a broken install)
        (
            ModuleNotFoundError(
                "No module named 'transformer_engine_torch'", name="transformer_engine_torch"
            ),
            "transformer_engine_torch",
        ),
        (ModuleNotFoundError("No module named 'some_dep'", name="some_dep"), "some_dep"),
        (ImportError("libtransformer_engine.so: undefined symbol: foo"), "undefined symbol"),
        # import failures of other types (a .so load, a version / arch check)
        (OSError("libcudart.so: cannot open shared object file"), "cannot open shared object"),
        (RuntimeError("TE requires a newer PyTorch"), "newer PyTorch"),
        (AssertionError("unsupported arch"), "unsupported arch"),
    ],
)
def test_a_broken_te_install_is_logged_and_skipped(monkeypatch, warnings, exc, match):
    """Any other import failure (a broken install): the condition does not raise (it runs outside
    the runner's error handling), skips the patch as before, and logs the failure with its
    traceback instead of skipping silently."""
    monkeypatch.setenv(patches.ENV_VAR, "128")
    _fail_te_gemm_import(monkeypatch, exc)

    assert patches._needs_te_gemm_workspace_patch(None) is False
    assert len(warnings) == 1 and "fails to import" in warnings[0]
    assert match in warnings[0] and "Traceback" in warnings[0]


def test_a_broken_te_install_does_not_stop_the_patch_phase(monkeypatch, warnings):
    """Through run_patches: the patch is skipped with a warning and the phase goes on."""
    from collections import defaultdict

    from primus.core.patches import patch_runner
    from primus.core.patches.patch_registry import PatchRegistry, register_patch

    # an isolated registry holding only this patch (other tests clear the global one)
    monkeypatch.setattr(PatchRegistry, "_patches_by_backend_phase", defaultdict(lambda: defaultdict(list)))
    monkeypatch.setattr(PatchRegistry, "_all_patches", [])
    register_patch(
        "megatron.te.hipblaslt_gemm_workspace",
        backend="megatron",
        phase="before_train",
        condition=patches._needs_te_gemm_workspace_patch,
    )(patches.patch_te_gemm_workspace_size)
    monkeypatch.setenv(patches.ENV_VAR, "128")
    _fail_te_gemm_import(monkeypatch, ImportError("libtransformer_engine.so: undefined symbol: foo"))
    monkeypatch.setattr(patch_runner, "log_rank_0", lambda *a, **k: None)

    applied = patch_runner.run_patches(
        backend="megatron",
        phase="before_train",
        enabled_ids=["megatron.te.hipblaslt_gemm_workspace"],
    )

    assert applied == 0
    assert any("undefined symbol" in w for w in warnings)
