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
