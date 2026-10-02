# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
A malformed PRIMUS_TE_GEMM_WORKSPACE_MIB has to be reported, not dropped.

The patch leaves TE's workspace alone for an invalid value, and a run that
needed the larger workspace then fails later inside hipBLASLt, far from the
setting that caused it.
"""

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
