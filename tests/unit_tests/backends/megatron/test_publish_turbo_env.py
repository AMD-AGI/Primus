###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU tests that the turbo backend helpers publish through ``_set_turbo_env``."""

from types import SimpleNamespace

import pytest

from primus.backends.megatron.core.extensions import primus_turbo as primus_turbo_ext

CASES = [
    (
        "_apply_turbo_gemm_backend_env",
        "turbo_gemm_backend",
        "PRIMUS_TURBO_GEMM_BACKEND",
    ),
    (
        "_apply_turbo_grouped_gemm_backend_env",
        "turbo_grouped_gemm_backend",
        "PRIMUS_TURBO_GROUPED_GEMM_BACKEND",
    ),
]


@pytest.mark.parametrize("helper,arg,variable", CASES, ids=["gemm", "grouped_gemm"])
def test_backend_env_is_published_via_set_turbo_env(monkeypatch, helper, arg, variable):
    calls = []
    monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(primus_turbo_ext, "_set_turbo_env", lambda name, value: calls.append((name, value)))
    monkeypatch.setattr(primus_turbo_ext, "get_args", lambda: SimpleNamespace(**{arg: "flydsl"}))
    monkeypatch.setattr(primus_turbo_ext, "warning_rank_0", lambda *args, **kwargs: None)

    fn = getattr(primus_turbo_ext, helper)
    fn.cache_clear()
    try:
        fn()
    finally:
        fn.cache_clear()

    assert calls == [(variable, "flydsl")]


@pytest.mark.parametrize("helper,arg,variable", CASES, ids=["gemm", "grouped_gemm"])
def test_default_backend_publishes_nothing(monkeypatch, helper, arg, variable):
    calls = []
    monkeypatch.setattr(primus_turbo_ext, "_set_turbo_env", lambda name, value: calls.append((name, value)))
    monkeypatch.setattr(primus_turbo_ext, "get_args", lambda: SimpleNamespace(**{arg: "default"}))

    fn = getattr(primus_turbo_ext, helper)
    fn.cache_clear()
    try:
        fn()
    finally:
        fn.cache_clear()

    assert calls == []
