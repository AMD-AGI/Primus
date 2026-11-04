###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU tests for ``_publish_turbo_env``. The helper itself is not modified."""

import os

from primus.backends.megatron.core.extensions import primus_turbo as primus_turbo_ext


def test_publish_turbo_env_uses_set_env(monkeypatch):
    calls = []

    def record(name, value):
        calls.append((name, value))

    monkeypatch.setattr(primus_turbo_ext, "_turbo_set_env", record)
    monkeypatch.delenv("PRIMUS_TURBO_TEST_SET_ENV", raising=False)
    primus_turbo_ext._publish_turbo_env("PRIMUS_TURBO_TEST_SET_ENV", "flydsl")
    assert calls == [("PRIMUS_TURBO_TEST_SET_ENV", "flydsl")]
    assert "PRIMUS_TURBO_TEST_SET_ENV" not in os.environ


def test_publish_turbo_env_writes_os_environ_when_set_env_is_missing(monkeypatch):
    monkeypatch.setattr(primus_turbo_ext, "_turbo_set_env", None)
    monkeypatch.delenv("PRIMUS_TURBO_TEST_PUBLISH", raising=False)
    primus_turbo_ext._publish_turbo_env("PRIMUS_TURBO_TEST_PUBLISH", "flydsl")
    assert os.environ["PRIMUS_TURBO_TEST_PUBLISH"] == "flydsl"
