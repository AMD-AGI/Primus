###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the train subcommand's runtime router (``_resolve_pretrain_runtime``).

This selects the active core runtime vs the legacy pretrain flow; run()/register
import heavy runtime modules and are out of scope.
"""

from __future__ import annotations

import pytest

pytest.importorskip("primus.cli.subcommands.train")

from primus.cli.subcommands.train import _resolve_pretrain_runtime  # noqa: E402


def test_runtime_defaults_to_core(monkeypatch):
    monkeypatch.delenv("PRIMUS_TRAIN_RUNTIME", raising=False)
    assert _resolve_pretrain_runtime(None) == "core"


def test_runtime_explicit_legacy(monkeypatch):
    monkeypatch.setenv("PRIMUS_TRAIN_RUNTIME", "legacy")
    assert _resolve_pretrain_runtime(None) == "legacy"


def test_runtime_explicit_core_case_insensitive(monkeypatch):
    monkeypatch.setenv("PRIMUS_TRAIN_RUNTIME", "CORE")
    assert _resolve_pretrain_runtime(None) == "core"


def test_runtime_invalid_value_warns_and_falls_back_to_core(monkeypatch, capsys):
    monkeypatch.setenv("PRIMUS_TRAIN_RUNTIME", "bogus")
    assert _resolve_pretrain_runtime(None) == "core"
    assert "Ignoring invalid" in capsys.readouterr().err
