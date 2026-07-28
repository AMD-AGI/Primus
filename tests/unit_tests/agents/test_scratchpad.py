###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the tuning-agent Scratchpad (plain-text note store)."""

from __future__ import annotations

import pytest

pytest.importorskip("primus.agents.tuning_agent.scratchpad")

from primus.agents.tuning_agent.scratchpad import Scratchpad  # noqa: E402


def test_scratchpad_creates_empty_file_with_parents(tmp_path):
    p = tmp_path / "nested" / "dir" / "sp.txt"
    sp = Scratchpad(p)
    assert p.exists()
    assert sp.read() == ""


def test_scratchpad_append_timestamps_and_accumulates(tmp_path):
    sp = Scratchpad(tmp_path / "sp.txt")
    msg = sp.append("first note")
    assert "scratchpad updated" in msg
    sp.append("second note")
    body = sp.read()
    assert "first note" in body and "second note" in body
    assert body.count("\n") == 2  # one line per append


def test_scratchpad_reset_clears(tmp_path):
    sp = Scratchpad(tmp_path / "sp.txt")
    sp.append("something")
    sp.reset()
    assert sp.read() == ""


def test_scratchpad_read_missing_file_returns_empty(tmp_path):
    sp = Scratchpad(tmp_path / "sp.txt")
    sp.path.unlink()  # remove the file created in __init__
    assert sp.read() == ""
