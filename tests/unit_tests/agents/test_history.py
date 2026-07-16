###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Unit tests for the tuning-agent trial history store (``history.py``).

``History`` is the JSONL-backed record of evaluated trials.  These tests
cover the dedup signature, append + reload round-trip, ``already_evaluated``,
``best`` objective selection (legal-only), and the LLM summary rendering.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("primus.agents.tuning_agent.history")

from primus.agents.tuning_agent.history import History, _sig  # noqa: E402


def _result(legal=True, tps=None, mem=None, reason="", source="simulate"):
    """Build a stand-in EvalResult exposing ``.as_dict()`` + ``.source``.

    ``History.add`` only relies on these two attributes, so a SimpleNamespace
    keeps the test independent of the (heavier) real EvalResult dataclass.
    """
    d = {
        "legal": legal,
        "tokens_per_s_per_gpu": tps,
        "memory_per_gpu_gb": mem,
        "reason": reason,
        "source": source,
    }
    return SimpleNamespace(as_dict=lambda: dict(d), source=source)


# ─────────────────────────────────────────────────────────────────────────────
# _sig
# ─────────────────────────────────────────────────────────────────────────────


def test_sig_is_order_independent():
    a = {"tp": 1, "pp": 2, "ep": 4}
    b = {"ep": 4, "pp": 2, "tp": 1}
    assert _sig(a) == _sig(b)


def test_sig_distinguishes_different_configs():
    assert _sig({"tp": 1}) != _sig({"tp": 2})


# ─────────────────────────────────────────────────────────────────────────────
# add + reload round-trip
# ─────────────────────────────────────────────────────────────────────────────


def test_add_persists_and_reloads(tmp_path):
    path = tmp_path / "history.jsonl"
    h = History(path=path)
    cfg = {"tp": 1, "pp": 2, "ep": 4}
    h.add(cfg, _result(legal=True, tps=12000.0, mem=80.0))

    assert path.is_file()
    assert h.already_evaluated(cfg) is True

    # A fresh load from disk reconstructs the same trial + dedup signature.
    h2 = History.load(path)
    assert len(h2.trials) == 1
    assert h2.trials[0].config == cfg
    assert h2.already_evaluated(cfg) is True
    assert h2.already_evaluated({"tp": 9, "pp": 9, "ep": 9}) is False


def test_load_missing_file_is_empty(tmp_path):
    h = History.load(tmp_path / "nope.jsonl")
    assert h.trials == []
    assert h.seen_signatures == set()


def test_load_skips_corrupt_lines(tmp_path):
    path = tmp_path / "h.jsonl"
    good = (
        '{"idx": 0, "timestamp": 1.0, "config": {"tp": 1}, '
        '"result": {"legal": true}, "source": "simulate", "notes": ""}'
    )
    path.write_text(good + "\n" + "{not json}\n" + "\n")
    h = History.load(path)
    assert len(h.trials) == 1
    assert h.trials[0].config == {"tp": 1}


# ─────────────────────────────────────────────────────────────────────────────
# best
# ─────────────────────────────────────────────────────────────────────────────


def test_best_picks_max_legal_objective(tmp_path):
    h = History(path=tmp_path / "h.jsonl")
    h.add({"tp": 1}, _result(legal=True, tps=10000.0))
    h.add({"tp": 2}, _result(legal=True, tps=15000.0))
    h.add({"tp": 4}, _result(legal=True, tps=12000.0))
    # An illegal trial with a higher tps must be ignored.
    h.add({"tp": 8}, _result(legal=False, tps=99999.0))

    best = h.best("tokens_per_s_per_gpu")
    assert best is not None
    assert best.config == {"tp": 2}
    assert best.result["tokens_per_s_per_gpu"] == 15000.0


def test_best_returns_none_when_no_legal(tmp_path):
    h = History(path=tmp_path / "h.jsonl")
    h.add({"tp": 1}, _result(legal=False, tps=10000.0))
    assert h.best() is None


# ─────────────────────────────────────────────────────────────────────────────
# summary_for_llm
# ─────────────────────────────────────────────────────────────────────────────


def test_summary_for_llm_empty():
    h = History(path=__import__("pathlib").Path("/tmp/does-not-matter.jsonl"))
    assert h.summary_for_llm() == "(no trials yet)"


def test_summary_for_llm_renders_rows_and_tail(tmp_path):
    h = History(path=tmp_path / "h.jsonl")
    h.add({"tp": 1, "pp": 1}, _result(legal=True, tps=10000.0, mem=80.0))
    h.add({"tp": 2, "pp": 2}, _result(legal=False, reason="TP=2 illegal here"))

    full = h.summary_for_llm()
    assert "#000" in full and "#001" in full
    # Legal row shows OK + formatted tps; illegal shows REJECT + reason.
    assert "OK" in full
    assert "REJECT" in full
    assert "10,000" in full

    # ``k`` keeps only the trailing k rows.
    tail = h.summary_for_llm(k=1)
    assert "#000" not in tail
    assert "#001" in tail
