###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the tuning-agent tool helpers (``tools.py``).

Covers the pure helpers (`_safe_load_json`, `_tag`) and the tools that only
read evaluator state; the rest of the tool-belt holds live history state.
"""

from __future__ import annotations

import pytest

pytest.importorskip("primus.agents.tuning_agent.tools")

from primus.agents.tuning_agent.legality import TrialConfig  # noqa: E402
from primus.agents.tuning_agent.tools import _safe_load_json, _tag  # noqa: E402


def test_safe_load_json_passthrough_dict():
    d = {"tp": 2}
    assert _safe_load_json(d) is d


def test_safe_load_json_parses_plain_json():
    assert _safe_load_json('{"tp": 2, "pp": 4}') == {"tp": 2, "pp": 4}


def test_safe_load_json_strips_code_fence():
    assert _safe_load_json('```json\n{"ep": 8}\n```') == {"ep": 8}
    assert _safe_load_json('```\n{"cp": 2}\n```') == {"cp": 2}


def test_safe_load_json_returns_none_on_invalid():
    assert _safe_load_json("not json") is None
    assert _safe_load_json(123) is None  # non-str, non-dict


def test_tag_encodes_parallelism_axes():
    cfg = TrialConfig(tp=2, pp=4, ep=8, cp=1, mbs=2)
    assert _tag(cfg, 7) == "007_tp2_pp4_ep8_cp1_mbs2"


def _tools(evaluator):
    import json
    from types import SimpleNamespace

    from primus.agents.tuning_agent.tools import build_tools

    cfg = SimpleNamespace(optimization=SimpleNamespace(budget=1))
    tools = build_tools(cfg, None, None, None, evaluator, None, [], quiet=True)
    return {t.__name__: t for t in tools}, json


def test_get_op_coverage_is_registered_and_reports_bringup_off():
    from types import SimpleNamespace

    tools, json = _tools(SimpleNamespace(op_bringup=None))
    assert json.loads(tools["get_op_coverage"]()) == {"op_bringup": "off"}


def test_get_op_coverage_returns_the_agent_summary():
    from types import SimpleNamespace

    summary = {"policy": "auto", "gaps_seen": {"fancy": {}}, "results": [], "published": {}}
    tools, json = _tools(SimpleNamespace(op_bringup=SimpleNamespace(summary=lambda: summary)))
    assert json.loads(tools["get_op_coverage"]()) == summary
