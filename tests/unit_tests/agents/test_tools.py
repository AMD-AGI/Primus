###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the tuning-agent tool helpers (``tools.py``).

Covers the pure helpers (`_safe_load_json`, `_tag`); the LLM tool-belt itself
holds live evaluator/history state and is out of scope here.
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
