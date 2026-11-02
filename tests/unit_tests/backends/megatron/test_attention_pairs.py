###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""primus.backends.megatron.core.attention_pairs (pure Python), against a brute-force count."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[4] / "primus/backends/megatron/core/attention_pairs.py"
_spec = importlib.util.spec_from_file_location("_attention_pairs", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
pairs = _mod.causal_visible_pairs


def brute(s, w):
    return sum(min(q + 1, w) if w else q + 1 for q in range(s))


@pytest.mark.parametrize("s", [1, 2, 7, 64])
def test_matches_brute_force_for_every_window(s):
    for w in range(1, s + 3):  # W = 1, ..., S - 1, S, S + 1, S + 2
        assert pairs(s, w) == brute(s, w), (s, w)


@pytest.mark.parametrize("window", [None, 0, -5])
def test_no_window_is_full_causal(window):
    assert pairs(10, window) == 55


def test_edge_windows():
    assert pairs(10, 1) == 10  # each query sees itself only
    assert pairs(10, 9) == 55 - 1  # only the last query misses one key
    assert pairs(10, 10) == pairs(10, 11) == 55


@pytest.mark.parametrize("window", [(2047, 0), [3, 0]])
def test_a_megatron_window_tuple_is_rejected_with_a_hint(window):
    with pytest.raises(TypeError, match="left \\+ 1"):
        pairs(4096, window)
