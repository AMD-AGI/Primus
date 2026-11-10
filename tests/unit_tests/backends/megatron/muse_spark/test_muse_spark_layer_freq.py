###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""primus/backends/megatron/core/models/muse_spark/layer_freq.py: Muse Spark's per-layer pattern helpers (torch-free)."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[5] / "primus/backends/megatron/core/models/muse_spark/layer_freq.py"


def _mod():
    spec = importlib.util.spec_from_file_location("layer_freq", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "raw, n, expected",
    [
        (None, 3, [1, 1, 1]),  # unset: upstream's default 1
        (1, 3, [1, 1, 1]),
        (2, 5, [1, 0, 1, 0, 1]),
        ([1, 0, 1], 3, [1, 0, 1]),
        ((0, 1), 2, [0, 1]),
        ("[1, 0]", 2, [1, 0]),
        ("([0]*1+[1]*3)", 4, [0, 1, 1, 1]),
        # an int string (a CLI / env override) means the same as the int
        ("2", 4, [1, 0, 1, 0]),
        (" 1 ", 2, [1, 1]),
    ],
)
def test_moe_layer_pattern(raw, n, expected):
    m = _mod()
    assert m.moe_layer_pattern(raw, n) == expected
    if isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        # an int string means "a ratio" to every reader, not "no pattern" in one of them
        assert m.parse_layer_pattern(raw) == int(raw)


@pytest.mark.parametrize("bad", [0, -1, -2, "0", "-2"])
def test_a_non_positive_int_moe_layer_freq_is_an_error(bad):
    # upstream divides by zero for 0 and builds MoE on every |N|-th layer for a negative N:
    # no all-dense answer that would undercount the MoE FLOPs
    with pytest.raises(ValueError, match="must be >= 1"):
        _mod().moe_layer_pattern(bad, 4)


@pytest.mark.parametrize("bad", [True, False, "", "  "])
def test_a_bool_or_blank_moe_layer_freq_is_an_error(bad):
    # a bool or a blank value is not a layout: it never silently means all-MoE
    with pytest.raises(ValueError, match="^moe_layer_freq: "):
        _mod().moe_layer_pattern(bad, 4)


def test_a_pattern_of_the_wrong_length_is_an_error():
    with pytest.raises(ValueError, match="moe_layer_freq has 2 entries"):
        _mod().moe_layer_pattern([1, 0], 3)


@pytest.mark.parametrize(
    "bad, match",
    [
        ([0.9, 1, 0.9, 1], "entries must be ints"),  # was truncated to 0: a layer silently dense
        ([True, False, True, False], "entries must be ints"),
        ([2, 0, 3, 0], "must be 0 \\(dense\\) or 1 \\(MoE\\)"),
        ("[2, 0, 3, 0]", "must be 0 \\(dense\\) or 1 \\(MoE\\)"),
        ([1] * 4097, "longer than 4096"),
    ],
)
def test_a_list_gets_the_same_checks_as_a_string(bad, match):
    # a list must hold 0/1 ints, as a string does: floats / bools / 2s are not flags
    with pytest.raises(ValueError, match=match):
        _mod().moe_layer_pattern(bad, len(bad))


def test_a_pattern_string_is_parsed_once(monkeypatch):
    m = _mod()
    calls = []
    real = m.parse_layer_pattern
    monkeypatch.setattr(m, "parse_layer_pattern", lambda v: calls.append(v) or real(v))
    assert m.moe_layer_pattern("[1,0,1,0]", 4) == [1, 0, 1, 0]
    assert len(calls) == 1


@pytest.mark.parametrize(
    "bad", ["abc", "[1,2", "__import__('os').getpid()", "[1]*'a'", "[1]*10**9", "[1.0, 0]"]
)
def test_a_pattern_string_runs_no_code_and_names_the_field(bad):
    # only pattern expressions parse (no eval); anything else is a ValueError naming the field
    with pytest.raises(ValueError, match="^moe_layer_freq: "):
        _mod().moe_layer_pattern(bad, 4)


def test_a_repetition_over_the_cap_is_refused_before_it_is_built():
    with pytest.raises(ValueError, match=r"repetition 1\*1000000000 is outside \[0, 4096\]"):
        _mod().moe_layer_pattern("[1]*1000000000", 4)


def test_the_module_imports_no_torch_or_megatron():
    text = _PATH.read_text()
    assert "import torch" not in text and "megatron.core" not in text.replace(
        "primus.backends.megatron.core", ""
    )


def test_count_arithmetic_is_accepted_like_megatron():
    m = _mod()
    assert m.parse_layer_pattern("[1]*(60+2)") == [1] * 62
    assert m.parse_layer_pattern("[0]*(4-1)+[1]") == [0, 0, 0, 1]
    with pytest.raises(ValueError, match="count"):
        m.parse_layer_pattern("[1]*(5000*5000)")


@pytest.mark.parametrize("src", ["[0 if i < 3 else 1 for i in range(62)]", "x", "foo()"])
def test_a_comprehension_or_name_is_refused_with_a_short_message(src):
    m = _mod()
    with pytest.raises(ValueError) as e:
        m.parse_layer_pattern(src)
    msg = str(e.value)
    assert "write the list out" in msg and "Name(" not in msg  # no ast.dump in the message
