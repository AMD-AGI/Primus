###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for primus.backends.megatron_bridge.patches.gemma4.gemma4_config_overrides

Tests cover:
    - _parse_value: Python literals, the "none" spelling, bare-string fallback
    - _parse_spec: ";"-separated entries, so list values keep their commas
    - _coerce_to_field: enum-typed fields, which are the reason this helper exists

The enum cases are the point of the file. ``model.attention_backend`` is an
``AttnBackend``, and megatron-core compares it by enum identity, so assigning the
plain string "unfused" does not raise -- it silently fails to match and the run
continues on the default backend. A tuning hook that reports success and changes
nothing makes an A/B comparison return two identical numbers, which reads as
"this setting does not matter" rather than "this setting was never applied".

A locally defined Enum stands in for AttnBackend so these stay fast and do not
drag in megatron-core.
"""

from enum import Enum

import pytest


class _Backend(Enum):
    """Stand-in for megatron-core's AttnBackend."""

    flash = 1
    fused = 2
    unfused = 3
    local = 4


@pytest.fixture(autouse=True)
def logged(monkeypatch):
    """Capture log_rank_0 calls and return the list of messages.

    Primus's logger is process-global and only initialised by a real run, so
    calling it from a bare unit test raises AttributeError on a None logger.
    These code paths log on purpose -- an override that gets ignored should say
    so -- so the calls are captured rather than suppressed, which also lets a
    test assert that the message was emitted.
    """
    from primus.backends.megatron_bridge.patches.gemma4 import (
        gemma4_config_overrides as mod,
    )

    messages: list[str] = []
    monkeypatch.setattr(mod, "log_rank_0", lambda msg, *a, **k: messages.append(str(msg)))
    return messages


# -----------------------------------------------------------------------------
# Test _parse_value
# -----------------------------------------------------------------------------


def test_parse_value_literals():
    """Python literals are parsed to their real types, not left as strings."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_config_overrides import (
        _parse_value,
    )

    assert _parse_value("6") == 6
    assert _parse_value("1.5") == 1.5
    assert _parse_value("True") is True
    assert _parse_value("False") is False
    assert _parse_value("[1, 2, 3]") == [1, 2, 3]


def test_parse_value_none_spellings():
    """'none' and 'null' mean None.

    Needed because the values worth clearing are spelled the way they appear in
    YAML, and ``recompute_granularity=none`` must become None rather than the
    four-character string "none", which is truthy.
    """
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_config_overrides import (
        _parse_value,
    )

    for spelling in ("none", "None", "null", "NULL"):
        assert _parse_value(spelling) is None, spelling


def test_parse_value_bare_string_falls_back():
    """Anything that is not a literal stays a string rather than raising."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_config_overrides import (
        _parse_value,
    )

    assert _parse_value("unfused") == "unfused"
    assert _parse_value("selective") == "selective"


# -----------------------------------------------------------------------------
# Test _parse_spec
# -----------------------------------------------------------------------------


def test_parse_spec_separator_is_semicolon():
    """Entries split on ';' so that list values can keep their commas."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_config_overrides import (
        _parse_spec,
    )

    entries = _parse_spec("model.num_layers=6;model.recompute_granularity=none")
    assert entries == [("model.num_layers", 6), ("model.recompute_granularity", None)]


def test_parse_spec_preserves_commas_in_lists():
    """A comma-separated list survives as one value."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_config_overrides import (
        _parse_spec,
    )

    ((path, value),) = _parse_spec("model.recompute_num_layers=[1, 2, 3]")
    assert path == "model.recompute_num_layers"
    assert value == [1, 2, 3]


def test_parse_spec_skips_malformed_entries_loudly(logged):
    """An entry with no '=' is dropped and reported, not applied.

    Reporting matters more than dropping: a typo that vanished quietly would
    leave the run using defaults while the operator believed otherwise.
    """
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_config_overrides import (
        _parse_spec,
    )

    entries = _parse_spec("model.num_layers=6;garbage;model.recompute_granularity=none")

    assert entries == [("model.num_layers", 6), ("model.recompute_granularity", None)]
    assert any("garbage" in m for m in logged)


def test_parse_spec_tolerates_whitespace_and_empty_chunks():
    """Trailing separators and spacing are harmless, so manifests stay editable."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_config_overrides import (
        _parse_spec,
    )

    assert _parse_spec(" model.num_layers = 6 ;;") == [("model.num_layers", 6)]


# -----------------------------------------------------------------------------
# Test _coerce_to_field -- the regression guard
# -----------------------------------------------------------------------------


def test_coerce_enum_by_name():
    """A string matching a member name becomes that member, not a string.

    This is the bug the helper exists for: without coercion the assignment
    succeeds and the setting is silently ignored.
    """
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_config_overrides import (
        _coerce_to_field,
    )

    result = _coerce_to_field(_Backend.flash, "unfused", "model.attention_backend")

    assert result is _Backend.unfused
    assert not isinstance(result, str)


def test_coerce_enum_by_value():
    """A raw value also resolves, since _parse_value turns "3" into an int."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_config_overrides import (
        _coerce_to_field,
    )

    assert _coerce_to_field(_Backend.flash, 3, "model.attention_backend") is _Backend.unfused


def test_coerce_enum_invalid_keeps_previous_value_and_says_so(logged):
    """An unresolvable name must not be assigned, and must not be silent.

    Leaving the previous value is the safe outcome: writing the raw string would
    put the config into a state megatron-core cannot match, which is the failure
    this helper is meant to prevent. It has to be reported, though, or a typo in
    an override becomes indistinguishable from a setting that had no effect.
    """
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_config_overrides import (
        _coerce_to_field,
    )

    result = _coerce_to_field(_Backend.flash, "does_not_exist", "model.attention_backend")

    assert result is _Backend.flash
    assert any("model.attention_backend" in m and "does_not_exist" in m for m in logged)
    # The valid options belong in the message, so the fix is obvious from the log.
    assert any("unfused" in m for m in logged)


def test_coerce_leaves_non_enum_fields_alone():
    """Ordinary fields pass through untouched, including None."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_config_overrides import (
        _coerce_to_field,
    )

    assert _coerce_to_field(30, 6, "model.num_layers") == 6
    assert _coerce_to_field("full", None, "model.recompute_granularity") is None
    assert _coerce_to_field(None, 6, "model.recompute_num_layers") == 6
    assert _coerce_to_field(True, False, "model.masked_softmax_fusion") is False


def test_coerce_accepts_an_enum_already():
    """An Enum that is already the right type is returned as-is."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_config_overrides import (
        _coerce_to_field,
    )

    assert _coerce_to_field(_Backend.flash, _Backend.local, "model.attention_backend") is _Backend.local
