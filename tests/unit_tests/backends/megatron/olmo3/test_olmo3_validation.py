###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 validation tests that need neither torch nor Megatron.

``olmo3_validation`` is pure Python, but its package ``primus.backends.megatron`` imports the
Megatron adapter (torch). It is loaded by file path, so these tests run (are never skipped) where
``test_olmo3_components.py`` is skipped for a missing ``megatron.core``.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[5] / "primus/backends/megatron/core/models/olmo3/olmo3_validation.py"
_spec = importlib.util.spec_from_file_location("_olmo3_validation_pure", _PATH)
olmo3_validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(olmo3_validation)

check_olmo3_unsupported = olmo3_validation.check_olmo3_unsupported
olmo3_window_from_settings = olmo3_validation.olmo3_window_from_settings
parse_olmo3_skip_freq = olmo3_validation.parse_olmo3_skip_freq


def test_enum_keys_refuse_none_strings_but_unset_values_stay_unset():
    """For the enum keys Megatron compares exactly, the strings none / null / ~ are values, not
    "unset" (position_embedding_type: none is "no position embedding"), so they are refused. Only
    Python None passes. Keys whose readers treat null as unset (window_attn_skip_freq) still do."""

    for key, bad in (
        ("position_embedding_type", "none"),
        ("position_embedding_type", "null"),
        ("normalization", "None"),
        ("normalization", "NONE"),
        ("softmax_type", "none"),
        ("softmax_type", "null"),
        ("transformer_impl", "null"),
        ("transformer_impl", "~"),
    ):
        with pytest.raises(ValueError, match=key):
            check_olmo3_unsupported(SimpleNamespace(**{key: bad}))
    for key, good in (
        ("position_embedding_type", "rope"),
        ("normalization", "RMSNorm"),
        ("softmax_type", "vanilla"),
        ("transformer_impl", "transformer_engine"),
    ):
        check_olmo3_unsupported(SimpleNamespace(**{key: good}))
        check_olmo3_unsupported(SimpleNamespace(**{key: None}))


def test_skip_freq_messages():
    """A negative int gets the >= 2 message, not a parse error."""

    for bad in (-4, "-4", " -4 "):
        with pytest.raises(ValueError, match="must be >= 2"):
            parse_olmo3_skip_freq(bad, 8)
    with pytest.raises(ValueError, match="cannot parse"):
        parse_olmo3_skip_freq("[1,0]**2", 4)


def test_skip_freq_parser_is_fast_and_safe():
    import time

    t0 = time.monotonic()
    for bad in ("[1]*10**10", "[1]*" + "9" * 4000, "(" * 2000 + ")" * 2000, "[1]*99999*99999"):
        with pytest.raises(ValueError, match="window_attn_skip_freq"):
            parse_olmo3_skip_freq(bad, 4)
    # generous bound: only an exponential blow-up (a real string evaluation) can exceed it
    assert time.monotonic() - t0 < 10.0


def test_window_form_spelling_is_one_rule():
    """'HF' / ' megatron ' read the same in the window reader and the refusal table."""

    for form, ws in (("HF", [4096, 0]), (" megatron ", [4095, 0])):
        assert olmo3_window_from_settings(ws, None, form) == ((4095, 0), 4096)
        check_olmo3_unsupported(SimpleNamespace(olmo3_window_size_form=form))
    for bad in ("hugging", 1):
        with pytest.raises(ValueError, match="olmo3_window_size_form"):
            check_olmo3_unsupported(SimpleNamespace(olmo3_window_size_form=bad))


def test_hf_form_refuses_an_odd_window_that_looks_like_the_megatron_form():
    """With olmo3_window_size_form: hf (the preset), the usual Megatron/TE override [4095, 0] would
    give W = 4095 (one key short): it is refused with both readings named. An odd W is taken when
    olmo3_sliding_window names it; even windows and the Megatron form are unchanged."""
    with pytest.raises(ValueError, match=r"window_size: \[4096, 0\]"):
        olmo3_window_from_settings([4095, 0], None, "hf")
    with pytest.raises(ValueError, match="olmo3_window_size_form: megatron"):
        olmo3_window_from_settings("[511, 0]", None, " HF ")
    assert olmo3_window_from_settings([4095, 0], 4095, "hf") == ((4094, 0), 4095)
    assert olmo3_window_from_settings([4096, 0], None, "hf") == ((4095, 0), 4096)
    assert olmo3_window_from_settings([4095, 0], None, "megatron") == ((4095, 0), 4096)
    assert olmo3_window_from_settings([512, 0], None, "hf") == ((511, 0), 512)


def test_norm_epsilon_must_match_the_layernorm_epsilon_the_model_reads():
    """The OLMo-3 model reads layernorm_epsilon only; a different norm_epsilon would be ignored."""
    check_olmo3_unsupported(SimpleNamespace(layernorm_epsilon=1e-6, norm_epsilon=1.0e-06))
    check_olmo3_unsupported(SimpleNamespace(layernorm_epsilon=1e-6))
    check_olmo3_unsupported(SimpleNamespace(layernorm_epsilon=1e-6, norm_epsilon=None))
    for bad in (1e-5, "x"):
        with pytest.raises(ValueError, match="norm_epsilon"):
            check_olmo3_unsupported(SimpleNamespace(layernorm_epsilon=1e-6, norm_epsilon=bad))
