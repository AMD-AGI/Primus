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
    Python None passes."""

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
        parse_olmo3_skip_freq("4k", 4)


def test_window_rule_one_source():
    """W is olmo3_sliding_window. window_size next to it is told apart by shape: an int is W, a pair
    is the Megatron/TE (W - 1, 0); a pair is never read as W, so no value names two windows."""
    assert olmo3_window_from_settings(4096, 4096) == ((4095, 0), 4096)
    assert olmo3_window_from_settings("4096", 4096) == ((4095, 0), 4096)
    assert olmo3_window_from_settings([4095, 0], 4096) == ((4095, 0), 4096)
    assert olmo3_window_from_settings(None, "4096") == ((4095, 0), 4096)
    # the old HF-form pair [W, 0] is refused: as a pair it is the TE window of W + 1 keys
    with pytest.raises(ValueError, match="disagrees"):
        olmo3_window_from_settings([4096, 0], 4096)
    # an override of only olmo3_sliding_window next to the preset's 4096 is refused, any W
    for w in (2048, 4095, 4097, 8192):
        with pytest.raises(ValueError, match="disagrees"):
            olmo3_window_from_settings(4096, w)
        assert olmo3_window_from_settings(w, w) == ((w - 1, 0), w)
        assert olmo3_window_from_settings(None, w) == ((w - 1, 0), w)
    with pytest.raises(ValueError, match="window_size: null"):
        olmo3_window_from_settings(4096, 2048)
    # without olmo3_sliding_window, window_size alone is refused
    for ws in ([4095, 0], 4096, "(2047,0)"):
        with pytest.raises(ValueError, match="olmo3_sliding_window"):
            olmo3_window_from_settings(ws, None)
    with pytest.raises(ValueError, match="olmo3_sliding_window"):
        olmo3_window_from_settings(None, None)
    for ws in ([4093, 0], [4097, 0], [2048, 0], 4095, True):
        with pytest.raises(ValueError, match="window_size|disagrees"):
            olmo3_window_from_settings(ws, 4096)
    check_olmo3_unsupported(SimpleNamespace(olmo3_window_size_form="anything"))  # no such key any more


def test_norm_epsilon_must_match_the_layernorm_epsilon_the_model_reads():
    """The OLMo-3 model reads layernorm_epsilon only; a different norm_epsilon would be ignored."""
    check_olmo3_unsupported(SimpleNamespace(layernorm_epsilon=1e-6, norm_epsilon=1.0e-06))
    check_olmo3_unsupported(SimpleNamespace(layernorm_epsilon=1e-6))
    check_olmo3_unsupported(SimpleNamespace(layernorm_epsilon=1e-6, norm_epsilon=None))
    for bad in (1e-5, "x"):
        with pytest.raises(ValueError, match="norm_epsilon"):
            check_olmo3_unsupported(SimpleNamespace(layernorm_epsilon=1e-6, norm_epsilon=bad))


@pytest.mark.parametrize(
    "bad",
    ["[[4096,0]]", "((4096,0", "[4096,0)", "(4096,0))", "4096,,0", "(4096,0,)", "[4096.0,0]", (4096.0, 0)],
)
def test_window_string_needs_one_balanced_pair(bad):
    """A window string is "W, 0" bare (Megatron's tuple_type) or in one balanced () / [] pair; the
    entries are ints, as Megatron's tuple_type reads them (no float, no empty entry)."""
    for good in ("[4095, 0]", "(4095,0)", "4095,0"):
        assert olmo3_window_from_settings(good, 4096) == ((4095, 0), 4096)
    with pytest.raises(ValueError, match="window_size"):
        olmo3_window_from_settings(bad, 4096)


def test_skip_freq_bound_is_the_real_length():
    """The bound is the length the expression builds: a long literal list of num_layers entries is
    accepted; digit-free or tuple elements, ** and huge factors are refused."""
    pattern = "[" + "1," * 31 + "0]*2"
    assert len(parse_olmo3_skip_freq(pattern, num_layers=64)) == 64
    assert parse_olmo3_skip_freq("([1]*3 + [0])*8", num_layers=32) == [1, 1, 1, 0] * 8
    for bad in ("[" + "()," * 10 + "]*131072", "[1]*10**10", "[1]*10000000000", "[1,2]*[1]", "[1]+2"):
        with pytest.raises(ValueError):
            parse_olmo3_skip_freq(bad, num_layers=4)


def test_int_skip_freq_has_the_string_bound():
    """With num_layers unknown, an int skip frequency has the same bound typed or as a string."""
    for v in (5000, "5000"):
        with pytest.raises(ValueError, match="4096"):
            parse_olmo3_skip_freq(v, num_layers=None)
    assert parse_olmo3_skip_freq(4, num_layers=None) == parse_olmo3_skip_freq("4", num_layers=None) == 4


def test_full_width_k_norm_needs_disjoint_kv_heads_per_tp_rank():
    """num_query_groups < TP replicates KV heads across ranks: the cross-rank k_norm statistic would
    count them twice, so it is refused; any TP up to num_query_groups is fine."""
    for tp, ng in ((1, 32), (2, 32), (8, 32), (32, 32), (2, 2)):
        check_olmo3_unsupported(SimpleNamespace(tensor_model_parallel_size=tp, num_query_groups=ng))
    for tp, ng in ((4, 2), (8, 4), (2, 1)):
        with pytest.raises(
            ValueError, match=r"tensor_model_parallel_size \(\d+\) must divide num_query_groups"
        ):
            check_olmo3_unsupported(SimpleNamespace(tensor_model_parallel_size=tp, num_query_groups=ng))
    # num_query_groups unset means MHA (num_attention_heads)
    check_olmo3_unsupported(SimpleNamespace(tensor_model_parallel_size=8, num_attention_heads=32))


def test_layernorm_recompute_is_refused():
    """OLMo-3's input / pre-MLP norms are IdentityOp (post-norm model): selective layernorm
    recompute would silently do nothing, so it is refused; core_attn / mlp and full recompute pass."""
    for modules in (
        ["layernorm"],
        ["core_attn", "layernorm"],
        "core_attn,layernorm",
        "[layernorm,mlp]",
        "['layernorm']",
        "layernorm mlp",
    ):
        with pytest.raises(ValueError, match="layernorm"):
            check_olmo3_unsupported(
                SimpleNamespace(recompute_granularity="selective", recompute_modules=modules)
            )
    for ok in (
        SimpleNamespace(recompute_granularity="selective", recompute_modules=["core_attn", "mlp"]),
        SimpleNamespace(recompute_granularity="selective", recompute_modules=None),
        SimpleNamespace(recompute_granularity="full", recompute_modules=["layernorm"]),
        SimpleNamespace(recompute_granularity=None, recompute_modules=["layernorm"]),
    ):
        check_olmo3_unsupported(ok)


def test_no_rope_freq_string_and_list_forms_agree():
    """An all-zero no_rope_freq is accepted in every form (list, "[0,0,0,0]", 0, "0"); a pattern
    that skips RoPE somewhere is refused in every form."""
    for ok in ([0, 0, 0, 0], "[0,0,0,0]", "[0, 0]", 0, "0", None):
        check_olmo3_unsupported(SimpleNamespace(no_rope_freq=ok))
    for bad in ([0, 1], "[0,1]", 4, "4"):
        with pytest.raises(ValueError, match="no_rope_freq"):
            check_olmo3_unsupported(SimpleNamespace(no_rope_freq=bad))


def test_missing_window_message_names_window_size_only_when_given():
    with pytest.raises(ValueError) as e:
        olmo3_window_from_settings(None, None)
    assert "window_size" not in str(e.value)
    with pytest.raises(ValueError, match="window_size=4096 without it"):
        olmo3_window_from_settings(4096, None)


def test_kv_heads_check_follows_group_query_attention():
    """With group_query_attention off (MHA), Megatron ignores num_query_groups (default 1) and uses
    num_attention_heads, so the check does too; with it on, TP must divide num_query_groups."""
    NS = SimpleNamespace
    for tp in (2, 8):
        check_olmo3_unsupported(
            NS(
                tensor_model_parallel_size=tp,
                num_query_groups=1,
                group_query_attention=False,
                num_attention_heads=32,
            )
        )
    check_olmo3_unsupported(NS(tensor_model_parallel_size=4, num_query_groups=8, group_query_attention=True))
    with pytest.raises(
        ValueError, match=r"tensor_model_parallel_size \(4\) must divide num_query_groups \(6\)"
    ):
        check_olmo3_unsupported(
            NS(tensor_model_parallel_size=4, num_query_groups=6, group_query_attention=True)
        )
    with pytest.raises(ValueError, match=r"tensor_model_parallel_size \(\d+\) must divide num_query_groups"):
        check_olmo3_unsupported(
            NS(tensor_model_parallel_size=4, num_query_groups=2, group_query_attention=True)
        )
    # a config object (no group_query_attention attribute) holds the real KV head count
    with pytest.raises(ValueError, match=r"tensor_model_parallel_size \(\d+\) must divide num_query_groups"):
        check_olmo3_unsupported(NS(tensor_model_parallel_size=4, num_query_groups=2))


def test_window_number_forms_agree_and_null_is_unset():
    """window_size and olmo3_sliding_window read numbers by one rule (ints only), and the YAML
    null spellings a CLI override passes ("null", "~", "none") leave window_size unset."""
    for ws in (None, "null", "~", "None", 4096, "4096", [4095, 0]):
        assert olmo3_window_from_settings(ws, 4096) == ((4095, 0), 4096)
    for ws, sw in (("4096.0", 4096), (None, "4096.0"), (None, 4096.0), (4096.0, 4096)):
        with pytest.raises(ValueError):
            olmo3_window_from_settings(ws, sw)


@pytest.mark.parametrize(
    "key,value",
    [
        ("enable_cuda_graph", True),
        ("external_cuda_graph", "true"),
        ("cuda_graph_impl", "local"),
        ("cuda_graph_impl", "transformer_engine"),
    ],
)
def test_cuda_graphs_are_refused(key, value):
    with pytest.raises(ValueError, match="CUDA graphs"):
        check_olmo3_unsupported(SimpleNamespace(**{key: value}))
    check_olmo3_unsupported(
        SimpleNamespace(enable_cuda_graph=False, external_cuda_graph=False, cuda_graph_impl="none")
    )
