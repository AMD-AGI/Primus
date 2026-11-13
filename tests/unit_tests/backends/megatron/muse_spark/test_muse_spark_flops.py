###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark FLOPs correction (pure Python; Megatron's count is stubbed)."""

from types import SimpleNamespace

import pytest

from primus.backends.megatron.core.models.muse_spark.layer_patterns import (
    normalize_layer_pattern,
    parse_window_size,
)
from primus.backends.megatron.patches.muse_spark_flops_patches import (
    make_muse_spark_num_floating_point_operations,
    muse_spark_flops_delta,
    num_sliding_layers,
    sliding_window_keys,
)


def _args(**kw):
    base = dict(
        model_type="muse_spark",
        seq_length=4096,
        kv_channels=64,
        num_attention_heads=128,
        hidden_size=7168,  # differs from the q width (64 * 128 = 8192): a swap is caught
        num_layers=4,
        num_experts=256,
        window_size=[2048, 0],
        window_attn_skip_freq=[1, 0, 1, 1],
    )
    return SimpleNamespace(**{**base, **kw})


def test_sliding_layers_are_charged_less_and_router_added():
    a = _args()
    d = muse_spark_flops_delta(a, 2)
    # q width kv_channels * heads = 8192 (per_layer), router width hidden_size = 7168 (router)
    # hand-computed for seq 4096, HF window 2048: a query at position i sees min(i+1, 2048)
    # keys, mean = (2048*2049/2 + 2048*2048) / 4096 = 1536.25; full causal mean = 4097/2 =
    # 2048.5. Megatron charges 6 * 2 * q * (S/2) per token per layer for QK^T + AV; each of
    # the 3 sliding layers (window_attn_skip_freq [1, 0, 1, 1]) is scaled by 1536.25 / 2048.5.
    per_layer = 12 * 8192 * 2048 * (1536.25 / 2048.5 - 1)
    attn = 3 * per_layer
    router = 6 * 7168 * 256 * 4
    assert d == pytest.approx(2 * 4096 * (attn + router), rel=1e-12)
    # sliding layers are charged less: the delta is below the router-only delta
    assert muse_spark_flops_delta(a, 1) < muse_spark_flops_delta(_args(window_attn_skip_freq=[0, 0, 0, 0]), 1)
    # no sliding layer -> router only
    assert muse_spark_flops_delta(_args(window_attn_skip_freq=[0, 0, 0, 0]), 1) == 4096 * router


def test_wrapper_applies_after_get_model_sets_the_model_type_enum():
    # Megatron's get_model replaces args.model_type with the ModelType enum before
    # train() counts FLOPs: the correction must still apply (decided at install time).
    from enum import Enum

    class ModelType(Enum):
        encoder_or_decoder = 1

    expected = 1.0e15 + muse_spark_flops_delta(_args(), 1)
    assert expected != 1.0e15
    # Megatron's order: the enum is set before the FIRST FLOPs call (an empty cache), so
    # nothing on the cache-miss path may depend on the string model_type
    from primus.backends.megatron.patches.muse_spark_flops_patches import (
        MUSE_SPARK_ARGS_ATTR,
    )

    wrapped = make_muse_spark_num_floating_point_operations(lambda a, b: 1.0e15)
    a = _args()
    setattr(a, MUSE_SPARK_ARGS_ATTR, True)  # the patch marks the run's args at install time
    a.model_type = ModelType.encoder_or_decoder
    assert wrapped(a, 1) == expected
    # and a fresh marked args object with the enum (a new cache miss) is corrected too
    b = _args()
    setattr(b, MUSE_SPARK_ARGS_ATTR, True)
    b.model_type = ModelType.encoder_or_decoder
    assert wrapped(b, 1) == expected
    # an unmarked args object with the enum is another model's: upstream's count
    c = _args()
    c.model_type = ModelType.encoder_or_decoder
    assert wrapped(c, 1) == 1.0e15


def test_the_wrapper_leaves_a_non_muse_model_alone():
    # a later non-Muse run in the same process keeps upstream's count (window and experts
    # set, so a Muse delta would be non-zero)
    wrapped = make_muse_spark_num_floating_point_operations(lambda a, b, **kw: 1.0e12)
    other = _args(model_type="gpt", window_size=(1024, 0), num_experts=8)
    assert wrapped(other, 1) == 1.0e12
    assert wrapped(other, 2, seqlen_squared_sum_in_batch=7, total_real_tokens_in_batch=9) == 1.0e12


@pytest.mark.parametrize(
    "model_type, expected", [("gpt", False), ("muse_spark", True), ("muse_spark_text", False)]
)
def test_patch_condition_is_muse_spark_only(model_type, expected):
    """The condition registered for the patch (not a re-import of the predicate); the HF
    text_config model_type ``muse_spark_text`` is not a Primus model type."""
    from primus.core.patches import PatchContext
    from primus.core.patches.patch_registry import PatchRegistry

    registered = PatchRegistry.get("megatron.muse_spark.flops_reporting")
    assert registered is not None and registered.phase == "before_train"
    ctx = PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"backend_args": SimpleNamespace(model_type=model_type)},
    )
    assert bool(registered.condition(ctx)) is expected


def test_patch_wraps_megatron_flops_once(monkeypatch):
    """The install path: Megatron's function is swapped once; a second install is a no-op."""
    import sys
    import types

    from primus.backends.megatron.patches import muse_spark_flops_patches as mod

    def megatron_flops(args, batch_size):
        return 1.0e15

    training = types.ModuleType("megatron.training.training")
    training.num_floating_point_operations = megatron_flops
    pkg_training = types.ModuleType("megatron.training")
    pkg_training.training = training
    pkg = sys.modules.get("megatron") or types.ModuleType("megatron")
    monkeypatch.setitem(sys.modules, "megatron", pkg)
    monkeypatch.setattr(pkg, "training", pkg_training, raising=False)
    monkeypatch.setitem(sys.modules, "megatron.training", pkg_training)
    monkeypatch.setitem(sys.modules, "megatron.training.training", training)
    monkeypatch.setattr(mod, "log_rank_0", lambda *_a, **_k: None)

    ctx = SimpleNamespace(extra={"backend_args": SimpleNamespace(model_type="muse_spark")})
    mod.patch_muse_spark_flops_reporting(ctx)
    assert getattr(ctx.extra["backend_args"], mod.MUSE_SPARK_ARGS_ATTR) is True  # run's args marked
    first = training.num_floating_point_operations
    assert first is not megatron_flops and first.__wrapped__ is megatron_flops
    mod.patch_muse_spark_flops_reporting(ctx)
    assert training.num_floating_point_operations is first  # not wrapped twice
    a = _args()
    assert first(a, 1) == 1.0e15 + muse_spark_flops_delta(a, 1)


def test_window_not_shorter_than_seq_is_no_correction():
    # sliding == full causal: only the router term (review: the +0.5 key bias)
    router = 6 * 7168 * 16 * 4
    a = _args(seq_length=1024, num_experts=16, window_attn_skip_freq=[1, 0, 1, 1])
    assert muse_spark_flops_delta(a, 1) == 1024 * router
    assert muse_spark_flops_delta(_args(seq_length=2048, num_experts=16), 1) == 2048 * router


@pytest.mark.parametrize("impl", ["megatron", "fallback"])
def test_sliding_layers_follow_megatron_int_and_none_patterns(monkeypatch, impl):
    # Megatron: None -> every layer is a window layer; int N -> layer % N != 0 (1-indexed).
    # "fallback": Megatron's helper cannot be imported, so the pure-Python copy is used.
    import sys

    from primus.backends.megatron.patches import muse_spark_flops_patches as mod

    if impl == "fallback":
        monkeypatch.setitem(sys.modules, "megatron.core.transformer.utils", None)
        assert mod._window_layer_fn() is mod._fallback_is_window_layer
    else:
        pytest.importorskip("megatron.core.transformer.utils")
        assert mod._window_layer_fn() is not mod._fallback_is_window_layer
    assert num_sliding_layers(_args(window_attn_skip_freq=None)) == 4
    assert num_sliding_layers(_args(window_attn_skip_freq=4)) == 3
    assert num_sliding_layers(_args(window_attn_skip_freq=2)) == 2
    assert num_sliding_layers(_args(window_attn_skip_freq=[0, 1, 1, 1])) == 3
    assert num_sliding_layers(_args(window_size=None, window_attn_skip_freq=None)) == 0
    router = 4096 * 6 * 7168 * 256 * 4
    for pattern in (None, 4):
        assert muse_spark_flops_delta(_args(window_attn_skip_freq=pattern), 1) < router
    assert muse_spark_flops_delta(_args(window_attn_skip_freq=None), 1) < muse_spark_flops_delta(
        _args(window_attn_skip_freq=4), 1
    )


@pytest.mark.parametrize("pattern", [None, 1, 2, 4, [1, 0, 1, 1], [0, 0, 0, 0]])
@pytest.mark.parametrize("window", [None, (2047, 0)])
def test_the_fallback_matches_megatrons_window_rule(pattern, window):
    utils = pytest.importorskip("megatron.core.transformer.utils")
    from primus.backends.megatron.patches import muse_spark_flops_patches as mod

    for layer in range(1, 5):
        assert mod._fallback_is_window_layer(window, pattern, layer) == bool(
            utils.is_layer_window_attention(window, pattern, layer)
        )


def test_router_term_counts_only_moe_layers():
    from primus.backends.megatron.patches.muse_spark_flops_patches import num_moe_layers

    nosw = dict(window_attn_skip_freq=[0, 0, 0, 0])
    router_layer = 4096 * 6 * 7168 * 256  # seq * 6 * hidden * experts, batch 1
    assert muse_spark_flops_delta(_args(**nosw, moe_layer_freq=[1, 0, 1, 0]), 1) == 2 * router_layer
    assert muse_spark_flops_delta(_args(**nosw, moe_layer_freq=2), 1) == 2 * router_layer
    assert muse_spark_flops_delta(_args(**nosw, moe_layer_freq=1), 1) == 4 * router_layer
    assert num_moe_layers(_args(moe_layer_freq=[1, 1, 1, 1], mtp_num_layers=1)) == 5
    assert num_moe_layers(_args(moe_layer_freq=[1, 1, 1, 0], mtp_num_layers=1)) == 3
    assert num_moe_layers(_args(num_experts=None)) == 0
    assert num_moe_layers(_args(moe_layer_freq="([1]*3+[0])")) == 3
    # unset is upstream's default 1 (every layer); no MoE layer is an all-0 list; an int
    # <= 0 (upstream: a division by zero, or every |N|-th layer) is an error, not 0 layers
    assert num_moe_layers(_args(moe_layer_freq=None)) == 4
    assert num_moe_layers(_args(moe_layer_freq=[0, 0, 0, 0])) == 0
    assert num_moe_layers(_args(moe_layer_freq=[1, 0, 1, 1], mtp_num_layers=None)) == 3
    for bad in (0, -2):
        with pytest.raises(ValueError, match="must be >= 1"):
            num_moe_layers(_args(moe_layer_freq=bad))


@pytest.mark.parametrize(
    "ws, expected",
    [
        ([2048, 0], 2048),
        ((2048, 0), 2048),
        ("(2048,0)", 2048),
        ("[2048, 0]", 2048),
        (2048, 2048),
        (None, None),
        ([-1, 0], None),
        ("(-1, 0)", None),
    ],
)
def test_sliding_window_keys_accepts_every_window_size_form(ws, expected):
    assert sliding_window_keys(SimpleNamespace(window_size=ws)) == expected


def test_string_window_size_is_counted_like_a_list():
    assert muse_spark_flops_delta(_args(window_size="(2048,0)"), 2) == muse_spark_flops_delta(_args(), 2)


@pytest.mark.parametrize("bad", ["(2048,0]", "2048,,0", "(abc)", "(2048, 5)"])
def test_the_sliding_layer_count_parses_a_string_window_size(bad):
    # the layer count tests the window only for truthiness, and any non-empty string is
    # truthy: without the parse (muse_spark_flops_patches.num_sliding_layers) a malformed
    # string would be counted as a window on every sliding layer. With it, it raises.
    from primus.backends.megatron.patches.muse_spark_flops_patches import (
        num_sliding_layers,
    )

    assert num_sliding_layers(_args(window_size="(2048,0)")) == num_sliding_layers(_args()) == 3
    with pytest.raises(ValueError, match="window_size"):
        num_sliding_layers(_args(window_size=bad))


@pytest.mark.parametrize(
    "bad",
    [
        "(2048)",
        "(a,0)",
        "(1,2,3)",
        [1],
        3.5,
        # malformed strings the reviewer found parsing silently
        "2048,,0",
        "((2048,0",
        "[(2048,0)]",
        "[[2048,0]]",
        "(2048,0]",
        "",
        "()",
        # out of range: left < -1, non-causal right
        [-5, 0],
        "(-7,0)",
        (2048, 5),
        "[2048, -1]",
        # non-int items
        [2048.0, 0],
        [True, 0],
        ["2048", 0],
    ],
)
def test_parse_window_size_rejects_malformed_values(bad):
    with pytest.raises(ValueError, match="window_size"):
        parse_window_size(bad)


@pytest.mark.parametrize(
    "ok, expected",
    [
        ("2048,0", (2048, 0)),
        (" ( 2048 , 0 ) ", (2048, 0)),
        ("[-1,0]", (-1, 0)),
        ((0, 0), (0, 0)),
    ],
)
def test_parse_window_size_accepts_one_pair(ok, expected):
    assert parse_window_size(ok) == expected


def test_flops_window_is_validated_like_the_config():
    # HF convention (the presets): an HF window of 0 or a negative one is an error, not
    # silently "full attention"
    for bad in ([0, 0], [-5, 0], (2048, 5)):
        with pytest.raises(ValueError, match="window_size"):
            sliding_window_keys(SimpleNamespace(window_size=bad))
    # TE convention: left excludes the query
    te = SimpleNamespace(window_size=(2047, 0), window_size_is_hf_sliding_window=False)
    assert sliding_window_keys(te) == 2048
    assert (
        sliding_window_keys(SimpleNamespace(window_size=(-1, 0), window_size_is_hf_sliding_window=False))
        is None
    )


@pytest.mark.parametrize("src", ["[]*100000000000000000000", "([0]*0)*100000000000000000000"])
def test_huge_repetition_of_an_empty_list_raises_a_named_value_error(src):
    with pytest.raises(ValueError, match="no_rope_freq"):
        normalize_layer_pattern(src, 62, "no_rope_freq")


def test_the_wrapper_reads_the_fixed_args_once(monkeypatch):
    from primus.backends.megatron.patches import muse_spark_flops_patches as mod

    a = _args()
    one, three = muse_spark_flops_delta(a, 1), muse_spark_flops_delta(a, 3)
    b = _args(seq_length=2048)  # a different args object with a different delta
    b_one = muse_spark_flops_delta(b, 1)
    assert b_one != one
    calls = []
    real = mod.muse_spark_flops_delta_per_sample
    monkeypatch.setattr(
        mod, "muse_spark_flops_delta_per_sample", lambda a, terms=None: calls.append(1) or real(a, terms)
    )
    wrapped = mod.make_muse_spark_num_floating_point_operations(lambda a, b: 0.0)
    assert wrapped(a, 1) == one
    assert wrapped(a, 3) == three  # only batch_size changes per call
    assert len(calls) == 1
    # a different args object is recomputed, not served from the cache
    assert wrapped(b, 1) == b_one
    assert len(calls) == 2


def test_the_wrapper_forwards_the_extra_megatron_arguments():
    # Megatron-core 0.19+ call num_floating_point_operations(args, batch_size,
    # seqlen_squared_sum_in_batch=..., total_real_tokens_in_batch=...)
    from primus.backends.megatron.patches import muse_spark_flops_patches as mod

    seen = []
    a = _args()
    wrapped = mod.make_muse_spark_num_floating_point_operations(
        lambda args, bs, *rest, **kw: seen.append((rest, kw)) or 5.0
    )
    out = wrapped(a, 2, seqlen_squared_sum_in_batch=7, total_real_tokens_in_batch=9)
    assert out == 5.0 + muse_spark_flops_delta(
        a, 2, seqlen_squared_sum_in_batch=7, total_real_tokens_in_batch=9
    )
    assert seen == [((), {"seqlen_squared_sum_in_batch": 7, "total_real_tokens_in_batch": 9})]


def test_packed_stats_of_a_full_bshd_batch_give_the_bshd_delta():
    a = _args()
    s = a.seq_length
    assert muse_spark_flops_delta(
        a, 8, seqlen_squared_sum_in_batch=8 * s * s, total_real_tokens_in_batch=8 * s
    ) == pytest.approx(muse_spark_flops_delta(a, 8), rel=1e-12)


def test_packed_documents_shorter_than_the_window_get_no_window_correction():
    # B = 8 samples of 4096 tokens packed with 512-token documents (window 2048): every
    # document fits in the window, so upstream's full-causal count is already right; only
    # the router term (per real token) is added
    from primus.backends.megatron.patches import muse_spark_flops_patches as mod

    a = _args()
    tokens = 8 * 4096
    sq = (tokens // 512) * 512 * 512
    d = muse_spark_flops_delta(a, 8, seqlen_squared_sum_in_batch=sq, total_real_tokens_in_batch=tokens)
    assert d == pytest.approx(mod.MuseSparkFlopsTerms(a).router_per_token * tokens, rel=1e-12)
    assert d > 0  # was negative: the full-sequence window correction was subtracted


def test_packed_documents_longer_than_the_window_are_corrected_at_their_length():
    # equal 3072-token documents: the window ratio is that of a 3072-token sequence
    from primus.backends.megatron.patches import muse_spark_flops_patches as mod

    a = _args()
    n_docs, L = 10, 3072
    t = mod.MuseSparkFlopsTerms(a)
    want = t.sliding_attention_delta(n_docs * L * L, L) + t.router_per_token * n_docs * L
    got = muse_spark_flops_delta(
        a, 8, seqlen_squared_sum_in_batch=n_docs * L * L, total_real_tokens_in_batch=n_docs * L
    )
    assert got == pytest.approx(want, rel=1e-12)
    w = t.window
    ratio = (w * L - w * (w - 1) // 2) / (L * (L + 1) // 2)
    assert t.sliding_attention_delta(n_docs * L * L, L) == pytest.approx(
        6 * 8192 * n_docs * L * L * (ratio - 1) * 3, rel=1e-12
    )


def test_the_wrapper_uses_the_packed_stats_per_call():
    from primus.backends.megatron.patches import muse_spark_flops_patches as mod

    a = _args()
    wrapped = mod.make_muse_spark_num_floating_point_operations(lambda args, bs, **kw: 0.0)
    short = wrapped(a, 8, seqlen_squared_sum_in_batch=64 * 512 * 512, total_real_tokens_in_batch=64 * 512)
    full = wrapped(a, 8)
    assert short > 0 > full  # short documents: router only; full sequences: window correction wins


def test_num_moe_layers_rejects_a_pattern_of_the_wrong_length():
    from primus.backends.megatron.patches import muse_spark_flops_patches as mod

    with pytest.raises(ValueError, match="expected num_layers=4"):
        mod.num_moe_layers(SimpleNamespace(num_experts=8, num_layers=4, moe_layer_freq=[1, 0]))


@pytest.mark.parametrize("flag, expected", [("false", 2048), ("0", 2048), ("true", 2047), (None, 2047)])
def test_sliding_window_keys_parses_the_convention_flag_strictly(flag, expected):
    # (2047, 0): TE's left window 2047 = 2048 keys; in the HF convention it is 2047 keys
    a = SimpleNamespace(window_size=(2047, 0), window_size_is_hf_sliding_window=flag)
    assert sliding_window_keys(a) == expected
    with pytest.raises(ValueError, match="window_size_is_hf_sliding_window"):
        sliding_window_keys(SimpleNamespace(window_size=(2047, 0), window_size_is_hf_sliding_window="ture"))
