###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark FLOPs correction (pure Python; Megatron's count is stubbed)."""

from types import SimpleNamespace

import pytest

from primus.backends.megatron.core.models.muse_spark.layer_patterns import (
    causal_visible_pairs,
    normalize_layer_pattern,
    parse_window_size,
)
from primus.backends.megatron.patches.muse_spark_flops_patches import (
    causal_mean_keys,
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
        hidden_size=8192,
        num_layers=4,
        num_experts=256,
        window_size=[2048, 0],
        window_attn_skip_freq=[1, 0, 1, 1],
    )
    return SimpleNamespace(**{**base, **kw})


def test_sliding_layers_are_charged_less_and_router_added():
    a = _args()
    d = muse_spark_flops_delta(a, 2)
    assert a.kv_channels * a.num_attention_heads == 8192  # q width: the 8192 in per_layer
    # hand-computed for seq 4096, HF window 2048: a query at position i sees min(i+1, 2048)
    # keys, mean = (2048*2049/2 + 2048*2048) / 4096 = 1536.25; full causal mean = 4097/2 =
    # 2048.5. Megatron charges 6 * 2 * q * (S/2) per token per layer for QK^T + AV; each of
    # the 3 sliding layers (window_attn_skip_freq [1, 0, 1, 1]) is scaled by 1536.25 / 2048.5.
    per_layer = 12 * 8192 * 2048 * (1536.25 / 2048.5 - 1)
    assert per_layer == pytest.approx(-50_343_933.0007, rel=1e-9)
    attn = 3 * per_layer
    router = 6 * 8192 * 256 * 4
    assert d == pytest.approx(2 * 4096 * (attn + router), rel=1e-12) and attn < 0
    # no sliding layer -> router only
    assert muse_spark_flops_delta(_args(window_attn_skip_freq=[0, 0, 0, 0]), 1) == 4096 * router


def test_wrapper_applies_after_get_model_sets_the_model_type_enum():
    # Megatron's get_model replaces args.model_type with the ModelType enum before
    # train() counts FLOPs: the correction must still apply (decided at install time).
    from enum import Enum

    class ModelType(Enum):
        encoder_or_decoder = 1

    wrapped = make_muse_spark_num_floating_point_operations(lambda a, b: 1.0e15)
    a = _args()
    expected = 1.0e15 + muse_spark_flops_delta(a, 1)
    assert wrapped(a, 1) == expected
    a.model_type = ModelType.encoder_or_decoder
    assert wrapped(a, 1) == expected and expected != 1.0e15


def test_patch_condition_is_muse_spark_only():
    from primus.backends.megatron.core.models.muse_spark.model_types import (
        is_muse_spark_model_type,
    )

    ctx = SimpleNamespace(extra={"backend_args": SimpleNamespace(model_type="gpt")})
    assert not is_muse_spark_model_type(ctx)
    ctx = SimpleNamespace(extra={"backend_args": SimpleNamespace(model_type="muse_spark")})
    assert is_muse_spark_model_type(ctx)
    ctx = SimpleNamespace(extra={"backend_args": SimpleNamespace(model_type="muse_spark_text")})
    assert not is_muse_spark_model_type(ctx)  # HF text_config model_type only


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
    first = training.num_floating_point_operations
    assert first is not megatron_flops and first.__wrapped__ is megatron_flops
    mod.patch_muse_spark_flops_reporting(ctx)
    assert training.num_floating_point_operations is first  # not wrapped twice
    a = _args()
    assert first(a, 1) == 1.0e15 + muse_spark_flops_delta(a, 1)


def test_window_not_shorter_than_seq_is_no_correction():
    # sliding == full causal: only the router term (review: the +0.5 key bias)
    router = 6 * 8192 * 16 * 4
    a = _args(seq_length=1024, num_experts=16, window_attn_skip_freq=[1, 0, 1, 1])
    assert muse_spark_flops_delta(a, 1) == 1024 * router
    assert muse_spark_flops_delta(_args(seq_length=2048, num_experts=16), 1) == 2048 * router


def test_sliding_layers_follow_megatron_int_and_none_patterns():
    # Megatron: None -> every layer is a window layer; int N -> layer % N != 0 (1-indexed)
    assert num_sliding_layers(_args(window_attn_skip_freq=None)) == 4
    assert num_sliding_layers(_args(window_attn_skip_freq=4)) == 3
    assert num_sliding_layers(_args(window_attn_skip_freq=2)) == 2
    assert num_sliding_layers(_args(window_attn_skip_freq=[0, 1, 1, 1])) == 3
    assert num_sliding_layers(_args(window_size=None, window_attn_skip_freq=None)) == 0
    router = 4096 * 6 * 8192 * 256 * 4
    for pattern in (None, 4):
        assert muse_spark_flops_delta(_args(window_attn_skip_freq=pattern), 1) < router
    assert muse_spark_flops_delta(_args(window_attn_skip_freq=None), 1) < muse_spark_flops_delta(
        _args(window_attn_skip_freq=4), 1
    )


def test_router_term_counts_only_moe_layers():
    from primus.backends.megatron.patches.muse_spark_flops_patches import num_moe_layers

    nosw = dict(window_attn_skip_freq=[0, 0, 0, 0])
    router_layer = 4096 * 6 * 8192 * 256  # seq * 6 * hidden * experts, batch 1
    assert muse_spark_flops_delta(_args(**nosw, moe_layer_freq=[1, 0, 1, 0]), 1) == 2 * router_layer
    assert muse_spark_flops_delta(_args(**nosw, moe_layer_freq=2), 1) == 2 * router_layer
    assert muse_spark_flops_delta(_args(**nosw, moe_layer_freq=1), 1) == 4 * router_layer
    assert num_moe_layers(_args(moe_layer_freq=[1, 1, 1, 1], mtp_num_layers=1)) == 5
    assert num_moe_layers(_args(moe_layer_freq=[1, 1, 1, 0], mtp_num_layers=1)) == 3
    assert num_moe_layers(_args(num_experts=None)) == 0
    assert num_moe_layers(_args(moe_layer_freq="([1]*3+[0])")) == 3


@pytest.mark.parametrize("s, w", [(4, 2), (7, 3), (10, None), (5, 9), (6, 0), (6, 6), (1, 1)])
def test_causal_pairs_and_mean_keys_brute_force(s, w):
    brute = sum(min(i + 1, w or s) for i in range(s))
    assert causal_visible_pairs(s, w) == brute
    assert causal_mean_keys(s, w) == brute / s


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
    calls = []
    real = mod.muse_spark_flops_delta_per_sample
    monkeypatch.setattr(mod, "muse_spark_flops_delta_per_sample", lambda a: calls.append(1) or real(a))
    wrapped = mod.make_muse_spark_num_floating_point_operations(lambda a, b: 0.0)
    assert wrapped(a, 1) == one
    assert wrapped(a, 3) == three  # only batch_size changes per call
    assert len(calls) == 1


@pytest.mark.parametrize(
    "freq, expected",
    [(None, 4), (1, 4), (2, 2), ([1, 0, 1, 1], 3), ("([1]*3+[0])", 3)],
)
def test_num_moe_layers_uses_the_shared_pattern(freq, expected):
    from primus.backends.megatron.patches import muse_spark_flops_patches as mod

    args = SimpleNamespace(num_experts=8, num_layers=4, moe_layer_freq=freq, mtp_num_layers=None)
    assert mod.num_moe_layers(args) == expected
    with pytest.raises(ValueError, match="expected num_layers=4"):
        mod.num_moe_layers(SimpleNamespace(num_experts=8, num_layers=4, moe_layer_freq=[1, 0]))
