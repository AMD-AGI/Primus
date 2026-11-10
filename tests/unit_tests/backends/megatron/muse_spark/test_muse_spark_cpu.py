###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark: CPU checks (no GPU, no Megatron model build).

* the layer-pattern parser (no eval, length check, cut for depth-reduced runs);
* derive_megatron_fields against the released config.json values;
* vocab padding of the converter = Megatron's training padding;
* the HF -> mcore weight mapping on a tiny random checkpoint (gated qkv packing,
  grouped expert keys, every text tensor consumed, the vision and audio towers skipped);
* the norms (cast before ``1 + w``) and the residual gate against plain formulas.
"""

import json
import math
import os
from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.checkpoint import native_loader_muse_spark as conv
from primus.backends.megatron.checkpoint import (
    native_loader_muse_spark_text as conv_text,
)
from primus.backends.megatron.core.models.muse_spark.layer_patterns import (
    normalize_layer_pattern,
    parse_layer_pattern,
)
from primus.backends.megatron.core.transformer.muse_spark.norms import (
    MuseSparkCenteredRMSNorm,
    MuseSparkInputScaledRMSNorm,
    residual_gate_coeffs,
)

HF_PREFIX = conv.HF_PREFIXES[0]  # the released checkpoint's text prefix (model.language_model.)

# text_config of meta-models/Muse-Spark-Open-816B-A42B (the values the port depends on)
LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention", "sliding_attention"] * 15 + [
    "sliding_attention",
    "full_attention",
]
TEXT = {
    "model_type": "muse_spark_text",
    "hidden_size": 8192,
    "num_attention_heads": 128,
    "num_key_value_heads": 16,
    "head_dim": 64,
    "num_hidden_layers": 62,
    "intermediate_size": 16384,
    "moe_intermediate_size": 4096,
    "moe_hidden_size": 4096,
    "num_local_experts": 256,
    "num_experts_per_tok": 8,
    "rms_norm_eps": 1e-5,
    "post_norm_eps": 1e-8,
    "final_logit_softcapping": 20.0,
    "output_multiplier": 0.17677669529663687,
    "qk_scale_factor": 5.4730064863875,
    "residual_gate_temperature": 0.3,
    "rope_theta": 500000.0,
    "sliding_window": 2048,
    "layer_types": LAYER_TYPES,
    "layer_rope_theta": [0.0 if t == "full_attention" else 500000.0 for t in LAYER_TYPES],
    "vocab_size": 202048,
    "tie_word_embeddings": False,
}


# ---------------------------------------------------------------- layer patterns
def test_pattern_parser_matches_released_layers():
    win = parse_layer_pattern("([1,0,1,1]*15+[1,0])")
    nope = parse_layer_pattern("([0,1,0,0]*15+[0,1])")
    assert len(win) == len(nope) == 62
    assert win == [1 if t == "sliding_attention" else 0 for t in LAYER_TYPES]
    assert nope == [1 - x for x in win]


def test_pattern_parser_has_no_eval():
    with pytest.raises(ValueError):
        parse_layer_pattern("__import__('os').getcwd()")
    with pytest.raises(ValueError):
        parse_layer_pattern("[1]*'a'")


def test_pattern_cut_and_length_check():
    assert normalize_layer_pattern("([1,0,1,1]*15+[1,0])", 4) == [1, 0, 1, 1]
    assert normalize_layer_pattern(4, 8) == 4
    # a cut is reported; an exact-length list is not
    notes = []
    assert normalize_layer_pattern([1, 0, 1] * 30, 4, "k", warn=notes.append) == [1, 0, 1, 1]
    assert len(notes) == 1 and "k" in notes[0] and "90 entries" in notes[0]
    assert (
        normalize_layer_pattern([1, 0, 1, 1], 4, "k", warn=notes.append) == [1, 0, 1, 1] and len(notes) == 1
    )
    # an int is bounded by num_layers; errors name the argument
    with pytest.raises(ValueError, match="window_attn_skip_freq.*num_layers=4"):
        normalize_layer_pattern(100, 4, "window_attn_skip_freq")
    with pytest.raises(ValueError, match="no_rope_freq: cannot parse"):
        normalize_layer_pattern("[1,", 4, "no_rope_freq")
    with pytest.raises(ValueError, match="no_rope_freq is an empty list"):
        normalize_layer_pattern("[1]*0", 4, "no_rope_freq")
    with pytest.raises(ValueError, match="fewer than num_layers"):
        normalize_layer_pattern("[1,0]", 4)
    with pytest.raises(ValueError, match="0 or 1"):
        normalize_layer_pattern("[1,2,1,1]", 4)
    # the whole pattern is validated before a depth-reduced cut
    with pytest.raises(ValueError, match="0 or 1"):
        normalize_layer_pattern("[1,0,1,1]*10+[2]", 4)
    with pytest.raises(ValueError, match="empty"):
        normalize_layer_pattern("[]", 4)
    for bad_layers in (None, 0, -1, True):
        with pytest.raises(ValueError, match="num_layers"):
            normalize_layer_pattern("([1,0,1,1]*15+[1,0])", bad_layers)


def test_every_layer_must_be_moe():
    from primus.backends.megatron.core.models.muse_spark.layer_patterns import (
        check_all_moe_layers,
    )

    for ok in (1, None, [1, 1, 1, 1], "[1]*4", "([1]*6)"):
        check_all_moe_layers(ok, 4)
    for bad in (2, [1, 0, 1, 1], "([1]*3+[0])"):
        with pytest.raises(ValueError, match="every layer"):
            check_all_moe_layers(bad, 4)


@pytest.mark.parametrize(
    "bad",
    [
        True,
        "-1",
        0,
        "[1,",
        "[[1,0]]",
        "[1]*30000000",
        "[1]*-3",
        5.0,
        [1, True],
        "-[1]",
        "(" * 400 + "1" + ")" * 400,
        "-" * 5000 + "1",
        "[" + "1," * 5000 + "1]",
        "[" + "1, " * 6000 + "1]",
    ],
)
def test_pattern_parser_rejects_malformed_input(bad):
    with pytest.raises(ValueError):
        normalize_layer_pattern(bad, 4)


# ---------------------------------------------------------------- config translation
def test_derive_fields_from_release_config():
    f = conv.derive_megatron_fields({"text_config": TEXT})
    assert f["window_size"] == (2048, 0)  # HF count; the transformer config converts to TE
    assert f["window_attn_skip_freq"] == parse_layer_pattern("([1,0,1,1]*15+[1,0])")
    assert f["no_rope_freq"] == parse_layer_pattern("([0,1,0,0]*15+[0,1])")
    assert f["moe_latent_size"] == 4096 and f["num_experts"] == 256 and f["kv_channels"] == 64
    assert f["rotary_base"] == 500000.0 and f["num_query_groups"] == 16 and f["norm_epsilon"] == 1e-5
    # one HF output_multiplier feeds both the router and the LM head
    assert f["moe_router_output_multiplier"] == f["output_logit_multiplier"] == TEXT["output_multiplier"]


def test_derive_fields_read_a_non_default_output_multiplier():
    # the release value equals the 1/sqrt(32) fallback: a different one shows the key is read
    f = conv.derive_megatron_fields({"text_config": {**TEXT, "output_multiplier": 0.5}})
    assert f["moe_router_output_multiplier"] == f["output_logit_multiplier"] == 0.5


def test_derive_fields_keep_moe_sizes_apart():
    # the release has moe_intermediate_size == moe_hidden_size; distinct values catch a swap
    f = conv.derive_megatron_fields({**TEXT, "moe_intermediate_size": 1536, "moe_hidden_size": 2048})
    assert f["moe_ffn_hidden_size"] == 1536 and f["moe_latent_size"] == 2048
    assert f["ffn_hidden_size"] == 16384


def test_derive_fields_defaults_match_transformer_config():
    pytest.importorskip("megatron.core")
    import dataclasses

    from primus.backends.megatron.core.models.muse_spark.muse_spark_transformer_config import (
        MuseSparkTransformerConfig,
    )

    defaults = {fl.name: fl.default for fl in dataclasses.fields(MuseSparkTransformerConfig)}
    dropped = (
        "qk_scale_factor",
        "output_multiplier",
        "final_logit_softcapping",
        "post_norm_eps",
        "residual_gate_temperature",
    )
    f = conv.derive_megatron_fields({k: v for k, v in TEXT.items() if k not in dropped})
    for name in (
        "qk_scale_factor",
        "moe_router_output_multiplier",
        "output_logit_multiplier",
        "final_logit_softcapping",
        "post_norm_eps",
        "residual_gate_temperature",
    ):
        assert f[name] == defaults[name], name


@pytest.mark.parametrize("window", [None, 0, -1, True, "2048", "missing"])
def test_derive_fields_reject_sliding_layers_without_a_window(window):
    # sliding_attention layers with no positive window would silently run full attention
    t = {**TEXT, "sliding_window": window}
    if window == "missing":
        del t["sliding_window"]
    with pytest.raises(ValueError, match="sliding_window must be a positive integer"):
        conv.derive_megatron_fields(t)


def test_derive_fields_fallbacks():
    t = {
        k: v
        for k, v in TEXT.items()
        if k not in ("layer_types", "layer_rope_theta", "head_dim", "sliding_window")
    }
    f = conv.derive_megatron_fields(t)
    assert f["window_attn_skip_freq"] == [0] * 62 and f["no_rope_freq"] == [0] * 62
    assert f["window_size"] is None and f["kv_channels"] == 8192 // 128
    t = {**t, "rope_theta": None, "rope_parameters": {"rope_type": "default", "rope_theta": 1e6}}
    assert conv.derive_megatron_fields(t)["rotary_base"] == 1e6


@pytest.mark.parametrize(
    "change, match",
    [
        ({"rope_parameters": {"rope_type": "yarn", "rope_theta": 500000.0, "factor": 4.0}}, "default RoPE"),
        ({"rope_scaling": {"type": "linear", "factor": 2.0}}, "default RoPE"),
        ({"layer_rope_theta": [0.0, 1e6] * 31}, "other than 0"),
        ({"layer_rope_theta": [0.0, 500000.0]}, "entries for 62 layers"),
    ],
)
def test_derive_fields_rejects_unsupported_rope(change, match):
    with pytest.raises(ValueError, match=match):
        conv.derive_megatron_fields({**TEXT, **change})


def _transformer_config(**kw):
    pytest.importorskip("megatron.core")
    from primus.backends.megatron.core.models.muse_spark.muse_spark_transformer_config import (
        MuseSparkTransformerConfig,
    )

    base = dict(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        window_size=(2048, 0),
        window_attn_skip_freq="([1,0,1,1]*15+[1,0])",
        no_rope_freq="([0,1,0,0]*15+[0,1])",
    )
    return MuseSparkTransformerConfig(**{**base, **kw})


@pytest.mark.parametrize("raw, want", [("false", False), ("0", False), ("true", True), (" Yes ", True)])
def test_the_config_parses_string_booleans_strictly(raw, want):
    # env-driven yaml gives strings: "false" must not stay a truthy string
    cfg = _transformer_config(embedding_norm=raw)
    assert cfg.embedding_norm is want
    te = _transformer_config(window_size=(128, 0), window_size_is_hf_sliding_window=raw)
    # HF convention (True): 128 keys -> TE left window 127; TE convention (False): kept
    assert te.window_size == ((127, 0) if want else (128, 0))
    assert te.window_size_is_hf_sliding_window is False


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -5.0, True])
def test_the_config_refuses_a_bad_softcap(bad):
    # configs built directly (converter, parity harness) get the build_args patch's check
    with pytest.raises(ValueError, match="final_logit_softcapping"):
        _transformer_config(final_logit_softcapping=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -0.3, -1.0, True])
def test_the_config_refuses_a_bad_residual_gate_temperature(bad):
    with pytest.raises(ValueError, match="residual_gate_temperature must be a finite number > 0"):
        _transformer_config(residual_gate_temperature=bad)


def test_the_config_normalizes_the_softcap():
    assert _transformer_config(final_logit_softcapping="20").final_logit_softcapping == 20.0
    assert _transformer_config(final_logit_softcapping=None).final_logit_softcapping is None


@pytest.mark.parametrize("name", ["embedding_norm", "window_size_is_hf_sliding_window"])
def test_the_config_refuses_a_boolean_typo(name):
    with pytest.raises(ValueError, match=name):
        _transformer_config(**{name: "ture"})


def test_hf_sliding_window_converts_to_te_once():
    import copy
    import dataclasses

    cfg = _transformer_config()
    assert cfg.window_size == (2047, 0) and cfg.window_size_is_hf_sliding_window is False
    assert cfg.window_attn_skip_freq == [1, 0, 1, 1] and cfg.no_rope_freq == [0, 1, 0, 0]
    assert dataclasses.replace(cfg).window_size == (2047, 0)
    assert copy.deepcopy(cfg).window_size == (2047, 0)
    assert _transformer_config(window_size=(2047, 0), window_size_is_hf_sliding_window=False).window_size == (
        2047,
        0,
    )
    assert _transformer_config(window_size="(2048, 0)").window_size == (2047, 0)
    with pytest.raises(ValueError, match="causal"):
        _transformer_config(window_size=(2048, 5))


def test_softmax_scale_folds_qk_scale_factor():
    import dataclasses

    cfg = _transformer_config()
    assert cfg.softmax_scale == pytest.approx(cfg.qk_scale_factor / math.sqrt(16))
    # a copy with a new qk_scale_factor / kv_channels re-derives the scale (no stale value)
    assert dataclasses.replace(cfg, qk_scale_factor=1.0).softmax_scale == pytest.approx(0.25)
    assert dataclasses.replace(cfg, kv_channels=64).softmax_scale == pytest.approx(cfg.qk_scale_factor / 8)
    assert dataclasses.replace(cfg).softmax_scale == cfg.softmax_scale
    # an explicit softmax_scale must agree with qk_scale_factor
    assert _transformer_config(softmax_scale=cfg.softmax_scale).softmax_scale == cfg.softmax_scale
    with pytest.raises(ValueError, match="qk_scale_factor"):
        _transformer_config(softmax_scale=0.125)


def test_config_refuses_shared_experts():
    # the layer spec builds no shared expert: a set size must raise, not be dropped
    assert _transformer_config().moe_shared_expert_intermediate_size is None
    with pytest.raises(ValueError, match="no shared expert"):
        _transformer_config(moe_shared_expert_intermediate_size=128)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"num_moe_experts": 8, "moe_latent_size": None}, "needs moe_latent_size"),
        ({"expert_tensor_parallel_size": 2}, "expert_tensor_parallel_size=1"),
        ({"mtp_num_layers": 1}, "no MTP layers"),
    ],
)
def test_config_refuses_settings_the_model_does_not_build(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _transformer_config(**kwargs)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_converter_padding_does_not_depend_on_tp(tp):
    # the rows depend only on the config: one TP1 checkpoint loads at every TP
    padded = conv.vocab_size_with_padding(202048, conv.MAKE_VOCAB_SIZE_DIVISIBLE_BY, tp)
    assert padded == 202048 == math.ceil(202048 / 64) * 64
    assert conv.vocab_size_with_padding(202049, 64, tp) == 202112


def test_padding_rejects_rows_that_do_not_divide_by_tp():
    with pytest.raises(ValueError, match="does not divide by tensor_model_parallel_size=3") as err:
        conv.vocab_size_with_padding(202048, 64, 3)
    # the multiple is fixed (args.make_vocab_size_divisible_by is not read): the fix named is the TP
    assert "choose a TP that divides 202048" in str(err.value)
    assert "make_vocab_size_divisible_by" not in str(err.value)
    with pytest.raises(ValueError, match="divisible_by=0 must be an integer >= 1"):
        conv.vocab_size_with_padding(202048, 0, 1)


@pytest.mark.parametrize(
    "vocab, tp, name",
    [
        (0, 1, "vocab_size"),
        (-5, 1, "vocab_size"),
        (202047.9, 1, "vocab_size"),
        (None, 1, "vocab_size"),
        ("abc", 1, "vocab_size"),
        (True, 1, "vocab_size"),
        (202048, 0, "tensor_model_parallel_size"),
        (202048, -2, "tensor_model_parallel_size"),
        (202048, 1.5, "tensor_model_parallel_size"),
    ],
)
def test_padding_rejects_invalid_inputs(vocab, tp, name):
    with pytest.raises(ValueError, match=f"{name}=.* must be an integer >= 1"):
        conv.vocab_size_with_padding(vocab, 64, tp)


def test_padding_takes_an_integral_float_and_no_tp():
    assert conv.vocab_size_with_padding(202048.0, 64, None) == 202048
    # the shared config-number parser: a numeric string is a number, a blank TP is 1
    assert conv.vocab_size_with_padding("202_048", 64, "") == 202048


def test_training_patch_and_converter_share_one_padding_helper():
    from primus.backends.megatron.core.models.muse_spark import layer_patterns as lp
    from primus.backends.megatron.patches.args import (
        muse_spark_layer_patterns_patches as p,
    )

    assert conv.vocab_size_with_padding is lp.vocab_size_with_padding is p.vocab_size_with_padding
    assert conv.MAKE_VOCAB_SIZE_DIVISIBLE_BY == lp.MAKE_VOCAB_SIZE_DIVISIBLE_BY == 64
    # no make_vocab_size_divisible_by in the args: the shared default, not Megatron's 128
    args = SimpleNamespace(vocab_size=202049, tensor_model_parallel_size=1)
    assert p.padded_vocab_of(args) == 202112
    # the module config's 128 (every experiment has it) is not read: the model rule is 64,
    # so no experiment has to repeat make_vocab_size_divisible_by to match the converter
    # 202048 is padded at the converter's 64, not at make_vocab_size_divisible_by=128
    args = SimpleNamespace(vocab_size=202048, tensor_model_parallel_size=1, make_vocab_size_divisible_by=128)
    assert p.padded_vocab_of(args) == 202048


@pytest.mark.parametrize(
    "vocab, padded", [(202048, 202048), (201856, 201856), (201857, 201920), (202049, 202112)]
)
def test_build_margs_pads_the_checkpoint_vocab_to_64_whatever_the_args_say(monkeypatch, vocab, padded):
    # the converter pads to the model's 64 (as the muse_spark_vocab training patch), never
    # to the args' make_vocab_size_divisible_by (the module config's 128): other rows would
    # not load into training
    arguments = pytest.importorskip("megatron.training.arguments")
    monkeypatch.setattr(arguments, "validate_args", lambda m, *a, **k: m)
    args = SimpleNamespace(
        hf_dir="hf",
        save_dir="out",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        dtype="bf16",
        seq_length=128,
        make_vocab_size_divisible_by=128,
        moe_grouped_gemm=True,
    )
    cfg = {"text_config": dict(TEXT, vocab_size=vocab)}
    fields = conv.derive_megatron_fields(cfg)
    margs = conv.build_margs(fields, args, cfg)
    assert (margs.vocab_size, margs.make_vocab_size_divisible_by, margs.padded_vocab_size) == (
        vocab,
        64,
        padded,
    )


def test_build_margs_reads_the_checkpoint_vocab_without_a_passed_config(tmp_path, monkeypatch):
    # convert() passes the loaded config; the two-argument build_margs form (the
    # logprob-parity harness) reads <hf_dir>/config.json, so neither falls back to the
    # tokenizer's vocab (201856 rows instead of the checkpoint's 202048)
    arguments = pytest.importorskip("megatron.training.arguments")
    monkeypatch.setattr(arguments, "validate_args", lambda m, *a, **k: m)
    cfg = {"text_config": dict(TEXT)}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    args = SimpleNamespace(
        hf_dir=str(tmp_path),
        save_dir="out",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        dtype="bf16",
        seq_length=128,
        make_vocab_size_divisible_by=128,
        moe_grouped_gemm=True,
    )
    margs = conv.build_margs(conv.derive_megatron_fields(cfg), args)  # no hf_cfg passed
    assert margs.vocab_size == margs.padded_vocab_size == TEXT["vocab_size"] == 202048
    assert conv._hf_cfg_of(SimpleNamespace()) is None


@pytest.mark.parametrize(
    "hf_cfg, hf_dir, match",
    [
        ({"text_config": {"hidden_size": 8}}, None, "the HF config has no vocab_size"),
        ({}, None, "the HF config has no vocab_size"),
        (None, None, r"no HF config \(args.hf_dir is unset\) has no vocab_size"),
    ],
)
def test_build_margs_requires_the_checkpoint_vocab(hf_cfg, hf_dir, match):
    # no silent tokenizer-sized head: a missing vocab_size is an error before any Megatron import
    args = SimpleNamespace(
        tensor_parallel_size=1, pipeline_parallel_size=1, expert_parallel_size=1, dtype="bf16", hf_dir=hf_dir
    )
    with pytest.raises(ValueError, match=match):
        conv.build_margs({}, args, hf_cfg)


# ---------------------------------------------------------------- depth cut
def test_cut_to_layers_cuts_every_per_layer_list():
    cfg = {"text_config": dict(TEXT), "architectures": ["MuseSparkForConditionalGeneration"]}
    cut = conv.cut_to_layers(cfg, 4)
    t = cut["text_config"]
    assert t["num_hidden_layers"] == 4
    assert t["layer_types"] == LAYER_TYPES[:4] and t["layer_rope_theta"] == TEXT["layer_rope_theta"][:4]
    assert t["vocab_size"] == TEXT["vocab_size"] and cut["architectures"] == cfg["architectures"]
    assert cfg["text_config"]["num_hidden_layers"] == 62  # the input is not changed
    f = conv.derive_megatron_fields(cut)
    assert f["num_layers"] == 4
    assert 55e9 < conv.estimate_num_params(f, 202048) < 57e9  # the proxy
    assert conv.cut_to_layers(cfg, None) is cfg and conv.cut_to_layers(cfg, 62) is cfg
    assert conv.cut_to_layers(TEXT, 2)["num_hidden_layers"] == 2  # a text-only config
    for bad in (0, 63):
        with pytest.raises(ValueError, match="the checkpoint has 62 layers"):
            conv.cut_to_layers(cfg, bad)


def test_the_depth_cut_maps_only_the_first_layers():
    t, margs, fields = _tiny_checkpoint(L=2)
    store = conv.MappedStore(_Store(t), log=lambda *a: None, num_layers=1)
    assert store.dropped and all(k.startswith(HF_PREFIX + "layers.1.") for k in store.dropped)
    assert not any(".layers.1." in k for k in store.all_keys())
    assert "model.vision_tower.patch_embed.weight" in store.skipped  # still reported as out of scope
    margs.num_layers = 1
    sd, consumed = conv.build_mcore_state_dict(_GroupedModel(16, 1), store, margs, fields, lambda *a: None)
    assert consumed == store.all_keys()  # the exact-mapping check holds for the cut
    full = conv.MappedStore(_Store(t), log=lambda *a: None)
    assert not full.dropped and full.all_keys() > store.all_keys()


@pytest.mark.parametrize(
    "key, value, match",
    [
        ("layer_types", ["sliding_attention"] * 6, "layer_types has 6 entries for 4 layers"),
        ("layer_types", ["sliding_attention"] * 2, "layer_types has 2 entries for 4 layers"),
        (
            "layer_types",
            ["sliding_attention", "chunked_attention"] * 2,
            "unsupported values \\['chunked_attention'\\]",
        ),
        ("mlp_layer_types", ["sparse"], "mlp_layer_types has 1 entries for 4 layers"),
    ],
)
def test_per_layer_lists_must_match_the_layers(key, value, match):
    cfg = {**TEXT, "num_hidden_layers": 4, "layer_rope_theta": [500000.0, 0, 500000.0, 500000.0]}
    cfg.pop("layer_types", None), cfg.pop("mlp_layer_types", None)
    cfg[key] = value
    with pytest.raises(ValueError, match=match):
        conv.derive_megatron_fields(cfg)


def test_kept_by_depth_cut():
    P = HF_PREFIX
    assert conv.kept_by_depth_cut(P + "layers.3.mlp.gate.weight", P, 4)
    assert not conv.kept_by_depth_cut(P + "layers.4.mlp.gate.weight", P, 4)
    assert not conv.kept_by_depth_cut(P + "layers.12.mlp.gate.weight", P, 4)  # 12, not 1
    assert conv.kept_by_depth_cut(P + "layers.12.mlp.gate.weight", P, None)
    for k in (P + "embed_tokens.weight", P + "norm.weight", "lm_head.weight", "model.audio_tower.layers.9.w"):
        assert conv.kept_by_depth_cut(k, P, 1)


def test_converter_rejects_sharded_layouts():
    args = SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1, expert_parallel_size=1)
    with pytest.raises(ValueError, match="builds with TP=PP=EP=1 only") as err:
        conv.build_margs({}, args)
    # the torch_dist checkpoint is resharded by Megatron on load, so 1/1/1 is all a user needs
    assert "Megatron reshards on load" in str(err.value) and "convert with 1/1/1" in str(err.value)


@pytest.mark.parametrize("dtype", ["bfloat16", "fp8", "FP32", "", None])
def test_converter_rejects_unknown_dtypes(dtype):
    # only bf16 / fp16 / fp32: anything else is an error, not a silent fp32 checkpoint
    args = SimpleNamespace(
        tensor_parallel_size=1, pipeline_parallel_size=1, expert_parallel_size=1, dtype=dtype
    )
    with pytest.raises(ValueError, match="is not supported .*bf16, fp16, fp32"):
        conv.build_margs({}, args)
    assert conv.SUPPORTED_DTYPES == ("bf16", "fp16", "fp32")


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"mlp_layer_types": ["dense"] * TEXT["num_hidden_layers"]}, "dense Muse Spark layers"),
        ({"num_shared_experts": 1}, "shared experts"),
    ],
)
def test_derive_fields_rejects_unsupported_layers_with_value_error(extra, match):
    # ValueError, not assert: the check stays under python -O
    with pytest.raises(ValueError, match=match):
        conv.derive_megatron_fields({"text_config": dict(TEXT, **extra)})


def test_checkpoint_vocab_reads_the_loaded_config():
    # the same text_config lookup as check_convert_size (no second read of config.json)
    assert conv.checkpoint_vocab({"text_config": {"vocab_size": 123}}) == 123
    assert conv.checkpoint_vocab({"vocab_size": 77}) == 77  # text-only config
    assert conv.checkpoint_vocab({"text_config": {}}) is None
    # a composite config with vocab_size at the top level only: the top level, as the
    # build_margs error says (text_config.vocab_size or the top level)
    assert conv.checkpoint_vocab({"text_config": {"hidden_size": 8}, "vocab_size": 55}) == 55
    assert conv.checkpoint_vocab({"text_config": {"vocab_size": 9}, "vocab_size": 55}) == 9
    assert conv.checkpoint_vocab(None) is None


# ---------------------------------------------------------------- weight mapping
class _Store:
    def __init__(self, tensors):
        self.t = tensors

    def get(self, name, dtype):
        return self.t[name].to(dtype)

    def all_keys(self):
        return set(self.t)


def _tiny_checkpoint(L=2, E=3, nh=4, ng=2, hd=2, hidden=8, latent=4, inter=6, vocab=10):
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    P = HF_PREFIX
    t = {
        P + "embed_tokens.weight": r(vocab, hidden),
        P + "norm.weight": r(hidden),
        "lm_head.weight": r(vocab, hidden),
    }
    for i in range(L):
        h = f"{P}layers.{i}."
        t.update(
            {
                h + "input_layernorm.weight": r(hidden),
                h + "self_attn.q_proj.weight": r(nh * hd, hidden),
                h + "self_attn.gate_proj.weight": r(nh * hd, hidden),
                h + "self_attn.k_proj.weight": r(ng * hd, hidden),
                h + "self_attn.v_proj.weight": r(ng * hd, hidden),
                h + "self_attn.o_proj.weight": r(hidden, nh * hd),
                h + "post_attention_residual_gate.gate": r(hidden),
                h + "pre_feedforward_layernorm.weight": r(hidden),
                h + "mlp.gate.weight": r(E, hidden),
                h + "mlp.gate.e_score_correction_bias": r(E),
                h + "mlp.pre_expert_proj.weight": r(latent, hidden),
                h + "mlp.pre_expert_norm.weight": r(latent),
                h + "mlp.experts.gate_up_proj": r(E, 2 * inter, latent),
                h + "mlp.experts.down_proj": r(E, latent, inter),
                h + "mlp.experts.post_expert_norm.weight": r(latent),
                h + "mlp.post_expert_proj.weight": r(hidden, latent),
                h + "post_feedforward_layernorm.weight": r(hidden),
                h + "post_feedforward_residual_gate.gate": r(hidden),
            }
        )
    t["model.vision_tower.patch_embed.weight"] = r(3, 3)
    t["model.vision_adapter.fc1.weight"] = r(3, 3)  # the released names (weights.txt)
    t["model.vision_projection.weight"] = r(3, 3)
    # the released audio tower names (skipped by the converter)
    t["model.audio_tower.proj.weight"] = r(3, 3)
    t["model.audio_tower.layers.0.norm.weight"] = r(3)
    margs = SimpleNamespace(
        params_dtype=torch.float32,
        num_attention_heads=nh,
        num_query_groups=ng,
        kv_channels=hd,
        hidden_size=hidden,
        num_layers=L,
        untie_embeddings_and_output_weights=True,
    )
    return t, margs, {"num_experts": E}


class _GroupedModel:
    def __init__(self, padded_vocab, L, expert_key="mlp.experts.linear_fc1.weight0"):
        self.keys = {"embedding.word_embeddings.weight": torch.empty(padded_vocab, 8)}
        self.embedding = SimpleNamespace(
            word_embeddings=SimpleNamespace(weight=self.keys["embedding.word_embeddings.weight"])
        )
        for i in range(L):
            if expert_key:
                self.keys[f"decoder.layers.{i}.{expert_key}"] = torch.empty(1)

    def state_dict(self):
        return self.keys


def test_mapping_keeps_router_and_gates_fp32_in_bf16():
    t, margs, fields = _tiny_checkpoint(L=1)
    margs.params_dtype = torch.bfloat16
    sd, _ = conv.build_mcore_state_dict(
        _GroupedModel(16, 1),
        conv.MappedStore(_Store(t), log=lambda *a: None),
        margs,
        fields,
        lambda *a: None,
    )
    fp32 = {k for k, v in sd.items() if v.dtype == torch.float32}
    assert fp32 == {
        "decoder.layers.0.self_attn_bda.gate",
        "decoder.layers.0.mlp_bda.gate",
        "decoder.layers.0.mlp.router.weight",
        "decoder.layers.0.mlp.router.expert_bias",
    }


def test_mapping_round_trip_grouped_experts():
    t, margs, fields = _tiny_checkpoint()
    store = conv.MappedStore(_Store(t), log=lambda *a: None)
    assert store.skipped == [
        "model.audio_tower.layers.0.norm.weight",
        "model.audio_tower.proj.weight",
        "model.vision_adapter.fc1.weight",
        "model.vision_projection.weight",
        "model.vision_tower.patch_embed.weight",
    ]
    sd, consumed = conv.build_mcore_state_dict(_GroupedModel(16, 2), store, margs, fields, lambda *a: None)
    assert consumed == store.all_keys()  # every text tensor is mapped; vision / audio skipped
    assert not any("audio" in k or "vision" in k for k in sd)

    P = HF_PREFIX + "layers.1."
    # gated qkv, per KV group: [q heads of the group; gate heads; k; v]
    qkv = sd["decoder.layers.1.self_attention.linear_qkv.weight"].view(2, -1, 8)
    q, gte, k, v = qkv.split([4, 4, 2, 2], dim=1)
    torch.testing.assert_close(q.reshape(-1, 8), t[P + "self_attn.q_proj.weight"])
    torch.testing.assert_close(gte.reshape(-1, 8), t[P + "self_attn.gate_proj.weight"])
    torch.testing.assert_close(k.reshape(-1, 8), t[P + "self_attn.k_proj.weight"])
    torch.testing.assert_close(v.reshape(-1, 8), t[P + "self_attn.v_proj.weight"])
    # grouped expert keys (the training layout), [gate; up] order kept
    for e in range(3):
        torch.testing.assert_close(
            sd[f"decoder.layers.1.mlp.experts.linear_fc1.weight{e}"], t[P + "mlp.experts.gate_up_proj"][e]
        )
        torch.testing.assert_close(
            sd[f"decoder.layers.1.mlp.experts.linear_fc2.weight{e}"], t[P + "mlp.experts.down_proj"][e]
        )
    assert not any("local_experts" in key for key in sd)
    # every other key: the HF tensor it comes from, and fp32 kept for router / gates
    one_to_one = {
        "input_layernorm.weight": "input_layernorm.weight",
        "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
        "self_attn_bda.gate": "post_attention_residual_gate.gate",
        "pre_mlp_layernorm.weight": "pre_feedforward_layernorm.weight",
        "mlp.router.weight": "mlp.gate.weight",
        "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
        "mlp.fc1_latent_proj.weight": "mlp.pre_expert_proj.weight",
        "mlp.pre_expert_norm.weight": "mlp.pre_expert_norm.weight",
        "mlp.post_expert_norm.weight": "mlp.experts.post_expert_norm.weight",
        "mlp.fc2_latent_proj.weight": "mlp.post_expert_proj.weight",
        "mlp_bda.post_norm.weight": "post_feedforward_layernorm.weight",
        "mlp_bda.gate": "post_feedforward_residual_gate.gate",
    }
    for i in range(2):
        for dst, src in one_to_one.items():
            torch.testing.assert_close(
                sd[f"decoder.layers.{i}.{dst}"],
                t[f"{HF_PREFIX}layers.{i}.{src}"].float(),
                check_dtype=False,
            )
    torch.testing.assert_close(sd["decoder.final_layernorm.weight"], t[HF_PREFIX + "norm.weight"])
    assert torch.equal(sd["output_layer.weight"][:10], t["lm_head.weight"])
    per_layer = len(one_to_one) + 1 + 2 * 3  # + qkv + fc1 / fc2 per expert
    assert len(sd) == 2 * per_layer + 3
    # vocab rows padded to the model's padded size; common.pad_vocab repeats the last
    # real row (as Megatron's converters do), the real rows are unchanged
    emb = sd["embedding.word_embeddings.weight"]
    assert emb.shape[0] == 16
    assert torch.equal(emb[:10], t["model.language_model.embed_tokens.weight"].to(emb.dtype))
    assert torch.all(emb[10:] == emb[9])


def test_mapping_sequential_experts_uses_local_expert_keys():
    # moe_grouped_gemm unset: SequentialMLP, one MLP per expert
    t, margs, fields = _tiny_checkpoint()
    store = conv.MappedStore(_Store(t), log=lambda *a: None)
    model = _GroupedModel(16, 2, expert_key="mlp.experts.local_experts.0.linear_fc1.weight")
    sd, consumed = conv.build_mcore_state_dict(model, store, margs, fields, lambda *a: None)
    assert consumed == store.all_keys()
    assert not any(".linear_fc1.weight0" in k or ".linear_fc2.weight0" in k for k in sd)
    for i in range(2):
        P = f"{HF_PREFIX}layers.{i}."
        for e in range(3):
            torch.testing.assert_close(
                sd[f"decoder.layers.{i}.mlp.experts.local_experts.{e}.linear_fc1.weight"],
                t[P + "mlp.experts.gate_up_proj"][e],
            )
            torch.testing.assert_close(
                sd[f"decoder.layers.{i}.mlp.experts.local_experts.{e}.linear_fc2.weight"],
                t[P + "mlp.experts.down_proj"][e],
            )


def test_mapping_rejects_a_model_without_known_expert_keys():
    t, margs, fields = _tiny_checkpoint(L=1)
    store = conv.MappedStore(_Store(t), log=lambda *a: None)
    with pytest.raises(RuntimeError, match="expert key layout"):
        conv.build_mcore_state_dict(
            _GroupedModel(16, 1, expert_key=None), store, margs, fields, lambda *a: None
        )


@pytest.mark.parametrize("tied", [False, True])
def test_mapping_emits_the_output_layer_only_when_untied(tied):
    t, margs, fields = _tiny_checkpoint(L=1)
    margs.untie_embeddings_and_output_weights = not tied
    if tied:
        del t["lm_head.weight"]  # a tied HF checkpoint has no separate head
    store = conv.MappedStore(_Store(t), log=lambda *a: None)
    sd, consumed = conv.build_mcore_state_dict(_GroupedModel(16, 1), store, margs, fields, lambda *a: None)
    assert ("output_layer.weight" in sd) is (not tied)
    assert consumed == store.all_keys()
    cfg = {**TEXT, "tie_word_embeddings": tied}
    assert conv.derive_megatron_fields(cfg)["untie_embeddings_and_output_weights"] is (not tied)


@pytest.mark.parametrize(
    "extra", ["model.mtp.0.weight", "model.multi_modal_projector.weight", "other.prefix.weight"]
)
def test_an_unknown_tensor_is_not_set_aside_and_fails_the_mapping_check(extra):
    # only the known vision / audio modules are skipped: anything else stays in the
    # exact-mapping check, so the mapping leaves it unconsumed and the converter refuses
    t, margs, fields = _tiny_checkpoint()
    t[extra] = torch.zeros(1)
    logs = []
    store = conv.MappedStore(_Store(t), log=logs.append)
    assert extra in store.all_keys() and extra not in store.skipped
    _, consumed = conv.build_mcore_state_dict(_GroupedModel(16, 2), store, margs, fields, lambda *a: None)
    assert store.all_keys() - consumed == {extra}
    # the set-aside modules are logged with their counts
    assert any("1 tensors under model.vision_tower.*" in m for m in logs)
    assert any("2 tensors under model.audio_tower.*" in m for m in logs)


def test_the_converter_skips_the_audio_and_vision_towers(monkeypatch):
    # the converter maps the text decoder only: the audio / vision tower tensors are
    # skipped (logged), so the exact-mapping check holds without them
    logs = []
    monkeypatch.setattr(conv.common, "SafetensorsStore", lambda d: _Store(_tiny_checkpoint()[0]))
    store = conv.MuseSparkConverter(log=logs.append).open_store("hf")
    assert not any(k.startswith(("model.audio_tower.", "model.vision_")) for k in store.all_keys())
    assert "model.audio_tower.proj.weight" in store.skipped
    assert any("not ported, skipped" in m and "model.audio_tower.*" in m for m in logs)


@pytest.mark.parametrize("prefix", ["model.language_model.", "model."])
def test_hf_prefix_detected_from_checkpoint_keys(prefix):
    # conditional-generation wrapper and text-only MuseSparkForCausalLM layouts
    t, margs, fields = _tiny_checkpoint()
    t = {(prefix + k[len(HF_PREFIX) :] if k.startswith(HF_PREFIX) else k): v for k, v in t.items()}
    if prefix == "model.":
        t = {k: v for k, v in t.items() if "vision_" not in k and "audio_tower" not in k}
    assert conv.detect_hf_prefix(t) == prefix
    store = conv.MappedStore(_Store(t), log=lambda *a: None)
    assert store.prefix == prefix
    sd, consumed = conv.build_mcore_state_dict(_GroupedModel(16, 2), store, margs, fields, lambda *a: None)
    assert consumed == store.all_keys()
    torch.testing.assert_close(sd["decoder.final_layernorm.weight"], t[prefix + "norm.weight"])


def test_unknown_hf_prefix_is_rejected():
    keys = {"transformer.wte.embed_tokens.weight", "lm_head.weight"}
    with pytest.raises(ValueError, match="unsupported Muse Spark checkpoint layout.*transformer"):
        conv.detect_hf_prefix(keys)


# ---------------------------------------------------------------- norms and gate
def test_centered_norm_casts_weight_before_adding_one():
    norm = MuseSparkCenteredRMSNorm(hidden_size=4, eps=1e-5).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.fill_(1e-3)  # 1 + 1e-3 rounds to 1.0 in bf16
    x = torch.randn(2, 4)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * (1 + norm.weight.float())
    torch.testing.assert_close(norm(x), ref)
    assert not torch.allclose(norm(x), x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5))


def test_input_scaled_norm():
    norm = MuseSparkInputScaledRMSNorm(hidden_size=4, eps=1e-8)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, -0.5, 0.0, 1.0]))
    x = torch.randn(3, 4)
    y = x * (1 + norm.weight)
    torch.testing.assert_close(norm(x), y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8))


def test_residual_gate_formula():
    # reference values (gate / temperature 0.3 through the HF formula, computed by hand):
    # beta = sigmoid(w), alpha = sqrt(max(1 - beta**2, 1e-3)), i.e. alpha**2 + beta**2 = 1
    gate = torch.tensor([-1.0, 0.0, 2.0, 10.0])
    alpha, beta = residual_gate_coeffs(gate, 0.3)
    torch.testing.assert_close(beta, torch.tensor([0.03444519566621117, 0.5, 0.9987289837369187, 1.0]))
    torch.testing.assert_close(
        alpha,
        torch.tensor([0.9994065881789636, 0.8660254037844386, 0.05040254997340609, math.sqrt(1e-3)]),
    )
    # the variance-preserving property where the floor does not act; the floor at a large gate
    torch.testing.assert_close(alpha[:3] ** 2 + beta[:3] ** 2, torch.ones(3))


def test_text_entry_point_names_the_same_api():
    for name in conv_text.__all__:
        assert getattr(conv_text, name) is getattr(conv, name)


@pytest.fixture(autouse=True)
def _unknown_host_memory(monkeypatch):
    # the size check reads the host's memory; tests give it explicitly (see below)
    monkeypatch.setattr(conv, "host_memory_available", lambda: None)


def test_converter_checks_the_released_size_against_host_memory():
    fields = conv.derive_megatron_fields({"text_config": TEXT})
    n = conv.estimate_num_params(fields, TEXT["vocab_size"])
    assert 800e9 < n < 840e9  # ~816B (the model name)
    plan = conv.conversion_memory_plan(fields, TEXT["vocab_size"], "bf16")
    # one model copy plus one layer: the full-scale node (3022 GiB of host RAM) holds it
    assert plan["model"] == 2 * n and plan["layer"] < plan["model"] / 50
    assert 1.5 * 2**40 < plan["total"] < 1.7 * 2**40 < 3022 * 2**30
    assert conv.check_convert_size(fields, {"text_config": TEXT}, available=3022 * 2**30) == n
    with pytest.raises(ValueError, match="host memory") as err:
        conv.check_convert_size(fields, {"text_config": TEXT}, available=1024 * 2**30)
    assert "pretrained_checkpoint" in str(err.value)
    # fp32 needs twice the memory
    fp32 = conv.conversion_memory_plan(fields, TEXT["vocab_size"], "fp32")
    assert fp32["model"] == 2 * plan["model"]
    with pytest.raises(ValueError, match="in fp32"):
        conv.check_convert_size(fields, {"text_config": TEXT}, dtype="fp32", available=3022 * 2**30)
    # unknown host memory: no memory refusal; no cap by default
    assert conv.check_convert_size(fields, {"text_config": TEXT}) == n


def test_host_memory_available_reads_meminfo_and_the_cgroup(monkeypatch):
    files = {
        "/proc/meminfo": "MemTotal: 100 kB\nMemAvailable:   2048 kB\n",
        "/sys/fs/cgroup/memory.max": "max",
        "/sys/fs/cgroup/memory.current": "10",
    }
    monkeypatch.undo()  # the autouse fixture's stub
    monkeypatch.setattr(conv, "_read_text", lambda path, default="": files.get(path, default))
    assert conv.host_memory_available() == 2048 * 1024
    # a cgroup limit below MemAvailable wins; its inactive page cache counts as free
    files.update({"/sys/fs/cgroup/memory.max": "5000", "/sys/fs/cgroup/memory.stat": "inactive_file 100\n"})
    assert conv.host_memory_available() == 5000 - 10 + 100
    files.clear()
    assert conv.host_memory_available() is None


def test_fp32_tensor_check_on_the_mapped_state_dict():
    def sd(layers=2, gate_dtype=torch.float32):
        out = {}
        for i in range(layers):
            p = f"decoder.layers.{i}."
            out[p + "self_attn_bda.gate"] = torch.zeros(4, dtype=gate_dtype)
            out[p + "mlp_bda.gate"] = torch.zeros(4)
            out[p + "mlp.router.weight"] = torch.zeros(2, 4)
            out[p + "mlp.router.expert_bias"] = torch.zeros(2)
            out[p + "mlp.fc1_latent_proj.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
        return out

    conv.check_fp32_tensors(sd(), 2)
    with pytest.raises(RuntimeError, match="must stay fp32"):
        conv.check_fp32_tensors(sd(gate_dtype=torch.bfloat16), 2)
    with pytest.raises(RuntimeError, match="expected 12"):
        conv.check_fp32_tensors(sd(), 3)


def test_convert_checks_the_fp32_dtypes_on_the_built_model(monkeypatch, tmp_path):
    """convert() checks the built model's dtypes (what gets saved), not the mapped dict:
    a router weight that model.to(bf16) left in bf16 stops the conversion."""

    def model_sd(router_dtype):
        out = {}
        for i in range(2):
            p = f"decoder.layers.{i}."
            out[p + "self_attn_bda.gate"] = torch.zeros(4)
            out[p + "mlp_bda.gate"] = torch.zeros(4)
            out[p + "mlp.router.weight"] = torch.zeros(2, 4, dtype=router_dtype)
            out[p + "mlp.router.expert_bias"] = torch.zeros(2)
        return out

    fields = dict(conv.derive_megatron_fields({"text_config": TEXT}), num_layers=2)
    mapped = []
    monkeypatch.setattr(conv.common, "ensure_megatron_on_path", lambda: None)
    monkeypatch.setattr(conv.common, "read_hf_config", lambda _d: {"text_config": TEXT})
    monkeypatch.setattr(conv, "derive_megatron_fields", lambda _c: fields)
    monkeypatch.setattr(conv, "check_convert_size", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        conv, "build_margs", lambda _f, _a, _c=None: SimpleNamespace(num_layers=2, iteration=1)
    )
    monkeypatch.setattr(conv.common, "init_megatron_single_process", lambda m, *_a, **_k: m)
    monkeypatch.setattr(conv.common, "SafetensorsStore", lambda _d: mapped.append("store"))
    for dtype, ok in ((torch.float32, True), (torch.bfloat16, False)):
        mapped.clear()
        monkeypatch.setattr(
            conv, "build_model", lambda _m, d=dtype: SimpleNamespace(state_dict=lambda: model_sd(d))
        )
        if ok:

            def _stop(*_a, **_k):
                raise LookupError("mapped")

            monkeypatch.setattr(conv, "MappedStore", _stop)
            with pytest.raises(LookupError):
                conv.convert(str(tmp_path), str(tmp_path / "out"))
            assert mapped == ["store"]
        else:
            with pytest.raises(RuntimeError, match="mlp.router.weight must stay fp32"):
                conv.convert(str(tmp_path), str(tmp_path / "out"))
            assert mapped == []  # stopped before reading any HF tensor


def test_converter_size_cap_counts_the_hf_weights():
    # the expected count is written out by hand from the HF config (the released
    # MuseSparkText weights), not from memory_check.param_split, which the cap calls:
    # a wrong count there (or in derive_megatron_fields) fails this test
    t = TEXT
    h, L, v = t["hidden_size"], t["num_hidden_layers"], t["vocab_size"]
    q = t["num_attention_heads"] * t["head_dim"]
    kv = t["num_key_value_heads"] * t["head_dim"]
    attn = h * q + h * q + 2 * h * kv + q * h  # q_proj, the output gate, k/v_proj, o_proj
    latent = t["moe_hidden_size"]
    router = t["num_local_experts"] * h
    latent_proj = 2 * h * latent  # into and out of the latent width
    experts = t["num_local_experts"] * 3 * latent * t["moe_intermediate_size"]  # gate, up, down
    want = L * (attn + router + latent_proj + experts) + 2 * v * h  # untied embed + lm_head
    fields = conv.derive_megatron_fields({"text_config": TEXT})
    assert conv.estimate_num_params(fields, v) == want


def _patch_ctx(params, backend):
    from primus.core.patches import PatchContext

    return PatchContext(
        backend="megatron",
        phase="build_args",
        extra={"module_config": SimpleNamespace(params=params), "backend_args": backend},
    )


@pytest.mark.parametrize("which", ["deepep", "mfsdp"])
def test_turbo_patches_clear_the_flag_on_module_config_params(which, monkeypatch):
    """The Turbo patches read module_config.params (get_args): the flag must be cleared there."""
    from primus.backends.megatron.patches.args import muse_spark_arg_compat_patches as m

    # EP 8 fits one 8-GPU node: pin the node size (the runner's env must not decide the
    # result); muse_spark_turbo_deepep_internode is unset (false) in these args
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("GPUS_PER_NODE", "8")

    common = dict(enable_primus_turbo=True, tensor_model_parallel_size=1, expert_tensor_parallel_size=1)
    if which == "deepep":
        fn, flag, extra = (
            m.patch_muse_spark_deepep_group_size,
            "use_turbo_deepep",
            dict(expert_model_parallel_size=1),
        )
    else:
        fn, flag, extra = (
            m.patch_muse_spark_mfsdp_turbo_grouped_gemm,
            "use_turbo_grouped_gemm",
            dict(use_megatron_fsdp=True, expert_model_parallel_size=8),
        )
    params = SimpleNamespace(**common, **extra, **{flag: True})
    backend = SimpleNamespace(**common, **extra)  # no Turbo flags, like the real backend_args
    fn(_patch_ctx(params, backend))
    assert getattr(params, flag) is False
    assert not hasattr(backend, flag)
    # a layout that supports it keeps the flag. backend_args is built from module_config.params
    # (adapter.convert_config), so the two agree on the layout keys in a real run.
    layout2 = {**extra, "expert_model_parallel_size": 8, "use_megatron_fsdp": False}
    params2 = SimpleNamespace(**common, **layout2, **{flag: True})
    backend2 = SimpleNamespace(**common, **layout2)
    fn(_patch_ctx(params2, backend2))
    assert getattr(params2, flag) is True
    # backend_args wins on a key both carry: it holds the CLI overrides and the earlier patches
    params3 = SimpleNamespace(**common, **layout2, **{flag: True})
    fn(_patch_ctx(params3, backend))  # backend: the unsupported layout of the first call
    assert getattr(params3, flag) is False


@pytest.mark.parametrize(
    "gpus, ep, opt_in, kept",
    [
        ("8", 8, None, True),  # the expert group fits one node
        ("4", 8, None, False),  # spans nodes: intranode-only build
        ("4", 8, True, True),  # rocSHMEM build opted in
        ("4", 8, "TRUE", True),  # an env-driven yaml string
        ("4", 8, False, False),
        ("4", 8, "0", False),
        ("8", 16, "yes", True),
    ],
)
def test_deepep_spanning_nodes_needs_the_internode_opt_in(monkeypatch, gpus, ep, opt_in, kept):
    from primus.backends.megatron.patches.args import muse_spark_arg_compat_patches as m

    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("GPUS_PER_NODE", gpus)
    args = SimpleNamespace(
        muse_spark_turbo_deepep_internode=opt_in,
        use_turbo_deepep=True,
        enable_primus_turbo=True,
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=ep,
    )
    reason = m.deepep_disable_reason(args)
    if kept:
        assert reason is None
    else:
        assert "spans nodes" in reason and f"{gpus} GPUs per node" in reason and m.INTERNODE_ARG in reason
    # an explicit local_gpus wins over the environment
    assert m.deepep_disable_reason(args, local_gpus=ep) is None


@pytest.mark.parametrize("raw", [0, -1, True, 1.5])
def test_convert_max_params_rejects_bad_values(raw):
    with pytest.raises(ValueError, match="max_params"):
        conv.check_convert_size(conv.derive_megatron_fields({"text_config": TEXT}), max_params=raw)


def test_the_cap_takes_a_config_string():
    # the hook passes the yaml value of native_ckpt_max_params as it is (int or string)
    fields = conv.derive_megatron_fields({"text_config": TEXT})
    n = conv.estimate_num_params(fields)
    assert conv.check_convert_size(fields, max_params="820_000_000_000") == n
    assert conv.check_convert_size(fields, max_params="9e11") == n
    assert conv.check_convert_size(fields, max_params=" ") == n  # blank: no cap
    with pytest.raises(ValueError, match="above native_ckpt_max_params"):
        conv.check_convert_size(fields, max_params="60e9")
    for bad in ("0", "-5", "lots", "1.5"):
        with pytest.raises(ValueError, match=conv.MAX_PARAMS_KEY):
            conv.check_convert_size(fields, max_params=bad)


def test_convert_refuses_an_existing_output_before_the_build(tmp_path, monkeypatch):
    # a rerun into a converted directory must fail before the (hours-long) build
    (tmp_path / "release").mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")
    monkeypatch.setattr(conv.common, "ensure_megatron_on_path", lambda: pytest.fail("the build started"))
    with pytest.raises(ValueError, match="already exists"):
        conv.convert(str(tmp_path / "hf"), str(tmp_path))
    conv.refuse_existing_output(str(tmp_path / "empty"))  # no output: no error


def _write_in_progress_marker(save_dir, host=None, pid=None, boot=None, pidns=None, token="t0"):
    """A marker as claim_output writes it (host, pid, boot id, pid namespace, token); this
    process's identity unless overridden."""
    my_host, my_pid, my_boot, my_ns = conv._process_identity()
    text = f"{host or my_host} {pid or my_pid} {boot or my_boot} {pidns or my_ns} {token}"
    (save_dir / conv.IN_PROGRESS_MARKER).write_text(text + "\n")


def _dead_pid():
    import subprocess
    import sys

    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid  # reaped: no such process now


def test_only_a_killed_runs_own_leftover_is_removed(tmp_path):
    # a tracker-less release/ is removed only with the in-progress marker of a dead
    # process of this host (this converter's own killed run), and only by claim_output
    part = tmp_path / "release"
    (part / "sub").mkdir(parents=True)
    (part / "sub" / "x.distcp").write_text("partial")
    (tmp_path / "other.txt").write_text("kept")
    _write_in_progress_marker(tmp_path, pid=_dead_pid())
    conv.refuse_existing_output(str(tmp_path))  # accepted, nothing deleted yet
    assert part.exists()
    logs = []
    conv.claim_output(str(tmp_path), log=logs.append)
    assert not part.exists() and (tmp_path / "other.txt").exists()
    assert "killed conversion" in logs[0]
    assert (tmp_path / conv.IN_PROGRESS_MARKER).read_text().strip() == conv._marker_text()
    conv.release_output(str(tmp_path))
    assert not (tmp_path / conv.IN_PROGRESS_MARKER).exists()


def test_the_marker_heartbeat_runs_while_a_killed_runs_leftover_is_removed(tmp_path, monkeypatch):
    # a long removal on shared storage must keep the fresh marker live (its mtime moves)
    import threading
    import time

    (tmp_path / "release").mkdir()
    _write_in_progress_marker(tmp_path, pid=_dead_pid())
    monkeypatch.setattr(conv, "MARKER_HEARTBEAT_S", 0.05)
    marker = tmp_path / conv.IN_PROGRESS_MARKER
    seen = {}
    real_rmtree = conv.shutil.rmtree

    def slow_rmtree(path, *a, **kw):
        start = marker.stat().st_mtime_ns
        time.sleep(0.5)
        seen["moved"] = marker.stat().st_mtime_ns != start
        seen["beating"] = any("heartbeat" in t.name for t in threading.enumerate())
        return real_rmtree(path, *a, **kw)

    monkeypatch.setattr(conv.shutil, "rmtree", slow_rmtree)
    conv.claim_output(str(tmp_path), log=lambda *a: None)
    assert seen == {"moved": True, "beating": True}
    assert not (tmp_path / "release").exists()


def test_an_unreadable_pid_namespace_is_judged_by_the_heartbeat(tmp_path, monkeypatch):
    # pid namespace "-" (unreadable /proc/self/ns) on both sides is not "the same namespace":
    # a dead-looking pid of another container must not make a live conversion "dead"
    host, _, boot, _ = conv._process_identity()
    monkeypatch.setattr(conv, "_process_identity", lambda: (host, 1, boot, "-"))
    _write_in_progress_marker(tmp_path, pid=_dead_pid(), pidns="-")
    assert conv._marker_owner(str(tmp_path)) == "live"  # fresh heartbeat, no pid check


@pytest.mark.parametrize("pid", ["live_other", "notapid"])
def test_a_same_host_marker_with_a_live_or_bad_pid_is_dead_once_its_heartbeat_is_stale(tmp_path, pid):
    # a reused pid (an unrelated running process, here pid 1) or a corrupt pid field: live
    # while the heartbeat is fresh, dead once it has not moved for MARKER_STALE_S
    host, _, boot, pidns = conv._process_identity()
    if "-" in (boot, pidns):
        pytest.skip("boot id / pid namespace unreadable here: no pid check")
    text = f"{host} {1 if pid == 'live_other' else 'notapid'} {boot} {pidns} t0"
    (tmp_path / conv.IN_PROGRESS_MARKER).write_text(text + "\n")
    assert conv._marker_owner(str(tmp_path)) == "live"
    old = os.path.getmtime(tmp_path / conv.IN_PROGRESS_MARKER) - conv.MARKER_STALE_S - 1
    os.utime(tmp_path / conv.IN_PROGRESS_MARKER, (old, old))
    assert conv._marker_owner(str(tmp_path)) == "dead"


@pytest.mark.parametrize("marker", [None, "other-host", "garbage"])
def test_a_tracker_less_directory_not_known_as_ours_is_refused_and_kept(tmp_path, marker):
    part = tmp_path / "release"
    (part / "precious").mkdir(parents=True)
    if marker == "other-host":  # a fresh heartbeat: that conversion may still run
        _write_in_progress_marker(tmp_path, host="some-other-host", pid=_dead_pid())
    elif marker == "garbage":
        (tmp_path / conv.IN_PROGRESS_MARKER).write_text("nonsense")
    for fn in (conv.refuse_existing_output, conv.claim_output):
        with pytest.raises(ValueError):
            fn(str(tmp_path))
    assert (part / "precious").is_dir()


def test_a_live_conversion_into_the_same_directory_is_refused(tmp_path):
    import subprocess
    import sys

    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _write_in_progress_marker(tmp_path, pid=p.pid)
        with pytest.raises(ValueError, match="being converted by another conversion"):
            conv.claim_output(str(tmp_path))  # also before any iter_ directory exists
    finally:
        p.kill()
        p.wait()


def test_another_hosts_marker_blocks_until_its_heartbeat_is_stale(tmp_path):
    # shared storage: a conversion killed on another host blocks only until its heartbeat
    # (the marker's mtime) has not moved for MARKER_STALE_S; then its leftover is ours
    (tmp_path / "release").mkdir()
    _write_in_progress_marker(tmp_path, host="some-other-host", pid=1)
    with pytest.raises(ValueError, match="being converted by another conversion.*some-other-host"):
        conv.refuse_existing_output(str(tmp_path))
    old = os.path.getmtime(tmp_path / conv.IN_PROGRESS_MARKER) - conv.MARKER_STALE_S - 1
    os.utime(tmp_path / conv.IN_PROGRESS_MARKER, (old, old))
    conv.claim_output(str(tmp_path), log=lambda *a: None)
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize("other", ["boot", "pidns"])
def test_a_same_host_marker_of_another_boot_or_namespace_uses_its_heartbeat(tmp_path, other):
    # a container sharing the host name (another pid namespace), or a reboot: its pid cannot
    # be checked here, so a dead-looking pid must not make a running conversion "dead"
    (tmp_path / "release").mkdir()
    _write_in_progress_marker(tmp_path, pid=_dead_pid(), **{other: "elsewhere"})
    assert conv._marker_owner(str(tmp_path)) == "live"  # fresh heartbeat
    with pytest.raises(ValueError, match="being converted"):
        conv.claim_output(str(tmp_path), log=lambda *a: None)
    assert (tmp_path / "release").is_dir()
    old = os.path.getmtime(tmp_path / conv.IN_PROGRESS_MARKER) - conv.MARKER_STALE_S - 1
    os.utime(tmp_path / conv.IN_PROGRESS_MARKER, (old, old))
    assert conv._marker_owner(str(tmp_path)) == "dead"


def test_the_marker_age_uses_the_file_servers_clock(tmp_path, monkeypatch):
    # a node whose clock runs 2 h ahead must not judge a fresh marker stale
    import time as _time

    _write_in_progress_marker(tmp_path, host="some-other-host", pid=1)
    real = _time.time
    monkeypatch.setattr(conv.time, "time", lambda: real() + 7200)
    assert conv._marker_owner(str(tmp_path)) == "live"


def test_no_local_clock_when_the_file_servers_clock_cannot_be_read(tmp_path, monkeypatch):
    # an unwritable directory: no fallback to this node's clock (it may be skewed)
    import time as _time

    _write_in_progress_marker(tmp_path, host="some-other-host", pid=1)
    marker = tmp_path / conv.IN_PROGRESS_MARKER
    os.utime(marker, (_time.time() - 10 * conv.MARKER_STALE_S,) * 2)
    monkeypatch.setattr(conv.tempfile, "mkstemp", lambda **_k: (_ for _ in ()).throw(PermissionError()))
    monkeypatch.setattr(conv.time, "time", lambda: (_ for _ in ()).throw(AssertionError("local clock")))
    assert conv._fs_now(str(tmp_path)) is None
    assert conv._marker_owner(str(tmp_path)) == "live"  # never judged dead without that clock
    record = tmp_path / "out.failed"
    record.write_text("run: x\nValueError: old\n")
    assert conv._record_age(str(record)) == float("inf")
    assert not conv._record_of_this_launch(str(record), "x")


def test_retire_compares_with_the_marker_judged_dead(tmp_path):
    # A judged the marker dead; meanwhile B retired it and wrote its own (live) marker. A's
    # retire must leave B's marker in place and refuse.
    path = tmp_path / conv.IN_PROGRESS_MARKER
    _write_in_progress_marker(tmp_path, pid=_dead_pid(), token="dead")
    judged = path.read_text().strip()
    _write_in_progress_marker(tmp_path, host="b-host", pid=1, token="b")  # B's, live
    with pytest.raises(ValueError, match="claimed by another conversion"):
        conv._retire_dead_marker(str(path), judged)
    assert "b-host" in path.read_text() and not list(tmp_path.glob("*.dead.*"))


def test_retire_keeps_a_third_marker_and_refuses_cleanly(tmp_path, monkeypatch):
    # the moved marker is not the dead one, and a third conversion created a marker before the
    # put-back: a ValueError (not FileExistsError), and no marker is lost
    path = tmp_path / conv.IN_PROGRESS_MARKER
    _write_in_progress_marker(tmp_path, host="b-host", pid=1, token="b")
    real_rename = os.rename

    def rename(src, dst):
        real_rename(src, dst)
        _write_in_progress_marker(tmp_path, host="c-host", pid=2, token="c")

    monkeypatch.setattr(conv.os, "rename", rename)
    with pytest.raises(ValueError, match="other conversions"):
        conv._retire_dead_marker(str(path), "the dead marker")
    assert "c-host" in path.read_text()
    (aside,) = tmp_path.glob("*.dead.*")
    assert "b-host" in aside.read_text()


def test_the_heartbeat_moves_the_marker(tmp_path):
    import time

    conv.claim_output(str(tmp_path), log=lambda *a: None)
    marker = tmp_path / conv.IN_PROGRESS_MARKER
    os.utime(marker, (0, 0))
    with conv.marker_heartbeat(str(tmp_path), every=0.05):
        time.sleep(0.3)
    assert os.path.getmtime(marker) > 0


@pytest.mark.parametrize(
    "layout, what",
    [
        ({"iter_0000500": True, "tracker": "500"}, "names iteration 500"),  # a training checkpoint
        ({"release": True, "tracker": "5"}, "names iteration 5"),  # tracker of an iteration
        ({"iter_0000500": True}, "iter_0000500"),  # an iteration directory, no tracker
        ({"iter_0000001": True, "release": True}, "iter_0000001"),  # beside a release/
    ],
)
def test_a_directory_with_another_checkpoint_is_refused_and_kept(tmp_path, layout, what):
    for name in layout:
        if name != "tracker":
            (tmp_path / name).mkdir()
    if "tracker" in layout:
        (tmp_path / "latest_checkpointed_iteration.txt").write_text(layout["tracker"])
    for fn in (conv.refuse_existing_output, conv.claim_output):
        with pytest.raises(ValueError, match=f"another checkpoint.*{what}"):
            fn(str(tmp_path))
    assert all((tmp_path / n).exists() for n in layout if n != "tracker")
    assert not (tmp_path / conv.IN_PROGRESS_MARKER).exists()


def test_a_finished_release_checkpoint_is_refused_and_kept(tmp_path):
    """load_validate_save moves the torch_dist save to release/ (main's layout): a rerun
    into that directory is refused as finished, and nothing is removed."""
    (tmp_path / "release").mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")
    for fn in (conv.refuse_existing_output, conv.claim_output):
        with pytest.raises(ValueError, match="release already exists"):
            fn(str(tmp_path))
    assert (tmp_path / "release").is_dir()
    assert not (tmp_path / conv.IN_PROGRESS_MARKER).exists()


def test_converter_refuses_a_legacy_torch_format(tmp_path):
    with pytest.raises(ValueError, match="torch_dist"):
        conv.MuseSparkConverter(log=lambda *a: None).convert(
            str(tmp_path / "hf"), str(tmp_path / "out"), ckpt_format="torch"
        )
    assert not (tmp_path / "out").exists()


def test_an_empty_stale_marker_is_retired_and_a_fresh_one_blocks(tmp_path):
    # a conversion killed between the marker's create and its write leaves it empty: once its
    # heartbeat is stale it is a dead marker (retired), not a lock forever; while fresh it blocks
    marker = tmp_path / conv.IN_PROGRESS_MARKER
    marker.write_text("")
    with pytest.raises(conv.OutputBusyError):
        conv.claim_output(str(tmp_path), log=lambda *a: None)
    old = os.path.getmtime(marker) - conv.MARKER_STALE_S - 10
    os.utime(marker, (old, old))
    assert conv._marker_owner(str(tmp_path)) == "dead"
    conv.claim_output(str(tmp_path), log=lambda *a: None)
    assert marker.read_text().strip() == conv._marker_text()


def test_claim_output_checks_the_directory_once(tmp_path, monkeypatch):
    calls = []
    real = conv._output_state
    monkeypatch.setattr(conv, "_output_state", lambda *a: calls.append(a) or real(*a))
    conv.claim_output(str(tmp_path), log=lambda *a: None)
    assert len(calls) == 1


def test_convert_claims_nothing_when_its_arguments_are_refused(tmp_path, monkeypatch):
    # build_margs' import-free checks (TP/PP/EP, dtype, vocab) run before the claim
    (tmp_path / "release").mkdir()
    _write_in_progress_marker(tmp_path, pid=_dead_pid())
    monkeypatch.setattr(conv.common, "ensure_megatron_on_path", lambda: "")
    monkeypatch.setattr(conv.common, "read_hf_config", lambda d: {"text_config": TEXT})
    with pytest.raises(ValueError, match="TP=PP=EP=1"):
        conv.convert(str(tmp_path / "hf"), str(tmp_path), tensor_parallel_size=2)
    with pytest.raises(ValueError, match="dtype"):
        conv.convert(str(tmp_path / "hf"), str(tmp_path), dtype="int4")
    assert (tmp_path / "release").is_dir()


def test_the_claim_is_exclusive(tmp_path, monkeypatch):
    # two conversions pass the check together; only one creates the marker (O_EXCL), the
    # other is refused and removes nothing
    (tmp_path / "release").mkdir()
    _write_in_progress_marker(tmp_path, pid=_dead_pid())  # a killed run's leftover
    real_refuse = conv.refuse_existing_output

    def racing_refuse(save_dir):
        out = real_refuse(save_dir)  # the check passes ("ours") ...
        (tmp_path / conv.IN_PROGRESS_MARKER).unlink()  # ... then the other wins the claim
        _write_in_progress_marker(tmp_path, host="winner-host", pid=1)
        return out

    monkeypatch.setattr(conv, "refuse_existing_output", racing_refuse)
    with pytest.raises(ValueError, match="another conversion"):
        conv.claim_output(str(tmp_path), log=lambda *a: None)
    assert (tmp_path / "release").is_dir()  # nothing removed by the loser
    assert "winner-host" in (tmp_path / conv.IN_PROGRESS_MARKER).read_text()


def test_a_retry_in_the_same_process_reclaims_its_own_marker(tmp_path):
    conv.claim_output(str(tmp_path), log=lambda *a: None)  # an attempt that failed later
    (tmp_path / "release").mkdir()
    conv.claim_output(str(tmp_path), log=lambda *a: None)  # the retry
    assert not (tmp_path / "release").exists()


class _WaitClock:
    def __init__(self, events=()):
        self.now, self.events = 0.0, sorted(events)

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.now += s
        while self.events and self.events[0][0] <= self.now:
            self.events.pop(0)[1]()


def _wait(tmp_path, clock):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    done, lock = tmp_path / "out.done", tmp_path / "out.converting.lock"
    conv.wait_for_hook_conversion(
        str(out), str(done), str(lock), log=lambda *a: None, clock=clock, sleep=clock.sleep
    )
    return clock.now


def _finish(out, lock, done):
    """What node 0 does at the end: the converter's tracker, then .done, then the lock goes."""
    (out / conv.TRACKER).write_text("1")
    done.touch()
    lock.unlink()


def test_the_hook_wait_follows_a_long_conversion(tmp_path):
    # node 0 converts for 5 h with a moving heartbeat: no timeout, returns when finished
    out, lock, done = tmp_path / "out", tmp_path / "out.converting.lock", tmp_path / "out.done"
    out.mkdir()
    lock.touch()
    marker = out / conv.IN_PROGRESS_MARKER
    marker.write_text("node0 1 - - t\n")
    base = os.path.getmtime(marker)

    def beat(t):
        # the heartbeat: os.utime moves the mtime (here to the fake time, seconds apart)
        os.utime(marker, (base + t, base + t))

    events = [(t, (lambda t=t: beat(t))) for t in range(60, 5 * 3600, 60)]
    events.append((5 * 3600, lambda: _finish(out, lock, done)))
    assert 5 * 3600 <= _wait(tmp_path, _WaitClock(events)) < 5 * 3600 + 10


def test_the_hook_wait_fails_fast_when_node_0_fails(tmp_path):
    # the hook removes its lock in a finally: no finished checkpoint -> an error after
    # WAIT_LOCK_GONE_S
    lock = tmp_path / "out.converting.lock"
    lock.touch()
    clock = _WaitClock([(100, lock.unlink)])
    with pytest.raises(RuntimeError, match="failed"):
        _wait(tmp_path, clock)
    assert clock.now < 100 + conv.WAIT_LOCK_GONE_S + 10


def test_the_hook_wait_stops_on_a_stale_heartbeat(tmp_path):
    # node 0 wrote its marker in this run, then stopped beating: after MARKER_STALE_S (this
    # node's monotonic clock since the marker last changed) the wait ends
    (tmp_path / "out").mkdir()
    (tmp_path / "out.converting.lock").touch()
    marker = tmp_path / "out" / conv.IN_PROGRESS_MARKER
    clock = _WaitClock([(10, lambda: marker.write_text("node0 1 - - t\n"))])
    with pytest.raises(RuntimeError, match="has not changed"):
        _wait(tmp_path, clock)
    assert 10 + conv.MARKER_STALE_S <= clock.now < 10 + conv.MARKER_STALE_S + 10


@pytest.mark.parametrize("skew", [-7200.0, 7200.0])
def test_the_hook_wait_does_not_compare_clocks(tmp_path, monkeypatch, skew):
    # a waiting node whose wall clock is 2 h off: node 0's .done is still taken
    import time as _time

    real = _time.time
    monkeypatch.setattr(conv.time, "time", lambda: real() + skew)
    out, lock, done = tmp_path / "out", tmp_path / "out.converting.lock", tmp_path / "out.done"
    lock.touch()
    clock = _WaitClock([(50, lambda: (out.mkdir(exist_ok=True), _finish(out, lock, done)))])
    assert 50 <= _wait(tmp_path, clock) < 60


def test_the_hook_wait_ignores_a_killed_runs_old_marker_and_done(tmp_path):
    # a rerun after a crash: the old run's marker and .done are still there (no tracker: the
    # checkpoint was never finished) while node 0 starts; the waiting nodes neither stop on
    # the old marker nor take the old .done, and finish on this run's finished checkpoint
    out, lock, done = tmp_path / "out", tmp_path / "out.converting.lock", tmp_path / "out.done"
    out.mkdir()
    marker = out / conv.IN_PROGRESS_MARKER
    marker.write_text("other-host 1 - - old\n")
    old = os.path.getmtime(marker) - 2000
    os.utime(marker, (old, old))
    done.touch()

    def node0_starts():
        done.unlink()
        lock.touch()
        marker.write_text("node0 1 - - new\n")

    clock = _WaitClock([(5, node0_starts), (300, lambda: _finish(out, lock, done))])
    assert 300 <= _wait(tmp_path, clock) < 310


def test_the_hook_wait_takes_a_conversion_that_finished_before_it_began(tmp_path):
    # node 0 finished between this node's tracker check and the start of its wait: the lock
    # is gone and never seen, the checkpoint is finished: the wait returns at once
    out, done = tmp_path / "out", tmp_path / "out.done"
    out.mkdir()
    (out / conv.TRACKER).write_text("1")
    done.touch()
    assert _wait(tmp_path, _WaitClock()) == 0


def test_the_hook_wait_stops_at_once_when_node_0_refuses_before_the_lock_is_seen(tmp_path):
    # node 0 fails fast (e.g. the 60B size cap) between two polls: the lock is never seen;
    # its failure record ends the wait at once, not after WAIT_START_S
    failed = tmp_path / "out.failed"
    clock = _WaitClock([(7, lambda: failed.write_text("run: -\nValueError: too big\n"))])
    with pytest.raises(RuntimeError, match="failed: ValueError: too big"):
        _wait(tmp_path, clock)
    assert clock.now < 20


def test_the_hook_wait_takes_this_launchs_failure_written_before_it_began(tmp_path, monkeypatch):
    # node 0 failed fast (size cap) and removed its lock before this node started waiting:
    # the record carries this launch's id, so the wait stops at once (not after 1800 s)
    monkeypatch.setenv("PRIMUS_CONVERT_RUN_ID", "launch-7")
    failed = tmp_path / "out.failed"
    failed.write_text(f"run: {conv.conversion_run_id()}\nValueError: too big\n")
    clock = _WaitClock()
    with pytest.raises(RuntimeError, match="failed: ValueError: too big"):
        _wait(tmp_path, clock)
    assert clock.now == 0


def test_the_hook_wait_ignores_an_old_record_with_a_reused_launch_id(tmp_path, monkeypatch):
    # a constant PRIMUS_CONVERT_RUN_ID (or a requeue that kept it): the previous launch's
    # record carries the same id but is older than WAIT_START_S, so it is not this launch's
    monkeypatch.setenv("PRIMUS_CONVERT_RUN_ID", "constant")
    out, lock, done = tmp_path / "out", tmp_path / "out.converting.lock", tmp_path / "out.done"
    failed = tmp_path / "out.failed"
    failed.write_text(f"run: {conv.conversion_run_id()}\nValueError: old failure\n")
    old = os.path.getmtime(failed) - conv.WAIT_START_S - 60
    os.utime(failed, (old, old))

    def node0_starts():
        failed.unlink()
        lock.touch()

    clock = _WaitClock(
        [(5, node0_starts), (100, lambda: (out.mkdir(exist_ok=True), _finish(out, lock, done)))]
    )
    assert 100 <= _wait(tmp_path, clock) < 110


def test_convert_writes_the_failure_record_for_an_os_error_before_the_build(tmp_path, monkeypatch):
    # an OSError in the output checks (here: save_dir is a file) is written to <save_dir>.failed
    monkeypatch.setenv("PRIMUS_CONVERT_RUN_ID", "launch-9")
    out = tmp_path / "out"
    out.write_text("not a directory")

    def not_a_directory(d):
        raise NotADirectoryError(d)

    monkeypatch.setattr(conv, "refuse_existing_output", not_a_directory)
    with pytest.raises(NotADirectoryError):
        conv.convert(str(tmp_path), str(out))
    run_id, message = conv.parse_run_record((tmp_path / "out.failed").read_text())
    assert run_id == "PRIMUS_CONVERT_RUN_ID=launch-9" and message.startswith("NotADirectoryError")


def test_the_hook_wait_stops_when_node_0_failed_before_the_converter_and_before_it_began(
    tmp_path, monkeypatch
):
    # node 0 took the lock, wrote its start record, failed in the convert-phase patches (no
    # .failed: the converter never ran) and removed the lock, all before this node waited:
    # the start record with this launch's id makes the gone lock count (60 s, not 1800 s)
    monkeypatch.setenv("PRIMUS_CONVERT_RUN_ID", "launch-10")
    (tmp_path / "out.failed").write_text("run: -\nValueError: an earlier launch\n")
    conv.hook_before_convert(str(tmp_path / "out"))
    assert not (tmp_path / "out.failed").exists()  # the previous launch's record is removed
    clock = _WaitClock()
    with pytest.raises(RuntimeError, match="ended without a finished checkpoint"):
        _wait(tmp_path, clock)
    assert conv.WAIT_LOCK_GONE_S <= clock.now < conv.WAIT_LOCK_GONE_S + 10


def test_the_hook_wait_ignores_another_launchs_start_record(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIMUS_CONVERT_RUN_ID", "launch-old")
    conv.hook_before_convert(str(tmp_path / "out"))
    monkeypatch.setenv("PRIMUS_CONVERT_RUN_ID", "launch-new")
    out, lock, done = tmp_path / "out", tmp_path / "out.converting.lock", tmp_path / "out.done"
    out.mkdir(exist_ok=True)

    def node0_starts():
        lock.touch()
        (out / conv.IN_PROGRESS_MARKER).write_text("node0 1 - - t\n")

    clock = _WaitClock([(5, node0_starts), (100, lambda: _finish(out, lock, done))])
    assert 100 <= _wait(tmp_path, clock) < 110


def test_a_fresh_record_with_a_reused_id_names_the_reuse(tmp_path, monkeypatch):
    # a constant user id makes a fresh earlier record look like this
    # launch's (documented): the error says so, so the user can tell it from a new failure
    monkeypatch.setenv("PRIMUS_CONVERT_RUN_ID", "fixed")
    (tmp_path / "out.failed").write_text(f"run: {conv.conversion_run_id()}\nValueError: old failure\n")
    with pytest.raises(RuntimeError, match="written before this wait began.*set a new value per launch"):
        _wait(tmp_path, _WaitClock())


@pytest.mark.parametrize(
    "run_id, record",
    [
        pytest.param(None, "ValueError: an old failure\n", id="legacy-record-no-id"),
        pytest.param(
            "launch-8", "run: PRIMUS_CONVERT_RUN_ID=launch-7\nValueError: old\n", id="other-launch-id"
        ),
    ],
)
def test_the_hook_wait_ignores_a_previous_launchs_failure_record(tmp_path, monkeypatch, run_id, record):
    # a failure record written before this launch began: node 0 removes it when it starts,
    # and the waiting node follows this launch's conversion to its end
    if run_id is None:
        monkeypatch.delenv("PRIMUS_CONVERT_RUN_ID", raising=False)
    else:
        monkeypatch.setenv("PRIMUS_CONVERT_RUN_ID", run_id)
    out, lock, done = tmp_path / "out", tmp_path / "out.converting.lock", tmp_path / "out.done"
    failed = tmp_path / "out.failed"
    failed.write_text(record)

    def node0_starts():
        failed.unlink()  # convert() removes it first
        lock.touch()

    clock = _WaitClock(
        [(5, node0_starts), (100, lambda: (out.mkdir(exist_ok=True), _finish(out, lock, done)))]
    )
    assert 100 <= _wait(tmp_path, clock) < 110


@pytest.mark.parametrize(
    "env, want",
    [
        ({"PRIMUS_CONVERT_RUN_ID": "x"}, "PRIMUS_CONVERT_RUN_ID=x"),
        # the converter reads the variable only (no id is derived from Slurm variables)
        ({"SLURM_JOB_ID": "12", "SLURM_STEP_ID": "3"}, None),
        ({"SLURM_JOB_ID": "12"}, None),
        # torchrun's id is not used: the hook runs before torchrun starts
        ({"TORCHELASTIC_RUN_ID": "abc"}, None),
        ({}, None),
    ],
)
def test_conversion_run_id(monkeypatch, env, want):
    for k in (
        "PRIMUS_CONVERT_RUN_ID",
        "SLURM_JOB_ID",
        "SLURM_STEP_ID",
        "SLURM_RESTART_COUNT",
        "TORCHELASTIC_RUN_ID",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert conv.conversion_run_id() == want


def test_a_refused_duplicate_conversion_leaves_the_running_ones_record(tmp_path, monkeypatch):
    # another conversion holds the marker (node 0 converting); a duplicate convert() is
    # refused with OutputBusyError and neither writes nor removes <save_dir>.failed
    save = tmp_path / "ckpt"
    save.mkdir()
    _write_in_progress_marker(save, host="node0-host", pid=1)  # fresh heartbeat: live
    failed = tmp_path / "ckpt.failed"
    failed.write_text("run: x\nValueError: node 0's own record\n")
    before = failed.read_text()
    with pytest.raises(conv.OutputBusyError, match="being converted by another conversion"):
        conv.convert(str(tmp_path / "hf"), str(save))
    assert failed.read_text() == before
    failed.unlink()
    with pytest.raises(conv.OutputBusyError):
        conv.convert(str(tmp_path / "hf"), str(save))
    assert not failed.exists()


def test_failure_records_round_trip(monkeypatch):
    monkeypatch.setenv("PRIMUS_CONVERT_RUN_ID", "r1")
    text = conv.format_failure_record(ValueError("too big"))
    assert text.startswith(conv.RUN_ID_PREFIX)
    assert conv.parse_run_record(text) == ("PRIMUS_CONVERT_RUN_ID=r1", "ValueError: too big")
    assert conv.parse_run_record("ValueError: old format\n") == (None, "ValueError: old format")
    assert conv.parse_run_record("run: -\nX: y\n") == (None, "X: y")


@pytest.mark.parametrize(
    "top, text, want_untied",
    [
        (True, {}, False),  # composite config: the top-level value applies
        (None, {"tie_word_embeddings": True}, False),
        (True, {"tie_word_embeddings": False}, True),  # text_config wins
        (None, {}, True),  # absent: untied (the released model)
    ],
)
def test_tie_word_embeddings_falls_back_to_the_top_level(top, text, want_untied):
    base = {k: v for k, v in TEXT.items() if k != "tie_word_embeddings"}
    cfg = {"text_config": {**base, **text}}
    if top is not None:
        cfg["tie_word_embeddings"] = top
    assert conv.derive_megatron_fields(cfg)["untie_embeddings_and_output_weights"] is want_untied


def test_convert_writes_its_failure_record(tmp_path, monkeypatch):
    save = tmp_path / "ckpt"
    failed = tmp_path / "ckpt.failed"
    failed.write_text("old\n")
    monkeypatch.setattr(conv.common, "read_hf_config", lambda d: {"text_config": TEXT})
    monkeypatch.setattr(conv.common, "ensure_megatron_on_path", lambda: "")
    monkeypatch.setattr(
        conv, "check_convert_size", lambda *a, **k: (_ for _ in ()).throw(ValueError("too big"))
    )
    with pytest.raises(ValueError, match="too big"):
        conv.convert(str(tmp_path / "hf"), str(save))
    assert failed.read_text().split("\n", 1)[1] == "ValueError: too big\n"
    assert failed.read_text().startswith("run: ")
    assert conv.failure_record_path(str(save) + "/") == str(failed)


@pytest.mark.parametrize("locked", [False, True])
def test_the_hook_wait_stops_when_node_0_never_starts(tmp_path, locked):
    # no lock at all, or a lock but no marker of this run (node 0 hung before the claim)
    if locked:
        (tmp_path / "out.converting.lock").touch()
    clock = _WaitClock()
    with pytest.raises(TimeoutError, match="did not start"):
        _wait(tmp_path, clock)
    assert conv.WAIT_START_S <= clock.now < conv.WAIT_START_S + 10


def test_the_hook_wait_reports_progress(tmp_path):
    out, lock, done = tmp_path / "out", tmp_path / "out.converting.lock", tmp_path / "out.done"
    out.mkdir()
    lock.touch()
    marker = out / conv.IN_PROGRESS_MARKER
    marker.write_text("node0 1 - - t\n")
    base = os.path.getmtime(marker)
    events = [(t, (lambda t=t: os.utime(marker, (base + t, base + t)))) for t in range(60, 1900, 60)]
    events.append((1900, lambda: _finish(out, lock, done)))
    clock = _WaitClock(events)
    logs = []
    conv.wait_for_hook_conversion(
        str(out), str(tmp_path / "out.done"), str(lock), log=logs.append, clock=clock, sleep=clock.sleep
    )
    progress = [m for m in logs if m.startswith("waiting for node 0")]
    assert len(progress) == 3 and "converting" in progress[-1]  # at 600, 1200, 1800 s
    assert logs[-1].startswith("node 0's conversion finished")


def test_the_waiting_node_logs_on_its_own_rank(tmp_path, monkeypatch, capsys):
    # log_info prints on rank 0 only: the waiting node's messages must still print
    hook = _convert_hook()
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "muse_spark_text"}))
    monkeypatch.setattr(conv, "wait_for_hook_conversion", lambda *a, log, **k: log("node 0 done"))
    monkeypatch.setenv("NODE_RANK", "2")
    monkeypatch.delenv("RANK", raising=False)
    hook._native_main(str(tmp_path), {"out": str(tmp_path / "ckpt")})
    out = capsys.readouterr().out
    assert "[NODE_RANK 2] Waiting for node 0" in out
    assert "[NODE_RANK 2] node 0 done" in out and "[NODE_RANK 2] Checkpoint ready" in out


def test_the_hook_uses_the_muse_spark_wait_on_the_other_nodes(tmp_path, monkeypatch):
    hook = _convert_hook()
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "muse_spark_text"}))
    seen = []
    monkeypatch.setattr(conv, "wait_for_hook_conversion", lambda *a, **k: seen.append(a))
    monkeypatch.setattr(hook, "wait_for_conversion", lambda *a, **k: pytest.fail("600 s wait used"))
    monkeypatch.setenv("NODE_RANK", "1")
    monkeypatch.delenv("RANK", raising=False)
    hook._native_main(str(tmp_path), {"out": str(tmp_path / "ckpt")})
    assert seen and str(seen[0][0]) == str(tmp_path / "ckpt")


@pytest.mark.parametrize(
    "cfg, want",
    [
        ({"text_config": {"torch_dtype": "float32"}}, "fp32"),
        ({"dtype": "float32"}, "fp32"),
        ({"text_config": {"dtype": "float16"}}, "fp16"),
        ({"torch_dtype": "bfloat16", "text_config": {"torch_dtype": "float32"}}, "bf16"),
        ({}, "bf16"),
        ({"torch_dtype": None, "dtype": "float32"}, "fp32"),  # a JSON null torch_dtype
        # text_config's dtype wins over the wrapper's dtype: the text decoder is converted
        ({"dtype": "float32", "text_config": {"dtype": "bfloat16"}}, "bf16"),
    ],
)
def test_the_hook_reads_the_muse_spark_dtype_keys(tmp_path, cfg, want):
    # main's generic reader (native_registry.dtype_from_hf_config through the hook) with the
    # family's dtype_from_config fallback; no second dtype parser
    hook = _convert_hook()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "muse_spark", "architectures": ["MuseSparkForConditionalGeneration"], **cfg}
        )
    )
    family = hook._detect_native_family(str(tmp_path))
    assert family == "muse_spark"
    assert hook._native_dtype_from_config(str(tmp_path), None, family) == want
    # other families read torch_dtype only (main's behaviour): no dtype / text_config fallback
    for other in ("llama", "qwen3", "deepseek"):
        want_other = {"float32": "fp32", "bfloat16": "bf16", None: "bf16"}[cfg.get("torch_dtype")]
        assert hook._native_dtype_from_config(str(tmp_path), None, other) == want_other


def _ctx(hf_config, config=None):
    from primus.backends.megatron.checkpoint import native_registry as registry

    config = config or {}
    return registry.ConvertContext(
        hf_path="/x",
        hf_config=hf_config,
        family=registry.get_family("muse_spark"),
        experiment={},
        options={},
        config_value=config.get,
    )


def test_the_prepare_step_refuses_a_wrapper_dtype_that_differs_from_the_text_dtype():
    from primus.backends.megatron.checkpoint import native_registry as registry

    cfg = {"torch_dtype": "float32", "text_config": {"dtype": "bfloat16"}}
    with pytest.raises(registry.NativeConvertRefused, match="native_ckpt_dtype"):
        conv.hook_prepare(_ctx(cfg))
    # an explicit native_ckpt_dtype chooses; equal dtypes pass
    plan = conv.hook_prepare(_ctx(cfg, {"native_ckpt_dtype": "bf16"}))
    assert plan.convert_kwargs == {} and plan.path_suffix == ""  # no depth: every layer
    conv.hook_prepare(_ctx({"torch_dtype": "bfloat16", "text_config": {"dtype": "bfloat16"}}))


_FULL_TRAINING = dict(
    num_layers=62,
    hidden_size=8192,
    ffn_hidden_size=16384,
    num_attention_heads=128,
    num_query_groups=16,
    kv_channels=64,
    num_experts=256,
    moe_ffn_hidden_size=4096,
    moe_latent_size=4096,
    swiglu=True,
    vocab_size=202048,
    untie_embeddings_and_output_weights=True,
    bf16=True,
    use_distributed_optimizer=True,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=8,
    expert_tensor_parallel_size=1,
    muse_spark_skip_memory_check=False,
)


def _train_ctx(hf_layers, config, argv=()):
    from primus.backends.megatron.checkpoint import native_registry as registry

    return registry.ConvertContext(
        hf_path="/x",
        hf_config={"text_config": {"num_hidden_layers": hf_layers}},
        family=registry.get_family("muse_spark"),
        experiment={},
        options={},
        argv=list(argv),
        config_value=config.get,
    )


def test_the_prepare_step_refuses_a_training_depth_above_the_checkpoint():
    from primus.backends.megatron.checkpoint import native_registry as registry

    with pytest.raises(registry.NativeConvertRefused, match="num_layers=62.*only 4 layers"):
        conv.check_training_fits(_train_ctx(4, _FULL_TRAINING), gpu_bytes=None)
    # the command-line override (--num_layers 4) is what the training builds
    ctx = _train_ctx(4, _FULL_TRAINING, ["train", "posttrain", "--num_layers", "4"])
    conv.check_training_fits(ctx, gpu_bytes=288 * 10**9, world_size=8, log=lambda *a: None)


def test_the_prepare_step_reads_the_training_args_once(monkeypatch):
    # check_training_fits returns the cut; hook_prepare reuses it (no second parse)
    skip = {**_FULL_TRAINING, "muse_spark_skip_memory_check": True}
    ctx = _train_ctx(62, skip, ["train", "posttrain", "--num_layers", "4"])
    assert conv.check_training_fits(ctx, gpu_bytes=None) == 4
    calls = []
    real = conv.training_args
    monkeypatch.setattr(conv, "training_args", lambda c: calls.append(c) or real(c))
    plan = conv.hook_prepare(ctx)
    assert len(calls) == 1 and plan.convert_kwargs == {"num_layers": 4}
    assert conv._cut_record_value(None) == conv.CUT_RECORD_ALL and conv._cut_record_value(4) == "4"


def test_a_shallower_fine_tune_converts_a_cut_of_the_released_checkpoint(tmp_path):
    # the proxy preset (or --num_layers 4) against the 62-layer release: the hook passes the
    # cut to convert(num_layers=4), into its own directory, and records the depth
    skip = {**_FULL_TRAINING, "muse_spark_skip_memory_check": True}  # no GPU-dependent check
    ctx = _train_ctx(62, skip, ["train", "posttrain", "--num_layers", "4"])
    assert conv.planned_cut(ctx) == 4
    plan = conv.hook_prepare(ctx)
    assert plan.convert_kwargs == {"num_layers": 4} and plan.path_suffix == "_layers4"
    out = tmp_path / "ckpt_native_torch_dist_layers4"
    plan.before_convert(out)
    assert (tmp_path / "ckpt_native_torch_dist_layers4.muse_spark_layers").read_text().strip() == "4"
    assert plan.check_existing(out) is None  # converted at this depth: reused
    # the full depth: no cut, and a checkpoint converted at 4 layers is not reused for it
    full = conv.hook_prepare(_train_ctx(62, skip, []))
    assert full.convert_kwargs == {} and full.path_suffix == ""
    assert "converted with 4 decoder layer(s)" in full.check_existing(out)
    # a checkpoint without the record (converted elsewhere) is reused as before
    assert full.check_existing(tmp_path / "elsewhere") is None


def test_the_hook_passes_the_cut_to_convert(monkeypatch):
    # the cut reaches MuseSparkConverter(num_layers=...) through convert()'s keyword
    seen = {}

    class _Conv:
        def __init__(self, log=None, num_layers=None):
            seen["num_layers"] = num_layers

        def convert(self, *a, **kw):
            return "ok"

    monkeypatch.setattr(conv, "MuseSparkConverter", _Conv)
    monkeypatch.setattr(conv, "refuse_existing_output", lambda d: (None, None))
    assert conv.convert("/hf", "/nonexistent-out", num_layers=4) == "ok"
    assert seen == {"num_layers": 4}


def test_the_prepare_step_refuses_a_layout_that_cannot_fit_before_converting():
    from primus.backends.megatron.checkpoint import native_registry as registry

    # the full model on one node of MI355X: refused before hours of conversion
    with pytest.raises(registry.NativeConvertRefused, match="checked before the conversion"):
        conv.check_training_fits(_train_ctx(62, _FULL_TRAINING), gpu_bytes=288 * 10**9, world_size=8)
    # the skip flag, or no visible GPU, leaves it to the before_train check
    skip = {**_FULL_TRAINING, "muse_spark_skip_memory_check": True}
    conv.check_training_fits(_train_ctx(62, skip), gpu_bytes=288 * 10**9, world_size=8)
    conv.check_training_fits(_train_ctx(62, _FULL_TRAINING), gpu_bytes=None, world_size=8)
    # the full-scale layout (TP 8 PP 5 EP 16, 14 layers first, 20 nodes) fits
    full = {
        **_FULL_TRAINING,
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 5,
        "expert_model_parallel_size": 16,
        "decoder_first_pipeline_num_layers": 14,
    }
    conv.check_training_fits(_train_ctx(62, full), gpu_bytes=288 * 10**9, world_size=160, log=lambda *a: None)


def test_the_registry_routes_muse_spark_and_its_options():
    from primus.backends.megatron.checkpoint import native_registry as registry

    fam = registry.get_family("muse_spark")
    assert fam.converter_module().convert is conv.convert
    assert fam.hook("wait") is conv.hook_wait and fam.model_type == "muse_spark"
    assert registry.read_options(fam, {"native_ckpt_max_params": 0}.get) == {"max_params": 0}
    assert registry.read_options(fam, {}.get) == {}  # unset: no cap (the memory check decides)


def test_convert_deletes_nothing_when_the_size_cap_refuses(tmp_path, monkeypatch):
    # the leftover of a killed run survives a conversion refused by the size cap
    (tmp_path / "release").mkdir()
    _write_in_progress_marker(tmp_path, pid=_dead_pid())
    monkeypatch.setattr(conv.common, "ensure_megatron_on_path", lambda: "")
    monkeypatch.setattr(conv.common, "read_hf_config", lambda d: {"text_config": TEXT})
    monkeypatch.setattr(
        conv, "check_convert_size", lambda *a, **k: (_ for _ in ()).throw(ValueError("too big"))
    )
    with pytest.raises(ValueError, match="too big"):
        conv.convert(str(tmp_path / "hf"), str(tmp_path))
    assert (tmp_path / "release").is_dir()


def test_convert_size_cap_follows_the_config_value():
    fields = conv.derive_megatron_fields({"text_config": TEXT})
    n = conv.estimate_num_params(fields)
    assert conv.check_convert_size(fields, max_params=n + 1) == n
    with pytest.raises(ValueError, match="above native_ckpt_max_params"):
        conv.check_convert_size(fields, max_params=n - 1)


@pytest.mark.parametrize("key", ["attention_bias", "mlp_bias"])
def test_derive_fields_rejects_linear_biases(key):
    with pytest.raises(ValueError, match="not supported"):
        conv.derive_megatron_fields({**TEXT, key: True})
    # False (and absent) is accepted for this key whatever the other key is
    fields = conv.derive_megatron_fields({**TEXT, key: False})
    assert fields["add_qkv_bias"] is False and fields["num_layers"] == TEXT["num_hidden_layers"]
    other = "mlp_bias" if key == "attention_bias" else "attention_bias"
    with pytest.raises(ValueError, match=f"{other}=True"):
        conv.derive_megatron_fields({**TEXT, key: False, other: True})


def test_dispatch_uses_model_type(monkeypatch):
    """The trainer needs no Muse Spark branch (its default branch serves it).
    get_model_provider returns the Muse Spark builder for ``model_type == "muse_spark"`` (a
    string literal: no other model imports the Muse Spark package); both use
    layer_patterns.MODEL_TYPE. The config schema check must see the branch, so
    MuseSparkTransformerConfig fields are validated."""
    import pathlib
    from functools import partial

    pytest.importorskip("megatron.core")
    from primus.backends.megatron.core.models.muse_spark import muse_spark_builders
    from primus.backends.megatron.core.models.muse_spark.layer_patterns import (
        MODEL_TYPE,
    )
    from primus.core.config.schema_check import _model_provider_branches
    from primus.core.utils import import_utils

    monkeypatch.setattr(import_utils, "log_rank_0", lambda *a, **k: None)
    provider = import_utils.get_model_provider(model_type=MODEL_TYPE)
    assert isinstance(provider, partial)
    assert provider.func is muse_spark_builders.model_provider
    assert provider.args == (muse_spark_builders.muse_spark_builder,)

    root = pathlib.Path(__file__).resolve().parents[5]
    _, branches, _ = _model_provider_branches(root, "megatron")
    assert branches.get(MODEL_TYPE, "").endswith("muse_spark.muse_spark_builders")


def test_convert_rejects_a_zero_cap_before_reading_weights(tmp_path):
    import json as _json

    (tmp_path / "config.json").write_text(_json.dumps({"text_config": TEXT}))
    with pytest.raises(ValueError, match="max_params"):
        conv.convert(
            str(tmp_path),
            str(tmp_path / "out"),
            dtype="bf16",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            expert_parallel_size=1,
            max_params=0,
        )


# ---------------------------------------------------------------- presets: vocab / shapes
def _preset(name):
    import pathlib

    import primus
    from primus.core.config.yaml_loader import parse_yaml

    return parse_yaml(str(pathlib.Path(primus.__file__).parent / "configs/models/megatron" / name))


@pytest.mark.parametrize("name", ["muse_Spark_Open_816B_A42B.yaml", "muse_Spark_Open_816B_A42B_proxy.yaml"])
def test_presets_set_the_checkpoint_vocab_and_freeze_the_router_bias(name):
    cfg = _preset(name)
    assert cfg["vocab_size"] == TEXT["vocab_size"] == 202048
    assert cfg["tokenizer_model"] == "meta-models/Muse-Spark-Open-816B-A42B"
    assert float(cfg["moe_router_bias_update_rate"]) == 0.0  # fine-tuning keeps the trained bias


def test_full_preset_shapes_match_the_hf_config():
    cfg = _preset("muse_Spark_Open_816B_A42B.yaml")
    assert (
        cfg["num_layers"],
        cfg["hidden_size"],
        cfg["num_attention_heads"],
        cfg["num_query_groups"],
        cfg["kv_channels"],
        cfg["num_experts"],
        cfg["moe_router_topk"],
        cfg["moe_ffn_hidden_size"],
        cfg["moe_latent_size"],
        cfg["ffn_hidden_size"],
    ) == (
        TEXT["num_hidden_layers"],
        TEXT["hidden_size"],
        TEXT["num_attention_heads"],
        TEXT["num_key_value_heads"],
        TEXT["head_dim"],
        TEXT["num_local_experts"],
        TEXT["num_experts_per_tok"],
        TEXT["moe_intermediate_size"],
        TEXT["moe_hidden_size"],
        TEXT["intermediate_size"],
    )


@pytest.mark.parametrize(
    "tokenizer, tp, padded, vocab",
    [
        ("HuggingFaceTokenizer", 1, 202048, 202048),
        ("HuggingFaceTokenizer", 8, 202048, 202048),  # same rows at every TP
        ("NullTokenizer", 1, 202048, 202047),  # mock data: EOD id 202047 stays inside the rows
        ("NullTokenizer", 4, 202048, 202047),
    ],
)
def test_vocab_patch_gives_the_checkpoint_rows(tokenizer, tp, padded, vocab):
    from primus.backends.megatron.patches.args import (
        muse_spark_layer_patterns_patches as p,
    )

    args = SimpleNamespace(
        model_type="muse_spark",
        vocab_size=202048,
        make_vocab_size_divisible_by=64,
        tensor_model_parallel_size=tp,
        tokenizer_type=tokenizer,
        padded_vocab_size=None,
    )
    p.patch_muse_spark_vocab(SimpleNamespace(extra={"backend_args": args}))
    assert args.padded_vocab_size == padded == conv.vocab_size_with_padding(202048, 64, tp)
    assert args.vocab_size == vocab
    p.patch_muse_spark_vocab(SimpleNamespace(extra={"backend_args": args}))  # set: unchanged
    assert args.padded_vocab_size == padded and args.vocab_size == vocab


def test_vocab_patch_keeps_an_explicit_padded_vocab():
    from primus.backends.megatron.patches.args import (
        muse_spark_layer_patterns_patches as p,
    )

    args = SimpleNamespace(
        model_type="muse_spark",
        vocab_size=202048,
        make_vocab_size_divisible_by=64,
        tensor_model_parallel_size=1,
        tokenizer_type="HuggingFaceTokenizer",
        padded_vocab_size=7,
    )
    p.patch_muse_spark_vocab(SimpleNamespace(extra={"backend_args": args}))
    assert args.padded_vocab_size == 7 and args.vocab_size == 202048


@pytest.mark.parametrize("multiple, warned", [(None, False), (64, False), (128, False), (256, True)])
def test_vocab_patch_warns_that_a_user_vocab_multiple_is_ignored(monkeypatch, multiple, warned):
    from primus.backends.megatron.patches.args import (
        muse_spark_layer_patterns_patches as p,
    )

    warns = []
    monkeypatch.setattr(p, "warning_rank_0", warns.append)
    monkeypatch.setattr(p, "log_kv_rank_0", lambda *a: None)
    args = SimpleNamespace(
        model_type="muse_spark",
        vocab_size=202048,
        make_vocab_size_divisible_by=multiple,
        tensor_model_parallel_size=1,
        tokenizer_type="HuggingFaceTokenizer",
        padded_vocab_size=None,
    )
    p.patch_muse_spark_vocab(SimpleNamespace(extra={"backend_args": args}))
    assert args.padded_vocab_size == 202048  # 64 is used whatever the key says
    assert bool(warns) is warned
    if warned:
        assert "make_vocab_size_divisible_by=256 is ignored" in warns[0]


def test_vocab_patch_fits_the_mock_eod_into_an_explicit_padded_vocab():
    from primus.backends.megatron.patches.args import (
        muse_spark_layer_patterns_patches as p,
    )

    args = SimpleNamespace(
        model_type="muse_spark",
        vocab_size=202048,
        tensor_model_parallel_size=1,
        tokenizer_type="NullTokenizer",
        padded_vocab_size=202048,
    )
    p.patch_muse_spark_vocab(SimpleNamespace(extra={"backend_args": args}))
    assert args.padded_vocab_size == 202048 and args.vocab_size == 202047  # EOD id 202047 fits
    args.padded_vocab_size, args.vocab_size = 202112, 202048  # already room for the EOD id
    p.patch_muse_spark_vocab(SimpleNamespace(extra={"backend_args": args}))
    assert args.vocab_size == 202048


def test_released_tokenizer_fits_the_preset_vocab():
    """The released tokenizer (when it can be loaded) has no id at or above vocab_size."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B")
    except OSError as e:  # offline / no cached copy / gated: only these mean "not available"
        pytest.skip(f"tokenizer not available: {type(e).__name__}")
    # a ValueError (unknown tokenizer class) or ImportError (missing backend) is a real
    # incompatibility and fails the test
    assert len(tok) <= _preset("muse_Spark_Open_816B_A42B.yaml")["vocab_size"]


def test_layer_patterns_patch_normalizes_backend_args_and_checks_them():
    """The build_args patch itself: both pattern keys parsed and cut back into backend_args,
    an absent key left absent, and the moe_layer_freq / softcap checks fed from the args."""
    from primus.backends.megatron.patches.args.muse_spark_layer_patterns_patches import (
        patch_muse_spark_layer_patterns,
    )

    full = "[0, 1, 1, 1, 0, 1, 1, 1]"
    backend = SimpleNamespace(num_layers=4, no_rope_freq=full, window_attn_skip_freq=full, moe_layer_freq=1)
    patch_muse_spark_layer_patterns(_patch_ctx(SimpleNamespace(), backend))
    assert backend.no_rope_freq == [0, 1, 1, 1]
    assert backend.window_attn_skip_freq == [0, 1, 1, 1]

    only_one = SimpleNamespace(num_layers=4, no_rope_freq="[1, 1, 1, 0]", moe_layer_freq=1)
    patch_muse_spark_layer_patterns(_patch_ctx(SimpleNamespace(), only_one))
    assert only_one.no_rope_freq == [1, 1, 1, 0]
    assert not hasattr(only_one, "window_attn_skip_freq")

    dense = SimpleNamespace(num_layers=4, no_rope_freq=full, moe_layer_freq="[1, 0, 1, 1]")
    with pytest.raises(ValueError):
        patch_muse_spark_layer_patterns(_patch_ctx(SimpleNamespace(), dense))

    capped = SimpleNamespace(num_layers=4, moe_layer_freq=1, final_logit_softcapping=-20.0)
    with pytest.raises(ValueError, match="final_logit_softcapping"):
        patch_muse_spark_layer_patterns(_patch_ctx(SimpleNamespace(), capped))


def test_the_converter_saves_torch_dist_through_mains_mapping_check(tmp_path, monkeypatch):
    # the mapping check, the save and the release/ layout are
    # native_convert_common.load_validate_save; the converter only sets torch_dist and
    # refuses an existing checkpoint
    ckpt = pytest.importorskip("megatron.training.checkpointing")
    save = tmp_path / "ckpt"
    saved = []

    def fake_save(iteration, *a, release=False, **k):  # writes what Megatron's save writes
        saved.append((iteration, release))
        (save / ("release" if release else f"iter_{iteration:07d}")).mkdir(parents=True)
        (save / "latest_checkpointed_iteration.txt").write_text("release" if release else str(iteration))

    monkeypatch.setattr(ckpt, "save_checkpoint", fake_save)

    class Model:
        def __init__(self, missing=(), unexpected=()):
            self.r = (list(missing), list(unexpected))

        def load_state_dict(self, sd, strict):
            assert strict is False
            return self.r

        def state_dict(self):
            return {"w": torch.zeros(4)}

    store = SimpleNamespace(all_keys=lambda: {"a", "b"})
    margs = SimpleNamespace(iteration=2)
    conv.prepare_torch_dist_save(margs, str(save))
    assert margs.ckpt_format == "torch_dist" and margs.save == str(save)
    logs = []
    for model, consumed, what in (
        (Model(missing=["w"]), {"a", "b"}, "missing_non_extra=1"),
        (Model(unexpected=["u"]), {"a", "b"}, "unexpected=1"),
        (Model(), {"a"}, "unconsumed=1"),
    ):
        with pytest.raises(RuntimeError, match=what):
            conv.common.load_validate_save(model, {}, consumed, store, margs, str(save), logs.append)
    assert saved == []
    conv.common.load_validate_save(
        Model(missing=["x._extra_state"]), {"k": 1}, {"a", "b"}, store, margs, str(save), logs.append
    )
    assert saved == [(2, True)]  # straight into release/ (main's torch_dist layout)
    assert (save / "release").is_dir() and not (save / "iter_0000002").exists()
    assert (save / "latest_checkpointed_iteration.txt").read_text() == "release"
    # a rerun into the finished directory is refused, and nothing is removed
    with pytest.raises(ValueError, match="release already exists"):
        conv.prepare_torch_dist_save(SimpleNamespace(iteration=1), str(save))
    assert (save / "release").is_dir()


def _convert_hook():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[5]
    path = root / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"
    spec = importlib.util.spec_from_file_location("muse_spark_convert_hook", path)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    return hook


@pytest.mark.parametrize(
    "cfg",
    [
        # each key alone: the architecture match returns first, so a combined case would
        # never reach the model_type entries
        {"architectures": ["MuseSparkForConditionalGeneration"]},
        {"model_type": "muse_spark"},
        {"model_type": "muse_spark_text"},
    ],
)
def test_the_post_training_hook_detects_a_muse_spark_checkpoint(tmp_path, cfg):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert _convert_hook()._detect_native_family(str(tmp_path)) == "muse_spark"


def test_the_post_training_hook_runs_the_muse_spark_converter(tmp_path, monkeypatch):
    # the hook's muse_spark branch dispatches to native_loader_muse_spark_text.convert (not
    # the DeepSeek fallback) with the hook's dtype / TP / PP / EP
    from primus.backends.megatron.checkpoint import native_convert_common as common
    from primus.backends.megatron.checkpoint import (
        native_loader_muse_spark_text as conv_text,
    )

    hook = _convert_hook()
    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: "")
    monkeypatch.setattr(common, "apply_convert_patches", lambda **k: 0)
    calls = []
    monkeypatch.setattr(conv_text, "convert", lambda *a, **k: calls.append((a, k)))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "muse_spark_text"}))
    out = str(tmp_path / "out")
    opts = {"family": "muse_spark", "dtype": "bf16", "convert_kwargs": {"max_params": "820_000_000_000"}}
    hook.native_convert_checkpoint(str(tmp_path), out, opts)
    tp_pp_ep = dict(tensor_parallel_size=1, pipeline_parallel_size=1, expert_parallel_size=1)
    want = dict(dtype="bf16", **tp_pp_ep, ckpt_format="torch_dist", max_params="820_000_000_000")
    assert calls == [((str(tmp_path), out), want)]


def test_the_post_training_hook_reads_the_convert_cap_from_the_config(tmp_path, monkeypatch):
    # sft_trainer params native_ckpt_max_params -> convert_kwargs["max_params"] through main's
    # registry option reading (params first, then the module; an explicit 0 is kept)
    hook = _convert_hook()
    params = SimpleNamespace(native_ckpt_max_params="820_000_000_000")
    module = SimpleNamespace(params=params)
    monkeypatch.setattr(hook, "load_primus_config", lambda *a: None)
    monkeypatch.setattr(
        hook, "get_module_config", lambda cfg, name: module if name == "sft_trainer" else None
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "muse_spark_text"}))

    def read():
        opts = {}
        hook._prepare_native_family(str(tmp_path), "muse_spark", opts, {}, "x.yaml", [])
        return opts["convert_kwargs"].get("max_params")

    assert read() == "820_000_000_000"
    params.native_ckpt_max_params = None
    assert read() is None
    module.native_ckpt_max_params = "70_000_000_000"
    assert read() == "70_000_000_000"
    params.native_ckpt_max_params = 0
    assert read() == 0


def test_the_post_training_hook_detects_nothing_without_a_local_config(tmp_path):
    hook = _convert_hook()
    assert hook._detect_native_family(str(tmp_path)) is None  # no config.json
    assert hook._detect_native_family("meta-models/Muse-Spark-Open-816B-A42B") is None  # a Hub id


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        (None, False),
        (1, True),
        (0, False),
        (" Yes ", True),
        ("off", False),
        ("", False),
        ("TRUE", True),
    ],
)
def test_the_muse_spark_bool_keys_share_one_parser(value, expected):
    from primus.backends.megatron.core.models.muse_spark.model_types import bool_flag

    assert bool_flag(value, "some_key") is expected


@pytest.mark.parametrize("value", [None, "", "   "])
@pytest.mark.parametrize("default", [True, False])
def test_an_unset_flag_takes_its_default(value, default):
    from types import SimpleNamespace

    from primus.backends.megatron.core.models.muse_spark.model_types import (
        bool_flag,
        config_flag,
    )

    assert bool_flag(value, "k", default) is default
    assert config_flag(SimpleNamespace(k=value), "k", default) is default
    assert config_flag(SimpleNamespace(), "k", default) is default
    assert config_flag(SimpleNamespace(k="false"), "k", True) is False  # an explicit value wins


def test_a_string_default_is_parsed_like_a_value():
    from primus.backends.megatron.core.models.muse_spark.model_types import bool_flag

    assert bool_flag(None, "k", "false") is False
    assert bool_flag("", "k", "on") is True
    with pytest.raises(ValueError, match="k="):
        bool_flag(None, "k", "ture")


@pytest.mark.parametrize(
    "name", ["muse_spark_fused_attn_norm", "embedding_norm", "window_size_is_hf_sliding_window"]
)
def test_the_config_keeps_the_default_of_an_unset_bool_field(name):
    pytest.importorskip("megatron.core")
    from primus.backends.megatron.core.models.muse_spark.muse_spark_transformer_config import (
        MuseSparkTransformerConfig,
    )

    assert MuseSparkTransformerConfig.__dataclass_fields__[name].default is True
    cfg = _transformer_config(**{name: None})
    if name == "window_size_is_hf_sliding_window":
        # unset = the default (HF convention): the HF window 2048 became TE's 2047
        assert cfg.window_size == (2047, 0) and cfg.window_size_is_hf_sliding_window is False
    else:
        assert getattr(cfg, name) is True


@pytest.mark.parametrize("value", ["ture", "maybe", "2", 2, 1.0, -1, [True]])
def test_the_muse_spark_bool_keys_refuse_typos(value):
    from primus.backends.megatron.core.models.muse_spark.model_types import bool_flag

    with pytest.raises(ValueError, match="some_key=.* is not a boolean"):
        bool_flag(value, "some_key")


# ---------------------------------------------------------------- streaming conversion
class _OwnModel(_GroupedModel):
    """A built model whose state dict holds real tensors (the mapped shapes) to copy into."""

    def __init__(self, sd, padded_vocab, L, dtypes=None):
        super().__init__(padded_vocab, L)
        dtypes = dtypes or {}
        self.keys = {k: torch.zeros(v.shape, dtype=dtypes.get(k, v.dtype)) for k, v in sd.items()}
        self.embedding = SimpleNamespace(
            word_embeddings=SimpleNamespace(weight=self.keys["embedding.word_embeddings.weight"])
        )

    def state_dict(self, keep_vars=False):
        return dict(self.keys)  # a new dict, as nn.Module's: rebinding an entry does not reach the model


def test_streaming_maps_one_layer_per_chunk_and_matches_the_full_mapping():
    t, margs, fields = _tiny_checkpoint(L=3)
    store = conv.MappedStore(_Store(t), log=lambda *a: None)
    full, consumed = conv.build_mcore_state_dict(_GroupedModel(16, 3), store, margs, fields, lambda *a: None)
    chunks = [
        dict(c) for c, _ in conv.iter_mcore_state_dict(_GroupedModel(16, 3), store, margs, fields, print)
    ]
    assert len(chunks) == 1 + 3  # embedding / norm / head, then one per layer
    for L, chunk in enumerate(chunks[1:]):
        assert {k.split(".")[2] for k in chunk} == {str(L)}  # a chunk holds its own layer only
    assert set().union(*chunks) == set(full)

    model = _OwnModel(full, 16, 3)
    before = dict(model.keys)  # the model's own tensors before the copy
    sd, streamed = conv.stream_into_model(model, store, margs, fields, lambda *a: None)
    assert streamed == consumed == store.all_keys()
    assert set(sd) == set(full)
    for k, v in full.items():
        # copied into the model's own tensor: the same object, written in place
        torch.testing.assert_close(before[k], v)
        assert sd[k].data_ptr() == before[k].data_ptr()


def test_streaming_keeps_the_models_dtypes_and_refuses_a_bad_mapping():
    t, margs, fields = _tiny_checkpoint(L=1)
    margs.params_dtype = torch.bfloat16
    store = conv.MappedStore(_Store(t), log=lambda *a: None)
    full, _ = conv.build_mcore_state_dict(_GroupedModel(16, 1), store, margs, fields, lambda *a: None)
    router, proj = "decoder.layers.0.mlp.router.weight", "decoder.layers.0.self_attention.linear_proj.weight"
    assert full[router].dtype == torch.float32 and full[proj].dtype == torch.bfloat16
    # model dtypes that differ from the mapped ones: the copy must cast into the model's own
    model = _OwnModel(full, 16, 1, dtypes={router: torch.bfloat16, proj: torch.float32})
    before = dict(model.keys)
    conv.stream_into_model(model, store, margs, fields, lambda *a: None)
    assert before[router].dtype == torch.bfloat16 and before[proj].dtype == torch.float32
    torch.testing.assert_close(before[router], full[router].to(torch.bfloat16))
    torch.testing.assert_close(before[proj], full[proj].float())

    missing_key = _OwnModel({k: v for k, v in full.items() if "router.weight" not in k}, 16, 1)
    with pytest.raises(RuntimeError, match="keys the model does not have"):
        conv.stream_into_model(missing_key, store, margs, fields, lambda *a: None)
    wrong = _OwnModel(dict(full, **{"decoder.layers.0.mlp_bda.gate": torch.zeros(3)}), 16, 1)
    with pytest.raises(RuntimeError, match="mlp_bda.gate: HF tensor shape"):
        conv.stream_into_model(wrong, store, margs, fields, lambda *a: None)


def test_the_save_writes_unsplit_expert_fc1_and_restores_the_config():
    experts = pytest.importorskip("megatron.core.transformer.moe.experts")

    class Experts(experts.SequentialMLP):
        def __init__(self, config):  # no parallel state needed: only the config is swapped
            torch.nn.Module.__init__(self)
            self.config = config
            self.inner = torch.nn.Module()
            self.inner.config = config

    cfg = SimpleNamespace(gated_linear_unit=True, hidden_size=8)
    model = torch.nn.Module()
    model.experts = Experts(cfg)
    model.other = torch.nn.Module()
    model.other.config = cfg  # not an expert module: left alone
    with conv.unsplit_expert_fc1_save(model) as n:
        assert n == 2
        assert model.experts.config.gated_linear_unit is False
        assert model.experts.inner.config.gated_linear_unit is False
        assert model.experts.config.hidden_size == 8 and model.other.config is cfg
        assert cfg.gated_linear_unit is True  # the shared config is not changed
    assert model.experts.config is cfg and model.experts.inner.config is cfg


def test_build_model_uses_the_shared_builder_with_the_family(monkeypatch):
    seen = []
    monkeypatch.setattr(
        conv.common, "build_mcore_gpt_model", lambda margs, **kw: seen.append((margs, kw)) or "model"
    )
    margs = SimpleNamespace(params_dtype=torch.bfloat16)
    assert conv.build_model(margs) == "model"
    assert seen == [(margs, {"family": "muse_spark"})]
    from primus.backends.megatron.checkpoint import native_registry as registry

    assert registry.model_type_for("muse_spark") == "muse_spark"


@pytest.mark.parametrize("save_dir", ["ckpt", "ckpt/", "/a/ckpt/"])
def test_the_records_are_siblings_of_the_checkpoint_directory(save_dir):
    # a trailing separator never puts a record inside the checkpoint directory
    base = save_dir.rstrip("/")
    assert conv.failure_record_path(save_dir) == base + ".failed"
    assert conv.start_record_path(save_dir) == base + ".started"
    assert conv.cut_record_path(save_dir) == base + conv.CUT_RECORD_SUFFIX


def test_parallel_reader_matches_safe_open(tmp_path):
    # the conversion's concurrent pread reader returns the bytes safe_open returns, for every
    # dtype the checkpoint has, with tensors split into several chunks and read by threads
    safetensors = pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(0)
    tensors = {
        "a.bf16": torch.randn(37, 129, generator=g).to(torch.bfloat16),
        "b.f32": torch.randn(5, 3, 7, generator=g),
        "c.scalar": torch.tensor(3.5),
        "d.empty": torch.zeros(0, 4),
        "e.i64": torch.arange(11),
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    reader = conv.ParallelSafetensorsReader(threads=4, chunk=64)  # many chunks per tensor
    try:
        with safetensors.safe_open(str(path), framework="pt") as f:
            for name in tensors:
                got, want = reader.read(str(path), name), f.get_tensor(name)
                assert got.dtype == want.dtype and got.shape == want.shape
                assert torch.equal(got, want)
    finally:
        reader.close()


def test_mapped_store_reads_through_the_parallel_reader(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    w = torch.randn(8, 4).to(torch.bfloat16)
    save_file({"model.language_model.embed_tokens.weight": w}, str(tmp_path / "model.safetensors"))
    store = conv.MappedStore(conv.common.SafetensorsStore(str(tmp_path)), log=lambda *a: None)
    got = store.get("model.language_model.embed_tokens.weight", torch.float32)
    assert got.dtype == torch.float32 and torch.equal(got, w.float())
