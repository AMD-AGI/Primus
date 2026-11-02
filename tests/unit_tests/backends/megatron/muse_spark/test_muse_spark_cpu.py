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
  grouped expert keys, every text tensor consumed, vision / audio exempt);
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


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_converter_padding_does_not_depend_on_tp(tp):
    # the rows depend only on the config: one TP1 checkpoint loads at every TP
    padded = conv.vocab_size_with_padding(202048, conv.MAKE_VOCAB_SIZE_DIVISIBLE_BY, tp)
    assert padded == 202048 == math.ceil(202048 / 64) * 64
    assert conv.vocab_size_with_padding(202049, 64, tp) == 202112


def test_padding_rejects_rows_that_do_not_divide_by_tp():
    with pytest.raises(ValueError, match="does not divide by tensor_model_parallel_size=3"):
        conv.vocab_size_with_padding(202048, 64, 3)
    with pytest.raises(ValueError, match="must be >= 1"):
        conv.vocab_size_with_padding(202048, 0, 1)


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
    args = SimpleNamespace(vocab_size=202049, tensor_model_parallel_size=1, make_vocab_size_divisible_by=128)
    assert p.padded_vocab_of(args) == 202112


def test_build_margs_reads_the_checkpoint_vocab_without_a_passed_config(tmp_path):
    # convert() passes the loaded config; the two-argument build_margs form (the
    # logprob-parity harness) reads <hf_dir>/config.json, so neither falls back to the
    # tokenizer's vocab (201856 rows instead of the checkpoint's 202048)
    cfg = {"text_config": dict(TEXT)}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    args = SimpleNamespace(hf_dir=str(tmp_path))
    assert conv._hf_cfg_of(args, cfg) is cfg
    read = conv._hf_cfg_of(args)
    assert conv.checkpoint_vocab(read) == TEXT["vocab_size"] == 202048
    assert conv.vocab_size_with_padding(conv.checkpoint_vocab(read), 64, 1) == 202048
    assert conv._hf_cfg_of(SimpleNamespace()) is None


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
    P = conv.HF_PREFIX
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
    t["model.vision_model.patch_embed.weight"] = r(3, 3)
    t["model.audio_model.proj.weight"] = r(3, 3)
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
    def __init__(self, padded_vocab, L):
        self.keys = {"embedding.word_embeddings.weight": torch.empty(padded_vocab, 8)}
        self.embedding = SimpleNamespace(
            word_embeddings=SimpleNamespace(weight=self.keys["embedding.word_embeddings.weight"])
        )
        for i in range(L):
            self.keys[f"decoder.layers.{i}.mlp.experts.linear_fc1.weight0"] = torch.empty(1)

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
    assert store.skipped == ["model.audio_model.proj.weight", "model.vision_model.patch_embed.weight"]
    sd, consumed = conv.build_mcore_state_dict(_GroupedModel(16, 2), store, margs, fields, lambda *a: None)
    assert consumed == store.all_keys()  # every text tensor is mapped; vision / audio exempt

    P = conv.HF_PREFIX + "layers.1."
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
                t[f"{conv.HF_PREFIX}layers.{i}.{src}"].float(),
                check_dtype=False,
            )
    torch.testing.assert_close(sd["decoder.final_layernorm.weight"], t[conv.HF_PREFIX + "norm.weight"])
    assert torch.equal(sd["output_layer.weight"][:10], t["lm_head.weight"])
    per_layer = len(one_to_one) + 1 + 2 * 3  # + qkv + fc1 / fc2 per expert
    assert len(sd) == 2 * per_layer + 3
    # vocab rows padded to the model's padded size; common.pad_vocab repeats the last
    # real row (as Megatron's converters do), the real rows are unchanged
    emb = sd["embedding.word_embeddings.weight"]
    assert emb.shape[0] == 16
    assert torch.equal(emb[:10], t["model.language_model.embed_tokens.weight"].to(emb.dtype))
    assert torch.all(emb[10:] == emb[9])


@pytest.mark.parametrize("prefix", ["model.language_model.", "model."])
def test_hf_prefix_detected_from_checkpoint_keys(prefix):
    # conditional-generation wrapper and text-only MuseSparkForCausalLM layouts
    t, margs, fields = _tiny_checkpoint()
    t = {(prefix + k[len(conv.HF_PREFIX) :] if k.startswith(conv.HF_PREFIX) else k): v for k, v in t.items()}
    if prefix == "model.":
        t = {k: v for k, v in t.items() if "vision_model" not in k and "audio_model" not in k}
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
    gate = torch.tensor([-1.0, 0.0, 2.0])
    alpha, beta = residual_gate_coeffs(gate, 0.3)
    w = gate / 0.3
    torch.testing.assert_close(beta, torch.sigmoid(w))
    torch.testing.assert_close(
        alpha, torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1 + torch.sigmoid(w)), min=1e-3))
    )


def test_text_entry_point_names_the_same_api():
    for name in conv_text.__all__:
        assert getattr(conv_text, name) is getattr(conv, name)


def test_converter_refuses_the_released_size(monkeypatch):
    fields = conv.derive_megatron_fields({"text_config": TEXT})
    n = conv.estimate_num_params(fields, TEXT["vocab_size"])
    assert 800e9 < n < 840e9  # ~816B (the model name)
    with pytest.raises(ValueError, match="single-process limit") as err:
        conv.check_convert_size(fields, {"text_config": TEXT})
    assert f"the {conv.MAX_PARAMS_KEY} config key overrides the limit" in str(err.value)
    assert "no converter route" in str(err.value)
    # a proxy (8 released layers) or the toy size passes a raised / the default limit
    small = dict(
        fields,
        num_layers=4,
        hidden_size=1024,
        num_attention_heads=16,
        num_query_groups=4,
        num_experts=16,
        moe_latent_size=512,
        moe_ffn_hidden_size=512,
    )
    assert conv.check_convert_size(small) < 1e9
    assert conv.check_convert_size(dict(fields, num_layers=8), max_params=200 * 10**9) > 60e9


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
    monkeypatch.setattr(conv, "build_margs", lambda _f, _a, _c=None: SimpleNamespace(num_layers=2))
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


def test_converter_size_cap_matches_the_memory_check():
    from primus.backends.megatron.core.models.muse_spark import memory_check as mc

    fields = conv.derive_megatron_fields({"text_config": TEXT})
    a = SimpleNamespace(**fields, attention_output_gate=True)
    assert conv.estimate_num_params(fields, TEXT["vocab_size"]) == sum(mc.param_split(a, TEXT["vocab_size"]))


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
    from primus.backends.megatron.patches.args import (
        muse_spark_recompute_cp_patches as m,
    )

    # EP 8 fits one 8-GPU node: pin the node size and the opt-in (the runner's env must
    # not decide the result)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    monkeypatch.delenv(m.INTERNODE_ENV, raising=False)

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
        ("4", 8, "1", True),  # rocSHMEM build opted in
        ("4", 8, "TRUE", True),
        ("4", 8, "0", False),
        ("8", 16, "yes", True),
    ],
)
def test_deepep_spanning_nodes_needs_the_internode_opt_in(monkeypatch, gpus, ep, opt_in, kept):
    from primus.backends.megatron.patches.args import (
        muse_spark_recompute_cp_patches as m,
    )

    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.setenv("GPUS_PER_NODE", gpus)
    if opt_in is None:
        monkeypatch.delenv(m.INTERNODE_ENV, raising=False)
    else:
        monkeypatch.setenv(m.INTERNODE_ENV, opt_in)
    args = SimpleNamespace(
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
        assert "spans nodes" in reason and f"{gpus} GPUs per node" in reason and m.INTERNODE_ENV in reason
    # an explicit local_gpus wins over the environment
    assert m.deepep_disable_reason(args, local_gpus=ep) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 60 * 10**9),
        ("", 60 * 10**9),
        (" 7_000 ", 7000),
        ("60e9", 60 * 10**9),
        ("1000", 1000),
        (1000, 1000),
    ],
)
def test_convert_max_params_parses_the_config_value(raw, expected):
    assert conv.parse_max_params(raw) == expected


@pytest.mark.parametrize("raw", ["60B", "-5", "0", "1.5", "nan", 0, -1, True, 1.5])
def test_convert_max_params_rejects_bad_values(raw):
    with pytest.raises(ValueError, match=conv.MAX_PARAMS_KEY):
        conv.parse_max_params(raw)
    with pytest.raises(ValueError, match=conv.MAX_PARAMS_KEY):
        conv.check_convert_size(conv.derive_megatron_fields({"text_config": TEXT}), max_params=raw)


def test_convert_size_cap_follows_the_config_value():
    fields = conv.derive_megatron_fields({"text_config": TEXT})
    n = conv.estimate_num_params(fields)
    assert conv.check_convert_size(fields, max_params=n + 1) == n
    with pytest.raises(ValueError, match="single-process limit"):
        conv.check_convert_size(fields, max_params=str(n - 1))


@pytest.mark.parametrize("key", ["attention_bias", "mlp_bias"])
def test_derive_fields_rejects_linear_biases(key):
    with pytest.raises(ValueError, match="not supported"):
        conv.derive_megatron_fields({**TEXT, key: True})
    assert conv.derive_megatron_fields({**TEXT, key: False})["add_qkv_bias"] is False


def test_dispatch_uses_model_type(monkeypatch):
    """The trainer reuses pretrain_gpt for ``PRETRAIN_GPT_MODEL_NAMES`` keys and
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
        MODEL_TYPES,
    )
    from primus.backends.megatron.megatron_pretrain_trainer import (
        PRETRAIN_GPT_MODEL_NAMES,
    )
    from primus.core.config.schema_check import _model_provider_branches
    from primus.core.utils import import_utils

    assert MODEL_TYPES == (MODEL_TYPE,)
    assert PRETRAIN_GPT_MODEL_NAMES[MODEL_TYPE] == "Muse-Spark"

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
    with pytest.raises(ValueError, match=conv.MAX_PARAMS_KEY):
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


@pytest.mark.parametrize(
    "name", ["muse_Spark_Open_816B_A42B.yaml", "muse_spark_816B_A42B_proxy.yaml", "muse_spark_toy.yaml"]
)
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
    except (OSError, ValueError, ImportError) as e:  # offline / no cached copy / gated (OSError subclasses)
        pytest.skip(f"tokenizer not available: {type(e).__name__}")
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


def test_one_exact_mapping_check_for_every_native_converter():
    # NativeConverter and load_validate_save (llama / qwen3 / deepseek) share one check
    from primus.backends.megatron.checkpoint import native_convert_common as common

    class Model:
        def __init__(self, missing=(), unexpected=()):
            self.r = (list(missing), list(unexpected))

        def load_state_dict(self, sd, strict):
            assert strict is False
            return self.r

    store = SimpleNamespace(all_keys=lambda: {"a", "b"})
    logs = []
    common.check_exact_mapping(Model(missing=["x._extra_state"]), {"k": 1}, {"a", "b"}, store, logs.append)
    assert logs[-1].startswith("mapping is COMPLETE and EXACT")
    for model, consumed, what in (
        (Model(missing=["w"]), {"a", "b"}, "missing_non_extra=1"),
        (Model(unexpected=["u"]), {"a", "b"}, "unexpected=1"),
        (Model(), {"a"}, "unconsumed=1"),
    ):
        with pytest.raises(RuntimeError, match=what):
            common.check_exact_mapping(model, {}, consumed, store, logs.append)
    assert not hasattr(__import__(conv.NativeConverter.__module__, fromlist=["x"]), "check_exact_mapping")


def test_load_validate_save_raises_before_any_save_on_a_bad_mapping(tmp_path, monkeypatch):
    # the llama / qwen3 / deepseek path: a bad mapping raises before the save dir is made or
    # save_checkpoint runs (no partial checkpoint); a good one saves once
    ckpt = pytest.importorskip("megatron.training.checkpointing")

    from primus.backends.megatron.checkpoint import native_convert_common as common

    saved = []
    monkeypatch.setattr(ckpt, "save_checkpoint", lambda *a, **k: saved.append(a))

    class Model:
        def __init__(self, missing=()):
            self.missing = list(missing)

        def load_state_dict(self, sd, strict):
            return self.missing, []

    store = SimpleNamespace(all_keys=lambda: {"a"})
    margs = SimpleNamespace(iteration=0)
    save_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="missing_non_extra=1"):
        common.load_validate_save(
            Model(missing=["w"]), {}, {"a"}, store, margs, str(save_dir), lambda m: None
        )
    assert saved == [] and not save_dir.exists()
    with pytest.raises(RuntimeError, match="unconsumed=1"):
        common.load_validate_save(Model(), {}, set(), store, margs, str(save_dir), lambda m: None)
    assert saved == [] and not save_dir.exists()
    common.load_validate_save(Model(), {}, {"a"}, store, margs, str(save_dir), lambda m: None)
    assert len(saved) == 1 and save_dir.is_dir()


def _fake_converter(tmp_path, monkeypatch, save, guard=None):
    """A NativeConverter whose steps are stubs: only the template's order and cleanup run."""
    from primus.backends.megatron.checkpoint import native_converter as nc

    common = nc.common
    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: None)
    monkeypatch.setattr(common, "read_hf_config", lambda _d: {})
    monkeypatch.setattr(common, "init_megatron_single_process", lambda m, _p, log=None: m)
    monkeypatch.setattr(common, "check_exact_mapping", lambda *a: None)
    if guard is not None:
        monkeypatch.setattr(common, "assert_bridge_free", guard)
    order = []

    class Fake(nc.NativeConverter):
        def derive_fields(self, cfg):
            return {}

        def check_config(self, fields, args, cfg):  # (fields, args, cfg), as build_margs
            order.append(("check_config", args.hf_dir))

        def build_margs(self, fields, args, cfg):
            return SimpleNamespace()

        def build_model(self, margs):
            return object()

        def open_store(self, hf_dir):
            return None

        def build_state_dict(self, model, store, margs, fields):
            return {}, set()

        def save(self, model, margs, save_dir):
            order.append("save")
            save(save_dir)

    return Fake(log=lambda *a: None), order


def _write_ckpt(save_dir):
    import os

    os.makedirs(os.path.join(save_dir, "iter_0000001"), exist_ok=True)
    with open(os.path.join(save_dir, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write("1")


def test_the_bridge_free_check_runs_before_the_save(tmp_path, monkeypatch):
    def guard(stage):
        raise AssertionError(f"megatron.bridge leaked ({stage})")

    c, order = _fake_converter(tmp_path, monkeypatch, _write_ckpt, guard)
    out = tmp_path / "out"
    with pytest.raises(AssertionError, match="leaked"):
        c.convert("hf", str(out))
    assert "save" not in order and not out.exists()  # nothing for skip-if-exists to accept


def test_a_failed_save_leaves_no_checkpoint(tmp_path, monkeypatch):
    def half_save(save_dir):
        _write_ckpt(save_dir)
        raise RuntimeError("disk full")

    c, order = _fake_converter(tmp_path, monkeypatch, half_save, lambda _s: None)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="disk full"):
        c.convert("hf", str(out))
    assert order == [("check_config", "hf"), "save"] and not out.exists()
    # a dir that was already there keeps its other files; only the tracker goes
    out.mkdir()
    (out / "user.txt").write_text("keep")
    with pytest.raises(RuntimeError):
        c.convert("hf", str(out))
    assert (out / "user.txt").exists() and not (out / "latest_checkpointed_iteration.txt").exists()


def test_a_partly_removed_checkpoint_keeps_no_tracker(tmp_path, monkeypatch):
    # rmtree fails part-way (EBUSY on NFS, ...): the tracker is gone first, and the log says so
    import shutil

    from primus.backends.megatron.checkpoint import native_converter as nc

    save = tmp_path / "ckpt"
    (save / "iter_0000001").mkdir(parents=True)
    (save / "latest_checkpointed_iteration.txt").write_text("1")
    monkeypatch.setattr(
        shutil,
        "rmtree",
        lambda p, onerror=None, **k: onerror(os.rmdir, str(p), (OSError, OSError(16, "busy"), None)),
    )
    logs = []
    nc.remove_partial_checkpoint(str(save), False, logs.append)
    assert not (save / "latest_checkpointed_iteration.txt").exists()
    assert "not all of the partial checkpoint" in logs[-1] and "busy" in logs[-1]


def test_a_failed_save_into_an_existing_dir_removes_only_what_it_added(tmp_path):
    from primus.backends.megatron.checkpoint import native_converter as nc

    save = tmp_path / "ckpt"
    save.mkdir()
    (save / "notes.txt").write_text("user file")
    before = nc.existing_entries(str(save))
    (save / "iter_0000001").mkdir()
    (save / "iter_0000001" / "__0_0.distcp").write_text("partial")
    (save / "latest_checkpointed_iteration.txt").write_text("1")
    logs = []
    nc.remove_partial_checkpoint(str(save), before, logs.append)
    assert sorted(p.name for p in save.iterdir()) == ["notes.txt"]
    assert "1 new entry " in logs[-1]  # the iter dir (the tracker is removed first)
    assert nc.existing_entries(str(tmp_path / "missing")) is None


def test_a_failed_save_into_an_empty_existing_dir_keeps_the_dir(tmp_path):
    # an empty pre-existing dir (mkdir-ed output dir, mount point) is set(), not None: the
    # save's entries go, the dir itself stays
    from primus.backends.megatron.checkpoint import native_converter as nc

    save = tmp_path / "out"
    save.mkdir()
    before = nc.existing_entries(str(save))
    assert before == set()
    (save / "iter_0000001").mkdir()
    (save / "latest_checkpointed_iteration.txt").write_text("1")
    (save / "common.pt").write_text("x")
    logs = []
    nc.remove_partial_checkpoint(str(save), before, logs.append)
    assert save.is_dir() and not any(save.iterdir())
    assert "the 2 new entries " in logs[-1]
    # created by this run (None): removed whole
    nc.remove_partial_checkpoint(str(save), None, logs.append)
    assert not save.exists()


def test_the_text_loader_has_no_old_store_alias():
    assert not hasattr(conv, "TextOnlyStore") and "TextOnlyStore" not in conv_text.__all__
