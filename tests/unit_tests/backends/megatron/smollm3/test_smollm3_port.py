###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU unit tests for the SmolLM3 port.

  1. native_loader_smollm3: HF ``no_rope_layers`` (1 = RoPE) -> Megatron
     ``no_rope_freq`` (1 = NoPE) inversion, the int/list collapse, input checks,
     the unsupported-feature errors and the eps override in build_margs.
  2. The checkpoint-convert hook routes model_type ``smollm3`` /
     ``SmolLM3ForCausalLM`` to the SmolLM3 converter.
  3. The preset matches the released config.json values.
  4. The converter's shared Llama driver, ``convert(family=...)``, runs this module's
     field functions (Megatron setup, model build and save are faked).

The skip_embedding_weight_decay patch is tested in
tests/unit_tests/backends/megatron/patches/test_embedding_weight_decay_patches.py.
Megatron-dependent tests use ``tests.utils.require_megatron`` (MEGATRON_PATH aware).
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from primus.backends.megatron.checkpoint import (  # noqa: E402
    native_loader_smollm3 as smol,
)
from tests.utils import require_megatron  # noqa: E402

REPO = Path(__file__).resolve().parents[5]


def _merged_preset(name):
    """The preset ``name`` with its ``extends`` chain merged (later files override)."""
    import yaml

    doc = yaml.safe_load((REPO / "primus/configs/models/megatron" / name).read_text()) or {}
    merged = {}
    for parent in doc.pop("extends", []) or []:
        merged.update(_merged_preset(parent))
    merged.update(doc)
    return merged


# HuggingFaceTB/SmolLM3-3B-Base@d78a42f79198 config.json (fields the converter reads)
SMOLLM3_CFG = {
    "architectures": ["SmolLM3ForCausalLM"],
    "model_type": "smollm3",
    "hidden_size": 2048,
    "intermediate_size": 11008,
    "num_hidden_layers": 36,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "max_position_embeddings": 65536,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000.0,
    "rope_scaling": None,
    "no_rope_layer_interval": 4,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "sliding_window": None,
    "use_sliding_window": False,
    "vocab_size": 128256,
}

NOPE_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]


# --------------------------------------------------------------------------- loader
def test_nope_pattern_from_interval_inverts_polarity():
    pattern = smol.nope_pattern(SMOLLM3_CFG)
    assert len(pattern) == 36
    assert [i for i, v in enumerate(pattern) if v] == NOPE_LAYERS


def test_nope_pattern_from_explicit_hf_list():
    # HF: 1 = RoPE, 0 = NoPE. Megatron: 1 = NoPE.
    cfg = {"num_hidden_layers": 4, "no_rope_layers": [1, 0, 1, 1]}
    assert smol.nope_pattern(cfg) == [0, 1, 0, 0]


def test_nope_pattern_none_when_all_rope():
    assert smol.nope_pattern({"num_hidden_layers": 4, "no_rope_layers": [1, 1, 1, 1]}) is None


def test_nope_pattern_unset_uses_hf_default_interval():
    # HF SmolLM3Config defaults no_rope_layer_interval to 4 when config.json omits both keys.
    assert smol.nope_pattern({"num_hidden_layers": 8}) == [0, 0, 0, 1, 0, 0, 0, 1]
    trimmed = {k: v for k, v in SMOLLM3_CFG.items() if k != "no_rope_layer_interval"}
    assert smol.nope_pattern(trimmed) == smol.nope_pattern(SMOLLM3_CFG)


def test_nope_pattern_explicit_list_wins_over_interval():
    # HF SmolLM3Config uses an explicit no_rope_layers verbatim and ignores the interval
    # (save_pretrained always writes the interval, default 4).
    base = {"num_hidden_layers": 4, "no_rope_layer_interval": 4}
    assert smol.nope_pattern({**base, "no_rope_layers": [1, 0, 1, 1]}) == [0, 1, 0, 0]
    assert smol.nope_pattern({**base, "no_rope_layers": [1, 1, 1, 1]}) is None
    # The interval is not validated when the list makes it irrelevant.
    cfg = {**base, "no_rope_layer_interval": 0, "no_rope_layers": [1, 1, 1, 0]}
    assert smol.nope_pattern(cfg) == [0, 0, 0, 1]


def test_nope_pattern_null_list_uses_interval_and_null_interval_is_rejected():
    # A null no_rope_layers is HF's "build from the interval" (missing interval = 4).
    assert smol.nope_pattern({"num_hidden_layers": 4, "no_rope_layers": None}) == [0, 0, 0, 1]
    # HF SmolLM3Config computes (i + 1) % None for an explicit null interval and fails.
    with pytest.raises(ValueError, match="no_rope_layer_interval"):
        smol.nope_pattern({"num_hidden_layers": 4, "no_rope_layer_interval": None})
    # ... also on the real convert path (with_hf_defaults must not fill the null with 4).
    with pytest.raises(ValueError, match="no_rope_layer_interval"):
        smol.derive_megatron_fields({**SMOLLM3_CFG, "no_rope_layer_interval": None})
    # With an explicit list the null interval is irrelevant, as in HF.
    cfg = smol.with_hf_defaults({**SMOLLM3_CFG, "no_rope_layer_interval": None, "no_rope_layers": [1] * 36})
    assert cfg["no_rope_layer_interval"] is None and smol.nope_pattern(cfg) is None


@pytest.mark.parametrize("interval", [0, -4, 1.5, "4", True])
def test_nope_pattern_rejects_bad_interval(interval):
    with pytest.raises(ValueError):
        smol.nope_pattern({"num_hidden_layers": 4, "no_rope_layer_interval": interval})


@pytest.mark.parametrize("bad", [[1, 1, 1, 0, 1, 1], [1, 1, 0]])
def test_nope_pattern_rejects_wrong_length(bad):
    with pytest.raises(ValueError):
        smol.nope_pattern({"num_hidden_layers": 4, "no_rope_layers": bad})


@pytest.mark.parametrize("bad", [[1, 2], [1, 1.5], [1, "1"], [1, None]])
def test_nope_pattern_rejects_non_binary(bad):
    with pytest.raises(ValueError):
        smol.nope_pattern({"num_hidden_layers": 2, "no_rope_layers": bad})


@pytest.mark.parametrize("bad", [[0, 2], [0, 1.0], [0, "1"]])
def test_megatron_no_rope_freq_rejects_non_binary(bad):
    with pytest.raises(ValueError):
        smol.megatron_no_rope_freq(bad)


def test_megatron_no_rope_freq_collapses_periodic_pattern():
    assert smol.megatron_no_rope_freq(smol.nope_pattern(SMOLLM3_CFG)) == 4
    assert smol.megatron_no_rope_freq([1, 1, 1]) == 1
    assert smol.megatron_no_rope_freq([0, 1, 0, 1, 0, 1]) == 2


def test_megatron_no_rope_freq_keeps_list_otherwise():
    assert smol.megatron_no_rope_freq([0, 1, 1, 0]) == [0, 1, 1, 0]
    # One period spanning the whole depth stays a list (both forms are valid for Megatron;
    # the int form is used only for a pattern that repeats).
    assert smol.megatron_no_rope_freq([0, 0, 0, 0, 0, 1]) == [0, 0, 0, 0, 0, 1]
    # 6 layers with interval 4: not periodic over the depth -> explicit list
    six = smol.nope_pattern({"num_hidden_layers": 6, "no_rope_layer_interval": 4})
    assert six == [0, 0, 0, 1, 0, 0]
    assert smol.megatron_no_rope_freq(six) == six
    assert smol.megatron_no_rope_freq(None) is None


def test_megatron_value_expands_back_to_pattern():
    pattern = smol.nope_pattern(SMOLLM3_CFG)
    assert smol.expand_no_rope_freq(smol.megatron_no_rope_freq(pattern), 36) == pattern


@pytest.mark.parametrize(
    "extra",
    [
        {"use_sliding_window": True, "sliding_window": 4096},
        {"layer_types": ["full_attention", "sliding_attention"] * 18},
        {"layer_types": ["chunked_attention"] * 36},
        {"rope_scaling": {"factor": 2.0}},  # no type: rejected, not guessed as default
        {"rope_scaling": "yarn"},
        {"mlp_bias": True},
        {"attention_bias": True},
        {"hidden_act": "gelu"},
        {"rope_scaling": {"type": "yarn", "factor": 2.0, "original_max_position_embeddings": 65536}},
        {"rope_scaling": {"rope_type": "yarn", "factor": 2.0}},
        {"rope_parameters": {"rope_type": "linear", "factor": 2.0}},
        # "default" with scaling parameters: rejected, not converted with plain RoPE
        {"rope_scaling": {"rope_type": "default", "factor": 2.0}},
        {"rope_parameters": {"type": "default", "rope_theta": 5e6, "factor": 2.0}},
    ],
)
def test_check_supported_rejects_unsupported_features(extra):
    # check_supported needs no Megatron, and derive_megatron_fields calls it first.
    with pytest.raises(NotImplementedError):
        smol.check_supported({**SMOLLM3_CFG, **extra})


@pytest.mark.parametrize(
    "extra",
    [
        {"rope_scaling": {"rope_type": "default"}},
        {"rope_parameters": {"rope_type": "default", "rope_theta": 5000000.0}},
        {"layer_types": ["full_attention"] * 36},
        {"sliding_window": 4096, "use_sliding_window": False},
        # HF builds sliding layers only when use_sliding_window AND sliding_window is set.
        {"use_sliding_window": True, "sliding_window": None},
        {"rope_parameters": {"rope_theta": 5000000.0}},
    ],
)
def test_check_supported_accepts_no_op_settings(extra):
    smol.check_supported({**SMOLLM3_CFG, **extra})


@pytest.mark.parametrize("key", ["rope_parameters", "rope_scaling"])
def test_check_supported_rejects_conflicting_rope_theta(key):
    with pytest.raises(ValueError, match="disagrees"):
        smol.check_supported({**SMOLLM3_CFG, key: {"rope_type": "default", "rope_theta": 1e4}})


@pytest.mark.parametrize("key", ["rope_parameters", "rope_scaling"])
def test_check_supported_takes_nested_rope_theta(key):
    """A rope_theta given only inside rope_parameters or rope_scaling is the base the
    converter builds with, not HF's class default 2e6."""
    cfg = {k: v for k, v in SMOLLM3_CFG.items() if k != "rope_theta"}
    resolved, _ = smol.check_supported({**cfg, key: {"rope_type": "default", "rope_theta": 1e6}})
    assert resolved["rope_theta"] == 1e6
    assert (
        smol.derive_megatron_fields({**cfg, key: {"rope_type": "default", "rope_theta": 1e6}})["rotary_base"]
        == 1e6
    )
    # Reviewer's repro: no top-level rope_theta, base only in rope_scaling.
    resolved, _ = smol.check_supported(
        {"num_hidden_layers": 8, key: {"rope_type": "default", "rope_theta": 1e6}}
    )
    assert resolved["rope_theta"] == 1e6


def test_check_supported_rejects_disagreeing_nested_thetas():
    cfg = {k: v for k, v in SMOLLM3_CFG.items() if k != "rope_theta"}
    with pytest.raises(ValueError, match="rope_scaling.rope_theta=3000000.0 disagrees"):
        smol.check_supported(
            {
                **cfg,
                "rope_parameters": {"rope_type": "default", "rope_theta": 1e6},
                "rope_scaling": {"rope_type": "default", "rope_theta": 3e6},
            }
        )


def test_with_hf_defaults_is_silent_on_a_full_config(recwarn):
    # The released config.json has no mlp_bias: the bias / activation defaults are not warned about.
    smol.with_hf_defaults({k: v for k, v in SMOLLM3_CFG.items() if k != "attention_bias"})
    assert not [w for w in recwarn if "omits" in str(w.message)]


def test_with_hf_defaults_follows_smollm3_config():
    # Missing keys get HF SmolLM3Config defaults, not the Llama converter's.
    dropped = ("rope_theta", "tie_word_embeddings", "rms_norm_eps")
    trimmed = {k: v for k, v in SMOLLM3_CFG.items() if k not in dropped}
    with pytest.warns(UserWarning, match=r"omits rms_norm_eps .*rope_theta \(using HF default 2000000.0\)"):
        cfg = smol.with_hf_defaults(trimmed)
    assert cfg["rope_theta"] == 2000000.0
    assert cfg["tie_word_embeddings"] is True
    assert cfg["rms_norm_eps"] == 1e-6
    # rope_parameters without a base: still the HF default, never None.
    cfg = smol.with_hf_defaults({**trimmed, "rope_parameters": {"rope_type": "default"}})
    assert cfg["rope_theta"] == 2000000.0
    with pytest.warns(UserWarning) as record:
        cfg = smol.with_hf_defaults(
            {**trimmed, "rope_parameters": {"rope_type": "default", "rope_theta": 3e6}}
        )
    assert cfg["rope_theta"] == 3e6
    # rope_theta came from rope_parameters: not reported as a fallback
    assert all("rope_theta" not in str(w.message) for w in record)


def test_with_hf_defaults_warns_on_missing_num_key_value_heads():
    trimmed = {k: v for k, v in SMOLLM3_CFG.items() if k != "num_key_value_heads"}
    with pytest.warns(UserWarning, match=r"omits num_key_value_heads \(using HF default 4\)"):
        smol.with_hf_defaults(trimmed)


@pytest.mark.parametrize("key", ["num_hidden_layers", "rms_norm_eps", "hidden_size", "tie_word_embeddings"])
def test_with_hf_defaults_rejects_explicit_null(key):
    # HF keeps an explicit null as None (it does not substitute its default), so a null
    # here is a config HF cannot build: rejected, not silently defaulted.
    with pytest.raises(ValueError, match=f"{key}=null"):
        smol.with_hf_defaults({**SMOLLM3_CFG, key: None})


def test_with_hf_defaults_null_special_cases():
    cfg = smol.with_hf_defaults({**SMOLLM3_CFG, "attention_bias": None, "mlp_bias": None})
    assert cfg["attention_bias"] is False and cfg["mlp_bias"] is False  # nn.Linear(bias=None)
    cfg = smol.with_hf_defaults({**SMOLLM3_CFG, "num_key_value_heads": None})
    assert cfg["num_key_value_heads"] == 16  # HF: null -> plain MHA
    with pytest.raises(ValueError, match="rope_theta=null"):
        smol.with_hf_defaults({**SMOLLM3_CFG, "rope_theta": None})
    cfg = smol.with_hf_defaults({**SMOLLM3_CFG, "rope_theta": None, "rope_parameters": {"rope_theta": 3e6}})
    assert cfg["rope_theta"] == 3e6


def test_expand_no_rope_freq_follows_megatron():
    # Literal patterns here; test_build_margs_cuts_pattern_to_num_layers and
    # test_megatron_nope_placement_matches_hf compare with TransformerConfig's own expansion.
    assert smol.expand_no_rope_freq(4, 8) == [0, 0, 0, 1, 0, 0, 0, 1]
    assert smol.expand_no_rope_freq([0, 1], 2) == [0, 1]
    assert smol.expand_no_rope_freq(None, 4) is None
    assert smol.expand_no_rope_freq([0, 0], 2) is None
    with pytest.raises(ValueError, match="does not divide"):
        smol.expand_no_rope_freq(4, 6)
    with pytest.raises(ValueError, match="bool"):
        smol.expand_no_rope_freq(True, 4)
    with pytest.raises(ValueError, match="entries"):
        smol.expand_no_rope_freq([0, 1], 4)
    # Length and entries are checked before the all-zero shortcut.
    with pytest.raises(ValueError, match="entries"):
        smol.expand_no_rope_freq([0, 0], 8)
    # [] is falsy: Megatron skips no_rope_freq for it, so it means no NoPE layer.
    assert smol.expand_no_rope_freq([], 4) is None
    with pytest.raises(ValueError, match="0/1"):
        smol.expand_no_rope_freq([0, 2, 0, 0], 4)
    assert smol.expand_no_rope_freq(0, 4) is None
    with pytest.raises(ValueError, match="positive"):
        smol.expand_no_rope_freq(-4, 8)
    for bad in (4.0, "4", {"a": 1}):
        with pytest.raises(ValueError, match="int or a list"):
            smol.expand_no_rope_freq(bad, 8)


def _model(num_layers, no_rope_freq):
    return SimpleNamespace(config=SimpleNamespace(num_layers=num_layers, no_rope_freq=no_rope_freq))


def test_check_model_nope():
    fields = {"no_rope_freq": [0, 0, 0, 1, 0, 0, 0, 1], "num_layers": 8, "hf_num_layers": 8}
    assert smol.check_model_nope(_model(8, 4), fields) == fields["no_rope_freq"]
    with pytest.raises(RuntimeError, match="no_rope_freq"):
        smol.check_model_nope(_model(8, None), fields)  # dropped arg -> all RoPE
    all_rope8 = {"no_rope_freq": None, "num_layers": 8, "hf_num_layers": 8}
    assert smol.check_model_nope(_model(8, None), all_rope8) is None
    # A model cut to its first 4 layers (layer-wise verifier) is checked on those layers.
    assert smol.check_model_nope(_model(4, 4), fields) == [0, 0, 0, 1]
    with pytest.raises(RuntimeError):
        smol.check_model_nope(_model(4, None), fields)
    # Deeper than config.json, or an int that Megatron could not expand: fail on that.
    with pytest.raises(RuntimeError, match="12 layers, config.json has 8"):
        smol.check_model_nope(_model(12, 4), fields)
    # The verifier lowers num_layers in the fields but keeps config.json's list.
    assert smol.check_model_nope(_model(4, 4), {**fields, "num_layers": 4}) == [0, 0, 0, 1]
    with pytest.raises(RuntimeError, match="invalid model.config.no_rope_freq"):
        smol.check_model_nope(_model(6, 4), fields)
    # A float or a wrong-length all-zero list is wrapped too, not escaped or accepted.
    with pytest.raises(RuntimeError, match="invalid model.config.no_rope_freq"):
        smol.check_model_nope(_model(8, 4.0), fields)
    all_rope = {"no_rope_freq": None, "num_layers": 8, "hf_num_layers": 8}
    with pytest.raises(RuntimeError, match="invalid model.config.no_rope_freq"):
        smol.check_model_nope(_model(8, [0, 0]), all_rope)
    # The depth guard works also when config.json has no NoPE layer (hf_num_layers).
    with pytest.raises(RuntimeError, match="12 layers, config.json has 8"):
        smol.check_model_nope(_model(12, None), all_rope)
    assert smol.check_model_nope(_model(4, None), {**all_rope, "num_layers": 4}) is None


def test_derive_fields_carries_hf_depth():
    fields = smol.derive_megatron_fields({**SMOLLM3_CFG, "no_rope_layers": [1] * 36})
    assert fields["no_rope_freq"] is None and fields["hf_num_layers"] == 36


def _margs_args():
    return SimpleNamespace(
        hf_dir="/nonexistent",
        save_dir="/nonexistent",
        dtype="fp32",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        seq_length=128,
        make_vocab_size_divisible_by=128,
    )


@pytest.mark.parametrize(
    "cfg_extra, depth, expected",
    [
        ({}, 6, [0, 0, 0, 1, 0, 0]),  # stock interval 4 cut to 6: list, not int 4
        ({}, 8, 4),  # cut to a multiple of the period: int form
        ({}, 2, None),  # no NoPE layer in the first 2 layers
        ({"no_rope_layers": [1, 0] + [1] * 34}, 4, [0, 1, 0, 0]),  # custom list cut to 4
    ],
)
def test_build_margs_cuts_pattern_to_num_layers(cfg_extra, depth, expected):
    require_megatron()
    fields = smol.derive_megatron_fields({**SMOLLM3_CFG, **cfg_extra})
    m = smol.build_margs({**fields, "num_layers": depth}, _margs_args())
    assert m.num_layers == depth
    assert m.no_rope_freq == expected
    # Megatron's own expansion of what build_margs gives is HF's pattern for the cut depth
    # (literal NoPE layers from config.json, not from the module under test).
    from megatron.core.transformer.transformer_config import TransformerConfig

    cfg = TransformerConfig(
        num_layers=depth, hidden_size=64, num_attention_heads=4, no_rope_freq=m.no_rope_freq
    )
    hf_nope = [0 if v else 1 for v in cfg_extra["no_rope_layers"][:depth]] if cfg_extra else None
    if hf_nope is None:
        hf_nope = [1 if i in NOPE_LAYERS else 0 for i in range(depth)]
    if any(hf_nope):
        assert list(cfg.no_rope_freq) == hf_nope
    else:
        assert not cfg.no_rope_freq


@pytest.mark.parametrize(
    "fields, depth",
    [
        ({"no_rope_freq": [0, 0, 0, 1], "hf_num_layers": 4, "num_layers": 4}, 8),
        ({"no_rope_freq": None, "hf_num_layers": 36, "num_layers": 36}, 40),
        ({"hf_num_layers": 4, "num_layers": 4}, 8),
        ({"no_rope_freq": None, "hf_num_layers": 0, "num_layers": 4}, 1),  # 0 is a depth, not "missing"
    ],
)
def test_nope_pattern_for_depth_rejects_deeper_than_config(fields, depth):
    with pytest.raises(ValueError, match="deeper"):
        smol.nope_pattern_for_depth(fields, depth)


def test_nope_pattern_for_depth_needs_a_depth():
    # hf_num_layers is required (derive_megatron_fields always sets it); no fallbacks.
    with pytest.raises(ValueError, match="no hf_num_layers"):
        smol.nope_pattern_for_depth({}, 2)
    with pytest.raises(ValueError, match="no hf_num_layers"):
        smol.nope_pattern_for_depth({"no_rope_freq": [0, 0, 0, 1], "num_layers": 4}, 4)
    with pytest.raises(ValueError, match="entries"):
        smol.nope_pattern_for_depth({"no_rope_freq": [0, 0, 0, 1], "hf_num_layers": 8}, 4)


def test_check_supported_sliding_window_follows_hf():
    n = SMOLLM3_CFG["num_hidden_layers"]
    sw = {"use_sliding_window": True, "sliding_window": 4096}
    # explicit all-full layer_types: HF builds full attention and ignores use_sliding_window
    smol.check_supported({**SMOLLM3_CFG, **sw, "layer_types": ["full_attention"] * n})
    # no NoPE layer: HF makes no layer sliding
    smol.check_supported({**SMOLLM3_CFG, **sw, "no_rope_layers": [1] * n})
    # default NoPE layers become sliding in HF
    with pytest.raises(NotImplementedError, match="sliding"):
        smol.check_supported({**SMOLLM3_CFG, **sw})
    with pytest.raises(NotImplementedError, match="sliding"):
        smol.check_supported(
            {**SMOLLM3_CFG, "layer_types": ["full_attention"] * (n - 1) + ["sliding_attention"]}
        )
    with pytest.raises(ValueError, match="entries"):
        smol.check_supported({**SMOLLM3_CFG, "layer_types": ["full_attention"] * (n - 1)})
    # A missing num_hidden_layers takes HF's default (36), so the length is still checked.
    no_depth = {k: v for k, v in SMOLLM3_CFG.items() if k != "num_hidden_layers"}
    smol.check_supported({**no_depth, "layer_types": ["full_attention"] * 36})
    with pytest.raises(ValueError, match="entries"):
        smol.check_supported({**no_depth, "layer_types": ["full_attention"] * 3})


@pytest.mark.parametrize("key", ["num_hidden_layers", "hidden_act"])
def test_check_supported_explicit_null_is_value_error(key):
    with pytest.raises(ValueError, match=f"{key}=null"):
        smol.check_supported({**SMOLLM3_CFG, key: None})


def test_derive_fields_and_build_margs():
    require_megatron()
    fields = smol.derive_megatron_fields(SMOLLM3_CFG)
    assert fields["num_layers"] == 36
    assert fields["num_query_groups"] == 4
    assert fields["rotary_base"] == 5000000.0
    assert fields["norm_epsilon"] == 1e-06
    assert fields["untie_embeddings_and_output_weights"] is False
    assert fields["hf_num_layers"] == 36
    # Full depth: the stock interval collapses to Megatron's int form (cut depths are
    # in test_build_margs_cuts_pattern_to_num_layers).
    m = smol.build_margs(fields, _margs_args())
    assert m.no_rope_freq == 4
    assert m.layernorm_epsilon == 1e-06
    assert m.rotary_base == 5000000.0
    assert m.untie_embeddings_and_output_weights is False


# Not 1e-5: that is Megatron's own default and would pass without the converter setting it.
@pytest.mark.parametrize("eps", [1e-6, 3e-6])
def test_llama_build_margs_sets_layernorm_epsilon(eps):
    """The shared Llama build_margs sets the attribute Megatron reads (layernorm_epsilon,
    the dest of --norm-epsilon), not the dead norm_epsilon."""
    require_megatron()
    from primus.backends.megatron.checkpoint import native_loader_llama as llama

    cfg = {k: v for k, v in SMOLLM3_CFG.items() if k not in ("model_type", "architectures")}
    fields = llama.derive_megatron_fields({**cfg, "model_type": "llama", "rms_norm_eps": eps})
    m = llama.build_margs(fields, _margs_args())
    assert m.layernorm_epsilon == eps
    assert not hasattr(m, "norm_epsilon")


_QWEN3_CFG = {
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1e6,
    "tie_word_embeddings": False,
}
_DEEPSEEK_CFG = {
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 128,
    "num_hidden_layers": 2,
    "first_k_dense_replace": 1,
    "num_attention_heads": 4,
    "vocab_size": 1024,
    "q_lora_rank": None,
    "kv_lora_rank": 64,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 16,
    "v_head_dim": 32,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000,
}


@pytest.mark.parametrize("family", ["qwen3", "deepseek"])
def test_qwen3_and_deepseek_build_margs_set_layernorm_epsilon(family, monkeypatch):
    """Like the Llama converter: build_margs sets layernorm_epsilon (what Megatron reads),
    so the converted model is built and validated at the HF rms_norm_eps, and it does not
    set the dead norm_epsilon. Runs on CPU: validate_args asks the GPU's compute capability
    when moe_grouped_gemm is on (DeepSeek), so that one call is stubbed."""
    require_megatron()
    import importlib

    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 4))
    conv = importlib.import_module(f"primus.backends.megatron.checkpoint.native_loader_{family}")
    cfg = _QWEN3_CFG if family == "qwen3" else _DEEPSEEK_CFG
    fields = conv.derive_megatron_fields(cfg)
    assert fields["norm_epsilon"] == 1e-6
    m = conv.build_margs(fields, _margs_args())  # any error fails the test (no skip)
    assert m.layernorm_epsilon == 1e-6
    assert not hasattr(m, "norm_epsilon")


def test_convert_checks_the_family_interface_first():
    """A family module without the converter functions is refused before Megatron starts."""
    import types

    from primus.backends.megatron.checkpoint import native_loader_llama as llama

    fam = types.ModuleType("native_loader_partial")
    fam.derive_megatron_fields = lambda cfg: {}
    with pytest.raises(TypeError, match="build_margs.*LOG_NAME"):
        llama.convert("/nonexistent", "/nonexistent-out", family=fam)
    for module in (llama, smol):  # both families meet the whole contract
        assert all(callable(getattr(module, f)) for f in llama._FAMILY_INTERFACE)
        assert isinstance(module.LOG_NAME, str)


def test_missing_epsilon_defaults():
    """config.json without rms_norm_eps: the shared Llama converter keeps its 1e-5
    fallback (unchanged by this port); the SmolLM3 converter uses HF SmolLM3Config's 1e-6."""
    from primus.backends.megatron.checkpoint import native_loader_llama as llama

    cfg = {k: v for k, v in SMOLLM3_CFG.items() if k != "rms_norm_eps"}
    assert llama.derive_megatron_fields(cfg)["norm_epsilon"] == 1e-5
    with pytest.warns(UserWarning, match="rms_norm_eps"):
        assert smol.derive_megatron_fields(cfg)["norm_epsilon"] == 1e-6


# --------------------------------------------------------------------------- hook
def _load_hook():
    path = REPO / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"
    spec = importlib.util.spec_from_file_location("convert_hook_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "cfg",
    [
        {"architectures": ["SmolLM3ForCausalLM"]},
        {"model_type": "smollm3"},
    ],
)
def test_hook_detects_smollm3(tmp_path, cfg):
    hook = _load_hook()
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert hook._detect_native_family(str(tmp_path)) == "smollm3"


def test_hook_dispatches_smollm3_to_its_converter(tmp_path, monkeypatch):
    """family smollm3 runs native_loader_smollm3.convert (Megatron setup is faked here;
    the conversion gates run it for real)."""
    hook = _load_hook()
    from primus.backends.megatron.checkpoint import native_convert_common as common
    from primus.backends.megatron.checkpoint import native_loader_smollm3 as conv

    calls = []
    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: None)
    monkeypatch.setattr(common, "apply_convert_patches", lambda **k: 0)
    monkeypatch.setattr(hook, "_native_dtype_from_config", lambda path, dtype: "bf16")
    monkeypatch.setattr(conv, "convert", lambda hf, out, **k: calls.append((hf, out)))
    hook.native_convert_checkpoint(str(tmp_path), str(tmp_path / "out"), {"family": "smollm3"})
    assert calls == [(str(tmp_path), str(tmp_path / "out"))]
    with pytest.raises(ValueError, match="no native converter for family 'smolm3'"):
        hook.native_convert_checkpoint(str(tmp_path), str(tmp_path / "out"), {"family": "smolm3"})


def test_hook_families_each_have_a_converter_module():
    """One registry: every family in _NATIVE_FAMILIES has primus...native_loader_<family>."""
    hook = _load_hook()
    ckpt = REPO / "primus/backends/megatron/checkpoint"
    for family in set(hook._NATIVE_FAMILIES.values()):
        assert (ckpt / f"native_loader_{family}.py").is_file(), family


# --------------------------------------------------------------------------- convert driver
class _DictStore:
    """Stands in for common.SafetensorsStore: HF tensors by name."""

    def __init__(self, tensors):
        self.tensors = tensors

    def has(self, name):
        return name in self.tensors

    def get(self, name, dtype):
        return self.tensors[name].to(dtype)


def _tiny_hf_tensors(layers, hidden, ffn, heads, kv_heads, vocab):
    head_dim = hidden // heads
    g = torch.Generator().manual_seed(0)

    def r(*shape):
        return torch.randn(*shape, generator=g)

    t = {"model.embed_tokens.weight": r(vocab, hidden), "model.norm.weight": r(hidden)}
    for i in range(layers):
        h = f"model.layers.{i}."
        t[h + "input_layernorm.weight"] = r(hidden)
        t[h + "post_attention_layernorm.weight"] = r(hidden)
        t[h + "self_attn.q_proj.weight"] = r(heads * head_dim, hidden)
        t[h + "self_attn.k_proj.weight"] = r(kv_heads * head_dim, hidden)
        t[h + "self_attn.v_proj.weight"] = r(kv_heads * head_dim, hidden)
        t[h + "self_attn.o_proj.weight"] = r(hidden, heads * head_dim)
        t[h + "mlp.gate_proj.weight"] = r(ffn, hidden)
        t[h + "mlp.up_proj.weight"] = r(ffn, hidden)
        t[h + "mlp.down_proj.weight"] = r(hidden, ffn)
    return t


@pytest.mark.parametrize("break_nope", [False, True])
def test_convert_with_smollm3_family_runs_smollm3_fields(tmp_path, monkeypatch, break_nope):
    """native_loader_smollm3.convert drives the shared Llama sequence with the SmolLM3
    functions: derive_megatron_fields (rope_theta, eps, NoPE pattern from config.json),
    build_margs (real Megatron argument parsing and validate_args) and
    build_mcore_state_dict (the NoPE check on the built model, then the weight mapping).
    Only the heavy common.* steps (process-group init, model build, save) are faked.
    A model built without the NoPE pattern stops the conversion before the save."""
    require_megatron()
    from megatron.core.transformer.transformer_config import TransformerConfig

    from primus.backends.megatron.checkpoint import native_convert_common as common

    L, H, F, NH, KV, V = 8, 64, 128, 4, 2, 200
    cfg = {
        **SMOLLM3_CFG,
        "num_hidden_layers": L,
        "hidden_size": H,
        "intermediate_size": F,
        "num_attention_heads": NH,
        "num_key_value_heads": KV,
        "vocab_size": V,
    }
    tensors = _tiny_hf_tensors(L, H, F, NH, KV, V)
    seen = {}

    def build_model(margs):
        seen["margs"] = margs
        no_rope = None if break_nope else margs.no_rope_freq
        config = TransformerConfig(
            num_layers=margs.num_layers,
            hidden_size=margs.hidden_size,
            num_attention_heads=margs.num_attention_heads,
            num_query_groups=margs.num_query_groups,
            no_rope_freq=no_rope,
        )
        padded = 256
        sd = {"embedding.word_embeddings.weight": torch.zeros(padded, H)}
        return SimpleNamespace(config=config, state_dict=lambda: sd)

    def save(model, sd, consumed, store, margs, save_dir, log):
        seen["sd"], seen["consumed"], seen["save_dir"] = sd, consumed, save_dir

    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: None)
    monkeypatch.setattr(common, "assert_bridge_free", lambda stage: None)
    monkeypatch.setattr(common, "read_hf_config", lambda hf_dir: dict(cfg))
    monkeypatch.setattr(common, "init_megatron_single_process", lambda margs, port, log=None: margs)
    monkeypatch.setattr(common, "build_mcore_gpt_model", build_model)
    monkeypatch.setattr(common, "SafetensorsStore", lambda hf_dir: _DictStore(tensors))
    monkeypatch.setattr(common, "load_validate_save", save)

    logs = []
    kwargs = dict(dtype="fp32", seq_length=128, log=lambda *a: logs.append(" ".join(map(str, a))))
    if break_nope:
        with pytest.raises(RuntimeError, match="no_rope_freq expands to"):
            smol.convert(str(tmp_path), str(tmp_path / "out"), **kwargs)
        assert "sd" not in seen
        return
    assert smol.convert(str(tmp_path), str(tmp_path / "out"), **kwargs) == str(tmp_path / "out")

    margs = seen["margs"]
    assert margs.layernorm_epsilon == 1e-6  # config.json rms_norm_eps, not Megatron's 1e-5
    assert margs.rotary_base == 5000000.0
    assert margs.no_rope_freq == 4
    assert margs.num_layers == L and margs.num_query_groups == KV
    assert seen["save_dir"] == str(tmp_path / "out")
    assert seen["consumed"] == set(tensors)  # every HF tensor mapped
    sd = seen["sd"]
    assert len([k for k in sd if k.startswith("decoder.layers.")]) == 6 * L
    emb = sd["embedding.word_embeddings.weight"]
    torch.testing.assert_close(emb[:V], tensors["model.embed_tokens.weight"])
    assert any("NoPE layers = [3, 7]" in line for line in logs)
    assert any("layernorm_epsilon=1e-06" in line for line in logs)


# --------------------------------------------------------------------------- preset
def test_preset_matches_config_json():
    preset = _merged_preset("smollm3_3B_Base.yaml")
    assert preset["num_layers"] == SMOLLM3_CFG["num_hidden_layers"]
    assert preset["hidden_size"] == SMOLLM3_CFG["hidden_size"]
    assert preset["ffn_hidden_size"] == SMOLLM3_CFG["intermediate_size"]
    assert preset["num_attention_heads"] == SMOLLM3_CFG["num_attention_heads"]
    assert preset["num_query_groups"] == SMOLLM3_CFG["num_key_value_heads"]
    assert preset["kv_channels"] == 128
    assert float(preset["rotary_base"]) == SMOLLM3_CFG["rope_theta"]
    assert float(preset["layernorm_epsilon"]) == SMOLLM3_CFG["rms_norm_eps"]
    # norm_epsilon is not read by Megatron for this model, but config tools read it
    # (merged from llama3_base.yaml); it must not show a value the model does not use.
    assert float(preset["norm_epsilon"]) == float(preset["layernorm_epsilon"])
    assert preset["untie_embeddings_and_output_weights"] is False
    assert smol.expand_no_rope_freq(preset["no_rope_freq"], 36) == smol.nope_pattern(SMOLLM3_CFG)


@pytest.mark.parametrize("num_layers", [4, 8, 36])
def test_megatron_nope_placement_matches_hf(num_layers):
    """Per-PR NoPE placement check: Megatron's TransformerConfig expands the preset's
    no_rope_freq: 4 to the same per-layer pattern as HF SmolLM3 (no_rope_layer_interval 4:
    every 4th layer, 1-based, skips RoPE)."""
    require_megatron()
    from megatron.core.transformer.transformer_config import TransformerConfig

    preset = _merged_preset("smollm3_3B_Base.yaml")
    cfg = TransformerConfig(
        num_layers=num_layers,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=1,
        no_rope_freq=preset["no_rope_freq"],
    )
    hf = smol.nope_pattern({**SMOLLM3_CFG, "num_hidden_layers": num_layers})
    assert list(cfg.no_rope_freq) == list(hf)
    # HF: layer i (0-based) uses RoPE unless (i + 1) % interval == 0.
    assert [i for i, v in enumerate(cfg.no_rope_freq) if v] == list(range(3, num_layers, 4))


# --------------------------------------------------------------------------- config -> args (per PR)
def _runtime_backend_args(tmp_path, monkeypatch, config, overrides=()):
    """The runtime's own config path for ``config``: _initialize_configuration (YAML,
    preset, module defaults, CLI overrides), then _initialize_trainer with the real
    MegatronAdapter.convert_config (Megatron's argparse defaults) and the merge of the
    Primus-only params. Backend setup, the build_args patches and the trainer class are
    faked (no GPU, no distributed). Returns the args the trainer is built with and the
    module config that the before_train patch conditions read."""
    import argparse
    from unittest.mock import patch

    from primus.backends.megatron.megatron_adapter import MegatronAdapter
    from primus.core.runtime.train_runtime import PrimusRuntime

    monkeypatch.setenv("PRIMUS_WORKSPACE", str(tmp_path / "output"))  # the experiment dirs
    built = {}

    class Trainer:
        def __init__(self, backend_args=None, **kwargs):
            built["backend_args"] = backend_args

    class Adapter(MegatronAdapter):
        def prepare_backend(self, module_config):
            pass

        def load_trainer_class(self, stage="pretrain", trainer_class=None):
            return Trainer

    runtime = PrimusRuntime(
        args=argparse.Namespace(config=str(REPO / config), data_path=str(tmp_path), backend_path=None)
    )
    runtime._initialize_configuration("pre_trainer", list(overrides))
    runtime.ctx.adapter = Adapter()
    with patch.object(PrimusRuntime, "_run_phase_patches"):
        runtime._initialize_trainer()
    return built["backend_args"], runtime.ctx.module_config


@pytest.mark.parametrize("platform", ["MI300X", "MI325X", "MI355X"])
def test_recipe_flags_reach_the_megatron_args(tmp_path, monkeypatch, platform):
    """Per-PR check of the config binding (the trainer E2E case is weekly): the example
    recipe's skip_embedding_weight_decay and reset_log_on_first_iteration reach the args
    that Megatron's parse_args returns (MegatronBaseTrainer patches parse_args to return
    backend_args), the before_train patch conditions see them, and the embedding patch's
    own backend-args checks pass. A CLI override turns the flag off end to end."""
    require_megatron()
    from primus.backends.megatron.patches import embedding_weight_decay_patches as ew
    from primus.backends.megatron.patches.training_log import (
        first_iteration_reset_patches as rp,
    )
    from primus.core.patches import PatchContext

    config = f"examples/megatron/configs/{platform}/smollm3_3B_Base-BF16-pretrain.yaml"
    args, module_config = _runtime_backend_args(tmp_path, monkeypatch, config)
    assert args.skip_embedding_weight_decay is True
    assert args.reset_log_on_first_iteration is True
    assert args.layernorm_epsilon == pytest.approx(1e-6) and args.no_rope_freq == 4
    ctx = PatchContext(backend="megatron", phase="before_train", extra={"module_config": module_config})
    assert ew.skip_embedding_wd_enabled(ctx) and rp.reset_log_on_first_iteration_enabled(ctx)
    assert ew.skip_embedding_wd_in_args(args) and ew.unsupported_reason(args) is None

    off, module_config = _runtime_backend_args(
        tmp_path, monkeypatch, config, ["skip_embedding_weight_decay=false"]
    )
    ctx = PatchContext(backend="megatron", phase="before_train", extra={"module_config": module_config})
    assert off.skip_embedding_weight_decay is False and not ew.skip_embedding_wd_enabled(ctx)
