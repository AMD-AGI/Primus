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
  4. The converter's ``convert`` (the Llama conversion steps) runs this module's
     field functions (Megatron setup, model build and save are faked).
  5. NoPE placement: on the config, and in a forward pass of a small GPTModel.

Megatron-dependent tests call ``require_megatron_training`` (skip when megatron.core is not
importable, as other Megatron unit tests do with ``pytest.importorskip``).
"""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from primus.backends.megatron.checkpoint import (  # noqa: E402
    native_loader_smollm3 as smol,
)
from tests.unit_tests.backends.megatron.checkpoint.native_loader_test_utils import (  # noqa: E402
    llama_layout_hf_tensors,
    load_convert_hook,
    margs_args,
    require_megatron_training,
    stub_convert_pipeline,
)

# The repository root (tests/unit_tests/backends/megatron/smollm3/ is five levels down).
REPO = Path(__file__).resolve().parents[5]


def _merged_preset(name):
    """The preset ``name`` as training loads it (``PresetLoader``: extends, deep merge, env)."""
    from primus.core.config.preset_loader import PresetLoader

    return PresetLoader.load(name, "megatron", config_type="models")


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
    assert pattern == [0, 0, 0, 1] * 9
    assert [i for i, v in enumerate(pattern) if v] == NOPE_LAYERS
    assert smol.nope_pattern({"num_hidden_layers": 6, "no_rope_layer_interval": 4}) == [0, 0, 0, 1, 0, 0]


@pytest.mark.parametrize(
    "extra, rope_layers, expected",
    [
        ({}, [1, 0, 1, 1], [0, 1, 0, 0]),  # HF: 1 = RoPE, 0 = NoPE. Megatron: 1 = NoPE.
        ({}, [1, 1, 1, 1], None),  # all RoPE
        # HF SmolLM3Config uses an explicit no_rope_layers verbatim and ignores the interval
        # (save_pretrained always writes it); the interval is then not validated either.
        ({"no_rope_layer_interval": 4}, [1, 0, 1, 1], [0, 1, 0, 0]),
        ({"no_rope_layer_interval": 0}, [1, 1, 1, 0], [0, 0, 0, 1]),
    ],
)
def test_nope_pattern_from_explicit_hf_list(extra, rope_layers, expected):
    assert smol.nope_pattern({"num_hidden_layers": 4, "no_rope_layers": rope_layers, **extra}) == expected


def test_nope_pattern_unset_uses_hf_default_interval():
    # HF SmolLM3Config defaults no_rope_layer_interval to 4 when config.json omits both keys.
    assert smol.nope_pattern({"num_hidden_layers": 8}) == [0, 0, 0, 1, 0, 0, 0, 1]
    trimmed = {k: v for k, v in SMOLLM3_CFG.items() if k != "no_rope_layer_interval"}
    assert smol.nope_pattern(trimmed) == smol.nope_pattern(SMOLLM3_CFG)


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


@pytest.mark.parametrize("interval", [0, -4, 1.5, 4.5, "4", True])
def test_nope_pattern_rejects_bad_interval(interval):
    with pytest.raises(ValueError, match="no_rope_layer_interval"):
        smol.nope_pattern({"num_hidden_layers": 4, "no_rope_layer_interval": interval})


@pytest.mark.parametrize("bad", [[1, 1, 1, 0, 1, 1], [1, 1, 0], 5, "1101"])
def test_nope_pattern_rejects_wrong_length(bad):
    with pytest.raises(ValueError, match="no_rope_layers"):
        smol.nope_pattern({"num_hidden_layers": 4, "no_rope_layers": bad})


@pytest.mark.parametrize("layers", [0, -1, 4.0, True])
def test_nope_pattern_rejects_bad_depth(layers):
    with pytest.raises(ValueError, match="num_hidden_layers"):
        smol.nope_pattern({"num_hidden_layers": layers})


def test_null_vocab_size_is_ignored():
    """vocab_size is not used (Megatron sizes the vocab from the tokenizer)."""
    smol.check_supported({**SMOLLM3_CFG, "vocab_size": None})


@pytest.mark.parametrize("bad", [[1, 2], [1, 1.5], [1, "1"], [1, None]])
def test_nope_pattern_rejects_non_binary(bad):
    with pytest.raises(ValueError):
        smol.nope_pattern({"num_hidden_layers": 2, "no_rope_layers": bad})


@pytest.mark.parametrize("bad", [[0, 2], [0, 1.0], [0, "1"]])
def test_megatron_no_rope_freq_rejects_non_binary(bad):
    with pytest.raises(ValueError):
        smol.megatron_no_rope_freq(bad)


@pytest.mark.parametrize(
    "pattern, expected",
    [
        # A pattern that repeats over the depth collapses to its int period.
        ([0, 0, 0, 1] * 9, 4),
        ([1, 1, 1], 1),
        ([1, 1], 1),
        ([0, 1, 0, 1, 0, 1], 2),
        # A one-layer pattern, or one period spanning the whole depth, stays a list
        # (both forms are valid for Megatron; the int form is used only for a repeat).
        ([1], [1]),
        ([0, 0, 1], [0, 0, 1]),
        ([0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]),
        # Not periodic over the depth (6 layers with interval 4): an explicit list.
        ([0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]),
        ([0, 1, 1, 0], [0, 1, 1, 0]),
        (None, None),
    ],
)
def test_megatron_no_rope_freq_int_or_list(pattern, expected):
    assert smol.megatron_no_rope_freq(pattern) == expected


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
        {"hidden_act": "swish"},  # HF ACT2FN maps "swish" to the same SiLU
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


def test_nope_pattern_accepts_what_hf_accepts():
    """HF computes (i + 1) % interval and tests no_rope_layers entries for truth: an
    integral float interval and 0.0/1.0 (or bool) entries build the same layers."""
    n = 8
    assert smol.nope_pattern({"num_hidden_layers": n, "no_rope_layer_interval": 4.0}) == smol.nope_pattern(
        {"num_hidden_layers": n, "no_rope_layer_interval": 4}
    )
    rope = [1.0, 1.0, 1.0, 0.0] * 2
    assert smol.nope_pattern({"num_hidden_layers": n, "no_rope_layers": rope}) == [0, 0, 0, 1] * 2
    assert (
        smol.nope_pattern({"num_hidden_layers": n, "no_rope_layers": [bool(r) for r in rope]})
        == [0, 0, 0, 1] * 2
    )
    with pytest.raises(ValueError):
        smol.nope_pattern({"num_hidden_layers": 2, "no_rope_layers": [0.5, 1.0]})
    # The check of a built model stays strict: no float entries there.
    with pytest.raises(ValueError):
        smol.expand_no_rope_freq([0.0, 1.0], 2)
    assert smol.expand_no_rope_freq([False, True], 2) == [0, 1]


def test_check_supported_rope_type_null_falls_back_to_type():
    smol.check_supported({**SMOLLM3_CFG, "rope_scaling": {"rope_type": None, "type": "default"}})
    with pytest.raises(NotImplementedError):
        smol.check_supported(
            {**SMOLLM3_CFG, "rope_scaling": {"rope_type": None, "type": "yarn", "factor": 2.0}}
        )


@pytest.mark.parametrize("theta", ["5e6", True, -1.0, [5e6], float("inf"), float("nan")])
def test_non_numeric_rope_theta_is_a_clear_error(theta):
    with pytest.raises(ValueError, match="not a positive finite number"):
        smol.check_supported({**SMOLLM3_CFG, "rope_theta": theta})
    cfg = {k: v for k, v in SMOLLM3_CFG.items() if k != "rope_theta"}
    with pytest.raises(ValueError, match="not a positive finite number"):
        smol.check_supported({**cfg, "rope_parameters": {"rope_type": "default", "rope_theta": theta}})


def test_missing_key_warning_goes_to_the_convert_log(recwarn):
    """convert passes its [smollm3-convert] log: the missing-key warning goes there (not
    to warnings.warn); vocab_size is not warned about (the tokenizer sizes the vocab)."""
    lines = []
    cfg = {k: v for k, v in SMOLLM3_CFG.items() if k not in ("rope_theta", "vocab_size")}
    smol.derive_megatron_fields(cfg, log=lines.append)
    assert len(lines) == 1 and lines[0].startswith("WARNING: SmolLM3 config.json omits rope_theta")
    assert "vocab_size" not in lines[0]
    assert not [w for w in recwarn if "omits" in str(w.message)]
    smol.with_hf_defaults({k: v for k, v in SMOLLM3_CFG.items() if k != "vocab_size"})
    assert not [w for w in recwarn if "omits" in str(w.message)]


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
    fields = {"nope_layers": [0, 0, 0, 1, 0, 0, 0, 1], "num_layers": 8, "hf_num_layers": 8}
    assert smol.check_model_nope(_model(8, 4), fields) == fields["nope_layers"]
    with pytest.raises(RuntimeError, match="no_rope_freq"):
        smol.check_model_nope(_model(8, None), fields)  # dropped arg -> all RoPE
    all_rope8 = {"nope_layers": None, "num_layers": 8, "hf_num_layers": 8}
    assert smol.check_model_nope(_model(8, None), all_rope8) is None
    # A model cut to its first 4 layers (num_layers lowered) is checked on those layers.
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
    all_rope = {"nope_layers": None, "num_layers": 8, "hf_num_layers": 8}
    with pytest.raises(RuntimeError, match="invalid model.config.no_rope_freq"):
        smol.check_model_nope(_model(8, [0, 0]), all_rope)
    # The depth guard works also when config.json has no NoPE layer (hf_num_layers).
    with pytest.raises(RuntimeError, match="12 layers, config.json has 8"):
        smol.check_model_nope(_model(12, None), all_rope)
    assert smol.check_model_nope(_model(4, None), {**all_rope, "num_layers": 4}) is None


def test_derive_fields_carries_hf_depth():
    fields = smol.derive_megatron_fields({**SMOLLM3_CFG, "no_rope_layers": [1] * 36})
    assert fields["nope_layers"] is None and fields["hf_num_layers"] == 36


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
    require_megatron_training()
    fields = smol.derive_megatron_fields({**SMOLLM3_CFG, **cfg_extra})
    m = smol.build_margs({**fields, "num_layers": depth}, margs_args())
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
        ({"nope_layers": [0, 0, 0, 1], "hf_num_layers": 4, "num_layers": 4}, 8),
        ({"nope_layers": None, "hf_num_layers": 36, "num_layers": 36}, 40),
        ({"hf_num_layers": 4, "num_layers": 4}, 8),
        ({"nope_layers": None, "hf_num_layers": 0, "num_layers": 4}, 1),  # 0 is a depth, not "missing"
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
        smol.nope_pattern_for_depth({"nope_layers": [0, 0, 0, 1], "num_layers": 4}, 4)
    with pytest.raises(ValueError, match="entries"):
        smol.nope_pattern_for_depth({"nope_layers": [0, 0, 0, 1], "hf_num_layers": 8}, 4)


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
    require_megatron_training()
    fields = smol.derive_megatron_fields(SMOLLM3_CFG)
    assert fields["num_layers"] == 36
    assert fields["num_query_groups"] == 4
    assert fields["rotary_base"] == 5000000.0
    assert fields["norm_epsilon"] == 1e-06
    assert fields["untie_embeddings_and_output_weights"] is False
    assert fields["hf_num_layers"] == 36
    # Full depth: the stock interval collapses to Megatron's int form (cut depths are
    # in test_build_margs_cuts_pattern_to_num_layers).
    m = smol.build_margs(fields, margs_args())
    assert m.no_rope_freq == 4
    assert m.layernorm_epsilon == 1e-06
    assert m.rotary_base == 5000000.0
    assert m.untie_embeddings_and_output_weights is False


def test_missing_epsilon_defaults():
    """config.json without rms_norm_eps: the shared Llama converter keeps its 1e-5
    fallback (unchanged by this port); the SmolLM3 converter uses HF SmolLM3Config's 1e-6."""
    from primus.backends.megatron.checkpoint import native_loader_llama as llama

    cfg = {k: v for k, v in SMOLLM3_CFG.items() if k != "rms_norm_eps"}
    assert llama.derive_megatron_fields(cfg)["norm_epsilon"] == 1e-5
    with pytest.warns(UserWarning, match="rms_norm_eps"):
        assert smol.derive_megatron_fields(cfg)["norm_epsilon"] == 1e-6


# --------------------------------------------------------------------------- hook
def test_build_margs_without_ckpt_format():
    """A build-and-load caller may pass args without ckpt_format: convert's default is used,
    and an explicit value is kept."""
    require_megatron_training()
    from types import SimpleNamespace

    from primus.backends.megatron.checkpoint import native_convert_common as common

    fields = smol.derive_megatron_fields(SMOLLM3_CFG)
    args = {k: v for k, v in vars(margs_args()).items() if k != "ckpt_format"}
    assert smol.build_margs(fields, SimpleNamespace(**args)).ckpt_format == common.DEFAULT_CKPT_FORMAT
    assert smol.build_margs(fields, margs_args()).ckpt_format == "torch"


@pytest.mark.parametrize(
    "cfg",
    [
        {"architectures": ["SmolLM3ForCausalLM"]},
        {"model_type": "smollm3"},
    ],
)
def test_hook_detects_smollm3(tmp_path, cfg):
    hook = load_convert_hook()
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert hook._detect_native_family(str(tmp_path)) == "smollm3"


def test_registry_selects_smollm3_for_released_config():
    """native_families/smollm3.py registers the family for the checkpoint's config.json."""
    from primus.backends.megatron.checkpoint import native_registry

    fam = native_registry.family_for_config(
        {"architectures": ["SmolLM3ForCausalLM"], "model_type": "smollm3"}
    )
    assert fam is not None and fam.name == "smollm3"
    assert fam.converter == "primus.backends.megatron.checkpoint.native_loader_smollm3"
    assert fam.model_type == "gpt"
    assert native_registry.family_for_config({"model_type": "smollm3"}).name == "smollm3"
    assert "SmolLM3" in native_registry.supported_text()
    assert fam.converter_module().convert is not None


def test_hook_dispatches_smollm3_to_its_converter(tmp_path, monkeypatch):
    """family smollm3 runs native_loader_smollm3.convert with the hook's layout and the
    family's convert_kwargs (Megatron setup is faked here; the conversion gates run it for real)."""
    hook = load_convert_hook()
    from primus.backends.megatron.checkpoint import native_convert_common as common
    from primus.backends.megatron.checkpoint import native_loader_smollm3 as conv

    calls = []
    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: None)
    monkeypatch.setattr(common, "apply_convert_patches", lambda **k: 0)
    monkeypatch.setattr(hook, "_native_dtype_from_config", lambda path, dtype, family=None: "bf16")
    monkeypatch.setattr(conv, "convert", lambda hf, out, **k: calls.append((hf, out, k)))
    hook.native_convert_checkpoint(
        str(tmp_path),
        str(tmp_path / "out"),
        {"family": "smollm3", "tensor_parallel_size": 2, "convert_kwargs": {"probe": 1}},
    )
    assert len(calls) == 1
    hf, out, kwargs = calls[0]
    assert (hf, out) == (str(tmp_path), str(tmp_path / "out"))
    assert kwargs["tensor_parallel_size"] == 2 and kwargs["dtype"] == "bf16" and kwargs["probe"] == 1


# --------------------------------------------------------------------------- convert driver
@pytest.mark.parametrize("broken", [None, "no_rope_freq", "layernorm_epsilon"])
def test_convert_with_smollm3_family_runs_smollm3_fields(tmp_path, monkeypatch, broken):
    """native_loader_smollm3.convert runs the Llama conversion sequence with the SmolLM3
    functions: derive_megatron_fields (rope_theta, eps, NoPE pattern from config.json),
    build_margs (real Megatron argument parsing and validate_args) and
    build_mcore_state_dict (the NoPE and epsilon checks on the built model, then the weight
    mapping). The model's TransformerConfig is built from the args by Megatron's own
    ``core_transformer_config_from_args`` (as gpt_builder does), so a field lost on the way from
    args to config fails the checks. Only process-group init, the module build and the save are
    faked. ``broken``: that arg is reset to Megatron's default before the config is built; the
    conversion must stop before the save."""
    require_megatron_training()
    import argparse

    from megatron.training.argument_utils import core_transformer_config_from_args
    from megatron.training.arguments import add_megatron_arguments

    defaults = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False)).parse_args([])

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
    tensors = llama_layout_hf_tensors(L, H, F, NH, KV, V)
    seen = {}

    def build_model(margs):
        seen["margs"] = margs
        args = SimpleNamespace(**vars(margs))
        if broken:
            setattr(args, broken, getattr(defaults, broken))
        config = core_transformer_config_from_args(args)
        padded = 256
        sd = {"embedding.word_embeddings.weight": torch.zeros(padded, H)}
        return SimpleNamespace(config=config, state_dict=lambda: sd)

    def save(model, sd, consumed, store, margs, save_dir, log):
        seen["sd"], seen["consumed"], seen["save_dir"] = sd, consumed, save_dir

    stub_convert_pipeline(monkeypatch, cfg, tensors, build_model, save)

    logs = []
    kwargs = dict(dtype="fp32", seq_length=128, log=lambda *a: logs.append(" ".join(map(str, a))))
    if broken:
        match = {"no_rope_freq": "no_rope_freq expands to", "layernorm_epsilon": "layernorm_epsilon=1e-05"}
        with pytest.raises(RuntimeError, match=match[broken]):
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
    # Tied embeddings: the head uses the embedding; the checkpoint has no lm_head to map.
    assert margs.untie_embeddings_and_output_weights is False
    assert "output_layer.weight" not in sd
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
    # norm_epsilon is not read by Megatron for this model; the merged config must not
    # show llama3_base.yaml's 1e-5 next to the value in use.
    assert float(preset["norm_epsilon"]) == float(preset["layernorm_epsilon"])
    assert preset["untie_embeddings_and_output_weights"] is False
    assert smol.expand_no_rope_freq(preset["no_rope_freq"], 36) == smol.nope_pattern(SMOLLM3_CFG)


@pytest.mark.parametrize("num_layers", [4, 8, 36])
def test_megatron_nope_placement_matches_hf(num_layers):
    """Per-PR NoPE config check: Megatron's TransformerConfig expands the preset's
    no_rope_freq: 4 to the same per-layer pattern as HF SmolLM3 (no_rope_layer_interval 4:
    every 4th layer, 1-based, skips RoPE).

    This checks the config only. That attention then skips RoPE on those layers is
    Megatron's Attention.forward (it sets rotary_pos_emb to None where no_rope_freq is 1,
    before the core attention, so it holds for TE and Turbo attention alike);
    test_attention_skips_rope_on_nope_layers checks that in a forward pass."""
    require_megatron_training()
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


def test_attention_skips_rope_on_nope_layers(request, monkeypatch):
    """Model-level NoPE check: a forward pass through the decoder of a small Megatron
    GPTModel built from the preset's no_rope_freq applies RoPE in exactly the layers where
    HF SmolLM3 does (no_rope_layers[i] == 1). Every layer gets the RoPE tensor; the layers
    that call apply_rotary_pos_emb are recorded."""
    require_megatron_training()
    # Requested after require_megatron_training: the fixture imports megatron.core, so without
    # Megatron-LM the test is skipped (or fails in CI) like its siblings, not an error.
    request.getfixturevalue("init_parallel_state")
    import megatron.core.transformer.attention as attention
    import megatron.core.transformer.transformer_block as transformer_block
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig

    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):  # GPTModel sets them
        monkeypatch.delenv(var, raising=False)
    # TE's CPU-offload context creates a CUDA stream; without it the block takes its
    # no-TE path (cpu_offloading is off), so the model builds on a host with no GPU.
    monkeypatch.setattr(transformer_block, "get_cpu_offload_context", None)
    num_layers, seq = 8, 8
    preset = _merged_preset("smollm3_3B_Base.yaml")
    config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=2,
        no_rope_freq=preset["no_rope_freq"],
        normalization="RMSNorm",
        attention_backend=AttnBackend.local,
        use_cpu_initialization=True,
    )
    model = GPTModel(
        config,
        transformer_layer_spec=get_gpt_layer_local_spec(normalization="RMSNorm"),
        vocab_size=64,
        max_sequence_length=seq,
        position_embedding_type="rope",
        rotary_base=int(preset["rotary_base"]),
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    current, rope_layers = [None], []

    class _CoreAttentionStub(torch.nn.Module):
        """RoPE is applied before the core attention; Megatron's DotProductAttention needs
        CUDA (its global buffer), so it is replaced by a shape-preserving stub."""

        def forward(self, query, key, value, *args, **kwargs):
            return query.reshape(query.shape[0], query.shape[1], -1)

    for module in model.modules():
        if isinstance(module, attention.Attention):
            module.register_forward_pre_hook(lambda m, a: current.__setitem__(0, m.layer_number))
            module.core_attention = _CoreAttentionStub()
    model.decoder.final_layernorm = torch.nn.Identity()  # TE RMSNorm needs CUDA; not under test
    rotary = attention.apply_rotary_pos_emb

    def recording_rotary(t, *args, **kwargs):
        if not rope_layers or rope_layers[-1] != current[0]:
            rope_layers.append(current[0])
        return rotary(t, *args, **kwargs)

    monkeypatch.setattr(attention, "apply_rotary_pos_emb", recording_rotary)
    # The decoder (all attention layers) gets one RoPE tensor, as from GPTModel's
    # RotaryEmbedding ([s, 1, 1, head_dim]; built here because that module needs CUDA).
    head_dim = config.kv_channels
    inv_freq = 1.0 / (float(preset["rotary_base"]) ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq).float(), inv_freq)
    rotary_pos_emb = torch.cat([freqs, freqs], dim=-1)[:, None, None, :].to(device)
    tokens = torch.randint(0, 64, (1, seq), device=device)
    mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=device), diagonal=1)[None, None]
    with torch.no_grad():
        hidden = model.embedding(tokens, torch.arange(seq, device=device)[None])
        model.decoder(hidden, mask, rotary_pos_emb=rotary_pos_emb)

    hf_nope = smol.nope_pattern({**SMOLLM3_CFG, "num_hidden_layers": num_layers})  # 1 = NoPE
    hf_rope_layers = [i + 1 for i in range(num_layers) if not hf_nope[i]]  # 1-based
    assert rope_layers == hf_rope_layers == [1, 2, 3, 5, 6, 7]


@pytest.mark.parametrize("use_turbo_attention", [False, True], ids=["te", "turbo"])
def test_shipped_attention_specs_keep_megatron_nope(monkeypatch, use_turbo_attention):
    """The example recipes build attention from Primus's TE spec provider: Primus-Turbo flash
    attention (MI300X/MI325X, use_turbo_attention) or TE attention (MI355X). Both only swap the
    *core* attention, which gets q and k after RoPE: the layer is Megatron's SelfAttention, whose
    forward applies the NoPE skip (attention.py: no_rope_freq[layer_number - 1] -> no RoPE, the
    path test_attention_skips_rope_on_nope_layers runs). This checks the shipped spec keeps that
    module and does not replace the attention forward. primus_turbo needs a GPU at import, so on
    a CPU host its classes are stand-ins (the spec wiring is what is under test)."""
    require_megatron_training()
    pytest.importorskip("transformer_engine")
    import sys
    import types

    from megatron.core.models.gpt import gpt_layer_specs
    from megatron.core.transformer.attention import Attention, SelfAttention

    turbo_name = "primus.backends.megatron.core.extensions.primus_turbo"
    provider_name = "primus.backends.megatron.core.extensions.transformer_engine_spec_provider"
    if not torch.cuda.is_available():
        stand_in = types.ModuleType(turbo_name)
        for name in (
            "PrimusTurboAttention",
            "PrimusTurboColumnParallelGroupedLinear",
            "PrimusTurboColumnParallelLinear",
            "PrimusTurboLayerNormColumnParallelLinear",
            "PrimusTurboLinear",
            "PrimusTurboRowParallelGroupedLinear",
            "PrimusTurboRowParallelLinear",
        ):
            setattr(stand_in, name, type(name, (), {}))
        stand_in._make_primus_turbo_norm_te_column_parallel_linear = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, turbo_name, stand_in)
        monkeypatch.delitem(sys.modules, provider_name, raising=False)
    provider = importlib.import_module(provider_name)
    if use_turbo_attention and provider.PrimusTurboAttention is None:
        pytest.skip("primus_turbo not importable")
    cfg = provider._build_default_primus_args()
    cfg.enable_primus_turbo = use_turbo_attention
    cfg.use_turbo_attention = use_turbo_attention
    monkeypatch.setattr(provider, "get_primus_args", lambda: cfg)
    # As the turbo te_spec_provider patch does for the run: the GPT spec builder uses Primus's provider.
    monkeypatch.setattr(gpt_layer_specs, "TESpecProvider", provider.PrimusTurboSpecProvider)
    spec = gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec()
    attn = spec.submodules.self_attention
    assert attn.module is SelfAttention
    assert attn.module.forward is Attention.forward  # the NoPE skip is in this forward
    expected = provider.PrimusTurboAttention if use_turbo_attention else provider.TEDotProductAttention
    assert attn.submodules.core_attention is expected


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
def test_recipe_reaches_the_megatron_args(tmp_path, monkeypatch, platform):
    """Per-PR check of the config binding (the trainer E2E case is weekly): the example
    recipe's SmolLM3 settings reach the args that Megatron's parse_args returns."""
    require_megatron_training()

    config = f"examples/megatron/configs/{platform}/smollm3_3B_Base-BF16-pretrain.yaml"
    args, module_config = _runtime_backend_args(tmp_path, monkeypatch, config)
    assert args.layernorm_epsilon == pytest.approx(1e-6) and args.no_rope_freq == 4
    assert args.rotary_base == 5000000 and args.untie_embeddings_and_output_weights is False


def test_check_model_eps():
    """The built model must use config.json's rms_norm_eps (a dropped override gives
    Megatron's 1e-5 while the weights still load)."""
    fields = {"norm_epsilon": 1e-6}
    ok = SimpleNamespace(config=SimpleNamespace(layernorm_epsilon=1e-6))
    assert smol.check_model_eps(ok, fields) == 1e-6
    for bad in (SimpleNamespace(config=SimpleNamespace(layernorm_epsilon=1e-5)), SimpleNamespace()):
        with pytest.raises(RuntimeError, match="layernorm_epsilon"):
            smol.check_model_eps(bad, fields)


def test_null_head_dim_is_rejected_and_explicit_list_does_not_warn_about_interval(recwarn):
    with pytest.raises(ValueError, match="head_dim=null"):
        smol.with_hf_defaults({**SMOLLM3_CFG, "head_dim": None})
    cfg = {k: v for k, v in SMOLLM3_CFG.items() if k != "no_rope_layer_interval"}
    n = cfg["num_hidden_layers"]
    smol.with_hf_defaults({**cfg, "no_rope_layers": [int((i + 1) % 4 != 0) for i in range(n)]})
    assert not [w for w in recwarn if "no_rope_layer_interval" in str(w.message)]
