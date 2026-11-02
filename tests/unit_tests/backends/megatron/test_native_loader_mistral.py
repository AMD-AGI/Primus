###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""CPU tests: Mistral routing to the native Llama converter, its HF config
translation, the HF -> mcore weight mapping on a tiny synthetic checkpoint, and
the Mistral-7B-v0.3 preset against config.json."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[4]
HOOK = REPO / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"

# config.json of mistralai/Mistral-7B-v0.3 @ caa1feb0e54d415e2df31207e5f4e273e33509b1
MISTRAL_V03 = {
    "architectures": ["MistralForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "max_position_embeddings": 32768,
    "model_type": "mistral",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_theta": 1000000.0,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "vocab_size": 32768,
}


def _hook():
    spec = importlib.util.spec_from_file_location("convert_checkpoints_hook", HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _loader():
    pytest.importorskip("torch")
    from primus.backends.megatron.checkpoint import native_loader_llama

    return native_loader_llama


def _rules():
    from primus.backends.megatron.checkpoint import llama_config_rules

    return llama_config_rules


LLAMA = dict(
    {k: v for k, v in MISTRAL_V03.items() if k != "sliding_window"},
    architectures=["LlamaForCausalLM"],
    model_type="llama",
)


def _family(tmp_path, cfg):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return _hook()._native_support_for_path(str(tmp_path))[0]


# ---------------------------------------------------------------------------
# Routing in the posttrain conversion hook
# ---------------------------------------------------------------------------
def test_mistral_v03_routes_to_native_llama(tmp_path):
    assert _family(tmp_path, MISTRAL_V03) == "llama"
    assert _family(tmp_path, dict(MISTRAL_V03, architectures=None)) == "llama"


def _without(cfg, keys):
    return {k: v for k, v in cfg.items() if k not in keys}


_LLAMA31_ROPE = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}

_NO_WINDOW_KEY = {k: v for k, v in MISTRAL_V03.items() if k != "sliding_window"}  # HF default 4096

# Configs the native converter does not support, each with a fragment of the
# reason it must give (so a case cannot pass on an earlier, unrelated rule): the
# hook keeps them on Megatron-Bridge with that reason and the converter rejects them.
_SW = "only full causal attention: sliding_window: null"
_FAMILIES = "they name different models"
_ARCH = "only LlamaForCausalLM or MistralForCausalLM"
_POS_NUM = "a positive finite number is required"
_POS_INT = "a positive integer is required"
_NULL = "HF has no value for it"
_BIAS = "the converter maps no bias tensors"
_LLAMA3_FIXED = "Megatron's llama3 RoPE scaling uses"
_FACTOR = "a finite number of at least 1 is required"
_NOT_LLAMA = "not a Llama or Mistral config"
_NO_CONVERTER = "no native converter"
# (cfg, the converter's reason[, the hook's reason when the hook's family table
# rejects the config before the Llama rules run])
_UNSUPPORTED = [
    # sliding window: Mistral-7B-v0.1, the HF default (4096) when the key is
    # missing, and Mistral named by only architectures or only model_type
    (dict(MISTRAL_V03, sliding_window=4096), _SW),
    (_NO_WINDOW_KEY, _SW),
    (dict(MISTRAL_V03, architectures=None, sliding_window=4096), _SW),
    (dict(MISTRAL_V03, model_type=None, sliding_window=4096), _SW),
    # no sliding_window (4096) and no max_position_embeddings (MistralConfig: 131072)
    (_without(MISTRAL_V03, ("sliding_window", "max_position_embeddings")), _SW),
    (dict(MISTRAL_V03, sliding_window="4096"), _SW),
    # a window as long as max_position_embeddings: seq_length may be longer (build_margs)
    (dict(MISTRAL_V03, sliding_window=32768), _SW),
    (dict(MISTRAL_V03, sliding_window=4096, max_position_embeddings=4096), _SW),
    (dict(MISTRAL_V03, hidden_act="gelu"), "only SwiGLU with silu"),
    (dict(MISTRAL_V03, rope_scaling={"rope_type": "yarn", "factor": 4.0}), "RoPE type 'yarn'"),
    (dict(MISTRAL_V03, attention_bias=True), _BIAS),
    (dict(LLAMA, attention_bias=True), _BIAS),
    (dict(LLAMA, mlp_bias=True), _BIAS),
    # bias flags are booleans, as tie_word_embeddings ("false" is not false)
    (dict(LLAMA, attention_bias="false"), "true or false is required"),
    (dict(MISTRAL_V03, mlp_bias=0), "true or false is required"),
    # the model shape must be set (no class defaults for the weights' shape)
    *[
        (_without(base, (k,)), f"['{k}'] missing")
        for base in (MISTRAL_V03, LLAMA)
        for k in ("hidden_size", "num_attention_heads", "num_hidden_layers", "intermediate_size")
    ],
    # rope_type: null is not "default"
    (dict(LLAMA, rope_scaling={"rope_type": None, "factor": 8.0}), "RoPE type None"),
    # architectures and model_type name different models
    (dict(LLAMA, model_type="qwen3"), _FAMILIES, "name different model families"),
    (dict(MISTRAL_V03, model_type="llama"), _FAMILIES),
    (dict(MISTRAL_V03, architectures=["LlamaForCausalLM"]), _FAMILIES),
    (dict(LLAMA, architectures=["LlamaForCausalLM", "MistralForCausalLM"]), _ARCH),
    # experts (a MoE config with a Mistral name)
    (dict(MISTRAL_V03, num_local_experts=8), "the converter maps dense models only"),
    # an architectures entry of another model or head, with a Llama / Mistral model_type
    (dict(LLAMA, architectures=["Qwen2ForCausalLM"]), _ARCH),
    (dict(MISTRAL_V03, architectures=["MixtralForCausalLM"]), _ARCH),
    (dict(LLAMA, architectures=["LlamaForSequenceClassification"]), _ARCH),
    (dict(LLAMA, architectures=["LlamaForCausalLM", "LlamaForSequenceClassification"]), _ARCH),
    # architectures that is not a string or a list of strings
    (dict(LLAMA, architectures=7), "architectures"),
    (dict(LLAMA, architectures={"name": "LlamaForCausalLM"}), "architectures"),
    (dict(LLAMA, architectures=["LlamaForCausalLM", 1]), "architectures"),
    # llama3 RoPE scaling without a usable factor
    (dict(LLAMA, rope_scaling={"rope_type": "llama3"}), _FACTOR),
    (dict(LLAMA, rope_scaling={"rope_type": "llama3", "factor": "x"}), _FACTOR),
    (dict(LLAMA, rope_scaling=dict(_LLAMA31_ROPE, factor=10**400)), _FACTOR),
    (dict(LLAMA, rope_scaling=dict(_LLAMA31_ROPE, factor=0.5)), _FACTOR),  # HF requires >= 1
    (dict(LLAMA, rope_scaling=dict(_LLAMA31_ROPE, factor=True)), _FACTOR),
    # HF requires every llama3 key; a missing one is not taken as Megatron's value
    *[
        (dict(LLAMA, rope_scaling={k: v for k, v in _LLAMA31_ROPE.items() if k != name}), f"{name} missing")
        for name in ("low_freq_factor", "high_freq_factor", "original_max_position_embeddings")
    ],
    (dict(LLAMA, rope_scaling={"rope_type": "llama3", "factor": 8.0}), "missing"),
    # llama3 RoPE scaling that Megatron's fixed llama3 scaling cannot reproduce
    (dict(LLAMA, rope_scaling=dict(_LLAMA31_ROPE, low_freq_factor=2.0)), _LLAMA3_FIXED),
    (
        dict(LLAMA, rope_scaling=dict(_LLAMA31_ROPE, high_freq_factor=8.0)),
        _LLAMA3_FIXED,
    ),
    (
        dict(
            LLAMA,
            rope_scaling=dict(_LLAMA31_ROPE, original_max_position_embeddings=4096),
        ),
        _LLAMA3_FIXED,
    ),
    (
        dict(LLAMA, rope_scaling=dict(_LLAMA31_ROPE, low_freq_factor="1")),
        _LLAMA3_FIXED,
    ),
    # rope_parameters and rope_scaling with a different scaling
    (
        dict(
            LLAMA,
            rope_parameters={"rope_type": "default", "rope_theta": 1e6},
            rope_scaling=_LLAMA31_ROPE,
            rope_theta=1e6,
        ),
        "differ",
    ),
    (
        dict(
            LLAMA,
            rope_parameters=dict(_LLAMA31_ROPE, factor=32.0, rope_theta=1e6),
            rope_scaling=_LLAMA31_ROPE,
            rope_theta=1e6,
        ),
        "differ",
    ),
    (dict(MISTRAL_V03, max_position_embeddings=None), _NULL),
    (dict(LLAMA, max_position_embeddings=None), _NULL),
    # Llama configs: the same activation / RoPE rules (llama3 RoPE is supported)
    (dict(LLAMA, hidden_act="gelu"), "only SwiGLU with silu"),
    (dict(LLAMA, rope_scaling={"rope_type": "yarn", "factor": 4.0}), "RoPE type 'yarn'"),
    (dict(LLAMA, rope_scaling={"type": "linear", "factor": 2.0}), "RoPE type 'linear'"),
    # Mistral accepts no RoPE scaling, also not llama3
    (dict(MISTRAL_V03, rope_scaling={"rope_type": "llama3", "factor": 8.0}), "RoPE type 'llama3'"),
    # transformers 5.x rope_parameters: scaling is checked there too
    (
        dict(
            _without(MISTRAL_V03, ("rope_theta",)),
            rope_parameters={"rope_theta": 1e6, "rope_type": "yarn", "factor": 4.0},
        ),
        "RoPE type 'yarn'",
    ),
    (
        dict(LLAMA, rope_parameters={"rope_theta": 1e6, "rope_type": "linear", "factor": 2.0}),
        "RoPE type 'linear'",
    ),
    # rope_theta and rope_parameters.rope_theta disagree
    (dict(MISTRAL_V03, rope_parameters={"rope_theta": 10000.0, "rope_type": "default"}), "differ"),
    (dict(MISTRAL_V03, rope_parameters="default"), "a JSON object is required"),
    # hidden_size not a multiple of the heads and no head_dim
    (dict(MISTRAL_V03, hidden_size=4097), "is not a multiple of num_attention_heads"),
    # null where HF has no value for it (it would fail in HF, not take a default)
    *[
        (dict(MISTRAL_V03, **{k: None}), k)
        for k in ("rms_norm_eps", "rope_theta", "hidden_act", "tie_word_embeddings")
    ],
    # not a Llama or Mistral config (Qwen2 has q/k/v biases but no attention_bias key)
    (dict(LLAMA, architectures=["Qwen2ForCausalLM"], model_type="qwen2"), _NOT_LLAMA, _NO_CONVERTER),
    (dict(LLAMA, architectures=None, model_type=None), _NOT_LLAMA, _NO_CONVERTER),
    # inf / nan / an int too large for a float
    (dict(MISTRAL_V03, rms_norm_eps=float("inf")), _POS_NUM),
    (dict(MISTRAL_V03, rope_theta=float("nan")), _POS_NUM),
    (dict(MISTRAL_V03, rope_theta=10**400), _POS_NUM),
    # 0 / 0.0 / negative / wrong-type values are rejected, not read as absent
    *[
        (dict(base, **{k: v}), why)
        for base in (MISTRAL_V03, LLAMA)
        for k, v, why in (
            ("num_key_value_heads", 0, _POS_INT),
            ("rms_norm_eps", 0.0, _POS_NUM),
            ("rope_theta", 0, _POS_NUM),
            ("head_dim", 0, _POS_INT),
            ("rms_norm_eps", -1e-5, _POS_NUM),
            ("num_key_value_heads", 3, "does not divide num_attention_heads"),  # 32 heads
            ("num_key_value_heads", 8.5, _POS_INT),
            ("hidden_size", 0, _POS_INT),
            ("num_attention_heads", None, _POS_INT),
            ("intermediate_size", "14336", _POS_INT),
            ("max_position_embeddings", 4096.5, _POS_INT),
            ("tie_word_embeddings", "false", "true or false is required"),
        )
    ],
]
UNSUPPORTED = [(cfg, why, hook_why[0] if hook_why else why) for cfg, why, *hook_why in _UNSUPPORTED]
SUPPORTED_MISTRAL = [
    MISTRAL_V03,
    dict(MISTRAL_V03, hidden_act="swish"),
    dict(MISTRAL_V03, rope_scaling=None),
    dict(MISTRAL_V03, rope_scaling={"rope_type": "default"}),  # HF: no scaling
    dict(MISTRAL_V03, attention_bias=False),
]


@pytest.mark.parametrize("cfg,_why,hook_why", UNSUPPORTED)
def test_unsupported_keeps_bridge_with_reason(tmp_path, cfg, _why, hook_why):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    family, reason = _hook()._native_support_for_path(str(tmp_path))
    assert family is None and hook_why in reason, reason


@pytest.mark.parametrize("cfg,why,_hook_why", UNSUPPORTED)
def test_converter_rejects_unsupported(cfg, why, _hook_why):
    with pytest.raises(ValueError, match="config not supported") as e:
        _rules().derive_megatron_fields(cfg)
    assert why in str(e.value), str(e.value)


def test_unsupported_cases_are_distinct():
    keys = [json.dumps(c, sort_keys=True) for c, _, _ in UNSUPPORTED]
    assert len(keys) == len(set(keys))


SUPPORTED_LLAMA = [
    LLAMA,
    dict(LLAMA, sliding_window=None),
    dict(LLAMA, sliding_window=4096),  # LlamaConfig has no sliding window; HF's Llama ignores the key
    # transformers 5.x layout
    dict(_without(LLAMA, ("rope_theta",)), rope_parameters={"rope_theta": 1e6, "rope_type": "default"}),
    dict(LLAMA, max_position_embeddings=4096.0),  # a JSON integer written as a float
    # Llama 3.1: RoPE scaling changes no weight; the training config sets use_rope_scaling
    dict(
        LLAMA,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    ),
    # both layouts with the same scaling (transformers 5.x writes rope_parameters)
    dict(
        LLAMA,
        rope_theta=5e5,
        rope_scaling=_LLAMA31_ROPE,
        rope_parameters=dict(_LLAMA31_ROPE, rope_theta=5e5),
    ),
]


@pytest.mark.parametrize("cfg", SUPPORTED_LLAMA + SUPPORTED_MISTRAL)
def test_supported_routes_native(tmp_path, cfg):
    assert _family(tmp_path, cfg) == "llama"
    assert _rules().derive_megatron_fields(cfg)["num_query_groups"] == 8


def test_rope_parameters_theta_is_used():
    cfg = dict(
        _without(MISTRAL_V03, ("rope_theta",)), rope_parameters={"rope_theta": 1e6, "rope_type": "default"}
    )
    assert _rules().derive_megatron_fields(cfg)["rotary_base"] == 1e6


@pytest.mark.parametrize(
    "content,reason",
    [
        (None, "no config.json"),
        ("{not json", "cannot read"),
        ("[1, 2]", "not a JSON object"),
        ('"mistral"', "not a JSON object"),
    ],
)
def test_native_support_for_path_reasons(tmp_path, content, reason):
    if content is not None:
        (tmp_path / "config.json").write_text(content)
    family, why = _hook()._native_support_for_path(str(tmp_path))
    assert family is None and reason in why


def test_hook_routing_imports_no_torch_or_megatron(tmp_path):
    """Detection is a config.json check: loading the hook and routing a Mistral
    config imports neither torch, nor primus.backends.megatron, nor megatron."""
    import subprocess
    import sys

    (tmp_path / "config.json").write_text(json.dumps(MISTRAL_V03))
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('hook', {str(HOOK)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        f"assert m._native_support_for_path({str(tmp_path)!r}) == ('llama', None)\n"
        "bad = [n for n in sys.modules if n == 'torch' or n.startswith(('megatron', 'primus.backends'))]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO, env={"PYTHONPATH": str(REPO)})


def test_hook_uses_the_converter_rules():
    """The hook loads the converter's rules file, and routes every Llama /
    Mistral name to the Llama converter."""
    rules, hook = _rules(), _hook()
    assert os.path.samefile(hook._RULES.__file__, rules.__file__)
    assert (
        "derive_megatron_fields" in vars(_loader())
        and _loader().derive_megatron_fields is rules.derive_megatron_fields
    )
    table = hook._NATIVE_FAMILIES
    assert {n for n in table if rules.config_family({"model_type": n}) == "mistral"} == set(
        rules.MISTRAL_NAMES
    )
    assert all(table[n] == "llama" for n in rules.MISTRAL_NAMES + rules.LLAMA_NAMES)


@pytest.mark.parametrize(
    "cfg,family",
    [
        (dict(LLAMA, model_type=None), "llama"),
        (dict(LLAMA, architectures=None), "llama"),
        (dict(LLAMA, model_type=""), "llama"),
        ({"architectures": ["Qwen3ForCausalLM"], "model_type": "llama"}, None),
        ({"architectures": ["MistralForCausalLM"], "model_type": "qwen3"}, None),
        ({"model_type": "qwen3"}, "qwen3"),
        ({"architectures": ["DeepseekV3ForCausalLM"]}, "deepseek"),
        ({"architectures": ["MixtralForCausalLM"], "model_type": "mixtral"}, None),
    ],
)
def test_other_families_unchanged(tmp_path, cfg, family):
    assert _family(tmp_path, cfg) == family


def test_reason_labels():
    rules = _rules()
    assert rules.unsupported_reason(dict(MISTRAL_V03, rope_theta="x")).startswith("rope_theta 'x'")
    assert rules.unsupported_reason(
        dict(MISTRAL_V03, rope_parameters={"rope_theta": 0, "rope_type": "default"})
    ).startswith("rope_parameters.rope_theta 0")


@pytest.mark.parametrize(
    "extra,want",
    [
        ({"torch_dtype": "float16"}, "fp16"),
        ({"dtype": "float32"}, "fp32"),
        ({"dtype": "bfloat16"}, "bf16"),
        ({}, "bf16"),
    ],
)
def test_native_dtype_reads_dtype_and_torch_dtype(tmp_path, extra, want):
    (tmp_path / "config.json").write_text(json.dumps(dict(_without(MISTRAL_V03, ("torch_dtype",)), **extra)))
    assert _hook()._native_dtype_from_config(str(tmp_path), None) == want


@pytest.mark.parametrize(
    "rope,opts,args,note",
    [
        (None, {}, {}, None),
        ({"rope_type": "default"}, {"use_rope_scaling": True, "rope_scaling_factor": 8.0}, {}, None),
        (
            _LLAMA31_ROPE,
            {"use_rope_scaling": True, "rope_scaling_factor": 8.0},
            {"use_rope_scaling": "true", "rope_scaling_factor": "8.0"},
            None,
        ),
        (
            _LLAMA31_ROPE,
            {},
            {"use_rope_scaling": "true", "rope_scaling_factor": "8.0"},
            "use_rope_scaling None, rope_scaling_factor None",
        ),
        (
            _LLAMA31_ROPE,
            {"use_rope_scaling": True, "rope_scaling_factor": 32.0},
            {"use_rope_scaling": "true", "rope_scaling_factor": "8.0"},
            "rope_scaling_factor 32.0",
        ),
        (
            dict(_LLAMA31_ROPE, factor=32),
            {"use_rope_scaling": False, "rope_scaling_factor": 8.0},
            {"use_rope_scaling": "true", "rope_scaling_factor": "32.0"},
            "factor 32.0",
        ),
        # a training-config factor that is not a number is replaced, not an error
        (
            _LLAMA31_ROPE,
            {"use_rope_scaling": True, "rope_scaling_factor": "x"},
            {"use_rope_scaling": "true", "rope_scaling_factor": "8.0"},
            "rope_scaling_factor 'x'",
        ),
    ],
)
def test_llama3_rope_args(rope, opts, args, note):
    """The hook passes the HF llama3 RoPE scaling to training; the note says
    when that replaces the training config's values."""
    got_args, got_note = _hook()._llama3_rope_args(dict(LLAMA, rope_scaling=rope), opts)
    assert got_args == args
    assert (got_note is None) if note is None else (note in got_note)


def test_llama3_rope_args_parse_as_training_args():
    """The emitted values parse back (primus/core/utils/arg_utils.py) to the
    types Megatron's options have."""
    from primus.core.utils.arg_utils import parse_cli_overrides

    args, _ = _hook()._llama3_rope_args(dict(LLAMA, rope_scaling=_LLAMA31_ROPE), {})
    parsed = parse_cli_overrides([f"{k}={v}" for k, v in args.items()])
    assert parsed == {"use_rope_scaling": True, "rope_scaling_factor": 8.0}


# main(): the backend choice for native_ckpt_convert unset / true / false,
# for a supported and an unsupported config.
@pytest.mark.parametrize("enabled", [None, True, False])
@pytest.mark.parametrize("supported", [True, False])
def test_main_backend_choice(tmp_path, monkeypatch, enabled, supported):
    hook = _hook()
    cfg = MISTRAL_V03 if supported else dict(MISTRAL_V03, sliding_window=4096)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    calls = []
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda _f: (str(tmp_path), None))
    monkeypatch.setattr(hook, "read_native_opts", lambda _f: {"enabled": enabled, "dtype": None, "out": None})
    monkeypatch.setattr(
        hook, "_native_main", lambda path, opts, family, *_a: calls.append(("native", family))
    )

    def bridge(*_a):
        calls.append(("bridge",))
        raise SystemExit("bridge")

    monkeypatch.setattr(hook, "convert_checkpoint", bridge)
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setattr("sys.argv", ["hook", "--config", str(tmp_path / "x.yaml")])
    if enabled is True and not supported:
        with pytest.raises(SystemExit) as e:
            hook.main()
        assert e.value.code == 1 and calls == []
        return
    try:
        hook.main()
    except SystemExit as e:
        assert str(e) == "bridge"
    want = [("native", "llama")] if supported and enabled is not False else [("bridge",)]
    assert calls == want


# ---------------------------------------------------------------------------
# Converter config translation
# ---------------------------------------------------------------------------
def test_derive_fields_mistral_v03():
    f = _loader().derive_megatron_fields(MISTRAL_V03)
    assert f == dict(
        hidden_size=4096,
        num_attention_heads=32,
        num_layers=32,
        max_position_embeddings=32768,
        norm_epsilon=1e-05,
        rotary_base=1000000.0,
        untie_embeddings_and_output_weights=True,
        ffn_hidden_size=14336,
        num_query_groups=8,
        kv_channels=128,
    )


_ABSENT = ("num_key_value_heads", "rms_norm_eps", "rope_theta", "head_dim")
_NULLABLE = ("num_key_value_heads", "head_dim")  # HF maps null to MHA / hidden // heads


def _defaults(f):
    return (f["num_query_groups"], f["norm_epsilon"], f["rotary_base"], f["kv_channels"])


@pytest.mark.parametrize("null,kv_heads", [(False, 8), (True, 32)])
def test_absent_or_null_keys_mistral(null, kv_heads):
    """Absent keys of a Mistral config convert as HF's MistralConfig reads them: 8 KV
    heads, eps 1e-6, rope_theta 10000, head_dim hidden/heads, and 131072 positions
    (with sliding_window: null). A null num_key_value_heads is MHA in MistralConfig
    (num_attention_heads), not 8; a null head_dim is hidden/heads."""
    cfg = _without(MISTRAL_V03, _ABSENT + ("max_position_embeddings",))
    if null:
        cfg.update(dict.fromkeys(_NULLABLE))
    f = _rules().derive_megatron_fields(cfg)
    assert _defaults(f) == (kv_heads, 1e-6, 10000.0, 128)
    assert f["max_position_embeddings"] == 131072


@pytest.mark.parametrize("null", [False, True])
def test_absent_or_null_keys_llama(null):
    """A Llama config keeps the converter's fallbacks from before this port: MHA,
    eps 1e-5 (not LlamaConfig's 1e-6), rope_theta 10000, head_dim hidden/heads."""
    cfg = _without(LLAMA, _ABSENT)
    if null:
        cfg.update(dict.fromkeys(_NULLABLE))
    assert _defaults(_rules().derive_megatron_fields(cfg)) == (32, 1e-5, 10000.0, 128)


# ---------------------------------------------------------------------------
# Weight mapping on a tiny synthetic Mistral-shaped checkpoint
# ---------------------------------------------------------------------------
class _Store:
    def __init__(self, tensors):
        self.t = tensors
        self.reads = []

    def get(self, name, dtype):
        self.reads.append(name)
        return self.t[name].to(dtype)

    def has(self, name):
        return name in self.t


def _tiny(vocab=200, padded=256):
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    hidden, nh, ng, ffn, layers = 64, 8, 2, 96, 2
    hd = hidden // nh
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    t = {
        "model.embed_tokens.weight": r(vocab, hidden),
        "model.norm.weight": r(hidden),
        "lm_head.weight": r(vocab, hidden),
    }
    for L in range(layers):
        h = f"model.layers.{L}."
        t[h + "input_layernorm.weight"] = r(hidden)
        t[h + "post_attention_layernorm.weight"] = r(hidden)
        t[h + "self_attn.q_proj.weight"] = r(nh * hd, hidden)
        t[h + "self_attn.k_proj.weight"] = r(ng * hd, hidden)
        t[h + "self_attn.v_proj.weight"] = r(ng * hd, hidden)
        t[h + "self_attn.o_proj.weight"] = r(hidden, nh * hd)
        t[h + "mlp.gate_proj.weight"] = r(ffn, hidden)
        t[h + "mlp.up_proj.weight"] = r(ffn, hidden)
        t[h + "mlp.down_proj.weight"] = r(hidden, ffn)
    cfg = dict(
        MISTRAL_V03,
        hidden_size=hidden,
        num_attention_heads=nh,
        num_key_value_heads=ng,
        intermediate_size=ffn,
        num_hidden_layers=layers,
        vocab_size=vocab,
    )
    fields = _rules().derive_megatron_fields(cfg)
    margs = SimpleNamespace(
        params_dtype=torch.float32,
        num_attention_heads=nh,
        num_query_groups=fields["num_query_groups"],
        kv_channels=fields["kv_channels"],
        hidden_size=hidden,
        num_layers=layers,
        untie_embeddings_and_output_weights=fields["untie_embeddings_and_output_weights"],
    )
    model = SimpleNamespace(
        state_dict=lambda: {"embedding.word_embeddings.weight": torch.empty(padded, hidden)}
    )
    return torch, t, fields, margs, model, (nh, ng, hd, hidden)


def test_weight_mapping_round_trip():
    torch, t, fields, margs, model, (nh, ng, hd, hidden) = _tiny()
    store = _Store(t)
    sd, consumed = _loader().build_mcore_state_dict(model, store, margs, fields, lambda *_: None)

    # every released tensor is read exactly once
    assert sorted(store.reads) == sorted(t)
    assert consumed == set(t)
    assert len(sd) == 3 + 6 * margs.num_layers
    assert torch.equal(sd["decoder.final_layernorm.weight"], t["model.norm.weight"])
    assert sd["embedding.word_embeddings.weight"].shape[0] == 256
    assert torch.equal(sd["embedding.word_embeddings.weight"][:200], t["model.embed_tokens.weight"])
    assert torch.equal(sd["output_layer.weight"][:200], t["lm_head.weight"])

    for L in range(margs.num_layers):
        p, h = f"decoder.layers.{L}.", f"model.layers.{L}."
        qkv = sd[p + "self_attention.linear_qkv.weight"]
        grouped = qkv.reshape(ng, (nh // ng + 2) * hd, hidden)
        q, k, v = torch.split(grouped, [(nh // ng) * hd, hd, hd], dim=1)
        assert torch.equal(q.reshape(-1, hidden), t[h + "self_attn.q_proj.weight"])
        assert torch.equal(k.reshape(-1, hidden), t[h + "self_attn.k_proj.weight"])
        assert torch.equal(v.reshape(-1, hidden), t[h + "self_attn.v_proj.weight"])
        assert torch.equal(sd[p + "self_attention.linear_proj.weight"], t[h + "self_attn.o_proj.weight"])
        gate, up = torch.chunk(sd[p + "mlp.linear_fc1.weight"], 2, dim=0)
        assert torch.equal(gate, t[h + "mlp.gate_proj.weight"])
        assert torch.equal(up, t[h + "mlp.up_proj.weight"])
        assert torch.equal(sd[p + "mlp.linear_fc2.weight"], t[h + "mlp.down_proj.weight"])
        assert torch.equal(
            sd[p + "self_attention.linear_qkv.layer_norm_weight"], t[h + "input_layernorm.weight"]
        )
        assert torch.equal(
            sd[p + "mlp.linear_fc1.layer_norm_weight"], t[h + "post_attention_layernorm.weight"]
        )


# ---------------------------------------------------------------------------
# Preset and recipes
# ---------------------------------------------------------------------------
def test_preset_matches_hf_config():
    pytest.importorskip("yaml")
    from primus.core.config.yaml_loader import parse_yaml

    preset = dict(parse_yaml(str(REPO / "primus/configs/models/megatron/mistral_7B_v0_3.yaml")))
    c = MISTRAL_V03
    assert preset["num_layers"] == c["num_hidden_layers"]
    assert preset["hidden_size"] == c["hidden_size"]
    assert preset["ffn_hidden_size"] == c["intermediate_size"]
    assert preset["num_attention_heads"] == c["num_attention_heads"]
    assert preset["num_query_groups"] == c["num_key_value_heads"]
    assert preset.get("kv_channels") in (None, c["hidden_size"] // c["num_attention_heads"])
    assert preset["max_position_embeddings"] == c["max_position_embeddings"]
    assert float(preset["rotary_base"]) == c["rope_theta"]
    assert float(preset["norm_epsilon"]) == c["rms_norm_eps"]
    assert preset["vocab_size"] == c["vocab_size"]
    assert preset["use_rope_scaling"] is False
    assert preset["untie_embeddings_and_output_weights"] is True
    assert preset["swiglu"] is True and preset["add_bias_linear"] is False
    assert preset["normalization"] == "RMSNorm"
    assert preset.get("window_size") is None


@pytest.mark.parametrize("sku", ["MI300X", "MI325X", "MI355X"])
def test_recipe_uses_preset(sku):
    yaml = pytest.importorskip("yaml")
    path = REPO / f"examples/megatron/configs/{sku}/mistral_7B_v0_3-BF16-pretrain.yaml"
    pre = yaml.safe_load(path.read_text())["modules"]["pre_trainer"]
    assert pre["model"] == "mistral_7B_v0_3.yaml"
    o = pre["overrides"]
    assert o["seq_length"] <= MISTRAL_V03["max_position_embeddings"]
    assert o["mock_data"] is True and o["train_data_path"] is None


@pytest.mark.parametrize("exists", [False, True])
def test_native_main_emits_llama3_rope_args(tmp_path, monkeypatch, capsys, exists):
    """_native_main reads the HF config, converts, and prints the extra.* lines
    that apply HF's llama3 RoPE scaling to the training run, on the convert path
    and on the already-converted path."""
    hook = _hook()
    (tmp_path / "config.json").write_text(json.dumps(dict(LLAMA, rope_scaling=_LLAMA31_ROPE)))
    converted = []
    monkeypatch.setattr(hook, "native_convert_checkpoint", lambda *a: converted.append(a))
    monkeypatch.setenv("RANK", "0")
    out = tmp_path / "out"
    if exists:
        out.mkdir()
        (out / "latest_checkpointed_iteration.txt").write_text("1")
    hook._native_main(str(tmp_path), {"enabled": None, "dtype": None, "out": str(out)}, "llama")
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("extra.")]
    assert lines == [
        "extra.use_rope_scaling=true",
        "extra.rope_scaling_factor=8.0",
        f"extra.pretrained_checkpoint={out}",
        "extra.finetune=true",
    ]
    if exists:
        assert converted == []
    else:
        assert converted and converted[0][2]["hf_config"]["rope_scaling"] == _LLAMA31_ROPE


_BIG_ROPE = dict(_LLAMA31_ROPE, low_freq_factor=2.0)  # natively unsupported: Megatron-Bridge


@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.parametrize("enabled,rope", [(False, _LLAMA31_ROPE), (None, _BIG_ROPE)])
def test_bridge_path_emits_llama3_rope_args(tmp_path, monkeypatch, capsys, exists, enabled, rope):
    """The Megatron-Bridge path passes the same llama3 RoPE arguments as the
    native path, on the convert and the already-converted paths."""
    hook = _hook()
    (tmp_path / "config.json").write_text(json.dumps(dict(LLAMA, rope_scaling=rope)))
    data = tmp_path / "data"
    megatron_path = data / "megatron_checkpoints" / tmp_path.name
    if exists:
        megatron_path.mkdir(parents=True)
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda _f: (str(tmp_path), None))
    monkeypatch.setattr(hook, "read_native_opts", lambda _f: {"enabled": enabled, "dtype": None, "out": None})
    monkeypatch.setattr(hook, "convert_checkpoint", lambda *a: None)
    monkeypatch.setattr(hook, "fix_common_pt_for_megatron_lm", lambda *a: None)
    monkeypatch.setenv("DATA_PATH", str(data))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr("sys.argv", ["hook", "--config", str(tmp_path / "x.yaml")])
    hook.main()
    out = capsys.readouterr().out
    assert [x for x in out.splitlines() if x.startswith("extra.")] == [
        "extra.use_rope_scaling=true",
        "extra.rope_scaling_factor=8.0",
        f"extra.pretrained_checkpoint={megatron_path}",
        "extra.finetune=true",
    ]
    assert "Using Megatron-Bridge" in out
    if rope is _BIG_ROPE:
        assert "Megatron cannot reproduce the rest of the HF llama3 RoPE scaling: low_freq_factor 2.0" in out


def test_llama3_rope_args_invalid_factor():
    """A factor Megatron cannot use (Bridge path only) is reported, not passed."""
    args, note = _hook()._llama3_rope_args(dict(LLAMA, rope_scaling=dict(_LLAMA31_ROPE, factor=0.5)), {})
    assert args == {} and "factor 0.5" in note


def test_native_main_mistral_emits_no_rope_args(tmp_path, monkeypatch, capsys):
    hook = _hook()
    (tmp_path / "config.json").write_text(json.dumps(MISTRAL_V03))
    monkeypatch.setattr(hook, "native_convert_checkpoint", lambda *a: None)
    monkeypatch.setenv("RANK", "0")
    out = tmp_path / "out"
    hook._native_main(str(tmp_path), {"enabled": None, "dtype": None, "out": str(out)}, "llama")
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("extra.")]
    assert lines == [f"extra.pretrained_checkpoint={out}", "extra.finetune=true"]


_PRIMUS_YAML = """work_group: amd
user_name: root
exp_name: native-opts
workspace: {workspace}
modules:
  post_trainer:
    framework: megatron
    config: sft_trainer.yaml
    model: llama3.1_8B.yaml
{module_keys}    overrides:
      stage: sft
      hf_path: {hf_path}
{override_keys}"""


@pytest.mark.parametrize(
    "module_keys,override_keys,want",
    [
        ("", "      use_rope_scaling: true\n      rope_scaling_factor: 8.0\n", (True, 8.0)),
        ("    use_rope_scaling: true\n    rope_scaling_factor: 4.0\n", "", (True, 4.0)),
        ("", "      native_ckpt_convert: false\n", (False, 8.0)),  # the llama3.1_8B preset's values
    ],
)
def test_read_native_opts_reads_rope_scaling(tmp_path, module_keys, override_keys, want):
    pytest.importorskip("yaml")
    config = tmp_path / "exp.yaml"
    config.write_text(
        _PRIMUS_YAML.format(
            workspace=tmp_path / "ws", hf_path=tmp_path, module_keys=module_keys, override_keys=override_keys
        )
    )
    opts = _hook().read_native_opts(str(config))
    assert (opts["use_rope_scaling"], opts["rope_scaling_factor"]) == want
    assert opts["enabled"] is (False if "native_ckpt_convert" in override_keys else None)
