###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""CPU tests: Mistral routing in the posttrain conversion hook, the native
Mistral converter's config checks and HF -> mcore weight mapping on a tiny
synthetic checkpoint, and the Mistral-7B-v0.3 preset and recipes against config.json."""

import importlib.util
import json
import sys
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
    from primus.backends.megatron.checkpoint import native_loader_mistral

    return native_loader_mistral


def _reason(hf_dir):
    """The Mistral checks' message for the config.json in ``hf_dir``, or None."""
    from primus.backends.megatron.checkpoint.native_config_mistral import check_config

    try:
        check_config(json.loads((Path(hf_dir) / "config.json").read_text()))
    except ValueError as e:
        return str(e)
    return None


def _without(cfg, keys):
    return {k: v for k, v in cfg.items() if k not in keys}


# ---------------------------------------------------------------------------
# Routing in the posttrain conversion hook
# ---------------------------------------------------------------------------
class _Bridge(Exception):
    pass


def _main(monkeypatch, tmp_path, cfg, enabled=None, logs=None, errors=None):
    """Run the hook's main() for the HF checkpoint ``tmp_path/hf`` with ``cfg`` and
    ``native_ckpt_convert`` = ``enabled``. Only the Megatron-Bridge conversion (its
    start is raised as _Bridge) and Megatron's process setup are stubbed: the native
    path runs main() -> _native_main -> native_convert_checkpoint -> convert.
    Returns (path, log lines)."""
    from primus.backends.megatron.checkpoint import native_convert_common as common

    hf = tmp_path / "hf"
    hf.mkdir(exist_ok=True)
    (hf / "config.json").write_text(json.dumps(cfg))
    hook = _hook()
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda f: (str(hf), None))
    monkeypatch.setattr(hook, "read_native_opts", lambda f: {"enabled": enabled, "dtype": None, "out": None})
    monkeypatch.setattr(common, "apply_convert_patches", lambda module_name: 0)
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("NODE_RANK", "0")
    logs = [] if logs is None else logs

    def log_info(msg):
        logs.append(msg)
        if "Megatron-Bridge conversion path" in msg:
            raise _Bridge

    monkeypatch.setattr(hook, "log_info", log_info)
    monkeypatch.setattr(hook, "log_error", (errors if errors is not None else []).append)
    monkeypatch.setattr(sys, "argv", ["hook", "--config", "/x.yaml"])
    try:
        hook.main()
    except _Bridge:
        return "bridge", logs
    return "native", logs


def test_mistral_v03_converts_natively_through_main(monkeypatch, tmp_path, capsys):
    """native_ckpt_convert unset: main() converts Mistral-7B-v0.3 with
    native_loader_mistral.convert and hands the checkpoint to the trainer."""
    m = _loader()
    calls = _fake_megatron(monkeypatch, m)
    path, logs = _main(monkeypatch, tmp_path, MISTRAL_V03)
    out = tmp_path / "data/megatron_checkpoints/hf_native"
    assert path == "native" and "Native (mistral) conversion: dtype=bf16 TP=1 PP=1 EP=1" in logs
    assert calls["store"] == str(tmp_path / "hf") and calls["save"] == str(out)
    assert calls["margs"][0] == m.derive_megatron_fields(MISTRAL_V03)
    assert f"extra.pretrained_checkpoint={out}" in capsys.readouterr().out


_NOT_NATIVE = [
    # Mistral-7B-v0.1 / Instruct-v0.1
    (dict(MISTRAL_V03, sliding_window=4096), "sliding_window=4096"),
    # no sliding_window key: HF reads 4096
    (_without(MISTRAL_V03, ("sliding_window",)), "no sliding_window"),
    # Mistral-Nemo style head_dim != hidden / heads
    (dict(MISTRAL_V03, hidden_size=5120, head_dim=128), "head_dim=128"),
    (
        dict(MISTRAL_V03, rope_scaling={"rope_type": "yarn", "factor": 4.0}),
        "factor': 4.0} is not supported: no RoPE scaling",
    ),
    (_without(MISTRAL_V03, ("rms_norm_eps",)), "no rms_norm_eps"),
    # other heads (model_type mistral names the family; the checks decline the head)
    (dict(MISTRAL_V03, architectures=["MistralForSequenceClassification"]), "architectures="),
    (dict(MISTRAL_V03, architectures=["MistralForTokenClassification"]), "architectures="),
    (dict(MISTRAL_V03, architectures=["MistralModel"]), "architectures="),
    # missing / null / string architectures: model_type mistral names the family, and
    # the checks decline it with their reason
    (dict(MISTRAL_V03, architectures=None), "architectures=None"),
    (_without(MISTRAL_V03, ("architectures",)), "architectures=None"),
    (dict(MISTRAL_V03, architectures="MistralForCausalLM"), "architectures='MistralForCausalLM'"),
    # quantized exports (packed qweight / scales) and tied embeddings: declined up front
    (dict(MISTRAL_V03, quantization_config={"quant_method": "gptq", "bits": 4}), "quantization_config"),
    (dict(MISTRAL_V03, tie_word_embeddings=True), "tie_word_embeddings=True"),
    # MoE: no native family
    (dict(MISTRAL_V03, architectures=["MixtralForCausalLM"], model_type="mixtral"), None),
]


@pytest.mark.parametrize("cfg,reason", _NOT_NATIVE)
def test_other_mistral_configs_keep_the_bridge_path(monkeypatch, tmp_path, cfg, reason):
    """native_ckpt_convert unset: a Mistral config the native converter does not
    convert keeps the Megatron-Bridge path, and main() logs why when the Mistral
    checks declined it (``reason``, None for an architecture with no native family)."""
    m = _loader()
    calls = _fake_megatron(monkeypatch, m)
    path, logs = _main(monkeypatch, tmp_path, cfg)
    assert path == "bridge" and calls == {}
    declines = [x for x in logs if "declines" in x]
    if reason is None:
        assert declines == []
    else:
        assert declines == [f"Native converter declines {tmp_path / 'hf'}: {_reason(tmp_path / 'hf')}"]
        assert reason in declines[0]
    # native_ckpt_convert: false: the Bridge path, with no native-converter message
    path, logs = _main(monkeypatch, tmp_path, cfg, enabled=False)
    assert path == "bridge" and not [x for x in logs if "declines" in x]


def test_forced_native_stops_with_the_converter_reason(monkeypatch, tmp_path):
    """native_ckpt_convert: true on a MistralForCausalLM config the converter rejects:
    main() logs one error, with the converter's specific reason (not "no native
    converter matches"), and exits with status 1 before Megatron is set up, with no
    checkpoint written."""
    m = _loader()
    calls = _fake_megatron(monkeypatch, m)
    errors = []
    with pytest.raises(SystemExit) as stop:
        _main(monkeypatch, tmp_path, dict(MISTRAL_V03, sliding_window=4096), enabled=True, errors=errors)
    assert stop.value.code == 1
    reason = _reason(tmp_path / "hf")
    assert "Mistral sliding_window=4096 is not supported" in reason
    assert len(errors) == 1
    assert f"the native converter declines the HF model at {tmp_path / 'hf'}: {reason}" in errors[0]
    assert "no native converter matches" not in errors[0]
    assert "native_ckpt_convert: false" in errors[0]
    assert calls == {}
    assert not (tmp_path / "data/megatron_checkpoints").exists()


def test_forced_native_without_a_family_says_no_converter_matches(monkeypatch, tmp_path):
    """native_ckpt_convert: true on an architecture with no native family: one error,
    naming the supported families."""
    m = _loader()
    calls = _fake_megatron(monkeypatch, m)
    errors = []
    with pytest.raises(SystemExit) as stop:
        _main(
            monkeypatch,
            tmp_path,
            dict(MISTRAL_V03, architectures=["MixtralForCausalLM"], model_type="mixtral"),
            enabled=True,
            errors=errors,
        )
    assert stop.value.code == 1
    assert len(errors) == 1 and "no native converter matches" in errors[0] and "Mistral-7B" in errors[0]
    assert calls == {}


def _install_decision(config_file):
    """Run 00_install_requirements.sh's decision step (its embedded Python) on an
    experiment config: "SKIP" (native, no Bridge dependencies) or "INSTALL"."""
    import re
    import subprocess

    script = (HOOK.parent / "00_install_requirements.sh").read_text()
    code = re.search(r"<<'PY'\n(.*?)\nPY\n", script, re.S).group(1)
    out = subprocess.run(
        [sys.executable, "-", str(config_file)], input=code, cwd=REPO, capture_output=True, text=True
    ).stdout
    return out.strip().rsplit("__DECISION__:", 1)[-1]


@pytest.mark.parametrize(
    "cfg,decision", [(MISTRAL_V03, "SKIP"), (dict(MISTRAL_V03, sliding_window=4096), "INSTALL")]
)
def test_install_hook_decision(tmp_path, cfg, decision):
    """00_install_requirements.sh skips the Bridge dependencies for Mistral-7B-v0.3
    (native conversion) and installs them for a config the native converter declines."""
    pytest.importorskip("yaml")
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps(cfg))
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        f"work_group: amd\nuser_name: root\nexp_name: t\nworkspace: {tmp_path / 'out'}\n"
        "modules:\n  post_trainer:\n    framework: megatron\n    config: sft_trainer.yaml\n"
        f"    model: mistral_7B_v0_3.yaml\n    overrides:\n      hf_path: {hf}\n"
    )
    assert _install_decision(exp) == decision


def test_install_hook_follows_the_same_decision(monkeypatch, tmp_path):
    """_detect_native_family (which 00_install_requirements.sh calls) finds the
    mistral family for v0.3, not for a config the native converter declines."""
    hook = _hook()
    logs = []
    monkeypatch.setattr(hook, "log_info", logs.append)
    for name, cfg, fam in (
        ("v03", MISTRAL_V03, "mistral"),
        ("head_dim_null", dict(MISTRAL_V03, head_dim=None), "mistral"),
        ("v01", dict(MISTRAL_V03, sliding_window=4096), None),
    ):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text(json.dumps(cfg))
        assert hook._detect_native_family(str(tmp_path / name)) == fam
        assert hook._detect_native_family(str(tmp_path / name)) == fam  # the same on a second detection
    detect = hook._detect_native_family_and_reason
    assert detect(str(tmp_path / "v03")) == ("mistral", None)
    fam, reason = detect(str(tmp_path / "v01"))
    assert fam is None and "sliding_window=4096" in reason  # the reason main() reports
    # each detection reads the config again: a config fixed in place is not kept declined
    (tmp_path / "v01" / "config.json").write_text(json.dumps(MISTRAL_V03))
    assert detect(str(tmp_path / "v01")) == ("mistral", None)
    # a config rewritten to another family, or with no architectures, carries no earlier reason
    (tmp_path / "v01" / "config.json").write_text(json.dumps(dict(MISTRAL_V03, sliding_window=4096)))
    assert detect(str(tmp_path / "v01"))[1]
    (tmp_path / "v01" / "config.json").write_text(json.dumps({"model_type": "llama", "architectures": []}))
    assert detect(str(tmp_path / "v01")) == ("llama", None)
    (tmp_path / "v01" / "config.json").unlink()
    assert detect(str(tmp_path / "v01")) == (None, None)
    assert logs == []  # detection logs nothing


@pytest.mark.parametrize(
    "extra",
    [
        # the declined Mistral config also lists Llama's architecture / names its model_type
        {"architectures": ["MistralForCausalLM", "LlamaForCausalLM"]},
        {"model_type": "llama"},
        # a Llama architecture with model_type mistral: the Mistral checks still run
        {"architectures": ["LlamaForCausalLM"]},
    ],
)
def test_declined_mistral_config_falls_to_no_other_family(monkeypatch, tmp_path, extra):
    """Once the Mistral checks decline a config that names the Mistral family (by
    architecture or model_type), no other native family (Llama, through another
    architecture or the model_type) converts it."""
    hook = _hook()
    monkeypatch.setattr(hook, "log_info", lambda msg: None)
    (tmp_path / "config.json").write_text(json.dumps(dict(MISTRAL_V03, sliding_window=4096, **extra)))
    fam, reason = hook._detect_native_family_and_reason(str(tmp_path))
    assert fam is None and reason


def test_llama_routing_unchanged(monkeypatch, tmp_path):
    """A Llama checkpoint still converts natively by default, with Llama's converter
    and no config check logged."""
    from primus.backends.megatron.checkpoint import native_convert_common as common
    from primus.backends.megatron.checkpoint import native_loader_llama as llama

    _loader()
    calls = []
    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: None)
    monkeypatch.setattr(llama, "convert", lambda hf, out, **kw: calls.append(hf))
    cfg = dict(MISTRAL_V03, architectures=["LlamaForCausalLM"], model_type="llama", sliding_window=4096)
    path, logs = _main(monkeypatch, tmp_path, cfg)
    assert path == "native" and calls == [str(tmp_path / "hf")]
    assert not [x for x in logs if "declines" in x]


@pytest.mark.parametrize(
    "cfg,family",
    [
        ({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}, "llama"),
        ({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}, "qwen3"),
        ({"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe"}, "qwen3"),
        ({"architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3"}, "deepseek"),
        ({"architectures": ["SomethingElse", "LlamaForCausalLM"], "model_type": "x"}, "llama"),
        ({"architectures": None, "model_type": "llama"}, "llama"),
        ({"model_type": "deepseek_v2"}, "deepseek"),
        ({"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2"}, None),
    ],
)
def test_other_families_detected_as_before(tmp_path, cfg, family):
    """Llama / Qwen3 / DeepSeek detection is unchanged: first architecture that names a
    family, else model_type, and no config check (they have no _NATIVE_CONFIG_CHECKS entry)."""
    hook = _hook()
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert hook._detect_native_family_and_reason(str(tmp_path)) == (family, None)


@pytest.mark.parametrize(
    "text,error",
    [
        ('["MistralForCausalLM"]', "is not a JSON object"),  # a list, not an object
        ("null", "is not a JSON object"),
        ("{not json", "Expecting property name"),  # json.JSONDecodeError, as before
    ],
)
def test_unreadable_config_raises(tmp_path, text, error):
    """config.json that is not a JSON object fails loudly (no silent Bridge fallback)."""
    hook = _hook()
    (tmp_path / "config.json").write_text(text)
    with pytest.raises(ValueError, match=error):
        hook._detect_native_family_and_reason(str(tmp_path))


def test_forced_bridge_runs_no_detection(monkeypatch, tmp_path):
    """native_ckpt_convert: false takes the Bridge path without loading any family checks."""
    hook = _hook()
    called = []
    monkeypatch.setattr(hook, "_detect_native_family_and_reason", lambda p: called.append(p))
    hf = tmp_path / "hf"
    hf.mkdir()
    monkeypatch.setattr(hook, "get_checkpoint_config", lambda f: (str(hf), None))
    monkeypatch.setattr(hook, "read_native_opts", lambda f: {"enabled": False, "dtype": None, "out": None})

    def log_info(msg):
        if "Megatron-Bridge conversion path" in msg:
            raise _Bridge

    monkeypatch.setattr(hook, "log_info", log_info)
    monkeypatch.setattr(sys, "argv", ["hook", "--config", "/x.yaml"])
    with pytest.raises(_Bridge):
        hook.main()
    assert called == []


def test_config_checks_registry_contract(monkeypatch, tmp_path):
    """Checks are registered per family in _NATIVE_CONFIG_CHECKS, loaded once per
    process, and a module without check_config fails with a clear error."""
    hook = _hook()
    assert "mistral" in hook._NATIVE_CONFIG_CHECKS
    assert all(f in hook._NATIVE_FAMILIES.values() for f in hook._NATIVE_CONFIG_CHECKS)
    check = hook._native_config_check("mistral")
    assert check is hook._native_config_check("mistral")  # cached: loaded once
    assert check(MISTRAL_V03) == 1e6
    assert hook._native_config_check("llama") is None

    bad = tmp_path / "primus/backends/megatron/checkpoint/native_config_broken.py"
    bad.parent.mkdir(parents=True)
    bad.write_text("def other(cfg):\n    return None\n")
    monkeypatch.setattr(hook, "PRIMUS_ROOT", tmp_path)
    monkeypatch.setitem(hook._NATIVE_CONFIG_CHECKS, "broken", str(bad.relative_to(tmp_path)))
    with pytest.raises(TypeError, match="has no check_config"):
        hook._native_config_check("broken")


def test_config_checks_need_only_the_standard_library():
    """The hook loads native_config_mistral.py by path (_NATIVE_CONFIG_CHECKS) on every
    rank and in the install hook, before torch / Megatron: it must import nothing but
    the standard library."""
    import ast

    src = (REPO / "primus/backends/megatron/checkpoint/native_config_mistral.py").read_text()
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or ".").split(".")[0])
    assert names <= {"json", "math", "os"}, names


# ---------------------------------------------------------------------------
# Converter config checks and translation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cfg,match",
    [
        # Mistral-7B-v0.1
        (dict(MISTRAL_V03, sliding_window=4096), "sliding_window=4096"),
        # HF ignores use_sliding_window for Mistral: the window still applies
        (dict(MISTRAL_V03, sliding_window=4096, use_sliding_window=False), "sliding_window=4096"),
        # no key: HF MistralConfig default 4096
        (_without(MISTRAL_V03, ("sliding_window",)), "no sliding_window key"),
        (_without(dict(MISTRAL_V03, use_sliding_window=False), ("sliding_window",)), "no sliding_window key"),
        # RoPE scaling (rope_scaling / rope_parameters)
        (dict(MISTRAL_V03, rope_scaling={"rope_type": "yarn", "factor": 4.0}), "rope_scaling"),
        (dict(MISTRAL_V03, rope_scaling={"factor": 2.0}), "rope_scaling"),
        (
            dict(MISTRAL_V03, rope_scaling={"rope_type": "default", "factor": 8.0}),
            "rope_scaling=.*no RoPE scaling",
        ),
        (
            dict(MISTRAL_V03, rope_scaling={"type": "linear", "rope_type": "default", "factor": 4}),
            "rope_scaling=.*no RoPE scaling",
        ),
        (dict(MISTRAL_V03, rope_scaling="default"), "rope_scaling='default'"),
        (
            dict(MISTRAL_V03, rope_parameters={"rope_type": "yarn", "factor": 4.0, "rope_theta": 1e6}),
            "rope_parameters=.*no RoPE scaling",
        ),
        # disagrees with rope_theta
        (
            dict(MISTRAL_V03, rope_parameters={"rope_type": "default", "rope_theta": 1e4}),
            "RoPE base given differently",
        ),
        (dict(MISTRAL_V03, rope_parameters={"rope_type": None}), "rope_parameters=.*no RoPE scaling"),
        (
            dict(MISTRAL_V03, rope_parameters={"rope_type": "default", "rope_theta": None}),
            "rope_parameters.rope_theta=None",
        ),
        (dict(MISTRAL_V03, rope_scaling=["x"]), "rope_scaling"),
        (dict(MISTRAL_V03, rope_scaling="linear"), "rope_scaling"),
        (dict(MISTRAL_V03, hidden_act="gelu"), "hidden_act"),
        (dict(MISTRAL_V03, attention_bias=True), "attention_bias"),
        (dict(MISTRAL_V03, mlp_bias=True), "mlp_bias"),
        (
            _without(MISTRAL_V03, ("num_key_value_heads",)),
            r"no num_key_value_heads: .*fallback \(num_attention_heads\).*default \(8\)",
        ),
        (_without(MISTRAL_V03, ("rms_norm_eps",)), r"no rms_norm_eps: .*fallback \(1e-5\).*default \(1e-6\)"),
        (dict(MISTRAL_V03, num_key_value_heads=None), "num_key_value_heads=None .*num_attention_heads"),
        (dict(MISTRAL_V03, num_key_value_heads=0), "num_key_value_heads=0"),
        (dict(MISTRAL_V03, num_key_value_heads=5), "num_key_value_heads=5"),
        (dict(MISTRAL_V03, head_dim=0), "head_dim=0"),
        (dict(MISTRAL_V03, head_dim=-1), "head_dim=-1"),
        # a head size other than hidden / heads is not verified with this converter
        (dict(MISTRAL_V03, head_dim=96), "head_dim=96"),
        (dict(MISTRAL_V03, hidden_size=4100), "hidden_size=4100 is not a multiple"),
        (dict(MISTRAL_V03, hidden_size=4097, head_dim=None), "hidden_size=4097 is not a multiple"),
        (dict(MISTRAL_V03, num_hidden_layers=0), "num_hidden_layers=0"),
        (dict(MISTRAL_V03, vocab_size=None), "vocab_size=None"),
        (dict(MISTRAL_V03, partial_rotary_factor=0.5), "partial_rotary_factor=0.5"),
        # an int above float range is rejected with ValueError, not OverflowError
        (dict(MISTRAL_V03, partial_rotary_factor=10**400), "partial_rotary_factor="),
        (dict(MISTRAL_V03, rms_norm_eps=10**400), "rms_norm_eps="),
        (dict(MISTRAL_V03, rope_theta=10**400), "rope_theta="),
        (dict(MISTRAL_V03, rope_theta=-1), "rope_theta=-1"),
        (dict(MISTRAL_V03, rope_theta=float("inf")), "rope_theta=inf"),
        (dict(MISTRAL_V03, head_dim=True), "head_dim=True"),
        (dict(MISTRAL_V03, head_dim=128.0), "head_dim=128.0"),
        (dict(MISTRAL_V03, partial_rotary_factor="1"), "partial_rotary_factor='1'"),
        (dict(MISTRAL_V03, tie_word_embeddings="no"), "tie_word_embeddings='no'"),
        # values of any type raise ValueError, never TypeError
        (dict(MISTRAL_V03, rope_scaling={"rope_type": ["default"]}), "rope_scaling"),
        (dict(MISTRAL_V03, rope_parameters={"type": {"a": 1}}), "rope_parameters"),
        (dict(MISTRAL_V03, rope_theta=[1e6]), "rope_theta"),
        (dict(MISTRAL_V03, hidden_act=["silu"]), "hidden_act"),
        (dict(MISTRAL_V03, sliding_window=[4096]), "sliding_window"),
        (dict(MISTRAL_V03, architectures="MistralForCausalLM"), "architectures"),
        (dict(MISTRAL_V03, architectures=None), "architectures=None"),
        (dict(MISTRAL_V03, architectures=[]), "architectures=\\[\\]"),
        (_without(MISTRAL_V03, ("model_type",)), "model_type=None"),
        (dict(MISTRAL_V03, model_type="llama"), "model_type='llama'"),
        (dict(MISTRAL_V03, num_attention_heads=[32]), "num_attention_heads"),
        (dict(MISTRAL_V03, attention_bias={"x": 1}), "attention_bias"),
        # booleans must be JSON booleans: 0 / 1 are not
        (dict(MISTRAL_V03, attention_bias=0), "attention_bias=0"),
        (dict(MISTRAL_V03, mlp_bias=0.0), "mlp_bias=0.0"),
        (dict(MISTRAL_V03, tie_word_embeddings=True), "tie_word_embeddings=True"),
        (dict(MISTRAL_V03, quantization_config={"quant_method": "awq"}), "quantization_config"),
        (dict(MISTRAL_V03, quantization_config={}), "quantization_config"),
        (dict(MISTRAL_V03, tie_word_embeddings=1), "tie_word_embeddings=1"),
        (dict(MISTRAL_V03, tie_word_embeddings=1.0), "tie_word_embeddings=1.0"),
        (dict(MISTRAL_V03, tie_word_embeddings=0), "tie_word_embeddings=0"),
        ([MISTRAL_V03], "JSON object"),
        (dict(MISTRAL_V03, rms_norm_eps=0), "rms_norm_eps=0"),
        (dict(MISTRAL_V03, rms_norm_eps=float("nan")), "rms_norm_eps=nan"),
        (dict(MISTRAL_V03, rms_norm_eps=float("inf")), "rms_norm_eps=inf"),
    ],
)
def test_converter_rejects_unsupported(cfg, match):
    with pytest.raises(ValueError, match=match):
        _loader().derive_megatron_fields(cfg)


def test_loader_interface_checks_then_is_llama():
    """derive_megatron_fields: the checks run first, the translation is Llama's
    (the one native_loader_llama.convert uses), for every accepted config."""
    m = _loader()
    from primus.backends.megatron.checkpoint import native_loader_llama as llama

    assert m.derive_megatron_fields(MISTRAL_V03) == llama.derive_megatron_fields(MISTRAL_V03)
    for extra in (
        {"head_dim": 128},
        {"head_dim": None},
        {"rope_parameters": {"rope_theta": 1e6}},
    ):  # top-level too
        cfg = dict(MISTRAL_V03, **extra)
        assert m.derive_megatron_fields(cfg) == llama.derive_megatron_fields(cfg)
    # head_dim: null -> kv_channels None, which Megatron's validate_args sets to hidden // heads
    assert m.derive_megatron_fields(dict(MISTRAL_V03, head_dim=None))["kv_channels"] is None
    with pytest.raises(ValueError, match="sliding_window=4096"):
        m.derive_megatron_fields(dict(MISTRAL_V03, sliding_window=4096))


@pytest.mark.parametrize(
    "extra",
    [
        {"partial_rotary_factor": None},  # null: the default, full rotary
        {"partial_rotary_factor": 1},
        {"hidden_act": "swish"},  # HF's alias of silu
        {"rope_parameters": {"rope_theta": 1e6}},  # no type stated: only the base
        {"rope_scaling": None, "rope_parameters": None},
        {"attention_bias": None, "mlp_bias": False},
        {"rope_scaling": {"rope_type": "default"}},
        {"rope_scaling": {}, "rope_parameters": {}},  # empty: no scaling, as HF reads it
        # transformers v5 re-save: the base only under rope_parameters
        {"rope_theta": None, "rope_parameters": {"rope_type": "default", "rope_theta": 1e6}},
    ],
)
def test_valid_hf_configs_accepted(extra):
    m = _loader()
    cfg = {k: v for k, v in dict(MISTRAL_V03, **extra).items() if v is not None or k != "rope_theta"}
    fields, base = m.derive_megatron_fields(cfg), m.derive_megatron_fields(MISTRAL_V03)
    assert fields == base


@pytest.mark.parametrize("tie", ["absent", None, False])
def test_untied_embeddings(tie):
    """tie_word_embeddings absent, null or false: untied (HF Mistral's default);
    true is declined (test_converter_rejects_unsupported)."""
    cfg = _without(MISTRAL_V03, ["tie_word_embeddings"])
    if tie != "absent":
        cfg["tie_word_embeddings"] = tie
    assert _loader().derive_megatron_fields(cfg)["untie_embeddings_and_output_weights"] is True


def test_null_rope_theta_uses_the_nested_or_default_base():
    """A top-level rope_theta: null is no base given, as common.hf_rope_theta reads it:
    the nested base, else HF's default 10000."""
    from primus.backends.megatron.checkpoint import native_convert_common as common
    from primus.backends.megatron.checkpoint.native_config_mistral import check_config

    nested = dict(MISTRAL_V03, rope_theta=None, rope_parameters={"rope_type": "default", "rope_theta": 1e6})
    assert check_config(nested) == common.hf_rope_theta(nested, 10000.0) == 1e6
    bare = dict(MISTRAL_V03, rope_theta=None)
    assert check_config(bare) == common.hf_rope_theta(bare, 10000.0) == 10000.0


def test_rope_parameters_theta_used():
    """rope_theta under rope_parameters (newer transformers) is read: the one checked
    base becomes rotary_base, also when it is given only there (a transformers v5
    re-save of Mistral-7B-v0.3), and a different nested base is declined."""
    m = _loader()
    nested = {"rope_type": "default", "rope_theta": 1e6}
    both = dict(MISTRAL_V03, rope_parameters=nested)
    assert m.derive_megatron_fields(both)["rotary_base"] == 1e6
    with pytest.raises(ValueError, match="RoPE base given differently"):
        m.derive_megatron_fields(dict(both, rope_theta=5e5))
    only_nested = dict(_without(MISTRAL_V03, ["rope_theta"]), rope_parameters=nested)
    assert m.derive_megatron_fields(only_nested) == m.derive_megatron_fields(MISTRAL_V03)
    assert m.derive_megatron_fields(only_nested)["rotary_base"] == 1e6
    only_scaling = dict(_without(MISTRAL_V03, ["rope_theta"]), rope_scaling={"rope_theta": 5e5})
    assert m.derive_megatron_fields(only_scaling)["rotary_base"] == 5e5
    # no rope_theta anywhere: HF's default
    assert m.derive_megatron_fields(_without(MISTRAL_V03, ["rope_theta"]))["rotary_base"] == 10000.0
    # a top-level int above float range is reported as out of range
    with pytest.raises(ValueError, match="must be a positive finite number"):
        m.derive_megatron_fields(dict(MISTRAL_V03, rope_theta=10**400))


def _fake_megatron(monkeypatch, m):
    """Stub Megatron and Llama's builders; record what convert gives them."""
    from primus.backends.megatron.checkpoint import native_convert_common as common
    from primus.backends.megatron.checkpoint import native_loader_llama as llama

    calls = {}
    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: None)
    monkeypatch.setattr(common, "assert_bridge_free", lambda where: None)
    margs = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        params_dtype="bf16",
    )

    def build_margs(fields, args):
        calls["margs"] = (fields, args)
        return margs

    monkeypatch.setattr(llama, "build_margs", build_margs)
    monkeypatch.setattr(common, "init_megatron_single_process", lambda margs, port, log: margs)
    monkeypatch.setattr(common, "build_mcore_gpt_model", lambda margs: "model")
    monkeypatch.setattr(common, "SafetensorsStore", lambda d: calls.setdefault("store", d))
    monkeypatch.setattr(llama, "build_mcore_state_dict", lambda *a: calls.setdefault("sd", a) and ({}, set()))
    monkeypatch.setattr(common, "load_validate_save", lambda *a: calls.setdefault("save", a[-2]))
    return calls


@pytest.mark.parametrize(
    "cfg",
    [
        MISTRAL_V03,
        dict(MISTRAL_V03, head_dim=128),
        dict(MISTRAL_V03, rope_parameters={"rope_type": "default", "rope_theta": 1e6}),
        # transformers v5 re-save: the base only under rope_parameters reaches rotary_base
        dict(
            _without(MISTRAL_V03, ["rope_theta"]), rope_parameters={"rope_type": "default", "rope_theta": 1e6}
        ),
    ],
)
def test_convert_is_llamas_convert(tmp_path, monkeypatch, cfg):
    """convert runs native_loader_llama.convert (build_margs, model build, weight
    mapping and save) on the HF directory itself, with this module's translation;
    Llama's module is not changed."""
    m = _loader()
    from primus.backends.megatron.checkpoint import native_loader_llama as llama

    original = llama.derive_megatron_fields
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    calls = _fake_megatron(monkeypatch, m)
    assert m.convert(str(tmp_path), "out", dtype="bf16", seq_length=512) == "out"
    fields, args = calls["margs"]
    assert fields == m.derive_megatron_fields(MISTRAL_V03)
    assert fields["kv_channels"] == 128 and fields["rotary_base"] == 1e6
    assert (args.hf_dir, args.save_dir, args.dtype, args.seq_length) == (str(tmp_path), "out", "bf16", 512)
    assert calls["store"] == str(tmp_path) and calls["save"] == "out"
    assert llama.derive_megatron_fields is original


def test_convert_checks_the_config_once(tmp_path, monkeypatch):
    """convert runs the Mistral checks once (inside Llama's translation step)."""
    m = _loader()
    (tmp_path / "config.json").write_text(json.dumps(MISTRAL_V03))
    _fake_megatron(monkeypatch, m)
    checked = []
    real = m.check_config
    monkeypatch.setattr(m, "check_config", lambda cfg: checked.append(1) or real(cfg))
    m.convert(str(tmp_path), "out")
    assert checked == [1]


def test_convert_rejects_before_converting(tmp_path, monkeypatch):
    m = _loader()
    (tmp_path / "config.json").write_text(json.dumps(dict(MISTRAL_V03, sliding_window=4096)))
    calls = _fake_megatron(monkeypatch, m)
    with pytest.raises(ValueError, match="sliding_window=4096.*native_ckpt_convert: false"):
        m.convert(str(tmp_path), "out")
    assert calls == {}


def test_direct_native_convert_reports_the_decline_reason(tmp_path, monkeypatch):
    """native_convert_checkpoint called without a family (a direct caller, not main())
    on a Mistral config the checks decline: the error carries the checks' reason, not
    "no native converter"; nothing is converted."""
    m = _loader()
    (tmp_path / "config.json").write_text(json.dumps(dict(MISTRAL_V03, sliding_window=4096)))
    calls = _fake_megatron(monkeypatch, m)
    with pytest.raises(ValueError) as e:
        _hook().native_convert_checkpoint(str(tmp_path), "out", {})
    assert "declines the HF model" in str(e.value) and "sliding_window=4096" in str(e.value)
    assert "no native converter" not in str(e.value)
    assert calls == {}


def test_mistral_family_dispatches_to_native_loader_mistral(tmp_path, monkeypatch):
    """native_convert_checkpoint converts the 'mistral' family with
    native_loader_mistral.convert (not Llama's or DeepSeek's)."""
    m = _loader()
    from primus.backends.megatron.checkpoint import native_convert_common as common
    from primus.backends.megatron.checkpoint import (
        native_loader_deepseek,
        native_loader_llama,
    )

    (tmp_path / "config.json").write_text(json.dumps(MISTRAL_V03))
    calls = []
    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: None)
    monkeypatch.setattr(common, "apply_convert_patches", lambda module_name: 0)
    monkeypatch.setattr(m, "convert", lambda hf, out, **kw: calls.append(("mistral", hf, out, kw)))
    for other in (native_loader_llama, native_loader_deepseek):
        monkeypatch.setattr(other, "convert", lambda *a, **kw: calls.append(("other", a, kw)))
    _hook().native_convert_checkpoint(str(tmp_path), "out", {})
    assert calls == [
        (
            "mistral",
            str(tmp_path),
            "out",
            dict(dtype="bf16", tensor_parallel_size=1, pipeline_parallel_size=1, expert_parallel_size=1),
        )
    ]


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
        add_qkv_bias=False,
    )


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
    fields = _loader().derive_megatron_fields(cfg)
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
    # the padded vocab rows copy the last real row (native_convert_common.pad_vocab)
    for name, hf in (
        ("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
        ("output_layer.weight", "lm_head.weight"),
    ):
        assert torch.equal(sd[name][200:], t[hf][-1:].expand(56, -1))

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
def test_recipe_resolves_to_mistral(monkeypatch, sku):
    """Each recipe, resolved per PR on CPU through Primus's config stack (env
    interpolation -> extends -> preset merge -> overrides), builds Mistral-7B-v0.3's
    shape on the Llama layer template, without a sliding window or RoPE scaling."""
    pytest.importorskip("yaml")
    from primus.core.config.primus_config import load_primus_config
    from primus.core.utils import file_utils

    monkeypatch.setattr(file_utils, "create_path_if_not_exists", lambda *a, **k: None)
    path = REPO / f"examples/megatron/configs/{sku}/mistral_7B_v0_3-BF16-pretrain.yaml"
    (module,) = load_primus_config(path, None).modules
    p, c = module.params, MISTRAL_V03
    assert (p.num_layers, p.hidden_size, p.ffn_hidden_size) == (32, 4096, c["intermediate_size"])
    assert (p.num_attention_heads, p.num_query_groups, p.kv_channels) == (32, 8, 128)
    assert p.group_query_attention is True
    assert float(p.rotary_base) == c["rope_theta"] and float(p.norm_epsilon) == c["rms_norm_eps"]
    assert p.use_rope_scaling is False and getattr(p, "window_size", None) is None
    assert (p.vocab_size, p.max_position_embeddings) == (c["vocab_size"], c["max_position_embeddings"])
    assert p.untie_embeddings_and_output_weights is True
    assert (p.swiglu, p.normalization, p.position_embedding_type) == (True, "RMSNorm", "rope")
    assert p.add_bias_linear is False
    assert p.seq_length <= c["max_position_embeddings"]
    assert p.mock_data is True and p.train_data_path is None
