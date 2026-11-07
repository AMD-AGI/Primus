###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The OLMo-3 native converter family registration (torch-free)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from primus.backends.megatron.checkpoint import native_registry

# the released allenai/Olmo-3-1025-7B config.json (the keys the registry reads)
_RELEASED = {
    "architectures": ["Olmo3ForCausalLM"],
    "model_type": "olmo3",
    "dtype": "bfloat16",
    "vocab_size": 100278,
}


def test_registration_selects_olmo3_for_the_released_config():
    fam = native_registry.family_for_config(_RELEASED)
    assert fam is not None and fam.name == "olmo3"
    assert fam.converter == "primus.backends.megatron.checkpoint.native_loader_olmo3"
    assert fam.model_type == "olmo3" and native_registry.model_type_for("olmo3") == "olmo3"
    # model_type alone selects it too; other families are not shadowed
    assert native_registry.family_for_config({"model_type": "olmo3"}).name == "olmo3"
    assert native_registry.family_for_config({"architectures": ["LlamaForCausalLM"]}).name == "llama"
    assert "OLMo-3" in native_registry.supported_text()


def test_released_dtype_key_gives_bf16(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({**_RELEASED, "dtype": "float32"}))
    fam = native_registry.get_family("olmo3")
    assert native_registry.dtype_from_hf_config(str(tmp_path), None, fam) == "fp32"


def test_prepare_settings_reach_convert(monkeypatch, tmp_path):
    """The prepare step's keywords are passed to the converter's convert() by the hook."""
    import importlib.util
    import sys
    import types

    fam = native_registry.get_family("olmo3")
    params = SimpleNamespace(seq_length=8192, olmo3_rope_layout="per_type", make_vocab_size_divisible_by=128)
    ctx = native_registry.ConvertContext(
        hf_path=str(tmp_path),
        hf_config=_RELEASED,
        family=fam,
        experiment={"tp": 8, "pp": 1, "ep": 1, "ckpt_format": "torch_dist"},
        options={},
        config_value=lambda name: getattr(params, name, None),
    )
    plan = fam.hook("prepare")(ctx)
    assert plan.convert_kwargs == {
        "seq_length": 8192,
        "rope_layout": "per_type",
        "make_vocab_size_divisible_by": 1024,
        "expected_padded_vocab_size": 100352,
    }

    # the hook passes opts["convert_kwargs"] to convert(): a stand-in converter module records it
    hook = (
        Path(__file__).resolve().parents[5]
        / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"
    )
    spec = importlib.util.spec_from_file_location("olmo3_family_hook", hook)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = []
    fake = types.ModuleType("fake_olmo3_conv")
    fake.convert = lambda *a, **k: calls.append(k)
    monkeypatch.setitem(sys.modules, fam.converter, fake)
    fake_common = types.ModuleType("fake_common")
    fake_common.ensure_megatron_on_path = lambda: None
    fake_common.apply_convert_patches = lambda **k: 0
    fake_common.check_ckpt_format = lambda f: f
    fake_common.DEFAULT_CKPT_FORMAT = "torch_dist"
    fake_common.assert_bridge_free = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "primus.backends.megatron.checkpoint.native_convert_common", fake_common)
    import primus.backends.megatron.checkpoint as ckpt_pkg

    monkeypatch.setattr(ckpt_pkg, "native_convert_common", fake_common, raising=False)
    (tmp_path / "config.json").write_text(json.dumps(_RELEASED))
    mod.native_convert_checkpoint(
        str(tmp_path), str(tmp_path / "out"), {"family": "olmo3", "convert_kwargs": plan.convert_kwargs}
    )
    assert len(calls) == 1
    assert {k: calls[0][k] for k in plan.convert_kwargs} == plan.convert_kwargs
    assert calls[0]["dtype"] == "bf16"


def _prepare(params, tp=1, hf_config=_RELEASED, argv=()):
    fam = native_registry.get_family("olmo3")
    ctx = native_registry.ConvertContext(
        hf_path="/hf",
        hf_config=hf_config,
        family=fam,
        experiment={"tp": tp, "pp": 1, "ep": 1, "ckpt_format": "torch_dist"},
        options={},
        argv=list(argv),
        config_value=lambda name: params.get(name),
    )
    return fam.hook("prepare")(ctx)


def test_prepare_accepts_whole_number_floats_and_strings():
    """YAML may give 8192.0 or "8192": a whole number is accepted as an int."""
    plan = _prepare({"seq_length": 8192.0, "make_vocab_size_divisible_by": "64"}, tp="2")
    assert plan.experiment is None  # no command-line TP: the experiment's settings stand
    assert plan.convert_kwargs["seq_length"] == 8192
    assert plan.convert_kwargs["make_vocab_size_divisible_by"] == 128


@pytest.mark.parametrize(
    "params, tp, key",
    [
        ({"seq_length": 8192.5}, 1, "seq_length"),
        ({"seq_length": "8192.0x"}, 1, "seq_length"),
        ({"seq_length": "1e4"}, 1, "seq_length"),
        ({"seq_length": "8192.0"}, 1, "seq_length"),
        ({}, 0, "tensor_model_parallel_size"),
        ({"seq_length": 0}, 1, "seq_length"),
        ({"seq_length": True}, 1, "seq_length"),
        ({"seq_length": float("inf")}, 1, "seq_length"),
        ({"make_vocab_size_divisible_by": -128}, 1, "make_vocab_size_divisible_by"),
        ({"olmo3_rope_layout": "auto"}, 1, "olmo3_rope_layout"),
        ({"olmo3_rope_layout": "yarn"}, 1, "olmo3_rope_layout"),
        ({}, 1.5, "tensor_model_parallel_size"),
    ],
)
def test_prepare_refuses_bad_settings_naming_the_key(params, tp, key):
    """A value that is not a positive whole number, or a rope layout convert() would refuse, stops
    the hook with NativeConvertRefused naming the key (no silent truncation)."""
    with pytest.raises(native_registry.NativeConvertRefused, match=key):
        _prepare(params, tp=tp)


def test_prepare_refuses_a_bad_vocab_size():
    with pytest.raises(native_registry.NativeConvertRefused, match="vocab_size"):
        _prepare({}, hf_config={**_RELEASED, "vocab_size": "big"})


@pytest.mark.parametrize(
    "argv",
    [
        ["--tensor_model_parallel_size", "8", "--make_vocab_size_divisible_by", "512"],
        ["--tensor-model-parallel-size=8", "make_vocab_size_divisible_by=512"],
    ],
)
def test_prepare_reads_command_line_overrides(argv):
    """The hook's command line (the training overrides) wins over the config file, as in training:
    TP 8 x divisor 512 pads 100278 rows to 102400, and the plan's experiment carries TP 8."""
    plan = _prepare({"seq_length": 4096}, tp=1, argv=argv + ["--seq_length", "8192"])
    assert plan.convert_kwargs == {
        "seq_length": 8192,
        "make_vocab_size_divisible_by": 4096,
        "expected_padded_vocab_size": 102400,
    }
    assert plan.path_suffix == "_vocab102400"
    assert plan.experiment == {"tp": 8, "pp": 1, "ep": 1, "ckpt_format": "torch_dist"}
    with pytest.raises(native_registry.NativeConvertRefused, match="tensor_model_parallel_size"):
        _prepare({}, argv=["--tensor_model_parallel_size", "0"])


@pytest.mark.parametrize("content", ["{bad", "[1, 2]", "\xff\xfe"])
def test_check_existing_reports_a_corrupt_settings_file(tmp_path, content):
    """A truncated / non-object settings file is a refusal reason (the check_existing contract),
    not a raw JSONDecodeError."""
    from primus.backends.megatron.checkpoint import olmo3_convert_hooks as hooks

    (tmp_path / hooks.CONVERT_SETTINGS_FILE).write_text(content)
    reason = hooks.check_existing(tmp_path, {"padded_vocab_size": 100352}, 128)
    assert "converter settings file" in reason and "Remove it" in reason and "= 128" in reason
    plan = _prepare({})
    assert "Remove it" in plan.check_existing(tmp_path)
    hooks.write_settings(tmp_path, {"padded_vocab_size": 100352})
    assert plan.check_existing(tmp_path) is None
