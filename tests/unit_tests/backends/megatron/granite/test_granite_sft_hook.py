###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The Granite entries of the posttrain checkpoint hook (01_convert_checkpoints.py): family
detection and the dispatch to the Granite converter. The hook's functions are called directly; the converters are replaced by
recorders, so no model is built."""

import importlib
import importlib.util
import inspect
import json
import pathlib
import sys
import types

import pytest


def _hook():
    path = (
        pathlib.Path(__file__).resolve().parents[5]
        / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"
    )
    spec = importlib.util.spec_from_file_location("granite_convert_hook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path, cfg):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return str(tmp_path)


def _fake_common():
    """A stand-in for native_convert_common with what the hook calls before convert()."""
    common = types.ModuleType(f"{_PKG}.native_convert_common")
    common.ensure_megatron_on_path = lambda: None
    common.apply_convert_patches = lambda module_name: 0
    common.DEFAULT_CKPT_FORMAT = "torch_dist"
    common.check_ckpt_format = lambda fmt: fmt
    return common


@pytest.mark.parametrize(
    "cfg, family",
    [
        ({"architectures": ["GraniteForCausalLM"]}, "granite"),
        ({"architectures": ["GraniteForCausalLM"], "model_type": "granite"}, "granite"),
        # A Granite task head shares model_type "granite": not a native family (Bridge path).
        ({"architectures": ["GraniteForSequenceClassification"], "model_type": "granite"}, None),
        # No architectures: model_type "granite" is the dense model (what the converter accepts).
        ({"model_type": "granite"}, "granite"),
        ({"model_type": "granite", "architectures": []}, "granite"),
        ({"architectures": ["LlamaForCausalLM"]}, "llama"),
        ({"model_type": "qwen3"}, "qwen3"),
        ({"architectures": ["DeepseekV3ForCausalLM"]}, "deepseek"),
        ({"model_type": "granitemoehybrid"}, None),
        # A single architecture written as a string is one name, not a list of letters.
        ({"architectures": "GraniteForCausalLM", "model_type": "granite"}, "granite"),
        ({"architectures": "GraniteForSequenceClassification", "model_type": "granite"}, None),
    ],
)
def test_detect_native_family(tmp_path, cfg, family):
    assert _hook()._detect_native_family(_write(tmp_path, cfg)) == family


@pytest.mark.parametrize(
    "cfg",
    [
        {"architectures": ["GraniteForCausalLM"]},
        {"architectures": ["GraniteForCausalLM"], "model_type": "granite"},
        {"architectures": "GraniteForCausalLM", "model_type": "granite"},
        {"model_type": "granite"},
        {"model_type": "granite", "architectures": []},
        {"model_type": "granite", "architectures": None},
        {"architectures": ["GraniteForSequenceClassification"], "model_type": "granite"},
        {"model_type": "granitemoehybrid"},
        {"model_type": "granite", "architectures": ["LlamaForCausalLM"]},
    ],
)
def test_hook_and_converter_agree_on_granite_configs(tmp_path, cfg):
    # One rule: a config the hook sends to the Granite converter passes the converter's
    # model-type check (an empty architectures list counts as none in both), and a Granite
    # config the hook does not send there is refused by it.
    conv = importlib.import_module(f"{_PKG}.native_loader_granite")
    routed = _hook()._detect_native_family(_write(tmp_path, cfg)) == "granite"
    try:
        conv._check_model_type(cfg)
        accepted = True
    except ValueError:
        accepted = False
    assert routed == accepted


_LOADERS = {
    "qwen3": "native_loader_qwen3",
    "llama": "native_loader_llama",
    "granite": "native_loader_granite",
    "deepseek": "native_loader_deepseek",
}
_PKG = "primus.backends.megatron.checkpoint"


@pytest.mark.parametrize(
    "cfg, family",
    [
        ({"architectures": ["GraniteForCausalLM"], "model_type": "granite"}, "granite"),
        ({"model_type": "granite"}, "granite"),
        ({"architectures": ["LlamaForCausalLM"]}, "llama"),
        ({"model_type": "qwen3"}, "qwen3"),
        ({"architectures": ["DeepseekV3ForCausalLM"]}, "deepseek"),
    ],
)
def test_native_convert_dispatches_to_the_family_converter(tmp_path, monkeypatch, cfg, family):
    # Each family reaches its own loader's convert(): a Granite checkpoint must not fall
    # through to another family's converter.
    calls = []
    common = _fake_common()
    monkeypatch.setitem(sys.modules, common.__name__, common)
    for fam, name in _LOADERS.items():
        loader = types.ModuleType(f"{_PKG}.{name}")
        loader.convert = lambda hf, out, _fam=fam, **kw: calls.append((_fam, hf, out, kw))
        monkeypatch.setitem(sys.modules, loader.__name__, loader)
    pkg = importlib.import_module(_PKG)
    for name in ("native_convert_common", *_LOADERS.values()):
        monkeypatch.setattr(pkg, name, sys.modules[f"{_PKG}.{name}"], raising=False)

    hf_path = _write(tmp_path, cfg)
    _hook().native_convert_checkpoint(hf_path, str(tmp_path / "out"), {"dtype": "bf16"})
    assert [c[0] for c in calls] == [family]
    assert calls[0][1:3] == (hf_path, str(tmp_path / "out"))
    assert calls[0][3]["dtype"] == "bf16"
    assert calls[0][3]["ckpt_format"] == "torch_dist"


@pytest.mark.parametrize("name", ["native_loader_granite", "native_loader_llama"])
def test_the_hook_calls_convert_with_its_real_signature(tmp_path, monkeypatch, name):
    # The recorder above accepts any keyword: bind the hook's call to the real convert()
    # signature, so a renamed or dropped keyword fails here and not in an SFT run.
    pytest.importorskip("torch")
    real = importlib.import_module(f"{_PKG}.{name}").convert
    signature = inspect.signature(real)
    bound = []
    common = _fake_common()
    monkeypatch.setitem(sys.modules, common.__name__, common)
    loader = types.ModuleType(f"{_PKG}.{name}")
    loader.convert = lambda *a, **kw: bound.append(signature.bind(*a, **kw))
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    pkg = importlib.import_module(_PKG)
    monkeypatch.setattr(pkg, "native_convert_common", common, raising=False)
    monkeypatch.setattr(pkg, name, loader, raising=False)
    cfg = (
        {"architectures": ["GraniteForCausalLM"], "model_type": "granite"}
        if name == "native_loader_granite"
        else {"architectures": ["LlamaForCausalLM"]}
    )
    _hook().native_convert_checkpoint(_write(tmp_path, cfg), str(tmp_path / "out"), {"dtype": "bf16"})
    assert len(bound) == 1


def test_granite_is_registered_with_the_native_registry():
    # native_families/granite.py registers the family: the converter, the granite model
    # provider, and the model_type rule the converter's _check_model_type also applies.
    registry = importlib.import_module(f"{_PKG}.native_registry")
    family = registry.get_family("granite")
    assert family.converter == f"{_PKG}.native_loader_granite"
    assert family.model_type == "granite"
    assert registry.model_type_for("granite") == "granite"
    assert family.model_type_needs_no_architectures
    assert "Granite" in registry.supported_text()
