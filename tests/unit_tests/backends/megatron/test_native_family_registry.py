###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
CPU tests for the family registry of Primus' native HF -> Megatron converter.

The registry (``native_registry``) holds what the post-train conversion hook
used to hard-code per family: the HF names that select a family, its converter
module, its options, the dtype fallback, the model_type its mcore model is built
with, and hook-side steps. These tests check that the built-in families behave
as before (same detection, dispatch and dtype), that a family registered from
outside the core files is detected, dispatched and given its options, and that
the messages list the registered families. Megatron is not needed.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from primus.backends.megatron.checkpoint import native_convert_common as common
from primus.backends.megatron.checkpoint import native_registry as registry
from primus.backends.megatron.checkpoint.native_registry import (
    ConvertPlan,
    NativeConvertRefused,
    NativeFamily,
    NativeOption,
    register_native_family,
)

_REPO = Path(__file__).resolve().parents[4]
_HOOK = _REPO / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("native_registry_hook_under_test", _HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hook = _load_hook()

# The hook's table before the registry: every name must still select the same family.
_TABLE_BEFORE_REGISTRY = {
    "qwen3": "qwen3",
    "qwen3_moe": "qwen3",
    "qwen3forcausallm": "qwen3",
    "qwen3moeforcausallm": "qwen3",
    "deepseek_v2": "deepseek",
    "deepseek_v3": "deepseek",
    "deepseekv2forcausallm": "deepseek",
    "deepseekv3forcausallm": "deepseek",
    "deepseek": "deepseek",
    "llama": "llama",
    "llamaforcausallm": "llama",
}
_FAKE_CONVERTER = "native_registry_test_fake_converter"


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    """Each test sees the built-ins (plus what it registers); the registry is restored after."""
    saved = (dict(registry._FAMILIES), dict(registry._NAMES), dict(registry._LOAD_ERRORS), registry._loaded)
    yield
    registry._FAMILIES.clear()
    registry._FAMILIES.update(saved[0])
    registry._NAMES.clear()
    registry._NAMES.update(saved[1])
    registry._LOAD_ERRORS.clear()
    registry._LOAD_ERRORS.update(saved[2])
    registry._loaded = saved[3]


@pytest.fixture
def fake_converter(monkeypatch):
    """A converter module whose ``convert`` records its call and writes a torch_dist checkpoint."""
    calls = []

    def convert(hf_dir, save_dir, **kwargs):
        calls.append((hf_dir, save_dir, kwargs))
        out = Path(save_dir)
        (out / "release").mkdir(parents=True, exist_ok=True)
        (out / "release" / "metadata.json").write_text("{}")
        (out / common.TRACKER_FILENAME).write_text("release")
        return save_dir

    module = types.ModuleType(_FAKE_CONVERTER)
    module.convert = convert
    monkeypatch.setitem(sys.modules, _FAKE_CONVERTER, module)
    return calls


@pytest.fixture
def no_megatron_setup(monkeypatch):
    """native_convert_checkpoint's Megatron path / patch steps, stubbed out."""
    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: "")
    monkeypatch.setattr(common, "apply_convert_patches", lambda module_name=None, platform=None: 0)


def _hf(tmp_path, cfg, name="model"):
    hf = tmp_path / "hf" / name
    hf.mkdir(parents=True)
    (hf / "config.json").write_text(json.dumps(cfg))
    return str(hf)


def _experiment(tp=1, pp=1, ep=1, ckpt_format="torch_dist"):
    return {"tp": tp, "pp": pp, "ep": ep, "ckpt_format": ckpt_format}


def _toy_family(**fields):
    base = dict(
        name="toy",
        label="Toy LM",
        converter=_FAKE_CONVERTER,
        architectures=("ToyForCausalLM",),
        model_types=("toy",),
    )
    base.update(fields)
    return register_native_family(NativeFamily(**base))


def _emitted(capsys):
    lines = capsys.readouterr().out.splitlines()
    return dict(line.split("=", 1) for line in lines if line.startswith("extra."))


# ---------------------------------------------------------------------------
# Built-in families: unchanged
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,family", sorted(_TABLE_BEFORE_REGISTRY.items()))
def test_builtin_names_select_the_same_family(tmp_path, name, family):
    # as an architecture (any case) and as a model_type, as the old table did
    assert hook._detect_native_family(_hf(tmp_path, {"architectures": [name.upper()]}, "a")) == family
    assert hook._detect_native_family(_hf(tmp_path, {"model_type": name}, "m")) == family


def test_builtin_detection_order_and_misses(tmp_path):
    # first matching architecture wins; then model_type; else None
    cfg = {"architectures": ["Unknown", "LlamaForCausalLM"], "model_type": "qwen3"}
    assert hook._detect_native_family(_hf(tmp_path, cfg, "a")) == "llama"
    cfg = {"architectures": ["Unknown"], "model_type": "qwen3_moe"}
    assert hook._detect_native_family(_hf(tmp_path, cfg, "b")) == "qwen3"
    assert hook._detect_native_family(_hf(tmp_path, {"architectures": ["MistralForCausalLM"]}, "c")) is None
    assert hook._detect_native_family(str(tmp_path / "no-config")) is None


def test_builtin_families_converters_and_messages():
    names = [f.name for f in registry.families()[:3]]
    assert names == ["qwen3", "deepseek", "llama"]
    for name in names:
        fam = registry.get_family(name)
        assert fam.converter == f"primus.backends.megatron.checkpoint.native_loader_{name}"
        assert fam.model_type == "gpt" and fam.options == () and fam.prepare is None and fam.wait is None
    assert registry.supported_text().startswith("supported: Qwen3 dense/MoE, DeepSeek-V2/V3, Llama 2/3/3.1")


@pytest.mark.parametrize(
    "cfg,expected",
    [
        ({"torch_dtype": "bfloat16"}, "bf16"),
        ({"torch_dtype": "float16"}, "fp16"),
        ({"torch_dtype": "float32"}, "fp32"),
        ({"torch_dtype": "int8"}, "bf16"),
        ({}, "bf16"),
        ({"dtype": "float32"}, "bf16"),  # built-ins read torch_dtype only, as before
    ],
)
def test_builtin_dtype_unchanged(tmp_path, cfg, expected):
    hf = _hf(tmp_path, {"architectures": ["LlamaForCausalLM"], **cfg})
    assert hook._native_dtype_from_config(hf, None, "llama") == expected
    assert hook._native_dtype_from_config(hf, None) == expected
    assert hook._native_dtype_from_config(hf, "fp16", "llama") == "fp16"


def test_builtin_dispatch_unchanged(tmp_path, monkeypatch, no_megatron_setup):
    from primus.backends.megatron.checkpoint import native_loader_llama

    calls = []
    monkeypatch.setattr(native_loader_llama, "convert", lambda hf, out, **kw: calls.append((hf, out, kw)))
    hf = _hf(tmp_path, {"architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16"})
    hook.native_convert_checkpoint(hf, "out", {"ckpt_format": None})
    assert calls == [
        (
            hf,
            "out",
            dict(
                dtype="bf16",
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                expert_parallel_size=1,
                ckpt_format="torch_dist",
            ),
        )
    ]


def test_unregistered_family_name_is_refused_not_sent_to_deepseek(tmp_path, no_megatron_setup):
    hf = _hf(tmp_path, {"architectures": ["LlamaForCausalLM"]})
    with pytest.raises(ValueError, match=r"no native converter family 'mistral' \(supported: Qwen3"):
        hook.native_convert_checkpoint(hf, "out", {"family": "mistral"})


def test_unknown_model_message_lists_registered_families(tmp_path, monkeypatch, capsys):
    _toy_family()
    hf = _hf(tmp_path, {"architectures": ["MistralForCausalLM"]})
    with pytest.raises(ValueError) as e:
        hook.native_convert_checkpoint(hf, "out", {})
    assert "Qwen3 dense/MoE, DeepSeek-V2/V3, Llama 2/3/3.1, Toy LM" in str(e.value)

    monkeypatch.setattr(hook, "get_checkpoint_config", lambda config: (hf, None))
    monkeypatch.setattr(
        hook,
        "read_native_opts",
        lambda config: {"enabled": True, "dtype": None, "out": None, "ckpt_format": None},
    )
    monkeypatch.setattr(sys, "argv", ["hook", "--config", "/exp.yaml"])
    with pytest.raises(SystemExit):
        hook.main()
    err = capsys.readouterr().err
    assert f"no native converter matches the HF model at {hf}" in err
    assert "Llama 2/3/3.1, Toy LM" in err and "native_ckpt_convert: false" in err


# ---------------------------------------------------------------------------
# Registering a family
# ---------------------------------------------------------------------------
def test_register_detects_and_dispatches(tmp_path, fake_converter, no_megatron_setup):
    _toy_family()
    hf = _hf(tmp_path, {"architectures": ["ToyForCausalLM"], "torch_dtype": "float32"})
    assert hook._detect_native_family(hf) == "toy"
    hook.native_convert_checkpoint(hf, str(tmp_path / "out"), {"ckpt_format": "torch"})
    assert fake_converter[0][2] == dict(
        dtype="fp32",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        ckpt_format="torch",
    )


def test_register_rejects_conflicts():
    fam = _toy_family()
    assert register_native_family(fam) is fam  # the same entry again: no-op
    with pytest.raises(ValueError, match="already registered"):
        _toy_family(label="Other")
    with pytest.raises(ValueError, match="'llamaforcausallm' already selects family 'llama'"):
        register_native_family(
            NativeFamily(name="x", label="X", converter="m", architectures=("LlamaForCausalLM",))
        )
    with pytest.raises(TypeError, match="NativeOption"):
        register_native_family(NativeFamily(name="y", label="Y", converter="m", options=("max_params",)))


def test_model_type_needs_no_architectures(tmp_path):
    # e.g. a model_type shared by task heads the converter does not build
    _toy_family(model_type_needs_no_architectures=True)
    assert hook._detect_native_family(_hf(tmp_path, {"model_type": "toy"}, "a")) == "toy"
    cfg = {"architectures": ["ToyForSequenceClassification"], "model_type": "toy"}
    assert hook._detect_native_family(_hf(tmp_path, cfg, "b")) is None
    assert hook._detect_native_family(_hf(tmp_path, {"architectures": "ToyForCausalLM"}, "c")) == "toy"


def test_family_modules_are_discovered(tmp_path, monkeypatch, capsys):
    pkg = tmp_path / "fams_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "toy.py").write_text(
        "from primus.backends.megatron.checkpoint.native_registry import NativeFamily, register_native_family\n"
        "register_native_family(NativeFamily(name='toy', label='Toy LM', converter='m',"
        " architectures=('ToyForCausalLM',)))\n"
    )
    (pkg / "broken.py").write_text("raise ImportError('needs a missing package')\n")
    (pkg / "_private.py").write_text("raise AssertionError('never imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(registry, "FAMILY_PACKAGE", "fams_pkg")
    registry._loaded = False
    try:
        errors = registry.load_family_modules()
        assert registry.family_for_config({"architectures": ["ToyForCausalLM"]}).name == "toy"
        assert list(errors) == ["fams_pkg.broken"]
        assert "fams_pkg.broken failed to import" in capsys.readouterr().err
        assert "failed to import: fams_pkg.broken" in registry.unknown_model_text("/m")
    finally:
        for name in ("fams_pkg", "fams_pkg.toy", "fams_pkg.broken"):
            sys.modules.pop(name, None)


def test_shipped_family_package_imports():
    # the real native_families package: every module imports and registers cleanly
    registry._loaded = False
    registry._LOAD_ERRORS.clear()
    assert registry.load_family_modules() == {}


# ---------------------------------------------------------------------------
# Options, dtype fallback, model_type
# ---------------------------------------------------------------------------
def test_options_defaults_and_explicit_values():
    fam = _toy_family(
        options=(
            NativeOption("native_ckpt_max_params", "max_params"),
            NativeOption("native_ckpt_seq_length", "seq_length", default=4096),
        )
    )
    assert registry.read_options(fam, {}.get) == {"seq_length": 4096}
    assert registry.read_options(fam, {"native_ckpt_max_params": 0}.get) == {
        "max_params": 0,
        "seq_length": 4096,
    }
    config = {"native_ckpt_max_params": "", "native_ckpt_seq_length": 8192}
    assert registry.read_options(fam, config.get) == {"seq_length": 8192}


def test_options_reach_the_converter(tmp_path, monkeypatch, capsys, fake_converter, no_megatron_setup):
    _toy_family(options=(NativeOption("native_ckpt_max_params", "max_params"),))
    monkeypatch.setattr(hook, "_config_reader", lambda config_file: {"native_ckpt_max_params": 10**9}.get)
    monkeypatch.setenv("NODE_RANK", "0")
    hf = _hf(tmp_path, {"architectures": ["ToyForCausalLM"]})
    out = tmp_path / "out"
    hook._native_main(hf, {"out": str(out), "ckpt_format": None}, _experiment(), config_file="exp.yaml")
    assert fake_converter[0][2]["max_params"] == 10**9
    assert _emitted(capsys)["extra.pretrained_checkpoint"] == str(out)


def test_config_reader_params_then_module_and_lazy(monkeypatch):
    loads = []
    params = types.SimpleNamespace(native_ckpt_max_params=0, seq_length=None)
    module = types.SimpleNamespace(params=params, seq_length=8192, native_ckpt_max_params=5)
    monkeypatch.setattr(hook, "load_primus_config", lambda path, cli: loads.append(path) or object())
    monkeypatch.setattr(
        hook, "get_module_config", lambda cfg, name: module if name == "post_trainer" else None
    )

    value = hook._config_reader("exp.yaml")
    assert loads == []  # nothing read until a family asks
    assert value("native_ckpt_max_params") == 0  # params first; an explicit 0 is kept
    assert value("seq_length") == 8192  # unset in params: the module's
    assert value("missing") is None
    assert len(loads) == 1
    assert hook._config_reader(None)("seq_length") is None


def test_family_dtype_fallback_only_without_torch_dtype(tmp_path):
    _toy_family(dtype_from_config=lambda cfg: cfg.get("dtype") or cfg.get("text_config", {}).get("dtype"))
    assert hook._native_dtype_from_config(_hf(tmp_path, {"dtype": "float32"}, "a"), None, "toy") == "fp32"
    cfg = {"text_config": {"dtype": "float16"}}
    assert hook._native_dtype_from_config(_hf(tmp_path, cfg, "b"), None, "toy") == "fp16"
    cfg = {"torch_dtype": "bfloat16", "dtype": "float32"}
    assert hook._native_dtype_from_config(_hf(tmp_path, cfg, "c"), None, "toy") == "bf16"


def test_hooks_given_as_strings_are_imported_on_use(tmp_path, monkeypatch):
    helpers = types.ModuleType("native_registry_test_helpers")
    helpers.dtype = lambda cfg: "float16"
    monkeypatch.setitem(sys.modules, "native_registry_test_helpers", helpers)
    _toy_family(dtype_from_config="native_registry_test_helpers:dtype")
    assert hook._native_dtype_from_config(_hf(tmp_path, {}), None, "toy") == "fp16"
    with pytest.raises(ValueError, match="module:function"):
        registry.resolve_hook("no_colon")


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, "gpt"),
        ({"model_type": "mamba"}, "mamba"),
        ({"family": "toy"}, "toy_mt"),
        ({"family": "llama"}, "gpt"),
    ],
)
def test_build_mcore_gpt_model_model_type(monkeypatch, kwargs, expected):
    _toy_family(model_type="toy_mt")
    seen = []

    def get_model_provider(model_type):
        seen.append(model_type)
        return lambda pre_process, post_process: torch.nn.Linear(2, 2)

    import primus.core.utils.import_utils as import_utils

    monkeypatch.setattr(import_utils, "get_model_provider", get_model_provider)
    model = common.build_mcore_gpt_model(types.SimpleNamespace(params_dtype=torch.float16), **kwargs)
    assert seen == [expected] and model.weight.dtype == torch.float16


# ---------------------------------------------------------------------------
# prepare / wait hooks
# ---------------------------------------------------------------------------
def test_prepare_plan_kwargs_suffix_and_before_convert(
    tmp_path, monkeypatch, capsys, fake_converter, no_megatron_setup
):
    seen = {}

    def prepare(ctx):
        seen.update(argv=ctx.argv, cfg=ctx.hf_config, experiment=ctx.experiment, options=ctx.options)
        return ConvertPlan(
            convert_kwargs={"make_vocab_size_divisible_by": 256},
            path_suffix="_vocab1024",
            before_convert=lambda path: (path.mkdir(parents=True), (path / "settings.json").write_text("{}")),
        )

    _toy_family(
        prepare=prepare, options=(NativeOption("native_ckpt_seq_length", "seq_length", default=4096),)
    )
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("NODE_RANK", "0")
    hf = _hf(tmp_path, {"architectures": ["ToyForCausalLM"]}, "toy-1b")
    hook._native_main(
        hf, {"out": None, "ckpt_format": None}, _experiment(tp=2), argv=["--seq_length", "8192"]
    )

    expected = tmp_path / "data" / "megatron_checkpoints" / "toy-1b_native_torch_dist_vocab1024"
    assert _emitted(capsys)["extra.pretrained_checkpoint"] == str(expected)
    assert (expected / "settings.json").is_file()
    assert seen == {
        "argv": ["--seq_length", "8192"],
        "cfg": {"architectures": ["ToyForCausalLM"]},
        "experiment": _experiment(tp=2),
        "options": {"seq_length": 4096},
    }
    kwargs = fake_converter[0][2]
    assert kwargs["make_vocab_size_divisible_by"] == 256 and kwargs["seq_length"] == 4096


def test_prepare_suffix_not_added_to_native_ckpt_out(
    tmp_path, monkeypatch, capsys, fake_converter, no_megatron_setup
):
    _toy_family(prepare=lambda ctx: ConvertPlan(path_suffix="_vocab1024"))
    monkeypatch.setenv("NODE_RANK", "0")
    out = tmp_path / "out"
    hook._native_main(
        _hf(tmp_path, {"model_type": "toy"}), {"out": str(out), "ckpt_format": None}, _experiment()
    )
    assert _emitted(capsys)["extra.pretrained_checkpoint"] == str(out)


def test_prepare_refusal_exits_before_conversion(tmp_path, capsys, fake_converter):
    def prepare(ctx):
        raise NativeConvertRefused("this checkpoint has a feature the converter does not support")

    _toy_family(prepare=prepare)
    with pytest.raises(SystemExit):
        hook._native_main(
            _hf(tmp_path, {"model_type": "toy"}), {"out": None, "ckpt_format": None}, _experiment()
        )
    assert fake_converter == []
    assert "does not support" in capsys.readouterr().err


def test_prepare_experiment_replaces_the_layout_checked(tmp_path, capsys, fake_converter):
    # e.g. a TP override read from the command line: a legacy torch checkpoint is then refused
    _toy_family(prepare=lambda ctx: ConvertPlan(experiment={**ctx.experiment, "tp": 4}))
    hf = _hf(tmp_path, {"model_type": "toy"})
    with pytest.raises(SystemExit):
        hook._native_main(
            hf, {"out": str(tmp_path / "o"), "ckpt_format": "torch"}, _experiment(ckpt_format="torch")
        )
    assert fake_converter == [] and "TP=4" in capsys.readouterr().err


def test_check_existing_refuses_reuse(tmp_path, monkeypatch, capsys, fake_converter):
    out = tmp_path / "out"
    (out / "release").mkdir(parents=True)
    (out / "release" / "metadata.json").write_text("{}")
    (out / common.TRACKER_FILENAME).write_text("release")
    verdict = {"reason": "holds 1024 vocab rows, this run builds 2048"}
    _toy_family(prepare=lambda ctx: ConvertPlan(check_existing=lambda path: verdict["reason"]))
    hf = _hf(tmp_path, {"model_type": "toy"})

    with pytest.raises(SystemExit):
        hook._native_main(hf, {"out": str(out), "ckpt_format": None}, _experiment())
    assert "1024 vocab rows" in capsys.readouterr().err

    verdict["reason"] = None
    hook._native_main(hf, {"out": str(out), "ckpt_format": None}, _experiment())
    assert fake_converter == [] and _emitted(capsys)["extra.pretrained_checkpoint"] == str(out)


def test_wait_hook_replaces_default_wait_on_other_nodes(tmp_path, monkeypatch, capsys, fake_converter):
    waits = []
    _toy_family(wait=lambda path, done, lock, rank: waits.append((path, done.name, lock.name, rank)))
    monkeypatch.setenv("NODE_RANK", "1")
    monkeypatch.setattr(hook, "wait_for_conversion", lambda *a: pytest.fail("default wait used"))
    out = tmp_path / "out"
    hook._native_main(
        _hf(tmp_path, {"model_type": "toy"}), {"out": str(out), "ckpt_format": None}, _experiment()
    )
    assert waits == [(out, "out.done", "out.converting.lock", 1)] and fake_converter == []


def test_builtin_waits_with_default(tmp_path, monkeypatch, capsys):
    waits = []
    monkeypatch.setenv("NODE_RANK", "1")
    monkeypatch.setattr(hook, "wait_for_conversion", lambda done, lock: waits.append(done.name))
    out = tmp_path / "out"
    hf = _hf(tmp_path, {"architectures": ["LlamaForCausalLM"]})
    hook._native_main(hf, {"out": str(out), "ckpt_format": None}, _experiment())
    assert waits == ["out.done"]


# ---------------------------------------------------------------------------
# tracked_ckpt_dir
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("content,expected", [("release", "release"), ("7", "iter_0000007"), ("x", None)])
def test_tracked_ckpt_dir(tmp_path, content, expected):
    assert common.tracked_ckpt_dir(str(tmp_path)) is None
    (tmp_path / common.TRACKER_FILENAME).write_text(content + "\n")
    got = common.tracked_ckpt_dir(str(tmp_path))
    assert got == (str(tmp_path / expected) if expected else None)
