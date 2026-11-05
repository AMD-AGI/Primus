###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
CPU tests for the checkpoint format of Primus' native HF -> Megatron conversion.

The converter writes ``torch_dist`` (resharded by Megatron on load, so any
TP / PP / EP) or the legacy ``torch`` format (TP1/PP1/EP1 only). These tests
cover the on-disk layout of each format, the format the post-train conversion
hook picks for an experiment, and that a converted checkpoint of one format is
never reused for a run that needs the other. Megatron is replaced by stand-in
modules, so the tests need neither the Megatron-LM submodule nor a GPU.
"""

import argparse
import importlib.machinery
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.checkpoint import native_convert_common as common

_REPO = Path(__file__).resolve().parents[4]
_HOOK = _REPO / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("native_convert_hook_under_test", _HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hook = _load_hook()


def _package(name, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    mod.__path__ = []
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _write_torch_dist(ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "metadata.json").write_text(json.dumps({"sharded_backend": "torch_dist"}))
    (ckpt_dir / "__0_0.distcp").write_bytes(b"")


def _write_torch(ckpt_dir: Path):
    (ckpt_dir / "mp_rank_00").mkdir(parents=True)
    (ckpt_dir / "mp_rank_00" / "model_optim_rng.pt").write_bytes(b"")


@pytest.fixture
def fake_save_checkpoint(monkeypatch):
    """Stand-in ``megatron.training.checkpointing.save_checkpoint``.

    Writes what Megatron writes for ``margs.ckpt_format`` at ``margs.iteration``:
    ``iter_XXXXXXX/metadata.json`` for torch_dist, ``iter_XXXXXXX/mp_rank_00`` for
    torch, plus the tracker file.
    """
    calls = []
    state = {}

    def save_checkpoint(iteration, models, optimizer, scheduler, num_floating_point_operations_so_far):
        margs = state["margs"]
        iter_dir = Path(margs.save) / f"iter_{iteration:07d}"
        if margs.ckpt_format == "torch_dist":
            _write_torch_dist(iter_dir)
        else:
            _write_torch(iter_dir)
        (Path(margs.save) / common.TRACKER_FILENAME).write_text(str(iteration))
        calls.append((iteration, margs.ckpt_format))

    checkpointing = _package("megatron.training.checkpointing", save_checkpoint=save_checkpoint)
    monkeypatch.setitem(sys.modules, "megatron", _package("megatron"))
    monkeypatch.setitem(
        sys.modules, "megatron.training", _package("megatron.training", checkpointing=checkpointing)
    )
    monkeypatch.setitem(sys.modules, "megatron.training.checkpointing", checkpointing)
    return state, calls


def _save(tmp_path, state, ckpt_format):
    model = torch.nn.Linear(4, 4, bias=False)
    sd = {"weight": torch.ones(4, 4)}
    store = SimpleNamespace(all_keys=lambda: {"hf.weight"})
    margs = SimpleNamespace(iteration=1, ckpt_format=ckpt_format, save=str(tmp_path / "out"))
    state["margs"] = margs
    common.load_validate_save(model, sd, {"hf.weight"}, store, margs, margs.save, log=lambda *a: None)
    return Path(margs.save)


# ---------------------------------------------------------------------------
# native_convert_common: saved layout per format
# ---------------------------------------------------------------------------
def test_default_format_is_torch_dist():
    assert common.DEFAULT_CKPT_FORMAT == "torch_dist"
    assert set(common.CKPT_FORMATS) == {"torch_dist", "torch"}


@pytest.mark.parametrize("bad", ["zarr", "torch_dcp", "", None])
def test_check_ckpt_format_rejects_unsupported(bad):
    with pytest.raises(ValueError, match="unsupported native checkpoint format"):
        common.check_ckpt_format(bad)


def test_torch_dist_save_uses_release_layout(tmp_path, fake_save_checkpoint):
    state, calls = fake_save_checkpoint
    out = _save(tmp_path, state, "torch_dist")

    assert calls == [(1, "torch_dist")]
    assert (out / common.TRACKER_FILENAME).read_text() == "release"
    assert (out / "release" / "metadata.json").is_file()
    assert not (out / "iter_0000001").exists()
    assert common.saved_ckpt_format(str(out)) == "torch_dist"


def test_legacy_torch_save_keeps_per_rank_layout(tmp_path, fake_save_checkpoint):
    state, calls = fake_save_checkpoint
    out = _save(tmp_path, state, "torch")

    assert calls == [(1, "torch")]
    assert (out / common.TRACKER_FILENAME).read_text() == "1"
    assert (out / "iter_0000001" / "mp_rank_00" / "model_optim_rng.pt").is_file()
    assert not (out / "release").exists()
    assert common.saved_ckpt_format(str(out)) == "torch"


def test_torch_dist_save_refuses_existing_release_dir(tmp_path, fake_save_checkpoint):
    state, calls = fake_save_checkpoint
    (tmp_path / "out" / "release").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="already has a 'release' directory"):
        _save(tmp_path, state, "torch_dist")
    assert calls == []


def test_incomplete_mapping_saves_nothing(tmp_path, fake_save_checkpoint):
    state, calls = fake_save_checkpoint
    margs = SimpleNamespace(iteration=1, ckpt_format="torch_dist", save=str(tmp_path / "out"))
    state["margs"] = margs
    store = SimpleNamespace(all_keys=lambda: {"hf.weight", "hf.unused"})
    with pytest.raises(RuntimeError, match="refusing to save"):
        common.load_validate_save(
            torch.nn.Linear(4, 4, bias=False),
            {"weight": torch.ones(4, 4)},
            {"hf.weight"},
            store,
            margs,
            margs.save,
            log=lambda *a: None,
        )
    assert calls == []


def test_saved_ckpt_format_missing_or_incomplete(tmp_path):
    assert common.saved_ckpt_format(str(tmp_path / "absent")) is None

    (tmp_path / common.TRACKER_FILENAME).write_text("release")
    assert common.saved_ckpt_format(str(tmp_path)) is None  # tracker, no release dir

    (tmp_path / "release").mkdir()
    assert common.saved_ckpt_format(str(tmp_path)) is None  # interrupted save: no metadata.json

    (tmp_path / common.TRACKER_FILENAME).write_text("not-a-number")
    assert common.saved_ckpt_format(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Family loaders: the requested format reaches Megatron's args
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_megatron_arguments(monkeypatch):
    arguments = _package(
        "megatron.training.arguments",
        add_megatron_arguments=lambda parser: parser,
        validate_args=lambda m: m,
    )
    monkeypatch.setitem(sys.modules, "megatron", _package("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.training", _package("megatron.training", arguments=arguments))
    monkeypatch.setitem(sys.modules, "megatron.training.arguments", arguments)


def _loader_args(ckpt_format):
    return argparse.Namespace(
        hf_dir="hf",
        save_dir="out",
        dtype="bf16",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        ckpt_format=ckpt_format,
        seq_length=128,
        make_vocab_size_divisible_by=128,
    )


_LLAMA_CFG = {
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "max_position_embeddings": 128,
    "intermediate_size": 128,
    "rms_norm_eps": 1e-6,
}


@pytest.mark.parametrize("ckpt_format", ["torch_dist", "torch"])
@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_loader_build_margs_sets_ckpt_format(fake_megatron_arguments, family, ckpt_format):
    conv = importlib.import_module(f"primus.backends.megatron.checkpoint.native_loader_{family}")
    margs = conv.build_margs(conv.derive_megatron_fields(dict(_LLAMA_CFG)), _loader_args(ckpt_format))
    assert margs.ckpt_format == ckpt_format


@pytest.mark.parametrize("family", ["llama", "qwen3", "deepseek"])
def test_loader_convert_rejects_unsupported_format_before_work(family):
    conv = importlib.import_module(f"primus.backends.megatron.checkpoint.native_loader_{family}")
    with pytest.raises(ValueError, match="unsupported native checkpoint format"):
        conv.convert("hf-dir-not-read", "out-not-written", ckpt_format="zarr", log=lambda *a: None)


# ---------------------------------------------------------------------------
# Conversion hook: format choice for the experiment
# ---------------------------------------------------------------------------
def _experiment(tp=1, pp=1, ep=1, ckpt_format="torch_dist"):
    return {"tp": tp, "pp": pp, "ep": ep, "ckpt_format": ckpt_format}


@pytest.mark.parametrize(
    "requested, experiment, expected",
    [
        (None, _experiment(), "torch_dist"),
        (None, _experiment(tp=2), "torch_dist"),
        (None, _experiment(pp=2, ep=8), "torch_dist"),
        (None, _experiment(ckpt_format="torch"), "torch"),  # TP1 legacy experiments: unchanged
        (None, _experiment(ckpt_format="torch_dcp"), "torch_dist"),
        ("torch_dist", _experiment(tp=4), "torch_dist"),
        ("torch", _experiment(), "torch"),
    ],
)
def test_resolve_format(requested, experiment, expected):
    assert hook._resolve_native_ckpt_format(requested, experiment) == expected


@pytest.mark.parametrize(
    "requested, experiment, message",
    [
        (None, _experiment(tp=2, ckpt_format="torch"), "cannot be loaded at TP=2 PP=1 EP=1"),
        ("torch", _experiment(pp=2), "cannot be loaded at TP=1 PP=2 EP=1"),
        ("torch", _experiment(ep=8), "cannot be loaded at TP=1 PP=1 EP=8"),
        ("torch_dist", _experiment(ckpt_format="torch"), "set ckpt_format: torch_dist"),
        ("zarr", _experiment(), "is not supported"),
    ],
)
def test_resolve_format_refuses_unloadable(requested, experiment, message, capsys):
    with pytest.raises(SystemExit):
        hook._resolve_native_ckpt_format(requested, experiment)
    assert message in capsys.readouterr().err


def test_read_experiment_checkpoint_settings(monkeypatch):
    params = SimpleNamespace(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        expert_model_parallel_size=None,
        ckpt_format="torch_dist",
    )
    module = SimpleNamespace(params=params)
    monkeypatch.setattr(hook, "load_primus_config", lambda path, cli: SimpleNamespace())
    monkeypatch.setattr(
        hook, "get_module_config", lambda cfg, name: module if name == "post_trainer" else None
    )

    assert hook.read_experiment_checkpoint_settings("exp.yaml") == {
        "tp": 2,
        "pp": 4,
        "ep": 1,
        "ckpt_format": "torch_dist",
    }


# ---------------------------------------------------------------------------
# Conversion hook: cache path and reuse
# ---------------------------------------------------------------------------
@pytest.fixture
def hf_llama(tmp_path):
    hf = tmp_path / "hf" / "tiny-llama"
    hf.mkdir(parents=True)
    (hf / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    return hf


@pytest.fixture
def fake_convert(monkeypatch):
    """Replace the converter: record the call and write the requested layout."""
    calls = []

    def native_convert_checkpoint(hf_path, megatron_path, opts):
        calls.append(dict(opts))
        out = Path(megatron_path)
        if opts["ckpt_format"] == "torch_dist":
            _write_torch_dist(out / "release")
            (out / common.TRACKER_FILENAME).write_text("release")
        else:
            _write_torch(out / "iter_0000001")
            (out / common.TRACKER_FILENAME).write_text("1")

    monkeypatch.setattr(hook, "native_convert_checkpoint", native_convert_checkpoint)
    monkeypatch.setenv("NODE_RANK", "0")
    return calls


def _emitted(capsys):
    lines = capsys.readouterr().out.splitlines()
    return dict(line.split("=", 1) for line in lines if line.startswith("extra."))


@pytest.mark.parametrize("ckpt_format", ["torch_dist", "torch"])
def test_default_output_path_includes_format(
    tmp_path, monkeypatch, capsys, hf_llama, fake_convert, ckpt_format
):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    hook._native_main(str(hf_llama), {"out": None, "ckpt_format": None}, _experiment(ckpt_format=ckpt_format))

    expected = tmp_path / "data" / "megatron_checkpoints" / f"tiny-llama_native_{ckpt_format}"
    assert [c["ckpt_format"] for c in fake_convert] == [ckpt_format]
    assert _emitted(capsys) == {"extra.pretrained_checkpoint": str(expected), "extra.finetune": "true"}
    assert common.saved_ckpt_format(str(expected)) == ckpt_format


def test_existing_checkpoint_of_same_format_is_reused(tmp_path, monkeypatch, capsys, hf_llama, fake_convert):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    hook._native_main(str(hf_llama), {"out": None, "ckpt_format": None}, _experiment(tp=2))
    capsys.readouterr()
    hook._native_main(str(hf_llama), {"out": None, "ckpt_format": None}, _experiment(tp=8, pp=2))

    assert len(fake_convert) == 1  # one torch_dist conversion serves every layout
    assert _emitted(capsys)["extra.pretrained_checkpoint"].endswith("tiny-llama_native_torch_dist")


def test_stale_legacy_checkpoint_is_not_reused_for_torch_dist(tmp_path, capsys, hf_llama, fake_convert):
    out = tmp_path / "ckpt"
    _write_torch(out / "iter_0000001")
    (out / common.TRACKER_FILENAME).write_text("1")

    with pytest.raises(SystemExit):
        hook._native_main(str(hf_llama), {"out": str(out), "ckpt_format": None}, _experiment(tp=2))

    assert fake_convert == []
    captured = capsys.readouterr()
    assert "holds a torch checkpoint, not torch_dist" in captured.err
    assert "extra.pretrained_checkpoint" not in captured.out


def test_incomplete_checkpoint_is_not_reused(tmp_path, capsys, hf_llama, fake_convert):
    out = tmp_path / "ckpt"
    (out / "release").mkdir(parents=True)
    (out / common.TRACKER_FILENAME).write_text("release")

    with pytest.raises(SystemExit):
        hook._native_main(str(hf_llama), {"out": str(out), "ckpt_format": None}, _experiment())

    assert fake_convert == []
    assert "holds a missing or incomplete checkpoint" in capsys.readouterr().err


def test_legacy_format_refused_before_conversion_at_tp2(
    tmp_path, monkeypatch, capsys, hf_llama, fake_convert
):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    with pytest.raises(SystemExit):
        hook._native_main(str(hf_llama), {"out": None, "ckpt_format": "torch"}, _experiment(tp=2))
    assert fake_convert == []
    assert not (tmp_path / "data").exists()
