###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Loading Granite checkpoints: the build_args patch checks the model args against the HF config
the converter saves with the checkpoint (``native_loader_granite``,
``granite_checkpoint_args_patches``)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit_tests.backends.megatron.granite._granite_common import (
    GRANITE_HF,
    shipped_preset,
)

pytest.importorskip("torch")
pytest.importorskip("megatron")

from primus.backends.megatron.checkpoint import (  # noqa: E402
    native_loader_granite as conv,
)


# ---------------------------------------------------------------------------
# Loading a Granite checkpoint: the build_args patch checks the args against its saved config
# ---------------------------------------------------------------------------
def _checkpoint_dir(path, iteration=1):
    """A directory that holds a Megatron checkpoint: its tracker file and the iteration's
    directory (``iter_{:07d}``)."""
    path.mkdir(parents=True, exist_ok=True)
    (path / f"iter_{iteration:07d}").mkdir(exist_ok=True)
    (path / f"iter_{iteration:07d}" / "common.pt").write_text("weights")  # a checkpoint file
    (path / conv.TRACKER_FILE).write_text(str(iteration))  # Megatron writes it last
    return path


def _copy_path(ckpt, iteration=1):
    return Path(ckpt) / f"iter_{iteration:07d}" / conv.HF_CONFIG_COPY


def _converted_checkpoint(tmp_path, cfg=GRANITE_HF, name="ckpt"):
    import json

    ckpt = _checkpoint_dir(tmp_path / name)
    _copy_path(ckpt).write_text(json.dumps(cfg))
    return str(ckpt)


def _finetune_args(tmp_path, **change):
    values = {**shipped_preset(), "pretrained_checkpoint": _converted_checkpoint(tmp_path), "load": None}
    values.update(change)
    return SimpleNamespace(**values)


def test_finetune_args_matching_the_checkpoint_pass(tmp_path):
    logs = []
    args = _finetune_args(tmp_path, vocab_size=None)
    assert conv.check_args_against_checkpoint(args, log=logs.append) == GRANITE_HF
    assert logs == [
        f"[Primus:Granite] model args match the HF config {_copy_path(tmp_path / 'ckpt')} (the loaded checkpoint)"
    ]


@pytest.mark.parametrize(
    "change, needle",
    [
        ({"residual_multiplier": 0.5}, "residual_multiplier: preset 0.5"),
        ({"mup_embedding_mult": 1.0}, "mup_embedding_mult"),
        ({"norm_epsilon": 1e-6}, "norm_epsilon"),
        ({"softmax_scale": 0.125}, "softmax_scale"),
    ],
)
def test_finetune_args_differing_from_the_checkpoint_are_refused(tmp_path, change, needle):
    with pytest.raises(ValueError, match="does not match the Granite checkpoint") as e:
        conv.check_args_against_checkpoint(_finetune_args(tmp_path, **change))
    assert needle in str(e.value)


def test_finetune_check_covers_a_resume_through_load(tmp_path):
    # --load (a resume, or a converted checkpoint given directly) is checked like
    # pretrained_checkpoint.
    args = _finetune_args(tmp_path, residual_multiplier=0.5)
    args.load, args.pretrained_checkpoint = args.pretrained_checkpoint, None
    with pytest.raises(ValueError, match="residual_multiplier: preset 0.5"):
        conv.check_args_against_checkpoint(args)


@pytest.mark.parametrize("text", ["{trunc", "[1, 2]"])
def test_finetune_check_names_a_corrupt_config_copy(tmp_path, text):
    args = _finetune_args(tmp_path)
    copy = _copy_path(tmp_path / "ckpt")
    copy.write_text(text)
    with pytest.raises(ValueError) as e:
        conv.check_args_against_checkpoint(args)
    assert str(copy) in str(e.value)


def test_finetune_check_skips_a_run_without_a_checkpoint(tmp_path):
    logs = []
    args = _finetune_args(tmp_path, residual_multiplier=0.5, pretrained_checkpoint=None)
    assert conv.check_args_against_checkpoint(args, log=logs.append) is None
    assert logs == []  # pretraining: nothing to check, nothing to say


@pytest.mark.parametrize("key", ["hf_path", "tokenizer_model"])
def test_finetune_check_never_reads_a_tokenizer_or_hf_directory(tmp_path, key):
    # A local hf_path / tokenizer_model directory may hold another model (here Llama) or
    # another Granite revision: only the config saved with the checkpoint is used.
    import json

    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"model_type": "llama"}))
    empty = _checkpoint_dir(tmp_path / "empty")
    args = _finetune_args(tmp_path, pretrained_checkpoint=str(empty), **{key: str(other)})
    # The other directory's config.json is never used in place of the missing copy: the
    # check is skipped (a mismatching Llama config would have raised).
    assert conv.check_args_against_checkpoint(args, log=lambda m: None) is None
    # With the saved copy, the local directory is still ignored.
    args.pretrained_checkpoint = str(tmp_path / "ckpt")
    assert conv.check_args_against_checkpoint(args) == GRANITE_HF


def test_a_checkpoint_without_the_copy_loads_unchecked_with_a_warning(tmp_path):
    # Checkpoints saved by training, converted through Megatron-Bridge or made by another tool
    # carry no copy: Megatron loads them as usual, the check is skipped with a warning.
    logs = []
    plain = _checkpoint_dir(tmp_path / "plain")
    args = _finetune_args(tmp_path, pretrained_checkpoint=str(plain), residual_multiplier=0.5)
    assert conv.check_args_against_checkpoint(args, log=logs.append) is None
    assert len(logs) == 1 and "WARNING" in logs[0] and conv.HF_CONFIG_COPY in logs[0]
    # A resume through `load` of such a checkpoint, without pretrained_checkpoint, is not
    # refused either.
    args = _finetune_args(tmp_path, load=str(plain), pretrained_checkpoint=None, residual_multiplier=0.5)
    assert conv.check_args_against_checkpoint(args, log=logs.append) is None


def test_a_checkpoint_with_an_unreadable_tracker_is_not_checked(tmp_path):
    # No fallback to a copy at the checkpoint root: it is not tied to any iteration. Megatron
    # itself reports the unreadable tracker when it loads.
    import json

    ckpt = tmp_path / "c"
    ckpt.mkdir()
    (ckpt / conv.TRACKER_FILE).write_text("garbage")
    (ckpt / conv.HF_CONFIG_COPY).write_text(json.dumps({**GRANITE_HF, "model_type": "llama"}))
    logs = []
    args = SimpleNamespace(**{**shipped_preset(), "load": str(ckpt)})
    assert conv.check_args_against_checkpoint(args, log=logs.append) is None
    assert "WARNING" in logs[0]


def test_the_loaded_checkpoint_is_the_one_checked(tmp_path):
    # Megatron loads `load` when it holds a checkpoint, else `pretrained_checkpoint`. The copy
    # of the loaded checkpoint wins over pretrained_checkpoint's.
    import json

    resume = _converted_checkpoint(tmp_path, cfg={**GRANITE_HF, "residual_multiplier": 0.5}, name="resume")
    args = _finetune_args(tmp_path, load=resume, residual_multiplier=0.5)
    assert conv.checkpoint_to_load(args) == resume
    assert conv.check_args_against_checkpoint(args, log=lambda m: None)["residual_multiplier"] == 0.5
    assert json.loads(_copy_path(tmp_path / "ckpt").read_text())["residual_multiplier"] == 0.22
    # `load` without a checkpoint (first run of a resumable job): pretrained_checkpoint is checked.
    args = _finetune_args(tmp_path, load=str(tmp_path / "empty_load"), residual_multiplier=0.5)
    assert conv.checkpoint_to_load(args) == str(tmp_path / "ckpt")
    with pytest.raises(ValueError, match="residual_multiplier"):
        conv.check_args_against_checkpoint(args)


def test_a_resume_of_a_finetuning_run_is_checked_against_its_pretrained_checkpoint(tmp_path):
    # The fine-tuning run saved into `load` (Megatron's saves carry no copy); its resume is
    # checked against the converted pretrained_checkpoint the weights were trained from.
    resume = _checkpoint_dir(tmp_path / "sft", iteration=50)
    logs = []
    ok = _finetune_args(tmp_path, load=str(resume))
    assert conv.check_args_against_checkpoint(ok, log=logs.append) == GRANITE_HF
    assert "pretrained_checkpoint" in logs[-1]
    bad = _finetune_args(tmp_path, load=str(resume), residual_multiplier=0.5)
    with pytest.raises(ValueError, match="residual_multiplier: preset 0.5") as e:
        conv.check_args_against_checkpoint(bad)
    assert "pretrained_checkpoint" in str(e.value)


def test_ckpt_step_selects_the_iteration_checked(tmp_path):
    import json

    save = _converted_checkpoint(tmp_path)  # iteration 1 carries GRANITE_HF
    _checkpoint_dir(Path(save), iteration=20)  # tracker -> 20, with another config
    _copy_path(save, 20).write_text(json.dumps({**GRANITE_HF, "residual_multiplier": 0.5}))
    latest = SimpleNamespace(**{**shipped_preset(), "load": save})
    with pytest.raises(ValueError, match="residual_multiplier"):
        conv.check_args_against_checkpoint(latest)
    older = SimpleNamespace(**{**shipped_preset(), "load": save, "ckpt_step": 1})
    assert conv.check_args_against_checkpoint(older, log=lambda m: None) == GRANITE_HF


def test_iter_dir_name_is_megatrons(tmp_path):
    # The naming rule matches Megatron's own get_checkpoint_name (run where megatron.training
    # imports, i.e. on a GPU node).
    import torch

    if not torch.cuda.is_available():
        pytest.skip("megatron.training.checkpointing probes the GPU at import")
    from megatron.training.checkpointing import get_checkpoint_name

    for iteration, release in ((1, False), (1234567, False), (0, True)):
        expected = get_checkpoint_name(str(tmp_path), iteration, release=release, return_base_dir=True)
        assert str(tmp_path / conv._iter_dir_name(iteration, release)) == expected


def test_iteration_dir_follows_megatron_naming(tmp_path):
    ckpt = tmp_path / "c"
    ckpt.mkdir()
    assert conv.iteration_dir(str(ckpt)) is None  # no tracker
    (ckpt / conv.TRACKER_FILE).write_text("release")
    assert conv.iteration_dir(str(ckpt)) == ckpt / "release"
    (ckpt / conv.TRACKER_FILE).write_text("42\n")
    assert conv.iteration_dir(str(ckpt)) == ckpt / "iter_0000042"
    assert conv.iteration_dir(str(ckpt), ckpt_step=7) == ckpt / "iter_0000007"


def _patch_ctx(params, backend):
    return SimpleNamespace(
        extra={
            "module_config": SimpleNamespace(params=SimpleNamespace(**params)),
            "backend_args": SimpleNamespace(**backend),
        }
    )


def test_granite_checkpoint_args_patch_runs_the_check_for_granite_only(monkeypatch):
    # The check runs once per run from a build_args patch, not from the model builder.
    # In that phase model_type is still only in module_config.params (the runtime merges the
    # params into the backend args after build_args), so the patch reads both, backend first.
    from primus.backends.megatron.patches import (
        granite_checkpoint_args_patches as patch,
    )

    calls = []
    monkeypatch.setattr(conv, "check_args_against_checkpoint", lambda args, log: calls.append(args))
    ctx = _patch_ctx
    granite = ctx({"model_type": "granite", "hidden_size": 1}, {"hidden_size": 2560, "load": "ckpt"})
    args = patch._granite_args(granite)
    assert (args.model_type, args.hidden_size, args.load) == ("granite", 2560, "ckpt")
    assert patch._granite_args(ctx({}, {"model_type": "granite"})).model_type == "granite"
    assert patch._granite_args(ctx({"model_type": "gpt"}, {})) is None
    assert patch._granite_args(SimpleNamespace(extra={})) is None
    patch.patch_granite_checkpoint_args(granite)  # register_patch returns the function itself
    assert len(calls) == 1 and calls[0].hidden_size == 2560 and calls[0].model_type == "granite"
    # The backend args override model_type away from granite: the condition is false, so the
    # patch runner does not call the patch.
    assert not patch.is_granite_run(ctx({"model_type": "granite"}, {"model_type": "gpt"}))


def test_the_patch_keeps_no_state_and_wraps_nothing():
    # No module-level run state, no save_checkpoint wrapper: the only registered Granite
    # checkpoint patch is the build_args check.
    from primus.backends.megatron.patches import (
        granite_checkpoint_args_patches as patch,
    )
    from primus.core.patches.patch_registry import PatchRegistry

    assert not [n for n in vars(patch) if n.startswith("_SAVE") or n == "_INSTALLED_WRAPPER"]
    assert PatchRegistry.get("megatron.granite.save_hf_config") is None
    assert PatchRegistry.get("megatron.granite.save_hf_config_final") is None


def test_a_refused_checkpoint_stops_the_run(tmp_path, monkeypatch):
    # run_patches swallows ordinary exceptions; the refusal must stop the run (SystemExit).
    from primus.backends.megatron.patches import (
        granite_checkpoint_args_patches as patch,
    )
    from primus.core.patches import patch_runner

    monkeypatch.setattr(patch, "log_rank_0", lambda *a, **k: None)
    ckpt = _converted_checkpoint(tmp_path)
    ctx = _patch_ctx(
        {"model_type": "granite"}, {**shipped_preset(), "load": ckpt, "residual_multiplier": 0.5}
    )
    # Through the real runner, with only this patch enabled (it would swallow a ValueError).
    with pytest.raises(SystemExit, match="residual_multiplier: preset 0.5"):
        patch_runner.run_patches(
            backend="megatron",
            phase="build_args",
            extra=ctx.extra,
            enabled_ids=["megatron.granite.checkpoint_args"],
        )


@pytest.mark.parametrize("error", [OSError("unreadable"), KeyError("x"), TypeError("t")])
def test_any_failure_of_the_check_stops_the_run(monkeypatch, error):
    from primus.backends.megatron.patches import (
        granite_checkpoint_args_patches as patch,
    )

    def fail(args, log):
        raise error

    monkeypatch.setattr(patch, "log_rank_0", lambda *a, **k: None)
    monkeypatch.setattr(conv, "check_args_against_checkpoint", fail)
    with pytest.raises(SystemExit, match=type(error).__name__):
        patch.patch_granite_checkpoint_args(_patch_ctx({"model_type": "granite"}, {}))


@pytest.mark.parametrize(
    "params, backend, expected",
    [
        ({"model_type": "granite"}, {}, True),  # only in the module params (build_args phase)
        ({}, {"model_type": "granite"}, True),
        ({"model_type": "gpt"}, {"model_type": "granite"}, True),  # backend wins
        ({"model_type": "granite"}, {"model_type": "gpt"}, False),  # backend wins
        ({"model_type": "granite"}, {"model_type": None}, False),  # a set backend value wins, even falsy
        ({}, {}, False),
    ],
)
def test_granite_run_predicate_follows_the_runtime_merge_rule(params, backend, expected):
    # The runtime merges module params into the backend args with the backend's value kept
    # (merge_namespace(backend_args, params, allow_override=False)). Every Granite patch tests
    # "this run builds Granite" with is_granite_run, so it must follow that rule.
    from primus.backends.megatron.patches import (
        granite_checkpoint_args_patches as patch,
    )

    c = _patch_ctx(params, backend)
    assert patch.is_granite_run(c) is expected


def test_granite_convert_runs_the_granite_steps(monkeypatch, tmp_path):
    # native_loader_granite.convert runs the steps of native_loader_llama.convert with the
    # Granite fields and margs, the Granite model provider and the Llama weight mapping.
    import json

    from primus.backends.megatron.checkpoint import native_convert_common as common

    calls = []
    margs = SimpleNamespace(params_dtype="bf16", iteration=1)
    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: None)
    monkeypatch.setattr(common, "assert_bridge_free", lambda where: None)
    monkeypatch.setattr(common, "read_hf_config", lambda d: GRANITE_HF)
    monkeypatch.setattr(conv, "build_margs", lambda fields, args: calls.append(("margs", fields)) or margs)
    monkeypatch.setattr(common, "init_megatron_single_process", lambda m, port, log: m)
    monkeypatch.setattr(conv, "build_model", lambda m: calls.append(("model", m)) or "model")
    monkeypatch.setattr(common, "SafetensorsStore", lambda d: "store")
    monkeypatch.setattr(
        conv, "build_mcore_state_dict", lambda *a: calls.append(("map", a[0], a[1])) or ({}, set())
    )

    def save(*a):
        # The copy is written after the checkpoint, into its iteration directory.
        calls.append(("save", a[-2], _copy_path(tmp_path).exists()))
        _checkpoint_dir(tmp_path, 1)

    monkeypatch.setattr(common, "load_validate_save", save)
    logs = []
    out = conv.convert("hf", str(tmp_path), log=lambda *a: logs.append(" ".join(map(str, a))))
    assert out == str(tmp_path)
    assert calls[0] == ("margs", conv.derive_megatron_fields(GRANITE_HF))
    assert calls[1:] == [("model", margs), ("map", "model", "store"), ("save", str(tmp_path), False)]
    assert any(
        "residual_mult=0.22" in line and "emb_mult=12" in line and line.startswith("arch:") for line in logs
    ), logs
    # The HF config is saved with the checkpoint for the fine-tuning run's args check.
    assert json.loads(_copy_path(tmp_path).read_text()) == GRANITE_HF
    assert not list(tmp_path.rglob(conv.HF_CONFIG_COPY + "*.tmp"))


def test_granite_convert_writes_no_copy_when_the_mapping_fails(monkeypatch, tmp_path):
    from primus.backends.megatron.checkpoint import native_convert_common as common

    margs = SimpleNamespace(params_dtype="bf16", iteration=1)
    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: None)
    monkeypatch.setattr(common, "assert_bridge_free", lambda where: None)
    monkeypatch.setattr(common, "read_hf_config", lambda d: GRANITE_HF)
    monkeypatch.setattr(conv, "build_margs", lambda fields, args: margs)
    monkeypatch.setattr(common, "init_megatron_single_process", lambda m, port, log: m)
    monkeypatch.setattr(conv, "build_model", lambda m: "model")
    monkeypatch.setattr(common, "SafetensorsStore", lambda d: "store")
    monkeypatch.setattr(conv, "build_mcore_state_dict", lambda *a: ({}, set()))

    def fail(*a):
        raise RuntimeError("native conversion mapping incomplete/incorrect")

    monkeypatch.setattr(common, "load_validate_save", fail)
    with pytest.raises(RuntimeError, match="mapping incomplete"):
        conv.convert("hf", str(tmp_path), log=lambda *a: None)
    assert not list(tmp_path.rglob(conv.HF_CONFIG_COPY + "*"))


def test_granite_convert_removes_the_tracker_when_the_copy_fails(monkeypatch, tmp_path):
    from primus.backends.megatron.checkpoint import native_convert_common as common

    margs = SimpleNamespace(params_dtype="bf16", iteration=1)
    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: None)
    monkeypatch.setattr(common, "assert_bridge_free", lambda where: None)
    monkeypatch.setattr(common, "read_hf_config", lambda d: GRANITE_HF)
    monkeypatch.setattr(conv, "build_margs", lambda fields, args: margs)
    monkeypatch.setattr(common, "init_megatron_single_process", lambda m, port, log: m)
    monkeypatch.setattr(conv, "build_model", lambda m: "model")
    monkeypatch.setattr(common, "SafetensorsStore", lambda d: "store")
    monkeypatch.setattr(conv, "build_mcore_state_dict", lambda *a: ({}, set()))

    monkeypatch.setattr(common, "load_validate_save", lambda *a: _checkpoint_dir(tmp_path, 1))

    def enospc(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(conv, "_write_json_atomically", enospc)
    with pytest.raises(OSError, match="No space"):
        conv.convert("hf", str(tmp_path), log=lambda *a: None)
    # Without the tracker, the conversion hook converts again instead of skipping.
    assert not (tmp_path / conv.TRACKER_FILE).exists()
