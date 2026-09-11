###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from runner.helpers.hooks.train.pretrain.megatron import prepare as megatron_prepare
from runner.helpers.hooks.train.pretrain.nemo_automodel import prepare as nemo_prepare


class FakePrimusConfig:
    def __init__(self, pre_trainer):
        self.pre_trainer = pre_trainer

    def get_module_config(self, name):
        assert name == "pre_trainer"
        return self.pre_trainer


def trainer_config(**overrides):
    values = {
        "data_path": None,
        "eval_interval": 0,
        "eval_iters": 0,
        "full_validation": False,
        "model_type": None,
        "stage": None,
        "test_data_path": None,
        "tokenizer_model": "tokenizer",
        "tokenizer_type": "HuggingFaceTokenizer",
        "train_data_path": None,
        "trainer_class": None,
        "valid_data_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def write_tokenized(prefix: Path):
    Path(f"{prefix}.bin").parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}.bin").write_text("bin", encoding="utf-8")
    Path(f"{prefix}.idx").write_text("idx", encoding="utf-8")


@pytest.fixture(autouse=True)
def clean_prepare_env(monkeypatch):
    for name in (
        "HF_TOKEN",
        "TOKENIZED_DATA_PATH",
        "TOKENIZED_EVAL_DATA_PATH",
        "TOKENIZED_TRAIN_DATA_PATH",
        "PRIMUS_DATA_PREP_POLL_SECONDS",
        "PRIMUS_DATA_PREP_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(megatron_prepare, "get_node_rank", lambda: 0)


def megatron_prepare_args(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare.py",
            "--primus_path",
            str(tmp_path),
            "--data_path",
            str(tmp_path / "data"),
            "--config",
            str(tmp_path / "config.yaml"),
            *extra,
        ],
    )
    return megatron_prepare.parse_args()


def test_nemo_prepare_accepts_unknown_training_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare.py",
            "--primus_path",
            str(tmp_path),
            "--data_path",
            str(tmp_path / "data"),
            "--config",
            str(tmp_path / "config.yaml"),
            "--micro_batch_size",
            "2",
            "--model.attention_backend",
            "aiter",
        ],
    )
    args = nemo_prepare.parse_args()
    assert args.primus_path == str(tmp_path)


def test_megatron_sft_skips_bookcorpus(monkeypatch, tmp_path):
    monkeypatch.setattr(
        megatron_prepare,
        "prepare_dataset",
        lambda **kwargs: pytest.fail("SFT must not prepare BookCorpus"),
    )
    config = FakePrimusConfig(trainer_config(stage="sft"))
    megatron_prepare.prepare_dataset_if_needed(config, tmp_path)


def test_existing_valid_data_path_does_not_create_validation_split(monkeypatch, tmp_path, capsys):
    train_prefix = tmp_path / "existing_train"
    write_tokenized(train_prefix)
    monkeypatch.setenv("TOKENIZED_DATA_PATH", str(train_prefix))
    config = FakePrimusConfig(
        trainer_config(eval_interval=10, eval_iters=2, valid_data_path=str(tmp_path / "valid"))
    )

    megatron_prepare.prepare_dataset_if_needed(config, tmp_path)

    output = capsys.readouterr().out
    assert f"extra.train_data_path={train_prefix}" in output
    assert "extra.valid_data_path" not in output


def test_existing_test_data_path_does_not_create_validation_split(monkeypatch, tmp_path, capsys):
    train_prefix = tmp_path / "existing_train"
    write_tokenized(train_prefix)
    monkeypatch.setenv("TOKENIZED_DATA_PATH", str(train_prefix))
    config = FakePrimusConfig(
        trainer_config(eval_interval=10, eval_iters=2, test_data_path=str(tmp_path / "test"))
    )

    megatron_prepare.prepare_dataset_if_needed(config, tmp_path)

    output = capsys.readouterr().out
    assert f"extra.train_data_path={train_prefix}" in output
    assert "extra.test_data_path" not in output
    assert "extra.valid_data_path" not in output


def test_generated_split_only_fills_unset_eval_data_keys():
    configured = megatron_prepare.configured_eval_data_path_keys(trainer_config(test_data_path="/data/test"))
    assert configured == {"test_data_path"}
    assert megatron_prepare.configured_eval_data_path_keys(trainer_config()) == set()


def test_no_validation_uses_whole_corpus_prefix(monkeypatch, tmp_path, capsys):
    train_prefix = tmp_path / "custom_whole"
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("TOKENIZED_DATA_PATH", str(train_prefix))
    calls = []

    def fake_prepare_dataset(**kwargs):
        calls.append(kwargs)
        write_tokenized(kwargs["tokenized_data_path"])

    monkeypatch.setattr(megatron_prepare, "prepare_dataset", fake_prepare_dataset)
    megatron_prepare.prepare_dataset_if_needed(FakePrimusConfig(trainer_config()), tmp_path)

    assert calls[0]["tokenized_eval_data_path"] is None
    assert f"extra.train_data_path={train_prefix}" in capsys.readouterr().out


def test_validation_uses_custom_train_and_eval_prefixes(monkeypatch, tmp_path, capsys):
    train_prefix = tmp_path / "custom_train"
    eval_prefix = tmp_path / "custom_eval"
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("TOKENIZED_TRAIN_DATA_PATH", str(train_prefix))
    monkeypatch.setenv("TOKENIZED_EVAL_DATA_PATH", str(eval_prefix))

    def fake_prepare_dataset(**kwargs):
        write_tokenized(kwargs["tokenized_data_path"])
        write_tokenized(kwargs["tokenized_eval_data_path"])
        megatron_prepare.write_split_metadata(
            megatron_prepare.split_metadata_path(kwargs["tokenized_eval_data_path"]),
            kwargs["test_size"],
            kwargs["seed"],
        )

    monkeypatch.setattr(megatron_prepare, "prepare_dataset", fake_prepare_dataset)
    config = FakePrimusConfig(trainer_config(eval_interval=10, eval_iters=2))
    megatron_prepare.prepare_dataset_if_needed(config, tmp_path)

    output = capsys.readouterr().out
    assert f"extra.train_data_path={train_prefix}" in output
    assert f"extra.valid_data_path={eval_prefix}" in output
    assert f"extra.test_data_path={eval_prefix}" in output


def test_run_preprocess_honors_arbitrary_custom_prefix(monkeypatch, tmp_path):
    requested_prefix = tmp_path / "custom" / "training-prefix"

    def fake_run(command, **kwargs):
        preprocess_prefix = Path(command[command.index("--output-prefix") + 1])
        write_tokenized(Path(f"{preprocess_prefix}_text_sentence"))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(megatron_prepare.subprocess, "run", fake_run)
    megatron_prepare.run_preprocess(
        tmp_path / "dataset.json",
        requested_prefix,
        "HuggingFaceTokenizer",
        "tokenizer",
    )

    assert Path(f"{requested_prefix}.bin").exists()
    assert Path(f"{requested_prefix}.idx").exists()


def test_validation_cache_is_keyed_by_seed_and_test_size(monkeypatch, tmp_path):
    train_prefix = tmp_path / "train"
    eval_prefix = tmp_path / "eval"
    preprocess_calls = []
    split_files = []

    def fake_download(train_json, valid_json, test_size, seed):
        split_files.append((train_json.name, valid_json.name, test_size, seed))

    def fake_preprocess(dataset_json, output_prefix, *args, **kwargs):
        preprocess_calls.append((dataset_json.name, output_prefix))
        write_tokenized(output_prefix)

    monkeypatch.setattr(megatron_prepare, "download_bookcorpus_split", fake_download)
    monkeypatch.setattr(megatron_prepare, "run_preprocess", fake_preprocess)

    def prepare(test_size, seed):
        megatron_prepare.prepare_dataset(
            tmp_path,
            "HuggingFaceTokenizer",
            "tokenizer",
            train_prefix,
            eval_prefix,
            test_size=test_size,
            seed=seed,
        )

    prepare(0.1, 11)
    prepare(0.1, 11)
    assert len(preprocess_calls) == 2
    prepare(0.1, 12)
    prepare(0.2, 12)

    assert len(preprocess_calls) == 6
    assert len({entry[0] for entry in split_files}) == 3
    assert megatron_prepare.split_metadata_matches(megatron_prepare.split_metadata_path(eval_prefix), 0.2, 12)


def test_split_seed_does_not_consume_the_training_seed(monkeypatch, tmp_path):
    """--seed belongs to Megatron; this hook must leave it for the training args.

    primus-cli forwards one argument list to the hook and to training, so a
    ``--seed`` the hook swallowed would both retune the split and go missing from
    the config overrides.
    """
    args, unknown = megatron_prepare_args(monkeypatch, tmp_path, "--seed", "1234")

    assert not hasattr(args, "seed")
    assert unknown == ["--seed", "1234"], "the training seed must pass straight through"
    assert args.split_seed == megatron_prepare.DEFAULT_SPLIT_SEED


def test_split_seed_comes_from_the_cli_flag_or_the_default(monkeypatch, tmp_path):
    """``--split_seed`` is the only entry point; there is no environment layer."""
    args, _ = megatron_prepare_args(monkeypatch, tmp_path)
    assert args.split_seed == megatron_prepare.DEFAULT_SPLIT_SEED == 42

    args, unknown = megatron_prepare_args(monkeypatch, tmp_path, "--split_seed", "9")
    assert args.split_seed == 9
    assert unknown == []


def test_split_seed_rename_keeps_existing_caches_valid(tmp_path):
    """Renaming the flag must not invalidate split caches already on disk.

    The cache key and the .split.json payload are keyed on "seed" regardless of
    what the flag is called, so at the unchanged default they still resolve to the
    filenames and metadata written before the rename.
    """
    assert megatron_prepare.split_cache_key(0.005, megatron_prepare.DEFAULT_SPLIT_SEED) == "6da01fc840cb"

    eval_prefix = tmp_path / "eval"
    metadata_path = megatron_prepare.split_metadata_path(eval_prefix)
    metadata_path.write_text('{"seed": 42, "test_size": 0.005}\n', encoding="utf-8")
    assert megatron_prepare.split_metadata_matches(metadata_path, 0.005, 42)


def test_rank0_failure_writes_failure_marker(monkeypatch, tmp_path):
    train_prefix = tmp_path / "train"
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("TOKENIZED_DATA_PATH", str(train_prefix))

    def fail_prepare(**kwargs):
        raise RuntimeError("preprocess failed")

    monkeypatch.setattr(megatron_prepare, "prepare_dataset", fail_prepare)
    config = FakePrimusConfig(trainer_config())
    with pytest.raises(RuntimeError, match="preprocess failed"):
        megatron_prepare.prepare_dataset_if_needed(config, tmp_path)

    _, failed_flag = megatron_prepare.dataset_coordination_paths(
        train_prefix,
        None,
        "HuggingFaceTokenizer",
        "tokenizer",
        0.005,
        42,
    )
    assert "preprocess failed" in failed_flag.read_text(encoding="utf-8")


def test_nonzero_rank_propagates_failure_and_times_out(monkeypatch, tmp_path):
    done = tmp_path / "done"
    failed = tmp_path / "failed"
    monkeypatch.setattr(megatron_prepare, "get_node_rank", lambda: 1)
    megatron_prepare.atomic_write(failed, "rank0 exploded\n")
    with pytest.raises(SystemExit):
        megatron_prepare.wait_for_dataset(done, failed)

    failed.unlink()
    monkeypatch.setenv("PRIMUS_DATA_PREP_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("PRIMUS_DATA_PREP_POLL_SECONDS", "0.005")
    with pytest.raises(SystemExit):
        megatron_prepare.wait_for_dataset(done, failed)
