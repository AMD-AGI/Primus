###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for tools/convergence_test.

The convergence test is only as good as its reading of the logs: a parser that
drops validation points, misplaces a run boundary, or silently falls back to an
older run produces a plausible curve for the wrong run. These pin down the
MaxText and Megatron log shapes, the TensorBoard reader, the baseline verdict,
MaxText's config precedence, and the MaxText dataset writer.

Pure CPU; no GPU, JAX, or Primus runtime required.
"""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_DIR = REPO_ROOT / "tools" / "convergence_test"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"convergence_{name}", TOOL_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot_loss = _load("plot_loss")
resolve_config = _load("resolve_config")
prepare = _load("prepare_dataset")
check_config = _load("check_config")


def maxtext_step(step, loss, tokens=1000):
    return (
        f"[20260929 19:48:42][rank-0/1][INFO]     completed step: {step}, seconds: 1.150, "
        f"TFLOP/s/device: 470.000, Tokens/s/device: 57000.000, total_weights: {tokens}, "
        f"loss: {loss + 0.01:.3f}, lm_loss: {loss:.3f}, perplexity: 1.000\n"
    )


def maxtext_eval(step, loss):
    return (
        f"[20260929 19:48:42][rank-0/1][INFO]     Completed eval after train step {step}, "
        f"loss={loss:.3f}, perplexity=1.000, total_weights=5000.0, avg_z_loss=0.000\n"
    )


def maxtext_run(steps, start_loss, eval_interval):
    """A MaxText run as it appears in a log: evaluations precede their step."""
    lines = ["\x1b[1m[INFO] Config param steps: %d\x1b[0m\n" % steps]
    for step in range(steps):
        if step % eval_interval == 0:
            lines.append(maxtext_eval(step, start_loss - step * 0.1 - 0.05))
        lines.append(maxtext_step(step, start_loss - step * 0.1))
    return lines


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------


def test_maxtext_steps_are_reported_from_one(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("".join(maxtext_run(steps=6, start_loss=11.0, eval_interval=3)))

    ((train, valid),) = plot_loss.parse_log(log)

    assert [r["iteration"] for r in train] == [1, 2, 3, 4, 5, 6]
    assert train[0]["loss"] == pytest.approx(11.0)  # lm_loss, not the total loss
    assert train[0]["total_loss"] == pytest.approx(11.01)
    assert train[0]["elapsed_ms"] == pytest.approx(1150.0)
    assert train[0]["step_tokens"] == 1000
    assert train[0]["total_iters"] == 6
    # "after train step N" evaluates the weights after N+1 updates
    assert [v["iteration"] for v in valid] == [1, 4]


def test_appended_maxtext_runs_split_on_the_early_evaluation(tmp_path):
    """The second run's first evaluation is logged before its first step, so it
    must open the new run rather than attach to the end of the first one."""
    log = tmp_path / "run.log"
    log.write_text("".join(maxtext_run(8, 11.0, 4) + maxtext_run(4, 10.0, 2)))

    runs = plot_loss.parse_log(log)

    assert len(runs) == 2
    (first_train, first_valid), (second_train, second_valid) = runs
    assert first_train[-1]["iteration"] == 8
    assert [v["iteration"] for v in first_valid] == [1, 5]
    assert [v["iteration"] for v in second_valid] == [1, 3]
    assert second_train[0]["loss"] == pytest.approx(10.0)
    assert second_train[0]["total_iters"] == 4


def test_megatron_line_still_parses(tmp_path):
    log = tmp_path / "debug.log"
    log.write_text(
        " iteration        1/    2000 | consumed samples:          128 | elapsed time per iteration (ms): "
        "1330.5 | learning rate: 3.000000E-06 | global batch size:   128 | lm loss: 1.214700E+01 | "
        "grad norm: 10.693 | number of skipped iterations:   0 | number of nan iterations:   0 |\n"
        " validation loss at iteration 1 | lm loss value: 1.100000E+01 |\n"
    )

    ((train, valid),) = plot_loss.parse_log(log)

    assert train[0]["iteration"] == 1 and train[0]["total_iters"] == 2000
    assert train[0]["loss"] == pytest.approx(12.147)
    assert train[0]["global_batch_size"] == 128
    assert valid == [{"iteration": 1, "loss": 11.0}]


def test_megatron_end_of_training_evaluation(tmp_path):
    """Megatron evaluates again after the last iteration. A probe has only that
    evaluation; a full run keeps its in-run point at the same iteration."""

    def iteration(i):
        return f" iteration {i:8d}/      30 | lm loss: 7.000000E+00 |\n"

    def validation(i, loss, where=""):
        return f" validation loss at iteration {i}{where} | lm loss value: {loss:.6E} |\n"

    log = tmp_path / "debug.log"
    log.write_text(
        iteration(30) + validation(30, 7.1, " on validation set") + validation(30, 7.2, " on test set")
    )
    ((_, valid),) = plot_loss.parse_log(log)
    assert valid == [{"iteration": 30, "loss": pytest.approx(7.1)}]

    log.write_text(iteration(30) + validation(30, 7.0) + validation(30, 7.1, " on validation set"))
    ((_, valid),) = plot_loss.parse_log(log)
    assert valid == [{"iteration": 30, "loss": pytest.approx(7.0)}]


def test_validation_comes_from_the_same_run_in_the_experiment_log(tmp_path):
    """The console has this run's iterations only; the experiment log has its
    evaluations, after those of an earlier run of the same config."""

    def run(losses, valid_loss):
        lines = [
            f" iteration {i:8d}/{len(losses):8d} | lm loss: {loss:.6E} |\n"
            for i, loss in enumerate(losses, 1)
        ]
        return (
            "".join(lines)
            + f" validation loss at iteration {len(losses)} | lm loss value: {valid_loss:.6E} |\n"
        )

    experiment = tmp_path / "exp" / "logs" / "pre_trainer" / "rank-7"
    experiment.mkdir(parents=True)
    (experiment / "debug.log").write_text(run([10.5, 9.0, 8.1], 8.2) + run([10.5, 9.1, 8.0], 8.05))
    console = tmp_path / "console.log"

    console.write_text("".join(run([10.5, 9.1, 8.0], 0).splitlines(True)[:3]))
    ((train, valid),) = plot_loss.parse_log(console)
    plot_loss.merge_validation(train, valid, tmp_path / "exp")
    assert valid == [{"iteration": 3, "loss": pytest.approx(8.05)}]

    # an earlier run is never mistaken for this one
    console.write_text("".join(run([10.5, 9.2, 8.3], 0).splitlines(True)[:3]))
    ((train, valid),) = plot_loss.parse_log(console)
    plot_loss.merge_validation(train, valid, tmp_path / "exp")
    assert valid == []


def test_directory_without_iterations_is_distinguishable(tmp_path):
    (tmp_path / "logs" / "pre_trainer" / "rank-0").mkdir(parents=True)
    (tmp_path / "logs" / "pre_trainer" / "rank-0" / "debug.log").write_text("Training stopped: boom\n")

    with pytest.raises(plot_loss.NoIterations):
        plot_loss.load_runs(tmp_path)
    with pytest.raises(plot_loss.NoIterations):
        plot_loss.load_runs(tmp_path / "missing.csv")


def test_csv_round_trip_keeps_unlogged_validation_points(tmp_path):
    train = [{"iteration": i, "loss": 10.0 - i, "step_tokens": 100.0} for i in (10, 20)]
    valid = [{"iteration": 15, "loss": 8.5}, {"iteration": 20, "loss": 7.9}]
    path = tmp_path / "run.csv"

    plot_loss.write_csv(path, train, valid)
    ((read_train, read_valid),) = plot_loss.parse_csv(path)

    assert [r["iteration"] for r in read_train] == [10, 20]
    assert read_train[0]["step_tokens"] == 100.0
    assert read_valid == valid


def test_summary_reads_back_a_csv_without_every_column(tmp_path, capsys):
    path = tmp_path / "old.csv"
    path.write_text(
        "iteration,loss,valid_loss,elapsed_ms,peak_mem_pct\n1,12.1,,900,21.8\n2,11.9,11.8,800,21.8\n"
    )

    ((train, valid),) = plot_loss.parse_csv(path)
    plot_loss.summarise("old", train, valid)

    assert "peak memory       : (21.8%)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "iterations, expect, missing",
    [
        ([1, 2, 3], 3, 0),
        ([1, 2, 3], 10, 7),
        ([1980, 1990, 2000], 2005, 0),  # log_interval does not divide train_iters
        ([1970, 1980, 1990], 2000, 10),
    ],
)
def test_stopped_early_allows_for_the_logging_stride(iterations, expect, missing):
    train = [{"iteration": i} for i in iterations]
    assert plot_loss.stopped_early(train, expect) == missing


# ---------------------------------------------------------------------------
# TensorBoard
# ---------------------------------------------------------------------------


def _varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _field(number, wire, payload):
    key = _varint((number << 3) | wire)
    if wire == 2:
        return key + _varint(len(payload)) + payload
    return key + payload


def _event(step, tag, value):
    """tensorboardX's scalar Event: {step, summary {value {tag, simple_value}}}."""
    scalar = _field(1, 2, tag.encode()) + _field(2, 5, struct.pack("<f", value))
    summary = _field(1, 2, scalar)
    payload = _field(1, 1, struct.pack("<d", 0.0)) + _field(2, 0, _varint(step)) + _field(5, 2, summary)
    return struct.pack("<Q", len(payload)) + b"\0" * 4 + payload + b"\0" * 4


def test_tensorboard_scalars_fill_in_what_the_log_omits(tmp_path):
    events = tmp_path / "tb" / "run"
    events.mkdir(parents=True)
    records = b"".join(
        _event(step, tag, value)
        for step in range(3)
        for tag, value in (
            ("learning/lm_loss", 11.0 - step * 0.1 + 1e-4),
            ("learning/current_learning_rate", 1e-4 * (step + 1)),
            ("learning/grad_norm", 1.0),
            ("learning/raw_grad_norm", 5.0 - step),
        )
    )
    records += _event(0, "eval/avg_loss", 10.9512)
    # A record still being written is ignored rather than misread.
    (events / "events.out.tfevents.1.host").write_bytes(records + struct.pack("<Q", 999))

    log = tmp_path / "run.log"
    log.write_text("".join(maxtext_run(3, 11.0, 5)))
    ((train, valid),) = plot_loss.parse_log(log)
    plot_loss.merge_tensorboard(train, valid, tmp_path / "tb")

    assert train[0]["loss"] == pytest.approx(11.0001, abs=1e-5)
    assert [r["lr"] for r in train] == pytest.approx([1e-4, 2e-4, 3e-4])
    assert [r["grad_norm"] for r in train] == pytest.approx([5.0, 4.0, 3.0])  # pre-clip
    assert "aux_loss" not in train[0]  # a dense model has no balancing loss
    assert valid[0]["loss"] == pytest.approx(10.9512, abs=1e-5)


def test_tensorboard_from_another_run_is_ignored(tmp_path):
    events = tmp_path / "tb"
    events.mkdir()
    (events / "events.out.tfevents.1.host").write_bytes(
        b"".join(_event(step, "learning/lm_loss", 3.0) for step in range(3))
    )
    log = tmp_path / "run.log"
    log.write_text("".join(maxtext_run(3, 11.0, 5)))
    ((train, valid),) = plot_loss.parse_log(log)

    plot_loss.merge_tensorboard(train, valid, events)

    assert train[0]["loss"] == pytest.approx(11.0)
    assert "lr" not in train[0]


# ---------------------------------------------------------------------------
# Baseline verdict
# ---------------------------------------------------------------------------


def _curve(offset, iterations=200):
    train = [{"iteration": i, "loss": 10.0 / i + offset} for i in range(1, iterations + 1)]
    valid = [{"iteration": i, "loss": 10.0 / i + offset} for i in range(50, iterations + 1, 50)]
    return train, valid


def test_baseline_passes_within_tolerance():
    assert plot_loss.compare_to_baseline(*_curve(0.01), *_curve(0.0), tolerance=0.05)


def test_baseline_fails_outside_tolerance():
    assert not plot_loss.compare_to_baseline(*_curve(0.08), *_curve(0.0), tolerance=0.05)


@pytest.mark.parametrize("train_offset, valid_offset", [(0.08, 0.0), (0.0, 0.08)])
def test_baseline_checks_training_and_validation_separately(train_offset, valid_offset):
    train, valid = _curve(train_offset)[0], _curve(valid_offset)[1]
    assert not plot_loss.compare_to_baseline(train, valid, *_curve(0.0), tolerance=0.05)


def test_baseline_fails_a_run_that_stopped_early():
    train, valid = _curve(0.0, iterations=150)
    assert not plot_loss.compare_to_baseline(train, valid, *_curve(0.0), tolerance=0.05)


# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------


@pytest.fixture
def lint():
    for findings in (check_config.ERRORS, check_config.WARNINGS, check_config.NOTES):
        findings.clear()
    return check_config


def _megatron(**params):
    return resolve_config.ResolvedConfig(
        path=Path("x.yaml"),
        framework="megatron",
        exp_root_path="./output/x",
        exp_name="x",
        params=params,
        effective=params,
    )


def test_lint_reads_exponents_written_without_a_dot(lint):
    # YAML 1.1 loads `lr: 1e-5` (no dot) as a string
    lint.check_schedule(_megatron(train_iters=1000, lr="1e-5", lr_decay_iters="1000", lr_warmup_iters=100))
    assert [w for w, _ in lint.WARNINGS] == [
        "lr is 1e-05, which is a throughput-benchmark value; the loss will barely move"
    ]


def test_a_probe_is_the_head_of_a_longer_schedule(lint):
    cfg = _megatron(
        train_iters=30,
        lr=1.5e-4,
        lr_decay_iters=1000,
        lr_warmup_iters=100,
        stderr_sink_level="DEBUG",
        disable_tensorboard=False,
        eval_iters=0,
    )
    lint.check_schedule(cfg)
    lint.check_observability(cfg)
    assert len(lint.WARNINGS) == 2  # the decay horizon, and no validation

    lint.WARNINGS.clear()
    lint.check_schedule(cfg, probe=True)
    lint.check_observability(cfg, probe=True)
    assert lint.WARNINGS == [] and lint.ERRORS == []
    assert "probe: the first 30 iterations of a 1000-iteration schedule" in lint.NOTES


def test_warmup_must_end_before_the_decay_horizon(lint):
    # Megatron asserts lr_warmup_steps < lr_decay_steps before the first iteration
    lint.check_schedule(_megatron(train_iters=50, lr=1.5e-4, lr_decay_iters=50, lr_warmup_iters=100))
    assert "not below the 50-iteration decay horizon" in lint.ERRORS[0][0]

    lint.ERRORS.clear()
    lint.check_schedule(_megatron(train_iters=50, lr=1.5e-4, lr_warmup_iters=100))  # decays over train_iters
    assert len(lint.ERRORS) == 1


def test_global_batch_divides_by_micro_batch_times_data_parallel(lint, monkeypatch):
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    monkeypatch.setenv("NNODES", "1")
    common = {"train_iters": 10, "micro_batch_size": 4, "seq_length": 4096}

    lint.check_budget(_megatron(global_batch_size=96, tensor_model_parallel_size=2, **common), None)
    assert lint.ERRORS == []  # data parallel 4: 96 = 4 x 4 x 6

    lint.check_budget(_megatron(global_batch_size=100, **common), None)
    assert "micro_batch_size x data-parallel size (4 x 8)" in lint.ERRORS[0][0]


# ---------------------------------------------------------------------------
# MaxText config precedence
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_maxtext(tmp_path, monkeypatch):
    configs = tmp_path / "src" / "maxtext" / "configs"
    (configs / "models").mkdir(parents=True)
    (configs / "base.yml").write_text(
        "steps: 150_001\nlearning_rate: 3.e-5\nvocab_size: 32000\nnum_experts: 1\n"
    )
    (configs / "models" / "tiny.yml").write_text("vocab_size: 32000\nnum_experts: 8\nbase_emb_dim: 64\n")
    monkeypatch.setenv("MAXTEXT_PATH", str(tmp_path))
    return configs


def test_maxtext_model_file_beats_plain_overrides(fake_maxtext):
    params = {
        "framework": "maxtext",
        "stderr_sink_level": "INFO",
        "base_config": "base.yml",
        "model_name": "tiny",
        "override_model_config": True,
        "steps": 20,
        "vocab_size": 64000,
        "override_model": {"base_emb_dim": 128},
    }

    effective, configs, model_file, shadowed = resolve_config._merge_maxtext(params)

    assert configs == fake_maxtext and model_file.name == "tiny.yml"
    assert effective["steps"] == 20  # overlay beats base.yml
    assert effective["learning_rate"] == pytest.approx(3e-5)  # inherited
    assert effective["vocab_size"] == 32000  # the model file beats the overlay...
    assert effective["base_emb_dim"] == 128  # ...but not override_model
    assert shadowed == {"vocab_size": (64000, 32000)}
    assert "stderr_sink_level" not in effective


def test_maxtext_global_batch_follows_the_device_count(monkeypatch):
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    monkeypatch.setenv("NNODES", "2")
    cfg = resolve_config.ResolvedConfig(
        path=Path("x.yaml"),
        framework="maxtext",
        exp_root_path="./output/x",
        exp_name="x",
        params={},
        effective={"per_device_batch_size": 1.5, "gradient_accumulation_steps": 2, "max_target_length": 4096},
    )
    assert cfg.global_batch_size == 48
    assert cfg.seq_length == 4096


def test_maxtext_datasets_are_keyed_by_sequence_length(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    assert resolve_config.default_dataset_dir("meta-llama/Llama-3.2-1B").name == "fineweb-edu-llama-3.2-1b"
    assert (
        resolve_config.default_dataset_dir("meta-llama/Llama-3.2-1B", "maxtext", 4096).name
        == "fineweb-edu-llama-3.2-1b-maxtext-4096"
    )


# ---------------------------------------------------------------------------
# MaxText dataset writer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, streams",
    [
        # One pass over one stream keeps the two splits disjoint...
        ("fineweb-edu", [("train", ["valid", "train"])]),
        # ...and a corpus with an official held-out split validates on it.
        ("c4", [("validation", ["valid"]), ("train", ["train"])]),
    ],
)
def test_splits_are_drawn_from_the_right_corpus_stream(monkeypatch, source, streams):
    calls = []
    monkeypatch.setattr(
        prepare,
        "_fill_from_stream",
        lambda _spec, split, splits, *rest: calls.append((split, [s[0] for s in splits])),
    )
    spec = resolve_config.parse_source(source)
    corpus = prepare.source_splits(spec)

    prepare.build_splits(spec, [(n, None, 1, corpus[n]) for n in ("valid", "train")], "tok", False, 1, 1, 1)

    assert calls == streams


def test_datasets_are_keyed_by_corpus(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    assert (
        resolve_config.default_dataset_dir("meta-llama/Llama-2-7b-hf", "maxtext", 4096, "c4").name
        == "c4-llama-2-7b-hf-maxtext-4096"
    )
    assert (
        resolve_config.default_dataset_dir(
            "Qwen/Qwen3-8B", "megatron", None, "HuggingFaceFW/fineweb:sample-10BT"
        ).name
        == "huggingfacefw-fineweb-sample-10bt-qwen3-8b"
    )


def test_hub_sources(monkeypatch):
    c4 = resolve_config.parse_source("c4")
    assert (c4["repo"], c4["name"], c4["tag"]) == ("allenai/c4", "en", "c4")
    assert resolve_config.source_splits(c4) == {"valid": "validation", "train": "train"}

    hub = resolve_config.parse_source("HuggingFaceFW/fineweb:sample-10BT")
    assert (hub["kind"], hub["repo"], hub["name"]) == ("hub", "HuggingFaceFW/fineweb", "sample-10BT")
    # no held-out split: validate on the head of the training stream
    assert resolve_config.source_splits(hub) == {"valid": "train", "train": "train"}
    assert resolve_config.parse_source("roneneldan/TinyStories")["name"] is None

    for bad in ("fineweb", "not a repo", "a/b/c"):
        with pytest.raises(ValueError):
            resolve_config.parse_source(bad)


def test_local_sources(tmp_path, monkeypatch):
    corpus = tmp_path / "My Corpus"
    corpus.mkdir()
    for name in ("b.jsonl", "a.jsonl", "notes.md"):
        (corpus / name).write_text('{"text": "x"}\n')

    by_dir = resolve_config.parse_source(str(corpus))
    by_glob = resolve_config.parse_source(str(corpus / "*.jsonl"))
    assert by_dir["kind"] == "local" and by_dir["builder"] == "json"
    assert [Path(f).name for f in by_dir["files"]] == ["a.jsonl", "b.jsonl"]  # sorted, .md ignored
    assert by_dir["tag"].startswith("local-my-corpus-") and by_glob["tag"].startswith("local-my-corpus-")
    assert by_dir["tag"] != by_glob["tag"]  # different selections, different datasets

    monkeypatch.chdir(tmp_path)
    assert resolve_config.parse_source("./My Corpus")["tag"] == by_dir["tag"]

    (corpus / "c.parquet").write_text("")
    with pytest.raises(ValueError, match="mixes file types"):
        resolve_config.parse_source(str(corpus))
    with pytest.raises(ValueError, match="no .* files match"):
        resolve_config.parse_source(str(tmp_path / "missing" / "*.jsonl"))


def test_local_corpus_is_read_through_its_text_column(tmp_path):
    pytest.importorskip("datasets")
    path = tmp_path / "docs.jsonl"
    path.write_text('{"content": "first document"}\n{"content": "second document"}\n')
    spec = resolve_config.parse_source(str(path))

    records = prepare.open_stream(spec, "train", "content")
    batches = list(prepare.stream_batches(records, "content", batch_size=8, min_chars=1))
    assert batches == [["first document", "second document"]]

    with pytest.raises(SystemExit, match="has no 'text' column .*content"):
        prepare.open_stream(spec, "train", "text")


def test_parquet_writer_chunks_documents_and_replaces_stale_shards(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    prefix = str(tmp_path / "train")
    (tmp_path / "train-00000-of-00009.parquet").write_text("stale")

    writer = prepare.ParquetShardWriter(prefix, num_shards=2, seq_length=4)
    for doc in ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12], [13, 14, 15]):
        writer.add(doc)
    writer.finalize()

    files = prepare.parquet_files(prefix)
    assert [Path(f).name for f in files] == ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
    rows = [pq.read_table(f).column("tokens").to_pylist() for f in files]
    # documents alternate between shards; a document's rows stay together and in order
    assert rows[0] == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10], [13, 14, 15]]
    assert rows[1] == [[11, 12]]


def test_pad_token_is_moved_off_real_tokens():

    class Tokenizer(SimpleNamespace):
        def convert_tokens_to_ids(self, token):
            return {"<|finetune_right_pad_id|>": 128004}.get(token)

    llama3 = Tokenizer(pad_token_id=None, unk_token_id=None, bos_token_id=128000, eos_token_id=128001)
    assert prepare.maxtext_pad_id(llama3) == 0  # MaxText's fallback: a real token
    prepare.ensure_pad_token(llama3)
    assert llama3.pad_token == "<|finetune_right_pad_id|>"

    mixtral = Tokenizer(pad_token_id=None, unk_token_id=0, bos_token_id=1, eos_token_id=2)
    prepare.ensure_pad_token(mixtral)
    assert not hasattr(mixtral, "pad_token")
    assert prepare.maxtext_pad_id(mixtral) == 0


# ---------------------------------------------------------------------------
# Requests: plan_request.py, make_config.py, jobs.py
# ---------------------------------------------------------------------------

plan_request = _load("plan_request")
make_config = _load("make_config")
jobs = _load("jobs")


@pytest.mark.parametrize(
    "said, key, precision",
    [
        ("llama 2 7b", "llama27b", None),
        ("Llama-2-7B FP8", "llama27b", "fp8"),
        ("llama2_7B-nanoo_fp8", "llama27b", "fp8"),
        ("Meta-Llama-3.1-8B-Instruct", "llama318b", None),
        ("mixtral_8x7B_v0.1", "mixtral8x7b", None),
        ("Qwen3-30B-A3B bf16", "qwen330ba3b", "bf16"),
        ("deepseek v2 lite", "deepseekv2lite", None),
    ],
)
def test_model_names_in_any_spelling(said, key, precision):
    assert plan_request.model_key(said) == key
    assert plan_request.split_precision(said)[1] == precision


def test_the_catalog_knows_recipes_and_examples():
    entries = plan_request.catalog()
    recipes = {(e.backend, e.name, e.precision) for e in entries if e.kind == "bundled"}
    assert ("megatron", "megatron/llama2_7B-FP8", "fp8") in recipes
    assert ("maxtext", "maxtext/llama2_7B-nanoo_fp8", "fp8") in recipes
    assert ("maxtext", "maxtext/mixtral_8x7B", "bf16") in recipes
    examples = [e for e in entries if e.kind == "example" and e.key == "qwen38b"]
    assert "megatron" in {e.backend for e in examples}
    assert not [e for e in entries if "mlperf" in e.name]  # benchmark variants are not models


@pytest.mark.parametrize(
    "image, backend",
    [
        ("rocm/primus:v26.7", "megatron"),
        ("rocm/jax-training:maxtext-v26.7", "maxtext"),
        ("unifiedtrainingdockers.azurecr.io/utd/ci:jax_ci_92ca3a6_20260925_maxdiffusion", "maxtext"),
        ("unifiedtrainingdockers.azurecr.io/utd/ci:primus_ci_4fd8ec2_20260927", "megatron"),
    ],
)
def test_backend_from_the_image_name(image, backend):
    assert plan_request.backend_from_image(image)[0] == backend


@pytest.mark.parametrize(
    "name, size, lr",
    [
        ("llama3.2_1B", 1, 3e-4),
        ("mixtral_8x7B_v0.1", 7, 1.5e-4),
        ("qwen3_30B_A3B", 30, 1e-4),
        ("grok1", None, 1.5e-4),
    ],
)
def test_learning_rate_follows_model_size(name, size, lr):
    assert make_config.model_size_b(name) == size
    assert make_config.peak_lr(size) == pytest.approx(lr)


def test_yaml_numbers_survive_yaml_1_1():
    yaml = pytest.importorskip("yaml")
    for value in (1e-08, 1.5e-05, 0.0001, 3, True, None):
        assert yaml.safe_load(f"x: {make_config._yaml_value(value)}")["x"] == value


def test_generated_megatron_moe_config_learns_its_router(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    monkeypatch.setenv("NNODES", "1")
    example = REPO_ROOT / "examples/megatron/configs/MI325X/qwen3_30B_A3B-FP8-pretrain.yaml"
    info = make_config.generate(example, tmp_path / "moe.yaml", iters=200)

    cfg = resolve_config.load(info["path"])
    get = cfg.effective.get
    assert (cfg.global_batch_size, cfg.train_iters, cfg.seq_length) == (128, 200, 4096)
    assert get("moe_router_force_load_balancing") is False
    assert get("use_turbo_deepep") is False and get("turbo_sync_free_moe_stage") == 0
    assert get("moe_aux_loss_coeff") == pytest.approx(0.01)
    assert get("micro_batch_size") == 1  # lowered for real routing
    shipped = resolve_config.load(example).effective
    for key in ("recompute_granularity", "fp8", "expert_model_parallel_size"):
        assert get(key) == shipped.get(key)  # what ships
    assert get("mock_data") is False and get("lr_decay_iters") == 200 and get("eval_interval") == 20


def test_maxtext_keys_the_model_file_sets_go_through_override_model(tmp_path):
    model_file = tmp_path / "tiny.yml"
    model_file.write_text("tokenizer_type: tiktoken\nbase_emb_dim: 64\n")

    class Cfg:
        num_devices = 8
        model_file = None
        values = {"per_device_batch_size": 4, "num_experts": 1}

        def get(self, key, default=None):
            return self.values.get(key, default)

    cfg = Cfg()
    cfg.model_file = model_file
    groups, override_model, _ = make_config.maxtext_overrides(cfg, 100, "./data/x", 7, "run")
    flat = {k: v for _, values in groups for k, v in values.items()}
    assert override_model == {"tokenizer_type": "huggingface"}
    assert "tokenizer_type" not in flat
    assert (flat["per_device_batch_size"], flat["gradient_accumulation_steps"]) == (4, 4)


def _plan(tmp_path, probe_first, budget):
    import json

    probe = ["--model", "m", "--probe", "20"] + (["--budget-hours", "2"] if budget else [])
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "status": "ready",
                "probe_first": probe_first,
                "driver_args": ["--model", "m", "--", "--lr", "2e-4"],
                "probe_driver_args": probe + ["--", "--lr", "2e-4"],
            }
        )
    )
    return plan


def test_a_plan_runs_its_probe_and_full_run_as_one_job(tmp_path):
    names = lambda stages: [s["name"] for s in stages]  # noqa: E731

    assert names(jobs._plan_stages(_plan(tmp_path, False, False))) == ["run"]
    stages = jobs._plan_stages(_plan(tmp_path, True, True))
    assert names(stages) == ["probe", "full run"] and stages[1]["budget_from_probe"]
    assert names(jobs._plan_stages(_plan(tmp_path, True, True), probe=True)) == ["probe"]
    assert names(jobs._plan_stages(_plan(tmp_path, True, False), no_probe=True)) == ["run"]
    # an explicit length skips the probe, and goes before the training overrides
    (stage,) = jobs._plan_stages(_plan(tmp_path, True, True), train_iters=700)
    assert stage["args"] == ["--model", "m", "--train-iters", "700", "--", "--lr", "2e-4"]


def test_status_follows_the_current_stage(tmp_path):
    def iterations(total, losses):
        return "".join(
            f" iteration {i:8d}/{total:8d} | elapsed time per iteration (ms): 3000.0 | lm loss: {loss:.6E} |\n"
            for i, loss in enumerate(losses, 1)
        )

    log = tmp_path / "driver.log"
    log.write_text(
        "[jobs] stage 1/2: probe\n"
        + iterations(20, [10.5 - i * 0.1 for i in range(20)])
        + "  recommended iters : 2210\n"
        + "[jobs] stage 2/2: full run\n"
        + "[convergence] launching: ./primus-cli container ...\n"
    )
    job = {"id": "x", "pid": 1, "dir": str(tmp_path), "log": str(log), "started": "2026-10-01T00:00:00"}

    info = jobs.progress(job)
    assert info["stage"] == "full run (stage 2/2)"
    assert info["phase"] == "starting the container and compiling"  # not the probe's iterations
    assert info["recommended_iterations"] == 2210

    log.write_text(log.read_text() + iterations(2210, [10.4, 10.0, 9.5]))
    info = jobs.progress(job)
    assert (info["phase"], info["iteration"], info["total"]) == ("training", 3, 2210)


def test_status_reads_a_finished_megatron_run(tmp_path):
    log = tmp_path / "driver.log"
    lines = [
        "[convergence] results      : /out/x_1.{log,csv,png}\n",
        "[convergence] launching: ./primus-cli container ...\n",
        "[convergence] console log: /out/x_1.log\n",
    ]
    lines += [
        f" iteration {i:8d}/      30 | elapsed time per iteration (ms): 3200.0 | lm loss: {10.5 - i * 0.1:.6E} |"
        " number of skipped iterations:   0 | number of nan iterations:   0 |\n"
        for i in range(1, 31)
    ]
    lines += [
        "[plot-loss] x: /out/x_1.log (30 points, 1 validations)\n",
        "\nx\n-\n",
        "  iterations        : 1 -> 30 (of 30)\n",
        "  initial vs ln(V)  : 10.40 vs 10.37 -- as expected\n",
        "[plot-loss] wrote /out/x_1.csv\n",
        "[plot-loss] wrote /out/x_1.png\n",
    ]
    log.write_text("".join(lines))
    (tmp_path / "exit_code").write_text("0\n")
    job = {
        "id": "x",
        "pid": 1,
        "dir": str(tmp_path),
        "log": str(log),
        "started": "2026-10-01T00:00:00",
        "driver_args": ["--model", "m"],
        "command": "run",
    }

    assert jobs.state(job) == ("finished", 0)
    info = jobs.progress(job)
    assert (info["iteration"], info["total"]) == (30, 30)
    assert info["seconds_per_iteration"] == pytest.approx(3.2)
    assert info["results"] == ["/out/x_1.csv", "/out/x_1.png"]
    assert "initial vs ln(V)  : 10.40 vs 10.37 -- as expected" in [s.strip() for s in info["summary"]]
    assert "PASS" in jobs.report(job)
