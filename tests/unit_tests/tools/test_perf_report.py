###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for tools/perf/report.py, the Excel report of a result directory.

Pure CPU. Needs pandas (and XlsxWriter for the workbook test); skipped
without them, as the report installs them on first use.
"""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "tools" / "perf" / "report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("perf_report", MODULE_PATH)
    assert spec and spec.loader, f"cannot load {MODULE_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.pd = pd
    return module


report = _load_module()


def run(**fields) -> dict:
    """A results.json row with the fields the report reads."""
    row = {
        "filename": "x.log",
        "backend": "megatron",
        "model_name": "llama3.1_8B",
        "precision": "BF16",
        "config_hash": "0b5862c3",
        "seq_len": 8192,
        "micro_batch_size": 4,
        "global_batch_size": 256,
        "nnodes": 1,
        "gpus_per_node": 8,
        "world_size": 8,
        "repeat": 1,
        "status": "ok",
        "hmean_tflops": 1100.0,
        "hmean_tps_per_gpu": 21000.0,
        "elapsed_sec": 600,
        "docker_image": "rocm/primus:v26.8",
        "image_id": "sha256:fe8e",
        "batch_stamp": "20261015-002731",
        "result_dir": "/tmp/results",
    }
    row.update(fields)
    return row


def test_throughput_is_the_metric_of_the_workload():
    assert report.throughput(run()) == (21000.0, "tokens/s")
    assert report.throughput(run(hmean_tps_per_gpu=None, hmean_samples_per_gpu=8.7)) == (8.7, "samples/s")
    # MaxDiffusion video models report both; frames are the headline.
    assert report.throughput(run(hmean_samples_per_gpu=0.01, hmean_frames_per_gpu=1.0))[1] == "frames/s"
    assert report.throughput(run(backend="dlrm", hmean_tps_per_gpu=None, rec_per_s=7.7e5)) == (7.7e5, "recommendations/s")
    assert report.throughput(run(hmean_tps_per_gpu=None)) == (None, None)


def test_repetitions_are_averaged_over_successful_runs_only():
    runs = [
        run(repeat=1, hmean_tflops=1000.0, hmean_tps_per_gpu=20000.0),
        run(repeat=2, hmean_tflops=1100.0, hmean_tps_per_gpu=22000.0),
        run(repeat=3, status="failed", failure_reason="oom", hmean_tflops=10.0, hmean_tps_per_gpu=10.0),
    ]
    (entry,) = report.aggregate(report.runs_frame(runs))
    assert (entry["runs_ok"], entry["runs_total"]) == (2, 3)
    assert entry["tflops"] == pytest.approx(1050.0)
    assert entry["throughput"] == pytest.approx(21000.0)
    assert entry["throughput_total"] == pytest.approx(21000.0 * 8)
    assert entry["cv"] == pytest.approx(70.71067 / 1050.0, rel=1e-4)
    assert entry["notes"] == "1 failed: oom"
    assert entry["minutes"] == pytest.approx(10.0)


def test_failed_multinode_attempts_do_not_count():
    runs = [
        run(),
        run(repeat=1, attempt=1, failed_attempt=1, status="failed", failure_reason="timeout"),
        run(repeat=1, attempt=2, failed_attempt=2, status="failed", failure_reason="timeout"),
    ]
    (entry,) = report.aggregate(report.runs_frame(runs))
    assert (entry["runs_ok"], entry["runs_total"]) == (1, 3)
    assert entry["notes"] == "2 failed attempts: timeout"
    # Only retries were lost, so the runs cell is amber rather than red.
    assert entry["only_retries"] is True
    # One successful run: no run-to-run spread to show.
    assert entry["cv"] is None


def test_runs_of_different_lengths_are_separate_configs():
    runs = [run(total_steps=50, elapsed_sec=720), run(total_steps=10, elapsed_sec=240)]
    entries = report.aggregate(report.runs_frame(runs))
    assert [(e["total_steps"], e["minutes"]) for e in entries] == [(10, 4.0), (50, 12.0)]


def test_configs_split_by_config_file_and_image_and_sort_by_backend():
    runs = [
        run(backend="dlrm", model_name="dlrm-v2", precision="TF32", hmean_tflops=None, rec_per_s=7e5, seq_len=None),
        run(precision="FP8", config_hash="874ffa7b"),
        run(),
        run(config_hash="deadbeef"),
        run(image_id="sha256:other"),
        # A failed setup step of a diffusion_dlrm run is not a config.
        run(backend="diffusion_dlrm", model_name="setup", precision=None, status="failed"),
    ]
    entries = report.aggregate(report.runs_frame(runs))
    # Within a model and precision: by image, then config hash.
    assert [(e["backend"], e["precision"], e["config_hash"], e["image"]) for e in entries] == [
        ("megatron", "BF16", "0b5862c3", "sha256:fe8e"),
        ("megatron", "BF16", "deadbeef", "sha256:fe8e"),
        ("megatron", "BF16", "0b5862c3", "sha256:other"),
        ("megatron", "FP8", "874ffa7b", "sha256:fe8e"),
        ("dlrm", "TF32", "0b5862c3", "sha256:fe8e"),
    ]


def test_config_without_successful_runs_says_so():
    (entry,) = report.aggregate(report.runs_frame([run(status="interrupted", failure_reason="interrupted")]))
    assert entry["tflops"] is None
    # The reason is not repeated when it is the status.
    assert entry["notes"] == "no successful run; 1 interrupted"


def test_backend_without_throughput_says_so():
    (entry,) = report.aggregate(report.runs_frame([run(backend="megatron_bridge", hmean_tps_per_gpu=None)]))
    assert entry["unit"] is None
    assert entry["notes"] == "no throughput in the log"


def test_pivot_puts_precisions_side_by_side():
    entries = report.aggregate(
        report.runs_frame(
            [
                run(precision="FP8", config_hash="b", hmean_tflops=1500.0),
                run(precision="BF16", hmean_tflops=1100.0),
                run(precision="BF16", config_hash="edited", hmean_tflops=1000.0),
                run(model_name="mamba_370M", precision=None, config_hash="c", hmean_tflops=300.0),
            ]
        )
    )
    means, counts = report.pivot(entries, "tflops")
    assert list(means.columns) == ["BF16", "FP8", "(none)"]
    assert means.loc[("llama3.1_8B", 8), "FP8"] == pytest.approx(1500.0)
    # Two Performance rows fall into one cell: their mean, and the count to flag it.
    assert means.loc[("llama3.1_8B", 8), "BF16"] == pytest.approx(1050.0)
    assert counts.loc[("llama3.1_8B", 8), "BF16"] == 2
    # A model with no precision in its name still appears.
    assert means.loc[("mamba_370M", 8), "(none)"] == pytest.approx(300.0)


def test_chart_labels_tell_images_and_configs_apart():
    entries = report.aggregate(
        report.runs_frame(
            [
                run(),
                run(docker_image="rocm/primus:v26.7", image_id="sha256:old"),
                run(config_hash="edited"),
                run(world_size=16, nnodes=2),
            ]
        )
    )
    labels = report.chart_labels(entries)
    assert "megatron: llama3.1_8B BF16 [rocm/primus:v26.7]" in labels
    assert "megatron: llama3.1_8B BF16 (16 GPUs) [rocm/primus:v26.8]" in labels
    # Same model, precision and image: the config hash tells them apart.
    assert sum(label.startswith("megatron: llama3.1_8B BF16 [rocm/primus:v26.8] #") for label in labels) == 2
    assert len(set(labels)) == len(labels)


def test_baseline_prefers_the_same_config():
    current = report.aggregate(report.runs_frame([run()]))[0]
    baseline = report.aggregate(
        report.runs_frame([run(config_hash="older", hmean_tflops=900.0), run(hmean_tflops=1000.0)])
    )
    base, how = report.match_baseline(current, baseline)
    assert (base["tflops"], how) == (1000.0, "same")
    assert report._change(current["tflops"], base["tflops"]) == pytest.approx(0.1)

    other = report.aggregate(report.runs_frame([run(config_hash="older", micro_batch_size=2, hmean_tflops=900.0)]))
    base, how = report.match_baseline(current, other)
    assert how == "changed (MBS 2, GBS 256)"
    assert report.match_baseline(report.aggregate(report.runs_frame([run(model_name="other")]))[0], other) == (None, None)


SINGLE_NODE_SUMMARY = """\
################################################################################
# System / Environment Snapshot
################################################################################
# Date           : 2026-10-15 00:27:32 CDT
# Hostname       : mi355-gpu-60
# Git commit    : b90866e28ca2544b823bcd0060044f3377884d6e
# Git dirty?    : yes (2 files modified)
################################################################################
# Batch Configuration
################################################################################
# Start time   : 2026-10-15 00:27:31 CDT
# Docker image : rocm/primus:v26.8
# Reps/config  : 3
# Total runs   : 8
# Configs to run (resolved path + content hash):
#   - 190f0585  /repo/examples/megatron/configs/MI355X/deepseek_v2_lite-BF16-pretrain.yaml
# diffusion_dlrm: tools/perf/diffusion_bench/config [222c8d07], after the configs above
#   Models     : flux-dev bs16; DLRM TF32 FP32
#   Run limits : none from the runner
################################################################################
[00:27:40] [run 1/8] megatron/pretrain/... rep 1/3
                    [ OK ] exit=0 elapsed=717s

################################################################################
# Primus Benchmark Batch Complete
################################################################################
# End time     : 2026-10-15 02:28:32 CDT
# Elapsed (sec): 7261
# Total runs   : 8
# Not started  : 0
################################################################################
"""


def test_batch_summary_is_read_by_section(tmp_path):
    path = tmp_path / "batch_summary_20261015-002731.txt"
    path.write_text(SINGLE_NODE_SUMMARY)
    info = report.parse_batch_summary(str(path))
    assert info["outcome"] == "Complete"
    assert info["system"]["Hostname"] == "mi355-gpu-60"
    assert info["config"]["Reps/config"] == "3"
    assert info["config"]["diffusion_dlrm Models"] == "flux-dev bs16; DLRM TF32 FP32"
    # The batch configuration's run limits, not the diffusion_dlrm block's.
    assert "Run limits" not in info["config"]
    assert info["config"]["Total runs"] == "8"
    assert info["final"]["Elapsed (sec)"] == "7261"
    # The config list is not a field.
    assert not any(key.startswith("-") for key in info["config"])


def test_interrupted_and_multinode_summaries(tmp_path):
    interrupted = tmp_path / "a.txt"
    interrupted.write_text(
        "# Batch Configuration\n# Start time   : x\n"
        "# Primus Benchmark Batch INTERRUPTED\n# Interrupted at: 2026-10-15 00:01:07 CDT\n"
    )
    info = report.parse_batch_summary(str(interrupted))
    assert info["outcome"] == "INTERRUPTED"
    assert info["final"]["Interrupted at"] == "2026-10-15 00:01:07 CDT"

    multinode = tmp_path / "b.txt"
    multinode.write_text(
        "# SLURM job      : 1234\n# Batch Configuration\n# NUM_REPS (target)     : 2\n"
        "# Primus Multi-Node Benchmark Batch STOPPED: GPUs busy\n# Elapsed (sec) : 100\n"
    )
    info = report.parse_batch_summary(str(multinode))
    assert info["outcome"] == "STOPPED: GPUs busy"
    assert info["system"]["SLURM job"] == "1234"
    assert info["config"]["NUM_REPS (target)"] == "2"
    rows = dict((label, value) for _, label, value in report.overview_rows("s", info, report.runs_frame([run()])))
    assert rows["Elapsed (h:mm:ss)"] == "0:01:40"
    assert rows["Repetitions per config"] == "2"


def test_refused_batch_shows_its_errors_and_lines_up_with_others(tmp_path):
    refused = tmp_path / "refused.txt"
    refused.write_text(
        "# Hostname       : node-1\n[ERROR] GPU 3 is busy (61000 MiB in use)\n"
        "# Primus Benchmark Batch REFUSED: a pre-run check failed (see above)\n"
    )
    complete = tmp_path / "complete.txt"
    complete.write_text(SINGLE_NODE_SUMMARY)
    refused_info, complete_info = report.parse_batch_summary(str(refused)), report.parse_batch_summary(str(complete))
    per_batch = [
        report.overview_rows("a", refused_info, report.runs_frame([run()]).iloc[0:0], keep_empty=True),
        report.overview_rows("b", complete_info, report.runs_frame([run()]), keep_empty=True),
    ]
    assert dict((l, v) for _, l, v in per_batch[0])["Errors"] == "GPU 3 is busy (61000 MiB in use)"
    # One block per section and rows in one order, whichever batch has which fields.
    labels = report.overview_labels(per_batch)
    sections = [section for section, _ in labels]
    assert sections == sorted(sections, key=report.SECTION_ORDER.index)
    runs_rows = [label for section, label in labels if section == "Runs"]
    assert runs_rows.index("Runs planned") < runs_rows.index("Result rows")
    # A row no batch has a value for is left out.
    assert ("Run settings", "FAIL_PATTERN") not in labels
    problems = report.batch_problems([("a", refused_info, None), ("b", complete_info, None)])
    assert problems == ["Batch a: REFUSED: a pre-run check failed (see above)"]


def test_stale_results_are_parsed_without_writing(tmp_path, monkeypatch):
    class FakeExtractor:
        PARSER_VERSION = 4
        CSV_FIELDS = ("filename", "backend", "status")

        @staticmethod
        def parse_log(path):
            return [{"filename": Path(path).name, "backend": "megatron", "status": "ok", "extra": 1}]

    monkeypatch.setattr(report, "_extractor", lambda: FakeExtractor)
    (tmp_path / "results.json").write_text(json.dumps({"parser_version": 3, "runs": [{"filename": "old"}]}))
    (tmp_path / "a.log").write_text("log\n")
    before = (tmp_path / "results.json").read_text()
    runs = report.load_runs(str(tmp_path))
    # An older parser's results are not trusted; the logs are parsed again ...
    assert [r["filename"] for r in runs] == ["a.log"]
    assert set(runs[0]) == {"filename", "backend", "status", "result_dir"}
    # ... and nothing is written into the directory.
    assert (tmp_path / "results.json").read_text() == before


def test_report_is_written(tmp_path, monkeypatch):
    xlsxwriter = pytest.importorskip("xlsxwriter")
    monkeypatch.setattr(report, "xlsxwriter", xlsxwriter, raising=False)
    monkeypatch.setattr(report, "load_dependencies", lambda allow_install: True)
    results = tmp_path / "results"
    results.mkdir()
    (results / "x.log").write_text("log\n")
    rows = [run(repeat=1), run(repeat=2, hmean_tflops=1200.0), run(repeat=3, status="failed", failure_reason="oom")]
    (results / "results.json").write_text(json.dumps({"parser_version": report._extractor().PARSER_VERSION, "runs": rows}))
    (results / "batch_summary_20261015-002731.txt").write_text(SINGLE_NODE_SUMMARY)

    # Named twice: counted once.
    assert report.main([str(results), str(results), "--baseline", str(results)]) == report.EXIT_OK
    out = results / "report.xlsx"
    with zipfile.ZipFile(out) as z:
        workbook = z.read("xl/workbook.xml").decode()
        for sheet in ("Overview", "Performance", "Pivot", "Runs", "Issues", "Chart"):
            assert f'name="{sheet}"' in workbook
    openpyxl = pytest.importorskip("openpyxl")
    ws = openpyxl.load_workbook(out)["Performance"]
    header = [c.value for c in ws[1]]
    first = {h: c.value for h, c in zip(header, ws[2])}
    assert first["Runs ok / total"] == "2 / 3"
    assert first["TFLOP/s per GPU"] == pytest.approx(1150.0)
    assert first["TFLOP/s change"] == pytest.approx(0.0)
    assert first["Baseline config"] == "same"
    assert not list(results.glob("*.tmp.xlsx"))

    other = tmp_path / "other"
    other.mkdir()
    assert report.main([str(results), str(other)]) == report.EXIT_ERROR  # several inputs need -o
    assert report.main([str(results), "-o", str(tmp_path / "missing" / "r.xlsx")]) == report.EXIT_ERROR
    (other / "results.json").write_text(json.dumps({"parser_version": report._extractor().PARSER_VERSION, "runs": []}))
    assert report.main([str(other)]) == report.EXIT_NO_RUNS
