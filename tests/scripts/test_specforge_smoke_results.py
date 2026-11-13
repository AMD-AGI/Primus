# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
"""CPU-only checks of the numerical regression gate and incomplete-run rejection."""

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from calibrate_specforge_smoke import calibrate, compare, workload_identity
from check_specforge_smoke import collect_timing, main, validate_run


class SmokeResultsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "control").mkdir()
        inboxes = self.root / "consumer-state/inboxes"
        inboxes.mkdir(parents=True)
        (inboxes / "inbox-rank0.jsonl.consumed_count").write_text("800", encoding="utf-8")
        checkpoint = self.root / "output" / "epoch0-step100"
        checkpoint.mkdir(parents=True)
        for name in ("training_state.pt", "training_state_rank0.pt"):
            (checkpoint / name).write_bytes(b"checkpoint")
        for name in ("consumer.done", "inference.done"):
            (self.root / "control" / name).write_text("0", encoding="utf-8")
        self.log = self.root / "control" / "consumer.log"
        self.lines = [
            f"step {step}: {{'train/loss': 3.0, 'train/acc': 0.2, 'train/grad_norm': 0.1}}\n"
            for step in range(5, 101, 5)
        ]
        self.log.write_text("".join(self.lines), encoding="utf-8")
        self.inputs = {
            "manifest": json.loads(
                (Path(__file__).parent / "specforge_smoke_fixture.json").read_text(encoding="utf-8")
            ),
            "runtime_primus_revision": "original",
            "runtime_specforge_revision": "original-specforge",
            "runtime_image": "original-image",
        }

    def test_complete_run(self):
        result = validate_run(self.root)
        self.assertEqual(result["final_window_observations"], 10)
        self.assertEqual(result["final_window_mean"]["train/loss"], 3)
        self.assertEqual(result["consumed_samples"], 800)

    def test_timing_excludes_queue_and_early_steps(self):
        (self.root / "launcher.log").write_text(
            "[2026-11-12 20:00:00] [INFO] queue submission\n"
            "[2026-11-12 23:59:30] [NODE-0] [INFO] primus-cli-direct.sh\n"
            "[2026-11-13 00:01:10] [NODE-0] [INFO] [direct] python3 finished successfully (code 0)\n",
            encoding="utf-8",
        )
        lines = [
            line.replace(
                "'train/loss':", f"'perf/optimizer_step_time_s': {10 if step <= 50 else 0.5}, 'train/loss':"
            )
            for step, line in zip(range(5, 101, 5), self.lines)
        ]
        self.log.write_text("".join(lines), encoding="utf-8")
        timing = validate_run(self.root)["timing"]
        self.assertEqual(timing["end_to_end_seconds"], 100)
        self.assertEqual(timing["steady_state_seconds_per_step"], 0.5)
        self.assertEqual(timing["notes"], [])

    def test_missing_or_invalid_timing_is_not_invented(self):
        timing = validate_run(self.root)["timing"]
        self.assertIsNone(timing["end_to_end_seconds"])
        self.assertIsNone(timing["steady_state_seconds_per_step"])
        self.assertEqual(len(timing["notes"]), 2)
        for value in (float("nan"), -1, 0):
            with self.subTest(value=value):
                values = dict.fromkeys(range(55, 101, 5), 0.5)
                values[100] = value
                self.assertIsNone(collect_timing(self.root, values)["steady_state_seconds_per_step"])

    def test_ambiguous_launcher_timing(self):
        (self.root / "launcher.log").write_text(
            "[2026-11-12 12:00:00] [INFO] primus-cli-direct.sh\n" * 2
            + "[2026-11-12 12:01:00] [INFO] [direct] python3 finished successfully (code 0)\n",
            encoding="utf-8",
        )
        self.assertIsNone(collect_timing(self.root, {})["end_to_end_seconds"])

    def test_checker_cli_requires_successful_launcher(self):
        inputs = self.root / "reference-inputs.json"
        inputs.write_text(json.dumps(self.inputs), encoding="utf-8")
        argv = ["check", "--run-dir", str(self.root), "--inputs", str(inputs), "--launcher-exit-code"]
        with patch("sys.argv", argv + ["1"]), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main()
        self.assertFalse((self.root / "summary.json").exists())
        with patch("sys.argv", argv + ["0"]), redirect_stdout(io.StringIO()):
            main()
        summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["step"], 100)
        self.assertEqual(summary["numerical_check"], "not_requested")
        self.assertEqual(summary["reference_inputs"], self.inputs)
        self.assertEqual(summary["workload_identity"], workload_identity(self.inputs))

    def test_checked_in_fixture(self):
        scripts = Path(__file__).resolve().parent
        repo = scripts.parents[1]
        manifest = json.loads((scripts / "specforge_smoke_fixture.json").read_text(encoding="utf-8"))
        spec = manifest["dataset"]
        payload = (repo / spec["fixture_path"]).read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), spec["fixture_sha256"])
        rows = [json.loads(line) for line in payload.decode("utf-8").split("\n") if line]
        ids = [row["id"] for row in rows]
        self.assertEqual(len(set(ids)), 800)
        self.assertEqual(
            manifest["training"]["steps"]
            * manifest["training"]["batch_size"]
            * manifest["training"]["accumulation_steps"],
            len(rows),
        )

    def test_checker_cli_accepts_new_software_with_baseline(self):
        inputs = self.root / "reference-inputs.json"
        inputs.write_text(json.dumps(self.inputs), encoding="utf-8")
        baseline = Path(__file__).parent / "specforge_smoke_baseline_mi355x.json"
        argv = [
            "check",
            "--run-dir",
            str(self.root),
            "--inputs",
            str(inputs),
            "--launcher-exit-code",
            "0",
            "--baseline",
            str(baseline),
        ]
        with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
            main()
        summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["numerical_check"], "passed")
        self.assertTrue(summary["review_notes"])
        self.assertNotEqual(
            summary["reference_identity"],
            json.loads(baseline.read_text(encoding="utf-8"))["reference_identity"],
        )

    def test_missing_and_duplicate_records(self):
        for lines in (self.lines[1:], self.lines + self.lines[-1:]):
            with self.subTest(lines=len(lines)):
                self.log.write_text("".join(lines), encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_run(self.root)

    def test_failed_role(self):
        (self.root / "control" / "consumer.done").write_text("1", encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_run(self.root)

    def test_incomplete_fixture_consumption(self):
        (self.root / "consumer-state/inboxes/inbox-rank0.jsonl.consumed_count").write_text(
            "792", encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            validate_run(self.root)

    def test_nonfinite_metric(self):
        self.log.write_text("".join(self.lines).replace("3.0", "1e999"), encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_run(self.root)

    def test_calibration_and_regression(self):
        paths = []
        for index in range(3):
            row = validate_run(self.root)
            row.update(
                reference_identity="same-inputs",
                workload_identity=workload_identity(self.inputs),
                run_id=str(index),
            )
            row["final_window_mean"]["train/loss"] += index * 0.01
            path = self.root / f"summary{index}.json"
            path.write_text(json.dumps(row), encoding="utf-8")
            paths.append(path)
        baseline = calibrate(paths)
        compare(row, baseline)
        self.assertAlmostEqual(baseline["bounds"]["train/loss"]["reference"], 3.01)
        self.assertAlmostEqual(baseline["bounds"]["train/acc"]["minimum"], 0.19)
        self.assertAlmostEqual(baseline["bounds"]["train/loss"]["maximum"], 3.01 * 1.05)
        for metric, value in (
            ("train/loss", 6),
            ("train/acc", 0.18),
            ("train/loss", -1),
            ("train/acc", 1.1),
            ("train/loss", float("nan")),
        ):
            with self.subTest(metric=metric, value=value):
                changed = json.loads(json.dumps(row))
                changed["final_window_mean"][metric] = value
                with self.assertRaises(ValueError):
                    compare(changed, baseline)
        for metric, value in (
            ("train/loss", baseline["bounds"]["train/loss"]["maximum"] + 1e-8),
            ("train/acc", baseline["bounds"]["train/acc"]["minimum"] - 1e-8),
        ):
            changed = copy.deepcopy(row)
            changed["final_window_mean"][metric] = value
            with self.assertRaises(ValueError):
                compare(changed, baseline)
        row["reference_identity"] = "different-inputs"
        compare(row, baseline)  # New software versions are allowed.
        for metric, value in (
            ("train/loss", 0.1),
            ("train/acc", 0.9),
            ("train/loss", baseline["bounds"]["train/loss"]["maximum"]),
            ("train/acc", baseline["bounds"]["train/acc"]["minimum"]),
        ):
            improved = copy.deepcopy(row)
            improved["final_window_mean"][metric] = value
            notes = compare(improved, baseline)
            if value in (0.1, 0.9):
                self.assertTrue(notes)
        row["workload_identity"] = "different-workload"
        with self.assertRaises(ValueError):
            compare(row, baseline)
        paths[0].write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaises(ValueError):
            calibrate(paths)

    def test_workload_identity_excludes_software_but_preserves_workload(self):
        original = workload_identity(self.inputs)
        changed = copy.deepcopy(self.inputs)
        changed["runtime_primus_revision"] = "new-main"
        changed["runtime_image"] = "new-image"
        changed["runtime_specforge_revision"] = "new-specforge"
        changed["manifest"]["runtime"] = {"specforge_revision": "new-specforge"}
        self.assertEqual(workload_identity(changed), original)
        for section, key, value in (
            ("dataset", "fixture_sha256", "changed"),
            ("dataset", "rows", 799),
            ("model", "revision", "new-model"),
            ("model", "tokenizer_revision", "new-tokenizer"),
            ("training", "seed", 123),
            ("training", "batch_size", 4),
        ):
            changed = copy.deepcopy(self.inputs)
            changed["manifest"][section][key] = value
            self.assertNotEqual(workload_identity(changed), original)
        changed = copy.deepcopy(self.inputs)
        changed["manifest"]["hardware"] = "different GPU"
        self.assertNotEqual(workload_identity(changed), original)

    def test_checked_in_policy_replays_historical_validation(self):
        scripts = Path(__file__).parent
        baseline = json.loads((scripts / "specforge_smoke_baseline_mi355x.json").read_text(encoding="utf-8"))
        history = json.loads(
            (scripts / "specforge_smoke_qualification_mi355x.json").read_text(encoding="utf-8")
        )
        observations = history["observations_shared_by_all_runs"]
        self.assertEqual([row["step"] for row in observations], list(range(5, 101, 5)))
        for run in history["runs"]:
            row = dict(run["summary"], workload_identity=workload_identity(history["reference_inputs"]))
            payload = (json.dumps(run["summary"], indent=2) + "\n").encode()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), run["summary_sha256"])
            for metric, mean in row["final_window_mean"].items():
                self.assertEqual(sum(item[metric] for item in observations[-10:]) / 10, mean)
            timing_observations = run["timing_observations"]
            self.assertEqual([item["logged_step"] for item in timing_observations], list(range(55, 101, 5)))
            self.assertEqual(
                sum(item["optimizer_step_time_s"] for item in timing_observations) / 10,
                run["timing"]["steady_state_seconds_per_step"],
            )
            self.assertGreater(run["timing"]["end_to_end_seconds"], 0)
            compare(row, baseline)


if __name__ == "__main__":
    unittest.main()
