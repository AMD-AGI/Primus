# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
"""CPU-only checks of the numerical regression gate and incomplete-run rejection."""

import json
import tempfile
import unittest
from pathlib import Path

from calibrate_specforge_smoke import calibrate, compare
from run_specforge_single_node_smoke import validate_run


class SmokeResultsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "control").mkdir()
        checkpoint = self.root / "output" / "epoch0-step200"
        checkpoint.mkdir(parents=True)
        for name in ("training_state.pt", "training_state_rank0.pt"):
            (checkpoint / name).write_bytes(b"checkpoint")
        for name in ("consumer.done", "inference.done"):
            (self.root / "control" / name).write_text("0", encoding="utf-8")
        self.log = self.root / "control" / "consumer.log"
        self.lines = [
            f"step {step}: {{'train/loss': 3.0, 'train/acc': 0.2, 'train/grad_norm': 0.1}}\n"
            for step in range(5, 201, 5)
        ]
        self.log.write_text("".join(self.lines), encoding="utf-8")

    def test_complete_run(self):
        result = validate_run(self.root)
        self.assertEqual(result["final_window_observations"], 10)
        self.assertEqual(result["final_window_mean"]["train/loss"], 3)

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

    def test_nonfinite_metric(self):
        self.log.write_text("".join(self.lines).replace("3.0", "1e999"), encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_run(self.root)

    def test_calibration_and_regression(self):
        paths = []
        for index in range(3):
            row = validate_run(self.root)
            row.update(reference_identity="same-inputs", run_id=str(index))
            row["final_window_mean"]["train/loss"] += index * 0.01
            path = self.root / f"summary{index}.json"
            path.write_text(json.dumps(row), encoding="utf-8")
            paths.append(path)
        baseline = calibrate(paths)
        compare(row, baseline)
        self.assertAlmostEqual(baseline["bounds"]["train/loss"]["reference"], 3.01)
        self.assertEqual(baseline["bounds"]["train/acc"]["absolute_tolerance"], 0.005)
        for metric, value in (("train/loss", 6), ("train/loss", 0.1), ("train/acc", 0.9)):
            with self.subTest(metric=metric, value=value):
                changed = json.loads(json.dumps(row))
                changed["final_window_mean"][metric] = value
                with self.assertRaises(ValueError):
                    compare(changed, baseline)
        row["reference_identity"] = "different-inputs"
        with self.assertRaises(ValueError):
            compare(row, baseline)
        paths[0].write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaises(ValueError):
            calibrate(paths)


if __name__ == "__main__":
    unittest.main()
