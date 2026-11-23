#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
"""Freeze bounds from three runs, then check independent runs without retuning."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

METRICS = ("train/loss", "train/acc")


def workload_identity(inputs):
    """Compare workload semantics; software revisions are diagnostic provenance."""
    manifest = inputs["manifest"]
    workload = {
        "dataset": {key: manifest["dataset"][key] for key in ("fixture_sha256", "rows")},
        "model": manifest["model"],
        "training": manifest["training"],
        "hardware": manifest["hardware"],
    }
    return hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()


def read_summary(path):
    row = json.loads(path.read_text(encoding="utf-8"))
    if row["step"] != 100 or row["final_window_observations"] != 10 or row["consumed_samples"] != 800:
        raise ValueError("Incomplete reference run")
    for metric in METRICS:
        value = row["final_window_mean"][metric]
        if not math.isfinite(value) or value < 0 or (metric == "train/acc" and value > 1):
            raise ValueError(f"Invalid metric: {metric}")
    return row


def calibrate(paths):
    if len(paths) != 3 or len({path.resolve() for path in paths}) != 3:
        raise ValueError("Require three distinct calibration summaries")
    rows = [read_summary(path) for path in paths]
    if len({row["reference_identity"] for row in rows}) != 1:
        raise ValueError("Calibration inputs differ")
    if len({row["workload_identity"] for row in rows}) != 1:
        raise ValueError("Calibration workloads differ")
    bounds = {}
    for key in METRICS:
        values = [row["final_window_mean"][key] for row in rows]
        center = median(values)
        limit = {"maximum": center * 1.05} if key == "train/loss" else {"minimum": max(0, center - 0.01)}
        bounds[key] = {"reference": center, **limit, "calibration_values": values}
    return {
        "schema_version": 2,
        "workload_identity": rows[0]["workload_identity"],
        "reference_identity": rows[0]["reference_identity"],
        "metric": "mean of logged training observations at steps 55,60,...,100",
        "rule": "one-sided; loss <= median * 1.05; accuracy >= max(0, median - 0.01); improvements pass",
        "policy_status": "provisional engineering limits; not calibrated across software versions",
        "calibration_summary_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
        "bounds": bounds,
    }


def compare(row, baseline):
    if baseline.get("schema_version") != 2:
        raise ValueError("Expected schema-version-2 baseline; review legacy policy before migration")
    if row["workload_identity"] != baseline["workload_identity"]:
        raise ValueError("Reference workload differs")
    failures = []
    review_notes = []
    for key, bound in baseline["bounds"].items():
        value = row["final_window_mean"][key]
        invalid = not math.isfinite(value) or value < 0 or (key == "train/acc" and value > 1)
        outside = value > bound["maximum"] if key == "train/loss" else value < bound["minimum"]
        if invalid or outside:
            failures.append(f"{key}: {value} violates {bound}")
        elif (key == "train/loss" and value < bound["reference"] * 0.95) or (
            key == "train/acc" and value > bound["reference"] + 0.01
        ):
            review_notes.append(f"Review large improvement in {key}: {value} versus {bound['reference']}")
    if failures:
        raise ValueError("Numerical regression: " + "; ".join(failures))
    return review_notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    make = sub.add_parser("calibrate")
    make.add_argument("--summaries", nargs=3, type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("validate")
    check.add_argument("--baseline", type=Path, required=True)
    check.add_argument("--summaries", nargs=2, type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "calibrate":
        result = calibrate(args.summaries)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    else:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        if len({path.resolve() for path in args.summaries}) != 2:
            raise ValueError("Require two distinct validation summaries")
        for path in args.summaries:
            if hashlib.sha256(path.read_bytes()).hexdigest() in baseline["calibration_summary_sha256"]:
                raise ValueError("Validation must not reuse a calibration summary")
            for note in compare(read_summary(path), baseline):
                print(f"{path.name}: {note}")
        print("Both independent validation runs passed")


if __name__ == "__main__":
    main()
