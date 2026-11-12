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
    bounds = {}
    for key in METRICS:
        values = [row["final_window_mean"][key] for row in rows]
        center = median(values)
        # Predeclared rule: twice the largest calibration deviation, with a
        # 1% relative loss floor or 0.5 percentage-point accuracy floor.
        floor = max(abs(center) * 0.01, 1e-6) if key == "train/loss" else 0.005
        tolerance = max(floor, 2 * max(abs(value - center) for value in values))
        bounds[key] = {"reference": center, "absolute_tolerance": tolerance, "calibration_values": values}
    return {
        "schema_version": 1,
        "reference_identity": rows[0]["reference_identity"],
        "metric": "mean of logged training observations at steps 55,60,...,100",
        "rule": "two-sided; max(2 * maximum absolute deviation from median, floor); loss floor=max(1% of median,1e-6); accuracy floor=0.005",
        "calibration_summary_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
        "bounds": bounds,
    }


def compare(row, baseline):
    if row["reference_identity"] != baseline["reference_identity"]:
        raise ValueError("Reference inputs differ")
    failures = []
    for key, bound in baseline["bounds"].items():
        value = row["final_window_mean"][key]
        if not math.isfinite(value) or abs(value - bound["reference"]) > bound["absolute_tolerance"]:
            failures.append(f"{key}: {value} outside {bound['reference']} +/- {bound['absolute_tolerance']}")
    if failures:
        raise ValueError("Numerical regression: " + "; ".join(failures))


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
            compare(read_summary(path), baseline)
        print("Both independent validation runs passed")


if __name__ == "__main__":
    main()
