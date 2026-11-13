#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
"""Check an existing 100-step reference run; never launch training."""

import argparse
import ast
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

from calibrate_specforge_smoke import compare, workload_identity

STEPS = 100
SAMPLES = 800
METRICS = ("train/loss", "train/acc", "train/grad_norm")


def collect_timing(run_dir, step_seconds):
    """Report timing only; missing/ambiguous evidence is never estimated."""
    result = {"end_to_end_seconds": None, "steady_state_seconds_per_step": None, "notes": []}
    launcher = run_dir / "launcher.log"
    text = launcher.read_text(encoding="utf-8", errors="replace") if launcher.exists() else ""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    prefix = r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?\[INFO\] "
    starts = re.findall(prefix + r"primus-cli-direct\.sh\s*$", text, re.MULTILINE)
    ends = re.findall(
        prefix + r"\[direct\] (?:python3|python|torchrun) finished successfully \(code 0\)\s*$",
        text,
        re.MULTILINE,
    )
    if len(starts) == len(ends) == 1:
        try:
            duration = (datetime.fromisoformat(ends[0]) - datetime.fromisoformat(starts[0])).total_seconds()
        except ValueError:
            duration = 0
        if duration > 0:
            result["end_to_end_seconds"] = duration
        else:
            result["notes"].append("Invalid or non-positive launcher timestamp interval")
    else:
        result["notes"].append("Missing or ambiguous timestamped direct-entry start/completion markers")
    expected = list(range(55, 101, 5))
    values = [step_seconds.get(step) for step in expected]
    if all(value is not None and math.isfinite(value) and value > 0 for value in values):
        result["steady_state_seconds_per_step"] = sum(values) / len(values)
    else:
        result["notes"].append("Missing or invalid perf/optimizer_step_time_s at logged steps 55–100")
    return result


def validate_run(run_dir):
    consumed_path = run_dir / "consumer-state/inboxes/inbox-rank0.jsonl.consumed_count"
    consumed = int(consumed_path.read_text(encoding="utf-8").strip())
    if consumed != SAMPLES:
        raise ValueError(f"Expected {SAMPLES} consumed conversation samples, got {consumed}")
    control = run_dir / "control"
    for name in ("consumer.done", "inference.done"):
        if (control / name).read_text(encoding="utf-8").strip() != "0":
            raise ValueError(f"Unsuccessful role status: {name}")
    text = (control / "consumer.log").read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    observations = {}
    step_seconds = {}
    for match in re.finditer(r"\bstep\s+(\d+):\s*(\{[^\n]*\})", text):
        step = int(match[1])
        if step in observations:
            raise ValueError(f"Duplicate metric record at step {step}")
        metrics = ast.literal_eval(match[2])
        for key in METRICS:
            if key not in metrics or not math.isfinite(float(metrics[key])):
                raise ValueError(f"Missing/non-finite {key} at step {step}")
        if metrics["train/loss"] < 0 or not 0 <= metrics["train/acc"] <= 1:
            raise ValueError(f"Invalid loss/accuracy at step {step}")
        observations[step] = {key: float(metrics[key]) for key in METRICS}
        if "perf/optimizer_step_time_s" in metrics:
            try:
                step_seconds[step] = float(metrics["perf/optimizer_step_time_s"])
            except (TypeError, ValueError):
                step_seconds[step] = None
    expected = set(range(5, STEPS + 1, 5))
    if set(observations) != expected:
        raise ValueError(
            f"Expected metrics at every fifth step through 100; missing={sorted(expected - observations.keys())}, unexpected={sorted(observations.keys() - expected)}"
        )
    checkpoints = []
    for shared in (run_dir / "output").glob("**/*-step100/training_state.pt"):
        rank = shared.with_name("training_state_rank0.pt")
        if shared.stat().st_size > 0 and rank.is_file() and rank.stat().st_size > 0:
            checkpoints.append(str(shared.parent.relative_to(run_dir)))
    if not checkpoints:
        raise ValueError("Missing nonempty step-100 shared/rank-0 checkpoint files")
    window = [metrics for step, metrics in observations.items() if 50 < step <= STEPS]
    return {
        "step": STEPS,
        "consumed_samples": consumed,
        "final_metrics": observations[STEPS],
        "final_window_observations": len(window),
        "final_window_mean": {key: sum(row[key] for row in window) / len(window) for key in METRICS},
        "checkpoints": checkpoints,
        "timing": collect_timing(run_dir, step_seconds),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--launcher-exit-code", type=int, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="Recorded reference conditions JSON")
    parser.add_argument("--baseline", type=Path, help="Optional previously calibrated numerical bounds")
    args = parser.parse_args()
    if args.launcher_exit_code != 0:
        parser.error(f"Launcher failed with exit code {args.launcher_exit_code}")
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    for key in ("runtime_primus_revision", "runtime_specforge_revision", "runtime_image"):
        if not inputs.get(key):
            parser.error(f"Missing software provenance: {key}")
    summary = validate_run(args.run_dir)
    summary["run_id"] = args.run_dir.name
    summary["reference_identity"] = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    summary["workload_identity"] = workload_identity(inputs)
    summary["reference_inputs"] = inputs
    summary["numerical_check"] = "not_requested"
    summary["review_notes"] = []
    if args.baseline:
        summary["review_notes"] = compare(summary, json.loads(args.baseline.read_text(encoding="utf-8")))
        summary["numerical_check"] = "passed"
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
