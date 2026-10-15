#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Parse a primus-bench results directory into summary.csv / summary.md / summary.json.

Usage: parse_results.py <run_dir>

Diffusion metrics come from AMDiffusionBenchmark's runs_summary.csv (falls back to the per-step
log lines when the summary is missing). DLRM metrics come from DLRMBenchmark's results.csv
(falls back to re-parsing the training log with the same logic as utils/process_training_log.py).
"""
import ast
import csv
import glob
import json
import math
import os
import re
import statistics
import sys

DIFFUSION_WARMUP_STEPS = 5
DLRM_NUM_SAMPLES = 20

STEP_RE = re.compile(r"Step (\d+) - (\{.*\})")
ITS_RE = re.compile(r"\b(\d+\.\d+)it/s\b")
SIT_RE = re.compile(r"\b(\d+\.\d+)s/it\b")

FIELDS = [
    "suite",
    "benchmark",
    "batch_size",
    "precision",
    "status",
    "wall_time_min",
    "step_time_gpu_s",
    "samples_per_s_per_gpu",
    "samples_per_s_total",
    "tflops_per_gpu",
    "rec_per_s",
    "rec_per_s_cv",
    "num_gpus",
    "metric_source",
    "log",
]


def to_float(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def read_kv(path):
    info = {}
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    info[k.strip()] = v.strip()
    return info


def read_status(run_dir):
    path = os.path.join(run_dir, "status.tsv")
    steps = {}
    if os.path.isfile(path):
        with open(path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                steps[row["step"]] = row
    return steps


def parse_diffusion_steps(log_path):
    rows = []
    with open(log_path, errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                try:
                    rows.append((int(m.group(1)), ast.literal_eval(m.group(2))))
                except (ValueError, SyntaxError):
                    # Ignore malformed or partially written step payloads in logs; keep best-effort parsing.
                    pass
    rows = [r for s, r in rows if s > DIFFUSION_WARMUP_STEPS] or [r for _, r in rows]

    def mean(key):
        vals = [to_float(r.get(key)) for r in rows]
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else None

    return {
        "avg_step_time_gpu": mean("step_time_gpu"),
        "avg_fps_gpu": mean("fps_gpu"),
        "avg_tflops": mean("tflops/s"),
        "num_steps": len(rows),
    }


def log_has_oom(path):
    try:
        with open(path, errors="replace") as f:
            return any(re.search(r"out of memory|OutOfMemoryError", l, re.I) for l in f)
    except OSError:
        return False


def parse_diffusion(run_dir, steps):
    results = []
    dirs = sorted(glob.glob(os.path.join(run_dir, "diffusion", "*_bs*")))
    for d in dirs:
        name = os.path.basename(d)
        model, bs = name.rsplit("_bs", 1)
        step = steps.get(f"diffusion_{name}", {})
        logs = glob.glob(os.path.join(d, "*_logs_*.txt"))
        row = {
            "suite": "diffusion",
            "benchmark": model,
            "batch_size": bs,
            "precision": "bf16",
            "status": step.get("status", "unknown"),
            "log": os.path.relpath(logs[0], run_dir) if logs else step.get("log"),
        }
        dur = to_float(step.get("duration_s"))
        row["wall_time_min"] = round(dur / 60, 1) if dur is not None else None

        summary = os.path.join(d, "runs_summary.csv")
        num_gpus = None
        metrics = {}
        if os.path.isfile(summary):
            with open(summary) as f:
                rows = list(csv.DictReader(f))
            if rows:
                s = rows[-1]
                num_gpus = to_float(s.get("num_processes"))
                metrics = {
                    "avg_step_time_gpu": to_float(s.get("avg_step_time_gpu")),
                    "avg_fps_gpu": to_float(s.get("avg_fps_gpu")),
                    "avg_tflops": to_float(s.get("avg_tflops")),
                }
                row["metric_source"] = "runs_summary.csv"
                if row["status"] in ("unknown", "success", "failed"):
                    row["status"] = s.get("status") or row["status"]
        if metrics.get("avg_fps_gpu") is None:
            if logs:
                parsed = parse_diffusion_steps(logs[0])
                if parsed["num_steps"]:
                    metrics = parsed
                    row["metric_source"] = f"log ({parsed['num_steps']} steps)"
                if row["status"] in ("unknown", "failed") and log_has_oom(logs[0]):
                    row["status"] = "oom"

        num_gpus = int(num_gpus) if num_gpus else 8
        fps = metrics.get("avg_fps_gpu")
        row.update(
            {
                "num_gpus": num_gpus,
                "step_time_gpu_s": metrics.get("avg_step_time_gpu"),
                "samples_per_s_per_gpu": fps,
                "samples_per_s_total": fps * num_gpus if fps is not None else None,
                "tflops_per_gpu": metrics.get("avg_tflops"),
            }
        )
        results.append(row)
    return results


def dlrm_batch_size(config_path):
    if os.path.isfile(config_path):
        with open(config_path) as f:
            for line in f:
                m = re.match(r"\s*export\s+BATCH_SIZE=(\d+)", line)
                if m:
                    return int(m.group(1))
    return None


def parse_dlrm_log(log_path, batch_size):
    its = []
    with open(log_path, errors="replace") as f:
        for line in f:
            if "Epoch" in line:
                m = ITS_RE.search(line)
                if m:
                    its.append(float(m.group(1)))
                    continue
                m = SIT_RE.search(line)
                if m:
                    its.append(1.0 / float(m.group(1)))
    rec = [v * batch_size for v in its]
    if len(rec) > 1:
        window = rec[-DLRM_NUM_SAMPLES:-1] if len(rec) > DLRM_NUM_SAMPLES else rec[:-1]
        mean = statistics.mean(window)
        return mean, statistics.pstdev(window) / (mean + 1e-3)
    if rec:
        return rec[0], 0.0
    return None, None


def parse_dlrm(run_dir, steps):
    results = []
    for d in sorted(glob.glob(os.path.join(run_dir, "dlrm", "*"))):
        prec = os.path.basename(d)
        step = steps.get(f"dlrm_{prec}", {})
        bs = dlrm_batch_size(os.path.join(d, "training_config.sh"))
        dur = to_float(step.get("duration_s"))
        row = {
            "suite": "dlrm",
            "benchmark": "dlrm-v2",
            "batch_size": bs,
            "precision": prec,
            "status": step.get("status", "unknown"),
            "num_gpus": 8,
            "wall_time_min": round(dur / 60, 1) if dur is not None else None,
            "log": os.path.relpath(os.path.join(d, "train.log"), run_dir),
        }
        rec, cv = None, None
        res_csv = os.path.join(d, "results.csv")
        if os.path.isfile(res_csv):
            with open(res_csv) as f:
                rows = list(csv.DictReader(f))
            if rows:
                rec = to_float(rows[-1].get("Recommendations/s (mean)"))
                cv = to_float(rows[-1].get("Recommendations/s (std/mean)"))
                row["metric_source"] = "results.csv"
        train_log = os.path.join(d, "train.log")
        if (not rec) and os.path.isfile(train_log) and bs:
            rec, cv = parse_dlrm_log(train_log, bs)
            row["metric_source"] = "train.log"
        if not rec and row["status"] == "success":
            row["status"] = "no-metrics"
        row["rec_per_s"] = rec
        row["rec_per_s_cv"] = cv
        results.append(row)
    return results


def fmt(v, nd=2):
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def write_markdown(path, info, steps, diffusion, dlrm):
    lines = ["# Primus benchmark summary", ""]
    for k in (
        "gpu_target",
        "gpu_product",
        "gpu_count",
        "hostname",
        "date",
        "config",
        "image",
        "dlrm_image",
        "hf_cache",
        "dry_run",
    ):
        if k in info:
            lines.append(f"- **{k}**: {info[k]}")
    lines.append("")

    if diffusion:
        lines += [
            "## Diffusion (AMDiffusionBenchmark)",
            "",
            "| Model | Batch size / GPU | Status | Step time GPU (s) | Samples/s/GPU | Samples/s (node) | TFLOPS/GPU | Wall time (min) |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
        for r in diffusion:
            lines.append(
                f"| {r['benchmark']} | {r['batch_size']} | {r['status']} | {fmt(r['step_time_gpu_s'], 3)} | "
                f"{fmt(r['samples_per_s_per_gpu'], 3)} | {fmt(r['samples_per_s_total'], 2)} | "
                f"{fmt(r['tflops_per_gpu'], 1)} | {fmt(r['wall_time_min'], 1)} |"
            )
        lines.append("")

    if dlrm:
        lines += [
            "## DLRM (DLRMBenchmark)",
            "",
            "| Precision | Batch size | Status | Recommendations/s (mean) | std/mean | Wall time (min) |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for r in dlrm:
            lines.append(
                f"| {r['precision']} | {r['batch_size'] or '-'} | {r['status']} | {fmt(r['rec_per_s'], 0)} | "
                f"{fmt(r['rec_per_s_cv'], 3)} | {fmt(r['wall_time_min'], 1)} |"
            )
        lines.append("")

    lines += [
        "## Steps",
        "",
        "| Step | Status | Exit code | Duration (min) | Log |",
        "|---|---|---:|---:|---|",
    ]
    for name, s in steps.items():
        dur = to_float(s.get("duration_s"))
        lines.append(
            f"| {name} | {s.get('status')} | {s.get('exit_code')} | "
            f"{fmt(dur / 60 if dur is not None else None, 1)} | {s.get('log')} |"
        )
    lines.append("")

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    return text


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(64)
    run_dir = os.path.abspath(sys.argv[1])
    if not os.path.isdir(run_dir):
        sys.exit(f"Not a directory: {run_dir}")

    info = read_kv(os.path.join(run_dir, "run_info.txt"))
    steps = read_status(run_dir)
    diffusion = parse_diffusion(run_dir, steps)
    dlrm = parse_dlrm(run_dir, steps)
    rows = diffusion + dlrm

    with open(os.path.join(run_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: round(r[k], 4) if isinstance(r.get(k), float) else r.get(k) for k in FIELDS})

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({"run_info": info, "steps": steps, "results": rows}, f, indent=2)

    text = write_markdown(os.path.join(run_dir, "summary.md"), info, steps, diffusion, dlrm)
    print()
    print(text)
    print(f"Summary written to {run_dir}/summary.{{md,csv,json}}")


if __name__ == "__main__":
    main()
