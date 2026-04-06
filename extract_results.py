#!/usr/bin/env python3
"""Extract benchmarking metrics from Primus training log files."""

import argparse
import csv
import glob
import os
import re
import sys


def extract_first_match(text, pattern):
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1).strip() if m else None


def parse_log(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    basename = os.path.basename(file_path)
    m = re.match(r"(.+?)-pretrain", basename)
    if not m:
        print(f"WARNING: Could not extract model_name from {basename}, skipping")
        return None
    model_name = m.group(1)

    config_file = extract_first_match(content, r"^Config:\s*(.+)$")

    warmup_iters_str = extract_first_match(content, r"log_avg_skip_iterations\s*:\s*(\d+)")
    warmup_iters = int(warmup_iters_str) if warmup_iters_str else 0

    train_iters = extract_first_match(content, r"train_iters\s*:\s*(\d+)")
    micro_batch_size = extract_first_match(content, r"micro_batch_size\s*:\s*(\d+)")
    global_batch_size = extract_first_match(content, r"global_batch_size\s*:\s*(\d+)")

    iter_pattern = re.compile(
        r"iteration\s+(\d+)/\s+\d+\s*\|"
        r".*?elapsed time per iteration \(ms\):\s*([\d.]+)"
        r".*?throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)"
        r"(?:.*?tokens per GPU \(tokens/s/GPU\):\s*([\d.]+))?"
        r".*?usage_ratio:.*?/([\d.]+)%"
    )

    iterations = []
    for m in iter_pattern.finditer(content):
        iter_num = int(m.group(1))
        iter_time = float(m.group(2))
        tflops = float(m.group(3))
        tps_per_gpu = float(m.group(4)) if m.group(4) else None
        memory_use_pct = float(m.group(5))
        iterations.append({
            "iter": iter_num,
            "iter_time": iter_time,
            "tflops": tflops,
            "tps_per_gpu": tps_per_gpu,
            "memory_use_pct": memory_use_pct,
        })

    measured = [it for it in iterations if it["iter"] > warmup_iters]
    if not measured:
        print(f"WARNING: No measured iterations (after removing {warmup_iters} warmup) in {basename}")
        return None

    n = len(measured)
    mean_iter_time = sum(it["iter_time"] for it in measured) / n
    mean_tflops = sum(it["tflops"] for it in measured) / n
    mean_memory = sum(it["memory_use_pct"] for it in measured) / n

    tps_values = [it["tps_per_gpu"] for it in measured if it["tps_per_gpu"] is not None]
    if tps_values:
        hmean_tps = len(tps_values) / sum(1.0 / v for v in tps_values)
    else:
        hmean_tps = None

    return {
        "file_path": file_path,
        "model_name": model_name,
        "config_file": config_file,
        "warmup_iters": warmup_iters,
        "train_iters": train_iters,
        "micro_batch_size": micro_batch_size,
        "global_batch_size": global_batch_size,
        "num_measured_iters": n,
        "mean_iter_time_ms": round(mean_iter_time, 2),
        "mean_tflops": round(mean_tflops, 2),
        "hmean_tps_per_gpu": round(hmean_tps, 2) if hmean_tps is not None else "",
        "mean_memory_use_pct": round(mean_memory, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract benchmark results from Primus training logs")
    parser.add_argument("input_dir", help="Directory containing .log files")
    parser.add_argument("-o", "--output", help="Output CSV path (default: <input_dir>/benchmark_summary.csv)")
    args = parser.parse_args()

    log_files = sorted(glob.glob(os.path.join(args.input_dir, "*.log")))
    if not log_files:
        print(f"No .log files found in {args.input_dir}")
        sys.exit(1)

    results = []
    for lf in log_files:
        row = parse_log(lf)
        if row:
            results.append(row)

    if not results:
        print("No results extracted.")
        sys.exit(1)

    fieldnames = [
        "file_path", "model_name", "config_file",
        "warmup_iters", "train_iters", "micro_batch_size", "global_batch_size",
        "num_measured_iters", "mean_iter_time_ms", "mean_tflops",
        "hmean_tps_per_gpu", "mean_memory_use_pct",
    ]

    out_path = args.output or os.path.join(args.input_dir, "benchmark_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
