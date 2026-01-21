###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import csv
import os
import re
import statistics
from pathlib import Path
from typing import Dict

ANSI_ESCAPE_PATTERN = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")

MEGATRON_ITERATION_PATTERN = re.compile(
    r"iteration\s+\d+/\s*\d+.*?"
    r"elapsed time per iteration\s*\(ms\):\s*([\d.]+)/([\d.]+).*?"
    r"hip mem usage/free/total/usage_ratio:\s*([\d.]+)GiB/([\d.]+)GiB/([\d.]+)GiB/([\d.]+)%.*?"
    r"throughput per GPU\s*\(TFLOP/s/GPU\):\s*([\d.]+)/([\d.]+).*?"
    r"tokens per GPU\s*\(tokens/s/GPU\):\s*([\d.]+)/([\d.]+)",
    re.DOTALL,
)

TORCHTITAN_ITERATION_PATTERN = re.compile(
    r"rank-0(?:/\d+)?"
    r"[\s\S]*?"
    r"step:\s*(\d+)\s+"
    r"loss:\s*([\d.]+)\s+"
    r"grad_norm:\s*([\d.]+)\s+"
    r"memory:\s*([\d.]+)GiB\(([\d.]+)%\)\s+"
    r"tps:\s*([\d,]+)\s+"
    r"tflops:\s*([\d.]+)\s+"
    r"mfu:\s*([\d.]+)%",
)


def remove_ansi_escape(text: str) -> str:
    return ANSI_ESCAPE_PATTERN.sub("", text)


def parse_metrics_for_torchtitan(file_path: str) -> Dict[str, float]:
    """Extract last iteration's performance metrics from the log file."""
    with open(file_path, "r", encoding="utf-8") as f:
        log_text = f.read()
    log_text = remove_ansi_escape(log_text)

    matches = TORCHTITAN_ITERATION_PATTERN.findall(log_text)
    if not matches:
        raise ValueError(f"No valid iteration metrics found in {file_path}")

    metrics = {"tps": [], "tflops": [], "memory": []}
    for m in matches:
        metrics["memory"].append(float(m[3]))
        metrics["tps"].append(float(m[5].replace(",", "")))
        metrics["tflops"].append(float(m[6]))

    return {
        "TFLOP/s/GPU": round(statistics.mean(metrics["tflops"]), 2),
        "Tokens/s/GPU": round(statistics.mean(metrics["tps"]), 1),
        "Mem Usage": round(statistics.mean(metrics["memory"]), 2),
        "Step Time (s)": "",
    }


def parse_last_metrics_from_log(file_path: str) -> Dict[str, float]:
    """Extract last iteration's performance metrics from the log file."""
    with open(file_path, "r", encoding="utf-8") as f:
        log_text = f.read()
    log_text = remove_ansi_escape(log_text)

    matches = MEGATRON_ITERATION_PATTERN.findall(log_text)
    if not matches:
        raise ValueError(f"No valid iteration metrics found in {file_path}")

    last = matches[-1]
    step_time_s = float(last[1]) / 1000
    mem_usage = float(last[2])
    tflops = float(last[7])
    tokens_per_gpu = float(last[9])

    return {
        "TFLOP/s/GPU": round(tflops, 2),
        "Step Time (s)": round(step_time_s, 3),
        "Tokens/s/GPU": round(tokens_per_gpu, 1),
        "Mem Usage": round(mem_usage, 2),
    }


def parse_log_file(file_path: Path) -> Dict[str, str]:

    dimension_info = {
        "Model": file_path.stem,
        "Framework": file_path.parts[-2],
        "GPU": file_path.parts[-3],
        "date": file_path.parts[-4],
    }

    if dimension_info["Framework"] == "megatron":
        metrics = parse_last_metrics_from_log(str(file_path))
    elif dimension_info["Framework"] == "torchtitan":
        metrics = parse_metrics_for_torchtitan(str(file_path))
    return {**dimension_info, **metrics}


def write_csv_report(data: list[Dict[str, str]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"✅ Report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse benchmark logs and generate CSV report")
    parser.add_argument(
        "--benchmark-log-dir",
        type=str,
        default="output/benchmarks",
        help=(
            "Directory containing benchmark log files. "
            "All files ending with .log in this directory will be parsed."
        ),
    )
    parser.add_argument(
        "--report-csv-path", type=str, default="output/benchmarks.csv", help="Output CSV file"
    )

    args = parser.parse_args()
    results = []

    log_dir = Path(args.benchmark_log_dir)
    for log_file in log_dir.rglob("*.log"):
        try:
            results.append(parse_log_file(log_file))
        except Exception as e:
            print(f"❌ Failed to parse {log_file}: {e}")

    if results:
        write_csv_report(results, args.report_csv_path)
    else:
        print("⚠️ No valid logs parsed.")


if __name__ == "__main__":
    main()
