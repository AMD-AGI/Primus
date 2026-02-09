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
    r"iteration\s+\d+\s*/\s*\d+.*?"
    r"elapsed time per iteration\s*\(ms\):\s*(?P<elapsed_ms>[\d.]+)(?:/(?P<elapsed_ms_avg>[\d.]+))?.*?"
    r"throughput per GPU\s*\(TFLOP/s/GPU\):\s*(?P<tflops>[\d.]+)/(?P<tflops_avg>[\d.]+).*?"
    r"tokens per GPU\s*\(tokens/s/GPU\):\s*(?P<tokens>[\d.]+)/(?P<tokens_avg>[\d.]+).*?"
    r"hip mem usage/free/total/usage_ratio:\s*"
    r"(?P<hip_used>[\d.]+)\s*G(?:i)?B/(?P<hip_free>[\d.]+)\s*G(?:i)?B/(?P<hip_total>[\d.]+)\s*G(?:i)?B/(?P<hip_ratio>[\d.]+)%",
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

# MaxText log pattern
# Example: "completed step: 49, seconds: 5.018, TFLOP/s/device: 107.127, Tokens/s/device: 6529.527, loss: 10.172"
MAXTEXT_ITERATION_PATTERN = re.compile(
    r"completed\s+step:\s*(\d+)"
    r".*?"
    r"seconds:\s*([\d.]+)"
    r".*?"
    r"TFLOP/s/device:\s*([\d.]+)"
    r".*?"
    r"Tokens/s/device:\s*([\d,]+(?:\.\d+)?)"
    r".*?"
    r"loss:\s*([\d.]+)",
    re.DOTALL | re.IGNORECASE,
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


def parse_metrics_for_maxtext(file_path: str) -> Dict[str, float]:
    """Extract performance metrics from MaxText log file.

    MaxText logs in format like:
    [INFO] completed step: 49, seconds: 5.018, TFLOP/s/device: 107.127,
           Tokens/s/device: 6529.527, total_weights: 262144, loss: 10.172

    Note: Same step may be logged multiple times, we deduplicate by keeping the last occurrence.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        log_text = f.read()
    log_text = remove_ansi_escape(log_text)

    matches = MAXTEXT_ITERATION_PATTERN.findall(log_text)
    if not matches:
        raise ValueError(f"No valid iteration metrics found in {file_path}")

    # Use dict to deduplicate by step number (keep last occurrence)
    step_metrics = {}
    for m in matches:
        # m[0] = step, m[1] = seconds, m[2] = TFLOP/s/device, m[3] = Tokens/s/device, m[4] = loss
        step_num = int(m[0])
        tokens_per_device = m[3].replace(",", "")
        step_metrics[step_num] = {
            "seconds": float(m[1]),
            "tflops": float(m[2]),
            "tps": float(tokens_per_device),
        }

    # Extract deduplicated metrics
    metrics = {"seconds": [], "tflops": [], "tps": []}
    for step_data in step_metrics.values():
        metrics["seconds"].append(step_data["seconds"])
        metrics["tflops"].append(step_data["tflops"])
        metrics["tps"].append(step_data["tps"])

    return {
        "TFLOP/s/GPU": round(statistics.mean(metrics["tflops"]), 2),
        "Tokens/s/GPU": round(statistics.mean(metrics["tps"]), 1),
        "Step Time (s)": round(statistics.mean(metrics["seconds"]), 3),
        "Mem Usage": "",  # MaxText doesn't log memory in this format
    }


def parse_last_metrics_from_log(file_path: str) -> Dict[str, float]:
    """Extract last iteration's performance metrics from the log file."""
    with open(file_path, "r", encoding="utf-8") as f:
        log_text = f.read()
    log_text = remove_ansi_escape(log_text)

    last_m = None
    for m in MEGATRON_ITERATION_PATTERN.finditer(log_text):
        last_m = m

    if not last_m:
        raise ValueError(f"No valid iteration metrics found in {file_path}")
    metric = last_m.groupdict()
    step_time_s = float(metric["elapsed_ms"]) / 1000.0
    mem_usage = float(metric["hip_used"])
    tflops_avg = float(metric["tflops_avg"])
    tokens_avg = float(metric["tokens_avg"])

    return {
        "TFLOP/s/GPU": round(tflops_avg, 2),
        "Step Time (s)": round(step_time_s, 3),
        "Tokens/s/GPU": round(tokens_avg, 1),
        "Mem Usage": round(mem_usage, 2),
    }


def parse_log_file(file_path: Path) -> Dict[str, str]:
    """Parse log file and extract metrics based on framework type."""
    dimension_info = {
        "Model": file_path.stem,
        "Framework": file_path.parts[-2],
        "GPU": file_path.parts[-3],
        "date": file_path.parts[-4],
    }

    framework = dimension_info["Framework"].lower()

    if framework == "megatron":
        metrics = parse_last_metrics_from_log(str(file_path))
    elif framework == "torchtitan":
        metrics = parse_metrics_for_torchtitan(str(file_path))
    elif framework == "maxtext":
        metrics = parse_metrics_for_maxtext(str(file_path))
    else:
        raise ValueError(f"Unknown framework: {framework}. Supported: megatron, torchtitan, maxtext")

    return {**dimension_info, **metrics}


def write_csv_report(data: list[Dict[str, str]], output_path: str) -> None:
    """Write benchmark data to CSV file with consistent field ordering."""
    if not data:
        print("⚠️ No data to write.")
        return

    # Define consistent field order
    # Dimension fields first, then metric fields, TFLOP/s/GPU at the end
    preferred_order = [
        "date",
        "GPU",
        "Framework",
        "Model",
        "Tokens/s/GPU",
        "Step Time (s)",
        "Mem Usage",
        "TFLOP/s/GPU",  # Moved to last position
    ]

    # Collect all unique fields from all records
    all_fields = set()
    for record in data:
        all_fields.update(record.keys())

    # Ensure all preferred fields are included (even if not present in any record)
    # This guarantees consistent CSV structure
    for field in preferred_order:
        all_fields.add(field)

    # Sort fields: preferred order first, then alphabetically for any extra fields
    fieldnames = []
    for field in preferred_order:
        if field in all_fields:
            fieldnames.append(field)
            all_fields.discard(field)  # Use discard instead of remove to avoid KeyError
    # Add any remaining fields (sorted alphabetically)
    fieldnames.extend(sorted(all_fields))

    # Ensure all records have all fields (fill missing with empty string)
    normalized_data = []
    for record in data:
        normalized_record = {field: record.get(field, "") for field in fieldnames}
        normalized_data.append(normalized_record)

    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_data)
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
