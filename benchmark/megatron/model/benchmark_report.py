###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict

ANSI_ESCAPE_PATTERN = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
ARG_PATTERN = re.compile(r"\]:\s+([a-zA-Z0-9_]+)\s+\.{3,}\s+(.*)$")
# Field order matches the current Megatron training-log line, where the memory
# segment is appended last (after tokens) rather than sitting between elapsed and
# throughput as in the old format. Groups: elapsed inst/avg, tflops inst/avg,
# tokens inst/avg, memory.
ITERATION_PATTERN = re.compile(
    r"iteration\s+\d+/.*?elapsed time per iteration \(ms\): ([\d.]+)/([\d.]+).*?"
    # Compute (TFLOP/s/GPU) accepts both the current "compute per GPU (...): X (avg Y)"
    # label and the legacy "throughput per GPU (...): X/Y" label. The space before
    # "(avg" is optional (some log sinks drop it).
    r"(?:compute|throughput) per GPU \(TFLOP/s/GPU\): ([\d.]+)(?:/|\s*\(avg\s*)([\d.]+)\)?.*?"
    # Tokens accepts both the current "tokens/s/GPU inst/harmonic mean: X/Y" label
    # and the legacy "tokens per GPU (tokens/s/GPU): X/Y" label.
    r"(?:tokens per GPU \(tokens/s/GPU\)|tokens/s/GPU inst/harmonic mean): ([\d.]+)/([\d.]+).*?"
    # Memory accepts both the current "hip mem usage/free/total/usage_ratio: X GB/..."
    # segment and the legacy "mem usages: X" segment. Only the used value is captured.
    r"(?:hip mem usage/free/total/usage_ratio:\s*|mem usages:\s*)([\d.]+)(?:\s*G(?:i)?B)?",
    re.DOTALL,
)


def remove_ansi_escape(text: str) -> str:
    return ANSI_ESCAPE_PATTERN.sub("", text)


def parse_arguments_from_log(file_path: str) -> Dict[str, str]:
    """Parse model arguments from a log file."""
    arguments = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = ARG_PATTERN.search(line)
            if match:
                key, value = match.group(1).strip(), match.group(2).strip()
                arguments[key] = remove_ansi_escape(value)
    return arguments


def parse_last_metrics_from_log(file_path: str) -> Dict[str, float]:
    """Extract last iteration's performance metrics from the log file."""
    with open(file_path, "r", encoding="utf-8") as f:
        log_text = f.read()

    matches = ITERATION_PATTERN.findall(log_text)
    if not matches:
        raise ValueError(f"No valid iteration metrics found in {file_path}")

    last = matches[-1]
    step_time_s = (float(last[0]) + float(last[1])) / 2000
    tflops = max(float(last[2]), float(last[3]))
    tokens_per_gpu = (float(last[4]) + float(last[5])) / 2
    mem_usage = float(last[6])

    return {
        "TFLOP/s/GPU": round(tflops, 2),
        "Step Time (s)": round(step_time_s, 3),
        "Tokens/s/GPU": round(tokens_per_gpu, 1),
        "Mem Usage": round(mem_usage, 2),
    }


def get_flag(arguments: Dict[str, str], key: str) -> str:
    return arguments.get(key, "")


def parse_log_file(model_name: str, file_path: str) -> Dict[str, str]:
    arguments = parse_arguments_from_log(file_path)
    recompute_info = (
        f"{get_flag(arguments, 'recompute_granularity')}/"
        f"{get_flag(arguments, 'recompute_method')}/"
        f"{get_flag(arguments, 'recompute_num_layers')}"
    )

    parsed_info = {
        "Model": model_name,
        "DP": get_flag(arguments, "data_parallel_size"),
        "TP": get_flag(arguments, "tensor_model_parallel_size"),
        "PP": get_flag(arguments, "pipeline_model_parallel_size"),
        "EP": get_flag(arguments, "expert_model_parallel_size"),
        "ETP": get_flag(arguments, "expert_tensor_parallel_size"),
        "FSDP": get_flag(arguments, "use_torch_fsdp2"),
        "Recompute (granularity/method/num_layers)": recompute_info,
    }

    metrics = parse_last_metrics_from_log(file_path)
    return {**parsed_info, **metrics}


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
        "--model",
        type=str,
        default=None,
        help=(
            "Specify the model name (e.g., 'llama2_7b') to parse only the corresponding log file "
            "(i.e., <model>.log). If not set, all .log files under --benchmark-log-dir will be parsed."
        ),
    )
    parser.add_argument(
        "--benchmark-log-dir",
        type=str,
        default="output/benchmarks",
        help=(
            "Directory containing benchmark log files. "
            "If --model is not specified, all files ending with .log in this directory will be parsed."
        ),
    )
    parser.add_argument(
        "--report-csv-path", type=str, default="output/benchmarks.csv", help="Output CSV file"
    )
    args = parser.parse_args()

    results = []

    if args.model:
        log_file = os.path.join(args.benchmark_log_dir, f"{args.model}.log")
        if os.path.isfile(log_file):
            results.append(parse_log_file(args.model, log_file))
        else:
            print(f"⚠️ Log file not found: {log_file}")
    else:
        for filename in os.listdir(args.benchmark_log_dir):
            if filename.endswith(".log"):
                model_name = Path(filename).stem
                log_path = os.path.join(args.benchmark_log_dir, filename)
                try:
                    results.append(parse_log_file(model_name, log_path))
                except Exception as e:
                    print(f"❌ Failed to parse {filename}: {e}")

    if results:
        write_csv_report(results, args.report_csv_path)
    else:
        print("⚠️ No valid logs parsed.")


if __name__ == "__main__":
    main()
