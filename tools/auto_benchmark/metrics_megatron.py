#!/usr/bin/env python3

import os
import re
from statistics import mean

LOG_DIR = "/workspace/Primus/tools/auto_benchmark/results/logs_megatron"

# ============================================================
# NOTE ON ITERATIONS AND AVERAGING
# ============================================================

NOTE_TEXT = """
NOTE:
- "Iterations" represents the number of training iterations USED to compute
  the averages shown above (after warm-up removal).

- All iteration records are extracted from the log file, sorted by iteration
  number, and the FIRST TWO iterations are discarded to remove warm-up effects.

- Iteration numbers do NOT need to be sequential. Valid examples:
    * iteration 1, iteration 2, iteration 3
    * iteration 1, iteration 5, iteration 10
    * iteration 1, iteration 10, iteration 20

- Numeric fields (elapsed time, TFLOPS/GPU, tokens/GPU) may appear as:
    1234
    1,234
    1,234.56
  Commas are removed before averaging.
"""

# ============================================================
# Regex patterns (comma-safe)
# ============================================================

NUM = r"[\d,]+(?:\.\d+)?"

ITERATION_REGEX = re.compile(
    rf"iteration\s+(\d+)/\s*\d+.*?"
    rf"elapsed time per iteration \(ms\):\s*({NUM}).*?"
    rf"throughput per GPU \(TFLOP/s/GPU\):\s*({NUM}).*?"
    rf"tokens per GPU \(tokens/s/GPU\):\s*({NUM}).*?"
    rf"global batch size:\s*(\d+)",
    re.IGNORECASE
)

# Filename metadata
FILENAME_REGEX = re.compile(
    r"(?P<model>.+?)_megatron_(?P<device>MI\d+X?)",
    re.IGNORECASE
)

PRECISION_REGEX = re.compile(r"(BF16|FP8)", re.IGNORECASE)

# ============================================================
# Helpers
# ============================================================

def is_megatron(filename: str) -> bool:
    name = filename.lower()
    return "megatron" in name or "megatorn" in name

def parse_filename(filename: str):
    m = FILENAME_REGEX.search(filename)
    if not m:
        return None

    p = PRECISION_REGEX.search(filename)
    precision = p.group(1).upper() if p else "-"

    return {
        "model": m.group("model"),
        "device": m.group("device"),
        "precision": precision,
    }

def to_float(num_str: str) -> float:
    """Convert numbers like '1,234.56' safely to float."""
    return float(num_str.replace(",", ""))

# ============================================================
# Log parsing (FULL FILE SCAN)
# ============================================================

def parse_log_file(path):
    records = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = ITERATION_REGEX.search(line)
            if m:
                records.append({
                    "iter": int(m.group(1)),
                    "elapsed_ms": to_float(m.group(2)),
                    "tflops_gpu": to_float(m.group(3)),
                    "tokens_gpu": to_float(m.group(4)),
                    "gbs": int(m.group(5)),
                })

    return records

def compute_averages(records):
    if len(records) <= 2:
        return None

    # Sort by iteration number (non-sequential safe)
    records = sorted(records, key=lambda x: x["iter"])

    # Drop first two warm-up iterations
    records = records[2:]

    return {
        "count": len(records),
        "elapsed_ms": mean(r["elapsed_ms"] for r in records),
        "tflops_gpu": mean(r["tflops_gpu"] for r in records),
        "tokens_gpu": mean(r["tokens_gpu"] for r in records),
        "gbs": records[0]["gbs"],
    }

# ============================================================
# Table printer
# ============================================================

def print_table(headers, rows):
    widths = [
        max(len(str(r[i])) for r in ([headers] + rows))
        for i in range(len(headers))
    ]

    def fmt(row):
        return "| " + " | ".join(
            str(row[i]).ljust(widths[i]) for i in range(len(row))
        ) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    print(sep)
    print(fmt(headers))
    print(sep)
    for r in rows:
        print(fmt(r))
    print(sep)

def print_note():
    print(NOTE_TEXT)

# ============================================================
# Main
# ============================================================

def main():
    rows = []

    for fname in sorted(os.listdir(LOG_DIR)):
        if not fname.endswith(".log"):
            continue

        if not is_megatron(fname):
            continue

        meta = parse_filename(fname)
        if not meta:
            continue

        path = os.path.join(LOG_DIR, fname)

        records = parse_log_file(path)
        stats = compute_averages(records)
        if not stats:
            continue

        rows.append([
            meta["model"],
            "megatron",
            meta["device"],
            meta["precision"],
            stats["count"],
            f"{stats['elapsed_ms']:.2f}",
            f"{stats['tflops_gpu']:.2f}",
            f"{stats['tokens_gpu']:.2f}",
            stats["gbs"],
        ])

    headers = [
        "Model", "Backend", "Device", "Precision",
        "Iterations",
        "Iter Time (ms)",
        "TFLOPS/GPU",
        "Tokens/GPU",
        "GBS",
    ]

    if rows:
        print_table(headers, rows)
        print_note()
    else:
        print("No Megatron logs found.")

if __name__ == "__main__":
    main()

