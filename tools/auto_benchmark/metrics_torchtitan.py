#!/usr/bin/env python3

import os
import re
from statistics import mean

LOG_DIR = "/workspace/Primus/tools/auto_benchmark/results/logs_torchtitan"

# ============================================================
# Regex patterns
# ============================================================

STEP_REGEX = re.compile(
    r"step:\s*(\d+).*?"
    r"memory:\s*([\d.]+)GiB.*?"
    r"tps:\s*([\d,]+(?:\.\d+)?).*?"
    r"tflops:\s*([\d,]+(?:\.\d+)?).*?"
    r"mfu:\s*([\d.]+)%"
)

# Dot-aligned config values
BS_REGEX = re.compile(
    r"training\.local_batch_size\s*\.{2,}\s*(\d+)"
)

SEQ_REGEX = re.compile(
    r"training\.seq_len\s*\.{2,}\s*(\d+)"
)

# Filename metadata
FILENAME_REGEX = re.compile(
    r"(?P<model>.+?)_torchtitan_(?P<device>MI\d+X?)",
    re.IGNORECASE
)

PRECISION_REGEX = re.compile(r"(BF16|FP8)", re.IGNORECASE)

# ============================================================
# Helpers
# ============================================================

def is_torchtitan(filename):
    return "torchtitan" in filename.lower()

def parse_filename(filename):
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

# ============================================================
# Log parsing (FULL FILE SCAN)
# ============================================================

def parse_log_file(path):
    """
    Fully scans the log file to extract:
      - BS from training.local_batch_size
      - SEQ from training.seq_len
      - Per-step performance metrics
    """
    bs = None
    seq = None
    steps = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            # Batch size
            if bs is None:
                m = BS_REGEX.search(line)
                if m:
                    bs = int(m.group(1))

            # Sequence length
            if seq is None:
                m = SEQ_REGEX.search(line)
                if m:
                    seq = int(m.group(1))

            # Step metrics
            m = STEP_REGEX.search(line)
            if m:
                steps.append({
                    "step": int(m.group(1)),
                    "memory": float(m.group(2)),
                    "tps": float(m.group(3).replace(",", "")),
                    "tflops": float(m.group(4).replace(",", "")),
                    "mfu": float(m.group(5)),
                })

    return bs if bs is not None else "-", seq if seq is not None else "-", steps

def compute_averages(steps):
    if len(steps) <= 2:
        return None

    steps = sorted(steps, key=lambda x: x["step"])
    steps = steps[2:]  # drop first two warm-up steps

    return {
        "count": len(steps),
        "memory": mean(s["memory"] for s in steps),
        "tps": mean(s["tps"] for s in steps),
        "tflops": mean(s["tflops"] for s in steps),
        "mfu": mean(s["mfu"] for s in steps),
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

# ============================================================
# Note printer
# ============================================================

def print_note():
    print("""
NOTE:
- "Steps" represents the number of training steps USED to compute averages.
- Steps are sorted by step number and the FIRST TWO steps are dropped
  to remove warm-up effects.
- Step numbers do NOT need to be sequential. Valid examples:
    * step 1, step 2, step 3
    * step 1, step 5, step 10
    * step 1, step 10, step 20
- TPS and TFLOPS values may contain commas (e.g. 1,202.17);
  commas are removed before averaging.
""")

# ============================================================
# Main
# ============================================================

def main():
    rows = []

    for fname in sorted(os.listdir(LOG_DIR)):
        if not fname.endswith(".log"):
            continue

        if not is_torchtitan(fname):
            continue

        meta = parse_filename(fname)
        if not meta:
            continue

        path = os.path.join(LOG_DIR, fname)

        bs, seq, steps = parse_log_file(path)
        stats = compute_averages(steps)
        if not stats:
            continue

        rows.append([
            meta["model"],
            "torchtitan",
            meta["device"],
            bs,
            seq,
            meta["precision"],
            stats["count"],
            f"{stats['memory']:.2f}",
            f"{stats['tps']:.2f}",
            f"{stats['tflops']:.2f}",
            f"{stats['mfu']:.2f}",
        ])

    headers = [
        "Model", "Backend", "Device",
        "BS", "Seq", "Precision",
        "Steps",
        "Mem(GiB)", "TPS", "TFLOPS", "MFU(%)"
    ]

    if rows:
        print_table(headers, rows)
        print_note()
    else:
        print("No TorchTitan logs found.")

if __name__ == "__main__":
    main()

