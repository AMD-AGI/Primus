#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Enhanced-ILD training loss of a Primus run: selected iterations, 10-step means, and a CSV.

Reads Megatron's iteration lines (``iteration N/ M | ... | hylo ild: ... |``) from the log of
the rank that prints them::

    python tools/hybrid/summarize_hylo_ild_loss.py /path/to/run/logs/<rank>/debug.log \\
        --csv loss.csv --reference-loss 33.33

The reference trainer logs every 10 steps, so its final logged training loss is a mean over the
last 10 steps; compare that with the last 10-step window here.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from typing import Dict, Optional, Sequence, Tuple

_ITER = re.compile(r"iteration\s+(\d+)/\s*(\d+) \|(.*)")
_NUM = re.compile(r"\s*([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)")
KEYS = (
    "hylo ild",
    "hylo hidden ild",
    "hylo mixer ild",
    "learning rate",
    "grad norm",
    "elapsed time per iteration (ms)",
)


def parse(path: str) -> Tuple[Dict[int, Dict[str, float]], Optional[int]]:
    """``{iteration: {key: value}}`` for lines that carry the ILD loss, and the planned total."""
    rows, total = {}, None
    with open(path, errors="replace") as f:
        for line in f:
            m = _ITER.search(line)
            if not m:
                continue
            iteration, total = int(m.group(1)), int(m.group(2))
            values = {}
            for part in m.group(3).split("|"):
                if ":" in part:
                    key, value = part.split(":", 1)
                    number = _NUM.match(value)
                    if number and key.strip() in KEYS:
                        values[key.strip()] = float(number.group(1))
            if "hylo ild" in values:
                rows[iteration] = values
    return rows, total


def window_mean(rows, first: int, last: int, key: str = "hylo ild") -> Tuple[float, int]:
    values = [rows[i][key] for i in range(first, last + 1) if i in rows and key in rows[i]]
    return (statistics.fmean(values), len(values)) if values else (float("nan"), 0)


def main(argv: Optional[Sequence[str]] = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("log")
    p.add_argument("--csv", help="write every iteration's values here")
    p.add_argument("--reference-loss", type=float, help="a final 10-step training loss to compare with")
    a = p.parse_args(argv)

    rows, total = parse(a.log)
    if not rows:
        raise SystemExit(f"{a.log}: no iteration lines with 'hylo ild'")
    last = max(rows)
    steps = [
        r["elapsed time per iteration (ms)"] for r in rows.values() if "elapsed time per iteration (ms)" in r
    ]
    median_ms = f"{statistics.median(steps):.0f} ms" if steps else "n/a"
    print(f"iterations logged {len(rows)}, last {last}/{total}, median step {median_ms}")
    print(f"{'iteration':>9s} {'ild':>9s} {'hidden':>9s} {'mixer':>9s} {'lr':>10s} {'grad norm':>9s}")
    nan = float("nan")
    for i in sorted({1, 10, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, last} & set(rows)):
        r = rows[i]
        print(
            f"{i:9d} {r['hylo ild']:9.4f} {r.get('hylo hidden ild', nan):9.4f} {r.get('hylo mixer ild', nan):9.4f} "
            f"{r.get('learning rate', nan):10.3e} {r.get('grad norm', nan):9.2f}"
        )
    print("10-step means:")
    windows = {}
    for first, end in ((1, 10), (last - 9, last)):
        if first < 1:
            continue
        mean, n = window_mean(rows, first, end)
        hidden, _ = window_mean(rows, first, end, "hylo hidden ild")
        mixer, _ = window_mean(rows, first, end, "hylo mixer ild")
        windows[f"{first}-{end}"] = {"ild": mean, "hidden": hidden, "mixer": mixer, "steps": n}
        extra = ""
        if a.reference_loss is not None and end == last:
            extra = f"  vs reference {a.reference_loss:.4f}: {mean - a.reference_loss:+.4f}"
        print(
            f"  {first:5d}-{end:<5d} ild {mean:.4f} (hidden {hidden:.4f}, mixer {mixer:.4f}, {n} steps){extra}"
        )
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("iteration",) + KEYS)
            for i in sorted(rows):
                writer.writerow((i,) + tuple(rows[i].get(k, "") for k in KEYS))
        print("wrote", a.csv)
    return {"last": last, "total": total, "windows": windows}


if __name__ == "__main__":
    main()
