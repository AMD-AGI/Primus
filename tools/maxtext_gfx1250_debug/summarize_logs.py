#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Steady-state metrics from MaxText run logs: mean of steps >= SKIP (default 2).

Usage: summarize_logs.py <log or dir> [...]   (env SKIP=2)
"""
import glob
import os
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(
    r"completed step: (\d+), seconds: ([\d.]+), TFLOP/s/device: ([\d.]+), "
    r"Tokens/s/device: ([\d.]+).*?loss: ([-\w.]+)"
)
SKIP = int(os.environ.get("SKIP", "2"))

paths = []
for arg in sys.argv[1:] or ["."]:
    paths += sorted(glob.glob(os.path.join(arg, "*.log"))) if os.path.isdir(arg) else [arg]

for path in paths:
    rows = STEP.findall(ANSI.sub("", open(path, errors="replace").read()))
    if not rows:
        print(f"{path}: no completed steps")
        continue
    steady = [r for r in rows if int(r[0]) >= SKIP] or rows
    mean = lambda i: statistics.mean(float(r[i]) for r in steady)
    print(
        f"{os.path.basename(path):34s} steps={len(rows):3d} loss {rows[0][4]}->{rows[-1][4]} "
        f"step_s={mean(1):.4f} tflops={mean(2):.2f} tok_s={mean(3):.1f}"
    )
