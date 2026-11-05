"""Summarize /tmp/q3_e2e_<name>.log runs over iterations 11-20.

tokens/s/GPU = GBS 512 x seq 4096 / 8 GPUs / mean iteration time; memory is the max-rank ROCm usage.
"""

import re
import statistics
import sys

TOKENS_PER_GPU_ITER = 512 * 4096 // 8
ITER = re.compile(r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\): ([\d.]+)")
TFLOPS = re.compile(r"per GPU \(TFLOP/s/GPU\): ([\d.]+)")
LOSS = re.compile(r"lm loss: ([\d.E+-]+)")
MEM = re.compile(r"rocm max mem usage/usage_ratio: ([\d.]+)GB")

print("| run | ms/iter | sd | TFLOP/s/GPU | tokens/s/GPU | max mem GB | loss@20 |")
print("|---|---|---|---|---|---|---|")
for name in sys.argv[1:]:
    rows = []
    for line in open(f"/tmp/q3_e2e_{name}.log", errors="ignore"):
        m = ITER.search(line)
        if m and int(m.group(1)) > 10:
            t, loss, mem = TFLOPS.search(line), LOSS.search(line), MEM.search(line)
            rows.append(
                (
                    float(m.group(2)),
                    float(t.group(1)) if t else 0.0,
                    loss.group(1) if loss else "-",
                    float(mem.group(1)) if mem else 0.0,
                )
            )
    if not rows:
        print(f"| {name} | no iterations |")
        continue
    ms = statistics.mean(r[0] for r in rows)
    print(
        f"| {name} | {ms:.1f} | {statistics.pstdev(r[0] for r in rows):.1f} | "
        f"{statistics.mean(r[1] for r in rows):.1f} | {TOKENS_PER_GPU_ITER / ms * 1e3:,.0f} | "
        f"{max(r[3] for r in rows):.1f} | {float(rows[-1][2]):.5f} |"
    )
