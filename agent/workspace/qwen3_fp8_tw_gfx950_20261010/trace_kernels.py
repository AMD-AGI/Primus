"""Aggregate GPU kernel time per name from a torch profiler trace (one training step).

Usage: python trace_kernels.py TRACE.json.gz [regex ...]
Prints the top kernels overall, then the kernels matching each regex.
"""

import collections
import gzip
import json
import re
import sys

path = sys.argv[1]
patterns = sys.argv[2:]
with gzip.open(path, "rt") as f:
    trace = json.load(f)

agg = collections.defaultdict(lambda: [0, 0.0])
total = 0.0
for e in trace["traceEvents"]:
    if e.get("ph") != "X" or e.get("cat") != "kernel":
        continue
    agg[e["name"]][0] += 1
    agg[e["name"]][1] += e["dur"]
    total += e["dur"]

rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
print(f"total kernel time {total / 1e3:.1f} ms, {len(rows)} distinct kernels")
for name, (n, us) in rows[:40]:
    print(f"{us / 1e3:9.1f} ms {n:6d} x {us / n:8.1f} us  {name[:150]}")
for p in patterns:
    rx = re.compile(p, re.I)
    sel = [(k, v) for k, v in rows if rx.search(k)]
    s = sum(v[1] for _, v in sel)
    print(f"\n== /{p}/ : {s / 1e3:.1f} ms in {len(sel)} kernels")
    for name, (n, us) in sel[:25]:
        print(f"{us / 1e3:9.1f} ms {n:6d} x {us / n:8.1f} us  {name[:170]}")
