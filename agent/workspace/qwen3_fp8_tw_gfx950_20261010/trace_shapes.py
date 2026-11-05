"""Group GPU kernel time by the input shapes of the enclosing CPU op (torch profiler trace).

Usage: python trace_shapes.py TRACE.json.gz OP_REGEX [KERNEL_REGEX]
For every kernel launched inside an op matching OP_REGEX (and matching KERNEL_REGEX),
prints total time per (op input dims, kernel name).
"""

import bisect
import collections
import gzip
import json
import re
import sys

path, op_rx = sys.argv[1], re.compile(sys.argv[2])
k_rx = re.compile(sys.argv[3]) if len(sys.argv) > 3 else re.compile(".")
with gzip.open(path, "rt") as f:
    events = json.load(f)["traceEvents"]

ops_by_tid = collections.defaultdict(list)
launch_by_corr = {}
kernels = []
for e in events:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat")
    if cat == "cpu_op" and op_rx.search(e["name"]):
        dims = e.get("args", {}).get("Input Dims")
        ops_by_tid[e["tid"]].append((e["ts"], e["ts"] + e["dur"], str(dims)))
    elif cat == "cuda_runtime":
        corr = e.get("args", {}).get("correlation")
        if corr is not None:
            launch_by_corr[corr] = (e["tid"], e["ts"])
    elif cat == "kernel" and k_rx.search(e["name"]):
        kernels.append(e)

for tid in ops_by_tid:
    ops_by_tid[tid].sort()
starts = {tid: [o[0] for o in ops] for tid, ops in ops_by_tid.items()}

agg = collections.defaultdict(lambda: [0, 0.0])
for k in kernels:
    launch = launch_by_corr.get(k.get("args", {}).get("correlation"))
    if launch is None:
        continue
    tid, ts = launch
    ops = ops_by_tid.get(tid)
    if not ops:
        continue
    i = bisect.bisect_right(starts[tid], ts) - 1
    while i >= 0 and ops[i][1] < ts:
        i -= 1
    if i < 0:
        continue
    key = (ops[i][2][:90], re.sub(r"\(.*", "", k["name"])[:70])
    agg[key][0] += 1
    agg[key][1] += k["dur"]

for (dims, name), (n, us) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:30]:
    print(f"{us / 1e3:8.1f} ms {n:5d} x {us / n:7.1f} us  {dims:90s} {name}")
