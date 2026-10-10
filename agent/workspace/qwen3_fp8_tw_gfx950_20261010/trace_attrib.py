"""Attribute GPU kernels to the CPU op stack that launched them (torch profiler trace).

Usage: python trace_attrib.py TRACE.json.gz REGEX [REGEX ...]
For each kernel matching a regex, prints total time grouped by the enclosing op chain.
"""

import bisect
import collections
import gzip
import json
import re
import sys

path, patterns = sys.argv[1], [re.compile(p) for p in sys.argv[2:]]
with gzip.open(path, "rt") as f:
    events = json.load(f)["traceEvents"]

ops_by_tid = collections.defaultdict(list)
launch_by_corr = {}
kernels = []
for e in events:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat")
    if cat in ("cpu_op", "user_annotation", "python_function"):
        ops_by_tid[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"]))
    elif cat == "cuda_runtime":
        corr = e.get("args", {}).get("correlation")
        if corr is not None:
            launch_by_corr[corr] = (e["tid"], e["ts"])
    elif cat == "kernel":
        kernels.append(e)

for tid in ops_by_tid:
    ops_by_tid[tid].sort()
starts = {tid: [o[0] for o in ops] for tid, ops in ops_by_tid.items()}


def stack_at(tid, ts, depth=4):
    ops = ops_by_tid.get(tid, [])
    i = bisect.bisect_right(starts.get(tid, []), ts)
    chain = [name for (s, e, name) in ops[max(0, i - 400) : i] if s <= ts <= e]
    keep = [n for n in chain if not n.startswith(("cudaLaunch", "hipLaunch", "aten::empty"))]
    return " < ".join(reversed(keep[-depth:]))


for rx in patterns:
    groups = collections.defaultdict(lambda: [0, 0.0])
    for k in kernels:
        if not rx.search(k["name"]):
            continue
        corr = k.get("args", {}).get("correlation")
        tid, ts = launch_by_corr.get(corr, (None, None))
        key = stack_at(tid, ts) if tid is not None else "?"
        groups[key][0] += 1
        groups[key][1] += k["dur"]
    print(f"\n== /{rx.pattern}/")
    for key, (n, us) in sorted(groups.items(), key=lambda kv: -kv[1][1])[:8]:
        print(f"{us / 1e3:8.1f} ms {n:5d} x {us / n:7.1f} us  {key[:260]}")
