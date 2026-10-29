#!/usr/bin/env python3
"""Per-shape GPU time of aten::add_ (or another op) in the first profiled step.

  add_shapes.py TRACE.pt.trace.json.gz [op]
"""
import gzip, json, sys
from collections import defaultdict

path = sys.argv[1]
op = sys.argv[2] if len(sys.argv) > 2 else "aten::add_"
with gzip.open(path, "rt") as f:
    ev = json.load(f)["traceEvents"]
steps = sorted((e for e in ev if e.get("ph") == "X" and e.get("name", "").startswith("ProfilerStep#")
                and e.get("cat") != "gpu_user_annotation"), key=lambda e: e["ts"])
t0, t1 = steps[0]["ts"], steps[0]["ts"] + steps[0]["dur"]
ops = defaultdict(list)
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "cpu_op" and e.get("name") == op:
        ops[e.get("tid")].append(e)
kern = {}
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "kernel" and t0 <= e["ts"] < t1:
        kern[e.get("args", {}).get("correlation")] = e
agg = defaultdict(lambda: [0, 0.0])
for e in ev:
    if e.get("ph") != "X" or e.get("cat") != "cuda_runtime":
        continue
    k = kern.get(e.get("args", {}).get("correlation"))
    if k is None:
        continue
    for o in ops[e.get("tid")]:
        if o["ts"] <= e["ts"] <= o["ts"] + o["dur"]:
            key = str(o.get("args", {}).get("Input Dims"))
            agg[key][0] += 1
            agg[key][1] += k["dur"]
            break
tot = sum(v[1] for v in agg.values())
print(f"total {tot/1e3:.2f} ms")
for key, (c, t) in sorted(agg.items(), key=lambda x: -x[1][1])[:20]:
    print(f"{t/1e3:8.2f} ms {c:5d}x  {t/c:8.1f} us  {key[:160]}")
