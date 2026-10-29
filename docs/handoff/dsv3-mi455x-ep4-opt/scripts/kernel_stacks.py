#!/usr/bin/env python3
"""Which CPU ops / Python frames launch kernels whose name matches a regex (first profiled step).

  kernel_stacks.py TRACE.pt.trace.json.gz REGEX [depth]
"""
import gzip
import json
import re
import sys
from collections import defaultdict

path, pat = sys.argv[1], re.compile(sys.argv[2])
depth = int(sys.argv[3]) if len(sys.argv) > 3 else 4
with gzip.open(path, "rt") as f:
    ev = json.load(f)["traceEvents"]
steps = sorted((e for e in ev if e.get("ph") == "X" and e.get("name", "").startswith("ProfilerStep#")
                and e.get("cat") != "gpu_user_annotation"), key=lambda e: e["ts"])
s = steps[0]
t0, t1 = s["ts"], s["ts"] + s["dur"]
rt = {}
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "cuda_runtime":
        c = e.get("args", {}).get("correlation")
        if c is not None:
            rt[c] = e
frames = defaultdict(list)
for e in ev:
    if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "python_function") and t0 - 1e6 <= e["ts"] < t1:
        frames[e.get("tid")].append(e)
agg = defaultdict(lambda: [0, 0.0])
for k in ev:
    if k.get("ph") != "X" or k.get("cat") != "kernel" or not (t0 <= k["ts"] < t1) or not pat.search(k["name"]):
        continue
    r = rt.get(k.get("args", {}).get("correlation"))
    if r is None:
        continue
    st = [e for e in frames[r.get("tid")] if e["ts"] <= r["ts"] <= e["ts"] + e["dur"]]
    st.sort(key=lambda e: e["ts"])
    names = [re.sub(r"^.*/(megatron|primus_turbo|primus|transformer_engine|torch)/", r"\1/", e["name"])
             for e in st if not e["name"].startswith("<built-in")]
    names = [n for n in names if not n.startswith("torch/") or "autograd" in n]
    key = " < ".join(reversed(names[-depth:]))
    agg[key][0] += 1
    agg[key][1] += k["dur"]
for key, (c, t) in sorted(agg.items(), key=lambda x: -x[1][1])[:25]:
    print(f"{t/1e3:8.2f} ms {c:5d}x  {key[:300]}")
