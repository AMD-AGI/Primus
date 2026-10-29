#!/usr/bin/env python3
"""CPU-side blocking calls in the first profiled step: hipMemcpyWithStream / *Synchronize / D2H copies,
grouped by CPU op stack; plus how far the CPU launch runs ahead of the GPU over the step."""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    ev = json.load(f)["traceEvents"]
steps = sorted((e for e in ev if e.get("ph") == "X" and e.get("name", "").startswith("ProfilerStep#")
                and e.get("cat") != "gpu_user_annotation"), key=lambda e: e["ts"])
s = steps[0]
t0, t1 = s["ts"], s["ts"] + s["dur"]
cpu = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation", "python_function")
              and t0 - 1e6 <= e["ts"] < t1), key=lambda e: e["ts"])
rts = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and t0 <= e["ts"] < t1]
SYNC = ("hipMemcpyWithStream", "hipStreamSynchronize", "hipEventSynchronize", "hipDeviceSynchronize",
        "hipMemcpy", "hipMemcpyDtoH")
agg = defaultdict(lambda: [0, 0.0])
total = 0.0
for r in rts:
    if r["name"] not in SYNC:
        continue
    st = [e["name"] for e in cpu if e.get("tid") == r.get("tid") and e["ts"] <= r["ts"] <= e["ts"] + e["dur"]
          and e.get("cat") != "python_function"]
    key = r["name"] + " :: " + " > ".join(st[-5:])
    agg[key][0] += 1
    agg[key][1] += r["dur"]
    total += r["dur"]
print(f"step {s['dur']/1e3:.1f} ms; blocking runtime calls total {total/1e3:.1f} ms")
for k, (c, t) in sorted(agg.items(), key=lambda x: -x[1][1])[:30]:
    print(f"  {t/1e3:8.1f} ms {c:5d}x  {k[:260]}")
