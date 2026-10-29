#!/usr/bin/env python3
"""For one trace: every Memcpy in the first profiled step with bytes, duration, the launching runtime
call and the CPU op stack (cpu_op events enclosing the runtime call), plus what other GPU work
overlaps it."""
import gzip
import json
import sys
from collections import Counter

path = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else "Memcpy"
with gzip.open(path, "rt") as f:
    ev = json.load(f)["traceEvents"]
steps = sorted((e for e in ev if e.get("ph") == "X" and e.get("name", "").startswith("ProfilerStep#")
                and e.get("cat") != "gpu_user_annotation"), key=lambda e: e["ts"])
s = steps[0]
t0, t1 = s["ts"], s["ts"] + s["dur"]
rt_by_corr = {}
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "cuda_runtime":
        c = e.get("args", {}).get("correlation")
        if c is not None:
            rt_by_corr[c] = e
cpu = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation")
              and t0 - 1e6 <= e["ts"] < t1), key=lambda e: e["ts"])
kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel" and t0 <= e["ts"] < t1]


def stack(t, tid):
    return [e["name"] for e in cpu if e.get("tid") == tid and e["ts"] <= t <= e["ts"] + e["dur"]]


mc = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") == "gpu_memcpy" and want in e["name"]
             and t0 <= e["ts"] < t1), key=lambda e: e["ts"])
summary = Counter()
for e in mc:
    a = e.get("args", {})
    r = rt_by_corr.get(a.get("correlation"))
    st = stack(r["ts"], r.get("tid")) if r else []
    ov = sum(min(k["ts"] + k["dur"], e["ts"] + e["dur"]) - max(k["ts"], e["ts"]) for k in kern
             if k["ts"] < e["ts"] + e["dur"] and k["ts"] + k["dur"] > e["ts"])
    key = " > ".join(st[-4:])
    summary[key] += e["dur"]
    print(f"{(e['ts']-t0)/1e3:9.2f}ms dur={e['dur']/1e3:7.2f}ms bytes={a.get('bytes')} "
          f"rt={r['name'] if r else '?'}({r['dur']/1e3 if r else 0:.2f}ms) kernel-overlap={ov/1e3:.2f}ms | {key[:200]}")
print("\nby stack (ms):")
for k, v in summary.most_common():
    print(f"  {v/1e3:8.1f}  {k[:220]}")
