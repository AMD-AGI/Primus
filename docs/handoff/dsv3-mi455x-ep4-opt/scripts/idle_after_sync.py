#!/usr/bin/env python3
"""Attribute every GPU idle gap (no kernel / memcpy on any stream) in the profiled steps to the most
recent blocking CPU call that ended before the gap (on any thread), labelled by its Python caller.

  idle_after_sync.py TRACE.pt.trace.json.gz
"""
import gzip
import json
import re
import sys
from collections import defaultdict

SYNC = {"hipMemcpyWithStream", "hipStreamSynchronize", "hipEventSynchronize", "hipDeviceSynchronize"}
path = sys.argv[1]
with gzip.open(path, "rt") as f:
    ev = json.load(f)["traceEvents"]
steps = sorted((e for e in ev if e.get("ph") == "X" and e.get("name", "").startswith("ProfilerStep#")
                and e.get("cat") != "gpu_user_annotation"), key=lambda e: e["ts"])
gpu = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
             key=lambda e: e["ts"])
syncs = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and e["name"] in SYNC
                and e["dur"] > 20), key=lambda e: e["ts"] + e["dur"])
py = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "python_function"]
py_by_tid = defaultdict(list)
for e in py:
    py_by_tid[e.get("tid")].append(e)


def label(r):
    st = [e["name"] for e in py_by_tid[r.get("tid")] if e["ts"] <= r["ts"] <= e["ts"] + e["dur"]]
    st = [s for s in st if not s.startswith("<built-in") and "torch/" not in s and "_contextlib" not in s]
    keep = [re.sub(r"^.*/(megatron|primus|primus_turbo|pretrain_gpt)", r"\1", s) for s in st[-3:]]
    return r["name"] + " @ " + " < ".join(reversed(keep))


labels = {id(r): label(r) for r in syncs}
agg = defaultdict(lambda: [0, 0.0, 0.0])
n = len(steps)
for s in steps:
    t0, t1 = s["ts"], s["ts"] + s["dur"]
    busy = []
    for e in gpu:
        if e["ts"] >= t1:
            break
        if e["ts"] + e["dur"] > t0:
            busy.append([max(e["ts"], t0), min(e["ts"] + e["dur"], t1)])
    busy.sort()
    merged = []
    for a, b in busy:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    prev = t0
    si = 0
    ss = [r for r in syncs if t0 - 2e5 <= r["ts"] + r["dur"] < t1]
    for a, b in merged + [[t1, t1]]:
        g = a - prev
        if g > 0:
            last = None
            for r in ss:
                if r["ts"] + r["dur"] <= prev + 5:
                    last = r
                else:
                    break
            key = labels[id(last)] if last is not None and prev - (last["ts"] + last["dur"]) < 30000 else "no recent sync"
            agg[key][0] += 1
            agg[key][1] += g
        prev = max(prev, b)
    for r in ss:
        if t0 <= r["ts"] + r["dur"] < t1:
            agg[labels[id(r)]][2] += r["dur"]
print(f"steps {n}; idle attributed to the last blocking call ended < 30 ms before the gap (ms per step)")
print(f"{'idle ms':>8} {'gaps':>6} {'cpu blocked ms':>14}  call")
for k, (c, g, b) in sorted(agg.items(), key=lambda x: -x[1][1]):
    print(f"{g/n/1e3:8.1f} {c/n:6.0f} {b/n/1e3:14.1f}  {k[:200]}")
