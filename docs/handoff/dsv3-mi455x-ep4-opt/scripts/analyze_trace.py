#!/usr/bin/env python3
"""Bottleneck view of one Primus torch-profiler trace (one rank, the CPU ProfilerStep windows).

  analyze_trace.py TRACE.pt.trace.json.gz [--top 40]

Prints per step (mean over profiled steps):
  - top GPU kernels / memcpy by total time, with count and stream
  - GPU idle gaps (no kernel on any stream) by size, and the CPU runtime / op that is running
    when each gap starts (what the GPU is waiting for)
"""
import argparse
import gzip
import json
from collections import defaultdict


def union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--gap-us", type=float, default=50.0)
    a = ap.parse_args()
    op = gzip.open if a.trace.endswith(".gz") else open
    with op(a.trace, "rt") as f:
        ev = json.load(f)["traceEvents"]
    steps = [e for e in ev if e.get("ph") == "X" and e.get("name", "").startswith("ProfilerStep#")
             and e.get("cat") != "gpu_user_annotation"]
    steps.sort(key=lambda e: e["ts"])
    gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    rt = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"]
    cpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "cpu_op"]
    n = len(steps)
    print(f"steps: {n}  " + " ".join(f"{s['name']}={s['dur']/1e3:.1f}ms" for s in steps))
    tot = defaultdict(float)
    cnt = defaultdict(int)
    streams = defaultdict(set)
    gaps_by_cause = defaultdict(lambda: [0, 0.0])
    gap_hist = defaultdict(lambda: [0, 0.0])
    idle_total = 0.0
    for s in steps:
        t0, t1 = s["ts"], s["ts"] + s["dur"]
        ks = [e for e in gpu if t0 <= e["ts"] < t1]
        for e in ks:
            tot[e["name"]] += e["dur"]
            cnt[e["name"]] += 1
            streams[e["name"]].add(e.get("args", {}).get("stream", "?"))
        busy = union([(max(e["ts"], t0), min(e["ts"] + e["dur"], t1)) for e in ks])
        prev = t0
        rts = sorted((e for e in rt if t0 <= e["ts"] < t1), key=lambda e: e["ts"])
        cps = sorted((e for e in cpu if t0 <= e["ts"] < t1), key=lambda e: e["ts"])
        for b0, b1 in busy + [[t1, t1]]:
            g = b0 - prev
            if g > 0:
                idle_total += g
                bucket = "<20us" if g < 20 else "<100us" if g < 100 else "<1ms" if g < 1000 else ">=1ms"
                gap_hist[bucket][0] += 1
                gap_hist[bucket][1] += g
                if g >= a.gap_us:
                    # CPU runtime call covering the gap start, else the innermost cpu op
                    cause = None
                    for e in rts:
                        if e["ts"] <= prev + 1 and e["ts"] + e["dur"] >= prev + 1:
                            cause = "rt:" + e["name"]
                    inner = None
                    for e in cps:
                        if e["ts"] > prev + 1:
                            break
                        if e["ts"] + e["dur"] >= prev + 1:
                            inner = e["name"]
                    key = f"{cause or '-'} | {inner or '-'}"
                    gaps_by_cause[key][0] += 1
                    gaps_by_cause[key][1] += g
            prev = max(prev, b1)
    print(f"\nGPU idle per step: {idle_total/n/1e3:.1f} ms")
    for k in ("<20us", "<100us", "<1ms", ">=1ms"):
        c, t = gap_hist[k]
        print(f"  gaps {k:>7}: {c/n:7.1f} per step, {t/n/1e3:7.1f} ms")
    print(f"\nidle gaps >= {a.gap_us:.0f} us by (CPU runtime call | innermost CPU op) at gap start:")
    for k, (c, t) in sorted(gaps_by_cause.items(), key=lambda x: -x[1][1])[:25]:
        print(f"  {t/n/1e3:7.1f} ms {c/n:6.1f}x  {k[:150]}")
    print(f"\ntop {a.top} GPU ops per step (ms, count, mean us, streams):")
    for k, t in sorted(tot.items(), key=lambda x: -x[1])[: a.top]:
        print(f"  {t/n/1e3:7.1f} {cnt[k]/n:6.0f} {t/cnt[k]:8.1f}  {sorted(map(str, streams[k]))}  {k[:110]}")


if __name__ == "__main__":
    main()
