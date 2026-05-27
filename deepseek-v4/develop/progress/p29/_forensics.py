"""P29 forensic helper — attribute the dominant `aten::sum` fp32 reduce
kernel to its launching cpu_op and (where possible) Python module.

Usage: python _forensics.py [trace.json]
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys

TARGET = (
    "reduce_kernel<512, 1, at::native::ReduceOp<float, "
    "at::native::func_wrapper_t<float, at::native::sum_functor<"
    "float, float, float>"
)


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        fp = argv[1]
    else:
        trace_dir = "output/amd/tas-mi355x-20260509/p28_profile_baseline_pp1_ep8_seq4096/tensorboard"
        fp = glob.glob(os.path.join(trace_dir, "*.pt.trace.json"))[0]
    print(f"loading {fp} ...")
    data = json.load(open(fp))
    evs = data["traceEvents"]
    print(f"  {len(evs)} events")

    kernels: list[dict] = []
    for e in evs:
        if e.get("ph") != "X":
            continue
        if e.get("cat") not in ("kernel", "Kernel", "gpu_op"):
            continue
        if TARGET in e.get("name", ""):
            kernels.append(e)
    print(f"  {len(kernels)} matching kernels")

    # Build External-id index over cpu_op / user_annotation events.
    def _ext(e: dict) -> int | None:
        a = e.get("args") or {}
        for k in ("External id", "external id", "correlation"):
            v = a.get(k)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
        return None

    ext_to_cpu: dict[int, list[dict]] = collections.defaultdict(list)
    for e in evs:
        if e.get("ph") != "X":
            continue
        if e.get("cat") in ("cpu_op", "user_annotation", "python_function"):
            eid = _ext(e)
            if eid is not None:
                ext_to_cpu[eid].append(e)
    print(f"  External-id -> cpu_op map size: {len(ext_to_cpu)}")

    # 1. Direct launcher attribution.
    direct = collections.Counter()
    direct_dur_us: collections.Counter[str] = collections.Counter()
    matched_eids: list[tuple[int, dict]] = []
    for k in kernels:
        eid = _ext(k)
        if eid is None:
            continue
        cpus = ext_to_cpu.get(eid, [])
        for c in cpus:
            direct[c["name"]] += 1
            direct_dur_us[c["name"]] += float(k.get("dur", 0))
            matched_eids.append((eid, c))
    print("\n[1] Direct launcher cpu_op via External id:")
    for n, c in direct.most_common(15):
        print(f"  {c:5d}  total {direct_dur_us[n] / 1e3:8.1f} ms   {n}")

    # 2. Stack walk — for each direct launcher event, find ALL cpu_op
    # events on the same (pid, tid) thread that ENCLOSE it in time
    # (parent in the call stack).  Aggregate by parent name.
    by_thread: dict[tuple[int, int], list[dict]] = collections.defaultdict(list)
    for e in evs:
        if e.get("ph") != "X":
            continue
        if e.get("cat") not in ("cpu_op", "user_annotation", "python_function"):
            continue
        by_thread[(e.get("pid"), e.get("tid"))].append(e)
    for thread_evs in by_thread.values():
        thread_evs.sort(key=lambda x: (x["ts"], -float(x.get("dur", 0))))

    parent_hist = collections.Counter()
    for eid, c in matched_eids:
        thread_evs = by_thread.get((c.get("pid"), c.get("tid")), [])
        ts0 = float(c["ts"])
        ts1 = ts0 + float(c.get("dur", 0))
        # Linear scan — the trace is small enough, and this gives us
        # every enclosing event.  Build path list, take the OUTERMOST
        # event that is NOT the child itself.
        for p in thread_evs:
            p_ts = float(p["ts"])
            p_dur = float(p.get("dur", 0))
            if p is c:
                continue
            if p_ts <= ts0 and (p_ts + p_dur) >= ts1:
                # Encloses c.
                parent_hist[p["name"]] += 1
    print("\n[2] Enclosing cpu_op (any depth) histogram (top 30):")
    for n, c in parent_hist.most_common(30):
        print(f"  {c:6d}  {n}")

    # 3. Outermost user_annotation per direct launcher (i.e. the
    # top-level Module.forward / ProfilerStep / ATen op above the
    # kernel).
    outer_hist = collections.Counter()
    for eid, c in matched_eids:
        thread_evs = by_thread.get((c.get("pid"), c.get("tid")), [])
        ts0 = float(c["ts"])
        ts1 = ts0 + float(c.get("dur", 0))
        outermost = None
        outermost_dur = -1.0
        for p in thread_evs:
            p_ts = float(p["ts"])
            p_dur = float(p.get("dur", 0))
            if p is c:
                continue
            if p_ts <= ts0 and (p_ts + p_dur) >= ts1:
                if p_dur > outermost_dur and p["name"] != "ProfilerStep":
                    name = p["name"]
                    if name.startswith("ProfilerStep#"):
                        continue
                    outermost = name
                    outermost_dur = p_dur
        if outermost:
            outer_hist[outermost] += 1
    print("\n[3] Outermost (longest-enclosing) user_annotation excluding ProfilerStep (top 20):")
    for n, c in outer_hist.most_common(20):
        print(f"  {c:6d}  {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
