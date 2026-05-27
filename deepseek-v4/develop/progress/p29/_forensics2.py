"""P29 forensics, second pass — pin every reduce_kernel to the deepest
Python source line in its call stack.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys

TARGET = (
    "reduce_kernel<512, 1, at::native::ReduceOp<float, "
    "at::native::func_wrapper_t<float, at::native::sum_functor<"
    "float, float, float>"
)

# A "Python source line" event in PyTorch profiler shows up with a name
# like "/abs/path/file.py(NN): func".
PY_LINE_RE = re.compile(r"\.py\((\d+)\):\s+\S+")


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

    kernels: list[dict] = []
    for e in evs:
        if e.get("ph") != "X":
            continue
        if e.get("cat") not in ("kernel", "Kernel", "gpu_op"):
            continue
        if TARGET in e.get("name", ""):
            kernels.append(e)
    print(f"  {len(kernels)} matching kernels")

    ext_to_cpu: dict[int, list[dict]] = collections.defaultdict(list)
    for e in evs:
        if e.get("ph") != "X":
            continue
        if e.get("cat") in ("cpu_op", "user_annotation", "python_function"):
            eid = _ext(e)
            if eid is not None:
                ext_to_cpu[eid].append(e)

    by_thread: dict[tuple[int, int], list[dict]] = collections.defaultdict(list)
    for e in evs:
        if e.get("ph") != "X":
            continue
        if e.get("cat") in ("cpu_op", "user_annotation", "python_function"):
            by_thread[(e.get("pid"), e.get("tid"))].append(e)
    for thread_evs in by_thread.values():
        thread_evs.sort(key=lambda x: (x["ts"], -float(x.get("dur", 0))))

    # For each reduce kernel: walk its enclosing chain on the launch
    # thread, pick the DEEPEST (smallest, most-specific) Python source
    # line, and aggregate.  Also separately track:
    #   - whether the reduce sits under autograd::engine::evaluate_function
    #     (-> BWD origin) or not (-> FWD origin)
    #   - the deepest matching Megatron / Primus / TE / V4 / training
    #     framework path
    fwd_py_hist = collections.Counter()
    bwd_py_hist = collections.Counter()
    fwd_v4_hist = collections.Counter()
    bwd_v4_hist = collections.Counter()
    fwd_module_hist = collections.Counter()
    bwd_module_hist = collections.Counter()
    fwd_dur_us = 0.0
    bwd_dur_us = 0.0

    V4_KEYWORDS = (
        "primus",
        "megatron",
        "transformer_engine",
        "deepseek_v4",
        "v4_attention",
        "v4_csa_attention",
        "compressor",
        "indexer",
    )

    def _is_v4(name: str) -> bool:
        low = name.lower()
        return any(k in low for k in V4_KEYWORDS)

    for k in kernels:
        eid = _ext(k)
        cpus = ext_to_cpu.get(eid, []) if eid is not None else []
        if not cpus:
            continue
        c = cpus[0]
        thread_evs = by_thread.get((c.get("pid"), c.get("tid")), [])
        ts0 = float(c["ts"])
        ts1 = ts0 + float(c.get("dur", 0))

        enclosing: list[dict] = [
            p
            for p in thread_evs
            if p is not c and float(p["ts"]) <= ts0 and (float(p["ts"]) + float(p.get("dur", 0))) >= ts1
        ]
        # Sort by inner-to-outer (smallest dur first).
        enclosing.sort(key=lambda x: float(x.get("dur", 0)))

        is_bwd = any(
            "autograd::engine::evaluate_function" in p["name"] or "FunctionMeta" in p["name"]
            for p in enclosing
        )

        deepest_py = None
        deepest_v4 = None
        deepest_module = None
        for p in enclosing:
            name = p["name"]
            if PY_LINE_RE.search(name) and deepest_py is None:
                deepest_py = name
            if _is_v4(name) and deepest_v4 is None:
                deepest_v4 = name
            if name.startswith("nn.Module:") and deepest_module is None:
                deepest_module = name
        target_py = bwd_py_hist if is_bwd else fwd_py_hist
        target_v4 = bwd_v4_hist if is_bwd else fwd_v4_hist
        target_mod = bwd_module_hist if is_bwd else fwd_module_hist
        if deepest_py:
            target_py[deepest_py] += 1
        if deepest_v4:
            target_v4[deepest_v4] += 1
        if deepest_module:
            target_mod[deepest_module] += 1
        if is_bwd:
            bwd_dur_us += float(k.get("dur", 0))
        else:
            fwd_dur_us += float(k.get("dur", 0))

    print(f"\nFWD origin: {sum(fwd_py_hist.values())} reduces, {fwd_dur_us / 1e3:.1f} ms total")
    print(f"BWD origin: {sum(bwd_py_hist.values())} reduces, {bwd_dur_us / 1e3:.1f} ms total")

    print("\nFWD: deepest Python source line (top 10):")
    for n, c in fwd_py_hist.most_common(10):
        print(f"  {c:5d}  {n}")
    print("\nBWD: deepest Python source line (top 10):")
    for n, c in bwd_py_hist.most_common(10):
        print(f"  {c:5d}  {n}")
    print("\nFWD: deepest nn.Module (top 10):")
    for n, c in fwd_module_hist.most_common(10):
        print(f"  {c:5d}  {n}")
    print("\nBWD: deepest nn.Module (top 10):")
    for n, c in bwd_module_hist.most_common(10):
        print(f"  {c:5d}  {n}")
    print("\nFWD: deepest V4-keyword frame (top 10):")
    for n, c in fwd_v4_hist.most_common(10):
        print(f"  {c:5d}  {n}")
    print("\nBWD: deepest V4-keyword frame (top 10):")
    for n, c in bwd_v4_hist.most_common(10):
        print(f"  {c:5d}  {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
