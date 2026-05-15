#!/usr/bin/env python3
"""Summarise V4 attention kernel times from a chrome trace JSON.

Reads the rank-0 trace produced by the P57 proxy + profiler window
(iter 6 -> 7) and prints per-kernel total/mean/launches for the V4
attention BWD / FWD families plus the dense / sparse FWD kernels. Used
to update `attention_perf.md` + `proxy_ep8.md` with proxy-actual numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

KERNEL_PREFIXES = (
    "_v4_attention_fwd_kernel",
    "_v4_attention_bwd_kernel",
    "_v4_attention_bwd_dq_kernel",
    "_v4_attention_bwd_dkv_kernel",
    "_v4_attention_bwd_dpool_kernel",
    "_v4_attention_bwd_dpool_atomic_free_kernel",
    "_v4_csa_attention_pool_fwd_kernel",
    "_v4_csa_attention_pool_sparse_fwd_kernel",
    "_v4_csa_attention_pool_sparse_merge_fwd_kernel",
    "_v4_csa_attention_pool_bwd_kernel",
    "_v4_csa_attention_pool_sparse_bwd_kernel",
    "_v4_csa_attention_pool_dq_bwd_kernel",
    "_v4_csa_attention_pool_dkv_bwd_kernel",
    "_v4_csa_attention_pool_dpool_segreduce_kernel",
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("trace", type=Path)
    p.add_argument("--top", type=int, default=40)
    args = p.parse_args()

    with args.trace.open() as fh:
        trace = json.load(fh)
    events = trace.get("traceEvents", [])

    by_name: dict[str, list[float]] = defaultdict(list)
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = (ev.get("cat") or "").lower()
        if "kernel" not in cat and cat != "gpu_op":
            continue
        name = ev.get("name", "")
        if not name:
            continue
        dur = ev.get("dur")
        if dur is None:
            continue
        by_name[name].append(float(dur) * 1e-3)

    rows = sorted(by_name.items(), key=lambda kv: -sum(kv[1]))

    print(f"# Top {args.top} GPU kernels by total time")
    print(f"# trace: {args.trace}")
    print(f"# total kernels distinct: {len(by_name)}")
    print()
    print(f"{'kernel':80s}  {'launches':>8s}  {'total_ms':>10s}  {'mean_ms':>9s}")
    for name, durs in rows[: args.top]:
        n = len(durs)
        tot = sum(durs)
        mean = tot / n if n else 0.0
        print(f"{name[:80]:80s}  {n:>8d}  {tot:>10.3f}  {mean:>9.4f}")

    print()
    print("=" * 80)
    print("V4 attention kernel families (matches ANY prefix in KERNEL_PREFIXES)")
    print("=" * 80)
    families = []
    for name, durs in by_name.items():
        for pref in KERNEL_PREFIXES:
            if name.startswith(pref):
                families.append((name, durs))
                break
    families.sort(key=lambda kv: -sum(kv[1]))
    print(f"{'kernel':80s}  {'launches':>8s}  {'total_ms':>10s}  {'mean_ms':>9s}")
    for name, durs in families:
        n = len(durs)
        tot = sum(durs)
        mean = tot / n if n else 0.0
        print(f"{name[:80]:80s}  {n:>8d}  {tot:>10.3f}  {mean:>9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
