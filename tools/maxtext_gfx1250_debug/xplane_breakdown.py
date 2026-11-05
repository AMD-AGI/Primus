#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU kernel-time breakdown from an xplane.pb (device 0 by default).

  xplane_breakdown.py <file.xplane.pb> [--top N] [--device N]
"""
import argparse
import collections
import re

try:
    from tensorflow.tsl.profiler.protobuf import xplane_pb2
except ImportError:
    from tsl.profiler.protobuf import xplane_pb2

CATS = [
    ("gemm_hipblaslt", re.compile(r"^Cijk_|hipblaslt|Custom_Cijk", re.I)),
    ("collective", re.compile(r"nccl|rccl|all_?gather|reduce_?scatter|all_?reduce|all_?to_?all", re.I)),
    ("memcpy", re.compile(r"memcpy|memset|copy_", re.I)),
    ("triton", re.compile(r"^triton_|triton", re.I)),
    (
        "xla_fusion",
        re.compile(
            r"fusion|loop_|input_|wrapped_|reduce|convert|transpose|broadcast|concatenate|slice|dynamic", re.I
        ),
    ),
]


def cat(name):
    for c, rx in CATS:
        if rx.search(name):
            return c
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    xs = xplane_pb2.XSpace()
    with open(a.path, "rb") as f:
        xs.ParseFromString(f.read())
    plane = next(p for p in xs.planes if p.name == f"/device:GPU:{a.device}")
    md = plane.event_metadata
    per_kernel = collections.Counter()
    counts = collections.Counter()
    t0, t1 = None, None
    for line in plane.lines:
        if not re.search(r"stream|queue", line.name, re.I) or re.search(
            r"XLA (Ops|Modules)|Steps", line.name
        ):
            continue
        for ev in line.events:
            name = md[ev.metadata_id].name
            per_kernel[name] += ev.duration_ps
            counts[name] += 1
            s = line.timestamp_ns * 1000 + ev.offset_ps
            t0 = s if t0 is None else min(t0, s)
            t1 = s + ev.duration_ps if t1 is None else max(t1, s + ev.duration_ps)
    total = sum(per_kernel.values())
    by_cat = collections.Counter()
    for k, v in per_kernel.items():
        by_cat[cat(k)] += v
    print(f"{a.path}: device {a.device}, lines={[l.name for l in plane.lines]}")
    print(f"busy kernel time {total/1e9:.1f} ms over span {(t1-t0)/1e9:.1f} ms")
    for c, v in by_cat.most_common():
        print(f"  {c:16s} {v/1e9:10.1f} ms {100*v/total:5.1f}%")
    print(f"top {a.top} kernels:")
    for k, v in per_kernel.most_common(a.top):
        print(f"  {100*v/total:5.1f}% {v/1e9:9.1f} ms n={counts[k]:6d} [{cat(k)}] {k[:130]}")


if __name__ == "__main__":
    main()
