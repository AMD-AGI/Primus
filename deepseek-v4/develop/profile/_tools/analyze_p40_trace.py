#!/usr/bin/env python3
"""Plan-6 P40 close-out trace analyser.

Quick attribution helper for the plan-6 close-out chrome trace.  Reads
the rank-0 ``pt.trace.json`` file and emits:

  * Steady-iter profiler window timing (``ProfilerStep#6`` event)
  * Σ kernel duration + GPU active/idle break-down
  * Top-30 kernels by total time inside the steady iter window
  * V4 attention kernel attribution (dense / HCA / CSA FWD + BWD)
  * Plan-6 Triton kernel attribution (P34 / P35 / P36 / P37)
  * Multi-stream overlap factor

Usage::

    python deepseek-v4/develop/profile/_tools/analyze_p40_trace.py \
        --trace output/.../rank[0].<id>.pt.trace.json \
        --out   /tmp/p40_summary.json

Lives next to ``render_baseline_report.py``; intentionally minimal so
the JSON output can drive the markdown / HTML close-out write-up.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import pathlib
from typing import Any


def load_trace(path: pathlib.Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    with path.open("r") as fh:
        return json.load(fh)


def is_kernel(ev: dict[str, Any]) -> bool:
    return ev.get("cat", "") in ("kernel", "Kernel", "gpu_op")


def is_complete(ev: dict[str, Any]) -> bool:
    return ev.get("ph") == "X" and ev.get("dur") is not None


def is_profiler_step(ev: dict[str, Any]) -> bool:
    return isinstance(ev.get("name", ""), str) and ev["name"].startswith("ProfilerStep#")


def overlap_intervals(intervals: list[tuple[float, float]]) -> float:
    """Return the union (in microseconds) of the given (start, end) pairs."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--steady-step", type=int, default=6, help="ProfilerStep#N to analyse")
    args = ap.parse_args()

    trace = load_trace(args.trace)
    events = trace.get("traceEvents", [])

    profiler_steps = [
        ev
        for ev in events
        if is_profiler_step(ev) and is_complete(ev) and ev.get("name") == f"ProfilerStep#{args.steady_step}"
    ]
    if not profiler_steps:
        steady_step = max(
            (int(ev["name"].split("#", 1)[1]) for ev in events if is_profiler_step(ev) and is_complete(ev)),
            default=-1,
        )
        profiler_steps = [
            ev
            for ev in events
            if is_profiler_step(ev) and is_complete(ev) and ev.get("name") == f"ProfilerStep#{steady_step}"
        ]
        args.steady_step = steady_step

    if not profiler_steps:
        raise SystemExit("no ProfilerStep events in the trace")

    step = profiler_steps[0]
    win_start = float(step["ts"])
    win_end = win_start + float(step["dur"])

    def in_window(ev: dict[str, Any]) -> bool:
        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0) or 0.0)
        return ts >= win_start and ts + dur <= win_end + 1.0

    kernel_intervals: list[tuple[float, float]] = []
    per_kernel: dict[str, list[float]] = collections.defaultdict(list)
    for ev in events:
        if is_complete(ev) and is_kernel(ev) and in_window(ev):
            ts = float(ev["ts"])
            dur = float(ev["dur"])
            kernel_intervals.append((ts, ts + dur))
            per_kernel[ev.get("name", "?")].append(dur)

    win_us = win_end - win_start
    gpu_active_us = overlap_intervals(kernel_intervals)
    sum_kernel_us = sum(dur for _, _ in kernel_intervals for dur in [_ - _] if False) + sum(  # placeholder
        float(ev["dur"]) for ev in events if is_complete(ev) and is_kernel(ev) and in_window(ev)
    )

    rows = []
    for name, durs in per_kernel.items():
        total = sum(durs)
        rows.append((name, len(durs), total, total / len(durs)))
    rows.sort(key=lambda r: r[2], reverse=True)

    v4_attn_families = {
        "_v4_attention_fwd_kernel",
        "_v4_attention_bwd_kernel",
        "_v4_attention_bwd_kernel_split_dq",
        "_v4_attention_bwd_kernel_split_dkv",
        "_v4_csa_attention_pool_fwd_kernel",
        "_v4_csa_attention_pool_sparse_fwd_kernel",
        "_v4_csa_attention_pool_sparse_bwd_kernel",
        "_v4_csa_attention_pool_sparse_bwd_kernel_segreduce",
    }
    plan6_families = {
        "P34 stack_grouped_weight": [
            "_stack_grouped_linear_weight_fwd_kernel",
            "_stack_grouped_linear_weight_bwd_kernel",
        ],
        "P35 RoPE": [
            "_apply_interleaved_partial_rope_fwd_kernel",
            "_apply_interleaved_partial_rope_bwd_kernel",
        ],
        "P36 Sinkhorn": [
            "_sinkhorn_fwd_kernel",
            "_sinkhorn_bwd_kernel",
        ],
        "P37 HC tail": [
            "_hc_compute_tail_fwd_kernel",
            "_hc_compute_tail_bwd_kernel",
        ],
    }

    def select(rows, predicate):
        return [r for r in rows if predicate(r[0])]

    summary: dict[str, Any] = {
        "trace": str(args.trace),
        "steady_step": args.steady_step,
        "profiler_window_ms": win_us / 1000.0,
        "gpu_active_ms_union": gpu_active_us / 1000.0,
        "gpu_idle_ms": (win_us - gpu_active_us) / 1000.0,
        "sum_kernel_dur_ms": sum_kernel_us / 1000.0,
        "multi_stream_overlap": (sum_kernel_us / gpu_active_us) if gpu_active_us > 0 else 0.0,
        "kernel_launches": sum(r[1] for r in rows),
        "top_30": [
            {
                "name": name,
                "count": count,
                "total_ms": total / 1000.0,
                "avg_ms": avg / 1000.0,
                "pct_window": (total / win_us * 100.0) if win_us else 0.0,
                "pct_gpu_active": (total / gpu_active_us * 100.0) if gpu_active_us else 0.0,
            }
            for name, count, total, avg in rows[:30]
        ],
        "v4_attention": [
            {
                "name": name,
                "count": count,
                "total_ms": total / 1000.0,
                "avg_ms": avg / 1000.0,
                "pct_window": (total / win_us * 100.0) if win_us else 0.0,
            }
            for name, count, total, avg in select(rows, lambda n: n in v4_attn_families)
        ],
        "plan6_triton": [
            {
                "phase": phase,
                "name": name,
                "count": count,
                "total_ms": total / 1000.0,
                "avg_ms": avg / 1000.0,
                "pct_window": (total / win_us * 100.0) if win_us else 0.0,
            }
            for phase, names in plan6_families.items()
            for name in names
            for cand_name, count, total, avg in rows
            if cand_name == name
            for _name in [cand_name]
        ],
    }

    # Detect descoped kernels (should NOT be present at default-off).
    descoped_candidates = [
        "_indexer_score_fwd_kernel",
        "_indexer_score_bwd_kernel",
        "_v4_router_post_fwd_kernel",
        "_v4_router_post_bwd_kernel",
    ]
    summary["descoped_present"] = [
        {"name": name, "count": count, "total_ms": total / 1000.0}
        for name, count, total, _ in rows
        if name in descoped_candidates
    ]

    out_path = args.out
    text = json.dumps(summary, indent=2)
    if out_path is None:
        print(text)
    else:
        out_path.write_text(text + "\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
