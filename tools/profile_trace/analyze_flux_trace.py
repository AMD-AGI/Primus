#!/usr/bin/env python3
"""Summarize FLUX GPU work and non-overlapped time from a PyTorch trace."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

GROUP_PATTERNS = {
    "RCCL": ("rccl", "nccl"),
    "AITER attention": ("aiter", "fmha", "flash_attn", "flashattention"),
    "FP8 GEMM": ("scaled_mm", "fp8_gemm", "flux_flydsl", "natural_wgrad", "hipblaslt", "hipblas"),
    "FP8 cast/scale": (
        "float8",
        "fp8",
        "amax",
        "abs_max",
        "abs_clamp",
        "dynamic_scale",
        "reciprocal",
    ),
    "norm": ("layer_norm", "rms_norm", "rmsnorm", "welford"),
    "RoPE": ("rotary", "rope"),
    "optimizer": ("adam", "foreach", "clip_grad", "grad_norm"),
    "pointwise/layout": (
        "gelu",
        "silu",
        "residual",
        "fused_add",
        "add_view",
        "rearrange",
        "permute",
        "cat",
        "split",
    ),
}


def _external_id(event: dict) -> object | None:
    args = event.get("args") or {}
    return args.get("External id", args.get("External ID"))


def _operator_names(events: list[dict]) -> tuple[dict[object, str], dict[object, object]]:
    names: dict[object, list[str]] = defaultdict(list)
    correlations = {}
    for event in events:
        external_id = _external_id(event)
        if external_id is None or event.get("ph") != "X":
            continue
        category = str(event.get("cat", "")).lower()
        if "cpu_op" in category or "user_annotation" in category:
            names[external_id].append(str(event.get("name", "")))
        if "runtime" in category and "correlation" in (event.get("args") or {}):
            correlations[event["args"]["correlation"]] = external_id
    return {key: " ".join(value) for key, value in names.items()}, correlations


def _step_windows(events: list[dict]) -> list[tuple[str, float, float]]:
    steps = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "gpu_user_annotation"
        and str(event.get("name", "")).startswith("ProfilerStep#")
    ]
    return sorted(
        ((str(event["name"]), float(event["ts"]), float(event["ts"] + event["dur"])) for event in steps),
        key=lambda item: item[1],
    )


def summarize_steps(trace: dict) -> list[dict[str, float | str]]:
    events = trace.get("traceEvents", [])
    accelerator_events = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]
    rows = []
    for name, start, stop in _step_windows(events):
        intervals = sorted(
            (
                max(start, float(event["ts"])),
                min(stop, float(event["ts"] + event.get("dur", 0.0))),
            )
            for event in accelerator_events
            if float(event.get("ts", 0.0)) < stop
            and float(event.get("ts", 0.0)) + float(event.get("dur", 0.0)) > start
        )
        merged: list[tuple[float, float]] = []
        for interval_start, interval_stop in intervals:
            if merged and interval_start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], interval_stop))
            else:
                merged.append((interval_start, interval_stop))
        active = sum(interval_stop - interval_start for interval_start, interval_stop in merged)
        rows.append(
            {
                "name": name,
                "wall_ms": (stop - start) / 1000.0,
                "active_ms": active / 1000.0,
                "idle_ms": (stop - start - active) / 1000.0,
            }
        )
    return rows


def classify(name: str) -> str:
    lowered = name.lower()
    if "cijk_" in lowered and "f8" in lowered:
        return "FP8 GEMM"
    if "cijk_" in lowered:
        return "BF16/other GEMM"
    for group, patterns in GROUP_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            return group
    if "triton" in lowered:
        return "other Triton"
    return "other GPU"


def summarize_trace(trace: dict) -> list[dict[str, float | int | str]]:
    events = trace.get("traceEvents", [])
    operators, correlations = _operator_names(events)
    windows = _step_windows(events)
    steps = max(1, len(windows))
    intervals: list[tuple[float, float, str]] = []
    work_us: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    kernels: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for event in events:
        category = str(event.get("cat", "")).lower()
        if event.get("ph") != "X" or "kernel" not in category:
            continue
        start = float(event.get("ts", 0.0))
        window_stop = next((stop for _, begin, stop in windows if begin <= start < stop), None)
        if windows and window_stop is None:
            continue
        duration = float(event.get("dur", 0.0))
        if window_stop is not None:
            duration = min(duration, window_stop - start)
        if duration <= 0:
            continue
        args = event.get("args") or {}
        external_id = _external_id(event)
        if external_id is None:
            external_id = correlations.get(args.get("correlation"))
        context = operators.get(external_id, "")
        group = classify(f"{event.get('name', '')} {context}")
        intervals.append((start, start + duration, group))
        work_us[group] += duration
        calls[group] += 1
        kernels[group][str(event.get("name", ""))] += duration

    changes: list[tuple[float, int, str]] = []
    for start, stop, group in intervals:
        changes.append((start, 1, group))
        changes.append((stop, -1, group))
    changes.sort(key=lambda item: (item[0], item[1]))

    exposed_us: dict[str, float] = defaultdict(float)
    active: dict[str, int] = defaultdict(int)
    previous = changes[0][0] if changes else 0.0
    for timestamp, delta, group in changes:
        active_groups = [name for name, count in active.items() if count > 0]
        if len(active_groups) == 1:
            exposed_us[active_groups[0]] += timestamp - previous
        active[group] += delta
        previous = timestamp

    rows = []
    for group in sorted(work_us, key=work_us.get, reverse=True):
        rows.append(
            {
                "group": group,
                "calls": calls[group],
                "calls_per_step": calls[group] / steps,
                "work_ms": work_us[group] / (1000.0 * steps),
                "exposed_ms": exposed_us[group] / (1000.0 * steps),
                "steps": steps,
                "top_kernels": sorted(
                    kernels[group].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5],
            }
        )
    return rows


def render_markdown(
    rows: list[dict[str, float | int | str]], step_rows: list[dict[str, float | str]] | None = None
) -> str:
    lines = []
    if step_rows:
        lines.extend(
            [
                "| GPU step | Wall (ms) | GPU active (ms) | GPU idle (ms) |",
                "|---|---:|---:|---:|",
            ]
        )
        for step in step_rows:
            lines.append(
                f"| {step['name']} | {step['wall_ms']:.3f} | "
                f"{step['active_ms']:.3f} | {step['idle_ms']:.3f} |"
            )
        lines.extend(
            [
                "",
                "> Exposed means the group was the only active kernel group; "
                "it is not a critical-path speedup estimate.",
                "",
            ]
        )
    lines.extend(
        [
            "| Kernel group | Calls/step | Work/step (ms) | Exposed/step (ms) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['calls_per_step']:.1f} | "
            f"{row['work_ms']:.3f} | {row['exposed_ms']:.3f} |"
        )
    for row in rows:
        lines.extend(["", f"### {row['group']} top kernels"])
        for name, duration in row["top_kernels"]:
            lines.append(f"- `{name}`: {duration / (1000.0 * row['steps']):.3f} ms/step")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="PyTorch Chrome trace JSON")
    parser.add_argument("--output", type=Path, help="Optional Markdown output path")
    args = parser.parse_args()

    with args.trace.open(encoding="utf-8") as handle:
        trace = json.load(handle)
    rows = summarize_trace(trace)
    report = render_markdown(rows, summarize_steps(trace))
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
