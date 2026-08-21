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
    "FP8 GEMM": ("scaled_mm", "fp8_gemm", "hipblaslt", "hipblas"),
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


def _operator_names(events: list[dict]) -> dict[object, str]:
    names: dict[object, list[str]] = defaultdict(list)
    for event in events:
        external_id = _external_id(event)
        if external_id is None or event.get("ph") != "X":
            continue
        category = str(event.get("cat", "")).lower()
        if "cpu_op" in category or "user_annotation" in category:
            names[external_id].append(str(event.get("name", "")))
    return {key: " ".join(value) for key, value in names.items()}


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
    operators = _operator_names(events)
    steps = max(
        1,
        len(
            {event.get("name") for event in events if str(event.get("name", "")).startswith("ProfilerStep#")}
        ),
    )
    intervals: list[tuple[float, float, str]] = []
    work_us: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    kernels: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for event in events:
        category = str(event.get("cat", "")).lower()
        if event.get("ph") != "X" or "kernel" not in category:
            continue
        duration = float(event.get("dur", 0.0))
        if duration <= 0:
            continue
        context = operators.get(_external_id(event), "")
        group = classify(f"{event.get('name', '')} {context}")
        start = float(event.get("ts", 0.0))
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


def render_markdown(rows: list[dict[str, float | int | str]]) -> str:
    lines = [
        "| Kernel group | Calls/step | Work/step (ms) | Exposed/step (ms) |",
        "|---|---:|---:|---:|",
    ]
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
        rows = summarize_trace(json.load(handle))
    report = render_markdown(rows)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
