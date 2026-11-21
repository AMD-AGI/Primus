###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Expected vs. ceiling bound for the training performance projection.

The *expected* bound is what today's stack achieves (the default projection).
The *ceiling* bound projects the same deployment with the software levers at
their physical best, while kernels stay Origami's / the SDPA simulator's:

* ``overlap`` -- TP all-reduce, MoE all-to-all, gradient reduce, FSDP and PP
  point-to-point traffic are hidden behind compute, limited only by the
  compute available to hide them.
* ``fusion``  -- element-wise passes (norm / residual, MoE permute + unpermute,
  activation, expert quant / dequant) are fused into neighbouring kernels and
  cost no extra HBM traffic.
* ``launch``  -- kernel launch latency and pipeline bubbles are zero
  (persistent kernels / graph capture, bubble-free schedule).

A measurement above the ceiling means the model is missing something; the
headroom report says which lever carries the remaining gap.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Dict, Iterable, Iterator, Optional, Sequence

LEVERS = ("overlap", "fusion", "launch")
BOUNDS = ("expected", "ceiling")

_active: frozenset = frozenset()


def parse_levers(text) -> tuple:
    """``"overlap,fusion"`` / ``["overlap"]`` / ``None`` -> validated lever tuple."""
    if text is None or text == "":
        return LEVERS
    items = text.split(",") if isinstance(text, str) else list(text)
    levers = tuple(dict.fromkeys(i.strip() for i in items if i.strip()))
    unknown = [lv for lv in levers if lv not in LEVERS]
    if unknown:
        raise ValueError(f"unknown ceiling lever(s) {unknown}; choose from {list(LEVERS)}")
    return levers


def at_ceiling(lever: str) -> bool:
    return lever in _active


def active_levers() -> tuple:
    return tuple(lv for lv in LEVERS if lv in _active)


@contextlib.contextmanager
def ceiling_levers(levers: Iterable[str]) -> Iterator[None]:
    """Project with ``levers`` at their physical best for the enclosed block."""
    global _active
    previous = _active
    _active = frozenset(parse_levers(tuple(levers)) if levers else ())
    try:
        yield
    finally:
        _active = previous


def check_simulate_only(args) -> None:
    """The ceiling rescales simulated components; measured layer times cannot be split."""
    if getattr(args, "load_benchmark", None) or getattr(args, "profiling_mode", "simulate") != "simulate":
        raise ValueError(
            "--bound ceiling / --headroom need --profiling-mode simulate without --load-benchmark: "
            "measured layer times cannot be split into the comm, element-wise and launch terms the "
            "ceiling removes."
        )


def _metrics(result: Optional[dict]) -> Dict[str, float]:
    if not result or not result.get("iteration_time_ms"):
        raise RuntimeError("projection produced no multinode result (is --target-nodes below the minimum?)")
    return {
        "iteration_time_ms": float(result["iteration_time_ms"]),
        "tokens_per_sec_per_gpu": float(result["tokens_per_sec_per_gpu"]),
    }


def _gain(base: Dict[str, float], other: Dict[str, float]) -> float:
    return 100.0 * (other["tokens_per_sec_per_gpu"] / base["tokens_per_sec_per_gpu"] - 1.0)


def headroom(
    project: Callable[[Sequence[str]], Optional[dict]],
    levers: Sequence[str] = LEVERS,
    measured_tokens_per_sec_per_gpu: Optional[float] = None,
) -> dict:
    """Expected, ceiling, each lever alone, and a biggest-first waterfall.

    ``project(levers)`` runs one projection with ``levers`` at ceiling and
    returns the multinode result dict.  Levers interact (hiding comm behind
    compute is worth less once fusion shrinks the compute), so the waterfall
    stacks them and its last step equals the full ceiling.
    """
    levers = parse_levers(tuple(levers))
    expected = _metrics(project(()))
    alone = []
    for lever in levers:
        m = _metrics(project((lever,)))
        alone.append({"lever": lever, "gain_pct": _gain(expected, m), **m})
    alone.sort(key=lambda r: r["gain_pct"], reverse=True)

    waterfall, enabled = [], []
    for row in alone:
        enabled.append(row["lever"])
        m = _metrics(project(tuple(enabled)))
        waterfall.append({"levers": list(enabled), "gain_pct": _gain(expected, m), **m})
    ceiling = {k: waterfall[-1][k] for k in expected} if waterfall else dict(expected)

    report = {
        "levers": list(levers),
        "expected": expected,
        "ceiling": ceiling,
        "headroom_pct": _gain(expected, ceiling),
        "lever_gains": alone,
        "waterfall": waterfall,
    }
    if measured_tokens_per_sec_per_gpu:
        meas = float(measured_tokens_per_sec_per_gpu)
        report["measured"] = {
            "tokens_per_sec_per_gpu": meas,
            "of_expected": meas / expected["tokens_per_sec_per_gpu"],
            "of_ceiling": meas / ceiling["tokens_per_sec_per_gpu"],
            "above_ceiling": meas > ceiling["tokens_per_sec_per_gpu"],
        }
    return report


def format_headroom(report: dict) -> str:
    exp, ceil = report["expected"], report["ceiling"]
    lines = [
        "=" * 100,
        "[Primus:Performance Projection] Headroom (expected vs ceiling)",
        "=" * 100,
        f"  Expected : {exp['tokens_per_sec_per_gpu']:>12,.0f} tokens/s/GPU   "
        f"iteration {exp['iteration_time_ms']:,.1f} ms",
        f"  Ceiling  : {ceil['tokens_per_sec_per_gpu']:>12,.0f} tokens/s/GPU   "
        f"iteration {ceil['iteration_time_ms']:,.1f} ms   (+{report['headroom_pct']:.1f}%)",
        "  Each lever alone:",
    ]
    for row in report["lever_gains"]:
        lines.append(f"    {row['lever']:<8} +{row['gain_pct']:5.1f}%")
    lines.append("  Waterfall (stacked, biggest first):")
    for row in report["waterfall"]:
        lines.append(
            f"    + {'+'.join(row['levers']):<24} {row['tokens_per_sec_per_gpu']:>12,.0f} tokens/s/GPU "
            f"(+{row['gain_pct']:.1f}%)"
        )
    meas = report.get("measured")
    if meas:
        verdict = "ABOVE CEILING - modelling gap" if meas["above_ceiling"] else "within ceiling"
        lines.append(
            f"  Measured : {meas['tokens_per_sec_per_gpu']:>12,.0f} tokens/s/GPU = "
            f"{meas['of_expected']:.2f}x expected, {meas['of_ceiling']:.2f}x ceiling ({verdict})"
        )
    lines.append("=" * 100)
    return "\n".join(lines)
