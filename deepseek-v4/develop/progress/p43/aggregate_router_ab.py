#!/usr/bin/env python3
"""Aggregate the 3-run × 2-side P43 router post-logits A/B."""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

_OUTPUT_BASE = Path(
    "/shared/amdgpu/home/wen_xie_qle/workspace/Primus-deepseek-v4/output/amd/tas-mi355x-20260515"
)
_PATTERN = re.compile(r"elapsed time per iteration .ms.: ([0-9.]+)")
# Steady-iter window: discard the first 5 iters (warmup + compile).
_STEADY_START = 5


def _read_iter_times(log_path: Path) -> list[float]:
    text = log_path.read_text(errors="replace")
    matches = _PATTERN.findall(text)
    return [float(x) for x in matches]


def _stats(name: str, values: list[float]) -> dict:
    if not values:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    print(f"P43 router A/B aggregator — base={_OUTPUT_BASE}")
    print()
    all_a: list[float] = []
    all_b: list[float] = []
    for triton in (0, 1):
        for run in (1, 2, 3):
            tag = f"p43_smoke_router_triton_{triton}_run{run}_pp1_ep8_seq4096"
            log = _OUTPUT_BASE / tag / "logs" / "pre_trainer" / "rank-7" / "debug.log"
            if not log.exists():
                print(f"   miss: {log}")
                continue
            times = _read_iter_times(log)
            steady = times[_STEADY_START:]
            print(
                f"  TRITON={triton} RUN={run}  total_iters={len(times)}  "
                f"steady_n={len(steady)}  mean={statistics.mean(steady):.2f}ms  "
                f"median={statistics.median(steady):.2f}ms"
            )
            (all_b if triton else all_a).extend(steady)

    print()
    a = _stats("A (eager, TRITON=0, 3 runs aggregated)", all_a)
    b = _stats("B (triton, TRITON=1, 3 runs aggregated)", all_b)
    for s in (a, b):
        print(
            f"  {s['name']:<48s}  n={s['n']:4d}  mean={s['mean']:.3f}ms  "
            f"median={s['median']:.3f}ms  stdev={s['stdev']:.3f}ms  "
            f"min={s['min']:.2f}  max={s['max']:.2f}"
        )

    delta = b["mean"] - a["mean"]
    # Welch's t-style noise floor: (stdev_a + stdev_b) / sqrt(min(n_a, n_b)).
    n_min = min(a["n"], b["n"])
    noise = (a["stdev"] + b["stdev"]) / (2 * (n_min**0.5))
    print()
    print(f"  delta B-A = {delta:+.3f} ms / iter")
    print(f"  combined noise floor estimate (sd_pooled/sqrt(n_min)) = ±{noise:.3f} ms")
    if abs(delta) < noise:
        print("  -> within noise band: descope reaffirmed; default OFF.")
    elif delta < 0:
        print("  -> B faster than A: candidate to flip default ON.")
    else:
        print("  -> B slower than A: regression; default stays OFF.")


if __name__ == "__main__":
    sys.exit(main())
