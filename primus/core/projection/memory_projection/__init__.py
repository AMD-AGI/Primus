###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Memory projection entry point.

Dispatches between two modes based on the ``--memory-mode`` CLI flag
(default ``simulate``):

* ``simulate``  — purely analytical, no GPU required.  Fast estimate
  built from the model profiler tree and parallelism config.
* ``benchmark`` — runs the same single-node layer benchmark as the perf
  projection, captures per-rank memory, and analytically extrapolates
  to the target cluster.  Anchored on real measurements; intended for
  OOM-accurate projections of a target cluster.
* ``both``      — runs ``simulate`` and ``benchmark`` and prints a
  side-by-side comparison.
"""


def launch_projection_from_cli(args, overrides):
    mode = getattr(args, "memory_mode", "simulate") or "simulate"

    if mode == "simulate":
        from .simulate import launch_projection_from_cli as _simulate

        return _simulate(args, overrides)

    if mode == "benchmark":
        from .benchmark import launch_projection_from_cli as _benchmark

        return _benchmark(args, overrides)

    if mode == "both":
        from .benchmark import launch_projection_from_cli as _benchmark
        from .reports import compare_simulate_vs_benchmark
        from .simulate import launch_projection_from_cli as _simulate

        print("\n" + "=" * 100)
        print("[Primus:Memory Projection] Running SIMULATE mode...")
        print("=" * 100)
        sim_result = _simulate(args, overrides)

        print("\n" + "=" * 100)
        print("[Primus:Memory Projection] Running BENCHMARK mode...")
        print("=" * 100)
        bench_projection = _benchmark(args, overrides)

        # Side-by-side, only on rank 0.  ``sim_result`` carries the
        # analytical total bytes; ``bench_projection`` is the
        # ``PerRankProjection`` returned by the bench launcher.
        import os as _os

        if int(_os.getenv("RANK", "0")) == 0 and sim_result and bench_projection is not None:
            # Pass the full simulate result dict (not just the int) so the
            # comparator can render component-wise deltas (params/grads/
            # optimizer vs. activations) instead of only a one-line total.
            compare_simulate_vs_benchmark(sim_result, bench_projection)

        return bench_projection

    raise ValueError(f"Unknown --memory-mode: {mode!r}. Expected one of: simulate, benchmark, both.")


__all__ = ["launch_projection_from_cli"]
