###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Pretty-printers for memory-projection results.
"""

from __future__ import annotations

from typing import Any, Dict


def _gb(b: int) -> str:
    return f"{b / (1024 ** 3):.3f} GB"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def print_per_rank_breakdown(
    projection,
    *,
    target_label: str = "target",
    total_vram_bytes: int = 0,
) -> None:
    """Print a clean per-rank memory projection report.

    Args:
        projection: A ``PerRankProjection`` (from
            :func:`primus.core.projection.memory_projection.extrapolation.extrapolate_per_rank_peak`).
        target_label: Free-form name printed in the header (e.g.
            ``"32n × 8 GPUs (target_dp=64, TP=2, PP=4, EP=8)"``).
        total_vram_bytes: When > 0, the report shows headroom vs. a known
            VRAM ceiling and a clear OOM-fits decision.
    """
    bd = projection.breakdown
    diag = projection.diagnostics

    point = projection.point_estimate_bytes
    upper = projection.upper_bound_bytes

    bench_alloc = diag.get("bench_global_peak_allocated_bytes", 0)
    bench_reserved = diag.get("bench_global_peak_reserved_bytes", 0)

    static = bd["analytical_at_target_corrected"]["static"]
    acts = bd["analytical_at_target_corrected"]["activations"]

    print("")
    print("=" * 100)
    print(f"[Primus:Memory Projection] Per-rank peak memory at {target_label}")
    print("=" * 100)
    print("")

    print(f"  Bench measurements (rank 0, world_size={diag.get('bench_world_size')}):")
    print(f"    Global peak allocated:  {_gb(bench_alloc)}")
    print(f"    Global peak reserved:   {_gb(bench_reserved)}")
    print("")

    print(f"  Analytical at target (world_size={diag.get('target_world_size')}):")
    print(f"    Params (bf16):          {_gb(static['params_bytes'])}")
    print(f"    Grads  (bf16):          {_gb(static['grads_bytes'])}")
    print(f"    Optimizer (fp32, DP):   {_gb(static['optimizer_bytes'])}")
    print(f"    Activations:            {_gb(acts['transformer_layers_bytes'])}")
    print(f"    DeepEP buffers:         {_gb(bd['analytical_at_target_corrected']['deepep_buffers_bytes'])}")
    print(f"    Comm buffers (~):       {_gb(bd['analytical_at_target_corrected']['comm_buffers_bytes'])}")
    print("")

    correction = bd["activation_correction"]
    if correction.get("applied_correction"):
        print("  Activation correction (measured / analytical at bench):")
        for layer_type, factor in correction["correction_factors"].items():
            print(f"    {layer_type:<6s}: ×{factor:.3f}")
        before = correction["uncorrected_activation_bytes"]
        after = correction["corrected_activation_bytes"]
        print(f"    Activation total before correction: {_gb(before)}")
        print(f"    Activation total after correction:  {_gb(after)}")
        print("")
    else:
        print("  Activation correction: NOT applied (no per-layer measurement)")
        print("")

    # New residual decomposition (preferred when present; falls back to
    # the legacy keys for older artifacts).
    framework_overhead = bd.get("framework_overhead_bytes")
    live_excess = bd.get("live_tensor_excess_bytes")
    residual_total = bd.get("residual_total_bytes")
    margin = bd["safety_margin"]
    if framework_overhead is None or live_excess is None or residual_total is None:
        # Legacy artifact: only the conflated *_allocated/_reserved
        # fields are available.  Show them under the old label.
        res_alloc = bd.get("residual_allocated_bytes", 0)
        res_reserved = bd.get("residual_reserved_bytes", 0)
        print("  Residual (bench peak − analytical at bench):")
        print(f"    From allocated peak:    {_gb(res_alloc)} (used for point estimate)")
        print(f"    From reserved peak:     {_gb(res_reserved)} (used for upper bound)")
    else:
        print("  Residual decomposition (anchored on bench measurement):")
        print(
            f"    Framework overhead:     {_gb(framework_overhead)}  "
            "(reserved − allocated − comm_baseline)"
        )
        print(
            f"    Live-tensor excess:     {_gb(live_excess)}  "
            "(allocated − analytical_at_bench, clamped ≥ 0)"
        )
        print(
            f"    Total residual:         {_gb(residual_total)}  "
            "(added to point; ×(1+margin) for upper)"
        )
    print(f"    Safety margin on UB:    {_pct(margin)}")
    print("")

    print("  ─────────────────────────────────────────────────────────────────")
    print(f"  Point estimate (per rank): {_gb(point)}")
    print(f"  Upper bound    (per rank): {_gb(upper)}")
    print("  ─────────────────────────────────────────────────────────────────")
    print("")
    print(f"  Analytical share of point estimate: {_pct(diag.get('analytical_share', 0))}")
    print(f"  Residual share of point estimate:   {_pct(diag.get('residual_share', 0))}")

    if total_vram_bytes > 0:
        headroom_point = total_vram_bytes - point
        headroom_upper = total_vram_bytes - upper
        print("")
        print(f"  VRAM available per GPU: {_gb(total_vram_bytes)}")
        print(
            f"  Headroom (point):       {_gb(headroom_point)}"
            f" ({'FITS' if headroom_point > 0 else 'OOM'})"
        )
        print(
            f"  Headroom (upper):       {_gb(headroom_upper)}"
            f" ({'FITS' if headroom_upper > 0 else 'AT-RISK'})"
        )

    if "warning" in diag:
        print("")
        print(f"  [WARNING] {diag['warning']}")

    print("=" * 100)


def compare_simulate_vs_benchmark(
    simulate_result_or_total,
    projection,
) -> None:
    """Side-by-side print of simulate mode vs. the benchmark projection.

    ``simulate_result_or_total`` may be either:
      * an int (legacy callers that only carry total bytes), or
      * the dict returned by ``simulate.project_from_config`` containing
        ``param_optimizer_bytes`` and ``activation_bytes`` so we can
        render component-wise deltas.

    Both numbers are *per-rank at the target shape*: simulate reads the
    target parallelism straight from the YAML; benchmark anchors on a
    measured bench peak and extrapolates to that same target.  A small
    delta (within ~10–20%) means the analytical model and the residual
    term are well-calibrated; a large positive delta (simulate »
    benchmark) usually means simulate is over-counting unsharded
    components, while a large negative delta means the bench captured
    overhead (allocator fragmentation, NCCL/RCCL workspaces, kernel
    scratch) that the analytical model under-estimates.
    """
    if isinstance(simulate_result_or_total, dict):
        sim = simulate_result_or_total
        sim_total = int(sim.get("total_bytes", 0))
        sim_param_opt = int(sim.get("param_optimizer_bytes", 0))
        sim_acts = int(sim.get("activation_bytes", 0))
    else:
        sim_total = int(simulate_result_or_total or 0)
        sim_param_opt = 0
        sim_acts = 0

    point = projection.point_estimate_bytes
    upper = projection.upper_bound_bytes

    bd = projection.breakdown
    diag = projection.diagnostics
    bench_static = bd.get("analytical_at_target_corrected", {}).get("static", {})
    bench_acts = bd.get("analytical_at_target_corrected", {}).get("activations", {})
    bench_deepep = bd.get("analytical_at_target_corrected", {}).get(
        "deepep_buffers_bytes", 0
    )
    bench_comm = bd.get("analytical_at_target_corrected", {}).get(
        "comm_buffers_bytes", 0
    )
    # Prefer the new total-residual field; fall back to the legacy
    # allocated-residual for back-compat with older fixtures.
    bench_residual = bd.get(
        "residual_total_bytes",
        bd.get("residual_allocated_bytes", 0),
    )
    # ``dataclasses.asdict`` does not serialize ``@property`` fields, so
    # the breakdown nested dicts carry only the raw byte fields.  Sum
    # them explicitly here (falling back to a ``"total"`` field for any
    # caller that explicitly populated one).
    bench_static_total = bench_static.get(
        "total",
        int(bench_static.get("params_bytes", 0))
        + int(bench_static.get("grads_bytes", 0))
        + int(bench_static.get("optimizer_bytes", 0)),
    )
    bench_acts_total = bench_acts.get(
        "total",
        int(bench_acts.get("transformer_layers_bytes", 0))
        + int(bench_acts.get("embedding_bytes", 0))
        + int(bench_acts.get("output_bytes", 0))
        + int(bench_acts.get("loss_bytes", 0)),
    )

    bench_world = diag.get("target_world_size", 0)
    bench_safety = bd.get("safety_margin", 0.0)

    def _delta(a: int, b: int) -> str:
        if b <= 0:
            return "  n/a"
        pct = (a - b) / b * 100.0
        return f"{pct:+6.1f}%"

    print("")
    print("=" * 100)
    print(
        f"[Primus:Memory Projection] simulate vs benchmark "
        f"(per rank at target shape, target_world={bench_world})"
    )
    print("=" * 100)
    print(f"  {'Component':<30s} {'simulate':>14s}  {'benchmark':>14s}  {'Δ':>8s}")
    print(f"  {'-' * 30} {'-' * 14}  {'-' * 14}  {'-' * 8}")
    if sim_param_opt > 0 or sim_acts > 0:
        print(
            f"  {'Params + grads + optimizer':<30s} "
            f"{_gb(sim_param_opt):>14s}  {_gb(bench_static_total):>14s}  "
            f"{_delta(sim_param_opt, bench_static_total):>8s}"
        )
        print(
            f"  {'Activations':<30s} "
            f"{_gb(sim_acts):>14s}  {_gb(bench_acts_total):>14s}  "
            f"{_delta(sim_acts, bench_acts_total):>8s}"
        )
        print(
            f"  {'DeepEP buffers':<30s} "
            f"{'(n/a)':>14s}  {_gb(bench_deepep):>14s}  {'-':>8s}"
        )
        print(
            f"  {'Comm buffers':<30s} "
            f"{'(n/a)':>14s}  {_gb(bench_comm):>14s}  {'-':>8s}"
        )
        print(
            f"  {'Residual (framework + excess)':<30s} "
            f"{'(n/a)':>14s}  {_gb(bench_residual):>14s}  {'-':>8s}"
        )
    print(f"  {'-' * 30} {'-' * 14}  {'-' * 14}  {'-' * 8}")
    print(
        f"  {'TOTAL (point estimate)':<30s} "
        f"{_gb(sim_total):>14s}  {_gb(point):>14s}  "
        f"{_delta(sim_total, point):>8s}"
    )
    upper_label = f"TOTAL (upper bound, {int(bench_safety * 100)}% margin)"
    print(
        f"  {upper_label:<30s} "
        f"{'(n/a)':>14s}  {_gb(upper):>14s}  {'-':>8s}"
    )
    print("=" * 100)
    print(
        "  Δ legend: simulate − benchmark, expressed as % of benchmark.  "
        "Positive → simulate over-estimates; negative → simulate under-estimates."
    )
    print("=" * 100)


def report_dict(projection) -> Dict[str, Any]:
    """Return a JSON-serializable dict snapshot of the projection report."""
    return {
        "point_estimate_bytes": projection.point_estimate_bytes,
        "upper_bound_bytes": projection.upper_bound_bytes,
        "breakdown": projection.breakdown,
        "diagnostics": projection.diagnostics,
    }
