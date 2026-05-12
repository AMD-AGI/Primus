"""Tests for `pilot.tools.trace_analyze` v0.4 report shape.

Focus: the four behaviours guaranteed by the v0.4 skill — generic section
names (no internal numbering), full kernels list, per-stream pipeline
rows, and a precise compute / comm / overlap / bubble decomposition.

Builds a tiny synthetic chrome-trace with two GPU streams (compute on
stream 7, comm on stream 14) so the assertions are deterministic.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from pilot.tools import trace_analyze as ta


def _build_synthetic_trace(out_path: Path) -> None:
    """Write a 2-stream chrome trace covering one ProfilerStep#0 iter.

    Iter window: [0, 100_000] us (100 ms).
    Stream 7 (compute): one big GEMM kernel [10_000, 90_000] (80 ms).
    Stream 14 (comm):   two short collectives [40_000, 50_000] and
                        [70_000, 80_000] (10 ms each).
    Bubble:             [0..10_000] + [90_000..100_000] = 20 ms.
    Overlap (comm ∧ compute) = 20 ms (both collectives sit inside the GEMM).
    Pure compute = 80 - 20 = 60 ms.  Pure comm = 0.  Bubble = 20 ms.

    Phase markers (autograd cpu_op + Optimizer.step):
    fwd:  iter_start (0)  →  bwd_start (30_000)
    bwd:  30_000 → 80_000  (50 ms; first/last evaluate_function brackets)
    optim: 90_000 → 95_000 (5 ms FusedAdam)
    """
    events = [
        # ProfilerStep#0 marker (CPU annotation, ph='X')
        {"name": "ProfilerStep#0", "ph": "X", "ts": 0, "dur": 100_000,
         "pid": 0, "tid": 1, "cat": "user_annotation"},
        # autograd::engine::evaluate_function — phase=backward markers
        {"name": "autograd::engine::evaluate_function: MulBackward0", "ph": "X",
         "ts": 30_000, "dur": 100, "pid": 0, "tid": 1, "cat": "cpu_op"},
        {"name": "autograd::engine::evaluate_function: AddBackward0", "ph": "X",
         "ts": 79_900, "dur": 100, "pid": 0, "tid": 1, "cat": "cpu_op"},
        # Optimizer.step marker
        {"name": "Optimizer.step#FusedAdam.step", "ph": "X",
         "ts": 90_000, "dur": 5_000, "pid": 0, "tid": 1, "cat": "user_annotation"},
        # GPU kernel on compute stream 7 — GEMM (matches `compute_gemm_*`)
        {"name": "Cijk_Alik_Bljk_demo_MT256x256x128", "ph": "X",
         "ts": 10_000, "dur": 80_000, "pid": 0, "tid": 7, "cat": "kernel"},
        # GPU kernel on comm stream 14 — RCCL collective
        {"name": "ncclDevKernel_Generic_1", "ph": "X",
         "ts": 40_000, "dur": 10_000, "pid": 0, "tid": 14, "cat": "kernel"},
        {"name": "ncclDevKernel_Generic_1", "ph": "X",
         "ts": 70_000, "dur": 10_000, "pid": 0, "tid": 14, "cat": "kernel"},
    ]
    payload = {"traceEvents": events, "schemaVersion": 1, "deviceProperties": []}
    with gzip.open(out_path, "wt") as f:
        json.dump(payload, f)


def test_trace_analyze_v04_report_shape(tmp_path: Path) -> None:
    trace_path = tmp_path / "rank0.pt.trace.json.gz"
    _build_synthetic_trace(trace_path)

    patterns = ta._load_patterns(None)  # default packaged patterns
    report = ta.analyze_trace(trace_path, patterns, pipeline_width=10)

    # ---- Iter window
    assert report["iter_wallclock_ms"] == 100.0
    assert report["iter_boundary_source"] == "profiler_step"

    # ---- Bucket roll-up: GEMM in compute_gemm, RCCL in comm_collective
    bkts = report["buckets"]
    assert "compute_gemm" in bkts and bkts["compute_gemm"]["kernel_count"] == 1
    assert "comm_collective" in bkts and bkts["comm_collective"]["kernel_count"] == 2

    # ---- Full kernels list (req #2): every distinct name appears
    full = {k["name_short"]: k for k in report["kernels_full"]}
    assert any("ncclDevKernel" in n for n in full)
    assert any("Cijk" in n or "MT256x256x128" in n for n in full)
    # Per-kernel `streams` populated
    for k in report["kernels_full"]:
        assert isinstance(k.get("streams"), list)
        assert len(k["streams"]) >= 1

    # ---- Cost breakdown (req #4): partition equals iter_wallclock
    cb = report["cost_breakdown"]
    parts = cb["pure_compute_ms"] + cb["pure_comm_ms"] + cb["overlap_ms"] + cb["bubble_ms"]
    assert abs(parts - report["iter_wallclock_ms"]) < 0.5  # ms drift tolerance
    # Numerical sanity: pure compute = 60 ms, pure comm = 0, overlap = 20 ms, bubble = 20 ms
    assert 55 <= cb["pure_compute_ms"] <= 65
    assert cb["pure_comm_ms"] == 0.0
    assert 18 <= cb["overlap_ms"] <= 22
    assert 18 <= cb["bubble_ms"] <= 22

    # ---- Per-stream pipeline (req #3): one row per CUDA stream observed
    pl = report["pipeline_timeline"]
    streams = {s["stream"]: s for s in pl["streams"]}
    assert 7 in streams
    assert 14 in streams
    assert pl["stream_count_total"] == 2
    # Comm overlay row exists (even though comm rides another stream)
    assert "comm_overlay_glyphs" in pl
    # And `legend` includes the comm-collective glyph somewhere
    legend_buckets = set(pl["legend"].values())
    assert "comm_collective" in legend_buckets

    # ---- Phase boundaries detected from autograd markers
    ph = report["phases"]
    assert ph["source"] == "autograd_engine"
    assert ph["fwd_ms"] == 30.0      # iter_start → first evaluate_function
    assert 49.5 <= ph["bwd_ms"] <= 50.5
    assert ph["optim_ms"] == 5.0
    # Per-phase bucket tables present. Phase tagging uses kernel midpoint:
    # the GEMM (10..90 ms; mid=50) and BOTH comm kernels (mid=45, 75) sit
    # inside the bwd window [30..80] — so all 3 GPU kernels land in bwd.
    by_phase = ph["buckets_by_phase"]
    bwd = by_phase["bwd"]
    assert any(r["bucket"].startswith("compute_gemm") for r in bwd), \
        "GEMM (mid=50ms) should fall in bwd window [30..80]"
    assert any(r["bucket"] == "comm_collective" for r in bwd), \
        "Both ncclDevKernel events (mid=45, 75) should fall in bwd window"
    # Forward and optim windows have no GPU kernels in this synthetic trace
    # (the GEMM straddles fwd/bwd but is assigned fully to bwd by midpoint)
    assert by_phase["fwd"] == []
    assert by_phase["optim"] == []
    # avg_ms is reported and equals self_ms / kernels
    for r in by_phase["bwd"]:
        if r["kernel_count"]:
            expected = round(r["self_ms"] / r["kernel_count"], 4)
            assert abs(r["avg_ms"] - expected) < 1e-6

    # ---- Markdown render (req #1): NO `§8.x` numbering in section headers
    md = ta.render_md(report)
    assert "§" not in md, "section headers must be generic (no §8.x numbering)"
    assert "## Per-iter cost — bucket roll-up" in md
    assert "## Per-iter cost — full kernel list" in md
    assert "## Compute / Comm / Overlap / Bubble decomposition" in md
    assert "## Iter pipeline timeline (per stream)" in md
    # Phase header chip + per-phase sub-tables
    assert "Per-phase wallclock" in md
    assert "fwd=" in md and "bwd=" in md and "optim=" in md
    assert "### Forward" in md
    assert "### Backward" in md
    assert "avg_ms/kernel" in md  # column added
    # Cap support: passing N truncates the kernel list table
    md_capped = ta.render_md(report, kernels_cost_list_cap=1)
    assert "showing top 1 of" in md_capped
