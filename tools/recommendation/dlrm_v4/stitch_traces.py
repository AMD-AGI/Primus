#!/usr/bin/env python3
"""Stitch per-rank Chrome traces from a dlrm_v4 run into one merged file.

When ``Profiler`` runs on multiple ranks, each rank writes its own file:

    <trace_dir>/trace_step{step}_rank{rank}.json

Each per-rank trace uses overlapping ``pid`` namespaces (CPU pid = OS pid;
GPU streams pid = 0..N), so concatenating the raw event lists would collapse
multiple ranks onto the same Perfetto track. This script:

* Identifies each pid as ``CPU`` / ``GPU`` / ``Spans`` (and other torch.profiler
  string-pid tracks) using the per-rank ``process_labels`` metadata events.
* Always drops the ``Spans`` track (low-signal in this codebase, large in
  visual clutter).
* Optionally filters to just ``cpu`` or ``gpu`` events via ``--include``.
* Sorts the surviving tracks into contiguous Perfetto sections:
  **all CPU tracks (rank 0..N) first, then all GPU tracks (rank 0..N, stream
  0..K)**.
* Remaps every event's ``pid`` and flow ``id`` so cross-rank events never
  collide on the same track or flow arrow.
* Preserves the profiler's step selection. The workload's Gin records eight
  steps and exports five of them (``Profiler.skip_head`` drops the first,
  ``Profiler.keep_active`` keeps the next five); pruning belongs to that
  exporter, not to this downstream merger.
* Rebuilds GPU-side annotation spans (``ProfilerStep#N`` included) from the
  GPU work each CPU span launched, as the training callback does, keeps
  ``ProfilerStep#N`` on each rank's main compute stream only, and applies
  the same ROCm display cleanup. Steps whose main-stream span or period is
  far above the window median are printed per rank with the host calls that
  spent the time (largest exclusive time), plus the rank whose host stalled
  longest. The ROCm cleanup may shorten a GPU slice by up to 5 us (listed in
  ``traceRendering``); use ``--no-rocm-layout`` for a copy whose GPU slices
  are unchanged. Input traces are never rewritten. These viewer files should
  not replace raw traces for timing analysis.
* Starts the window cleanly. Under the host-sync-free step the GPU is still
  running the dropped head step when the first exported CPU step begins, so
  the compute stream would open with ~100-140 ms of blank time. Per-rank
  exports keep that head step as a labelled lead-in (``args.lead_in``,
  ``ProfilerStep#N (lead-in, not exported)``, renamed here to
  ``## lead-in: step N (not exported) ##`` so tools that count
  ``ProfilerStep#`` spans see exactly the exported steps); this script clips
  it to the stitched window. The window starts at the earliest exported CPU
  step start across ranks (the cut). The trace starts once for all ranks:
  at the earliest start of head-step GPU work running when the earliest
  cut-crossing item starts (one pass, not iterated). From there every rank
  keeps its head-step GPU work on every stream (whole; the one item per
  stream still running at the trace start is dropped whole) and its host
  events, so no rank's GPU or CPU rows open blank while another rank
  finishes a long kernel (CPU annotations crossing the start are clipped;
  other host calls crossing it are dropped with their nested calls, never
  truncated; flows with a dropped end are dropped). Reported in
  ``traceRendering.lead_in``. Exports without a lead-in are unchanged.
* Keeps GPU work on each stream in launch (correlation) order before the
  rebuild. Some ROCm captures draw adjacent items on one in-order stream
  swapped or overlapping by 5-12 us, which nests a kernel inside another or
  adds a row; the repair exchanges timestamps inside such a cluster and trims
  the remaining overlap, moving no timestamp by more than 12 us. Every change
  is listed in ``traceRendering.gpu_stream_order_by_rank``.
* Leaves crossings outside the bounded repair unchanged and reports them in
  ``traceRendering``, like the training callback. One unresolved crossing does
  not disable the safe repairs elsewhere in the trace.

Because torch.profiler emits ``baseTimeNanoseconds`` from the same node clock,
timestamps line up directly across ranks — no time-shift needed for single-node
runs (multi-node would need clock-skew correction, not implemented here).

Examples
--------
Stitch step 52, default (CPU + GPU, drop Spans), gzip output::

    python tools/recommendation/dlrm_v4/stitch_traces.py <trace_dir> --step 52 --gzip

GPU-only view (skip CPU thread tree entirely — useful for kernel-level analysis)::

    python tools/recommendation/dlrm_v4/stitch_traces.py <trace_dir> --step 52 --include gpu --gzip

CPU-only view (host-side ops, profiler annotations, comm scheduling)::

    python tools/recommendation/dlrm_v4/stitch_traces.py <trace_dir> --step 52 --include cpu --gzip
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# The shared helpers have no torch/gin imports, so this CLI runs on a login
# host as well as in the training environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "primus/backends/recommendation/dlrm_v4/src"))
from generative_recommenders.dlrm_v4.trace_postprocessing import (
    _repair_gpu_nesting,
    clip_lead_in,
    exported_cpu_step_start,
    lead_in_trace_bounds,
    rebuild_gpu_annotation_spans,
    rename_lead_in_steps,
    serialize_stream_order,
)

# trace_step52_rank3.json or trace_3_rank0.json (legacy filename)
_RANK_RE = re.compile(r"trace_(?:step)?(\d+)_rank(\d+)\.json$")
_KEY_RE = re.compile(r"trace_(.+?)_rank\d+\.json$")

# Per-rank pid offset. Picked large enough that no real OS pid collides
# (Linux pids fit in 22 bits; 1e6 per rank gives ~10 ranks of headroom).
_PID_STRIDE = 1_000_000

# Per-rank flow-id offset. torch.profiler flow ids are int32/int64 — pack rank
# into the high bits so cross-rank flows can never link by accident.
_FLOW_ID_STRIDE = 1 << 40

# Sort-index sections in Perfetto. Lower = appears higher in the timeline UI.
# Each section reserves a wide range so within-section ordering (rank, stream)
# fits comfortably without overlapping the next section.
_SORT_BASE = {
    "cpu": 0,
    "gpu": 1_000_000,
    "other": 10_000_000,  # Traces / "" misc string-pid tracks
}

# `Spans` carries no useful content in our workloads (one X event per trace)
# and clutters the timeline — always dropped.
_ALWAYS_DROP_PIDS_STR = {"Spans"}


def _apply_gpu_steps(
    events: list[dict], mode: str, step_streams: str = "main"
) -> tuple[list[dict], dict | None]:
    """Apply the ``--gpu-steps`` policy to one rank's events.

    ``rebuild`` replaces every GPU-side annotation span (``ProfilerStep#N``
    included) with the extent of the GPU work its CPU span launched, linked by
    correlation id (see ``rebuild_gpu_annotation_spans``). Unlike a CPU time
    window, this stays right when the host runs ahead of the GPU.
    ``step_streams="main"`` keeps ``ProfilerStep#N`` on each rank's main
    compute stream only; ``"all"`` keeps it on every stream. ``drop``
    removes the GPU-side ``ProfilerStep`` spans; ``keep`` leaves the profiler's
    spans as emitted.
    """
    if mode == "keep":
        return events, None
    if mode == "drop":
        return [
            e
            for e in events
            if not (
                e.get("ph") == "X"
                and e.get("cat") == "gpu_user_annotation"
                and str(e.get("name", "")).startswith("ProfilerStep")
            )
        ], None
    report = rebuild_gpu_annotation_spans(events, step_streams)
    return events, report


def _classify_pid(pid_to_label: dict, pid_to_name: dict) -> dict:
    """Map original pid -> ('cpu'|'gpu'|'spans'|'other', stream_idx_or_0).

    Classification order, first match wins:
      1. pid (as a string) is in the always-drop set        -> 'spans'
      2. process_name is in the always-drop set             -> 'spans'
      3. process_labels == 'CPU'                            -> 'cpu'
      4. process_labels starts with 'GPU '                  -> 'gpu', stream id
      5. anything else (including unlabeled pids)           -> 'other'
    """
    all_pids = set(pid_to_label) | set(pid_to_name)
    out: dict = {}
    for pid in all_pids:
        label = pid_to_label.get(pid, "")
        name = pid_to_name.get(pid, "")
        if isinstance(pid, str) and pid in _ALWAYS_DROP_PIDS_STR:
            out[pid] = ("spans", 0)
            continue
        if name in _ALWAYS_DROP_PIDS_STR:
            out[pid] = ("spans", 0)
            continue
        if label == "CPU":
            out[pid] = ("cpu", 0)
        elif label.startswith("GPU"):
            try:
                stream_idx = int(label.split()[1])
            except (IndexError, ValueError):
                stream_idx = 0
            out[pid] = ("gpu", stream_idx)
        else:
            out[pid] = ("other", 0)
    return out


def _scan_pid_metadata(events: list[dict]) -> tuple[dict, dict]:
    """First pass: collect per-pid label and name from ``ph='M'`` events."""
    label: dict = {}
    name: dict = {}
    for e in events:
        if e.get("ph") != "M":
            continue
        pid = e.get("pid")
        if pid is None:
            continue
        if e.get("name") == "process_labels":
            label[pid] = e.get("args", {}).get("labels", "")
        elif e.get("name") == "process_name":
            name[pid] = e.get("args", {}).get("name", "")
    return label, name


def _new_sort_index(kind: str, rank: int, stream_idx: int) -> int:
    """Compute Perfetto sort_index so tracks group as: CPU(rank0..N), GPU(rank0..N, stream0..K), other."""
    base = _SORT_BASE.get(kind, _SORT_BASE["other"])
    return base + rank * 100 + stream_idx


def _new_pid(orig_pid, rank: int) -> object:
    """Remap a single pid into a per-rank namespace, preserving int vs str."""
    if isinstance(orig_pid, int):
        return orig_pid + rank * _PID_STRIDE
    if isinstance(orig_pid, str):
        try:
            return int(orig_pid) + rank * _PID_STRIDE
        except ValueError:
            return f"rank{rank}_{orig_pid}" if orig_pid else f"rank{rank}_misc"
    return orig_pid


def _process_one_rank(
    events: list[dict],
    rank: int,
    include: set[str],
    gpu_steps: str = "rebuild",
) -> list[dict]:
    """Filter + remap one rank's events. ``include`` is a subset of {'cpu','gpu','other'}."""
    events, _ = _apply_gpu_steps(events, gpu_steps)
    label, name = _scan_pid_metadata(events)
    classify = _classify_pid(label, name)

    out: list[dict] = []
    for e in events:
        pid = e.get("pid")
        if pid is None:
            out.append(e)
            continue
        # Always-drop check on the raw pid value first - Spans events in our
        # workloads have NO process_name/process_labels metadata, so the
        # classifier table doesn't list them. Catch them here directly.
        if isinstance(pid, str) and pid in _ALWAYS_DROP_PIDS_STR:
            continue
        kind, stream_idx = classify.get(pid, ("other", 0))
        if kind == "spans":  # always dropped
            continue
        if kind not in include:  # filtered by --include
            continue

        # Remap pid + flow id (per-rank namespace).
        e["pid"] = _new_pid(pid, rank)
        if "id" in e and e.get("ph") in ("s", "t", "f"):
            try:
                e["id"] = int(e["id"]) + rank * _FLOW_ID_STRIDE
            except (TypeError, ValueError):
                pass

        # Rewrite metadata: section-aware sort_index + rank-prefixed name.
        if e.get("ph") == "M":
            args = e.setdefault("args", {})
            if e.get("name") == "process_sort_index":
                args["sort_index"] = _new_sort_index(kind, rank, stream_idx)
            elif e.get("name") == "process_name":
                orig = args.get("name", "python")
                args["name"] = f"[Rank {rank}] {orig}"

        out.append(e)

    return out


def _group_by_step(trace_dir: Path) -> dict[str, dict[int, Path]]:
    """Map step-key (e.g. ``"step52"`` or ``"3"``) -> {rank: path}."""
    groups: dict[str, dict[int, Path]] = defaultdict(dict)
    for p in sorted(trace_dir.glob("trace_*_rank*.json")):
        m = _RANK_RE.search(p.name)
        if not m:
            continue
        prefix_match = _KEY_RE.match(p.name)
        key = prefix_match.group(1) if prefix_match else m.group(1)
        groups[key][int(m.group(2))] = p
    return dict(groups)


def stitch_one(
    rank_to_path: dict[int, Path],
    out_path: Path,
    *,
    include: set[str],
    gzip_out: bool,
    verbose: bool,
    gpu_steps: str = "rebuild",
    rocm_layout: bool = False,
    step_streams: str = "main",
) -> None:
    """Merge one (step, rank->path) group into a single trace file."""
    merged_events: list[dict] = []
    base: dict | None = None
    layout_reports: dict[int, dict] = {}
    rebuild_reports: dict[int, dict] = {}
    order_reports: dict[int, dict] = {}
    capture_notes: dict[int, list] = {}
    captures: dict[int, dict] = {}
    per_rank: list[tuple[int, list[dict], bool]] = []
    cpu_starts: dict[int, float] = {}

    for rank in sorted(rank_to_path):
        path = rank_to_path[rank]
        if verbose:
            sz_mb = path.stat().st_size / (1 << 20)
            print(f"  rank {rank}: {path.name} ({sz_mb:.1f} MB)", file=sys.stderr)
        with path.open() as f:
            trace = json.load(f)
        rocm = bool(trace.get("hip_runtime_version") or trace.get("hip_driver_version"))
        notes = (trace.get("traceRendering") or {}).get("capture_window_events")
        if notes:
            capture_notes[rank] = notes
        if trace.get("traceCapture"):
            captures[rank] = trace["traceCapture"]
        events = trace.get("traceEvents", [])
        if rocm and rocm_layout:
            # Before the rebuild, so annotation spans follow the repaired
            # launch order of the work they enclose.
            order = serialize_stream_order(events)
            if order["changes"] or order["unresolved"]:
                order_reports[rank] = order
                if order["unresolved"]:
                    print(
                        f"warning: rank {rank} keeps "
                        f"{len(order['unresolved'])} same-stream ordering "
                        "conflict(s) beyond the bounded launch-order repair; "
                        "see traceRendering.gpu_stream_order_by_rank.",
                        file=sys.stderr,
                    )
                if verbose:
                    sm = order["summary"]
                    print(
                        f"  rank {rank}: launch-order repair changed "
                        f"{sm['changed_events']} GPU item(s) "
                        f"({sm['launch_order_reassigned']} reordered, "
                        f"{sm['overlaps_trimmed']} overlaps trimmed; max "
                        f"{sm['max_shift_ns'] / 1000:.3f} us)",
                        file=sys.stderr,
                    )
        events, rebuild_report = _apply_gpu_steps(events, gpu_steps, step_streams)
        if rebuild_report is not None:
            rebuild_reports[rank] = rebuild_report
        if base is None:
            base = {k: v for k, v in trace.items() if k != "traceEvents"}
            base["distributedInfo"] = {
                **trace.get("distributedInfo", {}),
                "stitched_ranks": sorted(rank_to_path),
                "stitched_files": [p.name for p in rank_to_path.values()],
                "stitched_include": sorted(include),
                "stitched_gpu_steps": gpu_steps,
                "stitched_rocm_layout": rocm_layout,
                "stitched_step_streams": step_streams,
            }
        start = exported_cpu_step_start(events)
        if start is not None:
            cpu_starts[rank] = start
        per_rank.append((rank, events, rocm))

    assert base is not None, "no input traces provided"
    # Window start: the earliest exported CPU step start across ranks. One
    # rank alone cannot see it (the host lead differs per rank).
    lead_reports: dict[int, dict] = {}
    cut = min(cpu_starts.values()) if cpu_starts else None
    # Trace start (lead_in_trace_bounds): lead-in GPU work crossing the cut
    # is kept whole, and one rank's item running when the earliest of those
    # starts can move the start earlier once (not iterated). Every rank then
    # keeps its head-step GPU and CPU work from that one start, so no rank's
    # rows open blank while another rank finishes a long kernel.
    crossing_start = lead_start = cut
    if cut is not None:
        crossing_start, lead_start = lead_in_trace_bounds((events for _, events, _ in per_rank), cut)
    for rank, events, rocm in per_rank:
        if cut is not None:
            report = clip_lead_in(events, cut, lead_start)
            if report["lead_in_events"]:
                lead_reports[rank] = report
        if rocm and rocm_layout:
            report = _repair_gpu_nesting(events)
            if report["unresolved"]:
                print(
                    f"warning: rank {rank} retains {len(report['unresolved'])} "
                    "GPU crossing(s) outside the bounded display repair; "
                    "see traceRendering for the unchanged crossings. "
                    "Other bounded repairs remain applied.",
                    file=sys.stderr,
                )
            layout_reports[rank] = report
        rename_lead_in_steps(events)
        merged_events.extend(_process_one_rank(events, rank, include, "keep"))
    per_rank.clear()
    base["traceEvents"] = merged_events
    # Per-rank rendering reports from the training export are superseded by
    # the reports below, which describe this file.
    base.pop("traceRendering", None)
    # Each rank's export window report (Profiler trim) is kept per rank below,
    # not as rank 0's unlabelled top-level copy.
    base.pop("traceCapture", None)
    if layout_reports or rebuild_reports or capture_notes or captures or lead_reports or order_reports:
        base["traceRendering"] = {
            "scope": (
                "Display-only: GPU work on each stream kept in launch "
                "(correlation) order (a ROCm timestamp moves by at most 12 us; "
                "every change is listed in gpu_stream_order_by_rank), GPU "
                "annotation spans rebuilt from launch linkage, then bounded "
                "GPU boundary adjustments (the ROCm repair may shorten a "
                "kernel/memcpy slice by up to 5 us; every change is listed in "
                "gpu_nesting_by_rank). Stitch with --no-rocm-layout for a copy "
                "with unchanged GPU slices; input files are unchanged. "
                "Unresolved crossings remain visible."
            ),
        }
    if order_reports:
        base["traceRendering"]["gpu_stream_order_by_rank"] = order_reports
    if rebuild_reports:
        base["traceRendering"]["gpu_annotation_rebuild_by_rank"] = rebuild_reports
    if layout_reports:
        base["traceRendering"]["gpu_nesting_by_rank"] = layout_reports
    if capture_notes:
        base["traceRendering"]["capture_window_events_by_rank"] = capture_notes
    if captures:
        base["traceRendering"]["trace_capture_by_rank"] = captures
    if lead_reports:
        base["traceRendering"]["lead_in"] = {
            "scope": (
                "The dropped head step (not exported) is shown as a lead-in "
                "from trace_start_us, on GPU and CPU rows of every rank. "
                "window_start_us is the earliest exported CPU step start "
                "across ranks; crossing_start_us the earliest start, across "
                "ranks, of that step's GPU work still running at "
                "window_start_us (kept whole); trace_start_us the earliest "
                "start, across ranks, of its GPU work still running at "
                "crossing_start_us (computed once, not iterated). GPU side: "
                "on every rank and stream, the step's GPU work (kernels, "
                "memcpys, memsets) starting at or after trace_start_us is "
                "kept unchanged, so each rank's streams are busy from the "
                "top unless truly idle; work that started earlier is dropped "
                "whole (never truncated), including the one item per stream "
                "still running at trace_start_us. GPU annotations crossing "
                "trace_start_us are clipped to it "
                "(args.clipped_at_window_start). CPU side: the host events "
                "of that step from trace_start_us on every thread (CPU ops, "
                "runtime calls, annotations), so CPU rows start where GPU "
                "rows do. CPU annotations crossing trace_start_us are "
                "clipped to it unless nested in a dropped host call; other "
                "host calls crossing it are never truncated but dropped "
                "together with the calls nested in them, keeping CPU "
                "nesting valid. No kernel, memcpy, memset, runtime call or "
                "CPU op is moved or shortened. Lead-in events carry "
                "args.lead_in/owner_step and are not exported steps; launch "
                "arrows are kept only when both the launch and its GPU work "
                "are kept, other flows with a dropped end are dropped."
            ),
            "window_start_us": cut,
            "crossing_start_us": crossing_start,
            "trace_start_us": lead_start,
            "by_rank": lead_reports,
        }
        if verbose:
            kept = sum(r.get("kept_lead_in_events", 0) for r in lead_reports.values())
            print(
                f"  lead-in: window {cut:.3f} us, crossing work from "
                f"{crossing_start:.3f} us, GPU/CPU kept from "
                f"{lead_start:.3f} us, "
                f"{kept:,} lead-in event(s) "
                f"kept on {len(lead_reports)} rank(s)",
                file=sys.stderr,
            )
    stalls: dict[int, list] = defaultdict(list)
    for rank, rebuild in sorted(rebuild_reports.items()):
        for flag in (rebuild.get("step_wall") or {}).get("extended_steps", []):
            top = (flag.get("largest_host_self_time") or [None])[0]
            if top is not None:
                stalls[flag["step"]].append((top["self_ms"], rank, top["name"]))
            period = (
                f", period {flag['main_stream_period_ms']:.1f} ms vs median "
                f"{flag['median_period_ms']:.1f} ms"
                if "median_period_ms" in flag
                else ""
            )
            print(
                f"  rank {rank}: extended step {flag['step']} "
                f"({'+'.join(flag.get('flagged_by', ['span']))}): "
                f"{flag['main_stream_ms']:.1f} ms on the main stream vs median "
                f"{flag['median_ms']:.1f} ms{period}; largest host self time "
                + ", ".join(
                    f"{c['name'][:48]} {c['self_ms']:.1f} ms" for c in flag.get("largest_host_self_time", [])
                ),
                file=sys.stderr,
            )
    stall_summary = {}
    for step, hits in sorted(stalls.items()):
        self_ms, rank, name = max(hits)
        stall_summary[str(step)] = {
            "rank": rank,
            "name": name,
            "self_ms": self_ms,
            "ranks_flagged": sorted(r for _, r, _ in hits),
        }
        print(
            f"  extended step {step}: longest host stall on rank {rank} "
            f"({name[:60]} {self_ms:.1f} ms self time); flagged on "
            f"{len(hits)} rank(s)",
            file=sys.stderr,
        )
    if stall_summary:
        base.setdefault("traceRendering", {})["extended_step_host_stall"] = stall_summary

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_out:
        with gzip.open(out_path, "wt") as f:
            json.dump(base, f)
    else:
        with out_path.open("w") as f:
            json.dump(base, f)
    if verbose:
        sz_mb = out_path.stat().st_size / (1 << 20)
        print(
            f"  -> {out_path}  ({len(merged_events):,} events, {sz_mb:.1f} MB)",
            file=sys.stderr,
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("trace_dir", type=Path, help="Directory containing trace_*_rank*.json files.")
    ap.add_argument(
        "--step",
        type=str,
        default=None,
        help="Stitch only the given step key (e.g. '52' or 'step52'). "
        "Default: stitch every step group found.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Only valid when --step selects exactly "
        "one group. Default: <trace_dir>/trace_<step>.json[.gz] "
        "(or trace_<step>_cpu/_gpu when --include filters).",
    )
    ap.add_argument(
        "--include",
        choices=("cpu", "gpu", "both"),
        default="both",
        help="Which sections to keep: cpu-only tracks, gpu-only "
        "tracks, or both (default). 'Spans' is always dropped.",
    )
    ap.add_argument(
        "--gpu-steps",
        choices=("rebuild", "keep", "drop"),
        default="rebuild",
        help="GPU-side annotation spans. 'rebuild' (default) "
        "rebuilds every GPU annotation, ProfilerStep#N "
        "included, from the GPU work its CPU span launched "
        "(correlation ids), which stays correct when the host "
        "runs ahead of the GPU. 'keep' leaves the profiler's "
        "spans as emitted; 'drop' removes the GPU-side "
        "ProfilerStep spans.",
    )
    ap.add_argument(
        "--step-streams",
        choices=("main", "all"),
        default="main",
        help="With --gpu-steps rebuild, where GPU-side "
        "ProfilerStep#N spans go. 'main' (default): one per "
        "step on each rank's main compute stream (the stream "
        "with the most non-collective GPU work time). 'all': "
        "one on every stream the step launched work on.",
    )
    ap.add_argument("--gzip", action="store_true", help="Write gzip-compressed JSON (Perfetto auto-detects).")
    ap.add_argument(
        "--rocm-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply shared ROCm display cleanup to temporary copies "
        "(default: on for HIP traces; skipped for CUDA traces).",
    )
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    if not args.trace_dir.is_dir():
        print(f"error: {args.trace_dir} is not a directory", file=sys.stderr)
        return 2

    if args.include == "both":
        # 'other' covers torch.profiler string-pid tracks (Traces / misc) that
        # carry low-volume but legitimate annotations. Dropped under cpu/gpu
        # so each filtered view is clean.
        include = {"cpu", "gpu", "other"}
    else:
        include = {args.include}

    groups = _group_by_step(args.trace_dir)
    if not groups:
        print(f"error: no trace_*_rank*.json files under {args.trace_dir}", file=sys.stderr)
        return 2

    if args.step is not None:
        wanted = args.step if args.step.startswith("step") else f"step{args.step}"
        if wanted not in groups and args.step in groups:
            wanted = args.step
        if wanted not in groups:
            print(
                f"error: step {args.step!r} not found. " f"Available: {sorted(groups)}",
                file=sys.stderr,
            )
            return 2
        groups = {wanted: groups[wanted]}

    if args.out is not None and len(groups) != 1:
        print("error: --out requires --step to select exactly one group", file=sys.stderr)
        return 2

    for key, rank_map in sorted(groups.items()):
        if not args.quiet:
            print(
                f"stitching {key} ({len(rank_map)} ranks, include={args.include}):",
                file=sys.stderr,
            )
        if args.out is not None:
            out = args.out
        else:
            ext = ".json.gz" if args.gzip else ".json"
            # Default mode ("both") gets the bare filename; explicit cpu/gpu
            # filters tag the output so they can coexist in one directory.
            suffix = "" if args.include == "both" else f"_{args.include}"
            out = args.trace_dir / f"trace_{key}{suffix}{ext}"
        stitch_one(
            rank_map,
            out,
            include=include,
            gzip_out=args.gzip,
            verbose=not args.quiet,
            gpu_steps=args.gpu_steps,
            rocm_layout=args.rocm_layout,
            step_streams=args.step_streams,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
