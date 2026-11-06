# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
"""Lightweight Chrome-trace cleanup shared by export and offline tooling."""

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)

# GPU activity categories used to detect GPU stream rows and their busy time.
_GPU_KERNEL_CATS = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})


def _trace_time_ns(value: float) -> int:
    """Match Chrome-trace import's independent rounding of us to integer ns."""
    scaled = value * 1000
    whole = int(scaled)
    fraction = scaled - whole
    # Perfetto rounds half away from zero; Python round uses ties to even.
    # Adding 0.5 before conversion is also unsafe at large absolute timestamps:
    # float64 can already have a 1 ns quantum there.
    if fraction >= 0.5:
        return whole + 1
    if fraction <= -0.5:
        return whole - 1
    return whole


def _repair_gpu_nesting(
    events: list[dict], max_snap_us: float = 5.0
) -> dict:
    """Repair small GPU slice crossings in Perfetto's integer-ns domain.

    A JSON complete event that partially overlaps an open slice cannot join its
    nesting stack. Perfetto places it on an overflow track, which appears as a
    second stack on the same stream. ROCm's sub-us boundary noise and independent
    rounding of timestamps/durations can produce these crossings even when
    float-valued end timestamps appear equal.

    Keep valid nesting and exact-touch siblings intact. If an annotation's
    apparent child ends just beyond its parent, extend the annotation (and any
    affected enclosing annotations). Otherwise, shorten a small sibling overlap
    to the next slice's start, visiting the deepest open descendants first.
    Event starts and identities never change. Every cumulative end adjustment
    stays within ``max_snap_us`` of its original integer-ns endpoint. Larger or
    ambiguous crossings remain visible and are returned as ``unresolved``.

    This is display cleanup, not a source for performance measurements. Mutates
    durations/order and returns every boundary adjustment for artifact audits.
    """
    from collections import defaultdict

    if max_snap_us < 0:
        raise ValueError("max_snap_us must be nonnegative")
    max_snap_ns = _trace_time_ns(max_snap_us)
    annotation = "gpu_user_annotation"
    tracks = defaultdict(list)
    changes = []
    unresolved = []

    for index, event in enumerate(events):
        if (
            event.get("ph") != "X"
            or event.get("cat") not in _GPU_KERNEL_CATS | {annotation}
            or event.get("dur", 0) <= 0
        ):
            continue
        duration = _trace_time_ns(event["dur"])
        if duration <= 0:
            continue
        start = _trace_time_ns(event["ts"])
        end = start + duration
        tracks[(event.get("pid"), event.get("tid"))].append({
            "index": index,
            "event": event,
            "start": start,
            "end": end,
            "original_end": end,
        })

    def can_move_end(frame: dict, end: int) -> bool:
        return (
            end > frame["start"]
            and abs(end - frame["original_end"]) <= max_snap_ns
        )

    def move_end(frame: dict, end: int, reason: str, other: dict) -> None:
        if frame["end"] == end:
            return
        event = frame["event"]
        changes.append({
            "index": frame["index"],
            "name": event.get("name"),
            "cat": event.get("cat"),
            "pid": event.get("pid"),
            "tid": event.get("tid"),
            "start_ns": frame["start"],
            "original_end_ns": frame["original_end"],
            "old_end_ns": frame["end"],
            "new_end_ns": end,
            "delta_ns": end - frame["end"],
            "cumulative_delta_ns": end - frame["original_end"],
            "reason": reason,
            "other_index": other["index"],
            "other_name": other["event"].get("name"),
        })
        frame["end"] = end

    for track_key, slices in tracks.items():
        slices.sort(key=lambda frame: (
            frame["start"],
            -frame["end"],
            frame["event"].get("cat") != annotation,
            frame["index"],
        ))
        stack = []
        for current in slices:
            while stack:
                previous = stack[-1]
                if previous["end"] <= current["start"]:
                    stack.pop()
                    continue
                if current["end"] <= previous["end"]:
                    break

                overrun = current["end"] - previous["end"]
                overlap = previous["end"] - current["start"]
                near_child_end = (
                    previous["event"].get("cat") == annotation
                    and overrun <= max_snap_ns
                    and overrun < overlap
                )
                if near_child_end:
                    parents = []
                    for parent in reversed(stack):
                        if parent["end"] >= current["end"]:
                            break
                        parents.append(parent)
                    if all(
                        parent["event"].get("cat") == annotation
                        and can_move_end(parent, current["end"])
                        for parent in parents
                    ):
                        for parent in parents:
                            move_end(
                                parent, current["end"],
                                "contain_near_equal_child_end", current,
                            )
                        break

                if (
                    not near_child_end
                    and overlap <= max_snap_ns
                    and previous["start"] < current["start"]
                    and can_move_end(previous, current["start"])
                ):
                    move_end(
                        previous, current["start"],
                        "separate_sibling_overlap", current,
                    )
                else:
                    unresolved.append({
                        "track": track_key,
                        "index": previous["index"],
                        "other_index": current["index"],
                        "overlap_ns": overlap,
                        "overrun_ns": overrun,
                        "name": previous["event"].get("name"),
                        "other_name": current["event"].get("name"),
                    })
                stack.pop()
            stack.append(current)

        for frame in slices:
            if frame["end"] != frame["original_end"]:
                frame["event"]["dur"] = (
                    frame["end"] - frame["start"]
                ) / 1000

    # For equal imported start times, put containers before their children.
    # Metadata stays first and stable sorting preserves otherwise-equal events.
    events.sort(key=lambda event: (
        0 if event.get("ph") == "M" or "ts" not in event else 1,
        _trace_time_ns(event.get("ts", 0)),
        -_trace_time_ns(event.get("dur", 0)) if event.get("ph") == "X" else 0,
        event.get("cat") != annotation,
    ))
    return {
        "changes": changes,
        "unresolved": unresolved,
        "summary": {
            "boundary_adjustments": len(changes),
            "changed_events": len({change["index"] for change in changes}),
            "max_single_adjustment_ns": max(
                (abs(change["delta_ns"]) for change in changes), default=0
            ),
            "max_total_adjustment_ns": max(
                (abs(change["cumulative_delta_ns"]) for change in changes),
                default=0,
            ),
            "unresolved_crossings": len(unresolved),
        },
    }


def serialize_stream_order(
    events: list[dict], max_shift_us: float = 12.0
) -> dict:
    """Restore launch order of GPU work on each stream (bounded display repair).

    A HIP stream executes its work in submission order, and correlation ids
    are assigned in launch order, so on one stream the correlation order is
    the execution order. Some ROCm captures still record adjacent work items
    on one stream out of that order (the earlier launch drawn a few us after
    the later one started) or overlapping by 5-12 us (the earlier item's end
    recorded late). Such items nest inside each other or cross, which puts a
    second row on the stream, and an annotation rebuilt from launch linkage
    then contains a kernel launched outside it.

    Per stream, in correlation order:

    * ``launch_order``: inside each tangled cluster (a run of items whose
      start or end times are not monotone in correlation order) the
      cluster's own start times, sorted, are given to its items in
      correlation order, and likewise its end times. The cluster still
      covers exactly the same instants; only which item owns which
      timestamp changes. Start times reuse an existing item's ``ts`` value.
    * ``trim_overlap``: an item that still ends after the next item starts
      is shortened to end at that start (starts never move in this step).

    Every start and end stays within ``max_shift_us`` of its original
    integer-ns value, or the cluster/overlap is left unchanged and listed as
    unresolved. Streams with any work item lacking a correlation id are
    skipped. Changed events get ``args.stream_order_repaired``. Display
    cleanup only; raw per-rank traces keep the original timestamps. Run it
    before ``rebuild_gpu_annotation_spans`` so annotations follow the
    repaired work. Mutates ``events`` in place and returns a report.
    """
    from collections import defaultdict

    if max_shift_us < 0:
        raise ValueError("max_shift_us must be nonnegative")
    cap = _trace_time_ns(max_shift_us)
    streams: Dict[tuple, list] = defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in _GPU_KERNEL_CATS and "ts" in e:
            streams[(e.get("pid"), e.get("tid"))].append(e)

    changes: list = []
    unresolved: list = []
    skipped: list = []
    for track, items in streams.items():
        if any((e.get("args") or {}).get("correlation") is None for e in items):
            skipped.append(f"{track[0]}:{track[1]}")
            continue
        recs = []
        for e in items:
            s = _trace_time_ns(e["ts"])
            recs.append({
                "e": e, "corr": e["args"]["correlation"],
                "s": s, "e_ns": s + _trace_time_ns(e.get("dur", 0)),
                "ts": e["ts"],
            })
        recs.sort(key=lambda r: (r["corr"], r["s"]))
        for r in recs:
            r["s0"], r["e0"], r["ts0"] = r["s"], r["e_ns"], r["ts"]
        n = len(recs)
        if n < 2:
            continue
        # Cluster boundaries: after i when every earlier start/end is <= every
        # later start/end, so sorting inside clusters equals a global sort.
        suf_s = [0] * n
        suf_e = [0] * n
        suf_s[-1], suf_e[-1] = recs[-1]["s"], recs[-1]["e_ns"]
        for i in range(n - 2, -1, -1):
            suf_s[i] = min(recs[i]["s"], suf_s[i + 1])
            suf_e[i] = min(recs[i]["e_ns"], suf_e[i + 1])
        lo = 0
        max_s = max_e = None
        for i in range(n):
            max_s = recs[i]["s"] if max_s is None else max(max_s, recs[i]["s"])
            max_e = recs[i]["e_ns"] if max_e is None else max(max_e, recs[i]["e_ns"])
            if i + 1 < n and max_s <= suf_s[i + 1] and max_e <= suf_e[i + 1]:
                boundary = True
            else:
                boundary = i + 1 == n
            if not boundary:
                continue
            cluster = recs[lo:i + 1]
            lo = i + 1
            max_s = max_e = None
            if len(cluster) < 2:
                continue
            starts = sorted((r["s"], r["ts"]) for r in cluster)
            ends = sorted(r["e_ns"] for r in cluster)
            if all(
                r["s"] == st[0] and r["e_ns"] == en
                for r, st, en in zip(cluster, starts, ends)
            ):
                continue
            if any(
                abs(st[0] - r["s0"]) > cap or abs(en - r["e0"]) > cap
                for r, st, en in zip(cluster, starts, ends)
            ):
                unresolved.append({
                    "track": f"{track[0]}:{track[1]}",
                    "kind": "launch_order",
                    "correlations": [r["corr"] for r in cluster],
                    "max_shift_ns": max(
                        max(abs(st[0] - r["s0"]), abs(en - r["e0"]))
                        for r, st, en in zip(cluster, starts, ends)
                    ),
                })
                continue
            for r, st, en in zip(cluster, starts, ends):
                if r["s"] != st[0] or r["e_ns"] != en:
                    r["s"], r["ts"], r["e_ns"] = st[0], st[1], en
                    r["reason"] = "launch_order"
        for a, b in zip(recs, recs[1:]):
            if a["e_ns"] <= b["s"]:
                continue
            if b["s"] > a["s"] and abs(a["e0"] - b["s"]) <= cap:
                a["e_ns"] = b["s"]
                a["reason"] = (
                    "launch_order+trim_overlap"
                    if a.get("reason", "").startswith("launch_order")
                    else "trim_overlap"
                )
            else:
                unresolved.append({
                    "track": f"{track[0]}:{track[1]}",
                    "kind": "overlap",
                    "correlations": [a["corr"], b["corr"]],
                    "overlap_ns": a["e_ns"] - b["s"],
                })
        for r in recs:
            if r["s"] == r["s0"] and r["e_ns"] == r["e0"]:
                continue
            e = r["e"]
            e["ts"] = r["ts"]
            e["dur"] = (r["e_ns"] - r["s"]) / 1000
            args = e.setdefault("args", {})
            args["stream_order_repaired"] = True
            changes.append({
                "track": f"{track[0]}:{track[1]}",
                "correlation": r["corr"],
                "name": str(e.get("name"))[:120],
                "reason": r["reason"],
                "old_start_ns": r["s0"], "old_end_ns": r["e0"],
                "new_start_ns": r["s"], "new_end_ns": r["e_ns"],
                "start_shift_ns": r["s"] - r["s0"],
                "end_shift_ns": r["e_ns"] - r["e0"],
            })
    return {
        "scope": (
            "Display-only. On each GPU stream, work items are kept in "
            "launch (correlation) order: tangled clusters exchange their own "
            "start/end timestamps in correlation order, then an item still "
            "overlapping the next is shortened to the next start. Every "
            f"timestamp moves by at most {max_shift_us} us; raw traces keep "
            "the original values."
        ),
        "max_shift_us": max_shift_us,
        "changes": changes,
        "unresolved": unresolved,
        "skipped_streams_without_correlation": skipped,
        "summary": {
            "changed_events": len(changes),
            "launch_order_reassigned": sum(
                1 for c in changes if c["reason"].startswith("launch_order")
            ),
            "overlaps_trimmed": sum(
                1 for c in changes if "trim_overlap" in c["reason"]
            ),
            "max_shift_ns": max(
                (
                    max(abs(c["start_shift_ns"]), abs(c["end_shift_ns"]))
                    for c in changes
                ),
                default=0,
            ),
            "unresolved": len(unresolved),
        },
    }


def _repair_gpu_nesting_from_file(
    path: str, max_snap_us: float = 5.0
) -> dict:
    """Apply the shared GPU nesting repair to an exported display trace."""
    import json

    with open(path) as source:
        trace = json.load(source)
    report = _repair_gpu_nesting(trace.get("traceEvents", []), max_snap_us)
    trace.setdefault("traceRendering", {})["gpu_nesting"] = report
    with open(path, "w") as destination:
        json.dump(trace, destination)
    summary = report["summary"]
    logger.warning(
        "Repaired GPU nesting in %s: %d boundary adjustment(s), max %d ns; "
        "%d crossing(s) left unchanged",
        path, summary["boundary_adjustments"],
        summary["max_total_adjustment_ns"], summary["unresolved_crossings"],
    )
    return report


# CPU events that mark a host-side rendezvous inside a step. In the streaming
# loop the only one in a training step is the window-boundary barrier
# (``_window_boundary_barrier``); ``Profiler.align_ranks`` runs its barrier in
# the WARMUP step, which is never exported.
_BARRIER_RE = re.compile(r"^(c10d::barrier|nccl:[a-z_]*barrier)$")
# The training loop's label around that barrier when a profiler is recording
# (``_traced_window_boundary``); either marks the step as a boundary step.
_WINDOW_BOUNDARY_ANN = "## window_boundary ##"
# A CPU annotation that only a TRAINING step opens. Eval batches also advance
# the profiler, so a step without it is an eval (or other non-train) step.
_TRAIN_STEP_RE = re.compile(r"^## backward \d+ ##$")


def _step_number(name: str) -> int:
    m = re.match(r"^ProfilerStep#(\d+)$", name)
    return int(m.group(1)) if m else -1


def _classify_cpu_steps(
    events: list, cpu_spans: Dict[str, tuple], gpu_pids: set
) -> Dict[str, Dict[str, Any]]:
    """Flag each CPU ``ProfilerStep#N`` that is not an ordinary training step.

    ``barrier``: a host barrier ran inside the step (window boundary). The step
    starts from an empty GPU queue and waits for the slowest rank, so it is not
    a steady-state step. ``train``: the step opened ``## backward N ##``; a
    recorded step without it is an eval (or other non-train) step.
    """
    import bisect

    order = sorted(cpu_spans, key=lambda n: cpu_spans[n][0])
    starts = [cpu_spans[n][0] for n in order]
    flags = {n: {"barrier": False, "train": False} for n in order}

    def owner(ts: float) -> Optional[str]:
        i = bisect.bisect_right(starts, ts) - 1
        if i >= 0 and ts < cpu_spans[order[i]][1]:
            return order[i]
        return None

    for e in events:
        if e.get("ph") != "X" or e.get("pid") in gpu_pids or "ts" not in e:
            continue
        name = str(e.get("name", ""))
        if _BARRIER_RE.match(name) or (
            e.get("cat") == "user_annotation" and name == _WINDOW_BOUNDARY_ANN
        ):
            n = owner(e["ts"])
            if n is not None:
                flags[n]["barrier"] = True
        elif e.get("cat") == "user_annotation" and _TRAIN_STEP_RE.match(name):
            n = owner(e["ts"])
            if n is not None:
                flags[n]["train"] = True
    return flags


def _trim_warmup_from_trace(
    path: str,
    keep_n_active: int,
    keep_head_active: Optional[int] = None,
    skip_head: int = 0,
    skip_boundary_steps: bool = False,
    metric_log_frequency: Optional[int] = None,
    lead_in: bool = True,
) -> Optional[dict]:
    """Post-process a chrome trace to drop WARMUP steps, and optionally the tail.

    torch.profiler captures events during BOTH the WARMUP and RECORD phases
    of a schedule and writes them all to the exported trace. There is no
    built-in flag to exclude WARMUP from the export. We approximate it by:

      1) Finding all ``ProfilerStep#N`` spans in the file.
      2) Keeping only the last ``keep_n_active`` of them (sorted by start
         timestamp) as the "recorded" range.
      3) Dropping the first ``skip_head`` recorded steps (see below).
      4) If ``keep_head_active`` is set, narrowing the range further to its
         FIRST ``keep_head_active`` steps.
      5) Filtering ``traceEvents``: CPU events by the kept CPU step window,
         GPU work by the CPU step that LAUNCHED it (correlation id), and GPU
         annotations by the time of the CPU annotation they project
         (``External id``). Metadata events (``ph='M'``) are always preserved.

    ``skip_head``. ``Profiler.align_ranks`` drains the device and barriers all
    ranks at the end of the WARMUP step, so the first recorded step starts
    from an empty GPU queue. With the host-sync-free step the host normally
    runs one step ahead of the GPU; the first step after a drain has no such
    lead and is not a steady-state step (about -5% at 1n against the steps
    after it). Recording one step more and not exporting it keeps that
    rebuild step out of the artifact.

    ``lead_in`` (with ``skip_head``). Under the host-sync-free step the GPU is
    still running the dropped head step when the first exported CPU step
    begins (~110-145 ms at 1/2/4n). Trimming that work by launch leaves the
    compute stream blank at the top of the export. With ``lead_in`` the head
    step immediately before the export is kept as a labelled lead-in: its CPU
    events and the GPU work it launched stay in the file with
    ``args.lead_in = true`` and ``args.owner_step = N``, and its
    ``ProfilerStep#N`` annotations are renamed ``ProfilerStep#N (lead-in, not
    exported)`` so step counting, ``step_wall`` and the one-step-per-step
    layout ignore it. ``scripts/stitch_traces.py`` then clips the lead-in to
    one trace start across ranks (:func:`lead_in_trace_bounds`,
    :func:`clip_lead_in`), from which every rank keeps its head-step GPU
    work (whole) and CPU events; a per-rank file keeps the whole head step.

    ``keep_head_active``. The last recorded steps of a window carry the
    tracing session's teardown, and what that costs is cross-rank SKEW -- the
    eight ranks finish flushing at different times, so the tail steps go
    ragged. Job 36227 (``active=6``, exported tail == recorded tail) against
    job 36446 (``active=7``), same arm, 8 ranks:

                 36227 active=6            36446 active=7
        step 46  270.66 ms  spread 17.56   270.89 ms  spread 3.10
        step 56  279.30 ms  spread 25.06   280.82 ms  spread 0.66

    against ~1 ms of spread on the clean steps. Under the host-sync-free step
    the GPU work of the last exported step also finishes while the host is in
    the NEXT step, so at least one recorded step must follow the export.

    Boundary and eval steps. A step that ran the streaming window-boundary
    barrier (``c10d::barrier``) is not a steady-state step. Under the
    host-sync-free step the barrier first drains the run-ahead queue (~200 ms
    of ``hipStreamSynchronize`` at 1n, steps 46 and 56 of the ship captures;
    the synced trunk's barrier takes ~0.1 ms), so the step starts from an
    empty GPU queue. The 1n ship capture's step 56 (first batch of ``ts=24``)
    additionally holds a ~1.09 s host stall on rank 0 alone, inside
    ``_HSTUPreprocessAndAttentionFunction`` with no CPU op or runtime call
    under it, that the other ranks wait out in their input all-to-all. That
    stall is not the barrier, is absent at the step-46 boundary and in the
    trunk's step 56, and its cause is not identified. A step without
    ``## backward N ##`` is an eval step.
    Both are reported in ``traceCapture`` and logged. With
    ``skip_boundary_steps`` the export window moves past a barrier step when
    enough steps were recorded (it never moves into the last recorded step);
    otherwise it is left in place with a warning. File names never change.

    ``metric_log_frequency``. With it, steps whose host logs metrics
    (``(N + 1) % f == 0`` on a cold start: ``compute_and_log`` drains the
    queue at the end of that step) and the step after one are reported too.
    This mapping assumes ``ProfilerStep#N`` is train batch ``N``, which holds
    without a resume and without an eval before the window.

    Mutates the file in place and returns the ``traceCapture`` report (also
    stored in the file), or None when nothing was trimmed.
    """
    import json as _json
    with open(path) as f:
        d = _json.load(f)
    events = d.get("traceEvents", [])
    prior = d.get("traceCapture")
    if isinstance(prior, dict) and (prior.get("skip_head") or prior.get("lead_in_steps")):
        # Already trimmed with skip_head: the lead-in step is renamed, so a
        # second skip_head trim would drop one more exported step as a new
        # head and discard the old lead-in. Leave the file unchanged.
        logger.warning(
            f"{path} was already trimmed with skip_head "
            f"(exported {prior.get('exported_steps')}); not re-trimming"
        )
        return prior

    # ProfilerStep spans mark training-step boundaries; we filter by their
    # time ranges rather than by name index because step numbering can offset
    # between schedule_fn argument and the value printed in the trace.
    # torch.profiler emits one ProfilerStep#N span per CPU thread that ran
    # during that step, so dedupe by name first so "5 active steps" means
    # 5 distinct step numbers, not 5 spans.
    name_to_span: Dict[str, tuple] = {}
    # CPU-side extent of each step, kept separately. When the host runs ahead
    # of the GPU (a pipelined / sync-free step), the GPU copy of
    # ``ProfilerStep#N`` ends after CPU ``ProfilerStep#N+1`` has begun, so a
    # range built from the merged CPU+GPU spans leaks the next CPU step into
    # the export as an extra, unpaired step (iter-8/9 bench traces: one extra
    # ProfilerStep per rank -> trace_status=error).
    name_to_cpu: Dict[str, tuple] = {}
    for e in events:
        nm = e.get("name", "")
        # Exact match: a renamed lead-in step (an earlier trim) is not a step.
        if not _STEP_RE.match(str(nm)) or e.get("ph") != "X" or "ts" not in e:
            continue
        if e.get("cat") == _DEVICE_STEP_CAT:
            continue
        ts = e["ts"]
        end = ts + e.get("dur", 0)
        prev = name_to_span.get(nm)
        if prev is None:
            name_to_span[nm] = (ts, end)
        else:
            name_to_span[nm] = (min(prev[0], ts), max(prev[1], end))
        if e.get("cat") != "gpu_user_annotation":
            pc = name_to_cpu.get(nm)
            name_to_cpu[nm] = (ts, end) if pc is None else (min(pc[0], ts), max(pc[1], end))
    skip_head = max(0, int(skip_head or 0))
    if len(name_to_span) <= keep_n_active and not keep_head_active and not skip_head:
        return None
    sorted_names = sorted(name_to_span, key=lambda n: name_to_span[n])
    recorded = sorted_names[-keep_n_active:]
    head = min(skip_head, max(0, len(recorded) - 1))
    candidates = recorded[head:]
    n_keep = len(candidates)
    if keep_head_active is not None and 0 < keep_head_active < n_keep:
        n_keep = keep_head_active

    gpu_pids = {
        e.get("pid")
        for e in events
        if e.get("cat") in _GPU_KERNEL_CATS or e.get("cat") == "gpu_user_annotation"
    }
    flags = _classify_cpu_steps(events, name_to_cpu, gpu_pids)

    def bad(name: str) -> bool:
        return flags.get(name, {}).get("barrier", False)

    first = 0
    moved = False
    warnings: list = []
    window = candidates[first:first + n_keep]
    if skip_boundary_steps:
        while True:
            hits = [i for i, n in enumerate(window) if bad(n)]
            if not hits:
                break
            new_first = first + hits[-1] + 1
            # Keep at least one recorded step after the export (see above).
            if new_first + n_keep > len(candidates) - 1:
                warnings.append(
                    f"skip_boundary_steps: cannot move past "
                    f"{window[hits[-1]]}; only {len(candidates)} step(s) "
                    f"recorded after skip_head (need {new_first + n_keep + 1}). "
                    f"Raise Profiler.active."
                )
                break
            first, moved = new_first, True
            window = candidates[first:first + n_keep]
    active_names = window

    for n in active_names:
        if bad(n):
            warnings.append(
                f"{n} contains a host barrier (streaming window boundary): it "
                f"starts from an empty GPU queue and waits for the slowest rank"
            )
        if n in flags and not flags[n]["train"]:
            warnings.append(f"{n} has no '## backward N ##': not a training step (eval?)")
    # None (not []) when no frequency was given: the training callback does
    # not pass one and reports metric-log steps through Profiler.note_event
    # (traceRendering.capture_window_events) instead.
    metric_steps: Optional[list] = None
    if metric_log_frequency and metric_log_frequency > 0:
        metric_steps = []
        f_ = int(metric_log_frequency)
        for n in active_names:
            s = _step_number(n)
            if s < 0:
                continue
            if (s + 1) % f_ == 0:
                metric_steps.append(s)
                warnings.append(
                    f"{n} logs metrics (METRIC_LOG_FREQ={f_}): its host drains "
                    f"the GPU queue at the end of the step"
                )
            if s % f_ == 0 and s > 0:
                warnings.append(
                    f"{n} follows a metric-log step (METRIC_LOG_FREQ={f_}): it "
                    f"starts from an empty GPU queue"
                )

    active = [name_to_span[n] for n in active_names]
    t_start = min(s for s, _ in active)
    t_end = max(e for _, e in active)
    cpu_active = [name_to_cpu[n] for n in active_names if n in name_to_cpu]
    if cpu_active:
        c_start = min(s for s, _ in cpu_active)
        c_end = max(e for _, e in cpu_active)
    else:
        c_start, c_end = t_start, t_end

    # GPU-side events belong to the step whose CPU range LAUNCHED them (linked
    # by correlation id), not to whatever step the host was in when the kernel
    # started. Events with no resolvable launch fall back to the merged range.
    launch_ts: Dict[Any, float] = {}
    # GPU annotations carry no correlation id; they are Kineto's projection of
    # a CPU annotation with the same External id, so they follow that CPU
    # annotation. A time-window test would keep the NEXT step's input-pipeline
    # annotations (copy_batch_to_gpu N+3, start_sparse_data_dist N+2, ...)
    # whenever their GPU work runs inside the window while their CPU span was
    # trimmed away, leaving orphans with no CPU parent (42 at 1n, 160 at 4n).
    cpu_ann_ts: Dict[Any, float] = {}
    for e in events:
        if e.get("ph") == "X" and e.get("pid") not in gpu_pids and "ts" in e:
            args = e.get("args") or {}
            corr = args.get("correlation")
            if corr is not None:
                launch_ts[corr] = e["ts"]
            if e.get("cat") == "user_annotation":
                ext = args.get("External id")
                if ext is not None:
                    cpu_ann_ts[ext] = e["ts"]

    # Lead-in: the recorded, unexported step right before the export (see
    # ``lead_in`` above). Only with skip_head: without it the step before the
    # export is WARMUP.
    lead_name: Optional[str] = None
    if lead_in and head and active_names:
        i = sorted_names.index(active_names[0])
        prev = sorted_names[i - 1] if i > 0 else None
        if prev is not None and prev in recorded and prev in name_to_cpu:
            lead_name = prev
    l_start = name_to_cpu[lead_name][0] if lead_name is not None else c_start
    lead_step = _step_number(lead_name) if lead_name is not None else -1

    def _where(e: dict) -> Optional[str]:
        """'keep', 'lead' (lead-in) or None (drop)."""
        if e.get("ph") == "M":
            return "keep"
        ts = e.get("ts")
        if ts is None:
            return "keep"
        ref = ts
        if e.get("pid") in gpu_pids:
            if e.get("cat") == "gpu_user_annotation":
                ext = (e.get("args") or {}).get("External id")
                at = cpu_ann_ts.get(ext) if ext is not None else None
                if at is None:
                    if ext is not None and cpu_ann_ts:
                        # Its CPU annotation is not in the file at all (an
                        # earlier trim removed it): an orphan, never kept.
                        return None
                    return "keep" if t_start <= ts < t_end else None
                ref = at
            else:
                corr = (e.get("args") or {}).get("correlation")
                if corr is None and e.get("cat") == "ac2g":
                    corr = e.get("id")
                lt = launch_ts.get(corr) if corr is not None else None
                if lt is None:
                    return "keep" if t_start <= ts < t_end else None
                ref = lt
        if c_start <= ref < c_end:
            return "keep"
        if lead_name is not None and l_start <= ref < c_start:
            return "lead"
        return None

    kept = []
    n_lead = 0
    lead_label = (
        f"{lead_name}{_LEAD_IN_SUFFIX}" if lead_name is not None else None
    )
    for e in events:
        where = _where(e)
        if where is None:
            continue
        if where == "lead":
            n_lead += 1
            args = e.get("args")
            if not isinstance(args, dict):
                args = e["args"] = {}
            args["lead_in"] = True
            args["owner_step"] = lead_step
            if e.get("name") == lead_name and e.get("cat") in (
                "user_annotation", "gpu_user_annotation"
            ):
                e["name"] = lead_label
        kept.append(e)
    d["traceEvents"] = kept
    capture = {
        "recorded_steps": [_step_number(n) for n in recorded],
        "skipped_head_steps": [_step_number(n) for n in recorded[:head]],
        "exported_steps": [_step_number(n) for n in active_names],
        "skip_head": head,
        "skip_boundary_steps": bool(skip_boundary_steps),
        "moved_past_boundary": moved,
        "barrier_steps": sorted(
            _step_number(n) for n in recorded if bad(n)
        ),
        "non_train_steps": sorted(
            _step_number(n) for n in recorded
            if n in flags and not flags[n]["train"]
        ),
        "metric_log_steps": metric_steps,
        "lead_in_steps": [lead_step] if lead_name is not None else [],
        "lead_in_events": n_lead,
        "exported_cpu_start_us": c_start,
        "warnings": warnings,
    }
    d["traceCapture"] = capture
    with open(path, "w") as f:
        _json.dump(d, f)
    logger.warning(
        f"Trimmed WARMUP events from {path}: {len(events):,} -> {len(kept):,} "
        f"(kept {len(active)} step(s) {capture['exported_steps']}, active range "
        f"[{t_start:.0f}, {t_end:.0f}] us"
        + (f", skipped head {capture['skipped_head_steps']}" if head else "")
        + (f", lead-in {lead_name} ({n_lead:,} events)" if lead_name else "")
        + (f", head-trimmed to first {keep_head_active}" if keep_head_active else "")
        + ")"
    )
    for w in warnings:
        logger.warning(f"trace capture {path}: {w}")
    return capture


_ANNOTATION_CAT = "gpu_user_annotation"
# Category and thread-id range of the synthetic device-wide step rows ("GPU
# steps (even)/(odd)") that earlier exports carried. No longer written; a
# rebuild removes them so re-stitching an old export drops those rows.
_DEVICE_STEP_CAT = "gpu_step"
_DEVICE_STEP_TID_BASE = 2_000_000_000
_STEP_RE = re.compile(r"^ProfilerStep#(\d+)$")
# Name given to the lead-in (dropped head) step's annotations by the trim, so
# that only exported steps match _STEP_RE.
_LEAD_IN_SUFFIX = " (lead-in, not exported)"
_LEAD_IN_STEP_RE = re.compile(r"^ProfilerStep#(\d+) \(lead-in\b")
# Display name of the lead-in step in stitched traces (rename_lead_in_steps):
# it no longer starts with "ProfilerStep#", so per-step tools that match that
# prefix count only the exported steps.
_LEAD_IN_DISPLAY_RE = re.compile(r"^## lead-in: step (\d+) \(not exported\) ##$")


def _is_step_span_name(name: str) -> bool:
    """An exported or a lead-in ``ProfilerStep`` annotation name."""
    return bool(
        _STEP_RE.match(name)
        or _LEAD_IN_STEP_RE.match(name)
        or _LEAD_IN_DISPLAY_RE.match(name)
    )


def rename_lead_in_steps(events: list[dict]) -> int:
    """Rename lead-in step spans to ``## lead-in: step N (not exported) ##``.

    The per-rank export labels the dropped head step
    ``ProfilerStep#N (lead-in, not exported)``. Step parsers that match the
    ``ProfilerStep#`` prefix (HTA/TraceLens-style) would count it as a sixth,
    truncated step. The original name is kept in ``args.lead_in_step_name``;
    ``args.lead_in``/``owner_step`` are unchanged. Returns the count renamed.
    """
    renamed = 0
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in ("user_annotation", _ANNOTATION_CAT):
            continue
        m = _LEAD_IN_STEP_RE.match(str(e.get("name", "")))
        if not m:
            continue
        args = e.get("args")
        if not isinstance(args, dict):
            args = e["args"] = {}
        args["lead_in_step_name"] = e["name"]
        e["name"] = f"## lead-in: step {m.group(1)} (not exported) ##"
        renamed += 1
    return renamed
# Collective kernels (RCCL/NCCL/MSCCL), excluded when picking the main stream.
_COMM_KERNEL_RE = re.compile(r"nccl|msccl|rccl", re.IGNORECASE)


def _gpu_track_crossings(
    events: list[dict], cats: frozenset = frozenset({_ANNOTATION_CAT})
) -> list[dict]:
    """Partial overlaps among same-track GPU slices in Perfetto's ns domain.

    A slice that starts inside an open slice but ends after it cannot nest and
    is drawn on an overflow row. Returns one record per such crossing.
    """
    from collections import defaultdict

    tracks: Dict[tuple, list] = defaultdict(list)
    for index, event in enumerate(events):
        if event.get("ph") != "X" or event.get("cat") not in cats:
            continue
        start = _trace_time_ns(event["ts"])
        end = start + _trace_time_ns(event.get("dur", 0))
        tracks[(event.get("pid"), event.get("tid"))].append(
            (start, -end, event.get("cat") != _ANNOTATION_CAT, index, end)
        )
    crossings = []
    for track, slices in tracks.items():
        slices.sort()
        stack: list = []
        for start, _, _, index, end in slices:
            while stack and stack[-1][1] <= start:
                stack.pop()
            if stack and end > stack[-1][1]:
                crossings.append({
                    "track": f"{track[0]}:{track[1]}",
                    "name": events[stack[-1][0]].get("name"),
                    "other_name": events[index].get("name"),
                    "overlap_ns": stack[-1][1] - start,
                })
                stack.pop()
            stack.append((index, end))
    return crossings


# A step is flagged as extended when its main-stream span exceeds the window
# median by both this ratio and this many ms.
_EXTENDED_STEP_RATIO = 1.25
_EXTENDED_STEP_MIN_MS = 10.0


def _host_self_time(events: list[dict], pid: Any, t0: float, t1: float) -> list:
    """CPU events started in ``[t0, t1)`` on ``pid``, by exclusive time.

    Exclusive (self) time is an event's duration minus its direct children on
    the same thread, so a wait or stall is named by the innermost call that
    actually spent it: a host stall inside an op with no child calls, or a
    ``## splits poll ##``-style wait annotation, rather than its enclosing
    ``## forward N ##`` or the step itself.
    """
    from collections import defaultdict

    by_tid: Dict[Any, list] = defaultdict(list)
    for e in events:
        if (
            e.get("ph") == "X"
            and e.get("pid") == pid
            and e.get("cat") in ("cpu_op", "cuda_runtime", "user_annotation")
            and t0 <= e.get("ts", -1) < t1
            and not _STEP_RE.match(str(e.get("name", "")))
        ):
            by_tid[e.get("tid")].append(e)
    out = []
    for tid, evs in by_tid.items():
        evs.sort(key=lambda e: (e["ts"], -e.get("dur", 0)))
        selfs = {id(e): e.get("dur", 0) for e in evs}
        stack: list = []
        for e in evs:
            while stack and stack[-1]["ts"] + stack[-1].get("dur", 0) <= e["ts"]:
                stack.pop()
            if stack:
                selfs[id(stack[-1])] -= min(
                    e.get("dur", 0),
                    stack[-1]["ts"] + stack[-1].get("dur", 0) - e["ts"],
                )
            stack.append(e)
        out.extend((selfs[id(e)], e) for e in evs)
    out.sort(key=lambda x: -x[0])
    return out


def _step_wall_report(events: list[dict], main_stream: Dict[Any, Any]) -> dict:
    """Per-step main-stream extents and periods, and the steps far above them.

    A step's main-stream span runs from its first to its last compute-stream
    work item; its period runs from its first work item to the next step's.
    A host stall inside the step (a window-boundary barrier draining the
    run-ahead queue, a slow first forward of a new data window, an eval or a
    metrics drain) widens the span; a stall before the step's first launch
    (e.g. in a synced run) only shows in the period of the step before it,
    so a step is flagged when either exceeds the window median. For each
    flagged step the report gives the largest idle gap on the main stream,
    the longest CPU calls started inside the CPU step, and the CPU events with
    the most exclusive (self) time, which name the stall; the rank whose
    host stalled is the one whose top self-time event is longest.
    """
    from collections import defaultdict
    from statistics import median

    by_step: Dict[Any, Dict[int, list]] = defaultdict(dict)
    for e in events:
        if (
            e.get("ph") == "X"
            and e.get("cat") == _ANNOTATION_CAT
            and main_stream.get(e.get("pid")) == (e.get("pid"), e.get("tid"))
        ):
            m = _STEP_RE.match(str(e.get("name", "")))
            if m:
                start = _trace_time_ns(e["ts"])
                by_step[e.get("pid")][int(m.group(1))] = [
                    start, start + _trace_time_ns(e.get("dur", 0))
                ]
    cpu_steps: Dict[int, list] = {}
    for e in events:
        if e.get("ph") == "X" and e.get("cat") == "user_annotation":
            m = _STEP_RE.match(str(e.get("name", "")))
            if m:
                cpu_steps[int(m.group(1))] = [
                    e["ts"], e["ts"] + e.get("dur", 0), e.get("pid"), e.get("tid")
                ]
    cpu_order = sorted(cpu_steps)

    report: Dict[str, Any] = {
        "ratio": _EXTENDED_STEP_RATIO,
        "min_excess_ms": _EXTENDED_STEP_MIN_MS,
        "main_stream_step_ms": {},
        "main_stream_period_ms": {},
        "cpu_step_ms": {
            str(n): round((v[1] - v[0]) / 1000, 3) for n, v in sorted(cpu_steps.items())
        },
        "cpu_step_period_ms": {
            str(a): round((cpu_steps[b][0] - cpu_steps[a][0]) / 1000, 3)
            for a, b in zip(cpu_order, cpu_order[1:]) if b == a + 1
        },
        "extended_steps": [],
    }

    def far(value: float, med: float) -> bool:
        return value > med * _EXTENDED_STEP_RATIO and value - med > _EXTENDED_STEP_MIN_MS

    for pid, steps in by_step.items():
        order = sorted(steps)
        spans = {n: (steps[n][1] - steps[n][0]) / 1e6 for n in order}
        periods = {
            a: (steps[b][0] - steps[a][0]) / 1e6
            for a, b in zip(order, order[1:]) if b == a + 1
        }
        report["main_stream_step_ms"][str(pid)] = {
            str(n): round(v, 3) for n, v in spans.items()
        }
        report["main_stream_period_ms"][str(pid)] = {
            str(n): round(v, 3) for n, v in periods.items()
        }
        if len(steps) < 3:
            continue
        med = median(spans.values())
        med_period = median(periods.values()) if len(periods) >= 2 else None
        track = main_stream[pid]
        work = sorted(
            (_trace_time_ns(e["ts"]), _trace_time_ns(e["ts"]) + _trace_time_ns(e.get("dur", 0)))
            for e in events
            if e.get("ph") == "X" and e.get("cat") in _GPU_KERNEL_CATS
            and (e.get("pid"), e.get("tid")) == track
        )
        for n in order:
            a, b = steps[n]
            ms = spans[n]
            by = []
            if far(ms, med):
                by.append("span")
            if med_period is not None and n in periods and far(periods[n], med_period):
                by.append("period")
            if not by:
                continue
            gap, gap_at, prev_end = 0, a, None
            for ws, we in work:
                if we <= a or ws >= b:
                    continue
                if prev_end is not None and ws - prev_end > gap:
                    gap, gap_at = ws - prev_end, prev_end
                prev_end = we if prev_end is None else max(prev_end, we)
            flag: Dict[str, Any] = {
                "pid": pid,
                "step": n,
                "flagged_by": by,
                "main_stream_ms": round(ms, 3),
                "median_ms": round(med, 3),
                "largest_main_stream_gap_ms": round(gap / 1e6, 3),
                "largest_gap_offset_ms": round((gap_at - a) / 1e6, 3),
            }
            if n in periods:
                flag["main_stream_period_ms"] = round(periods[n], 3)
                if med_period is not None:
                    flag["median_period_ms"] = round(med_period, 3)
            cpu = cpu_steps.get(n)
            if cpu is not None:
                flag["cpu_step_ms"] = round((cpu[1] - cpu[0]) / 1000, 3)
                calls = sorted(
                    (
                        e for e in events
                        if e.get("ph") == "X"
                        and e.get("cat") in ("cpu_op", "cuda_runtime")
                        and e.get("pid") == cpu[2]
                        and cpu[0] <= e.get("ts", -1) < cpu[1]
                    ),
                    key=lambda e: -e.get("dur", 0),
                )[:3]
                flag["longest_cpu_calls"] = [
                    {
                        "name": str(e.get("name"))[:120],
                        "tid": e.get("tid"),
                        "offset_ms": round((e["ts"] - cpu[0]) / 1000, 3),
                        "ms": round(e.get("dur", 0) / 1000, 3),
                    }
                    for e in calls
                ]
                flag["largest_host_self_time"] = [
                    {
                        "name": str(e.get("name"))[:120],
                        "cat": e.get("cat"),
                        "tid": e.get("tid"),
                        "offset_ms": round((e["ts"] - cpu[0]) / 1000, 3),
                        "self_ms": round(self_us / 1000, 3),
                        "ms": round(e.get("dur", 0) / 1000, 3),
                    }
                    for self_us, e in _host_self_time(events, cpu[2], cpu[0], cpu[1])[:3]
                ]
            report["extended_steps"].append(flag)
    return report


def rebuild_gpu_annotation_spans(
    events: list[dict], step_streams: str = "main"
) -> dict:
    """Rebuild GPU-side annotation spans from the work each CPU span launched.

    Kineto projects every CPU ``user_annotation`` onto each GPU stream as a
    ``gpu_user_annotation`` covering only the kernels launched while that
    annotation was the *innermost* scope. Parents whose launches all sit in
    child scopes (``ProfilerStep#N``, ``## forward N ##``) come out as slivers.
    Time-window repairs (a GPU step = work starting inside the CPU step) are
    only right while the host waits for the GPU every step. Once the host runs
    ahead (host-sync-free step), a CPU step window covers the tail of the
    previous GPU step and the head of its own, so such spans cross their
    children by tens of ms and Perfetto stacks them on overflow rows.

    This replaces all Kineto ``gpu_user_annotation`` events with spans derived
    from launch linkage. Each kernel/memcpy/memset is linked to the CPU runtime
    call carrying the same ``args.correlation``. The call's enclosing CPU
    annotations are the open annotations on its thread; worker threads (e.g.
    ``pt_autograd_0``) also inherit the open annotations of the thread that owns
    ``ProfilerStep#N`` in the same process. For every CPU annotation and every
    GPU stream it has work on, the span runs from its first work item's start to
    its last work item's end, computed in Perfetto's integer-ns domain so the
    span exactly encloses its children. Spans are keyed by CPU event identity,
    so repeated annotation names stay separate.

    A CPU child's work is a subset of its parent's work, and sibling launches
    run in launch order within a stream, so the rebuilt spans nest without
    crossings and every step has the same structure. Kernel, memcpy and flow
    events are not modified.

    A lead-in step (``ProfilerStep#N (lead-in, not exported)``, see
    ``_trim_warmup_from_trace``) is placed like a step but is not counted in
    ``step_spans_ns`` or ``step_wall``; its spans carry ``args.lead_in``.

    ``ProfilerStep#N`` itself is kept on ONE stream per rank: the main
    compute stream, i.e. the stream with the most non-collective GPU work time
    (``step_streams="main"``, default). Its span covers the step's compute-
    stream work and every step boundary is visible there; copies on the side
    streams only repeated the step per stream, and those streams run the next
    batch's input work ahead of the compute stream. ``"all"`` keeps a step
    span on every stream the step touched. Other annotations are kept on every
    stream they have work on. Device-wide step rows written by earlier
    versions (``GPU steps (even)/(odd)``) are removed.

    The report's ``step_wall`` section lists each step's main-stream extent
    and flags steps far longer than the window's median, with the longest CPU
    calls and the largest main-stream idle gap inside them, so a stall in the
    capture window (a window-boundary barrier, an eval, a metrics drain) is
    named rather than read as step time.

    Traces without any launch linkage (no correlation ids) are left unchanged.
    Mutates ``events`` in place (replacement and ordering) and returns a
    JSON-serializable report for ``traceRendering``.
    """
    from collections import defaultdict

    if step_streams not in ("main", "all"):
        raise ValueError(f"unknown step_streams {step_streams!r}")

    gpu_pids = {
        e.get("pid") for e in events
        if e.get("cat") in _GPU_KERNEL_CATS or e.get("cat") == _ANNOTATION_CAT
    }
    work_by_corr: Dict[Any, list] = defaultdict(list)
    n_work = 0
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in _GPU_KERNEL_CATS:
            n_work += 1
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                work_by_corr[corr].append(e)

    launches_by_thread: Dict[tuple, list] = defaultdict(list)
    linked_corr = set()
    for e in events:
        if (
            e.get("ph") == "X"
            and e.get("pid") not in gpu_pids
            and e.get("cat") not in ("user_annotation", _ANNOTATION_CAT)
            and "ts" in e
        ):
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None and corr in work_by_corr and corr not in linked_corr:
                linked_corr.add(corr)
                launches_by_thread[(e.get("pid"), e.get("tid"))].append(e)

    old_spans = [
        e for e in events
        if e.get("ph") == "X" and e.get("cat") == _ANNOTATION_CAT
    ]
    report: Dict[str, Any] = {
        "method": "launch_linked",
        "scope": (
            "Display-only. GPU annotation spans are rebuilt from the GPU work "
            "launched inside each CPU annotation (correlation ids). Kernel, "
            "memcpy, memset and flow events are unchanged."
        ),
        "gpu_work_events": n_work,
        "linked_launches": len(linked_corr),
    }
    if not linked_corr:
        report["skipped"] = "no launch linkage (correlation ids) in trace"
        return report

    # Older exports carry synthetic device-wide step rows ("GPU steps
    # (even)/(odd)"). Drop them (slices and their row metadata). Earlier
    # rebuilt gpu_user_annotation spans are replaced below together with
    # Kineto's, so a second rebuild (e.g. the stitcher) is idempotent.
    prior_rows = [
        e for e in events
        if e.get("cat") == _DEVICE_STEP_CAT
        or (
            e.get("ph") == "M"
            and e.get("name") in ("thread_name", "thread_sort_index")
            and isinstance(e.get("tid"), int)
            and e["tid"] >= _DEVICE_STEP_TID_BASE
        )
    ]
    if prior_rows:
        prior_ids = {id(e) for e in prior_rows}
        events[:] = [e for e in events if id(e) not in prior_ids]
    report["prior_device_step_events_removed"] = len(prior_rows)

    cpu_ann_by_thread: Dict[tuple, list] = defaultdict(list)
    for e in events:
        if (
            e.get("ph") == "X"
            and e.get("cat") == "user_annotation"
            and e.get("pid") not in gpu_pids
            and "ts" in e
        ):
            cpu_ann_by_thread[(e.get("pid"), e.get("tid"))].append(e)
    step_threads_by_pid: Dict[Any, list] = defaultdict(list)
    for key, anns in cpu_ann_by_thread.items():
        anns.sort(key=lambda a: (a["ts"], -a.get("dur", 0)))
        if any(_STEP_RE.match(str(a.get("name", ""))) for a in anns):
            step_threads_by_pid[key[0]].append(key)

    def open_at(anns: list, times: list) -> list:
        """Open annotations (outermost first) at each sorted query time."""
        result = []
        stack: list = []
        i = 0
        for t in times:
            while i < len(anns) and anns[i]["ts"] <= t:
                a = anns[i]
                while stack and stack[-1]["ts"] + stack[-1].get("dur", 0) <= a["ts"]:
                    stack.pop()
                stack.append(a)
                i += 1
            result.append(
                [a for a in stack if t < a["ts"] + a.get("dur", 0)]
            )
        return result

    # Ancestor chains per launch: inherited step-thread stack, then own stack.
    chains: list = []
    inherited = 0
    for key, launches in launches_by_thread.items():
        launches.sort(key=lambda e: e["ts"])
        times = [e["ts"] for e in launches]
        own = open_at(cpu_ann_by_thread.get(key, []), times)
        outer = [[] for _ in launches]
        step_threads = step_threads_by_pid.get(key[0], [])
        if key not in step_threads:
            for step_key in step_threads:
                for j, anc in enumerate(open_at(cpu_ann_by_thread[step_key], times)):
                    outer[j].extend(anc)
            inherited += sum(1 for o in outer if o)
        for launch, o, a in zip(launches, outer, own):
            chains.append((launch, o + a))

    # (annotation identity, track) -> [start_ns, start_ts, end_ns, count]
    spans: Dict[tuple, list] = {}
    depth: Dict[int, int] = {}
    ann_by_id: Dict[int, dict] = {}
    # CPU ProfilerStep#N that launched an annotation's work: shown as
    # args.owner_step, so side-stream prefetch for step N that runs under the
    # main stream's ProfilerStep#N-1 still says which step it belongs to.
    step_of: Dict[int, int] = {}
    unannotated = 0
    for launch, chain in chains:
        if not chain:
            unannotated += len(work_by_corr[launch["args"]["correlation"]])
            continue
        step_no = None
        for a in chain:
            m = _STEP_RE.match(str(a.get("name", "")))
            if m:
                step_no = int(m.group(1))
                break
        for position, a in enumerate(chain):
            depth.setdefault(id(a), position)
            ann_by_id[id(a)] = a
            if step_no is not None:
                step_of.setdefault(id(a), step_no)
        for w in work_by_corr[launch["args"]["correlation"]]:
            start = _trace_time_ns(w["ts"])
            end = start + _trace_time_ns(w.get("dur", 0))
            track = (w.get("pid"), w.get("tid"))
            for a in chain:
                key = (id(a), track)
                cur = spans.get(key)
                if cur is None:
                    spans[key] = [start, w["ts"], end, 1]
                else:
                    if start < cur[0]:
                        cur[0], cur[1] = start, w["ts"]
                    if end > cur[2]:
                        cur[2] = end
                    cur[3] += 1

    new_spans = []
    for (aid, track), (start, start_ts, end, count) in spans.items():
        a = ann_by_id[aid]
        args = {"rebuilt_from_launches": True, "work_items": count}
        a_args = a.get("args") or {}
        ext = a_args.get("External id")
        if ext is not None:
            args["External id"] = ext
        if a_args.get("lead_in"):
            # Built from a lead-in CPU annotation (see _trim_warmup_from_trace).
            args["lead_in"] = True
            args["owner_step"] = a_args.get("owner_step")
        elif aid in step_of:
            args["owner_step"] = step_of[aid]
        new_spans.append({
            "ph": "X", "cat": _ANNOTATION_CAT, "name": a.get("name"),
            "pid": track[0], "tid": track[1],
            # Reuse the first work item's ts so both round to the same ns.
            "ts": start_ts, "dur": (end - start) / 1000,
            "args": args,
        })

    cpu_ext = {
        (a.get("args") or {}).get("External id")
        for anns in cpu_ann_by_thread.values() for a in anns
    }
    orphans = sum(
        1 for e in old_spans
        if (e.get("args") or {}).get("External id") not in cpu_ext
    )

    # ProfilerStep#N on the main compute stream only. A step span on every
    # stream it touched repeats the same step 3-4x per rank; side streams
    # (input copies, comm) run ahead into the next step and add no boundary
    # the compute stream does not already show. The main stream is the one
    # with the most non-communication GPU work time on that rank. Collective
    # kernels are excluded: a collective waiting for a late peer is "busy"
    # for the whole wait (1n step 56: a 1.09 s all_to_all on the comm stream
    # of every rank but the late one would otherwise win).
    busy: Dict[tuple, list] = defaultdict(lambda: [0, 0])
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in _GPU_KERNEL_CATS:
            b = busy[(e.get("pid"), e.get("tid"))]
            b[1] += 1
            if not _COMM_KERNEL_RE.search(str(e.get("name", ""))):
                b[0] += _trace_time_ns(e.get("dur", 0))
    step_tracks: Dict[Any, set] = defaultdict(set)
    for span, (aid, track) in zip(new_spans, spans):
        if _STEP_RE.match(str(span["name"])):
            step_tracks[track[0]].add(track)
    main_stream: Dict[Any, Any] = {
        pid: max(
            tracks, key=lambda t: (*busy.get(t, (0, 0)), str(t[1]))
        )
        for pid, tracks in step_tracks.items()
    }
    step_spans_dropped = 0
    if step_streams == "main":
        keep_new = []
        keep_keys = []
        for span, key in zip(new_spans, spans):
            track = key[1]
            if (
                _is_step_span_name(str(span["name"]))
                and main_stream.get(track[0]) != track
            ):
                step_spans_dropped += 1
                continue
            keep_new.append(span)
            keep_keys.append(key)
        new_spans = keep_new
        span_keys = keep_keys
    else:
        span_keys = list(spans)

    spans_by_stream: Dict[str, int] = defaultdict(int)
    step_streams_ns: Dict[str, Dict[str, list]] = defaultdict(dict)
    for span, (aid, track) in zip(new_spans, span_keys):
        spans_by_stream[f"{track[0]}:{track[1]}"] += 1
        if _STEP_RE.match(str(span["name"])):
            start, _, end, _ = spans[(aid, track)]
            step_streams_ns[span["name"]][f"{track[0]}:{track[1]}"] = [start, end]

    old_ids = {id(e) for e in old_spans}
    kept = [e for e in events if id(e) not in old_ids]
    events[:] = kept + new_spans
    new_ids = {
        id(s): depth.get(aid, 0) for s, (aid, _) in zip(new_spans, span_keys)
    }
    # Containers first on equal starts; for equal extents the CPU parent first.
    events.sort(key=lambda e: (
        0 if e.get("ph") == "M" or "ts" not in e else 1,
        _trace_time_ns(e.get("ts", 0)),
        -_trace_time_ns(e.get("dur", 0)) if e.get("ph") == "X" else 0,
        e.get("cat") != _ANNOTATION_CAT,
        new_ids.get(id(e), 0),
    ))

    report.update({
        "kineto_spans_removed": len(old_spans),
        "kineto_orphan_spans_removed": orphans,
        "rebuilt_spans": len(new_spans),
        "spans_by_stream": dict(spans_by_stream),
        "worker_launches_inheriting_step_thread": inherited,
        "work_outside_cpu_annotations": unannotated,
        "work_without_launch": n_work - sum(
            len(work_by_corr[c]) for c in linked_corr
        ),
        "step_streams": step_streams,
        "main_stream_by_pid": {
            str(pid): f"{track[0]}:{track[1]}" for pid, track in main_stream.items()
        },
        "step_spans_on_side_streams_removed": step_spans_dropped,
        "step_spans_ns": dict(step_streams_ns),
        "step_wall": _step_wall_report(events, main_stream),
        "annotation_crossings": len(_gpu_track_crossings(events)),
    })
    return report


def _rebuild_gpu_annotation_spans_from_file(
    path: str,
    step_streams: str = "main",
    capture_notes: Optional[List[dict]] = None,
) -> dict:
    """Apply :func:`rebuild_gpu_annotation_spans` to an exported trace file.

    ``capture_notes`` (from ``Profiler``) lists loop events that fell inside
    the recorded window -- window-boundary barriers, evals, metric drains --
    and is stored next to the rebuild report.
    """
    import json

    with open(path) as source:
        trace = json.load(source)
    report = rebuild_gpu_annotation_spans(
        trace.get("traceEvents", []), step_streams
    )
    rendering = trace.setdefault("traceRendering", {})
    rendering["gpu_annotation_rebuild"] = report
    if capture_notes is not None:
        rendering["capture_window_events"] = list(capture_notes)
    with open(path, "w") as destination:
        json.dump(trace, destination)
    logger.warning(
        "Rebuilt GPU annotation spans in %s from launch linkage: %s spans "
        "(replaced %s), step spans on %s, %s annotation crossing(s)",
        path, report.get("rebuilt_spans", 0),
        report.get("kineto_spans_removed", 0),
        report.get("main_stream_by_pid", {}),
        report.get("annotation_crossings", 0),
    )
    for flag in (report.get("step_wall") or {}).get("extended_steps", []):
        logger.warning("Extended step in %s: %s", path, flag)
    return report


def exported_cpu_step_start(events: list[dict]) -> Optional[float]:
    """Earliest CPU ``ProfilerStep#N`` start (exported steps only), in us.

    Lead-in steps are renamed by the trim and do not match, so this is the
    start of the first EXPORTED CPU step on this rank, or None.
    """
    gpu_pids = {
        e.get("pid") for e in events
        if e.get("cat") in _GPU_KERNEL_CATS or e.get("cat") == _ANNOTATION_CAT
    }
    starts = [
        e["ts"] for e in events
        if e.get("ph") == "X"
        and e.get("cat") == "user_annotation"
        and e.get("pid") not in gpu_pids
        and "ts" in e
        and _STEP_RE.match(str(e.get("name", "")))
    ]
    return min(starts) if starts else None


def _lead_gpu_work(events: list[dict]):
    """``(start_ns, end_ns, ts)`` of lead-in GPU work (kernel/memcpy/memset)."""
    for e in events:
        args = e.get("args")
        if (
            e.get("ph") == "X"
            and e.get("cat") in _GPU_KERNEL_CATS
            and "ts" in e
            and isinstance(args, dict)
            and args.get("lead_in")
        ):
            start = _trace_time_ns(e["ts"])
            yield start, start + _trace_time_ns(e.get("dur", 0)), e["ts"]


def lead_in_trace_start(events: list[dict], cut_us: float) -> Optional[float]:
    """Earliest start of lead-in GPU work that crosses ``cut_us``, in us.

    Such work is running when the exported window opens and is always kept
    whole by :func:`clip_lead_in`. The minimum across ranks is the first
    input of :func:`lead_in_trace_bounds`. None if no lead-in GPU work
    crosses the cut. Unchanged by :func:`clip_lead_in`.
    """
    cut_ns = _trace_time_ns(cut_us)
    best: Optional[Tuple[int, float]] = None
    for start, end, ts in _lead_gpu_work(events):
        if start < cut_ns < end and (best is None or start < best[0]):
            best = (start, ts)
    return best[1] if best is not None else None


def lead_in_gpu_start(events: list[dict], from_us: float) -> Optional[float]:
    """Earliest start of lead-in GPU work ending after ``from_us``, in us.

    It may be before ``from_us`` when an item crosses it. None if no lead-in
    GPU work ends after ``from_us``. See :func:`lead_in_trace_bounds`.
    """
    from_ns = _trace_time_ns(from_us)
    best: Optional[Tuple[int, float]] = None
    for start, end, ts in _lead_gpu_work(events):
        if end > from_ns and (best is None or start < best[0]):
            best = (start, ts)
    return best[1] if best is not None else None


def lead_in_trace_bounds(
    per_rank_events: Iterable[list[dict]], cut_us: float
) -> Tuple[float, float]:
    """``(crossing_start_us, trace_start_us)`` of a stitched lead-in.

    ``cut_us`` is the earliest exported CPU step start across ranks.

    1. ``crossing_start_us``: the earliest start, across ranks, of lead-in
       GPU work crossing the cut (:func:`lead_in_trace_start`); the cut if
       none crosses it. That work is kept whole, so the trace cannot start
       later than this.
    2. ``trace_start_us``: the earliest start, across ranks, of lead-in GPU
       work still running at ``crossing_start_us``
       (:func:`lead_in_gpu_start`), i.e. one rank's item crossing it can
       move the start earlier once. :func:`clip_lead_in` then keeps every
       rank's lead-in GPU work that starts at or after this instant, so no
       rank's stream opens blank while another rank finishes a long kernel.

    Computed in one pass, not iterated to a fixed point: following items
    that cross ``trace_start_us`` back in time cascades through staggered
    ranks (measured on the 1/2/4n MI350X captures: 17+ steps back to 57-113
    ms before the cut, almost half a step, without reducing blank time).
    The one item per stream still running at ``trace_start_us`` is dropped
    whole instead (see :func:`clip_lead_in`). Both values are fixed points
    of a re-stitch: the work that defines them starts at or after
    ``trace_start_us`` and is kept.
    """
    events_list = list(per_rank_events)
    crossing = cut_us
    for events in events_list:
        first = lead_in_trace_start(events, cut_us)
        if first is not None and first < crossing:
            crossing = first
    start = crossing
    for events in events_list:
        first = lead_in_gpu_start(events, crossing)
        if first is not None and first < start:
            start = first
    return crossing, start


def clip_lead_in(
    events: list[dict], cut_us: float, start_us: Optional[float] = None
) -> dict:
    """Clip lead-in events (``args.lead_in``) to the stitched window.

    ``cut_us`` is the start of the exported window: the earliest exported CPU
    step start across ranks. The lead-in exists so the GPU and CPU rows are
    not blank between the top of the trace and the first exported step while
    the GPU still finishes the dropped head step (owner_step N).
    ``start_us`` (never later than ``cut_us``) is where the stitched trace
    begins, on GPU and CPU rows alike. The stitcher passes the
    ``trace_start_us`` of :func:`lead_in_trace_bounds` (one value for all
    ranks); by default it is computed from this file alone the same way.
    No event's ``ts``/``dur`` is changed except display-only annotations
    clipped at ``start_us``.

    GPU side (tracks of kernels, memcpys, memsets, GPU annotations):

    * Lead-in GPU work (kernel, memcpy, memset) that starts at or after
      ``start_us`` is kept unchanged, on every stream, whether it ends
      before or after the cut. So every rank's streams show their head-step
      work from the trace start (blank only where the stream was idle).
    * Lead-in GPU work crossing ``cut_us`` is always kept whole (with the
      stitcher's start it starts at or after ``start_us`` anyway).
    * Other lead-in GPU work starting before ``start_us`` is dropped whole,
      never truncated: work that ended before the trace start, and the one
      item per stream still running at ``start_us`` (keeping it whole would
      move the trace start earlier; see :func:`lead_in_trace_bounds`).
    * Lead-in GPU annotations (display-only) that end at or before
      ``start_us`` are dropped; ones crossing it are shortened to start at
      ``start_us`` (or at an earlier kept item inside them on the same
      track), so they still enclose their children, and are marked
      ``args.clipped_at_window_start``.
    * Other lead-in GPU events before ``start_us`` (instants) are dropped.

    CPU side (every other track; host events of the head step):

    * A lead-in event that ends at or before ``start_us`` is dropped.
    * One that starts at or after ``start_us`` is kept unchanged.
    * A lead-in CPU annotation (``user_annotation``, display-only) crossing
      ``start_us`` is shortened to start at ``start_us`` and marked
      ``args.clipped_at_window_start`` -- unless it is nested in a dropped
      host call (next bullet), when it is dropped with that call.
    * Any other CPU event crossing ``start_us`` (CPU op, runtime call) is
      never truncated: it is dropped together with every event nested in it
      on its thread, so nesting stays valid (no partial overlap, no child
      without its parent) and the trace start stays at ``start_us``. These
      are the innermost host calls running at that instant (a few ms at
      most). Instants at or inside a dropped interval and before
      ``start_us`` are dropped.

    Flows: an ``ac2g`` arrow (id = correlation id) is kept only when both
    its launch and its GPU work are kept; it is dropped when either was
    dropped here. Another flow is dropped with all events of the same
    ``(cat, id)`` when any lead-in endpoint lies where events were dropped.
    No arrow dangles. A kept lead-in GPU annotation whose CPU annotation was
    dropped keeps its label and ``args.lead_in`` but its ``External id``
    moves to ``args.lead_in_external_id``.

    Events without ``args.lead_in`` are never touched, so a trace without a
    lead-in (or a second call with the same arguments) is unchanged. Lead-in
    events keep ``args.lead_in``/``owner_step`` and are never exported steps.
    Mutates ``events`` in place and returns a report.
    """
    from collections import defaultdict

    if start_us is None:
        start_us = lead_in_trace_bounds([events], cut_us)[1]
    start_us = min(start_us, cut_us)
    report: Dict[str, Any] = {
        "cut_us": cut_us, "start_us": start_us, "lead_in_events": 0,
    }

    def is_lead(e: dict) -> bool:
        args = e.get("args")
        return isinstance(args, dict) and bool(args.get("lead_in"))

    lead = [e for e in events if is_lead(e)]
    report["lead_in_events"] = len(lead)
    if not lead:
        return report
    cut_ns = _trace_time_ns(cut_us)
    start_ns = _trace_time_ns(start_us)
    gpu_pids = {
        e.get("pid") for e in events
        if e.get("cat") in _GPU_KERNEL_CATS or e.get("cat") == _ANNOTATION_CAT
    }

    def end_ns(e: dict) -> int:
        return _trace_time_ns(e["ts"]) + _trace_time_ns(e.get("dur", 0))

    def on_gpu(e: dict) -> bool:
        return e.get("pid") in gpu_pids

    drop: set = set()
    # Lead-in work crossing the cut that starts before start_ns (only when
    # the caller's start is later than it), kept whole, per GPU track.
    early_work: Dict[tuple, list] = defaultdict(list)
    # CPU events crossing start_ns that are dropped whole with their nest.
    cpu_holes: Dict[tuple, list] = defaultdict(list)
    kept_whole_before_start = 0
    gpu_dropped_across_start = 0
    for e in lead:
        if e.get("ph") != "X" or "ts" not in e:
            continue
        start, end = _trace_time_ns(e["ts"]), end_ns(e)
        cat = e.get("cat")
        if start >= start_ns:
            continue
        if cat in _GPU_KERNEL_CATS:
            if start < cut_ns < end:
                kept_whole_before_start += 1
                early_work[(e.get("pid"), e.get("tid"))].append(
                    (start, end, e["ts"])
                )
            else:
                drop.add(id(e))
                if end > start_ns:
                    gpu_dropped_across_start += 1
        elif end <= start_ns:
            drop.add(id(e))
        elif cat not in ("user_annotation", _ANNOTATION_CAT):
            drop.add(id(e))
            if not on_gpu(e):
                cpu_holes[(e.get("pid"), e.get("tid"))].append((start, end))

    def in_hole(e: dict) -> bool:
        holes = cpu_holes.get((e.get("pid"), e.get("tid")))
        if not holes or on_gpu(e) or "ts" not in e:
            return False
        s = _trace_time_ns(e["ts"])
        en = end_ns(e) if e.get("ph") == "X" else s
        return any(h0 <= s and en <= h1 for h0, h1 in holes)

    holed = 0
    if cpu_holes:
        for e in lead:
            if e.get("ph") == "X" and id(e) not in drop and in_hole(e):
                drop.add(id(e))
                holed += 1
    dropped_corr = {
        (e.get("args") or {}).get("correlation")
        for e in lead if id(e) in drop
    } - {None}

    clipped = 0
    for e in lead:
        if (
            e.get("ph") != "X"
            or id(e) in drop
            or e.get("cat") not in ("user_annotation", _ANNOTATION_CAT)
        ):
            continue
        start, end = _trace_time_ns(e["ts"]), end_ns(e)
        if on_gpu(e):
            new_ns, new_ts = start_ns, start_us
            for w_start, w_end, w_ts in early_work.get(
                (e.get("pid"), e.get("tid")), []
            ):
                if start <= w_start and w_end <= end and w_start < new_ns:
                    new_ns, new_ts = w_start, w_ts
        else:
            new_ns, new_ts = start_ns, start_us
        if start >= new_ns:
            continue
        e["ts"] = new_ts
        e["dur"] = (end - new_ns) / 1000
        e["args"]["clipped_at_window_start"] = True
        clipped += 1

    def flow_member_dropped(m: dict) -> bool:
        if not is_lead(m) or "ts" not in m:
            return False
        ts = _trace_time_ns(m["ts"])
        return ts < start_ns or in_hole(m)

    # Flows: drop whole (cat, id) groups that would dangle.
    flows: Dict[tuple, list] = defaultdict(list)
    for e in events:
        if e.get("ph") in ("s", "t", "f") and "id" in e:
            flows[(e.get("cat"), e.get("id"))].append(e)
    flow_drop = 0
    for (cat, fid), members in flows.items():
        if cat == "ac2g":
            bad = fid in dropped_corr
        else:
            bad = any(flow_member_dropped(m) for m in members)
        if bad:
            for m in members:
                drop.add(id(m))
            flow_drop += len(members)
    # Remaining non-X lead-in events (instants, counters) outside the window.
    for e in lead:
        if (
            e.get("ph") not in ("X", "s", "t", "f")
            and flow_member_dropped(e)
        ):
            drop.add(id(e))

    n_before = len(events)
    events[:] = [e for e in events if id(e) not in drop]
    # A kept lead-in GPU annotation whose CPU annotation ended before the
    # window has no CPU event in the file any more. Move its External id to
    # ``lead_in_external_id`` so it does not claim a link to an absent event;
    # ``lead_in``/``owner_step`` still say what it is.
    cpu_ext = {
        (e.get("args") or {}).get("External id")
        for e in events if e.get("cat") == "user_annotation"
    }
    unlinked = 0
    for e in events:
        args = e.get("args")
        if (
            e.get("cat") == _ANNOTATION_CAT
            and isinstance(args, dict)
            and args.get("lead_in")
            and "External id" in args
            and args["External id"] not in cpu_ext
        ):
            args["lead_in_external_id"] = args.pop("External id")
            unlinked += 1
    starts = [
        e["ts"] for e in events
        if "ts" in e and e.get("ph") != "M"
    ]
    cpu_starts = [
        e["ts"] for e in events
        if e.get("ph") == "X" and "ts" in e and not on_gpu(e)
    ]
    report.update({
        "dropped_events": n_before - len(events),
        "dropped_flow_events": flow_drop,
        "kept_lead_in_events": sum(1 for e in events if is_lead(e)),
        "kept_cpu_lead_in_events": sum(
            1 for e in events
            if is_lead(e) and e.get("ph") == "X" and not on_gpu(e)
        ),
        "gpu_work_kept_whole_before_start": kept_whole_before_start,
        "gpu_work_dropped_across_start": gpu_dropped_across_start,
        "kept_gpu_lead_in_work": sum(
            1 for e in events
            if is_lead(e) and e.get("ph") == "X"
            and e.get("cat") in _GPU_KERNEL_CATS
        ),
        "annotations_clipped": clipped,
        "cpu_calls_dropped_across_start": sum(
            len(v) for v in cpu_holes.values()
        ),
        "cpu_events_dropped_inside_those_calls": holed,
        "gpu_annotations_without_cpu_annotation": unlinked,
        "earliest_event_us": min(starts) if starts else None,
        "earliest_cpu_event_us": min(cpu_starts) if cpu_starts else None,
    })
    return report


def _deoverlap_gpu_slices(path: str, max_snap_us: float = 5.0) -> None:
    """Remove sub-microsecond kernel overlaps that break Perfetto's renderer.

    Perfetto draws all ``ph=="X"`` slices on a single track (one ``(pid, tid)``)
    as a strict nested stack ordered by start time: a slice that *opens* while a
    previous slice on the same track is still open is treated as that slice's
    child and is **clipped to the parent's end**. ROCm's roctracer reports
    per-stream kernel timestamps at ns granularity, so two back-to-back kernels
    on the same compute stream occasionally overlap by a fraction of a
    microsecond (e.g. an 88 ns ``elementwise`` epilogue ending 0.075 us *after*
    the next 21 ms ``_hstu_attn_bwd`` kernel begins). Perfetto then nests the
    long kernel inside the tiny one and clips it to a sub-pixel sliver, so the
    kernel "disappears" from the timeline even though it is fully present in the
    JSON.

    This pulls each slice's end back to just *before* the next slice's start
    whenever they overlap by less than ``max_snap_us`` (a measurement artifact,
    not real concurrency — kernels on one stream are serialized), leaving genuine
    nesting (a small kernel fully contained in a larger one) untouched. The
    adjustment is sub-microsecond and does not change any reported duration
    meaningfully. Mutates the file in place; best-effort.

    Critically, the slices are separated by a tiny ``_GAP_US`` (~1 ns) rather
    than snapped to an *exactly equal* end==start timestamp. A coincident
    end==start is just as fatal as an overlap in Perfetto: it nests the next
    slice inside the previous one and clips it to zero width (this is the ~1 ns
    gap that roctracer leaves between cleanly-rendered back-to-back kernels). So
    we also fix exact-touch (``a_end == b.ts``) boundaries, not just overlaps.
    """
    import json as _json
    from collections import defaultdict

    # ~1 ns. Matches the natural inter-kernel gap roctracer leaves between
    # back-to-back kernels that Perfetto already renders correctly. Must be
    # strictly > 0 so end != start after the nudge.
    _GAP_US = 0.001

    with open(path) as f:
        d = _json.load(f)
    events = d.get("traceEvents", [])

    tracks: Dict[tuple, list] = defaultdict(list)
    for e in events:
        if (
            e.get("ph") == "X"
            and e.get("cat") in _GPU_KERNEL_CATS
            and e.get("dur", 0) > 0
        ):
            tracks[(e.get("pid"), e.get("tid"))].append(e)

    snapped = 0
    max_clip = 0.0
    for sl in tracks.values():
        # Sort by start, then longest-first so a container precedes the slices
        # it nests; consecutive pairs are then either disjoint, properly nested,
        # or a tiny artifact overlap.
        sl.sort(key=lambda e: (e["ts"], -e["dur"]))
        for i in range(len(sl) - 1):
            a = sl[i]
            b = sl[i + 1]
            a_end = a["ts"] + a["dur"]
            b_end = b["ts"] + b["dur"]
            # Touching (a_end == b.ts) or partial overlap (a ends inside b) both
            # break rendering; true containment (a_end >= b_end) is valid nesting
            # and is left alone.
            if b["ts"] <= a_end < b_end:
                desired_end = b["ts"] - _GAP_US
                clip = a_end - desired_end
                if a["ts"] < desired_end and 0 < clip < max_snap_us:
                    a["dur"] = desired_end - a["ts"]
                    snapped += 1
                    if clip > max_clip:
                        max_clip = clip

    if snapped:
        with open(path, "w") as f:
            _json.dump(d, f)
        logger.warning(
            f"De-overlapped GPU slices in {path}: snapped {snapped} sub-us "
            f"overlap(s) (max {max_clip:.3f}us) so Perfetto renders every kernel"
        )


def _deoverlap_gpu_annotations(path: str, max_snap_us: float = 5.0) -> None:
    """Separate touching/overlapping *sibling* GPU annotations so Perfetto draws
    each one full width (the B200-style stacked layout).

    Same root cause as :func:`_deoverlap_gpu_slices`, but at the annotation
    boundary instead of the kernel boundary. The forward/backward phase
    annotations Kineto projects onto the GPU stream (``## item_forward ##``,
    ``## user_forward ##``, ``## multitask_module ##``, the ``## stu_* ##``
    pairs, ...) are emitted as a chain of siblings laid end-to-end: each is meant
    to end exactly where the next begins. Perfetto stores timestamps as int64 ns,
    and the absolute step timestamps are ~5.4e12 us where a float64's quantum is
    already ~1 ns, so a sibling boundary that should be coincident instead lands
    a few ns off. When the earlier sibling's end falls *at or after* the next
    sibling's start, Perfetto nests the next sibling inside it and clips it to a
    sub-pixel sliver — so e.g. the 100+ ms ``## user_forward ##`` span vanishes on
    some ranks/steps and renders on others purely by rounding luck.

    Unlike kernels (all flat on one stream), annotations form a real nesting
    hierarchy — ``## user_forward ##`` legitimately *contains* the ``## stu_* ##``
    spans and their kernels — so this cannot blindly snap consecutive slices. It
    walks the per-track slice stack (sorted by start, longest-first) and only
    snaps a slice ``a`` back when the next slice ``b`` is **not** contained in it
    (``b`` extends beyond ``a``'s end), i.e. they are siblings rather than
    parent/child. Real containment is left untouched, and a snap is skipped if it
    would clip into ``a``'s own descendants (kernels or child annotations).
    Mutates the file in place; best-effort. Run after :func:`_deoverlap_gpu_slices`
    so kernel boundaries are already clean.
    """
    import json as _json
    from collections import defaultdict

    # ~2 ns. The annotation boundaries sit at ~5.4e12 us where a float64's
    # quantum is ~0.98 ns, so a 1 ns nudge can round back onto the neighbour's
    # timestamp (an exact touch, which Perfetto still nests+clips). 2 ns (~2
    # quanta) reliably separates them and is still far below any visible width.
    _GAP_US = 0.002

    with open(path) as f:
        d = _json.load(f)
    events = d.get("traceEvents", [])

    # Stack the full per-track hierarchy over BOTH kernels and annotations so a
    # parent annotation knows the extent of its descendants (the snap guard),
    # but only annotation slices are ever trimmed.
    _ANN = "gpu_user_annotation"
    tracks: Dict[tuple, list] = defaultdict(list)
    for e in events:
        if (
            e.get("ph") == "X"
            and e.get("dur", 0) > 0
            and (e.get("cat") in _GPU_KERNEL_CATS or e.get("cat") == _ANN)
        ):
            tracks[(e.get("pid"), e.get("tid"))].append(e)

    snapped = 0
    max_clip = 0.0
    for sl in tracks.values():
        # Longest-first on ties so a container precedes the slices it nests.
        sl.sort(key=lambda e: (e["ts"], -e["dur"]))
        # Each frame: [event, max_descendant_end]. The stack holds the chain of
        # currently-open ancestors for the slice being placed.
        stack: list = []
        for b in sl:
            b_ts = b["ts"]
            b_end = b_ts + b["dur"]
            while stack:
                a = stack[-1][0]
                a_end = a["ts"] + a["dur"]
                if a_end < b_ts:
                    # a closed strictly before b begins -> disjoint sibling, pop.
                    frame = stack.pop()
                    eff = frame[0]["ts"] + frame[0]["dur"]
                    if stack:
                        stack[-1][1] = max(stack[-1][1], eff, frame[1])
                    continue
                if a_end < b_end:
                    # b starts at/inside a but extends past a's end => they are
                    # siblings (not parent/child), and a's tail nests+clips b in
                    # Perfetto. Snap a's end to just before b. This fires for both
                    # annotation tails (## item_forward ## overhanging
                    # ## user_forward ##) and kernel tails that straddle an
                    # annotation boundary (a layer-norm kernel ending a few ns
                    # past the start of the next phase span) -- both are sub-us
                    # roctracer/rounding artifacts, since kernels on one stream
                    # are serialized and phase spans are sequential.
                    desired_end = b_ts - _GAP_US
                    clip = a_end - desired_end
                    # Guard: only snap when a's deepest descendant ends at or
                    # before b's start. If a child (kernel or nested span)
                    # actually extends *past* b.ts, trimming a wouldn't fix b's
                    # clipping (the child would still nest b) and could drop a
                    # real child into b's territory, so leave it. A descendant
                    # ending exactly at the boundary is itself rounding noise and
                    # is clipped by <=1 ns, which is fine.
                    if (
                        a["ts"] < desired_end
                        and stack[-1][1] <= b_ts
                        and 0 < clip < max_snap_us
                    ):
                        a["dur"] = desired_end - a["ts"]
                        snapped += 1
                        if clip > max_clip:
                            max_clip = clip
                    frame = stack.pop()
                    eff = frame[0]["ts"] + frame[0]["dur"]
                    if stack:
                        stack[-1][1] = max(stack[-1][1], eff, frame[1])
                    continue
                # a_end >= b_end: a fully contains b -> b is a child, stop.
                break
            stack.append([b, b_ts])

    if snapped:
        with open(path, "w") as f:
            _json.dump(d, f)
        logger.warning(
            f"De-overlapped GPU annotations in {path}: snapped {snapped} sub-us "
            f"sibling overlap(s) (max {max_clip:.3f}us) so Perfetto renders every "
            f"annotation full width"
        )
