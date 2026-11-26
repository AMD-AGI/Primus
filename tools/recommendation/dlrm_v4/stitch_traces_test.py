#!/usr/bin/env python3
"""CPU-only regression tests for trace display boundaries and CLI defaults."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "primus/backends/recommendation/dlrm_v4/src"))
from generative_recommenders.dlrm_v4 import trace_postprocessing

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stitch_traces  # noqa: E402


def _slice(name, ts, dur, cat="kernel", pid=0, tid=4):
    return {
        "ph": "X",
        "name": name,
        "cat": cat,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "dur": dur,
    }


def _ns(value_us):
    # Perfetto rounds the timestamp and duration independently to nearest ns,
    # with half values away from zero. Do not round their floating-point sum.
    return int(Decimal(value_us * 1000).to_integral_value(rounding=ROUND_HALF_UP))


def _end_ns(event):
    return _ns(event["ts"]) + _ns(event["dur"])


def _metadata():
    return [
        {"ph": "M", "name": "process_labels", "pid": 374, "tid": 0, "args": {"labels": "CPU"}},
        {"ph": "M", "name": "process_labels", "pid": 0, "tid": 0, "args": {"labels": "GPU 0"}},
    ]


_CPU_PID, _MAIN_TID, _AUTOGRAD_TID = 374, 374, 375
_COMPUTE, _SIDE = 4, 1


def _synthetic_steps(host_ahead, steps=range(52, 56), stall=None):
    """Per-rank trace with linked launches for consecutive training steps.

    Each CPU ``ProfilerStep#N`` launches the next batch's input copy on a side
    stream, two forward kernels inside nested annotations, and a backward
    kernel from the autograd thread (whose own annotation has no step
    ancestor on that thread). With ``host_ahead`` the GPU lags the host by
    60 us of a 100 us step, so step N's kernels run during CPU step N+1 and the
    next step's side-stream copy runs while the compute stream still works on
    step N, as under the host-sync-free step. Without it the host waits for the
    GPU every step, so all GPU work sits inside its CPU step.

    ``stall=(step, us)`` makes that step a window-boundary step: a
    ``c10d::barrier`` CPU call whose barrier kernel runs on the compute stream
    first, then a host stall of ``us`` before the rest of the step.
    """
    events = _metadata()
    corr = [1000]
    ext = [5000]

    def annotate(name, ts, dur, tid=_MAIN_TID):
        ext[0] += 1
        event = _slice(name, ts, dur, "user_annotation", _CPU_PID, tid)
        event["args"] = {"External id": ext[0]}
        events.append(event)
        return event

    def launch(name, cpu_ts, tid, gpu_ts, gpu_dur, stream):
        corr[0] += 1
        call = _slice("hipLaunchKernel", cpu_ts, 2.0, "cuda_runtime", _CPU_PID, tid)
        call["args"] = {"correlation": corr[0]}
        kernel = _slice(name, gpu_ts, gpu_dur, "kernel", 0, stream)
        kernel["args"] = {"correlation": corr[0], "stream": stream}
        events.extend(
            [
                call,
                kernel,
                {
                    "ph": "s",
                    "id": corr[0],
                    "pid": _CPU_PID,
                    "tid": tid,
                    "ts": cpu_ts,
                    "cat": "ac2g",
                    "name": "ac2g",
                },
                {
                    "ph": "f",
                    "id": corr[0],
                    "pid": 0,
                    "tid": stream,
                    "ts": gpu_ts,
                    "cat": "ac2g",
                    "name": "ac2g",
                    "bp": "e",
                },
            ]
        )

    lag = 60.0 if host_ahead else 0.0
    shift = 0.0
    for step in steps:
        c0 = 1000.0 + (step - 52) * 100.0 + shift
        g0 = c0 + lag
        extra = stall[1] if stall and stall[0] == step else 0.0
        step_ann = annotate(f"ProfilerStep#{step}", c0, 99.0 + extra)
        # Kineto-style slivers that the rebuild must replace.
        sliver = _slice(f"ProfilerStep#{step}", g0 + 90.0, 1.0, "gpu_user_annotation")
        sliver["args"] = {"External id": step_ann["args"]["External id"]}
        events.append(sliver)
        if extra:
            events.append(_slice("c10d::barrier", c0 + 0.5, extra, "cpu_op", _CPU_PID, _MAIN_TID))
            launch("all_reduce_barrier", c0 + 0.6, _MAIN_TID, g0 + 0.6, 0.5, _COMPUTE)
            c0 += extra
            g0 += extra
            shift += extra
        annotate("## input_dist ##", c0 + 1.0, 4.0)
        # Next batch's copy: on the side stream, runs right after launch.
        side_ts = c0 + 5.0 if host_ahead else c0 + 2.0
        launch("copy_next_batch", c0 + 2.0, _MAIN_TID, side_ts, 8.0, _SIDE)
        annotate(f"## forward {step} ##", c0 + 10.0, 40.0)
        annotate("## user_forward ##", c0 + 12.0, 30.0)
        launch("fwd_a", c0 + 14.0, _MAIN_TID, g0 + 15.0, 20.0, _COMPUTE)
        launch("fwd_b", c0 + 30.0, _MAIN_TID, g0 + 36.0, 14.0, _COMPUTE)
        annotate(f"## backward {step} ##", c0 + 55.0, 40.0)
        annotate("## autograd_bwd ##", c0 + 60.0, 20.0, _AUTOGRAD_TID)
        launch("bwd", c0 + 62.0, _AUTOGRAD_TID, g0 + 55.0, 30.0, _COMPUTE)
        launch("optimizer", c0 + 90.0, _MAIN_TID, g0 + 88.0, 7.0, _COMPUTE)
    return events


def _launch_ranges(events, step_streams="main"):
    """(CPU annotation External id, stream) -> exact ns range of its work.

    With ``step_streams="main"`` the ProfilerStep ranges are kept on the
    compute stream only, as the rebuild keeps them.
    """
    calls = {e["args"]["correlation"]: e for e in events if e.get("cat") == "cuda_runtime"}
    anns = [e for e in events if e.get("cat") == "user_annotation"]
    steps = [a for a in anns if a["tid"] == _MAIN_TID and a["name"].startswith("ProfilerStep")]
    ranges = {}
    for kernel in (e for e in events if e.get("cat") == "kernel"):
        call = calls[kernel["args"]["correlation"]]
        t = call["ts"]
        owners = [a for a in anns if a["tid"] == call["tid"] and a["ts"] <= t < a["ts"] + a["dur"]]
        if call["tid"] != _MAIN_TID:
            owners += [a for a in anns if a["tid"] == _MAIN_TID and a["ts"] <= t < a["ts"] + a["dur"]]
        for a in owners:
            key = (a["args"]["External id"], kernel["tid"])
            lo, hi = ranges.get(key, (None, None))
            start, end = _ns(kernel["ts"]), _end_ns(kernel)
            ranges[key] = (start if lo is None else min(lo, start), end if hi is None else max(hi, end))
    if step_streams == "main":
        step_ext = {a["args"]["External id"] for a in steps}
        ranges = {k: v for k, v in ranges.items() if k[0] not in step_ext or k[1] == _COMPUTE}
    return ranges, steps


def _step_structure(events, stream):
    """Per GPU step on ``stream``: (depth, name) of everything nested in it."""
    import re

    gpu = sorted(
        (
            e
            for e in events
            if e.get("ph") == "X"
            and e.get("pid") == 0
            and e.get("tid") == stream
            and e.get("cat") in {"kernel", "gpu_user_annotation"}
        ),
        key=lambda e: (_ns(e["ts"]), -_ns(e["dur"])),
    )
    result = []
    for step in (
        e for e in gpu if e["name"].startswith("ProfilerStep") and e["cat"] == "gpu_user_annotation"
    ):
        lo, hi = _ns(step["ts"]), _end_ns(step)
        stack, layout = [], []
        for e in gpu:
            if e is step or not (lo <= _ns(e["ts"]) and _end_ns(e) <= hi):
                continue
            while stack and stack[-1] <= _ns(e["ts"]):
                stack.pop()
            layout.append((len(stack), re.sub(r"\d+", "N", e["name"])))
            stack.append(_end_ns(e))
        result.append(tuple(layout))
    return result


def _assert_steps_on_compute_stream_only(test, events, steps):
    """One GPU ProfilerStep per step, on the compute stream; no step rows."""
    from collections import Counter

    gpu_steps = [
        e for e in events if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("ProfilerStep")
    ]
    test.assertEqual({(e["pid"], e["tid"]) for e in gpu_steps}, {(0, _COMPUTE)})
    test.assertEqual(Counter(e["name"] for e in gpu_steps), Counter(f"ProfilerStep#{n}" for n in steps))
    test.assertFalse(any(e.get("cat") == "gpu_step" for e in events))
    test.assertFalse(
        any(
            e.get("ph") == "M"
            and isinstance(e.get("tid"), int)
            and e["tid"] >= trace_postprocessing._DEVICE_STEP_TID_BASE
            for e in events
        )
    )


class AnnotationRebuildTest(unittest.TestCase):
    def _check_exact(self, host_ahead, step_streams="main"):
        events = _synthetic_steps(host_ahead)

        def payload(evs):
            # Order may change (containers first); content must not.
            return sorted(
                json.dumps(e, sort_keys=True)
                for e in evs
                if e.get("cat") in {"kernel", "ac2g", "cuda_runtime"}
            )

        work = payload(events)
        expected, cpu_steps = _launch_ranges(events, step_streams)

        report = trace_postprocessing.rebuild_gpu_annotation_spans(events, step_streams)

        # Work, launch and flow events are untouched.
        self.assertEqual(payload(events), work)
        spans = [e for e in events if e.get("cat") == "gpu_user_annotation"]
        self.assertTrue(all(s["args"].get("rebuilt_from_launches") for s in spans))
        got = {(s["args"]["External id"], s["tid"]): (_ns(s["ts"]), _end_ns(s)) for s in spans}
        self.assertEqual(got, expected)
        self.assertEqual(len(spans), len(expected))
        # Every step owns its own forward, attention-like and backward work.
        self.assertEqual(
            trace_postprocessing._gpu_track_crossings(events, frozenset({"kernel", "gpu_user_annotation"})),
            [],
        )
        self.assertEqual(report["annotation_crossings"], 0)
        layouts = _step_structure(events, _COMPUTE)
        self.assertEqual(len(layouts), len(cpu_steps))
        self.assertEqual(len(set(layouts)), 1, layouts)
        self.assertEqual(
            [n for _, n in layouts[0]],
            [
                "## forward N ##",
                "## user_forward ##",
                "fwd_a",
                "fwd_b",
                "## backward N ##",
                "## autograd_bwd ##",
                "bwd",
                "optimizer",
            ],
        )
        return events, report, cpu_steps

    def test_host_ahead_step_spans_only_on_compute_stream(self):
        events, report, _ = self._check_exact(host_ahead=True)

        self.assertEqual(report["kineto_spans_removed"], 4)
        self.assertGreater(report["worker_launches_inheriting_step_thread"], 0)
        self.assertEqual(report["main_stream_by_pid"], {"0": f"0:{_COMPUTE}"})
        # The side-stream copy of each step is gone.
        self.assertEqual(report["step_spans_on_side_streams_removed"], 4)
        _assert_steps_on_compute_stream_only(self, events, range(52, 56))
        # The compute-stream step spans tile without overlap.
        compute = sorted(
            (
                e
                for e in events
                if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("ProfilerStep")
            ),
            key=lambda e: e["ts"],
        )
        for a, b in zip(compute, compute[1:]):
            self.assertLessEqual(_end_ns(a), _ns(b["ts"]))
        self.assertEqual(report["step_wall"]["extended_steps"], [])

    def test_step_streams_all_keeps_a_step_span_per_stream(self):
        events, report, _ = self._check_exact(host_ahead=True, step_streams="all")

        self.assertEqual(report["step_spans_on_side_streams_removed"], 0)
        streams = {
            e["tid"]
            for e in events
            if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("ProfilerStep")
        }
        self.assertEqual(streams, {_COMPUTE, _SIDE})

    def test_non_overlapping_steps_stay_inside_cpu_steps(self):
        events, report, cpu_steps = self._check_exact(host_ahead=False)

        _assert_steps_on_compute_stream_only(self, events, range(52, 56))
        windows = {s["name"]: (_ns(s["ts"]), _end_ns(s)) for s in cpu_steps}
        for span in (
            e
            for e in events
            if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("ProfilerStep")
        ):
            lo, hi = windows[span["name"]]
            self.assertLessEqual(lo, _ns(span["ts"]))
            self.assertLessEqual(_end_ns(span), hi)

    def test_rebuild_twice_is_idempotent(self):
        events = _synthetic_steps(host_ahead=True)
        first = trace_postprocessing.rebuild_gpu_annotation_spans(events)
        snapshot = sorted(json.dumps(e, sort_keys=True) for e in events)
        self.assertEqual(first["prior_device_step_events_removed"], 0)

        second = trace_postprocessing.rebuild_gpu_annotation_spans(events)

        self.assertEqual(second["prior_device_step_events_removed"], 0)
        self.assertEqual(second["rebuilt_spans"], first["rebuilt_spans"])
        self.assertEqual(second["kineto_spans_removed"], first["rebuilt_spans"])
        self.assertEqual(sorted(json.dumps(e, sort_keys=True) for e in events), snapshot)
        _assert_steps_on_compute_stream_only(self, events, range(52, 56))

    def test_waiting_collective_does_not_pick_the_comm_stream(self):
        # A collective waiting for a late peer is "busy" for the whole wait;
        # the compute stream must still be chosen.
        events = _synthetic_steps(host_ahead=True)
        events.append(
            {
                "ph": "X",
                "name": "hipLaunchKernel",
                "cat": "cuda_runtime",
                "pid": _CPU_PID,
                "tid": _MAIN_TID,
                "ts": 1003.0,
                "dur": 1.0,
                "args": {"correlation": 99},
            }
        )
        events.append(
            {
                "ph": "X",
                "name": "ncclDevKernel_Generic_1",
                "cat": "kernel",
                "pid": 0,
                "tid": 7,
                "ts": 1004.0,
                "dur": 5000.0,
                "args": {"correlation": 99},
            }
        )

        report = trace_postprocessing.rebuild_gpu_annotation_spans(events)

        self.assertEqual(report["main_stream_by_pid"], {"0": f"0:{_COMPUTE}"})

    def test_legacy_even_odd_step_rows_are_removed(self):
        events = _synthetic_steps(host_ahead=True)
        base = trace_postprocessing._DEVICE_STEP_TID_BASE
        for row, label in enumerate(("GPU steps (even)", "GPU steps (odd)")):
            events.append(
                {"ph": "M", "name": "thread_name", "pid": 0, "tid": base + row, "args": {"name": label}}
            )
            events.append(
                {
                    "ph": "M",
                    "name": "thread_sort_index",
                    "pid": 0,
                    "tid": base + row,
                    "args": {"sort_index": -100 + row},
                }
            )
        for step in range(52, 56):
            events.append(_slice(f"ProfilerStep#{step}", 1000.0 + step, 50.0, "gpu_step", 0, base + step % 2))

        report = trace_postprocessing.rebuild_gpu_annotation_spans(events)

        self.assertEqual(report["prior_device_step_events_removed"], 8)
        _assert_steps_on_compute_stream_only(self, events, range(52, 56))

    def test_extended_step_is_flagged_with_its_stall(self):
        events = _synthetic_steps(host_ahead=True, steps=range(52, 57), stall=(56, 50_000.0))

        report = trace_postprocessing.rebuild_gpu_annotation_spans(events)

        wall = report["step_wall"]
        self.assertEqual(sorted(wall["main_stream_step_ms"]["0"]), [str(n) for n in range(52, 57)])
        (flag,) = wall["extended_steps"]
        self.assertEqual(flag["step"], 56)
        self.assertGreater(flag["main_stream_ms"], 50.0)
        self.assertLess(flag["median_ms"], 0.1)
        self.assertGreater(flag["largest_main_stream_gap_ms"], 49.9)
        self.assertEqual(flag["longest_cpu_calls"][0]["name"], "c10d::barrier")
        self.assertAlmostEqual(flag["longest_cpu_calls"][0]["ms"], 50.0)
        self.assertEqual(flag["largest_host_self_time"][0]["name"], "c10d::barrier")
        self.assertIn("span", flag["flagged_by"])

    def test_host_self_time_names_the_innermost_stalled_call(self):
        # A long op with no child call (a host stall inside it) outranks its
        # enclosing annotations and an earlier, shorter barrier.
        events = [
            _slice("## forward 56 ##", 0.0, 1200.0, "user_annotation", 9, 1),
            _slice("hstu_preprocess", 10.0, 1100.0, "cpu_op", 9, 1),
            _slice("hipLaunchKernel", 1105.0, 2.0, "cuda_runtime", 9, 1),
            _slice("c10d::barrier", 1300.0, 200.0, "cpu_op", 9, 1),
            _slice("hipStreamSynchronize", 1300.5, 199.0, "cuda_runtime", 9, 1),
            _slice("## splits poll ##", 0.0, 1000.0, "user_annotation", 9, 2),
        ]
        ranked = trace_postprocessing._host_self_time(events, 9, 0.0, 2000.0)
        self.assertEqual(ranked[0][1]["name"], "hstu_preprocess")
        self.assertAlmostEqual(ranked[0][0], 1098.0)
        self.assertEqual(ranked[1][1]["name"], "## splits poll ##")
        self.assertAlmostEqual(dict((e["name"], t) for t, e in ranked)["c10d::barrier"], 1.0)

    def test_stall_before_first_launch_is_flagged_by_period(self):
        # Synced steps; the host stalls 50 ms after step 54's last launch, so
        # step 54's main-stream span is normal but its period is not.
        events = _synthetic_steps(host_ahead=False, steps=range(52, 58))
        cut = 1000.0 + (55 - 52) * 100.0
        for e in events:
            if e.get("ts", 0) >= cut:
                e["ts"] += 50_000.0
            elif e.get("name") == "ProfilerStep#54" and e.get("cat") == "user_annotation":
                e["dur"] += 50_000.0

        report = trace_postprocessing.rebuild_gpu_annotation_spans(events)

        (flag,) = report["step_wall"]["extended_steps"]
        self.assertEqual(flag["step"], 54)
        self.assertEqual(flag["flagged_by"], ["period"])
        self.assertGreater(flag["main_stream_period_ms"], 50.0)
        self.assertLess(flag["main_stream_ms"], 0.2)

    def test_parent_end_encloses_independently_rounded_child_end(self):
        events = [
            _slice("ProfilerStep#52", 0.0, 10.0, "user_annotation", 374, 374),
            _slice("first", 1.0004, 0.1),
            _slice("last", 2.0006, 1.0006),
        ]
        for corr, (cpu_ts, kernel) in enumerate(zip((1.0, 2.0), events[1:]), start=1):
            kernel["args"] = {"correlation": corr}
            call = _slice("hipLaunchKernel", cpu_ts, 0.1, "cuda_runtime", 374, 374)
            call["args"] = {"correlation": corr}
            events.append(call)
        # Subtracting float extrema yields a parent ending at 3001 ns,
        # although the last child's imported end is 3002 ns.
        naive_dur = 2.0006 + 1.0006 - 1.0004
        self.assertEqual(_ns(1.0004) + _ns(naive_dur), 3001)

        trace_postprocessing.rebuild_gpu_annotation_spans(events)

        (step,) = [e for e in events if e.get("cat") == "gpu_user_annotation"]
        self.assertEqual(_ns(step["ts"]), _ns(1.0004))
        self.assertEqual(_end_ns(step), 3002)

    def test_trace_without_launch_linkage_is_unchanged(self):
        events = _metadata() + [
            _slice("ProfilerStep#52", 0.0, 100.0, "user_annotation", 374, 374),
            _slice("ProfilerStep#52", 90.0, 1.0, "gpu_user_annotation"),
            _slice("kernel", 10.0, 5.0),
        ]
        original = copy.deepcopy(events)

        report = trace_postprocessing.rebuild_gpu_annotation_spans(events)

        self.assertEqual(events, original)
        self.assertIn("skipped", report)


class NestingRepairTest(unittest.TestCase):
    def test_annotation_one_ns_short_extends_to_preserve_child(self):
        parent = _slice("phase", 10.0, 10.0, "gpu_user_annotation")
        child = _slice("kernel", 12.0, 8.001)
        original_child = copy.deepcopy(child)

        report = trace_postprocessing._repair_gpu_nesting([parent, child])

        self.assertGreaterEqual(_end_ns(parent), _end_ns(child))
        self.assertEqual(child, original_child)
        self.assertEqual(report["summary"]["changed_events"], 1)
        self.assertEqual(report["summary"]["unresolved_crossings"], 0)

    def test_exact_touching_siblings_remain_unchanged(self):
        events = [
            _slice("forward", 10.0, 5.0, "gpu_user_annotation"),
            _slice("first kernel", 11.0, 3.0),
            _slice("backward", 15.0, 10.0, "gpu_user_annotation"),
            _slice("second kernel", 16.0, 8.0),
        ]
        original = copy.deepcopy(events)

        report = trace_postprocessing._repair_gpu_nesting(events)

        self.assertEqual(events, original)
        self.assertEqual(report["summary"]["changed_events"], 0)
        self.assertEqual(report["summary"]["unresolved_crossings"], 0)

    def test_large_crossing_is_reported_without_modification(self):
        events = [_slice("a", 10.0, 20.0), _slice("b", 20.0, 40.0)]
        original = copy.deepcopy(events)

        report = trace_postprocessing._repair_gpu_nesting(events, max_snap_us=5.0)

        self.assertEqual(events, original)
        self.assertEqual(report["summary"]["changed_events"], 0)
        self.assertGreater(report["summary"]["unresolved_crossings"], 0)
        self.assertTrue(report["unresolved"])

    def test_kernel_boundary_repair_preserves_launch_identity_and_payload(self):
        events = [_slice("a", 10.0, 5.002), _slice("b", 15.0, 4.0)]
        for index, event in enumerate(events):
            event["id"] = 900 + index
            event["args"] = {
                "External id": 100 + index,
                "correlation": 200 + index,
                "device": 0,
                "stream": 4,
                "shape": [32, 64],
            }
        original = copy.deepcopy(events)

        report = trace_postprocessing._repair_gpu_nesting(events)

        self.assertLessEqual(_end_ns(events[0]), _ns(events[1]["ts"]))
        self.assertEqual(report["summary"]["changed_events"], 1)
        for before, after in zip(original, events):
            self.assertEqual(
                {k: v for k, v in before.items() if k != "dur"},
                {k: v for k, v in after.items() if k != "dur"},
            )
        self.assertEqual(events[1], original[1])


class StitchCliTest(unittest.TestCase):
    def _run_cli(self, directory, *options):
        output = directory / "stitched.json"
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(stitch_traces.__file__)),
                str(directory),
                "--step",
                "52",
                "--out",
                str(output),
                *options,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(output.read_text())

    def test_cli_preserves_profiler_step_selection_and_inputs(self):
        events = _metadata()
        for step in range(52, 59):
            start = 1000.0 + (step - 52) * 100.0
            name = f"ProfilerStep#{step}"
            events.extend(
                [
                    _slice(name, start, 100.0, "user_annotation", 374, 374),
                    _slice(name, start + 80.0, 10.0, "gpu_user_annotation"),
                    _slice(f"kernel_step{step}", start + 10.0, 20.0),
                ]
            )
        trace = {"hip_runtime_version": 60700, "traceEvents": events}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw = directory / "original.raw.json"
            raw.write_text(json.dumps(trace))
            original = raw.read_bytes()
            source = directory / "trace_step52_rank0.json"
            source.write_bytes(original)
            # Step selection belongs to the profiler's active/head policy;
            # stitching must preserve the selection it receives.
            trace_postprocessing._trim_warmup_from_trace(str(source), 7, 5)
            selected = source.read_bytes()

            output = self._run_cli(directory)

            for category in ("user_annotation", "gpu_user_annotation"):
                names = {e["name"] for e in output["traceEvents"] if e.get("cat") == category}
                self.assertEqual(names, {f"ProfilerStep#{n}" for n in range(52, 57)})
            kernels = [e for e in output["traceEvents"] if e.get("cat") == "kernel"]
            self.assertEqual(len(kernels), 5)
            self.assertEqual(source.read_bytes(), selected)
            self.assertEqual(raw.read_bytes(), original)

            source.write_bytes(original)
            output = self._run_cli(directory)

            names = {e["name"] for e in output["traceEvents"] if e.get("cat") == "user_annotation"}
            self.assertEqual(names, {f"ProfilerStep#{n}" for n in range(52, 59)})
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(raw.read_bytes(), original)

    def test_trim_skips_head_and_drops_next_step_gpu_annotations(self):
        # Recorded 51-58 (warmup 50), skip_head=1, keep 5 -> export 52-56.
        events = _metadata()
        ext = 7000
        for step in range(50, 59):
            start = 1000.0 + (step - 50) * 100.0
            ext += 1
            events.append(_slice(f"ProfilerStep#{step}", start, 100.0, "user_annotation", 374, 374))
            fwd = _slice(f"## forward {step} ##", start + 5.0, 40.0, "user_annotation", 374, 374)
            fwd["args"] = {"External id": ext}
            events.append(fwd)
            # Host-ahead projection: the GPU copy of step N's annotation runs
            # during CPU step N+1.
            gpu = _slice(f"## forward {step} ##", start + 105.0, 40.0, "gpu_user_annotation")
            gpu["args"] = {"External id": ext}
            events.append(gpu)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace_step52_rank0.json"
            path.write_text(json.dumps({"traceEvents": events}))

            trace_postprocessing._trim_warmup_from_trace(str(path), 8, 5, skip_head=1)

            kept = json.loads(path.read_text())["traceEvents"]
        steps = {
            e["name"]
            for e in kept
            if e.get("cat") == "user_annotation" and e["name"].startswith("ProfilerStep")
        }
        self.assertEqual(
            steps, {f"ProfilerStep#{n}" for n in range(52, 57)} | {"ProfilerStep#51 (lead-in, not exported)"}
        )
        # GPU annotations follow their CPU annotation: 52-56 kept, including
        # 56's, which runs after the CPU window; 57's (inside the GPU range of
        # the kept steps) is dropped and 51's is kept only as the lead-in.
        gpu = sorted(
            e["name"]
            for e in kept
            if e.get("cat") == "gpu_user_annotation" and not (e.get("args") or {}).get("lead_in")
        )
        self.assertEqual(gpu, [f"## forward {n} ##" for n in range(52, 57)])
        lead = [
            e["name"]
            for e in kept
            if e.get("cat") == "gpu_user_annotation" and (e.get("args") or {}).get("lead_in")
        ]
        self.assertEqual(lead, ["## forward 51 ##"])

    def _boundary_window(self, last=62):
        """Warmup 50, recorded 51-``last``, boundary barrier in step 56."""
        events = _metadata()
        for step in range(50, last + 1):
            start = 1000.0 + (step - 50) * 100.0
            events.append(_slice(f"ProfilerStep#{step}", start, 100.0, "user_annotation", 374, 374))
            events.append(_slice(f"## backward {step} ##", start + 50.0, 40.0, "user_annotation", 374, 374))
            if step == 56:
                events.append(_slice("## window_boundary ##", start + 1.0, 5.0, "user_annotation", 374, 374))
        return events

    def _trim(self, events, *args, **kwargs):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace_step52_rank0.json"
            path.write_text(json.dumps({"traceEvents": events}))
            capture = trace_postprocessing._trim_warmup_from_trace(str(path), *args, **kwargs)
            stored = json.loads(path.read_text())["traceCapture"]
        self.assertEqual(capture, stored)
        return capture

    def test_trim_reports_boundary_step_and_keeps_it_by_default(self):
        capture = self._trim(self._boundary_window(), 12, 5, skip_head=1)
        self.assertEqual(capture["exported_steps"], [52, 53, 54, 55, 56])
        self.assertEqual(capture["barrier_steps"], [56])
        self.assertFalse(capture["moved_past_boundary"])
        # No frequency given (the training callback): not reported as empty.
        self.assertIsNone(capture["metric_log_steps"])

    def test_trim_skip_boundary_steps_moves_past_the_boundary(self):
        capture = self._trim(self._boundary_window(), 12, 5, skip_head=1, skip_boundary_steps=True)
        self.assertEqual(capture["exported_steps"], [57, 58, 59, 60, 61])
        self.assertTrue(capture["moved_past_boundary"])

    def test_trim_skip_boundary_steps_cannot_move_a_short_window(self):
        capture = self._trim(self._boundary_window(last=58), 8, 5, skip_head=1, skip_boundary_steps=True)
        self.assertEqual(capture["exported_steps"], [52, 53, 54, 55, 56])
        self.assertFalse(capture["moved_past_boundary"])
        self.assertTrue(any("cannot move" in w for w in capture["warnings"]))

    def test_cli_rebuilds_host_ahead_rank_and_records_rendering(self):
        events = _synthetic_steps(host_ahead=True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "trace_step52_rank0.json"
            source.write_text(
                json.dumps(
                    {
                        "hip_runtime_version": 60700,
                        "traceCapture": {"exported_steps": [52]},
                        "traceEvents": events,
                    }
                )
            )
            original = source.read_bytes()

            output = self._run_cli(directory)

            self.assertEqual(source.read_bytes(), original)
            rendering = output["traceRendering"]
            self.assertNotIn("traceCapture", output)
            self.assertEqual(rendering["trace_capture_by_rank"], {"0": {"exported_steps": [52]}})
            rebuild = rendering["gpu_annotation_rebuild_by_rank"]["0"]
            self.assertEqual(rebuild["annotation_crossings"], 0)
            self.assertEqual(rebuild["main_stream_by_pid"], {"0": f"0:{_COMPUTE}"})
            self.assertEqual(rebuild["step_spans_on_side_streams_removed"], 4)
            nesting = rendering["gpu_nesting_by_rank"]["0"]["summary"]
            self.assertEqual(nesting["unresolved_crossings"], 0)
            self.assertEqual(nesting["boundary_adjustments"], 0)
            kernels = [e for e in output["traceEvents"] if e.get("cat") == "kernel"]
            self.assertEqual(
                sorted((e["name"], e["ts"], e["dur"]) for e in kernels),
                sorted((e["name"], e["ts"], e["dur"]) for e in events if e.get("cat") == "kernel"),
            )

    def test_export_then_stitch_keeps_steps_on_compute_stream(self):
        events = _synthetic_steps(host_ahead=True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "trace_step52_rank0.json"
            source.write_text(json.dumps({"hip_runtime_version": 60700, "traceEvents": events}))
            # Training export path: rebuild, then the ROCm nesting repair.
            trace_postprocessing._rebuild_gpu_annotation_spans_from_file(str(source))
            trace_postprocessing._repair_gpu_nesting_from_file(str(source))
            exported = json.loads(source.read_text())["traceEvents"]
            _assert_steps_on_compute_stream_only(self, exported, range(52, 56))
            exported_bytes = source.read_bytes()

            out = directory / "out" / "stitched.json"
            stitch_traces.stitch_one(
                {0: source},
                out,
                include={"cpu", "gpu", "other"},
                gzip_out=False,
                verbose=False,
                rocm_layout=True,
            )
            output = json.loads(out.read_text())

            self.assertEqual(source.read_bytes(), exported_bytes)
            _assert_steps_on_compute_stream_only(self, output["traceEvents"], range(52, 56))
            rebuild = output["traceRendering"]["gpu_annotation_rebuild_by_rank"]["0"]
            self.assertEqual(rebuild["prior_device_step_events_removed"], 0)
            self.assertEqual(rebuild["annotation_crossings"], 0)

    def test_default_cuda_stitch_preserves_nonstep_gpu_events(self):
        gpu = [
            _slice("a", 10.0, 5.002),
            _slice("b", 15.0, 4.0),
            _slice("phase a", 30.0, 10.001, "gpu_user_annotation"),
            _slice("phase b", 40.0, 15.0, "gpu_user_annotation"),
        ]
        events = (
            _metadata()
            + gpu
            + [
                _slice("ProfilerStep#52", 0.0, 100.0, "user_annotation", 374, 374),
                _slice("ProfilerStep#52", 90.0, 1.0, "gpu_user_annotation"),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "trace_step52_rank0.json"
            source.write_text(json.dumps({"cuda_runtime_version": 12000, "traceEvents": events}))
            original = source.read_bytes()

            output = self._run_cli(directory)

            surviving = [
                e
                for e in output["traceEvents"]
                if e.get("cat") in {"kernel", "gpu_user_annotation"}
                and not e.get("name", "").startswith("ProfilerStep")
            ]
            self.assertEqual(surviving, gpu)
            self.assertEqual(source.read_bytes(), original)


def _exact_steps(events, cat="gpu_user_annotation"):
    import re

    return [e for e in events if e.get("cat") == cat and re.match(r"^ProfilerStep#\d+$", e.get("name", ""))]


def _assert_no_partial_overlap(test, events):
    """No two X slices on one track partially overlap (integer ns)."""
    from collections import defaultdict

    tracks = defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and "ts" in e:
            tracks[(e.get("pid"), e.get("tid"))].append((_ns(e["ts"]), _end_ns(e), e.get("name")))
    for track, spans in tracks.items():
        spans.sort(key=lambda x: (x[0], -x[1]))
        stack = []
        for start, end, name in spans:
            while stack and stack[-1][0] <= start:
                stack.pop()
            if stack:
                test.assertLessEqual(end, stack[-1][0], (track, name, stack[-1][1]))
            stack.append((end, name))


class LeadInTest(unittest.TestCase):
    """Host-ahead capture: dropped head step kept as a clipped lead-in."""

    # Warmup 50, recorded 51-58, skip_head=1, keep 5 -> export 52-56.
    _CUT = 1200.0  # CPU ProfilerStep#52 start after the shift in _raw

    def _raw(self, host_ahead=True, shift=0.0):
        events = _synthetic_steps(host_ahead, steps=range(50, 59))
        # _synthetic_steps puts step 52 at 1000 us; move so step 50 is there.
        for e in events:
            if "ts" in e:
                e["ts"] += 200.0 + shift
        return events

    def _export(self, directory, rank, events, lead_in=True):
        path = Path(directory) / f"trace_step52_rank{rank}.json"
        path.write_text(json.dumps({"hip_runtime_version": 60700, "traceEvents": events}))
        capture = trace_postprocessing._trim_warmup_from_trace(str(path), 8, 5, skip_head=1, lead_in=lead_in)
        trace_postprocessing._rebuild_gpu_annotation_spans_from_file(str(path))
        trace_postprocessing._repair_gpu_nesting_from_file(str(path))
        return path, capture

    def _stitch(self, paths):
        out = Path(paths[0]).parent / "out" / "stitched.json"
        stitch_traces.stitch_one(
            dict(enumerate(paths)),
            out,
            include={"cpu", "gpu", "other"},
            gzip_out=False,
            verbose=False,
            rocm_layout=True,
        )
        return json.loads(out.read_text())

    def test_second_skip_head_trim_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace_step52_rank0.json"
            path.write_text(json.dumps({"hip_runtime_version": 60700, "traceEvents": self._raw()}))
            first = trace_postprocessing._trim_warmup_from_trace(str(path), 8, 5, skip_head=1)
            before = path.read_text()
            again = trace_postprocessing._trim_warmup_from_trace(str(path), 5, skip_head=1)
            self.assertEqual(path.read_text(), before)
        self.assertEqual(again, first)
        self.assertEqual(again["exported_steps"], [52, 53, 54, 55, 56])
        self.assertEqual(again["lead_in_steps"], [51])

    def test_trim_keeps_renamed_lead_in_and_exports_five_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, capture = self._export(temporary, 0, self._raw())
            trace = json.loads(path.read_text())
        events = trace["traceEvents"]
        self.assertEqual(capture["exported_steps"], [52, 53, 54, 55, 56])
        self.assertEqual(capture["lead_in_steps"], [51])
        self.assertEqual(capture["exported_cpu_start_us"], self._CUT)
        self.assertGreater(capture["lead_in_events"], 0)
        cpu = sorted(e["name"] for e in _exact_steps(events, "user_annotation"))
        self.assertEqual(cpu, [f"ProfilerStep#{n}" for n in range(52, 57)])
        # One exported GPU step per step on the compute stream; the lead-in
        # step is renamed, tagged and also only on the compute stream.
        gpu = _exact_steps(events)
        self.assertEqual(sorted(e["name"] for e in gpu), [f"ProfilerStep#{n}" for n in range(52, 57)])
        self.assertEqual({e["tid"] for e in gpu}, {_COMPUTE})
        lead_steps = [e for e in events if e.get("name", "").startswith("ProfilerStep#51 (lead-in")]
        self.assertEqual({e["cat"] for e in lead_steps}, {"user_annotation", "gpu_user_annotation"})
        self.assertEqual({e["tid"] for e in lead_steps if e["cat"] == "gpu_user_annotation"}, {_COMPUTE})
        self.assertTrue(all(e["args"]["lead_in"] and e["args"]["owner_step"] == 51 for e in lead_steps))
        # Step 50 (WARMUP) and 57-58 (tail) are gone; step 51's are tagged.
        names = {e["name"] for e in events if e.get("cat") == "user_annotation"}
        for n in (50, 57, 58):
            self.assertNotIn(f"## forward {n} ##", names)
        fwd51 = [e for e in events if e.get("name") == "## forward 51 ##"]
        self.assertTrue(fwd51 and all(e["args"]["lead_in"] for e in fwd51))
        wall = trace["traceRendering"]["gpu_annotation_rebuild"]["step_wall"]
        self.assertEqual(sorted(wall["main_stream_step_ms"]["0"]), [str(n) for n in range(52, 57)])
        self.assertEqual(sorted(wall["cpu_step_ms"]), [str(n) for n in range(52, 57)])

    def _check_clean_start(self, output, cut, start=None):
        events = output["traceEvents"]
        start = cut if start is None else start
        timed = [e for e in events if e.get("ph") != "M" and "ts" in e]
        early = [e for e in timed if _ns(e["ts"]) < _ns(cut)]
        gpu_pids = {
            e["pid"]
            for e in events
            if e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}
        }
        # Before the window only lead-in events start: the head step's GPU
        # work (kept whole) and annotations, and its host events, all from
        # the trace start on.
        for e in early:
            self.assertTrue(e["args"]["lead_in"], e)
            self.assertEqual(e["args"]["owner_step"], 51)
            self.assertGreaterEqual(_ns(e["ts"]), _ns(start), e)
            if e["pid"] in gpu_pids:
                if e.get("ph") == "X":
                    self.assertGreater(_end_ns(e), _ns(start))
                    self.assertIn(e["cat"], {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"})
                else:
                    self.assertEqual(e["cat"], "ac2g")
        self.assertEqual(min(_ns(e["ts"]) for e in timed), _ns(start))
        # CPU rows open with the lead-in at the trace start on every rank
        # whose host is inside its head step then (no blank CPU row).
        self.assertEqual(
            min(_ns(e["ts"]) for e in timed if e["pid"] not in gpu_pids and e.get("ph") == "X"), _ns(start)
        )
        _assert_no_partial_overlap(self, events)
        # Every rank's compute stream opens with the lead-in step's work,
        # not with a blank before its first exported kernel; on the rank
        # whose exported CPU step starts first it is busy at the cut. (Rank 1
        # of the synthetic is idle 1185-1205 us between its own steps 50
        # and 51, which is real idle, not a trimming artifact.)
        firsts = []
        for pid in sorted({e["pid"] for e in events if e.get("cat") == "kernel"}):
            main = sorted(
                (e for e in events if e.get("cat") == "kernel" and e["pid"] == pid and e["tid"] == _COMPUTE),
                key=lambda e: e["ts"],
            )
            self.assertTrue(main[0]["args"].get("lead_in"), main[0])
            firsts.append(_ns(main[0]["ts"]))
        self.assertLessEqual(min(firsts), _ns(cut))
        # Exactly five exported GPU steps per rank, on the compute stream.
        from collections import Counter

        per_rank = Counter(e["pid"] for e in _exact_steps(events))
        self.assertTrue(per_rank and set(per_rank.values()) == {5}, per_rank)
        # No dangling flows: every remaining ac2g id still has both ends,
        # and its launch is in the file.
        flows = {}
        for e in events:
            if e.get("cat") == "ac2g":
                flows.setdefault(e["id"], set()).add(e["ph"])
        self.assertTrue(flows)
        self.assertTrue(all(v == {"s", "f"} for v in flows.values()), flows)
        # A GPU annotation's External id always names a CPU annotation in
        # the file; lead-in spans whose CPU side ended before the window
        # carry it as lead_in_external_id instead.
        cpu_ext = {e["args"]["External id"] for e in events if e.get("cat") == "user_annotation"}
        for e in events:
            if e.get("cat") == "gpu_user_annotation":
                ext = e["args"].get("External id")
                if ext is None:
                    self.assertTrue(e["args"]["lead_in"])
                    self.assertIn("lead_in_external_id", e["args"])
                else:
                    self.assertIn(ext, cpu_ext)
        rendering = output["traceRendering"]
        self.assertEqual(rendering["lead_in"]["window_start_us"], cut)
        self.assertEqual(rendering["lead_in"]["trace_start_us"], start)
        for rank, rebuild in rendering["gpu_annotation_rebuild_by_rank"].items():
            self.assertEqual(rebuild["annotation_crossings"], 0)
            self.assertEqual(rendering["gpu_nesting_by_rank"][rank]["summary"]["unresolved_crossings"], 0)

    def test_stitch_clips_lead_in_at_earliest_exported_cpu_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            # Rank 1's host is 30 us later than rank 0's (different lead).
            p0, _ = self._export(temporary, 0, self._raw())
            p1, _ = self._export(temporary, 1, self._raw(shift=30.0))
            inputs = [p0.read_bytes(), p1.read_bytes()]
            output = self._stitch([p0, p1])
            self.assertEqual([p0.read_bytes(), p1.read_bytes()], inputs)
            again = self._stitch([p0, p1])
        self.assertEqual(again, output)  # re-stitch is deterministic
        # fwd_b of step 51 crosses the cut on rank 0's compute stream from
        # 1196 us: the trace (GPU and CPU rows) starts there.
        self._check_clean_start(output, self._CUT, 1196.0)
        report = output["traceRendering"]["lead_in"]["by_rank"]
        self.assertGreater(report["0"]["dropped_events"], 0)
        self.assertGreater(report["0"]["dropped_flow_events"], 0)
        self.assertEqual(report["0"]["gpu_work_kept_whole_before_start"], 0)
        self.assertGreater(report["0"]["kept_gpu_lead_in_work"], 0)
        self.assertGreater(report["0"]["gpu_annotations_without_cpu_annotation"], 0)
        # Both ranks show their CPU step-51 tail from the trace start,
        # clipped and labelled. The stitched lead-in no longer starts with
        # "ProfilerStep#", so per-step parsers count only the exported steps.
        lead_cpu = sorted(
            (
                e
                for e in output["traceEvents"]
                if e.get("cat") == "user_annotation" and e["name"] == "## lead-in: step 51 (not exported) ##"
            ),
            key=lambda e: e["pid"],
        )
        self.assertEqual(len(lead_cpu), 2)
        for e in lead_cpu:
            self.assertEqual(e["args"]["lead_in_step_name"], "ProfilerStep#51 (lead-in, not exported)")
        self.assertFalse([e for e in output["traceEvents"] if "ProfilerStep#51" in str(e.get("name", ""))])
        lead_gpu = [
            e
            for e in output["traceEvents"]
            if e.get("cat") == "gpu_user_annotation" and e["name"] == "## lead-in: step 51 (not exported) ##"
        ]
        self.assertEqual({e["tid"] for e in lead_gpu}, {_COMPUTE})
        self.assertTrue(
            all(e["args"]["lead_in"] and e["args"]["owner_step"] == 51 for e in lead_cpu + lead_gpu)
        )
        # Every exported rebuilt GPU span names the CPU step that launched it.
        for e in output["traceEvents"]:
            if e.get("cat") == "gpu_user_annotation" and not e["args"].get("lead_in"):
                self.assertIn(e["args"].get("owner_step"), range(52, 57), e)
        for e in lead_cpu:
            self.assertEqual(e["ts"], 1196.0)
            self.assertTrue(e["args"]["clipped_at_window_start"])
        # CPU step 51 (99 us long in the synthetic) ends 1 us before 52.
        self.assertEqual([_end_ns(e) for e in lead_cpu], [_ns(self._CUT - 1.0), _ns(self._CUT + 29.0)])
        # Kernels are never shortened or moved by the clip.
        kernels = {(e["name"], e["ts"], e["dur"]) for e in output["traceEvents"] if e.get("cat") == "kernel"}
        raw = {
            (e["name"], e["ts"], e["dur"])
            for ev in (self._raw(), self._raw(shift=30.0))
            for e in ev
            if e.get("cat") == "kernel"
        }
        self.assertLessEqual(kernels, raw)

    def test_clip_twice_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self._export(temporary, 0, self._raw())
            events = json.loads(path.read_text())["traceEvents"]
        trace_postprocessing.clip_lead_in(events, self._CUT)
        snapshot = copy.deepcopy(events)
        report = trace_postprocessing.clip_lead_in(events, self._CUT)
        self.assertEqual(events, snapshot)
        self.assertEqual(report["dropped_events"], 0)
        self.assertEqual(report["annotations_clipped"], 0)

    def test_trace_without_lead_in_is_unchanged_by_clip(self):
        events = _synthetic_steps(host_ahead=False)
        snapshot = copy.deepcopy(events)
        report = trace_postprocessing.clip_lead_in(events, 1000.0 + 150.0)
        self.assertEqual(events, snapshot)
        self.assertEqual(report["lead_in_events"], 0)

    def test_synced_trace_stitches_identically_with_or_without_lead_in(self):
        # Trunk-style: the GPU finishes each step inside its CPU step, so the
        # whole lead-in ends before the window and the stitched events equal
        # those of an export without a lead-in.
        outputs = []
        for lead_in in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                path, capture = self._export(temporary, 0, self._raw(host_ahead=False), lead_in=lead_in)
                self.assertEqual(capture["lead_in_steps"], [51] if lead_in else [])
                outputs.append(self._stitch([path]))
        self.assertEqual(outputs[0]["traceEvents"], outputs[1]["traceEvents"])
        self.assertNotIn("lead_in", outputs[0]["traceRendering"])
        self.assertEqual(outputs[1]["traceRendering"]["lead_in"]["by_rank"]["0"]["kept_lead_in_events"], 0)


class CpuLeadInStartTest(unittest.TestCase):
    """CPU rows of the lead-in start where the GPU rows start.

    Rank 0 launches a long step-51 kernel (45 us on a third stream) that
    crosses the cut (CPU ProfilerStep#52 at 1200 us) from 1170 us, so the
    trace starts at 1170 us. Its autograd thread is inside a nested CPU op
    chain (evaluate_function > aten::sort > hipMemsetAsync) at that instant;
    rank 1's host runs 30 us later.
    """

    _CUT, _START, _LONG_TID = 1200.0, 1170.0, 7
    _export = LeadInTest._export
    _stitch = LeadInTest._stitch

    def _raw(self, shift=0.0):
        events = LeadInTest._raw(self, shift=shift)
        s = shift
        corr = [9000]

        def call(name, ts, dur, cat="cpu_op", tid=_AUTOGRAD_TID, gpu=None):
            ev = _slice(name, ts + s, dur, cat, _CPU_PID, tid)
            ev["args"] = {}
            events.append(ev)
            if gpu is not None:
                corr[0] += 1
                g_name, g_ts, g_dur, g_cat = gpu
                ev["args"]["correlation"] = corr[0]
                work = _slice(g_name, g_ts + s, g_dur, g_cat, 0, self._LONG_TID)
                work["args"] = {"correlation": corr[0], "stream": self._LONG_TID}
                events.extend(
                    [
                        work,
                        {
                            "ph": "s",
                            "id": corr[0],
                            "pid": _CPU_PID,
                            "tid": tid,
                            "ts": ts + s,
                            "cat": "ac2g",
                            "name": "ac2g",
                        },
                        {
                            "ph": "f",
                            "id": corr[0],
                            "pid": 0,
                            "tid": self._LONG_TID,
                            "ts": g_ts + s,
                            "cat": "ac2g",
                            "name": "ac2g",
                            "bp": "e",
                        },
                    ]
                )
            return ev

        # Main thread, inside step 51's input_dist gap: the long launch.
        call(
            "hipLaunchKernel",
            1106.0,
            1.0,
            "cuda_runtime",
            _MAIN_TID,
            ("bwd_dkdv_long", 1170.0, 45.0, "kernel"),
        )
        # Autograd thread: nested chain crossing 1170 us (inside the
        # ## autograd_bwd ## annotation, 1160-1180 us), then a kept call.
        call("autograd::engine::evaluate_function", 1165.0, 13.0)
        call("aten::sort", 1166.0, 10.0)
        call("hipMemsetAsync", 1171.0, 1.0, "cuda_runtime", gpu=("memset", 1216.0, 0.5, "gpu_memset"))
        call("aten::zeros", 1180.5, 2.5)
        call("hipLaunchKernel", 1181.0, 1.0, "cuda_runtime", gpu=("fill", 1217.0, 1.0, "kernel"))
        return events

    def _output(self):
        with tempfile.TemporaryDirectory() as temporary:
            p0, _ = self._export(temporary, 0, self._raw())
            p1, _ = self._export(temporary, 1, self._raw(shift=30.0))
            inputs = [p0.read_bytes(), p1.read_bytes()]
            output = self._stitch([p0, p1])
            again = self._stitch([p0, p1])
            self.assertEqual([p0.read_bytes(), p1.read_bytes()], inputs)
            per_rank = [json.loads(p.read_text())["traceEvents"] for p in (p0, p1)]
        self.assertEqual(again, output)  # re-stitch is deterministic
        return output, per_rank

    def test_cpu_rows_start_at_the_long_kernel(self):
        output, _ = self._output()
        events = output["traceEvents"]
        lead = output["traceRendering"]["lead_in"]
        self.assertEqual(lead["window_start_us"], self._CUT)
        self.assertEqual(lead["trace_start_us"], self._START)
        LeadInTest._check_clean_start(self, output, self._CUT, self._START)
        long_k = [e for e in events if e.get("name") == "bwd_dkdv_long" and e["pid"] == 0]
        self.assertEqual([(e["ts"], e["dur"]) for e in long_k], [(self._START, 45.0)])
        # Every rank's main CPU thread opens at the trace start with its
        # labelled step-51 annotation, clipped there.
        for rank in (0, 1):
            pid = stitch_traces._new_pid(_CPU_PID, rank)
            main = sorted(
                (e for e in events if e.get("ph") == "X" and e["pid"] == pid and e["tid"] == _MAIN_TID),
                key=lambda e: (e["ts"], -e["dur"]),
            )
            self.assertEqual(main[0]["ts"], self._START)
            self.assertEqual(main[0]["name"], "## lead-in: step 51 (not exported) ##")
            self.assertTrue(main[0]["args"]["clipped_at_window_start"])
        # No exported-step count change: five per rank on CPU and GPU.
        from collections import Counter

        self.assertEqual(
            Counter(e["pid"] for e in _exact_steps(events, "user_annotation")),
            Counter({stitch_traces._new_pid(_CPU_PID, r): 5 for r in (0, 1)}),
        )
        self.assertFalse([e for e in events if "ProfilerStep#51" in str(e.get("name", ""))])

    def test_crossing_cpu_calls_dropped_whole_with_their_nest(self):
        output, _ = self._output()
        events = output["traceEvents"]
        pid0 = stitch_traces._new_pid(_CPU_PID, 0)
        auto0 = {
            e["name"]: e
            for e in events
            if e.get("ph") == "X"
            and e["pid"] == pid0
            and e["tid"] == _AUTOGRAD_TID
            and e["args"].get("lead_in")
        }
        # The chain running at 1170 us is never truncated: dropped whole,
        # including the call nested in it that starts after 1170 us.
        for name in ("autograd::engine::evaluate_function", "aten::sort", "hipMemsetAsync"):
            self.assertNotIn(name, auto0)
        # Its enclosing annotation is clipped to the trace start; the call
        # after the chain is kept unchanged and labelled.
        self.assertEqual(auto0["## autograd_bwd ##"]["ts"], self._START)
        self.assertEqual((auto0["aten::zeros"]["ts"], auto0["aten::zeros"]["dur"]), (1180.5, 2.5))
        for name in ("aten::zeros", "hipLaunchKernel"):
            self.assertEqual(auto0[name]["args"]["lead_in"], True)
            self.assertEqual(auto0[name]["args"]["owner_step"], 51)
        report = output["traceRendering"]["lead_in"]["by_rank"]["0"]
        self.assertEqual(report["cpu_calls_dropped_across_start"], 2)
        self.assertEqual(report["cpu_events_dropped_inside_those_calls"], 1)
        # Flows: the kept launch keeps its arrow to its kernel; the dropped
        # memset launch loses its arrow, while its GPU memset (after the cut)
        # stays. The long kernel's launch ended before the trace start: its
        # arrow is dropped, the kernel kept.
        corr = {
            e["name"]: e["args"]["correlation"]
            for e in events
            if e.get("pid") == 0
            and e.get("tid") == self._LONG_TID
            and e.get("cat") in ("kernel", "gpu_memset")
        }
        self.assertEqual(set(corr), {"bwd_dkdv_long", "fill", "memset"})
        flows = {}
        for e in events:
            if e.get("cat") == "ac2g" and e["pid"] in (pid0, 0):
                flows.setdefault(e["id"], set()).add(e["ph"])
        # Rank 0's flow ids are unshifted (rank * _FLOW_ID_STRIDE).
        self.assertEqual(flows.get(corr["fill"]), {"s", "f"})
        self.assertNotIn(corr["memset"], flows)
        self.assertNotIn(corr["bwd_dkdv_long"], flows)

    def test_gpu_and_cpu_events_are_never_moved(self):
        output, per_rank = self._output()
        raw = {}
        for rank, events in enumerate(per_rank):
            for e in events:
                if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memset", "cuda_runtime", "cpu_op"):
                    raw[(rank, e["cat"], e["name"], e["args"].get("correlation"), e["ts"])] = e["dur"]
        for e in output["traceEvents"]:
            if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memset", "cuda_runtime", "cpu_op"):
                hits = [
                    d
                    for (_, c, n, k, t), d in raw.items()
                    if c == e["cat"] and n == e["name"] and k == e["args"].get("correlation") and t == e["ts"]
                ]
                self.assertTrue(hits and e["dur"] in hits, e)

    def test_clip_with_start_twice_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self._export(temporary, 0, self._raw())
            events = json.loads(path.read_text())["traceEvents"]
        start = trace_postprocessing.lead_in_trace_start(events, self._CUT)
        self.assertEqual(start, self._START)
        first = trace_postprocessing.clip_lead_in(events, self._CUT, start)
        self.assertGreater(first["kept_cpu_lead_in_events"], 0)
        self.assertEqual(trace_postprocessing.lead_in_trace_start(events, self._CUT), start)
        snapshot = copy.deepcopy(events)
        report = trace_postprocessing.clip_lead_in(events, self._CUT, start)
        self.assertEqual(events, snapshot)
        self.assertEqual(report["dropped_events"], 0)
        self.assertEqual(report["annotations_clipped"], 0)
        # A start after the cut is ignored (the cut is the latest start).
        report = trace_postprocessing.clip_lead_in(copy.deepcopy(snapshot), self._CUT, self._CUT + 50.0)
        self.assertEqual(report["start_us"], self._CUT)

    def test_without_crossing_gpu_work_cpu_side_starts_at_the_cut(self):
        events = LeadInTest._raw(self, host_ahead=False)
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self._export(temporary, 0, events)
            trimmed = json.loads(path.read_text())["traceEvents"]
        self.assertIsNone(trace_postprocessing.lead_in_trace_start(trimmed, self._CUT))

    def test_every_rank_keeps_gpu_work_from_the_gpu_start(self):
        # Regression: only GPU work crossing the cut used to be kept, so on
        # rank 0 the step-51 fwd_a (1175-1195 us, ended before the cut) was
        # dropped and the compute stream opened blank at 1196 us while the
        # long kernel ran from 1170 us. Every rank now keeps its head-step
        # GPU work overlapping [gpu_start, cut) on every stream.
        output, per_rank = self._output()
        events = output["traceEvents"]
        lead = output["traceRendering"]["lead_in"]
        self.assertEqual(lead["crossing_start_us"], self._START)
        self.assertEqual(lead["trace_start_us"], self._START)
        compute0 = sorted(
            (e for e in events if e.get("cat") == "kernel" and e["pid"] == 0 and e["tid"] == _COMPUTE),
            key=lambda e: e["ts"],
        )
        self.assertEqual(
            (compute0[0]["name"], compute0[0]["ts"], compute0[0]["dur"]), ("fwd_a", 1175.0, 20.0)
        )
        self.assertTrue(compute0[0]["args"]["lead_in"])
        self.assertEqual(compute0[0]["args"]["owner_step"], 51)
        # Every kept lead-in GPU item is in the per-rank export with the
        # same ts/dur, and every step-51 GPU item there that starts at or
        # after the trace start is in the stitched trace.
        for rank, raw in enumerate(per_rank):
            pid = stitch_traces._new_pid(0, rank)
            want = {
                (e["tid"], e["name"], e["ts"], e["dur"])
                for e in raw
                if e.get("ph") == "X"
                and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
                and e["args"].get("lead_in")
                and _ns(e["ts"]) >= _ns(self._START)
            }
            got = {
                (e["tid"], e["name"], e["ts"], e["dur"])
                for e in events
                if e.get("ph") == "X"
                and e.get("pid") == pid
                and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
                and e["args"].get("lead_in")
            }
            self.assertEqual(got, want)


class GpuLeadInStartTest(unittest.TestCase):
    """Work running at the crossing start moves the trace start once.

    On top of :class:`CpuLeadInStartTest` (long kernel crossing the cut from
    1170 us, the crossing start), rank 0 has a step-51 H2D copy on stream 9
    from 1150 to 1180 us, running at 1170 us: it is kept whole and the trace
    (GPU and CPU rows) starts at 1150 us. From there every stream keeps its
    step-51 work, also work that ended before 1170 us (stream 10's kernel at
    1156-1164 us). Stream 9's earlier copy (1140-1148 us) is before the
    trace and stays dropped; stream 10's kernel running at 1150 us
    (1145-1155 us) is dropped whole rather than moving the start again (one
    pass, so the start is a fixed point).
    """

    _CUT, _CROSSING, _START = 1200.0, 1170.0, 1150.0
    _LONG_TID = CpuLeadInStartTest._LONG_TID
    _export = LeadInTest._export
    _stitch = LeadInTest._stitch

    def _raw(self, shift=0.0):
        events = CpuLeadInStartTest._raw(self, shift=shift)
        if shift:
            return events
        corr = [9500]

        def launch(name, cpu_ts, gpu_ts, gpu_dur, cat, stream):
            corr[0] += 1
            call = _slice(
                "hipMemcpyAsync" if cat == "gpu_memcpy" else "hipLaunchKernel",
                cpu_ts,
                1.0,
                "cuda_runtime",
                _CPU_PID,
                _MAIN_TID,
            )
            call["args"] = {"correlation": corr[0]}
            work = _slice(name, gpu_ts, gpu_dur, cat, 0, stream)
            work["args"] = {"correlation": corr[0], "stream": stream}
            events.extend(
                [
                    call,
                    work,
                    {
                        "ph": "s",
                        "id": corr[0],
                        "pid": _CPU_PID,
                        "tid": _MAIN_TID,
                        "ts": cpu_ts,
                        "cat": "ac2g",
                        "name": "ac2g",
                    },
                    {
                        "ph": "f",
                        "id": corr[0],
                        "pid": 0,
                        "tid": stream,
                        "ts": gpu_ts,
                        "cat": "ac2g",
                        "name": "ac2g",
                        "bp": "e",
                    },
                ]
            )

        launch("h2d_early", 1103.0, 1140.0, 8.0, "gpu_memcpy", 9)
        launch("h2d_cross", 1104.0, 1150.0, 30.0, "gpu_memcpy", 9)
        launch("cross_start", 1105.0, 1145.0, 10.0, "kernel", 10)
        launch("between", 1107.0, 1156.0, 8.0, "kernel", 10)
        # Launched after the trace start: its arrow is kept.
        launch("late_fill", 1160.0, 1185.0, 2.0, "kernel", 10)
        return events

    _output = CpuLeadInStartTest._output

    def test_crossing_work_moves_the_trace_start_once(self):
        output, per_rank = self._output()
        events = output["traceEvents"]
        lead = output["traceRendering"]["lead_in"]
        self.assertEqual(lead["window_start_us"], self._CUT)
        self.assertEqual(lead["crossing_start_us"], self._CROSSING)
        self.assertEqual(lead["trace_start_us"], self._START)
        LeadInTest._check_clean_start(self, output, self._CUT, self._START)
        rank0 = {}
        for e in sorted(events, key=lambda e: -e.get("ts", 0)):
            if e.get("ph") == "X" and e["pid"] == 0 and e.get("cat") in ("kernel", "gpu_memcpy"):
                rank0[e["name"]] = e  # earliest item of each name
        self.assertEqual((rank0["h2d_cross"]["ts"], rank0["h2d_cross"]["dur"]), (1150.0, 30.0))
        self.assertEqual((rank0["between"]["ts"], rank0["between"]["dur"]), (1156.0, 8.0))
        self.assertEqual((rank0["fwd_a"]["ts"], rank0["fwd_a"]["dur"]), (1175.0, 20.0))
        self.assertEqual(rank0["late_fill"]["ts"], 1185.0)
        self.assertNotIn("h2d_early", rank0)
        self.assertNotIn("cross_start", rank0)
        report = lead["by_rank"]["0"]
        self.assertEqual(report["gpu_work_dropped_across_start"], 1)
        self.assertEqual(report["gpu_work_kept_whole_before_start"], 0)
        # CPU rows of both ranks open at the trace start.
        for rank in (0, 1):
            pid = stitch_traces._new_pid(_CPU_PID, rank)
            first = min(
                _ns(e["ts"])
                for e in events
                if e.get("ph") == "X" and e["pid"] == pid and e["tid"] == _MAIN_TID
            )
            self.assertEqual(first, _ns(self._START))
        # Arrows: kept for the launch kept after the trace start, dropped
        # for kept work whose launch ended before it.
        flows = {}
        for e in events:
            if e.get("cat") == "ac2g" and e["pid"] in (0, stitch_traces._new_pid(_CPU_PID, 0)):
                flows.setdefault(e["id"], set()).add(e["ph"])
        self.assertEqual(flows.get(rank0["late_fill"]["args"]["correlation"]), {"s", "f"})
        self.assertNotIn(rank0["h2d_cross"]["args"]["correlation"], flows)
        # Kept GPU work equals the per-rank export's ts/dur exactly.
        raw = {
            (e["tid"], e["name"], e["ts"], e["dur"])
            for e in per_rank[0]
            if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
        }
        for e in events:
            if (
                e.get("ph") == "X"
                and e["pid"] == 0
                and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
            ):
                self.assertIn((e["tid"], e["name"], e["ts"], e["dur"]), raw)

    def test_bounds_are_fixed_points_and_clip_twice_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self._export(temporary, 0, self._raw())
            events = json.loads(path.read_text())["traceEvents"]
        bounds = trace_postprocessing.lead_in_trace_bounds([events], self._CUT)
        self.assertEqual(bounds, (self._CROSSING, self._START))
        pristine = copy.deepcopy(events)
        # Default start: computed from this file the same way.
        default = trace_postprocessing.clip_lead_in(copy.deepcopy(pristine), self._CUT)
        self.assertEqual(default["start_us"], self._START)
        first = trace_postprocessing.clip_lead_in(events, self._CUT, self._START)
        self.assertEqual(first["start_us"], self._START)
        self.assertEqual(first["gpu_work_dropped_across_start"], 1)
        # Recomputing the bounds on the clipped events finds the same ones.
        self.assertEqual(trace_postprocessing.lead_in_trace_bounds([events], self._CUT), bounds)
        snapshot = copy.deepcopy(events)
        report = trace_postprocessing.clip_lead_in(events, self._CUT, self._START)
        self.assertEqual(events, snapshot)
        self.assertEqual(report["dropped_events"], 0)
        self.assertEqual(report["annotations_clipped"], 0)
        # A start after the cut is capped at the cut; work crossing the cut
        # is still kept whole (and reported) even then.
        late = copy.deepcopy(pristine)
        report = trace_postprocessing.clip_lead_in(late, self._CUT, self._CUT + 50.0)
        self.assertEqual(report["start_us"], self._CUT)
        self.assertGreater(report["gpu_work_kept_whole_before_start"], 0)
        self.assertIn("bwd_dkdv_long", {e.get("name") for e in late})
        _assert_no_partial_overlap(self, late)


def _work(name, corr, ts, dur, tid=7, pid=0):
    return {
        "ph": "X",
        "cat": "kernel",
        "name": name,
        "pid": pid,
        "tid": tid,
        "ts": ts,
        "dur": dur,
        "args": {"correlation": corr},
    }


def _spans_ns(events):
    return {e["args"]["correlation"]: (_ns(e["ts"]), _end_ns(e)) for e in events if e.get("cat") == "kernel"}


class StreamOrderTest(unittest.TestCase):
    """serialize_stream_order: bounded launch-order repair on one stream."""

    def _check_serial(self, events):
        spans = sorted(_spans_ns(events).items())
        for (_, (s0, e0)), (_, (s1, e1)) in zip(spans, spans[1:]):
            self.assertLessEqual(s0, e0)
            self.assertLessEqual(e0, s1)

    def test_clean_stream_is_unchanged(self):
        events = [_work("a", 1, 1000.0, 5.0), _work("b", 2, 1005.0, 3.0), _work("c", 3, 1010.0, 2.0)]
        snapshot = copy.deepcopy(events)
        report = trace_postprocessing.serialize_stream_order(events)
        self.assertEqual(events, snapshot)
        self.assertEqual(report["summary"]["changed_events"], 0)

    def test_swapped_pair_is_reordered_within_its_own_interval(self):
        # Rank-26 pattern: corr 34454 drawn right after 34455 (back to back).
        events = [
            _work("before", 1, 990.0, 5.0),
            _work("A", 2, 1001.373, 4.959),
            _work("B", 3, 1000.0, 1.373),
            _work("after", 4, 1010.0, 2.0),
        ]
        report = trace_postprocessing.serialize_stream_order(events)
        spans = _spans_ns(events)
        self.assertEqual(spans[2], (_ns(1000.0), _ns(1000.0) + 1373))
        self.assertEqual(spans[3], (_ns(1000.0) + 1373, _ns(1001.373) + 4959))
        self._check_serial(events)
        self.assertEqual(report["summary"]["launch_order_reassigned"], 2)
        self.assertEqual(report["summary"]["max_shift_ns"], 4959)
        self.assertTrue(
            all(e["args"].get("stream_order_repaired") for e in events if e["name"] in ("A", "B"))
        )
        self.assertFalse(events[0]["args"].get("stream_order_repaired"))

    def test_three_way_reversal_and_containment(self):
        # Z, Y, X drawn in reverse launch order; a fill inside a sort.
        events = [
            _work("X", 1, 1001.758, 4.642),
            _work("Y", 2, 1001.252, 0.506),
            _work("Z", 3, 1000.0, 1.252),
            _work("fill", 4, 1020.115, 4.959),
            _work("sort", 5, 1020.0, 37.959),
        ]
        report = trace_postprocessing.serialize_stream_order(events)
        self._check_serial(events)
        spans = _spans_ns(events)
        self.assertEqual(spans[1][0], _ns(1000.0))
        self.assertEqual(spans[5][1], _ns(1020.0) + 37959)  # union kept
        self.assertEqual(report["unresolved"], [])
        self.assertLessEqual(report["summary"]["max_shift_ns"], 12000)

    def test_late_end_overlap_is_trimmed_and_large_one_left(self):
        events = [
            _work("gemm", 1, 1000.0, 1363.412),
            _work("tail", 2, 2356.708, 6.319),
            _work("p", 3, 3000.0, 30.0),
            _work("q", 4, 3010.0, 5.0),
            _work("r", 5, 3020.0, 40.0),
        ]
        report = trace_postprocessing.serialize_stream_order(events)
        spans = _spans_ns(events)
        # The tail kernel stays put; the GEMM end moves back <= 12 us.
        self.assertEqual(spans[2][0], _ns(2356.708))
        self.assertEqual(spans[1][1], _ns(2356.708))
        # p overlaps q by 20 us and r by 10 us: beyond the bound, unchanged.
        self.assertEqual(spans[3], (_ns(3000.0), _ns(3000.0) + 30000))
        self.assertTrue(report["unresolved"])
        self.assertEqual(spans[4], (_ns(3010.0), _ns(3010.0) + 5000))

    def test_streams_without_correlation_are_skipped(self):
        events = [_work("a", 1, 1000.0, 10.0), _work("b", 2, 1005.0, 10.0)]
        del events[1]["args"]["correlation"]
        snapshot = copy.deepcopy(events)
        report = trace_postprocessing.serialize_stream_order(events)
        self.assertEqual(events, snapshot)
        self.assertEqual(report["skipped_streams_without_correlation"], ["0:7"])

    def test_stitch_repairs_order_before_rebuilding_annotations(self):
        # Kernel 11 (launched under "outer" only) is drawn 1 us after kernel
        # 12, the first kernel of "inner". After the repair, "inner" starts
        # after kernel 11 and contains only its own launch.
        cpu, gpu = 100, 0
        events = [
            {"ph": "M", "name": "process_labels", "pid": gpu, "args": {"labels": "GPU 0"}},
            {"ph": "M", "name": "process_labels", "pid": cpu, "args": {"labels": "CPU"}},
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "ProfilerStep#52",
                "pid": cpu,
                "tid": 1,
                "ts": 900.0,
                "dur": 200.0,
                "args": {"External id": 1},
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "outer",
                "pid": cpu,
                "tid": 1,
                "ts": 910.0,
                "dur": 100.0,
                "args": {"External id": 2},
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "inner",
                "pid": cpu,
                "tid": 1,
                "ts": 950.0,
                "dur": 20.0,
                "args": {"External id": 3},
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "hipLaunchKernel",
                "pid": cpu,
                "tid": 1,
                "ts": 920.0,
                "dur": 1.0,
                "args": {"correlation": 11},
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "hipLaunchKernel",
                "pid": cpu,
                "tid": 1,
                "ts": 955.0,
                "dur": 1.0,
                "args": {"correlation": 12},
            },
            _work("k11", 11, 1001.0, 5.0, tid=2, pid=gpu),
            _work("k12", 12, 1000.0, 1.0, tid=2, pid=gpu),
            _work("k13", 13, 1010.0, 50.0, tid=2, pid=gpu),
        ]
        events.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "hipLaunchKernel",
                "pid": cpu,
                "tid": 1,
                "ts": 965.0,
                "dur": 1.0,
                "args": {"correlation": 13},
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace_step52_rank0.json"
            path.write_text(json.dumps({"hip_runtime_version": 60700, "traceEvents": events}))
            out = Path(temporary) / "out.json"
            stitch_traces.stitch_one(
                {0: path},
                out,
                include={"cpu", "gpu", "other"},
                gzip_out=False,
                verbose=False,
                rocm_layout=True,
            )
            output = json.loads(out.read_text())
        inner = [
            e for e in output["traceEvents"] if e.get("cat") == "gpu_user_annotation" and e["name"] == "inner"
        ]
        self.assertEqual(len(inner), 1)
        k11 = [e for e in output["traceEvents"] if e.get("name") == "k11"][0]
        self.assertLessEqual(_end_ns(k11), _ns(inner[0]["ts"]))
        order = output["traceRendering"]["gpu_stream_order_by_rank"]["0"]
        self.assertEqual(order["summary"]["launch_order_reassigned"], 2)
        self.assertEqual(
            output["traceRendering"]["gpu_annotation_rebuild_by_rank"]["0"]["annotation_crossings"], 0
        )


if __name__ == "__main__":
    unittest.main()
