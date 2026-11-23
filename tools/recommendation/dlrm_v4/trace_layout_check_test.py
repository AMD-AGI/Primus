"""Unit tests for tools/recommendation/dlrm_v4/trace_layout_check.py (CPU-only, synthetic traces).

Run: python3 -m unittest scripts.trace_layout_check_test
"""

import copy
import gzip
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_layout_check as tlc  # noqa: E402

STRIDE = tlc.PID_STRIDE
CPU_PID, MAIN_TID, AG_TID = 340, 340, 9000
MAIN_STREAM, SIDE_STREAM = 4, 1
STEP_US = 1000.0  # 1 ms per step


class Builder:
    """Synthetic stitched trace: ranks x steps with forward/backward/optimizer."""

    def __init__(self, ranks=2, steps=5, first=52, kernel_prefix="k", step_us=STEP_US):
        self.ranks, self.steps, self.first = ranks, steps, first
        self.kernel_prefix, self.step_us = kernel_prefix, step_us
        self.ev = []
        self.ext = 1
        self.corr = 1
        self.extra_step_us = {}  # (rank, step) -> extra us inside forward (host+GPU)
        self.barrier_steps = set()  # steps with a window-boundary barrier
        self.extra_ann = None  # extra GPU/CPU annotation name inside forward
        self.drop_ann = False  # drop "## main_forward ##"

    # -- primitives -------------------------------------------------------
    def x(self, cat, name, pid, tid, ts, dur, **args):
        e = {"ph": "X", "cat": cat, "name": name, "pid": pid, "tid": tid, "ts": ts, "dur": dur}
        if args:
            e["args"] = args
        self.ev.append(e)
        return e

    def meta(self, r):
        cpid, gpid = CPU_PID + r * STRIDE, r + r * STRIDE
        M = lambda name, pid, tid, args: self.ev.append(
            {"ph": "M", "name": name, "pid": pid, "tid": tid, "ts": 0, "args": args}
        )
        M("process_name", cpid, 0, {"name": f"[Rank {r}] python"})
        M("process_labels", cpid, 0, {"labels": "CPU"})
        M("process_sort_index", cpid, 0, {"sort_index": r})
        M("process_name", gpid, 0, {"name": f"[Rank {r}] python"})
        M("process_labels", gpid, 0, {"labels": f"GPU {r}"})
        M("process_sort_index", gpid, 0, {"sort_index": 1_000_000 + r * 100})
        M("thread_name", cpid, MAIN_TID, {"name": f"thread {MAIN_TID} (python)"})
        M("thread_sort_index", cpid, MAIN_TID, {"sort_index": MAIN_TID})
        M("thread_name", cpid, AG_TID, {"name": f"thread {AG_TID} (pt_autograd_0)"})
        M("thread_sort_index", cpid, AG_TID, {"sort_index": AG_TID})
        for s in (SIDE_STREAM, MAIN_STREAM):
            M("thread_name", gpid, s, {"name": f"stream {s} "})
            M("thread_sort_index", gpid, s, {"sort_index": s})

    def launch(self, r, tid, ts, kname, gts, gdur, stream=MAIN_STREAM):
        cpid, gpid = CPU_PID + r * STRIDE, r + r * STRIDE
        c = self.corr
        self.corr += 1
        self.x("cuda_runtime", "hipLaunchKernel", cpid, tid, ts, 2.0, correlation=c)
        return self.x("kernel", kname, gpid, stream, gts, gdur, correlation=c)

    def ann(self, r, name, tid, ts, dur, gts, gdur, stream=MAIN_STREAM, gpu=True):
        cpid, gpid = CPU_PID + r * STRIDE, r + r * STRIDE
        ext = self.ext
        self.ext += 1
        self.x("user_annotation", name, cpid, tid, ts, dur, **{"External id": ext})
        if gpu:
            self.x("gpu_user_annotation", name, gpid, stream, gts, gdur, **{"External id": ext})
        return ext

    # -- trace ------------------------------------------------------------
    def build(self):
        for r in range(self.ranks):
            self.meta(r)
            t = 100.0
            for i in range(self.steps):
                n = self.first + i
                extra = self.extra_step_us.get((r, n), 0.0) + (
                    max(self.extra_step_us.get((q, n), 0.0) for q in range(self.ranks))
                    if any((q, n) in self.extra_step_us for q in range(self.ranks))
                    and (r, n) not in self.extra_step_us
                    else 0.0
                )
                D = self.step_us + extra
                g0 = t + 10.0  # GPU lags host by 10 us
                self.ann(r, f"ProfilerStep#{n}", MAIN_TID, t, D, g0, D - 20.0)
                fw = 0.4 * self.step_us + extra
                if n in self.barrier_steps:
                    cpid = CPU_PID + r * STRIDE
                    self.x("cpu_op", "c10d::barrier", cpid, MAIN_TID, t + 1.0, 3.0)
                # forward on the main thread
                self.ann(r, f"## forward {n} ##", MAIN_TID, t + 5, fw, g0 + 5, fw - 10)
                if not self.drop_ann:
                    self.ann(r, "## main_forward ##", MAIN_TID, t + 10, fw - 20, g0 + 10, fw - 30)
                if self.extra_ann:
                    self.ann(r, self.extra_ann, MAIN_TID, t + 11, 3, g0 + 11, 3)
                if (r, n) in self.extra_step_us:
                    # host stall with real self time (no HIP call) on this rank
                    cpid = CPU_PID + r * STRIDE
                    self.x("cpu_op", "slow_host_op", cpid, MAIN_TID, t + 20, extra - 5)
                elif extra:
                    cpid = CPU_PID + r * STRIDE
                    self.x(
                        "user_annotation",
                        "## splits poll ##",
                        cpid,
                        MAIN_TID,
                        t + 20,
                        extra - 5,
                        **{"External id": self.ext},
                    )
                    self.ext += 1
                self.launch(r, MAIN_TID, t + 15, f"{self.kernel_prefix}_fwd", g0 + 15, fw - 40)
                # backward on the autograd thread
                b0 = t + 5 + fw + 10
                bw = 0.4 * self.step_us
                self.ann(r, f"## backward {n} ##", MAIN_TID, b0, bw, g0 + fw + 15, bw - 10)
                self.ann(r, "## hstu_bwd ##", AG_TID, b0 + 5, bw - 20, g0 + fw + 20, bw - 30)
                self.launch(r, AG_TID, b0 + 10, f"{self.kernel_prefix}_bwd", g0 + fw + 25, bw - 40)
                # comm on a side stream
                self.launch(
                    r, MAIN_TID, b0 + bw + 5, "nccl_allreduce", g0 + fw + bw + 30, 20.0, stream=SIDE_STREAM
                )
                # optimizer
                o0 = b0 + bw + 20
                self.ann(r, f"## optimizer {n} ##", MAIN_TID, o0, 50, g0 + fw + bw + 60, 40)
                self.launch(r, MAIN_TID, o0 + 5, f"{self.kernel_prefix}_adam", g0 + fw + bw + 65, 30)
                t += D
        return {"traceEvents": self.ev}


def run(trace, ref=None, **kw):
    ref = tlc.build_profile(trace if ref is None else ref, "ref")
    return tlc.check_trace(trace, ref, **kw)


def checks(res, sev="irregular"):
    key = {
        "irregular": "irregularities",
        "allowed": "allowed_differences",
        "trunk_difference": "trunk_differences",
        "expected": "expected",
    }[sev]
    return [(i["check"], i["message"]) for i in res[key]]


class TestHelpers(unittest.TestCase):
    def test_norm_collapse_fmt(self):
        self.assertEqual(tlc.norm("## forward 52 ##"), "## forward N ##")
        self.assertEqual(tlc.collapse([1, 1, 2, 2, 1]), (1, 2, 1))
        self.assertEqual(tlc.fmt_ranks([0, 1, 2, 5, 7, 8]), "rank 0-2,5,7-8")
        self.assertEqual(tlc.fmt_ranks([None]), "trace")

    def test_nest_depths(self):
        mk = lambda s, t: tlc.Span({"ph": "X", "name": "a", "ts": s, "dur": t - s})
        a, b, c, d = mk(0, 10), mk(1, 5), mk(2, 3), mk(6, 9)
        out = {id(sp): (dep, p) for sp, dep, p in tlc.nest([d, c, b, a])}
        self.assertEqual(out[id(a)], (0, None))
        self.assertEqual(out[id(b)][0], 1)
        self.assertIs(out[id(c)][1], b)
        self.assertIs(out[id(d)][1], a)

    def test_split_ranks(self):
        ev = [{"pid": 5, "ph": "X"}, {"pid": STRIDE + 7, "ph": "X"}, {"pid": "Traces", "ph": "X"}]
        g = tlc.split_ranks(ev)
        self.assertEqual(sorted(k for k in g if k != "other"), [0, 1])
        self.assertEqual(len(g["other"]), 1)


class TestConformance(unittest.TestCase):
    def test_self_is_conformant(self):
        tr = Builder().build()
        res = run(tr)
        self.assertTrue(res["conformant"], res["groups"])
        self.assertEqual(res["n_ranks"], 2)

    def test_optimization_differences_tolerated(self):
        """New kernel names and shorter steps do not break conformance."""
        ref = Builder().build()
        cand = Builder(kernel_prefix="fused", step_us=700.0).build()
        res = run(cand, ref)
        self.assertTrue(res["conformant"], res["groups"])
        self.assertLess(
            res["timing"]["candidate_gpu_step_ms_overall"], res["timing"]["trunk_gpu_step_ms_overall"]
        )

    def test_new_annotation_is_allowed_missing_is_irregular(self):
        ref = Builder().build()
        b = Builder()
        b.extra_ann = "## fused_new_op ##"
        res = run(b.build(), ref)
        self.assertTrue(res["conformant"], res["groups"])
        allowed = " ".join(json.dumps(i["evidence"]) for i in res["allowed_differences"])
        self.assertIn("## fused_new_op ##", allowed)
        b = Builder()
        b.drop_ann = True
        res = run(b.build(), ref)
        self.assertFalse(res["conformant"])
        self.assertTrue(any("missing" in m for c, m in checks(res)))

    def test_legacy_row_and_extra_track(self):
        ref = Builder().build()
        cand = Builder().build()
        cand["traceEvents"].append(
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 99, "args": {"name": "GPU steps (even)"}}
        )
        cand["traceEvents"].append(
            {
                "ph": "X",
                "cat": "gpu_step",
                "name": "ProfilerStep#52",
                "pid": 0,
                "tid": 99,
                "ts": 110.0,
                "dur": 900.0,
            }
        )
        # extra unnamed host thread with HIP calls (not a known helper signature)
        cand["traceEvents"].append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "hipMemGetInfo",
                "pid": CPU_PID,
                "tid": 777,
                "ts": 200.0,
                "dur": 1.0,
            }
        )
        res = run(cand, ref)
        msgs = " | ".join(m for c, m in checks(res))
        self.assertIn("legacy synthetic step rows", msgs)
        self.assertIn("extra track(s) 'cpu:unnamed[cuda_runtime]'", msgs)
        self.assertIn("extra track(s) 'gpu:stream 99'", msgs)

    def test_helper_threads_are_optional_tracks(self):
        """NCCL watchdog / pin-memory threads appear only when they make a HIP
        call in the window: extra or missing ones are listed, not irregular."""
        hip = lambda name, tid, ts, pid=CPU_PID: {
            "ph": "X",
            "cat": "cuda_runtime",
            "name": name,
            "pid": pid,
            "tid": tid,
            "ts": ts,
            "dur": 1.0,
        }
        ref = Builder().build()
        for r in range(2):
            pid = CPU_PID + r * STRIDE
            ref["traceEvents"] += [
                hip("hipThreadExchangeStreamCaptureMode", 1419, 300.0, pid),
                hip("hipDevicePrimaryCtxGetState", 9459, 400.0, pid),
                hip("hipPointerGetAttributes", 9459, 401.0, pid),
                hip("hipHostMalloc", 9459, 402.0, pid),
            ]
        cand = Builder().build()
        cand["traceEvents"] += [
            hip("hipThreadExchangeStreamCaptureMode", 1425, 300.0),
            hip("hipThreadExchangeStreamCaptureMode", 1425, 400.0),
            hip("hipThreadExchangeStreamCaptureMode", 1425 + STRIDE, 300.0, CPU_PID + STRIDE),
        ]
        res = run(cand, ref)
        self.assertTrue(res["conformant"], res["groups"])
        allowed = [m for c, m in checks(res, "allowed") if "helper" in m]
        self.assertTrue(any("cpu:helper[pin_memory]" in m and "0 vs trunk 1" in m for m in allowed))
        r = tlc.Rank(0, tlc.split_ranks(cand["traceEvents"])[0])
        self.assertEqual(r.thread_role((CPU_PID, 1425)), "cpu:helper[nccl_watchdog]")

    def test_side_stream_names_reported_as_trunk_difference(self):
        ref = Builder().build()
        cand = Builder().build()
        cand["traceEvents"].append(
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "## splits poll ##",
                "pid": 0,
                "tid": SIDE_STREAM,
                "ts": 150.0,
                "dur": 2.0,
                "args": {"External id": 1},
            }
        )
        res = run(cand, ref)
        diff = [i for i in res["trunk_differences"] if "side-stream" in i["message"]]
        self.assertTrue(diff)
        self.assertEqual(diff[0]["evidence"]["new"], ["## splits poll ##"])

    def test_gpu_step_on_side_stream_and_missing(self):
        cand = Builder().build()
        ev = cand["traceEvents"]
        # copy step 53 of rank 0 onto the side stream
        st = next(
            e
            for e in ev
            if e.get("cat") == "gpu_user_annotation" and e["name"] == "ProfilerStep#53" and e["pid"] == 0
        )
        side = copy.deepcopy(st)
        side["tid"] = SIDE_STREAM
        ev.append(side)
        # delete GPU ProfilerStep#55 of rank 1
        ev.remove(
            next(
                e
                for e in ev
                if e.get("cat") == "gpu_user_annotation"
                and e["name"] == "ProfilerStep#55"
                and e["pid"] == 1 + STRIDE
            )
        )
        res = run(cand, Builder().build())
        msgs = [(i["rank"], i["step"], i["message"]) for i in res["irregularities"] if i["check"] == "steps"]
        self.assertIn(
            (0, 53, "GPU ProfilerStep#53 also on side stream(s) (layout v2: main stream only)"), msgs
        )
        self.assertTrue(any(r == 1 and s == 55 and m.startswith("0 GPU ProfilerStep#55") for r, s, m in msgs))

    def test_step_count_and_consecutive(self):
        res = run(Builder(steps=4).build(), Builder().build())
        self.assertTrue(any("4 CPU ProfilerSteps, expected 5" in m for c, m in checks(res)))

    def test_trunk_placement_reported(self):
        res = run(Builder().build())
        self.assertTrue(
            any("trunk GPU ProfilerStep placement" in m for c, m in checks(res, "trunk_difference"))
        )


class TestEdgesAndExtended(unittest.TestCase):
    def test_blank_lead_in(self):
        cand = Builder().build()
        # an early host event 300 us (30% of a step) before the first step
        cand["traceEvents"].append(
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::empty",
                "pid": CPU_PID,
                "tid": MAIN_TID,
                "ts": -200.0,
                "dur": 1.0,
            }
        )
        res = run(cand, Builder().build())
        edge = [i for i in res["irregularities"] if i["check"] == "edges" and i["rank"] == 0]
        self.assertEqual(len(edge), 1)
        self.assertIn("lead-in", edge[0]["message"])
        self.assertIn("precede the first CPU ProfilerStep", edge[0]["explanation"])

    def test_skip_head_explanation(self):
        cand = Builder().build()
        cand["traceEvents"].append(
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::empty",
                "pid": CPU_PID,
                "tid": MAIN_TID,
                "ts": -200.0,
                "dur": 1.0,
            }
        )
        cand["traceRendering"] = {
            "trace_capture_by_rank": {"0": {"skipped_head_steps": [51], "skip_head": 1}}
        }
        res = run(cand, Builder().build())
        edge = [i for i in res["irregularities"] if i["check"] == "edges" and i["rank"] == 0]
        self.assertIn("skipped head step(s) [51]", edge[0]["explanation"])

    def test_tail(self):
        cand = Builder().build()
        cand["traceEvents"].append(
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "export",
                "pid": CPU_PID,
                "tid": MAIN_TID,
                "ts": 100 + 5 * STEP_US + 400,
                "dur": 1.0,
            }
        )
        res = run(cand, Builder().build())
        self.assertTrue(
            any(
                i["check"] == "edges" and "tail" in i["message"] and i["rank"] == 0
                for i in res["irregularities"]
            )
        )

    def test_extended_step_single_rank_origin_with_barrier(self):
        b = Builder()
        b.extra_step_us[(0, 54)] = 2000.0  # rank 0 stalls; rank 1 waits in splits poll
        b.barrier_steps.add(54)
        res = run(b.build(), Builder().build(), min_excess_ms=0.0)
        ext = [i for i in res["irregularities"] if i["check"] == "extended" and i["step"] == 54]
        self.assertEqual({i["rank"] for i in ext}, {0, 1})
        e0 = next(i for i in ext if i["rank"] == 0)
        self.assertIn("c10d::barrier", e0["explanation"])
        self.assertIn("single-rank host stall: rank 0", e0["explanation"])
        self.assertIn("slow_host_op", e0["explanation"])

    def test_cpu_only_extended_step(self):
        b = Builder()
        tr = b.build()
        # lengthen CPU ProfilerStep#54 of rank 0 only (GPU unchanged); add a hipMemMap stall
        for e in tr["traceEvents"]:
            if e.get("cat") == "user_annotation" and e["name"] == "ProfilerStep#54" and e["pid"] == CPU_PID:
                e["dur"] += 2500.0
                t = e["ts"] + 1050.0
        tr["traceEvents"].append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "## start_sparse_data_dist 55 ##",
                "pid": CPU_PID,
                "tid": MAIN_TID,
                "ts": t,
                "dur": 2400.0,
                "args": {"External id": 99999},
            }
        )
        tr["traceEvents"].append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "hipMemMap",
                "pid": CPU_PID,
                "tid": MAIN_TID,
                "ts": t + 10,
                "dur": 2300.0,
            }
        )
        res = run(tr, Builder().build(), expected_steps=None, min_excess_ms=0.0)
        ext = [i for i in res["irregularities"] if i["check"] == "extended" and i["rank"] == 0]
        self.assertTrue(ext)
        x = ext[0]["explanation"]
        self.assertIn("host-side only", x)
        self.assertIn("allocator growth", x)

    def test_wait_step_names_cross_rank_allocator_origin(self):
        """Rank 0 is blocked in hipMemMap; rank 1 waits in '## splits poll ##'.
        The waiting rank's explanation names rank 0's allocator call, and the
        watchdog thread's HIP ticks do not cap the main-thread no-HIP gap."""
        b = Builder()
        b.extra_step_us[(0, 54)] = 2000.0
        tr = b.build()
        ev = tr["traceEvents"]
        stall = next(e for e in ev if e.get("name") == "slow_host_op")
        stall.update(cat="cuda_runtime", name="hipMemMap")
        poll = next(e for e in ev if e.get("name") == "## splits poll ##")
        for k in range(20):  # rank-1 watchdog thread: a HIP call every 100 us
            ev.append(
                {
                    "ph": "X",
                    "cat": "cuda_runtime",
                    "name": "hipThreadExchangeStreamCaptureMode",
                    "pid": poll["pid"],
                    "tid": 1419,
                    "ts": poll["ts"] + 100.0 * k,
                    "dur": 1.0,
                }
            )
        res = run(tr, Builder().build(), expected_steps=None, min_excess_ms=0.0)
        ext = {
            i["rank"]: i
            for i in res["irregularities"]
            if i["check"] == "extended" and i["step"] == 54 and i["message"].startswith("CPU")
        }
        self.assertEqual(set(ext), {0, 1})
        self.assertIn("allocator growth", ext[0]["explanation"])
        x1 = ext[1]["explanation"]
        self.assertIn("cross-rank origin: rank(s) [0]", x1)
        self.assertIn("hipMemMap", x1)
        self.assertGreater(ext[1]["evidence"]["host_gaps"]["no_hip_call_ms"], 1.5)
        self.assertIn("on the main thread", x1)
        self.assertEqual(res["wait_origins"]["54"]["ranks"], [0])
        self.assertTrue(res["wait_origins"]["54"]["allocator"])

    def test_window_boundary_step_is_expected(self):
        tr = Builder().build()
        ev = tr["traceEvents"]
        for r in range(2):
            pid, gpid = CPU_PID + r * STRIDE, r + r * STRIDE
            st = next(
                e
                for e in ev
                if e.get("cat") == "user_annotation" and e["name"] == "ProfilerStep#56" and e["pid"] == pid
            )
            g = next(
                e
                for e in ev
                if e.get("cat") == "gpu_user_annotation"
                and e["name"] == "ProfilerStep#56"
                and e["pid"] == gpid
            )
            st["dur"] += 2500.0
            t = st["ts"] + 1000.0
            ext = 99900 + r
            ev.append(
                {
                    "ph": "X",
                    "cat": "user_annotation",
                    "name": "## window_boundary ##",
                    "pid": pid,
                    "tid": MAIN_TID,
                    "ts": st["ts"] + 1.0,
                    "dur": 3.0,
                    "args": {"External id": ext},
                }
            )
            ev.append(
                {
                    "ph": "X",
                    "cat": "gpu_user_annotation",
                    "name": "## window_boundary ##",
                    "pid": gpid,
                    "tid": MAIN_STREAM,
                    "ts": g["ts"] + 1.0,
                    "dur": 2.0,
                    "args": {"External id": ext},
                }
            )
            ev.append(
                {
                    "ph": "X",
                    "cat": "user_annotation",
                    "name": "## window_boundary ##",
                    "pid": pid,
                    "tid": MAIN_TID,
                    "ts": t,
                    "dur": 2400.0,
                    "args": {"External id": ext + 50},
                }
            )
            ev.append(
                {
                    "ph": "X",
                    "cat": "cpu_op",
                    "name": "c10d::barrier",
                    "pid": pid,
                    "tid": MAIN_TID,
                    "ts": t + 1,
                    "dur": 2390.0,
                }
            )
            ev.append(
                {
                    "ph": "X",
                    "cat": "cuda_runtime",
                    "name": "hipStreamSynchronize",
                    "pid": pid,
                    "tid": MAIN_TID,
                    "ts": t + 2,
                    "dur": 2380.0,
                }
            )
            # next window's first GPU work right after the barrier (no tail)
            ev.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "next_window",
                    "pid": gpid,
                    "tid": MAIN_STREAM,
                    "ts": t + 2400.0,
                    "dur": 200.0,
                }
            )
        res = run(tr, Builder().build(), min_excess_ms=0.0)
        self.assertTrue(res["conformant"], res["groups"])
        exp = {(i["check"], i["step"]) for i in res["expected"]}
        self.assertIn(("extended", 56), exp)
        self.assertIn(("layout", 56), exp)


class TestIntegrity(unittest.TestCase):
    def _gpu(self, tr, name, pid=0):
        return next(
            e
            for e in tr["traceEvents"]
            if e.get("cat") == "gpu_user_annotation" and e["name"] == name and e["pid"] == pid
        )

    def test_orphan(self):
        tr = Builder().build()
        tr["traceEvents"].append(
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "## forward 51 ##",
                "pid": 0,
                "tid": MAIN_STREAM,
                "ts": 0.0,
                "dur": 5.0,
                "args": {"External id": 424242},
            }
        )
        res = run(tr, Builder().build())
        o = [i for i in res["irregularities"] if "orphan" in i["message"]]
        self.assertEqual(len(o), 1)
        self.assertIn("pre-window step numbers [51]", o[0]["explanation"])

    def test_lead_in_annotations_are_not_orphans(self):
        tr = Builder().build()
        # clipped lead-in: External id moved to lead_in_external_id
        tr["traceEvents"].append(
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "## forward 51 ##",
                "pid": 0,
                "tid": MAIN_STREAM,
                "ts": 0.0,
                "dur": 5.0,
                "args": {"lead_in": True, "owner_step": 51, "lead_in_external_id": 424242},
            }
        )
        # lead-in that still carries its External id (CPU side clipped away)
        tr["traceEvents"].append(
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "## backward 51 ##",
                "pid": 0,
                "tid": MAIN_STREAM,
                "ts": 6.0,
                "dur": 5.0,
                "args": {"lead_in": True, "owner_step": 51, "External id": 424243},
            }
        )
        res = run(tr, Builder().build())
        self.assertFalse([i for i in res["irregularities"] if "orphan" in i["message"]])
        li = [i for i in res["allowed_differences"] if "lead-in GPU annotation" in i["message"]]
        self.assertEqual(len(li), 1)
        self.assertEqual(li[0]["evidence"], {"lead_in_steps": [51], "unlinked": 2})

    def test_crossing(self):
        tr = Builder().build()
        g = self._gpu(tr, "## main_forward ##")
        g["dur"] += 400.0  # now crosses into backward on the main stream
        res = run(tr, Builder().build())
        self.assertTrue(any("crossing" in m for c, m in checks(res)))

    def test_same_stream_work_overlap_and_nesting(self):
        tr = Builder().build()
        k = [
            e
            for e in tr["traceEvents"]
            if e.get("cat") == "kernel" and e["pid"] == 0 and e["tid"] == MAIN_STREAM
        ]
        k.sort(key=lambda e: e["ts"])
        # a kernel nested inside the forward kernel, and one crossing its end
        tr["traceEvents"].append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "inner",
                "pid": 0,
                "tid": MAIN_STREAM,
                "ts": k[0]["ts"] + 3.0,
                "dur": 2.0,
                "args": {"correlation": 77777},
            }
        )
        tr["traceEvents"].append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": "late",
                "pid": 0,
                "tid": MAIN_STREAM,
                "ts": k[0]["ts"] + k[0]["dur"] - 3.0,
                "dur": 5.0,
                "args": {"correlation": 77778},
            }
        )
        res = run(tr, Builder().build())
        msgs = " | ".join(m for c, m in checks(res))
        self.assertIn("drawn inside another work item on the same stream", msgs)
        self.assertIn("same-stream kernel/memcpy crossing", msgs)
        clean = run(Builder().build(), Builder().build())
        self.assertFalse([m for c, m in checks(clean) if "same stream" in m or "same-stream" in m])

    def test_step_span_missing_own_kernels(self):
        tr = Builder().build()
        g = self._gpu(tr, "ProfilerStep#53")
        g["dur"] = 100.0  # cut short: its own backward/optimizer kernels fall outside
        res = run(tr, Builder().build())
        self.assertTrue(any("does not contain" in m for c, m in checks(res)))

    def test_duplicates(self):
        tr = Builder().build()
        tr["traceEvents"].append(copy.deepcopy(self._gpu(tr, "## optimizer 52 ##")))
        res = run(tr, Builder().build())
        self.assertTrue(any("duplicated slice" in m for c, m in checks(res)))

    def test_layout_change_within_trace(self):
        tr = Builder().build()
        # remove main_forward (CPU+GPU) in step 54 of rank 0 only
        ext = self._gpu(tr, "## forward 54 ##")["ts"]
        tr["traceEvents"] = [
            e
            for e in tr["traceEvents"]
            if not (
                e.get("name") == "## main_forward ##" and e["pid"] in (0, CPU_PID) and abs(e["ts"] - ext) < 20
            )
        ]
        res = run(tr, Builder().build())
        lay = [(i["rank"], i["step"]) for i in res["irregularities"] if i["check"] == "layout"]
        self.assertIn((0, 54), lay)

    def test_gpu_nesting_contradicts_cpu(self):
        tr = Builder().build()
        # move rank 0 step 53 main_forward GPU span out of its forward span (inside backward instead)
        g = [
            e
            for e in tr["traceEvents"]
            if e.get("cat") == "gpu_user_annotation" and e["name"] == "## main_forward ##" and e["pid"] == 0
        ][1]
        bw = self._gpu(tr, "## backward 53 ##")
        g["ts"], g["dur"] = bw["ts"] + 2, 3.0
        res = run(tr, Builder().build())
        self.assertTrue(any("contradicts the CPU annotation tree" in m for c, m in checks(res)))


class TestCli(unittest.TestCase):
    def test_main_exit_codes_and_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            ref, good, bad = (os.path.join(d, n) for n in ("ref.json.gz", "good.json", "bad.json"))
            with gzip.open(ref, "wt") as f:
                json.dump(Builder().build(), f)
            with open(good, "w") as f:
                json.dump(Builder(kernel_prefix="z").build(), f)
            b = Builder()
            b.drop_ann = True
            with open(bad, "w") as f:
                json.dump(b.build(), f)
            prof = os.path.join(d, "ref.profile.json")
            devnull = open(os.devnull, "w")
            old, olderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = devnull
            try:
                self.assertEqual(tlc.main([ref, "--save-ref-profile", prof]), 0)
                out = os.path.join(d, "v.json")
                rep = os.path.join(d, "v.txt")
                self.assertEqual(tlc.main([good, "--ref", prof, "--json", out, "--report", rep]), 0)
                with open(out) as f:
                    self.assertTrue(json.load(f)["conformant"])
                with open(rep) as f:
                    self.assertIn("CONFORMANT", f.read())
                self.assertEqual(tlc.main([bad, "--ref", ref, "--json", out]), 1)
                with open(out) as f:
                    v = json.load(f)
                self.assertFalse(v["conformant"])
                self.assertTrue(
                    all({"rank", "step", "track", "evidence"} <= set(i) for i in v["irregularities"])
                )
                self.assertEqual(tlc.main([os.path.join(d, "missing.json")]), 2)
            finally:
                sys.stdout, sys.stderr = old, olderr
                devnull.close()


if __name__ == "__main__":
    unittest.main()
