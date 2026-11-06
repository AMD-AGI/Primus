#!/usr/bin/env python3
"""Trace LAYOUT CONFORMANCE validator for stitched DLRMv4 traces.

Compares the *structure* (not the timing) of a stitched trace against a
reference "trunk" trace and flags irregular layout for investigation.
Kernel positions, kernel names and step latencies are expected to change
with optimizations; the checks below tolerate that and look only at layout:

  tracks      same CPU threads / GPU streams per rank, same naming style and
              row ordering as the reference; legacy rows ("GPU steps
              (even)/(odd)", cat ``gpu_step``) are flagged.
  steps       exactly one GPU ``ProfilerStep#N`` per exported step per rank,
              on the main compute stream only (layout-v2 standard), CPU
              ``ProfilerStep#N`` for each, consecutive numbers, expected count.
  layout      GPU annotation nesting (top two levels below the step) on the
              main stream identical across steps; GPU nesting consistent
              with the CPU annotation tree (External id); CPU annotation tree
              and main-stream annotation names compared with the reference
              (new names are listed as allowed differences, missing names or
              reordered common names are irregular).
  edges       no blank lead-in / tail on the main stream (> --edge-frac of
              the median step).
  extended    no step > --extended-ratio x the median of the other steps
              (GPU main-stream span and CPU step), with top CPU calls, idle
              gaps, barriers and capture notes as a first-pass explanation.
  integrity   no orphan GPU annotations, no annotation crossings, every
              main-stream work item launched in CPU step N lies inside GPU
              step N (correlation ids), no duplicated slices/rows.
  timing      per-step medians for candidate and reference (informational).

Usage:
  trace_layout_check.py CANDIDATE --ref TRUNK [--json OUT.json] [--report OUT.txt]
  trace_layout_check.py TRUNK --ref TRUNK          # intra-trace checks of the trunk
  trace_layout_check.py TRUNK --save-ref-profile P.json   # cache the reference
  trace_layout_check.py CANDIDATE --ref P.json     # reuse a cached reference

Exit status: 0 conformant, 1 irregular, 2 usage/load error.
CPU-only, standard library only.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import difflib
import gzip
import json
import re
import statistics
import sys
from typing import Any, Iterable

PID_STRIDE = 1_000_000           # stitch_traces._PID_STRIDE
WORK = {"kernel", "gpu_memcpy", "gpu_memset"}
GPU_ANN = "gpu_user_annotation"
CPU_ANN = "user_annotation"
STEP_RE = re.compile(r"^ProfilerStep#(\d+)$")
BARRIER_RE = re.compile(r"^(c10d::barrier|nccl:[a-z_]*barrier)$")
WINDOW_BOUNDARY_ANN = "## window_boundary ##"
SYNC_RE = re.compile(r"(Synchronize|aten::item|aten::_local_scalar_dense|hipStreamWaitEvent)")
LEGACY_ROW_RE = re.compile(r"GPU steps")
LEGACY_CAT = "gpu_step"
TOL_NS = 1_000                    # 1 us tolerance for containment/crossings
PROFILE_VERSION = 2
# Unnamed host helper threads that Kineto shows only when they make a HIP
# call inside the window (signature = the HIP calls they make). Their
# presence depends on sampling, so they are optional tracks.
#   nccl_watchdog: ProcessGroupNCCL watchdog loop (~100 ms tick), only
#                  hipThreadExchangeStreamCaptureMode.
#   pin_memory:    DataLoader pin-memory thread: is_pinned checks
#                  (PrimaryCtxGetState + PointerGetAttributes) and
#                  CachingHostAllocator misses (hipHostMalloc).
HELPER_THREADS = {
    "nccl_watchdog": frozenset({"hipThreadExchangeStreamCaptureMode"}),
    "pin_memory": frozenset({"hipDevicePrimaryCtxGetState", "hipPointerGetAttributes",
                             "hipHostMalloc", "hipHostFree", "hipHostRegister",
                             "hipHostUnregister"}),
}
ALLOC_RE = re.compile(r"hipMemMap|hipMalloc|hipMemCreate|hipMemSetAccess|hipExtMallocWithFlags")


def norm(name: Any) -> str:
    """Name with digit runs replaced by N (step numbers, tids, layer ids)."""
    return re.sub(r"\d+", "N", str(name))


def ns(us: float) -> int:
    return int(round(float(us) * 1000))


def ms(v_ns: float) -> float:
    return round(v_ns / 1e6, 3)


def median(v: Iterable[float]) -> float | None:
    v = list(v)
    return statistics.median(v) if v else None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_trace(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def split_ranks(events: list[dict]) -> dict[Any, list[dict]]:
    """Group events by rank (stitched pid = orig_pid + rank * PID_STRIDE).

    String pids (``Traces`` etc.) go under key ``"other"``.
    """
    out: dict[Any, list[dict]] = collections.defaultdict(list)
    for e in events:
        pid = e.get("pid")
        if isinstance(pid, int):
            out[pid // PID_STRIDE].append(e)
        else:
            try:
                out[int(pid) // PID_STRIDE].append(e)
            except (TypeError, ValueError):
                out["other"].append(e)
    return out


# --------------------------------------------------------------------------
# Per-rank model
# --------------------------------------------------------------------------

class Span:
    __slots__ = ("e", "name", "cat", "pid", "tid", "s", "t", "ext", "corr")

    def __init__(self, e: dict):
        self.e = e
        self.name = str(e.get("name", ""))
        self.cat = e.get("cat")
        self.pid = e.get("pid")
        self.tid = e.get("tid")
        self.s = ns(e.get("ts", 0))
        self.t = self.s + ns(e.get("dur", 0) or 0)
        a = e.get("args") or {}
        self.ext = a.get("External id")
        self.corr = a.get("correlation")

    @property
    def dur(self) -> int:
        return self.t - self.s


class Rank:
    """Structure of one rank's events."""

    def __init__(self, rank: Any, events: list[dict]):
        self.rank = rank
        self.meta = [e for e in events if e.get("ph") == "M"]
        self.x = [Span(e) for e in events if e.get("ph") == "X" and "ts" in e]
        self.flow_cats = collections.Counter(
            e.get("cat") for e in events if e.get("ph") in ("s", "f", "t"))
        self.proc_name: dict = {}
        self.proc_label: dict = {}
        self.proc_sort: dict = {}
        self.thread_names: dict = collections.defaultdict(list)
        self.thread_sort: dict = {}
        for m in self.meta:
            a = m.get("args") or {}
            pid, tid, n = m.get("pid"), m.get("tid"), m.get("name")
            if n == "process_name":
                self.proc_name[pid] = a.get("name")
            elif n == "process_labels":
                self.proc_label[pid] = a.get("labels")
            elif n == "process_sort_index":
                self.proc_sort[pid] = a.get("sort_index")
            elif n == "thread_name":
                self.thread_names[(pid, tid)].append(a.get("name"))
            elif n == "thread_sort_index":
                self.thread_sort[(pid, tid)] = a.get("sort_index")
        self.tracks: dict = collections.defaultdict(list)
        for sp in self.x:
            self.tracks[(sp.pid, sp.tid)].append(sp)
        self.gpu_pids = {sp.pid for sp in self.x if sp.cat in WORK or sp.cat == GPU_ANN}
        for pid, lab in self.proc_label.items():
            if str(lab).startswith("GPU"):
                self.gpu_pids.add(pid)
        self.cpu_pids = {sp.pid for sp in self.x if sp.pid not in self.gpu_pids}
        kc = collections.Counter((sp.pid, sp.tid) for sp in self.x if sp.cat == "kernel")
        self.main = max(kc, key=lambda k: (kc[k], -hash(k))) if kc else None
        # Steps
        self.cpu_steps: dict[int, list[Span]] = collections.defaultdict(list)
        self.gpu_steps: dict[int, list[Span]] = collections.defaultdict(list)
        for sp in self.x:
            m = STEP_RE.match(sp.name)
            if not m:
                continue
            if sp.cat == GPU_ANN:
                self.gpu_steps[int(m.group(1))].append(sp)
            elif sp.cat == CPU_ANN and sp.pid in self.cpu_pids:
                self.cpu_steps[int(m.group(1))].append(sp)
        self.main_thread = None
        for n, l in self.cpu_steps.items():
            self.main_thread = (l[0].pid, l[0].tid)
            break
        self.cpu_step_win = sorted(
            (l[0].s, l[0].t, n) for n, l in self.cpu_steps.items() if l)
        self._corr_step = None

    # -- helpers ------------------------------------------------------------
    def cpu_step_of(self, t_ns: int) -> int | None:
        starts = [w[0] for w in self.cpu_step_win]
        i = bisect.bisect_right(starts, t_ns) - 1
        if i >= 0 and t_ns < self.cpu_step_win[i][1]:
            return self.cpu_step_win[i][2]
        return None

    def corr_step(self) -> dict:
        """correlation id -> exported CPU step whose window contains the launch."""
        if self._corr_step is None:
            out = {}
            for sp in self.x:
                if sp.pid in self.cpu_pids and sp.corr is not None and sp.cat not in (CPU_ANN, GPU_ANN):
                    if sp.corr in out:
                        continue
                    out[sp.corr] = self.cpu_step_of(sp.s)
            self._corr_step = out
        return self._corr_step

    def launch_of(self) -> dict:
        out = {}
        for sp in self.x:
            if sp.pid in self.cpu_pids and sp.corr is not None and sp.cat == "cuda_runtime":
                out.setdefault(sp.corr, sp)
        return out

    def thread_role(self, key) -> str:
        pid, tid = key
        if pid in self.gpu_pids:
            return f"gpu:stream {tid}" + (" (main)" if key == self.main else "")
        if key == self.main_thread or (isinstance(pid, int) and tid == pid % PID_STRIDE):
            return "cpu:main"
        names = sorted({str(n) for n in self.thread_names.get(key, []) if n is not None})
        roles = sorted({m.group(1) for n in names for m in [re.search(r"\(([^)]*)\)\s*$", n)] if m}
                       - {"python"})
        if roles:
            return "cpu:" + "/".join(norm(r) for r in roles)
        sps = self.tracks.get(key, [])
        cats = sorted({str(sp.cat) for sp in sps})
        if cats == ["cuda_runtime"]:
            calls = {sp.name for sp in sps}
            for helper, sig in HELPER_THREADS.items():
                if calls <= sig:
                    return f"cpu:helper[{helper}]"
        return f"cpu:unnamed[{','.join(cats)}]"

    def main_work(self) -> list[Span]:
        if self.main is None:
            return []
        return sorted((sp for sp in self.tracks[self.main] if sp.cat in WORK), key=lambda s: s.s)

    def span_bounds(self) -> tuple[int, int]:
        return min(sp.s for sp in self.x), max(sp.t for sp in self.x)


# --------------------------------------------------------------------------
# Structure extraction (used for both candidate and reference)
# --------------------------------------------------------------------------

def track_profile(r: Rank) -> dict:
    """Normalized track structure of one rank."""
    keys = collections.Counter()
    detail = {}
    for key, sps in r.tracks.items():
        role = r.thread_role(key)
        keys[role] += 1
        cats = collections.Counter(sp.cat for sp in sps)
        names = collections.Counter(sp.name for sp in sps).most_common(3)
        detail.setdefault(role, []).append({
            "pid": key[0] % PID_STRIDE if isinstance(key[0], int) else key[0],
            "tid": key[1], "events": len(sps), "cats": dict(cats),
            "top_names": names,
            "steps": dict(collections.Counter(r.cpu_step_of(sp.s) for sp in sps))})
    # metadata-only GPU processes (empty device rows)
    empty_gpu = sorted(pid % PID_STRIDE for pid, lab in r.proc_label.items()
                       if str(lab).startswith("GPU") and not any(k[0] == pid for k in r.tracks))
    styles = collections.Counter()
    for pid, n in r.proc_name.items():
        styles["process_name:" + norm(n)] += 1
    for pid, n in r.proc_label.items():
        styles["process_labels:" + norm(n)] += 1
    for key, ns_ in r.thread_names.items():
        for n in ns_:
            styles["thread_name:" + norm(n)] += 1
    # Row ordering of tracks with events (sort by process then thread index).
    def sort_key(k):
        return (r.proc_sort.get(k[0], 1 << 60), r.thread_sort.get(k, 1 << 60), str(k))
    order = [r.thread_role(k) for k in sorted(r.tracks, key=sort_key)]
    # Track metadata anomalies
    dup_names = {f"{k[0] % PID_STRIDE}:{k[1]}": v for k, v in r.thread_names.items()
                 if len(set(map(str, v))) > 1}
    unnamed_tracks = sorted(f"{k[0] % PID_STRIDE}:{k[1]}" for k in r.tracks
                            if k not in r.thread_names and isinstance(k[0], int))
    legacy = [sp.name for sp in r.x if sp.cat == LEGACY_CAT][:3]
    legacy += [str(n) for k, v in r.thread_names.items() for n in v if LEGACY_ROW_RE.search(str(n))]
    legacy += [str(n) for n in r.proc_name.values() if LEGACY_ROW_RE.search(str(n))]
    return {"tracks": dict(keys), "detail": detail, "empty_gpu_pids": len(empty_gpu),
            "styles": dict(styles), "order": order, "dup_thread_names": dup_names,
            "unnamed_tracks": unnamed_tracks, "legacy_rows": legacy}


def nest(spans: list[Span], tie=None) -> list[tuple[Span, int, Span | None]]:
    """Laminar nesting: (span, depth, parent) in start order (1 us tolerance)."""
    order = sorted(spans, key=lambda s: (s.s, -s.t, tie(s) if tie else 0))
    stack: list[Span] = []
    out = []
    for sp in order:
        # pop until the top contains sp (ends at/after sp, within tolerance, and
        # sp does not start at the top's end)
        while stack and (sp.t > stack[-1].t + TOL_NS or (sp.s >= stack[-1].t and sp.dur > 0)):
            stack.pop()
        out.append((sp, len(stack), stack[-1] if stack else None))
        stack.append(sp)
    return out


def cpu_tree(r: Rank) -> dict:
    """CPU user_annotation parent/depth per span (per thread)."""
    parent, depth = {}, {}
    by_thread = collections.defaultdict(list)
    for sp in r.x:
        if sp.cat == CPU_ANN and sp.pid in r.cpu_pids:
            by_thread[(sp.pid, sp.tid)].append(sp)
    for key, sps in by_thread.items():
        for sp, d, p in nest(sps):
            parent[id(sp)] = p
            depth[id(sp)] = d
    # Roots on worker threads (autograd backward annotations) belong to the
    # innermost main-thread annotation covering them (e.g. "## backward N ##").
    main = by_thread.get(r.main_thread, [])
    mains = sorted((sp for sp in main if not STEP_RE.match(sp.name)), key=lambda s: s.s)
    starts = [sp.s for sp in mains]
    for key, sps in by_thread.items():
        if key == r.main_thread:
            continue
        for sp in sps:
            if parent[id(sp)] is not None:
                continue
            # Laminar nesting: every container of sp is on the ancestor chain
            # of the latest main-thread annotation starting at/before sp.
            i = bisect.bisect_right(starts, sp.s) - 1
            q = mains[i] if i >= 0 else None
            while q is not None and not (q.s <= sp.s and sp.t <= q.t + TOL_NS):
                q = parent.get(id(q))
            parent[id(sp)] = q
    return {"parent": parent, "depth": depth, "by_thread": by_thread}


def cpu_signature(r: Rank, tree: dict, step: int, max_depth: int = 3) -> tuple:
    """(thread role, depth, normalized name) of CPU annotations in a step window."""
    l = r.cpu_steps.get(step)
    if not l:
        return ()
    lo, hi = l[0].s, l[0].t
    seq = []
    for key, sps in sorted(tree["by_thread"].items(), key=lambda kv: r.thread_role(kv[0])):
        role = r.thread_role(key)
        inside = [sp for sp in sps if lo <= sp.s and sp.s < hi and not STEP_RE.match(sp.name)]
        base = min((tree["depth"][id(sp)] for sp in inside), default=0)
        for sp in sorted(inside, key=lambda s: (s.s, -s.t)):
            d = tree["depth"][id(sp)] - base
            if d < max_depth:
                seq.append((role, d, norm(sp.name)))
    return tuple(seq)


def collapse(seq: Iterable) -> tuple:
    """Run-length collapse consecutive identical items -> tuple of items."""
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return tuple(out)


def gpu_main_signature(r: Rank, step: int, max_depth: int = 2) -> tuple:
    """(depth, normalized name) of main-stream GPU annotations inside the step."""
    if r.main is None:
        return ()
    g = [sp for sp in r.gpu_steps.get(step, []) if (sp.pid, sp.tid) == r.main]
    if len(g) != 1:
        return ()
    st = g[0]
    anns = [sp for sp in r.tracks[r.main] if sp.cat == GPU_ANN and not STEP_RE.match(sp.name)
            and st.s - TOL_NS <= sp.s and sp.t <= st.t + TOL_NS]
    return tuple((d, norm(sp.name)) for sp, d, _ in nest(anns) if d < max_depth)


# --------------------------------------------------------------------------
# Reference profile
# --------------------------------------------------------------------------

def build_profile(trace: dict, label: str = "") -> dict:
    """Summarize a trace into the structure the candidate is compared with."""
    ranks = {k: Rank(k, v) for k, v in split_ranks(trace["traceEvents"]).items() if k != "other"}
    per_rank = {}
    gpu_sigs, cpu_sigs, gpu_names = collections.Counter(), collections.Counter(), collections.Counter()
    placement = collections.Counter()
    name_steps = collections.defaultdict(set)
    barrier_steps, leads, tails, n_pairs, cpu_names = set(), [], [], 0, set()
    side_names: set = set()
    step_ms = collections.defaultdict(list)
    cpu_ms = collections.defaultdict(list)
    for k, r in sorted(ranks.items(), key=lambda kv: kv[0]):
        tp = track_profile(r)
        tree = cpu_tree(r)
        steps = sorted(r.cpu_steps)
        for n in steps:
            gs = gpu_main_signature(r, n)
            gpu_sigs[gs] += 1
            cpu_sigs[collapse(cpu_signature(r, tree, n))] += 1
            for g in r.gpu_steps.get(n, []):
                placement["main" if (g.pid, g.tid) == r.main else "side"] += 1
                if (g.pid, g.tid) == r.main:
                    step_ms[n].append(ms(g.dur))
            cpu_ms[n].append(ms(r.cpu_steps[n][0].dur))
        cpu_names.update(norm(sp.name) for sp in r.x if sp.cat == CPU_ANN and sp.pid in r.cpu_pids)
        side_names.update(side_stream_names(r))
        for n in steps:
            n_pairs += 1
            for nm in main_step_names(r, n):
                gpu_names[nm] += 1
                name_steps[nm].add(n)
            c = r.cpu_steps[n][0]
            if _barriers_in(r, c.s, c.t):
                barrier_steps.add(n)
        e = check_edges(r, Findings(), 1e9)
        if e:
            leads.append(e["lead_blank_ms"])
            tails.append(e["tail_gap_ms"])
        per_rank[str(k)] = {"tracks": tp["tracks"], "empty_gpu_pids": tp["empty_gpu_pids"],
                            "styles": tp["styles"], "order": tp["order"],
                            "dup_thread_names": len(tp["dup_thread_names"]),
                            "unnamed_tracks": len(tp["unnamed_tracks"]),
                            "n_steps": len(steps)}
    mode = lambda c: c.most_common(1)[0][0] if c else ()
    return {
        "version": PROFILE_VERSION, "label": label, "n_ranks": len(ranks),
        "per_rank": per_rank,
        "gpu_sig": [list(x) for x in mode(gpu_sigs)],
        "cpu_sig": [list(x) for x in mode(cpu_sigs)],
        "gpu_main_names": sorted(gpu_names),
        "gpu_main_name_presence": {k: round(v / max(1, n_pairs), 3) for k, v in sorted(gpu_names.items())},
        "gpu_main_name_steps": {k: sorted(v) for k, v in sorted(name_steps.items())},
        "barrier_steps": sorted(barrier_steps),
        "cpu_names": sorted(cpu_names),
        "gpu_side_names": sorted(side_names),
        "lead_blank_ms_median": median(leads),
        "tail_gap_ms_median": median(tails),
        "step_placement": dict(placement),
        "gpu_step_ms_median": {str(n): median(v) for n, v in sorted(step_ms.items())},
        "cpu_step_ms_median": {str(n): median(v) for n, v in sorted(cpu_ms.items())},
    }


def main_step_names(r: Rank, step: int) -> set:
    """Normalized names of main-stream GPU annotations inside GPU step ``step``."""
    if r.main is None:
        return set()
    g = [sp for sp in r.gpu_steps.get(step, []) if (sp.pid, sp.tid) == r.main]
    if len(g) != 1:
        return set()
    st = g[0]
    return {norm(sp.name) for sp in r.tracks[r.main] if sp.cat == GPU_ANN and not STEP_RE.match(sp.name)
            and st.s - TOL_NS <= sp.s and sp.t <= st.t + TOL_NS}


def side_stream_names(r: Rank) -> set:
    """Normalized GPU annotation names on the side (non-main) GPU streams,
    exported steps only (lead-in spans excluded)."""
    out = set()
    for key, sps in r.tracks.items():
        if key[0] not in r.gpu_pids or key == r.main:
            continue
        for sp in sps:
            if sp.cat == GPU_ANN and not STEP_RE.match(sp.name) \
                    and not (sp.e.get("args") or {}).get("lead_in"):
                out.add(norm(sp.name))
    return out


def _tuplify(sig) -> tuple:
    return tuple(tuple(x) for x in sig)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

class Findings:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, check: str, severity: str, msg: str, *, rank=None, step=None,
            track=None, evidence=None, explanation=None):
        self.items.append({"check": check, "severity": severity, "message": msg,
                           "rank": rank, "step": step, "track": track,
                           "evidence": evidence or {}, "explanation": explanation})

    def irregular(self):
        return [i for i in self.items if i["severity"] == "irregular"]


def _profile_for_rank(ref: dict, rank) -> dict:
    pr = ref["per_rank"]
    if str(rank) in pr:
        return pr[str(rank)]
    # world-size mismatch: compare with rank 0 of the reference
    return pr[sorted(pr, key=lambda s: int(s))[0]]


def check_tracks(r: Rank, ref: dict, F: Findings):
    tp = track_profile(r)
    rp = _profile_for_rank(ref, r.rank)
    want, got = collections.Counter(rp["tracks"]), collections.Counter(tp["tracks"])
    for role in sorted(set(want) | set(got)):
        if role.startswith("cpu:helper[") and got[role] != want[role]:
            F.add("tracks", "allowed",
                  f"optional host helper track '{role}': {got[role]} vs trunk {want[role]} "
                  "(shown only when the thread makes a HIP call inside the window)",
                  rank=r.rank, track=role,
                  evidence={"tracks": tp["detail"].get(role, [])[:2]})
            continue
        if got[role] > want[role]:
            ev = tp["detail"].get(role, [])
            expl = None
            if role.startswith("cpu:unnamed"):
                top = collections.Counter()
                for d in ev:
                    top.update(dict(d["top_names"]))
                expl = ("unnamed host thread (no thread_name metadata) issuing only "
                        f"{','.join(sorted({c for d in ev for c in d['cats']}))}: "
                        f"{dict(top.most_common(3))}; likely a runtime/helper thread "
                        "(e.g. a HIP pointer-query or background worker) that trunk did not trace")
            F.add("tracks", "irregular",
                  f"extra track(s) '{role}': {got[role]} vs trunk {want[role]}",
                  rank=r.rank, track=role, evidence={"tracks": ev}, explanation=expl)
        elif got[role] < want[role]:
            F.add("tracks", "irregular",
                  f"missing track(s) '{role}': {got[role]} vs trunk {want[role]}",
                  rank=r.rank, track=role)
    if tp["empty_gpu_pids"] != rp["empty_gpu_pids"]:
        F.add("tracks", "irregular",
              f"metadata-only GPU process rows: {tp['empty_gpu_pids']} vs trunk {rp['empty_gpu_pids']}",
              rank=r.rank)
    ws, gs = set(rp["styles"]), set(tp["styles"])
    if gs - ws or ws - gs:
        F.add("tracks", "irregular", "track naming style differs from trunk",
              rank=r.rank, evidence={"new_styles": sorted(gs - ws), "missing_styles": sorted(ws - gs)})
    if tp["legacy_rows"]:
        F.add("tracks", "irregular", "legacy synthetic step rows present ('GPU steps (even/odd)' / cat gpu_step)",
              rank=r.rank, evidence={"examples": tp["legacy_rows"][:5]},
              explanation="older stitcher/postprocessing output; current layout drops these rows")
    ro = list(collapse(x.replace(" (main)", "") for x in rp["order"] if not x.startswith("cpu:helper[")))
    go = list(collapse(x.replace(" (main)", "") for x in tp["order"] if not x.startswith("cpu:helper[")))
    ro_c = [x for x in ro if x in go]
    go_c = [x for x in go if x in ro]
    if ro_c != go_c:
        F.add("tracks", "irregular", "track row ordering differs from trunk",
              rank=r.rank, evidence={"trunk": ro, "candidate": go})
    if len(tp["dup_thread_names"]) > rp["dup_thread_names"]:
        F.add("tracks", "irregular", "tracks with conflicting thread_name metadata (duplicated row labels)",
              rank=r.rank, evidence=tp["dup_thread_names"])
    main_ref = [x for x in rp["order"] if x.endswith("(main)")]
    main_got = [x for x in tp["order"] if x.endswith("(main)")]
    if main_ref and main_got and main_ref != main_got:
        F.add("tracks", "irregular", f"main compute stream differs: {main_got} vs trunk {main_ref}",
              rank=r.rank)
    return tp


def check_steps(r: Rank, F: Findings, expected_steps: int | None):
    nums = sorted(r.cpu_steps)
    track_main = f"{r.main[0] % PID_STRIDE}:{r.main[1]}" if r.main else None
    if expected_steps is not None and len(nums) != expected_steps:
        cap = _capture_info(_CTX.get("t") or {}, r.rank)
        expl = None
        if cap:
            expl = (f"capture metadata: exported_steps={cap.get('exported_steps')}, "
                    f"skipped_head_steps={cap.get('skipped_head_steps')}, recorded={cap.get('recorded_steps')}"
                    " (metadata may predate a later re-trim)")
        F.add("steps", "irregular", f"{len(nums)} CPU ProfilerSteps, expected {expected_steps}",
              rank=r.rank, evidence={"steps": nums}, explanation=expl)
    if nums and nums != list(range(nums[0], nums[0] + len(nums))):
        F.add("steps", "irregular", "CPU ProfilerStep numbers not consecutive",
              rank=r.rank, evidence={"steps": nums})
    for n in nums:
        if len(r.cpu_steps[n]) != 1:
            F.add("steps", "irregular", f"{len(r.cpu_steps[n])} CPU ProfilerStep#{n} spans",
                  rank=r.rank, step=n)
    for n in sorted(set(r.gpu_steps) | set(nums)):
        g = r.gpu_steps.get(n, [])
        on_main = [sp for sp in g if (sp.pid, sp.tid) == r.main]
        side = [sp for sp in g if (sp.pid, sp.tid) != r.main]
        if n not in r.cpu_steps:
            F.add("steps", "irregular", f"GPU ProfilerStep#{n} without a CPU ProfilerStep#{n}",
                  rank=r.rank, step=n)
        if len(on_main) != 1:
            F.add("steps", "irregular", f"{len(on_main)} GPU ProfilerStep#{n} on the main stream (want 1)",
                  rank=r.rank, step=n, track=track_main)
        if side:
            F.add("steps", "irregular",
                  f"GPU ProfilerStep#{n} also on side stream(s) (layout v2: main stream only)",
                  rank=r.rank, step=n,
                  track=sorted({f"{s.pid % PID_STRIDE}:{s.tid}" for s in side}))


def check_layout(r: Rank, ref: dict, tree: dict, F: Findings):
    nums = sorted(r.cpu_steps)
    gsigs = {n: gpu_main_signature(r, n) for n in nums}
    csigs = {n: cpu_signature(r, tree, n) for n in nums}
    track_main = f"{r.main[0] % PID_STRIDE}:{r.main[1]}" if r.main else None
    # (a) intra-trace consistency of the main-stream GPU layout
    c = collections.Counter(gsigs.values())
    if len(c) > 1:
        modal = c.most_common(1)[0][0]
        for n, s in gsigs.items():
            if s != modal:
                full = list(difflib.unified_diff([f"{x[0]}:{x[1]}" for x in modal],
                                                 [f"{x[0]}:{x[1]}" for x in s], lineterm="", n=0))[2:]
                d = full[:10]
                F.add("layout", _boundary_severity(r, n, full),
                      "main-stream GPU annotation layout (top 2 levels) differs from the other steps",
                      rank=r.rank, step=n, track=track_main, evidence={"diff_vs_modal_step": d},
                      explanation=_step_event_explanation(r, n, tree))
    # (a') intra-trace CPU annotation layout
    cc = collections.Counter(collapse(s) for s in csigs.values())
    if len(cc) > 1:
        modal = cc.most_common(1)[0][0]
        for n, s in csigs.items():
            if collapse(s) != modal:
                full = list(difflib.unified_diff(["/".join(map(str, x)) for x in modal],
                                                 ["/".join(map(str, x)) for x in collapse(s)],
                                                 lineterm="", n=0))[2:]
                d = full[:10]
                F.add("layout", _boundary_severity(r, n, full),
                      "CPU annotation tree (3 levels) differs from the other steps",
                      rank=r.rank, step=n, track="cpu", evidence={"diff_vs_modal_step": d},
                      explanation=_step_event_explanation(r, n, tree))
    # (b) GPU nesting consistent with CPU nesting
    viol = check_nesting_consistency(r, tree)
    if viol["count"]:
        F.add("layout", "irregular",
              f"{viol['count']} GPU annotation(s) whose GPU nesting contradicts the CPU annotation tree",
              rank=r.rank, track=viol["tracks"], evidence={"examples": viol["examples"]},
              explanation=("GPU spans are not nested under their CPU parent's GPU span; typical of "
                           "Kineto-native GPU annotation spans (not rebuilt from launches)"))
    # (c) vs reference
    if ref is None:
        return
    ref_g = _tuplify(ref["gpu_sig"])
    ref_c = _tuplify(ref["cpu_sig"])
    cmodal = collections.Counter(collapse(s) for s in csigs.values()).most_common(1)
    if cmodal:
        _compare_seq(F, r, cmodal[0][0], ref_c, "CPU annotation tree", lambda x: x[2],
                     lambda x: "/".join(map(str, x)), track="cpu")
    gmodal = collections.Counter(collapse(s) for s in gsigs.values()).most_common(1)
    pres = ref.get("gpu_main_name_presence") or {k: 1.0 for k in ref["gpu_main_names"]}
    core = {k for k, v in pres.items() if v > 0.5}
    occasional = set(pres) - core
    if r.main is not None:
        names = set()
        for n in nums:
            names |= main_step_names(r, n)
        new = sorted(names - set(pres))
        missing_core = sorted(core - names)
        missing_occ = sorted(occasional - names)
        if new:
            cpu_known = set(ref.get("cpu_names") or [])
            vis = sorted(x for x in new if x in cpu_known)
            nov = sorted(x for x in new if x not in cpu_known)
            if vis:
                F.add("layout", "trunk_difference",
                      "main-stream GPU spans for annotations trunk had on the CPU only "
                      "(trunk's Kineto spans omitted them; rebuilt spans show them)",
                      rank=r.rank, track=track_main, evidence={"names": vis})
            if nov:
                F.add("layout", "allowed",
                      "main-stream GPU annotation names absent from trunk entirely (optimization-driven?)",
                      rank=r.rank, track=track_main, evidence={"new": nov})
        if missing_core:
            F.add("layout", "irregular", "trunk main-stream GPU annotation names missing",
                  rank=r.rank, track=track_main, evidence={"missing": missing_core},
                  explanation="removed by an optimization (fusion/refactor) or lost by postprocessing")
        if missing_occ:
            steps_of = ref.get("gpu_main_name_steps", {})
            bs = ref.get("barrier_steps", [])
            F.add("layout", "trunk_difference",
                  "annotations present only in some trunk steps are absent",
                  rank=r.rank, track=track_main,
                  evidence={"absent": {k: steps_of.get(k) for k in missing_occ}, "trunk_barrier_steps": bs})
    if "gpu_side_names" in ref:
        got_side, want_side = side_stream_names(r), set(ref["gpu_side_names"])
        if got_side != want_side:
            F.add("layout", "trunk_difference",
                  "side-stream GPU annotation names differ from trunk "
                  "(rebuilt spans follow launches onto every stream; dist-init/"
                  "host-sync-free names are stack-specific)",
                  rank=r.rank, track="gpu:side",
                  evidence={"new": sorted(got_side - want_side),
                            "missing": sorted(want_side - got_side)})
    if gmodal and collapse(ref_g) != gmodal[0][0]:
        F.add("layout", "trunk_difference",
              "main-stream GPU nesting (top 2 levels) differs from trunk",
              rank=r.rank, track=track_main,
              evidence={"trunk_top_level": [x[1] for x in collapse(ref_g) if x[0] == 0][:12],
                        "candidate_top_level": [x[1] for x in gmodal[0][0] if x[0] == 0][:12]})


def _is_boundary_name(name: str) -> bool:
    return name == WINDOW_BOUNDARY_ANN or bool(BARRIER_RE.match(name))


def _boundary_severity(r: Rank, n: int, diff: list) -> str:
    """'expected' when step n holds the window-boundary barrier and the
    per-step layout difference consists only of the boundary/barrier labels."""
    c = r.cpu_steps.get(n)
    if not c or not _barriers_in(r, c[0].s, c[0].t):
        return "irregular"
    changed = [x[1:] for x in diff if x[:1] in "+-" and not x.startswith("@@")]
    # GPU lines are "depth:name", CPU lines "role/depth/name".
    names = [x.split("/", 2)[2] if x.count("/") >= 2 else x.split(":", 1)[-1] for x in changed]
    if changed and all(_is_boundary_name(nm) for nm in names):
        return "expected"
    return "irregular"


def _compare_seq(F, r, got, want, what, key, fmt, track=None):
    gk, wk = {key(x) for x in got}, {key(x) for x in want}
    new, missing = sorted(gk - wk), sorted(wk - gk)
    common = gk & wk
    g_c = [fmt(x) for x in got if key(x) in common]
    w_c = [fmt(x) for x in want if key(x) in common]
    if new:
        F.add("layout", "allowed", f"{what}: annotations not in trunk (optimization-driven?)",
              rank=r.rank, track=track, evidence={"new": new})
    if missing:
        F.add("layout", "irregular", f"{what}: trunk annotations missing", rank=r.rank, track=track,
              evidence={"missing": missing})
    if collapse(g_c) != collapse(w_c):
        d = list(difflib.unified_diff(list(collapse(w_c)), list(collapse(g_c)), lineterm="", n=0))[2:14]
        F.add("layout", "irregular", f"{what}: nesting/order of common annotations differs from trunk",
              rank=r.rank, track=track, evidence={"diff_trunk_to_candidate": d})


def check_nesting_consistency(r: Rank, tree: dict) -> dict:
    """Each GPU annotation should sit under the GPU span of its nearest CPU
    ancestor that has a span on the same stream, and under nothing else."""
    cpu_by_ext = {}
    for key, sps in tree["by_thread"].items():
        for sp in sps:
            if sp.ext is not None:
                cpu_by_ext.setdefault(sp.ext, sp)
    count, examples, tracks = 0, [], set()
    for key, sps in r.tracks.items():
        if key[0] not in r.gpu_pids:
            continue
        anns = [sp for sp in sps if sp.cat == GPU_ANN and not STEP_RE.match(sp.name)]
        if not anns:
            continue
        by_ext = collections.defaultdict(list)
        for sp in anns:
            by_ext[sp.ext].append(sp)
        tie = lambda s: tree["depth"].get(id(cpu_by_ext.get(s.ext)), 0)
        for sp, d, parent in nest(anns, tie=tie):
            c = cpu_by_ext.get(sp.ext)
            if c is None:
                continue
            anc, p = set(), tree["parent"].get(id(c))
            expected = None
            while p is not None:
                anc.add(p.ext)
                if expected is None and p.ext in by_ext:
                    # nearest CPU ancestor with a span on this stream, overlapping in time
                    for q in by_ext[p.ext]:
                        if q.s - TOL_NS <= sp.s and sp.t <= q.t + TOL_NS:
                            expected = q
                            break
                    if expected is None and any(q.t > sp.s and q.s < sp.t for q in by_ext[p.ext]):
                        expected = "escaped"
                p = tree["parent"].get(id(p))
            bad = expected == "escaped" or (parent is not None and parent.ext not in anc and sp.dur > TOL_NS)
            if bad:
                count += 1
                tracks.add(f"{key[0] % PID_STRIDE}:{key[1]}")
                if len(examples) < 5:
                    examples.append({"annotation": sp.name, "gpu_parent": parent.name if parent else None,
                                     "cpu_parent": tree["parent"].get(id(c)).name
                                     if tree["parent"].get(id(c)) else None,
                                     "escaped_cpu_parent_span": expected == "escaped",
                                     "track": f"{key[0] % PID_STRIDE}:{key[1]}"})
    return {"count": count, "examples": examples, "tracks": sorted(tracks)}


def _barriers_in(r: Rank, lo: int, hi: int) -> list[str]:
    out = []
    for sp in r.x:
        if sp.pid in r.cpu_pids and lo <= sp.s < hi and (BARRIER_RE.match(sp.name) or sp.name == WINDOW_BOUNDARY_ANN):
            out.append(sp.name)
    return sorted(set(out))


def _top_cpu_calls(r: Rank, lo: int, hi: int, k: int = 5) -> list:
    if r.main_thread is None:
        return []
    sps = [sp for sp in r.tracks[r.main_thread] if lo <= sp.s < hi and sp.cat in ("cpu_op", CPU_ANN, "cuda_runtime", "python_function")
           and not STEP_RE.match(sp.name)]
    nested = nest(sps)
    top = [sp for sp, d, _ in nested if d == 0]
    top.sort(key=lambda s: -s.dur)
    return [(sp.name, ms(sp.dur)) for sp in top[:k]]


WAIT_RE = re.compile(r"(poll|wait|Synchronize|barrier|Wait)", re.I)


def _top_self_time(r: Rank, lo: int, hi: int, k: int = 3) -> list:
    """Main-thread ops with the largest self time (dur minus direct children)."""
    if r.main_thread is None:
        return []
    sps = [sp for sp in r.tracks[r.main_thread] if lo <= sp.s < hi and not STEP_RE.match(sp.name)]
    selft = {}
    for sp, d, p in nest(sps):
        selft[id(sp)] = [sp, sp.dur]
        if p is not None and id(p) in selft:
            selft[id(p)][1] -= sp.dur
    top = sorted(selft.values(), key=lambda v: -v[1])[:k]
    return [(sp.name[:70], ms(v)) for sp, v in top]


def _dominant_chain(r: Rank, lo: int, hi: int, depth: int = 6) -> list:
    """Follow the longest main-thread call down while it covers >50% of its parent."""
    if r.main_thread is None:
        return []
    sps = [sp for sp in r.tracks[r.main_thread] if lo <= sp.s < hi and not STEP_RE.match(sp.name)]
    children = collections.defaultdict(list)
    roots = []
    for sp, d, p in nest(sps):
        (children[id(p)] if p is not None else roots).append(sp)
    chain, cur = [], max(roots, key=lambda s: s.dur, default=None)
    while cur is not None and len(chain) < depth:
        chain.append((cur.name[:70], ms(cur.dur)))
        nxt = max(children.get(id(cur), []), key=lambda s: s.dur, default=None)
        cur = nxt if nxt is not None and nxt.dur > 0.5 * cur.dur else None
    return chain


def _host_gaps(r: Rank, lo: int, hi: int) -> dict:
    """Largest window with no host event (any CPU thread) and with no HIP call
    on the step's main thread.

    The HIP gap is per thread: helper threads (the NCCL watchdog makes HIP
    calls every ~100 ms) would otherwise cap it at their tick period and make
    a long host wait read like a fixed 100 ms sleep.
    """
    ev = sorted((sp.s, sp.t) for sp in r.x if sp.pid in r.cpu_pids and sp.t > lo and sp.s < hi
                and not STEP_RE.match(sp.name))
    hip = sorted((sp.s, sp.t) for sp in r.x if (sp.pid, sp.tid) == r.main_thread
                 and sp.cat == "cuda_runtime" and sp.t > lo and sp.s < hi)

    def biggest(iv):
        best, cur = (0, lo, lo), lo
        for a, b in iv:
            if a > cur and a - cur > best[0]:
                best = (a - cur, cur, a)
            cur = max(cur, b)
        if hi > cur and hi - cur > best[0]:
            best = (hi - cur, cur, hi)
        return best

    def leaf_ops_covering(a, b):
        # deepest main-thread op that covers the gap midpoint
        mid = (a + b) // 2
        cov = [sp for sp in r.x if (sp.pid, sp.tid) == r.main_thread and sp.s <= mid < sp.t
               and not STEP_RE.match(sp.name)]
        cov.sort(key=lambda s: s.dur)
        return [c.name for c in cov[:3]]

    g_all, g_hip = biggest(ev), biggest(hip)
    return {"no_host_event_ms": ms(g_all[0]), "no_host_event_at_ms": ms(g_all[1] - lo),
            "no_hip_call_ms": ms(g_hip[0]), "no_hip_call_at_ms": ms(g_hip[1] - lo),
            "ops_covering_no_hip_window": leaf_ops_covering(g_hip[1], g_hip[2])}


def _main_idle_gaps(r: Rank, lo: int, hi: int, k: int = 3) -> list:
    w = [sp for sp in r.main_work() if sp.t > lo and sp.s < hi]
    gaps, cur, nxt = [], lo, None
    launch = r.launch_of()
    for sp in w:
        if sp.s > cur:
            gaps.append((sp.s - cur, cur, sp))
        cur = max(cur, sp.t)
    if hi > cur:
        gaps.append((hi - cur, cur, None))
    gaps.sort(key=lambda g: -g[0])
    out = []
    for g, at, after in gaps[:k]:
        d = {"gap_ms": ms(g), "at_ms": ms(at - lo)}
        if after is not None:
            d["next_kernel"] = after.name[:80]
            la = launch.get(after.corr)
            if la is not None:
                d["next_kernel_launch_lag_ms"] = ms(after.s - la.s)
                d["host_bound"] = (after.s - la.s) < 200_000 and la.s > at
        out.append(d)
    return out


def _capture_info(trace: dict, rank) -> dict:
    """traceRendering.trace_capture_by_rank[rank] (or traceCapture), if any."""
    tr = trace.get("traceRendering") or {}
    by = tr.get("trace_capture_by_rank") or {}
    return by.get(str(rank)) or trace.get("traceCapture") or {}


def _capture_notes(trace: dict, step: int, rank=None) -> list:
    """Capture-window events (window_boundary, metric_log, ...) noted at ``step``."""
    tr = trace.get("traceRendering") or {}
    pools = []
    by = tr.get("capture_window_events_by_rank") or {}
    if rank is not None and str(rank) in by:
        pools.append(by[str(rank)])
    for key in ("capture_window_events", "capture_notes", "notes"):
        if isinstance(tr.get(key), list):
            pools.append(tr[key])
    cap = trace.get("traceCapture") or {}
    if isinstance(cap.get("notes"), list):
        pools.append(cap["notes"])
    return [n for pool in pools for n in pool
            if isinstance(n, dict) and n.get("profiler_step") == step]


_CTX: dict = {}


def _step_event_explanation(r: Rank, n: int, tree: dict) -> str | None:
    l = r.cpu_steps.get(n)
    if not l:
        return None
    lo, hi = l[0].s, l[0].t
    bar = _barriers_in(r, lo, hi)
    parts = []
    if bar:
        parts.append(f"window-boundary/barrier in this step: {bar}")
    notes = _capture_notes(_CTX.get("t") or {}, n, r.rank)
    if notes:
        parts.append(f"capture notes: {[(x.get('kind'), x.get('profiler_step')) for x in notes]}")
    if n == min(r.cpu_steps):
        parts.append("first exported step (boundary step after the rank-align barrier)")
    if n == max(r.cpu_steps):
        parts.append("last exported step (export/teardown edge)")
    return "; ".join(parts) or None


def check_edges(r: Rank, F: Findings, frac: float, t_bounds=None):
    w = r.main_work()
    track_main = f"{r.main[0] % PID_STRIDE}:{r.main[1]}" if r.main else None
    on_main = sorted((sp for l in r.gpu_steps.values() for sp in l if (sp.pid, sp.tid) == r.main),
                     key=lambda s: s.s)
    if not w or not on_main:
        return {}
    med = median(sp.dur for sp in on_main)
    lo, hi = t_bounds or r.span_bounds()
    first, last = w[0].s, max(sp.t for sp in w)
    g0, g1 = on_main[0].s, on_main[-1].t
    lead_gap = first - lo
    # blank main-stream time between export start and first GPU step start
    blank_lead, cur = 0, lo
    for sp in w:
        if sp.s >= g0:
            break
        if sp.s > cur:
            blank_lead += sp.s - cur
        cur = max(cur, sp.t)
    if g0 > cur:
        blank_lead += g0 - cur
    tail_gap = hi - last
    blank_tail = max(0, hi - max(g1, max((sp.t for sp in w if sp.s < g1), default=g1)))
    cs = sorted(r.cpu_steps.items())
    cpu0 = cs[0][1][0].s if cs else None
    pre_cpu = [sp for sp in r.x if sp.pid in r.cpu_pids and sp.s < (cpu0 or lo)]
    corr = r.corr_step()
    prior_work = [sp for sp in w if sp.s < g0]
    prior_unowned = sum(1 for sp in prior_work if corr.get(sp.corr) is None)
    info = {"median_step_ms": ms(med), "lead_gap_ms": ms(lead_gap), "lead_blank_ms": ms(blank_lead),
            "tail_gap_ms": ms(tail_gap), "tail_blank_ms": ms(blank_tail),
            "first_cpu_step_to_first_gpu_step_ms": ms(g0 - cpu0) if cpu0 is not None else None}
    thr = frac * med
    if lead_gap > thr or blank_lead > thr:
        names = collections.Counter(sp.name for sp in pre_cpu if sp.cat in (CPU_ANN, "cpu_op")).most_common(4)
        expl = []
        if pre_cpu:
            expl.append(f"{len(pre_cpu)} host events precede the first CPU ProfilerStep by "
                        f"{ms((cpu0 or lo) - lo)} ms (kept lead-in / pre-step host work: {names})")
        if prior_work:
            expl.append(f"main stream runs {len(prior_work)} work items before the first GPU step "
                        f"({prior_unowned} launched outside any exported CPU step = previous-step GPU work)")
        cap = _capture_info(_CTX.get("t") or {}, r.rank)
        if cap.get("skipped_head_steps"):
            expl.append(f"capture skipped head step(s) {cap['skipped_head_steps']} (skip_head="
                        f"{cap.get('skip_head')}): their GPU work, trimmed from the export, most likely "
                        "occupies the main stream during the blank lead-in (gap filled by previous-step GPU work)")
        if cpu0 is not None and g0 - cpu0 > thr:
            expl.append(f"GPU step starts {ms(g0 - cpu0)} ms after its CPU step (host run-ahead queue: "
                        "the first exported step's GPU work waits behind earlier queued work)")
        if blank_lead > thr:
            expl.append(f"{ms(blank_lead)} ms of the lead-in is blank on the main stream")
        F.add("edges", "irregular",
              f"lead-in {ms(max(lead_gap, blank_lead))} ms > {frac:.0%} of median step {ms(med)} ms",
              rank=r.rank, track=track_main, evidence=info, explanation="; ".join(expl) or None)
    if tail_gap > thr:
        post = [sp for sp in r.x if sp.s > last]
        names = collections.Counter((sp.cat, sp.name) for sp in post).most_common(4)
        F.add("edges", "irregular",
              f"tail {ms(tail_gap)} ms after the last main-stream work > {frac:.0%} of median step",
              rank=r.rank, track=track_main, evidence=info,
              explanation=f"events after the last main-stream work: {names}")
    return info


def check_extended(r: Rank, F: Findings, ratio: float, min_excess_ms: float, tree: dict):
    track_main = f"{r.main[0] % PID_STRIDE}:{r.main[1]}" if r.main else None
    gpu = {n: [sp for sp in l if (sp.pid, sp.tid) == r.main] for n, l in r.gpu_steps.items()}
    gpu = {n: l[0] for n, l in gpu.items() if len(l) == 1}
    cpu = {n: l[0] for n, l in r.cpu_steps.items() if l}
    for kind, d in (("GPU", gpu), ("CPU", cpu)):
        if len(d) < 3:
            continue
        for n, sp in d.items():
            others = [o.dur for m, o in d.items() if m != n]
            med = median(others)
            if sp.dur > ratio * med and sp.dur - med > min_excess_ms * 1e6:
                c = cpu.get(n)
                lo, hi = (c.s, c.t) if c else (sp.s, sp.t)
                ev = {"step_ms": ms(sp.dur), "median_others_ms": ms(med),
                      "top_cpu_calls": _top_cpu_calls(r, lo, hi),
                      "dominant_cpu_chain": _dominant_chain(r, lo, hi, depth=10),
                      "top_self_time": _top_self_time(r, lo, hi),
                      "host_gaps": _host_gaps(r, lo, hi),
                      "main_stream_idle_gaps": _main_idle_gaps(r, sp.s, sp.t) if kind == "GPU" else
                      _main_idle_gaps(r, gpu[n].s, gpu[n].t) if n in gpu else []}
                expl = []
                hg = ev["host_gaps"]
                if hg["no_host_event_ms"] > 0.5 * (sp.dur - med) / 1e6:
                    expl.append(f"host freeze: {hg['no_host_event_ms']} ms with no host event on any "
                                "thread (no HIP calls; GC/page-fault/IO-like stall)")
                elif hg["no_hip_call_ms"] > 0.5 * (sp.dur - med) / 1e6:
                    expl.append(f"{hg['no_hip_call_ms']} ms with no HIP call on the main thread "
                                f"while it ran {hg['ops_covering_no_hip_window']}")
                syncs = sorted({s for s, _ in ev["top_cpu_calls"] if SYNC_RE.search(s)})
                if syncs:
                    expl.append(f"host synchronization among top calls: {syncs}")
                host_only = kind == "CPU" and n in gpu and gpu[n].dur <= ratio * median(
                    [o.dur for m, o in gpu.items() if m != n])
                if host_only:
                    expl.append("GPU step of the same number is NOT extended: host-side only "
                                "(the run-ahead queue hid it from the main stream)")
                ch = ev["dominant_cpu_chain"]
                ev["host_only"] = host_only
                ev["waits"] = bool(ch and WAIT_RE.search(ch[-1][0]))
                severity = "irregular"
                if (host_only and c is not None and _barriers_in(r, c.s, c.t)
                        and any(_is_boundary_name(a) for a, _ in ch)):
                    # The window-boundary barrier drains the run-ahead queue on
                    # the host; the GPU step is not extended.
                    severity = "expected"
                if ch and ch[0][1] > 0.5 * (sp.dur - med) / 1e6:
                    leaf = ch[-1][0]
                    kind_ = ("allocator growth (device memory map/alloc)"
                             if ALLOC_RE.search(leaf)
                             else "host blocked in a HIP call" if leaf.startswith(("hip", "cuda"))
                             else "long host call")
                    expl.append(f"{kind_}: {' > '.join(f'{a} ({b} ms)' for a, b in ch)}")
                if kind == "GPU":
                    hb = [g for g in ev["main_stream_idle_gaps"] if g.get("host_bound")]
                    if hb:
                        expl.append(f"main stream idle gap(s) waiting for host launches: "
                                    f"{[g['gap_ms'] for g in hb]} ms")
                x = _step_event_explanation(r, n, tree)
                if x:
                    expl.append(x)
                F.add("extended", severity,
                      f"{kind} step {n} is {sp.dur / med:.2f}x the median of the other steps",
                      rank=r.rank, step=n, track=track_main if kind == "GPU" else "cpu:main",
                      evidence=ev, explanation="; ".join(expl) or None)


def check_integrity(r: Rank, F: Findings):
    cpu_ext = {sp.ext for sp in r.x if sp.cat == CPU_ANN and sp.pid in r.cpu_pids}
    # Lead-in annotations (args.lead_in, from a skip_head export clipped by
    # stitch_traces.clip_lead_in) belong to the dropped head step; their CPU
    # side ran before the trace starts by design, and the clip moves a lost
    # link's External id to args.lead_in_external_id. Count them separately.
    def _is_lead(sp: Span) -> bool:
        a = sp.e.get("args") or {}
        return bool(a.get("lead_in")) or "lead_in_external_id" in a
    lead = [sp for sp in r.x if sp.cat == GPU_ANN and _is_lead(sp)]
    if lead:
        unlinked = sum(1 for sp in lead if sp.ext not in cpu_ext)
        steps = sorted({(sp.e.get("args") or {}).get("owner_step") for sp in lead} - {None})
        F.add("integrity", "allowed",
              f"{len(lead)} lead-in GPU annotation(s) of dropped head step(s) {steps} "
              f"({unlinked} without a CPU span, by design)",
              rank=r.rank, evidence={"lead_in_steps": steps, "unlinked": unlinked})
    orph = [sp for sp in r.x if sp.cat == GPU_ANN and sp.ext not in cpu_ext and not _is_lead(sp)]
    if orph:
        expl = None
        cs = sorted(r.cpu_steps.items())
        g0 = min((sp.s for l in r.gpu_steps.values() for sp in l), default=None)
        if cs:
            c0 = cs[0][1][0].s
            pre = sum(1 for sp in orph if (g0 is not None and sp.s < g0))
            nums = collections.Counter(int(m.group(1)) for sp in orph
                                       for m in [re.search(r" (\d+) ##$", sp.name)] if m)
            parts = []
            if pre:
                parts.append(f"{pre}/{len(orph)} lie before the first exported GPU step (lead-in)")
            old_steps = sorted(k for k in nums if k < cs[0][0])
            if old_steps:
                parts.append(f"names carry pre-window step numbers {old_steps}: CPU spans were trimmed "
                             "with the head step while their GPU spans were kept")
            if parts:
                expl = "; ".join(parts)
        F.add("integrity", "irregular", f"{len(orph)} orphan GPU annotation(s) (External id without CPU span)",
              rank=r.rank, evidence={"examples": sorted({sp.name for sp in orph})[:6],
                                     "tracks": sorted({f'{sp.pid % PID_STRIDE}:{sp.tid}' for sp in orph})},
              explanation=expl)
    # crossings among annotations/work on GPU tracks; and work drawn inside
    # other work on the same (in-order) stream, which adds a row too.
    n_cross, worst, ex = 0, 0, []
    w_cross, w_worst, w_ex = 0, 0, []
    nested, n_ex = 0, []
    for key, sps in r.tracks.items():
        if key[0] not in r.gpu_pids:
            continue
        tr = f"{key[0] % PID_STRIDE}:{key[1]}"
        items = sorted((sp for sp in sps if sp.cat == GPU_ANN or sp.cat in WORK),
                       key=lambda s: (s.s, -s.t, s.cat != GPU_ANN))
        stack: list[Span] = []
        for sp in items:
            while stack and stack[-1].t <= sp.s:
                stack.pop()
            if stack and sp.t > stack[-1].t:
                ov = stack[-1].t - sp.s
                if ov > TOL_NS and (sp.cat == GPU_ANN or stack[-1].cat == GPU_ANN):
                    n_cross += 1
                    worst = max(worst, ov)
                    if len(ex) < 4:
                        ex.append({"track": tr, "a": stack[-1].name[:60],
                                   "b": sp.name[:60], "overlap_us": round(ov / 1e3, 3)})
                elif ov > TOL_NS:
                    w_cross += 1
                    w_worst = max(w_worst, ov)
                    if len(w_ex) < 4:
                        w_ex.append({"track": tr, "a": stack[-1].name[:60],
                                     "b": sp.name[:60], "overlap_us": round(ov / 1e3, 3)})
                while stack and stack[-1].t < sp.t:
                    stack.pop()
            elif sp.cat in WORK and sp.dur > 0:
                outer = next((q for q in reversed(stack) if q.cat in WORK), None)
                if outer is not None and sp.s < outer.t:
                    nested += 1
                    if len(n_ex) < 4:
                        n_ex.append({"track": tr, "outer": outer.name[:60], "outer_corr": outer.corr,
                                     "inner": sp.name[:60], "inner_corr": sp.corr,
                                     "inner_starts_after_outer_us": round((sp.s - outer.s) / 1e3, 3)})
            stack.append(sp)
    if n_cross:
        F.add("integrity", "irregular", f"{n_cross} annotation crossing(s) > 1 us (max {ms(worst)} ms)",
              rank=r.rank, evidence={"examples": ex})
    if w_cross:
        F.add("integrity", "irregular",
              f"{w_cross} same-stream kernel/memcpy crossing(s) > 1 us (max {ms(w_worst)} ms)",
              rank=r.rank, evidence={"examples": w_ex},
              explanation="work on one in-order stream overlaps in the trace (raw ROCm timestamps); "
                          "draws a second row on that stream")
    if nested:
        F.add("integrity", "irregular",
              f"{nested} GPU work item(s) drawn inside another work item on the same stream",
              rank=r.rank, evidence={"examples": n_ex},
              explanation="an in-order stream cannot run two items at once; out-of-launch-order "
                          "ROCm timestamps nest a kernel under a kernel (an extra depth row)")
    # step spans vs own kernels
    corr = r.corr_step()
    track_main = f"{r.main[0] % PID_STRIDE}:{r.main[1]}" if r.main else None
    for n, l in sorted(r.gpu_steps.items()):
        g = [sp for sp in l if (sp.pid, sp.tid) == r.main]
        if len(g) != 1:
            continue
        st = g[0]
        own = [sp for sp in r.main_work() if corr.get(sp.corr) == n]
        if not own:
            F.add("integrity", "irregular", f"GPU ProfilerStep#{n} has no main-stream work launched in its CPU step",
                  rank=r.rank, step=n, track=track_main)
            continue
        out = [sp for sp in own if sp.s < st.s - TOL_NS or sp.t > st.t + TOL_NS]
        if out:
            F.add("integrity", "irregular",
                  f"GPU ProfilerStep#{n} does not contain {len(out)}/{len(own)} of its own main-stream work items",
                  rank=r.rank, step=n, track=track_main,
                  evidence={"outside_ms": round(sum(sp.dur for sp in out) / 1e6, 3),
                            "max_escape_ms": ms(max(max(st.s - sp.s, sp.t - st.t) for sp in out)),
                            "examples": [sp.name[:60] for sp in out[:3]]},
                  explanation="GPU step span is not derived from the step's own launches "
                              "(e.g. Kineto-native span cut at the CPU step boundary)")
        lo, hi = min(sp.s for sp in own), max(sp.t for sp in own)
        if st.s < lo - TOL_NS or st.t > hi + TOL_NS:
            foreign = [sp for sp in r.main_work() if st.s <= sp.s < st.t and corr.get(sp.corr) != n]
            F.add("integrity", "info",
                  f"GPU ProfilerStep#{n} extends beyond its own work (by {ms(max(0, lo - st.s))} ms head, "
                  f"{ms(max(0, st.t - hi))} ms tail; {len(foreign)} foreign work items inside)",
                  rank=r.rank, step=n, track=track_main)
    # duplicates
    seen = collections.Counter((sp.pid, sp.tid, sp.cat, sp.name, sp.s, sp.t) for sp in r.x)
    dups = {k: v for k, v in seen.items() if v > 1}
    if dups:
        F.add("integrity", "irregular", f"{sum(v - 1 for v in dups.values())} duplicated slice(s)",
              rank=r.rank, evidence={"examples": [f"{k[0] % PID_STRIDE}:{k[1]} {k[3][:50]} x{v}"
                                                  for k, v in list(dups.items())[:4]]})
    pm = collections.Counter((m.get("pid"), m.get("name")) for m in r.meta
                             if m.get("name") in ("process_name", "process_labels", "process_sort_index"))
    dpm = [f"{k[0] % PID_STRIDE}:{k[1]}" for k, v in pm.items() if v > 1]
    if dpm:
        F.add("integrity", "irregular", "duplicated process metadata rows", rank=r.rank,
              evidence={"examples": dpm[:6]})


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def check_trace(trace: dict, ref: dict | None, *, expected_steps: int | None = 5,
                edge_frac: float = 0.10, extended_ratio: float = 1.5,
                min_excess_ms: float = 5.0) -> dict:
    _CTX["t"] = trace
    F = Findings()
    ranks = {k: Rank(k, v) for k, v in split_ranks(trace["traceEvents"]).items() if k != "other"}
    edges, timing = {}, {"gpu": collections.defaultdict(list), "cpu": collections.defaultdict(list)}
    if ref is not None and ref.get("n_ranks") != len(ranks):
        F.add("tracks", "irregular", f"{len(ranks)} ranks vs trunk {ref.get('n_ranks')}")
    for k in sorted(ranks):
        r = ranks[k]
        if r.main is None:
            F.add("tracks", "irregular", "rank has no GPU kernels", rank=k)
            continue
        tree = cpu_tree(r)
        if ref is not None:
            check_tracks(r, ref, F)
        else:
            tp = track_profile(r)
            if tp["legacy_rows"]:
                F.add("tracks", "irregular", "legacy synthetic step rows present", rank=k,
                      evidence={"examples": tp["legacy_rows"][:5]})
        check_steps(r, F, expected_steps)
        check_layout(r, ref, tree, F)
        edges[k] = check_edges(r, F, edge_frac)
        check_extended(r, F, extended_ratio, min_excess_ms, tree)
        check_integrity(r, F)
        for n, l in r.gpu_steps.items():
            for sp in l:
                if (sp.pid, sp.tid) == r.main:
                    timing["gpu"][n].append(ms(sp.dur))
        for n, l in r.cpu_steps.items():
            timing["cpu"][n].append(ms(l[0].dur))
    # Explicit report of how trunk places GPU steps vs layout v2.
    if ref is not None:
        pl = ref.get("step_placement", {})
        F.add("steps", "trunk_difference",
              f"trunk GPU ProfilerStep placement: {pl.get('main', 0)} on main stream, {pl.get('side', 0)} on "
              "side streams; layout-v2 standard is main stream only",
              evidence=pl)
    origins = attribute_wait_steps(F, ranks, min_excess_ms)
    _aggregate_single_rank(F, len(ranks))
    timing_sum = {
        "candidate_gpu_step_ms_median": {n: median(v) for n, v in sorted(timing["gpu"].items())},
        "candidate_cpu_step_ms_median": {n: median(v) for n, v in sorted(timing["cpu"].items())},
        "candidate_gpu_step_ms_overall": median(x for v in timing["gpu"].values() for x in v),
    }
    if ref is not None:
        timing_sum["trunk_gpu_step_ms_median"] = ref.get("gpu_step_ms_median")
        timing_sum["trunk_cpu_step_ms_median"] = ref.get("cpu_step_ms_median")
        tv = [v for v in (ref.get("gpu_step_ms_median") or {}).values() if v is not None]
        timing_sum["trunk_gpu_step_ms_overall"] = median(tv)
    irr = F.irregular()
    return {"conformant": not irr, "n_ranks": len(ranks),
            "irregularities": irr,
            "allowed_differences": [i for i in F.items if i["severity"] == "allowed"],
            "expected": [i for i in F.items if i["severity"] == "expected"],
            "wait_origins": {str(k): v for k, v in origins.items()},
            "trunk_differences": [i for i in F.items if i["severity"] == "trunk_difference"],
            "info": [i for i in F.items if i["severity"] == "info"],
            "edges": {str(k): v for k, v in edges.items()},
            "timing": timing_sum,
            "groups": group_findings(irr)}


def _longest_host_blocks(r: Rank, lo: int, hi: int) -> list:
    """Main-thread HIP/op calls in [lo, hi) by self time that are not waits.

    A rank whose host is blocked in a non-wait call (allocator growth, a slow
    op) arrives late at the next collective; other ranks then show the time
    as a wait/poll. Returns [(self_ms, name, chain)] longest first.
    """
    if r.main_thread is None:
        return []
    sps = [sp for sp in r.tracks[r.main_thread] if lo <= sp.s < hi and not STEP_RE.match(sp.name)]
    selft, parent = {}, {}
    for sp, d, p in nest(sps):
        selft[id(sp)] = [sp, sp.dur]
        parent[id(sp)] = p
        if p is not None and id(p) in selft:
            selft[id(p)][1] -= sp.dur
    out = []
    for sp, v in selft.values():
        if sp.cat not in ("cuda_runtime", "cpu_op") or WAIT_RE.search(sp.name):
            continue
        chain, q = [], parent.get(id(sp))
        while q is not None:
            chain.append(q.name[:60])
            q = parent.get(id(q))
        out.append((ms(v), sp.name[:70], " > ".join(reversed(chain[:6]))))
    out.sort(key=lambda x: -x[0])
    return out[:3]


def attribute_wait_steps(F: Findings, ranks: dict, min_block_ms: float = 5.0):
    """Name the rank(s) that made other ranks wait in an extended step.

    For each step with an extended CPU step whose dominant host chain ends in
    a wait/poll, look at every rank's main thread in that CPU step for a long
    non-wait call (allocator growth, a slow op): at least half the longest
    such wait and at least ``min_block_ms``. Those ranks reach the following
    collective late; the waits on the other ranks are theirs.
    """
    waits: dict = collections.defaultdict(float)
    for i in F.items:
        if i["check"] == "extended" and i["message"].startswith("CPU") and i["evidence"].get("waits"):
            ch = i["evidence"].get("dominant_cpu_chain") or []
            if ch:
                waits[i["step"]] = max(waits[i["step"]], ch[-1][1])
    out = {}
    for n in sorted(waits):
        min_ms = max(min_block_ms, 0.5 * waits[n])
        blocks = []
        for k, r in ranks.items():
            c = r.cpu_steps.get(n)
            if not c:
                continue
            top = _longest_host_blocks(r, c[0].s, c[0].t)
            if top and top[0][0] >= min_ms:
                blocks.append((top[0][0], k, top[0][1], top[0][2]))
        blocks.sort(key=lambda b: (-b[0], str(b[1])))
        if not blocks:
            continue
        alloc = all(ALLOC_RE.search(b[2]) for b in blocks)
        names = sorted({b[2] for b in blocks})
        note = (f"cross-rank origin: rank(s) {sorted(b[1] for b in blocks)} blocked on the host in "
                f"{names} ({', '.join(f'r{b[1]} {b[0]} ms' for b in blocks[:6])}; under "
                f"{blocks[0][3]})"
                + ("; allocator growth (expandable-segment hipMemMap) waits for the device to "
                   "drain, so that rank reaches the next collective late and the other ranks' "
                   "poll/wait is spent waiting for it" if alloc else
                   "; the other ranks' poll/wait is spent waiting for these ranks at the next collective"))
        out[n] = {"ranks": [b[1] for b in blocks], "calls": names, "allocator": alloc,
                  "blocks": [{"rank": b[1], "ms": b[0], "name": b[2], "under": b[3]} for b in blocks]}
        for i in F.items:
            if i["check"] == "extended" and i["step"] == n and i["message"].startswith("CPU") \
                    and i["severity"] == "irregular":
                i["explanation"] = ((i["explanation"] + "; ") if i["explanation"] else "") + note
                i["evidence"]["cross_rank_origin"] = out[n]
    return out


def _aggregate_single_rank(F: Findings, n_ranks: int):
    """Annotate extended-step findings that hit a single rank."""
    by = collections.defaultdict(list)
    for i in F.items:
        if i["check"] == "extended":
            by[(i["step"], i["message"].split(" is ")[0])].append(i)
    for (step, _), l in by.items():
        if len(l) == 1 and n_ranks > 1:
            x = l[0]
            x["explanation"] = ((x["explanation"] + "; ") if x["explanation"] else "") + \
                f"single-rank (only rank {x['rank']} of {n_ranks})"
        elif n_ranks > 1 and len(l) == n_ranks:
            for x in l:
                x["explanation"] = ((x["explanation"] + "; ") if x["explanation"] else "") + "all ranks"
        if len(l) > 1:
            # Origin rank: the one whose top self-time op is real work while the
            # others' top self-time op is a wait/poll/sync (they wait for it).
            tops = {x["rank"]: (x["evidence"].get("top_self_time") or [("", 0)])[0] for x in l}
            waiting = [k for k, (n, _) in tops.items() if WAIT_RE.search(n)]
            working = [k for k, (n, _) in tops.items() if n and not WAIT_RE.search(n)]
            if waiting and len(working) == 1 and len(waiting) >= len(l) - 1:
                o = working[0]
                wn = collections.Counter(tops[k][0] for k in waiting).most_common(1)[0][0]
                note = (f"single-rank host stall: rank {o} spends {tops[o][1]} ms self time in "
                        f"'{tops[o][0]}' while {len(waiting)} other rank(s) wait in '{wn}' "
                        f"(~{median(tops[k][1] for k in waiting)} ms)")
                for x in l:
                    x["explanation"] = ((x["explanation"] + "; ") if x["explanation"] else "") + note


def group_findings(items: list[dict]) -> list[dict]:
    g = collections.OrderedDict()
    for i in items:
        msg = re.sub(r"\d+(\.\d+)?", "#", i["message"])
        per_step = i["check"] in ("extended", "layout")
        k = (i["check"], i["step"] if per_step else None, msg)
        x = g.setdefault(k, {"check": i["check"], "step": i["step"] if per_step else None,
                             "message": i["message"], "ranks": [], "steps": [],
                             "values_vary": False, "explanation": i["explanation"]})
        if i["rank"] not in x["ranks"]:
            x["ranks"].append(i["rank"])
        if i["step"] is not None and i["step"] not in x["steps"]:
            x["steps"].append(i["step"])
        if i["message"] != x["message"]:
            x["values_vary"] = True
    return list(g.values())


def fmt_ranks(rs: list) -> str:
    rs = [r for r in rs if r is not None]
    if not rs:
        return "trace"
    rs = sorted(rs)
    out, a = [], rs[0]
    for i in range(1, len(rs) + 1):
        if i == len(rs) or rs[i] != rs[i - 1] + 1:
            out.append(f"{a}" if a == rs[i - 1] else f"{a}-{rs[i - 1]}")
            if i < len(rs):
                a = rs[i]
    return "rank " + ",".join(out)


def render_report(label: str, ref_label: str, res: dict) -> str:
    L = [f"== Trace layout conformance: {label}", f"   reference: {ref_label}",
         f"   ranks: {res['n_ranks']}   verdict: {'CONFORMANT' if res['conformant'] else 'IRREGULAR'} "
         f"({len(res['irregularities'])} irregular findings in {len(res['groups'])} groups)"]
    t = res["timing"]
    L.append("-- step time (informational; median over ranks, ms)")
    L.append(f"   candidate GPU main-stream: {t['candidate_gpu_step_ms_median']}  overall {t['candidate_gpu_step_ms_overall']}")
    L.append(f"   candidate CPU ProfilerStep: {t['candidate_cpu_step_ms_median']}")
    if "trunk_gpu_step_ms_median" in t:
        L.append(f"   trunk GPU main-stream:     {t['trunk_gpu_step_ms_median']}  overall {t['trunk_gpu_step_ms_overall']}")
        L.append(f"   trunk CPU ProfilerStep:    {t['trunk_cpu_step_ms_median']}")
    e = res["edges"]
    if e:
        lead = [v.get("lead_blank_ms", 0) for v in e.values() if v]
        tail = [v.get("tail_gap_ms", 0) for v in e.values() if v]
        lag = [v.get("first_cpu_step_to_first_gpu_step_ms") for v in e.values() if v]
        if lead:
            L.append(f"-- edges: main-stream blank lead-in ms min/med/max {min(lead)}/{median(lead)}/{max(lead)}; "
                     f"tail {min(tail)}/{median(tail)}/{max(tail)}; first CPU->GPU step lag "
                     f"{min(lag)}/{median(lag)}/{max(lag)}")
    if res["groups"]:
        L.append("-- IRREGULARITIES (grouped across ranks)")
        for g in res["groups"]:
            st = (f" step {g['step']}" if g["step"] is not None else
                  f" steps {','.join(map(str, sorted(g['steps'])))}" if g.get("steps") else "")
            vv = " (values vary by rank; first shown)" if g.get("values_vary") else ""
            L.append(f" * [{g['check']}]{st} {fmt_ranks(g['ranks'])}: {g['message']}{vv}")
            if g["explanation"]:
                L.append(f"     why (first pass): {g['explanation']}")
        # one example with evidence per group
        L.append("-- evidence (first rank of each group)")
        seen = set()
        for i in res["irregularities"]:
            key = (i["check"], i["step"] if i["check"] in ("extended", "layout") else None,
                   re.sub(r"[\d.]+", "#", i["message"]))
            if key in seen:
                continue
            seen.add(key)
            ev = json.dumps(i["evidence"], default=str)
            L.append(f"   [{i['check']}] rank {i['rank']} step {i['step']} track {i['track']}: {ev[:600]}")
    if res.get("expected"):
        L.append("-- EXPECTED (window-boundary step; explained, not irregular)")
        for g in group_findings(res["expected"]):
            st = f" step {g['step']}" if g["step"] is not None else ""
            L.append(f" * [{g['check']}]{st} {fmt_ranks(g['ranks'])}: {g['message']}")
            if g["explanation"]:
                L.append(f"     why: {g['explanation'][:400]}")
    for title, k in (("ALLOWED DIFFERENCES vs trunk (listed, not irregular)", "allowed_differences"),
                     ("STRUCTURAL DIFFERENCES vs trunk (reported explicitly)", "trunk_differences")):
        items = res[k]
        if not items:
            continue
        L.append(f"-- {title}")
        seen = {}
        for i in items:
            key = (i["message"], json.dumps(i["evidence"], sort_keys=True, default=str))
            seen.setdefault(key, []).append(i["rank"])
        for (m, ev), rs in seen.items():
            L.append(f" * {fmt_ranks(rs)}: {m}: {ev[:500]}")
    info = res["info"]
    if info:
        c = collections.Counter(re.sub(r"[\d.]+", "#", i["message"]) for i in info)
        L.append("-- info")
        for m, n in c.most_common(6):
            L.append(f" * {n}x {m}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate", help="stitched trace (.json/.json.gz)")
    ap.add_argument("--ref", help="trunk trace or a profile saved with --save-ref-profile")
    ap.add_argument("--save-ref-profile", help="write the profile of CANDIDATE (as a reference) and exit")
    ap.add_argument("--json", help="write the JSON verdict here")
    ap.add_argument("--report", help="write the human report here")
    ap.add_argument("--expected-steps", type=int, default=5)
    ap.add_argument("--edge-frac", type=float, default=0.10)
    ap.add_argument("--extended-ratio", type=float, default=1.5)
    ap.add_argument("--min-excess-ms", type=float, default=5.0,
                    help="ignore extended steps whose excess over the median is below this")
    ap.add_argument("--label")
    a = ap.parse_args(argv)
    try:
        cand = load_trace(a.candidate)
    except Exception as exc:  # noqa: BLE001
        print(f"cannot load {a.candidate}: {exc!r}", file=sys.stderr)
        return 2
    if a.save_ref_profile:
        prof = build_profile(cand, a.label or a.candidate)
        with open(a.save_ref_profile, "w") as f:
            json.dump(prof, f, indent=1)
        print(f"wrote reference profile {a.save_ref_profile}")
        return 0
    ref, ref_label = None, "(none: intra-trace checks only)"
    if a.ref:
        if a.ref == a.candidate:
            ref = build_profile(cand, a.ref)
        else:
            try:
                obj = load_trace(a.ref)
            except Exception as exc:  # noqa: BLE001
                print(f"cannot load {a.ref}: {exc!r}", file=sys.stderr)
                return 2
            if "per_rank" in obj and obj.get("version") != PROFILE_VERSION:
                print(f"stale reference profile {a.ref} (version {obj.get('version')}, "
                      f"want {PROFILE_VERSION}); regenerate it with --save-ref-profile",
                      file=sys.stderr)
                return 2
            ref = obj if "per_rank" in obj else build_profile(obj, a.ref)
        ref_label = ref.get("label") or a.ref
    res = check_trace(cand, ref, expected_steps=a.expected_steps, edge_frac=a.edge_frac,
                      extended_ratio=a.extended_ratio, min_excess_ms=a.min_excess_ms)
    res["candidate"] = a.candidate
    res["reference"] = ref_label
    rep = render_report(a.label or a.candidate, ref_label, res)
    print(rep)
    if a.report:
        with open(a.report, "w") as f:
            f.write(rep + "\n")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1, default=str)
    return 0 if res["conformant"] else 1


if __name__ == "__main__":
    sys.exit(main())
