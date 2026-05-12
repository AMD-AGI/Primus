"""pilot.tools.trace_analyze — chrome-trace → bucketed evidence.

Authoritative protocol: ``skills/workflow/trace_analysis.md`` (v0.1; expect
2-3 iterations after the first real trace).

This is a **pure analyzer**: it reads the chrome-trace JSON files referenced
by ``trace_meta.json`` and writes:

  * ``state/runs/<run_id>/trace_analysis.md``     (human review)
  * ``state/runs/<run_id>/trace_analysis.json``   (DIAGNOSE input)
  * (optional) ``state/sessions/<sid>/trace_analyses/<run_id>.md`` symlink

It does NOT decide bottlenecks. Only buckets, ratios, and a small evidence
list. DIAGNOSE consumes the JSON and applies skill rules to derive a verdict.

CLI::

    python -m pilot.tools.trace_analyze run \\
        --trace-meta state/runs/<id>/profile/trace_meta.json \\
        [--patterns  pilot/tools/_trace_patterns.json] \\
        [--out-md    state/runs/<id>/trace_analysis.md] \\
        [--out-json  state/runs/<id>/trace_analysis.json] \\
        [--session   state/sessions/<sid>]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


_PILOT_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_PATTERNS: Path = _PILOT_ROOT / "tools" / "_trace_patterns.json"


# ---------------------------------------------------------------------------
# Errors / exit codes
# ---------------------------------------------------------------------------


class _TraceError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


_EXIT_OK = 0
_EXIT_STAGE_FAILED = 1
_EXIT_USAGE = 2
_EXIT_TOOL_ERROR = 3


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _resolve(p: str | Path) -> Path:
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return pp
    cwd_path = Path.cwd() / pp
    if cwd_path.exists():
        return cwd_path
    pilot_path = _PILOT_ROOT / pp
    return pilot_path if pilot_path.exists() else cwd_path


def _load_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


_BUCKET_ORDER_DEFAULT = [
    "comm_collective",
    "comm_moe_dispatch",
    "comm_alltoall_nvidia",
    "comm_allreduce_nvidia",
    "comm_allgather_nvidia",
    "comm_reduce_scatter_nvidia",
    "comm_p2p",
    "compute_attention",
    "compute_norm_act",
    "compute_gemm_fp8_grouped",
    "compute_gemm",
    "compute_fp8_prep",
    "compute_optim",
    "compute_other",
    "memcpy",
    "schedule_overhead",
]


def _classify_bucket_role(bid: str) -> str:
    """Map bucket id -> 'comm' | 'compute' | 'overhead' | 'other'.

    Used to compute the comm_ratio / compute_ratio headlines without having
    to hard-code bucket lists in the analyzer (so adding a new bucket to
    _trace_patterns.json automatically rolls up to the right total).
    """
    if bid.startswith("comm_"):
        return "comm"
    if bid.startswith("compute_"):
        return "compute"
    if bid in ("memcpy",):
        return "compute"
    if bid in ("schedule_overhead",):
        return "overhead"
    return "other"


def _load_patterns(path: str | Path | None) -> list[tuple[str, list[re.Pattern[str]]]]:
    p = _resolve(path) if path else _DEFAULT_PATTERNS
    if not p.exists():
        raise _TraceError("USAGE", f"patterns file not found: {p}")
    raw = _load_json(p)
    out: list[tuple[str, list[re.Pattern[str]]]] = []
    for item in raw.get("buckets", []):
        bid = item.get("id")
        if not bid:
            continue
        pats = [re.compile(s, re.IGNORECASE) for s in item.get("patterns") or []]
        out.append((bid, pats))
    return out


def _classify(name: str, patterns: list[tuple[str, list[re.Pattern[str]]]]) -> str | None:
    for bid, pats in patterns:
        for pat in pats:
            if pat.search(name):
                return bid
    return None


# ---------------------------------------------------------------------------
# Iter boundary
# ---------------------------------------------------------------------------


_PROFILER_STEP_RE = re.compile(r"^ProfilerStep#\d+$")


def _find_iter_boundary(events: list[dict[str, Any]]) -> tuple[int, int, str]:
    """Return (start_us, end_us, source) for the captured iteration.

    Strategy (in priority order):
    1. PyTorch profiler emits a top-level ``ProfilerStep#N`` user_annotation
       that wraps the entire ``schedule.step()`` window (CPU + GPU). When
       only ONE iter is captured (warmup=5, capture=1), this is the only
       reliable iter boundary.
    2. Two adjacent ``Optimizer.step#*`` X-events: use the gap (end-of-first
       to start-of-second) — this is the proper "iter = optim_done -> optim_start".
       Only valid if at least 2 captured iters.
    3. Single Optimizer marker: NOT a valid iter (it's just the optim window),
       fall through.
    4. Full GPU kernel timespan as last-resort fallback.
    """
    profiler_step_ranges: list[tuple[int, int]] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        name = e.get("name") or ""
        if not _PROFILER_STEP_RE.match(name):
            continue
        cat = (e.get("cat") or "").lower()
        # Prefer the cpu_op range (it spans full step). The gpu_user_annotation
        # mirrors it but on the GPU stream timeline.
        if cat not in ("user_annotation", "cpu_op"):
            continue
        ts = e.get("ts")
        dur = e.get("dur")
        if ts is not None and dur is not None:
            profiler_step_ranges.append((int(ts), int(ts) + int(dur)))
    if profiler_step_ranges:
        profiler_step_ranges.sort(key=lambda r: r[1] - r[0], reverse=True)
        s, e = profiler_step_ranges[0]
        return s, e, "profiler_step"

    optimizer_ranges: list[tuple[int, int]] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        name = (e.get("name") or "").lower()
        if "optimizer.step" not in name:
            continue
        ph = e.get("ph")
        ts = e.get("ts")
        dur = e.get("dur")
        if ph == "X" and ts is not None and dur is not None:
            optimizer_ranges.append((int(ts), int(ts) + int(dur)))
    optimizer_ranges.sort()
    if len(optimizer_ranges) >= 2:
        s0_end = optimizer_ranges[0][1]
        s1_start = optimizer_ranges[1][0]
        if s1_start > s0_end:
            return s0_end, s1_start, "optimizer_marker"

    kernel_ts = []
    for e in events:
        if not isinstance(e, dict):
            continue
        cat = (e.get("cat") or "").lower()
        if cat in ("kernel", "gpu_op", "gpu_user_annotation"):
            ts = e.get("ts")
            dur = e.get("dur") or 0
            if ts is not None:
                kernel_ts.append((int(ts), int(ts) + int(dur)))
    if not kernel_ts:
        return 0, 0, "empty"
    return min(s for s, _ in kernel_ts), max(e for _, e in kernel_ts), "wallclock_fallback"


# ---------------------------------------------------------------------------
# Phase (fwd / bwd / post-bwd / optim) boundary detection
# ---------------------------------------------------------------------------


def _find_phase_boundaries(
    events: list[dict[str, Any]], iter_start: int, iter_end: int
) -> dict[str, Any]:
    """Identify forward / backward / post-bwd-comm / optimizer windows in
    one iter using PyTorch + Megatron user-annotation markers.

    Algorithm:

    1. **Backward window** = [first ``autograd::engine::evaluate_function:*``
       cpu_op inside the iter, last such event's ``ts + dur``]. PyTorch's
       autograd engine ONLY emits these during the backward pass, so they
       are a tight, reliable bracket.
    2. **Optimizer window** = the union of every ``Optimizer.step#*``
       cpu_op / user_annotation event inside the iter. (Most workloads
       have one such event; we take the union to be defensive.)
    3. **Forward window** = ``[iter_start, bwd_start]``.
    4. **Post-bwd window** = ``[bwd_end, optim_start]`` — the gap typically
       used for grad reduction (DDP allreduce) or other epoch end work.
    5. **Optim end → iter end** is normally <1 ms (just the bookkeeping
       at the bottom of ``schedule.step``); we report it as
       `tail_ms` for completeness.

    Returns a dict with these fields (each ``*_us`` is None if the
    corresponding marker was not found):

    - ``source`` — one of ``"autograd_engine"`` / ``"missing"``
    - ``fwd_start_us``, ``fwd_end_us``, ``fwd_dur_us``
    - ``bwd_start_us``, ``bwd_end_us``, ``bwd_dur_us``
    - ``post_bwd_start_us``, ``post_bwd_end_us``, ``post_bwd_dur_us``
    - ``optim_start_us``, ``optim_end_us``, ``optim_dur_us``
    - ``tail_dur_us``  — iter_end - max(optim_end, bwd_end)
    """
    out: dict[str, Any] = {
        "source": "missing",
        "fwd_start_us": None, "fwd_end_us": None, "fwd_dur_us": None,
        "bwd_start_us": None, "bwd_end_us": None, "bwd_dur_us": None,
        "post_bwd_start_us": None, "post_bwd_end_us": None, "post_bwd_dur_us": None,
        "optim_start_us": None, "optim_end_us": None, "optim_dur_us": None,
        "tail_dur_us": None,
    }

    # ---- 1. Backward window ----
    bwd_starts: list[int] = []
    bwd_ends: list[int] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        if (e.get("cat") or "") != "cpu_op":
            continue
        name = e.get("name") or ""
        if not name.startswith("autograd::engine::evaluate_function:"):
            continue
        ts = e.get("ts")
        dur = e.get("dur") or 0
        if ts is None:
            continue
        ts_i = int(ts)
        if ts_i < iter_start or ts_i >= iter_end:
            continue
        bwd_starts.append(ts_i)
        bwd_ends.append(ts_i + int(dur))
    if bwd_starts and bwd_ends:
        bwd_start = min(bwd_starts)
        bwd_end = max(bwd_ends)
        out["source"] = "autograd_engine"
        out["bwd_start_us"] = bwd_start
        out["bwd_end_us"] = bwd_end
        out["bwd_dur_us"] = max(0, bwd_end - bwd_start)
    else:
        return out

    # ---- 2. Optimizer window ----
    opt_starts: list[int] = []
    opt_ends: list[int] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        name = e.get("name") or ""
        if "Optimizer.step" not in name:
            continue
        cat = (e.get("cat") or "").lower()
        if cat not in ("user_annotation", "cpu_op"):
            continue  # avoid double-counting gpu_user_annotation mirrors
        ts = e.get("ts")
        dur = e.get("dur") or 0
        if ts is None:
            continue
        ts_i = int(ts)
        if ts_i < iter_start or ts_i >= iter_end:
            continue
        opt_starts.append(ts_i)
        opt_ends.append(ts_i + int(dur))
    optim_start = min(opt_starts) if opt_starts else None
    optim_end = max(opt_ends) if opt_ends else None
    if optim_start is not None and optim_end is not None:
        out["optim_start_us"] = optim_start
        out["optim_end_us"] = optim_end
        out["optim_dur_us"] = max(0, optim_end - optim_start)

    # ---- 3. Forward window: [iter_start, bwd_start] ----
    out["fwd_start_us"] = iter_start
    out["fwd_end_us"] = bwd_start
    out["fwd_dur_us"] = max(0, bwd_start - iter_start)

    # ---- 4. Post-bwd window: [bwd_end, optim_start] ----
    if optim_start is not None and optim_start > bwd_end:
        out["post_bwd_start_us"] = bwd_end
        out["post_bwd_end_us"] = optim_start
        out["post_bwd_dur_us"] = optim_start - bwd_end

    # ---- 5. Tail window ----
    last_known_end = max(bwd_end, optim_end if optim_end is not None else 0)
    if iter_end > last_known_end:
        out["tail_dur_us"] = iter_end - last_known_end

    return out


# ---------------------------------------------------------------------------
# Overlap analysis
# ---------------------------------------------------------------------------


def _intervals_union_us(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals = sorted(intervals)
    out_total = 0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            out_total += cur_e - cur_s
            cur_s, cur_e = s, e
    out_total += cur_e - cur_s
    return out_total


def _intervals_intersection_us(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    if not a or not b:
        return 0
    a = sorted(a)
    b = sorted(b)
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


# ---------------------------------------------------------------------------
# Kernel-name shortener (for §8.2 cross-bucket top-N)
# ---------------------------------------------------------------------------


_NAME_NOISE_PREFIXES = (
    "void ",
    "std::enable_if<!(kattr_no_packed_fp32_ops_v<ck_tile::gfx950_t>), void>::type ",
)
_NAME_NOISE_SUBSTRINGS = (
    "at::native::(anonymous namespace)::",
    "at::native::",
    "transformer_engine::",
    "primus_turbo::",
    "::(anonymous namespace)::",
    "(anonymous namespace)::",
)


# Friendly tags we look for inside C++-mangled symbols (`_ZN...`); first
# match wins. Order: most-specific first.
_MANGLED_TAGS: tuple[tuple[str, str], ...] = (
    ("QuantGroupedGemmKernel", "ck_tile QuantGroupedGemmKernel"),
    ("FmhaBwdDQDKDVKernel", "ck_tile FmhaBwdDQDKDVKernel"),
    ("FmhaBwdOGradDotOKernel", "ck_tile FmhaBwdOGradDotOKernel"),
    ("FmhaFwdKernel", "ck_tile FmhaFwdKernel"),
    ("UniversalGemmKernel", "ck_tile UniversalGemmKernel"),
    ("primus_turbo", "primus_turbo"),
    ("transformer_engine", "transformer_engine"),
    ("cleanup", "cleanup"),
)


def _shorten_kernel_name(name: str, max_len: int = 80) -> str:
    """Pretty-print a chrome-trace kernel name for the global top-N table.

    Goal: keep the algorithmic identity (e.g. ``Cijk_Alik_Bljk_F8BS_..._MT256x256x128``,
    ``QuantGroupedGemmKernel``, ``ck_tile FmhaBwdDQDKDVKernel``) and drop
    template / namespace / argument-list noise. We are intentionally
    lossy — this is the human-readable column.
    """
    s = name.strip()

    # Itanium C++ mangling (`_ZN...`): walk the table of recognized tags
    # and try to dig out the GEMM tile shape sequence (MT<H>x<W>x<K>) so
    # the operator + tile fits in the table cell. Append A/B layout when
    # available so kernels that differ only in layout get distinct labels.
    if s.startswith("_Z"):
        for needle, label in _MANGLED_TAGS:
            if needle in s:
                tile = ""
                seq_idx = s.find("sequenceIJLi")
                if seq_idx > 0:
                    nums: list[str] = []
                    cur = seq_idx + len("sequenceIJLi")
                    while cur < len(s) and len(nums) < 3:
                        end = cur
                        while end < len(s) and s[end].isdigit():
                            end += 1
                        if end == cur:
                            break
                        nums.append(s[cur:end])
                        if s[end : end + 3] == "ELi":
                            cur = end + 3
                        else:
                            break
                    if len(nums) == 3:
                        tile = f"MT{nums[0]}x{nums[1]}x{nums[2]}"
                # ck_tile encodes A/B/C layouts after `tensor_layout4gemm`.
                # Three layout slots follow; each slot is either a literal
                # (`8RowMajor` / `11ColumnMajor`) or an Itanium substitution
                # ref (`SG_`, `SH_`, etc.) that points at a previously-named
                # type. We render up to 3 tokens. Refs are kept as-is in
                # the short name so kernels that differ only in which slot
                # gets the literal vs a ref are visually distinct.
                layout = ""
                lay_idx = s.find("tensor_layout4gemm")
                if lay_idx > 0:
                    region = s[lay_idx + len("tensor_layout4gemm") : lay_idx + 220]
                    tokens: list[str] = []
                    cur = 0
                    while cur < len(region) and len(tokens) < 3:
                        # Itanium length-prefixed identifier: <digits><name>
                        # We only care about the two known layouts.
                        if region[cur].isdigit():
                            num_end = cur
                            while num_end < len(region) and region[num_end].isdigit():
                                num_end += 1
                            try:
                                n = int(region[cur:num_end])
                            except ValueError:
                                break
                            ident = region[num_end : num_end + n]
                            if ident == "RowMajor":
                                tokens.append("R")
                            elif ident == "ColumnMajor":
                                tokens.append("C")
                            else:
                                tokens.append("?")
                            cur = num_end + n
                        elif region[cur] == "N":
                            # Nested-name `NSF_8RowMajorE`; skip the `NSx_`
                            # prefix and keep parsing.
                            m = re.match(r"NS[0-9A-Z]+_", region[cur:])
                            if not m:
                                cur += 1
                                continue
                            cur += m.end()
                        elif region[cur] == "S":
                            # Substitution ref (`SG_`, `SH_`, `SC_`, `S_`).
                            m = re.match(r"S[0-9A-Z]*_", region[cur:])
                            if not m:
                                break
                            tokens.append(m.group(0))
                            cur += m.end()
                        elif region[cur] == "E":
                            # End of nested-name; just step past.
                            cur += 1
                        else:
                            break
                    if tokens:
                        layout = "/".join(tokens)
                # ck_tile boolean template flags. There can be a series of
                # `LbN` bits both BEFORE and AFTER the layouts; we capture
                # up to 6 contiguous bits anywhere in a 200-char window
                # ending at the first `tensor_layout4gemm` occurrence to
                # disambiguate kernels that differ only in compile flags.
                flags = ""
                lay_pos = s.find("tensor_layout4gemm")
                # Region: 200 chars before layout (catches the leading
                # `Lb0Lb0Lb0Lb0` sequence) and 200 chars after.
                region = ""
                if lay_pos > 0:
                    region = s[max(0, lay_pos - 200) : lay_pos + 200]
                else:
                    region = s
                bits = re.findall(r"Lb([01])", region)
                if bits:
                    flags = "f" + "".join(bits[:8])
                pieces: list[str] = []
                if tile:
                    pieces.append(tile)
                if layout:
                    pieces.append(layout)
                if flags:
                    pieces.append(flags)
                short = f"{label} [{' '.join(pieces)}]" if pieces else label
                if len(short) > max_len:
                    short = short[: max_len - 1] + "…"
                return short

    for p in _NAME_NOISE_PREFIXES:
        if s.startswith(p):
            s = s[len(p) :]
    for sub in _NAME_NOISE_SUBSTRINGS:
        s = s.replace(sub, "")
    if "<" in s:
        head = s.split("<", 1)[0]
        head_compact = head.replace("ck_tile::", "ck_tile ")
        token_chunks = sum(1 for ch in (head_compact.count("_"), head_compact.count("::")))
        if token_chunks >= 1 and len(head_compact) >= 12:
            tile = ""
            tile_idx = s.find("MT")
            if tile_idx > 0 and tile_idx < 200:
                end = tile_idx
                while end < len(s) and s[end] not in (",", ">", " ", "<"):
                    end += 1
                tile = s[tile_idx:end]
            s = head_compact + (f" [{tile}]" if tile else "")
    if "(" in s:
        s = s.split("(", 1)[0]
    s = " ".join(s.split())
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


# ---------------------------------------------------------------------------
# Pipeline timeline (for §8.3)
# ---------------------------------------------------------------------------


_BUCKET_GLYPH: dict[str, str] = {
    "compute_gemm_fp8_grouped": "G",
    "compute_gemm": "g",
    "compute_attention": "A",
    "compute_norm_act": "N",
    "compute_fp8_prep": "F",
    "compute_optim": "O",
    "compute_other": "o",
    "comm_moe_dispatch": "M",
    "comm_collective": "C",
    "comm_alltoall_nvidia": "c",
    "comm_allreduce_nvidia": "c",
    "comm_allgather_nvidia": "c",
    "comm_reduce_scatter_nvidia": "c",
    "comm_p2p": "p",
    "memcpy": "m",
    "schedule_overhead": "s",
}
_GLYPH_BUBBLE = "."
_GLYPH_MIXED = "?"


_COMM_PRESENCE_THRESHOLD = 0.05  # if comm > 5% of cell wall, light it up
_CELL_MAJORITY_THRESHOLD = 0.30  # otherwise mark cell as "mixed"
_MAX_STREAM_ROWS = 8  # cap pipeline rows; overflow is grouped into 'other'


def _build_pipeline_timeline(
    iter_start_us: int,
    iter_end_us: int,
    per_event: list[tuple[int, int, str, str, int]],
    width: int = 80,
) -> dict[str, Any]:
    """Render the iter as a per-stream ASCII timeline.

    Output:

    - ``streams[]``: one entry per CUDA stream observed (sorted by
      total kernel wall-time desc). Each entry has ``glyphs`` (one char
      per cell, showing the majority bucket on THAT stream in THAT
      cell), plus ``wall_ms`` and the top-3 kernel names that ran on
      this stream.
    - ``comm_overlay_glyphs``: a comm-presence overlay; lights up the
      cell whenever ANY comm bucket ≥ 5% of the cell wall (across all
      streams). Survives the case where compute drowns out comm in the
      per-stream rows.
    - ``legend``: glyph → bucket id mapping.

    Each cell covers ``cell_us = (iter_end - iter_start) / width`` µs
    of wall time.
    """
    iter_dur = max(0, iter_end_us - iter_start_us)
    if iter_dur == 0 or width <= 0:
        return {
            "width_cells": 0,
            "cell_us": 0,
            "streams": [],
            "comm_overlay_glyphs": "",
            "legend": {},
        }
    width = max(1, min(width, 200))
    cell_us = max(1, iter_dur // width)

    # stream_id -> per-cell {bucket: us}, plus stream-level totals
    stream_cells: dict[int, list[dict[str, int]]] = defaultdict(
        lambda: [defaultdict(int) for _ in range(width)]
    )
    stream_total_us: dict[int, int] = defaultdict(int)
    stream_kernels: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # global per-cell (any stream) for the comm-presence overlay
    cell_bucket_us: list[dict[str, int]] = [defaultdict(int) for _ in range(width)]

    for s, e, bucket, name, tid in per_event:
        if e <= iter_start_us or s >= iter_end_us:
            continue
        s_clip = max(s, iter_start_us)
        e_clip = min(e, iter_end_us)
        dur_us = e_clip - s_clip
        if dur_us <= 0:
            continue
        stream_total_us[tid] += dur_us
        stream_kernels[tid][name] += dur_us
        i0 = (s_clip - iter_start_us) // cell_us
        i1 = (e_clip - iter_start_us - 1) // cell_us if e_clip > s_clip else i0
        i0 = max(0, min(width - 1, i0))
        i1 = max(0, min(width - 1, i1))
        for i in range(i0, i1 + 1):
            cell_lo = iter_start_us + i * cell_us
            cell_hi = cell_lo + cell_us
            ov = max(0, min(cell_hi, e_clip) - max(cell_lo, s_clip))
            if ov > 0:
                stream_cells[tid][i][bucket] += ov
                cell_bucket_us[i][bucket] += ov

    # ---- Per-stream rows -------------------------------------------------
    streams_sorted = sorted(stream_total_us.items(), key=lambda kv: kv[1], reverse=True)
    streams_view = streams_sorted[:_MAX_STREAM_ROWS]

    used_glyphs: set[str] = set()
    streams_out: list[dict[str, Any]] = []
    for tid, total_us in streams_view:
        cells = stream_cells[tid]
        glyphs: list[str] = []
        for i in range(width):
            bdict = cells[i]
            if not bdict:
                glyphs.append(" ")  # not active on this stream this cell
                continue
            top_bucket, top_us = max(bdict.items(), key=lambda kv: kv[1])
            coverage = top_us / cell_us if cell_us else 0
            if coverage < _CELL_MAJORITY_THRESHOLD and len(bdict) >= 2:
                glyphs.append(_GLYPH_MIXED)
                used_glyphs.add(_GLYPH_MIXED)
            else:
                g = _BUCKET_GLYPH.get(top_bucket, _GLYPH_MIXED)
                glyphs.append(g)
                used_glyphs.add(g)
        # Top 3 kernel names on this stream (by self_us)
        top_names = sorted(stream_kernels[tid].items(), key=lambda kv: kv[1], reverse=True)[:3]
        streams_out.append({
            "stream": tid,
            "wall_ms": round(total_us / 1000.0, 3),
            "wall_pct": round(total_us / iter_dur, 4) if iter_dur else 0.0,
            "glyphs": "".join(glyphs),
            "kernels_top": [
                {"name": _shorten_kernel_name(n), "self_ms": round(us / 1000.0, 3)}
                for n, us in top_names
            ],
        })

    # ---- Comm presence overlay ------------------------------------------
    overlay: list[str] = []
    for i in range(width):
        bdict = cell_bucket_us[i]
        if not bdict:
            overlay.append(_GLYPH_BUBBLE)
            continue
        comm_us = sum(us for b, us in bdict.items() if _classify_bucket_role(b) == "comm")
        if comm_us / cell_us >= _COMM_PRESENCE_THRESHOLD:
            comm_top = max(
                ((b, us) for b, us in bdict.items() if _classify_bucket_role(b) == "comm"),
                key=lambda kv: kv[1],
            )[0]
            g = _BUCKET_GLYPH.get(comm_top, _GLYPH_MIXED)
            overlay.append(g)
            used_glyphs.add(g)
        else:
            overlay.append(" ")

    legend: dict[str, str] = {}
    for g in sorted(used_glyphs):
        for b, gv in _BUCKET_GLYPH.items():
            if gv == g:
                legend[g] = b
                break
    if _GLYPH_BUBBLE in overlay or any(_GLYPH_BUBBLE in s["glyphs"] for s in streams_out):
        legend[_GLYPH_BUBBLE] = "bubble"
    if _GLYPH_MIXED in used_glyphs:
        legend[_GLYPH_MIXED] = "mixed (no bucket >30% of cell)"
    legend[" "] = "stream idle / comm < 5% of cell"

    return {
        "width_cells": width,
        "cell_us": cell_us,
        "streams": streams_out,
        "stream_count_total": len(streams_sorted),
        "comm_overlay_glyphs": "".join(overlay),
        "legend": legend,
    }


# ---------------------------------------------------------------------------
# Main analyze
# ---------------------------------------------------------------------------


def analyze_trace(
    trace_path: Path,
    patterns: list[tuple[str, list[re.Pattern[str]]]],
    *,
    pipeline_width: int = 80,
) -> dict[str, Any]:
    raw = _load_json(trace_path)
    events = raw.get("traceEvents") if isinstance(raw, dict) else raw
    if not isinstance(events, list):
        raise _TraceError("TOOL_ERROR", f"trace {trace_path} has no traceEvents list")

    iter_start_us, iter_end_us, iter_src = _find_iter_boundary(events)
    iter_dur_us = max(0, iter_end_us - iter_start_us)

    # Phase windows (fwd / bwd / post-bwd / optim) inside this iter.
    phases = _find_phase_boundaries(events, iter_start_us, iter_end_us)

    def _phase_for_us(t: int) -> str | None:
        """Tag a microsecond-timestamp with the phase it falls in.
        Returns one of fwd / bwd / post_bwd / optim / tail / None."""
        if phases.get("source") != "autograd_engine":
            return None
        if phases["fwd_start_us"] is not None and phases["fwd_start_us"] <= t < phases["fwd_end_us"]:
            return "fwd"
        if phases["bwd_start_us"] is not None and phases["bwd_start_us"] <= t < phases["bwd_end_us"]:
            return "bwd"
        if phases["post_bwd_start_us"] is not None and phases["post_bwd_start_us"] <= t < phases["post_bwd_end_us"]:
            return "post_bwd"
        if phases["optim_start_us"] is not None and phases["optim_start_us"] <= t < phases["optim_end_us"]:
            return "optim"
        return "tail"

    # Collect events that fall within the iter window.
    bucket_events: dict[str, list[tuple[int, int]]] = defaultdict(list)
    bucket_kernel_count: dict[str, int] = defaultdict(int)
    bucket_top: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    unknown_names: dict[str, dict[str, Any]] = {}
    all_kernel_intervals: list[tuple[int, int]] = []
    # Per-event record (s, e, bucket, name, tid) — used for the cross-bucket
    # full kernel list, per-stream pipeline rows, and comm-presence overlay.
    per_event: list[tuple[int, int, str, str, int]] = []
    # Cross-bucket per-name aggregates: name -> {bucket, wall_us(union),
    # self_us(sum), count, durs, intervals, streams}
    name_agg: dict[str, dict[str, Any]] = {}
    # Per-phase per-bucket intervals: phase -> bucket -> [(s, e), ...]
    phase_bucket_events: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    phase_bucket_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for e in events:
        if not isinstance(e, dict):
            continue
        cat = (e.get("cat") or "").lower()
        if cat not in ("kernel", "gpu_op"):
            continue
        ts = e.get("ts")
        dur = e.get("dur")
        if ts is None or dur is None:
            continue
        ts = int(ts)
        dur = int(dur)
        if iter_dur_us > 0 and (ts + dur < iter_start_us or ts > iter_end_us):
            continue
        clipped_s = max(ts, iter_start_us) if iter_dur_us > 0 else ts
        clipped_e = min(ts + dur, iter_end_us) if iter_dur_us > 0 else ts + dur
        clipped_dur = max(0, clipped_e - clipped_s)
        all_kernel_intervals.append((clipped_s, clipped_e))

        name = e.get("name") or "<unnamed>"
        bucket = _classify(name, patterns)
        if bucket is None:
            entry = unknown_names.setdefault(name, {"name": name, "self_time_us": 0, "count": 0})
            entry["self_time_us"] += clipped_dur
            entry["count"] += 1
            bucket = "compute_other"
        bucket_events[bucket].append((clipped_s, clipped_e))
        bucket_kernel_count[bucket] += 1
        top = bucket_top[bucket].setdefault(name, {"name": name, "self_time_us": 0, "count": 0})
        top["self_time_us"] += clipped_dur
        top["count"] += 1

        # Phase tagging: use kernel midpoint to assign a phase ownership
        # (cheap, deterministic, avoids splitting one kernel across phases
        # which would distort kernel_count).
        phase = _phase_for_us((clipped_s + clipped_e) // 2) if clipped_dur > 0 else None
        if phase is not None:
            phase_bucket_events[phase][bucket].append((clipped_s, clipped_e))
            phase_bucket_count[phase][bucket] += 1

        tid = e.get("tid")
        try:
            tid_int = int(tid) if tid is not None else -1
        except (TypeError, ValueError):
            tid_int = -1
        per_event.append((clipped_s, clipped_e, bucket, name, tid_int))
        agg = name_agg.setdefault(name, {
            "bucket": bucket,
            "self_us": 0,
            "count": 0,
            "durs": [],
            "intervals": [],
            "streams": set(),
        })
        agg["self_us"] += clipped_dur
        agg["count"] += 1
        agg["durs"].append(clipped_dur)
        agg["intervals"].append((clipped_s, clipped_e))
        agg["streams"].add(tid_int)

    # Bucket ratios (use SUM of kernel self-time per bucket; useful for
    # explaining "what was the GPU doing").
    bucket_self_us: dict[str, int] = {b: sum(e - s for s, e in iv) for b, iv in bucket_events.items()}
    total_kernel_us = sum(bucket_self_us.values())

    # Bucket-union (wall): comm/compute may overlap each other; for headline
    # ratios we want WALL fraction, not self-time fraction.
    bucket_wall_us: dict[str, int] = {
        b: _intervals_union_us(iv) for b, iv in bucket_events.items()
    }

    seen_buckets = sorted({*bucket_events.keys(), *_BUCKET_ORDER_DEFAULT})
    comm_buckets = [b for b in seen_buckets if _classify_bucket_role(b) == "comm"]
    compute_buckets = [b for b in seen_buckets if _classify_bucket_role(b) == "compute"]

    comm_intervals = sum((bucket_events[b] for b in comm_buckets), [])
    compute_intervals = sum((bucket_events[b] for b in compute_buckets), [])

    comm_wall_us = _intervals_union_us(comm_intervals)
    compute_wall_us = _intervals_union_us(compute_intervals)
    overlap_us = _intervals_intersection_us(comm_intervals, compute_intervals)

    all_kernel_wall_us = _intervals_union_us(all_kernel_intervals)
    bubble_us = max(0, iter_dur_us - all_kernel_wall_us) if iter_dur_us > 0 else 0

    def _ratio(num: int) -> float:
        return round(num / iter_dur_us, 4) if iter_dur_us > 0 else 0.0

    iter_ms = iter_dur_us / 1000.0
    longest_serial_comm_us = _longest_serial_window(comm_intervals, compute_intervals)

    # Per-bucket details
    buckets_out: dict[str, dict[str, Any]] = {}
    for b in _BUCKET_ORDER_DEFAULT:
        intervals = bucket_events.get(b) or []
        self_us = bucket_self_us.get(b, 0)
        wall_us = bucket_wall_us.get(b, 0)
        durs = [e - s for s, e in intervals]
        top_kernels = sorted(
            bucket_top.get(b, {}).values(),
            key=lambda x: x["self_time_us"],
            reverse=True,
        )[:5]
        buckets_out[b] = {
            "self_time_ms": round(self_us / 1000.0, 3),
            "self_time_pct": _ratio(self_us),
            "wall_time_ms": round(wall_us / 1000.0, 3),
            "wall_time_pct": _ratio(wall_us),
            "kernel_count": bucket_kernel_count.get(b, 0),
            "p50_kernel_ms": round(statistics.median(durs) / 1000.0, 3) if durs else 0.0,
            "p99_kernel_ms": round(_pct(durs, 99) / 1000.0, 3) if durs else 0.0,
            "top_kernels": [
                {
                    "name": k["name"],
                    "self_time_ms": round(k["self_time_us"] / 1000.0, 3),
                    "count": k["count"],
                }
                for k in top_kernels
            ],
        }

    ratios = {
        "comm_ratio": _ratio(comm_wall_us),
        "comm_collective_ratio": _ratio(bucket_wall_us.get("comm_collective", 0)),
        "comm_moe_dispatch_ratio": _ratio(bucket_wall_us.get("comm_moe_dispatch", 0)),
        "comm_p2p_ratio": _ratio(bucket_wall_us.get("comm_p2p", 0)),
        "compute_ratio": _ratio(compute_wall_us),
        "compute_gemm_ratio": _ratio(
            bucket_wall_us.get("compute_gemm_fp8_grouped", 0) + bucket_wall_us.get("compute_gemm", 0)
        ),
        "compute_attention_ratio": _ratio(bucket_wall_us.get("compute_attention", 0)),
        "compute_fp8_prep_ratio": _ratio(bucket_wall_us.get("compute_fp8_prep", 0)),
        "compute_norm_act_ratio": _ratio(bucket_wall_us.get("compute_norm_act", 0)),
        "compute_optim_ratio": _ratio(bucket_wall_us.get("compute_optim", 0)),
        "compute_other_ratio": _ratio(bucket_wall_us.get("compute_other", 0)),
        "memcpy_ratio": _ratio(bucket_wall_us.get("memcpy", 0)),
        "bubble_ratio": _ratio(bubble_us),
        "schedule_overhead_ratio": _ratio(bucket_wall_us.get("schedule_overhead", 0)),
    }

    overlap_pct_of_comm = round(overlap_us / comm_wall_us, 4) if comm_wall_us > 0 else 0.0

    # Coverage (self-time accounting; wall-time may double-count overlap)
    unknown_self_us = sum(v["self_time_us"] for v in unknown_names.values())
    unknown_ratio = round(unknown_self_us / total_kernel_us, 4) if total_kernel_us > 0 else 0.0

    warnings: list[str] = []
    if iter_src in ("wallclock_fallback", "single_optimizer", "empty"):
        warnings.append(f"iter_boundary_fallback: source={iter_src}; iter wallclock derived from kernel timespan")
    if iter_dur_us == 0:
        warnings.append("iter window is empty; no kernels matched the iter range")
    if unknown_ratio > 0.05:
        warnings.append(f"high unknown_ratio={unknown_ratio:.3f}; update _trace_patterns.json (top names listed below)")

    top_unknowns = sorted(
        unknown_names.values(), key=lambda x: x["self_time_us"], reverse=True
    )[:20]

    kernels_full: list[dict[str, Any]] = []
    for name, agg in name_agg.items():
        wall_us = _intervals_union_us(agg["intervals"])
        durs = agg["durs"]
        kernels_full.append({
            "name": name,
            "name_short": _shorten_kernel_name(name),
            "bucket": agg["bucket"],
            "wall_ms": round(wall_us / 1000.0, 3),
            "wall_pct": _ratio(wall_us),
            "self_ms": round(agg["self_us"] / 1000.0, 3),
            "calls": agg["count"],
            "p50_ms": round(statistics.median(durs) / 1000.0, 3) if durs else 0.0,
            "p99_ms": round(_pct(durs, 99) / 1000.0, 3) if durs else 0.0,
            "streams": sorted(agg["streams"]),
        })
    kernels_full.sort(key=lambda r: r["wall_ms"], reverse=True)
    # Disambiguate identical short names by appending `#N` (rank in
    # wall-desc order). Two passes: first count; second rewrite so even
    # the first occurrence carries `#1`. This makes it visually obvious
    # to the reader that several rows correspond to different mangled
    # symbols that share a short label (typical for PyTorch-generated
    # `vectorized_elementwise_kernel<...>` etc.).
    short_count: dict[str, int] = {}
    for k in kernels_full:
        short_count[k["name_short"]] = short_count.get(k["name_short"], 0) + 1
    seen_short: dict[str, int] = {}
    for k in kernels_full:
        ns = k["name_short"]
        if short_count.get(ns, 0) > 1:
            seen_short[ns] = seen_short.get(ns, 0) + 1
            k["name_short"] = f"{ns} #{seen_short[ns]}"

    # Map mangled-name -> disambiguated short name, then back-patch the
    # per-bucket drill-down so it shares the same labels as the full list.
    name_to_short = {k["name"]: k["name_short"] for k in kernels_full}
    for b, info in buckets_out.items():
        for tk in info.get("top_kernels", []):
            tk["name_short"] = name_to_short.get(tk["name"], _shorten_kernel_name(tk["name"]))

    pipeline_timeline = _build_pipeline_timeline(
        iter_start_us, iter_end_us, per_event, width=pipeline_width
    )

    # Compute / Comm / Overlap / Bubble decomposition (mutually exclusive
    # partition of iter_wallclock; the four numbers must sum to the iter).
    pure_compute_us = max(0, compute_wall_us - overlap_us)
    pure_comm_us = max(0, comm_wall_us - overlap_us)
    cost_breakdown = {
        "iter_wallclock_ms": round(iter_ms, 3),
        "pure_compute_ms": round(pure_compute_us / 1000.0, 3),
        "pure_compute_pct": _ratio(pure_compute_us),
        "pure_comm_ms": round(pure_comm_us / 1000.0, 3),
        "pure_comm_pct": _ratio(pure_comm_us),
        "overlap_ms": round(overlap_us / 1000.0, 3),
        "overlap_pct": _ratio(overlap_us),
        "bubble_ms": round(bubble_us / 1000.0, 3),
        "bubble_pct": _ratio(bubble_us),
        "compute_wall_ms": round(compute_wall_us / 1000.0, 3),
        "comm_wall_ms": round(comm_wall_us / 1000.0, 3),
        "longest_serialized_comm_ms": round(longest_serial_comm_us / 1000.0, 3),
    }
    decomp_sum_us = pure_compute_us + pure_comm_us + overlap_us + bubble_us
    if iter_dur_us > 0 and abs(decomp_sum_us - iter_dur_us) / iter_dur_us > 0.005:
        warnings.append(
            f"cost_breakdown_partition_drift: "
            f"sum(pure_compute+pure_comm+overlap+bubble)={decomp_sum_us / 1000.0:.2f} ms "
            f"vs iter_wallclock={iter_ms:.2f} ms "
            f"(drift={abs(decomp_sum_us - iter_dur_us) / 1000.0:.2f} ms)"
        )

    # Per-phase bucket roll-up (one summary block per phase). Each row in
    # the summary already contains avg_ms = self_ms / kernel_count, so the
    # renderer can drop straight into a markdown table.
    def _summarize_buckets(events_map: dict[str, list[tuple[int, int]]],
                           count_map: dict[str, int]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for b, iv in events_map.items():
            if not iv:
                continue
            wall_us = _intervals_union_us(iv)
            self_us = sum(e - s for s, e in iv)
            count = count_map.get(b, 0)
            durs = sorted(e - s for s, e in iv)
            p50 = durs[len(durs) // 2] if durs else 0
            # Clamp p99 index so small samples (n<100) still yield max-like
            # values rather than the smaller midpoint that `int(n*0.99)-1`
            # produces.
            p99 = durs[max(len(durs) // 2, min(len(durs) - 1, int(round(len(durs) * 0.99)) - 1))] if durs else 0
            rows.append({
                "bucket": b,
                "wall_ms": round(wall_us / 1000.0, 3),
                "self_ms": round(self_us / 1000.0, 3),
                "kernel_count": count,
                "avg_ms": round((self_us / count) / 1000.0, 4) if count else 0.0,
                "p50_ms": round(p50 / 1000.0, 3),
                "p99_ms": round(p99 / 1000.0, 3),
            })
        rows.sort(key=lambda r: r["wall_ms"], reverse=True)
        return rows

    phases_summary: dict[str, Any] = {
        "source": phases.get("source"),
        "fwd_ms": (phases["fwd_dur_us"] / 1000.0) if phases.get("fwd_dur_us") is not None else None,
        "bwd_ms": (phases["bwd_dur_us"] / 1000.0) if phases.get("bwd_dur_us") is not None else None,
        "post_bwd_ms": (phases["post_bwd_dur_us"] / 1000.0) if phases.get("post_bwd_dur_us") is not None else None,
        "optim_ms": (phases["optim_dur_us"] / 1000.0) if phases.get("optim_dur_us") is not None else None,
        "tail_ms": (phases["tail_dur_us"] / 1000.0) if phases.get("tail_dur_us") is not None else None,
        "buckets_overall": _summarize_buckets(bucket_events, bucket_kernel_count),
        "buckets_by_phase": {
            phase: _summarize_buckets(phase_bucket_events.get(phase, {}), phase_bucket_count.get(phase, {}))
            for phase in ("fwd", "bwd", "post_bwd", "optim", "tail")
        },
    }

    return {
        "schema_version": "0.4",
        "trace_path": str(trace_path),
        "iter_wallclock_ms": round(iter_ms, 3),
        "iter_boundary_source": iter_src,
        "total_kernel_count": sum(bucket_kernel_count.values()),
        "bubble_ms": round(bubble_us / 1000.0, 3),
        "buckets": buckets_out,
        "ratios": ratios,
        "kernels_full": kernels_full,
        "cost_breakdown": cost_breakdown,
        "phases": phases_summary,
        "pipeline_timeline": pipeline_timeline,
        "overlap": {
            "comm_compute_overlap_ms": round(overlap_us / 1000.0, 3),
            "comm_compute_overlap_pct_of_comm": overlap_pct_of_comm,
            "longest_serialized_comm_ms": round(longest_serial_comm_us / 1000.0, 3),
        },
        "memory": {
            "peak_alloc_mb": None,
            "peak_reserved_mb": None,
            "fragmentation_ratio": None,
        },
        "pipeline": {
            "pp_size": None,
            "p2p_send_recv_pairs": bucket_kernel_count.get("comm_p2p", 0) // 2,
            "estimated_bubble_pct": None,
        },
        "coverage": {
            "kernel_self_time_ms": round(total_kernel_us / 1000.0, 3),
            "unknown_self_time_ms": round(unknown_self_us / 1000.0, 3),
            "unknown_ratio": unknown_ratio,
        },
        "top_unknown_names": [
            {
                "name": v["name"],
                "self_time_ms": round(v["self_time_us"] / 1000.0, 3),
                "count": v["count"],
            }
            for v in top_unknowns
        ],
        "warnings": warnings,
    }


def _pct(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return s[f] + (s[c] - s[f]) * (k - f)


def _longest_serial_window(comm: list[tuple[int, int]], compute: list[tuple[int, int]]) -> int:
    """Largest contiguous comm window with NO compute overlap. Useful as a
    'are we synchronously bottlenecked on this collective?' indicator."""
    if not comm:
        return 0
    longest = 0
    compute_sorted = sorted(compute)
    for s, e in sorted(comm):
        # Subtract any compute overlap inside [s, e]
        cur = s
        cur_max = 0
        for cs, ce in compute_sorted:
            if ce < s:
                continue
            if cs > e:
                break
            if cs > cur:
                cur_max = max(cur_max, cs - cur)
            cur = max(cur, ce)
        cur_max = max(cur_max, e - cur)
        longest = max(longest, cur_max)
    return longest


# ---------------------------------------------------------------------------
# Per-rank merge
# ---------------------------------------------------------------------------


def run(
    trace_meta: dict[str, Any],
    *,
    patterns_path: str | Path | None = None,
    run_dir: Path | None = None,
    pipeline_width: int = 80,
) -> dict[str, Any]:
    patterns = _load_patterns(patterns_path)
    trace_files = trace_meta.get("trace_files") or []
    if not trace_files:
        raise _TraceError(
            "STAGE_FAILED",
            "trace_meta.trace_files is empty; profiler did not emit any trace. "
            "Re-run with profile enabled and confirm train_iters >= profile_step_end + 2.",
        )

    per_rank: list[dict[str, Any]] = []
    base_dir = run_dir if run_dir else Path(".")
    for tf in trace_files:
        path = base_dir / tf["path"]
        if not path.is_absolute():
            path = (base_dir / tf["path"]).resolve()
        analysis = analyze_trace(
            path,
            patterns,
            pipeline_width=pipeline_width,
        )
        analysis["rank"] = tf.get("rank")
        analysis["trace_bytes"] = tf.get("bytes")
        per_rank.append(analysis)

    if len(per_rank) == 1:
        report = per_rank[0]
    else:
        report = {
            "schema_version": "0.1",
            "iter_wallclock_ms": statistics.median([p["iter_wallclock_ms"] for p in per_rank]),
            "ranks": [p.get("rank") for p in per_rank],
            "per_rank": per_rank,
            "warnings": ["multi-rank merge not yet supported; per-rank analyses listed verbatim"],
        }
    report["run_id"] = trace_meta.get("run_id")
    return report


# ---------------------------------------------------------------------------
# Markdown render
# ---------------------------------------------------------------------------


def render_md(report: dict[str, Any], *, kernels_cost_list_cap: int | None = None) -> str:
    """Render the trace_analysis.md document per skill v0.4.

    Section names are STABLE — no internal numbering — so the report is
    reusable across consumers. ``kernels_cost_list_cap`` truncates the
    "Per-iter cost — full kernel list" section; default = unlimited.
    """
    lines: list[str] = []
    lines.append(f"# Trace Analysis — {report.get('run_id') or '<unknown>'}")
    lines.append("")
    iter_ms = report.get("iter_wallclock_ms", 0) or 0
    lines.append(f"**Iter wallclock**: {iter_ms:.2f} ms")
    # Per-phase wall-time breakdown (fwd / bwd / post-bwd-comm / optim / tail).
    phases = report.get("phases") or {}
    if phases.get("source") == "autograd_engine":
        def _fmt_phase(label: str, ms: float | None) -> str | None:
            if ms is None:
                return None
            pct = (ms / iter_ms) if iter_ms else 0
            return f"{label}={ms:.2f} ms ({pct:.1%})"
        chips = [
            _fmt_phase(lbl, phases.get(key))
            for lbl, key in (("fwd", "fwd_ms"), ("bwd", "bwd_ms"),
                             ("post_bwd_comm", "post_bwd_ms"),
                             ("optim", "optim_ms"), ("tail", "tail_ms"))
        ]
        chips_clean = [c for c in chips if c]
        if chips_clean:
            lines.append("**Per-phase wallclock**: " + " · ".join(chips_clean))
            lines.append("_phase boundaries: forward = iter_start → first `autograd::engine::evaluate_function:*`; "
                         "backward = first → last such marker; post_bwd_comm = bwd_end → optim_start; "
                         "optim = `Optimizer.step#*`._")
    elif phases.get("source") == "missing":
        lines.append("**Per-phase wallclock**: _phase markers not detected in trace (no `autograd::engine::evaluate_function:*` cpu_op events found within iter window)._")
    if "iter_boundary_source" in report:
        lines.append(f"**Iter boundary source**: `{report['iter_boundary_source']}`")
    if "total_kernel_count" in report:
        lines.append(f"**Kernels in iter**: {report['total_kernel_count']}")
    cov = report.get("coverage") or {}
    if cov:
        lines.append(f"**Coverage**: unknown_ratio={cov.get('unknown_ratio', 0):.3%}")
    lines.append("")

    buckets = report.get("buckets") or {}
    bubble_ms = report.get("bubble_ms", 0) or 0

    # ---- Per-iter cost — bucket roll-up --------------------------------
    def _render_bucket_table(rows: list[dict[str, Any]], label_for_total_ms: float) -> list[str]:
        """Render a single roll-up table; ``label_for_total_ms`` is the
        denominator for the wall-% column (iter wallclock or phase ms)."""
        body: list[str] = []
        body.append("| Bucket | wall_ms | wall % | self_ms | avg_ms/kernel | kernels | p50_ms | p99_ms |")
        body.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            pct = (r["wall_ms"] / label_for_total_ms) if label_for_total_ms else 0
            body.append(
                f"| `{r['bucket']}` | {r['wall_ms']:.2f} | {pct:.2%} | "
                f"{r['self_ms']:.2f} | {r['avg_ms']:.4f} | {r['kernel_count']} | "
                f"{r['p50_ms']:.3f} | {r['p99_ms']:.3f} |"
            )
        return body

    lines.append("## Per-iter cost — bucket roll-up")
    lines.append("")
    if phases.get("source") == "autograd_engine":
        lines.append("Three views: **overall** is the iter end-to-end roll-up; "
                     "**fwd / bwd / optim** isolate kernels whose midpoint falls inside that phase. "
                     "`avg_ms/kernel = self_ms / kernels`.")
    else:
        lines.append("`avg_ms/kernel = self_ms / kernels`. Per-phase tables not shown — phase markers were not detected.")
    lines.append("")
    lines.append("### Overall")
    lines.append("")
    overall_rows = (phases.get("buckets_overall") or [])
    if not overall_rows:
        # Fallback: synthesize from `buckets` (legacy path)
        for b in _BUCKET_ORDER_DEFAULT:
            d = buckets.get(b) or {}
            if not d.get("kernel_count"):
                continue
            overall_rows.append({
                "bucket": b,
                "wall_ms": d.get("wall_time_ms", 0) or 0,
                "self_ms": d.get("self_time_ms", 0) or 0,
                "kernel_count": d.get("kernel_count", 0),
                "avg_ms": (d.get("self_time_ms", 0) / d.get("kernel_count", 1))
                if d.get("kernel_count") else 0.0,
                "p50_ms": d.get("p50_kernel_ms", 0) or 0,
                "p99_ms": d.get("p99_kernel_ms", 0) or 0,
            })
        overall_rows.sort(key=lambda r: r["wall_ms"], reverse=True)
    lines.extend(_render_bucket_table(overall_rows, iter_ms))
    if bubble_ms > 0:
        bubble_pct = (bubble_ms / iter_ms) if iter_ms else 0
        lines.append(
            f"| `<bubble>` | {bubble_ms:.2f} | {bubble_pct:.2%} | {bubble_ms:.2f} | --- | --- | --- | --- |"
        )
    lines.append("")
    lines.append(f"_Iter wallclock total: **{iter_ms:.2f} ms** "
                 "(∑wall_ms ≈ iter ± ε due to overlap; bubble is wallclock − union(kernels))_")
    lines.append("")

    # Per-phase sub-tables
    if phases.get("source") == "autograd_engine":
        for phase_key, phase_label, ms_key in (
            ("fwd", "Forward", "fwd_ms"),
            ("bwd", "Backward", "bwd_ms"),
            ("post_bwd", "Post-bwd comm (grad reduction etc.)", "post_bwd_ms"),
            ("optim", "Optimizer", "optim_ms"),
        ):
            ph_rows = (phases.get("buckets_by_phase") or {}).get(phase_key) or []
            ph_ms = phases.get(ms_key) or 0
            if not ph_rows and ph_ms <= 0:
                continue
            lines.append(f"### {phase_label} ({ph_ms:.2f} ms wallclock)")
            lines.append("")
            if not ph_rows:
                lines.append("_No GPU kernels classified in this phase (likely all CPU work)._")
                lines.append("")
                continue
            lines.extend(_render_bucket_table(ph_rows, ph_ms))
            lines.append("")

    # ---- Per-iter cost — full kernel list ------------------------------
    kernels_full = report.get("kernels_full") or []
    if kernels_full:
        cap = kernels_cost_list_cap if kernels_cost_list_cap and kernels_cost_list_cap > 0 else None
        shown = kernels_full[:cap] if cap else kernels_full
        if cap:
            heading = (
                f"## Per-iter cost — full kernel list (showing top {len(shown)} of "
                f"{len(kernels_full)} distinct kernels by wall_ms)"
            )
        else:
            heading = f"## Per-iter cost — full kernel list ({len(shown)} distinct kernels)"
        lines.append(heading)
        lines.append("")
        lines.append("| # | kernel | bucket | wall_ms | wall % | calls | p50 | p99 | streams |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---|")
        for i, k in enumerate(shown, 1):
            streams_str = ",".join(str(s) for s in k.get("streams", []))
            lines.append(
                f"| {i} | `{k['name_short']}` | `{k['bucket']}` | "
                f"{k['wall_ms']:.2f} | {k['wall_pct']:.2%} | "
                f"{k['calls']} | {k['p50_ms']:.3f} | {k['p99_ms']:.3f} | {streams_str} |"
            )
        if cap and len(kernels_full) > cap:
            tail_wall = sum(r["wall_ms"] for r in kernels_full[cap:])
            lines.append(
                f"| --- | _… {len(kernels_full) - cap} more kernels …_ | --- | "
                f"{tail_wall:.2f} | --- | --- | --- | --- | --- |"
            )
        lines.append("")

    # ---- Compute / Comm / Overlap / Bubble decomposition ---------------
    cb = report.get("cost_breakdown") or {}
    if cb:
        lines.append("## Compute / Comm / Overlap / Bubble decomposition")
        lines.append("")
        lines.append("These four slices are **mutually exclusive** and partition iter wallclock exactly.")
        lines.append("")
        lines.append("| Slice | wall_ms | wall % | what it is |")
        lines.append("|---|---:|---:|---|")
        lines.append(
            f"| pure compute | {cb.get('pure_compute_ms', 0):.2f} | "
            f"{cb.get('pure_compute_pct', 0):.2%} | compute kernels live AND no comm kernels live |"
        )
        lines.append(
            f"| pure comm | {cb.get('pure_comm_ms', 0):.2f} | "
            f"{cb.get('pure_comm_pct', 0):.2%} | comm kernels live AND no compute kernels live |"
        )
        lines.append(
            f"| comm ∧ compute | {cb.get('overlap_ms', 0):.2f} | "
            f"{cb.get('overlap_pct', 0):.2%} | both compute and comm kernels live simultaneously |"
        )
        lines.append(
            f"| bubble | {cb.get('bubble_ms', 0):.2f} | "
            f"{cb.get('bubble_pct', 0):.2%} | no GPU kernel live (idle GPU) |"
        )
        lines.append("")
        lines.append("Aggregate (legacy union totals; double-count overlap):")
        lines.append("")
        lines.append("| Slice | wall_ms | wall % |")
        lines.append("|---|---:|---:|")
        lines.append(
            f"| total compute wall | {cb.get('compute_wall_ms', 0):.2f} | "
            f"{(cb.get('compute_wall_ms', 0) / iter_ms if iter_ms else 0):.2%} |"
        )
        lines.append(
            f"| total comm wall | {cb.get('comm_wall_ms', 0):.2f} | "
            f"{(cb.get('comm_wall_ms', 0) / iter_ms if iter_ms else 0):.2%} |"
        )
        lines.append(
            f"| longest serialized comm window | "
            f"{cb.get('longest_serialized_comm_ms', 0):.2f} | --- |"
        )
        lines.append("")

    # ---- Iter pipeline timeline (per stream) ---------------------------
    pl = report.get("pipeline_timeline") or {}
    if pl and pl.get("streams"):
        cell_ms = (pl.get("cell_us") or 0) / 1000.0
        width = pl.get("width_cells") or 0
        lines.append(
            f"## Iter pipeline timeline (per stream) — {width} cells × {cell_ms:.1f} ms each"
        )
        lines.append("")
        lines.append("```")
        max_label = max(
            len(f"stream {s['stream']:>3} →") for s in pl["streams"]
        ) if pl["streams"] else 12
        for s in pl["streams"]:
            label = f"stream {s['stream']:>3} →".ljust(max_label)
            tail = ", ".join(k["name"] for k in s.get("kernels_top", []))
            lines.append(
                f"  {label} {s['glyphs']}    {s['wall_ms']:.0f} ms  "
                f"({tail})" if tail else
                f"  {label} {s['glyphs']}    {s['wall_ms']:.0f} ms"
            )
        # Comm overlay row
        overlay = pl.get("comm_overlay_glyphs") or ""
        if overlay:
            label = "comm ≥5% →".ljust(max_label)
            lines.append(f"  {label} {overlay}    (cells where ANY comm bucket ≥ 5% of cell wall)")
        lines.append(f"  {'':<{max_label}} └{'─' * max(0, width - 2)}┘")
        start_label = "start of iter"
        end_label = "end of iter"
        gap = max(0, width - len(start_label) - len(end_label))
        lines.append(
            f"  {'':<{max_label}} {start_label}{' ' * gap}{end_label}"
        )
        lines.append("```")
        lines.append("")
        total_streams = pl.get("stream_count_total") or len(pl["streams"])
        if total_streams > len(pl["streams"]):
            lines.append(
                f"_Showing top {len(pl['streams'])} streams by wall_ms; "
                f"{total_streams - len(pl['streams'])} additional stream(s) omitted._"
            )
            lines.append("")
        if pl.get("legend"):
            items = [f"`{g}`={lbl}" for g, lbl in pl["legend"].items() if g.strip()]
            if " " in pl["legend"]:
                items.append(f"`<space>`={pl['legend'][' ']}")
            lines.append("Legend: " + ", ".join(items))
            lines.append("")

    # ---- Headline ratios -----------------------------------------------
    ratios = report.get("ratios") or {}
    if ratios:
        lines.append("## Headline ratios")
        lines.append("")
        lines.append("| Ratio | value |")
        lines.append("|---|---:|")
        for k, v in ratios.items():
            lines.append(f"| `{k}` | {v:.4f} |")
        lines.append("")

    # ---- Per-bucket drill-down -----------------------------------------
    lines.append("## Per-bucket drill-down (top 5 kernels per bucket)")
    lines.append("")
    for b in _BUCKET_ORDER_DEFAULT:
        d = buckets.get(b) or {}
        if not d.get("kernel_count"):
            continue
        lines.append(f"### `{b}` ({d.get('kernel_count', 0)} kernels, "
                     f"wall={d.get('wall_time_ms', 0):.2f} ms)")
        lines.append("")
        for k in d.get("top_kernels") or []:
            short = k.get("name_short") or _shorten_kernel_name(k["name"])
            lines.append(f"- `{short}`: {k['self_time_ms']:.2f} ms × {k['count']}")
        lines.append("")

    # ---- Top unknown ---------------------------------------------------
    unknowns = report.get("top_unknown_names") or []
    if unknowns:
        lines.append("## Top unknown kernels (extend `_trace_patterns.json`)")
        lines.append("")
        for u in unknowns[:20]:
            lines.append(f"- `{u['name']}`: {u['self_time_ms']:.2f} ms × {u['count']}")
        lines.append("")

    # ---- Warnings ------------------------------------------------------
    warnings = report.get("warnings") or []
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _cli() -> int:
    p = argparse.ArgumentParser(prog="pilot.tools.trace_analyze")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="Analyze the trace referenced by a trace_meta.json.")
    p_run.add_argument("--trace-meta", required=True)
    p_run.add_argument("--patterns", default=None)
    p_run.add_argument("--out-md", default=None)
    p_run.add_argument("--out-json", default=None)
    p_run.add_argument("--session", default=None, help="Session dir; will write trace_analyses/<run_id>.md symlink")
    p_run.add_argument("--pipeline-width", type=int, default=80,
                       help="ASCII pipeline timeline width in cells (40..200, default 80)")
    p_run.add_argument("--kernels-cost-list-cap", type=int, default=0,
                       help="Cap the 'Per-iter cost — full kernel list' to top N kernels by wall_ms "
                            "(0 = unlimited, default; recommended 50 for a quick skim)")
    args = p.parse_args()

    if args.cmd != "run":  # pragma: no cover
        return _EXIT_USAGE

    try:
        meta_path = _resolve(args.trace_meta)
        if not meta_path.exists():
            raise _TraceError("USAGE", f"trace_meta not found: {meta_path}")
        trace_meta = json.loads(meta_path.read_text())
        run_dir = meta_path.parent.parent  # state/runs/<id>/profile/trace_meta.json -> state/runs/<id>
        report = run(
            trace_meta,
            patterns_path=args.patterns,
            run_dir=run_dir,
            pipeline_width=args.pipeline_width,
        )
    except _TraceError as exc:
        _emit({"stage": "TRACE_ANALYZE", "status": "failed", "failure": {"kind": exc.kind, "message": str(exc)}})
        return _EXIT_USAGE if exc.kind == "USAGE" else _EXIT_STAGE_FAILED if exc.kind == "STAGE_FAILED" else _EXIT_TOOL_ERROR

    out_md_path = _resolve(args.out_md) if args.out_md else (run_dir / "trace_analysis.md")
    out_json_path = _resolve(args.out_json) if args.out_json else (run_dir / "trace_analysis.json")
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text(render_md(report, kernels_cost_list_cap=args.kernels_cost_list_cap))
    out_json_path.write_text(json.dumps(report, indent=2, default=str))

    if args.session:
        session_dir = _resolve(args.session)
        symlink_dir = session_dir / "trace_analyses"
        symlink_dir.mkdir(parents=True, exist_ok=True)
        link = symlink_dir / f"{trace_meta.get('run_id') or 'unknown'}.md"
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(out_md_path)
        except OSError as exc:  # pragma: no cover
            print(f"trace_analyze: could not create symlink {link}: {exc}", file=sys.stderr)

    _emit({
        "stage": "TRACE_ANALYZE",
        "status": "ok",
        "out_md": str(out_md_path),
        "out_json": str(out_json_path),
        "iter_wallclock_ms": report.get("iter_wallclock_ms"),
        "ratios": report.get("ratios"),
        "warnings": report.get("warnings"),
    })
    return _EXIT_OK


if __name__ == "__main__":
    sys.exit(_cli())
