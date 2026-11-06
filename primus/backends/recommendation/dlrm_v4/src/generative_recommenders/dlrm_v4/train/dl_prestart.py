"""Streaming DataLoader pre-start bookkeeping: startup timeline + read probes.

Two jobs, both one-time and startup-only (no per-step cost beyond a cached
bool check per worker fetch):

1. **Startup timeline** (always on, rank 0, ONE log line). The trainer's
   ``[boundary] train_ts=<first> train first-batch data-wait=`` line times the
   whole first ``train_pipeline.progress()`` call: the pipeline fill (the first
   batch pulls, i.e. the real data wait), the fx trace and the first
   forward/backward with every first-use JIT, tens of seconds on a cold
   container, and it was never split before. ``mark(name)`` records the first wall time
   of each named startup point; ``report()`` prints them relative to RUN_START
   (``set_run_start``) together with the worker-side first-sample-read stamps,
   so the wait splits into data vs compute. Rank 0 logs ``[dlstart] timeline
   ...`` (the line the readers parse). With ``DL_PRESTART_RANK_TIMELINE=1``
   (yambda_5b.gin item (109), gin default OFF) every other rank also logs the
   same fields as ``[dlstart] rank-timeline rank=<r> ...`` in a ``k:v ; ...``
   syntax that shares none of rank 0's anchors, so a slow first step names the
   rank that was slow (a rank-0-only line cannot tell a slow rank 0 from a rank
   0 waiting in a collective for another rank). Off, only rank 0 logs it.

2. **Worker first-read stamps** (the compliance proof for the pre-start).
   ``new_read_slot(name)`` allocates a shared double in the main process BEFORE
   the DataLoader forks; ``note_read(slot)`` is called by the dataset wrapper in
   the worker on its first fetch and stores the wall time once. With
   ``DL_PRESTART=1`` the worker pools are forked before INIT_STOP/RUN_START
   with an EMPTY sampler window, so no index reaches a worker and no sample is
   read, prefetched or collated before RUN_START; the report line shows
   ``workers_ready`` < 0 and ``train_first_read`` > 0 relative to RUN_START, and
   any read stamped before RUN_START is reported as a COMPLIANCE VIOLATION.

Wall times (``time.time()``) are used so worker and main-process stamps share
one clock (MLLOG ``time_ms`` is also wall time).
"""

from __future__ import annotations

import logging
import multiprocessing as _mp
import os
import time
from typing import Dict, List, Optional, Tuple

logger: logging.Logger = logging.getLogger(__name__)

# name -> first wall time (main process only).
_MARKS: Dict[str, float] = {}
# (label, wall) reference; RUN_START on a cold start, else loop entry.
_REF: Optional[Tuple[str, float]] = None
# slot name -> multiprocessing.Value('d'), created before the pools fork.
_READ_SLOTS: Dict[str, "_mp.sharedctypes.Synchronized"] = {}
_REPORTED: bool = False
# True until the first-step report; hot-path marks check this bool only.
ACTIVE: bool = True

# Report order (anything else marked is appended after these).
_ORDER: List[str] = [
    "prestart_begin",
    "prestart_workers_ready",
    "loop_entry",
    "dataset_built",
    "eval_pool_ready",
    "train_pools_ready",
    "first_progress_start",
    "pull1",
    "fx_begin",
    "idist0_begin",
    "idist0_done",
    "pull2",
    "fwd_begin",
    "fwd_end",
    "pull3",
    "enq3_done",
    "idist1_begin",
    "bwd_begin",
    "bwd_end",
    "opt_begin",
    "opt_end",
    "deq_wait_begin",
    "first_progress_done",
]


def set_run_start(wall: Optional[float] = None) -> None:
    """Record the RUN_START reference (call right after the RUN_START marker)."""
    global _REF
    _REF = ("run_start", time.time() if wall is None else float(wall))


def run_start_wall() -> Optional[float]:
    return _REF[1] if _REF is not None and _REF[0] == "run_start" else None


_EXPECT_RUN_START: List[bool] = [False]


def expect_run_start() -> None:
    """This process will log RUN_START (MLPerf cold start): from now on no
    sample may be handed out or prefetched until ``set_run_start`` ran."""
    _EXPECT_RUN_START[0] = True


def require_run_start(what: str) -> None:
    """Hard guard for the sampler release / cold prefetch: refuse to touch the
    dataset before RUN_START was logged (rules L549), instead of relying on
    call order."""
    if _EXPECT_RUN_START[0] and run_start_wall() is None:
        raise RuntimeError(
            f"[dlstart] {what} before RUN_START: the dataset must not be touched "
            "before the clock starts"
        )


# The reference data set (MLCommons reference b5e8e2a), relative to the data
# root: the hstu_cache_L<N> entries of its sha256sums_yambda_5b_cache.txt
# manifest, processed_5b/item_popularity.npy and the two shared_metadata item
# mappings its reader opens. scripts/stage_data_local.sh REF_FILES stages
# exactly this set; dl_prestart_reference_files_test.py checks they agree.
# "prestart" marks the files the trainer itself opens before RUN_START (dataset
# construction, as in the reference); anchor_ts and the PR 907 window index
# (ts_indices/ts_offsets) are read only after RUN_START.
_REF_CACHE_FILES: Tuple[Tuple[str, bool], ...] = (
    ("_READY", True),
    ("store_meta.json", True),
    ("flat_event_types.npy", True),
    ("flat_is_like.npy", True),
    ("flat_is_listen_plus.npy", True),
    ("flat_is_organic.npy", True),
    ("flat_is_skip.npy", True),
    ("flat_item_ids.npy", True),
    ("flat_played_ratio.npy", True),
    ("flat_timestamps.npy", True),
    ("flat_uid.npy", True),
    ("unique_uids.npy", True),
    ("user_start.npy", True),
    ("user_end.npy", True),
    ("positions_L{H}.npy", True),
    ("anchor_ts_L{H}.npy", False),
    # mlcommons/training PR 907 offline time-window index (user ruling
    # 2026-10-12: allowed, DL_WINDOW_INDEX=1). Read only after RUN_START.
    ("ts_indices_L{H}_W86400.npy", False),
    ("ts_offsets_L{H}_W86400.npy", False),
)
_REF_OTHER_FILES: Tuple[Tuple[str, bool], ...] = (
    ("processed_5b/item_popularity.npy", True),
    ("shared_metadata/artist_item_mapping.parquet", True),
    ("shared_metadata/album_item_mapping.parquet", True),
)


def reference_data_files(
    history_length: int = 4086, prestart_only: bool = False
) -> List[str]:
    """Reference data files relative to the data root (see _REF_CACHE_FILES)."""
    cache = f"processed_5b/hstu_cache_L{int(history_length)}"
    out = [
        f"{cache}/{name.format(H=int(history_length))}"
        for name, pre in _REF_CACHE_FILES
        if pre or not prestart_only
    ]
    out += [name for name, pre in _REF_OTHER_FILES if pre or not prestart_only]
    return out


def page_cache_snapshot(
    root: str, rank: int, history_length: Optional[int] = None
) -> Optional[str]:
    """Count page-cache-resident pages of the reference .npy files the trainer
    opens before RUN_START (``reference_data_files(prestart_only=True)``) and
    log them, separating the header page of each file from sample pages. Uses
    mincore(2) on a PROT_READ mapping, which reads nothing. Called right before
    RUN_START: after the launcher's cache flush, the sample-page count is the
    evidence that no data was read before the clock. Diagnostics only.

    Only explicit reference paths are touched (no directory glob), so no
    non-reference file in the data root is mapped before RUN_START (user rule
    2026-10-12). ``history_length`` defaults to $HISTORY_LENGTH (gin default
    4086)."""
    import ctypes

    import numpy as np

    try:
        if history_length is None:
            history_length = int(os.environ.get("HISTORY_LENGTH", "4086") or 4086)
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_long]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        page = os.sysconf("SC_PAGE_SIZE")
        prot_read, map_shared, chunk = 0x1, 0x01, 1 << 30
        files = [
            os.path.join(root, rel)
            for rel in reference_data_files(history_length, prestart_only=True)
            if rel.endswith(".npy") and os.path.isfile(os.path.join(root, rel))
        ]
        total = header = sample = 0
        hot: List[str] = []
        for p in files:
            size = os.path.getsize(p)
            if size == 0:
                continue
            total += size
            fd = os.open(p, os.O_RDONLY)
            try:
                addr = libc.mmap(None, size, prot_read, map_shared, fd, 0)
            finally:
                os.close(fd)
            if addr in (None, ctypes.c_void_p(-1).value):
                raise OSError(ctypes.get_errno(), f"mmap {p}")
            res = 0
            try:
                for off in range(0, size, chunk):
                    n = min(chunk, size - off)
                    vec = (ctypes.c_ubyte * ((n + page - 1) // page))()
                    if libc.mincore(ctypes.c_void_p(addr + off), n, vec) != 0:
                        raise OSError(ctypes.get_errno(), f"mincore {p}")
                    bits = np.frombuffer(vec, dtype=np.uint8) & 1
                    r = int(bits.sum())
                    if off == 0 and bits[0]:
                        header += page
                        r -= 1
                    res += r
            finally:
                libc.munmap(ctypes.c_void_p(addr), size)
            sample += res * page
            if res:
                hot.append(f"{os.path.basename(p)}:{res * page >> 10}KiB")
        line = (
            f"[dlstart] page cache at RUN_START (node of rank {rank}): sample pages "
            f"resident {sample >> 10} KiB, header pages {header >> 10} KiB, over "
            f"{len(files)} .npy files ({total / 2**30:.1f} GiB)"
            + (f"; resident: {', '.join(hot[:8])}" if hot else "")
        )
        logger.warning(line)
        return line
    except Exception as e:  # diagnostics only; never fails the run
        logger.warning("[dlstart] page-cache snapshot skipped: %r", e)
        return None


def mark(name: str) -> None:
    """Record the first wall time of a named startup point (idempotent)."""
    if name not in _MARKS:
        _MARKS[name] = time.time()


def marked(name: str) -> bool:
    return name in _MARKS


def new_read_slot(name: str):
    """Shared first-read stamp for one worker pool. MUST be created in the main
    process before that pool forks (fork inherits the shared page)."""
    slot = _READ_SLOTS.get(name)
    if slot is None:
        slot = _mp.get_context("fork").Value("d", 0.0)
        _READ_SLOTS[name] = slot
    return slot


def note_read(slot) -> None:
    """Worker side: stamp the first sample read of this pool (first writer wins)."""
    if slot is None:
        return
    with slot.get_lock():
        if slot.value == 0.0:
            slot.value = time.time()


def read_stamps() -> Dict[str, float]:
    """Pool name -> first-read wall time (pools that read nothing are omitted)."""
    out: Dict[str, float] = {}
    for name, slot in _READ_SLOTS.items():
        v = float(slot.value)
        if v > 0.0:
            out[name] = v
    return out


def compliance_violations() -> List[str]:
    """Read stamps that precede RUN_START (only meaningful on a cold start)."""
    rs = run_start_wall()
    if rs is None:
        return []
    return [
        f"{name}@{v - rs:+.3f}s" for name, v in sorted(read_stamps().items()) if v < rs
    ]


# Item (109): the per-rank startup timeline. Set by the gin binding through
# apply_env_bootstrap (gin default False -> "0"); a pre-set env var wins. Only
# "1" arms it; "0" (or unset) is the kill switch. Read at report() time.
RANK_TIMELINE_ENV: str = "DL_PRESTART_RANK_TIMELINE"


def rank_timeline_enabled() -> bool:
    """True iff ``DL_PRESTART_RANK_TIMELINE`` is "1" (item (109); off by default)."""
    return os.environ.get(RANK_TIMELINE_ENV, "0").strip() == "1"


def report(rank: int, prestarted: bool) -> Optional[str]:
    """One summary line (rank 0) + a violation warning on ANY rank. Idempotent.

    Rank 0 logs (and returns) the ``[dlstart] timeline`` line, unchanged. With
    ``DL_PRESTART_RANK_TIMELINE=1`` (item (109), off by default) every other
    rank logs the same fields as ``[dlstart] rank-timeline rank=<r> s rel.
    <ref> (prestart:on|off): name:+s ... ; split name:s ... ; verdict:OK|VIOLATION``
    (``k:v`` fields, ``;`` separators: no syntax shared with rank 0's line) and
    returns None. Off, other ranks log nothing beyond the violation warning."""
    global _REPORTED, ACTIVE
    ACTIVE = False
    if _REPORTED:
        return None
    _REPORTED = True
    bad = compliance_violations()
    if bad:
        logger.error(
            "[dlstart] COMPLIANCE VIOLATION rank=%d: sample read before RUN_START: %s",
            rank,
            ", ".join(bad),
        )
    if rank == 0:
        line = _timeline_line("[dlstart] timeline", prestarted, bad)
        logger.info(line)
        return line
    if not rank_timeline_enabled():
        return None
    # Same fields in a syntax that shares none of rank 0's anchors (no " | ", no
    # "k=v", no "compliance OK" / "COMPLIANCE VIOLATION"): the launcher capture
    # can glue another rank's fragment onto a torn rank-0 line, and the readers
    # must never take such a fragment for rank 0's values.
    body = (
        _timeline_line("", prestarted, bad)
        .replace(" | COMPLIANCE VIOLATION", " ; verdict:VIOLATION")
        .replace(" | compliance OK", " ; verdict:OK")
        .replace(" | ", " ; ")
        .replace("=", ":")
    )
    logger.info("[dlstart] rank-timeline rank=%s%s", rank, body)
    return None


def _timeline_line(tag: str, prestarted: bool, bad: List[str]) -> str:
    """The timeline fields after ``tag`` (rank 0's tag is ``[dlstart] timeline``)."""
    if _REF is not None:
        ref_label, ref = _REF
    else:
        ref_label, ref = "loop_entry", _MARKS.get("loop_entry", time.time())
    names = [n for n in _ORDER if n in _MARKS] + sorted(
        n for n in _MARKS if n not in _ORDER
    )
    parts = [f"{n}={_MARKS[n] - ref:+.3f}" for n in names]
    reads = read_stamps()
    parts += [f"{n}_first_read={v - ref:+.3f}" for n, v in sorted(reads.items())]
    # Split of the first progress(): data (start -> first batch at the main
    # process), pipeline fill (first -> third pull), the rest (fx trace +
    # first fwd/bwd/opt incl. first-use JIT; may include later pulls).
    split = []
    t0 = _MARKS.get("first_progress_start")
    t1 = _MARKS.get("pull1")
    tn = _MARKS.get("pull3", _MARKS.get("pull2", t1))
    te = _MARKS.get("first_progress_done")
    if t0 is not None and t1 is not None and te is not None:
        split.append(f"data_wait={t1 - t0:.3f}")
        split.append(f"fill_pulls={tn - t1:.3f}")
        split.append(f"after_pulls={te - tn:.3f}")
        split.append(f"first_progress={te - t0:.3f}")
    # Host-side sub-split of the first progress() (no syncs added; each segment
    # ends at a host return, so async GPU tails land in the next segment that
    # blocks on the device).
    for lab, a, b in (
        ("fx", "fx_begin", "idist0_begin"),
        ("idist0", "idist0_begin", "idist0_done"),
        ("fwd", "fwd_begin", "fwd_end"),
        ("enq3", "fwd_end", "enq3_done"),
        ("idist1", "idist1_begin", "bwd_begin"),
        ("bwd", "bwd_begin", "bwd_end"),
        ("opt", "opt_begin", "opt_end"),
        ("deq", "opt_end", "first_progress_done"),
    ):
        if a in _MARKS and b in _MARKS:
            split.append(f"{lab}={_MARKS[b] - _MARKS[a]:.3f}")
    return (
        tag
        + f" s rel. {ref_label} (prestart={'on' if prestarted else 'off'}): "
        + " ".join(parts)
        + (" | split " + " ".join(split) if split else "")
        + (" | COMPLIANCE VIOLATION" if bad else (" | compliance OK" if ref_label == "run_start" else ""))
    )


def _reset_for_tests() -> None:
    global _REF, _REPORTED
    global ACTIVE
    _MARKS.clear()
    _READ_SLOTS.clear()
    _REF = None
    _REPORTED = False
    ACTIVE = True
