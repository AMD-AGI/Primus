"""DL_COLD_PREFETCH: post-RUN_START parallel readahead of the first batches.

With the host page cache flushed before the trainer (MLPerf "Data State at
Start of Run"), the first training batches are built from cold storage by ONE
DataLoader worker per batch, i.e. a serial chain of synchronous mmap page
faults (queue depth ~1): ``positions[g] -> flat_uid[pos] -> user_start[uid] ->``
the per-sample scan window ``[max(user_start, pos - scan_window), pos]`` of five
``flat_*`` columns plus three scalar columns at ``pos``. In the START_TS=0
production shape the first batches come from tiny windows (1 batch each), so the
pipeline fill is several such serial batches back to back.

This module computes, in the rank's main process, exactly the byte ranges the
rank's first ``K`` batches will touch (same per-rank slice as
``StreamingWindowSampler.set_window``, same drop_last batching) and hands them to
the block layer in parallel (``posix_fadvise(WILLNEED)`` from a small thread
pool, or parallel ``pread`` in mode 2). The workers then hit the page cache.

Streaming read-ahead (``DL_COLD_PREFETCH_STREAM=1``, default off). The one-shot
prefetch covers only the first ``DL_COLD_PREFETCH_BATCHES`` batches. Early
windows hold one or two batches per rank (2 nodes: the first 8 batches span
windows 14..19). With the two-pool loader (DL_SINGLE_POOL=0) a window's pool
opens only when the window two places earlier has finished, so its
cross-window lookahead is ~2-4 batches; with the single chained pool
(DL_SINGLE_POOL=1, the gin default) the loader's lookahead crosses windows at
the full prefetch depth. Either way each later cold batch is still one
worker's serial fault chain. Measured
cost at 2 nodes (one allocation, kernel caches seeded): 9.0 s of step stalls in
the first 271 steps, and 71.4 s to 4.44M samples cold against 57.5 s with a warm
page cache. In streaming mode the same thread keeps walking the run's windows in
order after the first batch. It recomputes each window's per-rank batches
exactly as the sampler does (same indices, per-rank slice, drop_last, per-window
cap) and issues the same parallel WILLNEED per group of
``DL_COLD_PREFETCH_GROUP`` batches (default 4). It stays at most
``DL_COLD_PREFETCH_LEAD`` batches (default 64) ahead of the batches the training
pipeline has pulled (``note_pull`` / ``note_window_done``), skips batches already
pulled, and runs until the stream ends (``stop``); ``DL_COLD_PREFETCH_HORIZON``
caps it at that many per-rank batches.

Why the whole stream, not a horizon: the walk must not stop mid-run. Likely
mechanism (not measured directly): the WILLNEED walk caches exactly the pages
each batch needs, whereas without it the workers' first faults on a file region
read the kernel's 4 MiB mmap read-around, which also caches neighbouring users
that later batches need. A walk that stops then leaves the workers faulting on
regions nothing has touched yet, and every local rank hits the device at once.
What was measured, on one node emulating node 0 of a 4-node
run (iter-21 ioemu, 8 ranks, production loaders, cold page cache, 2870 steps at
215 ms, GPU cost = input wait beyond 150 ms of run-ahead slack): one-shot only
26.5 s, all in the first 100 steps; walk stopped at a 512-batch horizon 24.0 s,
21.7 s of it in the two windows right after the stop (all 8 ranks stalled at the
same steps); whole-stream walk 2.2 s, equal to the warm page cache (2.2 s, the
first windows' batch builds). Cost: the walk thread used ~1.8% of one core, but
the rank's process CPU rose ~25 ms per step (~12% of a core) with its index
preads on the 16-thread pool; stream mode now preads them inline (``_gather``).

``DL_COLD_PREFETCH_STREAM=auto`` runs the walk only when the run spans more
than one node, decided in ``start()`` (``resolve_stream``) from ``$NNODES``
(scripts/launch_slurm.sh exports it to every rank; train_ranker sizes the world
as NNODES x GPUS_PER_NODE and treats a missing NNODES as one node): on when it
is an integer > 1, off when it is 1, unset or not a positive integer. On one
node the walk did not leave the first training steps neutral: the first batch
arrived at the same time, but the first forward and backward were slower (on
four nodes they were not). Likely mechanism (not measured directly): the
pipeline fill arrives sooner and its ingestion competes with the training
thread for the GIL. One node therefore keeps the one-shot prefetch only.
``1``/``true``/``on``/``yes`` force the walk on at any node count (I/O
emulators that run one node's ranks of a larger world, A/Bs); unset (the
default), ``0``/``off``/``false``/``no`` or any other value keep it off. Rank 0
names the outcome of ``auto``: the armed line's stream fields end with
``auto=multi-node(...)``, or one ``stream=auto resolved off: ...`` line is
logged and the armed line carries no stream fields.

Eval holdout read-ahead (part of the streaming walk; ``DL_COLD_PREFETCH_EVAL``,
default on whenever the walk runs). Without the walk, the workers' faults on
the train data read the 4 MiB mmap read-around, and over the run that caches
most of the store, including the history ranges of the fixed eval holdout
(its anchors sit at the end of the timeline, past every train window). The
walk caches only the exact train ranges, so without more the first holdout
eval pass reads its data cold. Measured on a 4-node production-shape pair
(ship13 v1, walk on vs off, same allocation): first eval pass 105.2 s against
9.3 s (rank 0 first-batch wait 34.0 vs 3.5 s); every later pass 7.1-7.9 s in
both arms. So when the loop hands the walk an eval plan (``set_eval_plan``:
the fixed holdout anchors, the eval batch size, an optional per-pass batch
cap, and the training steps from the stream start to the first data-fraction
eval), the walk also issues the exact ranges of this rank's eval batches
(``StreamingWindowSampler.set_window`` slice, drop_last, cap; the same
``_ranges`` as a train batch). It starts once the consumer is within
``DL_COLD_PREFETCH_EVAL_LEAD`` pulled batches (default 1024) of that eval and
issues one group of ``DL_COLD_PREFETCH_GROUP`` eval batches at most every
``DL_COLD_PREFETCH_EVAL_GAP_S`` seconds (default 1.0), only while the train
read-ahead waits on the consumer (the train read-ahead keeps priority), so the
holdout reads are spread out instead of issued in one burst; when the eval is
fewer than ``_EvalWalk.URGENT`` (64) pulled batches away, or the train walk
reaches the end of the stream first, the rest is issued at once
(the eval that ends the run follows). A walk cut short by
``DL_COLD_PREFETCH_HORIZON`` does not: the read-around takes over again after
it. The holdout is the same at every pass, so one walk covers the run. Rank 0
logs one ``[cold-prefetch] eval holdout walk ...`` line when it is done
(``bytes=`` there, as in the train lines, counts the ranges this rank
requested; a node's ranks overlap). An exception in the eval read-ahead turns
only it off (one warning); the train walk goes on. Measured with its first
version, which also issued eval groups between train groups (4 nodes, same
production shape and seed, ship13 final + this): the holdout read-ahead
covered 62 batches per rank (12.1 GB requested on rank 0) in 26 s, 154 s
before the first eval, no train batch was late, and the first eval pass took
6.4 s (ship13 without the walk, other allocation: 6.6 s).

Rules (MLPerf):
  * Nothing here runs before ``release()``, which the loader calls when the
    sampler windows are installed, i.e. after RUN_START (inside the timed
    region, where overlapping I/O with compute is legal). ``start()`` refuses to
    run otherwise, and every file access goes through ``_open``/``_pread``/
    ``_fadvise`` so tests can prove it.
  * Advisory only: it never changes the bytes, order, windows or samples the
    model sees (the workers still read through their own mmaps).
  * Default off. Armed by ``DL_COLD_PREFETCH=1`` (fadvise) / ``=2`` (parallel
    pread), or by ``DL_PRESTART=2`` (pre-start + cold prefetch, for launchers
    that forward DL_PRESTART but not DL_COLD_PREFETCH). Optional
    ``DL_COLD_PREFETCH_BATCHES`` (default 8) and ``DL_COLD_PREFETCH_THREADS``
    (default 16). ``DL_COLD_PREFETCH_STREAM=1`` (or ``auto`` on more than one
    node) adds the streaming read-ahead to an armed prefetcher; it has no
    effect when the prefetcher is off.
Failures are swallowed with one warning line; the run is unaffected.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

PAGE: int = int(os.sysconf("SC_PAGE_SIZE")) if hasattr(os, "sysconf") else 4096
_CHUNK: int = 1 << 20  # max bytes per pread/fadvise call (spreads big runs over the pool)
_GAP_PAGES: int = 4  # merge runs separated by <= this many pages (fewer, larger requests)

# File-access hooks (tests monkeypatch these to prove nothing runs pre-release).
_open = os.open
_close = os.close
_pread = os.pread
_fadvise = getattr(os, "posix_fadvise", None)
_WILLNEED: int = getattr(os, "POSIX_FADV_WILLNEED", 3)

_RELEASED = threading.Event()
_RELEASE_WALL: List[float] = []
_THREADS: List[threading.Thread] = []

# Streaming read-ahead pacing. All three are written by the training (main)
# thread and only read by the prefetch thread; plain list cells, no locks
# (pacing is approximate by design).
#   _PULLED[0]  batches pulled so far by the chained window stream (warm-keep)
#   _DONE       [windows fully pulled, _PULLED value at that moment]
_PULLED: List[int] = [0]
_DONE: List[int] = [0, 0]
_STOP = threading.Event()
# Streaming mode merges like the one-shot (_GAP_PAGES, 4 pages, an a-priori
# default). Merging across a gap saves one WILLNEED call but reads the gap's
# bytes, so only gaps of a few pages pay off (per-call vs per-byte cost); merged
# runs are still issued in _willneed_cap-sized calls. Re-checked on synthetic
# stores only (no dataset file).
_STREAM_GAP_PAGES: int = _GAP_PAGES
_WAIT_S: float = 0.05  # pacing poll interval
# Default streaming horizon: the whole stream (see the module docstring for why a
# mid-run stop hurts). $DL_COLD_PREFETCH_HORIZON caps it in per-rank batches.
HORIZON_ALL: int = 1 << 62

# Eval holdout read-ahead (module docstring). _EVAL_PLAN[0] is set by the
# training (main) thread before the train stream starts (set_eval_plan) and
# only read by the prefetch thread; _STREAM_P0[0] is _PULLED at the stream's
# start (start()), the origin of the plan's step count.
_EVAL_PLAN: List[Optional[Dict[str, Any]]] = [None]
_STREAM_P0: List[int] = [0]
_EVAL_ON: Tuple[str, ...] = ("", "1", "true", "on", "yes")
_EVAL_OFF: Tuple[str, ...] = ("0", "off", "false", "no")


def _eval_env() -> str:
    return os.environ.get("DL_COLD_PREFETCH_EVAL", "").strip().lower()


def eval_prefetch_on() -> bool:
    """$DL_COLD_PREFETCH_EVAL: the eval holdout read-ahead (default on: unset or
    1/true/on/yes; it acts only together with the streaming walk).
    0/off/false/no turn it off, and so does any other value (fail closed; the
    armed line names it)."""
    return _eval_env() in _EVAL_ON


def set_eval_plan(
    indices: Any,
    batch_size: int,
    steps_to_first_eval: int,
    num_batches_cap: Optional[int] = None,
) -> None:
    """The fixed eval holdout the next eval pass reads (global anchor indices,
    as installed on the eval sampler), the eval batch size, the training steps
    from the start of the train stream to the first data-fraction eval, and
    the per-pass batch cap (None or <= 0: every batch). Called on the main
    thread before the train stream starts; it only stores the arguments (no
    file access). An unusable plan clears it (no eval read-ahead)."""
    try:
        bs = int(batch_size)
        at = int(steps_to_first_eval)
        cap = int(num_batches_cap or 0)
    except (TypeError, ValueError):
        _EVAL_PLAN[0] = None
        return
    if indices is None or bs <= 0 or at <= 0:
        _EVAL_PLAN[0] = None
        return
    _EVAL_PLAN[0] = {"indices": indices, "bs": bs, "at": at, "cap": max(0, cap)}


def _eval_rank_slice(indices: Any, rank: int, world_size: int) -> np.ndarray:
    """This rank's anchors of one eval pass: StreamingWindowSampler.set_window's
    round-robin slice ``g[:n][rank::W]`` of the multiple-of-``world_size``
    prefix (rank_eval_batches)."""
    g = np.asarray(indices, dtype=np.int64).reshape(-1)
    ws = max(1, int(world_size))
    n = (len(g) // ws) * ws
    return g[:n][int(rank)::ws]


def rank_eval_batches(
    indices: Any, rank: int, world_size: int, batch_size: int, cap: int = 0
) -> Tuple[np.ndarray, int]:
    """(per-rank anchor indices, batch count) of the batches one eval pass
    consumes: StreamingWindowSampler.set_window's round-robin slice of the
    multiple-of-world_size prefix, DataLoader drop_last, and the pass's
    num_eval_batches cap (with a cap, the eval pipeline and its workers still
    prefetch a few batches past it; those are not walked)."""
    per_rank = _eval_rank_slice(indices, rank, world_size)
    nb = len(per_rank) // max(1, int(batch_size))
    if cap and int(cap) > 0:
        nb = min(nb, int(cap))
    return per_rank, nb


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, "") or default)
    except ValueError:
        return default


def _eval_armed_fields(plan: Optional[Dict[str, Any]]) -> str:
    """The armed line's eval read-ahead fields (streaming walk only) for the
    plan snapshot the walk uses."""
    if not eval_prefetch_on():
        raw = _eval_env()
        return (" eval=off(DL_COLD_PREFETCH_EVAL)" if raw in _EVAL_OFF
                else f" eval=off(DL_COLD_PREFETCH_EVAL={raw!r} not 0/1)")
    if plan is None:
        return " eval=off(no plan)"
    try:
        _, nb = rank_eval_batches(plan["indices"], 0, 1, int(plan["bs"]), plan["cap"])
    except Exception:
        nb = -1
    if nb == 0:
        return " eval=off(empty holdout)"
    return (f" eval=on at={int(plan['at'])}"
            f" lead={max(0, _env_int('DL_COLD_PREFETCH_EVAL_LEAD', 1024))}"
            f" gap_s={max(0.0, _env_float('DL_COLD_PREFETCH_EVAL_GAP_S', 1.0)):g}"
            f" cap={int(plan['cap'])}")


def note_pull() -> None:
    """One batch pulled by the training pipeline (hot path: one int add)."""
    _PULLED[0] += 1


def note_window_done(n_done: int) -> None:
    """``n_done`` windows of the stream have been pulled completely."""
    _DONE[1] = _PULLED[0]
    _DONE[0] = int(n_done)


def stop() -> None:
    """End of the window stream: the streaming read-ahead exits (the stream's
    ``finally`` and the loader's teardown_close() both call this)."""
    _STOP.set()


_STREAM_FORCE_ON: Tuple[str, ...] = ("1", "true", "on", "yes")


def _stream_env() -> str:
    return os.environ.get("DL_COLD_PREFETCH_STREAM", "").strip().lower()


def stream_on() -> bool:
    """$DL_COLD_PREFETCH_STREAM forces the walk on (1/true/on/yes). ``auto`` is
    resolved per run by ``resolve_stream``."""
    return _stream_env() in _STREAM_FORCE_ON


def _pos_int_env(key: str) -> Tuple[int, str]:
    """(value, raw text) of ``$key``; value 0 unless a positive integer."""
    raw = os.environ.get(key, "").strip()
    try:
        val = int(raw)
    except ValueError:
        val = 0
    return max(0, val), raw


def _multi_node() -> Tuple[Optional[bool], str]:
    """Does the run span more than one node? (answer, evidence); the answer is
    None when unknown. Only $NNODES decides: scripts/launch_slurm.sh exports it
    to every rank, and train_ranker treats a missing NNODES as one node, so no
    other variable (e.g. a stray $LOCAL_WORLD_SIZE) can turn the walk on."""
    nnodes, n_raw = _pos_int_env("NNODES")
    if nnodes:
        return nnodes > 1, f"nnodes={nnodes}"
    return None, (f"NNODES={n_raw!r} not a positive integer" if n_raw else "NNODES unset")


def resolve_stream() -> Tuple[bool, str]:
    """Streaming read-ahead for this run: (on, auto reason). The reason is empty
    unless $DL_COLD_PREFETCH_STREAM=auto; then it is ``multi-node(<evidence>)``
    when the walk runs, otherwise why it stays off (one node, or unknown)."""
    raw = _stream_env()
    if raw != "auto":
        return raw in _STREAM_FORCE_ON, ""
    multi, why = _multi_node()
    if multi:
        return True, f"multi-node({why})"
    return False, f"{'single node' if multi is False else 'node count unknown'} ({why})"


def stream_lead() -> int:
    """$DL_COLD_PREFETCH_LEAD: batches the streaming walk stays ahead of the
    batches the training pipeline has pulled (default 64; read by start())."""
    return max(1, _env_int("DL_COLD_PREFETCH_LEAD", 64))

# yambda flat-store columns: (name, touched as the scan window [ss, pos] or the
# scalar [pos, pos]).
_SCAN_COLS: Tuple[str, ...] = (
    "flat_item_ids", "flat_timestamps",
    "flat_is_listen_plus", "flat_is_like", "flat_is_skip",
)
_SCALAR_COLS: Tuple[str, ...] = (
    "flat_played_ratio", "flat_event_types", "flat_is_organic",
)


# Pre-start level resolved by gin (prestart_streaming_loaders.enabled, which
# reads $DL_PRESTART first), set by utils.prestart_streaming_loaders. Level 2 =
# pre-start plus this prefetcher; an unset env var must not disable it.
_PRESTART_LEVEL: List[int] = [0]


def set_prestart_level(level: int) -> None:
    _PRESTART_LEVEL[0] = int(level)


def mode() -> int:
    """0 = off, 1 = fadvise WILLNEED, 2 = parallel pread. $DL_COLD_PREFETCH
    overrides; otherwise on (WILLNEED) when the pre-start level is 2."""
    raw = os.environ.get("DL_COLD_PREFETCH", "").strip()
    if raw:
        try:
            return max(0, min(2, int(raw)))
        except ValueError:
            return 0
    return 1 if _PRESTART_LEVEL[0] == 2 else 0


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "") or default)
    except ValueError:
        return default


def release() -> None:
    """Mark the sampler release (after RUN_START). Idempotent. Refuses to run
    before RUN_START was logged when this process logs it (hard guard)."""
    from generative_recommenders.dlrm_v4.train import dl_prestart as _dlstart

    _dlstart.require_run_start("sampler release / cold prefetch")
    if not _RELEASED.is_set():
        _RELEASE_WALL.append(time.time())
        _RELEASED.set()


def released() -> bool:
    return _RELEASED.is_set()


class _NpyFile:
    """fd + header geometry of one .npy column (metadata only)."""

    def __init__(self, path: str, expect_len: Optional[int]) -> None:
        self.fd: int = _open(path, os.O_RDONLY)
        try:
            self.size: int = os.fstat(self.fd).st_size
            head = _pread(self.fd, 4096, 0)
            f = io.BytesIO(head)
            ver = np.lib.format.read_magic(f)
            rd = (
                np.lib.format.read_array_header_1_0
                if ver[0] == 1
                else np.lib.format.read_array_header_2_0
            )
            shape, _, dtype = rd(f)
            self.hdr: int = f.tell()
            self.dtype: np.dtype = np.dtype(dtype)
            self.isz: int = int(self.dtype.itemsize)
            n = int(shape[0]) if shape else 0
            if expect_len is not None and n != int(expect_len):
                raise ValueError(f"{path}: header len {n} != array len {expect_len}")
            # Short file: only the elements actually present are addressable.
            self.n: int = max(0, min(n, (self.size - self.hdr) // max(1, self.isz)))
            self.ra_cap: int = _willneed_cap(os.fstat(self.fd).st_dev)
        except Exception:
            _close(self.fd)
            raise

    def close(self) -> None:
        try:
            _close(self.fd)
        except OSError:
            pass


def _page_runs(lo_b: np.ndarray, hi_b: np.ndarray, size: int, gap_pages: int):
    """Byte ranges -> sorted, merged, page-aligned [start, end) runs within size."""
    lo = (lo_b // PAGE) * PAGE
    hi = np.minimum(((hi_b + PAGE - 1) // PAGE) * PAGE, size)
    keep = hi > lo
    lo, hi = lo[keep], hi[keep]
    if lo.size == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e
    order = np.argsort(lo, kind="stable")
    lo, hi = lo[order], hi[order]
    cm = np.maximum.accumulate(hi)
    new = np.ones(lo.size, dtype=bool)
    new[1:] = lo[1:] > cm[:-1] + gap_pages * PAGE
    idx = np.flatnonzero(new)
    return lo[idx].astype(np.int64), np.maximum.reduceat(hi, idx).astype(np.int64)


_RA_CAPS: Dict[int, int] = {}


def _willneed_cap(st_dev: int) -> int:
    """Bytes one posix_fadvise(WILLNEED) call reads at most on this device.

    Linux clamps each WILLNEED request to max(io_pages, ra_pages) pages from
    its start (mm/readahead.c force_page_cache_ra; io_pages = max_sectors_kb,
    ra_pages = read_ahead_kb of the backing queue) and silently drops the rest,
    so longer runs are split into pieces of at most this size. Read once per
    device from sysfs; 128 KiB (the kernel's default read_ahead_kb) when the
    queue cannot be found. Never above _CHUNK."""
    cap = _RA_CAPS.get(st_dev)
    if cap is not None:
        return cap
    cap = 128 << 10
    try:
        base = f"/sys/dev/block/{os.major(st_dev)}:{os.minor(st_dev)}"
        q = os.path.join(base, "queue")
        if not os.path.isdir(q):  # a partition: the queue lives on its disk
            q = os.path.join(base, "..", "queue")
        kb = []
        for name in ("max_sectors_kb", "read_ahead_kb"):
            with open(os.path.join(q, name)) as fh:
                kb.append(int(fh.read().strip()))
        if max(kb) > 0:
            cap = max(kb) << 10
    except (OSError, ValueError):
        pass
    cap = max(PAGE, min(_CHUNK, cap))
    _RA_CAPS[st_dev] = cap
    return cap


def _chunks(starts: np.ndarray, ends: np.ndarray, chunk: int = _CHUNK):
    for s, e in zip(starts.tolist(), ends.tolist()):
        while s < e:
            n = min(chunk, e - s)
            yield s, n
            s += n


def _pmap(pool, fn, items) -> List[Any]:
    """``[fn(x) for x in items]`` on the pool, one task per contiguous slice
    rather than per item: per-item executor dispatch (a future, a queue hop and
    GIL hand-offs per call) costs more Python time than the tiny fadvise/pread
    calls themselves, in the training process. Results keep the item order."""
    items = list(items)
    if not items:
        return []
    n = max(1, min(len(items), int(getattr(pool, "_max_workers", 16) or 16)))
    step = -(-len(items) // n)
    out: List[Any] = []
    for part in pool.map(lambda sl: [fn(x) for x in sl],
                         [items[k:k + step] for k in range(0, len(items), step)]):
        out.extend(part)
    return out


def _willneed_loop(f: _NpyFile, starts, ends) -> None:
    """WILLNEED on [starts, ends) from the calling thread. The call only queues
    readahead (about 1.5 us per call on resident pages), so one thread keeps
    up, and a single thread contends for the GIL with the training thread
    far less than a pool of busy workers would."""
    fd = f.fd
    for s, n in _chunks(starts, ends, getattr(f, "ra_cap", _CHUNK)):
        _fadvise(fd, s, n, _WILLNEED)


def _issue(pool, f: _NpyFile, starts, ends, how: int, serial: bool = False) -> int:
    """Hand [starts, ends) to the block layer in parallel (``serial``: WILLNEED
    from the calling thread instead). Returns bytes."""
    if how == 1 and _fadvise is not None:
        if serial:
            _willneed_loop(f, starts, ends)
            return int((ends - starts).sum())
        fn = lambda a: _fadvise(f.fd, a[0], a[1], _WILLNEED)  # noqa: E731
        chunk = getattr(f, "ra_cap", _CHUNK)  # the kernel drops WILLNEED bytes past it
    else:
        fn = lambda a: len(_pread(f.fd, a[1], a[0]))  # noqa: E731
        chunk = _CHUNK
    _pmap(pool, fn, _chunks(starts, ends, chunk))
    return int((ends - starts).sum())


def _gather(pool, f: _NpyFile, idx: np.ndarray, stats: Dict[str, int],
            serial: bool = False) -> np.ndarray:
    """Values f[idx] read after a WILLNEED pass over their pages, so a dependent
    index level costs ~one I/O latency instead of len(idx) faults. ``serial``
    (the streaming walk, which now runs for the whole run): WILLNEED and the
    preads both from the calling thread. The WILLNEED pass already has every read
    in flight, and ~2.7k pooled preads per 4-batch group at 4 nodes cost ~21 ms
    of process CPU per step plus GIL hand-offs against the training thread;
    inline they are a few ms. Otherwise both go through the pool. Out-of-range
    idx yield -1 (the caller drops them)."""
    idx = np.asarray(idx, dtype=np.int64)
    ok = (idx >= 0) & (idx < f.n)
    u = np.unique(idx[ok])
    out = np.full(idx.shape, -1, dtype=np.int64)
    if u.size == 0:
        return out
    off = f.hdr + u * f.isz
    starts, ends = _page_runs(off, off + f.isz, f.size, 0)
    if _fadvise is not None:
        if serial:
            _willneed_loop(f, starts, ends)
        else:
            _pmap(pool, lambda a: _fadvise(f.fd, a[0], a[1], _WILLNEED),
                  _chunks(starts, ends, getattr(f, "ra_cap", _CHUNK)))
    if serial:
        bufs = [_pread(f.fd, e - s, s) for s, e in zip(starts.tolist(), ends.tolist())]
    else:
        bufs = _pmap(pool, lambda se: _pread(f.fd, se[1] - se[0], se[0]),
                     zip(starts.tolist(), ends.tolist()))
    stats["index_runs"] += int(starts.size)
    stats["index_bytes"] += int((ends - starts).sum())
    base = np.zeros(len(bufs), dtype=np.int64)
    base[1:] = np.cumsum([len(b) for b in bufs])[:-1]
    flat = np.frombuffer(b"".join(bufs), dtype=np.uint8)
    run = np.searchsorted(starts, off, side="right") - 1
    pos = base[run] + (off - starts[run])
    if np.any(pos + f.isz > flat.size):  # short read at EOF
        good = pos + f.isz <= flat.size
        u, pos = u[good], pos[good]
    if u.size == 0:
        return out
    vals = flat[pos[:, None] + np.arange(f.isz)].copy().view(f.dtype).reshape(-1)
    vals = vals.astype(np.int64)
    sel = np.searchsorted(u, idx).clip(0, u.size - 1)
    hit = ok & (u[sel] == idx)  # elements lost to a short read stay -1
    out[hit] = vals[sel[hit]]
    return out


def _window_ids(ds: Any, ts: int) -> np.ndarray:
    """``ds.train_window_indices(ts)`` without its per-window log line when the
    dataset supports ``quiet`` (the loader computes and logs the same window)."""
    fn = ds.train_window_indices
    try:
        import inspect

        quiet = "quiet" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        quiet = False
    return np.asarray(fn(ts, quiet=True) if quiet else fn(ts), dtype=np.int64)


def rank_window_batches(
    ds: Any, ts: int, rank: int, world_size: int, batch_size: int, skip: int = 0
) -> Tuple[np.ndarray, int]:
    """(per-rank anchor indices, batch count) of window ``ts``, exactly as
    StreamingWindowSampler.set_window + drop_last see them."""
    g = _window_ids(ds, ts)
    n = (len(g) // world_size) * world_size
    per_rank = g[:n][rank::world_size]
    if skip:
        per_rank = per_rank[skip:]
    return per_rank, len(per_rank) // batch_size


def first_rank_samples(
    ds: Any,
    ts_list: Sequence[int],
    rank: int,
    world_size: int,
    batch_size: int,
    k_batches: int,
    first_skip_samples: int = 0,
    max_windows: int = 64,
) -> Tuple[List[np.ndarray], List[int]]:
    """This rank's first ``k_batches`` batches as global anchor indices, one
    array per batch, mirroring StreamingWindowSampler.set_window + drop_last."""
    batches: List[np.ndarray] = []
    used_ts: List[int] = []
    for i, ts in enumerate(list(ts_list)[:max_windows]):
        g = _window_ids(ds, ts)
        n = (len(g) // world_size) * world_size
        per_rank = g[:n][rank::world_size]
        if i == 0 and first_skip_samples:
            per_rank = per_rank[first_skip_samples:]
        nb = len(per_rank) // batch_size
        for b in range(nb):
            batches.append(per_rank[b * batch_size:(b + 1) * batch_size])
            if not used_ts or used_ts[-1] != ts:
                used_ts.append(int(ts))
            if len(batches) >= k_batches:
                return batches, used_ts
    return batches, used_ts


def _files(ds: Any) -> Dict[str, _NpyFile]:
    """Open the store columns + positions (missing/mismatched files skipped)."""
    out: Dict[str, _NpyFile] = {}
    cache_dir = getattr(ds, "_cache_dir", None)
    store = getattr(ds, "store", None)
    if not cache_dir or store is None:
        return out
    names = [("positions", getattr(ds, "_positions_name", None), getattr(ds, "_positions", None))]
    for col in ("flat_uid",) + _SCAN_COLS + _SCALAR_COLS:
        names.append((col, f"{col}.npy", getattr(store, col, None)))
    names.append(("user_start", "user_start.npy", getattr(store, "user_start", None)))
    for role, fname, arr in names:
        if not fname or arr is None:
            continue
        try:
            out[role] = _NpyFile(os.path.join(cache_dir, fname), len(arr))
        except Exception:  # missing / unreadable / mismatched: skip this column
            pass
    return out


def _ranges(pool, files, samples: np.ndarray, scan_window: int, how: int,
            stats: Dict[str, int], gap_pages: int = _GAP_PAGES,
            serial: bool = False) -> None:
    pos_f, uid_f, us_f = files.get("positions"), files.get("flat_uid"), files.get("user_start")
    if pos_f is None:
        return
    fp = _gather(pool, pos_f, samples, stats, serial)
    fp = fp[fp >= 0]
    if fp.size == 0:
        return
    if uid_f is not None and us_f is not None:
        uid = _gather(pool, uid_f, fp, stats, serial)
        us = _gather(pool, us_f, uid, stats, serial)
        us = np.where(us >= 0, us, fp)  # unknown user start: scalar only
    else:
        us = fp
    ss = np.maximum(us, fp - int(scan_window))
    for col in _SCAN_COLS + _SCALAR_COLS:
        f = files.get(col)
        if f is None:
            continue
        lo = ss if col in _SCAN_COLS else fp
        lo_b = f.hdr + lo * f.isz
        hi_b = f.hdr + (fp + 1) * f.isz
        starts, ends = _page_runs(lo_b, hi_b, f.size, gap_pages)
        stats["ranges"] += int(starts.size)
        stats["bytes"] += _issue(pool, f, starts, ends, how, serial)


def _ahead(cum_at: Dict[int, int], i: int, b: int) -> Optional[int]:
    """Batches between the consumer and batch ``b`` of walked window ``i``
    (None: window ``i`` is already pulled completely). The consumer is in
    window ``_DONE[0]`` (every earlier window pulled), ``_PULLED - _DONE[1]``
    batches in (always 0 without warm-keep pull counting)."""
    d = _DONE[0]
    if i < d:
        return None
    p = max(0, _PULLED[0] - _DONE[1])
    if i == d:
        return b - p
    # d < i: the walk visits windows in order and only skips windows already
    # pulled, so window d was walked and has a start index.
    return cum_at[i] + b - (cum_at.get(d, cum_at[i]) + p)


def _run_start_wall() -> Optional[float]:
    try:
        from generative_recommenders.dlrm_v4.train import dl_prestart as _dls

        return _dls.run_start_wall()
    except Exception:
        return None


class _EvalWalk:
    """Read-ahead of this rank's eval holdout batches (module docstring), driven
    by the streaming walk's thread. ``pump()`` runs while the walk waits on the
    consumer: once the consumer is within ``lead`` pulled batches of the first
    data-fraction eval it issues one group of eval batches at most every
    ``gap_s`` seconds. ``pump(urgent=True)`` runs at the top of every train
    group and issues every remaining group at once when the eval is less than
    ``URGENT`` pulled batches away (the walk had no wait to use, e.g. a resume
    right before an eval). ``pump(force=True)`` does the same when the train
    walk reached the end of the stream. Inert without a plan or with
    $DL_COLD_PREFETCH_EVAL=0; a failure here only turns it off (one warning)
    and never stops the train walk."""

    URGENT: int = 64  # pulled batches before the eval: issue the rest at once

    def __init__(self, plan: Optional[Dict[str, Any]], pool, files, rank: int,
                 world_size: int, how: int, scan_window: int, group: int) -> None:
        self.on = False
        self.nb = 0
        self.b = 0
        self.rank = int(rank)
        if plan is None or not files or not eval_prefetch_on():
            return
        try:
            self.plan = plan
            self.pool, self.files = pool, files
            self.how, self.scan_window = how, int(scan_window)
            self.group = max(1, int(group))
            self.bs = int(plan["bs"])
            self.at = int(plan["at"])
            self.per_rank, self.nb = rank_eval_batches(
                plan["indices"], rank, world_size, self.bs, plan["cap"])
            if self.nb <= 0:
                return  # nothing to read ahead (the armed line says so)
            self.lead = max(0, _env_int("DL_COLD_PREFETCH_EVAL_LEAD", 1024))
            self.gap_s = max(0.0, _env_float("DL_COLD_PREFETCH_EVAL_GAP_S", 1.0))
            self.next_t = 0.0  # time.monotonic() of the next lead-paced group
            self.t_first: Optional[float] = None
            self.issue_s = 0.0
            self.at_pulled = -1
            self.trigger = ""
            self.st: Dict[str, int] = {"ranges": 0, "bytes": 0, "index_runs": 0,
                                       "index_bytes": 0}
            self.on = True
        except Exception as e:  # advisory only: the train walk goes on
            self._fail(e)

    def _fail(self, e: BaseException) -> None:
        self.on = False
        logger.warning("[cold-prefetch] eval holdout walk failed (ignored) rank=%d: %r",
                       self.rank, e)

    def pending(self) -> bool:
        return self.on and self.b < self.nb

    def pump(self, urgent: bool = False, force: bool = False) -> None:
        if not self.pending():
            return
        try:
            self._pump(urgent, force)
        except Exception as e:  # advisory only: the train walk goes on
            self._fail(e)

    def _pump(self, urgent: bool, force: bool) -> None:
        pulled = _PULLED[0] - _STREAM_P0[0]
        rest = force or pulled >= self.at - self.URGENT
        if urgent and not rest:
            return  # top of a train group: only an imminent eval
        if not rest:
            if pulled < self.at - self.lead:
                return
            if time.monotonic() < self.next_t:
                return
        if self.t_first is None:
            self.t_first = time.time()
            self.at_pulled = pulled
            self.trigger = "stream_end" if force else ("urgent" if rest else "lead")
        while self.b < self.nb and not _STOP.is_set():
            k = min(self.group, self.nb - self.b)
            t1 = time.time()
            _ranges(self.pool, self.files,
                    self.per_rank[self.b * self.bs:(self.b + k) * self.bs],
                    self.scan_window, self.how, self.st,
                    gap_pages=_STREAM_GAP_PAGES, serial=True)
            self.issue_s += time.time() - t1
            self.b += k
            if not rest:
                self.next_t = time.monotonic() + self.gap_s
                break
        if self.b >= self.nb and self.rank == 0:
            rs = _run_start_wall()
            logger.warning(
                "[cold-prefetch] eval holdout walk rank=%d batches=%d bs=%d ranges=%d "
                "bytes=%d index_runs=%d index_bytes=%d issue_s=%.3f trigger=%s "
                "at_pulled=%d first_eval_at=%d lead=%d gap_s=%.2f span_s=%.1f%s",
                self.rank, self.nb, self.bs, self.st["ranges"], self.st["bytes"],
                self.st["index_runs"], self.st["index_bytes"], self.issue_s,
                self.trigger, self.at_pulled, self.at, self.lead,
                self.gap_s, time.time() - (self.t_first or time.time()),
                f" t_rel_run_start={time.time() - rs:+.3f}" if rs else "",
            )


def _stream(pool, files, ds, ts_list, rank, world_size, batch_size,
            first_skip_samples, window_cap, how, scan_window, start_batch,
            horizon, lead, group, st: Dict[str, Any],
            eval_plan: Optional[Dict[str, Any]] = None) -> None:
    """Paced read-ahead of the stream's batches [start_batch, horizon), in the
    order the training pipeline pulls them (see the module docstring), plus the
    eval holdout read-ahead when the loop gave an eval plan (``_EvalWalk``)."""
    cum_at: Dict[int, int] = {}
    walked = 0  # batches walked (issued or found already pulled), the horizon budget
    skip_left = int(start_batch)  # leading batches the first wave already issued
    pulled0 = _PULLED[0]
    ev = _EvalWalk(eval_plan, pool, files, rank, world_size, how, scan_window, group)
    st["eval_batches"] = ev.nb if ev.on else -1
    cut = False  # the horizon stopped the walk (the read-around takes over again)
    # A mid-window resume re-enters window 0 at batch K: the stream pulls only
    # cap - K of its batches under a per-window cap (_WarmWindowStream._open_next).
    k0 = int(first_skip_samples) // max(1, int(batch_size))
    for i, ts in enumerate(ts_list):
        # Horizon: batches walked, or batches the consumer pulled (late windows
        # are skipped without being walked).
        if _STOP.is_set():
            break
        if walked >= horizon or _PULLED[0] - pulled0 >= horizon:
            cut = True
            break
        if i < _DONE[0]:  # pulled completely before we got here
            st["late_windows"] += 1
            continue
        per_rank, nb = rank_window_batches(
            ds, int(ts), rank, world_size, batch_size,
            first_skip_samples if i == 0 else 0,
        )
        if window_cap and int(window_cap) > 0:
            nb = min(nb, max(0, int(window_cap) - (k0 if i == 0 else 0)))
        cum_at[i] = walked
        st["windows"] += 1
        st["last_ts"] = int(ts)
        b = min(nb, skip_left)  # batches the first wave already issued
        skip_left -= b
        walked += b
        while b < nb and walked < horizon and not _STOP.is_set():
            ev.pump(urgent=True)  # eval holdout: the rest, if the eval is imminent
            k = min(group, nb - b, horizon - walked)
            ahead = _ahead(cum_at, i, b)
            if ahead is None:  # the consumer finished this window meanwhile
                st["late_batches"] += nb - b
                walked += nb - b
                break
            if ahead + k <= 0:  # whole group already pulled
                st["late_batches"] += k
            else:
                if ahead > lead:  # far enough ahead: wait for the consumer
                    w0 = time.time()
                    e0 = ev.issue_s if ev.on else 0.0
                    while not _STOP.is_set():
                        ahead = _ahead(cum_at, i, b)
                        if ahead is None or ahead <= lead:
                            break
                        ev.pump()  # eval holdout: one group when due (gap-paced)
                        _STOP.wait(_WAIT_S)
                    # eval issue time is reported by the eval line, not as waiting
                    st["wait_s"] += time.time() - w0 - ((ev.issue_s if ev.on else 0.0) - e0)
                    if _STOP.is_set():
                        break
                    if ahead is None:
                        continue  # re-evaluated at the loop top as consumed
                t1 = time.time()
                _ranges(pool, files, per_rank[b * batch_size:(b + k) * batch_size],
                        scan_window, how, st, gap_pages=_STREAM_GAP_PAGES,
                        serial=True)
                st["stream_issue_s"] += time.time() - t1
                st["stream_batches"] += k
            b += k
            walked += k
        if b < nb and walked >= horizon:  # the horizon stopped this window
            cut = True
            break
    # The train walk reached the end of the stream (not stopped, not cut by the
    # horizon) before the eval read-ahead finished: issue the rest now, the eval
    # that ends the run follows.
    if not _STOP.is_set() and not cut and ev.pending():
        ev.pump(force=True)
    st["eval_walked"] = ev.b if ev.on else -1


def _run(ds, ts_list, rank, world_size, batch_size, first_skip_samples, how,
         k_batches, threads, stream=False, window_cap=None, lead=64, horizon=1024,
         group=4, stream_auto="") -> None:
    t0 = time.time()
    c0 = time.thread_time()
    rel = _RELEASE_WALL[0] if _RELEASE_WALL else t0
    files: Dict[str, _NpyFile] = {}
    eval_plan = _EVAL_PLAN[0]  # one snapshot for the armed line and the walk

    def _rs() -> Optional[float]:
        try:
            from generative_recommenders.dlrm_v4.train import dl_prestart as _dls

            return _dls.run_start_wall()
        except Exception:
            return None

    try:
        if not _RELEASED.is_set():  # belt and braces: never before release
            logger.warning("[cold-prefetch] not released; skipped (rank=%d)", rank)
            return
        # Streaming mode: the first batch alone, then the paced walk from batch 2
        # on (one numbering); one-shot mode: the first k_batches in two waves.
        batches, used_ts = first_rank_samples(
            ds, ts_list, rank, world_size, batch_size, 1 if stream else k_batches,
            first_skip_samples,
        )
        files = _files(ds)
        stats: Dict[str, Any] = {"ranges": 0, "bytes": 0, "index_runs": 0, "index_bytes": 0}
        b1 = dict(stats)
        scan_window = int(getattr(ds, "_scan_window", 0))
        if not (batches and files):
            stream = False
        # (truthiness: gin binds the flag through env_int, so "off" is the int 0)
        if stream and not getattr(ds, "_use_offline_ts_indices", True):
            # Without the offline window index every window is an O(N) mask
            # over the whole anchor array: the walk would repeat the loader's
            # scans and still fall behind tiny windows. One-shot only.
            stream = False
            if rank == 0:
                logger.warning("[cold-prefetch] stream off: no offline window index "
                               "(DL_WINDOW_INDEX=0); one-shot prefetch only")
        with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="cold-prefetch") as pool:
            if batches and files:
                # Wave 1: the first batch alone (the critical one); wave 2: the rest.
                _ranges(pool, files, batches[0], scan_window, how, stats)
                b1 = dict(stats)
                if len(batches) > 1:
                    _ranges(pool, files, np.concatenate(batches[1:]), scan_window, how, stats)
            if rank == 0:
                rs = _rs()
                logger.warning(
                    "[cold-prefetch] armed mode=%s rank=%d batches=%d windows=%s files=%d "
                    "first_ranges=%d bytes=%d batch1_ranges=%d batch1_bytes=%d "
                    "index_runs=%d index_bytes=%d t_after_release=%+.3f issue_s=%.3f%s%s",
                    "fadvise" if how == 1 else "pread", rank, len(batches),
                    (f"{used_ts[0]}..{used_ts[-1]}" if used_ts else "-"), len(files),
                    stats["ranges"], stats["bytes"], b1["ranges"], b1["bytes"],
                    stats["index_runs"], stats["index_bytes"], t0 - rel, time.time() - t0,
                    f" t_rel_run_start={time.time() - rs:+.3f}" if rs else "",
                    (f" stream=on lead={lead}"
                     f" horizon={'all' if horizon >= HORIZON_ALL else horizon} group={group}"
                     f" cap={window_cap or 0}"
                     + (f" auto={stream_auto}" if stream_auto else "")
                     + _eval_armed_fields(eval_plan)) if stream else "",
                )
            if stream:
                st: Dict[str, Any] = {
                    "ranges": 0, "bytes": 0, "index_runs": 0, "index_bytes": 0,
                    "windows": 0, "late_windows": 0, "late_batches": 0,
                    "stream_batches": 0, "stream_issue_s": 0.0, "wait_s": 0.0,
                    "last_ts": -1,
                }
                ts0 = time.time()
                _stream(pool, files, ds, ts_list, rank, world_size, batch_size,
                        first_skip_samples, window_cap, how, scan_window, len(batches),
                        horizon, lead, group, st,
                        eval_plan=eval_plan)
                if rank == 0:
                    rs = _rs()
                    logger.warning(
                        "[cold-prefetch] stream done rank=%d batches=%d windows=%d "
                        "last_ts=%d late_windows=%d late_batches=%d ranges=%d bytes=%d "
                        "index_runs=%d index_bytes=%d issue_s=%.3f wait_s=%.1f "
                        "walk_s=%.1f cpu_s=%.2f pulled=%d stopped=%d "
                        "eval_batches=%d eval_walked=%d%s",
                        rank, st["stream_batches"], st["windows"], st["last_ts"],
                        st["late_windows"], st["late_batches"], st["ranges"],
                        st["bytes"], st["index_runs"], st["index_bytes"],
                        st["stream_issue_s"], st["wait_s"],
                        time.time() - ts0,
                        time.thread_time() - c0, _PULLED[0], int(_STOP.is_set()),
                        int(st.get("eval_batches", -1)), int(st.get("eval_walked", -1)),
                        f" t_rel_run_start={time.time() - rs:+.3f}" if rs else "",
                    )
    except Exception as e:  # advisory only: never fail or slow the run
        logger.warning("[cold-prefetch] failed (ignored) rank=%d: %r", rank, e)
    finally:
        for f in files.values():
            f.close()


def start(
    ds: Any,
    ts_list: Sequence[int],
    rank: int,
    world_size: int,
    batch_size: int,
    first_skip_samples: int = 0,
    window_cap: Optional[int] = None,
) -> Optional[threading.Thread]:
    """Launch the background prefetch (one-shot, plus the paced streaming
    read-ahead under DL_COLD_PREFETCH_STREAM=1, or =auto when the run spans
    more than one node) if armed and released. ``window_cap`` is the stream's
    per-window batch cap (num_train_batches)."""
    how = mode()
    if not how:
        return None
    if not _RELEASED.is_set():
        logger.warning("[cold-prefetch] start() before release ignored (rank=%d)", rank)
        return None
    stream, auto = resolve_stream()
    if auto and not stream and rank == 0:
        logger.warning("[cold-prefetch] stream=auto resolved off: %s", auto)
    _STOP.clear()
    _DONE[0], _DONE[1] = 0, _PULLED[0]  # the pacing counts from this stream's start
    _STREAM_P0[0] = _PULLED[0]  # origin of the eval plan's step count
    # Horizon: the whole stream by default. A walk that stops mid-run turned
    # into a node-wide stall right after the stop at 4 nodes (module docstring);
    # $DL_COLD_PREFETCH_HORIZON caps it in per-rank batches.
    horizon = _env_int("DL_COLD_PREFETCH_HORIZON", HORIZON_ALL)
    th = threading.Thread(
        target=_run,
        args=(ds, list(ts_list), rank, world_size, batch_size, first_skip_samples, how,
              max(1, _env_int("DL_COLD_PREFETCH_BATCHES", 8)),
              max(1, _env_int("DL_COLD_PREFETCH_THREADS", 16))),
        kwargs=dict(
            stream=stream,
            window_cap=window_cap,
            lead=stream_lead(),
            horizon=max(1, horizon),
            group=max(1, _env_int("DL_COLD_PREFETCH_GROUP", 4)),
            stream_auto=auto if stream else "",
        ),
        name="cold-prefetch",
        daemon=True,
    )
    th.start()
    _THREADS.append(th)
    return th


def _reset_for_tests() -> None:
    _EVAL_PLAN[0] = None
    _STREAM_P0[0] = 0
    _STOP.set()
    _RELEASED.clear()
    _RELEASE_WALL.clear()
    for th in _THREADS:
        th.join(timeout=10)
    _THREADS.clear()
    _STOP.clear()
    _PULLED[0] = 0
    _DONE[0] = 0
    _DONE[1] = 0
