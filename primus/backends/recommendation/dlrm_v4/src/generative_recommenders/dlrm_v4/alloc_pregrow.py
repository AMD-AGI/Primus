"""Caching-allocator pre-grow before RUN_START, and a growth report.

Why: with ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` (launch_slurm.sh)
every growth of a stream's expandable segment calls hipMemCreate/hipMemMap/
hipMemSetAccess, and on ROCm hipMemMap waits for the device to drain. Under the
host-sync-free step the host runs about one step ahead, so a growth stall costs
120-270 ms of host time on the rank that grows (seen under
``## start_sparse_data_dist ## > ## dedup_ec_indices ## >
fbgemm::jagged_unique_indices > aten::empty > hipMemMap``). That rank then
reaches its next KJT splits all-to-all late, and every peer's comm kernel spins
over its forward GEMMs while it waits.

The caching allocator never unmaps (no empty_cache in this tree) and caches
blocks per stream. So mapping each train-pipeline stream's segment once, up
front -- one allocate-and-free on that stream -- removes the mid-run hipMemMap
calls without changing any computed value.

When: ``pregrow_before_run_start()`` runs in train_ranker.py during
initialization, before INIT_STOP/RUN_START (like the C++ prebuild), so the
mapping cost is outside the timed region. The pipeline's memcpy / data_dist
streams do not exist yet at that point, so this module creates them (at the
priority each pipeline will use: torchrec's -1, or item (67)'s 0 for the train
pair at 2+ nodes), grows their pools, and each pipeline adopts its pair by
role in its constructor (``adopt_streams``).

Env (forwarded by launch_slurm.sh through the DLRM_HSTU_FP8_ prefix; the gin
default is bound by apply_env_bootstrap.DLRM_HSTU_FP8_ALLOC_PREGROW):
  DLRM_HSTU_FP8_ALLOC_PREGROW=spec  comma list of <pool>=<GiB>, pool in
                                    {memcpy, data_dist, main} (+ "_small" for the
                                    <=1 MiB pools) and margin=<GiB>. The value
                                    "rest" (one pool at most) takes all free HBM
                                    left after the other pools and the margin,
                                    so one spec fits 1n/2n/4n. "0" or "" = off.
  DLRM_HSTU_FP8_ALLOC_REPORT=1      log every step on which reserved memory
                                    grew, with per-stream totals (host only).
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_GIB = 1 << 30
# torchrec TrainPipelineSparseDist creates its memcpy / data_dist streams with
# this priority; adopt_streams() checks it against the streams it replaces.
_PIPE_STREAM_PRIORITY = -1

_KEYS = ("memcpy", "data_dist", "main", "memcpy_small", "data_dist_small",
         "main_small", "eval_memcpy", "eval_data_dist", "eval_memcpy_small",
         "eval_data_dist_small", "margin")

# Converge pass: map free memory above the margin that appeared DURING the grow
# only if it exceeds this (mem_get_info noise and code objects stay below it).
_CONVERGE_SLACK = 1 << 30
_CONVERGE_PASSES = 2

_pregrown = False
# Pre-grown (memcpy, data_dist) stream pairs: the train pair, then the eval
# pair when eval_* keys are set. _pair_roles[i] names _side_streams[i];
# adopt_streams() hands a pair to the pipeline of that role (item 67 makes the
# pairs differ in priority, so they are not interchangeable).
_side_streams: List[Dict[str, Any]] = []
_pair_roles: List[str] = []
_n_pairs = 0
_reporter_made = False


def _spec() -> str:
    spec = os.environ.get("DLRM_HSTU_FP8_ALLOC_PREGROW", "").strip()
    return "" if spec in ("0", "off", "false", "no") else spec


def report_on() -> bool:
    return os.environ.get("DLRM_HSTU_FP8_ALLOC_REPORT", "0") not in ("", "0")


def _rank() -> int:
    try:
        import torch.distributed as dist

        return dist.get_rank() if dist.is_initialized() else -1
    except Exception:
        return -1


def _parse_spec(spec: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, val = part.partition("=")
        name = name.strip()
        if name not in _KEYS:
            raise ValueError(f"DLRM_HSTU_FP8_ALLOC_PREGROW: unknown key {name!r}")
        val = val.strip()
        if val == "rest":
            if name == "margin":
                raise ValueError("DLRM_HSTU_FP8_ALLOC_PREGROW: margin cannot be 'rest'")
            out[name] = float("inf")
        else:
            f = float(val)
            if f != f or f in (float("inf"), float("-inf")) or f < 0:
                raise ValueError(
                    f"DLRM_HSTU_FP8_ALLOC_PREGROW: {name}={val!r} must be finite and >= 0"
                )
            out[name] = f
    if sum(1 for v in out.values() if v == float("inf")) > 1:
        raise ValueError("DLRM_HSTU_FP8_ALLOC_PREGROW: at most one 'rest' pool")
    return out


def _default_pool(seg: Dict[str, Any]) -> bool:
    # Private MemPools (e.g. DLRM_HSTU_FP8_TM_NONEXP's token-major pool) report
    # their own segment_pool_id: count them separately, never as a stream pool.
    return tuple(seg.get("segment_pool_id", (0, 0)) or (0, 0)) == (0, 0)


def _per_stream_reserved(names: Dict[int, str]) -> Dict[str, float]:
    tot: Dict[str, float] = {}
    for seg in torch.cuda.memory_snapshot():
        if not _default_pool(seg):
            key = "private_pool/" + str(seg.get("segment_type", "?"))
            tot[key] = tot.get(key, 0.0) + seg["total_size"] / _GIB
            continue
        key = names.get(int(seg.get("stream", 0)), "s%x" % int(seg.get("stream", 0)))
        key += "/" + str(seg.get("segment_type", "?"))
        tot[key] = tot.get(key, 0.0) + seg["total_size"] / _GIB
    return tot


def _pool_free_bytes(stream_id: int) -> int:
    n = 0
    for seg in torch.cuda.memory_snapshot():
        if int(seg.get("stream", 0)) != stream_id or seg.get("segment_type") != "large":
            continue
        if not _default_pool(seg):
            # A private pool's cached blocks are not this stream pool's free bytes.
            continue
        n += sum(b["size"] for b in seg.get("blocks", []) if b.get("state") == "inactive")
    return n


def _tail_free_bytes(stream_id: int) -> int:
    """Size of the free block at the mapped end of the stream's expandable large
    segment (0 if the last block is in use). Asking for this + X bytes makes the
    segment absorb that block and map X more (see the "rest" comment below)."""
    best_addr, tail = -1, 0
    for seg in torch.cuda.memory_snapshot():
        if int(seg.get("stream", 0)) != stream_id or seg.get("segment_type") != "large":
            continue
        if not seg.get("is_expandable", True):
            continue
        blocks = seg.get("blocks", [])
        addr = int(seg.get("address", 0))
        if blocks and addr > best_addr:
            best_addr = addr
            tail = int(blocks[-1]["size"]) if blocks[-1].get("state") == "inactive" else 0
    return tail


def _pool_stream(key: str) -> str:
    return key[: -len("_small")] if key.endswith("_small") else key


def _pair_priorities() -> Tuple[int, int]:
    """(train, eval) priorities for the pre-grown stream pairs.

    Each pair is created at the priority its pipeline will ask for
    (host_sync_free.train_pipe_stream_priority for the train pipeline, the A/B
    override pipe_stream_priority for both), so the pipeline keeps the adopted
    pair -- and its pre-grown pools -- instead of replacing it with fresh,
    empty streams. Unset everywhere = torchrec's -1, byte-identical behaviour.
    """
    from generative_recommenders.dlrm_v4 import host_sync_free as _hsf

    t = _hsf.train_pipe_stream_priority()
    e = _hsf.pipe_stream_priority()
    return (
        _PIPE_STREAM_PRIORITY if t is None else t,
        _PIPE_STREAM_PRIORITY if e is None else e,
    )


def guard_train_pair(main: Any, memcpy: Any, data_dist: Any) -> bool:
    """Item (67) placement guard for a train pair created at normal priority.

    Returns True (and turns item 67 off for this process) when either side
    stream shares the main stream's hardware queue: in-order with the forward,
    the deferred input dist's host syncs would wait behind it every step. Only
    the gin default is guarded; an explicit A/B override is left as asked.
    """
    from generative_recommenders.dlrm_v4 import host_sync_free as _hsf

    if _hsf.pipe_stream_priority() is not None:
        return False
    res = {
        name: _hsf.shares_hw_queue(main, st)
        for name, st in (("memcpy", memcpy), ("data_dist", data_dist))
    }
    hits = [name for name, shared in res.items() if shared]
    if hits:
        _hsf.disable_train_pipe_priority(
            "%s shares the main stream's hardware queue" % "+".join(hits)
        )
        return True
    unknown = [name for name, shared in res.items() if shared is None]
    if unknown:
        # The probe could not run: fail closed rather than claim a placement
        # that was never checked.
        _hsf.disable_train_pipe_priority(
            "placement unverified: the hw-queue probe could not run for %s"
            % "+".join(unknown)
        )
        return True
    logger.info(
        "DLRMV4 multi-node side-stream priority placement check: train "
        "memcpy/data_dist run beside main"
    )
    return False


def pregrow_before_run_start() -> None:
    """Create the pipeline side streams and grow the named pools (once).

    Called during initialization, before INIT_STOP/RUN_START. Allocates and
    frees device memory only; touches no data.
    """
    global _pregrown, _side_streams, _pair_roles, _n_pairs
    spec = _spec()
    if _pregrown or not spec or not torch.cuda.is_available():
        return
    _pregrown = True
    t0 = time.perf_counter()
    want = _parse_spec(spec)
    train_prio, eval_prio = _pair_priorities()
    streams: Dict[str, Any] = {
        "main": torch.cuda.current_stream(),
        "memcpy": torch.cuda.Stream(priority=train_prio),
        "data_dist": torch.cuda.Stream(priority=train_prio),
    }
    if train_prio != _PIPE_STREAM_PRIORITY and guard_train_pair(
        streams["main"], streams["memcpy"], streams["data_dist"]
    ):
        # Item (67) fell back (a side stream shares main's hardware queue).
        train_prio = _PIPE_STREAM_PRIORITY
        streams["memcpy"] = torch.cuda.Stream(priority=train_prio)
        streams["data_dist"] = torch.cuda.Stream(priority=train_prio)
    _side_streams = [{k: streams[k] for k in ("memcpy", "data_dist")}]
    _pair_roles = ["train"]
    if any(k.startswith("eval_") for k in want):
        for k in ("eval_memcpy", "eval_data_dist"):
            streams[k] = torch.cuda.Stream(priority=eval_prio)
        _side_streams.append(
            {"memcpy": streams["eval_memcpy"], "data_dist": streams["eval_data_dist"]}
        )
        _pair_roles.append("eval")
    _n_pairs = len(_side_streams)
    names = {int(st.cuda_stream): n for n, st in streams.items()}
    before = _per_stream_reserved(names)
    free, total = torch.cuda.mem_get_info()
    # Never leave less than `margin` GiB unmapped: the eval pipeline's own
    # streams, RCCL lazy peer buffers and non-torch device allocations
    # (hipBLASLt/Triton workspaces) still need room.
    margin = want.get("margin", 56.0) * _GIB
    budget = free - margin
    oom = False
    # Fixed-size pools first; a "rest" pool (inf) takes what is left.
    for name, gib in sorted(want.items(), key=lambda kv: kv[1] == float("inf")):
        if name == "margin" or gib <= 0:
            continue
        small = name.endswith("_small")
        st = streams[_pool_stream(name)]
        have = before.get(_pool_stream(name) + ("/small" if small else "/large"), 0.0)
        if gib == float("inf"):
            grow = int(budget)
            if small:
                grow = min(grow, 1 << 30)  # 1 MiB blocks: cap the small pool at 1 GiB
        else:
            grow = min(int((gib - have) * _GIB), int(budget))
        if grow <= 0:
            continue
        with torch.cuda.stream(st):
            if small:
                # Small pool (<=1 MiB requests): 1 MiB blocks, all live at once
                # so each one maps fresh pages, then freed back to the pool.
                ts = [
                    torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
                    for _ in range(grow >> 20)
                ]
                del ts
            else:
                # One block of the full target on that stream, then free it: the
                # expandable segment stays mapped and the block returns to that
                # stream's pool, where later allocations on the stream reuse it.
                # Asking for grow + the pool's cached free bytes makes the segment
                # extend by ~grow (the trailing free block is absorbed first).
                # For "rest" the request itself is capped at the budget: cached
                # free blocks that are not at the segment tail are not absorbed,
                # so grow + free could otherwise map past free - margin.
                req = grow + _pool_free_bytes(int(st.cuda_stream))
                if gib == float("inf"):
                    req = grow
                try:
                    t = torch.empty(req, dtype=torch.uint8, device="cuda")
                    del t
                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        "DLRMV4 alloc pregrow: rank=%d %s request %.2f GiB OOM; "
                        "pool left as is", _rank(), name, req / _GIB
                    )
                    oom = True
                    break
        budget -= grow
    torch.cuda.synchronize()
    free2, _ = torch.cuda.mem_get_info()
    # Converge: device memory held OUTSIDE the caching allocator can come back
    # while the pass above runs. The FP8 first-use warmup leaves ~6.2 GiB of it on
    # every rank; empty_cache() and 10 s of polling do not return it, but ~6.0 GiB
    # is free again by the end of the first pass (trigger not isolated: the
    # hipMemCreate/Map calls or the synchronize). The first pass sized the "rest"
    # pool before that, so it ended 6.0 GiB above the margin with a 6 GiB smaller
    # main pool (1n and 2n). Map what came back into the "rest" pool, so the pool
    # is the size a run without that memory gets. No-op when the first pass ends
    # within 1 GiB of the margin (every run without the warmup measured so far,
    # 1n and 2n: 56.0-56.1 GiB; 4n not measured).
    rest = [n for n, g in want.items() if g == float("inf")]
    # Ship-12 review: gated on item (35) having run on this rank, so FP8_WARMUP=0
    # (and the gin binding absent) restores the ship-11 pre-grow exactly instead
    # of relying on the measured no-op. sys.modules: never imports the warmup.
    _fw = sys.modules.get("generative_recommenders.dlrm_v4.train.fp8_warmup")
    warmup_ran = bool(getattr(_fw, "WARMUP_RAN", False))
    if rest and not rest[0].endswith("_small") and not oom and warmup_ran:
        st = streams[_pool_stream(rest[0])]
        for _ in range(_CONVERGE_PASSES):
            extra = int(free2 - margin)
            if extra <= _CONVERGE_SLACK:
                break
            req = _tail_free_bytes(int(st.cuda_stream)) + extra
            res0 = torch.cuda.memory_reserved()
            retries0 = int(torch.cuda.memory_stats().get("num_alloc_retries", 0))
            with torch.cuda.stream(st):
                try:
                    t = torch.empty(req, dtype=torch.uint8, device="cuda")
                    del t
                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        "DLRMV4 alloc pregrow: rank=%d converge request %.2f GiB OOM",
                        _rank(), req / _GIB,
                    )
                    break
            torch.cuda.synchronize()
            free3, _ = torch.cuda.mem_get_info()
            grown = torch.cuda.memory_reserved() - res0
            retried = int(torch.cuda.memory_stats().get("num_alloc_retries", 0)) > retries0
            logger.info(
                "DLRMV4 alloc pregrow converge: rank=%d %s pool %+.2f GiB (free %.1f GiB "
                "after the previous pass, %.1f GiB above the margin) -> free %.1f GiB",
                _rank(), rest[0], grown / _GIB, free2 / _GIB,
                (free2 - margin) / _GIB, free3 / _GIB,
            )
            free2 = free3
            if retried:
                # The allocator's OOM retry path releases every cached free block
                # (all pre-grown pools) before retrying; the pools are no longer
                # what the first pass mapped.
                logger.error(
                    "DLRMV4 alloc pregrow: rank=%d converge request hit the allocator "
                    "retry path (cached pools flushed); pre-grow incomplete", _rank()
                )
                break
            if grown < extra // 2:
                break  # served from an existing cached block (nothing mapped): stop
    # DLRM_HSTU_FP8_TM_NONEXP (default off): map the token-major FP8 operand pool
    # (plain, non-expandable segments) AFTER the pools above, i.e. out of the
    # margin: the main pool keeps its full "rest" (on the worst 1n rank it already
    # grows past it) and its peak demand drops by the token-major tensors moved
    # into this pool. Inert unless the knob is on. Ship-12 merge: placed after the
    # converge pass (item (35)), which sizes "rest" from the free memory above the
    # margin; the prewarm then takes its blocks from the margin as before.
    try:
        from generative_recommenders.ops.triton import fp8_tm_alloc

        blocks_env = os.environ.get("DLRM_HSTU_FP8_TM_NONEXP_BLOCKS", "")
        if (
            not blocks_env
            and fp8_tm_alloc.active()
            and fp8_tm_alloc.pooled_all_tiled()
        ):
            # DLRM_HSTU_FP8_TM_NONEXP_TILED=0 and every pooled operand is step-tiled
            # (ordinary allocator): nothing to prewarm, the margin stays unmapped; a
            # row-major operand after a latch maps the pool on demand (logged there).
            # An explicit DLRM_HSTU_FP8_TM_NONEXP_BLOCKS still prewarms.
            print(
                "[hstu-fp8-tm] prewarm skipped: every pooled operand is step-tiled",
                flush=True,
            )
        else:
            # Unset: 5 blocks (three xT + one dout_t + one spare), or 2 when the LN xT
            # stays out of the pool (DLRM_HSTU_FP8_TM_NONEXP_XT=0).
            if fp8_tm_alloc.prewarm(int(blocks_env or fp8_tm_alloc.default_blocks())):
                torch.cuda.synchronize()
                free2, _ = torch.cuda.mem_get_info()
    except Exception as error:  # pragma: no cover - defensive
        logger.warning("DLRMV4 alloc pregrow: tm_nonexp prewarm skipped: %s", error)
    after = _per_stream_reserved(names)
    logger.info(
        "armed OK: alloc pregrow before run_start: rank=%d %.1fs spec=%s "
        "free %.1f->%.1f GiB of %.1f; per-stream reserved GiB before=%s after=%s",
        _rank(),
        time.perf_counter() - t0,
        spec,
        free / _GIB,
        free2 / _GIB,
        total / _GIB,
        {k: round(v, 2) for k, v in before.items()},
        {k: round(v, 2) for k, v in after.items()},
    )


def adopt_streams(pipeline: Any, role: Optional[str] = None) -> bool:
    """Hand a pipeline its pre-grown stream pair; True if it was adopted.

    ``role`` ("train" / "eval") picks the pair made for that pipeline; None
    takes the next pair in build order (train first, then eval).
    """
    if not _side_streams:
        return False
    if role is None:
        i = 0
    elif role in _pair_roles:
        i = _pair_roles.index(role)
    else:
        return False
    which = _pair_roles.pop(i) if i < len(_pair_roles) else "train"
    s = _side_streams.pop(i)
    mine = getattr(pipeline, "_memcpy_stream", None)
    dd = getattr(pipeline, "_data_dist_stream", None)
    if mine is None or dd is None or {mine.priority, dd.priority} != {_PIPE_STREAM_PRIORITY}:
        # torchrec changed how it builds the streams: keep its streams (the
        # side pools then grow on demand) rather than change their priority.
        logger.warning(
            "DLRMV4 alloc pregrow: pipeline streams not adopted (torchrec "
            "priorities %s/%s, expected %d)",
            getattr(mine, "priority", None),
            getattr(dd, "priority", None),
            _PIPE_STREAM_PRIORITY,
        )
        return False
    pipeline._memcpy_stream = s["memcpy"]
    pipeline._data_dist_stream = s["data_dist"]
    logger.info("armed OK: alloc pregrow streams adopted by the %s pipeline", which)
    return True


class GrowthReporter:
    """Per-step host check of reserved bytes; logs only on growth."""

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline
        self._last = torch.cuda.memory_reserved()
        self._step = 0
        self._events = 0

    def after_step(self) -> None:
        self._step += 1
        cur = torch.cuda.memory_reserved()
        if cur == self._last:
            return
        self._events += 1
        p = self._pipeline
        names = {int(torch.cuda.current_stream().cuda_stream): "main"}
        for n in ("memcpy", "data_dist"):
            st = getattr(p, "_%s_stream" % n, None)
            if st is not None:
                names[int(st.cuda_stream)] = n
        logger.info(
            "DLRMV4 alloc growth: rank=%d step=%d events=%d reserved %.3f->%.3f GiB "
            "max_alloc %.3f GiB per-stream=%s",
            _rank(),
            self._step,
            self._events,
            self._last / _GIB,
            cur / _GIB,
            torch.cuda.max_memory_allocated() / _GIB,
            {k: round(v, 3) for k, v in _per_stream_reserved(names).items()},
        )
        self._last = cur


def make_reporter(pipeline: Any) -> Optional[GrowthReporter]:
    """First (train) pipeline only; None unless DLRM_HSTU_FP8_ALLOC_REPORT is set."""
    global _reporter_made
    if _reporter_made or not report_on() or not torch.cuda.is_available():
        return None
    _reporter_made = True
    logger.info("DLRMV4 alloc growth report armed")
    return GrowthReporter(pipeline)
