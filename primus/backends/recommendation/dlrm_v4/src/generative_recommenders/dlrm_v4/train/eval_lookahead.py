# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
"""Eval embedding lookahead (gin item (85)): issue batch N+1's embedding lookup and sequence
all-to-all while batch N's dense forward runs (x4c19 iter64, design B).

Problem (x4c19 iter63, probes/iter63/evalsync/RESULT.md; 1n eval traces).
Each warm eval batch idles the main stream ~5.6 ms, 81% of it waiting on the
forward's table-wise sequence all-to-all (``## All2All_Seq_fwd ##``). Today
dist_init, the EC TBE lookup, the K1 send permute and that all-to-all are all
issued at the head of forward(N+1) on the main stream, so the heavy sender
(rank 7: lookup + permute ~3.3 ms) joins late and every rank waits for it.

Mechanism. The forward-only eval pipeline (train/utils.py
``build_train_pipeline(role="eval")``) becomes an eval-only subclass
(``wrap``). At the end of ``progress(N)``, batch N+1's KJT all-to-all has just
been issued (``dequeue_batch``); the subclass then runs, on a dedicated side
stream S:

  S.wait_event(ev_c(N)); S.wait_stream(data_dist)
  dist_init(N+1) (KJT a2a wait + recat), TBE lookup, K1 send, seq all_to_all

and parks the EmbeddingCollection awaitable on batch N+1's pipeline context.
The forward of N+1 (``EvalLookaheadForward``) returns that awaitable lazily,
so the main stream waits at the stock first touch and K3'/decode,
construct_jagged_tensors and all dense work stay on main in stock order. A
callback on the awaitable records ``ev_c(N+1)`` on main right after the
embedding wait; S waits on it before any work of the next lookahead.

Invariants (DESIGN.md section 9.5, the must-handle list):
- S is dedicated: created and probed off the main stream's hardware queue
  before RUN_START (``prepare_before_run_start``), its large and small pools
  pre-grown from the allocator margin (one ``_POOL_GIB`` block, then
  SMALL_POOL_MIB of live 1 MiB blocks: alloc_pregrow's small-pool recipe),
  armed only if every rank armed it (one all_reduce).
  ``ev_c`` is the first S op of every lookahead. The receive wire and the
  Request dummy are ``record_stream(main)`` at issue; the send wire is covered
  by ``ev_c`` and the NCCL FIFO.
- The hit path's only main-after-S edge is each sharding's sequence
  all_to_all Work, which main waits at the first touch. So every pipelined
  module must issue one: only table_wise / column_wise / row_wise shardings
  arm; data_parallel (NoWait), any other type, world size 1 (the all_to_all is
  skipped) and torchrec sync collectives decline. The check is host-only and
  rank-uniform, at the build and again in phase A (x4c19 iter64 V1 F1).
- The per-rank stock fallback (phase A, VERIFY mismatch) keeps the stock
  collective order only with the deferred input dist, so TP_DEFER_INPUT_DIST
  other than "1" declines at the build (V1 F3). TP_EVAL_PIPELINE other than
  "1" (no eval pipeline) and TP_DEFER_INPUT_DIST other than "1" also skip the
  stream and the pool before RUN_START; the readiness vote still runs.
- dist_init runs on S after ``S.wait_stream(data_dist)``; it needs
  ``host_sync_free.dist_init_on_main()`` so ``p_kjtt_wait`` records the
  data-dist KJT a2a buffers on S. The kjt and module ctx are recorded on S.
- Batch 0 of every pass runs the stock path on main; the first lookahead of a
  pass is batch 1 and waits on ``ev_c(0)`` (covers the last train step's
  fused optimizer update and keeps every eval TBE forward in batch order).
- ``detach``/``reset``: unconsumed lookaheads are drained on main, then
  ``main.wait_stream(S)`` before training or a checkpoint touches TBE state.
- Collective order per rank is the stock one: seq(k+1) is issued after
  tensors(k+1) and before splits(k+2), on every path (lookahead or stock).
  Declined with EVAL_INTERIM_LOG>0; no lookahead for index >= the
  num_eval_batches cap (rank-uniform).
- Exceptions: phase A (host checks only, before any awaitable or module call)
  falls back to the stock path for the rest of the run on that rank. From
  ``request.wait()`` on, the error is logged and re-raised: a retry could issue
  a duplicate collective and hang every rank.

Numerics: BITWISE eval outputs (same kernels on the same inputs, same
collectives in the same per-rank order; only issue time and stream change).
Training is untouched: role "train" never takes the subclass, and with the
knob off nothing here runs (``prepare_before_run_start`` returns at once).

Switch: DLRM_HSTU_FP8_EVAL_EMB_LOOKAHEAD ("1" arms it; gin binds False, a
pre-set env var wins, "0" is the kill switch). Env-only diagnostics (not bound
in gin): ``_VERIFY=N`` (in-run byte compare against a stock recompute for
indices 1..N of every pass; +1 TBE forward and +1 sequence all_to_all per
sharding per verified batch on every rank), ``_ISSUE=end|top``, ``_PRIO=0|-1``,
``_POOL_GIB=<GiB>`` (default 4; 0 = neither pool pre-grown), ``_REPORT=1``
(per-pass counters).

VERIFY coverage (V1 F4): the recompute reuses the looked-ahead batch's own
KJT, so VERIFY does not independently check the side-stream dist_init (KJT
all_to_all wait + recat), and it compares the EC outputs only (not
predictions, labels or weights). The end-to-end bitwise check is eval-metric
equality across knob 0/1 arms run from identical weights (x4c19 iter64
kits/evlk_1n, check_evlk.py cross-arm verdict).

``cuda`` / ``dist`` / the probe are injectable for CPU unit tests
(train/tests/eval_lookahead_test.py), in the dd_own_h2d style; torch and
torchrec are imported lazily, so importing this module costs nothing.
"""

import logging
import math
import os
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple

logger: logging.Logger = logging.getLogger(__name__)

ENV = "DLRM_HSTU_FP8_EVAL_EMB_LOOKAHEAD"
VERIFY_ENV = ENV + "_VERIFY"
ISSUE_ENV = ENV + "_ISSUE"
PRIO_ENV = ENV + "_PRIO"
POOL_ENV = ENV + "_POOL_GIB"
REPORT_ENV = ENV + "_REPORT"

MAX_TRIES = 8
DEFAULT_POOL_GIB = 4.0
# The side-stream pool is carved from the free (unmapped) HBM that the
# allocator pre-grow leaves as its margin (56 GiB in yambda_5b.gin). Never
# leave less than this behind; below it the arming fails closed instead of
# risking the allocator's flush-and-retry path (which would empty every
# pre-grown pool).
POOL_MIN_FREE_AFTER_GIB = 8.0
# S's small pool (requests <= 1 MiB: Request dummies, block and length
# tensors, recat lengths, offsets) grows on first use too, and every
# expandable-segment growth is a hipMemMap device drain (alloc_pregrow.py's
# module doc). Pre-grown with alloc_pregrow's small-pool recipe, under the
# same _POOL_GIB switch (0 = neither pool) and the same fail-closed checks.
SMALL_POOL_MIB = 64
_GIB = 1 << 30
_MIB = 1 << 20
_POLL_S = 5e-5

# train/utils.py reads both with exactly these expressions (the CPU tests check
# its source): build_train_pipeline defers the next batch's input dist only on
# TP_DEFER_INPUT_DIST == "1"; streaming_train_eval_loop builds the eval
# pipeline only on TP_EVAL_PIPELINE == "1". launch_slurm.sh forwards both to
# every rank.
DEFER_ENV = "TP_DEFER_INPUT_DIST"
EVAL_PIPE_ENV = "TP_EVAL_PIPELINE"

# Sequence (EmbeddingCollection) sharding types whose output dist is torchrec's
# SequenceEmbeddingsAllToAll: alltoall_sequence returns a comm_ops.Request
# whose NCCL Work the main stream waits at the first touch, the only
# main-after-S edge of the hit path. data_parallel returns NoWait; any other
# type is not known to issue such a Work (fail closed).
ORDERED_SHARDINGS = frozenset(("table_wise", "column_wise", "row_wise"))

# Log line texts (every rank). The gin item block quotes the first three.
ARMED = "armed OK: eval emb lookahead stream ready"
INSTALLED = "DLRMV4 eval emb lookahead: eval pipeline subclass installed"
ACTIVE = "DLRMV4 eval emb lookahead ACTIVE: first looked-ahead batch consumed"
DECLINED = "DLRMV4 eval emb lookahead DECLINED"
DISABLED = "DLRMV4 eval emb lookahead DISABLED"
FENCE_MISSING = "DLRMV4 eval emb lookahead: consumed-event fence missing"
FAILURE = "DLRMV4 eval emb lookahead: failure after collectives may have been issued"
DRAINED = "DLRMV4 eval emb lookahead: drained"
VERIFY_LINE = "DLRMV4 eval emb lookahead VERIFY"
REPORT_LINE = "[eval-lookahead]"

# Attribute of a pipeline context that holds its issued lookahead
# ({module name: _Pending}). Per context, so stock reset() drops it.
STORE_ATTR = "_dlrm_evla"

_bad_env_logged: set = set()


# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------
def enabled() -> bool:
    """On only when DLRM_HSTU_FP8_EVAL_EMB_LOOKAHEAD is "1" (read per call).

    Absent = off in code; yambda_5b.gin binds False through
    apply_env_bootstrap ("0"); a pre-set env var wins, so =1 arms it for an
    A/B and =0 is the kill switch even if gin says True.
    """
    return os.environ.get(ENV, "").strip() == "1"


def _bad_env(name: str, raw: str, default: Any) -> None:
    key = (name, raw)
    if key not in _bad_env_logged:
        _bad_env_logged.add(key)
        logger.warning(
            "DLRMV4 eval emb lookahead: %s=%r not understood; using %r", name, raw, default
        )


def verify_count() -> int:
    """_VERIFY=N: byte-compare indices 1..N of every pass (0 = off)."""
    raw = os.environ.get(VERIFY_ENV, "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        _bad_env(VERIFY_ENV, raw, 0)
        return 0


def issue_mode() -> str:
    """_ISSUE=end (default: end of progress(N), for batch N+1) or top (top of
    progress(k), for batch k; torchrec EvalPipelineFusedSparseDist's depth)."""
    raw = os.environ.get(ISSUE_ENV, "").strip().lower()
    if raw in ("", "end"):
        return "end"
    if raw == "top":
        return "top"
    _bad_env(ISSUE_ENV, raw, "end")
    return "end"


def side_priority() -> int:
    """_PRIO=0 (default, normal priority) or -1 (high priority, diagnostic:
    host polls the KJT Works and ev_c before every issue so the high-priority
    queue never holds an unmet wait). ROCm clamps priorities to {-1, 0}."""
    raw = os.environ.get(PRIO_ENV, "").strip()
    if raw in ("", "0"):
        return 0
    if raw == "-1":
        return -1
    _bad_env(PRIO_ENV, raw, 0)
    return 0


def pool_gib() -> float:
    """_POOL_GIB: side-stream pool pre-grow in GiB (default 4; 0 = none)."""
    raw = os.environ.get(POOL_ENV, "").strip()
    if not raw:
        return DEFAULT_POOL_GIB
    try:
        v = float(raw)
    except ValueError:
        v = float("nan")
    if not math.isfinite(v) or v < 0:
        _bad_env(POOL_ENV, raw, DEFAULT_POOL_GIB)
        return DEFAULT_POOL_GIB
    return v


def report_on() -> bool:
    return os.environ.get(REPORT_ENV, "").strip() == "1"


def defer_input_dist() -> bool:
    """The pipeline defers the next batch's input dist (TP_DEFER_INPUT_DIST
    exactly "1", the default; the same read as train/utils.py
    build_train_pipeline)."""
    return os.environ.get(DEFER_ENV, "1") == "1"


def eval_pipeline_on() -> bool:
    """An eval pipeline can exist (TP_EVAL_PIPELINE exactly "1", the default;
    the same read as train/utils.py streaming_train_eval_loop, which also
    needs gin use_pipeline)."""
    return os.environ.get(EVAL_PIPE_ENV, "1") == "1"


def _env_decline_reason() -> Optional[str]:
    """Env-only reasons the lookahead can never run on this job (rank-uniform:
    the launcher forwards both variables to every rank). None = no reason."""
    if not eval_pipeline_on():
        return "%s=%r: eval runs the serial loop, no eval pipeline" % (
            EVAL_PIPE_ENV, os.environ.get(EVAL_PIPE_ENV)
        )
    if not defer_input_dist():
        # V1 F3: without the deferred input dist, stock progress(k) issues
        # batch k+1's input dist inline, before forward(k)'s sequence
        # all_to_all, while a lookahead rank issued that all_to_all at the end
        # of progress(k-1). After a per-rank fallback (phase A, VERIFY
        # mismatch) the ranks would pair different collectives: a hang.
        return (
            "%s=%r: without the deferred input dist a rank's stock fallback "
            "would pair different collectives with its peers" % (
                DEFER_ENV, os.environ.get(DEFER_ENV)
            )
        )
    return None


# --------------------------------------------------------------------------
# Small helpers (lazy torch; CPU-test friendly)
# --------------------------------------------------------------------------
def _torch_cuda() -> Any:
    import torch

    return torch.cuda


def _torch_dist() -> Any:
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover
        return None
    return dist


def _dist_up(dist: Any) -> bool:
    try:
        return dist is not None and dist.is_available() and dist.is_initialized()
    except Exception:  # noqa: BLE001
        return False


def _rank(dist: Any = None) -> int:
    dist = dist if dist is not None else _torch_dist()
    if _dist_up(dist):
        try:
            return int(dist.get_rank())
        except Exception:  # noqa: BLE001
            pass
    for key in ("RANK", "LOCAL_RANK"):
        v = os.environ.get(key, "").strip()
        if v:
            try:
                return int(v)
            except ValueError:
                pass
    return -1


def _rf(name: str) -> Any:
    """torch.profiler.record_function(name), or a no-op without torch."""
    try:
        from torch.profiler import record_function
    except Exception:  # noqa: BLE001
        return nullcontext()
    return record_function(name)


def _no_grad() -> Any:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return nullcontext()
    return torch.no_grad()


def _handle(stream: Any) -> Optional[int]:
    try:
        return int(stream.cuda_stream)
    except Exception:  # noqa: BLE001
        return None


def _has_callbacks(aw: Any) -> bool:
    """True for a torchrec Awaitable (``callbacks`` is a class property).

    Checked on the type: a LazyAwaitable resolves itself in __getattr__, so
    probing an instance for a missing attribute would trigger its wait()."""
    return isinstance(getattr(type(aw), "callbacks", None), property)


def _awaitables_per_sharding(aw: Any) -> List[Any]:
    # vars(): never route through LazyAwaitable.__getattr__ (it would wait).
    try:
        got = vars(aw).get("_awaitables_per_sharding")
    except TypeError:
        return []
    return list(got) if got else []


def _index(ctx: Any) -> int:
    try:
        return int(ctx.index)
    except Exception:  # noqa: BLE001
        return -1


def _ready_value(value: Any) -> Any:
    """An already-resolved awaitable for the model's lazy first touch."""
    try:
        from torchrec.distributed.types import LazyNoWait
    except Exception:  # noqa: BLE001 -- CPU tests without torchrec
        return value
    return LazyNoWait(value)


def _bits_equal(a: Any, b: Any) -> bool:
    from generative_recommenders.dlrm_v4.seq_a2a_rowperm import bits_equal

    return bits_equal(a, b)


def _jt_dicts_bits_equal(a: Any, b: Any) -> bool:
    """Byte compare of two EC outputs ({key: JaggedTensor}): same keys, then
    per key the contiguous values, lengths and offsets (and weights when
    present) by dtype, shape and integer view (catches -0.0/+0.0, NaN
    payloads)."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if list(a.keys()) != list(b.keys()):
        return False
    for k in a:
        x, y = a[k], b[k]
        for part in ("values", "lengths", "offsets"):
            if not _bits_equal(getattr(x, part)().contiguous(), getattr(y, part)().contiguous()):
                return False
        wx = x.weights_or_none() if hasattr(x, "weights_or_none") else None
        wy = y.weights_or_none() if hasattr(y, "weights_or_none") else None
        if (wx is None) != (wy is None):
            return False
        if wx is not None and not _bits_equal(wx.contiguous(), wy.contiguous()):
            return False
    return True


def _record_tensor(t: Any, stream: Any) -> bool:
    if t is None or not getattr(t, "is_cuda", False):
        return False
    t.record_stream(stream)
    return True


def _record_shared_caches(stream: Any) -> None:
    """Mark seq_a2a_rowperm's grow-only ``_arange_cache`` buffers in use on
    ``stream`` (U1 H11). They are shared by the rowperm fallback path on main
    and on S; a growth on one stream frees the old buffer into the allocating
    stream's pool. With both streams recorded, that block is reused only after
    both have passed their queued reads. Host-only bookkeeping (no kernel);
    the fused K1 path (production) does not read these buffers."""
    try:
        from generative_recommenders.dlrm_v4 import seq_a2a_rowperm as sarp

        for t in list(getattr(sarp, "_arange_cache", {}).values()):
            _record_tensor(t, stream)
    except Exception:  # noqa: BLE001 -- bookkeeping only
        pass


# --------------------------------------------------------------------------
# V1 F1: every pipelined module's output must be ordered by a collective Work
# --------------------------------------------------------------------------
def _sync_collectives_on() -> Optional[bool]:
    """torchrec's process-global sync-collectives switch
    (comm_ops.get_use_sync_collectives; when on, alltoall_sequence runs the
    all_to_all synchronously on the current stream and returns NoWait). False
    without torchrec (CPU unit tests inject their own modules); None when
    torchrec is there but the switch cannot be read (the caller fails closed).
    """
    try:
        from torchrec.distributed import comm_ops
    except ImportError:
        return False
    fn = getattr(comm_ops, "get_use_sync_collectives", None)
    if fn is None:
        return None
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001
        return None


def _sharded_modules(model: Any) -> List[Tuple[str, Any]]:
    """(type name, module) of every module of ``model`` that carries a torchrec
    sharding plan (``_sharding_type_to_sharding``, the attribute
    b1pp._seq_a2a_sites reads). Host only, read only. Empty when ``model``
    cannot be walked (phase A then checks the pipelined modules)."""
    walk = getattr(model, "modules", None)
    if not callable(walk):
        return []
    return [
        (type(m).__name__, m)
        for m in walk()
        if getattr(m, "_sharding_type_to_sharding", None) is not None
    ]


def _unordered_output_reason(mods: List[Tuple[str, Any]], world: int) -> Optional[str]:
    """V1 F1 guard; host only and rank-uniform (one sharding plan, one WORLD
    size and one process-global switch on every rank). The hit path orders
    the main stream after S's lookup and K1 only through each sharding's
    sequence all_to_all Work (main waits it at the first touch; the forward
    adds no S event wait). An output dist with no collective would let main
    read S's output unordered: a silent non-bitwise eval. Returns the reason
    to run stock, or None when every module of ``mods`` is covered.
    """
    if world <= 1:
        return (
            "world size %d: the sequence all_to_all is skipped (NoWait), so nothing "
            "would order the main stream after the side stream" % world
        )
    sync = _sync_collectives_on()
    if sync is None:
        return "torchrec sync-collectives switch unreadable (fail closed)"
    if sync:
        return (
            "torchrec sync collectives on: the sequence all_to_all returns NoWait, so "
            "nothing would order the main stream after the side stream"
        )
    for name, m in mods:
        st2s = getattr(m, "_sharding_type_to_sharding", None)
        if not isinstance(st2s, dict) or not st2s:
            return "module %s: sharding types unknown (fail closed)" % name
        bad = sorted({str(getattr(t, "value", t)) for t in st2s} - ORDERED_SHARDINGS)
        if bad:
            return (
                "module %s has a %s sharding: its output dist issues no sequence "
                "all_to_all for the main stream to wait on" % (name, "/".join(bad))
            )
    return None


# --------------------------------------------------------------------------
# Before RUN_START: the side stream S
# --------------------------------------------------------------------------
_prepared: Dict[str, Any] = {
    "done": False,
    "stream": None,
    "prio": 0,
    "tries": 0,
    "pool_gib": 0.0,
    "small_mib": 0,
    "ready": 0,
    "world": 1,
    "reason": None,
}


def prepared_stream() -> Any:
    """The armed side stream, or None (knob off, declined, or never prepared)."""
    return _prepared["stream"]


def _known_side_handles() -> Tuple[set, List[Any]]:
    """Handles of the streams alloc_pregrow created before us, and the eval
    pair (for the priority -1 placement check)."""
    handles: set = set()
    eval_pair: List[Any] = []
    try:
        from generative_recommenders.dlrm_v4 import alloc_pregrow as ap

        for role, pair in zip(list(ap._pair_roles), list(ap._side_streams)):
            for st in pair.values():
                h = _handle(st)
                if h is not None:
                    handles.add(h)
                if role == "eval":
                    eval_pair.append(st)
    except Exception:  # noqa: BLE001
        pass
    return handles, eval_pair


def _pick_stream(cuda: Any, probe: Callable[..., Any], prio: int) -> Tuple[Any, int, Optional[str]]:
    """Up to MAX_TRIES pooled streams at ``prio``; the first one that is not an
    existing stream's handle and that the hw-queue probe places beside main
    (and, at -1, beside the eval memcpy/data_dist pair). Fail closed: a probe
    that cannot run (None) ends the search not ready."""
    main = cuda.current_stream()
    main_h = _handle(main)
    known, eval_pair = _known_side_handles()
    if main_h is not None:
        known.add(main_h)
    seen = []
    for i in range(1, MAX_TRIES + 1):
        s = cuda.Stream(priority=prio)
        h = _handle(s)
        if h is None or h in known:
            seen.append("aliases an existing stream")
            continue
        known.add(h)
        shared = probe(main, s)
        if shared is None:
            return None, i, "hw-queue probe could not run (fail closed)"
        if shared:
            seen.append("shares main's hw queue")
            continue
        if prio == -1:
            hit = None
            for e in eval_pair:
                r = probe(e, s)
                if r is None:
                    return None, i, "hw-queue probe vs the eval pair could not run (fail closed)"
                if r:
                    hit = "shares the eval memcpy/data_dist hw queue"
                    break
            if hit:
                seen.append(hit)
                continue
        return s, i, None
    why = sorted(set(seen))
    return None, MAX_TRIES, "%d/%d candidates rejected (%s)" % (
        len(seen), MAX_TRIES, "; ".join(why) or "none usable"
    )


def _grow_on_current(nbytes: int) -> None:
    """One allocate-and-free on the current stream (maps that stream's pool)."""
    import torch

    t = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    del t


def _grow_small_on_current(nbytes: int) -> None:
    """alloc_pregrow's small-pool recipe (pregrow_before_run_start, the
    ``small`` branch): 1 MiB blocks (the caching allocator's small-pool
    limit), all live at once so each maps fresh pages, then freed back to the
    current stream's small pool."""
    import torch

    ts = [torch.empty(_MIB, dtype=torch.uint8, device="cuda") for _ in range(nbytes >> 20)]
    del ts


def _pregrow(
    cuda: Any,
    s: Any,
    gib: float,
    grow: Callable[[int], None],
    grow_small: Callable[[int], None],
) -> Optional[str]:
    """Pre-grow S's large pool by ``gib`` and its small pool by SMALL_POOL_MIB,
    from the margin; None on success. ``gib`` 0 grows neither. Fails closed
    when the two together would leave less than POOL_MIN_FREE_AFTER_GIB free,
    or when the allocator retry counter moved during either grow."""
    req = int(gib * _GIB)
    if req <= 0:
        return None
    small = SMALL_POOL_MIB * _MIB
    free, _total = cuda.mem_get_info()
    if free - req - small < POOL_MIN_FREE_AFTER_GIB * _GIB:
        return (
            "pool pre-grow %.1f GiB + %d MiB small does not fit (free %.1f GiB, "
            "keeps >= %.1f GiB)" % (gib, SMALL_POOL_MIB, free / _GIB, POOL_MIN_FREE_AFTER_GIB)
        )
    retries0 = int(cuda.memory_stats().get("num_alloc_retries", 0))
    with cuda.stream(s):
        grow(req)
        grow_small(small)
    cuda.synchronize()
    if int(cuda.memory_stats().get("num_alloc_retries", 0)) > retries0:
        logger.error(
            "DLRMV4 eval emb lookahead: pool pre-grow hit the allocator retry path "
            "(cached pools flushed) rank=%d", _rank()
        )
        return "pool pre-grow hit the allocator retry path"
    return None


def _vote_sum(ready: int, dist: Any, cuda: Any) -> int:
    """Sum of the 0/1 readiness votes over WORLD (all ready == MIN of 1)."""
    import torch

    t = torch.tensor([int(ready)], dtype=torch.int32, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def prepare_before_run_start(
    *,
    cuda: Any = None,
    dist: Any = None,
    probe: Optional[Callable[..., Any]] = None,
    grow: Optional[Callable[[int], None]] = None,
    vote: Optional[Callable[[int], int]] = None,
    grow_small: Optional[Callable[[int], None]] = None,
) -> bool:
    """Create, place and pre-grow the side stream S, once, before RUN_START.

    Called from train_ranker right after the allocator pre-grow (untimed
    initialization, the process group exists). Knob off: returns False at once
    (no stream, no allocation, no collective). Knob on: up to MAX_TRIES
    pooled streams at the chosen priority, accepted only when
    ``host_sync_free.shares_hw_queue(main, s) is False``; then S's large and
    small pools are pre-grown; then one all_reduce of the 0/1 readiness over
    WORLD. S is kept only if every rank is ready. With TP_EVAL_PIPELINE or
    TP_DEFER_INPUT_DIST other than "1" no stream or pool is made, but the
    rank still votes 0: whether a collective runs never depends on a per-rank
    env read. Every rank logs the arming line or the decline line. Returns
    True when armed.
    """
    if not enabled():
        return False
    if _prepared["done"]:
        return _prepared["stream"] is not None
    _prepared["done"] = True
    dist = dist if dist is not None else _torch_dist()
    rank = _rank(dist)
    world = int(dist.get_world_size()) if _dist_up(dist) else 1
    prio = side_priority()
    gib = pool_gib()
    stream, tries, reason = None, 0, None
    skip = _env_decline_reason()
    if skip is None:
        try:
            cuda = cuda if cuda is not None else _torch_cuda()
            if not cuda.is_available():
                reason = "CUDA/HIP not available"
            else:
                if probe is None:
                    from generative_recommenders.dlrm_v4 import host_sync_free as _hsf

                    probe = _hsf.shares_hw_queue
                stream, tries, reason = _pick_stream(cuda, probe, prio)
                if stream is not None:
                    reason = _pregrow(
                        cuda, stream, gib, grow or _grow_on_current,
                        grow_small or _grow_small_on_current,
                    )
                    if reason is not None:
                        stream = None
        except Exception as e:  # noqa: BLE001 -- never block startup on this
            stream, reason = None, "prepare raised %s: %s" % (type(e).__name__, e)
    ready = 1 if stream is not None else 0
    n_ready = ready
    if world > 1:
        try:
            n_ready = int(vote(ready)) if vote is not None else _vote_sum(ready, dist, cuda)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "DLRMV4 eval emb lookahead: readiness all_reduce failed (%s: %s) rank=%d",
                type(e).__name__, e, rank,
            )
            n_ready = -1
    armed = stream is not None and n_ready == world
    small_mib = SMALL_POOL_MIB if armed and int(gib * _GIB) > 0 else 0
    _prepared.update(
        stream=stream if armed else None,
        prio=prio,
        tries=tries,
        pool_gib=gib if armed else 0.0,
        small_mib=small_mib,
        ready=n_ready,
        world=world,
        reason=None if armed else (skip or reason or "peer ranks not ready"),
    )
    if armed:
        logger.info(
            "%s (prio=%d tries=%d pool=%.1fGiB small=%dMiB beside main; ranks ready "
            "%d/%d) rank=%d",
            ARMED, prio, tries, gib, small_mib, n_ready, world, rank,
        )
    elif skip is not None:
        # Same format as the build/configure declines: nothing was attempted.
        logger.warning("%s (%s): stock eval path rank=%d", DECLINED, skip, rank)
    else:
        not_ready = world - n_ready if n_ready >= 0 else world
        logger.warning(
            "%s (no usable side stream on %d/%d ranks: %s): stock eval path rank=%d",
            DECLINED, not_ready, world,
            reason or ("this rank ready; peer ranks not" if n_ready >= 0 else "vote failed"),
            rank,
        )
    return armed


# --------------------------------------------------------------------------
# The forward
# --------------------------------------------------------------------------
_FWD_CLASSES: Dict[Any, Any] = {}


def _make_forward_class(base: Any) -> Any:
    class EvalLookaheadForward(base):
        """PipelinedForward of the eval-lookahead pipeline.

        Hit (this batch was looked ahead): return the issued EC awaitable
        lazily (the main stream waits at the stock first touch). Miss: the
        stock ``PipelinedForward.__call__``. Both record ``ev_c(index)`` on
        main right after the embedding wait (awaitable callback).
        """

        # The pipeline's controller (_Lookahead); set on a per-pipeline subclass.
        _dlrm_evla_ctl: Any = None

        def __call__(self, *input, **kwargs):
            ctl = type(self)._dlrm_evla_ctl
            ctx = self._context
            store = getattr(ctx, STORE_ATTR, None) if ctx is not None else None
            hit = store.pop(self._name, None) if store else None
            if ctl is None:
                if hit is not None:  # pragma: no cover -- issue needs a controller
                    raise RuntimeError("eval lookahead forward without a controller")
                return super().__call__(*input, **kwargs)
            if hit is None:
                return ctl.forward_miss(
                    self,
                    lambda: super(EvalLookaheadForward, self).__call__(*input, **kwargs),
                )
            return ctl.forward_hit(self, hit)

    return EvalLookaheadForward


def forward_class(base: Any = None) -> Any:
    """``EvalLookaheadForward`` built on ``base`` (default: torchrec's
    PipelinedForward, imported lazily); one class per base."""
    if base is None:
        from torchrec.distributed.train_pipeline.runtime_forwards import (
            PipelinedForward as base,
        )
    cls = _FWD_CLASSES.get(base)
    if cls is None:
        cls = _make_forward_class(base)
        _FWD_CLASSES[base] = cls
    return cls


def __getattr__(name: str) -> Any:  # PEP 562: lazy EvalLookaheadForward
    if name == "EvalLookaheadForward":
        return forward_class()
    raise AttributeError(name)


# --------------------------------------------------------------------------
# The controller (one per eval pipeline)
# --------------------------------------------------------------------------
class _Pending:
    __slots__ = ("aw", "mctx", "kjt", "shardings")

    def __init__(self, aw: Any, mctx: Any, kjt: Any, shardings: int) -> None:
        self.aw = aw
        self.mctx = mctx
        self.kjt = kjt
        self.shardings = shardings


class _Lookahead:
    """State and logic of one eval-lookahead pipeline (see the module doc)."""

    def __init__(self, cuda: Any, hsf: Any, fwd_cls: Any) -> None:
        self.cuda = cuda
        self.hsf = hsf
        self.fwd_cls = fwd_cls
        self.bound_fwd: Any = None
        self.stream: Any = None
        self.prio = int(_prepared["prio"])
        self.issue_mode = issue_mode()
        self.verify_n = verify_count()
        self.report = report_on()
        self.installed = False
        self.on = False
        self.cap: Optional[int] = None
        self.ev: Any = None
        self.ev_idx = -1
        self.fence_warned = False
        self.active_logged = False
        self.pass_no = 1
        self.stats = self._zero()

    @staticmethod
    def _zero() -> Dict[str, int]:
        return {"issued": 0, "hit": 0, "stock": 0, "host_ahead": 0, "fenced": 0,
                "fence_missing": 0, "verified": 0, "mismatch": 0}

    def _c(self) -> Any:
        if self.cuda is None:
            self.cuda = _torch_cuda()
        return self.cuda

    def _h(self) -> Any:
        if self.hsf is None:
            from generative_recommenders.dlrm_v4 import host_sync_free as _hsf

            self.hsf = _hsf
        return self.hsf

    # ---- build / configure ------------------------------------------------
    def build_check(self, p: Any) -> Optional[str]:
        """Rank-uniform reasons to run the stock eval path (None = install)."""
        why = _env_decline_reason()  # V1 F3: TP_DEFER_INPUT_DIST (and TP_EVAL_PIPELINE)
        if why is not None:
            return why
        s = prepared_stream()
        if s is None:
            why = _prepared.get("reason") if _prepared.get("done") else None
            return "no prepared side stream (%s)" % (
                why or "prepare_before_run_start did not run before RUN_START"
            )
        # V1 F1, over every sharded module of the model (the pipelined modules
        # are not known before the first fill; phase A checks those).
        why = _unordered_output_reason(
            _sharded_modules(getattr(p, "_model", None)), int(_prepared["world"])
        )
        if why is not None:
            return why
        if not self._h().dist_init_on_main():
            return (
                "DLRM_HSTU_FP8_SYNCFREE or DLRM_HSTU_FP8_SYNCFREE_DISTINIT_MAIN off: "
                "p_kjtt_wait would not record the KJT a2a buffers on the side stream"
            )
        if not _kjt_wait_records_current():
            return "host_sync_free KJT all_to_all wait patch (p_kjtt_wait) not installed"
        if getattr(p, "_data_dist_stream", None) is None:
            return "pipeline has no data_dist stream"
        hs = _handle(s)
        pipe = (_handle(p._data_dist_stream), _handle(getattr(p, "_memcpy_stream", None)))
        if hs is not None and hs in pipe:
            return "side stream aliases a pipeline stream"
        self.stream = s
        return None

    def install(self, p: Any) -> None:
        self.bound_fwd = type(
            "EvalLookaheadForward", (self.fwd_cls,),
            {"_dlrm_evla_ctl": self, "__module__": __name__},
        )
        p._pipelined_forward_type = self.bound_fwd
        self.installed = True
        self.on = True
        logger.info(
            "%s (prio=%d issue=%s rank=%d)", INSTALLED, self.prio, self.issue_mode, _rank()
        )

    def decline(self, p: Any, reason: str, stock_fwd: Any, warn: bool = True) -> None:
        """Build/configure-time decline (rank-uniform): the stock forward type,
        nothing issued, S never used. (Forwards already installed by a fill
        would keep missing, i.e. run the stock path.) ``warn=False`` when the
        cause was already logged as a WARNING before RUN_START (one warning
        line per disengagement)."""
        self.on = False
        if stock_fwd is not None and not getattr(p, "_pipelined_modules", None):
            p._pipelined_forward_type = stock_fwd
            self.installed = False
        (logger.warning if warn else logger.info)(
            "%s (%s): stock eval path rank=%d", DECLINED, reason, _rank()
        )

    def disable(self, reason: str) -> None:
        """Runtime decline on this rank (phase A, VERIFY mismatch)."""
        if not self.on:
            return
        self.on = False
        logger.warning(
            "%s on rank %d: %s; stock eval path for the rest of the run",
            DISABLED, _rank(), reason,
        )

    # ---- issue ------------------------------------------------------------
    def _phase_a(self, p: Any, ctx: Any) -> Tuple[List[Any], List[str], Optional[str]]:
        """Host checks only: nothing here calls an awaitable or a module."""
        if getattr(p._model, "training", False):
            return [], [], "model in training mode (eval-only lookahead)"
        if not self._h().dist_init_on_main():
            return [], [], "dist_init_on_main() is off"
        S, main, dd = self.stream, self._c().current_stream(), p._data_dist_stream
        hs = _handle(S)
        if S is None or hs is None or hs in (_handle(main), _handle(dd)):
            return [], [], "side stream missing or aliases main/data_dist"
        mods = list(getattr(p, "_pipelined_modules", None) or [])
        if not mods:
            return [], [], "no pipelined modules"
        names: List[str] = []
        reqs = ctx.input_dist_tensors_requests
        mctxs = ctx.module_contexts
        for m in mods:
            f = m.forward
            if self.bound_fwd is None or not isinstance(f, self.bound_fwd):
                return [], [], "pipelined forward is %s, not this pipeline's EvalLookaheadForward" % (
                    type(f).__name__,
                )
            n = f.name
            if n in names:
                return [], [], "module %s pipelined twice" % n
            if n not in reqs:
                return [], [], "no input-dist request for %s at index %d" % (n, _index(ctx))
            if n not in mctxs:
                return [], [], "no module context for %s at index %d" % (n, _index(ctx))
            names.append(n)
        # V1 F1 on the modules actually pipelined (rank-uniform, so every rank
        # stops at the same index; still before any awaitable or module call).
        why = _unordered_output_reason(list(zip(names, mods)), int(_prepared["world"]))
        if why is not None:
            return [], [], why
        return mods, names, None

    def _poll_before_issue(self, p: Any, ctx: Any, names: List[str]) -> None:
        """Priority -1 diagnostic: make every wait S is about to enqueue
        already satisfied (no blocked barrier on a high-priority queue)."""
        hsf = self._h()
        for n in names:
            req = ctx.input_dist_tensors_requests[n]
            for a in list(getattr(req, "awaitables", None) or []):
                for w in list(getattr(a, "_awaitables", None) or []):
                    hsf._poll_until_done(w)
        cuda = self._c()
        evs = []
        if self.ev is not None and self.ev_idx == _index(ctx) - 1:
            evs.append(self.ev)
        else:
            ev = cuda.Event()
            ev.record(cuda.current_stream())
            evs.append(ev)
        ev = cuda.Event()
        ev.record(p._data_dist_stream)
        evs.append(ev)
        for ev in evs:
            while not ev.query():
                time.sleep(_POLL_S)

    def issue(self, p: Any) -> None:
        """Issue the lookahead for p.contexts[0] (idempotent)."""
        if not self.on or not p.contexts or not p.batches:
            return
        ctx = p.contexts[0]
        idx = _index(ctx)
        if idx < 1:
            return  # batch 0 of every pass runs the stock path on main
        if getattr(ctx, STORE_ATTR, None) is not None:
            return  # already issued
        if self.cap is not None and idx >= self.cap:
            return  # rank-uniform: the eval loop breaks before this batch
        # PHASE A: host checks only; a failure declines on this rank before any
        # awaitable or module call, so the stock forward still finds its request.
        try:
            mods, names, reason = self._phase_a(p, ctx)
        except Exception as e:  # noqa: BLE001
            mods, names, reason = [], [], "phase A raised %s: %s" % (type(e).__name__, e)
        if reason is not None:
            self.disable(reason)
            return
        cuda = self._c()
        main = cuda.current_stream()
        S, dd = self.stream, p._data_dist_stream
        with _rf("## eval_lookahead %d ##" % idx):
            # Fences: stream waits only, still before any awaitable or module
            # call, so a failure here is a phase-A decline as well.
            try:
                if self.prio == -1:
                    self._poll_before_issue(p, ctx, names)
                if self.ev is not None and self.ev_idx == idx - 1:
                    if self.report and not self.ev.query():
                        self.stats["host_ahead"] += 1
                    S.wait_event(self.ev)  # ev_c(idx-1): wire reuse, TBE order, lead bound
                    self.stats["fenced"] += 1
                else:
                    S.wait_stream(main)
                    self.stats["fence_missing"] += 1
                    if not self.fence_warned:
                        self.fence_warned = True
                        logger.warning(
                            "%s at index %d; side stream waited on the whole main stream rank=%d",
                            FENCE_MISSING, idx, _rank(),
                        )
                S.wait_stream(dd)
                _record_shared_caches(S)
            except Exception as e:  # noqa: BLE001
                self.disable("fence raised %s: %s" % (type(e).__name__, e))
                return
            store: Dict[str, _Pending] = {}
            try:
                with cuda.stream(S), _no_grad():
                    for m, name in zip(mods, names):
                        req = ctx.input_dist_tensors_requests[name]
                        with _rf("## runtime_forward_assemble_KJT ##"):
                            kjt = req.wait()  # p_kjtt_wait records the KJT a2a buffers on S
                        del ctx.input_dist_tensors_requests[name]
                        mctx = ctx.module_contexts.pop(name)
                        kjt.record_stream(S)  # may alias data-dist a2a outputs (stride 0)
                        mctx.record_stream(S)  # recat read by invert_permute on S
                        aw = m.compute_and_output_dist(mctx, kjt)
                        shardings = self._record_wires(aw, main)
                        store[name] = _Pending(aw, mctx, kjt, shardings)
                _record_shared_caches(main)
            except BaseException:
                logger.error("%s (index %d); re-raising rank=%d", FAILURE, idx, _rank())
                raise
            setattr(ctx, STORE_ATTR, store)
        self.stats["issued"] += 1

    @staticmethod
    def _record_wires(aw: Any, main: Any) -> int:
        """record_stream(main) on each sharding's receive wire and Request
        dummy (allocated on S, read on main). Returns the sharding count."""
        shards = _awaitables_per_sharding(aw)
        for sa in shards:
            req = getattr(sa, "_tensor_awaitable", None)
            if req is None:
                continue
            _record_tensor(getattr(req, "tensor", None), main)
            _record_tensor(getattr(req, "dummy_tensor", None), main)
        return len(shards)

    # ---- forward ----------------------------------------------------------
    def _verify_due(self, idx: int) -> bool:
        return self.installed and self.verify_n > 0 and 1 <= idx <= self.verify_n

    def _record_ev(self, idx: int) -> None:
        cuda = self._c()
        ev = cuda.Event()
        ev.record(cuda.current_stream())
        self.ev, self.ev_idx = ev, idx

    def _attach_ev(self, aw: Any, idx: int) -> None:
        """ev_c(idx): recorded on the waiting (main) stream right after the
        embedding wait + construct_jagged_tensors, by an awaitable callback."""
        if not _has_callbacks(aw):
            return  # no fence for this batch: the next lookahead falls back

        def _cb(ret: Any) -> Any:
            self._record_ev(idx)
            return ret

        aw.callbacks.append(_cb)

    def forward_hit(self, fwd: Any, pend: _Pending) -> Any:
        idx = _index(fwd._context)
        cur = self._c().current_stream()
        if fwd._stream is not None:
            cur.wait_stream(fwd._stream)  # as stock (runtime_forwards :124-127)
        pend.mctx.record_stream(cur)  # reverse_indices, input_features, RW perm
        pend.kjt.record_stream(cur)
        self.stats["hit"] += 1
        if not self.active_logged:
            self.active_logged = True
            logger.info(
                "%s (pass=%d index=%d shardings=%d rank=%d)",
                ACTIVE, self.pass_no, idx, pend.shardings, _rank(),
            )
        if self._verify_due(idx):
            return self._verify(fwd, pend.aw, pend.mctx, pend.kjt, idx, "lookahead")
        self._attach_ev(pend.aw, idx)
        return pend.aw

    def forward_miss(self, fwd: Any, stock_call: Callable[[], Any]) -> Any:
        ctx = fwd._context
        idx = _index(ctx)
        verify = self._verify_due(idx)
        cap: Dict[str, Any] = {}
        mctx = None
        if verify:
            # Keep VERIFY's extra all_to_all rank-uniform: a rank on the stock
            # path recomputes too (stock vs stock). The kjt comes from the
            # request's own callback chain, so the stock call itself is unchanged.
            req = ctx.input_dist_tensors_requests.get(fwd._name)
            mctx = ctx.module_contexts.get(fwd._name)
            if not _has_callbacks(req) or mctx is None:
                raise RuntimeError(
                    "DLRMV4 eval emb lookahead VERIFY: cannot capture the stock "
                    "input of %s at index %d" % (fwd._name, idx)
                )

            def _capture(kjt: Any) -> Any:
                cap["kjt"] = kjt
                return kjt

            req.callbacks.append(_capture)
        aw = stock_call()
        self.stats["stock"] += 1
        if verify:
            return self._verify(fwd, aw, mctx, cap["kjt"], idx, "stock")
        self._attach_ev(aw, idx)
        return aw

    def _verify(self, fwd: Any, aw: Any, mctx: Any, kjt: Any, idx: int, path: str) -> Any:
        """Eager byte compare against a stock recompute on main (diagnostic).

        One more compute_and_output_dist on main: +1 TBE forward and one more
        seq all_to_all per sharding, in the same slot on every rank. The
        recompute reads the same ``kjt`` (built on S on the lookahead path), so
        the side-stream dist_init is not independently checked here (V1 F4:
        cross-arm eval-metric equality is the end-to-end check)."""
        cuda = self._c()
        main = cuda.current_stream()
        res = aw.wait()
        if self.stream is not None:
            main.wait_stream(self.stream)  # the recompute reads the S-built kjt
        ref = fwd._module.compute_and_output_dist(mctx, kjt).wait()
        ok = _jt_dicts_bits_equal(res, ref)
        self.stats["verified"] += 1
        (logger.info if ok else logger.error)(
            "%s index=%d bitwise=%s rank=%d", VERIFY_LINE, idx, ok, _rank()
        )
        self._record_ev(idx)  # after the recompute: the fence covers it too
        if not ok:
            self.stats["mismatch"] += 1
            if path == "lookahead":
                self.disable("VERIFY byte mismatch at index %d" % idx)
            res = ref
        return _ready_value(res)

    # ---- pass boundaries --------------------------------------------------
    def drain(self, p: Any, where: str) -> int:
        """Wait (on main) and discard any lookahead still parked on a context."""
        n = 0
        contexts = list(getattr(p, "contexts", None) or [])
        if not contexts:
            return 0
        cuda = self._c()
        cur = cuda.current_stream()
        dd = getattr(p, "_data_dist_stream", None)
        for ctx in contexts:
            store = getattr(ctx, STORE_ATTR, None)
            if store is None:
                continue
            try:
                delattr(ctx, STORE_ATTR)
            except AttributeError:
                pass
            for pend in list(store.values()):
                try:
                    if dd is not None:
                        cur.wait_stream(dd)
                    pend.mctx.record_stream(cur)
                    pend.aw.wait()  # completes this rank's side of the issued a2a
                except BaseException:
                    logger.error("%s (index %d); re-raising rank=%d", FAILURE, _index(ctx), _rank())
                    raise
                n += 1
            store.clear()
        if n:
            logger.warning("%s %d unconsumed lookahead(s) at %s rank=%d", DRAINED, n, where, _rank())
        return n

    def join(self) -> None:
        """main.wait_stream(S): every S op (TBE forwards included) is ordered
        before whatever main runs next (training, checkpoint)."""
        if self.installed and self.stream is not None:
            self._c().current_stream().wait_stream(self.stream)

    def new_generation(self) -> None:
        """Never fence on an event from before a detach/attach or a pass end."""
        self.ev, self.ev_idx = None, -1

    def end_pass(self) -> None:
        if not self.installed:
            return
        st = self.stats
        if self.report:
            logger.info(
                "%s rank=%d pass=%d issued=%d hit=%d stock=%d host_ahead=%d/%d",
                REPORT_LINE, _rank(), self.pass_no, st["issued"], st["hit"], st["stock"],
                st["host_ahead"], st["fenced"],
            )
        if st["issued"] or st["hit"] or st["stock"]:
            self.pass_no += 1
        self.stats = self._zero()
        self.new_generation()


def _kjt_wait_records_current() -> bool:
    """host_sync_free.p_kjtt_wait is torchrec's KJT tensors wait (it records the
    data-dist a2a buffers on the waiting stream). True if torchrec is absent
    (CPU unit tests inject their own pipeline)."""
    try:
        import torchrec.distributed.dist_data as dd
    except Exception:  # noqa: BLE001
        return True
    fn = getattr(dd.KJTAllToAllTensorsAwaitable, "_wait_impl", None)
    return bool(getattr(dd, "_dlrm_lpk_patched", False)) and getattr(fn, "__name__", "") == "p_kjtt_wait"


# --------------------------------------------------------------------------
# The eval pipeline subclass
# --------------------------------------------------------------------------
def wrap(base: Any, *, forward_base: Any = None, cuda: Any = None, hsf: Any = None) -> Any:
    """The eval-lookahead subclass of ``base`` (ship14's _StreamingTrainPipeline).

    Only build_train_pipeline(role="eval") with the knob on calls this; the
    train pipeline class and its methods are untouched.
    """
    fwd_cls = forward_class(forward_base)
    stock_fwd = getattr(base, "_pipelined_forward_type", None)

    class EvalLookaheadPipeline(base):
        _pipelined_forward_type = fwd_cls  # replaced per instance in __init__

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            ctl = _Lookahead(cuda, hsf, fwd_cls)
            self._dlrm_evla = ctl
            try:
                reason = ctl.build_check(self)
            except Exception as e:  # noqa: BLE001
                reason = "build check raised %s: %s" % (type(e).__name__, e)
            if reason is None:
                ctl.install(self)
            else:
                # A decline before RUN_START already logged its WARNING.
                warned = bool(_prepared["done"]) and _prepared["stream"] is None
                ctl.decline(self, reason, stock_fwd, warn=not warned)

        def dlrm_evla_configure(self, num_eval_batches: Optional[int] = None,
                                interim_metrics: bool = False) -> None:
            """Once, right after the build (train/utils.py _run_eval_window)."""
            ctl = self._dlrm_evla
            ctl.cap = (
                int(num_eval_batches)
                if num_eval_batches is not None and int(num_eval_batches) > 0
                else None
            )
            if interim_metrics and ctl.on:
                ctl.decline(
                    self,
                    "EVAL_INTERIM_LOG>0: compute_and_log collectives sit between eval batches",
                    stock_fwd,
                )

        def fill_pipeline(self, dataloader_iter: Any) -> None:
            super().fill_pipeline(dataloader_iter)
            ctl = self._dlrm_evla
            if ctl.on and ctl.issue_mode == "top":
                ctl.issue(self)  # top of progress(k), k >= 1

        def progress(self, dataloader_iter: Any) -> Any:
            out = super().progress(dataloader_iter)
            ctl = self._dlrm_evla
            if ctl.on and ctl.issue_mode == "end":
                ctl.issue(self)  # end of progress(N): batch N+1
            return out

        def detach(self) -> Any:
            m = super().detach()
            ctl = self._dlrm_evla
            ctl.drain(self, "detach")
            ctl.join()
            ctl.new_generation()
            return m

        def attach(self, model: Any = None, sparse_dist: bool = True) -> None:
            ctl = self._dlrm_evla
            ctl.drain(self, "attach")  # never resolved across detach/attach
            ctl.new_generation()
            super().attach(model, sparse_dist)

        def reset(self) -> None:
            ctl = self._dlrm_evla
            ctl.drain(self, "reset")
            ctl.join()
            ctl.end_pass()
            super().reset()

    EvalLookaheadPipeline.__qualname__ = "EvalLookahead" + getattr(base, "__name__", "Pipeline")
    return EvalLookaheadPipeline


def _reset_for_tests() -> None:
    _prepared.update(done=False, stream=None, prio=0, tries=0, pool_gib=0.0, small_mib=0,
                     ready=0, world=1, reason=None)
    _bad_env_logged.clear()
