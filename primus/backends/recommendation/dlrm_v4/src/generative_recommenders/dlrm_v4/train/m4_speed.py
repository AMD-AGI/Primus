# pyre-strict
"""M4 speed estimator (API v2): per-rank device speed factors at train-window boundaries.

x4c19 half of M4, the speed-aware within-batch anchor dealing in comm's
dlrm_v4/train/m4_partition.py. Default OFF: unless DLRM_HSTU_FP8_M4 is "1",
install() installs nothing, main_forward_begin() returns at its first check,
no event is recorded, no collective runs and on_window_boundary() returns at
once. comm's gin item binds the switch; this module adds no second binding.

WHAT
  g_r is rank r's measured span time per unit of predicted cost (the same
  per-anchor law comm allocates with: COST_B*N + COST_C*N*(N+1)/2 ms, so the
  closed loop has the fixed point "equal measured spans"). Per training step s
  of a window, r_s = span_s / cost_s, where cost_s is the cost of the batch
  this rank trained at that step (note_window_costs, aligned by step index).
  Per rank, accepted r_s are pooled across windows until EVERY rank holds
  MIN_STEPS of them; the rank's statistic is the median of its pool. The
  gathered statistics are normalised to mean 1, smoothed by an EMA,
  renormalised and clamped to [0.9, 1.1]. comm targets rank r's cost share
  proportional to 1 / g_r (and, optionally, equalises g_r*load_r + F_r with
  the static offsets F of offsets_for).

SPAN "P_mf" (default; x4c19 iter61 proxy track + its verification)
  E_start: a torch.cuda.Event(enable_timing=True) recorded on the current
    (main compute) stream at the top of DlrmHSTU.main_forward, by the one call
    main_forward_begin(self, seq_embeddings) that dlrm_hstu.py makes there.
    main_forward runs after preprocess(), whose first touch of the EC output
    waits on the forward sequence all-to-all, so E_start follows that
    cross-rank sync point. Recorded only when module.training and
    torch.is_grad_enabled(), not while compiling or fx tracing (TorchRec's
    pipeline fx-traces the model; the call site is also skipped under
    TorchScript / fx tracing), not inside a backward graph task, and only for
    real tensors.
  E_end: recorded when the gradients w.r.t. ALL item-candidate embeddings
    (seq_embeddings[c].embedding for c in item_embedding_feature_names) are
    final: torch.autograd.graph.register_multi_grad_hook(mode="all") on those
    tensors (one tensor hook if there is only one). _item_forward concatenates
    them, so they always get a gradient, and that CatBackward0 is the last
    main_forward backward node (1920/1920 traced ship13 rank-steps), so E_end
    is the end of the dense backward; the EC expand backward and the backward
    sequence all-to-all come after it. NOT hooked, on purpose: the other
    seq_embeddings. Under EC_CAT_FUSE (gin default) artist/album are column
    views whose base the preprocessor consumes directly, so their hooks never
    fire; and a hook on an E-X2b/IX member (the item `values`, the cat
    flat/big) makes E-X2b decline at claim() (it checks _backward_hooks). A
    candidate tensor carrying any EC member tag is skipped. The hooks return
    None (no gradient is touched) and never synchronise; their handles are
    removed when the step is harvested. E_end goes on the stream E_start used.
  Between them: the merge, item/user forward (preprocessor, STU stack,
  postprocessor), multitask head and loss, and their backward down to the
  item-candidate input grads (95% of the per-rank main-stream busy time at
  1n/2n/4n on ship13; 93% on the ship14 S14 1n smoke traces; no collective
  wait inside at clean steps). Outside: embedding lookups, a2a glue, EC/TBE
  backward (with E-X2b also the preprocessor L1 dX), optimizer: the per-rank
  work offsets_for can add back at 1n. Host stalls DO land inside the span
  (progress() pulls batch N+2 and runs the deferred input dist of N+1 between
  the forward and the backward launch): the robust readout below rejects them.
  Diagnostic span "P_stu" (DLRM_HSTU_FP8_M4_SPAN=stu, v1's placement: forward
  pre-hook on STUStack._stu_layers[0], end at the grad of its input) is kept
  for A/B only: it over-corrects at 2n/4n (kernel-mix dependence).

READOUT (per rank; robust to host stalls inside the span)
  At boundary i >= 1, window k = i - 1 is closed: completed pairs are
  harvested (Event.query(), never waited on). Step 0 of the window (the first
  step after the barrier) is skipped. For the other measured steps m with a
  cost entry: r_m = elapsed_m / costs[m]. If the rank has a reference level
  (its own statistic at the last valid boundary), steps with
  |r_m / ref - 1| > 10% are dropped as stalls (unless they are more than half
  of a window of >= 8 steps: then the reference is treated as stale for that
  window). The remaining r_m are clipped to +-2.5% of their window median and
  appended to the rank's pool (bounded). Statistic = median of the pool.
  Window k = 0 (the first window of the process) is a warm-up window: its
  steps are not pooled and k = 0 is always None.

BOUNDARIES (main thread, right after _window_boundary_barrier)
  on_window_boundary(device) runs once per _traced_window_boundary call, i.e.
  at the start of EVERY train window i of the process; i == comm's run-local
  window index j (0 = the first window this process trains; a resume restarts
  at 0). i is counted here and nowhere else.
  i = 0: ARM (only when DLRM_HSTU_FP8_M4=1; the OFF path is collective-free).
    One all_gather of [config hash, offsets hash, offsets status, world,
    flag]. ON on every rank iff every rank is installed, not declined and runs
    the same diagnostic knobs; otherwise OFF on every rank, and no further
    work at any barrier. Offsets armed iff every rank parsed the same valid
    vector. Pre-window records (warm-ups) are dropped. Callbacks(0).
  i >= 1 while ON: CLOSE window k = i - 1. ONE all_gather_into_tensor (default
    process group) of [stat, n_accepted, n_measured, n_costs, flag | log-only:
    max/median step ratio, incomplete, orphan, hook_error, no_grad,
    live_overflow]; every decision is taken on the gathered rows in Python
    float64 arithmetic (math.fsum), so it is identical on all ranks:
      any flag != OK (a local decline or a local measurement error, from any
        thread) -> OFF on every rank from this boundary: k and every later k
        are decided None;
      k >= 1, EVERY rank n_accepted >= MIN_STEPS with a finite stat > 0 ->
        VALID: EMA update with the mean-1 vector; every rank resets its pool
        and takes its own stat as the new stall reference;
      published[k] = clamp(EMA / mean(EMA), 0.9, 1.1); None before the first
        valid boundary; an invalid boundary republishes the carried EMA. With
        FREEZE_AFTER=n, the vector of the n-th valid boundary is republished
        from then on. SANITIZE: a vector that is not W finite values > 0 with
        max/min <= 4 is published as None. offsets[k] = the armed offsets iff
        published[k] is not None.
    Then waiters are woken and rank 0 logs '[m4] factors k=..'.
  Callbacks(i) run at EVERY barrier i once armed (i = 0 included), also after
    a global OFF (comm's owner-hash check must keep running while windows
    dealt with earlier factors can still start); never when OFF at arming.
  Reading the gathered tensor is the only host sync; it follows the barrier.

DETERMINISM (comm's contract)
  factors_for(k) / offsets_for(k) return the DECIDED value of boundary k. For
  an undecided k they wait until boundary k is decided (vector or None), a
  global OFF covers k, or shutdown() is called. Rank-local state (a local
  decline, a hook error, an install failure) never changes an answer on one
  rank only: it travels in the next all_gather, and every rank switches at
  the same boundary.

API (API_VERSION 2)
  COST_B, COST_C (= COST_B_MS, COST_C_MS), anchor_costs(N).
  requested(), enabled(), set_enabled(on, reason), install(model),
  main_forward_begin(module, seq_embeddings), on_window_boundary(device),
  note_window_costs(j, costs), factors_for(k, timeout_s),
  offsets_for(k, timeout_s), register_boundary_callback(fn), shutdown(reason),
  status(); pure helpers decide(...), parse_offsets(raw, w).

KNOBS (env; must be identical on all ranks, checked at arming)
  DLRM_HSTU_FP8_M4 = "1" -> on (same parse as m4_partition.switch_on).
  DLRM_HSTU_FP8_M4_MIN_STEPS (default 32), DLRM_HSTU_FP8_M4_EMA (default 0.5).
  Diagnostic: DLRM_HSTU_FP8_M4_OFFSETS_MS (W comma-separated ms >= 0),
  DLRM_HSTU_FP8_M4_FREEZE_AFTER (n >= 1; unset/0 = off),
  DLRM_HSTU_FP8_M4_SPAN (mf | stu), DLRM_HSTU_FP8_M4_LOG_ALL (=1: per-rank
  n, measured/cost counts on the rank-0 boundary line; the per-rank stat,
  max/median step ratio, published vector and nonzero counters are always
  logged, at most 32 values per field).

NUMERICS
  No model math is touched: the events and hooks leave every tensor alone.
  With M4 on, the numerics change comes from comm's dealing only.
"""

from __future__ import annotations

import collections
import hashlib
import logging
import math
import os
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.fx._symbolic_trace import is_fx_tracing

logger: logging.Logger = logging.getLogger(__name__)

API_VERSION = 2

ENV_ON = "DLRM_HSTU_FP8_M4"
ENV_EMA = "DLRM_HSTU_FP8_M4_EMA"
ENV_MIN_STEPS = "DLRM_HSTU_FP8_M4_MIN_STEPS"
ENV_OFFSETS = "DLRM_HSTU_FP8_M4_OFFSETS_MS"  # diagnostic
ENV_FREEZE = "DLRM_HSTU_FP8_M4_FREEZE_AFTER"  # diagnostic
ENV_SPAN = "DLRM_HSTU_FP8_M4_SPAN"  # diagnostic
ENV_LOG_ALL = "DLRM_HSTU_FP8_M4_LOG_ALL"  # diagnostic (shared with m4_partition)

# Per-anchor cost law from synthetic-shape GPU timing only (x4c19 job 71809, uniform designs D3-D5,
# no dataset content; replaces the removed real-data iter58 fit): busy_ms = 7.62 + 38.157*T + 13.509*C
# per rank-step, T in 1e6 tokens, C = sum N(N+1)/2 in 1e9. comm's allocation reads
# COST_B_MS / COST_C_MS, so allocation and normalisation share one law.
COST_B = 38.157e-6  # ms per token
COST_C = 13.509e-9  # ms per causal (query, key) pair
COST_B_MS = COST_B
COST_C_MS = COST_C

CLAMP_LO = 0.9
CLAMP_HI = 1.1
DEFAULT_ALPHA = 0.5
DEFAULT_MIN_STEPS = 32
CLIP_FRAC = 0.025  # per-step ratio clip around the rank's window median
STALL_FRAC = 0.10  # drop a step deviating more than this from the rank's reference
STALE_MIN_STEPS = 8  # a window this long losing > half its steps -> reference stale
DEFAULT_TIMEOUT_S = 600.0
# Live step records (events not yet harvested). The host runs ~1-2 steps
# ahead of the GPU, so a handful is in flight; the bound only caps a stall.
MAX_LIVE = 64
# Harvested (step index, ms) pairs kept per window (largest 1n window: 597).
MAX_DONE = 1 << 16
# Pooled accepted ratios per rank (reset at every valid boundary).
MAX_ACC = 1 << 16
_POOL_MAX = 2 * MAX_LIVE + 8
# Decision row (decide()): [stat, n_accepted, n_measured, n_costs, flag].
_ROW = 5
# Gathered boundary row: the decision row + log-only fields.
LOG_COUNTERS = ("incomplete", "orphan", "hook_error", "no_grad", "live_overflow")
_BROW = _ROW + 1 + len(LOG_COUNTERS)  # + max/median step ratio + counters
# Gathered arming row (i = 0): [config hash, offsets hash, offsets status, world, flag].
_AROW = 5
FLAG_OK = 1.0
FLAG_OFF = 0.0  # declined on this rank (set_enabled(False), not installed)
FLAG_ERR = 2.0  # local measurement error on this rank
MAX_SPREAD = 4.0  # sanitize: a published vector must have max/min <= this
LOG_MAX = 32  # values per logged field
SPANS = ("mf", "stu")
# Tags the EC puts on E-X2b / Indexed-X / merge / cat-fuse member outputs
# (triton_ec_dgrad_segreduce.PORT_ATTR, ec_indexed_l1.IX_ATTR,
# ec_bf16_output.PREMERGED_ATTR / CATFUSED_ATTR): never hooked.
_MEMBER_TAGS = (
    "_dlrmv4_ec_dgrad_port",
    "_dlrmv4_ec_indexed_l1",
    "_dlrmv4_uih_cand_premerged",
    "_dlrmv4_catfused",
)


class M4FactorTimeout(RuntimeError):
    """factors_for() / offsets_for() waited longer than its timeout."""


def anchor_costs(n: Any) -> np.ndarray:
    """Predicted cost (ms, float64) of anchors of attention length N."""
    n = np.asarray(n, dtype=np.float64)
    return COST_B * n + COST_C * n * (n + 1.0) * 0.5


def requested() -> bool:
    """The env switch (comm's gin item binds it; env wins). Read per call."""
    return os.environ.get(ENV_ON, "0").strip() == "1"


def _env_alpha() -> float:
    raw = os.environ.get(ENV_EMA, "").strip()
    if not raw:
        return DEFAULT_ALPHA
    a = float(raw)
    if not (0.0 < a <= 1.0):
        raise ValueError(f"{ENV_EMA}={raw!r} must be in (0, 1]")
    return a


def _env_min_steps() -> int:
    raw = os.environ.get(ENV_MIN_STEPS, "").strip()
    if not raw:
        return DEFAULT_MIN_STEPS
    v = int(raw)
    if v < 1:
        raise ValueError(f"{ENV_MIN_STEPS}={raw!r} must be >= 1")
    return v


def _env_freeze() -> int:
    raw = os.environ.get(ENV_FREEZE, "").strip()
    if not raw:
        return 0
    v = int(raw)
    if v < 0:
        raise ValueError(f"{ENV_FREEZE}={raw!r} must be >= 0 (0 = off)")
    return v


def _env_span() -> str:
    raw = os.environ.get(ENV_SPAN, "").strip().lower()
    if raw in ("", "mf", "p_mf"):
        return "mf"
    if raw in ("stu", "p_stu"):
        return "stu"
    raise ValueError(f"{ENV_SPAN}={raw!r} must be 'mf' (default) or 'stu' (diagnostic)")


def parse_offsets(raw: Optional[str], w: int) -> Tuple[int, Optional[np.ndarray], str]:
    """(status, vector, reason) of DLRM_HSTU_FP8_M4_OFFSETS_MS for world size w.

    status 1: exactly w finite values >= 0 (ms) -> float64 [w]; 0: unset;
    -1: invalid (reason says why). Pure and deterministic in (raw, w).
    """
    raw = (raw or "").strip()
    if not raw:
        return 0, None, ""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != int(w):
        return -1, None, f"{len(parts)} values for world size {int(w)}"
    try:
        vals = [float(p) for p in parts]
    except ValueError as e:
        return -1, None, f"not a list of floats ({e})"
    if not all(math.isfinite(v) and v >= 0.0 for v in vals):
        return -1, None, "every value must be finite and >= 0"
    return 1, np.array(vals, dtype=np.float64), ""


class _Rec:
    """One training step: its events, its index in the window, its hook handles."""

    __slots__ = ("idx", "start", "end", "stream", "ended", "dead", "handles")

    def __init__(self, idx: int, start: Any, end: Any) -> None:
        self.idx = idx
        self.start = start
        self.end = end
        self.stream: Any = None
        self.ended = False
        self.dead = False
        self.handles: List[Any] = []


class _State:
    def __init__(self) -> None:
        self.cond = threading.Condition()
        # local_off: set_enabled(False) on this rank (travels in the next gather);
        # global_off / off_from: decided on gathered data (identical on all ranks).
        self.local_off: Optional[str] = None
        self.global_off: Optional[str] = None
        self.off_from: Optional[int] = None  # every undecided k >= off_from is None
        # None until boundary 0, then True (ON on every rank) / False (OFF for good).
        self.armed: Optional[bool] = None
        self.inert = False  # M4 not requested at the first boundary: nothing, ever
        self.stopped: Optional[str] = None
        self.installed = False
        self.hooked: Optional[str] = None
        self.span = "mf"
        self.require_cuda = True
        self.hooks_active = False
        self.remove_pending = False
        self.module_handles: List[Any] = []
        # Current-window measurement (main thread; hooks touch only their _Rec).
        self.step_idx = 0
        self.live: Deque[_Rec] = collections.deque()
        self.done: List[Tuple[int, float]] = []
        self.pool: List[Any] = []
        self.counters: Dict[str, int] = collections.Counter()
        self.cnt_snap: Dict[str, int] = {}  # counters at the previous boundary
        self.maxmed = float("nan")  # max/median step ratio of the closed window
        # Pooled accepted ratios and the stall reference (main thread).
        self.acc: Deque[float] = collections.deque(maxlen=MAX_ACC)
        self.ref: Optional[float] = None
        # Cross-thread state (under cond).
        self.costs: Dict[int, np.ndarray] = {}
        self.noted_k = False  # note_window_costs was called for the window just closed
        self.published: Dict[int, Optional[np.ndarray]] = {}
        self.offsets_pub: Dict[int, Optional[np.ndarray]] = {}
        self.last_k = -1
        self.callbacks: List[Callable[[int], None]] = []
        # Boundary bookkeeping (main thread).
        self.boundary = 0  # index i of the next on_window_boundary call
        self.ema: Optional[List[float]] = None
        self.n_valid = 0
        self.frozen: Optional[List[float]] = None
        self.alpha = DEFAULT_ALPHA
        self.min_steps = DEFAULT_MIN_STEPS
        self.freeze_after = 0
        self.log_all = False
        self.offsets: Optional[np.ndarray] = None
        self.offsets_desc = "none"
        self.last_rows: Optional[List[List[float]]] = None
        self.warned: set = set()


_S = _State()


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0") or "0")


def _world() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _print_all_ranks(msg: str) -> None:
    try:
        from generative_recommenders.ops.utils import arm_print_all_ranks
    except Exception:  # noqa: BLE001 -- logging only (unit tests run without the tree)
        print(f"{msg} [rank {_rank()}]", flush=True)
        return
    arm_print_all_ranks(msg)


def _vec_sha(v: Optional[np.ndarray]) -> str:
    if v is None:
        return "-"
    return hashlib.sha256(np.ascontiguousarray(v, dtype="<f8").tobytes()).hexdigest()[:16]


def _hash48(b: bytes) -> float:
    """48-bit digest as an exactly representable float64 (for the arming gather)."""
    return float(int.from_bytes(hashlib.sha256(b).digest()[:6], "little"))


def _config_hash(st: _State) -> float:
    s = (
        f"m4v{API_VERSION}|span={st.span}|alpha={st.alpha!r}|min={st.min_steps}|"
        f"freeze={st.freeze_after}|clip={CLIP_FRAC!r}|stall={STALL_FRAC!r}|"
        f"clamp={CLAMP_LO!r},{CLAMP_HI!r}|b={COST_B!r}|c={COST_C!r}"
    )
    return _hash48(s.encode())


def _median(v: Sequence[float]) -> float:
    return float(np.median(np.asarray(v, dtype=np.float64)))


def _warn_once(st: _State, key: str, msg: str, *args: Any) -> None:
    if key not in st.warned:
        st.warned.add(key)
        logger.warning(msg, *args)


# ----------------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------------


def _take_event(st: _State) -> Any:
    if st.pool:
        return st.pool.pop()
    # Looked up per call so tests can monkeypatch torch.cuda.Event.
    return torch.cuda.Event(enable_timing=True)


def _recycle(st: _State, *events: Any) -> None:
    for ev in events:
        if len(st.pool) < _POOL_MAX:
            st.pool.append(ev)


def _in_backward() -> bool:
    gt = getattr(torch._C, "_current_graph_task_id", None)
    if gt is not None:
        try:
            return gt() != -1
        except Exception:  # noqa: BLE001
            pass
    return threading.current_thread() is not threading.main_thread()


def _is_compiling() -> bool:
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001
        return False


def _drop_rec(st: _State, rec: _Rec, recycle: bool) -> None:
    for h in rec.handles:
        try:
            h.remove()
        except Exception:  # noqa: BLE001 -- the node may be gone already
            pass
    rec.handles = []
    if recycle:
        _recycle(st, rec.start, rec.end)


def _harvest(st: _State, final: bool = False) -> None:
    """Move completed pairs from live to done (oldest first, never blocks).

    A record whose end hook has not fired while a newer step already began
    had no (complete) backward: dropped as an orphan. With ``final`` (after
    the boundary barrier) everything is resolved: complete pairs harvested,
    the rest dropped and counted. Dropped events are not recycled (a late
    record could still land on them).
    """
    live = st.live
    while live:
        rec = live[0]
        if rec.dead:
            live.popleft()
            st.counters["hook_error"] += 1
            _drop_rec(st, rec, False)
            continue
        if not rec.ended:
            if final or len(live) > 1:
                live.popleft()
                st.counters["orphan"] += 1
                _drop_rec(st, rec, False)
                continue
            return
        if not (rec.start.query() and rec.end.query()):
            if final:
                live.popleft()
                st.counters["incomplete"] += 1
                _drop_rec(st, rec, False)
                continue
            return
        ms = float(rec.start.elapsed_time(rec.end))
        live.popleft()
        if len(st.done) < MAX_DONE:
            st.done.append((rec.idx, ms))
        else:
            st.counters["done_overflow"] += 1
        _drop_rec(st, rec, True)


def _make_end_hook(rec: _Rec) -> Callable[[Any], None]:
    def _m4_end_hook(_grads: Any) -> None:
        # Autograd thread (CUDA: the device thread), when the hooked gradients
        # are final (before their producer's backward runs). Returns None: the
        # gradient is untouched. One record per step (later calls are no-ops).
        if rec.ended or rec.dead:
            return None
        if _S.hooks_active:
            try:
                if rec.stream is not None:
                    rec.end.record(rec.stream)
                else:
                    rec.end.record()
                rec.ended = True
            except Exception:  # noqa: BLE001 -- a lost sample must not stop training
                rec.dead = True
        return None

    return _m4_end_hook


def _register_end_hooks(rec: _Rec, tensors: List[torch.Tensor]) -> None:
    """E_end hook: all of ``tensors`` have their final gradient (see SPAN)."""
    fn = _make_end_hook(rec)
    if len(tensors) > 1:
        multi = getattr(torch.autograd.graph, "register_multi_grad_hook", None)
        if multi is not None:
            rec.handles.append(multi(list(tensors), fn, mode="all"))
            return
    rec.handles.append(tensors[0].register_hook(fn))


def _begin_step(st: _State, tensors: List[torch.Tensor]) -> None:
    """E_start of one training step + its end hook (main thread)."""
    _harvest(st)
    idx = st.step_idx
    st.step_idx = idx + 1  # this step's index in the window, kept even if unmeasured
    if not tensors:
        st.counters["no_grad"] += 1
        return
    if st.require_cuda and not all(t.is_cuda for t in tensors):
        st.counters["not_cuda"] += 1
        return
    if len(st.live) >= MAX_LIVE:
        _drop_rec(st, st.live.popleft(), False)
        st.counters["live_overflow"] += 1
    rec = _Rec(idx, _take_event(st), _take_event(st))
    try:
        if st.require_cuda:
            rec.stream = torch.cuda.current_stream()
            rec.start.record(rec.stream)
        else:
            rec.start.record()
        _register_end_hooks(rec, tensors)
    except Exception as e:  # noqa: BLE001 -- a lost sample must not stop training
        rec.dead = True
        _drop_rec(st, rec, False)
        st.counters["hook_error"] += 1
        _warn_once(st, "hook_error", "[m4] speed estimator: step not measured (%s: %s)",
                   type(e).__name__, e)
        return
    st.live.append(rec)


def _member_tagged(t: torch.Tensor) -> bool:
    return any(getattr(t, a, None) is not None for a in _MEMBER_TAGS)


def main_forward_begin(module: torch.nn.Module, seq_embeddings: Any) -> None:
    """P_mf E_start + E_end hook; called at the top of DlrmHSTU.main_forward.

    A no-op unless the estimator is installed and active (OFF: one attribute
    check). Records only for a training forward with grad enabled. The end
    hook goes on the item-candidate embeddings only (see SPAN).
    """
    st = _S
    if not st.hooks_active or st.span != "mf":
        return None
    if not module.training or not torch.is_grad_enabled():
        return None
    if _is_compiling() or is_fx_tracing() or _in_backward():
        return None
    cfg = getattr(module, "_hstu_configs", None)
    names = getattr(cfg, "item_embedding_feature_names", None)
    if not names:
        st.counters["no_item_names"] += 1
        return None
    tensors: List[torch.Tensor] = []
    for c in names:
        try:
            t = getattr(seq_embeddings[c], "embedding", None)
        except Exception:  # noqa: BLE001 -- unexpected payload: not a measurable step
            return None
        if not isinstance(t, torch.Tensor):
            return None  # fx Proxy or an unexpected payload: not a training step
        if t.requires_grad:
            if _member_tagged(t):
                st.counters["member_skip"] += 1  # never hook an EC member output
            else:
                tensors.append(t)
    _begin_step(st, tensors)
    return None


def _stu_pre_hook(module: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    """Diagnostic span P_stu: forward pre-hook on STUStack._stu_layers[0]."""
    st = _S
    if not st.hooks_active or st.span != "stu":
        return None
    if not module.training or not torch.is_grad_enabled():
        return None
    if _is_compiling() or is_fx_tracing() or _in_backward():
        return None
    x = kwargs.get("x") if kwargs else None
    if x is None and args:
        x = args[0]
    if not isinstance(x, torch.Tensor):
        return None  # fx tracing (Proxy) or an unexpected call
    _begin_step(st, [x] if x.requires_grad else [])
    return None


def _remove_hooks(st: _State) -> None:
    """Main thread: module hooks off, live records dropped (handles removed)."""
    st.hooks_active = False
    for h in st.module_handles:
        try:
            h.remove()
        except Exception:  # noqa: BLE001
            pass
    st.module_handles = []
    for rec in st.live:
        rec.dead = True
        _drop_rec(st, rec, False)
    st.live.clear()
    st.done = []
    st.remove_pending = False


def install(model: torch.nn.Module, require_cuda: bool = True) -> bool:
    """Arm the measurement on the freshly built model (no-op when off).

    Called once by make_model() (after apply_env_bootstrap, so comm's gin
    binding is visible). P_mf: checks that the model has a main_forward that
    calls main_forward_begin (the dlrm_hstu.py hunk); P_stu: registers the
    layer-0 pre-hook. Malformed diagnostic knobs raise (identically on every
    rank). ``require_cuda=False`` is for CPU tests. True when installed.
    """
    st = _S
    if not requested():
        return False
    if st.local_off is not None or st.global_off is not None or st.armed is False:
        return False
    if st.installed:
        return True
    st.span = _env_span()
    st.alpha = _env_alpha()
    st.min_steps = _env_min_steps()
    st.freeze_after = _env_freeze()
    st.log_all = os.environ.get(ENV_LOG_ALL, "0").strip() == "1"
    if require_cuda and not torch.cuda.is_available():
        set_enabled(False, "CUDA not available")
        return False
    if st.span == "mf":
        found = None
        for name, m in model.named_modules():
            fn = getattr(type(m), "main_forward", None)
            if callable(fn):
                found = (name, m, fn)
                break
        if found is None:
            set_enabled(False, "no module with a main_forward (DlrmHSTU) in the model")
            return False
        name, m, fn = found
        code = getattr(fn, "__code__", None)
        if code is None or "main_forward_begin" not in code.co_names:
            set_enabled(
                False,
                f"{type(m).__name__}.main_forward does not call "
                "m4_speed.main_forward_begin (dlrm_hstu.py hunk missing)",
            )
            return False
        st.hooked = f"{name or '<model>'}.main_forward ({type(m).__name__})"
    else:
        found = None
        for name, m in model.named_modules():
            layers = getattr(m, "_stu_layers", None)
            if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                found = (name, m, layers)
                break
        if found is None:
            set_enabled(False, "no STU stack (_stu_layers) in the model")
            return False
        name, stack, layers = found
        st.module_handles.append(
            layers[0].register_forward_pre_hook(_stu_pre_hook, with_kwargs=True)
        )
        st.hooked = (
            f"{name or '<model>'}._stu_layers[0] ({type(layers[0]).__name__}) of a "
            f"{len(layers)}-layer {type(stack).__name__}"
        )
        if len(layers) < 2:
            logger.warning(
                "[m4] speed estimator (span=stu): 1-layer stack; with the "
                "targets-only last layer the layer-0 pre-hook never fires"
            )
    st.require_cuda = bool(require_cuda)
    st.installed = True
    st.hooks_active = True
    if _rank() == 0:
        logger.info(
            "[m4] speed estimator installed on %s (span=%s); arms at the first "
            "window boundary",
            st.hooked,
            "P_mf" if st.span == "mf" else "P_stu (diagnostic)",
        )
    return True


# ----------------------------------------------------------------------------
# on / off
# ----------------------------------------------------------------------------


def enabled() -> bool:
    """True while M4 is on for this rank (the single source of truth)."""
    st = _S
    return (
        requested()
        and not st.inert
        and st.local_off is None
        and st.global_off is None
        and st.armed is not False
    )


def set_enabled(on: bool, reason: str = "") -> bool:
    """Decline the estimator on this rank (a one-way latch); returns enabled().

    ``on=True`` never turns M4 on (only DLRM_HSTU_FP8_M4=1 does). ``on=False``
    from any thread, at any time: measurement stops on this rank at once; the
    flag travels in the next boundary's all_gather (boundary 0 included), and
    M4 then goes off on EVERY rank at that same boundary. Until then the
    answers of factors_for/offsets_for do not change on this rank.
    """
    if on:
        return enabled()
    st = _S
    with st.cond:
        first = st.local_off is None
        if first:
            st.local_off = reason or "set_enabled(False)"
        st.hooks_active = False
        st.remove_pending = True
    if threading.current_thread() is threading.main_thread():
        _remove_hooks(st)
    if first and requested() and not st.inert and st.armed is not False:
        _print_all_ranks(
            f"[m4] speed estimator disabled on this rank: {st.local_off} "
            + (
                "(all ranks go off at the next window boundary)"
                if st.armed
                else "(all ranks stay OFF: decided at the first window boundary)"
            )
        )
    return False


def shutdown(reason: str = "shutdown") -> None:
    """No more windows will be trained: wake every waiter (undecided k -> None).

    Call it only when no rank trains another window (e.g. from the window
    stream's teardown); it does not change the collective schedule.
    """
    st = _S
    with st.cond:
        if st.stopped is None:
            st.stopped = reason
        st.cond.notify_all()


def register_boundary_callback(fn: Callable[[int], None]) -> None:
    """fn(i) at every window boundary i while M4 is on (main thread).

    It runs after the decision of boundary k = i - 1 (nothing at i = 0), so
    factors_for(i - 1) does not block inside it; in registration order, at the
    same program point on every rank, so it may run collectives. Register
    identically on all ranks, before window 0. Exceptions propagate.
    """
    if not callable(fn):
        raise TypeError("m4_speed.register_boundary_callback needs a callable")
    with _S.cond:
        _S.callbacks.append(fn)


# ----------------------------------------------------------------------------
# costs, factors, offsets
# ----------------------------------------------------------------------------


def note_window_costs(j: int, costs: Any) -> None:
    """Record this rank's predicted cost (ms) per batch of run-local window j.

    comm's push() calls it for every window it installs, from the main or the
    pusher thread, before that window trains. costs[m] is the cost of this
    rank's m-th batch of window j, aligned with the window's first trained
    step. A window without a call has n_costs 0 (invalid window, EMA carried).
    """
    st = _S
    if st.inert or not requested() or st.global_off is not None:
        return
    c = np.array(costs, dtype=np.float64, copy=True).reshape(-1)
    if c.size and (not np.all(np.isfinite(c)) or np.any(c < 0.0)):
        raise ValueError(f"m4_speed.note_window_costs(j={j}): invalid costs")
    with st.cond:
        if int(j) < st.boundary - 1:
            st.counters["late_costs"] += 1  # that window is already closed
            return
        st.costs[int(j)] = c


def _wait_decided(k: int, timeout_s: float, what: str) -> Optional[np.ndarray]:
    st = _S
    k = int(k)
    if k < 0 or st.inert or not requested():
        return None
    deadline = time.monotonic() + float(timeout_s)
    with st.cond:
        while True:
            if k in st.published:
                v = (st.published if what == "factors" else st.offsets_pub).get(k)
                return None if v is None else v.copy()
            if st.off_from is not None and k >= st.off_from:
                return None  # decided None by a global OFF
            if st.stopped is not None:
                return None  # global stop: that window is never trained
            if threading.current_thread() is threading.main_thread():
                raise RuntimeError(
                    f"m4_speed.{what}_for({k}) on the main thread before boundary {k} "
                    f"is decided (last decided k={st.last_k}): the main thread decides "
                    "it at the barrier of window k+1, so waiting here would deadlock"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise M4FactorTimeout(
                    f"m4_speed.{what}_for({k}) timed out after {float(timeout_s):.0f} s: "
                    f"boundary {k} not decided (last decided k={st.last_k}, "
                    f"boundaries seen={st.boundary}); kill switch {ENV_ON}=0"
                )
            st.cond.wait(remaining)


def factors_for(k: int, timeout_s: float = DEFAULT_TIMEOUT_S) -> Optional[np.ndarray]:
    """Decided speed factors of boundary k (float64 [W], a copy) or None.

    None at once when M4 is not requested or k < 0. For an undecided k it
    waits for the decision, a global OFF or shutdown(); raises M4FactorTimeout
    after ``timeout_s`` (never a silent fallback) and RuntimeError on the main
    thread (the decider). Identical on all ranks for every k.
    """
    return _wait_decided(k, timeout_s, "factors")


def offsets_for(k: int, timeout_s: float = DEFAULT_TIMEOUT_S) -> Optional[np.ndarray]:
    """Static per-rank offsets F (ms, float64 [W] >= 0) published with factors[k].

    Same versioning, waiting and errors as factors_for; None when the
    offsets are unset or refused, or when factors[k] is None.
    """
    return _wait_decided(k, timeout_s, "offsets")


def decide(
    k: int,
    rows: List[List[float]],
    ema_prev: Optional[List[float]],
    alpha: float,
    min_steps: int,
) -> Tuple[bool, bool, Optional[List[float]], Optional[List[float]]]:
    """Pure decision on the gathered rows (identical on every rank).

    rows[r] = [stat, n_accepted, n_measured, n_costs, flag]. Returns (off,
    valid, ema, published). Python float64 arithmetic with math.fsum (exactly
    rounded), so the result does not depend on numpy's SIMD order on a host.
    """
    if not rows or any(len(r) != _ROW for r in rows):
        raise RuntimeError(f"m4_speed: malformed gathered rows {rows!r}")
    if any(r[4] != FLAG_OK for r in rows):
        return True, False, ema_prev, None
    valid = k >= 1 and all(
        r[1] >= min_steps and math.isfinite(r[0]) and r[0] > 0.0 for r in rows
    )
    ema = ema_prev
    if valid:
        w = len(rows)
        mean = math.fsum(r[0] for r in rows) / w
        v = [r[0] / mean for r in rows]
        if ema_prev is None or len(ema_prev) != w:
            ema = v
        else:
            ema = [alpha * a + (1.0 - alpha) * b for a, b in zip(v, ema_prev)]
    pub = None
    if ema is not None:
        m = math.fsum(ema) / len(ema)
        pub = [min(CLAMP_HI, max(CLAMP_LO, e / m)) for e in ema]
    return False, valid, ema, pub


def _local_row(st: _State, k: int) -> List[float]:
    """[stat, n_accepted, n_measured, n_costs, FLAG_OK] after window k (main thread).

    Also sets st.maxmed (log only): max / median of the window's step ratios.
    """
    _harvest(st, final=True)
    done = st.done
    with st.cond:
        costs = st.costs.pop(k, None)
        for old in [j for j in st.costs if j < k]:
            del st.costs[old]
    st.noted_k = costs is not None
    n_costs = 0 if costs is None else int(costs.shape[0])
    st.maxmed = float("nan")
    if k >= 1 and n_costs:
        rs = [
            ms / float(costs[idx])
            for idx, ms in done
            if 1 <= idx < n_costs and costs[idx] > 0.0 and ms > 0.0
        ]
        if rs:
            st.maxmed = max(rs) / _median(rs)
        if st.ref is not None and rs:
            ref = st.ref
            keep = [r for r in rs if abs(r / ref - 1.0) <= STALL_FRAC]
            if len(rs) >= STALE_MIN_STEPS and 2 * len(keep) < len(rs):
                st.counters["stall_ref_stale"] += 1  # a drift, not stalls: keep all
                keep = rs
            st.counters["stall"] += len(rs) - len(keep)
            rs = keep
        if rs:
            m = _median(rs)
            lo, hi = m * (1.0 - CLIP_FRAC), m * (1.0 + CLIP_FRAC)
            st.counters["clipped"] += sum(1 for r in rs if r < lo or r > hi)
            st.acc.extend(min(hi, max(lo, r)) for r in rs)
    n_acc = len(st.acc)
    stat = _median(st.acc) if n_acc else float("nan")
    return [stat, float(n_acc), float(len(done)), float(n_costs), FLAG_OK]


def _log_fields(st: _State) -> List[float]:
    """Log-only part of the gathered row: max/median step ratio + counter deltas."""
    snap = st.cnt_snap
    vals = [st.maxmed] + [
        float(st.counters.get(c, 0) - snap.get(c, 0)) for c in LOG_COUNTERS
    ]
    st.cnt_snap = dict(st.counters)
    return vals


def _sanitize(pub: Optional[List[float]], w: int) -> str:
    """"" or why a vector must not be published (then None is published)."""
    if pub is None:
        return ""
    if len(pub) != w:
        return f"length {len(pub)} != world {w}"
    if not all(math.isfinite(x) and x > 0.0 for x in pub):
        return "non-finite or non-positive entry"
    if max(pub) > MAX_SPREAD * min(pub):
        return f"spread > {MAX_SPREAD:g}x"
    return ""


def _reset_window(st: _State) -> None:
    for rec in st.live:
        rec.dead = True
        _drop_rec(st, rec, False)
    st.live.clear()
    st.done = []
    st.step_idx = 0


def _all_gather_rows(vals: List[float], device: Optional[torch.device]) -> List[List[float]]:
    """ONE all_gather_into_tensor on the default process group (if any)."""
    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized()):
        return [list(vals)]
    w = dist.get_world_size()
    if w <= 1:
        return [list(vals)]
    dev = torch.device(device) if device is not None else torch.device("cpu")
    host = torch.tensor(vals, dtype=torch.float64)
    if dev.type == "cuda":
        # Pinned + non_blocking: no host sync before the read below.
        inp = host.pin_memory().to(dev, non_blocking=True)
    else:
        inp = host
    out = torch.empty(w * len(vals), dtype=torch.float64, device=dev)
    dist.all_gather_into_tensor(out, inp)
    # The one host sync (the queue is already drained by the barrier).
    return out.view(w, len(vals)).tolist()


def _global_off(st: _State, i: int, reason: str) -> None:
    """OFF on every rank from boundary i (called with identical args everywhere)."""
    with st.cond:
        if st.global_off is None:
            st.global_off = reason
        st.off_from = max(0, i - 1)
        st.hooks_active = False
        st.cond.notify_all()
    _remove_hooks(st)
    st.acc.clear()


def on_window_boundary(device: Optional[torch.device] = None) -> None:
    """Window boundary i (see BOUNDARIES in the module docstring).

    Called by utils._traced_window_boundary right after the barrier, once per
    train window of the process, on the main thread. Returns at once when M4
    is not requested or was OFF at arming.
    """
    st = _S
    if st.inert:
        return
    if st.armed is None and not requested():
        st.inert = True  # OFF: nothing installed, nothing gathered, ever
        st.armed = False
        return
    i = st.boundary
    st.boundary = i + 1  # the ONLY place the boundary index advances
    if i == 0:
        _arm(st, device)  # runs callbacks(0) iff armed
        return
    if st.armed and st.global_off is None:
        _close_window(st, i, device)
    elif st.remove_pending or st.live:
        _remove_hooks(st)
    if st.armed:
        # Every barrier once armed, also after a global OFF: comm's owner-hash
        # check must keep running while earlier-dealt windows can start.
        _run_callbacks(st, i)


def _arm(st: _State, device: Optional[torch.device]) -> None:
    """Boundary i = 0: the arming gather; ON or OFF on every rank."""
    w = _world()
    if st.local_off is not None:
        why = st.local_off
    elif not st.installed:
        why = "no hook point (install() not called or it declined)"
    else:
        why = ""
    ostat, ovec, oreason = parse_offsets(os.environ.get(ENV_OFFSETS, ""), w)
    ohash = _hash48(np.ascontiguousarray(ovec, dtype="<f8").tobytes()) if ovec is not None else 0.0
    row = [_config_hash(st), ohash, float(ostat), float(w), FLAG_OFF if why else FLAG_OK]
    rows = _all_gather_rows(row, device)
    if not rows or any(len(r) != _AROW for r in rows):
        raise RuntimeError(f"m4_speed: malformed arming rows {rows!r}")
    bad = [r for r, x in enumerate(rows) if x[4] != FLAG_OK]
    if bad or len({x[0] for x in rows}) > 1:
        if bad:
            reason = f"not armed on rank(s) {bad}" + (f" (this rank: {why})" if why else "")
        else:
            reason = f"diagnostic M4 knobs differ across ranks (config hashes {[int(x[0]) for x in rows]})"
        st.armed = False
        _global_off(st, 0, reason)
        _print_all_ranks(f"[m4] speed estimator OFF: {reason}")
        return
    st.armed = True
    stats = [int(x[2]) for x in rows]
    if all(s == 0 for s in stats):
        st.offsets, st.offsets_desc = None, "none"
    elif all(s == 1 for s in stats) and len({x[1] for x in rows}) == 1 and ovec is not None:
        st.offsets, st.offsets_desc = ovec, _vec_sha(ovec)
    else:
        st.offsets, st.offsets_desc = None, "refused"
        _print_all_ranks(
            f"[m4] offsets REFUSED ({ENV_OFFSETS}): status by rank {stats} "
            "(1 valid, 0 unset, -1 invalid)"
            + ("; values differ across ranks" if all(s == 1 for s in stats) else "")
            + f"; this rank: {oreason or 'valid'}; factors are published without offsets"
        )
    _reset_window(st)  # drop pre-window records (warm-ups, the pipeline's fx trace)
    st.cnt_snap = dict(st.counters)
    _print_all_ranks(
        "armed OK: [m4] speed estimator ON span="
        + ("P_mf" if st.span == "mf" else "P_stu (DIAGNOSTIC)")
        + f" (API v{API_VERSION}): {st.hooked}; E_end "
        + ("when all item-candidate embedding grads are final" if st.span == "mf"
           else "at the grad of the layer-0 input")
        + f"; world={w} min_steps={st.min_steps} ema_alpha={st.alpha:g} "
        f"clamp=[{CLAMP_LO:g},{CLAMP_HI:g}] clip={CLIP_FRAC:g} stall={STALL_FRAC:g} "
        f"freeze={st.freeze_after or 'off'} offsets={st.offsets_desc}; factors from "
        f"boundary k>=1; kill switch {ENV_ON}=0"
    )
    _run_callbacks(st, 0)


def _close_window(st: _State, i: int, device: Optional[torch.device]) -> None:
    """Boundary i >= 1: close window k = i - 1, gather, decide, publish."""
    k = i - 1
    if st.remove_pending:
        _remove_hooks(st)
    try:
        row = _local_row(st, k)
    except Exception as e:  # noqa: BLE001 -- must still join the gather
        row = [float("nan"), 0.0, 0.0, 0.0, FLAG_ERR]
        _warn_once(st, "row_error", "[m4] speed estimator: window k=%d measurement "
                   "failed on this rank (%s: %s); M4 goes off on every rank", k,
                   type(e).__name__, e)
    if st.local_off is not None:
        row[4] = FLAG_OFF
    row = row + _log_fields(st)
    _reset_window(st)
    rows = _all_gather_rows(row, device)
    if not rows or any(len(r) != _BROW for r in rows):
        raise RuntimeError(f"m4_speed: malformed gathered rows {rows!r}")
    st.last_rows = rows
    off, valid, ema, pub = decide(k, [r[:_ROW] for r in rows], st.ema, st.alpha, st.min_steps)
    if off:
        dis = [r for r, x in enumerate(rows) if x[4] == FLAG_OFF]
        err = [r for r, x in enumerate(rows) if x[4] not in (FLAG_OK, FLAG_OFF)]
        reason = "; ".join(
            x for x in (
                f"disabled on rank(s) {dis}" if dis else "",
                f"measurement error on rank(s) {err}" if err else "",
            ) if x
        )
        reason = f"{reason} before boundary k={k}"
        _global_off(st, i, reason)
        _print_all_ranks(f"[m4] speed estimator OFF from window i={i}: {reason}")
        return
    st.ema = ema
    if valid:
        st.n_valid += 1
        st.ref = rows[_rank()][0] if len(rows) > 1 else rows[0][0]
        st.acc.clear()
        if st.frozen is None and st.freeze_after > 0 and st.n_valid >= st.freeze_after:
            st.frozen = pub
    if st.frozen is not None:
        pub = st.frozen
    bad = _sanitize(pub, len(rows))
    if bad:
        st.counters["unpublishable"] += 1
        if _rank() == 0:
            logger.warning("[m4] WARNING: factors k=%d not publishable (%s): %r -> None", k, bad, pub)
        pub = None
    vec = None if pub is None else np.array(pub, dtype=np.float64)
    offs = None if (vec is None or st.offsets is None) else st.offsets.copy()
    with st.cond:
        st.published[k] = vec
        st.offsets_pub[k] = offs
        st.last_k = k
        st.cond.notify_all()
    if _rank() == 0:
        _log_boundary(st, k, valid, vec, offs, rows)


def _fmt(vals: Sequence[float], fmt: str = "%.4f") -> str:
    v = list(vals)
    txt = ",".join("nan" if not math.isfinite(x) else fmt % x for x in v[:LOG_MAX])
    return "[" + txt + (",..." if len(v) > LOG_MAX else "") + "]"


def _log_boundary(
    st: _State,
    k: int,
    valid: bool,
    vec: Optional[np.ndarray],
    offs: Optional[np.ndarray],
    rows: List[List[float]],
) -> None:
    n = [int(r[1]) for r in rows]
    meas = [int(r[2]) for r in rows]
    nc = [int(r[3]) for r in rows]
    cnt = ""
    for j, c in enumerate(LOG_COUNTERS):
        per = [int(r[_ROW + 1 + j]) for r in rows]
        if any(per):
            cnt += " %s=%s" % (c, per[:LOG_MAX])
    extra = ""
    if st.log_all:
        extra = " n_r=%s meas=%s costs=%s" % (n[:LOG_MAX], meas[:LOG_MAX], nc[:LOG_MAX])
    logger.warning(
        "[m4] factors k=%d valid=%d n=%d..%d min=%s max=%s vec=%s offsets=%s "
        "nvalid=%d frozen=%d stat=%s maxmed=%s pub=%s%s%s",
        k,
        int(valid),
        min(n),
        max(n),
        "-" if vec is None else "%.4f" % float(vec.min()),
        "-" if vec is None else "%.4f" % float(vec.max()),
        _vec_sha(vec),
        "none" if offs is None else _vec_sha(offs),
        st.n_valid,
        int(st.frozen is not None),
        _fmt([r[0] for r in rows], "%.5f"),
        _fmt([r[_ROW] for r in rows], "%.3f"),
        "-" if vec is None else _fmt(vec.tolist(), "%.5f"),
        cnt,
        extra,
    )
    if k >= 1 and max(meas) > 0 and max(nc) == 0:
        if st.noted_k:
            # comm notes EMPTY costs for a round-robin window it cannot price (beyond
            # the N table, nb = 0, a resumed partial window): no sample, by contract
            if "empty_costs" not in st.warned:
                st.warned.add("empty_costs")
                logger.info("[m4] window k=%d: empty costs noted (no sample; expected "
                            "beyond the N table / resumed partial window)", k)
        else:
            _warn_once(st, "no_costs", "[m4] WARNING: window k=%d trained but "
                       "note_window_costs was never called for it (a path without an "
                       "M4 dealer must call m4_speed.set_enabled(False, reason))", k)
    if min(nc) == 0 < max(nc):
        _warn_once(st, "costs_partial", "[m4] WARNING: window k=%d: costs noted on some "
                   "ranks only %s (the dealer path should be rank-identical)", k, nc)
    if k >= 1 and max(nc) > 0 and max(meas) == 0:
        _warn_once(st, "no_steps", "[m4] WARNING: window k=%d has costs but no measured "
                   "step on any rank (span end hooks not reached?)", k)
    if any(m != c for m, c in zip(meas, nc) if m > 0 and c > 0):
        _warn_once(st, "mismatch", "[m4] WARNING: window k=%d measured steps %s != cost "
                   "entries %s on some rank (j<->k mapping, an extra training forward, "
                   "or lost samples)", k, meas[:LOG_MAX], nc[:LOG_MAX])


def _run_callbacks(st: _State, i: int) -> None:
    with st.cond:
        cbs = list(st.callbacks)
    for fn in cbs:
        fn(i)


def status() -> Dict[str, Any]:
    """Snapshot for logs and tests."""
    st = _S
    with st.cond:
        return {
            "requested": requested(),
            "enabled": enabled(),
            "armed": st.armed,
            "inert": st.inert,
            "local_off": st.local_off,
            "global_off": st.global_off,
            "off_from": st.off_from,
            "stopped": st.stopped,
            "installed": st.installed,
            "hooked": st.hooked,
            "span": st.span,
            "n_module_hooks": len(st.module_handles),
            "hooks_active": st.hooks_active,
            "boundary": st.boundary,
            "last_k": st.last_k,
            "n_valid": st.n_valid,
            "frozen": None if st.frozen is None else list(st.frozen),
            "published": {
                k: (None if v is None else v.copy()) for k, v in st.published.items()
            },
            "offsets_published": {
                k: (None if v is None else v.copy()) for k, v in st.offsets_pub.items()
            },
            "offsets": None if st.offsets is None else st.offsets.copy(),
            "offsets_desc": st.offsets_desc,
            "ema": None if st.ema is None else list(st.ema),
            "ref": st.ref,
            "acc": len(st.acc),
            "counters": dict(st.counters),
            "live": len(st.live),
            "live_handles": sum(len(r.handles) for r in st.live),
            "done": len(st.done),
            "costs_pending": sorted(st.costs),
            "callbacks": len(st.callbacks),
            "params": {
                "alpha": st.alpha,
                "min_steps": st.min_steps,
                "freeze_after": st.freeze_after,
                "log_all": st.log_all,
            },
        }


def _reset_for_tests() -> None:
    """Fresh module state (tests only; production never calls it)."""
    global _S
    old = _S
    _remove_hooks(old)
    with old.cond:
        old.stopped = old.stopped or "reset"
        old.cond.notify_all()
    _S = _State()
