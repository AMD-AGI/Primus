"""M4 (gin item (82)): speed-aware within-batch anchor dealing, N built on the fly (M4-OTF; comm; default OFF).

WHAT
  The streaming sampler deals every global batch round-robin: global block b of
  ``bs * W`` consecutive anchors of a window (dataset order), position p -> rank
  ``p % W`` (``_ChainedWindowBatchSampler.push``). Per-GPU speed differs under the
  power cap and persists within an allocation, so the step waits for the slowest
  GPU every step. M4 keeps EVERY global batch exactly as it is (same anchors, same
  window order, same per-window batch count, same dropped tail) and only changes
  WHICH rank receives which ``bs`` anchors of each block: the anchors are dealt so
  that predicted per-rank time ``g_r * sum(cost) (+ F_r)`` is balanced, where ``g_r``
  is the rank's measured time per unit predicted cost and ``F_r`` an optional fixed
  per-rank offset (``m4_speed``, x4c19). Each rank still gets exactly ``bs`` anchors
  per step, kept in dataset order inside the rank.

  cost_i = b*N_i + c*N_i*(N_i+1)/2 ms, N_i = the anchor's sequence length, b/c = the 1n
  synthetic-shape law (x4c19 job 71809; m4_speed's COST_B_MS/COST_C_MS when exported, so allocation
  and speed normalisation share one law). N is computed from the pool flags, which include
  the label-bearing listen flag; its classification as training_rules.adoc L258 metadata is
  PENDING a ruling (L268/L270: SWG pre-clearance needed before any ON submission).

M4-OTF: N IS BUILT ON THE FLY, AFTER RUN_START (user ruling 2026-10-12; replaces the withdrawn
  offline N table of ship15 34d35d6)
  Nothing derived from training data is read, verified or known before RUN_START: no table, no
  checksum, no file. maybe_create (loader construction, before RUN_START under DL_PRESTART) only
  checks the configuration, keeps a reference to the dataset object and starts the per-rank
  partition helper (a fresh interpreter that never receives a path). activate() (stream start,
  main thread) is the RUN-START GATE: without dl_prestart.run_start_wall() (a resume, or no MLPerf
  logger: train_ranker logs RUN_START and calls set_run_start only on a cold start) it reads
  nothing and turns M4 off through m4_speed.set_enabled(False, ..), which travels in m4_speed's
  boundary-0 arming gather, so EVERY rank deals round-robin. With the gate open it reads the
  window index (dataset._ensure_streaming_index; the mlcommons/training PR 907 offline index,
  DL_WINDOW_INDEX=1, which the stream reads anyway and which the user ruled allowed on
  2026-10-12) and starts m4_otf: one builder process per node (elected without data or
  collectives) computes each window's N after an explicit release and a 'go' (default: at window
  boundary 1), in increasing window order with deadline-driven jumps, into a sealed anonymous
  memfd that reaches the ranks of the node over a unix socket (m4_otf docstring: protocol,
  knobs). The helper receives a window's N as a file descriptor (SCM_RIGHTS), maps it read-only
  and partitions on the window buffer. In-run working memory only: created after RUN_START,
  anonymous, never on disk, never reused across runs, freed at teardown, used only to deal
  anchors (never a model input).

PARTITION (per global block; integer-exact, identical on every rank; N-PRESERVING)
  Each rank's attention normalizer is max_seq_len = the max N of its LOCAL batch, and
  every HSTU layer scales its SiLU scores by 1/N, so moving an anchor to a rank with a
  different batch max would change its computed function. Ranks are therefore grouped
  by their round-robin batch max N; an anchor only moves inside its group, and every
  rank first keeps its own first round-robin anchor that attains that max. Every
  rank's max N and every anchor's normalizer stay the round-robin values.
  Inside a group the other anchors are sorted by N descending (stable: ties by
  position) and cut into W_g strata of bs-1; round k gives the k-th anchor of every
  stratum to the group's ranks: the costliest to the rank with the largest remaining
  deficit (target_r - load_r, target_r = group total * w_r / sum(w), w_r =
  round(2^24 / g_r), g normalised to mean 1); ties to the lower rank. Every rank gets
  exactly bs anchors. Costs are int64 nanoseconds, so no floating-point order can make
  two ranks disagree.

HOST-LOCAL DEALING (M4 v3 = comm partition v2 + x4c19 iter62 M4-HL; DLRM_HSTU_FP8_M4_HOST_LOCAL)
  With host-local dealing the groups are keyed by (block, host, round-robin batch max N)
  instead of (block, max N), host(r) = r // (world_size / NNODES) (train_ranker:
  rank = node_rank * gpus_per_node + local_rank, WORLD_SIZE = NNODES * GPUS_PER_NODE).
  An anchor then only moves between ranks of ONE host, so every host's per-step anchor
  set equals its round-robin set. The streaming page walk (cold_prefetch,
  DL_COLD_PREFETCH_STREAM) advises every round-robin anchor's byte ranges from SOME
  local rank, into the node's shared page cache, at the same batch index and lead, so it
  stays exact for the host: M4 no longer declines the walk. One node: host-local ==
  global (identical owner arrays), so the walk never needs host-local dealing there.
  With hosts=None (or all ranks on one host) every key, hence every result (owner,
  load), is byte-identical to the global dealing above; the owner hash is byte-identical
  for hosts=None (maybe_create never passes a map on one node) and covers the map otherwise. Host-local dealing can only
  balance inside a host: it keeps ~0.89 of the global gain at 2n on the M4 exam
  allocation (x4c19 iter62 replay; 0.63-0.91 over ship13 2n speed vectors, 0.53-0.80
  at 4n), which is the price of an exact walk.

NUMERICS (not bitwise vs round-robin; same global batches and objective in exact arithmetic)
  Unchanged by construction: every anchor's 1/N normalizer and every rank's max N (hence
  the N-dependent DSC_K ds scale), the per-rank batch size (KJT stride) and the
  DDP-averaged per-rank-mean loss. Changed (which anchors share a rank batch): fp32
  summation order of the per-rank loss and of gradients before the DDP all-reduce; which
  per-rank dropout draw meets which anchor; per-rank backward/preproc FP8 amax histories
  (per-rank _SiteScale; forward scales are MAX-all-reduced); per-rank fp16 partial sums of
  the sparse all-to-all. Needs an end-to-end convergence pair before shipping. A window dealt
  round-robin (not ready, or any fallback below) is byte-identical to the stock slice.

DETERMINISM PROTOCOL (the partition of window j must be identical on all ranks)
  ``push`` runs on the single-pool pusher thread at rank-dependent times, so the
  factor vector is VERSIONED: window j (run-local index into train_ts_list) uses
  ``m4_speed.factors_for(k(j))``, the vector published at the window-boundary
  barrier that starts window k(j)+1. k(j) is computed from the batch counts of the
  windows already pushed (identical on every rank) so that, by the pusher rule
  ("stay >= 2 windows and >= 2 x depth batches ahead") and the loader depth, the
  trainer has necessarily passed that barrier before any rank can push window j;
  ``factors_for`` still blocks until the vector is published and RAISES on timeout
  (never a silent local fallback). k(j) <= 0 -> round-robin.
  READINESS (M4-OTF): at the same barrier k(j)+1 every rank all-gathers a bitmap of the
  run-local windows [k(j)+1, k(j)+129) whose N it holds; the AND over ranks is recorded there
  (the m4_speed boundary callback, main thread). Window j uses M4 iff its bit is set in the AND
  of boundary k(j)+1; the pusher waits (bounded by the factor timeout, then RAISES) until that
  boundary's AND is recorded. A late N is ignored for M4 (only its costs are still used).
  The elected rank of each node also sends its builder a schedule hint (need[j] = k(j)+1 from
  lag_k over the pushed and predicted batch counts, m4_otf step 3) so the builder has window j
  before boundary k(j)+1; it never enters a decision. Once m4_speed is off for good on a rank
  (enabled() False), no rank computes round-robin costs any more and the elected rank stops its
  builder (the all_gather keeps running at every boundary).
  Every window's owner hash is all-gathered at the barrier that starts it (the same one
  all_gather, int64 [owner hash, readiness words], registered when the train stream starts;
  relies on m4_speed calling it on every rank at every barrier, i = 0 included and
  collective-free) and any cross-rank mismatch raises BEFORE the window trains.

FALLBACKS (round-robin, byte-identical to the stock path; identical on all ranks; counted
  per reason in the end-of-stream summary)
  k(j) <= 0 (no factors yet: the first ~236-241 steps); factors_for returned None; an
  unusable factor vector (wrong length, non-finite, <= 0, spread > 4x: logged once per
  version); skip_samples > 0 (resumed first window); the window's indices are not the
  offline window slice (shuffle / holdout / in-window sort); otf-not-ready (some rank did not
  hold the window's N at boundary k(j)+1: builder late, failed, or never connected).
  COSTS for m4_speed: an M4 window's per-batch costs come from the helper; a round-robin
  window's from this rank's round-robin anchors, computed in-process when its N is held at
  push, otherwise at the first window boundary that has it (at the latest the boundary that
  starts the window; else no sample for that window); none at all once m4_speed is off for good.

SWITCHES AND DECLINES
  DLRM_HSTU_FP8_M4 (gin item, code row binds it OFF; env wins): on/off; parsed as strip() == "1".
  M4 ON with m4_speed missing, or a dataset without the streaming offline ts index
  (DL_WINDOW_INDEX=0), RAISES at loader construction (fail closed; configuration only).
  DLRM_HSTU_FP8_M4_HOST_LOCAL (gin-bound "0"; env wins; read once at loader
  construction, identical on every rank):
    "auto" = host-local dealing exactly when the streaming page walk runs AND the job
             spans more than one node, global dealing otherwise (walk off, or one node).
             The walk runs when the prefetcher is armed (cold_prefetch.mode() > 0:
             DL_COLD_PREFETCH=1|2, or DL_PRESTART=2, the level prestart_streaming_loaders sets
             before it builds the train loader) AND cold_prefetch.resolve_stream() turns the
             stream on (DL_COLD_PREFETCH_STREAM 1/true/on/yes, or auto on more than one node):
             the same two gates _stream_single / cold_prefetch.start apply;
    "1"    = host-local dealing whenever the job spans more than one node;
    "0"    = global dealing always. With the walk on across hosts this is the
             ship15 ON form (it was the lab_v3 arm "global with the walk"): the walk still advises
             each rank's round-robin ranges on that rank's host while global dealing moves
             ~37% (2n) / ~53% (4n) of a step's anchors to another host, so some moved
             anchors read pages their host's walk did not advise (x4c19 iter62 model:
             0.3-2.8% of the walk's volume at 2n). The walk's coverage is a performance
             matter only (the bytes read are identical); logged once per rank ("walk
             coverage approximate", which must be ABSENT in a production ON arm).
  NUMERICS of the mode: host-local and global dealing are both M4 (not bitwise vs
  round-robin, N-preserving) and are NOT bitwise to each other (different owner arrays).
  Kill switches: DLRM_HSTU_FP8_M4=0 (M4 off); DLRM_HSTU_FP8_M4_HOST_LOCAL=0 forces global.
  The one-shot DL_COLD_PREFETCH_BATCHES prefetch is correct in every mode: it reads only
  batches inside the round-robin prefix (M4 deals its first window after ~240 steps).
  Declines (identically on every rank, logged "DECLINED"): DL_SINGLE_POOL=0
  (double-buffered set_window path); world_size <= 1; DLRM_HSTU_FP8_M4_HOST_LOCAL not
  auto/1/0; host-local dealing asked for while NNODES does not divide world_size, or a
  positive $GPUS_PER_NODE disagrees with world_size / NNODES (host map unknown); and,
  possibly on some ranks only, host-local dealing on a rank whose $NODE_RANK is not its host
  in the node-major map (rank layout not node-major: m4_speed's arming gather at window
  boundary 0 then turns M4 off on every rank);
  m4_speed.enabled() False at construction; a dataset configuration other than the one the
  on-the-fly N rule assumes (m4_otf.nrule_config_mismatch, review C5: $HISTORY_STRATEGY
  interleaved, $HISTORY_LENGTH 4086, scan_window 20000, $MAX_SEQ_LEN 4096, 1 candidate, 8
  length-1 contextual features; dataset attributes only, before the helper starts);
  train-loader prefetch depth 0 (NUM_WORKERS=0: the lag rule has no bound). The run-start
  gate (logged "m4otf gate closed") also turns M4 off
  on every rank through the arming gather. (Partition v2 declined DL_COLD_PREFETCH_STREAM=1 at every
  node count, and missed "auto" on more than one node; v3 replaces both with the modes
  above.)
  Diagnostic env (lab arms; identical on all ranks): DLRM_HSTU_FP8_M4_LOG_ALL=1 (per-window
  line on every rank), DLRM_HSTU_FP8_M4_FACTOR_TIMEOUT_S (factors_for and readiness wait,
  default 600), DLRM_HSTU_FP8_M4_HELPER_TIMEOUT_S (one helper call, default 120),
  DLRM_HSTU_FP8_M4_LAG_MARGIN (batches, default 32); m4_otf's DLRM_HSTU_FP8_M4_OTF_GO_BATCHES,
  _OTF_J0, _OTF_LEAD (builder pacing; node-local, need not be identical).

LOG LINES
  "DLRMV4 m4 dealer created rank=../W (N on the fly after RUN_START (m4otf), law b=.. c=.. ms from
  <src>, helper pid .., ready ..s; dealing=global|host-local(hosts=..) (DLRM_HSTU_FP8_M4_HOST_LOCAL=..,
  walk=on|off[ (why)], nnodes=..); armed when the train stream starts)" at loader construction;
  with DLRM_HSTU_FP8_M4_HOST_LOCAL=0 and the walk on across hosts also "DLRMV4 m4 dealing=global
  with the streaming page walk on across .. hosts (DLRM_HSTU_FP8_M4_HOST_LOCAL=0; the ship15 form;
  the runtime text still reads "examination setting"): ... walk coverage approximate (rank ..)";
  at stream start (m4_otf): "DLRMV4 m4otf builder elected rank=.. (local clients expected ..)" and
  "DLRMV4 armed OK: m4otf release after run_start (+..s) rank=.. builder pid .." (elected rank),
  "DLRMV4 m4otf rank=..: client of the node builder .." (the other local ranks), then
  "DLRMV4 armed OK: m4 speed-aware dealing rank=.. depth=.. lag_margin=.. law=..
  dealing=global|host-local(hosts=..) N=on-the-fly (m4otf builder=elected|client connected=..)";
  builder stderr (m4_otf docstring, every line with t=<epoch ms>): "builder started", "release
  received", "go", "data mapped", "first memfd_create", per window "[m4otf] node=.. j=.. ts=..
  anchors=.. build_ms=.. progress=.. need=.. pick=.. t=..", "builder exit";
  "DLRMV4 armed OK: m4 engaged rank=.. at j=.." at the first M4-dealt window;
  "[m4] factors rank=0 k=.. vec=.. wq=[..] g=[..]" once per new factor vector;
  "[m4] rank=.. j=.. ts=.. k=.. mode=m4|rr[ reason=..] nb=.. vec=.. owner=.. ms=.. otf=.. costs=.."
  per window (rank 0; every rank with DLRM_HSTU_FP8_M4_LOG_ALL=1);
  "DLRMV4 armed OK: m4 owner-hash check at window boundary j=.." once, at the first barrier;
  "DLRMV4 m4 summary rank=..: windows m4=.. rr=..; rr reasons {..}" and "DLRMV4 m4otf summary
  rank=..: windows built=.. ready_at_decision=.. not_ready=.. deferred_costs=.." at stream end;
  after an m4_speed OFF, on the elected rank: "DLRMV4 m4otf builder stopped rank=..".
  Must be ABSENT in an ON arm: "MISMATCH", "DECLINED", "m4 helper", "unusable", "m4otf builder
  failed", "m4otf gate closed", "refused a client", "not a local rank", and (except with
  DLRM_HSTU_FP8_M4_HOST_LOCAL=0 on more than one node, the ship15 form) "walk coverage
  approximate". Expected dealing with HOST_LOCAL="0" (gin):
  dealing=global at every node count; with HOST_LOCAL="auto": global at one node,
  host-local(hosts=NNODES) on more nodes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from generative_recommenders.dlrm_v4.train import m4_otf

logger = logging.getLogger(__name__)

ENV_ON = "DLRM_HSTU_FP8_M4"
ENV_TIMEOUT = "DLRM_HSTU_FP8_M4_FACTOR_TIMEOUT_S"  # diagnostic
ENV_LOG_ALL = "DLRM_HSTU_FP8_M4_LOG_ALL"  # diagnostic
ENV_LAG_MARGIN = "DLRM_HSTU_FP8_M4_LAG_MARGIN"  # diagnostic (lab arms; must be identical on all ranks)
ENV_HELPER_TIMEOUT = "DLRM_HSTU_FP8_M4_HELPER_TIMEOUT_S"  # diagnostic
ENV_HOST_LOCAL = "DLRM_HSTU_FP8_M4_HOST_LOCAL"  # gin-bound: auto | 1 | 0

# Cost law from synthetic-shape GPU timing only (x4c19 job 71809, D3-D5): ms per token, ms per causal pair.
COST_B_MS = 38.157e-6
COST_C_MS = 13.509e-9
N_LIMIT = 4096  # N are int16 lengths <= 4095 (HISTORY_LENGTH 4086 + contextual + candidate)
COST_QUANTUM_MS = 1e-6  # 1 ns
WEIGHT_QUANTUM = 1 << 24
LAG_MARGIN_BATCHES = 32
# The trainer pipeline (TrainPipelineSparseDist) pulls batch N+2 while training N (L = 2), so
# the margin must be >= L for the pusher never to wait for an unpublished version; clamp.
MIN_LAG_MARGIN = 3
# Largest usable max(g)/min(g) (measured GPU spread < 1.2x): keeps w <= 2^24 * 4, so the
# integer target products (group total * w) stay < 2^62 for any block (asserted).
MAX_FACTOR_SPREAD = 4.0
DEFAULT_FACTOR_TIMEOUT_S = 600.0
DEFAULT_HELPER_TIMEOUT_S = 120.0  # per partition call (healthy: 0.3-3.4 s per window at 4n)
HELPER_START_TIMEOUT_S = 300.0  # helper interpreter start (ping at loader construction; no data)
_HASH_MASK = (1 << 63) - 1


def switch_on() -> bool:
    return os.environ.get(ENV_ON, "0").strip() == "1"


def cost_lut(b_ms: float = COST_B_MS, c_ms: float = COST_C_MS, n_limit: int = N_LIMIT) -> np.ndarray:
    """int64 cost of an anchor of length n (index) in COST_QUANTUM_MS units; strictly increasing."""
    n = np.arange(n_limit, dtype=np.float64)
    lut = np.rint((b_ms * n + c_ms * n * (n + 1.0) * 0.5) / COST_QUANTUM_MS).astype(np.int64)
    if np.any(np.diff(lut[1:]) <= 0):
        raise ValueError("m4: cost LUT is not strictly increasing for n >= 1")
    return lut


def factors_valid(g, world_size: int) -> str:
    """"" or why a published factor vector cannot be used (then the window is dealt
    round-robin; the vector is identical on every rank, so the fallback is too)."""
    try:
        a = np.asarray(g, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return "not-numeric"
    if a.shape[0] != world_size:
        return f"len {a.shape[0]} != world_size {world_size}"
    if not np.all(np.isfinite(a)):
        return "non-finite"
    if np.any(a <= 0.0):
        return "non-positive"
    if float(a.max()) > MAX_FACTOR_SPREAD * float(a.min()):
        return f"spread > {MAX_FACTOR_SPREAD:g}x"
    return ""


def weights_from_factors(g: np.ndarray, world_size: int) -> np.ndarray:
    """int64 target weights w_r = round(2^24 / g_r); g = time per unit cost (mean ~1)."""
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    if g.shape[0] != world_size or not np.all(np.isfinite(g)) or np.any(g <= 0.0):
        raise ValueError(f"m4: invalid speed factors {g!r} for world_size={world_size}")
    g = g / g.mean()  # only relative speed matters; also bounds w (no int64 overflow in targets)
    w = np.rint(WEIGHT_QUANTUM / g).astype(np.int64)
    if np.any(w <= 0) or int(w.max()) >= (1 << 31):
        raise ValueError(f"m4: weights out of range from factors {g!r}")
    return w


def _check_headroom(total_max: int, w_max: int) -> None:
    if int(total_max) * int(w_max) >= (1 << 62):
        raise ValueError(f"m4: target product {total_max} * {w_max} overflows the int64 headroom")


def partition_owner(n_blocks: np.ndarray, lut: np.ndarray, weights_q: np.ndarray, bs: int) -> np.ndarray:
    """Owner rank (int16) of every anchor of ``n_blocks`` [nb, bs*W] (lengths N).

    Exactly ``bs`` anchors per rank in every block; deterministic integer arithmetic.
    """
    return partition_owner_load(n_blocks, lut, weights_q, bs)[0]


def partition_owner_load(n_blocks: np.ndarray, lut: np.ndarray, weights_q: np.ndarray, bs: int):
    """(owner int16 [nb, G], load int64 [nb, W]): see partition_owner."""
    n_blocks = np.asarray(n_blocks)
    if n_blocks.ndim != 2:
        raise ValueError("m4: n_blocks must be 2-D [nb, bs*W]")
    nb, G = n_blocks.shape
    W = int(weights_q.shape[0])
    if G != bs * W:
        raise ValueError(f"m4: block width {G} != bs*W = {bs}*{W}")
    owner = np.empty((nb, G), dtype=np.int16)
    if nb == 0:
        return owner, np.zeros((0, W), dtype=np.int64)
    nmin, nmax = int(n_blocks.min()), int(n_blocks.max())
    if nmin < 0 or nmax >= lut.shape[0]:
        raise ValueError(f"m4: N out of range [{nmin}, {nmax}]")
    costs = lut[n_blocks]  # int64 [nb, G]
    # Descending N == descending cost (strictly increasing LUT); uint16 key -> numpy radix sort.
    key = (lut.shape[0] - 1) - n_blocks.astype(np.uint16)
    order = np.argsort(key, axis=1, kind="stable")
    sc = np.take_along_axis(costs, order, axis=1)
    total = costs.sum(axis=1)
    _check_headroom(int(total.max()), int(weights_q.max()))
    wsum = int(weights_q.sum())
    target = (total[:, None] * weights_q[None, :].astype(np.int64)) // wsum
    load = np.zeros((nb, W), dtype=np.int64)
    # Stratified rounds: the sorted block is cut into W strata of bs anchors
    # (stratum 0 = the bs costliest); round k takes the k-th anchor of every
    # stratum, so each round spans the block's whole cost range (enough room
    # to move a few percent of a rank's load) and the rounds' items shrink
    # with k (coarse moves first, fine trimming last).
    sc3 = sc.reshape(nb, W, bs)
    own3 = np.empty((nb, W, bs), dtype=np.int16)
    rows = np.arange(nb)[:, None]
    for k in range(bs):
        chunk = sc3[:, :, k]  # [nb, W], non-increasing along W
        # largest deficit (target - load) first; stable -> ties to the lower rank
        ranks = np.argsort(load - target, axis=1, kind="stable")
        own3[:, :, k] = ranks
        load[rows, ranks] += chunk
    np.put_along_axis(owner, order, own3.reshape(nb, G), axis=1)
    return owner, load


def offsets_from_ms(f, world_size: int) -> Optional[np.ndarray]:
    """int64 per-rank fixed time offsets (COST_QUANTUM_MS units) from ms; None -> None."""
    if f is None:
        return None
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    if f.shape[0] != world_size or not np.all(np.isfinite(f)):
        raise ValueError(f"m4: invalid offsets {f!r} for world_size={world_size}")
    return np.rint(np.maximum(f, 0.0) / COST_QUANTUM_MS).astype(np.int64)


def host_keys(hosts, world_size: int) -> Optional[np.ndarray]:
    """int64 [W] host index of every rank (None -> one host). Raises on a bad map."""
    if hosts is None:
        return None
    h = np.asarray(hosts, dtype=np.int64).reshape(-1)
    if h.shape[0] != world_size or np.any(h < 0) or np.any(h >= world_size):
        raise ValueError(f"m4: invalid host map {h!r} for world_size={world_size}")
    return h


def partition_owner_load_np(n_blocks: np.ndarray, lut: np.ndarray, weights_q: np.ndarray, bs: int,
                            offsets_q: Optional[np.ndarray] = None, hosts: Optional[np.ndarray] = None):
    """N-PRESERVING dealing: (owner int16 [nb, G], load int64 [nb, W]).

    ``hosts`` (optional int64 [W], M4-HL): host index of every rank. The groups are then
    keyed by (block, host, rr_max), so an anchor only moves between ranks of its own host;
    hosts=None (one host) gives exactly the global dealing.

    Every rank's attention normalizer is max_seq_len = the max N of its LOCAL batch
    (host_sync_free / hstu_flydsl_mha: max_uih_len + candidates, + contextual), and every
    HSTU layer scales its SiLU scores by 1/N. To keep the computed function of every anchor
    exactly the round-robin one, ranks are grouped by their round-robin batch max N
    (rr_max); an anchor may only move between ranks of its own group, and every rank first
    keeps its own first round-robin anchor that attains rr_max. Hence each rank's max N and
    each anchor's normalizer equal the round-robin values. Inside a group the remaining
    W_g*(bs-1) anchors are dealt by the stratified rounds of partition_owner_load
    (all (block, group) sub-problems at once, padded to W slots; integer-exact).

    ``offsets_q`` (optional, int64 [W] >= 0, cost quanta): per-rank fixed time outside the
    measured span (e.g. 1n table-wise owner work). Targets then equalise g_r*load_r + F_r
    inside each group: load_r = (T_g + sum_s F_s/g_s) / sum_s(1/g_s) / g_r - F_r / g_r.
    """
    n_blocks = np.asarray(n_blocks)
    if n_blocks.ndim != 2:
        raise ValueError("m4: n_blocks must be 2-D [nb, bs*W]")
    nb, G = n_blocks.shape
    W = int(weights_q.shape[0])
    if G != bs * W:
        raise ValueError(f"m4: block width {G} != bs*W = {bs}*{W}")
    owner = np.empty((nb, G), dtype=np.int16)
    if nb == 0:
        return owner, np.zeros((0, W), dtype=np.int64)
    nmin, nmax = int(n_blocks.min()), int(n_blocks.max())
    if nmin < 0 or nmax >= lut.shape[0]:
        raise ValueError(f"m4: N out of range [{nmin}, {nmax}]")
    wq = weights_q.astype(np.int64)
    costs = lut[n_blocks]  # int64 [nb, G]
    n3 = n_blocks.reshape(nb, bs, W)  # [b, m, r] = position m*W + r (round-robin rank r)
    rr_max = n3.max(axis=1).astype(np.int64)  # [nb, W]
    # pre-assigned anchor of every rank: its first round-robin anchor attaining rr_max
    m_star = np.argmax(n3 == rr_max[:, None, :].astype(n3.dtype), axis=1)  # [nb, W]
    r_idx = np.arange(W)
    pre_pos = m_star * W + r_idx[None, :]  # [nb, W] positions in the block
    # sub-problems = (block, host, rr_max value); one host -> (block, rr_max value)
    hk = host_keys(hosts, W)
    if hk is None:
        key = np.arange(nb, dtype=np.int64)[:, None] * 4096 + rr_max  # [nb, W]
    else:
        nh = int(hk.max()) + 1
        key = (np.arange(nb, dtype=np.int64)[:, None] * nh + hk[None, :]) * 4096 + rr_max  # [nb, W]
    ukey, sub_id = np.unique(key.reshape(-1), return_inverse=True)
    sub_id = sub_id.reshape(nb, W)
    S = ukey.shape[0]
    Wg = np.bincount(sub_id.reshape(-1), minlength=S)  # ranks per sub-problem
    # slot of every rank inside its sub-problem (ranks ascending)
    order_r = np.argsort(sub_id.reshape(-1), kind="stable")  # flat (b, r) sorted by sub-problem, then b*W+r
    seg_start = np.concatenate([[0], np.cumsum(Wg)[:-1]])
    slot_flat = np.empty(nb * W, dtype=np.int64)
    slot_flat[order_r] = np.arange(nb * W) - np.repeat(seg_start, Wg)
    slot = slot_flat.reshape(nb, W)
    rank_of_slot = np.full((S, W), -1, dtype=np.int64)
    rank_of_slot[sub_id.reshape(-1), slot_flat] = np.tile(r_idx, nb)
    # targets and initial loads (pre-assigned anchors)
    pre_cost = np.take_along_axis(costs, pre_pos, axis=1)  # [nb, W]
    # exact integer group totals (all round-robin anchors of the group's ranks)
    tot_int = np.zeros(S, dtype=np.int64)
    np.add.at(tot_int, sub_id.reshape(-1), costs.reshape(nb, bs, W).sum(axis=1).reshape(-1))
    _check_headroom(int(tot_int.max()), int(wq.max()))
    w_slot = np.zeros((S, W), dtype=np.int64)
    w_slot[sub_id.reshape(-1), slot_flat] = np.tile(wq, nb)
    wsum = w_slot.sum(axis=1)
    if offsets_q is None:
        target = (tot_int[:, None] * w_slot) // np.maximum(wsum, 1)[:, None]
    else:
        oq = np.asarray(offsets_q, dtype=np.int64).reshape(-1)
        if oq.shape[0] != W or np.any(oq < 0):
            raise ValueError("m4: offsets_q must be int64 [W] >= 0")
        f_slot = np.zeros((S, W), dtype=np.int64)
        f_slot[sub_id.reshape(-1), slot_flat] = np.tile(oq, nb)
        fw = (f_slot * w_slot) // WEIGHT_QUANTUM  # F_r / g_r in cost quanta
        a = tot_int + fw.sum(axis=1)
        _check_headroom(int(a.max()), int(wq.max()))
        target = (a[:, None] * w_slot) // np.maximum(wsum, 1)[:, None] - fw
    load = np.zeros((S, W), dtype=np.int64)
    load[sub_id.reshape(-1), slot_flat] = pre_cost.reshape(-1)
    # remaining anchors, grouped by sub-problem, sorted by N desc (ties: position)
    is_pre = np.zeros((nb, G), dtype=bool)
    np.put_along_axis(is_pre, pre_pos, True, axis=1)
    owner_rr_rank = np.tile(r_idx, bs)[None, :].repeat(nb, axis=0)  # [nb, G] round-robin rank of each position
    sub_of_pos = np.take_along_axis(sub_id, owner_rr_rank, axis=1)  # [nb, G]
    rem_b, rem_p = np.nonzero(~is_pre)  # row-major: block, then position
    rem_sub = sub_of_pos[rem_b, rem_p]
    rem_n = n_blocks[rem_b, rem_p].astype(np.int64)
    srt = np.lexsort((rem_p, -rem_n, rem_sub))  # by sub-problem, N desc, position
    rem_b, rem_p, rem_sub = rem_b[srt], rem_p[srt], rem_sub[srt]
    R = bs - 1
    owner[rem_b, rem_p] = -1  # every remaining position is written below
    if R > 0:
        cnt = Wg * R
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        q = np.arange(rem_sub.shape[0]) - np.repeat(start, cnt)  # local index inside the sub-problem
        m_of = q // R  # stratum (< W_g)
        k_of = q % R  # round
        item_cost = lut[n_blocks[rem_b, rem_p]]
        slot_won = np.zeros(rem_sub.shape[0], dtype=np.int64)  # W_g == 1: everything to slot 0
        # Sub-problems are solved per width class (next power of two >= W_g), padded with
        # dummy slots whose key (load - target = 2^61) sorts after every real slot; strata
        # >= W_g are empty, so the dummies receive nothing. The real slots' order per round
        # is the same at any padded width (stable argsort), hence so is the result.
        wclass = np.left_shift(1, np.ceil(np.log2(np.maximum(Wg, 1))).astype(np.int64))
        one = np.flatnonzero(wclass[rem_sub] == 1)  # single-rank groups keep their anchors
        np.add.at(load, (rem_sub[one], 0), item_cost[one])
        for wc in np.unique(wclass):
            wc = int(wc)
            if wc <= 1:
                continue
            subs = np.flatnonzero(wclass == wc)
            Sc = subs.shape[0]
            loc = np.full(S, -1, dtype=np.int64)
            loc[subs] = np.arange(Sc)
            sel = np.flatnonzero(loc[rem_sub] >= 0)
            ls, ms_, ks = loc[rem_sub[sel]], m_of[sel], k_of[sel]
            it = np.zeros((R, Sc, wc), dtype=np.int64)
            it[ks, ls, ms_] = item_cost[sel]
            ld = np.zeros((Sc, wc), dtype=np.int64)
            tg = np.full((Sc, wc), -(np.int64(1) << 61), dtype=np.int64)
            wmax = min(wc, W)
            ld[:, :wmax] = load[subs, :wmax]
            real = ~(rank_of_slot[subs, :wmax] < 0)
            tg[:, :wmax] = np.where(real, target[subs, :wmax], tg[:, :wmax])
            ld[:, :wmax] = np.where(real, ld[:, :wmax], 0)
            osl = np.empty((R, Sc, wc), dtype=np.int16)
            rows = np.arange(Sc)[:, None]
            for k in range(R):
                ordk = np.argsort(ld - tg, axis=1, kind="stable")  # largest deficit first; dummies last
                osl[k] = ordk
                ld[rows, ordk] += it[k]
            load[subs, :wmax] = np.where(real, ld[:, :wmax], load[subs, :wmax])
            slot_won[sel] = osl[ks, ls, ms_]
        owner[rem_b, rem_p] = rank_of_slot[rem_sub, slot_won]
    np.put_along_axis(owner, pre_pos, np.tile(r_idx, nb).reshape(nb, W).astype(np.int16), axis=1)
    # per-rank loads back in rank order
    load_r = np.zeros((nb, W), dtype=np.int64)
    load_r[np.repeat(np.arange(nb), W), np.tile(r_idx, nb)] = load[sub_id.reshape(-1), slot_flat]
    return owner, load_r


def lag_k(j: int, nbs: List[int], depth: int, margin: int = LAG_MARGIN_BATCHES) -> int:
    """Newest boundary k whose factors window j may use (-1: none).

    nbs[i] = batch count of run-window i (all i < j known when j is pushed).
    The pusher may push window j once the sampler has popped window s, leaving a deque
    (s+1..j-1) with < 2 windows or < 2*depth batches; the code takes the earliest such
    pop conservatively (s = j-2, moved down while batches(s..j-1) < 2*depth), which is
    never later than the true earliest pop (proved by the protocol review). The trainer
    is then at most `depth` (DataLoader prefetch) + L (= 2, pipeline pull-ahead) batches
    behind the sampler, i.e. at or after batch P[s] - depth - L. The vector of boundary
    k is published at the barrier that starts window k+1 (batch P[k+1]); require
    P[s] - P[k+1] >= depth + margin with margin >= MIN_LAG_MARGIN > L.
    """
    if j < 2:
        return -1
    P = [0]
    for i in range(j):
        P.append(P[-1] + int(nbs[i]))
    s = j - 2
    while s - 1 >= 0 and (P[j] - P[s]) < 2 * depth:
        s -= 1
    best = -1
    for k in range(0, s):
        if P[s] - P[k + 1] >= depth + margin:
            best = k
        else:
            break
    return best


def lag_k_all(nbs: List[int], depth: int, margin: int = LAG_MARGIN_BATCHES) -> List[int]:
    """[lag_k(j, nbs, depth, margin) for j in range(len(nbs) + 1)] in O(n log n) (batch counts >= 0, so
    the prefix sums are non-decreasing and both scans of lag_k are bisections). Used only for the
    M4-OTF builder's schedule hint; deal() keeps lag_k."""
    import bisect

    P = [0]
    for x in nbs:
        P.append(P[-1] + int(x))
    out = []
    for j in range(len(nbs) + 1):
        if j < 2:
            out.append(-1)
            continue
        # s = the largest s' in [1, j-2] with P[j] - P[s'] >= 2 * depth, else 0
        s = max(0, bisect.bisect_right(P, P[j] - 2 * depth, 0, j - 1) - 1)
        # the largest k in [0, s-1] with P[k+1] <= P[s] - depth - margin, else -1
        out.append(bisect.bisect_right(P, P[s] - depth - margin, 1, s + 1) - 2)
    return out


def predicted_nb(size: int, W: int, bs: int, nb_cap=None) -> int:
    """deal()'s batch count of a window of ``size`` anchors with skip_samples = 0 (a cold start)."""
    if size < 0 or W <= 0 or bs <= 0:
        return 0
    nb = (int(size) // W) // bs
    return min(nb, max(0, int(nb_cap))) if nb_cap is not None else nb


def otf_nbs(nbs: List[int], sizes: List[int], W: int, bs: int, nb_cap=None) -> List[int]:
    """Batch count of every run-local window: pushed ones as dealt, the rest predicted (``predicted_nb``)."""
    full = [int(x) for x in nbs[:len(sizes)]]
    return full + [predicted_nb(sizes[j], W, bs, nb_cap) for j in range(len(full), len(sizes))]


def otf_need(full_nbs: List[int], depth: int, margin: int = LAG_MARGIN_BATCHES) -> List[int]:
    """M4-OTF schedule hint (m4_otf step 3): need[j] = k(j) + 1, the window boundary whose readiness AND
    decides window j, or j if window j cannot use M4 (k(j) < 1); k = lag_k over ``full_nbs``."""
    ks = lag_k_all(full_nbs[:-1], depth, margin) if full_nbs else []
    return [k + 1 if k >= 1 else j for j, k in enumerate(ks)]


def _h63(*parts: bytes) -> int:
    h = hashlib.sha1()
    for p in parts:
        h.update(p)
    return int.from_bytes(h.digest()[:8], "little") & _HASH_MASK


# -- compute kernels (shared by the in-process backend, the dealer and the helper process) -----------
# n_win = one window's N (int16, the window's anchors in window order): offset 0 is the window's first
# anchor (the withdrawn table form read the same values at off[ts]).
def compute_rr_costs(n_win, lut, b0, nb, W, bs, rank) -> np.ndarray:
    """ms cost per batch of this rank's round-robin anchors (blocks b0 .. b0+nb-1 of the window)."""
    G = W * bs
    lo = b0 * G
    blocks = np.asarray(n_win[lo:lo + nb * G]).reshape(nb, G)
    return lut[blocks[:, rank::W]].sum(axis=1) * COST_QUANTUM_MS


def compute_m4(n_win, lut, j, ts, n_anchors, nb, W, bs, wq, rank, oq=None, hosts=None):
    """(positions of this rank's anchors in [0, nb*G) block-major ascending, ms cost per batch, hash)."""
    G = W * bs
    blocks = np.asarray(n_win[0:nb * G]).reshape(nb, G)
    owner, load = partition_owner_load_np(blocks, lut, wq, bs, oq, hosts)
    pos = np.flatnonzero((owner == rank).reshape(-1)).astype(np.int64)
    if pos.shape[0] != nb * bs:
        raise RuntimeError(f"m4: rank {rank} got {pos.shape[0]} anchors, want {nb * bs}")
    # hosts=None keeps the global hash byte-identical; a host map is part of the hashed input
    h = _h63(b"m4", np.int64([j, ts, n_anchors, nb]).tobytes(), wq.tobytes(),
             (b"" if oq is None else np.asarray(oq, np.int64).tobytes()),
             (b"" if hosts is None else b"hl" + np.asarray(hosts, np.int64).tobytes()), owner.tobytes())
    return pos, load[:, rank] * COST_QUANTUM_MS, h


def window_array(fd: int) -> np.ndarray:
    """Read-only int16 view of a window's N (a sealed memfd from m4_otf); the mapping goes with the
    last view. The fd may be closed right after this call."""
    import mmap

    size = os.fstat(fd).st_size
    if size % 2:
        raise ValueError(f"m4: window N buffer of {size} bytes is not int16")
    if size == 0:
        return np.zeros(0, dtype=np.int16)
    # MAP_PRIVATE: some kernels (Ubuntu 6.8.0-60/-71) refuse a read-only MAP_SHARED mapping of a
    # write-sealed memfd with EPERM. The seal rules out writes, so both map the same pages.
    return np.frombuffer(mmap.mmap(fd, size, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ), dtype=np.int16)


class LocalBackend:
    """In-process compute (unit tests). Production uses HelperBackend: the stratified-round
    loop is ~bs small numpy calls that hold the GIL; in the trainer process they would
    steal the GIL from the main (kernel-launch) thread."""

    def __init__(self, lut: np.ndarray, W: int, rank: int) -> None:
        self._lut, self._W, self._r = lut, W, rank

    def m4(self, j, ts, n_anchors, nb, bs, wq, oq=None, hosts=None, fd=-1):
        return compute_m4(window_array(fd), self._lut, j, ts, n_anchors, nb, self._W, bs, wq, self._r, oq, hosts)

    def close(self, timeout_s: float = 5.0) -> None:
        pass


def _helper_loop(conn) -> None:
    """Helper process body: first message = (W, rank, b_ms, c_ms) (no path, no data); then answers
    (seq, 'ping') and (seq, 'm4', ...) + the window's N as a file descriptor (SCM_RIGHTS) with
    (seq, 'ok'|'error', payload) until EOF or None."""
    from multiprocessing import reduction

    try:
        W, rank, b_ms, c_ms = conn.recv()
    except (EOFError, OSError):
        return
    try:
        lut = cost_lut(b_ms, c_ms)
        err = ""
    except BaseException as e:  # reported on the first request
        lut, err = None, f"{type(e).__name__}: {e}"
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        seq = msg[0] if isinstance(msg, tuple) and msg else -1
        kind = msg[1] if isinstance(msg, tuple) and len(msg) > 1 else None
        fd = -1
        if kind == "m4":
            try:
                fd = reduction.recv_handle(conn)  # the window's sealed N buffer; the helper never gets a path
            except (EOFError, OSError, RuntimeError):
                return
        try:
            if err:
                raise RuntimeError(err)
            body = msg[2:]
            if kind == "ping":
                rep = (seq, "ok", os.getpid())
            elif kind == "m4":
                j, ts, n_anchors, nb, bs, wq_bytes, oq_bytes, hosts_bytes = body
                wq = np.frombuffer(wq_bytes, dtype=np.int64)
                oq = None if oq_bytes is None else np.frombuffer(oq_bytes, dtype=np.int64)
                hosts = None if hosts_bytes is None else np.frombuffer(hosts_bytes, dtype=np.int64)
                n_win = window_array(fd)
                os.close(fd)
                fd = -1
                rep = (seq, "ok", compute_m4(n_win, lut, j, ts, n_anchors, nb, W, bs, wq, rank, oq, hosts))
                del n_win  # unmaps the window
            else:
                raise ValueError(f"unknown request {kind!r}")
        except Exception as e:
            rep = (seq, "error", f"{type(e).__name__}: {e}")
        finally:
            if fd >= 0:
                os.close(fd)
        try:
            conn.send(rep)
        except (EOFError, OSError):
            return


def _helper_entry(fd: int, ppid: int = 0, pdeath: int = 0) -> None:
    """``python -c`` entry of the helper (a fresh interpreter: imports numpy + this module only).

    SIGINT is ignored (a Ctrl-C to the process group is the trainer's to handle; the helper
    ends on EOF). With ``pdeath`` (the spawning thread is the trainer's main thread, which
    lives as long as the rank) the kernel SIGKILLs the helper when the trainer dies, so its
    lifetime no longer depends on forked DataLoader workers that inherited the socket."""
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if pdeath:
        try:
            import ctypes

            ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGKILL), 0, 0, 0)  # PR_SET_PDEATHSIG
        except Exception:
            pass
    if ppid and os.getppid() != ppid:
        return  # the trainer died before the parent-death signal was set
    from multiprocessing.connection import Connection

    conn = Connection(fd)
    try:
        _helper_loop(conn)
    finally:
        try:
            conn.close()
        except OSError:
            pass


_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class HelperBackend:
    """One helper process per rank: a fresh interpreter (subprocess + socketpair), so the
    partition's ~bs small numpy calls per window never contend for the trainer's GIL and
    the child does not import the trainer's __main__ (multiprocessing spawn would).
    Started at loader construction with no path and no data; a window's N reaches it as a
    file descriptor (SCM_RIGHTS) with each 'm4' request.
    Requests come from one thread at a time (the k0 loop on the main thread at stream
    start, then the single-pool pusher thread); the socket wait releases the GIL.
    Every request carries a sequence number the helper echoes. Any transport failure,
    timeout or out-of-sequence reply POISONS the backend (helper killed and reaped, every
    later call raises): a late reply can never be read as the answer to a newer request.
    The helper ends on EOF/None, on close(), or (parent-death signal) when the trainer dies."""

    def __init__(self, W: int, rank: int, timeout_s: float,
                 b_ms: float = COST_B_MS, c_ms: float = COST_C_MS) -> None:
        import socket
        import subprocess
        import sys
        from multiprocessing.connection import Connection

        self._dead = ""
        self._seq = 0
        self._lock = threading.Lock()
        self._timeout = float(timeout_s)
        a, b = socket.socketpair()
        try:
            fd = b.fileno()
            code = (
                "import sys; sys.path.insert(0, %r); "
                "from generative_recommenders.dlrm_v4.train.m4_partition import _helper_entry; "
                "_helper_entry(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))" % _PKG_ROOT
            )
            pdeath = 1 if threading.current_thread() is threading.main_thread() else 0
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            self._p = subprocess.Popen(
                [sys.executable, "-B", "-c", code, str(fd), str(os.getpid()), str(pdeath)],
                pass_fds=(fd,), close_fds=True, stdin=subprocess.DEVNULL, env=env,
            )
        except BaseException:
            a.close()
            b.close()
            raise
        b.close()
        self._conn = Connection(a.detach())
        self.pid = int(self._p.pid)
        try:
            self._conn.send((int(W), int(rank), float(b_ms), float(c_ms)))
        except BaseException as e:
            self._poison(f"init send failed: {type(e).__name__}: {e}")
            raise RuntimeError(f"m4 helper pid {self.pid} did not start (returncode={self._p.poll()}): "
                               f"{type(e).__name__}: {e}; kill switch {ENV_ON}=0") from e

    def _poison(self, why: str) -> None:
        if not self._dead:
            self._dead = why
        try:
            self._conn.close()
        except Exception:
            pass
        try:
            self._p.kill()
        except Exception:
            pass
        try:
            self._p.wait(timeout=5.0)
        except Exception:
            pass

    def _call(self, kind: str, *body, timeout_s: Optional[float] = None, fd: Optional[int] = None):
        from multiprocessing import reduction

        tmo = self._timeout if timeout_s is None else float(timeout_s)
        with self._lock:
            if self._dead:
                raise RuntimeError(f"m4 helper pid {self.pid} is unusable ({self._dead}); "
                                   f"kill switch {ENV_ON}=0")
            self._seq += 1
            seq = self._seq
            try:
                self._conn.send((seq, kind) + tuple(body))
                if fd is not None:
                    reduction.send_handle(self._conn, int(fd), self.pid)  # SCM_RIGHTS: a duplicate, not a path
                if not self._conn.poll(tmo):
                    raise TimeoutError(f"no answer within {tmo:.0f}s")
                rep = self._conn.recv()
                if not (isinstance(rep, tuple) and len(rep) == 3 and rep[0] == seq):
                    raise RuntimeError(f"out-of-sequence reply (want seq {seq})")
            except BaseException as e:
                self._poison(f"{kind!r} failed: {type(e).__name__}: {e}")
                raise RuntimeError(
                    f"m4 helper pid {self.pid} failed during {kind!r} (returncode={self._p.poll()}): "
                    f"{type(e).__name__}: {e}; kill switch {ENV_ON}=0"
                ) from e
        if rep[1] != "ok":
            raise RuntimeError(f"m4 helper failed on {kind!r}: {rep[2]}")
        return rep[2]

    def ping(self):
        """Wait for the helper's interpreter (loader construction; no data involved)."""
        return self._call("ping", timeout_s=max(self._timeout, HELPER_START_TIMEOUT_S))

    def m4(self, j, ts, n_anchors, nb, bs, wq, oq=None, hosts=None, fd=-1):
        return self._call("m4", int(j), int(ts), int(n_anchors), int(nb), int(bs), wq.tobytes(),
                          None if oq is None else np.asarray(oq, np.int64).tobytes(),
                          None if hosts is None else np.asarray(hosts, np.int64).tobytes(), fd=int(fd))

    def close(self, timeout_s: float = 5.0) -> None:
        """Bounded: never blocks longer than ~2 x timeout_s; the helper is reaped (no zombie)."""
        import subprocess

        got = self._lock.acquire(timeout=max(0.0, float(timeout_s)))
        try:
            if got and not self._dead:
                try:
                    self._conn.send(None)
                except Exception:
                    pass
            if got:
                try:
                    self._conn.close()
                except Exception:
                    pass
            if not self._dead:
                self._dead = "closed"
        finally:
            if got:
                self._lock.release()
        try:
            self._p.wait(timeout=float(timeout_s) if got else 0.0)
        except subprocess.TimeoutExpired:
            try:
                self._p.kill()
                self._p.wait(timeout=float(timeout_s))
            except Exception:
                pass
        except Exception:
            pass


class M4Dealer:
    """Per-rank dealer used by _ChainedWindowBatchSampler.push (main process)."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        backend,
        dataset=None,
        speed=None,
        hosts: Optional[np.ndarray] = None,
        lut: Optional[np.ndarray] = None,
        local_world_size: int = 0,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        # M4-HL: int64 [W] host of every rank (None = global dealing); identical on every rank
        self._hosts = host_keys(hosts, self.world_size)
        self._be = backend
        # The dataset object only: its window index is read at activate (after RUN_START).
        self._ds = dataset
        self._off: Optional[np.ndarray] = None
        self._tsi = None
        self._lut = cost_lut() if lut is None else lut
        self._lws = int(local_world_size) or self.world_size
        self._otf: Optional[m4_otf.OTFRuntime] = None
        self._otf_name: Optional[str] = None  # election socket name (None: m4_otf.socket_name(); tests)
        self._speed = speed
        self._nbs: List[int] = []
        self._depth = 0
        m = int(os.environ.get(ENV_LAG_MARGIN, LAG_MARGIN_BATCHES))
        if m < MIN_LAG_MARGIN:
            logger.warning("DLRMV4 m4: %s=%d below the pipeline pull-ahead bound; using %d",
                           ENV_LAG_MARGIN, m, MIN_LAG_MARGIN)
            m = MIN_LAG_MARGIN
        self._margin = m
        self._hash: Dict[int, int] = {}
        self._pending: Dict[int, Tuple[int, int, int, int]] = {}  # j -> (ts, b0, nb, bs): rr costs awaiting N
        self._lock = threading.Lock()
        self._timeout = float(os.environ.get(ENV_TIMEOUT, DEFAULT_FACTOR_TIMEOUT_S))
        self._log_all = os.environ.get(ENV_LOG_ALL, "0") == "1"
        self._checked = 0
        self.counts = {"m4": 0, "rr": 0}
        self._reasons: Dict[str, int] = {}
        self.deferred_costs = 0
        self.dropped_costs = 0
        self._engaged = False
        self._last_vec = None
        self._bad_vec_logged = set()
        self._closed = False
        # M4-OTF builder schedule (elected rank): batch size and cap seen by deal(), the batch counts the
        # last schedule assumed, how many of them were pushed then
        self._bs = 0
        self._nb_cap = None
        self._sched_nbs: Optional[List[int]] = None
        self._sched_pushed = 0
        self._speed_is_off = False

    # -- configuration -------------------------------------------------------------
    def set_depth(self, depth: int) -> None:
        self._depth = max(0, int(depth))

    def bind_index(self, ts_offsets, ts_indices) -> None:
        """The offline window index (activate, after RUN_START; tests)."""
        self._off = np.asarray(ts_offsets, dtype=np.int64)
        self._tsi = ts_indices

    def activate(self, depth: int, ts_list) -> bool:
        """Called once by _stream_single on the main thread before window 0 (all ranks alike).
        Declines (rank-identical, config-level) when the loader has no prefetch depth
        (NUM_WORKERS=0: the lag rule has no bound). RUN-START GATE: without RUN_START in this
        process nothing is read and M4 goes off on every rank. Otherwise reads the window index,
        registers the boundary check and starts m4_otf (election, builder, client, release)."""
        global _ACTIVE
        self.set_depth(depth)
        if self._depth <= 0:
            if self._speed is not None:
                self._speed.set_enabled(False, "train loader prefetch depth 0")
            logger.warning("DLRMV4 m4 speed-aware dealing DECLINED on rank %d: prefetch depth 0", self.rank)
            self.close()
            return False
        rs = _run_start_wall()
        if rs is None:
            # No data is read; the flag travels in m4_speed's boundary-0 arming gather: M4 off on every rank.
            if self._speed is not None:
                self._speed.set_enabled(False, "m4otf gate closed: no run_start in this process")
            logger.warning("DLRMV4 m4otf gate closed on rank %d: no run_start in this process (resume or no MLPerf "
                           "logger): no data read, M4 off on every rank from window boundary 0 (round-robin)",
                           self.rank)
            self.close()
            return False
        if _ACTIVE:
            raise RuntimeError("m4: a second dealer was activated in this process")
        ds = self._ds
        if hasattr(ds, "_ensure_streaming_index"):
            ds._ensure_streaming_index()  # after RUN_START; the stream's first push reads it anyway
        off = getattr(ds, "_ts_offsets", None)
        tsi = getattr(ds, "_ts_indices", None)
        if off is None or tsi is None:
            raise RuntimeError("DLRM_HSTU_FP8_M4=1 needs the offline ts index (DL_WINDOW_INDEX=1)")
        self.bind_index(off, tsi)
        _ACTIVE.append(self)
        if self._speed is not None:
            self._speed.register_boundary_callback(self.boundary_check)
        ts = [int(t) for t in ts_list]
        nw = self._off.shape[0] - 1
        sizes = [int(self._off[t + 1] - self._off[t]) if 0 <= t < nw else -1 for t in ts]
        self._otf = m4_otf.OTFRuntime.start(self.rank, ts, sizes, m4_otf.data_paths(ds), self._lws, rs,
                                            name=self._otf_name)
        logger.warning(
            "DLRMV4 armed OK: m4 speed-aware dealing rank=%d/%d depth=%d lag_margin=%d law=%s dealing=%s "
            "N=on-the-fly (m4otf builder=%s connected=%s, windows=%d); kill switch %s=0",
            self.rank, self.world_size, self._depth, self._margin, getattr(self, "law", None),
            "global" if self._hosts is None else "host-local(hosts=%d)" % (int(self._hosts.max()) + 1),
            "elected" if self._otf.elected else "client", "yes" if self._otf.connected() else "NO",
            len(ts), ENV_ON,
        )
        return True

    def close(self, timeout_s: float = 5.0) -> None:
        """Stop the OTF runtime and the helper (teardown_close, after the pusher join; a declined or
        abandoned dealer). Bounded; never raises."""
        if self._closed:
            return
        self._closed = True
        otf = self._otf
        if otf is not None:
            try:
                otf.close(timeout_s)
            except Exception as e:  # teardown must go on
                logger.warning("[teardown] m4otf close failed on rank %d: %s: %s", self.rank, type(e).__name__, e)
        try:
            self._be.close(timeout_s)
        except Exception as e:  # teardown must go on
            logger.warning("[teardown] m4 helper close failed on rank %d: %s: %s", self.rank, type(e).__name__, e)

    def stream_closed(self, why: str = "train window stream closed") -> None:
        """End-of-stream: wake any factors_for waiter (m4_speed.shutdown, when provided) and any
        readiness waiter; close the OTF client, its windows and (elected rank) the builder."""
        fn = getattr(self._speed, "shutdown", None)
        if fn is not None:
            try:
                fn(why)
            except Exception as e:  # the stream's finally must go on
                logger.warning("DLRMV4 m4 rank=%d: m4_speed.shutdown failed: %s: %s", self.rank, type(e).__name__, e)
        otf = self._otf
        if otf is not None:
            try:
                otf.close()
            except Exception as e:  # the stream's finally must go on
                logger.warning("DLRMV4 m4otf rank=%d: close failed: %s: %s", self.rank, type(e).__name__, e)

    def summary(self) -> None:
        with self._lock:
            c = dict(self.counts)
            r = dict(self._reasons)
        logger.warning("DLRMV4 m4 summary rank=%d: windows m4=%d rr=%d; rr reasons %s", self.rank, c["m4"], c["rr"], r)
        otf = self._otf
        if otf is not None:
            logger.warning("DLRMV4 m4otf summary rank=%d: windows built=%d ready_at_decision=%d not_ready=%d "
                           "deferred_costs=%d dropped_costs=%d rejected=%d builder=%s%s", self.rank, otf.received,
                           otf.book.ready, otf.book.not_ready, self.deferred_costs, self.dropped_costs, otf.rejected,
                           otf.role(), (" stopped=(%s)" % otf.stopped) if otf.stopped else "")

    # -- helpers -------------------------------------------------------------------
    def _window_ok(self, ts: int, gi: np.ndarray) -> str:
        if self._off is None:
            return "no-window-index"
        if ts < 0 or ts + 1 >= self._off.shape[0]:
            return "ts-out-of-index"
        lo, hi = int(self._off[ts]), int(self._off[ts + 1])
        if gi.shape[0] != hi - lo:
            return "indices-not-window-slice(len)"
        if self._tsi is not None and gi.shape[0] and not np.array_equal(gi, self._tsi[lo:hi]):
            return "indices-not-window-slice"
        return ""

    def _factors(self, k: int) -> Optional[np.ndarray]:
        if k < 1 or self._speed is None:
            return None
        return self._speed.factors_for(k, timeout_s=self._timeout)

    def _offsets(self, k: int) -> Optional[np.ndarray]:
        """Optional per-rank fixed offsets (ms) published with factors[k] (m4_speed.offsets_for)."""
        fn = getattr(self._speed, "offsets_for", None)
        if fn is None or k < 1:
            return None
        return fn(k, timeout_s=self._timeout)

    def _ready(self, j: int, boundary: int) -> bool:
        """M4-OTF: did EVERY rank hold window j's N at window boundary ``boundary`` (identical on all ranks)."""
        otf = self._otf
        return otf is not None and otf.book.decided_ready(j, boundary, self._timeout)

    def _speed_off(self) -> bool:
        """M4 off for good on this rank (m4_speed.enabled() False: a global OFF, or this rank's own decline,
        which turns M4 off on every rank at the next gather; one-way latches): no later window can use N,
        and m4_speed ignores (global OFF) or no longer needs (local decline) round-robin costs."""
        if self._speed_is_off:
            return True
        fn = getattr(self._speed, "enabled", None)
        if fn is None:
            return False
        try:
            off = not fn()
        except Exception:  # an estimator that cannot answer: keep going (costs are only samples)
            return False
        self._speed_is_off = bool(off)
        return self._speed_is_off

    def _rr_costs(self, j: int, ts: int, b0: int, nb: int, bs: int) -> Optional[np.ndarray]:
        """ms cost per batch of this rank's round-robin anchors of window j (= ts), or None (N not held)."""
        otf = self._otf
        fd = otf.dup_fd(j, ts) if otf is not None else None
        if fd is None:
            return None
        try:
            n_win = window_array(fd)
        finally:
            os.close(fd)
        return compute_rr_costs(n_win, self._lut, b0, nb, self.world_size, bs, self.rank)

    # -- main entry ----------------------------------------------------------------
    def deal(self, j: int, ts: int, global_indices, skip_samples: int, nb_cap, bs: int) -> Tuple[np.ndarray, int]:
        """(per_rank index array, nb) for run-window j; also records nb, costs, hash."""
        t0 = time.perf_counter()
        W, r = self.world_size, self.rank
        gi = np.asarray(global_indices)
        n = (gi.shape[0] // W) * W
        if skip_samples < 0 or skip_samples > n // W:
            raise ValueError(
                f"skip_samples={skip_samples} out of [0, {n // W}] for rank={r} world_size={W}"
            )
        nb = (n // W - skip_samples) // bs
        if nb_cap is not None:
            nb = min(nb, max(0, int(nb_cap)))
        with self._lock:
            if j != len(self._nbs):
                raise RuntimeError(f"m4: windows pushed out of order (j={j}, have {len(self._nbs)})")
            nbs = list(self._nbs)
            self._bs = int(bs)
            if j >= 1:
                self._nb_cap = nb_cap  # identical for every window after the first (utils nb_cap)
        k = lag_k(j, nbs, self._depth, self._margin) if self._depth > 0 else -1
        g = None
        fd = None
        otf = "-"
        why = self._window_ok(ts, gi)
        if skip_samples:
            reason = "skip_samples"
        elif why:
            reason = why
        elif nb == 0:
            reason = "empty"
        elif k < 1:
            reason = f"no-factors(k={k})"
        else:
            g = self._factors(k)
            reason = "" if g is not None else f"factors-None(k={k})"
            if g is not None:
                bad = factors_valid(g, W)
                if bad:
                    if k not in self._bad_vec_logged:
                        self._bad_vec_logged.add(k)
                        logger.warning("DLRMV4 m4 rank=%d: factors[k=%d] unusable (%s): %r; window j=%d dealt "
                                       "round-robin", r, k, bad, g, j)
                    reason, g = f"bad-factors(k={k})", None
            if g is not None:
                # M4-OTF readiness: the AND recorded at boundary k + 1 (identical on every rank)
                if not self._ready(j, k + 1):
                    reason, g, otf = f"otf-not-ready(k={k})", None, "not-ready@%d" % (k + 1)
                else:
                    otf = "ready@%d" % (k + 1)
                    fd = self._otf.dup_fd(j, ts)
                    if fd is None:
                        if not self._otf.closed():
                            raise RuntimeError(f"m4otf: window j={j} was ready at boundary {k + 1} on every rank but "
                                               f"its N is not held on rank {r}; kill switch {ENV_ON}=0")
                        reason, g = "stream-closed", None  # teardown: this window never trains
        G = bs * W
        defer = None
        if g is None:
            # stock round-robin slice (byte-identical to the stock push)
            per_rank = np.array(gi[:n][r::W][skip_samples:], copy=True)
            costs_ms = None
            if not why and skip_samples % bs == 0 and nb > 0 and not self._speed_off():
                costs_ms = self._rr_costs(j, ts, skip_samples // bs, nb, bs)
                if costs_ms is None:
                    defer = (ts, skip_samples // bs, nb, bs)  # noted at a later window boundary if N arrives
            h = _h63(b"rr", np.int64([j, ts, n, skip_samples, nb]).tobytes())
            mode, vec = "rr", "-"
        else:
            try:
                wq = weights_from_factors(g, W)
                oq = offsets_from_ms(self._offsets(k), W)
                pos, costs_ms, h = self._be.m4(j, ts, gi.shape[0], nb, bs, wq, oq, self._hosts, fd)
            finally:
                os.close(fd)
            if pos.shape[0] != nb * bs:
                raise RuntimeError(f"m4: rank {r} got {pos.shape[0]} anchors, want {nb * bs}")
            if self._hosts is not None:
                # host-local self-check, O(nb*bs): every position this rank serves has its round-robin rank on this
                # rank's host (catches a host map lost on the way to the partition; never fires on correct code)
                rr = (pos % G) % W
                if not np.all(self._hosts[rr] == self._hosts[r]):
                    raise RuntimeError(f"m4: host-local deal of window j={j} gave rank {r} anchors of another host")
            per_rank = np.asarray(gi[: nb * G])[pos]
            mode, vec = "m4", "%016x" % _h63(wq.tobytes())
        with self._lock:
            self._nbs.append(int(nb))
            self._hash[j] = int(h)
            self.counts[mode] += 1
            if mode == "rr":
                key = reason.split("(")[0] or "none"
                self._reasons[key] = self._reasons.get(key, 0) + 1
        if mode == "m4" and not self._engaged:
            self._engaged = True
            logger.warning("DLRMV4 armed OK: m4 engaged rank=%d at j=%d ts=%d (k=%d)", r, j, ts, k)
        if mode == "m4" and r == 0 and vec != self._last_vec:
            self._last_vec = vec
            logger.warning("[m4] factors rank=0 k=%d vec=%s wq=%s g=%s", k, vec, wq.tolist(),
                           np.round(np.asarray(g, np.float64), 6).tolist())
        if self._speed is not None:
            # exactly once per window on every rank; empty when there are no costs yet (nb = 0, mismatched
            # window, partial skip, N not held: m4_speed treats it as no sample unless a later note
            # for the same window (deferred costs) overwrites it before the window closes
            self._speed.note_window_costs(
                j, np.asarray(costs_ms, dtype=np.float64) if costs_ms is not None else np.zeros(0, np.float64))
            if defer is not None:
                with self._lock:
                    self._pending[j] = defer
        if r == 0 or self._log_all:
            logger.warning(
                "[m4] rank=%d j=%d ts=%d k=%d mode=%s%s nb=%d vec=%s owner=%016x ms=%.1f otf=%s costs=%s",
                r, j, ts, k, mode, (" reason=" + reason) if reason else "", nb, vec, h,
                (time.perf_counter() - t0) * 1e3, otf,
                "helper" if mode == "m4" else ("now" if costs_ms is not None else ("deferred" if defer else "none")),
            )
        return per_rank, nb

    # -- cross-rank check (m4_speed boundary callback, main thread) ----------------
    def window_hash(self, j: int) -> Optional[int]:
        with self._lock:
            return self._hash.get(j)

    def _gather(self, vals: List[int]) -> Optional[List[List[int]]]:
        """ONE all_gather of int64 [len(vals)] on the default process group; None without one."""
        import torch
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() <= 1:
            return None
        dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        t = torch.tensor(vals, dtype=torch.int64, device=dev)
        out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(out, t)
        return [[int(v) for v in row] for row in torch.stack(out).cpu().tolist()]

    def boundary_check(self, i: int) -> None:
        """Window boundary i (every rank, main thread): (a) receive newly built windows, (b) readiness bitmap
        of windows [i, i + 128), (c) ONE all_gather of [owner hash of window i, readiness words], (d) the
        owner-hash check, (e) the readiness AND of boundary i, (f) deferred round-robin costs, (g) builder
        progress / go (elected rank)."""
        if i == 0:
            return  # window 0 is always round-robin (lag_k(0) = -1); keep barrier 0 collective-free
        otf = self._otf
        w0 = w1 = 0
        if otf is not None:
            otf.drain(i)
            w0, w1 = otf.bits(i)
        h = self.window_hash(i)
        mine = -1 if h is None else int(h)
        rows = self._gather([mine, w0, w1])
        if rows is not None:
            vals = [x[0] for x in rows]
            if mine == -1 or any(v != vals[0] for v in vals):
                raise RuntimeError(
                    f"DLRMV4 m4 owner-hash MISMATCH at window j={i}: "
                    f"{['%016x' % v if v >= 0 else 'none' for v in vals]} (rank {self.rank}); "
                    f"kill switch DLRM_HSTU_FP8_M4=0"
                )
            self._checked += 1
            if self._checked == 1:
                logger.warning(
                    "DLRMV4 armed OK: m4 owner-hash check at window boundary j=%d: %d ranks agree (%016x)",
                    i, len(vals), vals[0],
                )
        else:
            rows = [[mine, w0, w1]]  # no process group (one process): nothing to check, own readiness
        if otf is None:
            return
        a0 = a1 = -1
        for x in rows:
            a0 &= int(x[1])
            a1 &= int(x[2])
        otf.book.record(i, a0, a1)
        if self._speed_off():
            # no later window can use N: no more round-robin costs, and the elected rank stops the builder
            # (held windows stay valid for windows already decided ready; the gather above keeps running)
            with self._lock:
                dropped = len(self._pending)
                self._pending.clear()
            self.dropped_costs += dropped
            otf.stop_builder(f"M4 is off for good from window boundary {i} (m4_speed)")
            return
        self._deferred(i)
        if otf.elected and otf.builder is not None and not otf.stopped:
            self._send_schedule(otf)
            with self._lock:
                done = sum(self._nbs[:i])
            otf.progress(i, done)

    def _send_schedule(self, otf) -> None:
        """Elected rank: the builder's schedule hint ('sched', need; m4_otf step 3) once the batch size is
        known, and again only if a pushed window's batch count differs from the one it assumed (never in
        a cold start, where the prediction is exact). Hint only: decisions are the readiness AND."""
        with self._lock:
            nbs = list(self._nbs)
            bs, cap = self._bs, self._nb_cap
        if bs <= 0 or self._depth <= 0 or otf.builder is None or otf.builder.dead:
            return
        basis = self._sched_nbs
        if basis is not None:
            if all(j < len(basis) and nbs[j] == basis[j] for j in range(self._sched_pushed, len(nbs))):
                self._sched_pushed = len(nbs)
                return
        full = otf_nbs(nbs, otf.sizes, self.world_size, bs, cap)
        if otf.schedule(otf_need(full, self._depth, self._margin)):
            self._sched_nbs = full
            self._sched_pushed = len(nbs)

    def _deferred(self, i: int) -> None:
        """Round-robin costs of windows pushed before their N was held: note them once N is here; window i
        (starting now) is the last chance, older ones are closed."""
        with self._lock:
            pend = sorted(self._pending.items())
        for j, (ts, b0, nb, bs) in pend:
            costs = self._rr_costs(j, ts, b0, nb, bs) if j >= i else None
            if costs is None and j > i:
                continue  # N not here yet; window j starts at a later boundary
            if costs is not None:
                if self._speed is not None:
                    self._speed.note_window_costs(j, np.asarray(costs, dtype=np.float64))
                self.deferred_costs += 1
            else:
                self.dropped_costs += 1  # no sample for that window
            with self._lock:
                self._pending.pop(j, None)


_DISABLED_REASON: List[str] = []
_ACTIVE: List["M4Dealer"] = []
_CREATED: List["M4Dealer"] = []


def decline_if_requested(rank: int, reason: str) -> None:
    """A training path that never builds the single-pool train loader: with M4 on, disable
    m4_speed (identically on every rank) so it does no per-window work."""
    if not switch_on():
        return
    try:
        from generative_recommenders.dlrm_v4.train import m4_speed as m4s
    except ImportError as e:  # pragma: no cover - depends on the x4c19 module
        raise RuntimeError(f"DLRM_HSTU_FP8_M4=1 but m4_speed is unavailable: {e}") from e
    _DISABLED_REASON.append(reason)
    m4s.set_enabled(False, reason)
    logger.warning("DLRMV4 m4 speed-aware dealing DECLINED on rank %d: %s", rank, reason)


def _run_start_wall() -> Optional[float]:
    """Wall time of this process's MLPerf RUN_START (dl_prestart.set_run_start), or None (gate closed)."""
    from generative_recommenders.dlrm_v4.train import dl_prestart

    return dl_prestart.run_start_wall()


def maybe_create(dataset, rank: int, world_size: int, single_pool: bool) -> Optional[M4Dealer]:
    """Create the dealer when M4 is ON (main process, before RUN_START when DL_PRESTART builds the loaders).

    Configuration only: nothing derived from training data is read here (M4-OTF); the dealer keeps
    a reference to the dataset object and the helper starts without any path. Config-level declines
    are identical on every rank. With M4 ON, a missing m4_speed or a dataset without the streaming
    offline ts index RAISES (fail closed); a dataset configured otherwise than the on-the-fly N rule
    assumes DECLINES before the helper starts (m4_otf.nrule_config_mismatch; review C5).
    """
    if not switch_on():
        return None
    # A second train loader in this process (NUM_WORKERS=0 + DL_PRESTART: prestart builds one,
    # declines, and the loop builds another): close the earlier, never-activated dealer.
    for d in [d for d in _CREATED if d not in _ACTIVE]:
        logger.warning("DLRMV4 m4: closing the unused dealer of an earlier train loader (rank %d)", rank)
        d.close()
    _CREATED[:] = [d for d in _CREATED if d in _ACTIVE]
    try:
        from generative_recommenders.dlrm_v4.train import m4_speed as m4s
    except ImportError as e:  # pragma: no cover - depends on the x4c19 module
        raise RuntimeError(f"DLRM_HSTU_FP8_M4=1 but m4_speed is unavailable: {e}") from e
    reason = ""
    hosts = None
    nnodes = _nnodes()
    walk, walk_why = _walk_on()
    hl_mode = os.environ.get(ENV_HOST_LOCAL, "").strip().lower() or "auto"
    walk_global = False  # global dealing with the walk on across hosts (HOST_LOCAL=0: the ship15 form)
    if not single_pool:
        reason = "DL_SINGLE_POOL=0 (double-buffered set_window path)"
    elif world_size <= 1:
        reason = "world_size<=1"
    elif hl_mode not in ("auto", "1", "0"):
        reason = f"{ENV_HOST_LOCAL}={hl_mode!r} is not auto/1/0 (fail closed)"
    elif nnodes > 1 and (hl_mode == "1" or (hl_mode == "auto" and walk)):
        hosts, reason = host_map(world_size, nnodes)
        if hosts is not None:
            reason = _layout_check(rank, hosts)
    elif nnodes > 1 and walk:
        walk_global = True  # hl_mode == "0"
    if reason:
        _DISABLED_REASON.append(reason)
        m4s.set_enabled(False, reason)
        logger.warning("DLRMV4 m4 speed-aware dealing DECLINED on rank %d: %s", rank, reason)
        return None
    if hasattr(m4s, "enabled") and not m4s.enabled():
        logger.warning("DLRMV4 m4 speed-aware dealing DECLINED on rank %d: m4_speed disabled", rank)
        return None
    ds = dataset
    missing = [a for a in ("_ensure_streaming_index", "_cache_dir", "_positions_name", "_streaming_window_seconds")
               if not hasattr(ds, a)]
    if missing or not getattr(ds, "_use_offline_ts_indices", True):
        raise RuntimeError("DLRM_HSTU_FP8_M4=1 needs the streaming dataset with the offline ts index "
                           f"(DL_WINDOW_INDEX=1); missing {missing or ['_use_offline_ts_indices']}")
    # Review C5 (iter59): the on-the-fly N rule is fixed (m4_otf.NRULE_*). Fail closed, before the helper
    # starts, unless the dataset's configuration is the one it assumes (attributes only; nothing is read).
    differs = m4_otf.nrule_config_mismatch(ds)
    if differs:
        reason = (f"the dataset configuration differs from the m4otf N rule's ({differs}); N would not be the "
                  "model's sequence length")
        _DISABLED_REASON.append(reason)
        m4s.set_enabled(False, reason)
        logger.warning("DLRMV4 m4 speed-aware dealing DECLINED on rank %d: %s", rank, reason)
        return None
    t0 = time.time()
    # One cost law for allocation and for the speed estimator's normalisation (m4_speed):
    # the loop then converges to equal measured spans even if the law is off.
    b_ms = float(getattr(m4s, "COST_B_MS", getattr(m4s, "COST_B", COST_B_MS)))
    c_ms = float(getattr(m4s, "COST_C_MS", getattr(m4s, "COST_C", COST_C_MS)))
    lut = cost_lut(b_ms, c_ms)
    backend = HelperBackend(world_size, rank,
                            float(os.environ.get(ENV_HELPER_TIMEOUT, DEFAULT_HELPER_TIMEOUT_S)), b_ms=b_ms, c_ms=c_ms)
    try:
        backend.ping()  # the helper's interpreter is up (before RUN_START under DL_PRESTART); no data
    except BaseException:
        backend.close()
        raise
    law_src = "m4_speed" if hasattr(m4s, "COST_B_MS") or hasattr(m4s, "COST_B") else "local default"
    lws = world_size // nnodes if world_size % nnodes == 0 else world_size
    dealer = M4Dealer(rank, world_size, backend, ds, speed=m4s, hosts=hosts, lut=lut, local_world_size=lws)
    dealer.law = (b_ms, c_ms, law_src)
    _CREATED.append(dealer)
    logger.warning(
        "DLRMV4 m4 dealer created rank=%d/%d (N on the fly after RUN_START (m4otf), law b=%.4g c=%.4g ms from %s, "
        "helper pid %d, ready %.1fs; dealing=%s (%s=%s, walk=%s%s, nnodes=%d); armed when the train stream starts)",
        rank, world_size, b_ms, c_ms, law_src, backend.pid, time.time() - t0,
        "global" if hosts is None else "host-local(hosts=%d)" % nnodes, ENV_HOST_LOCAL, hl_mode,
        "on" if walk else "off", (" (%s)" % walk_why) if walk_why else "", nnodes,
    )
    if walk_global:
        logger.warning(
            "DLRMV4 m4 dealing=global with the streaming page walk on across %d hosts (%s=0; examination "
            "setting): each host's walk advises its ranks' round-robin ranges, so anchors dealt to another host "
            "may read pages that host's walk did not advise; walk coverage approximate (rank %d)",
            nnodes, ENV_HOST_LOCAL, rank,
        )
    return dealer


def _nnodes() -> int:
    """$NNODES as train_ranker and cold_prefetch read it (missing / not a positive integer -> 1)."""
    try:
        n = int(os.environ.get("NNODES", "").strip() or 1)
    except ValueError:
        n = 1
    return max(1, n)


def _walk_on() -> Tuple[bool, str]:
    """(does the streaming page walk run, why): the walk's own two gates, read the way the
    train stream applies them (_stream_single starts cold_prefetch only when mode() > 0, and
    cold_prefetch.start runs the stream only when resolve_stream() is on).

    mode() > 0: DL_COLD_PREFETCH=1|2, or the pre-start level 2 (DL_PRESTART=2), which
    prestart_streaming_loaders sets before it builds the train loader (hence before
    maybe_create) on both the pre-start and the legacy DL_PRESTART=0 path.
    resolve_stream(): DL_COLD_PREFETCH_STREAM 1/true/on/yes force the stream on; auto runs it on
    more than one node ($NNODES). Partition v2 declined only the force-on values, so
    DL_COLD_PREFETCH_STREAM=auto on >= 2 nodes ran the walk under global dealing.
    No cold_prefetch module -> no walk."""
    try:
        from generative_recommenders.dlrm_v4.train.cold_prefetch import mode as _cp_mode
        from generative_recommenders.dlrm_v4.train.cold_prefetch import resolve_stream as _cp_resolve
    except ImportError:  # a stripped tree without the walk
        return False, "cold_prefetch not importable (no streaming walk)"
    if not _cp_mode():
        return False, "prefetcher off (cold_prefetch.mode()=0)"
    on, why = _cp_resolve()
    return bool(on), str(why or "")


def _layout_check(rank: int, hosts: np.ndarray) -> str:
    """'' unless this rank's host in the map is not its node: $NODE_RANK (exported by
    launch_slurm.sh to every rank; train_ranker reads it) must equal hosts[rank]. A launcher
    that does not lay ranks out node-major would make 'host-local' dealing move anchors off
    their physical host while every log said host-local. This decline can fire on SOME ranks
    only; m4_speed's arming gather at window boundary 0 then turns M4 off on every rank
    (round-robin everywhere). Unset / non-integer NODE_RANK: no check."""
    raw = os.environ.get("NODE_RANK", "").strip()
    try:
        nr = int(raw)
    except ValueError:
        return ""
    if 0 <= rank < hosts.shape[0] and int(hosts[rank]) != nr:
        return (f"host-local dealing: rank {rank} runs on NODE_RANK={nr} but the node-major host map puts it on "
                f"host {int(hosts[rank])} (rank layout is not node-major)")
    return ""


def host_map(world_size: int, nnodes: int) -> Tuple[Optional[np.ndarray], str]:
    """(int64 [W] host of every rank, decline reason): host(r) = r // (world_size / nnodes).

    train_ranker: rank = node_rank * gpus_per_node + local_rank and WORLD_SIZE = NNODES *
    GPUS_PER_NODE (GPUS_PER_NODE = $GPUS_PER_NODE or the visible device count). A world size
    NNODES does not divide, or a positive $GPUS_PER_NODE that disagrees with world_size / NNODES,
    leaves the host of a rank unknown: decline (identically on every rank) rather than guess."""
    if nnodes <= 1:
        return None, ""
    if world_size % nnodes:
        return None, f"host-local dealing: NNODES={nnodes} does not divide world_size={world_size}"
    gpn = world_size // nnodes
    raw = os.environ.get("GPUS_PER_NODE", "").strip()
    try:
        env_gpn = int(raw) if raw else 0
    except ValueError:
        env_gpn = 0
    if env_gpn > 0 and env_gpn != gpn:
        return None, (f"host-local dealing: GPUS_PER_NODE={env_gpn} != world_size/NNODES = "
                      f"{world_size}/{nnodes} (host map unknown)")
    return np.arange(world_size, dtype=np.int64) // gpn, ""
