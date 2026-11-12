# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
"""Drain-free training step: move the per-step host reads off the GPU queue.

Arbor 2026-10-02 iter 8. Trace attribution of the base tree (5ae3127, 1n,
TB1/TB2 steps 54-56; probes/iter8/{idle,comm}) found one blocking read per
step that EMPTIES the launch queue: ``torchrec.metrics.gauc.get_auc_states``
does ``int(num_candidates.max().item())`` right after ``progress()`` has
queued fwd+bwd+opt, so the host blocks ~110-130 ms until the GPU drains, and
every later launch (metric ops, loss all_reduce, MLLOG, the next forward's
preprocess) lands on an empty queue. Critical-rank cost: a 5-6 ms GPU bubble
per step. Removing it alone only moves the drain to the next host read in
line -- the forward's ``fx_length_stats`` ``.tolist()`` and the per-step
MLPerf train-loss ``float()`` -- so all three go together:

1. ``Samples.to`` (CPU -> device) computes the batch's length statistics from
   the CPU lengths it is about to copy (max/sum of num_candidates, per-key max
   of the UIH lengths) and tags them onto the DEVICE lengths tensors.
2. ``fx_length_stats`` and GAUC's ``max_length`` read those tags (validated
   against the exact view geometry; anything unrecognised falls back to the
   device read). The integers are identical by construction: the same
   reductions over the same values, before vs after the H2D copy.
3. The MLPerf train-loss value is read one step late from a non-blocking
   pinned D2H copy; the pending event is emitted before ANY other MLLOG event
   (pre-emit hook on MLPerfLogger), so ordering and metadata are unchanged.

4. (iters 12-15) torchrec's KJT input_dist reads ``length_per_key`` back from
   the device after the values all_to_all (a pageable 112 B D2H that became
   the next drain point). The per-key counts are host-known on the sender, so
   they ride along as K extra rows of the splits all_to_all (non-fused and
   fused ``FusedKJTListSplitsAwaitable`` paths; one pinned H2D for the rows,
   and torchrec's debug ``_safe_tolist`` is answered from the host copy).
   Knob ``DLRM_HSTU_FP8_SYNCFREE_LPK`` (default 1).
5. (iter 15) The forward's ``uih_features[key]`` / ``candidates`` KJT access
   computed ``length_per_key`` with a pageable 120 B D2H -- the drain point
   once (4) landed. ``Samples.to`` now caches it from the CPU lengths
   (``KeyedJaggedTensor.to`` carries ``_length_per_key``). Knob
   ``DLRM_HSTU_FP8_SYNCFREE_FWDLPK`` (default 1). Same integers.
   1n same-allocation ABBA (job 54854): sync-free +4.95/+5.00% vs off
   (control spread 0.31%), +1.3/+2.3% over (4) alone. 2n (job 54852): still
   -2.5/-2.8% vs off -- at 2n the regression grows with the host lead
   (draining at the forward instead: -1.5/-1.8%), so the single-node gate
   stays (probes/iter15/FINDINGS.md).

6. (iter 19) Multi-node: the +31 us/dispatch penalty came from the
   HIGH-priority torchrec data-dist queue holding blocked waits while the host
   ran ahead. ``dist_init_on_main`` moves the forward's dist_init wait to the
   main stream (with the record_stream bookkeeping that makes that safe) and
   ``split_wait_poll_on`` issues the deferred splits wait only once the
   collective has finished. One configuration, prio -1 kept, at 1n/2n/4n
   (probes/iter19/FINDINGS.md). Knobs ``..._DISTINIT_MAIN``,
   ``..._SPLITWAIT_POLL`` (default 1 when sync-free is active). With that,
   the single-node gate of (5) was removed (2026-10-03): ``enabled()`` follows
   the knob alone at every node count. (Item 67, 2026-10-08: once item 66
   removed the false H2D gate, the TRAIN pipeline's pair runs at priority 0
   at 2+ nodes -- train_pipe_stream_priority(); 1n and eval keep -1.)

A one-step lag fence (``StepFence``) keeps the host at most one step ahead of
the GPU, which bounds queued memory/allocator state the same way the drain
used to, without emptying the queue.

Same math on every sample: nothing is skipped, cached across steps, or
reordered on the device. Kill switch: ``DLRM_HSTU_FP8_SYNCFREE=0`` (the
``DLRM_HSTU_FP8_`` prefix is what launch_slurm.sh forwards into the
container; the knob itself has nothing to do with FP8). Debug:
``DLRM_HSTU_FP8_SYNCFREE_VERIFY=N`` cross-checks the host values against the
device read for the first N uses of each kind and raises on a mismatch.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

logger: logging.Logger = logging.getLogger(__name__)

_ENV = "DLRM_HSTU_FP8_SYNCFREE"
_VERIFY_ENV = "DLRM_HSTU_FP8_SYNCFREE_VERIFY"
_HOST_CAND = "_dlrm_host_cand_stats"  # (stride, max_c, sum_c) on cand lengths
_HOST_UIH = "_dlrm_host_key_max"  # (stride, [max per key]) on uih lengths

_enabled_logged: Optional[bool] = None
_verify_left = {"len": 0, "gauc": 0, "lpk": 0, "fwdlpk": 0}
_verify_init = False
_announced = set()


def _multi_node() -> bool:
    try:
        if int(os.environ.get("NNODES", "1") or 1) > 1:
            return True
    except ValueError:
        pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size() > max(1, torch.cuda.device_count())
    except Exception:
        pass
    return False


_multinode_noop_logged = False


def enabled() -> bool:
    """On when the knob is set, at every node count.

    History: iters 8-15 gated this to single-node (2n ABBA -3.56/-3.61%, job
    53941; still -2.5/-2.8% at iter 15, job 54852). Iter 19 found the 2n/4n
    mechanism (blocked waits on torchrec's high-priority data-dist queue) and
    removed it with dist_init_on_main() + split_wait_poll_on(), one
    configuration for 1n/2n/4n (probes/iter19, iter21). The node-count gate
    is gone; DLRM_HSTU_FP8_SYNCFREE_MULTINODE is accepted and ignored.
    """
    # Read per call (cheap): the gin env bootstrap may set it after import.
    global _enabled_logged, _multinode_noop_logged
    on = os.environ.get(_ENV, "0") == "1"
    if not _multinode_noop_logged and os.environ.get(_ENV + "_MULTINODE"):
        _multinode_noop_logged = True
        logger.info(
            "%s_MULTINODE=%s is a no-op: the host-sync-free step no longer "
            "has a node-count gate (only %s decides)",
            _ENV, os.environ.get(_ENV + "_MULTINODE"), _ENV,
        )
    if _enabled_logged is not on:
        _enabled_logged = on
        logger.info(
            "DLRMV4 host-sync-free step: active=%d (%s=%s, multi_node=%d)",
            int(on), _ENV, os.environ.get(_ENV, "0"), int(_multi_node()),
        )
    return on


_PRIO_ENV = _ENV.replace("SYNCFREE", "PIPE_STREAM_PRIO")  # DLRM_HSTU_FP8_PIPE_STREAM_PRIO
_DISTINIT_ENV = _ENV + "_DISTINIT_MAIN"  # DLRM_HSTU_FP8_SYNCFREE_DISTINIT_MAIN


def _prio_value(name: str) -> Optional[int]:
    """Read a stream-priority knob: unset/empty = None, else -1 or 0.

    torch on ROCm clamps stream priorities to {-1 (high), 0 (normal)}; any
    other value would come back clamped, the pipeline would then rebuild its
    streams and strand the pre-grown pools, so it is refused up front.
    """
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return None
    try:
        p = int(v)
    except ValueError:
        p = None
    if p not in (-1, 0):
        raise ValueError(
            "%s=%r: expected -1 (torchrec's high priority) or 0 (normal)" % (name, v)
        )
    return p


def pipe_stream_priority() -> Optional[int]:
    """Explicit priority for torchrec's pipeline memcpy / data-dist streams.

    None (the default) keeps torchrec's own choice (-1, high priority) for
    every pipeline; train_pipe_stream_priority() adds item (67) for the train
    pipeline at 2+ nodes. ``DLRM_HSTU_FP8_PIPE_STREAM_PRIO=<p>`` overrides it
    for every pipeline; it exists for A/B arms (iter 16-18: prio 0 was the
    per-topology workaround for the 2n/4n sync-free penalty; iter 19 replaced
    it with one configuration for every node count: dist_init_on_main() +
    split_wait_poll_on()).
    """
    return _prio_value(_PRIO_ENV)


_PRIO_MN_ENV = _PRIO_ENV + "_MN"  # DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN


def _spans_nodes() -> bool:
    """True when the job runs on more than one node (item 67's gate).

    One rule for items (64) and (67): this calls _env_bootstrap._spans_nodes,
    which also decides item (64)'s multi-node default at bootstrap and stays
    torch-free there. NNODES (exported by launch_slurm.sh at every node
    count) or SLURM_NNODES decides when set; only without either is the
    process-group size compared with the ranks of one node (LOCAL_WORLD_SIZE,
    else the visible devices). Imported here, not at module level: the
    bootstrap module imports gin.
    """
    from generative_recommenders.dlrm_v4.train import _env_bootstrap

    return _env_bootstrap._spans_nodes()


def train_pipe_stream_priority() -> Optional[int]:
    """Priority for the TRAIN pipeline's memcpy / data-dist streams (item 67).

    The A/B override ``DLRM_HSTU_FP8_PIPE_STREAM_PRIO`` (every pipeline) wins.
    Otherwise ``DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN=<p>`` (gin item 67 binds 0;
    -1 opts out) applies to the train pipeline only, and only when the job
    spans more than one node. None keeps torchrec's own choice (-1, high
    priority): 1-node runs and the eval pipeline are unchanged.

    Why (MNC iter 8-9): with item (66) the deferred next-batch input dist no
    longer waits on the following batch's H2D, so it has ~a step of slack
    before the next forward reads it. Its EC dedup / bucketize / permute
    kernels then gain nothing from pre-empting the main stream, and at high
    priority they slow the critical rank's main-stream kernels (~0.35 ms of
    4n step per ms of extra data-dist kernel work, probes dd_k1/dd_k2).

    Placement: on ROCm 7.2 HIP binds a stream once, at creation, into the pool
    of its priority (GPU_MAX_HW_QUEUES per priority, least-referenced reuse);
    at priority 0 that is the NORMAL pool, which also holds the main stream.
    On ROCm 10.2 (rocm/primus:v26.9) HIP defaults
    DEBUG_HIP_IGNORE_STREAM_PRIORITY=1: all priorities share one pool of
    GPU_MAX_HW_QUEUES queues, -1 streams keep their first queue, and
    priority-0 streams are re-placed by load whenever they go idle, so a
    placement check before RUN_START is only a snapshot. The pair must not
    land on main's queue (FIFO with the forward); alloc_pregrow / the pipeline
    check that with shares_hw_queue() and fall back to -1 if it does.
    """
    p = pipe_stream_priority()
    if p is not None:
        return p
    v = _prio_value(_PRIO_MN_ENV)  # validated at every node count
    if v is None or not _spans_nodes():
        return None
    return v


_mn_inactive_logged = False


def log_train_pipe_priority_inactive(role: str) -> bool:
    """Say once per process that item (67) is bound but the job is 1-node.

    Called where a pipeline has picked its side-stream priority and got None.
    Logs one info line, only for the train pipeline, only when
    DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN binds 0, the A/B override is unset and
    the job does not span nodes (the same gate as
    train_pipe_stream_priority()). Log only: no stream, priority, probe or
    allocation changes. True if it logged now.
    """
    global _mn_inactive_logged
    if (
        _mn_inactive_logged
        or role != "train"
        or pipe_stream_priority() is not None
        or _prio_value(_PRIO_MN_ENV) in (None, -1)
        or _spans_nodes()
    ):
        return False
    _mn_inactive_logged = True
    logger.info(
        "DLRMV4 multi-node side-stream priority inactive: single node, train "
        "pipeline keeps torchrec priority -1"
    )
    return True


def disable_train_pipe_priority(reason: str) -> None:
    """Turn item (67) off for the rest of this process (fallback to -1)."""
    os.environ[_PRIO_MN_ENV] = "-1"
    logger.warning(
        "DLRMV4 multi-node side-stream priority fallback: train pipeline side "
        "streams keep torchrec's priority -1 (%s)",
        reason,
    )


def shares_hw_queue(
    a: "torch.cuda.Stream", b: "torch.cuda.Stream", spin_ms: float = 40.0
) -> Optional[bool]:
    """True if stream ``b`` cannot run while stream ``a`` runs, i.e. HIP put
    both on one hardware queue (in-order). None if the probe could not run.

    A one-block spin kernel (torch.cuda._sleep: one CU) runs ~spin_ms on ``a``;
    a tiny spin is then issued on ``b``. On separate hardware queues the tiny
    one finishes long before the spin on ``a`` does; on a shared queue it has
    to wait for it. Synthetic, no training data; run before RUN_START.
    """
    try:
        if not torch.cuda.is_available() or not hasattr(torch.cuda, "_sleep"):
            return None
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        cycles = 1 << 20
        with torch.cuda.stream(a):
            e0.record(a)
            torch.cuda._sleep(cycles)
            e1.record(a)
        e1.synchronize()
        ms = e0.elapsed_time(e1)
        if not ms or ms <= 0:
            return None
        cycles = max(1, int(cycles * spin_ms / ms))
        done_a = torch.cuda.Event()
        done_b = torch.cuda.Event()
        with torch.cuda.stream(a):
            torch.cuda._sleep(cycles)
            done_a.record(a)
        with torch.cuda.stream(b):
            torch.cuda._sleep(64)
            done_b.record(b)
        done_b.synchronize()
        shared = bool(done_a.query())
        done_a.synchronize()
        return shared
    except Exception as e:  # never block training on the probe
        logger.warning("DLRMV4 hw-queue probe skipped: %s", e)
        return None


def dist_init_on_main() -> bool:
    """Run the forward's input-dist completion (dist_init) on the main stream.

    Iter 18 root cause of the multi-node sync-free regression (trace
    probes/iter15/sfv3_a/traces/TD, 2n, ranks 0 and 9, steps 53-57): the
    data-dist stream kernels that sit queued behind an unmet event for 100+ ms
    -- the +31 us/dispatch penalty of probes/iter16 -- are launched by
    torchrec ``PipelinedForward.__call__``: ``with stream(data_dist):
    request.wait()`` makes the HIGH-priority data-dist stream wait on the
    current batch's lengths/values all_to_all, which sits on the NCCL stream
    behind the previous backward's collectives, and the KJT dist_init kernels
    (``dist_init:recat_values``, 28/step >20 ms late) queue behind it; the
    next batch's ``dedup_ec_indices`` / start kernels (16 + 7/step) then queue
    behind those on the same FIFO stream. With the host a step ahead, that
    holds a blocked high-priority queue through the whole backward.

    torchrec's own ``PipelinedForward._use_main_stream_for_dist_init`` (set by
    torchrec itself for ``clear_data_dist_inputs``) issues that wait and those
    kernels on the current (main, normal-priority) stream instead. The forward
    waits on them either way (``current_stream().wait_stream(data_dist)`` right
    after), so the main stream reaches them in the same order; only the HW
    queue that holds the blocked barrier changes, and the pipeline streams keep
    torchrec's priority (-1), which measured +2.5% at 1n. Same kernels, same
    values, same collective issue order.

    On whenever the host-sync-free step is active; ``..._DISTINIT_MAIN=0``
    turns it off for A/B.
    """
    return enabled() and os.environ.get(_DISTINIT_ENV, "1") != "0"


_SPLITPOLL_ENV = _ENV + "_SPLITWAIT_POLL"  # DLRM_HSTU_FP8_SYNCFREE_SPLITWAIT_POLL


def split_wait_poll_on() -> bool:
    """Gate the splits all_to_all wait on a non-blocking completion query.

    Iter 19 trace (probes/iter19/sf2nt/traces/TU1, 2n, 16 ranks, steps 53-56;
    probes/iter19/blocked.py): with dist_init on main, the remaining block on
    the HIGH-priority data-dist queue is the deferred next-batch splits wait
    that ``dequeue_batch`` issues after backward N (``SplitsAllToAllAwaitable
    wait()``: work.wait() on the splits all_to_all, then the .tolist() copy)
    -- 56 blocks, mean 119 ms, kernels queued behind; the control has none.

    The host blocks on that .tolist() anyway. Instead of enqueueing the stream
    wait first and blocking in the copy, poll ``Work.is_completed()`` (an
    event query; it never blocks the GPU or a queue) and issue the stock wait +
    copy only once the collective has finished: the data-dist stream then only
    ever sees an already-satisfied wait. Same collective, same stream, same
    values. (Iter 19 W/W' held the wait on a priority-0 helper stream instead:
    2n/4n matched the prio-0 topology rule, but at 1n the helper shared the
    main stream's HW queue -- same trace stream id -- so the copy waited behind
    the whole backward and the host drained every step: 1n +1-2% vs +5.7%.)

    On with dist_init_on_main(); ``..._SPLITWAIT_POLL=0`` turns it off for A/B.
    """
    return dist_init_on_main() and os.environ.get(_SPLITPOLL_ENV, "1") != "0"


_SPLIT_RS_ENV = _ENV + "_SPLIT_RECORD_STREAM"  # DLRM_HSTU_FP8_SYNCFREE_SPLIT_RECORD_STREAM
_split_rs_announced = False


def _split_record_stream_on() -> bool:
    """record_stream the splits all_to_all buffers on the waiting stream.

    Only meaningful with dist_init_on_main(). Default off (the shipped path is
    unchanged); ``..._SPLIT_RECORD_STREAM=1`` turns it on.
    """
    return dist_init_on_main() and os.environ.get(_SPLIT_RS_ENV, "0") == "1"


def _announce_split_record_stream() -> None:
    global _split_rs_announced
    if not _split_rs_announced:
        _split_rs_announced = True
        logger.warning(
            "armed OK: %s=1: splits all_to_all buffers record_stream()ed on the "
            "waiting stream", _SPLIT_RS_ENV
        )


def _poll_until_done(work: Any) -> None:
    # Bounded only by the collective itself (the process-group watchdog still
    # aborts a hung collective); sleep releases the GIL for the loader threads.
    try:
        while not work.is_completed():
            time.sleep(5e-5)
    except Exception:  # noqa: BLE001 -- any query problem: fall back to wait()
        return


def _verify_budget(kind: str) -> bool:
    global _verify_init
    if not _verify_init:
        n = int(os.environ.get(_VERIFY_ENV, "0") or 0)
        _verify_left["len"] = n
        _verify_left["gauc"] = n
        _verify_left["lpk"] = n
        _verify_left["fwdlpk"] = n
        _verify_init = True
    if _verify_left.get(kind, 0) > 0:
        _verify_left[kind] -= 1
        return True
    return False


def _announce(kind: str) -> None:
    if kind not in _announced:
        _announced.add(kind)
        if "FALLBACK" in kind:
            logger.info("DLRMV4 host-sync-free: %s", kind)
        else:
            logger.info("DLRMV4 host-sync-free: %s armed (host value used)", kind)


def _fixed_stride_rows(kjt: Any) -> Optional[Tuple[int, int]]:
    """(num_keys, stride) when the KJT lengths are a dense [keys, stride]."""
    try:
        if kjt.variable_stride_per_key():
            return None
        nk = len(kjt.keys())
        stride = int(kjt.stride())
        lengths = kjt.lengths()
    except Exception:
        return None
    if nk <= 0 or stride <= 0 or lengths.numel() != nk * stride:
        return None
    return nk, stride


def host_stats_from_cpu(
    uih_kjt: Any, cand_kjt: Any
) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, List[int]]]]:
    """Length statistics of a CPU batch, or None if it is not a plain CPU batch."""
    gc = _fixed_stride_rows(cand_kjt)
    gu = _fixed_stride_rows(uih_kjt)
    if gc is None or gu is None:
        return None
    cl = cand_kjt.lengths()
    ul = uih_kjt.lengths()
    if cl.device.type != "cpu" or ul.device.type != "cpu":
        return None
    nkc, sc = gc
    nku, su = gu
    # numpy, not torch: a torch CPU reduction here runs on the intra-op thread
    # pool, which contends with the dataloader/pin threads and measured 5-8 ms
    # per call in the trace (probes/iter8); numpy over ~10k ints is ~10 us.
    # Integer max/sum: identical values either way.
    cn = cl.numpy().reshape(nkc, sc)[0]
    max_c = int(cn.max()) if sc > 0 else 0
    sum_c = int(cn.sum(dtype="int64"))
    key_max = [int(v) for v in ul.numpy().reshape(nku, su).max(axis=1)]
    return (sc, max_c, sum_c), (su, key_max)


def _fwd_lpk_on() -> bool:
    # A/B switch (iter 15): DLRM_HSTU_FP8_SYNCFREE_FWDLPK=0 restores the device
    # read of the uih/candidate length_per_key in the forward preprocess.
    return os.environ.get(_ENV + "_FWDLPK", "1") == "1"


def seed_host_length_per_key(*kjts: Any) -> None:
    """Cache length_per_key on CPU KJTs before the H2D copy (iter 15).

    ``KeyedJaggedTensor.to`` carries ``_length_per_key`` to the device copy, so
    the forward's ``kjt[key]`` (offset_per_key -> length_per_key) needs no
    device-to-host read. At 2n that pageable 120 B read was where the host
    drained every step once the splits path stopped draining (iter15 trace
    r1). Same integers: a per-key sum of the same CPU lengths.
    """
    if not _fwd_lpk_on():
        return
    for kjt in kjts:
        try:
            if getattr(kjt, "_length_per_key", None) is not None:
                continue
            g = _fixed_stride_rows(kjt)
            if g is None or kjt.lengths().device.type != "cpu":
                continue
            nk, stride = g
            a = kjt.lengths().numpy().reshape(nk, stride).sum(axis=1, dtype="int64")
            kjt._length_per_key = [int(v) for v in a]
        except Exception:  # pragma: no cover - never fatal, stock path
            continue


def verify_host_length_per_key(*kjts: Any) -> None:
    """VERIFY mode: compare the carried host length_per_key with the device."""
    if not _fwd_lpk_on():
        return
    for kjt in kjts:
        host = getattr(kjt, "_length_per_key", None)
        g = _fixed_stride_rows(kjt)
        if host is None or g is None or not _verify_budget("fwdlpk"):
            continue
        nk, _ = g
        dev = [int(v) for v in kjt.lengths().view(nk, -1).sum(1).tolist()]
        if dev != list(host):
            raise RuntimeError(
                f"host-sync-free forward length_per_key mismatch: host={host} device={dev}"
            )
        logger.info("DLRMV4 host-sync-free VERIFY fwdlpk OK %s", host)
    _announce("forward length_per_key")


def tag_device_lengths(uih_kjt: Any, cand_kjt: Any, stats: Any) -> None:
    """Attach the CPU-computed statistics to the device lengths tensors."""
    if stats is None:
        return
    cand_stats, uih_stats = stats
    try:
        setattr(cand_kjt.lengths(), _HOST_CAND, cand_stats)
        setattr(uih_kjt.lengths(), _HOST_UIH, uih_stats)
    except Exception:  # pragma: no cover - never fatal, just falls back
        return


def _row0_of_tagged(t: torch.Tensor, attr: str) -> Optional[Any]:
    base = t._base
    if base is None:
        return None
    tag = getattr(base, attr, None)
    if tag is None:
        return None
    stride = tag[0]
    if t.dim() != 1 or t.numel() != stride:
        return None
    if t.storage_offset() != base.storage_offset():
        return None
    return tag


def host_max_candidates(num_candidates: torch.Tensor) -> Optional[int]:
    """Host-known max(num_candidates) for row 0 of a tagged candidates lengths."""
    if not enabled():
        return None
    tag = _row0_of_tagged(num_candidates, _HOST_CAND)
    if tag is None:
        return None
    return tag[1]


def host_length_stats(
    num_candidates: torch.Tensor, uih_seq_lengths: torch.Tensor
) -> Optional[Tuple[int, int, int]]:
    """(max_c, max_u, sum_c) from tags, or None (caller does the device read)."""
    if not enabled():
        return None
    ctag = _row0_of_tagged(num_candidates, _HOST_CAND)
    if ctag is None:
        return None
    ubase = uih_seq_lengths._base
    if ubase is None or uih_seq_lengths.dim() != 1:
        return None
    utag = getattr(ubase, _HOST_UIH, None)
    if utag is None:
        return None
    ustride, key_max = utag
    if uih_seq_lengths.numel() != ustride:
        return None
    off = uih_seq_lengths.storage_offset() - ubase.storage_offset()
    if off < 0 or off % ustride != 0:
        return None
    k = off // ustride
    if k >= len(key_max):
        return None
    out = (ctag[1], key_max[k], ctag[2])
    if _verify_budget("len"):
        dev = torch.stack(
            [
                num_candidates.max().to(torch.int64),
                uih_seq_lengths.max().to(torch.int64),
                num_candidates.sum().to(torch.int64),
            ]
        ).tolist()
        if tuple(dev) != out:
            raise RuntimeError(
                f"host-sync-free length stats mismatch: host={out} device={tuple(dev)}"
            )
        logger.info("DLRMV4 host-sync-free VERIFY len OK %s", out)
    _announce("fx_length_stats")
    return out


# zero-fills (ops/zero_fold.py): torchrec's gauc.to_3d is asynchronous_complete_cumsum(seq_lengths) (a
# hipMemsetAsync for offsets[0] + a scan) then jagged_2d_to_dense; get_auc_states calls it three times with the same
# num_candidates. One offsets tensor serves all three (the same int64 values). The first call also runs gauc.to_3d
# and compares raw bits (one sync); a mismatch or error -> gauc.to_3d for the rest of the process.
_GAUC_OFFSETS_STATE: Dict[str, Any] = {"checked": False, "off": False}


def _gauc_to_3d(gauc: Any, num_candidates: torch.Tensor, max_length: int) -> Any:
    from generative_recommenders.ops.zero_fold import enabled as _zero_fold_on

    st = _GAUC_OFFSETS_STATE
    if st["off"] or not _zero_fold_on():
        return lambda t: gauc.to_3d(t, num_candidates, max_length)
    try:
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(num_candidates)

        def to_3d(t: torch.Tensor) -> torch.Tensor:
            return torch.ops.fbgemm.jagged_2d_to_dense(t, offsets, max_length)

    except Exception as exc:  # fail closed
        st["off"] = True
        logger.warning("DLRMV4 zero-fills gauc offsets reuse FAILED CLOSED -> gauc.to_3d (%s)", exc)
        return lambda t: gauc.to_3d(t, num_candidates, max_length)
    if not st["checked"]:
        st["checked"] = True

        def checked(t: torch.Tensor) -> torch.Tensor:
            got, ref = to_3d(t), gauc.to_3d(t, num_candidates, max_length)
            if got.dtype != ref.dtype or got.shape != ref.shape or not torch.equal(
                got.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)
            ):
                st["off"] = True
                logger.warning("DLRMV4 zero-fills gauc offsets reuse: raw-bit MISMATCH vs gauc.to_3d -> gauc.to_3d")
                return ref
            return got

        logger.info(
            "DLRMV4 armed OK: zero-fills gauc offsets reuse -- one complete_cumsum for the three to_3d calls "
            "(first call raw-bit checked vs gauc.to_3d; kill switch DLRM_HSTU_FP8_ZERO_FOLD=0)"
        )
        return checked
    return to_3d


def install_gauc_patch() -> None:
    """Route GAUC's max_length through the host tag (exact same value)."""
    import torchrec.metrics.gauc as gauc

    if getattr(gauc, "_dlrm_hostlen_patched", False):
        return
    orig = gauc.get_auc_states

    @torch.compiler.disable
    def get_auc_states(labels, predictions, weights, num_candidates):
        max_length = host_max_candidates(num_candidates)
        if max_length is None:
            return orig(labels, predictions, weights, num_candidates)
        if _verify_budget("gauc"):
            dev = int(num_candidates.max().item())
            if dev != max_length:
                raise RuntimeError(
                    f"host-sync-free GAUC max_length mismatch: host={max_length} device={dev}"
                )
            logger.info("DLRMV4 host-sync-free VERIFY gauc OK %d", max_length)
        _announce("gauc max_length")
        predictions_perm = predictions.permute(1, 0)
        labels_perm = labels.permute(1, 0)
        weights_perm = weights.permute(1, 0)
        to_3d = _gauc_to_3d(gauc, num_candidates, max_length)
        predictions_3d = to_3d(predictions_perm).permute(2, 0, 1)
        labels_3d = to_3d(labels_perm).permute(2, 0, 1)
        weights_3d = to_3d(weights_perm).permute(2, 0, 1)
        return gauc.compute_gauc_3d(predictions_3d, labels_3d, weights_3d)

    gauc.get_auc_states = get_auc_states
    gauc._dlrm_hostlen_patched = True


class PendingScalar:
    """A device scalar copied to pinned host memory without blocking."""

    def __init__(self, value: torch.Tensor) -> None:
        self._dev = value.detach()
        if self._dev.is_cuda:
            self._host = self._dev.to("cpu", non_blocking=True)
            self._event = torch.cuda.Event()
            self._event.record()
        else:
            self._host = self._dev
            self._event = None

    def get(self) -> float:
        if self._event is not None:
            self._event.synchronize()
        return float(self._host)


class StepFence:
    """Keep the host at most ``lag`` steps ahead of the GPU without draining.

    ``step()`` records an event on the current stream and waits for the one
    recorded ``lag`` calls ago, so the queue always still holds the newest
    ``lag`` steps of work.
    """

    def __init__(self, lag: int = 1) -> None:
        self._lag = max(1, int(lag))
        self._events: List[Any] = []

    def step(self) -> bool:
        """Record this step's end; wait for the one ``lag`` steps back.

        Returns True once there was an earlier step to wait on (every call
        after the first ``lag``), False before that.
        """
        if not torch.cuda.is_available():
            return True
        ev = torch.cuda.Event()
        ev.record()
        self._events.append(ev)
        waited = False
        while len(self._events) > self._lag:
            self._events.pop(0).synchronize()
            waited = True
        return waited

    def drain(self) -> None:
        while self._events:
            self._events.pop(0).synchronize()


# ---------------------------------------------------------------------------
# Host-known length_per_key for the KJT input_dist all-to-all (iter 12).
#
# Trace finding (probes/iter12, operator 2n ABBA job 54372): with the three
# per-step reads gone, the drain moved into the NEXT batch's input_dist:
# KJTAllToAllTensorsAwaitable.wait() -> KeyedJaggedTensor.dist_init() ->
# kjt.sync() -> length_per_key() does lengths.view(keys, stride).sum(1).tolist(),
# a 112 B pageable D2H that waits for the KJT values all_to_all, which sits in
# FIFO order on the NCCL stream behind the current step's DDP allreduces and
# Seq_bwd all_to_all. The host therefore still drained every step (129-140 ms,
# 96/96 rank-steps at 2n; 0.75/step at 1n), and while it was blocked in that
# copy the small backward kernels ran ~20-50 us longer each (+11.8 ms/step,
# 48/48 pairs, both nodes).
#
# Fix (same values, same collectives): every sender already knows its per-key
# value counts on the host (dist_splits needs them), so K extra rows are added
# to the splits all_to_all that start_sparse_data_dist already issues and
# reads back. Row j carries, for each destination rank r, the value count of
# the j-th key that r receives. The receiver sums row j over sources to get the
# length_per_key of its post-all_to_all KJT and builds it with that list, so
# sync() reads nothing from the device. Host check every call:
# sum(length_per_key) == values.numel() (both host ints). VERIFY mode also
# compares against the device reduction.
# ---------------------------------------------------------------------------

_LPK_ATTR = "_dlrm_host_lpk"
_lpk_pending: List[Any] = []


def _lpk_rows(input_kjt: Any, splits: List[int]) -> Optional[List[List[int]]]:
    """K rows of len(splits): row j[r] = value count of key offset_r + j."""
    if input_kjt.variable_stride_per_key():
        return None
    lpk = input_kjt.length_per_key()  # cached by dist_splits; host list
    k = max(splits) if splits else 0
    rows = [[0] * len(splits) for _ in range(k)]
    off = 0
    for r, n in enumerate(splits):
        for j in range(n):
            rows[j][r] = int(lpk[off + j])
        off += n
    return rows


def _lpk_on() -> bool:
    # Sub-switch for A/B: DLRM_HSTU_FP8_SYNCFREE_LPK=0 restores the device read.
    return os.environ.get(_ENV + "_LPK", "1") == "1"


_b1pp_mod: Any = None


def _b1pp() -> Any:
    """dlrm_v4/b1pp.py (comm iter 46), imported at the first dist_init that needs it."""
    global _b1pp_mod
    if _b1pp_mod is None:
        from generative_recommenders.dlrm_v4 import b1pp as _mod

        _b1pp_mod = _mod
    return _b1pp_mod


def install_kjt_lpk_patch() -> None:
    """Patch torchrec's KJT all_to_all to carry host length_per_key (gated)."""
    import torchrec.distributed.dist_data as dd
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    if getattr(dd, "_dlrm_lpk_patched", False):
        return

    split_init = dd.SplitsAllToAllAwaitable.__init__
    split_wait = dd.SplitsAllToAllAwaitable._wait_impl
    kjts_init = dd.KJTAllToAllSplitsAwaitable.__init__
    kjts_wait = dd.KJTAllToAllSplitsAwaitable._wait_impl
    kjtt_wait = dd.KJTAllToAllTensorsAwaitable._wait_impl

    def p_split_init(self, input_tensors, pg, *args, **kwargs):
        extra = _lpk_pending.pop() if _lpk_pending else None
        self._dlrm_lpk_n = 0
        if extra and input_tensors:
            dev = input_tensors[0].device
            dt = input_tensors[0].dtype
            input_tensors = list(input_tensors) + [
                torch.tensor(row, device=dev, dtype=dt) for row in extra
            ]
            self._dlrm_lpk_n = len(extra)
        split_init(self, input_tensors, pg, *args, **kwargs)

    def p_split_wait(self):
        work = getattr(self, "_splits_awaitable", None)
        if _split_record_stream_on():
            # The splits all_to_all's input/output buffers come from the
            # data-dist stream's pool, like the lengths/values buffers that
            # p_kjtt_wait already protects. With dist_init on main, the next
            # batch's splits input (staged on data-dist a step ahead) could
            # reuse this input block while NCCL is still sending it to a
            # cross-node peer, which then receives a stale/garbage row
            # (4-node `_get_recat` int32 overflow, always from an inter-node
            # source). Mark both in use on the current stream before the wait,
            # so their reuse is ordered after this stream's wait on the
            # collective.
            cur = torch.cuda.current_stream()
            for t in (getattr(self, "_input_tensor", None), getattr(self, "_output_tensor", None)):
                if t is not None and t.is_cuda:
                    t.record_stream(cur)
            _announce_split_record_stream()
        if split_wait_poll_on() and work is not None and hasattr(work, "is_completed"):
            # split_wait_poll_on(): issue the stream wait + .tolist() copy only
            # once the splits all_to_all has finished, so the high-priority
            # data-dist stream never holds a blocked barrier.
            with torch.autograd.profiler.record_function("## splits poll ##"):
                _poll_until_done(work)
        ret = split_wait(self)
        n = getattr(self, "_dlrm_lpk_n", 0)
        if n:
            self._dlrm_lpk_rows = ret[-n:]
            ret = ret[:-n]
        return ret

    def p_kjts_init(self, pg, input, splits, *args, **kwargs):
        rows = None
        if enabled() and _lpk_on() and pg.size() > 1:
            rows = _lpk_rows(input, list(splits))
        _lpk_pending.clear()
        if rows:
            _lpk_pending.append(rows)
        try:
            kjts_init(self, pg, input, splits, *args, **kwargs)
        finally:
            _lpk_pending.clear()
        self._dlrm_lpk_k = len(rows) if rows else 0

    def p_kjts_wait(self):
        out = kjts_wait(self)
        k = getattr(self, "_dlrm_lpk_k", 0)
        sa = getattr(self, "_splits_awaitable", None)
        src = getattr(sa, "_dlrm_lpk_rows", None)
        if k and src is not None:
            mine = self._splits[self._pg.rank()]
            # row j, column s: count of my j-th key from source s
            setattr(out, _LPK_ATTR, [int(sum(src[j])) for j in range(mine)])
        return out

    def p_kjtt_wait(self):
        if dist_init_on_main():
            # This wait now runs on the main stream, but every buffer of the
            # lengths/values all_to_all lives in the data-dist stream's
            # allocator pool (dist_data.py KJTAllToAllTensorsAwaitable):
            # - outputs: read here by the recat permute;
            # - inputs: c10d releases its stash of them in work.wait(), which
            #   assumes the waiting stream is the one they were allocated on
            #   (on the default path data-dist itself waits, so its next
            #   allocations are ordered after the NCCL send).
            # With the wait on main, the next batch's input dist (issued on
            # data-dist right after the forward, host a step ahead) can reuse
            # either set while NCCL is still sending or before the permute has
            # read it. iter 18/19 U arms died of that (GPU memory-access fault
            # in every 1n/2n/4n run). Mark both in use on the current stream
            # BEFORE work.wait(), so the allocator's reuse event is recorded
            # after main's wait on the collective (and after the permute).
            cur = torch.cuda.current_stream()
            extra = [self._recat] if self._recat is not None else []
            for t in list(self._output_tensors) + list(self._input_tensors or []) + extra:
                if t is not None and t.is_cuda:
                    t.record_stream(cur)
        lpk = getattr(self, _LPK_ATTR, None)
        if (
            lpk is None
            or self._workers == 1
            or self._recat is None
            or self._stride_per_rank is None
            or self._input.variable_stride_per_key()
            or len(lpk) != len(self._keys)
        ):
            if self._workers > 1 and enabled() and _lpk_on():
                _announce(
                    "kjt length_per_key FALLBACK (device read) keys=%d lpk=%s recat=%s"
                    % (len(self._keys), "none" if lpk is None else len(lpk),
                       self._recat is not None)
                )
            if self._workers > 1 and _b1pp().recat_on():
                _b1pp().log_fallback(
                    "recat", "host-sync-free length_per_key path not taken (torchrec dist_init)"
                )
            return kjtt_wait(self)
        for aw in self._awaitables:
            aw.wait()
        tensors = self._output_tensors
        lengths, values = tensors[0], tensors[1]
        weights = tensors[2] if len(tensors) == 3 else None
        if sum(lpk) != values.numel():
            raise RuntimeError(
                f"host length_per_key {lpk} sums to {sum(lpk)} != values {values.numel()}"
            )
        spr = self._stride_per_rank
        stride = spr[0]
        offsets = None  # None: torchrec computes the KJT offsets when asked (stock)
        with torch.autograd.profiler.record_function("## dist_init:recat_values ##"):
            if all(s == stride for s in spr):
                if stride > 0:
                    # B1'' part (B), dlrm_v4/b1pp.py (DLRM_HSTU_FP8_KJT_RECAT_FUSED,
                    # absent = off): two Triton kernels give the same permuted
                    # lengths/values plus the KJT offsets. When the TBE lookup
                    # takes the whole KJT (one sequence TBE group: 2n/4n RW) its
                    # offsets cumsum then has nothing to do; KJT.split children
                    # (1n, several groups) recompute theirs as stock. None = not
                    # engaged: stock.
                    fused = _b1pp().kjt_recat(self._recat, lengths, values, weights, stride)
                    if fused is not None:
                        lengths, values, offsets = fused
                    else:
                        lengths, values, weights = torch.ops.fbgemm.permute_2D_sparse_data(
                            self._recat, lengths.view(-1, stride), values, weights,
                            values.numel(),
                        )
                    lengths = lengths.view(-1)
                elif _b1pp().recat_on():
                    _b1pp().log_fallback("recat", "stride 0 (empty batch: no permute in either path)")
            else:
                if _b1pp().recat_on():
                    _b1pp().log_fallback(
                        "recat", "unequal stride per rank (stock permute_1D_sparse_data)",
                        "strides=%s", spr,
                    )
                lengths, values, weights = torch.ops.fbgemm.permute_1D_sparse_data(
                    self._recat, lengths.view(-1), values, weights, values.numel(),
                )
        kjt = KeyedJaggedTensor(
            keys=self._keys,
            values=values,
            weights=weights,
            lengths=lengths,
            offsets=offsets,
            stride=sum(spr),
            length_per_key=list(lpk),
        )
        if _verify_budget("lpk"):
            dev = (
                [int(v) for v in lengths.view(len(self._keys), -1).sum(1).tolist()]
                if lengths.numel()
                else [0] * len(self._keys)
            )
            if dev != list(lpk):
                raise RuntimeError(
                    f"host-sync-free length_per_key mismatch: host={lpk} device={dev}"
                )
            logger.info("DLRMV4 host-sync-free VERIFY lpk OK %s", lpk)
        _announce("kjt length_per_key")
        return kjt.sync()

    # ---- Fused splits path (iter 14). ------------------------------------
    # The train pipeline's start_sparse_data_dist does NOT build a
    # KJTAllToAllSplitsAwaitable per KJT: runtime_forwards.KJTAllToAllForward
    # returns a KJTSplitsAllToAllMeta and FusedKJTListSplitsAwaitable issues ONE
    # splits all_to_all for all of them, then constructs the
    # KJTAllToAllTensorsAwaitable directly. The hooks above never saw it, so at
    # 2n (job 54719 traces: splits a2a 48 elems in 320/320, aten::sum [14,16384]
    # + 112 B pageable D2H every step) the host still drained. Same rows, same
    # check, on this path: append K rows to each meta's splits_tensors (K is
    # max(splits), rank-invariant), strip them per meta after the fused wait,
    # and hand the summed list to the tensors awaitable built for that input.
    import torchrec.distributed.embedding_sharding as es

    fused_init = es.FusedKJTListSplitsAwaitable.__init__
    fused_wait = es.FusedKJTListSplitsAwaitable._wait_impl
    kjtt_init = dd.KJTAllToAllTensorsAwaitable.__init__
    Meta = es.KJTSplitsAllToAllMeta
    _POST = "_dlrm_lpk_post"

    class _Done:
        def __init__(self, v: Any) -> None:
            self._v = v

        def wait(self) -> Any:
            return self._v

    def p_fused_init(self, requests, contexts, pg):
        ks = []
        host_vals: Dict[int, List[int]] = {}
        for req in requests:
            for aw in req.awaitables:
                k = 0
                if (
                    isinstance(aw, Meta)
                    and enabled()
                    and _lpk_on()
                    and aw.pg is not None
                    and aw.pg.size() > 1
                    and aw.splits_tensors
                ):
                    rows = _lpk_rows(aw._input, list(aw.splits))
                    if rows:
                        dev = aw.splits_tensors[0].device
                        dt = aw.splits_tensors[0].dtype
                        # ONE pinned, non-blocking H2D for all K rows: a per-row
                        # torch.tensor(..., device=) would issue K pageable
                        # synchronous copies per step.
                        blk = torch.tensor(rows, dtype=dt).pin_memory()
                        blk = blk.to(dev, non_blocking=True)
                        extra = list(blk.unbind(0))
                        for t, r in zip(extra, rows):
                            host_vals[id(t)] = r
                        aw.splits_tensors = list(aw.splits_tensors) + extra
                        k = len(rows)
                ks.append(k)
        # The stock init runs its debug overflow check, _safe_tolist (a D2H plus
        # a stream sync), on EVERY splits tensor. For the appended rows the values
        # are already on the host, so return them without a device read.
        prev = es._safe_tolist
        if host_vals:
            def _tolist(t):
                v = host_vals.get(id(t))
                return list(v) if v is not None else prev(t)

            es._safe_tolist = _tolist
        try:
            fused_init(self, requests, contexts, pg)
        finally:
            es._safe_tolist = prev
        self._dlrm_lpk_ks = ks

    def p_fused_wait(self):
        ks = getattr(self, "_dlrm_lpk_ks", None)
        if not ks or not any(ks) or not self._splits_awaitable:
            return fused_wait(self)
        full = self._splits_awaitable.wait()
        new: List[Any] = []
        lens: List[int] = []
        off = 0
        for aw, n, k in zip(self._awaitables, self._lengths, ks):
            ch = full[off:off + n]
            off += n
            if k:
                rows, ch = ch[-k:], ch[:-k]
                mine = aw.splits[aw.pg.rank()]
                lpk = [int(sum(rows[j])) for j in range(mine)]
                try:
                    setattr(aw._input, _POST, lpk)
                except Exception:  # noqa: BLE001 -- slotted KJT: stock path
                    pass
            new.extend(ch)
            lens.append(len(ch))
        self._splits_awaitable = _Done(new)
        self._lengths = lens
        _announce("kjt length_per_key (fused splits)")
        return fused_wait(self)

    def p_kjtt_init(self, *args, **kwargs):
        kjtt_init(self, *args, **kwargs)
        inp = getattr(self, "_input", None)
        lpk = getattr(inp, _POST, None)
        if lpk is not None:
            try:
                delattr(inp, _POST)
            except Exception:  # noqa: BLE001
                pass
            setattr(self, _LPK_ATTR, lpk)

    es.FusedKJTListSplitsAwaitable.__init__ = p_fused_init
    es.FusedKJTListSplitsAwaitable._wait_impl = p_fused_wait
    dd.KJTAllToAllTensorsAwaitable.__init__ = p_kjtt_init

    dd.SplitsAllToAllAwaitable.__init__ = p_split_init
    dd.SplitsAllToAllAwaitable._wait_impl = p_split_wait
    dd.KJTAllToAllSplitsAwaitable.__init__ = p_kjts_init
    dd.KJTAllToAllSplitsAwaitable._wait_impl = p_kjts_wait
    dd.KJTAllToAllTensorsAwaitable._wait_impl = p_kjtt_wait
    dd._dlrm_lpk_patched = True
