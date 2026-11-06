# Copyright (c) Meta Platforms, Inc. and affiliates.
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

# pyre-strict
import collections
import copy
import gc
import inspect
import logging
import math
import os
import threading
import time
from collections.abc import Iterator
from datetime import timedelta
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import gin
import torch
import torchrec
from generative_recommenders.dlrm_v4.checkpoint import save_dmp_checkpoint, WINDOW_COMPLETE
from generative_recommenders.dlrm_v4.configs import (
    get_embedding_table_config,
    get_hstu_configs,
)
from generative_recommenders.dlrm_v4.datasets.dataset import (
    collate_fn,
    configure_fast_batch,
    configure_fast_batch_onestore,
    configure_pin_free_shm,
    Dataset,
    eval_fast_batch,
    fast_batch_enabled,
    fast_batch_knob_source,
    fast_batch_onestore_knob_source,
    fast_batch_onestore_level,
    fast_batch_scope,
    pin_free_shm_enabled,
    pin_free_shm_knob_source,
    Samples,
)
from generative_recommenders.dlrm_v4.utils import get_dataset, MetricsLogger, Profiler
from generative_recommenders.dlrm_v4 import host_sync_free as _hsf
from generative_recommenders.dlrm_v4 import alloc_pregrow as _alloc_pregrow
from generative_recommenders.dlrm_v4.train import cold_prefetch as _cold_prefetch
from generative_recommenders.dlrm_v4.train import dd_own_h2d as _dd_own_h2d
from generative_recommenders.dlrm_v4.train import dl_prestart as _dlstart
from generative_recommenders.dlrm_v4.train import m4_partition as _m4p
from generative_recommenders.dlrm_v4.train.run_knobs import (
    eval_pool_log_line,
    metric_cadence_log_line,
    metric_log_due,
    metric_logging_enabled,
    resolve_eval_dataloader_kwargs,
    mlperf_train_loss_log_line,
    resolve_mlperf_loss_every,
)
# Imported EAGERLY, and it must stay that way: `enable_bf16_ec_output`,
# `configure_ec_bf16_output` and `enable_segment_reduce` are `@gin.configurable`,
# and gin only resolves a binding for a configurable that is REGISTERED when it
# parses.  train_ranker.py runs its second, strict parse (no `skip_unknown`)
# before it calls `make_model`, so a lazy import here would make every
# `configure_ec_bf16_output.* = ...` / `enable_segment_reduce.* = ...` line in
# the gin die with "No configurable matching".  This module is cheap to import
# (logging, typing, gin, torch); the triton-carrying `ec_segment_reduce` is the
# one that stays lazy, and its knobs are forwarded through
# `enable_segment_reduce` for exactly this reason.
from generative_recommenders.dlrm_v4.train.ec_bf16_output import (
    enable_bf16_ec_output,
    enable_ec_cat_fuse,
    enable_ec_merge_uih_candidate,
    enable_segment_reduce,
    note_ec_indexed_l1_chain,
)
from generative_recommenders.common import HammerKernel
from generative_recommenders.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig
from generative_recommenders.ops.flydsl import DEFAULT_BWD_BACKEND
from torch import distributed as dist
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data.distributed import _T_co, DistributedSampler
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.sharding_plan import get_default_sharders
from torchrec.distributed.types import ShardedTensor, ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

TORCHREC_TYPES: Set[Type[Union[EmbeddingBagCollection, EmbeddingCollection]]] = {
    EmbeddingBagCollection,
    EmbeddingCollection,
}

# Embedding placement vocabulary -> torchrec compute kernel. Used by
# make_optimizer_and_shard to translate the gin/env placement strings
# ("hbm"/"uvm"/"uvm_caching") into ParameterConstraints. "auto" (or anything not
# in this map) means "no constraint": the planner decides from the HBM cap.
_PLACEMENT_TO_KERNEL: Dict[str, EmbeddingComputeKernel] = {
    "hbm": EmbeddingComputeKernel.FUSED,
    "uvm": EmbeddingComputeKernel.FUSED_UVM,
    "uvm_caching": EmbeddingComputeKernel.FUSED_UVM_CACHING,
}

# Per-table sharding-type vocabulary -> torchrec ShardingType. Used by
# make_optimizer_and_shard to pin a table's shard layout via ParameterConstraints
# (e.g. move a hot, data-skewed table off ROW_WISE to COLUMN_WISE so its
# embedding all-to-all load is balanced by rank instead of routed by row/value).
# "auto" (or anything not in this map) means "no constraint": the planner decides.
# Short aliases (rw/cw/tw/twrw) are accepted alongside the canonical names.
_SHARDING_TO_TYPE: Dict[str, ShardingType] = {
    "row_wise": ShardingType.ROW_WISE,
    "column_wise": ShardingType.COLUMN_WISE,
    "table_wise": ShardingType.TABLE_WISE,
    "table_row_wise": ShardingType.TABLE_ROW_WISE,
    "table_column_wise": ShardingType.TABLE_COLUMN_WISE,
    "data_parallel": ShardingType.DATA_PARALLEL,
    "rw": ShardingType.ROW_WISE,
    "cw": ShardingType.COLUMN_WISE,
    "tw": ShardingType.TABLE_WISE,
    "twrw": ShardingType.TABLE_ROW_WISE,
}


@gin.configurable
def seed_everything(seed: int = -1, rank: int = 0) -> None:
    """Seed all RNGs (same value on every rank) for reproducible dense weight init.

    Call right before make_model(), after setup() (process group needed for the
    broadcast) and the gin parse. seed < 0 ($SEED unset) draws a fresh random seed
    per run (rank 0 broadcasts; exported to $SEED); seed >= 0 reproduces a run.
    Data order/split are independent of this seed (StreamingWindowSampler/$SPLIT_SALT).
    """
    import random

    import numpy as np

    pinned = seed >= 0
    if not pinned:
        # rank 0 draws a random seed and broadcasts it so all ranks agree (an
        # identical seed on every rank is REQUIRED, else dense weight init
        # diverges across ranks and DDP/AllReduce trains garbage).
        if dist.is_available() and dist.is_initialized():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            drawn = int.from_bytes(os.urandom(4), "little") if rank == 0 else 0
            _seed_t = torch.tensor([drawn], dtype=torch.int64, device=device)
            dist.broadcast(_seed_t, src=0)
            seed = int(_seed_t.item())
        else:
            seed = int.from_bytes(os.urandom(4), "little")
    # Export the resolved value so the run is reproducible after the fact.
    os.environ["SEED"] = str(seed)

    logger.info(
        f"[rank {rank}] seeding all RNGs with SEED={seed} "
        f"({'pinned via $SEED' if pinned else 'random per-run; set $SEED to reproduce'})"
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@gin.configurable
def decorrelate_runtime_rng(rank: int = 0, enabled: bool = True) -> None:
    """Re-seed torch/cuda with $SEED + rank so HSTU dropout draws different masks
    per data-parallel rank (seed_everything's identical seed would draw the same).

    MUST run after make_model() + make_optimizer_and_shard() so init stays
    identical across ranks; it perturbs only forward-time stochasticity.
    Reproducible (pure fn of $SEED + rank; RNG state checkpointed). enabled=False
    keeps the legacy identical-mask behavior.
    """
    if not enabled:
        logger.info(
            f"[rank {rank}] decorrelate_runtime_rng disabled; dropout masks "
            f"identical across ranks"
        )
        return
    base = int(os.environ.get("SEED", "1"))
    offset_seed = base + int(rank)
    torch.manual_seed(offset_seed)
    torch.cuda.manual_seed_all(offset_seed)
    logger.info(
        f"[rank {rank}] decorrelated runtime RNG: SEED={base} + rank={rank} "
        f"=> {offset_seed} (per-rank dropout masks)"
    )


def setup(
    rank: int,
    world_size: int,
    master_port: int,
    device: torch.device,
    master_addr: str = "localhost",
) -> dist.ProcessGroup:
    # Default "localhost" keeps the single-node path unchanged; multi-node
    # launches pass the rank-0 host so every node rendezvouses at the same addr.
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    BACKEND = dist.Backend.NCCL
    # Process-group / NCCL watchdog timeout (seconds). Env-overridable so a
    # diagnostic run can use a short, finite timeout that trips the NCCL flight
    # recorder dump (TORCH_NCCL_TRACE_BUFFER_SIZE + TORCH_NCCL_DUMP_ON_TIMEOUT)
    # on a collective desync instead of hanging for the full default.
    TIMEOUT = int(os.environ.get("PG_TIMEOUT_S", "1800"))

    # set device BEFORE init_process_group so NCCL binds this rank to its
    # own GPU; otherwise every rank's first CUDA context lands on GPU 0,
    # leaving stale allocations and triggering OOMs on rank 0.
    torch.cuda.set_device(device)

    # NOTE: RNG seeding for reproducible weight init lives in seed_everything(),
    # which train_ranker calls right before make_model() (after setup() so the
    # process group is initialized for the cross-rank seed broadcast, and after
    # the full gin parse so the gin-configurable $SEED is bound). Seeding here
    # would be too early to be gin-configurable.

    # RCCL reads its settings when the first communicator is created, so the
    # fabric guard (no GDR flush on ionic NICs with RCCL's built-in transport
    # corrupts inter-node data, plans/dlrmv4 09) must run before it.
    from generative_recommenders.dlrm_v4 import fabric_guard

    fabric_guard.apply(rank)

    # initialize the process group
    #
    # The default PG timeout must match TIMEOUT (not the 600s NCCL default):
    # checkpoint saves go through DCP collectives on *this* default PG, and the
    # 560GB sparse-embedding write is both slow on shared NFS and badly skewed
    # across ranks (shards range ~37GB..~95GB), so the fastest rank can sit in
    # the post-write allgather/barrier well past 600s waiting for the slowest
    # rank. The stock 600s watchdog then SIGABRTs an otherwise-healthy job.
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            device_id=device,
            timeout=timedelta(seconds=TIMEOUT),
        )

    pg = dist.new_group(
        backend=BACKEND,
        timeout=timedelta(seconds=TIMEOUT),
    )

    # Which ROCr IPC mode the communicators were connected with (DMA-BUF default
    # vs legacy KFD IPC, HSA_ENABLE_IPC_MODE_LEGACY; launch_slurm.sh): one line
    # per node, read once here, before RUN_START (dlrm_v4/train/ipc_mode.py).
    try:
        from generative_recommenders.dlrm_v4.train import ipc_mode as _ipc_mode

        _ipc_mode.log_rocr_ipc_mode(
            rank=rank,
            local_rank=device.index if device.index is not None else rank,
            world_size=world_size,
        )
    except Exception:
        pass

    return pg


def cleanup() -> None:
    dist.destroy_process_group()


def _collect_and_freeze_gc() -> None:
    """Run the one-time full Python GC now and freeze its survivors.

    A gen-2 collection walks every tracked object (~300-450 ms on a rank). Each
    rank otherwise runs its first one wherever its own promotion count crosses
    CPython's 25% threshold, at a different step per rank; every one of those
    pauses stalls the other ranks at the next collective. Collecting here, before
    the first window's rendezvous, puts that pause on all ranks at once, and
    gc.freeze() moves the setup-time objects out of later collections' scans.
    Frozen objects are still freed by reference counting.

    Gated by $DLRM_HSTU_FP8_GC_FREEZE (gin: apply_env_bootstrap.
    DLRM_HSTU_FP8_GC_FREEZE; default "1" when absent). Not an FP8 knob: the
    prefix is only what launch_slurm.sh forwards into the container. With "0"
    nothing is collected or frozen and CPython's own GC schedule applies (the
    pre-fix behaviour), so the stall fix can be A/B'd per run.
    """
    _knob = os.environ.get("DLRM_HSTU_FP8_GC_FREEZE", "1")
    if _knob.strip().lower() in ("0", "false", "off", "no"):
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            logger.info(
                "gc_freeze OFF (DLRM_HSTU_FP8_GC_FREEZE=%s) -- no full GC or "
                "gc.freeze() before the first window rendezvous",
                _knob,
            )
        return
    gc.collect()
    gc.freeze()
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        logger.info(
            "armed OK: gc_freeze ACTIVE -- full GC run before the first window "
            "rendezvous, %d objects frozen",
            gc.get_freeze_count(),
        )


def _traced_window_boundary(
    profiler: Optional[Profiler],
    device: torch.device,
    world_size: int,
    train_ts: int,
) -> None:
    """``_window_boundary_barrier``, labelled in the trace when tracing.

    Untraced (``profiler is None``) this is exactly the barrier. Traced, the
    event is noted on the profiler, and while a window is recording the
    barrier is wrapped in a ``## window_boundary ##`` range: under the
    host-sync-free step the barrier drains the run-ahead queue, so a capture
    step containing it is not a steady-state step. Outside a recording window
    the barrier runs unlabelled.
    """
    # M4 speed estimator boundary after the barrier (no-op unless DLRM_HSTU_FP8_M4=1)
    from generative_recommenders.dlrm_v4.train import m4_speed as _m4_speed

    if profiler is None:
        _window_boundary_barrier(device, world_size, train_ts)
        _m4_speed.on_window_boundary(device)
        return
    profiler.note_event("window_boundary", train_ts=train_ts)
    if not profiler.is_recording():
        _window_boundary_barrier(device, world_size, train_ts)
        _m4_speed.on_window_boundary(device)
        return
    with torch.profiler.record_function("## window_boundary ##"):
        _window_boundary_barrier(device, world_size, train_ts)
        _m4_speed.on_window_boundary(device)


def _window_boundary_barrier(
    device: torch.device, world_size: int, train_ts: int
) -> None:
    """Collective rendezvous at a streaming window boundary.

    The per-window data prep (``window_indices``: an O(N) mask over the ~18 GB
    mmap'd ``anchor_ts`` array) can complete at very different times across
    ranks. The embedding input-dist all-to-all that follows is a collective, so
    if a fast rank reaches it while a slow rank is still in prep, the NCCL
    stream desyncs and the job deadlocks (one rank a collective behind the
    rest). Synchronizing here makes prep-time skew harmless: every rank waits
    until all ranks have a ready window before any issues the first forward.

    Cost is one near-zero-payload barrier per window (299 total over a full
    run). In the healthy case prep already overlapped the previous window's
    compute via the prefetcher, so the barrier returns immediately; it only
    blocks for the real prep skew it is there to absorb.
    """
    if not (dist.is_available() and dist.is_initialized()) or world_size <= 1:
        return
    t0 = time.time()
    if device.type == "cuda":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()
    waited = time.time() - t0
    # Surface non-trivial skew (the thing this barrier exists to absorb) so a
    # node with a slow rank is visible without trawling the flight recorder.
    if waited > 5.0:
        logger.warning(
            "[window-barrier] train_ts=%d: waited %.1fs at boundary "
            "rendezvous (per-rank data-prep skew)",
            train_ts,
            waited,
        )
    # Test/debug observability: the healthy-path barrier is otherwise SILENT
    # (the skew warning above only fires on >5s waits), so the resume e2e test
    # has no signal that the boundary rendezvous actually executed. When
    # WINDOW_BARRIER_DEBUG=1, rank 0 emits exactly one line per crossed window
    # so the test can assert the barrier ran at EVERY boundary (regression guard
    # for the desync deadlock the barrier fixes). Off by default — zero prod cost.
    if os.environ.get("WINDOW_BARRIER_DEBUG") == "1" and dist.get_rank() == 0:
        logger.info(
            "[window-barrier] train_ts=%d rendezvous complete (waited %.3fs)",
            train_ts,
            waited,
        )


class HammerToTorchDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        fast_batch: Optional[bool] = None,
    ) -> None:
        self.dataset: Dataset = dataset
        # This wrapper's fast-batch setting: None = $DL_FAST_BATCH (train
        # pools and every non-eval loader); the eval loaders' wrappers carry
        # $DL_FAST_BATCH_EVAL (see _eval_dataset). Set before the pools fork;
        # the workers inherit it.
        self.fast_batch: Optional[bool] = fast_batch

    def __getitem__(self, idx: int) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        with fast_batch_scope(self.fast_batch):
            self.dataset.load_query_samples([idx])
            sample = self.dataset.get_sample(idx)
            self.dataset.unload_query_samples([idx])
        return sample

    def __getitems__(
        self, indices: List[int]
    ) -> Union[List[Tuple[KeyedJaggedTensor, KeyedJaggedTensor]], Samples]:
        if self.fast_batch is None:
            return self._getitems(indices)
        with fast_batch_scope(self.fast_batch):
            return self._getitems(indices)

    def _getitems(
        self, indices: List[int]
    ) -> Union[List[Tuple[KeyedJaggedTensor, KeyedJaggedTensor]], Samples]:
        if fast_batch_enabled():
            # $DL_FAST_BATCH: the dataset builds the collated batch directly
            # (bit-for-bit what collate_fn makes of the per-sample list below);
            # collate_fn passes a Samples through. None = layout not covered.
            build_batch = getattr(self.dataset, "build_batch", None)
            batch = build_batch(indices) if build_batch is not None else None
            if batch is not None:
                self.dataset.unload_query_samples(indices)
                return batch
        self.dataset.load_query_samples(indices)
        samples = [self.dataset.get_sample(i) for i in indices]
        self.dataset.unload_query_samples(indices)
        return samples


class _ReadProbedDataset(HammerToTorchDataset):
    """The same dataset (shares ``.dataset``), plus a one-time worker-side
    first-read wall stamp into a shared slot (see dl_prestart). The slot is
    created in the main process before the pool forks; ``_noted`` is
    per-process (the fork copies False), so after the first fetch the probe
    costs one attribute check per worker fetch."""

    def __init__(
        self, base: HammerToTorchDataset, slot, fast_batch: Optional[bool] = None
    ) -> None:
        super().__init__(base.dataset, fast_batch=fast_batch)
        self._slot = slot
        self._noted: bool = False

    def _note(self) -> None:
        if not self._noted:
            self._noted = True
            _dlstart.note_read(self._slot)

    def __getitem__(self, idx: int) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        self._note()
        return super().__getitem__(idx)

    def __getitems__(
        self, indices: List[int]
    ) -> List[Tuple[KeyedJaggedTensor, KeyedJaggedTensor]]:
        self._note()
        return super().__getitems__(indices)


# yambda_5b.gin item (61) precondition: why the last fast_batch_pools call
# kept the train pools on the per-sample path although the resolved
# DL_FAST_BATCH (its gin default: the variable is not set) selects the fast
# builder, namely that item (37)'s single chained train pool is off; None when
# it did not. Recomputed by every call; _log_fast_batch_pools names it in each
# rank's train-pool line.
_FAST_BATCH_TRAIN_HELD: Optional[str] = None


@gin.configurable
def fast_batch_pools(
    train_pools: Optional[int] = None,
    eval_pools: Optional[int] = None,
) -> None:
    """yambda_5b.gin items (32) and (61): resolve the two fast-batch switches
    for this process and install them (datasets/dataset.py
    configure_fast_batch).

    ``train_pools`` = $DL_FAST_BATCH (item (61), gin dlfb/env_int): the train
    pools and every loader that is not an eval loader. ``eval_pools`` =
    $DL_FAST_BATCH_EVAL (item (32), gin dlfbe/env_int): every eval loader (the
    persistent eval pool and the per-call eval loaders). > 0 = the batched
    builder (DLRMv4YambdaDataset.build_batch), 0 = the per-sample path, and for
    ``eval_pools`` < 0 = follow ``train_pools``. None (no gin binding: tests,
    tools that do not parse yambda_5b.gin) keeps the env-only code defaults
    (DL_FAST_BATCH off; DL_FAST_BATCH_EVAL unset = follow it).

    Item (61) precondition: the batched train builder ships for item (37)'s
    single chained train pool. The check reads item (37)'s own switch,
    window_single_pool(), exactly as the loader does. When ``train_pools``
    selects the builder from the gin default (DL_FAST_BATCH unset or empty in
    the environment) and window_single_pool() is off (DL_SINGLE_POOL <= 0 in
    the environment, its gin setting off, or no item (37) binding: the loader
    then builds the two alternating pools), the train pools keep the
    per-sample path and each rank's "[fast-batch] train pools" line says why.
    A DL_FAST_BATCH set in the environment is applied as given
    (DL_FAST_BATCH=1 still selects the fast builder). Only that switch is
    checked: the loader shapes that never build the single pool although
    window_single_pool() is on (PERSISTENT_LOADER=0 or DOUBLE_BUFFER=0) are
    not held, so their train loaders take the batched builder too.

    Called in the main process before any pool is built
    (make_train_test_dataloaders, prestart_streaming_loaders,
    streaming_train_eval_loop); the forked workers inherit the result.
    Idempotent: the same gin + env give the same values."""
    global _FAST_BATCH_TRAIN_HELD
    _FAST_BATCH_TRAIN_HELD = None
    if (
        train_pools is not None
        and int(train_pools) > 0
        and not os.environ.get("DL_FAST_BATCH", "").strip()
        and not window_single_pool()
    ):
        # The why: DL_SINGLE_POOL as set in the environment (empty = unset,
        # as env_int reads it); when it is unset, the gin config decided.
        single_env = os.environ.get("DL_SINGLE_POOL") or ""
        _FAST_BATCH_TRAIN_HELD = (
            f"gin default {int(train_pools)} held because item (37)'s single "
            "train pool is off: window_single_pool() False, "
            + (
                f"env DL_SINGLE_POOL={single_env}"
                if single_env
                else "DL_SINGLE_POOL unset: its gin setting or no item (37) binding"
            )
            + "; env DL_FAST_BATCH=1 overrides"
        )
        train_pools = 0
    configure_fast_batch(train_pools, eval_pools)


@gin.configurable
def pin_free_shm_release(enabled: Optional[int] = None) -> None:
    """yambda_5b.gin item (55): resolve $DL_PIN_FREE_SHM (gin dlpfs/env_int)
    for this process and install it (datasets/dataset.py
    configure_pin_free_shm). > 0 = after Samples.pin_memory has made the
    pinned copy, the pin thread releases the received batch's shared-memory
    pages without the GIL; 0 = the storage destructor frees them (the kill
    switch). None (no gin binding: tests, tools that do not parse
    yambda_5b.gin) keeps the env-only code default (off). Called in the
    trainer process before any DataLoader pin thread starts
    (prestart_streaming_loaders, streaming_train_eval_loop); the pin threads
    are threads of this process and read the installed value. Idempotent: the
    same gin + env give the same value."""
    configure_pin_free_shm(enabled)


@gin.configurable
def fast_batch_onestore(level: Optional[int] = None) -> None:
    """yambda_5b.gin item (56): resolve $DL_FAST_BATCH_ONESTORE (gin
    dlfbos/env_int) for this process and install it (datasets/dataset.py
    configure_fast_batch_onestore). > 0 = every batch the fast builder
    (DLRMv4YambdaDataset.build_batch) makes is laid out in ONE storage, the
    UIH/candidate tensors being views of the two halves of the merged
    tensors, and the pin thread pins such a batch with one pinned allocation
    and one copy; 0 = one storage per tensor (the kill switch). Acts only
    where the fast builder runs: item (32)'s eval loaders and item (61)'s
    train pools ($DL_FAST_BATCH). None (no gin binding: tests, tools that
    do not parse yambda_5b.gin) keeps the env-only code default (off). Called
    in the trainer process before any pool is built
    (make_train_test_dataloaders, prestart_streaming_loaders,
    streaming_train_eval_loop): the forked workers inherit the value and the
    pin threads, threads of this process, read it. Idempotent: the same gin +
    env give the same value."""
    configure_fast_batch_onestore(level)


def _eval_dataset(base: HammerToTorchDataset, slot=None) -> HammerToTorchDataset:
    """What every eval loader reads: ``base``'s samples with the eval loaders'
    fast-batch setting ($DL_FAST_BATCH_EVAL, eval_fast_batch()) as the
    wrapper's own (None = follow $DL_FAST_BATCH: ``base`` as is), plus the
    dl_prestart read probe for the persistent eval pool (``slot``)."""
    fb = eval_fast_batch()
    if slot is not None:
        return _ReadProbedDataset(base, slot, fast_batch=fb)
    return base if fb is None else HammerToTorchDataset(base.dataset, fast_batch=fb)


# Pool kinds whose fast-batch line was logged (every rank, once per process).
_FAST_BATCH_POOLS_LOGGED: Set[str] = set()


def _log_fast_batch_pools(rank: int, pools: Dict[str, HammerToTorchDataset]) -> None:
    """One line per pool kind (every rank since ship 12: rank-0-only lines get
    lost in the merged log pipe; tagged rank=R) naming the builder its wrapper
    uses:
    ``armed OK: [fast-batch] <kind> pools: fast builder (build_batch; ...)`` or
    ``[fast-batch] <kind> pools: per-sample path (...)``, with the switch value
    and where it came from (env, gin default, code default; for
    $DL_FAST_BATCH held off by the item (61) precondition, the reason:
    fast_batch_pools). Silent when nothing selects the fast builder, no eval
    setting is given and nothing is held (the code default without
    yambda_5b.gin items (32) and (61))."""
    eval_setting = eval_fast_batch()
    train_src = fast_batch_knob_source("train")
    # Item (61) precondition: the reason fast_batch_pools held the train
    # switch off, while it still describes the installed value.
    held = (
        _FAST_BATCH_TRAIN_HELD
        if not fast_batch_enabled() and train_src == "gin default"
        else None
    )
    if not fast_batch_enabled() and eval_setting is None and held is None:
        return
    for kind, ds in pools.items():
        if kind in _FAST_BATCH_POOLS_LOGGED:
            continue
        _FAST_BATCH_POOLS_LOGGED.add(kind)
        own = getattr(ds, "fast_batch", None)
        if own is None:
            on = fast_batch_enabled()
            knob = f"DL_FAST_BATCH={int(on)}, {held or train_src}"
            if kind == "eval":
                knob = f"DL_FAST_BATCH_EVAL unset or < 0: follows {knob}"
        else:
            on = bool(own)
            knob = f"DL_FAST_BATCH_EVAL={int(on)}, {fast_batch_knob_source('eval')}"
        if on:
            logger.warning(
                "armed OK: [fast-batch] %s pools: fast builder (build_batch; %s) "
                "rank=%d",
                kind,
                knob,
                rank,
            )
        else:
            logger.warning(
                "[fast-batch] %s pools: per-sample path (%s) rank=%d", kind, knob, rank
            )


# True once this process logged the item (55) setting line.
_PIN_FREE_SHM_LOGGED: bool = False


def _log_pin_free_shm(rank: int, pinned: Optional[bool] = None) -> None:
    """Every rank (tagged rank=R, like the ship-12 setting lines: rank-0-only
    lines get lost in the merged log pipe), once per process: the
    $DL_PIN_FREE_SHM setting the pin threads use and where it came from (env,
    gin default). ``pinned`` False (no loader pins its batches,
    DL_PIN_MEMORY=0, so no pin thread runs) marks the setting inert; None =
    not known here. Not the arming line: every rank process's pin thread logs
    ``armed OK: [pin-free-shm] ...`` at the first batch whose shared memory it
    released. Silent on the env-only code default (no yambda_5b.gin item (55)
    and the variable unset)."""
    global _PIN_FREE_SHM_LOGGED
    if _PIN_FREE_SHM_LOGGED:
        return
    src = pin_free_shm_knob_source()
    if src == "code default":
        return
    _PIN_FREE_SHM_LOGGED = True
    if pin_free_shm_enabled():
        logger.warning(
            "[pin-free-shm] pin threads: release each received batch's shared "
            "memory after pinning (DL_PIN_FREE_SHM=1, %s%s) rank=%d",
            src,
            "; inert: no loader pins, DL_PIN_MEMORY=0" if pinned is False else "",
            rank,
        )
    else:
        logger.warning(
            "[pin-free-shm] pin threads: shared memory left to the storage "
            "destructor (DL_PIN_FREE_SHM=0, %s) rank=%d",
            src,
            rank,
        )


# True once this process logged the item (56) setting line.
_ONESTORE_LOGGED: bool = False


def _fast_builder_pool_kinds(pools: Dict[str, HammerToTorchDataset]) -> List[str]:
    """The kinds in ``pools`` whose wrapper builds with the fast builder (the
    rule _log_fast_batch_pools reports: the wrapper's own fast_batch, else
    $DL_FAST_BATCH)."""
    kinds = []
    for kind, ds in pools.items():
        own = getattr(ds, "fast_batch", None)
        if fast_batch_enabled() if own is None else bool(own):
            kinds.append(kind)
    return kinds


def _log_fast_batch_onestore(
    rank: int, pools: Optional[Dict[str, HammerToTorchDataset]] = None
) -> None:
    """Every rank (tagged rank=R), once per process: the
    $DL_FAST_BATCH_ONESTORE setting, where it came from (env, gin default)
    and which of ``pools`` build with the fast builder, the only batches the
    item lays out ("inert" when none does; ``pools`` None = not known here).
    Not an arming line: worker 0 of a pool that uses the fast builder appends
    "[onestore] ..." to its "armed OK: [fast-batch]" line, and every rank
    process's pin thread logs "armed OK: [onestore] ..." at the first batch
    it pinned as one storage. Silent on the env-only code default (no
    yambda_5b.gin item (56) and the variable unset)."""
    global _ONESTORE_LOGGED
    if _ONESTORE_LOGGED:
        return
    src = fast_batch_onestore_knob_source(show_raw=True)
    if src == "code default":
        return
    _ONESTORE_LOGGED = True
    if fast_batch_onestore_level():
        where = ""
        if pools is not None:
            kinds = _fast_builder_pool_kinds(pools)
            where = (
                "; fast-builder pools: " + ", ".join(kinds)
                if kinds
                else "; inert: no pool uses the fast builder"
            )
        logger.warning(
            "[onestore] fast-builder batches: one storage per batch, UIH/candidate "
            "tensors are views of the merged tensors, pinned with one copy "
            "(DL_FAST_BATCH_ONESTORE=2, %s%s) rank=%d",
            src,
            where,
            rank,
        )
    else:
        logger.warning(
            "[onestore] fast-builder batches: one storage per tensor "
            "(DL_FAST_BATCH_ONESTORE=0, %s) rank=%d",
            src,
            rank,
        )


def _draw_like_iterator_creation() -> None:
    """Consume exactly the global CPU RNG draw a DataLoader iterator creation
    makes (torch.utils.data _BaseDataLoaderIter.__init__: ``_base_seed`` from
    ``loader.generator``, None here). The pre-start creates the iterators early
    with the RNG state saved/restored, and replays this draw at the point the
    iterator WOULD have been created, so the main-process RNG stream (and every
    later consumer of it) is identical to DL_PRESTART=0."""
    torch.empty((), dtype=torch.int64).random_()


class _ChainedRanges:
    """O(1) __len__ + O(log K) __getitem__ over a sequence of `range`s.

    Lets `torch.utils.data.Subset(dataset, _ChainedRanges([r1, r2, ...]))`
    avoid materializing a Python list of all per-block indices (which at
    multi-billion totals is ~28 B/int and dominates host RAM).
    """

    def __init__(self, ranges: List[range]) -> None:
        self._ranges: List[range] = list(ranges)
        offsets = [0]
        for r in self._ranges:
            offsets.append(offsets[-1] + len(r))
        self._offsets: List[int] = offsets

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, idx: int) -> int:
        import bisect
        if idx < 0:
            idx += self._offsets[-1]
        if idx < 0 or idx >= self._offsets[-1]:
            raise IndexError(idx)
        bucket = bisect.bisect_right(self._offsets, idx) - 1
        return self._ranges[bucket][idx - self._offsets[bucket]]


class ChunkDistributedSampler(DistributedSampler[_T_co]):
    """
    Each rank reads a contiguous chunk (trunk) of the input data
    """

    def __init__(
        self,
        dataset: TorchDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 1,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * 1001 + int(self.rank))
            indices_t = torch.randperm(self.num_samples, generator=g)
        else:
            indices_t = torch.arange(self.num_samples)
        assert self.drop_last is True, (
            "drop_last must be True for ChunkDistributedSampler"
        )
        indices_t = indices_t + (self.num_samples * int(self.rank))
        assert indices_t.numel() == self.num_samples
        # Iterate via the numpy view, NOT directly over the tensor: iter(Tensor)
        # calls Tensor.unbind(0) which eagerly materializes one zero-dim Tensor
        # object per element (~600 B each). For 40 M eval / 525 M train samples
        # that's 24 GB / 315 GB of [heap] growth per rank, blowing host RAM
        # before the first batch. numpy's iter yields one Python int at a time
        # with O(1) extra memory.
        indices_np = indices_t.numpy()
        return (int(x) for x in indices_np)

    def set_epoch(self, epoch: int) -> None:
        logger.warning(f"Setting epoch to {epoch}")
        self.epoch = epoch


def _log_kernel_plan(path: str) -> None:
    """Log one greppable line naming the kernels this run will ACTUALLY use.

    The dispatch path on its own is misleading, and increasingly so: it selects
    an if/elif ladder, while narrower knobs replace individual kernels inside
    that ladder. $HSTU_ATTN_BWD swaps the HSTU attention backward for FlyDSL by
    default, so a "TRITON" run is really Triton-forward + FlyDSL-backward.

    Where each value came from is already logged by `apply_env_bootstrap`; this
    line is the resolved outcome, so that answering "what ran?" from a log is
    one grep rather than an inference across the gin file and the environment.
    """
    source = "env HSTU_HAMMER_KERNEL" if os.environ.get("HSTU_HAMMER_KERNEL") else "gin"
    bwd = (
        os.environ.get("HSTU_ATTN_BWD", DEFAULT_BWD_BACKEND).strip().lower()
        or DEFAULT_BWD_BACKEND
    )
    # HSTU_ATTN_BWD is read only under HammerKernel.TRITON, so off that path say
    # it is inert rather than printing a backend no kernel here will consult.
    if path == "TRITON":
        attn = f"fwd=triton bwd={bwd}"
    elif path:
        attn = f"fwd/bwd={path.lower()} (HSTU_ATTN_BWD={bwd} inert off the TRITON path)"
    else:
        path, source = "unset", "model default"
        attn = f"fwd/bwd=model default (HSTU_ATTN_BWD={bwd} applies only if that is TRITON)"
    logger.warning(
        "[kernel-plan] path=%s (from %s) | hstu_attn %s | triton_configs=%s",
        path,
        source,
        attn,
        "full-search" if os.environ.get("TRITON_FULL_AUTOTUNE") == "1" else "pinned",
    )


@gin.configurable
def make_model(
    dataset: str,
    bf16_training: bool = False,
    hammer_kernel: Optional[str] = None,
    ec_gather_select: bool = False,
    transducer_bf16_dtype: bool = False,
    ec_bf16_output: bool = False,
    ec_segment_reduce: bool = False,
    ec_merge_uih_candidate: bool = False,
    fuse_linear_norm: bool = False,
    fuse_ln_residual: bool = False,
) -> Tuple[torch.nn.Module, DlrmHSTUConfig, Dict[str, EmbeddingConfig]]:
    hstu_config = get_hstu_configs(dataset)
    table_config = get_embedding_table_config(dataset)

    # Hoist the EC-output bf16 cast above TorchRec's dedup expand. Must be
    # installed before the sharded EmbeddingCollection runs, i.e. before the
    # model is built. The forward is bitwise identical; the backward still
    # accumulates in fp32. See dlrm_v4/train/ec_bf16_output.py.
    if ec_bf16_output:
        if not ec_gather_select:
            raise ValueError(
                "make_model.ec_bf16_output=True requires ec_gather_select=True: "
                "the hoist replaces the gather-select expand and preserves its "
                "scatter_add_ backward. Enable EC_GATHER_SELECT or disable "
                "EC_BF16_OUTPUT rather than running an arm nothing has measured."
            )
        enable_bf16_ec_output()

    # Replace that hoisted backward's scatter_add_ with the sort-then-segment
    # reduce kernel. Gated separately from the hoist because it is the third
    # rung of the measured ladder and was measured as its own increment
    # (+4.37% e2e on top of ec_bf16_output). See dlrm_v4/train/ec_segment_reduce.py.
    if ec_segment_reduce:
        if not ec_bf16_output:
            raise ValueError(
                "make_model.ec_segment_reduce=True requires ec_bf16_output=True: "
                "the segment kernel replaces the backward of the HOISTED expand, "
                "so with the hoist off it is never reached and the arm would "
                "train inert -- which reads as a null result rather than as a "
                "broken arm. Enable EC_BF16_OUTPUT or disable EC_SEGMENT_REDUCE."
            )
        enable_segment_reduce()

    # Fold DlrmHSTU.main_forward's uih+candidate concat_2D_jagged into the
    # hoisted expand's index (and give each key its own expand so the split's
    # backward cat disappears). Forward bitwise identical; backward is a
    # reduction-order change only. See ec_bf16_output.py.
    if ec_merge_uih_candidate and not ec_bf16_output:
        # Default-on binding: rolling back the hoist (EC_BF16_OUTPUT=0) must
        # still run, so this disarms loudly instead of raising.
        logger.warning(
            "make_model.ec_merge_uih_candidate=True ignored: it lives inside "
            "the ec_bf16_output hoist, which is off. main_forward concats."
        )
    elif ec_merge_uih_candidate:
        emb_names = set(hstu_config.user_embedding_feature_names) | set(
            hstu_config.item_embedding_feature_names
        )
        enable_ec_merge_uih_candidate(
            [
                (u, c)
                for (u, c) in hstu_config.merge_uih_candidate_feature_mapping
                if u in emb_names and c in emb_names
            ]
        )
        if os.environ.get("DLRM_HSTU_FP8_EC_CAT_FUSE", "0") != "0":
            # Same list DlrmHSTU builds as _additional_embedding_features (the
            # keys ContextualPreprocessor concatenates along dim=1), in order.
            enable_ec_cat_fuse(
                [
                    u
                    for (u, c) in hstu_config.merge_uih_candidate_feature_mapping
                    if c in hstu_config.item_embedding_feature_names
                    and u in hstu_config.user_embedding_feature_names
                    and u != hstu_config.uih_post_id_feature_name
                ]
            )

    # Item (118) PREPROC_DEDUP (DLRM_HSTU_FP8_PREPROC_DEDUP, default off; the knob alone decides): registers the
    # consumer chain the dedup path runs on (ec_bf16_output.note_preproc_dedup_chain). Arms nothing by itself;
    # knob off: nothing is registered, imported or printed.
    from generative_recommenders.dlrm_v4.train import ec_bf16_output as _ebo_ppdd

    if _ebo_ppdd.ppdd_enabled():
        _ebo_ppdd.note_preproc_dedup_chain(
            fuse_linear_norm=fuse_linear_norm,
            fuse_ln_residual=fuse_ln_residual,
            bf16_training=bf16_training,
            ec_segment_reduce=ec_segment_reduce,
            transducer_bf16_dtype=transducer_bf16_dtype,
            uih_post_id=hstu_config.uih_post_id_feature_name,
        )

    # Indexed-X L1 (DLRM_HSTU_FP8_EC_INDEXED_L1, default off): the EC issues (U, idx) placeholders only when every
    # consumer of the hoisted members is Indexed-X-aware. Registration only; nothing happens with the knob off.
    note_ec_indexed_l1_chain(
        fuse_linear_norm=fuse_linear_norm,
        fuse_ln_residual=fuse_ln_residual,
        bf16_training=bf16_training,
        transducer_bf16_dtype=transducer_bf16_dtype,
        uih_post_id=hstu_config.uih_post_id_feature_name,
    )

    # bf16 autocast is off by default: on the PYTORCH attn backend the
    # pt_hstu_attention QK einsum backward overflows in bf16 at long
    # sequences (NaN at step 1 when N>1k). Safe with TRITON; flip via
    # `make_model.bf16_training = True` in the gin.
    model = DlrmHSTU(
        hstu_configs=hstu_config,
        embedding_tables=table_config,
        is_inference=False,
        bf16_training=bf16_training,
        use_gather_select=ec_gather_select,
        transducer_bf16_dtype=transducer_bf16_dtype,
        ec_bf16_output=ec_bf16_output,
        fuse_linear_norm=fuse_linear_norm,
        fuse_ln_residual=fuse_ln_residual,
    )
    # M4 speed estimator (dlrm_v4/train/m4_speed.py; no-op unless DLRM_HSTU_FP8_M4=1).
    from generative_recommenders.dlrm_v4.train import m4_speed as _m4_speed

    _m4_speed.install(model)
    if fuse_linear_norm and ((not dist.is_initialized()) or dist.get_rank() == 0):
        # POSITIVE arming proof, emitted by the feature itself rather than by an
        # echo of the env var. launch_slurm.sh forwards an env ALLOW-LIST and
        # silently drops anything not on it, so "we exported it" is not evidence
        # that the process ever saw it.
        logger.info(
            "armed OK: fuse_linear_norm ACTIVE -- ContextualPreprocessor MLPs "
            "use FusedLinearNormMLP; each Linear's bias grad is folded into the "
            "following norm backward, deleting the standalone aten::sum over dx"
        )
    if fuse_ln_residual and ((not dist.is_initialized()) or dist.get_rank() == 0):
        logger.info(
            "armed OK: fuse_ln_residual ACTIVE -- ContextualPreprocessor's two "
            "`output_seq_embeddings + mlp(...)` adds are folded into the MLP's "
            "final norm epilogue, deleting 2 x [M, 512] read-modify-write"
        )
    if ec_gather_select and ((not dist.is_initialized()) or dist.get_rank() == 0):
        logger.info(
            "DLRMV4 EC gather-select ENABLED (use_gather_select=True): the dedup "
            "expand uses torch.gather, so its backward is scatter_add_ rather than "
            "the torch-2.12-regressed index_add_"
        )

    # HSTU attention/compute kernel backend. Precedence:
    #   HSTU_HAMMER_KERNEL env var  >  make_model.hammer_kernel gin  >  model default.
    # The env var stays as an ad-hoc override (e.g. forcing PYTORCH for a one-off
    # debug run) without editing the gin. Note: the fused TRITON path avoids
    # materializing the dense [B, H, N, N] attention-score tensor that the PYTORCH
    # path allocates (~32 GiB at N=2048, bs=1024), so TRITON is both faster and
    # far lighter on HBM. On older ROCm, TRITON could hit PassManager errors at
    # some shapes (make_ttgir) — fall back to PYTORCH via the gin/env if so.
    kernel_choice = (
        os.environ.get("HSTU_HAMMER_KERNEL", "").upper()
        or (hammer_kernel.upper() if hammer_kernel else "")
    )
    if kernel_choice:
        model.set_hammer_kernel(HammerKernel[kernel_choice])
    _log_kernel_plan(kernel_choice)

    return (
        model,
        hstu_config,
        table_config,
    )


@gin.configurable()
def dense_optimizer_factory_and_class(
    optimizer_name: str,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    momentum: float,
    learning_rate: float,
) -> Tuple[
    Type[Optimizer], Dict[str, Any], Callable[[Iterable[torch.Tensor]], Optimizer]
]:
    global _DENSE_ADAM_FUSED
    kwargs: Dict[str, Any] = {"lr": learning_rate}
    if optimizer_name == "Adam":
        optimizer_cls = torch.optim.Adam
        kwargs.update({"betas": betas, "eps": eps, "weight_decay": weight_decay})
        # yambda_5b.gin item (40): DLRM_HSTU_FP8_DENSE_ADAM_FUSED, read here (at
        # construction, after the gin env bootstrap). The subclass keeps
        # torch.optim.Adam's state/param_groups/state_dict and steps every
        # eligible group in one bitwise Triton launch (anything else: torch).
        # Only "0" (or absent) disarms.
        if os.environ.get("DLRM_HSTU_FP8_DENSE_ADAM_FUSED", "0") != "0":
            from generative_recommenders.ops.triton import dense_adam_fused

            optimizer_cls = dense_adam_fused.FusedDenseAdam
            _DENSE_ADAM_FUSED = True
            logger.info(
                "[dense optimizer] %s=%s: FusedDenseAdam (one-launch bitwise "
                "foreach-Adam twin; clip_grad_norm_ multiply folded into it)",
                dense_adam_fused.ENV_KNOB,
                os.environ.get(dense_adam_fused.ENV_KNOB),
            )
    elif optimizer_name == "SGD":
        optimizer_cls = torch.optim.SGD
        kwargs.update({"weight_decay": weight_decay, "momentum": momentum})
    elif optimizer_name == "AdamW":
        optimizer_cls = torch.optim.AdamW
        kwargs.update({"betas": betas, "eps": eps, "weight_decay": weight_decay})
    else:
        raise Exception("Unsupported optimizer!")

    optimizer_factory = lambda params: optimizer_cls(params, **kwargs)

    logger.info(
        f"[dense optimizer] {optimizer_name} learning_rate={learning_rate} "
        f"(resolved from gin; override via $DENSE_LR)"
    )
    return optimizer_cls, kwargs, optimizer_factory


# Set by dense_optimizer_factory_and_class when it selected FusedDenseAdam
# (yambda_5b.gin item (40)); the clip call sites below then fold the grad
# multiply of the params it owns into its step.
_DENSE_ADAM_FUSED: bool = False


# Progress heartbeat for the launcher's stall watchdog (runner plans/dlrmv4 09).
# The trainer log is block-buffered behind launch_slurm.sh's sed filter, so it
# can sit still for minutes in healthy training and the watchdog had to allow
# 15 minutes. With DLRM_PROGRESS_FILE set, each node's local rank 0 rewrites
# <file>.r<rank> ("<step> <unix time>", tmp + rename) at most every 20 s while
# train steps or eval batches complete; a hang stops it within one step.
_PROGRESS_FILE: Optional[str] = os.environ.get("DLRM_PROGRESS_FILE") or None
_progress_last = [0.0]


def _progress_beat(rank: int, step: int) -> None:
    if _PROGRESS_FILE is None or rank % 8 != 0:
        return
    now = time.time()
    if now - _progress_last[0] < 20.0:
        return
    _progress_last[0] = now
    path = f"{_PROGRESS_FILE}.r{rank}"
    try:
        with open(path + ".tmp", "w") as f:
            f.write(f"{step} {int(now)}\n")
        os.replace(path + ".tmp", path)
    except OSError:
        pass


def _clip_grad_norm_(
    parameters: Iterable[torch.Tensor], max_norm: float, optimizer: Any
) -> torch.Tensor:
    """torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm), or, with
    item (40) armed, its bitwise twin that hands the clamped coefficient to the
    FusedDenseAdam inside `optimizer` (same norm, same coefficient, same fp32
    grad products, applied in the fused step instead of a _foreach_mul_)."""
    if _DENSE_ADAM_FUSED:
        from generative_recommenders.ops.triton.dense_adam_fused import (
            clip_grad_norm_fold_,
        )

        return clip_grad_norm_fold_(parameters, max_norm, optimizer)
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


@gin.configurable()
def sparse_optimizer_factory_and_class(
    optimizer_name: str,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    momentum: float,
    learning_rate: float,
) -> Tuple[
    Type[Optimizer], Dict[str, Any], Callable[[Iterable[torch.Tensor]], Optimizer]
]:
    kwargs: Dict[str, Any] = {"lr": learning_rate}
    if optimizer_name == "Adam":
        optimizer_cls = torch.optim.Adam
        beta1, beta2 = betas
        kwargs.update(
            {"beta1": beta1, "beta2": beta2, "eps": eps, "weight_decay": weight_decay}
        )
    elif optimizer_name == "SGD":
        optimizer_cls = torchrec.optim.SGD
        kwargs.update({"weight_decay": weight_decay, "momentum": momentum})
    elif optimizer_name == "RowWiseAdagrad":
        optimizer_cls = torchrec.optim.RowWiseAdagrad
        beta1, beta2 = betas
        kwargs.update(
            {
                "eps": eps,
                "beta1": beta1,
                "beta2": beta2,
                "weight_decay": weight_decay,
            }
        )
    else:
        raise Exception("Unsupported optimizer!")

    optimizer_factory = lambda params: optimizer_cls(params, **kwargs)

    logger.info(
        f"[sparse optimizer] {optimizer_name} learning_rate={learning_rate} "
        f"(resolved from gin; override via $SPARSE_LR)"
    )
    return optimizer_cls, kwargs, optimizer_factory


_FBGEMM_LOWMEM_PATCHED = False


def _patch_fbgemm_lowmem_clamp_cast(enabled: bool = True, rank0: bool = False) -> None:
    """Replace fbgemm's quant clamp+cast with a memory-frugal equivalent.

    ``enabled`` is the gin/env-driven kill switch (see
    ``make_optimizer_and_shard.qcomm_lowmem_clamp_cast`` /
    ``$QCOMM_LOWMEM_CODEC``). Default ON; pass ``enabled=False`` to fall back to
    stock fbgemm (e.g. to reproduce the pre-patch OOM, or if a future fbgemm
    version changes the codec and the patch needs revalidation).

    fbgemm's ``fp32_to_fp16_with_clamp`` (and the bf16 variant) does
    ``torch.clamp(tensor, MIN, MAX).half()``. ``torch.clamp(...)`` allocates a
    SECOND full-size fp32 tensor (same numel as the input) *before* the cast, so
    the transient peak is input(fp32) + clamp_temp(fp32) + output(fp16) ~= 2.5x
    the input. On a skewed row-wise-sharded batch the hottest shard's embedding
    tensor is huge (observed 81.5 GiB clamp temp), and that extra fp32 copy is
    exactly the allocation that OOMs the rank — which then exits the train loop
    while peers block forever in the a2a (a 30-min NCCL-watchdog hang). See
    HANG_ROOTCAUSE.md / flight-recorder dump for the full diagnosis.

    Cast FIRST then clamp IN PLACE: ``tensor.half().clamp_(MIN, MAX)``. This
    allocates only the fp16 output (no full-size fp32 temp), cutting the peak by
    the size of the input tensor, while being numerically identical: an fp32
    value above HALF_MAX casts to +inf, which clamp_ maps back to HALF_MAX (and
    NaNs pass through unchanged), matching clamp-then-cast bit for bit. Safe to
    do in place because the codec ``encode()`` runs inside the qcomm autograd
    ``Function.forward`` (grad disabled), so there is no graph to corrupt.
    """
    global _FBGEMM_LOWMEM_PATCHED
    if not enabled:
        if rank0:
            logger.warning(
                "[qcomm-lowmem] DISABLED (qcomm_lowmem_clamp_cast=False / "
                "QCOMM_LOWMEM_CODEC=0) — running stock fbgemm clamp+cast, which "
                "allocates a full-size fp32 clamp temp and can OOM->hang the "
                "hottest row-wise embedding shard on skewed batches."
            )
        return
    if _FBGEMM_LOWMEM_PATCHED:
        return
    try:
        from fbgemm_gpu import quantize_comm, quantize_utils

        _HMIN = quantize_utils.TORCH_HALF_MIN
        _HMAX = quantize_utils.TORCH_HALF_MAX
        _BMIN = quantize_utils.TORCH_BFLOAT16_MIN
        _BMAX = quantize_utils.TORCH_BFLOAT16_MAX

        def _lowmem_fp16(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.half().clamp_(_HMIN, _HMAX)

        def _lowmem_bf16(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.bfloat16().clamp_(_BMIN, _BMAX)

        # Patch BOTH the definition module and quantize_comm, which imported the
        # names directly (``from .quantize_utils import fp32_to_fp16_with_clamp``)
        # so its module-level reference must be overridden too.
        for _mod in (quantize_utils, quantize_comm):
            if hasattr(_mod, "fp32_to_fp16_with_clamp"):
                _mod.fp32_to_fp16_with_clamp = _lowmem_fp16
            if hasattr(_mod, "fp32_to_bf16_with_clamp"):
                _mod.fp32_to_bf16_with_clamp = _lowmem_bf16

        _FBGEMM_LOWMEM_PATCHED = True
        if rank0:
            logger.info(
                "[qcomm-lowmem] patched fbgemm fp32->fp16/bf16 clamp+cast to "
                "cast-then-clamp_ (drops the full-size fp32 clamp temp; avoids "
                "OOM on skewed row-wise embedding a2a)"
            )
    except Exception as e:  # noqa: BLE001 — patch is best-effort, never fatal
        if rank0:
            logger.warning(
                "[qcomm-lowmem] could not patch fbgemm clamp+cast (%s: %s); "
                "running with stock (higher-peak) quantizer",
                type(e).__name__,
                e,
            )


def _maybe_apply_qcomm_a2a(
    sharders: List[Any],
    device: torch.device,
    forward_precision: str = "fp32",
    backward_precision: str = "fp32",
    lowmem_clamp_cast: bool = True,
) -> List[Any]:
    """Optionally quantize the embedding all-to-all payload via TorchRec qcomm.

    The yambda-5b embedding shuffle is the dominant, bandwidth-bound (multi-node)
    collective (~14.5 GB/rank fp32); a bf16/fp16 wire dtype halves it. Quant/
    dequant happen inside the comm op, transparent to the lookup consumer. Ported
    from the DLRMv2 R2 lever, retargeted from ``EmbeddingBagCollectionSharder`` to
    the sequence ``EmbeddingCollectionSharder`` this model uses.

    Forward and backward are configured independently because they have different
    numerical needs (TorchRec golden_training/train_dlrm.py recommends
    forward=fp16, backward=bf16): the forward carries bounded embedding
    activations where fp16's extra mantissa helps, while gradients have a wider
    range that can overflow fp16, so bf16 (fp32 exponent range) is safer there.
    bf16 and fp16 are both 2 bytes, so the wire volume / perf is identical — the
    choice is purely numerical.

    Args (set via gin on ``make_optimizer_and_shard``, env-overridable). Each is
    one of ``fp32`` (that direction unquantized) | ``bf16`` | ``fp16``. If BOTH
    are fp32 the sharders are returned untouched (identical to baseline trunk).
    """
    _COMM = {"bf16": "BF16", "fp16": "FP16", "fp32": "FP32"}
    fwd = (forward_precision or "fp32").strip().lower()
    bwd = (backward_precision or "fp32").strip().lower()
    rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    for name, p in (("forward", fwd), ("backward", bwd)):
        if p not in _COMM:
            # Misconfigured precision: fail loudly rather than silently running
            # fp32. A typo in SPARSE_A2A_{FWD,BWD} must not pass as "no quant".
            raise ValueError(
                f"DLRMV4 qcomm a2a: unknown {name} precision {p!r} "
                f"(want one of fp32|bf16|fp16)"
            )
    if fwd == "fp32" and bwd == "fp32":
        return sharders
    # Before building the codec, swap fbgemm's clamp+cast for a memory-frugal
    # equivalent — see _patch_fbgemm_lowmem_clamp_cast for why (avoids a full
    # extra fp32 temp that OOMs the hottest row-wise shard on skewed batches).
    # Gated by `lowmem_clamp_cast` (gin/env); ON by default.
    _patch_fbgemm_lowmem_clamp_cast(enabled=lowmem_clamp_cast, rank0=rank0)
    try:
        from torchrec.distributed.embedding import EmbeddingCollectionSharder
        from torchrec.distributed.fbgemm_qcomm_codec import (
            CommType,
            get_qcomm_codecs_registry,
            QCommsConfig,
        )

        qcfg = QCommsConfig(
            forward_precision=getattr(CommType, _COMM[fwd]),
            backward_precision=getattr(CommType, _COMM[bwd]),
        )
        registry = get_qcomm_codecs_registry(qcfg, device=device)
    except Exception as e:  # noqa: BLE001
        # A configured quantized a2a that fails to build is a hard error. Silently
        # downgrading to fp32 would change numerics/throughput with no signal, and
        # a partial failure (one rank fp32, others fp16) would also desync the
        # collectives. Raise on every rank so the whole job aborts consistently.
        raise RuntimeError(
            f"DLRMV4 qcomm a2a: failed to enable configured quantization "
            f"(forward={fwd} backward={bwd}): {type(e).__name__}: {e}"
        ) from e

    new_sharders = []
    replaced = False
    for s in sharders:
        if type(s).__name__ == "EmbeddingCollectionSharder" and not replaced:
            new_sharders.append(
                EmbeddingCollectionSharder(qcomm_codecs_registry=registry)
            )
            replaced = True
        else:
            new_sharders.append(s)
    if not replaced:
        # Codec registry built fine, but there was no EmbeddingCollectionSharder to
        # bind it to, so the quantized a2a would be silently inert. Treat this as a
        # hard failure too — "configured but not applied" is the bug we want caught.
        raise RuntimeError(
            f"DLRMV4 qcomm a2a: quantization configured (forward={fwd} "
            f"backward={bwd}) but no EmbeddingCollectionSharder was found to attach "
            f"the qcomm codec registry to; refusing to run with quantization "
            f"silently disabled"
        )
    if rank0:
        logger.info(
            "DLRMV4 qcomm a2a ENABLED: forward=%s backward=%s "
            "replaced_ec_sharder=%s",
            fwd,
            bwd,
            replaced,
        )
    return new_sharders


def _maybe_apply_ec_index_dedup(
    sharders: List[Any],
    enabled: bool,
) -> List[Any]:
    """Optionally enable TorchRec's in-batch index dedup on the EC sharder.

    Runs as a FINAL pass AFTER get_default_sharders() and _maybe_apply_qcomm_a2a,
    so it covers BOTH sharder construction paths with a single knob:
      * qcomm ON  (SPARSE_A2A_* != fp32, the default): the EC sharder was already
        rebuilt with a qcomm codec registry inside _maybe_apply_qcomm_a2a.
      * qcomm OFF (SPARSE_A2A_* == fp32): that function early-returns the bare
        get_default_sharders() list untouched.
    In either case we rebuild the one EmbeddingCollectionSharder in place,
    PRESERVING its existing fused_params + qcomm codec registry, and only flip
    use_index_dedup. That ordering guarantees dedup composes with (never clobbers)
    the fp16/bf16 a2a instead of depending on which qcomm branch ran.

    use_index_dedup collapses duplicate ids within a batch to unique rows before
    the input all-to-all (torch.ops.fbgemm.jagged_unique_indices), does the
    embedding lookup once, then scatters back via reverse_indices. For a
    non-pooled EmbeddingCollection (this sequence model) that reconstruction is
    exact, so the forward is numerically lossless — unlike the fp16 a2a it does
    not perturb AUC/NE. The buffers it registers (_hash_size_cumsum_tensor_*,
    _hash_size_offset_tensor_*) are persistent=False, so it does NOT change the
    on-disk shard layout and can be toggled on an existing checkpoint with no
    reshard. Payoff scales with the in-batch duplicate rate: large under
    user-major batching (STREAMING_SHUFFLE_FRACTION=0, the default, where one
    user's UIH re-reads the same item/artist/album ids), shrinking toward
    neutral/negative as batch diversity rises.

    The knob only exists on EmbeddingCollectionSharder, which is the sole default
    sharder this model actually binds (a pure sequence EmbeddingCollection), so
    no other sharder needs touching. Configured via gin on
    ``make_optimizer_and_shard`` (env-overridable $EC_INDEX_DEDUP).
    """
    if not enabled:
        return sharders

    from torchrec.distributed.embedding import EmbeddingCollectionSharder

    rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    new_sharders = []
    replaced = False
    for s in sharders:
        if type(s).__name__ == "EmbeddingCollectionSharder" and not replaced:
            new_sharders.append(
                EmbeddingCollectionSharder(
                    fused_params=s.fused_params,
                    qcomm_codecs_registry=s.qcomm_codecs_registry,
                    use_index_dedup=True,
                )
            )
            replaced = True
        else:
            new_sharders.append(s)
    if not replaced:
        # Dedup configured but no EC sharder to bind it to would mean it is
        # silently inert — the same "configured but not applied" bug class the
        # qcomm path guards against. Fail loudly on every rank instead.
        raise RuntimeError(
            "DLRMV4 EC index dedup configured (use_index_dedup=True) but no "
            "EmbeddingCollectionSharder was found to enable it on; refusing to "
            "run with dedup silently disabled"
        )
    if rank0:
        logger.info("DLRMV4 EC index dedup ENABLED (use_index_dedup=True)")
    return new_sharders


def _embedding_table_names(
    model: torch.nn.Module,
    embedding_table_configs: Optional[Dict[str, EmbeddingConfig]],
) -> List[str]:
    """All embedding table names the planner will place.

    Prefers the authoritative `embedding_table_configs` (keys == table names ==
    planner parameter names). Falls back to walking the model's EBC/EC modules
    when the configs are not passed in.
    """
    if embedding_table_configs:
        return list(embedding_table_configs.keys())
    names: List[str] = []
    for _, module in model.named_modules():
        if type(module) in TORCHREC_TYPES:
            if isinstance(module, EmbeddingBagCollection):
                names.extend(c.name for c in module.embedding_bag_configs())
            elif isinstance(module, EmbeddingCollection):
                names.extend(c.name for c in module.embedding_configs())
    return names


def _build_placement_constraints(
    model: torch.nn.Module,
    embedding_placement: str,
    embedding_placement_overrides: Dict[str, str],
    embedding_table_configs: Optional[Dict[str, EmbeddingConfig]],
    embedding_sharding_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, ParameterConstraints]:
    """Translate gin/env placement + sharding strings into ParameterConstraints.

    Two orthogonal per-table knobs are merged into one constraint per table:

    * Placement (compute kernel / memory tier):
      ``embedding_placement_overrides.get(name, embedding_placement)``.
      ``auto``/empty -> no compute-kernel constraint (planner decides from HBM).
    * Sharding type (shard layout): ``embedding_sharding_overrides.get(name)``.
      ``auto``/empty (or absent) -> no sharding-type constraint (planner decides,
      which is ROW_WISE for the large yambda tables). Use e.g. ``column_wise``
      to move a hot, data-skewed table off ROW_WISE so its embedding all-to-all
      is balanced by rank instead of routed by (hot) row.

    A table is added to the returned dict only if at least one knob is set for it
    (so with everything ``auto`` we return {} and the plan is byte-identical to
    the legacy path). Unknown values raise ValueError.
    """
    embedding_sharding_overrides = embedding_sharding_overrides or {}
    valid_place = set(_PLACEMENT_TO_KERNEL) | {"auto", ""}
    for where, val in [
        ("embedding_placement", embedding_placement),
        *[
            (f"embedding_placement_overrides[{k}]", v)
            for k, v in embedding_placement_overrides.items()
        ],
    ]:
        if val not in valid_place:
            raise ValueError(
                f"Invalid embedding placement {val!r} for {where}; "
                f"expected one of {sorted(valid_place - {''})}."
            )
    valid_shard = set(_SHARDING_TO_TYPE) | {"auto", ""}
    for k, v in embedding_sharding_overrides.items():
        if v not in valid_shard:
            raise ValueError(
                f"Invalid embedding sharding {v!r} for "
                f"embedding_sharding_overrides[{k}]; "
                f"expected one of {sorted(valid_shard - {''})}."
            )

    names = _embedding_table_names(model, embedding_table_configs)
    unknown = (
        set(embedding_placement_overrides) | set(embedding_sharding_overrides)
    ) - set(names)
    if unknown:
        logger.warning(
            "[emb-placement] override(s) for unknown table(s) %s ignored; "
            "known tables: %s",
            sorted(unknown),
            sorted(names),
        )

    constraints: Dict[str, ParameterConstraints] = {}
    resolved_place: Dict[str, str] = {}
    resolved_shard: Dict[str, str] = {}
    for name in names:
        placement = embedding_placement_overrides.get(name, embedding_placement)
        sharding = embedding_sharding_overrides.get(name, "auto")
        resolved_place[name] = placement or "auto"
        resolved_shard[name] = sharding or "auto"
        kernel = _PLACEMENT_TO_KERNEL.get(placement)
        stype = _SHARDING_TO_TYPE.get(sharding)
        kwargs: Dict[str, Any] = {}
        if kernel is not None:
            kwargs["compute_kernels"] = [kernel.value]
        if stype is not None:
            kwargs["sharding_types"] = [stype.value]
        if kwargs:
            constraints[name] = ParameterConstraints(**kwargs)

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        logger.info(
            "[emb-placement] placement(global=%r overrides=%s) sharding(overrides=%s) "
            "-> resolved_placement=%s resolved_sharding=%s "
            "(constrained=%d/%d tables; the rest are planner-auto)",
            embedding_placement,
            embedding_placement_overrides or {},
            embedding_sharding_overrides or {},
            resolved_place,
            resolved_shard,
            len(constraints),
            len(names),
        )
    return constraints


class _DdpDeferredWaitState:
    """State of `_ddp_allreduce_deferred_wait_hook` (one per DDP instance).

    DLRM_DDP_DEFERRED_WAIT_VERIFY=N (diagnostic, default 0): the first N bucket
    all-reduces also run the reducer's built-in path on a copy of the same local
    bucket and compare the two results byte for byte (see the hook).
    """

    def __init__(
        self, process_group: dist.ProcessGroup, batched_grad_copy: bool
    ) -> None:
        self.process_group = process_group
        self.world_size = process_group.size()
        self.batched_grad_copy = batched_grad_copy
        self.verify_total = int(os.environ.get("DLRM_DDP_DEFERRED_WAIT_VERIFY", "0") or 0)
        self.verify_left = self.verify_total
        self.verify_n = 0
        self.verify_bad: Optional[torch.Tensor] = None

    def scale(self, t: torch.Tensor) -> None:
        if self.batched_grad_copy:
            t.div_(self.world_size)
        else:
            t.mul_(1.0 / self.world_size)


def _ddp_bucket_result(fut: torch.futures.Future[List[torch.Tensor]]) -> torch.Tensor:
    return fut.value()[0]


def _ddp_at_end_of_backward(fn: Callable[[], None]) -> None:
    """Run `fn` as an autograd final callback when called from inside a node
    (the reducer's per-parameter hooks), else now. Final callbacks run after the
    graph's last node, before backward() returns, with the stream backward() was
    called from current. Outside a node -- the static-graph first iteration
    reduces every bucket from its own final callback (Reducer::delay_all_reduce)
    and finalizes right after -- a queued callback would run only after that
    finalize, so `fn` runs at once (the reducer's built-in ordering)."""
    current_node = getattr(torch._C, "_current_autograd_node", None)
    if current_node is not None and current_node() is not None:
        try:
            torch.autograd.Variable._execution_engine.queue_callback(fn)
            return
        except RuntimeError:
            pass
    fn()


def _ddp_allreduce_deferred_wait_hook(
    state: _DdpDeferredWaitState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """DDP bucket all-reduce that leaves the backward compute stream running.

    The reducer's built-in path (no hook) calls `allreduce(bucket)->getFuture()`
    on the autograd thread right after issuing each bucket. torch 2.16's default
    c10d "nccl" backend (nccl2, `c10d::nccl2::WorkNCCL::getFuture`) makes the
    *current* stream wait for the collective inside getFuture(); legacy
    ProcessGroupNCCL returns a future completed on the NCCL stream and waits
    nowhere. On the autograd thread the current stream is the backward compute
    stream, so under nccl2 the rest of backward queues behind every bucket
    all_reduce (and behind the slowest rank's arrival) instead of overlapping it.

    This hook issues the same all_reduce and makes the compute stream wait for it
    only at the end of backward (`work.wait()` from an autograd final callback,
    which runs on the stream backward() was called from), i.e. before
    Reducer::finalize_backward, clip-grad-norm and the optimizer consume the
    reduced gradients: legacy ProcessGroupNCCL's ordering. It adds no stream: a
    side stream carrying the wait would share one of the GPU's few hardware
    queues (GPU_MAX_HW_QUEUES) with the trainer's compute / data-dist / RCCL /
    copy streams, and a wait packet there blocks every stream on that queue. The
    returned future is a completed CPU future (no stream semantics), so the
    reducer's own `future_work->wait()` neither blocks the host nor adds a wait.

    Same bits as the built-in path: with a hook registered the reducer skips its
    own 1/world scaling, so the hook applies the op the reducer would have run
    on the same bucket bytes, on the same stream, before the same all_reduce:
    one flat `div_(world)` (Reducer::flush_deferred_copies, batched_grad_copy)
    or `mul(1.0 / world)` as a double scalar (the per-parameter `mul_out` by
    `wrapped_scalar_tensor(1. / div_factor_)`). Dense buckets only (the dense
    DDP here has no sparse gradients); no DDP Join (div_factor_ = world size).
    """
    buf = bucket.buffer()
    if buf.device.type != "cuda":
        state.scale(buf)
        work = dist.all_reduce(buf, group=state.process_group, async_op=True)
        return work.get_future().then(_ddp_bucket_result)
    raw = None
    if state.verify_left > 0:
        # Counted at issue: the deferred callbacks of one backward run only after
        # all its buckets are issued, so a count kept there would verify (and log)
        # the wrong number of buckets.
        state.verify_left -= 1
        raw = buf.clone()
    state.scale(buf)
    work = dist.all_reduce(buf, group=state.process_group, async_op=True)
    ref_work = None
    if raw is not None:
        # Diagnostic: the reducer's built-in path on the same local bucket, issued
        # right after (same order on every rank): its scaling op, then
        # allreduce(...)->getFuture() with the compute stream current.
        state.scale(raw)
        ref_work = dist.all_reduce(raw, group=state.process_group, async_op=True)
        ref_work.get_future()

    def wait_reduced() -> None:
        work.wait()  # a stream wait (no host block unless TORCH_NCCL_BLOCKING_WAIT)
        if ref_work is None or raw is None:
            return
        ref_work.wait()
        raw.record_stream(torch.cuda.current_stream(buf.device))
        bad = (buf.view(torch.uint8) != raw.view(torch.uint8)).sum()
        if state.verify_bad is None:
            state.verify_bad = torch.zeros((), dtype=torch.int64, device=buf.device)
        state.verify_bad += bad
        state.verify_n += 1
        if state.verify_n == state.verify_total:
            logger.info(
                "DLRMV4 ddp deferred wait VERIFY rank %d: %d bucket all_reduces, "
                "%d bytes differ from the reducer's built-in path",
                dist.get_rank(),
                state.verify_n,
                int(state.verify_bad.item()),
            )

    _ddp_at_end_of_backward(wait_reduced)
    # Python-object result: PythonCommHook::parseHookResult requires one.
    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(buf)
    return fut


def _ddp_getfuture_waits_current_stream(
    process_group: dist.ProcessGroup, device: torch.device
) -> Tuple[bool, str]:
    """(True, name) when the group's backend for `device` is torch's nccl2 family
    (ProcessGroupNCCL2, ProcessGroupNCCLLazy: c10d::nccl2::WorkNCCL, whose
    getFuture() makes the current stream wait for the collective)."""
    try:
        backend = process_group._get_backend(device)
    except Exception:  # noqa: BLE001 - no backend for this device
        return False, "?"
    import torch.distributed.distributed_c10d as c10d

    nccl2_types = tuple(
        t
        for t in (
            getattr(c10d, "ProcessGroupNCCL2", None),
            getattr(c10d, "ProcessGroupNCCLLazy", None),
        )
        if t is not None
    )
    return bool(nccl2_types) and isinstance(backend, nccl2_types), type(
        backend
    ).__name__


def _make_dense_data_parallel_wrapper(ddp_kwargs: Optional[Dict[str, Any]] = None):
    """DefaultDataParallelWrapper(ddp_kwargs) that registers the deferred-wait hook.

    Kill switch DLRM_DDP_DEFERRED_WAIT=0: the plain DefaultDataParallelWrapper
    (no comm hook, the reducer's built-in all_reduce path).
    """
    from torchrec.distributed.model_parallel import DefaultDataParallelWrapper

    kwargs: Dict[str, Any] = {} if ddp_kwargs is None else {"ddp_kwargs": ddp_kwargs}
    if os.environ.get("DLRM_DDP_DEFERRED_WAIT", "1").strip().lower() in (
        "0",
        "false",
        "off",
        "no",
    ):
        logger.info(
            "DLRMV4 ddp deferred wait: OFF (DLRM_DDP_DEFERRED_WAIT=0) -- the "
            "reducer's built-in bucket all_reduce"
        )
        return DefaultDataParallelWrapper(**kwargs)

    class _DeferredWaitDataParallelWrapper(DefaultDataParallelWrapper):
        def wrap(self, dmp, env, device) -> None:  # pyre-ignore[2]
            super().wrap(dmp, env, device)
            ddp = dmp._dmp_wrapped_module
            if not isinstance(ddp, torch.nn.parallel.DistributedDataParallel):
                return
            if getattr(ddp, "_comm_hooks", None):
                # A reducer takes one hook: torchrec's fp16/bf16 compression hook
                # (allreduce_comm_precision, unset in this trainer) or an earlier
                # wrap() of this DMP already registered one.
                logger.warning(
                    "DLRMV4 ddp deferred wait: not registered, the DDP already "
                    "has comm hook(s) %s",
                    [h[0].__qualname__ for h in ddp._comm_hooks],
                )
                return
            needed, backend = _ddp_getfuture_waits_current_stream(
                ddp.process_group, device
            )
            rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
            if not needed:
                # e.g. legacy ProcessGroupNCCL (TORCH_DIST_USE_NCCL2=0): its
                # getFuture() waits nowhere, the built-in path already defers.
                if rank0:
                    logger.info(
                        "DLRMV4 ddp deferred wait: not registered (backend=%s: "
                        "its Work.getFuture() does not wait on the current "
                        "stream; the reducer's built-in all_reduce already "
                        "overlaps backward)",
                        backend,
                    )
                return
            state = _DdpDeferredWaitState(
                ddp.process_group, bool(getattr(ddp, "batched_grad_copy", False))
            )
            ddp.register_comm_hook(state, _ddp_allreduce_deferred_wait_hook)
            if rank0:
                logger.info(
                    "DLRMV4 armed OK: ddp deferred wait -- dense DDP bucket "
                    "all_reduce through _ddp_allreduce_deferred_wait_hook "
                    "(compute stream waits for it at the end of backward, not "
                    "at issue; backend=%s world=%d %s; kill switch "
                    "DLRM_DDP_DEFERRED_WAIT=0)",
                    backend,
                    state.world_size,
                    "div_" if state.batched_grad_copy else "mul_",
                )

    return _DeferredWaitDataParallelWrapper(**kwargs)


def _dense_data_parallel_wrapper():
    """torchrec's default dense DDP wrapper, plus DDP batched_grad_copy.

    The returned wrapper also registers `_ddp_allreduce_deferred_wait_hook`
    (kill switch DLRM_DDP_DEFERRED_WAIT=0): see that hook for why.

    iter63 small-kill (b). Default OFF in code; the gin binding
    apply_env_bootstrap.DLRM_HSTU_FP8_DDP_BATCHED_COPY turns it on. Off
    (absent, or the kill switch DLRM_HSTU_FP8_DDP_BATCHED_COPY=0) returns
    exactly the wrapper DMP builds when given none (DefaultDataParallelWrapper(): bucket_cap_mb=25,
    static_graph=True, gradient_as_bucket_view=True, no comm hook).

    The dense optimizer's zero_grad frees every grad (torchrec's
    KeyedOptimizerWrapper.zero_grad calls the torch default set_to_none=True),
    so the reducer cannot alias grads to bucket views and runs one
    `mul_out(bucket_view, grad, 1/world)` per parameter per step: 69 launches,
    0.38-0.53 ms (mean 0.46) per rank-step (ship10 1n traces, iter-63 bench
    TB1/TB2, ranks 0-7, steps 53-56). batched_grad_copy (torch 2.12 reducer)
    defers those to one `_foreach_copy_` plus one flat
    `bucket.gradients.div_(world)` per bucket (2 buckets here). On CUDA/ROCm
    both multiply by (float)(1.0 / world) -- mul_out by a CPU-scalar tensor,
    div_ by a CPU scalar via aten's reciprocal multiply -- for ANY world size,
    so the bucket bytes, the allreduce, clip-grad-norm and the optimizer see
    identical bits (on CPU/gloo div_ is a true division: identical only for
    power-of-two world sizes).
    """
    from torchrec.distributed.model_parallel import DefaultDataParallelWrapper

    value = os.environ.get("DLRM_HSTU_FP8_DDP_BATCHED_COPY", "").strip().lower()
    if value not in ("1", "true", "on", "yes"):
        return _make_dense_data_parallel_wrapper()
    ddp_params = inspect.signature(
        torch.nn.parallel.DistributedDataParallel.__init__
    ).parameters
    if "batched_grad_copy" not in ddp_params:
        logger.warning(
            "DLRMV4 ddp batched copy: this torch's DistributedDataParallel has "
            "no batched_grad_copy; keeping the per-parameter mul_out"
        )
        return _make_dense_data_parallel_wrapper()
    if "ddp_kwargs" not in inspect.signature(
        DefaultDataParallelWrapper.__init__
    ).parameters:
        logger.warning(
            "DLRMV4 ddp batched copy: this torchrec's DefaultDataParallelWrapper "
            "has no ddp_kwargs; keeping the per-parameter mul_out"
        )
        return _make_dense_data_parallel_wrapper()
    logger.info(
        "DLRMV4 armed OK: ddp batched copy -- dense DDP batched_grad_copy=True "
        "(per-bucket _foreach_copy_ + flat div_ instead of a mul_out per "
        "parameter; kill switch DLRM_HSTU_FP8_DDP_BATCHED_COPY=0)"
    )
    return _make_dense_data_parallel_wrapper({"batched_grad_copy": True})


@gin.configurable
def make_optimizer_and_shard(
    model: torch.nn.Module,
    device: torch.device,
    world_size: int,
    local_world_size: Optional[int] = None,
    hbm_cap_gb: int = 260,
    ddr_cap_gb: int = 0,
    sparse_a2a_forward_precision: str = "fp32",
    sparse_a2a_backward_precision: str = "fp32",
    qcomm_lowmem_clamp_cast: bool = True,
    ec_index_dedup: bool = False,
    embedding_placement: str = "auto",
    embedding_placement_overrides: Optional[Dict[str, str]] = None,
    embedding_sharding_overrides: Optional[Dict[str, str]] = None,
    embedding_table_configs: Optional[Dict[str, EmbeddingConfig]] = None,
) -> Tuple[DistributedModelParallel, torch.optim.Optimizer]:
    dense_opt_cls, dense_opt_args, dense_opt_factory = (
        dense_optimizer_factory_and_class()
    )

    sparse_opt_cls, sparse_opt_args, sparse_opt_factory = (
        sparse_optimizer_factory_and_class()
    )
    # Fuse sparse optimizer to backward step
    for k, module in model.named_modules():
        if type(module) in TORCHREC_TYPES:
            for _, param in module.named_parameters(prefix=k):
                if param.requires_grad:
                    apply_optimizer_in_backward(
                        sparse_opt_cls, [param], sparse_opt_args
                    )
    sharders = get_default_sharders()
    sharders = _maybe_apply_qcomm_a2a(
        sharders,
        device,
        forward_precision=sparse_a2a_forward_precision,
        backward_precision=sparse_a2a_backward_precision,
        lowmem_clamp_cast=qcomm_lowmem_clamp_cast,
    )
    # Final pass: enable in-batch index dedup on the EC sharder regardless of
    # whether the qcomm branch above rebuilt it (fp16 default) or returned the
    # bare get_default_sharders() list (fp32). Preserves the qcomm registry.
    sharders = _maybe_apply_ec_index_dedup(sharders, enabled=ec_index_dedup)
    # local_world_size = GPUs per node so the planner respects the intra-node
    # (xGMI/NVLink) vs inter-node hierarchy when placing shards. Defaults to
    # world_size for the single-node case (no behavior change).
    logger.info(
        "[hbm-cap] make_optimizer_and_shard: hbm_cap_gb=%s (planner Topology hbm_cap=%d bytes), "
        "ddr_cap_gb=%s, world_size=%s local_world_size=%s",
        hbm_cap_gb,
        hbm_cap_gb * 1024 * 1024 * 1024,
        ddr_cap_gb,
        world_size,
        local_world_size or world_size,
    )
    # Resolve per-table embedding placement (gin/env-driven). Global default
    # `embedding_placement` applies to every table; `embedding_placement_overrides`
    # (table name -> placement) wins per table. Tables resolving to "auto" carry
    # no constraint (planner decides from hbm_cap). When nothing is constrained we
    # pass constraints=None so the plan is byte-identical to the legacy path.
    constraints = _build_placement_constraints(
        model=model,
        embedding_placement=embedding_placement,
        embedding_placement_overrides=embedding_placement_overrides or {},
        embedding_sharding_overrides=embedding_sharding_overrides or {},
        embedding_table_configs=embedding_table_configs,
    )
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=local_world_size or world_size,
            world_size=world_size,
            compute_device="cuda",
            hbm_cap=hbm_cap_gb * 1024 * 1024 * 1024,
            # 0 (default) leaves the planner no host-DDR budget, so every table
            # must fit HBM and a "uvm"/"uvm_caching" placement cannot be planned.
            # Give it a budget to allow UVM-backed tables.
            ddr_cap=ddr_cap_gb * 1024 * 1024 * 1024,
        ),
        constraints=constraints or None,
    )
    pg = dist.GroupMember.WORLD
    env = ShardingEnv.from_process_group(pg)  # pyre-ignore [6]
    pg = env.process_group

    plan = planner.collective_plan(model, sharders, pg)

    # Authoritative placement log: report the compute kernel the planner ACTUALLY
    # assigned to each table (vs the [emb-placement] line above, which reports what
    # was requested). "fused" = HBM; "fused_uvm"/"fused_uvm_caching" = UVM-backed.
    # Rank 0 only, best-effort (never break the build over a logging shape change).
    if (dist.get_rank() if dist.is_initialized() else 0) == 0:
        try:
            for module_path, param_plans in plan.plan.items():
                for param_name, ps in param_plans.items():
                    logger.info(
                        "[emb-placement] plan: %s.%s -> compute_kernel=%s "
                        "sharding_type=%s",
                        module_path,
                        param_name,
                        getattr(ps, "compute_kernel", "?"),
                        getattr(ps, "sharding_type", "?"),
                    )
        except Exception as e:  # logging only; must never fail the build
            logger.warning("[emb-placement] could not dump plan kernels: %s", e)

    # Re-seed right before DMP materializes/inits the sharded embedding tables.
    # The per-table seeded init_fn (configs.get_embedding_table_config) handles
    # the eager path, but the fused FBGEMM TBE path inits weights on-device and
    # may bypass init_fn, drawing from the global RNG instead. Re-seeding here
    # (same value on every rank) makes embedding init reproducible run-to-run for
    # a fixed sharding plan (Tier 1). Dense params are already initialized in
    # make_model, so this does not perturb them.
    _emb_seed = int(os.environ.get("SEED", "1"))
    torch.manual_seed(_emb_seed)
    torch.cuda.manual_seed_all(_emb_seed)
    logger.info(f"[emb-init] re-seeded RNGs before DMP with SEED={_emb_seed}")

    # Shard model
    model = DistributedModelParallel(
        module=model,
        device=device,
        plan=plan,
        sharders=sharders,
        data_parallel_wrapper=_dense_data_parallel_wrapper(),
    )

    # `transducer_bf16_dtype` sets a plain Python attribute on the transducer in
    # DlrmHSTU.__init__, so it has to survive DMP wrapping and torchrec's
    # meta-device to_empty() path to do anything. It should -- plain attributes
    # survive both -- but if it ever stops, the only symptom is that the
    # ~4.9 ms/step/rank of fp32<->bf16 round-trip casts quietly come back, with
    # no error anywhere. Log it once so a silent no-op is greppable.
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        _inner = getattr(model, "module", model)
        _td = getattr(
            getattr(_inner, "_hstu_transducer", None), "_training_dtype", None
        )
        if _td is not None:
            logger.info(
                "DLRMV4 transducer _training_dtype after sharding: %s%s",
                _td,
                ""
                if _td == torch.bfloat16
                else "  (fp32 -- the entry-cast round trip is ACTIVE; expect "
                "~4.9 ms/step/rank of extra [tokens,512] casts)",
            )

    # --- startup init checksum (reproducibility probe) -------------------------
    # Right after DMP materializes real weights, log a deterministic fingerprint
    # of every parameter so two builds with the same $SEED + sharding plan can be
    # diffed for byte-level init reproducibility. For sharded embeddings we
    # all-reduce the per-shard (count, sum, sumsq) so the fingerprint covers the
    # WHOLE table independent of how rows split across ranks; replicated dense
    # params use rank 0's local copy.
    # OFF BY DEFAULT: the fp64 reductions below (.sum(dtype=float64) /
    # vector_norm(dtype=float64)) materialize a full fp64 copy of each local
    # embedding shard (~2x the fp32 shard, i.e. >150 GiB for the big tables),
    # which leaves almost no HBM headroom after sharding and will OOM the build on
    # any node with residual memory. Only enable for explicit reproducibility
    # checks, ideally with a smaller batch / on a clean node. Enable with
    # INIT_CHECKSUM=1.
    if os.environ.get("INIT_CHECKSUM", "0") == "1":
        import hashlib

        _rank = dist.get_rank() if dist.is_initialized() else 0
        _fps: List[str] = []
        for _name, _p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            _sharded = isinstance(_p, ShardedTensor)
            if _sharded:
                _shards = _p.local_shards()
                _loc = _shards[0].tensor if _shards else None
            else:
                _loc = _p
            if _loc is None or _loc.numel() == 0:
                _cnt, _sm, _sq = 0.0, 0.0, 0.0
            else:
                _det = _loc.detach()
                _cnt = float(_det.numel())
                _sm = _det.sum(dtype=torch.float64).item()
                _nrm = torch.linalg.vector_norm(
                    _det, ord=2, dtype=torch.float64
                ).item()
                _sq = _nrm * _nrm
            if _sharded and dist.is_initialized():
                _stat = torch.tensor(
                    [_cnt, _sm, _sq], dtype=torch.float64, device=device
                )
                dist.all_reduce(_stat, op=dist.ReduceOp.SUM)
                _cnt, _sm, _sq = _stat.tolist()
            _fps.append(f"{_name}|{int(_cnt)}|{_sm:.6f}|{_sq:.6f}")
            if _rank == 0:
                logger.info(
                    f"[init-checksum] {'sharded' if _sharded else 'dense'} "
                    f"{_name} n={int(_cnt)} sum={_sm:.6f} sumsq={_sq:.6f}"
                )
        if _rank == 0:
            _digest = hashlib.sha256("\n".join(_fps).encode()).hexdigest()[:16]
            logger.info(
                f"[init-checksum] SEED={os.environ.get('SEED', '?')} "
                f"params={len(_fps)} digest={_digest}"
            )

    # Create keyed optimizer
    all_optimizers = []
    all_params = {}
    non_fused_sparse_params = {}
    for k, v in in_backward_optimizer_filter(model.named_parameters()):
        if v.requires_grad:
            if isinstance(v, ShardedTensor):
                non_fused_sparse_params[k] = v
            else:
                all_params[k] = v

    if non_fused_sparse_params:
        all_optimizers.append(
            (
                "sparse_non_fused",
                KeyedOptimizerWrapper(
                    params=non_fused_sparse_params, optim_factory=sparse_opt_factory
                ),
            )
        )

    if all_params:
        all_optimizers.append(
            (
                "dense",
                KeyedOptimizerWrapper(
                    params=all_params,
                    optim_factory=dense_opt_factory,
                ),
            )
        )
    output_optimizer = CombinedOptimizer(all_optimizers)
    output_optimizer.init_state(set(model.sparse_grad_parameter_names()))
    return model, output_optimizer


@gin.configurable
def make_streaming_dataloader(
    dataset: HammerToTorchDataset,
    ts: Optional[int] = None,
    batch_size: int = 0,
    num_workers: int = 0,
    prefetch_factor: int = 0,
    train_only: bool = False,
    indices: Optional["np.ndarray"] = None,
    pin_memory: bool = False,
) -> DataLoader:
    # `indices` (explicit anchor index array) is used by the eval path to
    # iterate the FIXED user-holdout set, which spans a window range rather than
    # a single ts. Otherwise restrict to window `ts`; train_only=True drops
    # held-out eval users so the non-persistent TRAIN loader never trains on
    # them (no-leakage guarantee).
    if indices is not None:
        dataset.dataset.set_active_indices(indices)  # pyre-ignore [16]
    else:
        assert ts is not None, "make_streaming_dataloader needs ts or indices"
        dataset.dataset.set_ts(ts, train_only=train_only)  # pyre-ignore [16]
    total_items = dataset.dataset.get_item_count()
    subset = torch.utils.data.Subset(dataset, range(total_items))
    # shuffle=False keeps temporal order within the window: a non-shuffling
    # DistributedSampler hands rank r the strided slice indices[r::num_replicas]
    # (round-robin), so all ranks stay on the same time front and consume the
    # window in index order. Fork ctx mirrors the train path (COW-share the
    # mmap'd store instead of pickling it into every worker).
    mp_ctx = "fork" if num_workers and num_workers > 0 else None
    # In-process loading (num_workers=0) has no prefetch queue; DataLoader rejects
    # a non-None prefetch_factor outright.
    if mp_ctx is None:
        prefetch_factor = None
    dataloader = DataLoader(
        dataset=subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=DistributedSampler(subset, shuffle=False, drop_last=True),
        multiprocessing_context=mp_ctx,
        pin_memory=bool(pin_memory),
    )
    return dataloader


class StreamingWindowSampler(torch.utils.data.Sampler):
    """Per-rank sampler whose index list is swapped each window.

    Yields this rank's round-robin slice of the active window's GLOBAL anchor
    indices (into the dataset's ``_positions``). Because indices are global, a
    single DataLoader with ``persistent_workers=True`` can be reused across all
    windows: the main process re-iterates this sampler each window and ships the
    new indices to the already-forked workers, which map any global index via
    the shared mmap. No per-window worker respawn / dataset re-pickle.

    Round-robin striding (rank r gets ``indices[r::world_size]``) over the
    time-sorted window keeps every rank on the same time front; the window is
    truncated to a multiple of ``world_size`` so all ranks get equal counts
    (required for DDP collective lockstep).
    """

    def __init__(self, rank: int, world_size: int) -> None:
        self._rank: int = rank
        self._world_size: int = world_size
        self._indices: List[int] = []

    def set_window(self, global_indices, skip_samples: int = 0) -> None:
        """Install this window's per-rank index list, optionally fast-forwarding.

        ``skip_samples`` drops the first N per-rank samples from the list so the
        next ``__iter__`` starts at sample N+1 in this rank's slice. Used on
        resume to skip batches that were already trained: pass
        ``skip_samples = batch_size * batches_completed`` and the dataloader
        emits batches starting at exactly the next unseen batch.

        The skip is safe because the sample order is fully deterministic given
        (global_indices, rank, world_size): we re-derive the same per-rank list
        as the pre-crash run, just hand back a tail slice of it.
        """
        n = (len(global_indices) // self._world_size) * self._world_size
        per_rank = global_indices[:n][self._rank :: self._world_size].tolist()
        if skip_samples < 0 or skip_samples > len(per_rank):
            raise ValueError(
                f"skip_samples={skip_samples} out of [0, {len(per_rank)}] "
                f"for rank={self._rank} world_size={self._world_size}"
            )
        self._indices = per_rank[skip_samples:]

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


@gin.configurable
def make_persistent_streaming_dataloader(
    dataset: HammerToTorchDataset,
    sampler: StreamingWindowSampler,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool = False,
) -> DataLoader:
    """One reusable DataLoader for the whole streaming run. ``sampler`` is
    mutated per window via ``set_window``; workers persist across windows.

    ``pin_memory`` ($DL_PIN_MEMORY, on by default) routes each batch through
    ``Samples.pin_memory()`` on the loader's pin thread, so the per-step
    host-to-device copy is one real DMA instead of a blocking stage through a
    driver bounce buffer. Saves ~77 ms/step on 8x MI350X at bs 1024/rank
    (666 -> 589); see docs/04-technical-guides/recommendation/dlrm_v4/perf-notes.md.

    The copy itself stays blocking -- once the source is pinned it costs ~2 ms,
    and non_blocking=True measured no better while requiring event-based lifetime
    management for buffers the loader recycles.
    """
    use_workers = bool(num_workers and num_workers > 0)
    if isinstance(sampler, _ChainedWindowBatchSampler):
        # window_single_pool: one pool whose batch sampler chains the windows.
        # Batch composition is fixed by the sampler (per-window round-robin
        # slice, drop_last per window), so batch_size / drop_last are not
        # passed to the DataLoader (they are mutually exclusive with it).
        sampler.set_batch_size(int(batch_size))
        return DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if use_workers else None,
            persistent_workers=use_workers,
            multiprocessing_context="fork" if use_workers else None,
            pin_memory=bool(pin_memory),
        )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if use_workers else None,
        sampler=sampler,
        persistent_workers=use_workers,
        multiprocessing_context="fork" if use_workers else None,
        pin_memory=bool(pin_memory),
    )


@gin.configurable
def window_single_pool(enabled: int = 0) -> bool:
    """One persistent train worker pool whose batch sampler chains the windows.

    WHAT: instead of two pools that ping-pong windows (the freed pool is reset
    and re-primed for the window after next at every boundary), ONE persistent
    pool serves the whole run: its batch sampler yields window k's batches and
    then window k+1's, with the same per-window per-rank slice, the same
    drop_last per window and the same per-window cap. Window indices are still
    computed ahead in a background thread (inside the timed region). The
    prefetch depth (PREFETCH_FACTOR x NUM_WORKERS) simply continues across each
    boundary, so input production always follows consumption: no reset and no
    re-prime burst at the boundary, and half the train worker processes.
    Needs the persistent double-buffered loader. With the chained window
    stream (TP_WARM_WINDOWS=1) each window is bounded by its batch count; the
    per-window consumers (TP_WARM_WINDOWS=0 or TRAIN_PIPELINE=0) get a
    _WindowView of the shared iterator. Kill switch: DL_SINGLE_POOL=0 (two
    pools).
    """
    try:
        return int(enabled) > 0
    except (TypeError, ValueError):
        return False


def _np_copy(a):
    import numpy as _np

    return _np.array(a, copy=True)


class _ChainedWindowBatchSampler:
    """Batch sampler that serves a run's windows back to back (window_single_pool).

    ``push`` installs a window (background thread or main thread) exactly like
    ``StreamingWindowSampler.set_window`` + ``BatchSampler(drop_last=True)``:
    the window is truncated to a multiple of world_size, rank r takes
    ``indices[r::world_size]``, the first ``skip_samples`` are dropped and
    the window yields ``len // batch_size`` batches, optionally capped. The
    iterator yields each batch as the same Python int list the per-window
    sampler would have produced. While not ``live`` (pre-fork before
    RUN_START) iterators yield nothing, so no index reaches a worker.
    """

    def __init__(self, rank: int, world_size: int) -> None:
        self._rank = rank
        self._world_size = world_size
        self._bs = 0
        self._cv = threading.Condition()
        self._wins: "collections.deque" = collections.deque()
        self._done = False
        self._err: Optional[BaseException] = None
        self.live = False
        # Main-thread stall accounting (an iterator had to wait for a window
        # that was not pushed yet).
        self.waits = 0
        self.wait_s = 0.0
        # M4 speed-aware dealing (m4_partition.M4Dealer; None = stock round-robin).
        self.m4: Optional["_m4p.M4Dealer"] = None

    def set_batch_size(self, bs: int) -> None:
        self._bs = int(bs)

    def push(
        self, ts, global_indices, skip_samples: int = 0, nb_cap=None, win_index=None
    ) -> int:
        """Install the next window; returns its batch count for this rank.

        ``win_index`` is the run-local window index (position in train_ts_list);
        with M4 on (``self.m4``) the window is dealt by m4_partition, which falls
        back to exactly this stock round-robin slice whenever it has no speed
        factors for the window."""
        assert self._bs > 0, "batch size not set"
        if self.m4 is not None and win_index is not None:
            per_rank, nb = self.m4.deal(
                int(win_index), ts, global_indices, skip_samples, nb_cap, self._bs
            )
            with self._cv:
                self._wins.append((ts, per_rank, nb))
                self._cv.notify_all()
            return nb
        n = (len(global_indices) // self._world_size) * self._world_size
        per_rank = global_indices[:n][self._rank :: self._world_size]
        if skip_samples < 0 or skip_samples > len(per_rank):
            raise ValueError(
                f"skip_samples={skip_samples} out of [0, {len(per_rank)}] "
                f"for rank={self._rank} world_size={self._world_size}"
            )
        # Own copy of this rank's slice (do not pin the whole window array).
        per_rank = _np_copy(per_rank[skip_samples:])
        nb = len(per_rank) // self._bs
        if nb_cap is not None:
            nb = min(nb, max(0, int(nb_cap)))
        with self._cv:
            self._wins.append((ts, per_rank, nb))
            self._cv.notify_all()
        return nb

    def queued_batches(self) -> int:
        with self._cv:
            return sum(w[2] for w in self._wins)

    def queued_windows(self) -> int:
        with self._cv:
            return len(self._wins)

    def finish(self, err: Optional[BaseException] = None) -> None:
        with self._cv:
            self._done = True
            self._err = err
            self._cv.notify_all()

    def __iter__(self):
        return _ChainedWindowBatchIter(self)


class _ChainedWindowBatchIter:
    def __init__(self, s: "_ChainedWindowBatchSampler") -> None:
        self._s = s
        self._cur = None
        self._nb = 0
        self._b = 0

    def __iter__(self):
        return self

    def __next__(self):
        s = self._s
        if not s.live:
            raise StopIteration
        while True:
            if self._cur is not None and self._b < self._nb:
                b = self._b
                self._b += 1
                return self._cur[b * s._bs : (b + 1) * s._bs].tolist()
            t0 = None
            with s._cv:
                while not s._wins and not s._done:
                    if t0 is None:
                        t0 = time.perf_counter()
                    s._cv.wait(timeout=1.0)
                if t0 is not None:
                    s.waits += 1
                    s.wait_s += time.perf_counter() - t0
                if not s._wins:
                    if s._err is not None:
                        raise RuntimeError(
                            "[wbnd] single-pool window preparation failed"
                        ) from s._err
                    raise StopIteration
                _ts, self._cur, self._nb = s._wins.popleft()
                self._b = 0
                s._cv.notify_all()  # wake the pusher: one window fewer queued


class _WindowView:
    """One window of the chained single-pool iterator, for per-window consumers
    that iterate to StopIteration (TRAIN_PIPELINE=0 / TP_WARM_WINDOWS=0)."""

    def __init__(self, it, n: int) -> None:
        self._it = it
        self._left = int(n)

    def __iter__(self):
        return self

    def __next__(self):
        if self._left <= 0:
            raise StopIteration
        self._left -= 1
        return next(self._it)

    def drain(self) -> None:
        while self._left > 0:
            self._left -= 1
            next(self._it)


class _PrefetchingWindowLoader:
    """Double-buffered window loader for the persistent streaming path.

    Holds ``n_buffers`` pre-forked persistent worker pools that ping-pong: while
    the current window trains on one pool, the *next* window's index selection
    (``window_indices``) and first-batch prefetch are prepared on another pool
    in a background thread. By the time training advances, that window is warm,
    so the per-window reset (mask + first-batch stall) is hidden behind GPU
    compute (~0 dead time at the boundary).

    Worker pools are forked once on the main thread at the start of ``stream``;
    afterwards only iterator resets happen (no forks), so background-thread
    preparation cannot fork while other threads hold locks.
    """

    def __init__(
        self,
        dataset: "HammerToTorchDataset",
        sampler_factory,
        dl_factory,
        n_buffers: int = 2,
    ) -> None:
        self._dataset = dataset
        # window_single_pool: ONE persistent pool, windows chained in its batch
        # sampler (see _ChainedWindowBatchSampler / _stream_single).
        self._single: bool = bool(window_single_pool())
        if self._single:
            n_buffers = 1
        self._n = n_buffers
        _first = sampler_factory()
        if self._single:
            self._chain: Optional[_ChainedWindowBatchSampler] = (
                _ChainedWindowBatchSampler(_first._rank, _first._world_size)
            )
            self._samplers = [self._chain]
            # M4 (default OFF): speed-aware within-batch dealing of each global
            # batch (m4_partition). Configuration only here: nothing is read; N
            # is built on the fly after RUN_START (m4_otf).
            self._chain.m4 = _m4p.maybe_create(
                dataset.dataset, _first._rank, _first._world_size, single_pool=True
            )
        else:
            _m4p.maybe_create(
                dataset.dataset, _first._rank, _first._world_size, single_pool=False
            )
            self._chain = None
            self._samplers = [_first] + [
                sampler_factory() for _ in range(n_buffers - 1)
            ]
        self._dls = [dl_factory(s) for s in self._samplers]
        # SHIP-11 item (31) arming line: the train-pool worker count this run uses.
        logger.warning(
            "armed OK: train pool num_workers=%d prefetch_factor=%s",
            self._dls[0].num_workers, self._dls[0].prefetch_factor)
        self._iters: List[Optional[object]] = [None] * n_buffers
        # Buffers whose worker pool was forked by prestart() (empty window, RNG
        # neutral); their first _prepare replays the iterator-creation RNG draw.
        self._prestarted: List[bool] = [False] * n_buffers
        # $DL_WBND_DIAG=1: per-window prepare/activate timing lines (diagnostic,
        # off by default).
        self._diag: bool = os.environ.get("DL_WBND_DIAG", "0") == "1"
        # Teardown after RUN_STOP (teardown_close, $DL_TEARDOWN_CLOSE_POOLS):
        # the live stream's stop event (single pool) and its background
        # threads (the single-pool pusher, or the double-buffered _prepare
        # threads by buffer). Recorded once when stream() starts.
        self._stream_stop: Optional[threading.Event] = None
        self._stream_threads: List[Optional[threading.Thread]] = []
        self._closed: bool = False

    def prestart(self) -> None:
        """Fork every pool NOW with an EMPTY sampler window (DL_PRESTART).

        Called before INIT_STOP/RUN_START. ``iter()`` on a persistent-worker
        DataLoader forks the workers and primes the prefetch from the sampler;
        with an empty window the sampler yields nothing, so NO index reaches a
        worker: no sample is read, prefetched or collated. The later
        ``_prepare`` installs the real window and calls ``iter()`` again, which
        on a persistent loader is a reset of the already-running workers (no
        fork). The iterator creation draws one global CPU RNG value
        (``_base_seed``); the state is restored here and the draw is replayed in
        ``_prepare`` so the main-process RNG stream matches DL_PRESTART=0.
        """
        import numpy as _np

        rng_state = torch.get_rng_state()
        try:
            for b in range(self._n):
                if self._single:
                    # Not live: the chained sampler yields nothing, so the
                    # fork's priming sends no index.
                    self._chain.live = False
                else:
                    self._samplers[b].set_window(_np.empty(0, dtype=_np.int64))
                self._iters[b] = iter(self._dls[b])
                self._prestarted[b] = True
        finally:
            torch.set_rng_state(rng_state)

    def _prepare(self, buf: int, ts: int, skip_samples: int = 0) -> None:
        # train_window_indices() is the O(N) mask (+ uid-hash filter for the
        # holdout); numpy releases the GIL for it, so it overlaps the main
        # thread's GPU dispatch. iter() then kicks off this pool's background
        # prefetch. This is a TRAIN-only loader, so held-out eval users are
        # excluded here. `skip_samples` is non-zero only for the very first
        # window after a mid-window resume; subsequent windows always start at 0.
        t0 = time.time()
        self._samplers[buf].set_window(
            self._dataset.dataset.train_window_indices(ts),
            skip_samples=skip_samples,
        )
        t_idx = time.time()
        if self._prestarted[buf]:
            # Only ever true for the primed buffers, whose first _prepare runs on
            # the main thread (stream() priming loop) -- same thread and same
            # point in the RNG stream as the iterator creation it replaces.
            self._prestarted[buf] = False
            _draw_like_iterator_creation()
        self._iters[buf] = iter(self._dls[buf])
        if self._diag:
            t1 = time.time()
            _it = self._iters[buf]
            logger.warning(
                "[wbnd] prepare rank=%d ts=%d buf=%d idx_ms=%.1f reset_ms=%.1f "
                "prime=%d/%d t0=%.3f t1=%.3f",
                self._samplers[buf]._rank, ts, buf, (t_idx - t0) * 1e3,
                (t1 - t_idx) * 1e3, getattr(_it, "_send_idx", -1),
                getattr(_it, "_prefetch_factor", 0)
                * getattr(_it, "_num_workers", 0),
                t0, t1,
            )

    def _stream_single(
        self,
        ts_list: List[int],
        first_skip_samples: int,
        warm: bool,
        per_window_cap: Optional[int],
        first_window_trained: int,
        window_cap: Optional[int] = None,
    ):
        """window_single_pool: one persistent pool, windows chained in its batch
        sampler. Yields the same (ts, iterator, num_batches) triples as the
        double-buffered path. With ``warm`` (chained warm stream) the iterator
        is the same object for every window and the caller bounds each window
        by num_batches (_WarmWindowStream); without it each window gets a
        _WindowView of that iterator which ends after the window's batches.
        The cold-page-cache read-ahead (``window_cap``, $DL_COLD_PREFETCH_STREAM)
        is started, paced and stopped exactly as in the double-buffered path."""
        n = len(ts_list)
        chain = self._chain
        dl = self._dls[0]
        rank = chain._rank
        cap = per_window_cap if per_window_cap else None

        def nb_cap(i: int) -> Optional[int]:
            # Same bound as _WarmWindowStream._open_next.
            if cap is None:
                return None
            return max(0, cap - (first_window_trained if i == 0 else 0))

        # Every rank (rank=R/W): the per-rank arming check reads these lines.
        logger.warning(
            "armed OK: [wbnd] single train pool rank=%d/%d: %d windows chained "
            "in one persistent pool of %d workers, depth %s (DL_SINGLE_POOL=0 "
            "restores the double-buffered pools)",
            rank, chain._world_size, n, dl.num_workers,
            dl.prefetch_factor * max(1, dl.num_workers)
            if dl.prefetch_factor else 0,
        )
        _cold_prefetch.release()
        _cp_on = bool(_cold_prefetch.mode()) and not _cold_prefetch._THREADS
        _cp_args = (
            (self._dataset.dataset, ts_list, rank, chain._world_size,
             int(chain._bs or 1), first_skip_samples, window_cap)
            if _cp_on
            else ()
        )
        _cp_thread: Optional[threading.Thread] = None
        _cp_early = bool(self._prestarted[0])
        if _cp_on and _cp_early:
            _cp_thread = _cold_prefetch.start(*_cp_args)
        stop = threading.Event()
        # teardown_close() sets it too: this generator can outlive the loop.
        self._stream_stop = stop
        try:
            counts: List[Optional[int]] = [None] * n
            pushed = [threading.Event() for _ in range(n)]
            depth = int(dl.prefetch_factor or 0) * max(1, int(dl.num_workers or 0))
            if chain.m4 is not None and not chain.m4.activate(depth, ts_list):
                chain.m4 = None
            # The double-buffered loader installs its first two windows on the
            # main thread at stream start; do the same, and keep going until the
            # first prime (depth batches) can be served without waiting: no
            # other thread may run across the fork the first iter() makes
            # without DL_PRESTART, and the prime must never block on a window
            # nobody pushes yet.
            k0 = 0
            while k0 < n and (k0 < 2 or chain.queued_batches() < depth):
                counts[k0] = chain.push(
                    ts_list[k0],
                    self._dataset.dataset.train_window_indices(ts_list[k0]),
                    skip_samples=first_skip_samples if k0 == 0 else 0,
                    nb_cap=nb_cap(k0),
                    win_index=k0,
                )
                pushed[k0].set()
                k0 += 1
            if k0 == n:
                chain.finish()
            # CPU RNG: the double-buffered loader creates (or, pre-started,
            # replays) one iterator per pool it primes here, i.e. min(n, 2)
            # draws (one when the run has a single train window); keep the
            # main-process RNG stream identical.
            chain.live = True
            n_draws = min(n, 2)
            if self._prestarted[0]:
                self._prestarted[0] = False
                for _ in range(n_draws):
                    _draw_like_iterator_creation()
                it = iter(dl)  # reset of the pre-forked pool + full prime
            else:
                it = iter(dl)  # creates the pool (one draw) + full prime
                for _ in range(n_draws - 1):
                    _draw_like_iterator_creation()
            self._iters[0] = it
            if _cp_on and not _cp_early:
                _cp_thread = _cold_prefetch.start(*_cp_args)
            _dlstart.mark("train_pools_ready")

            def _pusher() -> None:
                try:
                    for j in range(k0, n):
                        with chain._cv:
                            # Stay >= 2 windows and >= 2 x depth batches ahead
                            # of the sampler (which runs ~depth batches ahead
                            # of the trainer), so the main thread never waits
                            # for indices.
                            while not stop.is_set() and len(chain._wins) >= 2 and (
                                sum(w[2] for w in chain._wins) >= 2 * depth
                            ):
                                chain._cv.wait(timeout=0.5)
                        if stop.is_set():
                            return
                        t0 = time.time()
                        idx = self._dataset.dataset.train_window_indices(ts_list[j])
                        counts[j] = chain.push(
                            ts_list[j], idx, 0, nb_cap(j), win_index=j
                        )
                        pushed[j].set()
                        if self._diag:
                            logger.warning(
                                "[wbnd] push rank=%d ts=%d nb=%d idx_ms=%.1f t=%.3f",
                                rank, ts_list[j], counts[j],
                                (time.time() - t0) * 1e3, time.time(),
                            )
                    chain.finish()
                except BaseException as e:  # surfaced via the sampler / generator
                    chain.finish(e)
                    raise

            th = threading.Thread(target=_pusher, daemon=True)
            th.start()
            self._stream_threads = [th]
            view: Optional[_WindowView] = None
            for i in range(n):
                while not pushed[i].wait(timeout=1.0):
                    if chain._err is not None:
                        raise RuntimeError(
                            f"[wbnd] single-pool preparation of train window "
                            f"{ts_list[i]} failed"
                        ) from chain._err
                    if not th.is_alive() and not pushed[i].is_set():
                        raise RuntimeError(
                            f"[wbnd] single-pool pusher exited before window {ts_list[i]}"
                        )
                if self._diag:
                    logger.warning(
                        "[wbnd] activate rank=%d ts=%d buf=0 released=%d catchup=0 "
                        "cap=%d t=%.3f sampler_waits=%d wait_s=%.3f",
                        rank, ts_list[i], getattr(it, "_send_idx", -1), depth,
                        time.time(), chain.waits, chain.wait_s,
                    )
                if warm:
                    yield ts_list[i], it, counts[i]
                else:
                    # Per-window consumer (TRAIN_PIPELINE=0 / TP_WARM_WINDOWS=0):
                    # it iterates each window to StopIteration, so hand it a view
                    # of the shared iterator that ends after this window's
                    # (capped) batches; batches an early-broken window left
                    # behind are dropped first, as the per-window pools did.
                    if view is not None:
                        view.drain()
                    view = _WindowView(it, counts[i])
                    yield ts_list[i], view, counts[i]
                # The consumer is back for the next window: window i has been
                # pulled completely (read-ahead pacing; one list store).
                if _cp_thread is not None:
                    _cold_prefetch.note_window_done(i + 1)
        finally:
            stop.set()
            with chain._cv:
                chain._cv.notify_all()
            if _cp_thread is not None:
                _cold_prefetch.stop()
            if chain.m4 is not None:
                chain.m4.stream_closed()
                chain.m4.summary()

    def stream(
        self,
        ts_list: List[int],
        first_skip_samples: int = 0,
        window_cap: Optional[int] = None,
        warm: bool = False,
        per_window_cap: Optional[int] = None,
        first_window_trained: int = 0,
    ):
        """Stream (ts, iterator, num_batches) triples. `first_skip_samples` is
        applied ONLY to the first ts in ``ts_list`` (the mid-window-resumed
        window); every subsequent window starts at sample 0 of its own per-rank
        list.

        ``num_batches`` is this window's per-rank batch count, read off the
        DataLoader before the buffer can be recycled. It is rank-identical
        (``set_window`` truncates to a multiple of world_size and ``drop_last``
        makes it ``len(sampler) // batch_size``) and already reflects any
        ``skip_samples``, so a caller can bound a window by count instead of
        waiting for StopIteration -- see ``_WarmWindowStream``.

        ``window_cap`` (the caller's per-window batch cap, num_train_batches)
        only tells the cold-page-cache read-ahead how many batches of each
        window are pulled; the stream itself is unchanged by it.

        ``warm=True``: the caller is the chained warm stream
        (``_WarmWindowStream``), which bounds each window by ``num_batches``.
        ``warm``, ``per_window_cap`` and ``first_window_trained`` are only used
        by the single-pool mode: it then hands out its one shared iterator
        instead of a per-window view, and its sampler must stop each window at
        the same (capped) count the chained window stream pulls. The
        double-buffered pools ignore them: every pool reset sends its full
        prefetch depth at once (one-shot prime).
        """
        n = len(ts_list)
        if n == 0:
            return
        if self._single:
            yield from self._stream_single(
                ts_list, first_skip_samples, warm, per_window_cap,
                first_window_trained, window_cap=window_cap,
            )
            return
        threads: List[Optional[threading.Thread]] = [None] * self._n
        self._stream_threads = threads  # teardown_close() joins the live ones
        # Sampler release point (after RUN_START): the priming below hands the
        # workers their first real indices. $DL_COLD_PREFETCH (default off)
        # readaheads this rank's first batches' byte ranges in parallel from
        # here on; it starts after the priming when the priming still has to
        # fork pools (no extra thread may run across a fork).
        _cold_prefetch.release()
        # Off (default): no args are built and nothing starts. One-shot: only
        # the first stream() of the run (the cold first window) is prefetched;
        # $DL_COLD_PREFETCH_STREAM=1 keeps it reading ahead of this stream.
        _cp_on = bool(_cold_prefetch.mode()) and not _cold_prefetch._THREADS
        _cp_args = (
            (
                self._dataset.dataset,
                ts_list,
                self._samplers[0]._rank,
                self._samplers[0]._world_size,
                int(self._dls[0].batch_size or 1),
                first_skip_samples,
                window_cap,
            )
            if _cp_on
            else ()
        )
        _cp_thread: Optional[threading.Thread] = None
        _cp_early = all(self._prestarted[: min(self._n, n)])
        if _cp_on and _cp_early:
            _cp_thread = _cold_prefetch.start(*_cp_args)
        try:
            # Prime the first n_buffers windows on the main thread (forks all
            # pools, or -- after prestart() -- resets the already-forked ones).
            for b in range(min(self._n, n)):
                skip = first_skip_samples if b == 0 else 0
                self._prepare(b, ts_list[b], skip_samples=skip)
            if _cp_on and not _cp_early:
                _cp_thread = _cold_prefetch.start(*_cp_args)
            _dlstart.mark("train_pools_ready")
            for i in range(n):
                buf = i % self._n
                if threads[buf] is not None:
                    threads[buf].join()
                    threads[buf] = None
                if self._diag:
                    _it = self._iters[buf]
                    logger.warning(
                        "[wbnd] activate rank=%d ts=%d buf=%d released=%d "
                        "catchup=0 cap=%d t=%.3f",
                        self._samplers[buf]._rank, ts_list[i], buf,
                        getattr(_it, "_send_idx", -1),
                        getattr(_it, "_prefetch_factor", 0)
                        * getattr(_it, "_num_workers", 0),
                        time.time(),
                    )
                yield ts_list[i], self._iters[buf], len(self._dls[buf])
                # The consumer is back for the next window: window i has been
                # pulled completely (read-ahead pacing; one list store).
                if _cp_thread is not None:
                    _cold_prefetch.note_window_done(i + 1)
                # This pool is now free; prefetch the window n_buffers ahead.
                # No skip on subsequent windows — only the first prepared window
                # carries `first_skip_samples`.
                j = i + self._n
                if j < n:
                    th = threading.Thread(
                        target=self._prepare, args=(buf, ts_list[j]), daemon=True
                    )
                    th.start()
                    threads[buf] = th
        finally:
            if _cp_thread is not None:
                _cold_prefetch.stop()

    def teardown_close(
        self, timeout_s: float = 5.0, warn: Optional[Callable[[str], None]] = None
    ) -> int:
        """Teardown after RUN_STOP ($DL_TEARDOWN_CLOSE_POOLS; pool_close.py):
        stop this loader's background threads, then shut its worker pools
        down. Returns the number of pools this call shut down.

        1. Single pool: set the stream's stop event and wake the pusher (what
           the stream generator's ``finally`` does, but that generator can
           outlive the loop), stop the cold-page-cache read-ahead, then join
           the pusher. Double-buffered pools: join the background
           ``_prepare`` threads; a pool whose reset is still running past the
           bound is left open (warned).
        2. ``_shutdown_workers()`` on each live iterator: torch joins the
           pin-memory thread first, then the workers.
        3. ``_iters[b]`` and the DataLoader's cached ``_iterator`` -> None.

        Main thread, once the stream's consumer is gone. A mid-window stop can
        leave the pusher waiting on a full sampler queue (>= 2 windows and
        >= 2 x depth batches) and up to depth batches outstanding in the pool;
        both are covered. Every wait is bounded by ``timeout_s``; failures go
        to ``warn`` and are never raised. A second call is a no-op.
        """
        from generative_recommenders.dlrm_v4.train import pool_close as _pool_close

        if self._closed:
            return 0
        self._closed = True
        say = warn if warn is not None else logger.warning
        busy: Set[int] = set()
        try:
            stop = self._stream_stop
            if stop is not None:
                stop.set()
                chain = self._chain
                if chain is not None and chain._cv.acquire(timeout=timeout_s):
                    try:
                        chain._cv.notify_all()
                    finally:
                        chain._cv.release()
            if stop is not None or self._stream_threads:
                _cold_prefetch.stop()
            for b, th in enumerate(list(self._stream_threads)):
                if th is None:
                    continue
                th.join(timeout_s)
                if th.is_alive():
                    if not self._single:
                        busy.add(b)
                    say(
                        f"[teardown] train loader thread {th.name} still running "
                        f"after {timeout_s:.1f} s; continuing"
                    )
            # M4: stop the per-rank partition helper (bounded; never raises)
            _m4 = getattr(self._chain, "m4", None) if self._chain is not None else None
            if _m4 is not None:
                _m4.close(timeout_s)
        except Exception as e:
            say(f"[teardown] stopping the train stream failed: {type(e).__name__}: {e}")
        n = 0
        for b in range(self._n):
            if b in busy:
                say(f"[teardown] train pool {b} left open: its window reset still runs")
                continue
            try:
                dl = self._dls[b]
                its = (self._iters[b], getattr(dl, "_iterator", None))
                for k, it in enumerate(its):
                    if it is not None and (k == 0 or it is not its[0]):
                        n += _pool_close.shutdown_iterator(it, timeout_s, say)
                self._iters[b] = None
                if getattr(dl, "_iterator", None) is not None:
                    dl._iterator = None
            except Exception as e:
                say(f"[teardown] train pool {b} close failed: {type(e).__name__}: {e}")
        return n


class _WarmWindowStream:
    """Chains per-window iterators into one uninterrupted batch stream.

    The TorchRec pipeline keeps two batches in flight. Handing it a fresh
    iterator per window makes it *drain* at every seam -- the window's last two
    batches train with nothing queued behind them -- and then refill from cold
    on the next window, paying the H2D copy and the embedding input_dist with an
    idle compute stream. On interior windows that is ~0.33 ms/step amortized; at
    START_TS=0, where a window can be shorter than a single step, it happens
    almost every step and eats most of the pipeline's benefit.

    Chaining removes the seam entirely: when the current window's iterator runs
    out, this object rolls onto the next one, which ``_PrefetchingWindowLoader``
    has already opened on a *separate* worker pool in a background thread. So the
    pipeline pulls the next window's first batches while the current window's
    tail is still training.

    Windows are still trained one at a time. ``advance()`` hands back the next
    window's ``(ts, num_batches)`` and the caller trains exactly that many
    batches, so per-window metrics, eval, checkpointing and the boundary barrier
    keep their current meaning. Two properties make that safe:

      * The counts are rank-identical, so every rank leaves a window on the same
        batch and the collectives stay in lockstep.
      * Pulling runs ahead of training, so ``advance()`` may be called for a
        window whose batches are already queued. Trained *order* is unchanged --
        only the prefetch depth crosses the boundary.

    Because pulling runs ahead, a short window can be opened before the previous
    one finishes training; ``_queue`` holds those opened-but-not-yet-trained
    windows so ``advance()`` still returns them in order.

    ``per_window_cap`` (num_train_batches) truncates every window. It is applied
    to the *pull* limit, not just the train loop: without that, the pipeline
    would read ahead into batches the capped window never trains and those reads
    would have to be thrown away — and once the cap lands within the prefetch
    depth of the window's real end, the discarded reads include the NEXT window's
    first batch, which a chained iterator cannot un-pull. Limiting pulls instead
    means every batch read is a batch trained, so nothing is ever discarded and
    the trained set matches the uncapped-iterator path exactly.

    ``first_window_trained`` is the mid-window resume offset. The cap counts
    batches from the window's own batch 0, so a window re-entered at batch K only
    has ``cap - K`` left even though its (already shortened) iterator holds more.
    """

    def __init__(
        self,
        window_stream,
        per_window_cap: Optional[int] = None,
        first_window_trained: int = 0,
    ) -> None:
        self._gen = iter(window_stream)
        self._cap = per_window_cap if per_window_cap else None
        self._already_trained = first_window_trained
        self._active = None
        self._left: Optional[int] = None
        self._queue: "collections.deque" = collections.deque()
        self._pulls: int = 0
        # $DLQ_LOG_EVERY=N (diagnostic, default 0 = off): every N pulls log the
        # trainer's wait in next() and the active pool's in-flight depth.
        try:
            self._dlq_every = int(os.environ.get("DLQ_LOG_EVERY", "0") or 0)
        except ValueError:
            self._dlq_every = 0
        self._dlq_n = 0
        self._dlq_w: List[float] = []
        self._dlq_q: List[int] = []

    def _open_next(self) -> None:
        """Open the next window for pulling. Raises StopIteration at run end."""
        ts, it, num_batches = next(self._gen)
        if self._cap is not None:
            num_batches = min(num_batches, max(0, self._cap - self._already_trained))
        # Only the first window carries the resume offset.
        self._already_trained = 0
        self._active = it
        self._left = num_batches
        self._queue.append((ts, num_batches))

    def _dlq_note(self, wait_ms: float) -> None:
        it = self._active
        sent = getattr(it, "_send_idx", None)
        inflight = (sent - getattr(it, "_rcvd_idx", 0)) if sent is not None else -1
        self._dlq_n += 1
        self._dlq_w.append(wait_ms)
        self._dlq_q.append(inflight)
        if len(self._dlq_w) >= self._dlq_every:
            w = sorted(self._dlq_w)
            q = sorted(self._dlq_q)
            rk = (
                dist.get_rank()
                if dist.is_available() and dist.is_initialized()
                else os.environ.get("LOCAL_RANK", "?")
            )
            logger.warning(
                "[dlq] rank=%s pull=%d left_in_window=%d wait_ms sum=%.1f p50=%.2f "
                "p90=%.2f max=%.1f n_gt5ms=%d inflight min=%d p50=%d ready_info=%d",
                rk, self._dlq_n, self._left or 0, sum(w), w[len(w) // 2],
                w[int(len(w) * 0.9)], w[-1], sum(1 for x in w if x > 5.0),
                q[0], q[len(q) // 2],
                len(getattr(it, "_task_info", {}) or {}) if inflight >= 0 else -1,
            )
            self._dlq_w.clear()
            self._dlq_q.clear()

    def advance(self):
        """(ts, num_batches) of the next window to TRAIN. Raises StopIteration
        once every window has been handed out."""
        while not self._queue:
            self._open_next()
        return self._queue.popleft()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self._active is None or self._left == 0:
                # Raises StopIteration at the end of the run, which drains the
                # pipeline exactly as the per-window iterators used to.
                self._open_next()
            _dq_t0 = time.perf_counter() if self._dlq_every > 0 else 0.0
            try:
                batch = next(self._active)
            except StopIteration:
                # The window ran dry before delivering its declared count, so
                # `advance()` handed the train loop a bound the data cannot meet
                # and the loop would train the NEXT window's batches under this
                # window's ts. len(DataLoader) is exact for a sized sampler with
                # drop_last, so this should be unreachable; shout rather than
                # quietly mis-attribute.
                if self._left:
                    logger.warning(
                        "[pipeline] window stream ended %d batch(es) short of its "
                        "declared count; window batch attribution may shift",
                        self._left,
                    )
                self._active = None
                continue
            self._left -= 1
            if self._dlq_every > 0:
                self._dlq_note((time.perf_counter() - _dq_t0) * 1000.0)
            # Consumer position for the cold-page-cache read-ahead (one int add).
            _cold_prefetch.note_pull()
            if self._pulls < 3:
                # Startup timeline only: wall time of the first three batches
                # the pipeline pulls (dl_prestart report splits the first
                # progress() into data wait vs the rest).
                self._pulls += 1
                _dlstart.mark(f"pull{self._pulls}")
            return batch


# Built by prestart_streaming_loaders() before INIT_STOP/RUN_START and consumed
# (exactly once) by streaming_train_eval_loop. None = legacy in-loop build.
_PRESTART_BUNDLE: Optional[Dict[str, Any]] = None


def _take_prestart_bundle() -> Optional[Dict[str, Any]]:
    global _PRESTART_BUNDLE
    bundle, _PRESTART_BUNDLE = _PRESTART_BUNDLE, None
    return bundle


@gin.configurable
def prestart_streaming_loaders(
    hstu_config: DlrmHSTUConfig,
    embedding_table_configs: Dict[str, EmbeddingConfig],
    rank: int,
    world_size: int,
    # $DL_PRESTART kill switch (gin dlps/env_int). Mirrors of the loop's own
    # gin bindings (same macros) so the pre-start builds exactly the pools the
    # loop would build.
    enabled: int = 0,
    persistent_loader: int = 0,
    double_buffer: int = 0,
    eval_num_workers: int = -1,
    eval_every_n_windows: int = 1,
    eval_every_data_pct: float = 0.0,
) -> bool:
    """DataLoader pre-start: move the train/eval DataLoader INITIALIZATION ahead
    of INIT_STOP/RUN_START, never the data.

    Before RUN_START this builds the dataset object (mmaps + metadata, the same
    construction the loop does), the double-buffered train pools and the
    persistent eval pool, and forks every worker -- each with an EMPTY sampler
    window, so no index is sent to a worker and no training or eval sample is
    read, prefetched or collated before RUN_START. ``iter()`` on these
    persistent loaders later is a reset of running workers, not a fork. The
    global CPU RNG state is saved/restored around the forks and the draws are
    replayed at the loop's original iterator-creation points, so the run is
    RNG-identical to DL_PRESTART=0. Proof of compliance: the
    ``[dlstart] timeline`` line (worker-ready < 0 < first read, rel. RUN_START).

    Returns True when the bundle was built. Only the production double-buffered
    persistent path is supported; anything else stays on the legacy build.
    """
    global _PRESTART_BUNDLE
    _cold_prefetch.set_prestart_level(enabled)
    if not enabled:
        if rank == 0:
            logger.warning(
                "[dlstart] DL_PRESTART=0: dataloader pools build/fork after RUN_START (legacy)"
            )
        return False
    if not (persistent_loader and double_buffer):
        if rank == 0:
            logger.warning(
                "[dlstart] DL_PRESTART=1 ignored: needs persistent_loader=1 and "
                "double_buffer=1 (got %s/%s); legacy in-loop build",
                persistent_loader,
                double_buffer,
            )
        return False
    import numpy as _np

    _dlstart.mark("prestart_begin")
    # Items (32)/(61) fast-batch switches, resolved before the pools below fork
    # (the workers inherit them; the eval pool's wrapper carries its own).
    fast_batch_pools()
    # Item (56) $DL_FAST_BATCH_ONESTORE, installed in this (the trainer)
    # process before the pools below fork (their workers lay out the fast
    # batches) and start their pin threads (which pin them).
    fast_batch_onestore()
    # Item (55) $DL_PIN_FREE_SHM, installed in this (the trainer) process
    # before the pools below start their pin threads.
    pin_free_shm_release()
    dataset_class, kwargs = get_dataset()
    kwargs["embedding_config"] = embedding_table_configs
    dataset = HammerToTorchDataset(
        dataset=dataset_class(hstu_config=hstu_config, is_inference=False, **kwargs)
    )
    train_ds = _ReadProbedDataset(dataset, _dlstart.new_read_slot("train"))
    prefetcher = _PrefetchingWindowLoader(
        dataset=dataset,
        sampler_factory=lambda: StreamingWindowSampler(rank, world_size),
        dl_factory=lambda s: make_persistent_streaming_dataloader(
            dataset=train_ds, sampler=s
        ),
    )
    # Only PERSISTENT worker pools can be pre-forked: with num_workers=0 every
    # iter() builds a fresh iterator (no reset), so there is nothing to start
    # early and the RNG replay would double-count. Stay on the legacy build.
    if not prefetcher._dls[0].persistent_workers:
        if rank == 0:
            logger.warning(
                "[dlstart] DL_PRESTART=1 ignored: train pools have no persistent "
                "workers (NUM_WORKERS=0); legacy in-loop build"
            )
        return False
    eval_enabled = eval_every_n_windows > 0 or bool(
        eval_every_data_pct and eval_every_data_pct > 0
    )
    eval_sampler: Optional[StreamingWindowSampler] = None
    eval_dl: Optional[DataLoader] = None
    rng_state = torch.get_rng_state()
    try:
        # Same fork order as the loop: eval pool first, then the train pools.
        if eval_enabled and hasattr(dataset.dataset, "eval_holdout_indices"):
            eval_sampler = StreamingWindowSampler(rank, world_size)
            eval_dl = make_persistent_streaming_dataloader(
                dataset=_eval_dataset(dataset, _dlstart.new_read_slot("eval")),
                sampler=eval_sampler,
                **resolve_eval_dataloader_kwargs(eval_num_workers),
            )
            if eval_dl.persistent_workers:
                eval_sampler.set_window(_np.empty(0, dtype=_np.int64))
                iter(eval_dl)  # forks the eval workers; the empty window sends nothing
            else:
                # EVAL_NUM_WORKERS=0: nothing to pre-fork; the loop builds it.
                eval_sampler, eval_dl = None, None
        prefetcher.prestart()
    finally:
        torch.set_rng_state(rng_state)
    _dlstart.mark("prestart_workers_ready")
    _PRESTART_BUNDLE = {
        # CPU RNG state the worker base seeds were drawn from (loop-entry guard).
        "rng_state": rng_state,
        "dataset": dataset,
        "prefetcher": prefetcher,
        "eval_sampler": eval_sampler,
        "eval_dl": eval_dl,
    }
    if rank == 0:
        logger.warning(
            "armed OK: [dlstart] DL_PRESTART=1: dataset + %s%s forked "
            "before RUN_START with empty sampler windows (no sample read); "
            "first-read stamps in the [dlstart] timeline line",
            (
                "1 chained train pool (single pool)"
                if getattr(prefetcher, "_single", False)
                else f"{prefetcher._n} train pools"
            ),
            " + eval pool" if eval_dl is not None else "",
        )
    pools: Dict[str, HammerToTorchDataset] = {"train": train_ds}
    if eval_dl is not None:
        pools["eval"] = eval_dl.dataset
    _log_fast_batch_pools(rank, pools)
    _log_fast_batch_onestore(rank, pools)
    pin_dls = list(prefetcher._dls) + ([eval_dl] if eval_dl is not None else [])
    _log_pin_free_shm(rank, pinned=any(bool(dl.pin_memory) for dl in pin_dls))
    return True



# Pinned-pool pre-grow (ttt-fix). Every DataLoader pool pins each batch into one block of torch's caching host
# allocator (Samples.pin_memory -> _pin_one_storage: torch.empty(nbytes, pin_memory=True), rounded up to a power of
# two), and a pool's blocks are first allocated by its pin thread when it first fills its prefetch queue:
#   * train pool: num_workers x prefetch_factor batches in flight (6 x 8 at yambda-5b), each in the 128 MiB class at
#     BATCH_SIZE 1024 -> 48-49 hipHostMalloc(128 MiB) in the first ~3.5 s after RUN_START. On ROCm 10.2 each takes
#     40-93 ms on the pin thread, and the trainer's own HIP calls in steps 1-5 slow down meanwhile (synchronous
#     memcpys up to 82 ms, stream syncs up to 135 ms; full-run probe 383263): the first steps run 2.0-2.4 s behind
#     v26.3 on the same node (383553 / 383554).
#   * eval pool: eval_num_workers x prefetch_factor (8 x 8) batches, each in the 256 MiB class -> 63-64
#     hipHostMalloc(256 MiB) in the first eval (55-142 ms each, 4.8-5.3 s per rank while the eval waits for its
#     batches; first eval 17.3-17.7 s against 14.2-14.5 s on v26.3).
# Here, after every pool is forked (no pinned mapping crosses a fork) and before RUN_START, the same blocks are
# allocated once and returned to the allocator's free list, where the pools' requests of those classes find them.
# Counts are the pools' own prefetch depths (+1 for the batch being consumed); the classes scale with the batch size
# from the measured 128 / 256 MiB at 1024 samples. Host memory only: no data, no RNG, no device work.
# Kill switch DL_PIN_PREWARM=0. Before pinning, _pin_free_check makes sure every NUMA node has the memory FREE
# (reclaiming page cache if needed) and otherwise fails the run here: pinned memory cannot be evicted, so pinning on
# top of a nearly full page cache makes the kernel reclaim during training, which stalls the ranks (yambda_5b.gin (121)).
_PIN_PREWARM_ENV: str = "DL_PIN_PREWARM"
_PIN_PREWARM_CLASS_AT_1024: int = 256 << 20  # eval batch
_PIN_PREWARM_TRAIN_CLASS_AT_1024: int = 128 << 20  # train batch


def _pin_class(at_1024: int, batch_size: int) -> int:
    return 1 << int(math.ceil(math.log2(max(1, (at_1024 * max(1, batch_size)) // 1024))))


def _pool_depth(dl: Any) -> Tuple[int, int, Any]:
    workers = max(1, int(getattr(dl, "num_workers", 0) or 0))
    return int(getattr(dl, "prefetch_factor", 0) or 0) * workers, workers, getattr(dl, "prefetch_factor", None)


_PIN_HEADROOM_ENV: str = "DL_PIN_PREWARM_HEADROOM_GIB"  # free memory to keep per NUMA node beyond the pin (128)


def _numa_free_bytes() -> Dict[int, Tuple[int, int, int]]:
    """{node: (MemFree, FilePages, Shmem)} in bytes from /sys/devices/system/node/node*/meminfo."""
    out: Dict[int, Tuple[int, int, int]] = {}
    base = "/sys/devices/system/node"
    try:
        names = os.listdir(base)
    except OSError:
        return out
    for d in names:
        if not (d.startswith("node") and d[4:].isdigit()):
            continue
        vals: Dict[str, int] = {}
        try:
            with open(f"{base}/{d}/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 4:
                        vals[parts[2].rstrip(":")] = int(parts[3]) * 1024
        except (OSError, ValueError):
            continue
        out[int(d[4:])] = (vals.get("MemFree", 0), vals.get("FilePages", 0), vals.get("Shmem", 0))
    return out


_PIN_RECLAIM_ENV: str = "DL_PIN_PREWARM_RECLAIM"  # 1 (default): reclaim page cache when a node is short; 0: never


def _pin_reclaim(target_free_bytes: int) -> str:
    """Run the page-cache balloon (host_mem_reclaim) once per node; return its report line.

    One rank per node does it: ranks of a node share the container's /tmp, so the first to take the lock runs the
    balloon in a subprocess (its memory is returned when it exits) and the others wait, then re-check.
    """
    import fcntl
    import subprocess
    import sys

    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    with open("/tmp/dlrm_pin_reclaim.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            G = 1 << 30
            if all(f >= target_free_bytes for f, _, _ in _numa_free_bytes().values()):
                return "already reclaimed by another rank"
            env = dict(os.environ)
            env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
            r = subprocess.run(
                [sys.executable, "-m", "generative_recommenders.dlrm_v4.train.host_mem_reclaim",
                 "--target-free-gib", f"{target_free_bytes / G + 8:.0f}"],
                env=env, capture_output=True, text=True, timeout=1800)
            return (r.stdout.strip().splitlines() or [f"rc={r.returncode} {r.stderr.strip()[-300:]}"])[-1]
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _pin_free_check(rank: int, per_rank_bytes: int, reclaim: Optional[int] = None,
                    headroom_gib: Optional[int] = None) -> None:
    """Make sure every NUMA node has truly FREE memory for its share of the pre-grow (before RUN_START).

    MemFree, not MemAvailable: page cache counts as 'available', but pinning on top of it makes the kernel reclaim
    during training, which stalls the ranks. Every node is checked against an even share (where hipHostMalloc places
    pages is not assumed). If a node is short, the page cache is reclaimed first (unprivileged, untimed; see
    host_mem_reclaim.py); if memory is still short after that, it is held by processes, not cache, and the run is
    terminated here, before RUN_START, with the numbers, instead of stalling later (yambda_5b.gin item (121)).
    """
    nodes = _numa_free_bytes()
    if not nodes:
        return
    local = int(os.environ.get("LOCAL_WORLD_SIZE", "0") or 0) or max(1, torch.cuda.device_count())
    G = 1 << 30
    if headroom_gib is None:
        headroom_gib = int(float(os.environ.get(_PIN_HEADROOM_ENV, "512")))
    head = int(headroom_gib) * G
    share = (per_rank_bytes * local + len(nodes) - 1) // len(nodes)
    need = share + head

    def _state(ns: Dict[int, Tuple[int, int, int]]) -> str:
        return "; ".join(f"node{n} free {f / G:.0f} GiB, page cache {(c - sh) / G:.0f} GiB"
                         for n, (f, c, sh) in sorted(ns.items()))

    if rank == 0:
        logger.warning("[pin-prewarm] free-memory check: need %.0f GiB pinned per NUMA node (%d local ranks x %.1f GiB) "
                       "+ %.0f GiB headroom; %s", share / G, local, per_rank_bytes / G, head / G, _state(nodes))
    short = [n for n, (f, _, _) in sorted(nodes.items()) if f < need]
    if not short:
        return
    if reclaim is None:
        reclaim = 0 if os.environ.get(_PIN_RECLAIM_ENV, "1").strip().lower() in ("0", "off", "false", "no") else 1
    reclaim = bool(int(reclaim))
    if reclaim:
        t0 = time.perf_counter()
        report = _pin_reclaim(need)
        nodes = _numa_free_bytes()
        short = [n for n, (f, _, _) in sorted(nodes.items()) if f < need]
        logger.warning("[pin-prewarm] reclaimed page cache in %.0f s [rank %d]: %s -> %s",
                       time.perf_counter() - t0, rank, report, _state(nodes))
        if not short:
            return
    msg = ("[pin-prewarm] NUMA node(s) %s have too little FREE memory for the pinned pre-grow%s: need %.0f GiB pinned + "
           "%.0f GiB headroom per node; %s. %s Pinning on top of it makes training stall mid-run, so the run stops "
           "here, before RUN_START. Free that memory (other jobs or containers on this node), run on another node, or "
           "disable the pre-grow with %s=0."
           % (",".join(map(str, short)), " even after reclaiming page cache" if reclaim else "", share / G, head / G,
              _state(nodes),
              "That memory is held by processes, not page cache." if reclaim else
              "Page-cache reclaim is off (%s=0); set it to 1 to free cached memory first." % _PIN_RECLAIM_ENV,
              _PIN_PREWARM_ENV))
    raise RuntimeError(msg)


@gin.configurable
def pin_pool_prewarm(
    rank: int,
    enabled: Optional[int] = None,
    reclaim: Optional[int] = None,
    headroom_gib: Optional[int] = None,
) -> None:
    """yambda_5b.gin item (121): pinned-pool pre-grow (T27) with its free-memory check. enabled / reclaim /
    headroom_gib come from gin (dlppw/dlppr/dlpph env_int: $DL_PIN_PREWARM, $DL_PIN_PREWARM_RECLAIM,
    $DL_PIN_PREWARM_HEADROOM_GIB win when set); None (no gin binding: tests, tools) falls back to the env with the
    code defaults 1 / 1 / 512."""
    if enabled is None:
        raw = os.environ.get(_PIN_PREWARM_ENV, "1").strip().lower()
        enabled = 0 if raw in ("0", "off", "false", "no") else 1
    if not int(enabled):
        if rank == 0:
            logger.warning("[pin-prewarm] OFF (%s=0)", _PIN_PREWARM_ENV)
        return
    bundle = _PRESTART_BUNDLE
    if not bundle or not torch.cuda.is_available():
        if rank == 0:
            logger.warning("[pin-prewarm] not run: no pre-forked pools")
        return
    plan: List[Tuple[str, int, int, str]] = []  # (pool, count, bytes, why)
    eval_dl = bundle.get("eval_dl")
    bs = int(getattr(eval_dl, "batch_size", 0) or 1024) if eval_dl is not None else 1024
    pre = bundle.get("prefetcher")
    tdepth = 0
    for dl in list(getattr(pre, "_dls", None) or []):
        if bool(getattr(dl, "pin_memory", False)):
            tdepth += _pool_depth(dl)[0]
    if tdepth > 0:
        tbs = 0
        for dl in list(getattr(pre, "_dls", None) or []):
            tbs = int(getattr(dl, "batch_size", 0) or 0) or tbs
        plan.append(("train", tdepth + 1, _pin_class(_PIN_PREWARM_TRAIN_CLASS_AT_1024, tbs or bs),
                     f"depth {tdepth}"))
    if eval_dl is not None and bool(getattr(eval_dl, "pin_memory", False)):
        depth, workers, pf = _pool_depth(eval_dl)
        plan.append(("eval", depth + 1, _pin_class(_PIN_PREWARM_CLASS_AT_1024, bs),
                     f"depth {depth} = {workers} workers x prefetch {pf}"))
    if not plan:
        if rank == 0:
            logger.warning("[pin-prewarm] not run: no pinned pool")
        return
    _pin_free_check(rank, sum(count * nbytes for _, count, nbytes, _ in plan), reclaim=reclaim, headroom_gib=headroom_gib)
    t0 = time.perf_counter()
    held: List[torch.Tensor] = []
    done: List[str] = []
    try:
        for pool, count, nbytes, why in plan:
            n0 = len(held)
            for _ in range(count):
                held.append(torch.empty(nbytes, dtype=torch.uint8, pin_memory=True))
            done.append(f"{pool} {len(held) - n0} x {nbytes >> 20} MiB ({why}, +1)")
    except Exception as error:  # never abort a run over a pre-grow
        logger.warning("[pin-prewarm] stopped after %d blocks: %s", len(held), error)
    del held  # back to the caching host allocator's free list (no stream use recorded)
    logger.warning(
        "armed OK: [pin-prewarm] %s pinned blocks cached in %.2fs before run_start (batch %d) [rank %d]; "
        "kill switch %s=0", " + ".join(done) or "no", time.perf_counter() - t0, bs, rank, _PIN_PREWARM_ENV)


@gin.configurable
def make_train_test_dataloaders(
    batch_size: int,
    dataset_type: str,
    hstu_config: DlrmHSTUConfig,
    train_split_percentage: float,
    embedding_table_configs: Dict[str, EmbeddingConfig],
    new_path_prefix: str = "",
    num_workers: int = 0,
    num_blocks: int = 1,
    prefetch_factor: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    dataset_class, kwargs = get_dataset(
        name=dataset_type, new_path_prefix=new_path_prefix
    )
    kwargs["embedding_config"] = embedding_table_configs
    # Items (32)/(61) fast-batch switches (gin + env), before any loader is built.
    fast_batch_pools()
    # Item (56) fast-batch layout (gin + env), before any loader forks.
    fast_batch_onestore()

    # Create dataset
    dataset = HammerToTorchDataset(
        dataset=dataset_class(hstu_config=hstu_config, is_inference=False, **kwargs)
    )
    total_items = dataset.dataset.get_item_count()
    items_per_block = total_items // num_blocks
    train_size_per_block = round(train_split_percentage * items_per_block)
    # Avoid `extend(range(...))` which materializes a Python list of all sample
    # indices — at 3.2B yambda samples × 28 bytes/int ≈ 90 GB/rank just for
    # train_inds. Subset accepts any sequence with O(1) __len__ and __getitem__,
    # so pass range objects (or a tiny chained view) directly.
    if num_blocks == 1:
        train_inds = range(0, train_size_per_block)
        test_inds = range(train_size_per_block, items_per_block)
    else:
        train_inds = _ChainedRanges([
            range(i * items_per_block, i * items_per_block + train_size_per_block)
            for i in range(num_blocks)
        ])
        test_inds = _ChainedRanges([
            range(i * items_per_block + train_size_per_block, (i + 1) * items_per_block)
            for i in range(num_blocks)
        ])
    train_set = torch.utils.data.Subset(dataset, train_inds)
    test_set = torch.utils.data.Subset(_eval_dataset(dataset), test_inds)

    # When the parent rank is started via mp.start_processes(start_method="spawn"),
    # torch.multiprocessing's default Process context is also "spawn". DataLoader
    # then pickles `self._dataset` to send to each worker — which for our mmap'd
    # 211 GB yambda store materializes the entire dataset into the parent's anon
    # memory (~230 GB/rank). Forcing "fork" lets workers inherit the parent's
    # mmap'd pages via COW with zero extra anon.
    mp_ctx = "fork" if num_workers and num_workers > 0 else None
    # In-process loading (num_workers=0) has no prefetch queue; DataLoader rejects
    # a non-None prefetch_factor outright.
    if mp_ctx is None:
        prefetch_factor = None
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=ChunkDistributedSampler(train_set, drop_last=True, shuffle=True),
        multiprocessing_context=mp_ctx,
    )
    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=eval_batch_size if eval_batch_size is not None else batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=ChunkDistributedSampler(test_set, drop_last=True, shuffle=True),
        multiprocessing_context=mp_ctx,
    )
    return train_dataloader, test_dataloader


# THROWAWAY DIAG state: per-embedding-table lookup stats accumulated across a
# fixed window of steps (DIAG_EMB_STEPS, default 100) so the reported numbers are
# averaged/aggregated rather than a single noisy batch. Rank-0 only.
_EMB_DIAG_ACC: Dict[str, Dict[str, Any]] = {}
_EMB_DIAG_NBATCH: int = 0
# Cap (per-batch lookups) below which we also track a TRUE global-unique set
# across the whole window (cheap for contextual/cross tables, total == batch
# size). Sequential tables (item/artist/album) blow past this and only get the
# per-batch averages.
_EMB_DIAG_GLOBAL_CAP: int = 1 << 17  # 131072


def _train_metric_step(
    metric_logger: MetricsLogger,
    metric_log_frequency: int,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    num_candidates: torch.Tensor,
) -> None:
    """Per-step training-side metric hook shared by every train loop.

    metric_log_frequency > 0: the full ``metric_logger.update`` (torchmetrics
    state accumulation + step bookkeeping), unchanged.
    metric_log_frequency <= 0 (METRIC_LOG_FREQ=0, training metrics OFF): only
    ``advance_step`` -- the global_step / samples_count bookkeeping that LR
    warmup, checkpoint/eval cadences and MLLOG depend on -- with no metric
    work on the device.
    """
    if metric_logging_enabled(metric_log_frequency):
        metric_logger.update(
            mode="train",
            predictions=predictions,
            labels=labels,
            weights=weights,
            num_candidates=num_candidates,
        )
    else:
        metric_logger.advance_step(num_candidates, mode="train")


def _log_unique_embedding_diag(
    sample, rank: int, step: int, max_steps: int = 100, log_every: int = 50
) -> None:
    """Diagnostic: aggregate per-embedding-table lookup stats over a step window.

    Quantifies the user-major batching concern — when consecutive sliding-window
    anchors come from the same few users, a batch reads very few UNIQUE embedding
    rows (low unique/total), so embedding lookups are highly redundant. Covers the
    base id tables AND the cross-feature tables (user_x_* / *_x_hour hashed
    combos); the user_x_* tables should be the most redundant under shuffle OFF.

    Accumulates over ``max_steps`` batches and emits an aggregate summary (mean
    per-batch unique%/hot%/top10%, plus a true global-unique over the whole
    window for the small contextual/cross tables). Rank-0 only, non-fatal.
    """
    if rank != 0:
        return
    global _EMB_DIAG_NBATCH
    try:
        from generative_recommenders.dlrm_v4.configs import YAMBDA_5B_CROSS_SPECS

        cross_caps = {name: n for (name, _k, n, _s) in YAMBDA_5B_CROSS_SPECS}

        def _table_of(key: str):
            # cross tables match by exact name; resolve BEFORE substring fallbacks
            # so e.g. 'user_x_artist' isn't misread as the artist_id table.
            if key in cross_caps:
                return key, cross_caps[key]
            if key == "uid" or key.endswith("_uid") or key.endswith(".uid"):
                return "uid", 0
            if "artist" in key:
                return "artist_id", 0
            if "album" in key:
                return "album_id", 0
            if "item" in key and key.endswith("id"):
                return "item_id", 0
            return None, 0

        for tag, kjt in (
            ("uih", sample.uih_features_kjt),
            ("cand", sample.candidates_features_kjt),
        ):
            for key in kjt.keys():
                table, cap = _table_of(key)
                if table is None:
                    continue
                vals = kjt[key].values()
                total = int(vals.numel())
                if total == 0:
                    continue
                u, counts = torch.unique(vals, return_counts=True)
                uniq = int(u.numel())
                hot1 = int(counts.max().item())
                k = min(10, uniq)
                topk = int(torch.topk(counts, k).values.sum().item())

                slot = _EMB_DIAG_ACC.setdefault(
                    f"{tag}.{key}",
                    {
                        "table": table,
                        "cap": cap,
                        "n": 0,
                        "tot": 0,
                        "uniq": 0,
                        "upct": 0.0,
                        "upct_min": 100.0,
                        "upct_max": 0.0,
                        "hot1pct": 0.0,
                        "topkpct": 0.0,
                        "glob": None,  # running global-unique id tensor (small tables)
                    },
                )
                upct = 100.0 * uniq / total
                slot["n"] += 1
                slot["tot"] += total
                slot["uniq"] += uniq
                slot["upct"] += upct
                slot["upct_min"] = min(slot["upct_min"], upct)
                slot["upct_max"] = max(slot["upct_max"], upct)
                slot["hot1pct"] += 100.0 * hot1 / total
                slot["topkpct"] += 100.0 * topk / total
                if total <= _EMB_DIAG_GLOBAL_CAP:
                    prev = slot["glob"]
                    merged = u if prev is None else torch.cat([prev, u])
                    slot["glob"] = torch.unique(merged)

        _EMB_DIAG_NBATCH += 1
        n = _EMB_DIAG_NBATCH
        if metric_log_due(n, log_every) or n >= max_steps:
            lines = [f"emb-diag AGGREGATE over {n} batches (step<= {step}):"]
            for name in sorted(_EMB_DIAG_ACC):
                s = _EMB_DIAG_ACC[name]
                c = max(1, s["n"])
                cap_s = f" cap={s['cap']/1e6:.0f}M" if s["cap"] else ""
                glob_s = ""
                if s["glob"] is not None:
                    g = int(s["glob"].numel())
                    glob_s = (
                        f" | global_uniq={g} over {s['tot']} seen "
                        f"({s['tot']/max(1,g):.1f}x reuse)"
                    )
                lines.append(
                    f"  {name}[{s['table']}]{cap_s}: "
                    f"avg_tot={s['tot']/c:.0f} "
                    f"avg_uniq%={s['upct']/c:.1f} "
                    f"(min={s['upct_min']:.1f} max={s['upct_max']:.1f}) "
                    f"avg_hot1%={s['hot1pct']/c:.1f} "
                    f"avg_top10%={s['topkpct']/c:.1f}"
                    f"{glob_s}"
                )
            logger.info("\n".join(lines))
    except Exception as e:  # diagnostic must never break training
        logger.warning(f"emb-diag failed: {e}")


@gin.configurable
def train_loop(
    rank: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    device: torch.device,
    num_epochs: int,
    num_batches: Optional[int] = None,
    output_trace: bool = False,
    metric_log_frequency: int = 1,
    checkpoint_frequency: int = 100,
    start_batch_idx: int = 0,
    streaming_diag_unique_emb: bool = False,
    # lr_scheduler: to-do: Add a scheduler
) -> None:
    model.train()
    batch_idx: int = start_batch_idx
    profiler = Profiler(rank) if output_trace else None

    for epoch in range(num_epochs):
        dataloader.sampler.set_epoch(epoch)  # pyre-ignore [16]
        for sample in dataloader:
            if streaming_diag_unique_emb and batch_idx < int(
                os.environ.get("DIAG_EMB_STEPS", "100")
            ):
                _log_unique_embedding_diag(
                    sample,
                    rank,
                    batch_idx,
                    max_steps=int(os.environ.get("DIAG_EMB_STEPS", "100")),
                    log_every=metric_log_frequency,
                )
            optimizer.zero_grad()
            sample.to(device)
            (
                _,
                _,
                aux_losses,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = model.forward(
                sample.uih_features_kjt,
                sample.candidates_features_kjt,
            )
            # pyre-ignore
            sum(aux_losses.values()).backward()
            optimizer.step()
            _train_metric_step(
                metric_logger,
                metric_log_frequency,
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=sample.candidates_features_kjt.lengths().view(
                    len(sample.candidates_features_kjt.keys()), -1
                )[0],
            )
            # (legacy inverted gate kept for > 0; <= 0 = training metrics OFF)
            if (
                metric_logging_enabled(metric_log_frequency)
                and batch_idx % metric_log_frequency != 0
            ):
                metric_logger.compute_and_log(
                    mode="train",
                    additional_logs={
                        "losses": aux_losses,
                    },
                )
            if batch_idx % checkpoint_frequency == 0 and batch_idx > 0:
                save_dmp_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    metric_logger=metric_logger,
                    rank=rank,
                    batch_idx=batch_idx,
                )
            batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            if num_batches is not None and batch_idx >= num_batches:
                break
        if num_batches is not None and batch_idx >= num_batches:
            break


@gin.configurable
def eval_loop(
    rank: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    metric_logger: MetricsLogger,
    device: torch.device,
    metric_log_frequency: int = 1,
    num_batches: Optional[int] = None,
    output_trace: bool = False,
    # lr_scheduler: to-do: Add a scheduler
) -> None:
    model.eval()
    # Exclude eval wall-time from the train step-time window (see _run_eval_window).
    metric_logger.pause_perf("eval")
    batch_idx: int = 0
    profiler = Profiler(rank) if output_trace else None
    metric_logger.reset(mode="eval")
    with torch.no_grad():
        for sample in dataloader:
            sample.to(device)
            (
                _,
                _,
                _,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = model.forward(
                sample.uih_features_kjt,
                sample.candidates_features_kjt,
            )
            metric_logger.update(
                mode="eval",
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=sample.candidates_features_kjt.lengths().view(
                    len(sample.candidates_features_kjt.keys()), -1
                )[0],
            )
            if (
                metric_logging_enabled(metric_log_frequency)
                and batch_idx % metric_log_frequency != 0
            ):
                metric_logger.compute_and_log(mode="eval")
            batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            if num_batches is not None and batch_idx >= num_batches:
                break
    metric_logger.compute_and_log(mode="eval")
    for k, v in metric_logger.compute(mode="eval").items():
        print(f"{k}: {v}")
    metric_logger.resume_perf("eval")


class _PipelineModelWrapper(torch.nn.Module):
    """Adapt ``DlrmHSTU.forward`` to the ``(loss, output)`` contract that
    ``TrainPipelineSparseDist`` expects.

    The wrapped ``model`` is the same DMP instance handed to the pipeline as
    ``model=``; the pipeline rewrites its sharded ``EmbeddingCollection`` in
    place, so calling it here is what lets the embedding all-to-all overlap the
    dense forward/backward compute.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._model = model

    def forward(
        self, batch: Samples
    ) -> Tuple[torch.Tensor, Tuple[Any, ...]]:
        # The model runs in `_pipeline_mode`: it takes the whole batch as its
        # single arg and reads the pre-merged sparse KJT off it. This keeps the
        # EmbeddingCollection input a plain getattr on the batch placeholder so
        # TorchRec pipelines its input_dist (instead of skipping it for "input
        # modifications").
        if _dlstart.ACTIVE:
            _dlstart.mark("fwd_begin")
        (
            _,
            _,
            aux_losses,
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
        ) = self._model(batch)
        if _dlstart.ACTIVE:
            # first forward's host return (no sync: GPU tail lands in the next
            # host sync, i.e. the deferred input dist / backward)
            _dlstart.mark("fwd_end")
        loss = sum(aux_losses.values())
        num_candidates = batch.candidates_features_kjt.lengths().view(
            len(batch.candidates_features_kjt.keys()), -1
        )[0]
        output = (
            aux_losses,
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
            num_candidates,
        )
        return loss, output


_IDIST_GATE_ENV: str = "DLRM_HSTU_FP8_IDIST_AFTER_PREV_FWD"


def _idist_gate_topology() -> Tuple[int, int, Optional[int]]:
    """(world size, GPUs per node, NNODES) for the idist gate's node scope.

    The same facts the launcher works from (train_ranker.py: NNODES,
    GPUS_PER_NODE or the visible GPU count); the world size comes from the
    initialized process group when there is one. NNODES is None when the
    variable is absent, empty or not an integer; the node scope then rests
    on the world size against GPUs per node alone.
    """

    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, "") or default)
        except ValueError:
            return default

    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
    else:
        world = _env_int("WORLD_SIZE", 1)
    local = (
        _env_int("GPUS_PER_NODE", 0)
        or _env_int("LOCAL_WORLD_SIZE", 0)
        or max(1, torch.cuda.device_count())
    )
    nnodes = _env_int("NNODES", 0) or None
    return world, local, nnodes


def _idist_gate_resolve(
    raw: Optional[str],
    defer_input_dist: bool,
    world: int,
    local: int,
    nnodes: Optional[int],
) -> Tuple[int, bool, str]:
    """DLRM_HSTU_FP8_IDIST_AFTER_PREV_FWD -> (mode, gate on, why).

    mode 0: unset, empty, 0/false/off/no (the code default) -> off.
    mode 1: 1/true/on/yes (the gin binding) -> on only for a single-node run
      (NNODES <= 1 or unset, and world size <= GPUs per node), off on more
      nodes.
    mode 2: on at every node count.
    Any other value -> mode -1, off (logged as a warning by the caller).
    The gate also needs the deferred next-batch input dist
    (TP_DEFER_INPUT_DIST=1, the default).
    """
    v = (raw or "").strip().lower()
    if v in ("", "0", "false", "off", "no"):
        return 0, False, "mode 0, the code default: no gate"
    if v in ("1", "true", "on", "yes"):
        mode = 1
    elif v == "2":
        mode = 2
    else:
        return -1, False, f"unrecognized value {raw!r}, expected 0, 1 or 2: no gate"
    topo = (
        f"world={world} gpus_per_node={local} "
        f"NNODES={'unset' if nnodes is None else nnodes}"
    )
    if not defer_input_dist:
        return mode, False, (
            f"mode {mode}, but TP_DEFER_INPUT_DIST=0: no deferred input dist "
            "to gate"
        )
    if mode == 1 and ((nnodes or 1) > 1 or world > local):
        return 1, False, (
            f"mode 1, multi-node: off until 2n/4n evidence; {topo}; =2 gates "
            "at every node count"
        )
    if mode == 1:
        return 1, True, f"mode 1, single node: {topo}"
    return 2, True, f"mode 2, every node count: {topo}"


def _wait_issued_collectives(objs: Iterable[Any]) -> int:
    """Order the CURRENT stream after every c10d collective reachable from
    torchrec input-dist state, without resolving anything.

    Walks containers and the attributes of torchrec / generative_recommenders
    objects (KJTList*Awaitable, FusedKJTListSplitsAwaitable,
    KJTAllToAll*Awaitable, SplitsAllToAllAwaitable, ...) and calls
    ``Work.wait()`` on each ``torch.distributed.Work`` found: a stream wait on
    the collective's end event (host-blocking only under
    TORCH_NCCL_BLOCKING_WAIT). No split read, no new collective, so every rank
    keeps the same collective order. Call it on the stream that allocated the
    collectives' buffers (data-dist) before dropping awaitables whose
    collectives may still be in flight: torch 2.16's default c10d "nccl"
    backend (nccl2) keeps only an async collective's INPUT alive, so a dropped
    output buffer goes straight back to that stream's pool while RCCL may still
    write it (legacy ProcessGroupNCCL, torch 2.12, stashed outputs too).
    Returns the number of Work handles waited.
    """
    import torch.distributed as dist

    work_type = getattr(dist, "Work", None)
    if work_type is None:
        return 0
    n = 0
    seen: Set[int] = set()
    stack: List[Tuple[Any, int]] = [(o, 0) for o in objs]
    while stack:
        o, depth = stack.pop()
        if (
            o is None
            or depth > 8
            or isinstance(o, (torch.Tensor, str, bytes, int, float, bool))
            or id(o) in seen
        ):
            continue
        seen.add(id(o))
        if isinstance(o, work_type):
            o.wait()
            n += 1
        elif isinstance(o, (list, tuple, set)):
            stack.extend((x, depth + 1) for x in o)
        elif isinstance(o, dict):
            stack.extend((x, depth + 1) for x in o.values())
        elif hasattr(o, "__dict__") and (type(o).__module__ or "").startswith(
            ("torchrec", "generative_recommenders")
        ):
            stack.extend((x, depth + 1) for x in vars(o).values())
    return n


@gin.configurable
def build_train_pipeline(
    model: torch.nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    defer_next_input_wait: bool = False,
    role: str = "train",
) -> Any:
    """Build a ``TrainPipelineSparseDist`` for the DMP-wrapped HSTU model.

    The 3-stage pipeline overlaps (1) H2D transfer of batch N+2, (2) the sparse
    data-dist all-to-all of batch N+1's embedding lookup, and (3) dense fwd/bwd
    of batch N, on separate CUDA streams. Requires the model to be wrapped with
    ``DistributedModelParallel`` (see ``make_optimizer_and_shard``).

    ``defer_next_input_wait`` queues the current backward/update before waiting
    for next-batch split sizes. The wait still finishes before advancing the
    batch queue. The production Gin enables it; False restores the earlier wait.

    ``role`` is "train" or "eval" (the forward-only eval pipeline); only the
    train pipeline takes item (67)'s multi-node side-stream priority.
    """
    if role not in ("train", "eval"):
        raise ValueError("role must be 'train' or 'eval', got %r" % (role,))
    # Lazy import: keeps module import working on torchrec builds that move or
    # rename the pipeline, and matches the reference Primus-DLRM setup.
    from torchrec.distributed.train_pipeline import TrainPipelineSparseDist

    if type(defer_next_input_wait) is not bool:
        raise ValueError("defer_next_input_wait must be a boolean")
    logger.info(
        "DLRMV4 pipeline overlap: defer_next_input_wait=%s",
        defer_next_input_wait,
    )

    # Switch the (DMP-wrapped) HSTU model into pipeline mode so both the fx trace
    # and the live forward consume the batch as a single arg and read the
    # pre-merged sparse KJT off it — required for the embedding input_dist to be
    # pipelined. Eval call sites pass the batch the same way (see train_eval_loop).
    underlying = model.module if hasattr(model, "module") else model
    underlying._pipeline_mode = True

    # TP_DEFER_INPUT_DIST=0 issues next-batch input_dist before forward. The
    # next-batch split wait has its own independent control above.
    defer_input_dist = os.environ.get("TP_DEFER_INPUT_DIST", "1") == "1"

    # DLRM_HSTU_FP8_IDIST_AFTER_PREV_FWD (_idist_gate_resolve below): 0 = off
    # (the code default: unset, empty, 0/false/off/no), 1 = on for single-node
    # runs only, 2 = on at every node count; the gin binding sets 1. Acts only
    # with the deferred input dist. The deferred next-batch
    # input dist (KJT permute + EC index dedup sorts on the HIGH-priority
    # data-dist stream, plus the dedup's .item() host syncs) starts on the GPU
    # as soon as the host issues it. Where that lands in the GPU timeline is
    # set by how fast the host enqueued the forward, so any change that trims
    # host work in the forward moves it. Arbor rccl iter-65 traces (1n, rank 0):
    # with the control's host speed it ran right after the previous forward's
    # GPU end (overlapping backward, harmless); with the glue fusions
    # (DLRM_HSTU_FP8_ACTION_ENC_FUSE, ..._CONTEXTUAL_FUSE, ..._EC_MERGED_INDEX_FUSE,
    # ..._LAZY_FLOPS_STASH) the host got
    # ~15 ms earlier and the same sorts landed under the targets-only layer's
    # LayerNorm + uvqk FP8 GEMM: the LN ran +0.16 ms, the GEMM +0.20-0.25 ms
    # per rank-step, and a 15 us merge-sort sat 2.1 ms behind the GEMM.
    # Host-side gate: before issuing batch k+2's input dist from progress(k+1),
    # poll (no GPU-queue barrier: a blocked barrier on the high-priority
    # data-dist queue costs every other dispatch ~31 us, host_sync_free notes)
    # until the main stream has finished forward k -- an event recorded right
    # after forward k was queued. The host is >= 1 step ahead and already
    # blocks later on the same batch's split sizes, so the GPU always has
    # backward k queued while the host waits. The pre-existing split wait in
    # dequeue_batch releases only at ~forward k+1 start + 4 ms, strictly later
    # than this gate's condition (forward k end), so the gate never lengthens
    # the host cycle at 1n; it only moves where the host waits inside a step.
    # Training steps only (forward-only eval keeps its own placement). Pure
    # scheduling: same kernels, same values, same collective issue order.
    # Node scope: the evidence is single-node. On two nodes (gate on) the GPU
    # span still dropped, but the next-batch input dist's side-stream
    # _block_bucketize_* kernels (row-wise sharding, multi-node only) ran
    # longer in early backward, so mode 1 keeps the gate off on more than one
    # node until 2n/4n evidence decides it; mode 2 overrides that.
    idist_raw = os.environ.get(_IDIST_GATE_ENV)
    idist_mode, idist_after_prev_fwd, idist_why = _idist_gate_resolve(
        idist_raw, defer_input_dist, *_idist_gate_topology()
    )
    (logger.warning if idist_mode < 0 else logger.info)(
        "DLRMV4 idist gate: %s=%r -> %s (%s)",
        _IDIST_GATE_ENV,
        idist_raw,
        "ON" if idist_after_prev_fwd else "OFF",
        idist_why,
    )

    # progress() owns backward() and optimizer.step(), so there is no in-loop
    # hook point for gradient clipping. Override the pipeline's own step hook,
    # which runs after _backward() and before optimizer.step() -- the same slot
    # the sequential path clips in. (A full-backward hook on the DMP wrapper is
    # not a substitute: it fires on grads w.r.t. module *inputs*, which the
    # sharded embedding path does not produce.)
    class _StreamingTrainPipeline(TrainPipelineSparseDist):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Next-batch input dist waits only on its own batch's H2D event, not
            # on the whole memcpy stream (which by then also holds batch N+2's
            # copy). Kill switch DLRM_HSTU_FP8_DD_OWN_H2D=0 (dd_own_h2d.py).
            # Created first: the attach()/reset() overrides below use it.
            self._dd_gate = _dd_own_h2d.OwnH2DGate()
            super().__init__(*args, **kwargs)
            # Host-sync-free step only (see host_sync_free.dist_init_on_main):
            # keep the forward's input-dist completion off the high-priority
            # data-dist queue so that queue never holds a blocked barrier with
            # kernels behind it while the host runs a step ahead.
            if _hsf.dist_init_on_main():
                from torchrec.distributed.train_pipeline.runtime_forwards import (
                    PipelinedForward,
                )

                PipelinedForward._use_main_stream_for_dist_init = True
                # The all-to-all outputs this moves onto the main stream are
                # record_stream()ed there in host_sync_free.p_kjtt_wait.
                logger.info("DLRMV4 host-sync-free: dist_init on main stream armed")
                if _hsf.split_wait_poll_on():
                    logger.info(
                        "DLRMV4 host-sync-free: split wait poll-gated armed"
                    )
            # Allocator pre-grow (gin apply_env_bootstrap.DLRM_HSTU_FP8_ALLOC_PREGROW):
            # the side streams whose pools were mapped before RUN_START replace
            # torchrec's before any batch is issued on them; each pipeline
            # takes the pair made for its role (item 67 gives them different
            # priorities at 2+ nodes).
            adopted = _alloc_pregrow.adopt_streams(self, role)
            self._alloc_reporter = _alloc_pregrow.make_reporter(self)
            # Item (67): the train pipeline's side streams run at normal
            # priority when the job spans >1 node (gin
            # DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN); DLRM_HSTU_FP8_PIPE_STREAM_PRIO
            # is the A/B override for every pipeline. None = torchrec's -1.
            prio = (
                _hsf.train_pipe_stream_priority()
                if role == "train"
                else _hsf.pipe_stream_priority()
            )
            if prio is None and self._memcpy_stream is not None:
                # 1 node with item (67) bound: one info line per process,
                # train pipeline only (log only; the pair keeps torchrec's -1).
                _hsf.log_train_pipe_priority_inactive(role)
            if prio is not None and self._memcpy_stream is not None:
                # Item (67)'s default path (train pipeline, multi-node
                # priority, no A/B override): its priority-0 pair must be the
                # pre-grown pair whose hardware-queue placement alloc_pregrow
                # probed before RUN_START. No placement probe runs here.
                item34 = (
                    role == "train"
                    and prio != -1
                    and _hsf.pipe_stream_priority() is None
                )
                if (
                    self._memcpy_stream.priority == prio
                    and self._data_dist_stream.priority == prio
                    and (adopted or not item34)
                ):
                    # Already at this priority (the pre-grown pair was made at
                    # it by alloc_pregrow._pair_priorities): keep it and its pools.
                    logger.info(
                        "DLRMV4 pipeline streams: kept %s pair memcpy prio=%d data_dist prio=%d (%s)",
                        "pre-grown" if adopted else "torchrec",
                        self._memcpy_stream.priority,
                        self._data_dist_stream.priority,
                        role,
                    )
                elif item34:
                    # Any other train pair at this point (the allocator
                    # pre-grow, item (11), is off, or its pair was not adopted
                    # at the requested priority) would need the probe inside
                    # the timed region: fall back to -1 instead.
                    _hsf.disable_train_pipe_priority(
                        "no pre-RUN_START placement probe for this pair "
                        "(allocator pre-grow off or its pair not adopted)"
                    )
                    prio = -1
                    if (
                        self._memcpy_stream.priority != -1
                        or self._data_dist_stream.priority != -1
                    ):
                        # Not reached with torchrec 1.7, which creates its
                        # pair at -1. Rebuilt before any batch is issued.
                        self._memcpy_stream = torch.cuda.Stream(priority=-1)
                        self._data_dist_stream = torch.cuda.Stream(priority=-1)
                else:
                    # Rebuilt before any batch is issued on them.
                    memcpy = torch.cuda.Stream(priority=prio)
                    data_dist = torch.cuda.Stream(priority=prio)
                    if (
                        role == "train"
                        and prio != -1
                        and _alloc_pregrow.guard_train_pair(
                            torch.cuda.current_stream(), memcpy, data_dist
                        )
                    ):
                        # Item (67) fell back: keep torchrec's own -1 pair.
                        prio = -1
                    else:
                        self._memcpy_stream = memcpy
                        self._data_dist_stream = data_dist
                    logger.info(
                        "DLRMV4 pipeline streams: memcpy prio=%d data_dist prio=%d (%s)",
                        self._memcpy_stream.priority,
                        self._data_dist_stream.priority,
                        role,
                    )
                if role == "train" and _hsf.pipe_stream_priority() is None:
                    if self._data_dist_stream.priority != -1:
                        logger.info(
                            "armed OK: multi-node train pipeline stream priority "
                            "%d (DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN)",
                            self._data_dist_stream.priority,
                        )
                    else:
                        logger.info(
                            "DLRMV4 multi-node side-stream priority off: train "
                            "pipeline keeps torchrec priority -1"
                        )

        def progress(self, dataloader_iter):
            out = super().progress(dataloader_iter)
            if self._alloc_reporter is not None:
                self._alloc_reporter.after_step()
            return out

        def _optimizer_step(self) -> None:
            if _dlstart.ACTIVE:
                _dlstart.mark("opt_begin")
            if grad_clip_norm and grad_clip_norm > 0:
                _clip_grad_norm_(model.parameters(), grad_clip_norm, self._optimizer)
            super()._optimizer_step()
            if _dlstart.ACTIVE:
                _dlstart.mark("opt_end")

        def _backward(self, losses) -> None:
            if _dlstart.ACTIVE:
                _dlstart.mark("bwd_begin")
            super()._backward(losses)
            if _dlstart.ACTIVE:
                # host return of autograd: every bwd kernel queued, plus any
                # host syncs inside bwd (ec_segment_reduce U, FlyDSL launch)
                _dlstart.mark("bwd_end")

        def _init_pipelined_modules(self, batch, context, pipelined_forward=None):
            if _dlstart.ACTIVE and not self._pipelined_modules:
                # fx trace + module rewrite (_rewrite_model) starts here; it
                # ends where start_sparse_data_dist(batch0) is entered
                _dlstart.mark("fx_begin")
            if pipelined_forward is None:
                return super()._init_pipelined_modules(batch, context)
            return super()._init_pipelined_modules(batch, context, pipelined_forward)

        # progress() queues the NEXT batch's input_dist -- EC dedup plus the
        # splits all-to-all, ~2.5 ms of host work -- before it launches THIS
        # batch's forward. The compute stream has nothing queued in that window,
        # so it idles at the top of every step while the CPU works on a batch it
        # will not train for another step; the splits wait also needs the sizes on
        # the host, so the CPU blocks there too. Deferring both to the
        # wait_sparse_data_dist that follows the forward means the forward's
        # kernels are already queued, so the GPU has ~130 ms of work to chew on
        # while the CPU pays for the next batch's prep.
        _deferred_input_dist = None
        _deferred_split_wait = None
        # Event recorded on the main stream right after the previous
        # progress()'s forward was queued (DLRM_HSTU_FP8_IDIST_AFTER_PREV_FWD).
        _prev_fwd_done = None
        # The compute stream the forward is queued on, retained at the first
        # gated call; the gate stands down if a later call runs elsewhere.
        _gate_stream = None
        _gate_failed = False
        _idist_gate_announced = False

        def _gate_next_input_dist(self) -> None:
            """Hold the next-batch input dist until the previous forward is
            done on the GPU; then remember where this forward ends.

            Runs in wait_sparse_data_dist right before the deferred
            start_sparse_data_dist, i.e. after this step's forward (and the
            batch N+2 H2D issue) are queued, with the main stream current.
            Deadlock-free: the polled event covers only forward k-1 (and work
            queued before it), which the host queued in the previous
            progress() and which depends only on collectives every rank
            issued before reaching this point; this step's backward is queued
            after the poll, but the GPU still holds backward k-1 then.
            """
            if (
                not idist_after_prev_fwd
                or self._gate_failed
                or not self._model.training
            ):
                # Off, failed once, or a forward-only (eval) pipeline: no gate.
                self._prev_fwd_done = None
                return
            stream = torch.cuda.current_stream()
            if self._gate_stream is None:
                self._gate_stream = stream
            elif stream != self._gate_stream:
                # Not the retained compute stream: do not gate, start afresh.
                self._prev_fwd_done = None
                return
            prev = self._prev_fwd_done
            done = torch.cuda.Event()
            done.record(stream)
            self._prev_fwd_done = done
            if prev is None:
                return
            if not self._idist_gate_announced:
                type(self)._idist_gate_announced = True
                logger.warning(
                    "DLRMV4 armed OK: idist gate ON (%s) -- next-batch input "
                    "dist gated on the previous forward's GPU end (host poll, "
                    "no queue barrier; kill switch "
                    "DLRM_HSTU_FP8_IDIST_AFTER_PREV_FWD=0)",
                    idist_why,
                )
            # Host poll (as host_sync_free._poll_until_done): never a GPU-queue
            # barrier; the sleep releases the GIL for the loader threads. The
            # labelled range shows in a trace only when the gate held the host.
            try:
                if prev.query():
                    return
                with torch.profiler.record_function("## idist_gate ##"):
                    while not prev.query():
                        time.sleep(5e-5)
            except Exception as e:  # noqa: BLE001 -- query problem: stop gating
                self._gate_failed = True
                self._prev_fwd_done = None
                logger.warning(
                    "DLRMV4 idist gate disabled after a query error (%s); the "
                    "next-batch input dist is no longer gated",
                    e,
                )

        def start_sparse_data_dist(self, batch, context):
            # Only the next-batch call is deferred. _init_pipelined_modules and
            # attach() call this for the CURRENT batch (contexts[0]) to bring the
            # module's input dists up before the forwards are overridden, and
            # that has to run inline.
            if (
                defer_input_dist
                and batch is not None
                and self.contexts
                and context is not self.contexts[0]
            ):
                self._deferred_input_dist = (batch, context)
                return
            if _dlstart.ACTIVE:
                _dlstart.mark("idist0_begin")
            # Inline (current-batch) path: stock torchrec wait; drop the
            # batch's unused H2D event.
            self._dd_gate.discard(context)
            super().start_sparse_data_dist(batch, context)

        def _start_sparse_data_dist_own_h2d(self, batch, context):
            # torchrec TrainPipelineSparseDist.start_sparse_data_dist with its
            # _wait_for_batch(batch, memcpy_stream) replaced by a wait on this
            # batch's own H2D event (dd_own_h2d.py). Here the memcpy stream
            # already holds batch N+2's copy, which batch N+1 does not read.
            if batch is None:
                return
            if not self._dd_gate.on:
                super().start_sparse_data_dist(batch, context)
                return
            from torchrec.distributed.train_pipeline import utils as _tp_utils

            with torch.autograd.profiler.record_function(
                f"## start_sparse_data_dist {context.index} ##"
            ):
                with self._stream_context(self._data_dist_stream):
                    if not self._dd_gate.wait(batch, context):
                        _tp_utils._wait_for_batch(batch, self._memcpy_stream)
                    with _tp_utils.use_context_for_postprocs(
                        self._pipelined_postprocs, context
                    ):
                        _tp_utils._start_data_dist(
                            self._pipelined_modules, batch, context
                        )

        def wait_sparse_data_dist(self, context):
            deferred = self._deferred_input_dist
            if deferred is not None and deferred[1] is context:
                if _dlstart.ACTIVE:
                    _dlstart.mark("idist1_begin")
                # rccl's idist gate (host poll on the previous forward) first,
                # then the deferred input dist gated on its own H2D event.
                self._gate_next_input_dist()
                self._deferred_input_dist = None
                self._start_sparse_data_dist_own_h2d(*deferred)
            if defer_next_input_wait and self.contexts and context is not self.contexts[0]:
                # Current backward does not consume the next batch's lengths.
                # Keep its splits launch here, but let backward be queued before
                # the host reads those sizes and issues the lengths/values A2As.
                assert len(self.contexts) >= 2 and context is self.contexts[1]
                assert self._deferred_split_wait is None
                self._deferred_split_wait = context
                return
            super().wait_sparse_data_dist(context)
            if _dlstart.ACTIVE:
                # batch0 split sizes are on the host here (real host wait)
                _dlstart.mark("idist0_done")

        def dequeue_batch(self):
            pending = self._deferred_split_wait
            if pending is not None:
                assert len(self.contexts) >= 2 and pending is self.contexts[1]
                # Finish before sliding the queue, including drain/eval paths.
                # Leave pending intact if the collective raises; abort normally.
                # (Host-sync-free: the splits wait inside is issued only after
                # the collective has finished; host_sync_free.split_wait_poll_on.)
                if _dlstart.ACTIVE:
                    _dlstart.mark("deq_wait_begin")
                super().wait_sparse_data_dist(pending)
                self._deferred_split_wait = None
            super().dequeue_batch()

        def attach(self, model=None, sparse_dist: bool = True) -> None:
            # Eval calls detach() with two batches still in flight, and progress()
            # re-attaches lazily on the next call. torchrec's attach() re-runs the
            # model rewrite and, inside _pipeline_model, re-issues batches[0]'s
            # input dist through the stock KJTAllToAll forwards that detach()
            # restored: the splits all_to_all is issued into contexts[0] (with
            # NEW module contexts) but never waited, and fill_pipeline() returns
            # early while len(batches) >= 2. The forward would then resolve the
            # tensors requests captured before detach() against those new module
            # contexts (a 0-row embedding tensor; construct_jagged_tensors dies
            # in torch.split). So finish the input dist that super().attach()
            # just started, exactly as fill_pipeline() does for batch 0.
            #
            # Never drop an issued collective un-waited. This override used to
            # clear that splits all_to_all and issue a second input dist: on
            # torch 2.16 the default c10d "nccl" backend is nccl2, which keeps
            # only an async collective's INPUT alive, so the dropped
            # all_to_all's output block went straight back to the data-dist pool
            # and the re-issue's tensors were carved from it while RCCL had yet
            # to write it (7 of 8 ranks drop it unfinished; the first train
            # batch after an eval then gets another collective's splits: "host
            # length_per_key ... != values", "shape '[-1, 1024]' is invalid",
            # permute_2D faults or a hang, within the first 1-5 resume
            # switches). Legacy ProcessGroupNCCL (torch 2.12) stashed outputs
            # too, which hid it.
            # docs/04-technical-guides/recommendation/dlrm_v4/rocm-port.md T17.
            resume_in_flight = (
                sparse_dist
                and bool(self.contexts)
                and bool(self.batches)
                and self.batches[0] is not None
            )
            # New pipeline generation: never poll an event from before detach.
            self._prev_fwd_done = None
            stale: List[Any] = []
            if resume_in_flight:
                # batches[0]'s pre-detach input-dist requests, superseded below.
                # Held here until the data-dist stream has waited on their
                # collectives; cleared now so _fuse_input_dist_splits and the
                # wait below see only the requests super().attach() issues.
                ctx0 = self.contexts[0]
                stale.extend(ctx0.input_dist_tensors_requests.values())
                stale.extend(ctx0.input_dist_splits_requests.values())
                stale.extend(aw for _, aw in ctx0.fused_splits_awaitables)
                ctx0.input_dist_tensors_requests.clear()
                ctx0.input_dist_splits_requests.clear()
                ctx0.fused_splits_awaitables.clear()
            super().attach(model, sparse_dist)
            self._dd_gate.clear()
            if not resume_in_flight:
                return
            context = self.contexts[0]
            if stale and self._data_dist_stream is not None:
                with self._stream_context(self._data_dist_stream):
                    _wait_issued_collectives(stale)
            del stale
            self._deferred_input_dist = None
            self._deferred_split_wait = None
            self._prev_fwd_done = None
            self._set_module_context(context)
            # Waits (on data-dist) for the splits all_to_all super().attach()
            # issued, then issues the lengths/values all_to_all into
            # context.input_dist_tensors_requests (stock batch-0 path).
            super().wait_sparse_data_dist(context)

        def reset(self) -> None:
            self._deferred_input_dist = None
            self._deferred_split_wait = None
            self._prev_fwd_done = None
            self._dd_gate.clear()
            # reset() drops every in-flight context: the eval pass's
            # num_eval_batches break leaves one batch's lengths/values
            # all_to_all issued, and an early train-window exit can too. Order
            # the data-dist stream after those collectives first, so their
            # output buffers (data-dist pool; nccl2 does not keep them alive)
            # are not reused while RCCL may still write them.
            if self.contexts and self._data_dist_stream is not None:
                pending: List[Any] = []
                for ctx in self.contexts:
                    pending.extend(ctx.input_dist_tensors_requests.values())
                    pending.extend(ctx.input_dist_splits_requests.values())
                    pending.extend(aw for _, aw in ctx.fused_splits_awaitables)
                if pending:
                    with self._stream_context(self._data_dist_stream):
                        _wait_issued_collectives(pending)
            super().reset()

        def copy_batch_to_gpu(self, dataloader_iter):
            # copy_batch_to_gpu allocates the context (and its index) before it
            # learns the iterator is exhausted, so a failed enqueue still
            # consumes an index. The streaming loop hands the pipeline one
            # iterator per window, so those failures are routine -- once at the
            # end of every window, plus once for every window that yields no
            # batch at all -- and the index runs tens of batches ahead of the
            # step count, which makes the "## forward N ##" labels in a trace
            # impossible to line up with a step. Hand the index back when the
            # batch turns out to be None so it keeps counting real batches.
            batch, context = super().copy_batch_to_gpu(dataloader_iter)
            if batch is None:
                self._next_index -= 1
            else:
                # "this batch's H2D is done" event, recorded right behind its
                # copies; the deferred input dist waits on it (dd_own_h2d.py).
                self._dd_gate.after_copy(self._memcpy_stream, batch, context)
                if _dlstart.ACTIVE and self._next_index == 3:
                    # third H2D issued (pull3 + non_blocking copy enqueue)
                    _dlstart.mark("enq3_done")
            return batch, context

    # Eval embedding lookahead (dlrm_v4/train/eval_lookahead.py; gin
    # apply_env_bootstrap.DLRM_HSTU_FP8_EVAL_EMB_LOOKAHEAD, default False): only
    # the forward-only eval pipeline, and only with the knob on, becomes an
    # eval-only subclass that issues batch N+1's EC lookup and sequence
    # all-to-all on a side stream at the end of progress(N). Role "train" always
    # builds _StreamingTrainPipeline itself, with the same arguments.
    pipeline_cls = _StreamingTrainPipeline
    if role == "eval":
        from generative_recommenders.dlrm_v4.train import eval_lookahead as _evla

        if _evla.enabled():
            pipeline_cls = _evla.wrap(_StreamingTrainPipeline)
    return pipeline_cls(
        model=model,
        optimizer=optimizer,
        device=device,
        execute_all_batches=True,
        # Issue batch N+2's H2D copy after batch N's forward is launched rather
        # than before it. The compute stream sits idle at the top of every step
        # until the CPU gets around to launching the forward, and the H2D issue
        # is ~1.6 ms of that wait; moving it behind the forward hands the GPU its
        # dense work sooner. (It also keeps the copy off the PCIe link while the
        # embedding lookup is reading from it, which is why torchrec offers it.)
        enqueue_batch_after_forward=True,
        custom_model_fwd=_PipelineModelWrapper(model),
    )


@gin.configurable
def train_eval_loop(
    rank: int,
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    device: torch.device,
    num_epochs: int,
    num_train_batches: Optional[int] = None,
    num_eval_batches: Optional[int] = None,
    train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
    output_trace: bool = False,
    metric_log_frequency: int = 1,
    checkpoint_frequency: int = 100,
    eval_frequency: int = 1,
    start_train_batch_idx: int = 0,
    start_eval_batch_idx: int = 0,
    use_pipeline: bool = False,
    # lr_scheduler: to-do: Add a scheduler
) -> None:
    train_batch_idx: int = start_train_batch_idx
    eval_batch_idx: int = start_eval_batch_idx
    profiler = Profiler(rank) if output_trace else None
    assert train_dataloader is not None and eval_dataloader is not None

    eval_data_iterator = iter(eval_dataloader)
    train_data_iterator = iter(train_dataloader)

    # 3-stage TorchRec pipeline (overlaps embedding a2a with dense compute).
    # When enabled, progress() owns H2D copy, sparse-dist, fwd/bwd and the
    # optimizer step; grad clipping moves to a full-backward hook (see builder).
    train_pipeline = (
        build_train_pipeline(model, optimizer, device) if use_pipeline else None
    )

    for epoch in range(num_epochs):
        train_dataloader.sampler.set_epoch(epoch)  # pyre-ignore [16]
        while True:
            model.train()
            if train_pipeline is not None:
                try:
                    (
                        aux_losses,
                        mt_target_preds,
                        mt_target_labels,
                        mt_target_weights,
                        num_candidates,
                    ) = train_pipeline.progress(train_data_iterator)
                except StopIteration:
                    train_data_iterator = iter(train_dataloader)
                    break
            else:
                try:
                    sample = next(train_data_iterator)
                except StopIteration:
                    train_data_iterator = iter(train_dataloader)
                    break
                optimizer.zero_grad()
                sample.to(device)
                (
                    _,
                    _,
                    aux_losses,
                    mt_target_preds,
                    mt_target_labels,
                    mt_target_weights,
                ) = model.forward(
                    sample.uih_features_kjt,
                    sample.candidates_features_kjt,
                )
                # pyre-ignore
                sum(aux_losses.values()).backward()
                _clip_grad_norm_(model.parameters(), 1.0, optimizer)
                optimizer.step()
                num_candidates = sample.candidates_features_kjt.lengths().view(
                    len(sample.candidates_features_kjt.keys()), -1
                )[0]
            _train_metric_step(
                metric_logger,
                metric_log_frequency,
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=num_candidates,
            )
            if metric_log_due(train_batch_idx, metric_log_frequency):
                metric_logger.compute_and_log(
                    mode="train",
                    additional_logs={
                        "losses": aux_losses,
                    },
                )
            if train_batch_idx % checkpoint_frequency == 0 and train_batch_idx > 0:
                save_dmp_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    metric_logger=metric_logger,
                    rank=rank,
                    batch_idx=train_batch_idx,
                )
            train_batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            if train_batch_idx % eval_frequency == 0:
                model.eval()
                eval_batch_idx: int = 0
                with torch.no_grad():
                    while True:
                        try:
                            sample = next(eval_data_iterator)
                        except StopIteration:
                            eval_data_iterator = iter(eval_dataloader)
                            sample = next(eval_data_iterator)
                        sample.to(device)
                        (
                            _,
                            _,
                            _,
                            mt_target_preds,
                            mt_target_labels,
                            mt_target_weights,
                        ) = (
                            # In pipeline mode the model takes the batch as one
                            # arg (see _PipelineModelWrapper / DlrmHSTU.forward).
                            model.forward(sample)
                            if use_pipeline
                            else model.forward(
                                sample.uih_features_kjt,
                                sample.candidates_features_kjt,
                            )
                        )
                        metric_logger.update(
                            mode="eval",
                            predictions=mt_target_preds,
                            labels=mt_target_labels,
                            weights=mt_target_weights,
                            num_candidates=sample.candidates_features_kjt.lengths().view(
                                len(sample.candidates_features_kjt.keys()), -1
                            )[0],
                        )
                        eval_batch_idx += 1
                        if output_trace:
                            assert profiler is not None
                            profiler.step()
                        if metric_log_due(eval_batch_idx, metric_log_frequency):
                            metric_logger.compute_and_log(mode="eval")
                        if (
                            num_eval_batches is not None
                            and eval_batch_idx >= num_eval_batches
                        ):
                            break
                    for k, v in metric_logger.compute(mode="eval").items():
                        print(f"{k}: {v}")
                model.train()
            # `num_train_batches` cap: None or 0 = run the whole window. >0 caps
            # batches per window (mostly the streaming-resume test driver uses
            # this to keep test windows short).
            if num_train_batches and train_batch_idx >= num_train_batches:
                break


def checkpoint_time_interval_for_saving(checkpoint_time_interval_s: float) -> float:
    """Skip time polling only when the effective checkpoint path is empty.

    Use Gin's current scope and reference resolution, just as the later save
    call does. CKPT_PATH alone is insufficient: a Gin binding can override it.
    Resolve once at loop entry, where the run's configuration is fixed. If the
    destination cannot be determined, retain the requested cadence.
    """
    if checkpoint_time_interval_s <= 0:
        return checkpoint_time_interval_s
    try:
        bindings = gin.get_bindings(save_dmp_checkpoint, resolve_references=False)
        # Gin resolves references with deepcopy. Resolve only the destination:
        # other save arguments may have evaluated references that this loop
        # overrides explicitly and must not construct merely to check the path.
        path = (
            copy.deepcopy(bindings["path"])
            if "path" in bindings
            else inspect.signature(save_dmp_checkpoint).parameters["path"].default
        )
    except Exception:
        # This is a best-effort optimization, not a new prerequisite for a
        # checkpoint-enabled run. Preserve polling if Gin resolution fails.
        logger.warning(
            "Could not resolve checkpoint destination; retaining time-based polling",
            exc_info=True,
        )
        return checkpoint_time_interval_s
    # Match save_dmp_checkpoint's exact no-op condition. Unknown/non-string
    # destinations are not evidence that saving is disabled.
    return 0.0 if isinstance(path, str) and path == "" else checkpoint_time_interval_s


def select_in_window_checkpoint_reason(
    *,
    train_batch_idx: int,
    global_step: int,
    elapsed_since_last_save: float,
    in_window_checkpoint_frequency: int,
    checkpoint_step_frequency: int,
    checkpoint_time_interval_s: float,
) -> Optional[str]:
    """Decide which (if any) in-window checkpoint cadence fires this batch.

    Pure / distributed-agnostic so it can be unit-tested without a real run.
    The caller computes `elapsed_since_last_save` (broadcast from rank 0 in the
    streaming loop) so all ranks pass the same value and reach the same verdict.

    Precedence (at most one save per batch): per-window-local batch count >
    monotonic global step > wall-clock interval. Returns the trigger reason
    string, or None when no cadence fires. A cadence is disabled when its
    frequency/interval is 0 / 0.0.

    Counter conventions match the loop: `train_batch_idx` is already
    post-incremented (>=1 on the first batch), and `global_step` is guarded
    >0 so step 0 doesn't trivially satisfy `% N == 0`.
    """
    if (
        in_window_checkpoint_frequency > 0
        and train_batch_idx % in_window_checkpoint_frequency == 0
    ):
        return "in_window_batch"
    if (
        checkpoint_step_frequency > 0
        and global_step > 0
        and global_step % checkpoint_step_frequency == 0
    ):
        return "global_step"
    if (
        checkpoint_time_interval_s > 0
        and elapsed_since_last_save >= checkpoint_time_interval_s
    ):
        return "time_interval"
    return None


def data_pct_to_global_steps(
    *,
    data_pct: float,
    total_train_samples: int,
    global_batch_size: int,
) -> int:
    """Convert a fraction of the epoch's training data into a global train-step
    count -- the single source of truth for the percent -> steps conversion.

    Used by BOTH the data-fraction eval cadence (EVAL_EVERY_DATA_PCT ->
    eval_interval_steps) and the skip-eval threshold (SKIP_EVAL_EPOCH_PCT ->
    skip_eval_until_step) so a percentage is turned into steps identically in
    both places. The fraction is taken in SAMPLE space against the epoch
    denominator MLPerf's EPOCH_NUM uses, then divided ONCE by the global batch
    size (per-rank batch x world_size) -- the only progress counter that is both
    rank-invariant (all ranks step in lockstep) and checkpoint-restored. Rounds
    UP so the result lands at or after the requested fraction, never before it.
    Returns 0 for a disabled / degenerate config (any non-positive input).
    """
    if data_pct <= 0 or total_train_samples <= 0 or global_batch_size <= 0:
        return 0
    return int(
        math.ceil(data_pct * float(total_train_samples) / float(global_batch_size))
    )


def first_data_pct_eval_steps(
    gstep0: int, eval_interval_steps: int, skip_eval_until_step: int
) -> int:
    """Training steps from global step ``gstep0`` to the first data-fraction
    eval of the loop below: it evaluates after the step that makes the global
    step g when g > 0, g % eval_interval_steps == 0 and g >=
    skip_eval_until_step. 0 when the cadence is off (eval_interval_steps <= 0).
    Used only to time item (36)'s eval holdout read-ahead."""
    if eval_interval_steps <= 0:
        return 0
    lo = max(int(gstep0) + 1, int(skip_eval_until_step), 1)
    first = -(-lo // int(eval_interval_steps)) * int(eval_interval_steps)
    return first - int(gstep0)


def resolve_skip_eval_until_step(
    *,
    skip_eval_epoch_pct: float,
    total_train_samples: int,
    global_batch_size: int,
) -> int:
    """Convert SKIP_EVAL_EPOCH_PCT into a global train-step threshold.

    Pure / distributed-agnostic so it can be unit-tested without a real run.
    Returns the first global_step at which periodic eval is allowed to run;
    every scheduled eval boundary at a STRICTLY smaller global_step is skipped.
    0 means "no skipping" (feature off, or a degenerate denominator).

    The fraction is expressed against `total_train_samples` -- the same epoch
    denominator MLPerf's EPOCH_NUM uses -- so `skip_eval_epoch_pct` reads
    directly off an epoch-vs-convergence plot. It is converted to samples, then
    to steps at the GLOBAL batch size (per-rank batch x world_size), because
    global_step is the only progress counter that is both rank-invariant (all
    ranks step in lockstep) and checkpoint-restored. Deriving the threshold from
    a per-rank sample tally instead could let ranks disagree about whether a
    given boundary is skipped, and a disagreement deadlocks the next collective.

    Rounds UP so the first eval lands at or after the requested fraction, never
    before it.

    A NEGATIVE value (the gin default) means "the MLPerf rule": the first eval
    lands at CEIL(FLOOR((4480*GBS + 135331840)/3) / eval_interval) * eval_interval
    samples, with eval_interval = CEIL(0.001 * total / GBS) * GBS (training_rules
    eval table, dlrm_v4_hstu), which is what the compliance checker requires for
    every global batch size.
    """
    if skip_eval_epoch_pct < 0:
        if total_train_samples <= 0 or global_batch_size <= 0:
            return 0
        interval = (
            math.ceil(0.001 * float(total_train_samples) / float(global_batch_size))
            * global_batch_size
        )
        first = math.floor((4480 * global_batch_size + 135331840) / 3)
        first = int(math.ceil(first / interval)) * interval
        return first // global_batch_size
    return data_pct_to_global_steps(
        data_pct=skip_eval_epoch_pct,
        total_train_samples=total_train_samples,
        global_batch_size=global_batch_size,
    )


def _validate_split_contract(
    saved: Optional[Dict[str, Any]],
    live: Dict[str, Any],
    rank: int,
) -> None:
    """Guarantee the train:eval split (and the inputs the resume skip-offset
    depends on) are unchanged across a crash/resume.

    `saved` is the contract recovered from the checkpoint (None on cold start or
    legacy pre-holdout checkpoints). Any mismatch is fatal: continuing would
    either desync the mid-window skip (duplicate/skip batches) or reassign users
    so that previously held-out eval users get trained (leakage). Set
    ALLOW_SPLIT_MISMATCH=1 to override (e.g. intentionally resuming a legacy
    checkpoint into a holdout run, accepting the risk).
    """
    allow = os.environ.get("ALLOW_SPLIT_MISMATCH", "0") == "1"
    if saved is None:
        # Legacy / cold-start checkpoint with no recorded contract. Only a
        # problem if this run actually holds users out (tsp < 1.0): we cannot
        # prove the earlier run used the same split.
        if live.get("train_split_percentage", 1.0) < 1.0 and not allow:
            raise RuntimeError(
                "Resuming a checkpoint with NO saved split contract into a "
                f"user-holdout run (train_split_percentage="
                f"{live['train_split_percentage']}). The earlier run's split "
                "cannot be verified, so held-out eval users may have been "
                "trained. Set ALLOW_SPLIT_MISMATCH=1 to override."
            )
        return
    mismatches = {
        k: (saved.get(k), live.get(k))
        for k in live
        if saved.get(k) != live.get(k)
    }
    if mismatches:
        msg = (
            "Split/resume contract mismatch between checkpoint and current run: "
            + ", ".join(
                f"{k}: checkpoint={s!r} current={c!r}" for k, (s, c) in mismatches.items()
            )
            + ". Resuming would desync the skip offset and/or leak held-out "
            "users into training."
        )
        if allow:
            if rank == 0:
                logger.warning("%s ALLOW_SPLIT_MISMATCH=1 set — continuing anyway.", msg)
        else:
            raise RuntimeError(msg + " Set ALLOW_SPLIT_MISMATCH=1 to override.")
    elif rank == 0:
        logger.info("Split/resume contract verified against checkpoint: %s", live)


@gin.configurable
def streaming_train_eval_loop(
    rank: int,
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    device: torch.device,
    num_train_ts: int,
    hstu_config: DlrmHSTUConfig,
    embedding_table_configs: Dict[str, EmbeddingConfig],
    num_train_batches: Optional[int] = None,
    num_eval_batches: Optional[int] = None,
    output_trace: bool = False,
    # Training-side metric cadence (console/TB). 0 = training metrics OFF:
    # no per-step torchmetrics update and no train compute_and_log; step
    # bookkeeping (MetricsLogger.advance_step) and eval are unaffected.
    metric_log_frequency: int = 1,
    # Interim eval logging cadence gate. 0 (default) = OFF: skip the per-batch
    # compute_and_log DURING an eval pass. Those interim calls only log
    # partial-holdout metrics (progress/debug); the value we keep -- the durable
    # metrics.jsonl, early-stop, and MLPerf EVAL_ACCURACY -- is the end-of-pass
    # compute() over the whole holdout, so OFF is numerically neutral for the
    # recorded eval AUC. >0 restores the old interim logging. $EVAL_INTERIM_LOG.
    eval_interim_metrics: int = 0,
    # MLPerf `train_loss` event cadence, in global train steps, INDEPENDENT of
    # metric_log_frequency (the console/TB cadence). 0 = fall back to
    # metric_log_frequency (preserves prior coupled behavior). Wired to
    # $MLPERF_TRAIN_LOSS_LOG_FREQ via gin.
    mlperf_train_loss_log_frequency: int = 0,
    # num_workers of the persistent EVAL DataLoader pool only (double-buffer
    # path); the train pools keep the gin num_workers ($NUM_WORKERS). < 0 =
    # inherit NUM_WORKERS (pre-2026-10-05 behaviour). Wired to $EVAL_NUM_WORKERS
    # via gin (default 8). Input pipeline only: same eval set, order and metric.
    eval_num_workers: int = -1,
    checkpoint_frequency: int = 100,
    start_ts: int = 0,
    persistent_loader: bool = False,
    eval_every_n_windows: int = 1,
    # Data-fraction eval cadence (mutually exclusive with eval_every_n_windows).
    # >0 = eval every this FRACTION of the run's total training data, converted
    # once to a global train-step interval. 0.0 = OFF (use the per-window
    # cadence). Wired to $EVAL_EVERY_DATA_PCT via gin.
    eval_every_data_pct: float = 0.0,
    # SKIP_EVAL_EPOCH_PCT: suppress scheduled (periodic) eval passes until the
    # run has trained this FRACTION of the epoch (0.027 = 2.7% of
    # total_train_samples). Early in a run the model sits far below the AUC
    # target, so those eval passes cost a full holdout forward sweep for no
    # signal. 0.0 (default in code) = OFF (eval from the first scheduled point).
    # Resolved ONCE into a global train-step threshold (see
    # resolve_skip_eval_until_step) and compared against the monotonic,
    # checkpoint-restored global_step, so the skipped set is identical on a cold
    # start and on every resume, and is rank-invariant. The end-of-run final
    # eval is never skipped. Wired to $SKIP_EVAL_EPOCH_PCT via gin.
    skip_eval_epoch_pct: float = 0.0,
    double_buffer: bool = False,
    # --- fixed user-holdout eval set ---
    # Window range the fixed eval set is drawn from. None -> default to
    # original_end_ts (start_ts + num_train_ts), the window just past training.
    eval_holdout_ts: Optional[int] = None,
    eval_holdout_num_windows: int = 1,
    # --- resume / mid-window-exact-once knobs ---
    resume_train_ts: Optional[int] = None,
    resume_batch_idx_in_window: int = WINDOW_COMPLETE,
    # Split contract recovered from the checkpoint (None on cold start or
    # legacy checkpoints). Validated below against the live split so a resumed
    # run cannot silently train a different user-split (would leak).
    resume_split_contract: Optional[Dict[str, Any]] = None,
    # True iff no checkpoint was loaded (genuine fresh run). Distinguishes a
    # cold start (safe to establish a new split) from a resume that merely lacks
    # a contract (legacy/non-streaming checkpoint), which the guard must reject.
    resume_cold_start: bool = False,
    in_window_checkpoint_frequency: int = 0,
    # --- global step / wall-clock checkpoint cadences ---
    checkpoint_step_frequency: int = 0,
    checkpoint_time_interval_s: float = 0.0,
    # --- gradient clipping (streaming path, dense params). 0.0 = OFF, which
    #     preserves legacy streaming behavior. Wired to $GRAD_CLIP_NORM via gin. ---
    grad_clip_norm: float = 0.0,
    # --- LR warmup (streaming path). Linearly ramp LR 0 -> base over the first
    #     N GLOBAL train steps, then hold. Scales BOTH the dense optimizer and
    #     the in-backward fused sparse-embedding optimizer. 0 = OFF (byte-
    #     identical). Wired to $LR_WARMUP_STEPS via gin. ---
    lr_warmup_steps: int = 0,
    # --- LR warmup floor (streaming path). Absolute LR the ramp STARTS from at
    #     gstep 0 (per group, clamped to <= that group's base LR). The ramp then
    #     goes start -> base over `lr_warmup_steps`. 0.0 (default) reproduces the
    #     0 -> base ramp exactly (byte-identical). A small nonzero floor turns the
    #     warmup into a "partial warmup" that reclaims the near-dead head of the
    #     ramp (trading a little seed-divergence damping for faster time-to-target).
    #     Only has any effect when lr_warmup_steps > 0. Wired to
    #     $LR_WARMUP_START_LR via gin. ---
    lr_warmup_start_lr: float = 0.0,
    # --- diagnostic: log per-batch unique/total embedding-id counts ---
    streaming_diag_unique_emb: bool = False,
    # --- 3-stage TorchRec pipeline (TrainPipelineSparseDist). Overlaps the H2D
    #     copy of batch N+2 and the embedding input_dist of batch N+1 with the
    #     dense fwd/bwd of batch N. progress() then owns zero_grad, forward,
    #     backward, grad clipping and optimizer.step(). Wired to $TRAIN_PIPELINE
    #     via gin; ON by default ($TRAIN_PIPELINE=0 restores the sequential
    #     path). Compatible with in-window checkpointing on every cadence -- see
    #     the trained-count note above the pipeline build for why the 2 in-flight
    #     batches do not break the resume offset. ---
    use_pipeline: bool = True,
    # --- test-only failure injection knob ---
    die_at_step: int = -1,
    # MLPerf logger (rank-0-gated); None disables all MLPerf event emission.
    mlperf_logger: Optional[Any] = None,
    # Skip the hourly clock broadcast when the resolved save path is empty.
    # False restores polling for controlled comparisons or rollback.
    skip_checkpoint_polling_when_disabled: bool = True,
    # Teardown after RUN_STOP ($DL_TEARDOWN_CLOSE_POOLS, train_ranker.py): a
    # list receives this loop's persistent input pools (train loader, eval
    # pool), which the caller closes once the loop has returned
    # (pool_close.close_pools). None (the default) records nothing.
    teardown_pools: Optional[List[Any]] = None,
) -> None:
    """Streaming train+eval loop with per-window (and optionally mid-window)
    checkpoints.

    Resume semantics (set by train_ranker after `load_dmp_checkpoint` returns):
      - resume_train_ts=None: cold start; honor `start_ts` as-is.
      - resume_train_ts=N, resume_batch_idx_in_window=WINDOW_COMPLETE(-1):
        previous run finished window N cleanly. Start at N+1 from sample 0.
      - resume_train_ts=N, resume_batch_idx_in_window=K (K>=0): previous run
        crashed mid-window after K completed batches. Re-enter window N and
        skip the first K batches of THIS rank's per-rank sample list (deterministic
        slice since `window_indices(N)` is a pure function of the anchor_ts cache).

    Checkpoint cadences (all independent; any combination may be enabled):
      - `checkpoint_frequency`: window-granularity. End-of-window save every
        Nth train_ts (and always on the final window). Uses WINDOW_COMPLETE.
      - `in_window_checkpoint_frequency`: per-window-local batch count. Fires
        every N batches *within* a window (counter resets each window).
      - `checkpoint_step_frequency`: global-step granularity. Fires whenever
        the monotonic `metric_logger.global_step['train']` hits a multiple of
        N — i.e. a true "every 1000 steps" trigger that spans windows and
        survives resume (global_step is restored from the checkpoint).
      - `checkpoint_time_interval_s`: wall-clock granularity. Fires when at
        least this many seconds have elapsed since the last save (e.g. 3600
        for hourly). Rank 0 owns the clock and broadcasts the decision so all
        ranks save together (avoids the collective barrier in
        `save_dmp_checkpoint` deadlocking on a split decision).

    All in-window triggers (`in_window_checkpoint_frequency`,
    `checkpoint_step_frequency`, `checkpoint_time_interval_s`) route through
    `_save_mid_window`, which stamps `batch_idx_in_window=K` so a crash leaves
    a resumable partial-window checkpoint. End-of-window saves
    (`checkpoint_frequency`) always use the WINDOW_COMPLETE sentinel. 0 / 0.0
    disables a given cadence (the default for all three fine-grained ones).

    `die_at_step` is a test-only hook: when `metric_logger.global_step['train']`
    reaches this value, the process exits with code 42 right after the in-window
    save fires. Used by the failure-injection test to crash at a deterministic
    boundary and then resume.
    """
    # Exactly one eval cadence may be active. eval_every_n_windows defaults to 1
    # (eval every window), so enabling the data-fraction cadence REQUIRES
    # explicitly disabling the per-window one (EVAL_EVERY_N_WINDOWS=0). Fail fast
    # on a contradictory config rather than silently picking one.
    if (eval_every_data_pct and eval_every_data_pct > 0) and eval_every_n_windows > 0:
        raise ValueError(
            "Conflicting eval cadences: eval_every_data_pct="
            f"{eval_every_data_pct} (>0) AND eval_every_n_windows="
            f"{eval_every_n_windows} (>0). They are mutually exclusive. To use "
            "the data-fraction cadence set EVAL_EVERY_N_WINDOWS=0; to use the "
            "per-window cadence set EVAL_EVERY_DATA_PCT=0."
        )
    # MLPerf train_loss cadence: independent of metric_log_frequency. <= 0 (the
    # gin default 0, submission logging profile) = OFF: no train_loss event and
    # no loss all_reduce; MLPerf 6.1 does not require it.
    mlperf_loss_every = resolve_mlperf_loss_every(
        metric_log_frequency, mlperf_train_loss_log_frequency
    )
    if rank == 0:
        # Arming lines for the metric cadence and the MLPerf train_loss cadence
        # (yambda_5b.gin defaults METRIC_LOG_FREQ=0 and MLPERF_TRAIN_LOSS_LOG_FREQ=0,
        # both OFF: the submission logging profile; perf runs opt in explicitly).
        logger.warning(metric_cadence_log_line(metric_log_frequency, mlperf_loss_every))
        logger.warning(mlperf_train_loss_log_line(mlperf_loss_every))
    profiler = Profiler(rank) if output_trace else None
    # In-window checkpointing is safe under the pipeline even though it keeps up
    # to 2 batches in flight beyond the one being trained. Every trigger below
    # stamps `batch_idx_in_window=train_batch_idx`, which counts batches TRAINED,
    # not batches pulled off the iterator, and the resume skip is derived from
    # that same trained count. A batch that was prefetched but never contributed
    # a gradient is simply re-read on resume, so nothing is trained twice and
    # nothing is dropped.
    # The diagnostic reads the batch, which the pipeline consumes internally.
    if use_pipeline and streaming_diag_unique_emb:
        logger.warning(
            "streaming_diag_unique_emb is unavailable under use_pipeline=True "
            "(the pipeline owns the batch); disabling it for this run. "
            "Set TRAIN_PIPELINE=0 to run the diagnostic."
        )
        streaming_diag_unique_emb = False
    # Normalize the per-window caps: <=0 (the env-binding default) means "no cap
    # = consume the full window". The eval-break check below is `is not None and
    # eval_batch_idx >= num_eval_batches`, so a literal 0 would (wrongly) break
    # after the first batch — map it to None instead for the full-holdout eval.
    if num_eval_batches is not None and num_eval_batches <= 0:
        num_eval_batches = None
    if num_train_batches is not None and num_train_batches <= 0:
        num_train_batches = None
    # Built once and reused across windows: the model rewrite is idempotent
    # (_init_pipelined_modules short-circuits once `_pipelined_modules` is set),
    # and progress() refills its queue from whichever window iterator it is
    # handed. Clipping is passed through so both paths clip identically.
    train_pipeline = (
        build_train_pipeline(model, optimizer, device, grad_clip_norm=grad_clip_norm)
        if use_pipeline
        else None
    )
    if train_pipeline is not None and rank == 0:
        logger.info(
            "[pipeline] TrainPipelineSparseDist enabled (grad_clip_norm=%s)",
            grad_clip_norm,
        )
    # Eval gets its OWN pipeline instance rather than borrowing the training
    # one, which is holding this window's in-flight batches -- handing it the
    # eval iterator would consume them. The same class serves both: torchrec
    # guards zero_grad/backward/optimizer behind `self._model.training`, so
    # under model.eval() progress() is a forward-only pass that still overlaps
    # the H2D copy and the embedding input_dist. Built on first eval so a perf
    # run that never evals does not pay for it. $TP_EVAL_PIPELINE=0 falls back
    # to the serial loop.
    use_eval_pipeline: bool = (
        use_pipeline and os.environ.get("TP_EVAL_PIPELINE", "1") == "1"
    )
    eval_pipeline: Optional[Any] = None
    _dlstart.mark("loop_entry")
    # DL_PRESTART bundle (dataset + pre-forked pools, built before RUN_START by
    # prestart_streaming_loaders); only the double-buffered persistent path uses
    # it, which is the only path prestart builds for.
    _pre = _take_prestart_bundle() if (persistent_loader and double_buffer) else None
    if _pre is not None:
        if rank == 0 and not torch.equal(_pre["rng_state"], torch.get_rng_state()):
            # A global CPU RNG draw happened between the pre-start and here: the
            # main stream is still replayed exactly, but worker base seeds now
            # differ from DL_PRESTART=0 (harmless for the RNG-free yambda
            # get_sample; would matter for a dataset that uses worker RNG).
            logger.warning(
                "[dlstart] CPU RNG moved between pre-start and loop entry: worker "
                "seeds differ from DL_PRESTART=0 (main RNG stream unaffected)"
            )
        dataset = _pre["dataset"]
    else:
        dataset_class, kwargs = get_dataset()
        kwargs["embedding_config"] = embedding_table_configs
        dataset = HammerToTorchDataset(
            dataset=dataset_class(hstu_config=hstu_config, is_inference=False, **kwargs)
        )
    _dlstart.mark("dataset_built")
    # Items (32)/(61) fast-batch switches (gin + env; same values the pre-start
    # resolved), before any pool below forks. The per-call eval loaders read
    # through eval_dataset ($DL_FAST_BATCH_EVAL); the persistent eval pool
    # gets the same setting from _eval_dataset(dataset, slot). Train loaders
    # keep ``dataset`` ($DL_FAST_BATCH).
    fast_batch_pools()
    # Item (56) $DL_FAST_BATCH_ONESTORE (gin + env; same value the pre-start
    # installed), before any pool below forks or starts a pin thread.
    fast_batch_onestore()
    # Item (55) $DL_PIN_FREE_SHM (gin + env; same value the pre-start
    # installed), before any loader below starts a pin thread.
    pin_free_shm_release()
    eval_dataset = _eval_dataset(dataset)
    _log_fast_batch_pools(rank, {"train": dataset, "eval": eval_dataset})
    _log_fast_batch_onestore(rank, {"train": dataset, "eval": eval_dataset})
    # Persistent path: build ONE DataLoader + a stateful sampler whose indices
    # are swapped per window, so workers fork once and are reused across all
    # windows (eliminates the per-window dataloader respawn + first-batch
    # warmup). The non-persistent path recreates a DataLoader per window.
    window_sampler: Optional[StreamingWindowSampler] = None
    persistent_dl: Optional[DataLoader] = None
    world_size = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    if persistent_loader:
        window_sampler = StreamingWindowSampler(rank=rank, world_size=world_size)
        persistent_dl = make_persistent_streaming_dataloader(
            dataset=dataset, sampler=window_sampler
        )
        if teardown_pools is not None:
            # Iterated only without double_buffer (_window_iter); closed by
            # the caller after RUN_STOP.
            teardown_pools.append(persistent_dl)
    # Item (55) setting line (a no-op after the pre-start logged it); every
    # loader of this run takes the same gin pin_memory as persistent_dl.
    _log_pin_free_shm(
        rank,
        pinned=None if persistent_dl is None else bool(persistent_dl.pin_memory),
    )

    # The fixed user-holdout eval is yambda-specific (needs window_indices +
    # the split API). Other streaming datasets (synthetic) keep the legacy
    # per-window eval. Detect support once.
    supports_holdout = hasattr(dataset.dataset, "eval_holdout_indices")

    # Fixed eval-holdout window range. Captured from the REQUESTED (start_ts,
    # num_train_ts) BEFORE the resume block mutates them, so it is identical on
    # cold start and on every resume (the supervisor relaunches with the same
    # START_TS / NUM_TRAIN_TS). Defaults to the window just past training.
    requested_end_ts = start_ts + num_train_ts
    # Eval-cadence anchor: the ORIGINAL requested start_ts, captured BEFORE the
    # resume block rebases start_ts. `_should_eval` keys the every-N-windows
    # cadence off the absolute window ts relative to THIS anchor, so the eval
    # grid (e.g. 150,160,170,...) is identical on cold start and on every resume.
    # (Keying off the per-call loop index instead would re-anchor the grid to
    # whatever window a mid-run resume happens to restart from.)
    eval_anchor_ts = start_ts
    # None (Python default) or <0 (the env-binding default) both mean "use the
    # window just past training", which is stable across resume.
    eval_holdout_ts_resolved = (
        eval_holdout_ts
        if (eval_holdout_ts is not None and eval_holdout_ts >= 0)
        else requested_end_ts
    )

    # Per-rank batch size: the persistent loader carries it directly; the
    # per-window path uses the same gin %batch_size (env BATCH_SIZE, default
    # 1024 — matches make_streaming_dataloader.batch_size). Hoisted out of the
    # data-pct block below because SKIP_EVAL_EPOCH_PCT needs the same value to
    # convert its epoch fraction into a global-step threshold.
    bs = (
        persistent_dl.batch_size
        if persistent_dl is not None
        else int(os.environ.get("BATCH_SIZE", "1024"))
    )
    global_batch_size = bs * world_size

    # Data-fraction eval cadence: convert eval_every_data_pct into a global
    # train-step interval ONCE, over the ORIGINAL requested window range
    # [eval_anchor_ts, requested_end_ts). Keying the later trigger off
    # `global_step % eval_interval_steps` (global_step is monotonic and
    # checkpoint-restored) makes the eval grid identical on cold start and on
    # every resume, exactly like checkpoint_step_frequency. 0 => disabled.
    eval_interval_steps = 0
    if eval_every_data_pct and eval_every_data_pct > 0:
        if hasattr(dataset.dataset, "total_train_anchors"):
            # total_train_anchors does a full-range gather over the mmap'd uid
            # array for ~billions of positions + a uid hash. It is slow
            # (minutes, single-threaded) AND, run on every rank independently,
            # a large per-rank skew source: a fast rank finishes and races into
            # the first embedding all-to-all while slow ranks are still hashing,
            # desyncing the NCCL collective stream and hanging the job. The
            # result is a pure function of the (identical) dataset + split, so
            # compute it ONCE on rank 0 and broadcast the scalar; ranks 1..N
            # skip the gather entirely (no 8x mmap/CPU contention, no skew).
            if world_size > 1 and torch.distributed.is_initialized():
                _tta = (
                    dataset.dataset.total_train_anchors(  # pyre-ignore[16]
                        eval_anchor_ts, requested_end_ts - eval_anchor_ts
                    )
                    if rank == 0
                    else 0
                )
                _tta_t = torch.tensor([_tta], dtype=torch.int64, device=device)
                torch.distributed.broadcast(_tta_t, src=0)
                total_train_anchors = int(_tta_t.item())
            else:
                total_train_anchors = dataset.dataset.total_train_anchors(  # pyre-ignore[16]
                    eval_anchor_ts, requested_end_ts - eval_anchor_ts
                )
            # Percent -> global-steps via the SAME conversion as the skip-eval
            # threshold (data_pct_to_global_steps): fraction taken in sample
            # space, divided once by the global batch size, rounded up. max(1, .)
            # because an eval INTERVAL must be >= 1 step, whereas the skip
            # threshold may legitimately resolve to 0 (feature off). total_train_
            # steps is the epoch length in full global batches, kept for the log.
            total_train_steps = total_train_anchors // max(1, global_batch_size)
            eval_interval_steps = max(
                1,
                data_pct_to_global_steps(
                    data_pct=eval_every_data_pct,
                    total_train_samples=total_train_anchors,
                    global_batch_size=global_batch_size,
                ),
            )
            if rank == 0:
                logger.info(
                    "[data-pct-eval] eval_every_data_pct=%.6g -> "
                    "eval_interval_steps=%d (total_train_anchors=%d bs=%d "
                    "world_size=%d total_train_steps=%d over windows [%d, %d))",
                    eval_every_data_pct,
                    eval_interval_steps,
                    total_train_anchors,
                    bs,
                    world_size,
                    total_train_steps,
                    eval_anchor_ts,
                    requested_end_ts,
                )
        elif rank == 0:
            logger.warning(
                "[data-pct-eval] dataset %s has no total_train_anchors(); "
                "data-fraction eval is DISABLED (no per-window eval either, "
                "since EVAL_EVERY_N_WINDOWS must be 0 to reach here) — only the "
                "final eval will run.",
                type(dataset.dataset).__name__,
            )

    # The split is an immutable run contract: a silent change across resume
    # would both desync the mid-window skip offset AND turn held-out eval users
    # into trained users (leakage). Build the live contract and validate the
    # one recovered from the checkpoint against it; abort on any mismatch unless
    # ALLOW_SPLIT_MISMATCH=1 is set (e.g. deliberately resuming a legacy run).
    live_split_contract: Optional[Dict[str, Any]] = None
    if supports_holdout:
        live_split_contract = {
            "train_split_percentage": dataset.dataset._train_split_percentage,  # pyre-ignore[16]
            "split_salt": dataset.dataset._split_salt,  # pyre-ignore[16]
            "eval_holdout_ts": eval_holdout_ts_resolved,
            "eval_holdout_num_windows": eval_holdout_num_windows,
            "batch_size": persistent_dl.batch_size if persistent_dl is not None else None,
            "world_size": world_size,
        }
        # Only validate on an actual resume. On a genuine cold start there is no
        # prior split to verify and establishing this run's split is always safe;
        # validating there would wrongly reject every fresh holdout run. A resume
        # that lacks a contract (legacy/non-streaming checkpoint) is NOT a cold
        # start and is still validated (and rejected) below.
        if not resume_cold_start:
            _validate_split_contract(resume_split_contract, live_split_contract, rank)

    # Apply resume hint: advance start_ts past the last completed window, or
    # re-enter the partial window with a per-rank skip on its first iter.
    # Shrink num_train_ts by the same amount so the resumed run finishes at
    # the same final timestamp (start_ts + num_train_ts) as a fresh run would
    # — i.e. resumed and uninterrupted produce identical total work.
    first_skip_samples = 0
    if resume_train_ts is not None:
        original_end_ts = start_ts + num_train_ts
        if resume_batch_idx_in_window == WINDOW_COMPLETE:
            new_start = resume_train_ts + 1
            if rank == 0:
                logger.info(
                    "Resuming from completed train_ts=%d → start_ts=%d "
                    "(num_train_ts %d → %d)",
                    resume_train_ts, new_start,
                    num_train_ts, max(0, original_end_ts - new_start),
                )
            start_ts = new_start
        else:
            if rank == 0:
                logger.info(
                    "Resuming mid-window at train_ts=%d batch_idx_in_window=%d "
                    "(skipping batches already trained)",
                    resume_train_ts,
                    resume_batch_idx_in_window,
                )
            start_ts = resume_train_ts
            # `batch_size` is per-rank from the persistent dataloader (set via
            # gin `make_persistent_streaming_dataloader.batch_size`). The
            # skip-samples-per-rank below maps "K batches done" → "K * bs
            # samples in this rank's index list", since each batch draws bs
            # samples from this rank's deterministic round-robin slice.
            assert persistent_dl is not None, (
                "Mid-window resume requires persistent_loader=True"
            )
            first_skip_samples = resume_batch_idx_in_window * persistent_dl.batch_size
        num_train_ts = max(0, original_end_ts - start_ts)
        if num_train_ts == 0 and rank == 0:
            logger.info(
                "Resume target already reached (end_ts=%d, start_ts=%d) — "
                "no further training windows; skipping straight to final eval.",
                original_end_ts, start_ts,
            )

    if rank == 0:
        logger.info(
            "[grad-clip] streaming path gradient clipping %s (max_norm=%.4g via $GRAD_CLIP_NORM)",
            "ENABLED" if (grad_clip_norm and grad_clip_norm > 0) else "OFF",
            grad_clip_norm,
        )

    def _window_iter(ts: int, skip_samples: int = 0):
        # TRAIN-only iterator: both branches exclude held-out eval users via
        # train_window_indices / set_ts(train_only=True). (Eval uses the fixed
        # holdout set, never this helper.)
        if persistent_loader:
            assert window_sampler is not None and persistent_dl is not None
            window_sampler.set_window(
                dataset.dataset.train_window_indices(ts),  # pyre-ignore [16]
                skip_samples=skip_samples,
            )
            return iter(persistent_dl)
        if skip_samples != 0:
            raise NotImplementedError(
                "skip_samples>0 requires persistent_loader=True"
            )
        return iter(
            make_streaming_dataloader(dataset=dataset, ts=ts, train_only=True)
        )
    # Windows are [start_ts, start_ts + num_train_ts); each step trains window T
    # then evals window T+1, so the last eval window is start_ts + num_train_ts,
    # which must be < num_windows(). Anchors require >= history_length prior
    # events, so the earliest windows are near-empty warm-up — use start_ts to
    # begin at a dense window. Clamp instead of failing.
    if hasattr(dataset.dataset, "num_windows"):
        available = dataset.dataset.num_windows()  # pyre-ignore [16]
        max_count = max(0, available - 1 - start_ts)
        if num_train_ts > max_count:
            logger.warning(
                f"start_ts={start_ts} + num_train_ts={num_train_ts} exceeds "
                f"available windows ({available}); clamping num_train_ts to {max_count}."
            )
            num_train_ts = max_count
    # Wall-clock anchor for time-based checkpointing. Mutable single-element
    # list so the nested train loop can reset it after each save. Starts at
    # loop entry so the first time-trigger fires ~interval seconds in.
    effective_checkpoint_time_interval_s = (
        checkpoint_time_interval_for_saving(checkpoint_time_interval_s)
        if skip_checkpoint_polling_when_disabled
        else checkpoint_time_interval_s
    )
    if checkpoint_time_interval_s > 0 and effective_checkpoint_time_interval_s == 0:
        if rank == 0:
            logger.info(
                "Time-based checkpoint polling disabled: "
                "save_dmp_checkpoint.path is empty"
            )
    last_ckpt_time = [time.time()]

    def _broadcast_elapsed() -> float:
        """Seconds since the last save, owned by rank 0 and broadcast to all
        ranks. save_dmp_checkpoint runs a collective barrier, so every rank must
        feed the same wall-clock value into the cadence decision — otherwise a
        split verdict (rank 0 saves, rank 1 doesn't) would deadlock. Broadcasting
        rank 0's elapsed keeps the (pure) decision identical everywhere."""
        elapsed = time.time() - last_ckpt_time[0]
        if torch.distributed.is_initialized() and world_size > 1:
            t = torch.tensor([elapsed], device=device, dtype=torch.float64)
            torch.distributed.broadcast(t, src=0)
            elapsed = float(t.item())
        return elapsed

    def _save_mid_window(train_ts: int, batch_idx_in_window: int) -> None:
        """In-window checkpoint helper. Snapshots the same state as the
        end-of-window save but stamps `batch_idx_in_window=K` instead of
        WINDOW_COMPLETE so the resume path knows to skip K batches.
        Uses train_ts as the numeric subdir name — every save into the same
        train_ts overwrites the previous in-window snapshot (via atomic
        replace), so disk stays bounded to keep_last_n train_ts dirs."""
        save_dmp_checkpoint(
            model=model,
            optimizer=optimizer,
            metric_logger=metric_logger,
            rank=rank,
            batch_idx=train_ts,
            train_ts=train_ts,
            batch_idx_in_window=batch_idx_in_window,
            device=device,
            split_contract=live_split_contract,
        )

    # --- LR warmup (streaming path) ------------------------------------------
    # Linearly ramp LR from 0 -> base over the first `lr_warmup_steps` GLOBAL
    # train steps, then hold at base. Keyed off the monotonic, checkpoint-
    # restored global_step so the ramp is identical on cold start and on every
    # resume (a run that crashes mid-warmup resumes at the correct multiplier;
    # a run already past warmup is a no-op). Scales BOTH LR channels: the
    # dense/non-fused optimizer passed in as `optimizer`, AND the in-backward
    # fused sparse-embedding optimizer (model.fused_optimizer), whose per-table
    # LR dominates this model. `lr_warmup_steps=0` (default) => OFF and the base
    # LRs are never touched (byte-identical to the pre-warmup path).
    _warmup_groups = list(optimizer.param_groups)
    _warmup_fused_opt = getattr(model, "fused_optimizer", None)
    if _warmup_fused_opt is not None:
        _warmup_groups += list(_warmup_fused_opt.param_groups)
    _warmup_base_lrs = [pg["lr"] for pg in _warmup_groups]
    # Per-group start LR for the ramp: the configured floor, clamped so it can
    # never exceed a group's base (a floor >= base would mean "no ramp" for that
    # group, which we express by starting it AT base). 0.0 => classic 0 -> base.
    _warmup_start_lrs = [min(float(lr_warmup_start_lr), b) for b in _warmup_base_lrs]
    if lr_warmup_steps and lr_warmup_steps > 0 and rank == 0:
        logger.info(
            "[lr-warmup] linear warmup over %d global steps across %d param "
            "group(s); start LRs: %s -> base LRs: %s",
            lr_warmup_steps,
            len(_warmup_groups),
            ", ".join(f"{s:.3e}" for s in _warmup_start_lrs),
            ", ".join(f"{b:.3e}" for b in _warmup_base_lrs),
        )

    def _apply_lr_warmup(gstep: int) -> None:
        # OFF, or ramp already finished (kernel + param_groups already hold the
        # base LR from the last sync at gstep==lr_warmup_steps): nothing to do.
        if not (lr_warmup_steps and lr_warmup_steps > 0) or gstep > lr_warmup_steps:
            return
        progress = min(1.0, float(gstep) / float(lr_warmup_steps))
        # Ramp each group from its (clamped) start LR up to its base LR. With
        # start=0 this collapses to `base * progress` (the original schedule).
        for pg, base, start in zip(
            _warmup_groups, _warmup_base_lrs, _warmup_start_lrs
        ):
            pg["lr"] = start + (base - start) * progress
        # Push the warmed LR into the FBGEMM TBE backward kernel. The in-backward
        # fused embedding optimizer only calls
        # `emb_module.set_learning_rate(param_groups[0]["lr"])` from its
        # step()/zero_grad() (torchrec/distributed/batched_embedding_kernel.py),
        # and this loop steps the dense `optimizer` — NOT model.fused_optimizer —
        # so WITHOUT this explicit call the embedding tables would keep training
        # at the construction-time base LR and warmup would silently apply to the
        # dense params only. `.step()` on the fused (in-backward) optimizer does
        # no parameter update; it only re-syncs the LR to the kernel.
        if _warmup_fused_opt is not None:
            _warmup_fused_opt.step()
        if rank == 0 and metric_log_due(gstep, metric_log_frequency):
            _fused_lr = ""
            if _warmup_fused_opt is not None:
                _fused_lr = (
                    f" fused_pg_lr={_warmup_fused_opt.param_groups[0]['lr']:.3e}"
                )
                # Best-effort readback of the ACTUAL kernel LR to confirm
                # set_learning_rate() propagated. model.fused_optimizer is a
                # (possibly nested) CombinedOptimizer whose leaves carry the TBE
                # module as `_emb_module`, so recurse to the first leaf. Purely
                # diagnostic; wrapped so it can never break training.
                def _first_tbe_kernel_lr(opt, depth=0):
                    if depth > 6 or opt is None:
                        return None
                    _em = getattr(opt, "_emb_module", None)
                    if _em is not None and hasattr(_em, "get_learning_rate"):
                        return float(_em.get_learning_rate())
                    for _sub in getattr(opt, "_optims", []):
                        _child = _sub[1] if isinstance(_sub, tuple) else _sub
                        _lr = _first_tbe_kernel_lr(_child, depth + 1)
                        if _lr is not None:
                            return _lr
                    _wrapped = getattr(opt, "_optimizer", None)
                    if _wrapped is not None:
                        return _first_tbe_kernel_lr(_wrapped, depth + 1)
                    return None

                try:
                    _klr = _first_tbe_kernel_lr(_warmup_fused_opt)
                    if _klr is not None:
                        _fused_lr += f" tbe_kernel_lr={_klr:.3e}"
                except Exception:
                    pass
            logger.info(
                "[lr-warmup] gstep=%d progress=%.4f dense_lr=%.3e%s",
                gstep,
                progress,
                _warmup_groups[0]["lr"],
                _fused_lr,
            )

    # Host-sync-free step (dlrm_v4/host_sync_free.py): one fence for the whole
    # run, so the one-step host lead carries across window boundaries.
    _step_fence = _hsf.StepFence(lag=1)

    # [dlstart] startup timeline: one report after the first training step of
    # this process (rank 0 line; every rank with item (109)
    # DL_PRESTART_RANK_TIMELINE=1; a compliance warning on any rank).
    _dl_report_pending = [True]

    def _dl_first_progress_done() -> None:
        _dl_report_pending[0] = False
        _dlstart.mark("first_progress_done")
        try:
            _dlstart.report(rank, prestarted=_pre is not None)
        except Exception as e:  # diagnostics must never break training
            logger.warning("[dlstart] timeline report failed: %r", e)

    def _run_train_window(
        train_data_iterator,
        train_ts: int,
        start_batch_idx: int = 0,
        label: Optional[str] = None,
        do_eval: Optional[Callable[[int, int], None]] = None,
        window_num_batches: Optional[int] = None,
    ) -> None:
        # `start_batch_idx` is set when we're re-entering a window that was
        # interrupted mid-way (in_window resume); the dataloader iterator was
        # already advanced past those batches via the sampler skip, and we
        # account for them in the local counter so in-window saves and the
        # die_at_step hook fire at the right relative offsets.
        #
        # `window_num_batches` is set only under pipeline warm-keep, where
        # `train_data_iterator` runs on into the NEXT window instead of stopping
        # (see _WarmWindowStream): the window then ends on its own rank-identical
        # batch count rather than on StopIteration. It already accounts for
        # `start_batch_idx` (the sampler skip shortened the window).
        train_batch_idx = start_batch_idx
        first_wait: Optional[float] = None
        # Distinguishes a clean end-of-window -- where any in-flight batches
        # belong to the NEXT window and must be KEPT to stay warm -- from an
        # early break, whose in-flight batches belong to THIS window and have to
        # be dropped.
        window_done = False
        while True:
            if (
                window_num_batches is not None
                and train_batch_idx - start_batch_idx >= window_num_batches
            ):
                window_done = True
                break
            model.train()
            _t_next = time.perf_counter() if (label and rank == 0) else None
            if _dl_report_pending[0]:
                _dlstart.mark("first_progress_start")
            if train_pipeline is not None:
                # progress() owns the whole step: it pulls from the iterator,
                # copies batch N+2 H2D, runs the embedding input_dist for batch
                # N+1, then zero_grad/fwd/bwd/clip/step for batch N. LR warmup is
                # applied first so this step's update uses the warmed LR, exactly
                # as in the sequential path below.
                _apply_lr_warmup(metric_logger.global_step["train"])
                try:
                    (
                        aux_losses,
                        mt_target_preds,
                        mt_target_labels,
                        mt_target_weights,
                        num_candidates,
                    ) = train_pipeline.progress(train_data_iterator)
                except StopIteration:
                    # End of the data stream (last window of the run under
                    # warm-keep): the pipeline is already empty.
                    window_done = True
                    break
                if _t_next is not None and first_wait is None:
                    first_wait = time.perf_counter() - _t_next
                if _dl_report_pending[0]:
                    _dl_first_progress_done()
            else:
                try:
                    sample = next(train_data_iterator)
                except StopIteration:
                    window_done = True
                    break
                if _t_next is not None and first_wait is None:
                    first_wait = time.perf_counter() - _t_next
                if streaming_diag_unique_emb and train_batch_idx < int(
                    os.environ.get("DIAG_EMB_STEPS", "100")
                ):
                    _log_unique_embedding_diag(
                        sample,
                        rank,
                        train_batch_idx,
                        max_steps=int(os.environ.get("DIAG_EMB_STEPS", "100")),
                        log_every=metric_log_frequency,
                    )
                # LR warmup: set the scaled LR on BOTH channels (dense
                # param_groups + the fused embedding TBE via set_learning_rate)
                # BEFORE the forward/backward, so the in-backward embedding
                # update THIS step uses the warmed LR (the dense optimizer reads
                # it at .step() below). No-op when lr_warmup_steps=0.
                _apply_lr_warmup(metric_logger.global_step["train"])
                optimizer.zero_grad()
                sample.to(device)
                (
                    _,
                    _,
                    aux_losses,
                    mt_target_preds,
                    mt_target_labels,
                    mt_target_weights,
                ) = model.forward(
                    sample.uih_features_kjt,
                    sample.candidates_features_kjt,
                )
                # pyre-ignore
                sum(aux_losses.values()).backward()
                # Gradient clipping for the streaming path. Clips dense params
                # (the sparse embedding tables use a fused optimizer and are
                # unaffected, same as the non-streaming path's
                # clip_grad_norm_). OFF by default (grad_clip_norm=0.0 via
                # $GRAD_CLIP_NORM) so legacy streaming runs are byte-for-byte
                # unchanged; set >0 to enable.
                if grad_clip_norm and grad_clip_norm > 0:
                    _clip_grad_norm_(model.parameters(), grad_clip_norm, optimizer)
                optimizer.step()
                num_candidates = sample.candidates_features_kjt.lengths().view(
                    len(sample.candidates_features_kjt.keys()), -1
                )[0]
            # METRIC_LOG_FREQ=0: advance_step only (step/sample bookkeeping, no
            # torchmetrics work); the train compute_and_log below is skipped too.
            _train_metric_step(
                metric_logger,
                metric_log_frequency,
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=num_candidates,
            )
            # MLPerf train_loss event on its own cadence (decoupled from the
            # console/TB metric cadence below). Called every step; the cross-rank
            # all-reduce only fires on the cadence, gated by the rank-identical
            # global_step inside the method, so it stays in lockstep.
            metric_logger.maybe_log_mlperf_train_loss(aux_losses, every=mlperf_loss_every)
            # Host-sync-free step (dlrm_v4/host_sync_free.py): with the per-step
            # drains gone, bound the host to one step ahead of the GPU by waiting
            # on the previous step's end event -- the queue still holds this
            # whole step, so the wait never empties it.
            if _hsf.enabled():
                _step_fence.step()
            # Cadence is keyed on the GLOBAL train step, not `train_batch_idx`:
            # the latter resets to 0 per window, so `% metric_log_frequency` was
            # true on batch 0 of every window and the cadence collapsed to
            # "once per window" for windows shorter than the frequency. Since
            # compute_and_log also resets the perf accumulators, that produced
            # report intervals of wildly unequal length (1..N steps) whose
            # step_ms/sps are not comparable to each other. global_step["train"]
            # is incremented once per batch in update() on every rank, so this
            # gate stays rank-identical and the AUC all-gather cannot desync.
            if metric_log_due(metric_logger.global_step["train"], metric_log_frequency):
                if profiler is not None:
                    profiler.note_event(
                        "metric_log", global_step=metric_logger.global_step["train"]
                    )
                metric_logger.compute_and_log(
                    mode="train",
                    additional_logs={
                        "losses": aux_losses,
                    },
                )
            if _dl_report_pending[0]:
                # Sequential path (TRAIN_PIPELINE=0): the first step's
                # fwd/bwd/opt is done here (the pipelined path reports right
                # after progress(), which owns the whole step).
                _dl_first_progress_done()
            train_batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            # Fine-grained in-window checkpoint triggers. All stamp
            # batch_idx_in_window so a crash here leaves a resumable partial
            # checkpoint, and all fire AFTER the metric update so restored
            # state reflects the just-completed batch. Triggers are mutually
            # short-circuited (one save per batch max) but evaluated on the
            # same deterministic counters across all ranks, so the collective
            # inside save_dmp_checkpoint stays in lockstep.
            gstep = metric_logger.global_step["train"]
            # Wall-clock elapsed is broadcast from rank 0 so every rank feeds
            # the same value into the (otherwise pure) cadence decision.
            elapsed = (
                _broadcast_elapsed()
                if effective_checkpoint_time_interval_s > 0
                else 0.0
            )
            save_reason = select_in_window_checkpoint_reason(
                train_batch_idx=train_batch_idx,
                global_step=gstep,
                elapsed_since_last_save=elapsed,
                in_window_checkpoint_frequency=in_window_checkpoint_frequency,
                checkpoint_step_frequency=checkpoint_step_frequency,
                checkpoint_time_interval_s=effective_checkpoint_time_interval_s,
            )
            if save_reason is not None:
                if rank == 0:
                    logger.info(
                        "checkpoint trigger=%s train_ts=%d batch=%d global_step=%d",
                        save_reason,
                        train_ts,
                        train_batch_idx,
                        gstep,
                    )
                _save_mid_window(train_ts, train_batch_idx)
                # Reset the wall-clock anchor on ANY save so the next time
                # trigger is measured from the most recent checkpoint.
                last_ckpt_time[0] = time.time()
            _progress_beat(rank, gstep)
            # Data-fraction eval cadence: run the full-holdout eval whenever the
            # monotonic global step crosses a multiple of eval_interval_steps
            # (i.e. every eval_every_data_pct of the training data). Keyed off
            # global_step (checkpoint-restored) so the eval grid is identical
            # across resume. Mid-window-safe: eval sets model.eval(), so restore
            # train mode + dataset.is_eval afterward. do_eval is None unless the
            # data-pct cadence is enabled.
            if (
                do_eval is not None
                and eval_interval_steps > 0
                and gstep > 0
                and gstep % eval_interval_steps == 0
            ):
                # SKIP_EVAL_EPOCH_PCT: skip this boundary (keep training) while
                # the run has not yet trained the requested fraction of the
                # epoch. The threshold is a pure function of the epoch
                # denominator + global batch size, and gstep is the
                # checkpoint-restored global step, so the skip set is identical
                # across resume and on every rank.
                if gstep < skip_eval_until_step:
                    if rank == 0:
                        logger.info(
                            "[data-pct-eval] SKIP eval train_ts=%d global_step=%d "
                            "< skip_eval_until_step=%d "
                            "(SKIP_EVAL_EPOCH_PCT=%.6g)",
                            train_ts,
                            gstep,
                            skip_eval_until_step,
                            skip_eval_epoch_pct,
                        )
                else:
                    if rank == 0:
                        logger.info(
                            "[data-pct-eval] trigger eval train_ts=%d global_step=%d "
                            "(interval=%d)",
                            train_ts,
                            gstep,
                            eval_interval_steps,
                        )
                    do_eval(train_ts, gstep)
                    model.train()
                    dataset.dataset.is_eval = False  # pyre-ignore [16]
                    # Data-fraction eval may hit the MLPerf target and emit RUN_STOP
                    # (via _do_eval_*). Stop the window immediately so we don't train
                    # past the convergence point; the outer window loop checks the
                    # same flag and breaks too.
                    if mlt.run_stopped:
                        break
            # Test-only: deterministic crash for the failure-injection test.
            # Triggered AFTER the save above, so on resume we re-enter at
            # batch_idx_in_window=train_batch_idx and emit batches [K+1, end).
            if (
                die_at_step >= 0
                and metric_logger.global_step["train"] >= die_at_step
            ):
                if rank == 0:
                    logger.warning(
                        "die_at_step=%d hit at train_ts=%d batch=%d global_step=%d "
                        "→ sys.exit(42)",
                        die_at_step,
                        train_ts,
                        train_batch_idx,
                        metric_logger.global_step["train"],
                    )
                # Distributed barrier so all ranks exit together rather than
                # leaving a few ranks hanging on NCCL ops.
                torch.distributed.barrier()
                import sys
                sys.exit(42)
            # `num_train_batches` cap: None or 0 = run the whole window. >0 caps
            # batches per window (mostly the streaming-resume test driver uses
            # this to keep test windows short). Under warm-keep the cap is already
            # folded into `window_num_batches` (and into the stream's own read
            # limit), so the check at the top of the loop owns the exit; breaking
            # here instead would read as an early break and reset away the next
            # window's prefetched batches.
            if (
                window_num_batches is None
                and num_train_batches
                and train_batch_idx >= num_train_batches
            ):
                break
        # Any early break — num_train_batches cap, mid-window convergence stop —
        # leaves up to two prefetched batches of THIS window queued, which would
        # otherwise be trained as the first steps of the NEXT window. Drop them.
        # The break conditions are rank-identical, so every rank discards the
        # same batches.
        #
        # A clean end-of-window is the opposite case and must NOT reset. Without
        # warm-keep there is nothing queued anyway (progress() raises
        # StopIteration only once its queue is empty, i.e. it already drained);
        # with warm-keep the queued batches are the next window's, and keeping
        # them is the entire point — resetting here would reintroduce the
        # per-seam drain plus a cold refill and, worse, silently drop batches
        # the next window still owes.
        if train_pipeline is not None and train_pipeline.batches and not window_done:
            if rank == 0:
                logger.info(
                    "[pipeline] window train_ts=%d exited early; dropping %d "
                    "prefetched batch(es)",
                    train_ts,
                    len(train_pipeline.batches),
                )
            train_pipeline.reset()
        if label and rank == 0 and first_wait is not None:
            logger.info(
                f"[boundary] {label} train first-batch data-wait={first_wait * 1000:.1f}ms"
            )

    def _run_eval_window(
        eval_data_iterator, label: Optional[str] = None
    ) -> Dict[str, float]:
        # DO NOT add a checkpoint trigger anywhere inside this function. The eval
        # data iterator's position is not serializable, so a checkpoint taken
        # mid-eval could not be resumed deterministically. `_maybe_checkpoint`
        # only fires after a completed eval window or mid-train-window, so any
        # restored state always sits on a completed-eval boundary -- which is
        # also why the eval reset below is safe across resume.
        #
        # Exclude this eval pass's wall-time from the train step-time window so
        # step_ms stays canonical even when eval coincides with a train interval;
        # the duration is reported separately (window_eval_time_ms + total_eval
        # below). Resumed unconditionally at the end of this function.
        metric_logger.pause_perf("eval")
        if profiler is not None:
            profiler.note_event("eval", label=label or "")
        # The pipeline replaces the sharded embedding modules' forward with a
        # PipelinedForward that expects its own prefetched input_dist context, so
        # a plain eval forward through them is invalid. detach() restores the
        # original forwards, and eval itself then runs correctly.
        #
        # Training resumes on the next progress(), which re-attaches lazily.
        # torchrec's attach() re-issues batches[0]'s input dist (stock
        # KJTAllToAll path) but never completes it; _StreamingTrainPipeline.attach
        # completes it with one wait_sparse_data_dist and stream-waits the
        # superseded pre-detach collectives before dropping them
        # (docs/04-technical-guides/recommendation/dlrm_v4/rocm-port.md T17).
        if train_pipeline is not None:
            train_pipeline.detach()
        model.eval()
        # Reset eval metrics so each pass reports a clean number over the FIXED
        # holdout set. Without this, lifetime/window eval metrics would keep
        # accumulating across eval steps (the old behavior, made worse now that
        # every step sees the identical set), making the eval-AUC trajectory
        # uninterpretable. With the reset, each eval point == AUC over the whole
        # fixed holdout at that train step -> directly comparable across steps.
        metric_logger.reset(mode="eval")
        eval_batch_idx = 0
        first_wait: Optional[float] = None
        _t_enter = time.perf_counter() if (label and rank == 0) else None
        nonlocal eval_pipeline
        if use_eval_pipeline and eval_pipeline is None:
            eval_pipeline = build_train_pipeline(
                model, optimizer, device, grad_clip_norm=grad_clip_norm,
                role="eval",
            )
            if rank == 0:
                logger.info(
                    "[pipeline] eval runs pipelined (forward-only progress())"
                )
            # Eval embedding lookahead (present only when the knob built the
            # eval-only subclass): the batch cap bounds its issue, and interim
            # eval metrics (collectives between batches) decline it.
            _evla_configure = getattr(eval_pipeline, "dlrm_evla_configure", None)
            if _evla_configure is not None:
                _evla_configure(num_eval_batches, bool(eval_interim_metrics))
        with torch.no_grad():
            while True:
                _t_next = time.perf_counter() if (label and rank == 0) else None
                if eval_pipeline is not None:
                    # progress() owns the pull, the H2D copy and the input dist,
                    # and returns the wrapper's output tuple. It raises
                    # StopIteration once the window's batches are all consumed.
                    # The fetch is not separable from the forward here, so the
                    # first-batch timing below covers the whole fill.
                    try:
                        (
                            _,
                            mt_target_preds,
                            mt_target_labels,
                            mt_target_weights,
                            num_candidates,
                        ) = eval_pipeline.progress(eval_data_iterator)
                    except StopIteration:
                        break
                    if _t_next is not None and first_wait is None:
                        first_wait = time.perf_counter() - _t_next
                else:
                    try:
                        sample = next(eval_data_iterator)
                    except StopIteration:
                        break
                    if _t_next is not None and first_wait is None:
                        first_wait = time.perf_counter() - _t_next
                    sample.to(device)
                    (
                        _,
                        _,
                        _,
                        mt_target_preds,
                        mt_target_labels,
                        mt_target_weights,
                    ) = (
                        # build_train_pipeline leaves the model in
                        # `_pipeline_mode`, where forward takes the batch as its
                        # single argument and reads the pre-merged sparse KJT
                        # off it.
                        model.forward(sample)
                        if train_pipeline is not None
                        else model.forward(
                            sample.uih_features_kjt,
                            sample.candidates_features_kjt,
                        )
                    )
                    num_candidates = sample.candidates_features_kjt.lengths().view(
                        len(sample.candidates_features_kjt.keys()), -1
                    )[0]
                metric_logger.update(
                    mode="eval",
                    predictions=mt_target_preds,
                    labels=mt_target_labels,
                    weights=mt_target_weights,
                    num_candidates=num_candidates,
                )
                eval_batch_idx += 1
                _progress_beat(rank, -eval_batch_idx)
                if output_trace:
                    assert profiler is not None
                    profiler.step()
                if eval_interim_metrics and metric_log_due(
                    eval_batch_idx, metric_log_frequency
                ):
                    metric_logger.compute_and_log(mode="eval")
                if num_eval_batches is not None and eval_batch_idx >= num_eval_batches:
                    break
            # Hand the sharded modules back before training resumes, and drop
            # any batches still queued -- the `num_eval_batches` break leaves
            # two in flight, and they belong to an iterator this window closes.
            # reset() after detach() so the next window rewrites the model from
            # the true original forwards rather than from the train pipeline's.
            if eval_pipeline is not None:
                eval_pipeline.detach()
                eval_pipeline.reset()
            _eval_metrics = metric_logger.compute(mode="eval")
            for k, v in _eval_metrics.items():
                print(f"{k}: {v}")
        if label and rank == 0 and _t_enter is not None:
            _eval_total = time.perf_counter() - _t_enter
            _fw = (first_wait * 1000) if first_wait is not None else float("nan")
            logger.info(
                f"[boundary] {label} eval first-batch data-wait={_fw:.1f}ms "
                f"total_eval={_eval_total * 1000:.1f}ms batches={eval_batch_idx}"
            )
            # Dedicated per-eval metrics sink. One JSON line per eval boundary
            # capturing the END-OF-PASS metric over the FIXED holdout -- the single
            # correct value for that eval point (no interim/averaging ambiguity).
            # Rank 0 only; append-only so it survives restarts and the trajectory
            # accumulates across resumes. Written next to the main run log
            # ("<LOG>.metrics.jsonl"), falling back to cwd if LOG is unset.
            import json
            import re as _re

            _log = os.environ.get("LOG")
            if _log:
                _base = _log[:-4] if _log.endswith(".log") else _log
                _metrics_path = f"{_base}.metrics.jsonl"
            else:
                _metrics_path = "streaming_eval_metrics.jsonl"
            _ts_m = _re.search(r"train_ts=(\d+)", label)
            _rec = {
                "label": label,
                "train_ts": int(_ts_m.group(1)) if _ts_m else None,
                "global_step": int(metric_logger.global_step.get("train", -1)),
                "eval_batches": eval_batch_idx,
                "total_eval_ms": round(_eval_total * 1000, 1),
                "wall_time": time.time(),
            }
            for _k, _v in _eval_metrics.items():
                try:
                    _rec[_k] = float(_v)
                except (TypeError, ValueError):
                    pass
            try:
                with open(_metrics_path, "a") as _f:
                    _f.write(json.dumps(_rec) + "\n")
            except OSError as _e:
                logger.warning("failed to write metrics sink %s: %s", _metrics_path, _e)
        metric_logger.resume_perf("eval")
        # Return metrics (on every rank) so the MLPerf eval hooks can consume them.
        return _eval_metrics

    def _maybe_checkpoint(train_ts: int) -> None:
        if (
            train_ts % checkpoint_frequency == 0 and train_ts > 0
        ) or train_ts == start_ts + num_train_ts - 1:
            # End-of-window save: stamp WINDOW_COMPLETE so resume advances past
            # this train_ts. `device` enables per-rank RNG snapshot for
            # bit-equal resume of dropout-bearing modules.
            save_dmp_checkpoint(
                model=model,
                optimizer=optimizer,
                metric_logger=metric_logger,
                rank=rank,
                batch_idx=train_ts,
                train_ts=train_ts,
                batch_idx_in_window=WINDOW_COMPLETE,
                device=device,
                split_contract=live_split_contract,
            )
            last_ckpt_time[0] = time.time()

    # Apply start_ts shift from resume (may have moved past the original start).
    # num_train_ts is the requested *count*; preserve it so the loop runs for
    # the same total number of windows post-resume as a fresh run would have.
    train_ts_list = list(range(start_ts, start_ts + num_train_ts))
    n_train = len(train_ts_list)

    def _should_eval(i: int) -> bool:
        """Whether to run the full-holdout eval after training window index `i`.

        Single cadence knob `eval_every_n_windows`:
          * <=0 -> eval disabled entirely (train-only; e.g. perf benchmarking or
            the resume test). The eval dataloader is not even built.
          * 1 (default) -> eval after every window.
          * K>1 -> eval when the ABSOLUTE window ts is on the grid anchored at
            `eval_anchor_ts` (the original start_ts), i.e. ts in {anchor,
            anchor+K, anchor+2K, ...}, and ALWAYS on the final window so the
            trajectory ends with an eval point. Anchoring to the absolute ts
            (not the per-call loop index `i`) keeps the eval grid (e.g.
            150,160,170,...) stable across a mid-run resume, which rebases
            start_ts/`train_ts_list` to the resume window.
        """
        if eval_every_n_windows <= 0:
            return False
        if eval_every_n_windows == 1:
            return True
        return (train_ts_list[i] - eval_anchor_ts) % eval_every_n_windows == 0 or i == n_train - 1

    def _skip_early_window_eval(i: int) -> bool:
        """SKIP_EVAL_EPOCH_PCT for the PER-WINDOW cadence: skip eval-eligible
        windows until the run has trained `skip_eval_epoch_pct` of the epoch, so
        early (low-signal) eval passes are not run. Keyed off the monotonic,
        checkpoint-restored global_step against the once-resolved
        `skip_eval_until_step`, so the skip set is stable across a mid-run
        resume and identical on every rank. The final window's eval is never
        skipped so the trajectory always ends with an eval point.
        """
        if skip_eval_until_step <= 0 or i == n_train - 1:
            return False
        return metric_logger.global_step["train"] < skip_eval_until_step

    # Fixed eval set: held-out users' anchors over the resolved holdout window
    # range, computed ONCE and reused at every eval step. Same anchors every
    # step -> stable, comparable eval-AUC curve, and bounded eval time
    # (~(1 - train_split_percentage) of a window). Cached inside the dataset so
    # re-deriving it (e.g. on resume) returns the identical set. None for
    # datasets without holdout support (synthetic) -> legacy per-window eval.
    eval_global_indices: Optional["np.ndarray"] = None
    if supports_holdout:
        eval_global_indices = dataset.dataset.eval_holdout_indices(  # pyre-ignore [16]
            eval_holdout_ts_resolved, eval_holdout_num_windows
        )
        if rank == 0:
            logger.info(
                "Fixed eval holdout: ts=[%d, %d) -> %d anchors (train_split_percentage=%s)",
                eval_holdout_ts_resolved,
                eval_holdout_ts_resolved + eval_holdout_num_windows,
                len(eval_global_indices),
                dataset.dataset._train_split_percentage,  # pyre-ignore[16]
            )

    # --- MLPerf run tracking --------------------------------------------------
    # total_train_samples = epoch_num denominator (global trainable samples over
    # the window range), computed once and logged as TRAIN_SAMPLES. Also the
    # denominator SKIP_EVAL_EPOCH_PCT is expressed against, so it is needed even
    # when MLPerf logging is off.
    total_train_samples = 0
    if mlperf_logger is not None or skip_eval_epoch_pct != 0:
        _idx_fn = getattr(
            dataset.dataset, "train_window_indices", None
        ) or getattr(dataset.dataset, "window_indices", None)
        if _idx_fn is not None:
            for _ts in train_ts_list:
                total_train_samples += int(_idx_fn(_ts).size)
        if rank == 0:
            logger.info(
                "MLPerf: total_train_samples=%d over %d windows",
                total_train_samples,
                n_train,
            )
    # SKIP_EVAL_EPOCH_PCT -> global-step threshold, resolved ONCE here (both
    # cadences below consult it). Depends only on the epoch denominator and the
    # global batch size, both of which are identical on every rank and across a
    # resume, so the skipped set is too.
    skip_eval_until_step = resolve_skip_eval_until_step(
        skip_eval_epoch_pct=skip_eval_epoch_pct,
        total_train_samples=total_train_samples,
        global_batch_size=global_batch_size,
    )
    if rank == 0 and skip_eval_epoch_pct != 0:
        logger.info(
            "[skip-eval] skip_eval_epoch_pct=%.6g%s -> skip_eval_until_step=%d "
            "(total_train_samples=%d global_batch_size=%d -> "
            "skip_until_samples=%d)",
            skip_eval_epoch_pct,
            " (MLPerf rule)" if skip_eval_epoch_pct < 0 else "",
            skip_eval_until_step,
            total_train_samples,
            global_batch_size,
            int(skip_eval_until_step * global_batch_size),
        )

    if mlperf_logger is not None:
        # Wire the logger + LR getter so MetricsLogger.compute emits train_loss.
        metric_logger.mlperf_logger = mlperf_logger

        def _current_lr() -> float:
            return float(optimizer.param_groups[0]["lr"])

        metric_logger.lr_getter = _current_lr

    # Centralized MLPerf run-boundary state machine: owns block/eval/run markers,
    # SAMPLES_COUNT/EPOCH_NUM progress metadata, and the per-window-AUC vs
    # auc_threshold convergence decision. Every method no-ops when mlperf_logger
    # is None, so the loop below calls them unconditionally.
    from generative_recommenders.dlrm_v4.train.mlperf_logging_utils import (
        MLPerfRunTracker,
    )

    mlt = MLPerfRunTracker(
        logger=mlperf_logger,
        metric_logger=metric_logger,
        total_train_samples=total_train_samples,
        rank=rank,
        device=device,
    )
    mlt.log_dataset_sizes(
        eval_samples=eval_global_indices.size
        if eval_global_indices is not None
        else None
    )

    # Pre-forked eval pool, published for the final eval below. Building a fresh
    # eval loader after training forks workers while the train pipeline still
    # holds its in-flight batches; those were copied on a side stream, so the
    # child frees blocks with recorded stream uses and the allocator's
    # cudaEventRecord fails with "initialization error" (no CUDA context in a
    # forked child). Reusing this pool makes the final eval a worker reset
    # instead of a fork. Stays None when eval is disabled entirely (train-only
    # perf runs build no pool), which keeps those runs' fork count unchanged.
    final_eval_dl: Optional[DataLoader] = None

    if persistent_loader and double_buffer:
        # Double-buffered: next window prepared in the background during the
        # current window's compute. Eval (if enabled) uses its own pre-forked
        # pool, primed up front on the main thread so no fork races a bg thread.
        if _pre is not None:
            # Pools already forked before RUN_START (empty windows).
            prefetcher = _pre["prefetcher"]
        else:
            # Same pools; the probe wrapper only stamps the first worker read
            # for the [dlstart] startup timeline.
            _train_ds = _ReadProbedDataset(dataset, _dlstart.new_read_slot("train"))
            prefetcher = _PrefetchingWindowLoader(
                dataset=dataset,
                sampler_factory=lambda: StreamingWindowSampler(rank, world_size),
                dl_factory=lambda s: make_persistent_streaming_dataloader(
                    dataset=_train_ds, sampler=s
                ),
            )
        eval_sampler: Optional[StreamingWindowSampler] = None
        eval_dl: Optional[DataLoader] = None
        # Eval iterator is built one window ahead: the eval pool (idle while the
        # current train window runs) prefetches the next eval window's first
        # batches concurrently with train compute, so eval starts warm. yambda's
        # sample content depends only on the sampler window, not is_eval, so
        # prefetching during train is safe.
        eval_iter: Optional[Iterator] = None
        # Build/fork the eval pool when EITHER cadence needs it: the per-window
        # cadence (eval_every_n_windows>0) or the data-fraction cadence
        # (eval_interval_steps>0). Both are never simultaneously on (validated
        # at entry), so this is "eval is enabled at all".
        if (eval_every_n_windows > 0 or eval_interval_steps > 0) and len(
            train_ts_list
        ) > 0:
            eval_sampler = StreamingWindowSampler(rank, world_size)
            # $EVAL_NUM_WORKERS (gin default 8; < 0 inherits NUM_WORKERS, the
            # pre-2026-10-05 pool). The holdout eval pass is input-pipeline
            # bound at 4 workers/rank (~255-285 ms per 1024-sample batch, as
            # long as a training step); 8 workers halve it (Arbor iter-13/15:
            # 64-70 s -> 33-40 s per 1n eval, identical eval_accuracy). Only
            # this persistent eval pool changes; train pools keep NUM_WORKERS.
            _eval_dl_kwargs = resolve_eval_dataloader_kwargs(eval_num_workers)
            if rank == 0:
                logger.warning(eval_pool_log_line(_eval_dl_kwargs))
            _pre_eval = _pre.get("eval_dl") if _pre is not None else None
            if _pre_eval is None:
                eval_dl = make_persistent_streaming_dataloader(
                    dataset=_eval_dataset(
                        dataset, _dlstart.new_read_slot("eval")
                    ),
                    sampler=eval_sampler,
                    **_eval_dl_kwargs,
                )
            # CRITICAL: fork the eval worker pool HERE, on the main thread,
            # BEFORE prefetcher.stream() below spins up its background prep
            # thread. The pool is persistent_workers=True, so this first iter()
            # is the ONLY fork; every later iter() merely resets and reuses these
            # workers (no fork), so it can never deadlock against the background
            # thread holding an allocator/GIL-released lock. (Deferring this
            # first fork into the loop — as a sparse-eval cadence naively might —
            # hangs the run.) _should_eval(0) is always True when eval is enabled
            # (0 % K == 0). The eval set is the FIXED holdout (same every step),
            # so we install it on the sampler ONCE here; later evals just call
            # iter() again to replay the identical set (no set_window churn).
            if _pre_eval is None:
                eval_sampler.set_window(eval_global_indices)
                eval_iter = iter(eval_dl)
            else:
                # DL_PRESTART: the eval pool was forked before RUN_START with an
                # empty window. Replay the iterator-creation RNG draw the legacy
                # iter() makes here, then install the fixed holdout set.
                eval_sampler = _pre["eval_sampler"]
                eval_dl = _pre_eval
                _draw_like_iterator_creation()
                eval_sampler.set_window(eval_global_indices)
                if eval_every_n_windows > 0:
                    # The per-window path consumes this iterator at window 0.
                    eval_iter = iter(eval_dl)
                # else (data-fraction cadence): _do_eval_db opens its own
                # iter(eval_dl) at each eval, so the legacy startup fill was
                # read and then discarded unconsumed by that reset -- ~prefetch
                # x workers eval batches read inside the timed region while the
                # first train batches load. Skipping it changes no eval: same
                # holdout set, order and metric.
            final_eval_dl = eval_dl
            _dlstart.mark("eval_pool_ready")
        elif _pre is not None and _pre.get("eval_dl") is not None:
            # DL_PRESTART forked an eval pool this run never uses (its eval
            # predicate differs from the loop's). Shut the idle workers down
            # instead of leaking them for the whole run.
            if rank == 0:
                logger.warning("[dlstart] pre-started eval pool unused; shutting it down")
            try:
                _it = getattr(_pre["eval_dl"], "_iterator", None)
                if _it is not None:
                    _it._shutdown_workers()
            except Exception as _e:  # never fail a run over an idle pool
                logger.warning("[dlstart] eval pool shutdown failed: %s", _e)

        # Data-fraction eval callback (double-buffer path). Fired mid-window by
        # _run_train_window on the global-step cadence. Reuses the already-forked
        # persistent eval pool: iter(eval_dl) here runs on the MAIN thread (a
        # reset, not a fork — the only fork was the up-front iter() above), so it
        # stays safe alongside the background window-prefetch thread.
        def _do_eval_db(train_ts: int, gstep: int) -> None:
            # Data-fraction eval boundary: this closes the current MLPerf block,
            # runs the holdout eval with full EVAL_START/EVAL_STOP + EVAL_ACCURACY
            # + convergence, then opens the next block. The block thus brackets
            # exactly one eval_interval_steps of training (MLPerf block == work
            # between two evals), instead of one timestamp window.
            dataset.dataset.is_eval = True  # pyre-ignore [16]
            assert eval_dl is not None
            mlt.block_stop()
            mlt.eval_start()
            eval_metrics = _run_eval_window(
                iter(eval_dl),
                label=f"eval_holdout@train_ts={train_ts}@step={gstep}",
            )
            # Emits RUN_STOP (sets mlt.run_stopped) if the target is met;
            # _run_train_window / the window loop break on that flag.
            mlt.eval_stop(eval_metrics)
            if not mlt.run_stopped:
                mlt.block_start()

        _db_do_eval = _do_eval_db if eval_interval_steps > 0 else None

        # Block placement depends on the eval cadence. Per-window cadence
        # (eval_every_n_windows>0): one block per timestamp window. Otherwise
        # (data-fraction cadence, or no eval): a single block spans the whole
        # run, split at each data-fraction eval boundary by _do_eval_db. Open
        # the first block here for the latter; the boundary helper + the
        # post-loop stop handle the rest.
        _per_window_blocks = eval_every_n_windows > 0
        if not _per_window_blocks:
            mlt.block_start()

        # Pipeline warm-keep ($TP_WARM_WINDOWS, default 1): chain the per-window
        # iterators so the pipeline prefetches across window seams instead of
        # draining and refilling at each one. Needs the prefetcher's independent
        # per-buffer worker pools, so it rides the double-buffer path only, and
        # it is meaningless without the pipeline.
        #
        # A num_train_batches cap is handed to the stream so it caps its own
        # reads (see _WarmWindowStream): every batch pulled is then a batch
        # trained, so the cap needs no drop and warm-keep stays measurable under
        # the short-window perf harness.
        _warm_windows = (
            train_pipeline is not None
            and os.environ.get("TP_WARM_WINDOWS", "1") == "1"
        )
        # Item (36)'s eval holdout read-ahead: with the streaming walk on
        # ($DL_COLD_PREFETCH_STREAM), the walk also issues this rank's holdout
        # ranges ahead of the first data-fraction eval, which the walk would
        # otherwise leave cold (cold_prefetch module docstring). It is timed
        # by the warm-keep stream's pull counts (the only consumer position
        # the walk sees), so it is planned only with warm-keep on. Metadata
        # only: no file access here; inert without the walk.
        if (
            _warm_windows
            and eval_dl is not None
            and eval_interval_steps > 0
            and eval_global_indices is not None
        ):
            _cold_prefetch.set_eval_plan(
                eval_global_indices,
                int(getattr(eval_dl, "batch_size", 0) or 0),
                first_data_pct_eval_steps(
                    int(metric_logger.global_step.get("train", 0)),
                    eval_interval_steps,
                    skip_eval_until_step,
                ),
                num_eval_batches,
            )
        else:
            _cold_prefetch.set_eval_plan(None, 0, 0)
        if train_pipeline is not None and rank == 0:
            logger.info(
                "[pipeline] window warm-keep %s",
                "ON (prefetch crosses window boundaries)"
                if _warm_windows
                else "OFF (pipeline drains/refills at each window boundary)",
            )

        def _window_feed():
            """Yield (ts, iterator, num_batches) per training window.

            Warm-keep hands back the SAME continuous iterator every window, with
            num_batches bounding it; otherwise each window gets its own iterator
            and ends on StopIteration (num_batches=None), exactly as before.
            """
            # Only the FIRST window after a mid-window resume needs the skip
            # (handed via prefetcher.stream's first_skip_samples). The skip is
            # zero on cold start (resume_train_ts is None → first_skip_samples=0)
            # and on completed-window resume (mid-window slice is 0 too).
            stream = prefetcher.stream(
                train_ts_list,
                first_skip_samples=first_skip_samples,
                window_cap=num_train_batches,
                warm=_warm_windows,
                per_window_cap=num_train_batches,
                first_window_trained=(
                    resume_batch_idx_in_window
                    if resume_batch_idx_in_window > 0
                    else 0
                ),
            )
            if not _warm_windows:
                for _ts, _it, _n in stream:
                    yield _ts, _it, None
                return
            warm = _WarmWindowStream(
                stream,
                per_window_cap=num_train_batches,
                first_window_trained=(
                    resume_batch_idx_in_window
                    if resume_batch_idx_in_window > 0
                    else 0
                ),
            )
            while True:
                try:
                    _ts, _n = warm.advance()
                except StopIteration:
                    return
                yield _ts, warm, _n

        if teardown_pools is not None:
            # $DL_TEARDOWN_CLOSE_POOLS: the caller closes these after RUN_STOP
            # (eval_dl is also final_eval_dl).
            teardown_pools.append(prefetcher)
            if eval_dl is not None:
                teardown_pools.append(eval_dl)

        for i, (train_ts, train_data_iterator, window_num_batches) in enumerate(
            _window_feed()
        ):
            dataset.dataset.is_eval = False  # pyre-ignore [16]
            # First iteration after a mid-window resume carries
            # resume_batch_idx_in_window so in-window saves and the die_at_step
            # hook keep accurate counters; otherwise count from 0.
            start_batch = (
                resume_batch_idx_in_window
                if i == 0 and resume_batch_idx_in_window > 0
                else 0
            )
            if i == 0:
                _collect_and_freeze_gc()
            # Rendezvous all ranks at the window boundary BEFORE the first
            # forward of this window. The prefetcher has already handed back a
            # ready iterator (this window's window_indices mmap scan is done),
            # but that O(N) scan over the ~18 GB anchor_ts array can finish at
            # very different times across ranks. Without this barrier a fast
            # rank issues the first embedding all-to-all while a slow rank is
            # still in prep, desyncing the NCCL collective stream and hanging
            # the job (observed at a window boundary via the flight recorder:
            # ranks split across consecutive collective seq ids). This only
            # absorbs prep skew (one near-zero sync per window); it does not
            # serialize the background prefetch of future windows.
            if rank == 0 and os.environ.get("DL_WBND_DIAG", "0") == "1":
                # Diagnostic (off by default): window -> global step map for
                # per-boundary step-time analysis.
                logger.warning(
                    "[wbnd] window ts=%d first_step=%d num_batches=%s t=%.3f",
                    train_ts,
                    int(metric_logger.global_step.get("train", -1)),
                    window_num_batches,
                    time.time(),
                )
            _traced_window_boundary(profiler, device, world_size, train_ts)
            if _per_window_blocks:
                mlt.block_start()
            _run_train_window(
                train_data_iterator,
                train_ts=train_ts,
                start_batch_idx=start_batch,
                label=f"train_ts={train_ts}",
                do_eval=_db_do_eval,
                window_num_batches=window_num_batches,
            )
            if _per_window_blocks:
                mlt.block_stop()
            should_stop = False
            if _should_eval(i) and not _skip_early_window_eval(i):
                dataset.dataset.is_eval = True  # pyre-ignore [16]
                assert eval_sampler is not None and eval_dl is not None
                mlt.eval_start()
                eval_metrics = _run_eval_window(
                    eval_iter, label=f"eval_holdout@train_ts={train_ts}"
                )
                should_stop = mlt.eval_stop(eval_metrics)
                # Re-arm the (already-forked) eval pool for the NEXT eval. The
                # holdout set is fixed, so the sampler window is unchanged; we
                # only need a fresh iter() to replay it. iter() reuses the
                # persistent workers — no fork, safe alongside the bg thread.
                next_eval_i = next(
                    (j for j in range(i + 1, n_train) if _should_eval(j)), None
                )
                if next_eval_i is not None:
                    eval_iter = iter(eval_dl)
            elif _should_eval(i) and rank == 0:
                logger.info(
                    "[per-window-eval] SKIP eval train_ts=%d global_step=%d "
                    "< skip_eval_until_step=%d (SKIP_EVAL_EPOCH_PCT=%.6g)",
                    train_ts,
                    metric_logger.global_step["train"],
                    skip_eval_until_step,
                    skip_eval_epoch_pct,
                )
            _maybe_checkpoint(train_ts)
            # should_stop: per-window convergence. mlt.run_stopped:
            # data-fraction convergence (RUN_STOP fired mid-window by _do_eval_db).
            if should_stop or mlt.run_stopped:
                # MLPerf target reached: RUN_STOP already emitted; stop training.
                break

        # Close the run-spanning block for the data-fraction / no-eval case.
        # Idempotent: a no-op if the last eval boundary already closed it (i.e.
        # convergence stopped the run) or if per-window blocks were used.
        if not _per_window_blocks:
            mlt.block_stop()
    else:
        # M4 (default OFF) needs the single-pool train loader: decline here.
        _m4p.decline_if_requested(
            rank, "no single-pool train loader (persistent_loader=0 or double_buffer=0)"
        )
        # Data-fraction eval callback (non-double-buffer path). Builds a fresh
        # eval dataloader per call over the FIXED holdout set (or the legacy
        # next-window eval when the dataset has no holdout support).
        def _do_eval_nb(train_ts: int, gstep: int) -> None:
            # Data-fraction eval boundary (non-double-buffer path). See _do_eval_db:
            # close the current MLPerf block, run the eval with full markers +
            # convergence, then open the next block so a block brackets one
            # eval_interval_steps of training rather than a timestamp window.
            dataset.dataset.is_eval = True  # pyre-ignore [16]
            mlt.block_stop()
            mlt.eval_start()
            if eval_global_indices is not None:
                eval_metrics = _run_eval_window(
                    iter(
                        make_streaming_dataloader(
                            dataset=eval_dataset, indices=eval_global_indices
                        )
                    ),
                    label=f"eval_holdout@train_ts={train_ts}@step={gstep}",
                )
            else:
                eval_metrics = _run_eval_window(
                    iter(
                        make_streaming_dataloader(dataset=eval_dataset, ts=train_ts + 1)
                    ),
                    label=f"eval@train_ts={train_ts}@step={gstep}",
                )
            mlt.eval_stop(eval_metrics)
            if not mlt.run_stopped:
                mlt.block_start()

        _nb_do_eval = _do_eval_nb if eval_interval_steps > 0 else None

        # Warm-keep needs the prefetcher's independent per-buffer worker pools:
        # this path mutates ONE sampler per window, so the next window's iterator
        # cannot be opened while the current one is still being pulled. The
        # pipeline therefore drains and refills at every boundary here; say so
        # rather than silently running slower than the double-buffer path.
        if (
            train_pipeline is not None
            and os.environ.get("TP_WARM_WINDOWS", "1") == "1"
            and rank == 0
        ):
            logger.info(
                "[pipeline] window warm-keep unavailable without double_buffer=1"
                " (single shared sampler); pipeline drains at each boundary."
            )

        # See the double-buffer branch: per-window blocks for the per-window
        # cadence, else a single run-spanning block split at data-fraction eval
        # boundaries by _do_eval_nb.
        _per_window_blocks = eval_every_n_windows > 0
        if not _per_window_blocks:
            mlt.block_start()

        for i, train_ts in enumerate(train_ts_list):
            dataset.dataset.is_eval = False  # pyre-ignore [16]
            skip = first_skip_samples if i == 0 else 0
            start_batch = (
                resume_batch_idx_in_window
                if i == 0 and resume_batch_idx_in_window > 0
                else 0
            )
            if i == 0:
                _collect_and_freeze_gc()
            # See the double-buffer path: rendezvous all ranks at the window
            # boundary before the first forward so per-rank data-prep skew
            # cannot desync the NCCL collective stream and hang the job.
            _traced_window_boundary(profiler, device, world_size, train_ts)
            if _per_window_blocks:
                mlt.block_start()
            _run_train_window(
                _window_iter(train_ts, skip_samples=skip),
                train_ts=train_ts,
                start_batch_idx=start_batch,
                do_eval=_nb_do_eval,
            )
            if _per_window_blocks:
                mlt.block_stop()
            should_stop = False
            if _should_eval(i) and not _skip_early_window_eval(i):
                dataset.dataset.is_eval = True  # pyre-ignore [16]
                mlt.eval_start()
                if eval_global_indices is not None:
                    eval_metrics = _run_eval_window(
                        iter(
                            make_streaming_dataloader(
                                dataset=eval_dataset, indices=eval_global_indices
                            )
                        ),
                        label=f"eval_holdout@train_ts={train_ts}",
                    )
                else:
                    # Legacy per-window eval (datasets without user holdout).
                    eval_metrics = _run_eval_window(
                        iter(
                            make_streaming_dataloader(
                                dataset=eval_dataset, ts=train_ts + 1
                            )
                        )
                    )
                should_stop = mlt.eval_stop(eval_metrics)
            elif _should_eval(i) and rank == 0:
                logger.info(
                    "[per-window-eval] SKIP eval train_ts=%d global_step=%d "
                    "< skip_eval_until_step=%d (SKIP_EVAL_EPOCH_PCT=%.6g)",
                    train_ts,
                    metric_logger.global_step["train"],
                    skip_eval_until_step,
                    skip_eval_epoch_pct,
                )
            _maybe_checkpoint(train_ts)
            # should_stop: per-window convergence. mlt.run_stopped:
            # data-fraction convergence (RUN_STOP fired mid-window by _do_eval_nb).
            if should_stop or mlt.run_stopped:
                # MLPerf target reached: RUN_STOP already emitted; stop training.
                break

        # Close the run-spanning block for the data-fraction / no-eval case
        # (idempotent; no-op under per-window blocks or after a convergence stop).
        if not _per_window_blocks:
            mlt.block_stop()

    # Final eval over the fixed user-holdout set (legacy final-window eval
    # otherwise). Skipped if the MLPerf target already stopped the run mid-run.
    if not mlt.run_stopped:
        dataset.dataset.is_eval = True  # pyre-ignore [16]
        mlt.eval_start()
        if eval_global_indices is not None:
            # The pre-forked pool's sampler already holds eval_global_indices
            # (set once at startup, never re-windowed), so iter() here replays
            # the identical holdout set without forking.
            final_metrics = _run_eval_window(
                iter(final_eval_dl)
                if final_eval_dl is not None
                else iter(
                    make_streaming_dataloader(
                        dataset=eval_dataset, indices=eval_global_indices
                    )
                ),
                label="eval_holdout@final",
            )
        else:
            final_metrics = _run_eval_window(
                iter(
                    make_streaming_dataloader(dataset=eval_dataset, ts=num_train_ts)
                ),
                label="eval@final",
            )
        mlt.eval_stop(final_metrics)
        if rank == 0:
            for k, v in final_metrics.items():
                print(f"{k}: {v}")
        # End-of-run RUN_STOP: SUCCESS if final metric met target, else ABORTED.
        mlt.finalize(final_metrics)
