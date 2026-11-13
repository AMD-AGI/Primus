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

# pyre-unsafe
"""
Dataset implementations for DLRMv4.

This module provides dataset classes for loading and processing recommendation
data, including sample containers, collation functions, and random data generation.
"""

import copy
import logging
import mmap
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Pipelineable


logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger("dlrmv4_dataset")


# $DL_FAST_BATCH (default 0 = off): build each DataLoader batch with numpy
# straight into the key-major batch layout (DLRMv4YambdaDataset.build_batch,
# called from HammerToTorchDataset.__getitems__) instead of one Python-list
# KJT per sample + collate_fn. Batches are bit-for-bit identical. Read once
# per process; forked DataLoader workers inherit the cached value. This is the
# train pools' switch (and that of every loader that is not an eval loader);
# yambda_5b.gin item (61) sets it (gin default 1) through configure_fast_batch.
_FAST_BATCH: Optional[bool] = None

# $DL_FAST_BATCH_EVAL: the eval loaders' own switch, so the eval passes can use
# build_batch while the train pools stay on the per-sample path (or the
# reverse). > 0 = eval loaders use build_batch, 0 = the per-sample path, < 0 or
# unset = follow $DL_FAST_BATCH. yambda_5b.gin item (32) resolves it (gin
# default 1, env override) through configure_fast_batch; without that call
# (tests, tools that do not parse the gin) eval_fast_batch() reads the env.
# The eval loaders' HammerToTorchDataset carries the result (``fast_batch``,
# set in the main process before the pools fork, inherited by the workers);
# while that wrapper fetches, fast_batch_enabled() returns it in place of
# $DL_FAST_BATCH, so the whole fetch -- build_batch, or the per-sample builder
# it falls back to -- follows the eval setting.
_FAST_BATCH_EVAL: Optional[int] = None
# True once configure_fast_batch installed a gin value for that switch (arming
# lines: "gin default" vs "code default").
_FAST_BATCH_FROM_GIN: Dict[str, bool] = {"train": False, "eval": False}
_FAST_BATCH_SCOPE = threading.local()


def configure_fast_batch(
    train_pools: Optional[int] = None, eval_pools: Optional[int] = None
) -> None:
    """Install the switches resolved by gin (yambda_5b.gin items (32), (61), via
    train/utils.py fast_batch_pools: env_int, so a set env var wins over the
    gin default) in this process. Called in the main process before any
    DataLoader pool forks; the workers inherit the values. None leaves that
    switch on its env-only code default. Idempotent."""
    global _FAST_BATCH, _FAST_BATCH_EVAL
    if train_pools is not None:
        _FAST_BATCH = int(train_pools) > 0
        _FAST_BATCH_FROM_GIN["train"] = True
    if eval_pools is not None:
        _FAST_BATCH_EVAL = int(eval_pools)
        _FAST_BATCH_FROM_GIN["eval"] = True


def eval_fast_batch() -> Optional[bool]:
    """The eval loaders' setting: True (build_batch), False (per-sample path)
    or None (follow $DL_FAST_BATCH). The configure_fast_batch value when gin
    set one, else $DL_FAST_BATCH_EVAL (unset or empty = None; a non-integer is
    off, as for DL_FAST_BATCH). Not cached: the main process calls it when it
    builds an eval loader's wrapper, which carries the result to the workers."""
    value = _FAST_BATCH_EVAL
    if value is None:
        raw = os.environ.get("DL_FAST_BATCH_EVAL", "").strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            logger.warning(
                "DL_FAST_BATCH_EVAL=%r is not an integer; eval loaders use the "
                "per-sample path (set DL_FAST_BATCH_EVAL=1 to enable)",
                raw,
            )
            return False
    return None if value < 0 else value > 0


class fast_batch_scope:
    """``with fast_batch_scope(on):`` fast_batch_enabled() returns ``on`` in
    this thread until exit (an eval loader's fetch). None = no override."""

    def __init__(self, on: Optional[bool]) -> None:
        self._on = on
        self._prev: Optional[bool] = None

    def __enter__(self) -> None:
        self._prev = getattr(_FAST_BATCH_SCOPE, "on", None)
        if self._on is not None:
            _FAST_BATCH_SCOPE.on = bool(self._on)

    def __exit__(self, *exc) -> None:
        _FAST_BATCH_SCOPE.on = self._prev


def fast_batch_knob() -> str:
    """The switch behind fast_batch_enabled() at this point, as NAME=0|1 (the
    per-worker arming line; only the eval loaders' wrappers open a scope)."""
    scoped = getattr(_FAST_BATCH_SCOPE, "on", None)
    if scoped is not None:
        return f"DL_FAST_BATCH_EVAL={int(scoped)}"
    return f"DL_FAST_BATCH={int(fast_batch_enabled())}"


def fast_batch_knob_source(which: str) -> str:
    """Where the value of switch ``which`` ("train" = DL_FAST_BATCH, "eval" =
    DL_FAST_BATCH_EVAL) came from: "env", "gin default" or "code default"."""
    name = "DL_FAST_BATCH" if which == "train" else "DL_FAST_BATCH_EVAL"
    if os.environ.get(name, "").strip():
        return "env"
    return "gin default" if _FAST_BATCH_FROM_GIN.get(which) else "code default"


def fast_batch_enabled() -> bool:
    """Whether the batch being built uses build_batch: inside an eval loader's
    fetch, that loader's own setting (fast_batch_scope); else $DL_FAST_BATCH
    (the gin item (61) value once configure_fast_batch ran, else the env;
    resolved once per process, inherited by forked workers)."""
    scoped = getattr(_FAST_BATCH_SCOPE, "on", None)
    if scoped is not None:
        return scoped
    global _FAST_BATCH
    if _FAST_BATCH is None:
        raw = os.environ.get("DL_FAST_BATCH", "0")
        try:
            _FAST_BATCH = int(raw or 0) > 0
        except ValueError:
            # Same convention as the other DL_* integer knobs (a non-integer is
            # off), but say so: DL_FAST_BATCH=true would otherwise silently run
            # the old path.
            logger.warning(
                "DL_FAST_BATCH=%r is not an integer; fast batch build is OFF "
                "(set DL_FAST_BATCH=1 to enable)",
                raw,
            )
            _FAST_BATCH = False
    return _FAST_BATCH


def new_long_tensor(numel: int) -> torch.Tensor:
    """Uninitialized 1-D int64 CPU tensor. Inside a DataLoader worker it is
    allocated in shared memory (the default_collate idiom), so handing the
    batch to the main process sends a file descriptor instead of copying it
    into a fresh shared-memory segment."""
    if numel > 0 and torch.utils.data.get_worker_info() is not None:
        elem = torch.empty(0, dtype=torch.long)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        return elem.new(storage).resize_(numel)
    return torch.empty(numel, dtype=torch.long)


# $DL_FAST_BATCH_ONESTORE (2 = on, 0 = off; gin item (56)): the one-storage
# layout of the batches build_batch makes (the fast builder: $DL_FAST_BATCH,
# $DL_FAST_BATCH_EVAL). Every output tensor of such a batch is a view of ONE
# storage (shared memory inside a DataLoader worker) instead of one storage
# per tensor: the merged values and lengths are two 4 KiB-aligned slices of
# it, and the UIH and candidate values and lengths are views of the two
# halves of those slices (merged = [uih | candidates] already), so the batch
# holds every value once (about half the bytes) and the worker skips the two
# concatenations. torch keeps one Python object per storage, so the pickle
# memo sends the shared storage once: a received batch costs one
# file-descriptor hand-off and one mmap instead of six. Samples.pin_memory
# pins it with one pinned allocation and one copy (_pin_one_storage), and
# $DL_PIN_FREE_SHM releases it with one madvise. Values, dtypes, shapes and
# strides are unchanged; storage_offset differs (the tensors are views).
# Contract, host side: the UIH/candidate KJTs share memory with the merged
# KJT until the H2D copy, so nothing may write a batch tensor on the host
# before it; no in-tree code does (train/tests/onestore_item49_test.py scans
# for such writes). Contract, device side: the H2D copy stays per tensor
# (KeyedJaggedTensor.to copies each tensor on its own), so the device tensors
# do not alias. A single-buffer H2D of the storage would carry the aliasing
# to the device, where an in-place op or torchrec's KJT.clear_storage() on
# one KJT would change the others. _pin_one_storage copies the storage whole
# only if the batch's tensors cover it up to the alignment padding; any other
# batch (views into a larger storage; the per-sample builder + collate_fn) is
# pinned per tensor, as before.
# yambda_5b.gin item (56) resolves the switch (gin default 2, env override,
# kill switch DL_FAST_BATCH_ONESTORE=0) through configure_fast_batch_onestore,
# called by train/utils.py fast_batch_onestore in the trainer process before
# any DataLoader pool forks: the forked workers (build_batch) inherit the
# installed value and the pin threads (threads of that process) read it.
# Without that call (tests, tools that do not parse the gin)
# fast_batch_onestore_level() reads the env: unset, empty, 0 or not an
# integer = off. Any value > 0 selects the layout and resolves to 2 (the
# arming lines then say "env: <raw value>" when that differs from 2).
# Read once per process.
_ONESTORE: Optional[int] = None
# True once configure_fast_batch_onestore installed a gin value (arming
# lines: "gin default" vs "code default").
_ONESTORE_FROM_GIN: bool = False
# Slice alignment inside a batch storage, in int64 elements (4 KiB).
_ONESTORE_ALIGN: int = 512
# _pin_one_storage size guard: storage bytes the batch's tensors may leave
# uncovered (the alignment padding of build_batch's two slices).
_ONESTORE_SLACK: int = 2 * _ONESTORE_ALIGN * 8
# pid that logged its first one-storage pin (one line per process; decided
# under the lock: a process can run several pin threads).
_PIN_ONCE_LOGGED_PID: Optional[int] = None
_PIN_ONCE_LOCK = threading.Lock()


def configure_fast_batch_onestore(level: Optional[int] = None) -> None:
    """Install the $DL_FAST_BATCH_ONESTORE value resolved by gin (yambda_5b.gin
    item (56), via train/utils.py fast_batch_onestore: env_int, so a set env
    var wins over the gin default) in this process: > 0 = the one-storage
    layout (level 2), else off. Called in the trainer process before any
    DataLoader pool forks; the workers inherit it and the pin threads read
    it there. None leaves the env-only code default (off). Idempotent."""
    global _ONESTORE, _ONESTORE_FROM_GIN
    if level is not None:
        _ONESTORE = 2 if int(level) > 0 else 0
        _ONESTORE_FROM_GIN = True


def fast_batch_onestore_knob_source(show_raw: bool = False) -> str:
    """Where the $DL_FAST_BATCH_ONESTORE value came from: "env", "gin default"
    or "code default". ``show_raw``: "env: <raw value>" when the variable's
    raw value is not the resolved level (e.g. 1 or 7, both resolved to 2)."""
    raw = os.environ.get("DL_FAST_BATCH_ONESTORE", "").strip()
    if raw:
        if show_raw and raw != str(fast_batch_onestore_level()):
            return "env: %s" % raw
        return "env"
    return "gin default" if _ONESTORE_FROM_GIN else "code default"


def fast_batch_onestore_level() -> int:
    """2 = the one-storage layout (UIH/candidate tensors aliased into the
    merged tensors), 0 = off: the configure_fast_batch_onestore value once gin
    set one, else the env (> 0 = 2; unset, empty, 0 or not an integer = 0)."""
    global _ONESTORE
    if _ONESTORE is None:
        raw = os.environ.get("DL_FAST_BATCH_ONESTORE", "0")
        try:
            _ONESTORE = 2 if int(raw or 0) > 0 else 0
        except ValueError:
            logger.warning(
                "DL_FAST_BATCH_ONESTORE=%r is not an integer; one-storage "
                "batches are OFF (set DL_FAST_BATCH_ONESTORE=2 to enable)",
                raw,
            )
            _ONESTORE = 0
    return int(_ONESTORE)


def fast_batch_onestore_enabled() -> bool:
    return fast_batch_onestore_level() > 0


def fast_batch_onestore_knob() -> str:
    """The resolved switch and its source, for the arming lines:
    ``DL_FAST_BATCH_ONESTORE=<level> (<env | env: raw | gin default | code
    default>)``."""
    return "DL_FAST_BATCH_ONESTORE=%d (%s)" % (
        fast_batch_onestore_level(),
        fast_batch_onestore_knob_source(show_raw=True),
    )


def new_long_tensors(numels: List[int]) -> List[torch.Tensor]:
    """Uninitialized 1-D int64 CPU tensors that are views of ONE storage, each
    starting on a 4 KiB boundary. Inside a DataLoader worker the storage is in
    shared memory, as in new_long_tensor. Any live view keeps the whole
    storage alive."""
    offsets = []
    total = 0
    for n in numels:
        offsets.append(total)
        total += -(-int(n) // _ONESTORE_ALIGN) * _ONESTORE_ALIGN
    nbytes = max(total, 1) * 8
    if torch.utils.data.get_worker_info() is not None:
        storage = torch.UntypedStorage._new_shared(nbytes)
    else:
        storage = torch.UntypedStorage(nbytes)
    return [
        torch.empty(0, dtype=torch.long).set_(storage, off, (int(n),), (1,))
        for off, n in zip(offsets, numels)
    ]


def _pin_one_storage(samples: "Samples") -> bool:
    """$DL_FAST_BATCH_ONESTORE: pin a batch whose KJT tensors all live in ONE
    unpinned CPU storage with one pinned allocation and one copy of that
    storage, then rebuild each KJT on views of the pinned copy (same storage
    offsets, sizes and strides). KeyedJaggedTensor.pin_memory still builds
    the pinned KJTs: Tensor.pin_memory() returns an already-pinned tensor as
    is, so every other attribute is carried exactly as on the per-tensor
    path. Size guard: only when the tensors cover the storage up to the
    alignment padding (_ONESTORE_SLACK); views into a larger storage would
    pin all of it. Returns False, and changes nothing, for any other batch."""
    kjts = [
        samples.uih_features_kjt,
        samples.candidates_features_kjt,
        samples.merged_sparse_features,
    ]
    if any(
        type(k) is not KeyedJaggedTensor or k._inverse_indices is not None
        for k in kjts
    ):
        return False
    tensors = [
        t
        for k in kjts
        for t in (k._values, k._weights, k._lengths, k._offsets)
        if t is not None
    ]
    if not tensors or any(t.device.type != "cpu" for t in tensors):
        return False
    base = tensors[0].untyped_storage()
    ptr = base.data_ptr()
    if ptr == 0 or any(t.untyped_storage().data_ptr() != ptr for t in tensors[1:]):
        return False
    if tensors[0].is_pinned():
        return False
    spans = [
        (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in tensors
    ]
    if base.nbytes() > _covered_bytes(spans) + _ONESTORE_SLACK:
        return False
    pinned = torch.empty(base.nbytes(), dtype=torch.uint8, pin_memory=True)
    pinned.copy_(torch.empty(0, dtype=torch.uint8).set_(base))
    pst = pinned.untyped_storage()

    def view(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        return torch.empty(0, dtype=t.dtype).set_(
            pst, t.storage_offset(), t.size(), t.stride()
        )

    out = []
    for k in kjts:
        k2 = copy.copy(k)
        k2._values = view(k._values)
        k2._weights = view(k._weights)
        k2._lengths = view(k._lengths)
        k2._offsets = view(k._offsets)
        out.append(k2.pin_memory())
    (
        samples.uih_features_kjt,
        samples.candidates_features_kjt,
        samples.merged_sparse_features,
    ) = out
    global _PIN_ONCE_LOGGED_PID
    pid = os.getpid()
    with _PIN_ONCE_LOCK:
        first = _PIN_ONCE_LOGGED_PID != pid
        if first:
            _PIN_ONCE_LOGGED_PID = pid
    if first:
        logger.warning(
            "armed OK: [onestore] %s: batch pinned as one storage (pid %d, "
            "%d tensors, %d B)",
            fast_batch_onestore_knob(),
            pid,
            len(tensors),
            base.nbytes(),
        )
    return True


# $DL_PIN_FREE_SHM (1 = on, 0 = off; gin item (55)): once Samples.pin_memory has
# copied a batch into pinned memory, release the shared-memory pages of the
# batch it was given with one madvise(MADV_REMOVE) per distinct shared
# storage, called through ctypes, which drops the GIL for the call. A batch
# from a DataLoader worker arrives as one shared-memory storage per KJT
# tensor (one unlinked shm file each), or as ONE with $DL_FAST_BATCH_ONESTORE
# (gin item (56); released once, after its one pinned copy). Without the knob
# the pin thread frees them when it drops the batch: each storage destructor
# unmaps its file and, as the file's last user, frees its pages -- with the
# GIL held, next to the trainer's host thread. With the knob the destructor
# finds no pages left. Values are unchanged: the pinned copy is complete
# before anything is freed.
#
# Contract: the caller does not read the unpinned input after pin_memory.
# True for the DataLoader pin thread, the only caller in this tree
# (train/tests/pin_free_shm_item48_test.py checks the callers); a reader of a
# released storage would see zeros. A storage is released only if all of
# these hold, otherwise it is left to its destructor:
#   * sharing strategy file_descriptor, CPU, is_shared(), page-aligned data
#     pointer (the mapping starts there; file_system storages carry a header);
#   * not a file mapping (torch.from_file(shared=True): .filename is set);
#   * the pinned replacement does not alias it (Tensor.pin_memory() returns an
#     already-pinned tensor as is);
#   * size guard: storage nbytes <= 2 x (bytes of this batch's tensors in it)
#     + 64 KiB, so views covering less than about half of a larger shared
#     storage (e.g. a shared dataset array whose pages later batches need)
#     never release it. The bytes are the UNION of the tensors' byte ranges,
#     so a one-storage batch counts every byte once: the merged values/lengths
#     that the UIH/candidate halves alias cover the storage up to the
#     alignment padding, and it is released.
# Not detectable, so collate must build fresh tensors per batch (true here:
# kjt_batch_func / torch.cat, and build_batch allocates a fresh storage per
# batch):
#   * a storage the batch covers whole but that another process keeps (a
#     worker returning one cached shared tensor in every batch);
#   * views covering about half or more of a larger shared storage: they pass
#     the size guard, and the whole storage would be released.
# yambda_5b.gin item (55) resolves the switch (gin default 1, env override,
# kill switch DL_PIN_FREE_SHM=0) through configure_pin_free_shm, called by
# train/utils.py pin_free_shm_release in the trainer process before any
# DataLoader pin thread starts (the pin threads are threads of that process;
# forked workers never pin). Without that call (tests, tools that do not parse
# the gin) pin_free_shm_enabled() reads the env: unset, empty, 0 or not an
# integer = off.
# Read once per process. "armed OK: [pin-free-shm]" once per process (first
# released batch); "armed OK: [onestore-free-shm]" once per process (first
# released batch that was pinned as one storage, item (56)); "[pin-free-shm]
# WARN" once per process if the size guard kept a storage, once for any other
# kept storage or failure, with counts.
_FREE_SHM: Optional[bool] = None
# True once configure_pin_free_shm installed a gin value (arming line: "gin
# default" vs "code default").
_FREE_SHM_FROM_GIN: bool = False
_MADV_REMOVE: int = 9  # <linux/mman.h>
_LIBC_MADVISE = None
_PAGE: int = mmap.PAGESIZE
_FREE_SHM_SLACK: int = 64 * 1024
# pids that printed the arming line / the one-storage arming line / the
# size-guard WARN / the other WARN
_FREE_SHM_ARMED_PID: Optional[int] = None
_FREE_SHM_ONESTORE_PID: Optional[int] = None
_FREE_SHM_WARN_SIZE_PID: Optional[int] = None
_FREE_SHM_WARN_OTHER_PID: Optional[int] = None
# Process totals (all pin threads), see pin_free_shm_stats().
_FREE_SHM_LOCK = threading.Lock()
_FREE_SHM_STATS: Dict[str, int] = dict.fromkeys(
    (
        "batches",  # pin_memory calls with the knob on
        "freed",  # storages released
        "freed_bytes",
        "skip_size",  # kept by the size guard
        "skip_size_bytes",
        "skip_alias",  # kept: the pinned copy aliases it
        "skip_file",  # kept: file mapping
        "madvise_err",  # madvise returned an error
        "errors",  # exceptions in the release path (batch still pinned)
        "onestore_batches",  # of "batches": pinned as one storage
        "onestore_freed",  # of "freed": storages of those batches
    ),
    0,
)


def configure_pin_free_shm(enabled: Optional[int] = None) -> None:
    """Install the $DL_PIN_FREE_SHM value resolved by gin (yambda_5b.gin item
    (55), via train/utils.py pin_free_shm_release: env_int, so a set env var
    wins over the gin default) in this process. Called in the trainer process
    before any DataLoader pin thread starts; the pin threads read it there.
    None leaves the env-only code default (off). Idempotent."""
    global _FREE_SHM, _FREE_SHM_FROM_GIN
    if enabled is not None:
        _FREE_SHM = int(enabled) > 0
        _FREE_SHM_FROM_GIN = True


def pin_free_shm_knob_source() -> str:
    """Where the $DL_PIN_FREE_SHM value came from: "env", "gin default" or
    "code default"."""
    if os.environ.get("DL_PIN_FREE_SHM", "").strip():
        return "env"
    return "gin default" if _FREE_SHM_FROM_GIN else "code default"


def pin_free_shm_enabled() -> bool:
    """$DL_PIN_FREE_SHM > 0: the configure_pin_free_shm value once gin set one,
    else the env. Unset, empty, 0 or not an integer = off."""
    global _FREE_SHM
    if _FREE_SHM is None:
        raw = os.environ.get("DL_PIN_FREE_SHM", "0")
        try:
            _FREE_SHM = int(raw or 0) > 0
        except ValueError:
            logger.warning(
                "DL_PIN_FREE_SHM=%r is not an integer; GIL-free shared-memory "
                "release is OFF (set DL_PIN_FREE_SHM=1 to enable)",
                raw,
            )
            _FREE_SHM = False
    return bool(_FREE_SHM)


def pin_free_shm_stats() -> Dict[str, int]:
    """Totals of $DL_PIN_FREE_SHM in this process (copy)."""
    with _FREE_SHM_LOCK:
        return dict(_FREE_SHM_STATS)


def _samples_tensors(samples: "Samples") -> List[torch.Tensor]:
    """Every tensor KeyedJaggedTensor.pin_memory() copies, for the 3 KJTs."""
    out = []
    for f in ("uih_features_kjt", "candidates_features_kjt", "merged_sparse_features"):
        k = getattr(samples, f, None)
        for a in ("_values", "_weights", "_lengths", "_offsets"):
            t = getattr(k, a, None)
            if isinstance(t, torch.Tensor):
                out.append(t)
        inv = getattr(k, "_inverse_indices", None)
        if isinstance(inv, tuple) and len(inv) == 2:
            if isinstance(inv[1], torch.Tensor):
                out.append(inv[1])
    return out


def _madvise_remove(ptr: int, size: int) -> int:
    """madvise(ptr, size, MADV_REMOVE) without the GIL. 0 or an errno."""
    global _LIBC_MADVISE
    import ctypes

    fn = _LIBC_MADVISE
    if fn is None:
        fn = ctypes.CDLL(None, use_errno=True).madvise
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        fn.restype = ctypes.c_int
        _LIBC_MADVISE = fn
    if fn(ptr, size, _MADV_REMOVE) == 0:
        return 0
    return ctypes.get_errno() or -1


def _covered_bytes(spans: List[Tuple[int, int]]) -> int:
    """Size of the union of the [start, end) byte ranges."""
    total, cur_s, cur_e = 0, 0, -1
    for s, e in sorted(spans):
        if s > cur_e:
            total += max(0, cur_e - cur_s)
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    return total + max(0, cur_e - cur_s)


def _free_shared_pages(
    src: List[torch.Tensor], pinned: List[torch.Tensor]
) -> Dict[str, int]:
    """Release the pages of every distinct shared CPU storage behind ``src``
    (the unpinned input) that passes the $DL_PIN_FREE_SHM guards; ``pinned``
    are the tensors that replace it. Returns this call's counts."""
    r = dict.fromkeys(
        (
            "freed",
            "freed_bytes",
            "skip_size",
            "skip_size_bytes",
            "skip_size_own",
            "skip_alias",
            "skip_file",
            "madvise_err",
            "errno",
        ),
        0,
    )
    if torch.multiprocessing.get_sharing_strategy() != "file_descriptor":
        return r
    # storage pointer -> (storage, byte ranges of this batch's tensors in it)
    groups: Dict[int, Tuple[torch.UntypedStorage, List[Tuple[int, int]]]] = {}
    for t in src:
        if t.device.type != "cpu":
            continue
        st = t.untyped_storage()
        p = st.data_ptr()
        if p == 0:
            continue
        g = groups.get(p)
        if g is None:
            g = groups[p] = (st, [])
        n = t.numel() * t.element_size()
        if n:
            g[1].append((t.data_ptr(), t.data_ptr() + n))
    if not groups:
        return r
    taken = []  # (storage start, storage end, data pointer) of the pinned copy
    for t in pinned:
        pst = t.untyped_storage()
        q = pst.data_ptr()
        taken.append((q, q + max(pst.nbytes(), 1), t.data_ptr()))
    for p, (st, spans) in groups.items():
        if p % _PAGE or not st.is_shared():
            continue
        if getattr(st, "filename", None) is not None:
            r["skip_file"] += 1
            continue
        nb = st.nbytes()
        end = p + nb
        if any((q < end and p < qe) or p <= d < end for q, qe, d in taken):
            r["skip_alias"] += 1
            continue
        own = _covered_bytes(spans)
        if nb > 2 * own + _FREE_SHM_SLACK:
            r["skip_size"] += 1
            r["skip_size_bytes"] += nb
            r["skip_size_own"] += own
            continue
        err = _madvise_remove(p, -(-nb // _PAGE) * _PAGE)
        if err == 0:
            r["freed"] += 1
            r["freed_bytes"] += nb
        else:
            r["madvise_err"] += 1
            r["errno"] = err
    return r


def _release_source_pages(
    src: List[torch.Tensor], pinned: List[torch.Tensor], one_store: bool = False
) -> None:
    """$DL_PIN_FREE_SHM after a completed pin: release, count, log once.
    ``one_store``: the batch was pinned as one storage ($DL_FAST_BATCH_ONESTORE,
    _pin_one_storage); counted and logged on its own as well. The once-per-
    process line decisions are taken under _FREE_SHM_LOCK with the counts (a
    process runs several pin threads: train pools and the eval pool).
    Never raises (the pinned batch is complete either way)."""
    global _FREE_SHM_ARMED_PID, _FREE_SHM_WARN_SIZE_PID, _FREE_SHM_WARN_OTHER_PID
    global _FREE_SHM_ONESTORE_PID
    pid = os.getpid()
    try:
        r = _free_shared_pages(src, pinned)
    except Exception as e:  # noqa: B902
        with _FREE_SHM_LOCK:
            _FREE_SHM_STATS["batches"] += 1
            _FREE_SHM_STATS["errors"] += 1
            if one_store:
                _FREE_SHM_STATS["onestore_batches"] += 1
            warn = _FREE_SHM_WARN_OTHER_PID != pid
            if warn:
                _FREE_SHM_WARN_OTHER_PID = pid
        if warn:
            logger.warning(
                "[pin-free-shm] WARN: release failed (%r); the batch is pinned "
                "and its shared memory is left to the destructor (pid %d, "
                "one line per process)",
                e,
                pid,
            )
        return
    with _FREE_SHM_LOCK:
        _FREE_SHM_STATS["batches"] += 1
        for k in (
            "freed",
            "freed_bytes",
            "skip_size",
            "skip_size_bytes",
            "skip_alias",
            "skip_file",
            "madvise_err",
        ):
            _FREE_SHM_STATS[k] += r[k]
        if one_store:
            _FREE_SHM_STATS["onestore_batches"] += 1
            _FREE_SHM_STATS["onestore_freed"] += r["freed"]
        armed = bool(r["freed"]) and _FREE_SHM_ARMED_PID != pid
        if armed:
            _FREE_SHM_ARMED_PID = pid
        armed_one = one_store and bool(r["freed"]) and _FREE_SHM_ONESTORE_PID != pid
        if armed_one:
            _FREE_SHM_ONESTORE_PID = pid
        warn_size = bool(r["skip_size"]) and _FREE_SHM_WARN_SIZE_PID != pid
        if warn_size:
            _FREE_SHM_WARN_SIZE_PID = pid
        warn_other = bool(
            r["skip_alias"] or r["skip_file"] or r["madvise_err"]
        ) and _FREE_SHM_WARN_OTHER_PID != pid
        if warn_other:
            _FREE_SHM_WARN_OTHER_PID = pid
    if armed:
        logger.warning(
            "armed OK: [pin-free-shm] DL_PIN_FREE_SHM=%d (%s): freed %d shared "
            "storage(s), %d B, GIL-free after pinning (pid %d)",
            int(pin_free_shm_enabled()),
            pin_free_shm_knob_source(),
            r["freed"],
            r["freed_bytes"],
            pid,
        )
    if armed_one:
        logger.warning(
            "armed OK: [onestore-free-shm] %s DL_PIN_FREE_SHM=%d (%s): batch "
            "pinned as one storage, then freed %d shared storage(s), %d B, kept "
            "%d (pid %d)",
            fast_batch_onestore_knob(),
            int(pin_free_shm_enabled()),
            pin_free_shm_knob_source(),
            r["freed"],
            r["freed_bytes"],
            r["skip_size"] + r["skip_alias"] + r["skip_file"] + r["madvise_err"],
            pid,
        )
    if warn_size:
        logger.warning(
            "[pin-free-shm] WARN: size guard kept %d shared storage(s) of %d B "
            "holding only %d B of this batch's tensors (> 2x + %d B: a view "
            "into a larger shared storage is never released); released %d "
            "storage(s) of this batch (pid %d, one line per process; totals "
            "in pin_free_shm_stats())",
            r["skip_size"],
            r["skip_size_bytes"],
            r["skip_size_own"],
            _FREE_SHM_SLACK,
            r["freed"],
            pid,
        )
    if warn_other:
        logger.warning(
            "[pin-free-shm] WARN: kept %d storage(s) the pinned copy aliases, "
            "%d file mapping(s); madvise failed for %d (errno %d); released "
            "%d storage(s) of this batch (pid %d, one line per process)",
            r["skip_alias"],
            r["skip_file"],
            r["madvise_err"],
            r["errno"],
            r["freed"],
            pid,
        )


@dataclass
class Samples(Pipelineable):
    """
    Container for batched samples with user interaction history and candidate features.

    Attributes:
        uih_features_kjt: User interaction history features as KeyedJaggedTensor.
        candidates_features_kjt: Candidate item features as KeyedJaggedTensor.
    """

    uih_features_kjt: KeyedJaggedTensor
    candidates_features_kjt: KeyedJaggedTensor
    # UIH + candidate features concatenated into the single KJT that the model's
    # sharded EmbeddingCollection consumes. Pre-built here (dataloader/CPU) rather
    # than inside DlrmHSTU.forward so the embedding lookup's input is a plain
    # attribute of the batch — which lets TorchRec's TrainPipelineSparseDist hoist
    # its input_dist into the prefetch stage (otherwise the runtime cat +
    # from_lengths_sync counts as an "input modification" and the embedding
    # collection is left un-pipelined).
    merged_sparse_features: KeyedJaggedTensor

    def to(self, device: torch.device, non_blocking: bool = False) -> "Samples":
        """
        Move all tensors to the specified device (in place) and return self.

        Returning ``self`` (rather than ``None``) and accepting ``non_blocking``
        makes ``Samples`` conform to TorchRec's ``Pipelineable`` protocol so it
        can be driven by ``TrainPipelineSparseDist``. Existing call sites that
        use ``sample.to(device)`` for its side effect continue to work unchanged.

        With the host-sync-free step on (``DLRM_HSTU_FP8_SYNCFREE``), a CPU ->
        device move also tags the device lengths with the length statistics
        computed from the CPU copy, so the forward and the GAUC metric need no
        device-to-host read (dlrm_v4/host_sync_free.py).
        """
        from generative_recommenders.dlrm_v4 import host_sync_free as _hsf

        stats = None
        if _hsf.enabled() and torch.device(device).type != "cpu":
            stats = _hsf.host_stats_from_cpu(
                self.uih_features_kjt, self.candidates_features_kjt
            )
            if stats is not None:
                _hsf.seed_host_length_per_key(
                    self.uih_features_kjt, self.candidates_features_kjt
                )
        for attr in vars(self):
            setattr(
                self,
                attr,
                getattr(self, attr).to(device=device, non_blocking=non_blocking),
            )
        _hsf.tag_device_lengths(
            self.uih_features_kjt, self.candidates_features_kjt, stats
        )
        if stats is not None:
            _hsf.verify_host_length_per_key(
                self.uih_features_kjt, self.candidates_features_kjt
            )
        return self

    def record_stream(self, stream: torch.Stream) -> None:
        """Record the contained KJTs on ``stream`` (Pipelineable protocol).

        Required by ``TrainPipelineSparseDist`` so the prefetched batch's H2D
        copy on the side stream is not freed before compute consumes it.
        """
        self.uih_features_kjt.record_stream(stream)
        self.candidates_features_kjt.record_stream(stream)
        self.merged_sparse_features.record_stream(stream)

    def pin_memory(self) -> "Samples":
        """Pin the contained KJTs' host memory (Pipelineable protocol).

        With $DL_FAST_BATCH_ONESTORE (the yambda_5b.gin item (56) default) a
        batch whose KJT tensors all live in one storage (build_batch's
        one-storage layout) is pinned with one allocation and one copy
        (_pin_one_storage); any other batch takes the per-tensor path. With
        $DL_PIN_FREE_SHM=1 (the yambda_5b.gin item (55) default) the input's
        shared-memory pages are released once the pinned copy is made (one
        storage per one-storage batch): do not read the unpinned input after
        this call (see _free_shared_pages)."""
        src = _samples_tensors(self) if pin_free_shm_enabled() else None
        one = fast_batch_onestore_enabled() and _pin_one_storage(self)
        if not one:
            self.uih_features_kjt = self.uih_features_kjt.pin_memory()
            self.candidates_features_kjt = self.candidates_features_kjt.pin_memory()
            self.merged_sparse_features = self.merged_sparse_features.pin_memory()
        if src:
            _release_source_pages(src, _samples_tensors(self), one_store=one)
            del src
        return self

    def batch_size(self) -> int:
        """
        Get the batch size of the samples.

        Returns:
            Number of samples in the batch.
        """
        return self.uih_features_kjt.stride()


def merge_uih_candidate_kjts(
    uih_features: KeyedJaggedTensor,
    candidates_features: KeyedJaggedTensor,
) -> KeyedJaggedTensor:
    """Concatenate the UIH and candidate KJTs into the single KJT consumed by the
    model's ``EmbeddingCollection``.

    Must mirror ``DlrmHSTU.preprocess`` exactly (key order = uih + candidates,
    values/lengths concatenated in that order). Built on the dataloader side so
    the model can read it straight off the batch and TorchRec can pipeline the
    embedding ``input_dist``.
    """
    return KeyedJaggedTensor.from_lengths_sync(
        keys=uih_features.keys() + candidates_features.keys(),
        values=torch.cat(
            [uih_features.values(), candidates_features.values()],
            dim=0,
        ),
        lengths=torch.cat(
            [uih_features.lengths(), candidates_features.lengths()],
            dim=0,
        ),
    )


def collate_fn(
    samples: Union[List[Tuple[KeyedJaggedTensor, KeyedJaggedTensor]], Samples],
) -> Samples:
    """
    Collate multiple samples into a batched Samples object.

    Args:
        samples: List of (uih_features, candidates_features) tuples, or an
            already-batched Samples ($DL_FAST_BATCH: HammerToTorchDataset.
            __getitems__ built the batch directly), which is returned as is.

    Returns:
        Batched Samples object with concatenated features.
    """
    if isinstance(samples, Samples):
        return samples
    (
        uih_features_kjt_list,
        candidates_features_kjt_list,
    ) = list(zip(*samples))

    uih_features_kjt = kjt_batch_func(uih_features_kjt_list)
    candidates_features_kjt = kjt_batch_func(candidates_features_kjt_list)
    return Samples(
        uih_features_kjt=uih_features_kjt,
        candidates_features_kjt=candidates_features_kjt,
        merged_sparse_features=merge_uih_candidate_kjts(
            uih_features_kjt, candidates_features_kjt
        ),
    )


class Dataset:
    """
    Base dataset class for DLRMv4.

    Provides the interface for loading, accessing, and managing samples
    for recommendation model training and inference.

    Args:
        hstu_config: HSTU model configuration.
        **args: Additional arguments (unused in base class).
    """

    def __init__(self, hstu_config: DlrmHSTUConfig, **args):
        self.arrival = None
        self.image_list = []
        self.label_list = []
        self.image_list_inmemory = {}
        self.last_loaded = -1.0

    def preprocess(self, use_cache=True):
        """
        Preprocess the dataset.

        Args:
            use_cache: Whether to use cached preprocessed data.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Dataset:preprocess")

    def get_item_count(self):
        """
        Get the total number of items in the dataset.

        Returns:
            Number of items.
        """
        return len(self.image_list)

    def load_query_samples(self, sample_list):
        """
        Load specified samples into memory.

        Args:
            sample_list: List of sample indices to load.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Dataset:load_query_samples")

    def unload_query_samples(self, sample_list):
        """
        Unload specified samples from memory.

        Args:
            sample_list: List of sample indices to unload.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Dataset:unload_query_samples")

    def get_sample(self, id: int):
        """
        Get a single sample by ID.

        Args:
            id: Sample identifier.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Dataset:get_sample")

    def get_samples(self, id_list: List[int]) -> Samples:
        """
        Get multiple samples and collate them into a batch.

        Args:
            id_list: List of sample identifiers.

        Returns:
            Collated Samples object containing the batch.
        """
        list_samples = [self.get_sample(ix) for ix in id_list]
        return collate_fn(list_samples)


@torch.jit.script
def kjt_batch_func(
    kjt_list: List[KeyedJaggedTensor],
) -> KeyedJaggedTensor:
    """
    Batch multiple KeyedJaggedTensors into a single tensor.

    Uses FBGEMM operations for efficient batching and reordering of
    jagged tensor data.

    Args:
        kjt_list: List of KeyedJaggedTensors to batch.

    Returns:
        Batched KeyedJaggedTensor with reordered indices and lengths.
    """
    bs_list = [kjt.stride() for kjt in kjt_list]
    bs = sum(bs_list)
    batched_length = torch.cat([kjt.lengths() for kjt in kjt_list], dim=0)
    batched_indices = torch.cat([kjt.values() for kjt in kjt_list], dim=0)
    bs_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(
        torch.tensor(bs_list)
    ).int()
    batched_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(batched_length)
    reorder_length = torch.ops.fbgemm.reorder_batched_ad_lengths(
        batched_length, bs_offset, bs
    )
    reorder_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(reorder_length)
    reorder_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
        batched_offset, batched_indices, reorder_offsets, bs_offset, bs
    )
    out = KeyedJaggedTensor(
        keys=kjt_list[0].keys(),
        lengths=reorder_length.long(),
        values=reorder_indices.long(),
    )
    return out


def get_random_data(
    contexual_features: List[str],
    hstu_uih_keys: List[str],
    hstu_candidates_keys: List[str],
    uih_max_seq_len: int,
    max_num_candidates: int,
    value_bound: int = 1000,
):
    """
    Generate random sample data for testing and debugging.

    Creates synthetic user interaction history and candidate features
    with random values.

    Args:
        contexual_features: List of contextual feature names.
        hstu_uih_keys: List of UIH feature keys.
        hstu_candidates_keys: List of candidate feature keys.
        uih_max_seq_len: Maximum sequence length for UIH.
        max_num_candidates: Maximum number of candidates.
        value_bound: Upper bound for random values.

    Returns:
        Tuple of (uih_features_kjt, candidates_features_kjt).
    """
    uih_non_seq_feature_keys = contexual_features
    uih_seq_feature_keys = [
        k for k in hstu_uih_keys if k not in uih_non_seq_feature_keys
    ]
    uih_seq_len = torch.randint(
        int(uih_max_seq_len * 0.8),
        uih_max_seq_len + 1,
        (1,),
    ).item()
    uih_lengths = torch.tensor(
        [1 for _ in uih_non_seq_feature_keys]
        + [uih_seq_len for _ in uih_seq_feature_keys]
    )
    # logging.info(f"uih_lengths: {uih_lengths}")
    uih_values = torch.randint(
        1,
        value_bound,
        # pyre-ignore[6]
        (uih_seq_len * len(uih_seq_feature_keys) + len(uih_non_seq_feature_keys),),
    )
    uih_features_kjt = KeyedJaggedTensor(
        keys=uih_non_seq_feature_keys + uih_seq_feature_keys,
        lengths=uih_lengths.long(),
        values=uih_values.long(),
    )
    num_candidates = torch.randint(
        1,
        max_num_candidates + 1,
        (1,),
    ).item()
    candidates_lengths = num_candidates * torch.ones(len(hstu_candidates_keys))
    candidates_values = torch.randint(
        1,
        value_bound,
        (num_candidates * len(hstu_candidates_keys),),  # pyre-ignore[6]
    )
    candidates_features_kjt = KeyedJaggedTensor(
        keys=hstu_candidates_keys,
        lengths=candidates_lengths.long(),
        values=candidates_values.long(),
    )
    return uih_features_kjt, candidates_features_kjt


class DLRMv4RandomDataset(Dataset):
    """
    Dataset that generates random synthetic data for DLRMv4.

    Useful for testing and benchmarking without real data dependencies.

    Args:
        hstu_config: HSTU model configuration.
        num_aggregated_samples: Total number of samples to generate.
        is_inference: Whether the dataset is used for inference mode.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        num_aggregated_samples: int = 10000,
        is_inference: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            hstu_config=hstu_config,
        )
        self.hstu_config: DlrmHSTUConfig = hstu_config
        self._max_num_candidates: int = hstu_config.max_num_candidates
        self._max_num_candidates_inference: int = (
            hstu_config.max_num_candidates_inference
        )
        self._max_seq_len: int = hstu_config.max_seq_len
        self._uih_keys: List[str] = hstu_config.hstu_uih_feature_names
        self._candidates_keys: List[str] = hstu_config.hstu_candidate_feature_names
        self._contextual_feature_to_max_length: Dict[str, int] = (
            hstu_config.contextual_feature_to_max_length
        )
        self._max_uih_len: int = (
            self._max_seq_len
            - self._max_num_candidates
            - (
                len(self._contextual_feature_to_max_length)
                if self._contextual_feature_to_max_length
                else 0
            )
        )
        self._is_inference = is_inference

        self.contexual_features = []
        if hstu_config.contextual_feature_to_max_length is not None:
            self.contexual_features = [
                p[0] for p in hstu_config.contextual_feature_to_max_length
            ]

        self.num_aggregated_samples = num_aggregated_samples
        self.items_in_memory = {}

    def get_sample(self, id: int) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        """
        Get a sample by ID from in-memory storage.

        Args:
            id: Sample identifier.

        Returns:
            Tuple of (uih_features_kjt, candidates_features_kjt).
        """
        return self.items_in_memory[id]

    def get_item_count(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            Number of aggregated samples.
        """
        return self.num_aggregated_samples

    def unload_query_samples(self, sample_list):
        """
        Clear all samples from memory.

        Args:
            sample_list: Ignored; clears all samples.
        """
        self.items_in_memory = {}

    def load_query_samples(self, sample_list):
        """
        Generate and load random samples into memory.

        Args:
            sample_list: List of sample IDs to generate.
        """
        max_num_candidates = (
            self._max_num_candidates_inference
            if self._is_inference
            else self._max_num_candidates
        )
        self.items_in_memory = {}
        for sample in sample_list:
            self.items_in_memory[sample] = get_random_data(
                contexual_features=self.contexual_features,
                hstu_uih_keys=self.hstu_config.hstu_uih_feature_names,
                hstu_candidates_keys=self.hstu_config.hstu_candidate_feature_names,
                uih_max_seq_len=self._max_uih_len,
                max_num_candidates=max_num_candidates,
            )
        self.last_loaded = time.time()
