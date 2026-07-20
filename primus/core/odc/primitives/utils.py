# Adapted from ODC (https://github.com/sail-sg/odc), which is distributed under
# the MIT License per its package metadata (pyproject.toml / setup.py
# classifiers). The upstream repository ships no LICENSE file or per-file
# copyright headers; upstream copyright is held by the ODC authors (Sea AI Lab).
#
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# See LICENSE for license information.

"""
ODC primitives utilities — ROCm symmetric-memory management.

ODC allocates symmetric memory and enumerates same-node peer views. On ROCm
there are two backends, selected by ``ODC_P2P_BACKEND``:

* ``mori`` (default):
    - ``init_shmem`` initializes MORI-SHMEM from the PyTorch process
      group via ``mori.shmem.shmem_torch_process_group_init("default")``.
    - ``shmem_create_tensor`` calls ``mori.shmem.mori_shmem_create_tensor``.
    - Same-node peer views are obtained in a single MORI call:
      ``mori_shmem_create_tensor_list_intra_node`` which returns the
      list of symmetric views across all same-node ranks. Because that
      API allocates *and* returns the peer-view list together, we
      structure ``SymmBufferRegistry.allocate_symm_buffer`` to use it,
      keeping the same external contract (a single tensor handed back
      to the caller, plus a separately stored peer list).
    - ``shmem_free_tensor_sync`` and ``finalize_distributed`` map to
      ``mori.shmem.mori_shmem_free_tensor`` / ``shmem_finalize``.
* ``rocshmem``: a host-API rocSHMEM backend (single-node / IPC-only)
  implemented in ``_rocshmem_backend.py``.

NOTE on environment variables (mori backend):
    Set ``MORI_SHMEM_HEAP_SIZE`` (e.g. ``"4G"``) *before* the first MORI
    call. Also set ``MORI_SOCKET_IFNAME`` (defaults to the value of
    ``NCCL_SOCKET_IFNAME`` if available) for the MORI bootstrap.
"""

import logging
import os
from functools import reduce
from typing import List

import torch
import triton
import triton.language as tl

from odc.primitives import __syncthreads
from odc.runtime_config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ODC P2P backend selector (config item odc_p2p_backend). Default "mori"
# preserves the existing, verified behaviour byte-for-byte; "rocshmem" selects
# the host-API backend (single-node / IPC-only) implemented in
# _rocshmem_backend.py. Read HERE at import time -- the ODC integration patch
# populates the runtime config before odc.primitives is first imported.
# ---------------------------------------------------------------------------
_P2P_BACKEND = get_config().p2p_backend.lower()
_USE_ROCSHMEM = _P2P_BACKEND == "rocshmem"


# ---------------------------------------------------------------------------
# Backend-specific imports
# ---------------------------------------------------------------------------
if _USE_ROCSHMEM:
    from . import _rocshmem_backend as _rs
else:
    import mori.shmem as _mori_shmem


# ---------------------------------------------------------------------------
# init_shmem — initialize the symmetric-memory runtime.
# ---------------------------------------------------------------------------
def init_shmem():
    """Initialize the symmetric-memory runtime on the global process group.

    mori backend:
        Bootstrap MORI-SHMEM from PyTorch's WORLD process group by name.
        Requires that PyTorch distributed is already initialized.
        ``MORI_SHMEM_HEAP_SIZE`` and ``MORI_SOCKET_IFNAME`` should be
        exported in the environment before this call.

    rocshmem backend:
        Bootstrap the rocSHMEM host-API runtime via a unique-id broadcast.
    """
    assert torch.distributed.is_initialized()

    if _USE_ROCSHMEM:
        # rocSHMEM host-API backend: bootstrap via unique-id broadcast.
        _rs.init()
        return

    # MORI requires MORI_SHMEM_HEAP_SIZE to be set. Inherit a sensible
    # default if the user did not export one — this is purely a guard;
    # production paths should always set it explicitly.
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "4G")

    # MORI uses its own TCP bootstrap socket. If the user set
    # NCCL_SOCKET_IFNAME but not MORI_SOCKET_IFNAME, mirror it so they
    # stay aligned.
    if "MORI_SOCKET_IFNAME" not in os.environ and "NCCL_SOCKET_IFNAME" in os.environ:
        os.environ["MORI_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"].lstrip("=")

    # Two init methods (select via the odc_mori_init config item):
    #   "pg"  (default): shmem_torch_process_group_init("default")
    #   "uid": shmem_init_attr(WITH_UNIQUEID, rank, world, uid) — does NOT
    #          touch PyTorch process-group registration. Use this when the
    #          host framework (e.g. Megatron) has already created many
    #          named process groups and the PG-based init crashes
    #          (observed: `free(): invalid pointer` during
    #          shmem_torch_process_group_init under Megatron).
    init_method = get_config().mori_init
    logger.info(
        "init_shmem (MORI): heap=%s, sock_ifname=%s, method=%s",
        os.environ.get("MORI_SHMEM_HEAP_SIZE"),
        os.environ.get("MORI_SOCKET_IFNAME", "<auto>"),
        init_method,
    )

    if init_method == "uid":
        rank_id = torch.distributed.get_rank()
        num_ranks = torch.distributed.get_world_size()
        # rank 0 generates the 128-byte unique id; broadcast to all ranks.
        uid_holder = [_mori_shmem.shmem_get_unique_id() if rank_id == 0 else None]
        torch.distributed.broadcast_object_list(uid_holder, src=0)
        torch.distributed.barrier()
        unique_id = uid_holder[0]
        _mori_shmem.shmem_init_attr(
            _mori_shmem.MORI_SHMEM_INIT_WITH_UNIQUEID,
            rank_id,
            num_ranks,
            unique_id,
        )
        return

    # method == "pg": register WORLD as "default", then PG-based init.
    world_group = torch.distributed.group.WORLD
    try:
        torch._C._distributed_c10d._register_process_group("default", world_group)
    except RuntimeError:
        # Already registered — ignore.
        pass
    _mori_shmem.shmem_torch_process_group_init("default")


# ---------------------------------------------------------------------------
# Symmetric tensor creation / peer-view enumeration / free
# ---------------------------------------------------------------------------
def shmem_create_tensor(shape, dtype) -> torch.Tensor:
    """Allocate a symmetric tensor on the symmetric heap.

    The full allocate + same-node peer enumeration is done in one call inside
    ``SymmBufferRegistry.allocate_symm_buffer`` (see below). This top-level
    helper is kept for callers that want a bare local tensor without the
    peer list (e.g. when only the local view is needed). On the mori backend
    it calls ``mori_shmem_create_tensor``, which allocates from MORI's
    symmetric heap.
    """
    torch.cuda.synchronize()
    if _USE_ROCSHMEM:
        tensor = _rs.create_tensor(shape, dtype)
    else:
        tensor = _mori_shmem.mori_shmem_create_tensor(shape, dtype)
    torch.cuda.synchronize()
    return tensor


def get_same_node_tensors(tensor, rank, local_world_size) -> List[torch.Tensor]:
    """Return the list of peer-view tensors for the given symmetric ``tensor``
    across the same-node ranks (``rank // local_world_size``'s peers).

    This API is only safe when ``tensor`` was created by the paired
    ``mori_shmem_create_tensor_list_intra_node`` (see
    ``SymmBufferRegistry.allocate_symm_buffer``); the list is retrieved
    from the registry. As a stand-alone fall-back, we synthesize it via
    ``shmem_ptr_p2p`` and wrap raw device pointers with
    ``torch.frombuffer``-style construction.
    """
    # When the tensor was created via the registry path, the registry
    # already stores the peer list — callers should prefer
    # ``SymmBufferRegistry.get_peer_tensors``.
    # As a stand-alone fallback for ad-hoc symmetric tensors, build
    # peer views via shmem_ptr_p2p + torch.Tensor reconstruction.
    peer_tensors: List[torch.Tensor] = []
    local_rank = rank % local_world_size
    rank_on_same_node_start = rank - local_rank
    for peer in range(rank_on_same_node_start, rank_on_same_node_start + local_world_size):
        if peer == rank:
            peer_tensors.append(tensor)
            continue
        peer_ptr = _mori_shmem.shmem_ptr_p2p(tensor.data_ptr(), rank, peer)
        if peer_ptr == 0:
            raise RuntimeError(
                f"shmem_ptr_p2p returned 0 for peer {peer}: no XGMI P2P route "
                "from PE {rank}. This usually means peer is on a different node "
                "(requires RDMA, not P2P)."
            )
        # Build a torch.Tensor view at peer_ptr.
        peer_tensors.append(_tensor_from_raw_ptr(peer_ptr, tensor.shape, tensor.dtype, tensor.device))
    return peer_tensors


def _tensor_from_raw_ptr(raw_ptr: int, shape, dtype, device) -> torch.Tensor:
    """Wrap a raw device pointer as a torch.Tensor without taking ownership.

    Used on ROCm to materialize peer-view tensors from
    ``mori.shmem.shmem_ptr_p2p``. The returned tensor shares storage with
    the original symmetric allocation (which is owned by the registry);
    no deleter is attached, so the caller must guarantee that the symm
    allocation outlives all peer views.
    """
    import ctypes

    # Compute total bytes
    if isinstance(shape, int):
        shape = (shape,)
    nbytes = reduce(lambda a, b: a * b, shape) * torch._utils._element_size(dtype)

    # Build a ctypes array view at raw_ptr (zero-copy) — only used to
    # pass through PyTorch's from_blob. PyTorch will create a tensor that
    # references this storage. The CObject is intentionally kept alive
    # via attribute attachment so GC does not free the wrapper.
    array_t = (ctypes.c_uint8 * nbytes).from_address(raw_ptr)
    t = torch.frombuffer(array_t, dtype=torch.uint8, count=nbytes).view(dtype).view(*shape).to(device)
    # NOTE: ``torch.frombuffer`` materializes a CPU tensor; ``.to(device)`` would
    # COPY. We can't use that path for a real peer-view (would defeat the purpose).
    # Instead use the explicit cuda path:
    raise NotImplementedError(
        "Direct raw-ptr → torch.Tensor wrapping is not implemented in this "
        "iteration. Use SymmBufferRegistry.allocate_symm_buffer (which uses "
        "mori_shmem_create_tensor_list_intra_node) — that path is the one "
        "exercised by ODC's gather/scatter primitives. The bare "
        "get_same_node_tensors fall-back is provided only as a placeholder."
    )


def shmem_free_tensor_sync(tensor):
    torch.cuda.synchronize()
    if _USE_ROCSHMEM:
        _rs.free_tensor(tensor)
    else:
        _mori_shmem.mori_shmem_free_tensor(tensor)
    torch.cuda.synchronize()


def finalize_distributed():
    if _USE_ROCSHMEM:
        _rs.finalize()
    else:
        _mori_shmem.shmem_finalize()


# ---------------------------------------------------------------------------
# Symmetric buffer registry (the heart of ODC memory management)
# ---------------------------------------------------------------------------
class SymmBufferRegistry:
    def __init__(self):
        self.local_tensor = {}
        self.local_tensor_to_keys = {}
        self.updated = set()
        self.peer_tensors = {}
        self.allocations = []

    @classmethod
    def get_instance(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = SymmBufferRegistry()
        return cls._instance

    # we'll mark all symm buffer as dirty, and next update_symm_buffer will copy the data to the symm buffer
    def flush(self):
        self.updated.clear()

    def update_symm_buffer(self, buffer_key, values):
        values = values.contiguous()
        if buffer_key not in self.local_tensor:
            self.allocate_symm_buffer(buffer_key, values.shape, values.dtype)

        if buffer_key not in self.updated:
            self.updated.add(buffer_key)
            self.local_tensor[buffer_key].copy_(values)
            # Make sure updated buffer is visible to all ranks
            torch.distributed.barrier()
        return self.local_tensor[buffer_key]

    @classmethod
    def set_shmem_flag(cls, tensor):
        tensor._odc_is_shmem = True

    @classmethod
    def is_shmem_tensor(cls, tensor):
        return hasattr(tensor, "_odc_is_shmem") and tensor._odc_is_shmem

    def allocate_symm_buffer(self, key, shape, dtype):
        """Allocate a symmetric tensor and record same-node peer views.

        mori backend: one-shot via ``mori_shmem_create_tensor_list_intra_node``,
        which returns ``list[torch.Tensor]`` of length ``local_world_size``,
        each entry being the same symmetric allocation viewed from the
        corresponding peer's address space. ``tensors[my_local_rank]`` is
        the local view; the entire list is recorded as the peer-view list.

        rocshmem backend: one-shot ``_rs.alloc_peer_tensors`` (equivalent).
        """
        assert key not in self.local_tensor
        local_world_size = get_local_world_size()
        rank = torch.distributed.get_rank()
        local_rank = rank % local_world_size

        if _USE_ROCSHMEM:
            # rocSHMEM host-API: one-shot symmetric alloc + same-node peer-view
            # list (equivalent to MORI's mori_shmem_create_tensor_list_intra_node).
            tensor, peer_tensors = _rs.alloc_peer_tensors(shape, dtype, local_world_size, rank)
            self.allocations.append(tensor)
        else:
            # node_start == 0 covers BOTH the single-node case and node 0 of a
            # multi-node run. In that case the same-node PEs are exactly
            # [0, local_world_size), so we call MORI's one-shot helper VERBATIM
            # — byte-for-byte the original single-node behaviour (single-node
            # protected, zero change).
            #
            # node_start > 0 (only reachable on node>0 of a multi-node run):
            # MORI's helper hardcodes GLOBAL PEs range(lws) == [0, lws), which
            # are cross-node from this node (ptr_p2p -> 0 -> bogus tensor at
            # address 0, also corrupting the "local" view). We instead allocate
            # the symmetric tensor once and enumerate THIS node's real PE range
            # [node_start, node_start + lws).
            node_start = rank - local_rank
            if node_start == 0:
                peer_tensors = _mori_shmem.mori_shmem_create_tensor_list_intra_node(
                    shape, dtype, local_world_size
                )
                assert len(peer_tensors) == local_world_size, (
                    f"mori_shmem_create_tensor_list_intra_node returned "
                    f"{len(peer_tensors)} tensors, expected {local_world_size}"
                )
                tensor = peer_tensors[local_rank]
            else:
                base_tensor = _mori_shmem.mori_shmem_create_tensor(shape, dtype)
                peer_tensors = [
                    _mori_shmem.symm_mori_shmem_tensor(base_tensor, node_start + i)
                    for i in range(local_world_size)
                ]
                assert len(peer_tensors) == local_world_size, (
                    f"expected {local_world_size} same-node peer views, " f"got {len(peer_tensors)}"
                )
                # peer_tensors[local_rank] resolves to peer == my_pe, i.e. the
                # true local allocation (base_tensor) — right object to free.
                tensor = peer_tensors[local_rank]
            # Keep the same allocation list to track for finalize.
            self.allocations.append(tensor)

        self.set_shmem_flag(tensor)
        assert len(peer_tensors) == local_world_size

        # ranks inside the same node must be contiguous.
        self.local_tensor[key] = tensor
        self.peer_tensors[key] = peer_tensors
        self.local_tensor_to_keys[self.local_tensor[key].data_ptr()] = key
        logger.info(
            f"Rank {rank} create tensor {key} with shape {shape} and dtype {dtype} "
            f"and ptr {self.local_tensor[key].data_ptr():#x}"
        )
        return tensor

    def has_key(self, key):
        return key in self.local_tensor

    def get_or_create_symm_buffer(self, key, shape, dtype):
        if self.has_key(key):
            return self.local_tensor[key]
        return self.allocate_symm_buffer(key, shape, dtype)

    def get_symm_buffer(self, key):
        if self.has_key(key):
            return self.local_tensor[key]
        raise ValueError(f"Symm buffer {key} not found")

    def get_peer_tensors(self, local_tensor):
        # Returns tensors in the same node.
        buffer_key = self.local_tensor_to_keys[local_tensor.data_ptr()]
        return self.peer_tensors[buffer_key]

    def finalize(self):
        for t in self.allocations:
            shmem_free_tensor_sync(t)
        self.local_tensor.clear()
        self.local_tensor_to_keys.clear()
        self.updated.clear()
        self.peer_tensors.clear()

    def memory_allocated(self):
        return sum(t.nbytes for t in self.allocations)


# ---------------------------------------------------------------------------
# Process group helpers — unchanged from upstream
# ---------------------------------------------------------------------------
local_world_pg = None


def get_local_world_pg(pg: torch.distributed.ProcessGroup):
    local_world_size = get_local_world_size()
    assert torch.distributed.get_world_size() == torch.distributed.get_world_size(
        group=pg
    ), "Cached AG only supports pure data parallelism"
    rank = torch.distributed.get_rank()
    global local_world_pg
    if local_world_pg is None:
        for i in range(0, torch.distributed.get_world_size(), local_world_size):
            ranks = list(range(i, i + local_world_size))
            new_gp = torch.distributed.new_group(ranks=ranks, backend="nccl")
            if rank in ranks:
                local_world_pg = new_gp
    assert local_world_pg is not None
    return local_world_pg


def get_local_world_size():
    if "RAY_LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["RAY_LOCAL_WORLD_SIZE"])
    else:
        return int(os.environ["LOCAL_WORLD_SIZE"])


stream = None


def get_comm_stream():
    global stream
    if stream is None:
        stream = torch.cuda.Stream()
    return stream


class BufferSplitter:
    def __init__(self):
        self.round_data_size = 2**6

    def get_max_global_buffer_size(self):
        # config item odc_max_buffer_size (default 64 MiB).
        max_buffer_size = int(get_config().max_buffer_size)
        return max_buffer_size

    def get_global_buffer_size(self, original_buffer_shape):
        original_size = reduce(lambda x, y: x * y, original_buffer_shape)
        max_buffer_size = self.get_max_global_buffer_size()
        if max_buffer_size <= 0:
            return original_size
        buf_size = min(max_buffer_size, original_size)
        return buf_size

    def round_to_data_size(self, size):
        return (size + self.round_data_size - 1) // self.round_data_size * self.round_data_size

    def get_local_buffer_size(self, original_buffer_shape, world_size):
        original_size = reduce(lambda x, y: x * y, original_buffer_shape)
        max_buffer_size = self.get_max_global_buffer_size()
        if max_buffer_size <= 0:
            return self.round_to_data_size(original_size)
        assert (
            max_buffer_size % world_size == 0
        ), f"ODC_MAX_BUFFER_SIZE: {max_buffer_size} % world_size: {world_size} != 0"
        local_max_buffer_size = max_buffer_size // world_size
        buf_size = min(local_max_buffer_size, original_size)
        return self.round_to_data_size(buf_size)


# ---------------------------------------------------------------------------
# sync_cta — CTA-internal synchronization helper
# ---------------------------------------------------------------------------
# Elect a single thread (lane 0) to do the atomic_add using the Triton vector
# mask pattern (no NVVM tid intrinsic exists on AMDGPU).
@triton.jit
def sync_cta(signal_ptr, expected):
    # Use a length-1 lane vector with mask to ensure only lane 0 of
    # the wave performs the atomic_add. This is the Triton-native way
    # to elect a "first thread".
    offsets = tl.arange(0, 1)
    mask = offsets == 0
    tl.atomic_add(signal_ptr + offsets, 1, mask=mask)
    __syncthreads()
    r = 0
    while r < expected:
        signals = tl.load(signal_ptr + offsets, mask=mask, volatile=True)
        r = tl.max(signals)
