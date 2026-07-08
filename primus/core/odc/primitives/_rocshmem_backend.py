# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""ODC rocSHMEM P2P backend, consuming the rocSHMEM ops from Primus-Turbo.

Activated by ``ODC_P2P_BACKEND=rocshmem`` (the default is ``mori``). The
rocSHMEM host/GDA surface is no longer loaded from an in-tree ``librs_host*.so``
via ctypes; it is now provided by Primus-Turbo as two pybind submodules of
``primus_turbo.pytorch._C``:

  * ``odc_rocshmem_host`` — single-node host-API surface (XGMI IPC path):
    ``rs_get_uid`` / ``rs_init_uid`` / ``rs_malloc`` / ``rs_ptr`` /
    ``rs_barrier`` / ``rs_finalize``. Selected when ``ODC_ROCSHMEM_GDA`` != 1.
  * ``odc_rocshmem_gda`` — multi-node GPU-direct (GDA) surface: the same
    host-compatible ``rs_*`` bootstrap plus the device ``gda_gather`` /
    ``gda_reduce_scatter_acc`` launchers. Selected when ``ODC_ROCSHMEM_GDA=1``.

Single-node (host) path: this backend uses ONLY the host API to manage the
symmetric heap and resolve peer pointers. It deliberately does NOT link any
rocSHMEM *device* bitcode into Triton — every on-device P2P op is plain Triton
``tl.load`` / ``tl.store`` / ``tl.atomic_*`` to XGMI-mapped peer addresses (see
the ``rocshmem`` branch in ``shmem_triton.py``). The heavy gather / scatter data
movement is host-side torch ``.copy_()`` on peer-view tensors (XGMI peer
load/store); the only device primitives exercised are ``int_p`` /
``int_wait_until_equals`` for the scatter-accumulate hand-shake. Those translate
a local symmetric address to a peer address with a per-PE affine delta —
``rocshmem_ptr`` was empirically verified to be affine (``rs_ptr(x, pe) - x`` is
constant across symmetric addresses ``x``), so a single ``delta[pe]`` resolves
every symmetric pointer.

pybind surface consumed (see Primus-Turbo csrc/pytorch/dist/odc_rocshmem_*.cpp)::

    int       rs_uid_bytes();
    bytes     rs_get_uid();                     # returns the uid as `bytes`
    void      rs_init_uid(int rank, int nranks, bytes uid);
    int       rs_my_pe();
    int       rs_n_pes();
    long long rs_malloc(size_t n);              # symmetric-heap device ptr
    long long rs_ptr(long long p, int pe);      # peer mapping of symmetric ptr p
    void      rs_barrier();
    void      rs_finalize();
    # GDA submodule only, device launchers (peers passed as a Python list):
    int       gda_gather(target, src, nbytes, list peers, stride_bytes);
    int       gda_reduce_scatter_acc(...);      # etc.
"""

import ctypes
import logging
import os
from functools import reduce

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# c10::ScalarType enum codes (stable across torch versions).
_DTYPE_CODE = {
    torch.uint8: 0,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 5,
    torch.float32: 6,
    torch.float64: 7,
    torch.bool: 11,
    torch.bfloat16: 15,
}

# Max same-node PEs supported by the device delta table (MI300X node == 8 GPUs).
MAX_LOCAL_PES = 8

_FROM_BLOB_CPP = r"""
#include <torch/extension.h>
// Wrap an externally-owned device pointer as a torch.Tensor (no deleter: the
// rocSHMEM symmetric heap owns the memory and frees it at rs_finalize()).
torch::Tensor odc_rs_from_blob(int64_t ptr, std::vector<int64_t> sizes,
                               int64_t dtype_code, int64_t device_index) {
    auto dtype = static_cast<c10::ScalarType>(dtype_code);
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_index);
    return torch::from_blob(reinterpret_cast<void*>(ptr), sizes, options);
}
"""

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_lib = None
_from_blob = None
_my_pe = -1
_n_pes = -1
_ref_base = None  # reference symmetric address used for affine peer deltas
_allocations = []  # keep python tensor objects alive for the run
_initialized = False

_gda_enabled = False  # GPU-direct (GDA) device path: ODC_ROCSHMEM_GDA=1
_ctypes_gda_lib = False  # True when ODC_ROCSHMEM_LIB overrides embedded turbo GDA


def gda_enabled():
    """True when the GPU-direct (GDA) device-kernel cross-node path is active."""
    return _gda_enabled


def _is_gda():
    return os.environ.get("ODC_ROCSHMEM_GDA", "0") == "1"


def _load_ctypes_gda(so):
    """Load proven in-tree librs_host_gda.so when ODC_ROCSHMEM_LIB is set.

    The embedded turbo ``odc_rocshmem_gda`` path linked against rocshmem_combined
    can fault on first device gather; this override uses the monolithic RDC .so
    that matches the in-tree dual baseline.
    """
    lib = ctypes.CDLL(so)
    lib.rs_uid_bytes.restype = ctypes.c_int
    lib.rs_get_uid.argtypes = [ctypes.c_char_p]
    lib.rs_init_uid.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
    lib.rs_my_pe.restype = ctypes.c_int
    lib.rs_n_pes.restype = ctypes.c_int
    lib.rs_malloc.restype = ctypes.c_longlong
    lib.rs_malloc.argtypes = [ctypes.c_size_t]
    lib.rs_ptr.restype = ctypes.c_longlong
    lib.rs_ptr.argtypes = [ctypes.c_longlong, ctypes.c_int]
    lib.rs_barrier.restype = None
    lib.rs_finalize.restype = None
    if hasattr(lib, "gda_gather"):
        lib.gda_gather.restype = ctypes.c_int
        lib.gda_gather.argtypes = [
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        lib.gda_reduce_scatter_acc.restype = ctypes.c_int
        lib.gda_reduce_scatter_acc.argtypes = [
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
        ]
        if hasattr(lib, "gda_stage_fence"):
            lib.gda_stage_fence.restype = ctypes.c_int
            lib.gda_stage_fence.argtypes = [ctypes.c_longlong, ctypes.c_longlong, ctypes.c_size_t]
        if hasattr(lib, "gda_hdp_flush"):
            lib.gda_hdp_init.restype = ctypes.c_int
            lib.gda_hdp_init.argtypes = []
            lib.gda_hdp_flush.restype = ctypes.c_int
            lib.gda_hdp_flush.argtypes = []
        if hasattr(lib, "gda_strided_touch"):
            lib.gda_strided_touch.restype = ctypes.c_int
            lib.gda_strided_touch.argtypes = [
                ctypes.c_longlong,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_longlong,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
        if hasattr(lib, "gda_reduce_scatter_acc_async"):
            lib.gda_reduce_scatter_acc_async.restype = ctypes.c_int
            lib.gda_reduce_scatter_acc_async.argtypes = lib.gda_reduce_scatter_acc.argtypes
            lib.gda_rs_overlap_sync.restype = ctypes.c_int
            lib.gda_rs_overlap_sync.argtypes = []
        if hasattr(lib, "gda_gather_async"):
            lib.gda_gather_async.restype = ctypes.c_int
            lib.gda_gather_async.argtypes = [
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_size_t,
                ctypes.c_longlong,
            ]
    return lib


def _load_backend():
    """Resolve the ODC rocSHMEM backend submodule from Primus-Turbo.

    When ``ODC_ROCSHMEM_LIB`` points at a GDA ``librs_host_gda.so``, load that
    monolithic binding via ctypes instead of the embedded turbo GDA kernels.
    """
    global _ctypes_gda_lib
    so = os.environ.get("ODC_ROCSHMEM_LIB")
    if so and os.path.isfile(so) and _is_gda():
        _ctypes_gda_lib = True
        logger.info("rocSHMEM GDA backend: ctypes override %s", so)
        return _load_ctypes_gda(so)
    _ctypes_gda_lib = False
    try:
        import primus_turbo  # noqa: F401  -- package init (also loads the _C ext)
        import primus_turbo.pytorch._C as _C
    except ImportError as e:
        raise ImportError(
            "ODC rocSHMEM backend now consumes the rocSHMEM ops from Primus-Turbo "
            "(primus_turbo.pytorch._C.odc_rocshmem_host / odc_rocshmem_gda). Install "
            "Primus-Turbo built with the ODC rocSHMEM ops, or put its build tree on "
            "PYTHONPATH so `import primus_turbo` resolves it."
        ) from e
    name = "odc_rocshmem_gda" if _is_gda() else "odc_rocshmem_host"
    mod = getattr(_C, name, None)
    if mod is None:
        available = [n for n in dir(_C) if "odc" in n.lower()]
        raise RuntimeError(
            f"primus_turbo.pytorch._C has no submodule '{name}'. This primus_turbo "
            f"build was compiled without the ODC rocSHMEM ops (DISABLE_ROCSHMEM). "
            f"ODC submodules present: {available}"
        )
    return mod


def _build_from_blob():
    from torch.utils.cpp_extension import load_inline

    mod = load_inline(
        name="odc_rs_from_blob",
        cpp_sources=[_FROM_BLOB_CPP],
        functions=["odc_rs_from_blob"],
        verbose=False,
    )
    return mod.odc_rs_from_blob


def _wrap(ptr, shape, dtype, dev_index):
    if isinstance(shape, int):
        shape = (shape,)
    return _from_blob(int(ptr), list(shape), _DTYPE_CODE[dtype], int(dev_index))


def _nbytes(shape, dtype):
    if isinstance(shape, int):
        shape = (shape,)
    numel = reduce(lambda a, b: a * b, shape, 1)
    return numel * torch._utils._element_size(dtype)


def init():
    """Bootstrap rocSHMEM from PyTorch's WORLD process group via a unique-id
    broadcast.

    BOTH the single-node host (``odc_rocshmem_host``) and the multi-node
    GPU-direct (``odc_rocshmem_gda``) submodules use the SAME unique-id
    bootstrap: rank 0 generates the uid (``rs_get_uid``, returned as ``bytes``),
    it is broadcast over the torch WORLD process group, and every rank calls
    ``rs_init_uid`` (which maps to ``rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID)``).
    The GDA transport exchanges the uid over a TCP socket bootstrap
    (``ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME``), so NO MPI job / mpirun is required —
    the run can be launched with plain torchrun.
    """
    global _lib, _from_blob, _my_pe, _n_pes, _initialized, _gda_enabled
    if _initialized:
        return
    assert dist.is_initialized(), "torch.distributed must be initialized first"
    _lib = _load_backend()
    _from_blob = _build_from_blob()

    _gda_enabled = _is_gda() and hasattr(_lib, "gda_gather")
    if _gda_enabled:
        # GDA multi-node bootstrap carries rank-0's address inside the uid and
        # exchanges it over a TCP socket. Default the bootstrap NIC to eth0 unless
        # the deployment overrode it; this replaces the old MPI/mpirun bootstrap.
        os.environ.setdefault("ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME", "eth0")

    rank = dist.get_rank()
    world = dist.get_world_size()
    if _ctypes_gda_lib:
        n = _lib.rs_uid_bytes()
        buf = (ctypes.c_char * n)()
        uidb = None
        if rank == 0:
            _lib.rs_get_uid(buf)
            uidb = bytes(buf)
        obj = [uidb]
        dist.broadcast_object_list(obj, src=0)
        uidb = obj[0]
        ctypes.memmove(buf, uidb, n)
        _lib.rs_init_uid(rank, world, buf)
    else:
        # rs_get_uid() returns the uid as `bytes` (pybind); broadcast it as a python
        # object and hand the raw bytes back to rs_init_uid on every rank.
        uidb = _lib.rs_get_uid() if rank == 0 else None
        obj = [uidb]
        dist.broadcast_object_list(obj, src=0)
        uidb = obj[0]
        _lib.rs_init_uid(rank, world, uidb)

    _my_pe = _lib.rs_my_pe()
    _n_pes = _lib.rs_n_pes()
    assert (
        _my_pe == rank and _n_pes == world
    ), f"rocSHMEM PE mismatch: my_pe={_my_pe} rank={rank} n_pes={_n_pes} world={world}"
    logger.info(
        "init_shmem (rocSHMEM %s, uid bootstrap): my_pe=%d n_pes=%d",
        "GDA" if _gda_enabled else "host-API",
        _my_pe,
        _n_pes,
    )
    _initialized = True


def _ensure_peer_deltas(base, local_world_size, rank):
    """On the first allocation, compute the affine peer deltas and push them to
    the Triton device layer so int_p / int_g / int_wait_until_equals can
    translate a local symmetric address to a peer address on-device."""
    global _ref_base
    if _ref_base is not None:
        return
    assert local_world_size <= MAX_LOCAL_PES, (
        f"rocshmem backend supports up to {MAX_LOCAL_PES} same-node PEs, got "
        f"local_world_size={local_world_size}"
    )
    _ref_base = base
    node_start = rank - rank % local_world_size
    # Index deltas by LOCAL position (pe - node_start), NOT global pe: on node>0
    # the same-node PEs are [node_start, node_start+lws), which would overflow a
    # global-indexed 8-slot table. The device kernel translates a runtime global
    # pe to a local position via the baked node_start (see _rs_peer_delta).
    deltas = [0] * MAX_LOCAL_PES  # indexed by LOCAL position
    for i in range(local_world_size):
        pe = node_start + i
        if pe == _my_pe:
            deltas[i] = 0
        else:
            peer = _lib.rs_ptr(base, pe)
            if peer == 0:
                raise RuntimeError(f"rs_ptr(base, {pe}) == 0: no XGMI P2P route to same-node peer")
            deltas[i] = peer - base
    from .shmem_triton import set_rocshmem_peer_deltas

    set_rocshmem_peer_deltas(deltas, node_start)
    logger.info(
        "rocSHMEM peer deltas (node_start=%d, local_pos -> delta bytes): %s",
        node_start,
        deltas,
    )


def alloc_peer_tensors(shape, dtype, local_world_size, rank):
    """Allocate one symmetric buffer and return ``(local_tensor, peer_tensors)``
    where ``peer_tensors[i]`` is the same allocation viewed from the address
    space of same-node PE ``node_start + i`` (length ``local_world_size``).

    Equivalent to MORI's ``mori_shmem_create_tensor_list_intra_node``.
    """
    base = _lib.rs_malloc(_nbytes(shape, dtype))
    if base == 0:
        raise RuntimeError(
            f"rs_malloc({_nbytes(shape, dtype)} bytes) returned NULL — symmetric "
            f"heap exhausted? Raise the rocSHMEM heap size."
        )
    dev = torch.cuda.current_device()
    if _gda_enabled:
        # GDA (USE_IPC=OFF): no intra-node IPC peer views; all cross-node AND
        # same-node transfers go through device get/put. Return the local view
        # plus dummy peer entries (never used on the GDA path).
        local_tensor = _wrap(base, shape, dtype, dev)
        _allocations.append(local_tensor)
        return local_tensor, [local_tensor] * local_world_size
    _ensure_peer_deltas(base, local_world_size, rank)

    node_start = rank - rank % local_world_size
    peer_tensors = []
    for i in range(local_world_size):
        pe = node_start + i
        p = base if pe == _my_pe else _lib.rs_ptr(base, pe)
        if p == 0:
            raise RuntimeError(f"rs_ptr(base, {pe}) == 0 (no XGMI route to peer {pe})")
        peer_tensors.append(_wrap(p, shape, dtype, dev))
    local_tensor = peer_tensors[rank % local_world_size]
    _allocations.append(local_tensor)
    return local_tensor, peer_tensors


def create_tensor(shape, dtype):
    """Bare local symmetric tensor (no peer list)."""
    base = _lib.rs_malloc(_nbytes(shape, dtype))
    if base == 0:
        raise RuntimeError("rs_malloc returned NULL (symmetric heap exhausted?)")
    t = _wrap(base, shape, dtype, torch.cuda.current_device())
    _allocations.append(t)
    return t


def free_tensor(tensor):
    # rocSHMEM host binding exposes no per-allocation free; the whole symmetric
    # heap is released by rs_finalize(). No-op here (buffers live for the run).
    pass


def barrier():
    _lib.rs_barrier()


# ---------------------------------------------------------------------------
# GPU-direct (GDA) device-kernel launchers (only valid when gda_enabled()).
# Pointers are raw device addresses (int) into the GDA symmetric heap.
# ---------------------------------------------------------------------------
def gda_gather(target_ptr, src_ptr, nbytes, peers, stride_bytes):
    """Device gather: for each cross-node peer in ``peers`` (global PE/rank), pull
    its shard (at symmetric ``src_ptr``) into ``target_ptr + peer*stride_bytes``."""
    if len(peers) == 0:
        return
    if _ctypes_gda_lib:
        n = len(peers)
        arr = (ctypes.c_int * n)(*[int(p) for p in peers])
        rc = _lib.gda_gather(
            int(target_ptr), int(src_ptr), int(nbytes), arr, n, int(stride_bytes)
        )
    else:
        # pybind binds gda_gather(target, src, nbytes, std::vector<int> peers, stride)
        rc = _lib.gda_gather(
            int(target_ptr), int(src_ptr), int(nbytes), [int(p) for p in peers], int(stride_bytes)
        )
    if rc != 0:
        raise RuntimeError(f"gda_gather hipError={rc}")


def gda_reduce_scatter_acc(
    acc_ptr,
    input_ptr,
    seg_off_bytes,
    shard_elems,
    n_pes,
    scratch_ptr,
    scratch_stride_bytes,
    dtype_code,
    nblocks,
):
    """Device pull-based reduce-scatter accumulate: acc_fp32[i] += sum over all
    PEs of input[seg_off + i] (pulled from each PE). Race-free (on-chip sum)."""
    rc = _lib.gda_reduce_scatter_acc(
        int(acc_ptr),
        int(input_ptr),
        int(seg_off_bytes),
        int(shard_elems),
        int(n_pes),
        int(scratch_ptr),
        int(scratch_stride_bytes),
        int(dtype_code),
        int(nblocks),
    )
    if rc != 0:
        raise RuntimeError(f"gda_reduce_scatter_acc hipError={rc}")


def gda_stage_fence(dst_ptr, src_ptr, nbytes):
    """Copy src->dst (device) and SYSTEM-fence so the staged symmetric buffer is
    visible to the NIC before a remote PE's device getmem (HDP-flush substitute)."""
    rc = _lib.gda_stage_fence(int(dst_ptr), int(src_ptr), int(nbytes))
    if rc != 0:
        raise RuntimeError(f"gda_stage_fence hipError={rc}")


def gda_hdp_init():
    """Resolve this rank's GPU HDP flush register (call after set_device).
    Returns 0 on success; nonzero means the register could not be resolved."""
    if not hasattr(_lib, "gda_hdp_init"):
        return -99
    return int(_lib.gda_hdp_init())


def gda_hdp_flush():
    """Flush this GPU's HDP cache so prior symmetric writes are NIC-visible (the
    proper GPUDirect-RDMA write-visibility primitive). Returns 1 if flushed."""
    return int(_lib.gda_hdp_flush())


def gda_strided_touch(
    input_ptr,
    seg_off_bytes,
    seg_bytes,
    n_pes,
    stride_bytes,
    touch_bytes,
    scratch_ptr,
    scratch_stride_bytes,
    nblocks,
):
    """Strided page-touch warm-up: a tiny RDMA read at every ``stride_bytes`` page
    of my shard segment on every PE, priming all pages/NICs (deterministic
    read-triggered settle) at minimal volume vs the full-shard throwaway RS."""
    rc = _lib.gda_strided_touch(
        int(input_ptr),
        int(seg_off_bytes),
        int(seg_bytes),
        int(n_pes),
        int(stride_bytes),
        int(touch_bytes),
        int(scratch_ptr),
        int(scratch_stride_bytes),
        int(nblocks),
    )
    if rc != 0:
        raise RuntimeError(f"gda_strided_touch hipError={rc}")


def gda_reduce_scatter_acc_async(
    acc_ptr,
    input_ptr,
    seg_off_bytes,
    shard_elems,
    n_pes,
    scratch_ptr,
    scratch_stride_bytes,
    dtype_code,
    nblocks,
):
    """Comm/compute OVERLAP: launch reduce-scatter on a side stream WITHOUT syncing.
    Caller must gda_rs_overlap_sync() before re-staging input or consuming acc."""
    rc = _lib.gda_reduce_scatter_acc_async(
        int(acc_ptr),
        int(input_ptr),
        int(seg_off_bytes),
        int(shard_elems),
        int(n_pes),
        int(scratch_ptr),
        int(scratch_stride_bytes),
        int(dtype_code),
        int(nblocks),
    )
    if rc != 0:
        raise RuntimeError(f"gda_reduce_scatter_acc_async launch err={rc}")


def gda_rs_overlap_sync():
    """Wait for all pending overlapped reduce-scatter kernels on the side stream."""
    rc = _lib.gda_rs_overlap_sync()
    if rc != 0:
        raise RuntimeError(f"gda_rs_overlap_sync hipError={rc}")


def gda_gather_async(target_ptr, src_ptr, nbytes, peers, stride_bytes, stream):
    """Approach 1: launch all-gather kernel on the given HIP `stream` WITHOUT syncing,
    so FSDP2 prefetch overlaps it with compute. Reassembly + consumer order via the
    same stream (gather reads stable params -> no settle/barrier needed)."""
    if len(peers) == 0:
        return
    if _ctypes_gda_lib:
        n = len(peers)
        arr = (ctypes.c_int * n)(*[int(p) for p in peers])
        rc = _lib.gda_gather_async(
            int(target_ptr),
            int(src_ptr),
            int(nbytes),
            arr,
            n,
            int(stride_bytes),
            int(stream),
        )
    else:
        rc = _lib.gda_gather_async(
            int(target_ptr),
            int(src_ptr),
            int(nbytes),
            [int(p) for p in peers],
            int(stride_bytes),
            int(stream),
        )
    if rc != 0:
        raise RuntimeError(f"gda_gather_async launch err={rc}")


def dtype_code(dtype):
    return _DTYPE_CODE[dtype]


def finalize():
    global _initialized
    if _lib is not None and _initialized:
        _lib.rs_finalize()
        _initialized = False
