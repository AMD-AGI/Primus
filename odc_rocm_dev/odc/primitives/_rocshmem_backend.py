"""ODC rocSHMEM host-API P2P backend (single-node / IPC-only).

Activated by ``ODC_P2P_BACKEND=rocshmem`` (the default is ``mori``). This
backend uses ONLY rocSHMEM's *host* API (``librs_host5.so``:
``rs_init_uid`` / ``rs_malloc`` / ``rs_ptr`` / ``rs_barrier`` / ``rs_finalize``)
to manage the symmetric heap and resolve peer pointers. It deliberately does
NOT link any rocSHMEM *device* bitcode into Triton — every on-device P2P op is
plain Triton ``tl.load`` / ``tl.store`` / ``tl.atomic_*`` to XGMI-mapped peer
addresses (see the ``rocshmem`` branch in ``nvshmem_triton.py``).

Single-node only. The heavy gather / scatter data movement is host-side torch
``.copy_()`` on peer-view tensors (XGMI peer load/store); the only device
primitives exercised are ``int_p`` / ``int_wait_until_equals`` for the
scatter-accumulate hand-shake. Those translate a local symmetric address to a
peer address with a per-PE affine delta — ``rocshmem_ptr`` was empirically
verified to be affine (``rs_ptr(x, pe) - x`` is constant across symmetric
addresses ``x``), so a single ``delta[pe]`` resolves every symmetric pointer.

Host-API ABI (extern "C" in ``librs_host5.so``)::

    int       rs_uid_bytes();
    void      rs_get_uid(char* out);
    void      rs_init_uid(int rank, int nranks, const char* bytes);
    int       rs_my_pe();
    int       rs_n_pes();
    long long rs_malloc(size_t n);          // symmetric-heap device ptr
    long long rs_ptr(long long p, int pe);  // peer mapping of symmetric ptr p
    void      rs_barrier();
    void      rs_finalize();
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

# RO (Reverse-Offload) host-driven cross-node path. Enabled by ODC_ROCSHMEM_RO=1
# together with the RO-capable binding (librs_host_ro.so), which adds host
# put/get primitives forwarded over the MPI/UCX conduit from the CPU (no
# device-side completion polling -> avoids MORI's GPU-initiated IBGDA hang).
_ro_enabled = False
_gda_enabled = False  # GPU-direct (GDA) device path: ODC_ROCSHMEM_GDA=1
_mpi = None  # ctypes handle to libmpi (kept alive; MPI_Init_thread lives here)


def ro_enabled():
    """True when the host-driven cross-node RO path is active."""
    return _ro_enabled


def gda_enabled():
    """True when the GPU-direct (GDA) device-kernel cross-node path is active."""
    return _gda_enabled


def _is_gda():
    return os.environ.get("ODC_ROCSHMEM_GDA", "0") == "1"


def _default_host_lib():
    """Project-relative fallback path to the rocSHMEM host binding.

    The binary ships inside the project at
    ``odc_rocm_dev/rocshmem_runtime/host_bindings/librs_host5.so`` so the
    backend works on any base image that mounts only the project directory
    (no dependence on the container's ``/root`` layer). The multinode RO
    binding lives next to it under ``ro_backend/librs_host_ro.so`` and is
    selected when ``ODC_ROCSHMEM_RO=1``.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    runtime = os.path.normpath(os.path.join(here, "..", "..", "rocshmem_runtime"))
    if _is_gda():
        return os.path.join(runtime, "gda_backend", "librs_host_gda.so")
    if os.environ.get("ODC_ROCSHMEM_RO", "0") == "1":
        return os.path.join(runtime, "ro_backend", "librs_host_ro.so")
    return os.path.join(runtime, "host_bindings", "librs_host5.so")


def _resolve_host_lib():
    """Resolve the rocSHMEM host binding .so path.

    Resolution order (first hit wins), so deployment can override without
    touching code:
      1. ``ODC_ROCSHMEM_LIB`` — explicit full path to the .so (back-compat).
      2. ``ODC_RS_HOST_LIB``  — alias for the explicit full path.
      3. ``ROCSHMEM_LIB_DIR``/librs_host[_ro].so — a directory holding the .so.
      4. project-relative default under ``rocshmem_runtime/`` (see above).
    """
    for var in ("ODC_ROCSHMEM_LIB", "ODC_RS_HOST_LIB"):
        v = os.environ.get(var)
        if v:
            return v
    lib_dir = os.environ.get("ROCSHMEM_LIB_DIR")
    if lib_dir:
        ro = os.environ.get("ODC_ROCSHMEM_RO", "0") == "1"
        name = "librs_host_ro.so" if ro else "librs_host5.so"
        return os.path.join(lib_dir, name)
    return _default_host_lib()


def _load_lib():
    so = _resolve_host_lib()
    if not os.path.exists(so):
        raise FileNotFoundError(
            f"rocSHMEM host binding not found at {so}. Set ODC_ROCSHMEM_LIB "
            f"(or ODC_RS_HOST_LIB) to the path of librs_host5.so, or "
            f"ROCSHMEM_LIB_DIR to the directory containing it. The binary "
            f"ships in the project at odc_rocm_dev/rocshmem_runtime/."
        )
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
    # RO-capable bindings expose host-driven cross-node primitives. Bind them
    # if present (librs_host_ro.so); absent on the single-node librs_host5.so.
    if hasattr(lib, "rs_putmem"):
        lib.rs_is_remote.restype = ctypes.c_int
        lib.rs_is_remote.argtypes = [ctypes.c_longlong, ctypes.c_int]
        lib.rs_putmem.restype = None
        lib.rs_putmem.argtypes = [ctypes.c_longlong, ctypes.c_longlong, ctypes.c_size_t, ctypes.c_int]
        lib.rs_getmem.restype = None
        lib.rs_getmem.argtypes = [ctypes.c_longlong, ctypes.c_longlong, ctypes.c_size_t, ctypes.c_int]
        lib.rs_int_p.restype = None
        lib.rs_int_p.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_int]
        lib.rs_quiet.restype = None
        lib.rs_quiet.argtypes = []
        lib.rs_fence.restype = None
        lib.rs_fence.argtypes = []
    # GDA binding (librs_host_gda.so): MPI bootstrap + device-kernel launchers.
    if hasattr(lib, "gda_gather"):
        lib.rs_init_mpi.restype = None
        lib.rs_init_mpi.argtypes = []
        lib.rs_is_remote.restype = ctypes.c_int
        lib.rs_is_remote.argtypes = [ctypes.c_longlong, ctypes.c_int]
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


# MPI thread-support level (MPI_THREAD_MULTIPLE == 3 in the MPI standard). The
# RO conduit runs a background progress thread, so the job MUST be initialized
# with MULTIPLE or one-sided window progress can deadlock.
_MPI_THREAD_MULTIPLE = 3


def _ensure_mpi_initialized():
    """Initialize MPI with THREAD_MULTIPLE before rocSHMEM-RO bootstrap.

    The RO binding links system OpenMPI (libmpi.so.40) and creates one-sided
    MPI windows inside ``rs_init_uid``. Those windows span the WHOLE job's
    MPI_COMM_WORLD, so the process MUST be part of a real MPI job (launched via
    ``srun --mpi=pmix`` / ``mpirun``); a torchrun launch yields a singleton
    MPI_COMM_WORLD and ``MPI_Win_create`` then fails / cannot reach peers.
    """
    global _mpi
    _mpi = ctypes.CDLL("libmpi.so.40", mode=ctypes.RTLD_GLOBAL)
    inited = ctypes.c_int(0)
    _mpi.MPI_Initialized(ctypes.byref(inited))
    if inited.value:
        return
    provided = ctypes.c_int(0)
    rc = _mpi.MPI_Init_thread(None, None, _MPI_THREAD_MULTIPLE, ctypes.byref(provided))
    if rc != 0:
        raise RuntimeError(f"MPI_Init_thread failed rc={rc}")
    if provided.value < _MPI_THREAD_MULTIPLE:
        logger.warning(
            "MPI thread level %d < MPI_THREAD_MULTIPLE(%d): RO progress may stall",
            provided.value,
            _MPI_THREAD_MULTIPLE,
        )


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
    broadcast (mirrors the verified single-node init)."""
    global _lib, _from_blob, _my_pe, _n_pes, _initialized, _ro_enabled, _gda_enabled
    if _initialized:
        return
    assert dist.is_initialized(), "torch.distributed must be initialized first"
    _lib = _load_lib()
    _from_blob = _build_from_blob()

    _gda_enabled = _is_gda() and hasattr(_lib, "gda_gather")
    if _gda_enabled:
        # GDA bootstrap: rocshmem_init() over MPI_COMM_WORLD (the validated probe
        # path). Needs MPI + UCX-OSC from the mpirun launch; MPI_Win_create for
        # the host setup uses the UCX OSC component. THREAD_MULTIPLE is safe.
        _ensure_mpi_initialized()
        _lib.rs_init_mpi()
        _my_pe = _lib.rs_my_pe()
        _n_pes = _lib.rs_n_pes()
        assert _my_pe == dist.get_rank() and _n_pes == dist.get_world_size(), (
            f"GDA PE mismatch: my_pe={_my_pe} rank={dist.get_rank()} "
            f"n_pes={_n_pes} world={dist.get_world_size()}"
        )
        logger.info("init_nvshmem (rocSHMEM GDA): my_pe=%d n_pes=%d", _my_pe, _n_pes)
        _initialized = True
        return

    _ro_enabled = os.environ.get("ODC_ROCSHMEM_RO", "0") == "1" and hasattr(_lib, "rs_putmem")
    if _ro_enabled:
        # RO bootstrap needs an already-running MPI job (THREAD_MULTIPLE) before
        # the symmetric heap / one-sided windows are created in rs_init_uid.
        _ensure_mpi_initialized()

    rank = dist.get_rank()
    world = dist.get_world_size()
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

    _my_pe = _lib.rs_my_pe()
    _n_pes = _lib.rs_n_pes()
    assert (
        _my_pe == rank and _n_pes == world
    ), f"rocSHMEM PE mismatch: my_pe={_my_pe} rank={rank} n_pes={_n_pes} world={world}"
    logger.info("init_nvshmem (rocSHMEM host-API): my_pe=%d n_pes=%d", _my_pe, _n_pes)
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
    from .nvshmem_triton import set_rocshmem_peer_deltas

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
# Host-driven RO cross-node primitives (only valid when ro_enabled()).
#
# These forward to the rocSHMEM RO conduit from the CPU. ``dest``/``src`` are
# raw device pointers (int) into the symmetric heap; ``pe`` is the GLOBAL PE
# (== torch rank). ``src`` (get) / ``dest`` (put) must be SYMMETRIC addresses
# so the runtime resolves the matching object on the remote PE.
# ---------------------------------------------------------------------------
def is_remote(base_ptr, pe):
    """1 if PE ``pe`` is inter-node (must use RO put/get); 0 if intra-node."""
    return int(_lib.rs_is_remote(int(base_ptr), int(pe)))


def putmem(dest_ptr, src_ptr, nbytes, pe):
    """Blocking host put: copy ``nbytes`` from local symmetric ``src_ptr`` into
    the symmetric object ``dest_ptr`` on PE ``pe`` (forwarded over the conduit)."""
    _lib.rs_putmem(int(dest_ptr), int(src_ptr), int(nbytes), int(pe))


def getmem(dest_ptr, src_ptr, nbytes, pe):
    """Blocking host get: pull ``nbytes`` from symmetric ``src_ptr`` on PE
    ``pe`` into local symmetric ``dest_ptr``."""
    _lib.rs_getmem(int(dest_ptr), int(src_ptr), int(nbytes), int(pe))


def int_p(dest_ptr, value, pe):
    """Host int store to the symmetric int slot ``dest_ptr`` on PE ``pe`` (used
    for the cross-node scatter-accumulate request/ack handshake)."""
    _lib.rs_int_p(int(dest_ptr), int(value), int(pe))


def quiet():
    """Complete all outstanding RO puts/gets issued by this PE."""
    _lib.rs_quiet()


def fence():
    """Order RO puts to a given PE (point-to-point ordering)."""
    _lib.rs_fence()


# ---------------------------------------------------------------------------
# GPU-direct (GDA) device-kernel launchers (only valid when gda_enabled()).
# Pointers are raw device addresses (int) into the GDA symmetric heap.
# ---------------------------------------------------------------------------
def gda_gather(target_ptr, src_ptr, nbytes, peers, stride_bytes):
    """Device gather: for each cross-node peer in ``peers`` (global PE/rank), pull
    its shard (at symmetric ``src_ptr``) into ``target_ptr + peer*stride_bytes``."""
    n = len(peers)
    if n == 0:
        return
    arr = (ctypes.c_int * n)(*[int(p) for p in peers])
    rc = _lib.gda_gather(int(target_ptr), int(src_ptr), int(nbytes), arr, n, int(stride_bytes))
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
    n = len(peers)
    if n == 0:
        return
    arr = (ctypes.c_int * n)(*[int(p) for p in peers])
    rc = _lib.gda_gather_async(
        int(target_ptr), int(src_ptr), int(nbytes), arr, n, int(stride_bytes), int(stream)
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
