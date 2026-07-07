# Adapted from ODC (https://github.com/sail-sg/odc), whose upstream Triton
# device API (odc/primitives/nvshmem_triton.py) this ROCm reimplementation is
# derived from. ODC is distributed under the MIT License per its package
# metadata (pyproject.toml / setup.py classifiers); the upstream repository
# ships no LICENSE file or per-file copyright headers, and upstream copyright is
# held by the ODC authors (Sea AI Lab).
#
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# See LICENSE for license information.

"""
ODC Triton device API — ROCm implementation.

ODC's Triton kernels reach symmetric peer memory through a small set of
device primitives. On ROCm there are two backends, selected by the
``ODC_P2P_BACKEND`` environment variable:

* ``mori`` (default): re-exports MORI-SHMEM's device API. MORI provides the
  device bitcode (``libmori_shmem_device.bc``, JIT-compiled by
  ``mori.ir.find_bitcode()``) that is linked into the Triton kernels.
* ``rocshmem``: a host-API rocSHMEM backend (single-node / IPC-only) that
  links NO device bitcode — every P2P op is plain Triton to an XGMI-mapped
  peer address.

Module-level public names (stable across backends):
    putmem_nbi_block, getmem_nbi_block, quiet, int_p, int_g,
    int_atomic_compare_swap, int_atomic_swap,
    tid, __syncthreads,
    LIB_SHMEM_PATH, SHMEM_EXTERN_LIBS

Design notes
------------
1. ``putmem_nbi_block``, ``getmem_nbi_block``, ``int32_p`` and other MORI
   device functions take an *extra* ``qp`` (queue-pair index) argument. The
   ODC wrappers below thread a fixed ``qp=0`` inside ``@triton.jit``-decorated
   thin shims so callers in ODC need not change.
2. MORI has no single-int blocking get, so ``int_g`` is implemented via a
   4-byte ``getmem_thread`` into a scratch register.
3. ``tid(axis=0)`` was used by ``utils.sync_cta`` to elect a "first thread"
   in a wave that performs an atomic_add. ``llvm.nvvm.read.ptx.sreg.tid.x``
   does not exist on AMDGPU. We provide a Triton-native alternative: an
   ``arange(0, 1)`` lane vector with mask ``offsets == 0`` restricts execution
   to lane 0, so ``tid`` is implemented as a constant that the caller compares
   against 0.
4. ``__syncthreads`` is implemented via Triton's IR barrier
   (``create_barrier``), which lowers to ``s_barrier`` on AMDGPU.
5. ``SHMEM_EXTERN_LIBS`` is ``mori.ir.triton.get_extern_libs()`` which locates
   the JIT-compiled MORI device bitcode for the current GPU + NIC. The Triton
   compile hook ``install_hook()`` is called at import time.
"""

import os

import triton
import triton.language as tl
from packaging import version
from triton.language import core

# ---------------------------------------------------------------------------
# ODC P2P backend selector. Default "mori" keeps the existing, verified
# behaviour byte-for-byte; "rocshmem" selects the host-API backend
# (single-node / IPC-only) defined in the dedicated branch below.
# ---------------------------------------------------------------------------
_P2P_BACKEND = os.environ.get("ODC_P2P_BACKEND", "mori").lower()


def is_triton_version_supported():
    return version.parse(triton.__version__) >= version.parse("3.4.0")


# ===========================================================================
# Branch A: rocSHMEM host-API backend (single-node / IPC-only).
#
# No device bitcode is linked into Triton (SHMEM_EXTERN_LIBS == {}); every
# P2P op is plain Triton to an XGMI-mapped peer address. A LOCAL symmetric
# address is translated to a peer address by adding a per-PE affine byte delta
# (rocshmem_ptr is affine — empirically verified). The delta table is populated
# once at init by _rocshmem_backend.alloc_peer_tensors (before any kernel
# launch) via set_rocshmem_peer_deltas() below.
#
# REGRESSION GUARD (mirrors the mori branch): no RDMA / device-library symbol
# ever enters these kernels — they are pure Triton. Use a per-rank/per-run
# TRITON_CACHE_DIR so a rank never loads another rank's cached kernel (the
# baked deltas differ per process).
# ===========================================================================
if _P2P_BACKEND == "rocshmem":
    # Empty extern-libs: Triton compiles these kernels with NO device bitcode.
    SHMEM_EXTERN_LIBS = {}
    LIB_NAME = "librocshmem_host"
    LIB_SHMEM_PATH = ""

    # Per-PE affine pointer deltas (bytes), indexed by LOCAL position
    # (global pe - node_start), baked into the int_p / int_g /
    # int_wait_until_equals kernels at compile time. Triton only lets @jit
    # functions read globals that are tl.constexpr, so the deltas are stored as
    # constexpr (their values are captured AND versioned in the compile cache
    # key per the used_global_vals mechanism -> a rank never reuses another
    # rank's kernel even on a shared TRITON_CACHE_DIR). ``_RS_NODE_START`` is the
    # first global rank on this node so a runtime global pe maps to a local slot.
    _RS_D0 = tl.constexpr(0)
    _RS_D1 = tl.constexpr(0)
    _RS_D2 = tl.constexpr(0)
    _RS_D3 = tl.constexpr(0)
    _RS_D4 = tl.constexpr(0)
    _RS_D5 = tl.constexpr(0)
    _RS_D6 = tl.constexpr(0)
    _RS_D7 = tl.constexpr(0)
    _RS_NODE_START = tl.constexpr(0)

    def set_rocshmem_peer_deltas(deltas, node_start=0):
        """Populate the device peer-delta table. Called once at init, BEFORE any
        kernel launch. ``deltas[i]`` == peer_addr(x, node_start+i) - x for
        symmetric x (indexed by LOCAL position i). ``node_start`` is this node's
        first global rank, used to map a runtime global pe to a local slot."""
        global _RS_D0, _RS_D1, _RS_D2, _RS_D3, _RS_D4, _RS_D5, _RS_D6, _RS_D7
        global _RS_NODE_START
        d = list(deltas) + [0] * (8 - len(deltas))
        _RS_D0 = tl.constexpr(d[0])
        _RS_D1 = tl.constexpr(d[1])
        _RS_D2 = tl.constexpr(d[2])
        _RS_D3 = tl.constexpr(d[3])
        _RS_D4 = tl.constexpr(d[4])
        _RS_D5 = tl.constexpr(d[5])
        _RS_D6 = tl.constexpr(d[6])
        _RS_D7 = tl.constexpr(d[7])
        _RS_NODE_START = tl.constexpr(node_start)

    @triton.jit
    def _rs_peer_delta(pe):
        # Select the baked delta for runtime global ``pe`` as int64 via a masked
        # sum over LOCAL positions (exactly one term is non-zero for a same-node
        # pe; all terms are int64).
        p = pe.to(tl.int64) - _RS_NODE_START
        d = (p == 0).to(tl.int64) * _RS_D0
        d += (p == 1).to(tl.int64) * _RS_D1
        d += (p == 2).to(tl.int64) * _RS_D2
        d += (p == 3).to(tl.int64) * _RS_D3
        d += (p == 4).to(tl.int64) * _RS_D4
        d += (p == 5).to(tl.int64) * _RS_D5
        d += (p == 6).to(tl.int64) * _RS_D6
        d += (p == 7).to(tl.int64) * _RS_D7
        return d

    @triton.jit
    def _rs_peer_addr(local_ptr, pe):
        # Translate a LOCAL symmetric pointer to the peer's XGMI-mapped address.
        delta = _rs_peer_delta(pe).to(tl.uint64, bitcast=True)
        return local_ptr.to(tl.uint64, bitcast=True) + delta

    @triton.jit
    def int_p(dest, value, pe):
        """Same-node single-int put: SYSTEM-scope release atomic store to the
        peer's XGMI-mapped fine-grained slot (visible to a peer CPU-side read).
        ``pe == my_pe`` -> delta 0 -> local store."""
        peer = _rs_peer_addr(dest, pe).to(tl.pointer_type(tl.int32), bitcast=True)
        tl.atomic_xchg(peer, value, sem="release", scope="sys")

    @triton.jit
    def int_g(src, pe):
        """Single int read from a peer slot. A volatile load is fine for a
        one-shot read; spin loops must use int_wait_until_equals."""
        peer = _rs_peer_addr(src, pe).to(tl.pointer_type(tl.int32), bitcast=True)
        return tl.load(peer, volatile=True)

    @triton.jit
    def int_wait_until_equals(ptr, expected, pe):
        """Spin until the peer slot == ``expected`` using a SYSTEM-scope acquire
        atomic load (atomic_add of 0). This bypasses the MI300X L2 staleness that
        makes a naive volatile-load spin never observe a peer's ack
        (verified: ~sub-ms wait instead of a hang). The pure-Triton equivalent
        of mori's int32_wait_until_equals."""
        peer = _rs_peer_addr(ptr, pe).to(tl.pointer_type(tl.int32), bitcast=True)
        got = expected - 1
        while got != expected:
            got = tl.atomic_add(peer, 0, sem="acquire", scope="sys")

    @triton.jit
    def quiet():
        # Single-node: bulk transfer is host-side copy_ and the same-node
        # signalling kernels never call quiet(). No-op shim so the module
        # imports; never on the single-node hot path.
        pass

    @triton.jit
    def putmem_nbi_block(dest, source, nbytes, pe):
        # Single-node bulk moves are host-side copy_; this shim exists only so
        # the (never-launched-on-single-node) cross-node kernels compile.
        pass

    @triton.jit
    def getmem_nbi_block(dest, source, nbytes, pe):
        pass

    @triton.jit
    def int_p_remote(dest, value, pe):
        # Cross-node is unsupported in the single-node host-API backend.
        pass

    @triton.jit
    def int_wait_until_equals_remote(scratch_ptr, src_ptr, expected, pe):
        pass

    @triton.jit
    def int_atomic_compare_swap(dest, cond, value, pe):
        peer = _rs_peer_addr(dest, pe).to(tl.pointer_type(tl.int32), bitcast=True)
        return tl.atomic_cas(peer, cond, value, sem="acq_rel", scope="sys")

    @triton.jit
    def int_atomic_swap(dest, value, pe):
        peer = _rs_peer_addr(dest, pe).to(tl.pointer_type(tl.int32), bitcast=True)
        return tl.atomic_xchg(peer, value, sem="acq_rel", scope="sys")

    @core.extern
    def tid(axis: core.constexpr, _semantic=None):
        # Platform-portable: return constant 0 (see mori branch rationale).
        del axis
        return core.full((), 0, dtype=tl.int32, _semantic=_semantic)

    @core.extern
    def __syncthreads(_semantic=None):
        return tl.tensor(_semantic.builder.create_barrier(), tl.void)


# ===========================================================================
# Branch B: MORI backend (default) — re-export MORI device API.
# ===========================================================================
else:
    import mori.shmem as _ms  # noqa: F401  (used by users importing this module)
    from mori.ir import triton as _mt
    from mori.ir.triton import get_extern_libs as _mori_get_extern_libs
    from mori.ir.triton import install_hook as _mori_install_hook

    # Install MORI's Triton compile hook. After install_hook(), Triton
    # compilation of any kernel that uses MORI device functions will be linked
    # against libmori_shmem_device.bc.
    _mori_install_hook()

    # SHMEM_EXTERN_LIBS is a dict {LIB_NAME: LIB_SHMEM_PATH}; MORI returns the
    # same shape from ``get_extern_libs()``.
    SHMEM_EXTERN_LIBS = _mori_get_extern_libs()
    LIB_NAME = "libmori_shmem"
    # LIB_SHMEM_PATH kept for backward compatibility with code that probes the
    # dict; on ROCm this is the path to the JIT-compiled MORI bitcode.
    if isinstance(SHMEM_EXTERN_LIBS, dict) and SHMEM_EXTERN_LIBS:
        LIB_SHMEM_PATH = next(iter(SHMEM_EXTERN_LIBS.values()))
    else:
        LIB_SHMEM_PATH = ""

    # -----------------------------------------------------------------------
    # Device API: thin @triton.jit shims that fold the extra ``qp`` argument.
    # -----------------------------------------------------------------------
    # NOTE: MORI device functions take an extra qp index. ODC's signatures pass
    # (dest, src, nbytes, pe). We hard-code qp=0 for now (single-QP is the
    # default); if/when ODC wants multi-QP, change here.

    @triton.jit
    def putmem_nbi_block(dest, source, nbytes, pe):
        return _mt.putmem_nbi_block(dest, source, nbytes, pe, 0)

    @triton.jit
    def getmem_nbi_block(dest, source, nbytes, pe):
        return _mt.getmem_nbi_block(dest, source, nbytes, pe, 0)

    @triton.jit
    def quiet():
        return _mt.quiet_thread()

    @triton.jit
    def int_p(dest, value, pe):
        """Write a 32-bit int to ``dest`` on rank ``pe`` from this rank.

        SAME-NODE ONLY.
        ------------------------------------------------------------------
        Resolve the peer pointer via ``ptr_p2p`` (XGMI direct mapping)
        and write through it with ``tl.store``. This is the recommended
        same-node path in MORI's own examples.

        IMPORTANT (regression guard): this primitive is compiled into the
        SAME-NODE request kernel that runs on every single-node iteration.
        It must NOT reference any RDMA device function (``int32_p`` etc.).
        Linking an RDMA primitive into this kernel activates MORI's IBGDA
        path, which on a host without a usable RoCE/IB NIC silently breaks
        the same-node store (empirically: ODC grad reduce produced zero
        gradients on ~half the iters). Cross-node puts live in the separate
        ``int_p_remote`` below, which is only ever compiled into the
        cross-node-only kernels (never launched on a single node).

        Known limitation (investigated empirically on MI300X + ROCm 7.2)
        ---------------------------------------------------------------
        ROCm's XGMI cache coherence between the writer GPU and the
        reader GPU's CPU-side D2H copy is not strict: a P2P-written
        int32 can take O(1 second) before the destination GPU's
        ``cudaMemcpy(D2H)`` returns the new value.
        """
        my_pe = _mt.my_pe()
        if pe == my_pe:
            tl.store(dest.to(tl.pointer_type(tl.int32)), value)
        else:
            peer_raw = _mt.ptr_p2p(dest.to(tl.uint64, bitcast=True), my_pe, pe)
            peer_ptr = peer_raw.to(tl.pointer_type(tl.int32), bitcast=True)
            tl.store(peer_ptr, value)

    @triton.jit
    def int_p_remote(dest, value, pe):
        """Cross-node single-int put: write ``value`` to the symmetric address
        ``dest`` on the remote PE ``pe`` over RDMA.

        Uses MORI's ``int32_p`` which the device API dispatches onto the RDMA
        (IBGDA) transport for a cross-node PE (and P2P for a same-node PE, so
        it is also safe if called with a same-node peer). ``dest`` is the
        LOCAL symmetric address; MORI resolves the remote PE's matching
        symmetric offset internally (qp=0).

        This primitive is intentionally SEPARATE from ``int_p`` so that the
        RDMA device symbol is only ever linked into the cross-node-only
        kernels (``shmem_*_remote_node_kernel`` / ``shmem_cross_node_*``),
        which are never compiled/launched on a single node. This keeps the
        single-node code path unchanged.
        """
        _mt.int32_p(dest.to(tl.uint64, bitcast=True), value, pe, 0)

    @triton.jit
    def int_g(src, pe):
        """Read a 32-bit int from ``src`` on rank ``pe`` to this rank (single read).

        SAME-NODE ONLY (same regression-guard rationale as ``int_p``).
        ------------------------------------------------------------------
        Resolve the peer pointer via ``ptr_p2p`` (XGMI direct mapping) and
        read with ``tl.load``. NOTE: a *single* read is fine, but do NOT
        use this inside a hot in-kernel spin loop — on ROCm the GPU L2
        caches the peer address and a repeated volatile load never
        observes a cross-process update. Use ``int_wait_until_equals``
        (same node) or ``int_wait_until_equals_remote`` (cross node) for
        spin-waiting instead.
        """
        my_pe = _mt.my_pe()
        if pe == my_pe:
            return tl.load(src.to(tl.pointer_type(tl.int32)), volatile=True)
        peer_raw = _mt.ptr_p2p(src.to(tl.uint64, bitcast=True), my_pe, pe)
        peer_ptr = peer_raw.to(tl.pointer_type(tl.int32), bitcast=True)
        return tl.load(peer_ptr, volatile=True)

    @triton.jit
    def int_wait_until_equals(ptr, expected, pe):
        """Block until the int32 at ``ptr`` on rank ``pe`` equals ``expected``.

        This is the correct replacement for a naive
            ``while r != expected: quiet(); r = int_g(...)``
        spin loop on ROCm.

        Root cause it fixes
        -------------------
        A naive spin ``while ...: tl.load(volatile=True)`` does not observe
        cross-process updates on MI300X: the GPU L2 caches the peer address
        and a long-lived in-kernel spin never sees a cross-process peer's
        ack write, causing the scatter_accumulate "60s hang".

        MORI's ``int32_wait_until_equals`` is implemented as
            ``while (AtomicLoadRelaxedSystem(addr) != val) {}``
        i.e. a SYSTEM-SCOPE atomic load that bypasses/refreshes the L2, so
        the cross-process / cross-GPU write becomes visible. Verified on
        MI300X: replacing the volatile spin with this turns a hang into a
        ~0.19 ms wait. We pass the PEER's P2P-resolved address so the
        same-node read path is exercised.
        """
        my_pe = _mt.my_pe()
        if pe == my_pe:
            target = ptr.to(tl.uint64, bitcast=True)
        else:
            target = _mt.ptr_p2p(ptr.to(tl.uint64, bitcast=True), my_pe, pe)
        _mt.int32_wait_until_equals(target, expected)

    @triton.jit
    def int_wait_until_equals_remote(scratch_ptr, src_ptr, expected, pe):
        """Cross-node spin-wait: block until the int32 at ``src_ptr`` on rank
        ``pe`` equals ``expected``.

        Why a separate primitive (vs ``int_wait_until_equals``)
        ------------------------------------------------------
        MORI's ``int32_wait_until_equals(addr, val)`` polls a *local*
        address (``while AtomicLoadRelaxedSystem(addr) != val``). The
        same-node ``int_wait_until_equals`` makes that work by passing the
        peer's XGMI-mapped address (the peer's memory is visible in this
        GPU's address space). A cross-node peer has **no** such mapping —
        ``ptr_p2p`` returns 0 and there is no local address that aliases
        the remote slot — so we must actively pull the value over RDMA.

        Implementation
        --------------
        Repeatedly RDMA-``get`` the 4-byte remote slot into a local
        symmetric scratch and compare. ``getmem`` carries no atomic-alignment
        constraint (unlike a remote atomic), so any int32 slot works.

        Args:
            scratch_ptr: a LOCAL symmetric/registered int32 scratch slot
                (RDMA needs the get destination to be a registered MR;
                MORI symmetric-heap tensors qualify).
            src_ptr: the LOCAL symmetric address of the slot to watch; the
                value is fetched from the same symmetric offset on ``pe``.
            expected: the int32 value to wait for.
            pe: the remote PE that owns the authoritative slot.
        """
        got = expected - 1
        while got != expected:
            _mt.getmem_nbi_block(
                scratch_ptr.to(tl.uint64, bitcast=True),
                src_ptr.to(tl.uint64, bitcast=True),
                4,
                pe,
                0,
            )
            _mt.quiet_thread()
            got = tl.load(scratch_ptr.to(tl.pointer_type(tl.int32)), volatile=True)

    @triton.jit
    def int_atomic_compare_swap(dest, cond, value, pe):
        # MORI: atomic_uint32_fetch_thread(dest, val, cmp, op, pe, qp)
        # CAS op code = 1 (per mori/src/shmem/atomic_ops.hpp; if different,
        # adjust). Only used by ODC tests, not by the production gather/
        # scatter path, so a slightly-incorrect op code would not affect
        # the FSDP integration.
        # We expose the same call shape; the op code is hard-coded as the
        # MORI CAS-fetch primitive.
        # TODO: cross-check op code with mori headers when wiring tests.
        return _mt.atomic_uint32_fetch_thread(dest, value, cond, 1, pe, 0)

    @triton.jit
    def int_atomic_swap(dest, value, pe):
        # MORI atomic SWAP op code = 0 (placeholder; not used in production).
        return _mt.atomic_uint32_fetch_thread(dest, value, 0, 0, pe, 0)

    # -----------------------------------------------------------------------
    # tid / __syncthreads — platform-portable shims
    # -----------------------------------------------------------------------
    # ODC uses tid(axis=0) only to elect "first thread does the atomic_add"
    # in sync_cta. The Triton-native way to do this is a mask pattern:
    #     offsets = tl.arange(0, 1)
    #     tl.atomic_add(signal_ptr + offsets, 1, mask=offsets == 0)
    # rather than ``if tidx == 0: tl.atomic_add(signal_ptr, 1)``.
    # We patch sync_cta in utils.py to use the mask form; here, ``tid``
    # returns a constant 0 so that ``if tidx == 0`` branches taken in
    # ODC's existing kernels (request_accumulation_same_node_kernel and
    # wait_accumulation_same_node_kernel) still execute as before. In a
    # ``@triton.jit`` kernel, all lanes follow the same control-flow path
    # at the IR level, so ``if tid == 0`` becomes a conditional that
    # produces the same result on every lane — semantically equivalent to
    # "everybody does it" plus relying on the operation being idempotent
    # under repetition. For ``int_p`` (memory store) this is fine because
    # we issue the same write from every lane and MORI's underlying
    # ``mori_shmem_int32_p`` is a wave-collective store (single transaction
    # per wave).
    @core.extern
    def tid(axis: core.constexpr, _semantic=None):
        # axis is a Triton constexpr (0/1/2). Return constant 0 — all lanes
        # are treated identically. See note above for correctness rationale.
        # We return a scalar int32 value.
        del axis  # unused on ROCm path
        return core.full((), 0, dtype=tl.int32, _semantic=_semantic)

    @core.extern
    def __syncthreads(_semantic=None):
        # Platform-portable: Triton IR barrier. Lowers to s_barrier on AMDGPU.
        return tl.tensor(_semantic.builder.create_barrier(), tl.void)
