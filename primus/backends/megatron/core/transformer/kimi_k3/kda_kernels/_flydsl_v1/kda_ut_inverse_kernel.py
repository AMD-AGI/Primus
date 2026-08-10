###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native FlyDSL kernel for ``(I − L)^{-1}``, ``L`` strictly lower triangular.

Pass 4 profiled the production forward and found this stage — spelled as
Neumann doubling over batched torch GEMMs in :func:`..ops._ut_inverse_doubling`
— at **576 µs of a 4227 µs on-device forward, the largest single item outside
the two existing kernels** (``wp9_p4/PROGRESS.md`` §1). It is not arithmetic
bound and never was:

* ``L`` is nilpotent, so ``Σ_{k<2n} L^k = (Σ_{k<n} L^k)(I + L^n)`` inverts it in
  ``log2(C) = 6`` doublings, which is **10 batched GEMMs** on ``[NB, 64, 64]``.
* every one of those reads two 100 MB operands and writes a 100 MB result, so
  the stage moves ~3 GB of HBM for ``C³/3 ≈ 87 k`` FMAs per chunk — 1.1 GFLOP
  in total, which an MI355X does in single-digit microseconds.

The whole matrix is 16 KB. Keeping it on-chip and doing the substitution
directly turns 3 GB of traffic into 200 MB, and the arithmetic drops as well
because forward substitution does ``C³/3`` FMAs where doubling does ``5·C³``.

Geometry
--------
One workgroup of ``C`` threads per chunk; **lane ``c`` owns column ``c`` of
``P``**, held in ``C`` registers. ``L`` is staged in LDS (``C²`` fp32 = 16 KB at
``C = 64``) because every lane reads the same ``L[i, j]`` — an LDS broadcast,
one bank access for the whole wave.

    P[i, c] = δ_ic + Σ_{j<i} L[i, j] · P[j, c]

is ``P = I + L P``, i.e. exactly ``(I − L) P = I``, and exactly the recurrence
the eager reference runs serially in Python
(``_eager/reference.py``, and ``test_kda_flydsl_ut_inverse_matches_forward_
substitution`` pins the two together).

Because lane ``c`` only ever reads ``P[j, c]`` — a value **it** produced — the
column-parallel decomposition needs no cross-lane communication and no barrier
beyond the one after the LDS fill. The ``i``/``j`` loops are unrolled at build
time so ``P[j]`` is a register indexed by a compile-time constant; that is
``C(C−1)/2 = 2016`` fused multiply-adds of straight-line code at ``C = 64``,
which is the same order as the score kernel's unrolled contraction.

Scope: fp32, ``C = 64`` — the production chunk size, and the only one where
this stage costs anything. :func:`..ops.ut_inverse` falls back to the doubling
for every other dtype and width, which is also what keeps the fp64 gradient
test on the torch path.
"""

from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import math as math_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, range_constexpr, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1._lld_shim import (
    ensure_usable_lld,
)
from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1._stream import (
    with_current_stream,
)

_LLVM_GEP_DYNAMIC = -2147483648  # LLVM kDynamicIndex sentinel

# Widths the unrolled substitution is built for. 64 is Kimi K3's chunk size;
# 32 and 16 are the unit-test widths and cost nothing to support.
SUPPORTED_C = (16, 32, 64)

__all__ = ["build_kda_ut_inverse", "supports_ut_geometry", "SUPPORTED_C"]


def supports_ut_geometry(chunk_size: int) -> Optional[str]:
    """``None`` when the kernel can run this width, else why it cannot."""
    if int(chunk_size) not in SUPPORTED_C:
        return f"chunk_size={chunk_size} is not one of {list(SUPPORTED_C)}"
    return None


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def build_kda_ut_inverse(chunk_size: int, waves_per_eu: int = 2):
    """Build the launcher for one width.

    Returns ``launch(Low, P, nb)`` over flat fp32 tensors, both ``[nb, C, C]``
    row-major. ``Low`` is read as strictly lower triangular: the kernel does not
    mask it, it relies on ``L[i, j] = 0`` for ``j ≥ i``, which is what
    ``Akk``'s own mask guarantees (``wp9_opt/RESULTS.md`` §2 checks it is
    exactly ``0.0``). The diagonal and upper triangle of the result are ``I``
    and ``0`` respectively, both written.
    """
    ensure_usable_lld()
    arch = get_rocm_arch()
    if not arch.startswith("gfx9"):
        raise RuntimeError(f"kda_ut_inverse targets CDNA (gfx9); got {arch!r}")
    reason = supports_ut_geometry(chunk_size)
    if reason is not None:
        raise ValueError(f"the FlyDSL UT-inverse kernel cannot run this width: {reason}")

    C = int(chunk_size)
    BLOCK = C  # one thread per column of P

    allocator = SmemAllocator(None, arch=arch, global_sym_name=f"kda_ut_smem_C{C}")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + C * C * 4

    @flyc.kernel(known_block_size=[BLOCK, 1, 1], name=f"kda_ut_inverse_C{C}")
    def kda_ut_inverse_kernel(Low: fx.Tensor, P: fx.Tensor):
        f32 = T.f32
        vec1_f32 = T.vec(1, f32)

        l_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Low)
        p_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), P)

        base = allocator.get_base()
        lds = SmemPtr(base, lds_off, f32, shape=(C * C,)).get()

        def gep(bptr, elem_idx):
            return _llvm.GEPOp(
                _llvm_ptr_ty(),
                bptr,
                [arith.index_cast(T.i64, elem_idx)],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=f32,
                noWrapFlags=0,
            ).result

        def load_f32(bptr, elem_idx):
            return _llvm.LoadOp(f32, gep(bptr, elem_idx)).result

        def store_f32(val, bptr, elem_idx):
            _llvm.StoreOp(val, gep(bptr, elem_idx))

        def lds_get(elem_idx):
            v = vector.load_op(vec1_f32, lds, [elem_idx])
            return vector.extract(v, static_position=[0], dynamic_position=[])

        def lds_put(elem_idx, val):
            vector.store(vector.from_elements(vec1_f32, [val]), lds, [elem_idx])

        bid = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)
        chunk0 = bid * arith.index(C * C)

        # Cooperative fill: thread `c` takes column `c` of every row, so each
        # row is one coalesced 4C-byte transaction.
        for i in range_constexpr(C):
            off = arith.index(i * C) + tid
            lds_put(off, load_f32(l_ptr, chunk0 + off))
        gpu.barrier()

        c_one = arith.constant(1.0, type=f32)
        c_zero = arith.constant(0.0, type=f32)

        # P[i, c] = delta(i, c) + sum_{j<i} L[i, j] P[j, c].
        # `preg[j]` is lane `c`'s own P[j, c]; nothing crosses lanes.
        preg = []
        for i in range_constexpr(C):
            acc = arith.select(arith.cmpi(arith.CmpIPredicate.eq, tid, arith.index(i)), c_one, c_zero)
            for j in range_constexpr(i):
                acc = math_dialect.fma(lds_get(arith.index(i * C + j)), preg[j], acc)
            preg.append(acc)
            store_f32(acc, p_ptr, chunk0 + arith.index(i * C) + tid)

    @flyc.jit
    def launch_kda_ut_inverse(
        Low: fx.Tensor,
        P: fx.Tensor,
        nb: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        grid_x = arith.index_cast(T.index, nb)
        launcher = kda_ut_inverse_kernel(Low, P)
        for op in ctx.gpu_module_body.operations:
            if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, int(waves_per_eu))
                op.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get(f"{BLOCK},{BLOCK}")
        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    _hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_hints):
            return launch_kda_ut_inverse(*args, **kwargs)

    return with_current_stream(_launch)
