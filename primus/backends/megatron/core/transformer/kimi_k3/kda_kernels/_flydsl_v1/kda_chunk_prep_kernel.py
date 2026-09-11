###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native FlyDSL kernel for the chunk's decay-weighted operand preparation.

Profiling took the production forward down to 2749 µs on device and found that
**~1350 µs of it — half — was torch elementwise glue that `fla` never performs**
The largest connected piece of that glue is the
decay chain: six ops that between them read and write ~3.3 GB of HBM to produce
three tensors from three, with one ``exp`` of real arithmetic per element.

===========================================  =============================
torch, six launches                          here, one
===========================================  =============================
``Γ  = cg.exp()``                            ``Γ = exp(cg)`` in a register
``QΓ = qf * Γ``                              stored straight into ``qw``'s
                                             top half at the operand dtype
``KΓ = kf * Γ``                              stored fp32
``E  = (chunk_total − cg).exp()``            ``E = exp(ct − cg)`` in a register
``KG = kf * E``                              stored at the operand dtype
``dec = chunk_total.exp()``                  stored fp32
===========================================  =============================

Reads ``qf``, ``kf``, ``cg`` once each and writes four tensors: ~1.0 GB against
~3.3 GB, and one launch against six (plus the ``qw`` slice copy the top-half
store absorbs, and half of ``_transposed_operand``'s traffic, because ``KG``
now leaves here already at the operand dtype).

Geometry
--------
One workgroup of 256 threads per chunk, thread ``t`` owning channel
``d = t % K`` for the rows ``t // K, t // K + BLOCK/K, …``. Every global access
is contiguous in ``d``, so every load and store is a full coalesced
transaction, and no LDS is needed at all: the only value shared across rows is
``ct[d] = cg[C−1, d]``, which each thread reads once because its ``d`` is
fixed.

``KG`` is deliberately **not** transposed here. The sweep wants it as
``[NB, K, C]`` and doing that in-kernel would need an LDS stage with a padded
row stride; leaving it to :func:`..sweep._transposed_operand` costs one launch
and a 200 MB bf16→bf16 copy, against 300 MB before, and keeps this kernel a
pure coalesced map. The backward wants the **un**-transposed layout anyway.

There is one deliberate benign race: the ``BLOCK/K`` threads that share a
channel ``d`` all store the same ``dec[d] = exp(ct[d])``, computed from the
same input by the same instruction, so the store is a duplicate of identical
bytes rather than a conflict. Predicating it would cost a branch to save
nothing.

gfx950 / CDNA4.
"""

import math
from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1._lld_shim import (
    ensure_usable_lld,
)
from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1._stream import (
    with_current_stream,
)

_LOG2E = math.log2(math.e)  # exp(x) == exp2(x * log2(e)); only exp2 is exposed
_LLVM_GEP_DYNAMIC = -2147483648  # LLVM kDynamicIndex sentinel

BLOCK_SIZE = 256

__all__ = ["build_kda_chunk_prep", "supports_prep_geometry", "BLOCK_SIZE"]


def supports_prep_geometry(chunk_size: int, k_dim: int) -> Optional[str]:
    """``None`` when the kernel can run this geometry, else why it cannot."""
    c, kd = int(chunk_size), int(k_dim)
    if kd <= 0 or kd > BLOCK_SIZE or BLOCK_SIZE % kd != 0:
        return f"head_dim={kd} does not divide the {BLOCK_SIZE}-thread block"
    rstep = BLOCK_SIZE // kd
    if c % rstep != 0:
        return f"chunk_size={c} is not a multiple of the {rstep} rows filled at once"
    return None


def _nat_exp(x, log2e_const, fastmath):
    """``exp(x)`` via the hardware ``v_exp_f32``; see the score kernel's copy.

    Module scope because the AST rewriter cannot resolve ``rocdl`` from a helper
    defined inside a traced body, and ``rocdl.exp2`` rather than
    ``ArithValue.exp2()`` because the latter lowers to an external
    ``__ocml_exp2_f32`` and drags the device-bitcode link into every compile.
    """
    scaled = arith.MulFOp(x, log2e_const, fastmath=fastmath).result
    return rocdl.exp2(T.f32, scaled)


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def build_kda_chunk_prep(chunk_size: int, k_dim: int, op_bf16: bool = True, waves_per_eu: int = 2):
    """Build the launcher for one ``(chunk_size, head_dim, operand dtype)``.

    Returns ``launch(Qf, Kf, CG, QW, KGam, KG, Dec, nb)`` over flat tensors:

    ========  ==================  ==============================================
    ``Qf``    ``[NB, C, K]`` f32  already scaled by ``softmax_scale``
    ``Kf``    ``[NB, C, K]`` f32
    ``CG``    ``[NB, C, K]`` f32  within-chunk cumulative log-decay, ``≤ 0``
    ``QW``    ``[NB, 2C, K]`` op  **only rows ``[0, C)`` are written** — the
                                  bottom half is the caller's ``W = M(Γ⊙K)``
    ``KGam``  ``[NB, C, K]`` f32  ``Γ ⊙ K``, an fp32 GEMM operand
    ``KG``    ``[NB, C, K]`` op   ``K ⊙ exp(ct − cg)``, not transposed
    ``Dec``   ``[NB, K]`` f32     ``exp(ct)``
    ========  ==================  ==============================================
    """
    ensure_usable_lld()
    arch = get_rocm_arch()
    if not arch.startswith("gfx9"):
        raise RuntimeError(f"kda_chunk_prep targets CDNA (gfx9); got {arch!r}")
    reason = supports_prep_geometry(chunk_size, k_dim)
    if reason is not None:
        raise ValueError(f"the FlyDSL chunk-prep kernel cannot run this geometry: {reason}")

    C, KD = int(chunk_size), int(k_dim)
    RSTEP = BLOCK_SIZE // KD  # rows filled per iteration
    NITER = C // RSTEP
    op_name = "bf16" if op_bf16 else "f32"

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1], name=f"kda_chunk_prep_{op_name}")
    def kda_chunk_prep_kernel(
        Qf: fx.Tensor,
        Kf: fx.Tensor,
        CG: fx.Tensor,
        QW: fx.Tensor,
        KGam: fx.Tensor,
        KG: fx.Tensor,
        Dec: fx.Tensor,
    ):
        f32 = T.f32
        op_t = T.bf16 if op_bf16 else T.f32
        fm = arith.FastMathFlags.fast

        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Qf)
        k_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Kf)
        cg_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), CG)
        qw_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), QW)
        kgam_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), KGam)
        kg_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), KG)
        dec_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Dec)

        def gep(bptr, elem_idx, elem_ty):
            return _llvm.GEPOp(
                _llvm_ptr_ty(),
                bptr,
                [arith.index_cast(T.i64, elem_idx)],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=elem_ty,
                noWrapFlags=0,
            ).result

        def load_f32(bptr, elem_idx):
            return _llvm.LoadOp(f32, gep(bptr, elem_idx, f32)).result

        def store_f32(val, bptr, elem_idx):
            _llvm.StoreOp(val, gep(bptr, elem_idx, f32))

        # Build-time choice by dict lookup rather than an `if` in the traced
        # body, the discipline the sweep kernel's docstring records.
        def _store_bf16(val, bptr, elem_idx):
            _llvm.StoreOp(arith.trunc_f(op_t, val), gep(bptr, elem_idx, op_t))

        store_op = {True: _store_bf16, False: store_f32}[bool(op_bf16)]

        c_log2e = arith.constant(_LOG2E, type=f32)

        def nat_exp(x):
            return _nat_exp(x, c_log2e, fm)

        bid = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)
        I_KD = arith.index(KD)

        d = tid % I_KD
        r0 = tid // I_KD
        base = bid * arith.index(C * KD)
        qw_base = bid * arith.index(2 * C * KD)

        # ct[d] = cg[C-1, d]: the chunk's total log-decay in this channel. Fixed
        # per thread, so it is read once for the whole row loop.
        ct = load_f32(cg_ptr, base + arith.index((C - 1) * KD) + d)
        store_f32(nat_exp(ct), dec_ptr, bid * I_KD + d)

        for i in range_constexpr(NITER):
            row = r0 + arith.index(i * RSTEP)
            off = row * I_KD + d
            cgv = load_f32(cg_ptr, base + off)
            gam = nat_exp(cgv)
            e_fac = nat_exp(arith.SubFOp(ct, cgv, fastmath=fm).result)
            qv = load_f32(q_ptr, base + off)
            kv = load_f32(k_ptr, base + off)
            store_op(arith.MulFOp(qv, gam, fastmath=fm).result, qw_ptr, qw_base + off)
            store_f32(arith.MulFOp(kv, gam, fastmath=fm).result, kgam_ptr, base + off)
            store_op(arith.MulFOp(kv, e_fac, fastmath=fm).result, kg_ptr, base + off)

    @flyc.jit
    def launch_kda_chunk_prep(
        Qf: fx.Tensor,
        Kf: fx.Tensor,
        CG: fx.Tensor,
        QW: fx.Tensor,
        KGam: fx.Tensor,
        KG: fx.Tensor,
        Dec: fx.Tensor,
        nb: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        grid_x = arith.index_cast(T.index, nb)
        launcher = kda_chunk_prep_kernel(Qf, Kf, CG, QW, KGam, KG, Dec)
        for op in ctx.gpu_module_body.operations:
            if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, int(waves_per_eu))
                op.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get(f"{BLOCK_SIZE},{BLOCK_SIZE}")
        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    _hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_hints):
            return launch_kda_chunk_prep(*args, **kwargs)

    return with_current_stream(_launch)
