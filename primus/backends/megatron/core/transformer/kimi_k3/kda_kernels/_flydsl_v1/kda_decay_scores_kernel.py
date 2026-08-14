###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native FlyDSL kernel for KDA's two intra-chunk decay-weighted score matrices.

This is the stage the FlyDSL backend exists for. Per chunk of ``C`` steps,
with ``cg`` the within-chunk cumulative log-decay
(``cg[r, d] = Σ_{i≤r} g[i, d] ≤ 0``), it computes

    Aqk[r, c] = Σ_d q[r, d] · exp(cg[r, d] − cg[c, d]) · k[c, d]   for c ≤ r
    Akk[r, c] = Σ_d k[r, d] · exp(cg[r, d] − cg[c, d]) · k[c, d]   for c < r

and zero above those diagonals. Both share the ``k`` operand and the decay
factors, so one pass produces both.

Why a kernel
------------
The eager reference (:mod:`..._eager.reference`) builds these matrices one
column at a time, which is ``C`` launches and keeps a ``[B, H, NC, C, C, K]``
intermediate alive through the backward pass. The two-matmul form the
published algorithm uses instead —
``A = Tril[(Q ⊙ Γ)(K / Γ)ᵀ]`` — is unusable as written: with
``gate_lower_bound = -5`` and ``C = 64`` the ``1/Γ`` factor reaches
``exp(320)``, which overflows fp32 (max ``≈ exp(88.7)``), so a direct
port produces ``inf`` and then ``nan``.

How the overflow is avoided
---------------------------
Both matrices are assembled from ``[SB, SB]`` blocks with ``SB = 16``, and
each block references the cumulative log-decay to a row chosen so that no
representable range is exceeded. For a block at row-block ``i``,
column-block ``j``, with reference row ``n``::

    A[r, c] = Σ_d (q[r,d]·exp(cg[r,d] − cg[n,d])) · (k[c,d]·exp(cg[n,d] − cg[c,d]))

which is exact for any ``n``, so ``n`` is free to be chosen for numerics:

* ``j < i`` — ``n`` is the **first row of row-block i**. ``cg`` is
  non-increasing in the row index, so both exponents are ``≤ 0`` and both
  factors lie in ``(0, 1]``. Nothing can overflow.
* ``j == i`` — ``n`` is the block **midpoint**. The exponents are then
  sign-indefinite but bounded by ``±(SB/2)·|g|_max = ±40``, and the
  discarded above-diagonal entries reach at most ``exp((SB−1)·5) =
  exp(75) ≈ 3.7e32``, inside fp32's ``3.4e38``. This is the equivalent of
  the secondary 16-tile ``fla``'s ``safe_gate`` path uses
  (``fla/ops/kda/chunk_intra.py``, the ``SAFE_GATE`` branches).

``SB = 16`` is the largest sub-block with a *provable* margin: ``SB = 32``
would permit ``exp(155)`` and only survives by luck of the data.

Layout and geometry
-------------------
Every tensor is passed flat. ``NB = B · H · NC`` chunks are independent, so
the grid is one workgroup per chunk and the kernel needs no cross-workgroup
communication.

* ``Q``, ``K``, ``CG``: ``[NB, C, K_DIM]`` fp32
* ``Aqk``, ``Akk``: ``[NB, C, C]`` fp32

``BLOCK_SIZE`` is fixed at 256 threads, which makes the compute phase
exactly one output element per thread for ``SB = 16`` and the fill phase a
whole-row-per-``K_DIM``-slice mapping. That requires ``256 % K_DIM == 0``,
i.e. ``K_DIM ∈ {32, 64, 128, 256}``; the launcher checks it and the Python
caller falls back to another backend otherwise.

gfx950 / CDNA4. This first version is a VALU (plain-FMA) kernel rather than
an MFMA one: it removes the ``C``-launch column loop and the quadratic
backward intermediate, which are the documented costs, while keeping the
data layout simple enough to be auditable against the eager reference.
Moving the ``[16, K] × [K, 16]`` contraction onto ``mfma_f32_16x16x32_bf16``
is the obvious next step.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import math as math_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1._lld_shim import (
    ensure_usable_lld,
)
from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1._stream import (
    with_current_stream,
)

_LOG2E = math.log2(math.e)  # exp(x) == exp2(x * log2(e)); only exp2 is exposed
_LLVM_GEP_DYNAMIC = -2147483648  # LLVM kDynamicIndex sentinel

BLOCK_SIZE = 256
SUB_BLOCK = 16
SUPPORTED_K = (32, 64, 128, 256)
# Channels per LDS access in the contraction. 4 is one `ds_read_b128`, the widest
# the ISA has; every supported `K_DIM` is a multiple of it, and the summation
# order is unchanged (the FMAs are still emitted in ascending `d`), so the output
# is bit-identical to reading one channel at a time.
LDS_RUN = 4

__all__ = ["build_kda_decay_scores", "BLOCK_SIZE", "SUB_BLOCK", "SUPPORTED_K"]


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def lds_runs(count: int):
    """``[(offset, width), ...]`` covering ``count`` fp32 with the widest accesses.

    Resolved here, at **build** scope. The AST rewriter turns a ``while`` in a
    traced body into an ``scf.while`` whose condition is a Python ``bool``
    (``AttributeError: 'bool' object has no attribute 'owner'``), and a plain
    ``range`` into a traced loop whose induction variable cannot index a Python
    list — so the body may only walk a ready-made list with ``range_constexpr``.
    """
    out, i = [], 0
    while i < count:
        width = 4 if count - i >= 4 else (2 if count - i >= 2 else 1)
        out.append((i, width))
        i += width
    return out


def _nat_exp(x, log2e_const, fastmath):
    """``exp(x)`` via the hardware ``v_exp_f32``.

    Two things about this helper are load-bearing.

    It must live at **module** scope, not inside the ``@flyc.kernel`` body: the
    AST rewriter does not make ``rocdl`` resolvable from a function defined
    inside a traced body, and doing so raises ``NameError: name 'rocdl' is not
    defined`` at trace time. Helpers reaching for a dialect module therefore go
    outside the body and are simply *called* from it.

    And it uses ``rocdl.exp2`` rather than ``arith.ArithValue(x).exp2()``,
    because the latter lowers to an external ``__ocml_exp2_f32`` call and so
    drags the ROCm device-bitcode link into every compile. Both were measured
    bit-identical to ``torch.exp2`` on this hardware.
    """
    scaled = arith.MulFOp(x, log2e_const, fastmath=fastmath).result
    return rocdl.exp2(T.f32, scaled)


def build_kda_decay_scores(chunk_size: int, k_dim: int, waves_per_eu: int = 2):
    """Build the launcher for one ``(chunk_size, k_dim)`` geometry.

    Returns a callable
    ``launch(Q, K, CG, Aqk, Akk, num_chunks)`` taking flat fp32 tensors.
    """
    ensure_usable_lld()
    arch = get_rocm_arch()
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"kda_decay_scores targets gfx950 (CDNA4); got {arch!r}")

    C = int(chunk_size)
    KD = int(k_dim)
    SB = SUB_BLOCK
    if C % SB != 0:
        raise ValueError(f"chunk_size ({C}) must be a multiple of the sub-block {SB}")
    if KD not in SUPPORTED_K:
        raise ValueError(f"k_dim ({KD}) must be one of {list(SUPPORTED_K)}")
    NSB = C // SB
    if BLOCK_SIZE != SB * SB:
        raise ValueError("the compute phase assumes one output element per thread")

    # Fill-phase decomposition: thread -> (channel, row-slice).
    ROW_STEP = BLOCK_SIZE // KD  # rows filled concurrently
    ROWS_PER_THREAD = SB // ROW_STEP
    if ROW_STEP == 0 or SB % ROW_STEP != 0:
        raise ValueError(f"cannot map {BLOCK_SIZE} threads onto [{SB}, {KD}] tiles")

    # Padding breaks the LDS bank conflict a bare K_DIM stride would create: the
    # compute phase reads column `d` of 16 different rows at once, and
    # 128 % 32 == 0 would put all 16 in the same bank. +4 keeps every row start
    # 4-element aligned, so a later vectorised read stays legal.
    STRIDE = KD + 4
    RUNS = lds_runs(LDS_RUN)
    # (row-block, col-block) pairs of the lower triangle, in the order emitted.
    PAIRS = [(i, j) for i in range(NSB) for j in range(i + 1)]
    ZERO_PAIRS = [(i, j) for i in range(NSB) for j in range(i + 1, NSB)]

    allocator = SmemAllocator(None, arch=arch, global_sym_name=f"kda_scores_smem_C{C}_K{KD}")
    lds_lq_off = allocator._align(allocator.ptr, 16)
    lds_lk_off = allocator._align(lds_lq_off + SB * STRIDE * 4, 16)
    lds_rg_off = allocator._align(lds_lk_off + SB * STRIDE * 4, 16)
    allocator.ptr = lds_rg_off + SB * STRIDE * 4

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def kda_decay_scores_kernel(
        Q: fx.Tensor,  # [NB, C, KD] fp32 flat
        Kk: fx.Tensor,  # [NB, C, KD] fp32 flat
        CG: fx.Tensor,  # [NB, C, KD] fp32 flat
        Aqk: fx.Tensor,  # [NB, C, C] fp32 flat
        Akk: fx.Tensor,  # [NB, C, C] fp32 flat
    ):
        f32 = T.f32
        fm_fast = arith.FastMathFlags.fast
        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        k_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Kk)
        cg_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), CG)
        aqk_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Aqk)
        akk_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Akk)

        base_ptr = allocator.get_base()
        lds_lq = SmemPtr(base_ptr, lds_lq_off, f32, shape=(SB * STRIDE,)).get()
        lds_lk = SmemPtr(base_ptr, lds_lk_off, f32, shape=(SB * STRIDE,)).get()
        lds_rg = SmemPtr(base_ptr, lds_rg_off, f32, shape=(SB * STRIDE,)).get()

        chunk = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)

        c_zero_f = arith.constant(0.0, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)
        KD_IDX = arith.index(KD)
        C_IDX = arith.index(C)
        row_base = chunk * C_IDX  # first row of this chunk, in rows
        vec1_f32 = T.vec(1, f32)

        def _gep(bptr, elem_idx):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            return _llvm.GEPOp(
                _llvm_ptr_ty(),
                bptr,
                [idx_i64],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=f32,
                noWrapFlags=0,
            ).result

        def g_load(bptr, elem_idx):
            return _llvm.LoadOp(f32, _gep(bptr, elem_idx)).result

        def g_store(val, bptr, elem_idx):
            _llvm.StoreOp(val, _gep(bptr, elem_idx))

        def lds_read(lds, elem_idx):
            vec = vector.load_op(vec1_f32, lds, [elem_idx])
            return vector.extract(vec, static_position=[0], dynamic_position=[])

        def lds_read_run(lds, base_idx, runs):
            """Consecutive fp32 from as few LDS accesses as possible.

            This kernel is LDS-bandwidth bound: its one-output-element-per-thread
            mapping reads three LDS words per two FMAs, which is 24 GB of LDS
            traffic per call at production geometry and 310 µs at the hardware's
            78 TB/s — against 294 µs measured. Issuing the contraction four
            channels at a time turns twelve ``ds_read_b32`` into three
            ``ds_read_b128``. ``STRIDE = KD + 4`` keeps every row start 16-byte
            aligned and the base steps by the run width, so the wide access is
            legal. ``runs`` comes from :func:`lds_runs` at build scope.
            """
            out = []
            for ri in range_constexpr(len(runs)):
                off, width = runs[ri]
                vec = vector.load_op(T.vec(width, f32), lds, [base_idx + arith.index(off)])
                for j in range_constexpr(width):
                    out.append(vector.extract(vec, static_position=[j], dynamic_position=[]))
            return out

        def lds_write(lds, elem_idx, val):
            vector.store(vector.from_elements(vec1_f32, [val]), lds, [elem_idx])

        def nat_exp(x):
            return _nat_exp(x, c_log2e, fm_fast)

        # ---- fill-phase indices ----
        fill_d = tid % KD_IDX
        fill_rsub = tid // KD_IDX
        # ---- compute-phase indices: one [SB, SB] output element per thread ----
        out_r = tid // arith.index(SB)
        out_c = tid % arith.index(SB)

        def row_elem(row_in_chunk, d):
            """Flat element index of ``[chunk, row_in_chunk, d]`` in a [NB, C, KD]."""
            return (row_base + row_in_chunk) * KD_IDX + d

        def out_elem(row_in_chunk, col_in_chunk):
            """Flat element index of ``[chunk, row, col]`` in a [NB, C, C]."""
            return (row_base + row_in_chunk) * C_IDX + col_in_chunk

        NUM_PAIRS = len(PAIRS)
        for pair_idx in range_constexpr(NUM_PAIRS):
            i_blk, j_blk = PAIRS[pair_idx]
            is_diag = i_blk == j_blk
            # reference row: sub-block midpoint on the diagonal (bounded, signed
            # exponents), first row of the row-block off it (all exponents <= 0)
            ref_row = i_blk * SB + (SB // 2) if is_diag else i_blk * SB
            ref_idx = arith.index(ref_row)
            l_row0 = arith.index(i_blk * SB)
            r_row0 = arith.index(j_blk * SB)

            cg_ref = g_load(cg_ptr, row_elem(ref_idx, fill_d))
            for m in range_constexpr(ROWS_PER_THREAD):
                rr = fill_rsub + arith.index(m * ROW_STEP)
                lds_off = rr * arith.index(STRIDE) + fill_d

                # left operand rows: q and k share one decay factor
                l_elem = row_elem(l_row0 + rr, fill_d)
                l_fac = nat_exp(arith.SubFOp(g_load(cg_ptr, l_elem), cg_ref, fastmath=fm_fast).result)
                lds_write(
                    lds_lq, lds_off, arith.MulFOp(g_load(q_ptr, l_elem), l_fac, fastmath=fm_fast).result
                )
                lds_write(
                    lds_lk, lds_off, arith.MulFOp(g_load(k_ptr, l_elem), l_fac, fastmath=fm_fast).result
                )

                # right operand rows: k with the reciprocal decay factor
                r_elem = row_elem(r_row0 + rr, fill_d)
                r_fac = nat_exp(arith.SubFOp(cg_ref, g_load(cg_ptr, r_elem), fastmath=fm_fast).result)
                lds_write(
                    lds_rg, lds_off, arith.MulFOp(g_load(k_ptr, r_elem), r_fac, fastmath=fm_fast).result
                )
            gpu.barrier()

            # The contraction is unrolled at compile time rather than run as an
            # `scf.for_`: `K_DIM` is a build-time constant, and an unrolled chain
            # of `fma` keeps the kernel body a plain function. A traced
            # `scf.for_` needs a `yield` in the kernel body, which would make the
            # body a Python generator function and interacts badly with the
            # enclosing constexpr pair loop.
            l_off = out_r * arith.index(STRIDE)
            r_off = out_c * arith.index(STRIDE)
            val_qk = c_zero_f
            val_kk = c_zero_f
            for d_i in range_constexpr(KD // LDS_RUN):
                d_idx = arith.index(d_i * LDS_RUN)
                rg = lds_read_run(lds_rg, r_off + d_idx, RUNS)
                lq = lds_read_run(lds_lq, l_off + d_idx, RUNS)
                lk = lds_read_run(lds_lk, l_off + d_idx, RUNS)
                for w in range_constexpr(LDS_RUN):
                    val_qk = math_dialect.fma(lq[w], rg[w], val_qk)
                    val_kk = math_dialect.fma(lk[w], rg[w], val_kk)

            if is_diag:
                # o_t reads the POST-update state, so Aqk keeps its diagonal;
                # Akk is the strictly-lower delta-correction matrix.
                keep_qk = arith.cmpi(arith.CmpIPredicate.sle, out_c, out_r)
                keep_kk = arith.cmpi(arith.CmpIPredicate.slt, out_c, out_r)
                val_qk = arith.select(keep_qk, val_qk, c_zero_f)
                val_kk = arith.select(keep_kk, val_kk, c_zero_f)

            elem = out_elem(l_row0 + out_r, r_row0 + out_c)
            g_store(val_qk, aqk_ptr, elem)
            g_store(val_kk, akk_ptr, elem)
            gpu.barrier()

        # Blocks strictly above the diagonal are never contracted; write their
        # zeros here so the kernel fully defines its output and the caller can
        # allocate with `empty` rather than `zeros`.
        for zp in range_constexpr(len(ZERO_PAIRS)):
            i_blk, j_blk = ZERO_PAIRS[zp]
            elem = out_elem(arith.index(i_blk * SB) + out_r, arith.index(j_blk * SB) + out_c)
            g_store(c_zero_f, aqk_ptr, elem)
            g_store(c_zero_f, akk_ptr, elem)

    @flyc.jit
    def launch_kda_decay_scores(
        Q: fx.Tensor,
        Kk: fx.Tensor,
        CG: fx.Tensor,
        Aqk: fx.Tensor,
        Akk: fx.Tensor,
        num_chunks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        grid_x = arith.index_cast(T.index, num_chunks)
        launcher = kda_decay_scores_kernel(Q, Kk, CG, Aqk, Akk)

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
            return launch_kda_decay_scores(*args, **kwargs)

    return with_current_stream(_launch)
