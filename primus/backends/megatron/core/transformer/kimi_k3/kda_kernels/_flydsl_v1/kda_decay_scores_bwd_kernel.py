###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native FlyDSL kernel for the **adjoint** of KDA's two score matrices.

Profiling measured :meth:`..ops._DecayScores.backward` at **12871 µs of a
23164 µs backward — 56 %**, because it recomputed
:func:`..ops.decay_scores_torch` under ``enable_grad`` and differentiated it:
~40 elementwise ops and batched GEMMs on 100 MB tensors, doubled by autograd,
for ~8 GFLOP of real arithmetic. This kernel replaces all of it with one launch.

The adjoint, and why it is the same kernel family
-------------------------------------------------
With ``E[r,c,d] = exp(cg[r,d] − cg[c,d])`` the forward is

    Aqk[r,c] = Σ_d q[r,d]·E[r,c,d]·k[c,d]     (c ≤ r, else 0)
    Akk[r,c] = Σ_d k[r,d]·E[r,c,d]·k[c,d]     (c < r, else 0)

so with the upstream gradients masked to those same triangles — the masked-off
entries are outputs that are *identically* zero, so they carry no gradient —
four contractions fall out, and each is the forward's own blocked
decay-weighted contraction with **the contraction axis swapped**:

===============  ==================================================  =========
quantity         definition                                          contracts
===============  ==================================================  =========
``dq[r,d]``      ``Σ_c dAqk[r,c]·E[r,c,d]·k[c,d]``                    over ``c``
``A2[r,d]``      ``Σ_c dAkk[r,c]·E[r,c,d]·k[c,d]``                    over ``c``
``A1[c,d]``      ``Σ_r dAqk[r,c]·E[r,c,d]·q[r,d]``                    over ``r``
``A3[c,d]``      ``Σ_r dAkk[r,c]·E[r,c,d]·k[r,d]``                    over ``r``
===============  ==================================================  =========

and then, elementwise and free,

    dk  = A1 + A2 + A3
    dcg = q·dq + k·(A2 − A1 − A3)

``dq`` and ``A2`` share their ``k·E`` operand; ``A1`` and ``A3`` share the
transposed ``dA`` access. So one workgroup produces all four.

The ``1/Γ`` guard, re-derived for the swapped axis
--------------------------------------------------
Every ``[SB, SB]`` block still factors ``E[r,c,d]`` through a reference row
``n``, ``E = exp(cg[r,d]−cg[n,d])·exp(cg[n,d]−cg[c,d]) = lf[r,d]·rf[c,d]``, and
``n`` is still chosen so that **both factors are ≤ 1** off the diagonal:

* the ``Σ_c`` direction (``dq``, ``A2``) owns row-block ``b`` and contracts all
  earlier columns at once, so ``n`` is the **first row of block b**: rows ``r``
  are ``≥ n`` and columns ``c`` are ``< n``, hence ``lf ≤ 1`` and ``rf ≤ 1``.
  This is exactly the forward's choice.
* the ``Σ_r`` direction (``A1``, ``A3``) owns *column*-block ``b`` and contracts
  all later rows, so the forward's choice would put ``rf = exp(cg[n]−cg[c])``
  with ``c > n`` and let it reach ``exp(75)``. ``n`` is therefore the **last row
  of block b**: columns ``c`` are ``≤ n`` and rows ``r`` are ``> n``, so again
  both factors are ``≤ 1``.
* the diagonal block keeps the forward's **midpoint** reference, so
  ``|exponent| ≤ (SB/2)·|g|_max = 40``.

Both factors ``≤ 1`` also means no *gratuitous* underflow: the product is the
true value, so whenever a factor underflows to zero the true entry is itself
below ``fp32``'s smallest normal.

On the diagonal the largest intermediate is
``SB · |dA|·|k| · exp(40) ≈ 3.8e18 · |dA|·|k|``, because the upstream gradient
is masked *before* it is contracted and so the above-diagonal ``lf·rf`` products
that force the forward's ``exp(75) = 3.7e32`` bound are never formed at all.
**The margin at ``SUB_BLOCK = 16`` is therefore wider here than in the
forward**, and 32 would still be the first unprovable size.
``test_kda_flydsl_scores_bwd_kernel_survives_a_saturated_gate`` pins it with
``g = −5`` everywhere.

Layout and geometry
-------------------
One workgroup per chunk, ``NB = B·H·NC`` of them, and a compile-time loop over
the ``NSB = C/SB`` row/column blocks. Block ``b`` is visited exactly once, as
the row owner for the ``b`` pairs to its left, as the column owner for the
``NSB−1−b`` pairs below it, and once on the diagonal — ``NSB+1`` sub-block
contractions whatever ``b`` is, so the compile-time schedule is balanced by
construction. All four accumulators for block ``b`` live in **registers**
(``MR·NR`` elements each), which is why the owner is a whole block rather than
a pair: nothing has to be re-read or atomically combined.

* ``Q``, ``K``, ``CG``, ``DQ``, ``DK``, ``DCG``: ``[NB, C, K_DIM]`` fp32
* ``DAqk``, ``DAkk``: ``[NB, C, C]`` fp32

fp32 VALU throughout, like the forward kernel and for the same two reasons:
``v_mfma_f32_16x16x4f32`` SIGABRTs in this ``flydsl`` build, and a bf16 MFMA path
would round ``dA`` and ``k`` at the point they
are consumed, which the gradients' 11× tolerance margin could absorb but which
has to be *measured* before it is shipped.

Two tracing rules shape the code below.

* Every build-time choice is made by indexing a dict of closures rather than by
  an ``if`` statement: the AST rewriter routes every
  ``if`` through ``scf_if_dispatch``.
* **Every** ``for`` in the body iterates ``range_constexpr``, including the
  two-deep ``MR``/``NR`` register loops. ``range_constexpr`` *is* ``range``
  (``flydsl/expr/__init__.py``) — it is a marker the rewriter matches by name
  (``ast_rewriter.py`` ``_is_range_constexpr``), and a plain ``range`` loop
  becomes a traced loop whose induction variable is an ``ArithValue``, which
  then cannot index the Python list an accumulator lives in.

gfx950 / CDNA4.
"""

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
from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.kda_decay_scores_kernel import (
    _LLVM_GEP_DYNAMIC,
    _LOG2E,
    BLOCK_SIZE,
    SUB_BLOCK,
    SUPPORTED_K,
    _llvm_ptr_ty,
    _nat_exp,
    lds_runs,
)

# `d` threads per row group, which fixes the accumulator tile: `TR = 256/TD`
# threads down the rows, `MR = SB/TR` rows and `NR = KD/TD` channels per thread.
# Swept at production geometry over `TD in {16, 32, 64, 128}` x
# `waves_per_eu in {1, 2, 4}` (`bench/tune_kernels.py`, every setting bit-equal):
#
#     TD          16     32     64    128
#     best us   1705   1698   1801   2992
#
# 32 wins, so the shipped value does not change; every setting is bit-equal.
# What the sweep rules out is worth as much as what it picks: 16 through 64 land
# within 6 % of each other, so **register pressure and occupancy are not this
# kernel's problem** — the one large effect is 128, which collapses to `NR = 1`
# and loses the wide LDS access below entirely (1.8x). `waves_per_eu` moved from
# 2 to 4 for 2 %.
THREADS_D = 32

__all__ = ["build_kda_decay_scores_bwd", "supports_bwd_geometry", "BLOCK_SIZE", "SUB_BLOCK"]


def supports_bwd_geometry(chunk_size: int, k_dim: int):
    """``None`` when the kernel can run this geometry, else why it cannot."""
    if chunk_size % SUB_BLOCK != 0:
        return f"chunk_size={chunk_size} is not a multiple of the {SUB_BLOCK}-row sub-block"
    if k_dim not in SUPPORTED_K:
        return f"head_dim={k_dim} is not one of {list(SUPPORTED_K)}"
    if BLOCK_SIZE != SUB_BLOCK * SUB_BLOCK:
        return f"the [{SUB_BLOCK},{SUB_BLOCK}] gradient tile needs {SUB_BLOCK ** 2} threads"
    td = min(THREADS_D, k_dim)
    tr = BLOCK_SIZE // td
    if SUB_BLOCK % tr != 0 or k_dim % td != 0:
        return f"cannot map {BLOCK_SIZE} threads onto a [{SUB_BLOCK}, {k_dim}] accumulator"
    if BLOCK_SIZE % k_dim != 0 or SUB_BLOCK % (BLOCK_SIZE // k_dim) != 0:
        return f"cannot map {BLOCK_SIZE} threads onto a [{SUB_BLOCK}, {k_dim}] staging tile"
    return None


def build_kda_decay_scores_bwd(
    chunk_size: int,
    k_dim: int,
    waves_per_eu: int = 4,  # swept with THREADS_D; see the note above it
    owned_blocks=None,
):
    """Build the launcher for one ``(chunk_size, k_dim)`` geometry.

    Returns a callable ``launch(Q, K, CG, DAqk, DAkk, DQ, DK, DCG, num_chunks)``
    over flat fp32 tensors. ``DAqk``/``DAkk`` are read as autograd hands them
    over — dense, including above the diagonals — and masked inside the kernel.

    ``owned_blocks`` restricts which row/column blocks this kernel finalises;
    ``None``, the default, means all of them in one launch. Each owned block
    writes a disjoint set of output rows, so one-kernel-per-block is exactly
    equivalent arithmetically — and measured **7764 µs against 1963** at
    production geometry, four launches each costing what the fused one costs,
    because a launch that does a quarter of the arithmetic still reads the whole
    chunk. The parameter is kept only because that is the measurement that
    established this kernel is bound by its global traffic and not by its
    instruction count.
    """
    ensure_usable_lld()
    arch = get_rocm_arch()
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"kda_decay_scores_bwd targets gfx950 (CDNA4); got {arch!r}")

    C, KD, SB = int(chunk_size), int(k_dim), SUB_BLOCK
    reason = supports_bwd_geometry(C, KD)
    if reason is not None:
        raise ValueError(f"the KDA score-adjoint kernel cannot run this geometry: {reason}")
    NSB = C // SB
    OWNED = list(range(NSB)) if owned_blocks is None else [int(b) for b in owned_blocks]

    # accumulator mapping: thread -> (MR rows of block b) x (NR channels)
    TD = min(THREADS_D, KD)
    TR = BLOCK_SIZE // TD
    MR = SB // TR
    NR = KD // TD
    # staging mapping: thread -> (one channel, FRPT rows)
    FROW = BLOCK_SIZE // KD
    FRPT = SB // FROW
    # How to cover the thread's NR consecutive channels with the widest legal LDS
    # accesses. Resolved here because the traced body may only walk a ready-made
    # list; see :func:`..kda_decay_scores_kernel.lds_runs`.
    NR_RUNS = lds_runs(NR)

    # +4: the contraction reads channel `d` of 16 rows at once, and a bare K_DIM
    # stride would put all 16 in one LDS bank for any K_DIM divisible by 32.
    STRIDE = KD + 4
    # +1 because the column role reads the gradient tile down a column.
    DSTRIDE = SB + 1
    # Descending, so the Sigma_r contractions add their smallest terms first;
    # see :func:`contract_over_r`. Both orders are resolved here, at build scope,
    # so the traced body only ever indexes a ready-made Python list.
    R_ORDER = list(range(SB - 1, -1, -1))
    I_ORDER = {b: list(range(NSB - 1, b, -1)) for b in range(NSB)}
    J_ORDER = {b: list(range(b)) for b in range(NSB)}

    tag = f"C{C}_K{KD}_" + "".join(str(b) for b in OWNED)
    allocator = SmemAllocator(None, arch=arch, global_sym_name=f"kda_scores_bwd_smem_{tag}")
    off_r = allocator._align(allocator.ptr, 16)  # k * rf  (contracted over c)
    off_q = allocator._align(off_r + SB * STRIDE * 4, 16)  # q * lf  (contracted over r)
    off_k = allocator._align(off_q + SB * STRIDE * 4, 16)  # k * lf  (contracted over r)
    off_ga = allocator._align(off_k + SB * STRIDE * 4, 16)  # dAqk tile
    off_gk = allocator._align(off_ga + SB * DSTRIDE * 4, 16)  # dAkk tile
    allocator.ptr = off_gk + SB * DSTRIDE * 4

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1], name=f"kda_scores_bwd_{tag}")
    def kda_decay_scores_bwd_kernel(
        Q: fx.Tensor,  # [NB, C, KD] fp32 flat
        Kk: fx.Tensor,  # [NB, C, KD]
        CG: fx.Tensor,  # [NB, C, KD]
        DAqk: fx.Tensor,  # [NB, C, C]
        DAkk: fx.Tensor,  # [NB, C, C]
        DQ: fx.Tensor,  # [NB, C, KD] out
        DK: fx.Tensor,  # [NB, C, KD] out
        DCG: fx.Tensor,  # [NB, C, KD] out
    ):
        f32 = T.f32
        fm = arith.FastMathFlags.fast
        vec1_f32 = T.vec(1, f32)

        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        k_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Kk)
        cg_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), CG)
        gq_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DAqk)
        gk_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DAkk)
        dq_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DQ)
        dk_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DK)
        dcg_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DCG)

        base = allocator.get_base()
        lds_r = SmemPtr(base, off_r, f32, shape=(SB * STRIDE,)).get()
        lds_q = SmemPtr(base, off_q, f32, shape=(SB * STRIDE,)).get()
        lds_k = SmemPtr(base, off_k, f32, shape=(SB * STRIDE,)).get()
        lds_ga = SmemPtr(base, off_ga, f32, shape=(SB * DSTRIDE,)).get()
        lds_gk = SmemPtr(base, off_gk, f32, shape=(SB * DSTRIDE,)).get()

        chunk = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)

        zero_f = arith.constant(0.0, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)
        KD_IDX, C_IDX = arith.index(KD), arith.index(C)
        I_STRIDE, I_DSTRIDE = arith.index(STRIDE), arith.index(DSTRIDE)
        row_base = chunk * C_IDX

        def gep(bptr, elem_idx):
            return _llvm.GEPOp(
                _llvm_ptr_ty(),
                bptr,
                [arith.index_cast(T.i64, elem_idx)],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=f32,
                noWrapFlags=0,
            ).result

        def g_load(bptr, elem_idx):
            return _llvm.LoadOp(f32, gep(bptr, elem_idx)).result

        def g_store(val, bptr, elem_idx):
            _llvm.StoreOp(val, gep(bptr, elem_idx))

        def lds_read(lds, elem_idx):
            v = vector.load_op(vec1_f32, lds, [elem_idx])
            return vector.extract(v, static_position=[0], dynamic_position=[])

        def lds_read_run(lds, base_idx):
            """The thread's ``NR`` **consecutive** channels, in as few accesses as possible.

            The contraction's ``NR`` channel reads are what this kernel spends
            most of its LDS instructions on, and with ``my_d`` contiguous (see
            below) they are adjacent, so they go out as ``ds_read_b128`` instead
            of ``NR`` separate ``ds_read_b32``. Alignment holds because
            ``STRIDE = KD + 4`` makes every row start 16-byte aligned and
            ``my_d[0] = acc_d · NR`` is a multiple of ``NR``.
            """
            out = []
            for ri in range_constexpr(len(NR_RUNS)):
                off, width = NR_RUNS[ri]
                vec = vector.load_op(T.vec(width, f32), lds, [base_idx + arith.index(off)])
                for j in range_constexpr(width):
                    out.append(vector.extract(vec, static_position=[j], dynamic_position=[]))
            return out

        def lds_write(lds, elem_idx, val):
            vector.store(vector.from_elements(vec1_f32, [val]), lds, [elem_idx])

        def nat_exp(x):
            return _nat_exp(x, c_log2e, fm)

        def sub_f(a, b):
            return arith.SubFOp(a, b, fastmath=fm).result

        def add_f(a, b):
            return arith.AddFOp(a, b, fastmath=fm).result

        def mul_f(a, b):
            return arith.MulFOp(a, b, fastmath=fm).result

        def relem(row_in_chunk, d):
            """Flat index of ``[chunk, row_in_chunk, d]`` in a ``[NB, C, KD]``."""
            return (row_base + row_in_chunk) * KD_IDX + d

        def aelem(r, c):
            """Flat index of ``[chunk, r, c]`` in a ``[NB, C, C]``."""
            return (row_base + r) * C_IDX + c

        # ---- ids: staging tiles, gradient tiles, accumulators ----
        fill_d = tid % KD_IDX
        fill_r = tid // KD_IDX
        grad_r = tid // arith.index(SB)
        grad_c = tid % arith.index(SB)
        acc_d = tid % arith.index(TD)
        acc_r = tid // arith.index(TD)
        my_row = [acc_r + arith.index(m * TR) for m in range_constexpr(MR)]
        # Thread `acc_d` owns `NR` **consecutive** channels, not `NR` channels
        # strided by `TD`. Both mappings cover the same channels and produce the
        # same sums in the same order, so the output is bit-identical -- but only
        # this one makes the `NR` reads of each contraction adjacent, so they
        # become one `ds_read_b128` (see `lds_read_run`), and makes the `cg`
        # loads and the three output stores `global_*_dwordx4`. The strided
        # mapping is the natural one to write and it is why this kernel spent 5.9x
        # the forward kernel's time for 2x its arithmetic.
        my_d = [acc_d * arith.index(NR) + arith.index(n) for n in range_constexpr(NR)]
        my_d0 = acc_d * arith.index(NR)

        def zeros():
            return [[zero_f for _ in range_constexpr(NR)] for _ in range_constexpr(MR)]

        # ---- staging ----
        def fill_right(row0, ref_row):
            """``lds_r[c][d] = k[row0+c, d] · exp(cg[ref,d] − cg[row0+c,d])``."""
            cg_ref = g_load(cg_ptr, relem(ref_row, fill_d))
            for m in range_constexpr(FRPT):
                rr = fill_r + arith.index(m * FROW)
                e = relem(arith.index(row0) + rr, fill_d)
                fac = nat_exp(sub_f(cg_ref, g_load(cg_ptr, e)))
                lds_write(lds_r, rr * I_STRIDE + fill_d, mul_f(g_load(k_ptr, e), fac))

        def fill_left(row0, ref_row):
            """``lds_q/lds_k[r][d] = q/k[row0+r, d] · exp(cg[row0+r,d] − cg[ref,d])``."""
            cg_ref = g_load(cg_ptr, relem(ref_row, fill_d))
            for m in range_constexpr(FRPT):
                rr = fill_r + arith.index(m * FROW)
                e = relem(arith.index(row0) + rr, fill_d)
                fac = nat_exp(sub_f(g_load(cg_ptr, e), cg_ref))
                off = rr * I_STRIDE + fill_d
                lds_write(lds_q, off, mul_f(g_load(q_ptr, e), fac))
                lds_write(lds_k, off, mul_f(g_load(k_ptr, e), fac))

        def fill_diag(row0, ref_row):
            """All three tiles for the diagonal block, from one pass over its rows.

            ``fill_right`` and ``fill_left`` on the same rows with the same
            reference would read ``cg`` twice and ``k`` twice; the diagonal is the
            only block that needs both directions, and merging them is worth 10 %
            of this kernel's global traffic.
            """
            cg_ref = g_load(cg_ptr, relem(ref_row, fill_d))
            for m in range_constexpr(FRPT):
                rr = fill_r + arith.index(m * FROW)
                e = relem(arith.index(row0) + rr, fill_d)
                cg_row = g_load(cg_ptr, e)
                kv = g_load(k_ptr, e)
                lf = nat_exp(sub_f(cg_row, cg_ref))
                rf = nat_exp(sub_f(cg_ref, cg_row))
                off = rr * I_STRIDE + fill_d
                lds_write(lds_r, off, mul_f(kv, rf))
                lds_write(lds_q, off, mul_f(g_load(q_ptr, e), lf))
                lds_write(lds_k, off, mul_f(kv, lf))

        def _keep_all(val):
            return val

        def _keep_le(val):
            return arith.select(arith.cmpi(arith.CmpIPredicate.sle, grad_c, grad_r), val, zero_f)

        def _keep_lt(val):
            return arith.select(arith.cmpi(arith.CmpIPredicate.slt, grad_c, grad_r), val, zero_f)

        # The forward writes exact zeros above its two diagonals, so the upstream
        # gradient there is the gradient of a constant and must be dropped. Off
        # the diagonal block every entry is kept, hence the identity closure.
        MASK_Q = {False: _keep_all, True: _keep_le}
        MASK_K = {False: _keep_all, True: _keep_lt}

        def fill_grads(rblk, cblk, is_diag):
            e = aelem(arith.index(rblk * SB) + grad_r, arith.index(cblk * SB) + grad_c)
            off = grad_r * I_DSTRIDE + grad_c
            lds_write(lds_ga, off, MASK_Q[is_diag](g_load(gq_ptr, e)))
            lds_write(lds_gk, off, MASK_K[is_diag](g_load(gk_ptr, e)))

        # ---- the two contraction directions ----
        def contract_over_c(t_dq, t_a2):
            """``t[r,d] += Σ_c dA[r,c] · lds_r[c,d]`` — the row owner's direction."""
            for c in range_constexpr(SB):
                ci = arith.index(c)
                bv = lds_read_run(lds_r, ci * I_STRIDE + my_d0)
                for m in range_constexpr(MR):
                    ga = lds_read(lds_ga, my_row[m] * I_DSTRIDE + ci)
                    gk = lds_read(lds_gk, my_row[m] * I_DSTRIDE + ci)
                    for n in range_constexpr(NR):
                        t_dq[m][n] = math_dialect.fma(ga, bv[n], t_dq[m][n])
                        t_a2[m][n] = math_dialect.fma(gk, bv[n], t_a2[m][n])

        def contract_over_r(t_a1, t_a3):
            """``t[c,d] += Σ_r dA[r,c] · lds_q/lds_k[r,d]`` — the column owner's.

            Runs ``r`` **downwards**. The staged operand carries
            ``lf[r,d] = exp(cg[r,d] − cg[ref,d])`` and ``cg`` is non-increasing in
            ``r``, so descending ``r`` adds the smallest terms first, which is the
            better fp32 order. ``contract_over_c`` needs no such reversal: its
            ``rf[c,d] = exp(cg[ref,d] − cg[c,d])`` *grows* with ``c``, so ascending
            already is smallest-first.
            """
            for ro in range_constexpr(SB):
                ri = arith.index(R_ORDER[ro])
                qv = lds_read_run(lds_q, ri * I_STRIDE + my_d0)
                kv = lds_read_run(lds_k, ri * I_STRIDE + my_d0)
                for m in range_constexpr(MR):
                    ga = lds_read(lds_ga, ri * I_DSTRIDE + my_row[m])
                    gk = lds_read(lds_gk, ri * I_DSTRIDE + my_row[m])
                    for n in range_constexpr(NR):
                        t_a1[m][n] = math_dialect.fma(ga, qv[n], t_a1[m][n])
                        t_a3[m][n] = math_dialect.fma(gk, kv[n], t_a3[m][n])

        def scale_into(acc, fac, tmp):
            for m in range_constexpr(MR):
                for n in range_constexpr(NR):
                    acc[m][n] = math_dialect.fma(fac[m][n], tmp[m][n], acc[m][n])

        # Measured and rejected: giving each sub-block a fresh accumulator and
        # folding the partial sums in afterwards, so that the last block's row
        # owner carries four fp32 chains of 16 instead of one of 48. It changed
        # *nothing* — the kernel's output was bit-identical against the fp64
        # oracle and its time was 1760 µs against 1754 — because this kernel
        # compiles with `unsafe_fp_math`, under which LLVM is free to reassociate
        # the fold straight back into one chain. Summation order here can only be
        # controlled by the order the FMAs are *emitted* in (which is why
        # `contract_over_r` runs `r` downwards), not by adding structure.

        # ======================= one owned block per step ======================
        for bi in range_constexpr(len(OWNED)):
            b = OWNED[bi]
            r0 = b * SB
            rows_g = [arith.index(r0) + my_row[m] for m in range_constexpr(MR)]
            acc_dq, acc_a2, acc_a1, acc_a3 = zeros(), zeros(), zeros(), zeros()
            # `cg` at the rows this thread owns feeds all three phases' decay
            # factors, so it is loaded once rather than once per phase.
            cg_own = [
                [g_load(cg_ptr, relem(rows_g[m], my_d[n])) for n in range_constexpr(NR)]
                for m in range_constexpr(MR)
            ]

            # ---- row owner, all earlier column blocks, ref = first row of b ----
            # cg is non-increasing in the row index, so rows of block b sit at or
            # below the reference and every earlier column above it: both
            # factors land in (0, 1].
            ref_first = arith.index(r0)
            cgf = [g_load(cg_ptr, relem(ref_first, my_d[n])) for n in range_constexpr(NR)]
            lf_row = [
                [nat_exp(sub_f(cg_own[m][n], cgf[n])) for n in range_constexpr(NR)]
                for m in range_constexpr(MR)
            ]
            t_dq, t_a2 = zeros(), zeros()
            for jj in range_constexpr(len(J_ORDER[b])):
                j = J_ORDER[b][jj]
                fill_right(j * SB, ref_first)
                fill_grads(b, j, False)
                gpu.barrier()
                contract_over_c(t_dq, t_a2)
                gpu.barrier()
            scale_into(acc_dq, lf_row, t_dq)
            scale_into(acc_a2, lf_row, t_a2)

            # ---- the diagonal block, ref = midpoint: |exponent| <= (SB/2)*5 ----
            ref_mid = arith.index(r0 + SB // 2)
            cgm = [g_load(cg_ptr, relem(ref_mid, my_d[n])) for n in range_constexpr(NR)]
            lf_diag = [
                [nat_exp(sub_f(cg_own[m][n], cgm[n])) for n in range_constexpr(NR)]
                for m in range_constexpr(MR)
            ]
            rf_diag = [
                [nat_exp(sub_f(cgm[n], cg_own[m][n])) for n in range_constexpr(NR)]
                for m in range_constexpr(MR)
            ]
            fill_diag(r0, ref_mid)
            fill_grads(b, b, True)
            gpu.barrier()
            d_dq, d_a2, d_a1, d_a3 = zeros(), zeros(), zeros(), zeros()
            contract_over_c(d_dq, d_a2)
            contract_over_r(d_a1, d_a3)
            gpu.barrier()
            scale_into(acc_dq, lf_diag, d_dq)
            scale_into(acc_a2, lf_diag, d_a2)
            scale_into(acc_a1, rf_diag, d_a1)
            scale_into(acc_a3, rf_diag, d_a3)

            # ---- column owner, all later row blocks, ref = LAST row of b ----
            # Not the first row: columns of block b sit *after* it, which would
            # make rf = exp(cg[first] - cg[c]) reach exp((SB-1)*5) = exp(75).
            # Referencing the last row puts the columns at or above it and the
            # later rows below it, so both factors are again in (0, 1].
            ref_last = arith.index(r0 + SB - 1)
            cgl = [g_load(cg_ptr, relem(ref_last, my_d[n])) for n in range_constexpr(NR)]
            rf_col = [
                [nat_exp(sub_f(cgl[n], cg_own[m][n])) for n in range_constexpr(NR)]
                for m in range_constexpr(MR)
            ]
            t_a1, t_a3 = zeros(), zeros()
            for ii in range_constexpr(len(I_ORDER[b])):
                i = I_ORDER[b][ii]
                fill_left(i * SB, ref_last)
                fill_grads(i, b, False)
                gpu.barrier()
                contract_over_r(t_a1, t_a3)
                gpu.barrier()
            scale_into(acc_a1, rf_col, t_a1)
            scale_into(acc_a3, rf_col, t_a3)

            # ---- dk and dcg are elementwise once the four sums are in hand ----
            for m in range_constexpr(MR):
                for n in range_constexpr(NR):
                    e = relem(rows_g[m], my_d[n])
                    dqv, a1, a2, a3 = acc_dq[m][n], acc_a1[m][n], acc_a2[m][n], acc_a3[m][n]
                    g_store(dqv, dq_ptr, e)
                    g_store(add_f(add_f(a1, a2), a3), dk_ptr, e)
                    g_store(
                        math_dialect.fma(
                            g_load(q_ptr, e),
                            dqv,
                            mul_f(g_load(k_ptr, e), sub_f(sub_f(a2, a1), a3)),
                        ),
                        dcg_ptr,
                        e,
                    )

    @flyc.jit
    def launch_kda_decay_scores_bwd(
        Q: fx.Tensor,
        Kk: fx.Tensor,
        CG: fx.Tensor,
        DAqk: fx.Tensor,
        DAkk: fx.Tensor,
        DQ: fx.Tensor,
        DK: fx.Tensor,
        DCG: fx.Tensor,
        num_chunks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        grid_x = arith.index_cast(T.index, num_chunks)
        launcher = kda_decay_scores_bwd_kernel(Q, Kk, CG, DAqk, DAkk, DQ, DK, DCG)
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
            return launch_kda_decay_scores_bwd(*args, **kwargs)

    return with_current_stream(_launch)
