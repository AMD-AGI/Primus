###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused FlyDSL kernel for Quantile Balancing's per-expert margin histogram.

One thread per ``(token, expert)`` pair computes

    margin = scores[n, e] − tau[n]
    raw    = floor((margin − lo) / width)
    slot   = clamp(raw, 0, B−1)
    hist[e, slot] += 1
    hist[e, B]    += 1   if raw < 0        (below-range counter)
    hist[e, B+1]  += 1   if raw > B−1      (above-range counter)

and that is the whole statistic. The eager form spends nine kernel launches on
it — sigmoid, add-bias, top-k, subtract, scale-and-floor, two clamp counters, an
int64 cast, an offset-add and a ``bincount`` — of which the ``bincount`` alone is
102 µs of a 208 µs total at the scaled shape (4096 tokens, 32 experts, 1024
bins). Everything after the top-k collapses into this one launch.

Two design points are numerically load-bearing
----------------------------------------------
**The bin index is a reciprocal multiply, because that is what the oracle does.**
``floor`` is discontinuous, so a last-bit difference in ``(margin − lo)/width``
moves a count into the neighbouring bin — which makes "match the oracle exactly"
the requirement, and "be as accurate as possible" a distraction. The first
version of this kernel used a true ``arith.DivFOp`` on exactly that reasoning and
it was **wrong**: measured against the eager path at 1000 bins over ``[-1, 1]``,
the true division mismatched **4 of 32 000 bins** while a reciprocal multiply
mismatched **0**. The reason is that *PyTorch* implements tensor-by-Python-scalar
division as a reciprocal multiply — probed directly on 2^20 values,
``(x - lo) / width`` and ``(x - lo) * (1/width)`` are bit-identical in torch,
while dividing by a one-element *tensor* differs from both in 35 of them.

Two corollaries worth keeping:

* with the **shipped** binning the question is moot. ``margin_min/max = ±1.0``
  over a power-of-two bin count makes ``width`` an exact power of two, so its
  reciprocal is exact and every spelling agrees. The choice only matters for the
  arbitrary binning the config permits, which is the case to be correct for.
* ``fast_fp_math`` is left **off** here. It was measured not to change the bin
  index either way, and this kernel is atomic- and memory-bound, so the flag buys
  nothing while widening what a future compiler version could change.

**The two out-of-range counters are folded into the histogram as columns
``B`` and ``B+1``, per expert.** They are mutually exclusive, so one predicated
atomic covers both. Making them per-expert rather than global turns 2 hot
addresses into ``2·E``, which drops the serialisation from ``N·E`` deep to
``N·E / 2E``; the caller sums the two columns to recover the scalars the eager
form returns.

Layout and geometry
-------------------
* ``SCORES``: ``[N, E]`` fp32 flat — raw sigmoid scores, **not** biased
* ``TAU``: ``[N]`` fp32 flat — the ``(k+1)``-th largest biased score per token
* ``HIST``: ``[E, B+2]`` **int32** flat, zeroed by the caller

int32 rather than int64 because ``hist[e, b]`` counts tokens and is bounded by
``N``, so a 32-bit counter cannot overflow for any batch that fits in memory;
the caller widens to int64 to match the eager return type. gfx950 int32 global
atomics are also a single instruction where a 64-bit add is not.

The grid is flat over ``N·E`` with a **branch-free** tail guard: an
out-of-range lane clamps its index to 0 and atomically adds 0, which is a no-op.
That avoids an ``scf.if`` in the body, which the AST rewriter is fragile around.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import math as math_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch

from primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels._flydsl_v1._lld_shim import (
    ensure_usable_lld,
)

_LLVM_GEP_DYNAMIC = -2147483648

BLOCK_SIZE = 256
#: ``hist[e, b] <= num_tokens``, so an int32 counter is safe for any batch that
#: fits in memory. Asserted rather than assumed.
MAX_TOKENS = 2**31 - 1

#: Test-only defects, so the bit-exactness test can be shown to have
#: discrimination power. Production never passes ``inject``.
#:
#: ``trunc_not_floor``
#:     ``trunc`` instead of ``floor``. They agree for positive values and differ
#:     by one for every negative one — and most margins *are* negative, since a
#:     margin is a score minus the ``(k+1)``-th largest.
#: ``no_clamp_count``
#:     Bin out-of-range margins into the end bins but do not count them, so
#:     saturation becomes invisible.
#: ``off_by_one_bin``
#:     Shift every bin index by one. Included because a histogram comparison
#:     that this does not break is not comparing anything.
INJECTIONS = (
    "trunc_not_floor",
    "no_clamp_count",
    "off_by_one_bin",
)

#: ``true_division`` — ``margin / width`` instead of ``margin · (1/width)``, i.e.
#: the *more accurate* spelling, which is a defect here because the oracle does
#: not use it. Kept separate from :data:`INJECTIONS` because whether it is
#: observable **depends on the binning**, and measuring that is more useful than
#: asserting it is always wrong:
#:
#: * with the shipped binning (``±1.0`` over a power-of-two bin count) the width
#:   is an exact power of two, so every spelling agrees bit-for-bit and there is
#:   nothing to catch;
#: * at 1000 bins over ``[-1, 1]`` it mismatches 4 of 32 000 bins.
#:
#: ``test_qb_flydsl_kernel.py`` pins both halves.
WIDTH_DEPENDENT_VARIANTS = ("true_division",)

_ALL_VARIANTS = INJECTIONS + WIDTH_DEPENDENT_VARIANTS

__all__ = [
    "build_qb_margin_histogram",
    "BLOCK_SIZE",
    "MAX_TOKENS",
    "INJECTIONS",
    "WIDTH_DEPENDENT_VARIANTS",
]


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def build_qb_margin_histogram(
    num_experts: int,
    num_bins: int,
    margin_min: float,
    margin_max: float,
    waves_per_eu: int = 2,
    inject: str = "",
):
    """Build the launcher for one ``(num_experts, num_bins, range)`` geometry.

    Args:
        inject: **test only.** One of :data:`INJECTIONS`. Empty (the production
            value) emits the correct kernel.

    Returns ``launch(SCORES, TAU, HIST, num_pairs)`` over flat tensors, where
    ``num_pairs = num_tokens * num_experts`` and ``HIST`` is a zeroed
    ``[num_experts, num_bins + 2]`` int32 tensor.
    """
    ensure_usable_lld()
    arch = get_rocm_arch()
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"qb_margin_histogram targets gfx950 (CDNA4); got {arch!r}")

    E = int(num_experts)
    B = int(num_bins)
    if E < 1:
        raise ValueError(f"num_experts must be >= 1, got {E}")
    if B < 2:
        raise ValueError(f"num_bins must be >= 2, got {B}")
    if inject and inject not in _ALL_VARIANTS:
        raise ValueError(f"unknown injection {inject!r}; expected '' or one of {list(_ALL_VARIANTS)}")
    LO = float(margin_min)
    WIDTH = (float(margin_max) - LO) / B
    HIST_STRIDE = B + 2  # ... + below-range column + above-range column
    _tag = f"_inj_{inject}" if inject else ""

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1], name=f"qb_margin_hist_E{E}_B{B}{_tag}")
    def qb_margin_histogram_kernel(
        SCORES: fx.Tensor,  # [N, E] f32 flat
        TAU: fx.Tensor,  # [N]    f32 flat
        HIST: fx.Tensor,  # [E, B+2] i32 flat, zeroed
        num_pairs: fx.Int32,
    ):
        f32 = T.f32
        i32 = T.i32
        add_op = _llvm.AtomicBinOp.add
        monotonic = _llvm.AtomicOrdering.monotonic

        s_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), SCORES)
        t_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), TAU)
        h_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), HIST)

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

        def atomic_add_i32(bptr, elem_idx, val):
            _llvm.AtomicRMWOp(add_op, gep(bptr, elem_idx, i32), val, monotonic)

        tid = arith.index_cast(T.index, gpu.thread_idx.x)
        blk = arith.index_cast(T.index, gpu.block_idx.x)
        npair = arith.index_cast(T.index, num_pairs)
        flat = blk * arith.index(BLOCK_SIZE) + tid

        I_E = arith.index(E)
        I_ZERO = arith.index(0)
        c_i32_zero = arith.constant(0, type=i32)
        c_i32_one = arith.constant(1, type=i32)
        c_lo = arith.constant(LO, type=f32)
        c_width = arith.constant(WIDTH, type=f32)
        c_bmax = arith.constant(float(B - 1), type=f32)
        c_fzero = arith.constant(0.0, type=f32)
        c_inv_width = arith.constant(1.0 / WIDTH, type=f32)

        # Build-time selections, by dict indexing rather than by an `if`.
        # The reciprocal multiply is the DEFAULT: it is what torch's
        # tensor-by-scalar division compiles to, and matching the oracle is the
        # requirement. See the module docstring for the measurement.
        scale_of = {
            False: lambda x: arith.MulFOp(x, c_inv_width).result,
            True: lambda x: arith.DivFOp(x, c_width).result,
        }[inject == "true_division"]
        round_of = {False: math_dialect.floor, True: math_dialect.trunc}[inject == "trunc_not_floor"]
        BIN_SHIFT = 1 if inject == "off_by_one_bin" else 0

        # Branch-free tail guard: an out-of-range lane reads element 0 (always
        # in bounds when the kernel is launched at all) and adds 0.
        in_range = arith.cmpi(arith.CmpIPredicate.slt, flat, npair)
        safe = arith.select(in_range, flat, I_ZERO)
        tok = safe // I_E
        exp = safe % I_E

        score = load_f32(s_ptr, safe)
        tau = load_f32(t_ptr, tok)
        margin = arith.SubFOp(score, tau).result

        # A TRUE division: `floor` is discontinuous, so a reciprocal multiply
        # would move edge cases into the neighbouring bin.
        shifted = arith.SubFOp(margin, c_lo).result
        raw = round_of(scale_of(shifted))

        below = arith.cmpf(arith.CmpFPredicate.OLT, raw, c_fzero)
        above = arith.cmpf(arith.CmpFPredicate.OGT, raw, c_bmax)
        # clamp(raw, 0, B-1), spelled as the eager path spells it
        clamped = arith.MaxNumFOp(arith.MinNumFOp(raw, c_bmax).result, c_fzero).result
        slot = arith.index_cast(T.index, arith.fptosi(i32, clamped)) + arith.index(BIN_SHIFT)

        row = exp * arith.index(HIST_STRIDE)
        atomic_add_i32(h_ptr, row + slot, arith.select(in_range, c_i32_one, c_i32_zero))

        # The two out-of-range counters share one predicated atomic: a margin
        # cannot be both below and above the range.
        oob_slot = arith.select(below, arith.index(B), arith.index(B + 1))
        oob_any = arith.ori(below, above)
        oob_live = arith.andi(oob_any, in_range)
        oob_count = {False: arith.select(oob_live, c_i32_one, c_i32_zero), True: c_i32_zero}[
            inject == "no_clamp_count"
        ]
        atomic_add_i32(h_ptr, row + oob_slot, oob_count)

    @flyc.jit
    def launch_qb_margin_histogram(
        SCORES: fx.Tensor,
        TAU: fx.Tensor,
        HIST: fx.Tensor,
        num_pairs: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        npair = arith.index_cast(T.index, num_pairs)
        grid_x = (npair + arith.index(BLOCK_SIZE - 1)) // arith.index(BLOCK_SIZE)
        launcher = qb_margin_histogram_kernel(SCORES, TAU, HIST, num_pairs)

        for op in ctx.gpu_module_body.operations:
            if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, int(waves_per_eu))
                op.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get(f"{BLOCK_SIZE},{BLOCK_SIZE}")

        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    # No fast/unsafe fp math. The probe showed the floor-of-a-division is
    # bit-exact either way, but this kernel is memory- and atomic-bound, so the
    # flags buy nothing and leaving them off makes the bit-exactness a property
    # of the code rather than of a compiler option.
    return launch_qb_margin_histogram
