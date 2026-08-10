###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused FlyDSL forward kernel for the Kimi K3 attention-residual mixer.

One workgroup per token computes, for ``C = num_blocks + 1`` candidates,

    ss[c]   = Σ_d v[c,d]²
    dot[c]  = Σ_d v[c,d] · w[d]
    r[c]    = rsqrt(ss[c]/H + eps)
    s[c]    = dot[c] · r[c]
    p       = softmax(s)
    out[d]  = Σ_c p[c] · v[c,d]

where ``v`` is the concatenation of the ``num_blocks`` checkpoints and the
running stream, and ``w`` is the fused rank-1 scorer.

Why this is the fusion, and what it removes
-------------------------------------------
The eager form is not arithmetic-bound; it is materialisation-bound. It writes
six full-size ``[tokens, C, hidden]`` intermediates — the ``cat``, the fp32
up-cast, the variance, the normalised ``k``, the score product, and the matmul's
read — and five of them are fp32. Measured at ``tokens = 4096, C = 4,
hidden = 2048``: ~1.07 GB of traffic for a 67 MB bf16 input, 404 µs forward.

Two algebraic facts collapse all of it:

* ``scores = rsqrt(mean(v²) + eps) · <v, w>``. The RMS normalisation never has
  to be *applied* to ``v``; the reduction it scales is ``<v, w>``, and both
  reductions come out of **one** pass. That kills the ``k`` tensor.
* ``out`` is a convex combination of ``v`` itself, so the second pass reads the
  same bytes and writes only ``[tokens, hidden]``.

So the kernel reads ``v`` twice in its input dtype and writes one output, i.e.
~151 MB against the eager path's ~1.07 GB at that shape. The ``cat`` never
happens: the two source tensors are passed as separate pointers and the
candidate axis is a build-time constant, so which pointer a candidate lives
behind is resolved during tracing.

Layout and geometry
-------------------
Every tensor is passed flat.

* ``BR``: ``[N, NB, H]`` in the model dtype (bf16 or fp32)
* ``PS``: ``[N, H]`` in the model dtype
* ``W``: ``[H]`` fp32 — the fused scorer, formed outside the kernel
* ``OUT``: ``[N, H]`` in the model dtype
* ``RSAV``, ``DSAV``: ``[N, C]`` fp32 — ``r`` and ``dot``, saved for the
  backward so it never has to re-reduce them

``BLOCK`` is 256 threads when ``H % 256 == 0`` and 64 otherwise, and ``H`` must
be a multiple of 64; the launcher checks it and the Python caller falls back to
eager. Each thread owns the strided slice ``d = tid + j·BLOCK``, which keeps
every global access coalesced across the workgroup.

gfx950 / CDNA4. There is no MFMA here on purpose: every contraction is a
reduction of one vector against another, i.e. rank-1, so an MFMA would be fed a
single row and waste 15/16 of the tile. The arithmetic is VALU FMA and the
kernel is bound by HBM, which is the thing being optimised.

Three FlyDSL tracing rules are load-bearing here:

* ``rocdl`` is unreachable from a helper defined *inside* a traced body, so
  :func:`_nat_exp` and :func:`_rsqrt_refined` sit at module scope and are
  called from the body.
* a Python ``if`` inside a traced body does not propagate branch-local
  rebindings out, so every build-time choice — which pointer a candidate lives
  behind, which element type to load — is made by **indexing a list built
  before the body**, never by an ``if`` statement.
* ``exp`` is not exposed; ``rocdl.exp2`` (hardware ``v_exp_f32``) is.
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

from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._flydsl_v1._lld_shim import (
    ensure_usable_lld,
)

_LOG2E = math.log2(math.e)  # exp(x) == exp2(x * log2(e)); only exp2 is exposed
_LLVM_GEP_DYNAMIC = -2147483648  # LLVM kDynamicIndex sentinel

#: Widest workgroup used. 256 threads is one output element per thread for
#: ``H = 256`` and a clean strided slice for every larger multiple.
MAX_BLOCK = 256
#: Narrowest workgroup, one full wave.
MIN_BLOCK = 64
#: The mixer is only ever built at the model's hidden size, so a
#: multiple-of-64 requirement covers every real shape (2048, 7168) and every
#: test shape worth having.
HIDDEN_ALIGN = 64
#: Candidate count ceiling. ``C = ceil(num_layers / attn_res_block_size) + 1``,
#: which is 9 for the 93-layer release and 4 for the scaled config. The
#: candidate loop is fully unrolled, so a very large ``C`` would blow up the
#: instruction cache rather than fail; the cap makes that a refusal instead.
MAX_CANDIDATES = 33

#: Test-only defects the builder can emit, so the parity test can be shown to
#: have discrimination power. A test that passes against a broken kernel is
#: worthless, and the only way to know is to break the kernel — in the emitted
#: MLIR, not in the torch glue around it. Production never passes ``inject``,
#: and :func:`build_attn_res_mixer_fwd` rejects an unknown name, so a typo in a
#: test cannot silently become "no injection at all".
#:
#: ``mix_normalised``
#:     Mix the RMS-normalised candidates instead of the raw ones. The easiest
#:     thing to get wrong and the one the reference is most explicit about
#:     (``modeling_kimi_linear.py:1083`` builds ``k``, ``:1087`` mixes ``v``).
#: ``score_unnormalised``
#:     The mirror image: score the raw candidates, skipping the RMS scaling.
#: ``no_softmax_max``
#:     Drop the max subtraction in the softmax. Algebraically a no-op, so it
#:     only shows up as overflow at a large score spread — which is why the
#:     test for it uses a wide ``proj_weight`` rather than the default fixture.
#: ``drop_eps``
#:     Leave ``eps`` out of the rsqrt. Only observable when a candidate has
#:     (near-)zero norm, which is not hypothetical: ``block_residual`` starts as
#:     literal zeros at layer 0 of a fresh model.
FWD_INJECTIONS = (
    "mix_normalised",
    "score_unnormalised",
    "no_softmax_max",
    "drop_eps",
)

#: Build-time variants that are **measured** to sit inside the fp32 parity band,
#: i.e. things one might expect to be defects and that are not. They are kept
#: buildable so the claim stays testable rather than becoming folklore; each has
#: a test that records the measurement.
#:
#: ``no_newton``
#:     Use the raw ``v_rsq_f32`` with no Newton refinement. Measured inside the
#:     ``atol = rtol = 1e-5`` band: gfx950's ``V_RSQ_F32`` is accurate to about
#:     ``2^-22``, the score is ``dot · r`` and only feeds a softmax, so the error
#:     never reaches the output. The refinement ships anyway — it is three VALU
#:     ops **per candidate**, against ``C·H`` FMAs of real work — but it is not
#:     load-bearing, and saying so is more useful than implying it is.
#: ``stream_first``
#:     Put the running stream first in the candidate order instead of last.
#:     Measured **bit-identical**, and it has to be: softmax is
#:     permutation-equivariant and the output is a sum over every candidate, so
#:     the mixer's arithmetic cannot see candidate order at all. Candidate order
#:     is therefore only checkable one level up, where a checkpoint slot has to
#:     line up with the layer that wrote it — which is what
#:     ``test_attention_residual.py``'s block-level tests do.
FWD_NEUTRAL_VARIANTS = ("no_newton", "stream_first")

_ALL_FWD_VARIANTS = FWD_INJECTIONS + FWD_NEUTRAL_VARIANTS

__all__ = [
    "build_attn_res_mixer_fwd",
    "supports_mixer_geometry",
    "block_size_for",
    "MAX_BLOCK",
    "MIN_BLOCK",
    "HIDDEN_ALIGN",
    "MAX_CANDIDATES",
    "FWD_INJECTIONS",
    "FWD_NEUTRAL_VARIANTS",
]


def block_size_for(hidden: int) -> int:
    """Workgroup width for ``hidden``: 256 when it divides, else 64."""
    return MAX_BLOCK if hidden % MAX_BLOCK == 0 else MIN_BLOCK


def supports_mixer_geometry(hidden: int, num_candidates: int):
    """``None`` when the kernel can run this geometry, else why it cannot."""
    if hidden % HIDDEN_ALIGN != 0:
        return f"hidden={hidden} is not a multiple of {HIDDEN_ALIGN}"
    if num_candidates < 1:
        return f"num_candidates={num_candidates} must be >= 1"
    if num_candidates > MAX_CANDIDATES:
        return f"num_candidates={num_candidates} exceeds the unrolled cap {MAX_CANDIDATES}"
    return None


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _nat_exp(x, log2e_const, fastmath):
    """``exp(x)`` via the hardware ``v_exp_f32``.

    Module scope, and ``rocdl.exp2`` rather than ``arith.ArithValue(x).exp2()``:
    both are load-bearing for the same reasons as
    ``kda_decay_scores_kernel._nat_exp`` — the AST rewriter cannot resolve
    ``rocdl`` from a helper defined inside a traced body, and the ``ArithValue``
    spelling lowers to an external ``__ocml_exp2_f32`` call that drags the ROCm
    device-bitcode link into every compile.
    """
    scaled = arith.MulFOp(x, log2e_const, fastmath=fastmath).result
    return rocdl.exp2(T.f32, scaled)


def _rsqrt_refined(x, c_three, c_half, fastmath):
    """``1/sqrt(x)`` to ~1 ULP: ``v_rsq_f32`` plus one Newton step.

    ``rocdl.rsq`` alone is a hardware approximation good to about ``2^-22``,
    which is at the edge of the ~1e-7 band this project accepts on KDA and would
    dominate the fp32 parity error for no reason. One Newton–Raphson step on
    ``f(r) = r^-2 - x``, ``r <- r·(3 - x·r²)/2``, converges quadratically and
    costs three VALU ops **per candidate** (not per element), i.e. ``C`` of them
    per workgroup against ``C·H`` FMAs of real work.
    """
    r = rocdl.rsq(T.f32, x)
    rr = arith.MulFOp(r, r, fastmath=fastmath).result
    xrr = arith.MulFOp(x, rr, fastmath=fastmath).result
    corr = arith.SubFOp(c_three, xrr, fastmath=fastmath).result
    half_corr = arith.MulFOp(corr, c_half, fastmath=fastmath).result
    return arith.MulFOp(r, half_corr, fastmath=fastmath).result


def build_attn_res_mixer_fwd(
    hidden: int,
    num_blocks: int,
    elem_dtype: str,
    eps: float,
    waves_per_eu: int = 2,
    inject: str = "",
):
    """Build the forward launcher for one ``(hidden, num_blocks, dtype, eps)``.

    Args:
        hidden: candidate width ``H``.
        num_blocks: ``NB``; ``C = NB + 1`` candidates.
        elem_dtype: ``"f32"`` or ``"bf16"`` — the dtype of ``BR``, ``PS`` and
            ``OUT``. ``W``, ``RSAV`` and ``DSAV`` are always fp32.
        eps: RMSNorm epsilon, baked in as a constant.
        inject: **test only.** One of :data:`FWD_INJECTIONS`, which emits a
            deliberately defective kernel so the parity test can be shown to
            catch it. Empty (the production value) emits the correct kernel.

    Returns:
        ``launch(BR, PS, W, OUT, RSAV, DSAV, num_tokens)`` over flat tensors.
    """
    ensure_usable_lld()
    arch = get_rocm_arch()
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"attn_res_mixer targets gfx950 (CDNA4); got {arch!r}")

    H = int(hidden)
    NB = int(num_blocks)
    C = NB + 1
    reason = supports_mixer_geometry(H, C)
    if reason is not None:
        raise ValueError(f"attn_res_mixer cannot run this geometry: {reason}")
    if elem_dtype not in ("f32", "bf16"):
        raise ValueError(f"elem_dtype must be 'f32' or 'bf16'; got {elem_dtype!r}")
    if inject and inject not in _ALL_FWD_VARIANTS:
        raise ValueError(f"unknown injection {inject!r}; expected '' or one of {list(_ALL_FWD_VARIANTS)}")

    BLOCK = block_size_for(H)
    EPT = H // BLOCK  # hidden elements per thread
    EPSILON = 0.0 if inject == "drop_eps" else float(eps)

    # 2C reduction rows (ss and dot per candidate), one f32 slot per thread.
    RED_ROWS = 2 * C
    allocator = SmemAllocator(None, arch=arch, global_sym_name=f"attnres_fwd_smem_H{H}_C{C}_{elem_dtype}")
    lds_red_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_red_off + RED_ROWS * BLOCK * 4

    # Recursive-doubling all-reduce strides. After steps 1, 2, 4, ... BLOCK/2
    # with wraparound addition every lane holds the whole workgroup's sum, so
    # the reduction needs no conditional and no designated leader.
    RD_STRIDES = [1 << i for i in range(int(math.log2(BLOCK)))]
    assert (1 << len(RD_STRIDES)) == BLOCK, "recursive doubling needs a power-of-two block"

    _tag = f"_inj_{inject}" if inject else ""

    @flyc.kernel(
        known_block_size=[BLOCK, 1, 1],
        name=f"attn_res_mixer_fwd_{elem_dtype}_H{H}_C{C}{_tag}",
    )
    def attn_res_mixer_fwd_kernel(
        BR: fx.Tensor,  # [N, NB, H] elem_dtype flat
        PS: fx.Tensor,  # [N, H]     elem_dtype flat
        W: fx.Tensor,  # [H]        f32 flat
        OUT: fx.Tensor,  # [N, H]     elem_dtype flat
        RSAV: fx.Tensor,  # [N, C]     f32 flat
        DSAV: fx.Tensor,  # [N, C]     f32 flat
    ):
        f32 = T.f32
        elem_t = {"f32": T.f32, "bf16": T.bf16}[elem_dtype]
        fm = arith.FastMathFlags.fast
        vec1_f32 = T.vec(1, f32)

        br_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), BR)
        ps_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), PS)
        w_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), W)
        out_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), OUT)
        rsav_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), RSAV)
        dsav_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DSAV)

        base = allocator.get_base()
        lds_red = SmemPtr(base, lds_red_off, f32, shape=(RED_ROWS * BLOCK,)).get()

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

        def _load_elem_f32(bptr, elem_idx):
            raw = _llvm.LoadOp(elem_t, gep(bptr, elem_idx, elem_t)).result
            return arith.ExtFOp(f32, raw).result

        def _store_elem_f32(val, bptr, elem_idx):
            _llvm.StoreOp(arith.trunc_f(elem_t, val), gep(bptr, elem_idx, elem_t))

        # Build-time dtype choice by dict indexing, never by an `if` statement:
        # a Python `if` inside a traced body does not propagate its
        # branch-local rebindings out.
        load_elem = {"f32": load_f32, "bf16": _load_elem_f32}[elem_dtype]
        store_elem = {"f32": store_f32, "bf16": _store_elem_f32}[elem_dtype]

        def lds_get(elem_idx):
            v = vector.load_op(vec1_f32, lds_red, [elem_idx])
            return vector.extract(v, static_position=[0], dynamic_position=[])

        def lds_put(elem_idx, val):
            vector.store(vector.from_elements(vec1_f32, [val]), lds_red, [elem_idx])

        c_zero = arith.constant(0.0, type=f32)
        c_half = arith.constant(0.5, type=f32)
        c_three = arith.constant(3.0, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)
        c_inv_h = arith.constant(1.0 / float(H), type=f32)
        c_eps = arith.constant(EPSILON, type=f32)

        def nat_exp(x):
            return _nat_exp(x, c_log2e, fm)

        tok = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)
        I_BLOCK = arith.index(BLOCK)
        I_H = arith.index(H)
        I_C = arith.index(C)

        # Per-candidate base element offsets. The last candidate is the running
        # stream and lives behind a different pointer; the choice is made here,
        # at trace time, as a list comprehension.
        row_br = tok * arith.index(NB * H)
        row_ps = tok * I_H
        blocks = [(br_ptr, row_br + arith.index(c * H)) for c in range(NB)]
        stream = [(ps_ptr, row_ps)]
        # "the checkpoints, then the stream" is the reference's order
        # (modeling_kimi_linear.py:1082); `stream_first` inverts it.
        cand = {False: blocks + stream, True: stream + blocks}[inject == "stream_first"]

        # ---------------- phase 1: one pass over v, two reductions ------------
        # `w` is loaded once per element and reused across candidates, so the
        # scorer costs H reads per workgroup rather than C*H.
        acc_ss = [c_zero for _ in range(C)]
        acc_dot = [c_zero for _ in range(C)]
        for j in range_constexpr(EPT):
            d = tid + arith.index(j * BLOCK)
            wv = load_f32(w_ptr, d)
            for c in range_constexpr(C):
                x = load_elem(cand[c][0], cand[c][1] + d)
                acc_ss[c] = math_dialect.fma(x, x, acc_ss[c])
                acc_dot[c] = math_dialect.fma(x, wv, acc_dot[c])

        for c in range_constexpr(C):
            lds_put(arith.index((2 * c) * BLOCK) + tid, acc_ss[c])
            lds_put(arith.index((2 * c + 1) * BLOCK) + tid, acc_dot[c])
        gpu.barrier()

        for s in range_constexpr(len(RD_STRIDES)):
            partner = (tid + arith.index(RD_STRIDES[s])) % I_BLOCK
            sums = [
                arith.AddFOp(
                    lds_get(arith.index(row * BLOCK) + tid),
                    lds_get(arith.index(row * BLOCK) + partner),
                    fastmath=fm,
                ).result
                for row in range_constexpr(RED_ROWS)
            ]
            gpu.barrier()  # every read done before any write
            for row in range_constexpr(RED_ROWS):
                lds_put(arith.index(row * BLOCK) + tid, sums[row])
            gpu.barrier()  # every write done before the next read

        # ---------------- phase 2: scores and softmax, redundantly ------------
        # C <= 33 and every thread already has the totals, so recomputing the
        # softmax per thread is cheaper than broadcasting it back through LDS.
        # Build-time selections, by dict indexing rather than by an `if`:
        #   how rsqrt is computed, and whether the score uses it at all.
        rsqrt_of = {
            False: lambda x: _rsqrt_refined(x, c_three, c_half, fm),
            True: lambda x: rocdl.rsq(f32, x),
        }[inject == "no_newton"]
        score_of = {
            False: lambda dt, r: arith.MulFOp(dt, r, fastmath=fm).result,
            True: lambda dt, r: dt,
        }[inject == "score_unnormalised"]

        rs = []
        dots = []
        scores = []
        for c in range_constexpr(C):
            ss = lds_get(arith.index((2 * c) * BLOCK) + tid)
            dt = lds_get(arith.index((2 * c + 1) * BLOCK) + tid)
            mean_sq = arith.MulFOp(ss, c_inv_h, fastmath=fm).result
            r = rsqrt_of(arith.AddFOp(mean_sq, c_eps, fastmath=fm).result)
            rs.append(r)
            dots.append(dt)
            scores.append(score_of(dt, r))

        smax_real = scores[0]
        for c in range_constexpr(C - 1):
            smax_real = arith.MaxNumFOp(smax_real, scores[c + 1]).result
        smax = {False: smax_real, True: c_zero}[inject == "no_softmax_max"]
        exps = [nat_exp(arith.SubFOp(scores[c], smax, fastmath=fm).result) for c in range(C)]
        zsum = exps[0]
        for c in range_constexpr(C - 1):
            zsum = arith.AddFOp(zsum, exps[c + 1], fastmath=fm).result
        inv_z = arith.DivFOp(arith.constant(1.0, type=f32), zsum, fastmath=fm).result
        probs = [arith.MulFOp(exps[c], inv_z, fastmath=fm).result for c in range(C)]

        # Save `r` and `dot` for the backward. Thread `t` writes candidate
        # `t % C`: every thread holds bit-identical values (same LDS totals,
        # same arithmetic), so the duplicate stores from threads congruent mod C
        # write the same bytes. Branch-free, which keeps the body free of the
        # `scf.if` the AST rewriter is fragile around.
        save_c = tid % I_C
        sel_r = rs[0]
        sel_d = dots[0]
        for c in range_constexpr(C - 1):
            hit = arith.cmpi(arith.CmpIPredicate.eq, save_c, arith.index(c + 1))
            sel_r = arith.select(hit, rs[c + 1], sel_r)
            sel_d = arith.select(hit, dots[c + 1], sel_d)
        store_f32(sel_r, rsav_ptr, tok * I_C + save_c)
        store_f32(sel_d, dsav_ptr, tok * I_C + save_c)

        # ---------------- phase 3: second pass, the convex combination --------
        # The mixture is over the UN-normalised candidates: `probs[c]` alone.
        # `mix_normalised` folds `rs[c]` in, which is what mixing the
        # RMS-normalised `k` would amount to.
        mix_w = {
            False: probs,
            True: [arith.MulFOp(probs[c], rs[c], fastmath=fm).result for c in range(C)],
        }[inject == "mix_normalised"]
        for j in range_constexpr(EPT):
            d = tid + arith.index(j * BLOCK)
            acc = c_zero
            for c in range_constexpr(C):
                x = load_elem(cand[c][0], cand[c][1] + d)
                acc = math_dialect.fma(mix_w[c], x, acc)
            store_elem(acc, out_ptr, row_ps + d)

    @flyc.jit
    def launch_attn_res_mixer_fwd(
        BR: fx.Tensor,
        PS: fx.Tensor,
        W: fx.Tensor,
        OUT: fx.Tensor,
        RSAV: fx.Tensor,
        DSAV: fx.Tensor,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        grid_x = arith.index_cast(T.index, num_tokens)
        launcher = attn_res_mixer_fwd_kernel(BR, PS, W, OUT, RSAV, DSAV)

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
            return launch_attn_res_mixer_fwd(*args, **kwargs)

    return _launch
