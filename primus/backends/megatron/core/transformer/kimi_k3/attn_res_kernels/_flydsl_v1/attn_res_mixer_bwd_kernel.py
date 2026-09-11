###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused FlyDSL backward kernel for the Kimi K3 attention-residual mixer.

The analytic adjoint of :mod:`.attn_res_mixer_kernel`, not autograd through a
recomputation. Writing it by hand is worth it here for the same reason the
forward is: the eager backward is 2.1x the eager forward (measured 857 µs
against 404 µs at ``tokens = 4096, C = 4, hidden = 2048``) purely because
autograd replays every one of the six full-size intermediates and then adds its
own. The adjoint itself is short enough to be checked line by line against
``torch.autograd``, which ``test_attn_res_flydsl_kernel.py`` does.

The adjoint
-----------
With ``ms[c] = ss[c]/H``, ``r[c] = rsqrt(ms[c] + eps)``,
``s[c] = dot[c]·r[c]``, ``p = softmax(s)`` and
``out[d] = Σ_c p[c]·v[c,d]``::

    dp[c]    = <dout, v[c]>
    ds[c]    = p[c] · (dp[c] − Σ_c' p[c']·dp[c'])          softmax adjoint
    d_dot[c] = ds[c] · r[c]
    d_r[c]   = ds[c] · dot[c]
    d_ss[c]  = −0.5 · d_r[c] · r[c]³ / H
    dv[c,d]  = p[c]·dout[d]  +  d_dot[c]·w[d]  +  2·d_ss[c]·v[c,d]
    dW[d]    = Σ_{n,c} d_dot[n,c] · v[n,c,d]

``r`` and ``dot`` come from the forward's ``RSAV`` / ``DSAV`` (``[N, C]`` fp32,
64 KB at the scaled shape), so ``p`` is recovered without re-reducing anything
and the only reduction the backward performs is ``dp``. That makes it structurally
the same kernel as the forward: one pass to reduce, a tiny per-candidate middle,
one pass to write.

``dW`` is a reduction over **tokens**, i.e. across workgroups. Rather than
atomics on ``H`` heavily-contended addresses, or a grid restructure, the kernel
emits the per-token partial ``dw_tok[n,d] = Σ_c d_dot[n,c]·v[n,c,d]`` — free,
because ``d_dot[c]`` and ``v[c,d]`` are already in registers in the write pass —
and the caller finishes it with one ``sum(0)``. That keeps the whole path fp32
and deterministic, which a bf16 GEMV against ``v`` would not: the parameter
gradient would then be rounded to bf16 before it ever reaches Megatron's fp32
``main_grad``.

Layout
------
* ``BR``: ``[N, NB, H]``, ``PS``: ``[N, H]``, ``DOUT``: ``[N, H]`` — model dtype
* ``W``: ``[H]``, ``RSAV`` / ``DSAV``: ``[N, C]`` — fp32
* ``DBR``: ``[N, NB, H]``, ``DPS``: ``[N, H]`` — model dtype (they are the
  gradients of module inputs, so they follow the input dtype)
* ``DW_TOK``: ``[N, H]`` fp32, ``DDOT``: ``[N, C]`` fp32

Same geometry rules, same workgroup mapping and the same three FlyDSL tracing
rules as the forward; see that module's docstring.
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
from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._flydsl_v1.attn_res_mixer_kernel import (
    block_size_for,
    supports_mixer_geometry,
)

_LOG2E = math.log2(math.e)
_LLVM_GEP_DYNAMIC = -2147483648

#: Test-only defects the builder can emit, so the gradient parity test can be
#: shown to have discrimination power. Each is a plausible mistake in a
#: hand-derived adjoint, which is the whole reason to distrust one.
#:
#: ``no_softmax_jacobian``
#:     ``ds = p ⊙ dp`` instead of ``p ⊙ (dp − Σ p·dp)``. The classic softmax
#:     adjoint bug: right shape, right magnitude, wrong direction, and it
#:     vanishes whenever the probabilities are near-uniform.
#: ``drop_rms_term``
#:     Omit ``2·d_ss·v`` from ``dv``, i.e. forget that ``v`` also reaches the
#:     output through the RMS scale of its own score.
#: ``drop_score_term``
#:     Omit ``d_dot·w`` from ``dv``, i.e. forget the scoring path entirely.
#: ``wrong_r_power``
#:     ``r²`` instead of ``r³`` in ``d_ss``. A one-character slip that leaves
#:     every gradient finite and correlated with the truth.
BWD_INJECTIONS = (
    "no_softmax_jacobian",
    "drop_rms_term",
    "drop_score_term",
    "wrong_r_power",
)

__all__ = ["build_attn_res_mixer_bwd", "BWD_INJECTIONS"]


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _nat_exp(x, log2e_const, fastmath):
    """``exp(x)`` via the hardware ``v_exp_f32``; module scope, see the forward."""
    scaled = arith.MulFOp(x, log2e_const, fastmath=fastmath).result
    return rocdl.exp2(T.f32, scaled)


def build_attn_res_mixer_bwd(
    hidden: int,
    num_blocks: int,
    elem_dtype: str,
    waves_per_eu: int = 2,
    inject: str = "",
):
    """Build the backward launcher for one ``(hidden, num_blocks, dtype)``.

    ``eps`` is not a parameter: the backward consumes the forward's saved ``r``,
    so the epsilon is already folded into it. That is deliberate — it removes
    the one way the two kernels could silently disagree.

    Args:
        inject: **test only.** One of :data:`BWD_INJECTIONS`. Empty (the
            production value) emits the correct adjoint.

    Returns ``launch(BR, PS, W, DOUT, RSAV, DSAV, DBR, DPS, DW_TOK, DDOT,
    num_tokens)`` over flat tensors.
    """
    ensure_usable_lld()
    arch = get_rocm_arch()
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"attn_res_mixer_bwd targets gfx950 (CDNA4); got {arch!r}")

    H = int(hidden)
    NB = int(num_blocks)
    C = NB + 1
    reason = supports_mixer_geometry(H, C)
    if reason is not None:
        raise ValueError(f"attn_res_mixer_bwd cannot run this geometry: {reason}")
    if elem_dtype not in ("f32", "bf16"):
        raise ValueError(f"elem_dtype must be 'f32' or 'bf16'; got {elem_dtype!r}")
    if inject and inject not in BWD_INJECTIONS:
        raise ValueError(f"unknown injection {inject!r}; expected '' or one of {list(BWD_INJECTIONS)}")

    BLOCK = block_size_for(H)
    EPT = H // BLOCK

    # One reduction row per candidate: dp[c] only. r and dot are read back from
    # the forward's saved tensors.
    RED_ROWS = C
    allocator = SmemAllocator(None, arch=arch, global_sym_name=f"attnres_bwd_smem_H{H}_C{C}_{elem_dtype}")
    lds_red_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_red_off + RED_ROWS * BLOCK * 4

    RD_STRIDES = [1 << i for i in range(int(math.log2(BLOCK)))]
    assert (1 << len(RD_STRIDES)) == BLOCK, "recursive doubling needs a power-of-two block"

    _tag = f"_inj_{inject}" if inject else ""

    @flyc.kernel(
        known_block_size=[BLOCK, 1, 1],
        name=f"attn_res_mixer_bwd_{elem_dtype}_H{H}_C{C}{_tag}",
    )
    def attn_res_mixer_bwd_kernel(
        BR: fx.Tensor,
        PS: fx.Tensor,
        W: fx.Tensor,
        DOUT: fx.Tensor,
        RSAV: fx.Tensor,
        DSAV: fx.Tensor,
        DBR: fx.Tensor,
        DPS: fx.Tensor,
        DW_TOK: fx.Tensor,
        DDOT: fx.Tensor,
    ):
        f32 = T.f32
        elem_t = {"f32": T.f32, "bf16": T.bf16}[elem_dtype]
        fm = arith.FastMathFlags.fast
        vec1_f32 = T.vec(1, f32)

        br_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), BR)
        ps_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), PS)
        w_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), W)
        do_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DOUT)
        rsav_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), RSAV)
        dsav_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DSAV)
        dbr_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DBR)
        dps_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DPS)
        dwt_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DW_TOK)
        ddot_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DDOT)

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

        load_elem = {"f32": load_f32, "bf16": _load_elem_f32}[elem_dtype]
        store_elem = {"f32": store_f32, "bf16": _store_elem_f32}[elem_dtype]

        def lds_get(elem_idx):
            v = vector.load_op(vec1_f32, lds_red, [elem_idx])
            return vector.extract(v, static_position=[0], dynamic_position=[])

        def lds_put(elem_idx, val):
            vector.store(vector.from_elements(vec1_f32, [val]), lds_red, [elem_idx])

        c_zero = arith.constant(0.0, type=f32)
        c_fzero = c_zero
        c_one = arith.constant(1.0, type=f32)
        c_two = arith.constant(2.0, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)
        c_neg_half_inv_h = arith.constant(-0.5 / float(H), type=f32)

        def nat_exp(x):
            return _nat_exp(x, c_log2e, fm)

        tok = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)
        I_BLOCK = arith.index(BLOCK)
        I_H = arith.index(H)
        I_C = arith.index(C)

        row_br = tok * arith.index(NB * H)
        row_ps = tok * I_H
        # (source pointer, source base, gradient pointer, gradient base) per
        # candidate. The last candidate is the running stream; the choice of
        # pointer is made here at trace time, never by an `if` in the body.
        cand = [
            (br_ptr, row_br + arith.index(c * H), dbr_ptr, row_br + arith.index(c * H)) for c in range(NB)
        ]
        cand = cand + [(ps_ptr, row_ps, dps_ptr, row_ps)]

        # ---------------- phase 1: dp[c] = <dout, v[c]> -----------------------
        acc_dp = [c_zero for _ in range(C)]
        for j in range_constexpr(EPT):
            d = tid + arith.index(j * BLOCK)
            dg = load_elem(do_ptr, row_ps + d)
            for c in range_constexpr(C):
                x = load_elem(cand[c][0], cand[c][1] + d)
                acc_dp[c] = math_dialect.fma(dg, x, acc_dp[c])

        for c in range_constexpr(C):
            lds_put(arith.index(c * BLOCK) + tid, acc_dp[c])
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
            gpu.barrier()
            for row in range_constexpr(RED_ROWS):
                lds_put(arith.index(row * BLOCK) + tid, sums[row])
            gpu.barrier()

        # ---------------- phase 2: the per-candidate adjoint ------------------
        # `p` is recovered from the forward's saved r and dot, so the softmax is
        # recomputed exactly as the forward computed it -- including the same
        # max subtraction, which is what makes the two agree bit-for-bit.
        rs = [load_f32(rsav_ptr, tok * I_C + arith.index(c)) for c in range(C)]
        dots = [load_f32(dsav_ptr, tok * I_C + arith.index(c)) for c in range(C)]
        scores = [arith.MulFOp(dots[c], rs[c], fastmath=fm).result for c in range(C)]
        smax = scores[0]
        for c in range_constexpr(C - 1):
            smax = arith.MaxNumFOp(smax, scores[c + 1]).result
        exps = [nat_exp(arith.SubFOp(scores[c], smax, fastmath=fm).result) for c in range(C)]
        zsum = exps[0]
        for c in range_constexpr(C - 1):
            zsum = arith.AddFOp(zsum, exps[c + 1], fastmath=fm).result
        inv_z = arith.DivFOp(c_one, zsum, fastmath=fm).result
        probs = [arith.MulFOp(exps[c], inv_z, fastmath=fm).result for c in range(C)]

        dps_red = [lds_get(arith.index(c * BLOCK) + tid) for c in range(C)]
        dot_p = arith.MulFOp(probs[0], dps_red[0], fastmath=fm).result
        for c in range_constexpr(C - 1):
            dot_p = math_dialect.fma(probs[c + 1], dps_red[c + 1], dot_p)

        # Build-time selections, by dict indexing rather than by an `if`.
        ds_of = {
            False: lambda c: arith.MulFOp(
                probs[c], arith.SubFOp(dps_red[c], dot_p, fastmath=fm).result, fastmath=fm
            ).result,
            True: lambda c: arith.MulFOp(probs[c], dps_red[c], fastmath=fm).result,
        }[inject == "no_softmax_jacobian"]
        r_pow_of = {
            False: lambda r, r2: arith.MulFOp(r2, r, fastmath=fm).result,
            True: lambda r, r2: r2,
        }[inject == "wrong_r_power"]

        d_dot = []
        two_d_ss = []
        for c in range_constexpr(C):
            ds = ds_of(c)
            d_dot.append(arith.MulFOp(ds, rs[c], fastmath=fm).result)
            # d_ss = -0.5/H * ds * dot * r^3, and the write pass needs 2*d_ss
            d_r = arith.MulFOp(ds, dots[c], fastmath=fm).result
            r2 = arith.MulFOp(rs[c], rs[c], fastmath=fm).result
            r3 = r_pow_of(rs[c], r2)
            d_ss = arith.MulFOp(
                arith.MulFOp(d_r, r3, fastmath=fm).result, c_neg_half_inv_h, fastmath=fm
            ).result
            two_d_ss.append(arith.MulFOp(c_two, d_ss, fastmath=fm).result)

        # The three terms of `dv`, each independently switchable off. `dw_acc`
        # keeps the true `d_dot` regardless, so a dropped term shows up in the
        # input gradients rather than being masked by a matching change in dW.
        rms_w = {False: two_d_ss, True: [c_fzero for _ in range(C)]}[inject == "drop_rms_term"]
        score_w = {False: d_dot, True: [c_fzero for _ in range(C)]}[inject == "drop_score_term"]

        # Same branch-free duplicate-store trick as the forward's RSAV/DSAV.
        save_c = tid % I_C
        sel = d_dot[0]
        for c in range_constexpr(C - 1):
            hit = arith.cmpi(arith.CmpIPredicate.eq, save_c, arith.index(c + 1))
            sel = arith.select(hit, d_dot[c + 1], sel)
        store_f32(sel, ddot_ptr, tok * I_C + save_c)

        # ---------------- phase 3: the input gradients ------------------------
        for j in range_constexpr(EPT):
            d = tid + arith.index(j * BLOCK)
            dg = load_elem(do_ptr, row_ps + d)
            wv = load_f32(w_ptr, d)
            dw_acc = c_zero
            for c in range_constexpr(C):
                x = load_elem(cand[c][0], cand[c][1] + d)
                dv = math_dialect.fma(probs[c], dg, arith.MulFOp(score_w[c], wv, fastmath=fm).result)
                dv = math_dialect.fma(rms_w[c], x, dv)
                store_elem(dv, cand[c][2], cand[c][3] + d)
                dw_acc = math_dialect.fma(d_dot[c], x, dw_acc)
            store_f32(dw_acc, dwt_ptr, row_ps + d)

    @flyc.jit
    def launch_attn_res_mixer_bwd(
        BR: fx.Tensor,
        PS: fx.Tensor,
        W: fx.Tensor,
        DOUT: fx.Tensor,
        RSAV: fx.Tensor,
        DSAV: fx.Tensor,
        DBR: fx.Tensor,
        DPS: fx.Tensor,
        DW_TOK: fx.Tensor,
        DDOT: fx.Tensor,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        grid_x = arith.index_cast(T.index, num_tokens)
        launcher = attn_res_mixer_bwd_kernel(BR, PS, W, DOUT, RSAV, DSAV, DBR, DPS, DW_TOK, DDOT)

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
            return launch_attn_res_mixer_bwd(*args, **kwargs)

    return _launch
