###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native FlyDSL kernel for KDA's inter-chunk state sweep.

Profiling measured this stage at **7902 µs of a 6199 µs forward** at production
geometry — 95 %, against 340 µs for the score kernel — because it was 64
serialised iterations of a Python ``for`` loop over batched torch GEMMs far too
small to fill an MI355X, with the ``[K, V]`` state round-tripping through HBM on
every one of them. This kernel runs the whole sweep in
**one launch** with the state resident on-chip.

The recurrence
--------------
Both the forward sweep and its adjoint are first-order linear recurrences in a
``[K, V]`` state with the *same* shape, so one kernel serves both::

    T_n      = Yc_n + SGN_T * (A_n @ S_n)            # [C, V]
    S_{n+1}  = dec_n * S_n + SGN_X * (Xt_n @ T_n)    # [K, V]   (+ E_n)

``dec_n`` is a per-``k`` scalar, broadcast along ``V``.

===============  ==========================  ==========================
                 forward                     backward
===============  ==========================  ==========================
state            ``S_n``                     ``dS_{n+1}``
``A``            ``W = M(G*K)``              ``KG``
``Yc``           ``U = M V``                 ``Aqk^T dO``
``Xt``           ``KG^T``                    ``W^T``
``SGN_T``        ``-1``                      ``+1``
``SGN_X``        ``+1``                      ``-1``
``E``            --                          ``QG^T dO``
order            ``n = 0 .. NC-1``           ``n = NC-1 .. 0``
``T_n`` is       ``vtilde_n``                ``d vtilde_n``
===============  ==========================  ==========================

On the forward the ``A`` operand is the **stacked** ``[QG; W]`` of shape
``[2C, K]``, so the same group also produces ``Rq_n = QG_n @ S_n`` — the only
other term the chunk output needs. Everything left outside is then per-chunk and
embarrassingly parallel: ``O_n = Aqk_n @ T_n + Rq_n`` and the five input
adjoints are plain batched GEMMs over all ``NC`` chunks at once, with no
recurrence. That is the whole point of the split: **only what is genuinely
sequential runs sequentially.**

Grid and residency
------------------
One workgroup per ``(B*H, V/BV)`` — the shape ``fla``'s ``chunk_h`` uses — and
the chunk loop lives *inside* the workgroup, so the state never leaves the CU.
``BV`` is chosen by measurement in :func:`..sweep._pick_block_v`, and the
measurement contradicted the obvious reasoning: **16 beats 64 by 1.5×** even
though it re-reads the state-independent operands four times more, because at
64 the grid is one workgroup per CU and therefore one wave per SIMD.
It lives in **LDS**, not in registers: MFMA's accumulator layout
(``m = (lane//16)*4 + slot``, ``n = lane%16``) is not its operand layout
(``row = lane%16``, ``k = (lane//16)*8 + j``), so a state that is both
accumulated into and read back as an operand has to be re-laid-out between the
two, and LDS is the cheapest place to do it. The HBM round trip — the actual
cost — is gone either way.

Two arithmetic modes, chosen by the caller's dtype
--------------------------------------------------
``"mfma"``
    ``v_mfma_f32_16x16x32_bf16``, bf16 operands and fp32 accumulate — what
    ``fla`` does, and the fast path. The ``A``/``Xt`` fragments come straight out
    of global memory (``row = lane%16`` with 8 contiguous ``k`` is exactly one
    16 B ``dwordx4``); the state fragment comes out of LDS.
``"valu"``
    plain fp32 FMA, ``VPT`` columns of ``V`` per thread. Slower, and the reason
    it exists is that the fp32 parity test has to exercise *this* kernel rather
    than a different code path. ``v_mfma_f32_16x16x4f32`` is present in the
    installed ``flydsl.expr.rocdl`` but SIGABRTs inside an
    ``llvm::cast<VectorType>`` however its ``cbsz/abid/blgp`` flags are spelled
    (i32 Values or Python ints), so no fp32 MFMA is usable here.

Two FlyDSL tracing rules shape the code below, both re-confirmed here:

* ``rocdl`` is not resolvable from a helper defined *inside* a traced body, so
  :func:`_mfma` sits at module scope and is merely called from the body.
* a build-time ``if`` whose branch contains an MFMA is the exact pattern that
  kills the AST rewriter (``'NoneType' object has no attribute '_CAPIPtr'``
  inside ``_unwrap_mfma_operand``; the in-tree DSv4 forward still dies of it).
  So *every* build-time choice here is made by indexing a dict of closures, not
  by an ``if`` statement.

gfx950 / CDNA4.
"""

from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import math as math_dialect
from flydsl._mlir.dialects import scf
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

_LLVM_GEP_DYNAMIC = -2147483648  # LLVM kDynamicIndex sentinel

BLOCK_SIZE = 256
WAVE = 64
NWAVES = BLOCK_SIZE // WAVE

# v_mfma_f32_16x16x32_bf16, lane layout ground-truthed empirically
MI_TILE = 16
MI_K = 32
LANE_K = 8  # operand elements per lane per instruction
ACC = 4  # accumulator elements per lane

VPT = 4  # "valu" mode: V columns per thread

MODES = ("mfma", "valu")

# Measurement-only variants of the chunk loop's inner contraction. **Every one of
# them computes the wrong answer** — they exist so the loop's cost can be
# attributed by deletion instead of by argument, which is how the two errors in
# `bench/FINDINGS.md` §5c were found. Only `"full"` is reachable from any
# production path; `bench/probe_sweep_bound.py` is the sole caller of the others.
#
# ``lds1``  the MFMA B fragment is built from **one** LDS read replicated
#           ``LANE_K`` times instead of ``LANE_K`` strided reads. MFMA count,
#           global loads and loop structure are untouched, so the delta is the
#           b-fragment LDS traffic and nothing else.
# ``areuse`` every ``ks`` step loads the **same** A fragment, so LICM collapses
#           ``klen`` global loads into one. The delta is the A operand's global
#           traffic.
# ``nobar``  the two barriers are removed from the chunk loop. This races (the
#           state and T are exchanged through LDS), so the answer is wrong -- but
#           `fla` carries its state in *registers* across the whole loop and so
#           has no barrier in it at all, and this is what that difference is worth.
# ``nostore`` the Rq and T global stores are dropped (T still reaches LDS, so the
#           recurrence still runs). Prices the loop's write traffic.
# ``noy``    the Yc load is replaced by a constant. Prices the loop's read traffic
#           on the one operand that is neither an MFMA operand nor the state.
PROBES = ("full", "lds1", "areuse", "nobar", "nostore", "noy")

__all__ = ["build_kda_state_sweep", "BLOCK_SIZE", "MODES", "supports_sweep_geometry"]


def _mfma(acc_t, a_frag, b_frag, acc):
    """One ``v_mfma_f32_16x16x32_bf16``. Module scope: see the module docstring."""
    return rocdl.mfma_f32_16x16x32_bf16(acc_t, [a_frag, b_frag, acc])


def supports_sweep_geometry(chunk_size: int, k_dim: int, v_dim: int, block_v: int = 64) -> Optional[str]:
    """``None`` when the kernel can run this geometry, else why it cannot."""
    for name, val in (("chunk_size", chunk_size), ("head_dim", k_dim), ("v_head_dim", v_dim)):
        if val % MI_TILE != 0:
            return f"{name}={val} is not a multiple of the {MI_TILE}-row MFMA tile"
    if v_dim % block_v != 0:
        return f"v_head_dim={v_dim} is not a multiple of block_v={block_v}"
    if block_v % MI_TILE != 0 or block_v % VPT != 0:
        return f"block_v={block_v} is not a multiple of {MI_TILE} and {VPT}"
    nvl = block_v // VPT
    if BLOCK_SIZE % nvl != 0:
        return f"block_v={block_v} does not split {BLOCK_SIZE} threads into whole V groups"
    if (chunk_size // MI_TILE) % NWAVES != 0 or (k_dim // MI_TILE) % NWAVES != 0:
        return (
            f"chunk_size={chunk_size} / head_dim={k_dim} do not give a whole number of "
            f"{MI_TILE}-row tiles per wave ({NWAVES} waves)"
        )
    ng = BLOCK_SIZE // nvl
    if chunk_size % ng != 0 or k_dim % ng != 0:
        return f"chunk_size={chunk_size} / head_dim={k_dim} are not multiples of {ng}"
    if chunk_size % MI_K != 0 or k_dim % MI_K != 0:
        return f"chunk_size={chunk_size} / head_dim={k_dim} are not multiples of {MI_K}"
    return None


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def build_kda_state_sweep(
    chunk_size: int,
    k_dim: int,
    v_dim: int,
    block_v: int = 64,
    mode: str = "mfma",
    emit_rq: bool = True,
    emit_states: bool = True,
    has_e: bool = False,
    sgn_t: float = -1.0,
    sgn_x: float = 1.0,
    reverse: bool = False,
    waves_per_eu: int = 1,
    probe: str = "full",
):
    """Build the launcher for one sweep configuration.

    Returns a callable
    ``launch(Amat, Yc, Xt, Dec, Ein, S0, Rq, Tout, Sout, Sf, nbh, nc)``
    over flat tensors. Only ``Amat`` and ``Xt`` — the two MFMA **A** operands —
    are in the operand dtype (bf16 for ``mode="mfma"``, fp32 for ``"valu"``);
    everything else is fp32. ``Yc``, ``Rq`` and ``Tout`` are on the direct path
    to the chunk output and are never fed to an MFMA, so rounding them would
    cost accuracy for nothing: keeping them fp32 was measured to take the bf16
    output error from 4.5e-3 back to 2.6e-3.

    Shapes, with ``NB = B*H*NC`` and ``MO = 2*C`` when ``emit_rq`` else ``C``:

    ==========  ====================  ==========================================
    ``Amat``    ``[NB, MO, K]``       stacked ``[QG; W]`` on the forward; op dtype
    ``Yc``      ``[NB, C, V]``
    ``Xt``      ``[NB, K, C]``        op dtype
    ``Dec``     ``[NB, K]``
    ``Ein``     ``[NB, K, V]``        1 element when ``has_e`` is false
    ``S0``      ``[B*H, K, V]``
    ``Rq``      ``[NB, C, V]``        1 element when ``emit_rq`` is false
    ``Tout``    ``[NB, C, V]``
    ``Sout``    ``[NB, K, V]``        1 element when ``emit_states`` is false
    ``Sf``      ``[B*H, K, V]``
    ==========  ====================  ==========================================
    """
    ensure_usable_lld()
    arch = get_rocm_arch()
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"kda_state_sweep targets gfx950 (CDNA4); got {arch!r}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}")
    if probe not in PROBES:
        raise ValueError(f"probe must be one of {PROBES}; got {probe!r}")

    C, KD, VD, BV = int(chunk_size), int(k_dim), int(v_dim), int(block_v)
    reason = supports_sweep_geometry(C, KD, VD, BV)
    if reason is not None:
        raise ValueError(f"the fused KDA state sweep cannot run this geometry: {reason}")

    NVB = VD // BV  # V blocks per (b, h)
    MO_A = 2 * C if emit_rq else C  # rows of the Amat operand
    A_ROW_T = C if emit_rq else 0  # first Amat row feeding the T phase
    NNT = BV // MI_TILE  # MFMA output column tiles
    NVL = BV // VPT  # VALU: threads spanning one V block
    NG = BLOCK_SIZE // NVL  # VALU: concurrent output rows
    SPILL = (KD * BV) // BLOCK_SIZE  # LDS<->global state copy steps

    # Per-mode loop extents: how many "row units" a group covers and how long
    # its contraction is. Resolved here, at build scope, so the traced body needs
    # no branch on the mode.
    if mode == "mfma":
        G1_ROWS, G1_LEN = C // MI_TILE, KD // MI_K
        G2_ROWS, G2_LEN = KD // MI_TILE, C // MI_K
    else:
        G1_ROWS, G1_LEN = C, KD
        G2_ROWS, G2_LEN = KD, C
    # (Amat row offset, sink name) for each phase of group 1.
    G1_PHASES = [(0, "rq"), (A_ROW_T, "t")] if emit_rq else [(0, "t")]

    # LDS: the state and T, both with V contiguous so a 16-lane group covers one
    # whole [*, BV] row without a bank conflict.
    allocator = SmemAllocator(None, arch=arch, global_sym_name=f"kda_sweep_smem_C{C}_K{KD}_V{BV}_{mode}_{probe}")
    lds_s_off = allocator._align(allocator.ptr, 16)
    lds_t_off = allocator._align(lds_s_off + KD * BV * 4, 16)
    allocator.ptr = lds_t_off + C * BV * 4

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1], name=f"kda_state_sweep_{mode}_{probe}")
    def kda_state_sweep_kernel(
        Amat: fx.Tensor,
        Yc: fx.Tensor,
        Xt: fx.Tensor,
        Dec: fx.Tensor,
        Ein: fx.Tensor,
        S0: fx.Tensor,
        Rq: fx.Tensor,
        Tout: fx.Tensor,
        Sout: fx.Tensor,
        Sf: fx.Tensor,
        nc: fx.Int32,
    ):
        f32 = T.f32
        op_t = T.bf16 if mode == "mfma" else T.f32
        fm = arith.FastMathFlags.fast
        vec1_f32 = T.vec(1, f32)
        vecv_f32 = T.vec(VPT, f32)
        acc_t = T.vec(ACC, f32)
        frag_t = T.vec(LANE_K, op_t)

        a_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Amat)
        y_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Yc)
        x_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Xt)
        d_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Dec)
        e_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Ein)
        s0_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), S0)
        rq_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Rq)
        t_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Tout)
        so_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Sout)
        sf_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Sf)

        base = allocator.get_base()
        lds_s = SmemPtr(base, lds_s_off, f32, shape=(KD * BV,)).get()
        lds_t = SmemPtr(base, lds_t_off, f32, shape=(C * BV,)).get()

        # ------------------------------- memory -------------------------------
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

        def load_frag(bptr, elem_idx):
            """``LANE_K`` contiguous operand-dtype elements, as one MFMA fragment."""
            return _llvm.LoadOp(frag_t, gep(bptr, elem_idx, op_t)).result

        def load_vec4(bptr, elem_idx):
            """Four contiguous f32 in one 16 B load ("valu" mode only)."""
            return _llvm.LoadOp(vecv_f32, gep(bptr, elem_idx, f32)).result

        def lds_get(lds, elem_idx):
            v = vector.load_op(vec1_f32, lds, [elem_idx])
            return vector.extract(v, static_position=[0], dynamic_position=[])

        def lds_put(lds, elem_idx, val):
            vector.store(vector.from_elements(vec1_f32, [val]), lds, [elem_idx])

        # --------------------------------- ids --------------------------------
        bid = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)
        nc_i = arith.index_cast(T.index, nc)

        I_BV, I_VD, I_KD, I_C, I_MOA = (arith.index(x) for x in (BV, VD, KD, C, MO_A))
        I_MI, I_ONE = arith.index(MI_TILE), arith.index(1)

        bh = bid // arith.index(NVB)
        vb = bid % arith.index(NVB)
        v0 = vb * I_BV  # first V column this workgroup owns
        nb0 = bh * nc_i  # this (b, h)'s first chunk row in the [NB, ...] tensors

        lane = tid % arith.index(WAVE)
        wave = tid // arith.index(WAVE)
        frag_row = lane % I_MI  # operand row  = lane % 16
        kgrp = lane // I_MI  # operand k group = lane // 16
        acc_m0 = kgrp * arith.index(ACC)  # accumulator row base
        lvq = tid % arith.index(NVL)
        lg = tid // arith.index(NVL)
        v_lo = lvq * arith.index(VPT)  # first V column this thread owns (valu)

        c_sgn_t = arith.constant(float(sgn_t), type=f32)
        c_sgn_x = arith.constant(float(sgn_x), type=f32)

        # ------------------------------- sinks --------------------------------
        # Called once per accumulator element, so both arithmetic modes share
        # every store. `row` is an output row; `v_lds` a column within the block.
        #
        # Each sink comes with a `prefetch` that issues the **global loads it will
        # need** and returns them. The group functions call every prefetch before
        # entering the accumulation loop, which is the whole point: `T` needs
        # `Yc[nb, row, v]` and the state needs `Dec[nb, kk]`, neither depends on
        # the accumulator, and issuing them at the point of use put a full HBM
        # latency on the critical path *twice per chunk* — on a 64-step serial
        # recurrence with only three workgroups per CU, with nothing to hide it.
        # `fla` reaches 234 µs here against our 542 with **four times fewer**
        # workgroups, so the gap was never parallelism.
        def pre_none(nb, row, v_lds):
            return None

        _c_one_f = arith.constant(1.0, type=f32)

        def pre_y(nb, row, v_lds):
            return {
                True: lambda: _c_one_f,
                False: lambda: load_f32(y_ptr, (nb * I_C + row) * I_VD + v0 + v_lds),
            }[probe == "noy"]()

        def pre_dec(nb, kk, v_lds):
            return load_f32(d_ptr, nb * I_KD + kk)

        _ST = {True: lambda v, p, e: None, False: store_f32}[probe == "nostore"]

        def sink_rq(nb, row, v_lds, val, pre):
            _ST(val, rq_ptr, (nb * I_C + row) * I_VD + v0 + v_lds)

        def sink_t(nb, row, v_lds, val, pre):
            off = (nb * I_C + row) * I_VD + v0 + v_lds
            tv = math_dialect.fma(c_sgn_t, val, pre)
            _ST(tv, t_ptr, off)
            lds_put(lds_t, row * I_BV + v_lds, tv)

        def _state_core(nb, kk, v_lds, val, dec):
            li = kk * I_BV + v_lds
            return li, math_dialect.fma(
                dec, lds_get(lds_s, li), arith.MulFOp(c_sgn_x, val, fastmath=fm).result
            )

        def _sink_state_plain(nb, kk, v_lds, val, pre):
            li, upd = _state_core(nb, kk, v_lds, val, pre)
            lds_put(lds_s, li, upd)

        def _sink_state_with_e(nb, kk, v_lds, val, pre):
            li, upd = _state_core(nb, kk, v_lds, val, pre)
            e = load_f32(e_ptr, (nb * I_KD + kk) * I_VD + v0 + v_lds)
            lds_put(lds_s, li, arith.AddFOp(upd, e, fastmath=fm).result)

        sink_state = {True: _sink_state_with_e, False: _sink_state_plain}[has_e]
        _SINKS = {"rq": (sink_rq, pre_none), "t": (sink_t, pre_y)}
        G1_BLOCKS = [(r0, *_SINKS[w]) for r0, w in G1_PHASES]

        # ------------------- operand builds, and the probe variants -------------
        def _b_full(lds_src, k0, v_lds):
            return vector.from_elements(
                frag_t,
                [
                    arith.trunc_f(op_t, lds_get(lds_src, (k0 + arith.index(j)) * I_BV + v_lds))
                    for j in range_constexpr(LANE_K)
                ],
            )

        def _b_one_read(lds_src, k0, v_lds):
            """MEASUREMENT ONLY — wrong answer. One LDS read, replicated."""
            one = arith.trunc_f(op_t, lds_get(lds_src, k0 * I_BV + v_lds))
            return vector.from_elements(frag_t, [one for _ in range_constexpr(LANE_K)])

        _BUILD_B = {"full": _b_full, "lds1": _b_one_read, "areuse": _b_full, "nobar": _b_full, "nostore": _b_full, "noy": _b_full}[probe]
        # `areuse` makes the A fragment's address loop-invariant so LICM hoists it.
        _A_K = {
            "full": lambda k0: k0,
            "lds1": lambda k0: k0,
            "areuse": lambda k0: kgrp * arith.index(LANE_K),
            "nobar": lambda k0: k0,
            "nostore": lambda k0: k0,
            "noy": lambda k0: k0,
        }[probe]

        # ------------------------------ contraction ----------------------------
        # `blocks` is a list of (row offset into the A operand, sink); they share
        # the state, so they also share every B fragment. On the forward that
        # halves group 1's LDS traffic, because Rq and T contract the *same*
        # state against two stacked row blocks of the same operand.
        def group_mfma(nb, ap, a_rows, a_inner, nrow_units, klen, lds_src, blocks):
            nblk = len(blocks)
            for mi in range_constexpr(nrow_units // NWAVES):
                mt = wave + arith.index(mi * NWAVES)
                row_f = mt * I_MI + frag_row
                a_base = [(nb * a_rows + arith.index(r0) + row_f) * a_inner for r0, _, _ in blocks]
                for nt in range_constexpr(NNT):
                    v_lds = arith.index(nt * MI_TILE) + frag_row  # acc column = lane % 16
                    # every global load the sinks will need, issued *before* the
                    # accumulation so its latency hides behind the MFMAs
                    pre = [
                        [
                            blocks[b][2](nb, mt * I_MI + acc_m0 + arith.index(s), v_lds)
                            for s in range_constexpr(ACC)
                        ]
                        for b in range_constexpr(nblk)
                    ]
                    acc = [arith.constant_vector(0.0, acc_t) for _ in range_constexpr(nblk)]
                    for ks in range_constexpr(klen):
                        k0 = arith.index(ks * MI_K) + kgrp * arith.index(LANE_K)
                        b_frag = _BUILD_B(lds_src, k0, v_lds)
                        for b in range_constexpr(nblk):
                            acc[b] = _mfma(acc_t, load_frag(ap, a_base[b] + _A_K(k0)), b_frag, acc[b])
                    for b in range_constexpr(nblk):
                        for s in range_constexpr(ACC):
                            blocks[b][1](
                                nb,
                                mt * I_MI + acc_m0 + arith.index(s),
                                v_lds,
                                vector.extract(acc[b], static_position=[s], dynamic_position=[]),
                                pre[b][s],
                            )

        def group_valu(nb, ap, a_rows, a_inner, nrow_units, klen, lds_src, blocks):
            # Register-blocked over *every* output row the thread owns, across all
            # row blocks, so one LDS read of the state feeds `RPT * len(blocks)`
            # rows. Without this the loop reads LDS once per FMA quad and is 8x
            # LDS-bound; with it a thread does 32 FMAs per LDS read.
            rpt = nrow_units // NG
            keys = [(bi, ri) for bi in range(len(blocks)) for ri in range(rpt)]
            row_of = {}
            a_base = {}
            for bi, ri in keys:
                row_of[(bi, ri)] = lg + arith.index(ri * NG)
                a_base[(bi, ri)] = (nb * a_rows + arith.index(blocks[bi][0]) + row_of[(bi, ri)]) * a_inner
            # as in `group_mfma`: the sinks' global loads go out before the
            # accumulation, not at the point of use
            pre = {
                (bi, ri, j): blocks[bi][2](nb, row_of[(bi, ri)], v_lo + arith.index(j))
                for bi, ri in keys
                for j in range(VPT)
            }
            acc = {key: [arith.constant(0.0, type=f32) for _ in range(VPT)] for key in keys}
            for k4 in range_constexpr(klen // VPT):
                a4 = {key: load_vec4(ap, a_base[key] + arith.index(k4 * VPT)) for key in keys}
                for jj in range_constexpr(VPT):
                    s4 = vector.load_op(vecv_f32, lds_src, [arith.index((k4 * VPT + jj) * BV) + v_lo])
                    sv = [vector.extract(s4, static_position=[j], dynamic_position=[]) for j in range(VPT)]
                    for key in keys:
                        a_v = vector.extract(a4[key], static_position=[jj], dynamic_position=[])
                        for j in range(VPT):
                            acc[key][j] = math_dialect.fma(a_v, sv[j], acc[key][j])
            for bi, ri in keys:
                for j in range_constexpr(VPT):
                    blocks[bi][1](
                        nb,
                        row_of[(bi, ri)],
                        v_lo + arith.index(j),
                        acc[(bi, ri)][j],
                        pre[(bi, ri, j)],
                    )

        group = {"mfma": group_mfma, "valu": group_valu}[mode]

        # ------------------------- state <-> global ---------------------------
        def copy_state(bptr, row_base, scale, to_lds):
            for i in range_constexpr(SPILL):
                li = tid + arith.index(i * BLOCK_SIZE)
                off = (row_base * scale + li // I_BV) * I_VD + v0 + li % I_BV
                to_lds(bptr, li, off)

        def _in(bptr, li, off):
            lds_put(lds_s, li, load_f32(bptr, off))

        def _out(bptr, li, off):
            store_f32(lds_get(lds_s, li), bptr, off)

        def _store_states(nb):
            copy_state(so_ptr, nb, I_KD, _out)

        def _skip_states(nb):
            return None

        store_states = {True: _store_states, False: _skip_states}[emit_states]
        chunk_of = {True: lambda it: nc_i - I_ONE - it, False: lambda it: it}[bool(reverse)]

        copy_state(s0_ptr, bh, I_KD, _in)

        # ==================== the sequential chunk loop ======================
        _BAR = {True: lambda: None, False: gpu.barrier}[probe == "nobar"]
        for it in scf.for_(arith.index(0), nc_i, I_ONE):
            nb = nb0 + chunk_of(it)
            _BAR()
            store_states(nb)
            group(nb, a_ptr, I_MOA, I_KD, G1_ROWS, G1_LEN, lds_s, G1_BLOCKS)
            _BAR()
            group(nb, x_ptr, I_KD, I_C, G2_ROWS, G2_LEN, lds_t, [(0, sink_state, pre_dec)])

        gpu.barrier()
        copy_state(sf_ptr, bh, I_KD, _out)

    @flyc.jit
    def launch_kda_state_sweep(
        Amat: fx.Tensor,
        Yc: fx.Tensor,
        Xt: fx.Tensor,
        Dec: fx.Tensor,
        Ein: fx.Tensor,
        S0: fx.Tensor,
        Rq: fx.Tensor,
        Tout: fx.Tensor,
        Sout: fx.Tensor,
        Sf: fx.Tensor,
        nbh: fx.Int32,
        nc: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        grid_x = arith.index_cast(T.index, nbh) * arith.index(NVB)
        launcher = kda_state_sweep_kernel(Amat, Yc, Xt, Dec, Ein, S0, Rq, Tout, Sout, Sf, nc)
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
            return launch_kda_state_sweep(*args, **kwargs)

    return with_current_stream(_launch)
