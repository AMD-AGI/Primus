###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################
#
# --- Vendored into Primus -------------------------------------------------
# Source: primus_turbo 0.3.1.dev183,
#         primus_turbo/flydsl/utils/attn_helper.py
# Copied verbatim and then modified for MiniMax Sparse Attention (MSA): the
# dense kernel walks a contiguous span of KV tiles, MSA walks an arbitrary list
# of them taken from a per-query block table. See README.md in this directory
# for what changed and why this is a copy rather than a subclass.
#
# Kept here rather than under third_party/ because it is an edited fork, not a
# tracked upstream checkout. The Apache-2.0 header above is the upstream file's
# and still governs this file; Primus itself is MIT (see the repository LICENSE).
# --------------------------------------------------------------------------

"""Shared MLIR-dialect helpers and ``s_waitcnt`` bit-field constants for the gfx950
dual-wave, software-pipelined flash-attention kernels.
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace as _TargetAddressSpace
from flydsl.expr import arith, buffer_ops, const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
from primus_turbo.flydsl.utils.prims import LOG2E


def dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    if dtype_str == "fp8":
        return fx.Float8E4M3FN
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f32', 'f16', 'bf16', or 'fp8')")


# s_waitcnt bitfield encoding
_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3

_LDS_ALIAS_DOMAIN = '#llvm.alias_scope_domain<id = "flydsl.dualwave_swp.lds">'

# Wait and low-level ROCDL wrappers


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (n & _VMCNT_LO_MASK) | _LGKMCNT_EXPCNT_BASE | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    rocdl.s_waitcnt(val)


def _s_waitcnt(val):
    rocdl.s_waitcnt(val)


def _sched_barrier(val):
    rocdl.sched_barrier(val)


def _s_barrier():
    rocdl.s_barrier()


def _s_setprio(val):
    rocdl.s_setprio(val)


def _dualwave_sync_barrier():
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _s_nop(x):
    if not isinstance(x, int) or not 0 <= x <= 15:
        raise ValueError("s_nop immediate must be a Python int in [0, 15]")
    llvm.inline_asm(ir.Type.parse("!llvm.void"), [], f"s_nop {x}", "", has_side_effects=True)


def _ds_read_tr16_b64(traits, result_type, base_ptr, imm_bytes, buf_id):
    """gfx950 ds_read_b64_tr_b16 at a constant byte offset from a per-lane V base.
    Emitted as the rocdl op, not inline asm: the op carries a real LDS memory effect so
    SIInsertWaitcnts places lgkmcnt itself, whereas asm would need hand-written waits.
    """
    scope_name = _dualwave_lds_scope("v", buf_id)
    ptr = buffer_ops.get_element_ptr(base_ptr, byte_offset=int(imm_bytes), elem_type=T.i8)
    return rocdl.ds_read_tr16_b64_(
        result_type,
        as_mlir_value(ptr),
        alias_scopes=_dualwave_lds_alias_scopes(scope_name),
        noalias_scopes=_dualwave_lds_noalias_scopes(scope_name, traits.LDS_SCOPE_NAMES),
    )


# Arithmetic and inline-asm primitives


def _fadd(a, b, fm_fast):
    return arith.addf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fsub(a, b, fm_fast):
    return arith.subf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fmul(a, b, fm_fast):
    return arith.mulf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fmax(a, b, fm_fast):
    return arith.MaxNumFOp(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast).result


def _mfma_acc(a, b, c, _mma_atom, mfma_acc_vec_type):
    return fly.mma_atom_call_ssa([mfma_acc_vec_type], _mma_atom, a, b, c)


def _concat_vectors(lhs, rhs):
    lhs_vec = Vec(lhs)
    rhs_vec = Vec(rhs)
    return lhs_vec.shuffle(
        rhs_vec,
        list(range(lhs_vec.numel)) + [lhs_vec.numel + i for i in range(rhs_vec.numel)],
    )


def _bitcast_i32(value):
    return as_mlir_value(ArithValue(value).bitcast(fx.Int32.ir_type))


def _bitcast_f32(value):
    return as_mlir_value(ArithValue(value).bitcast(fx.Float32.ir_type))


def _apply_dualwave_mask_pair(s_values, rel_i32, neg_inf_i32, pair_thresholds, cmp):
    for p in range_constexpr(len(pair_thresholds)):
        thr_x, thr_y = pair_thresholds[p]
        idx_x = p * 2
        idx_y = p * 2 + 1
        x_bits = _bitcast_i32(s_values[idx_x])
        y_bits = _bitcast_i32(s_values[idx_y])
        asm_str = (
            f"v_cmp_{cmp}_i32_e64 $0, $6, {int(thr_x)}\n\t"
            f"v_cmp_{cmp}_i32_e64 $1, $6, {int(thr_y)}\n\t"
            "v_cndmask_b32_e64 $2, $4, $7, $0\n\t"
            "v_cndmask_b32_e64 $3, $5, $7, $1"
        )
        ret_struct_ty = ir.Type.parse("!llvm.struct<(i64, i64, i32, i32)>")
        ret = llvm.inline_asm(
            ret_struct_ty,
            [
                as_mlir_value(x_bits),
                as_mlir_value(y_bits),
                as_mlir_value(rel_i32),
                as_mlir_value(neg_inf_i32),
            ],
            asm_str,
            "=s,=s,=v,=v,2,3,v,v,~{vcc}",
            has_side_effects=True,
        )
        new_x = llvm.extractvalue(T.i32, ret, [2])
        new_y = llvm.extractvalue(T.i32, ret, [3])
        s_values[idx_x] = _bitcast_f32(new_x)
        s_values[idx_y] = _bitcast_f32(new_y)


def _swap_halves(dw):
    pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_i32_ty, as_mlir_value(dw), as_mlir_value(dw), False, False)
    lo_res = llvm.extractvalue(T.i32, swapped, [0])
    hi_res = llvm.extractvalue(T.i32, swapped, [1])
    return lo_res, hi_res


def _o_pack_2dw(traits, v_o, dc, store_group, elem_dtype):
    r_base = store_group * 4
    if const_expr(traits.DTYPE_STR == "bf16"):
        lo = rocdl.cvt_pk_bf16_f32(
            Vec(v_o[dc])[r_base],
            Vec(v_o[dc])[r_base + 1],
        )
        hi = rocdl.cvt_pk_bf16_f32(
            Vec(v_o[dc])[r_base + 2],
            Vec(v_o[dc])[r_base + 3],
        )
        return lo, hi

    o_f16 = []
    for i in range_constexpr(4):
        o_f16.append(fx.Float32(Vec(v_o[dc])[r_base + i]).to(elem_dtype))
    pack = Vec.from_elements(o_f16, elem_dtype).bitcast(fx.Int32)
    return as_mlir_value(pack[0]), as_mlir_value(pack[1])


def _packed_o_128_vec(traits, v_o, dc, g, lane_div_32, elem_dtype):
    d0_a, d1_a = _o_pack_2dw(traits, v_o, dc, 2 * g, elem_dtype)
    d0_b, d1_b = _o_pack_2dw(traits, v_o, dc, 2 * g + 1, elem_dtype)
    is_hi_half = ArithValue(lane_div_32 != fx.Index(0))
    y0_a_lo, y0_a_hi = _swap_halves(d0_a)
    y1_a_lo, y1_a_hi = _swap_halves(d1_a)
    y0_b_lo, y0_b_hi = _swap_halves(d0_b)
    y1_b_lo, y1_b_hi = _swap_halves(d1_b)
    y0_a, y1_a = is_hi_half.select(y0_a_lo, y0_a_hi), is_hi_half.select(y1_a_lo, y1_a_hi)
    y0_b, y1_b = is_hi_half.select(y0_b_lo, y0_b_hi), is_hi_half.select(y1_b_lo, y1_b_hi)
    w0 = is_hi_half.select(y0_b, as_mlir_value(d0_a))
    w1 = is_hi_half.select(y1_b, as_mlir_value(d1_a))
    w2 = is_hi_half.select(as_mlir_value(d0_b), y0_a)
    w3 = is_hi_half.select(as_mlir_value(d1_b), y1_a)
    return Vec.from_elements([fx.Int32(w) for w in (w0, w1, w2, w3)], fx.Int32)


def _anchor_v_o(traits, v_o):
    """Pin v_o accumulators at the current source position."""
    acc_irs = [as_mlir_value(v_o[dc]) for dc in range_constexpr(traits.D_CHUNKS)]
    ret_ty = ir.Type.parse(f"!llvm.struct<({', '.join(['vector<16xf32>'] * traits.D_CHUNKS)})>")
    constraints = ",".join(["=v"] * traits.D_CHUNKS + [str(i) for i in range(traits.D_CHUNKS)])
    ret = llvm.inline_asm(
        ret_ty,
        acc_irs,
        "",
        constraints,
        has_side_effects=True,
    )
    return [llvm.extractvalue(acc_irs[dc].type, ret, [dc]) for dc in range_constexpr(traits.D_CHUNKS)]


def _anchor_v_p(traits, v_p, elem_dtype):
    # Fixed-reference-max forward: P is never rescaled, so there is no ordering left to pin.
    return v_p


def _score_lists_to_vecs(v_s_lists):
    s_lo, s_hi = v_s_lists
    return (
        Vec.from_elements([as_mlir_value(v) for v in s_lo], fx.Float32).ir_value(),
        Vec.from_elements([as_mlir_value(v) for v in s_hi], fx.Float32).ir_value(),
    )


def _reduce_score_pair(v_s, initial, reducer, fm_fast):
    s_lo, s_hi = v_s
    acc = initial
    for r in range_constexpr(16):
        acc = reducer(acc, s_lo[r], fm_fast)
    for r in range_constexpr(16):
        acc = reducer(acc, s_hi[r], fm_fast)
    return acc


def _lane_pair_reduce(v, reducer, fm_fast):
    v_i32 = _bitcast_i32(v)
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_ty, v_i32, v_i32, False, True)
    lhs_i32 = llvm.extractvalue(T.i32, swapped, [0])
    rhs_i32 = llvm.extractvalue(T.i32, swapped, [1])
    return reducer(_bitcast_f32(lhs_i32), _bitcast_f32(rhs_i32), fm_fast)


# Descriptor, LDS, and buffer helpers


def _lds_alias_scope_array(names):
    attrs = [f'#llvm.alias_scope<id = "{name}", domain = {_LDS_ALIAS_DOMAIN}>' for name in names]
    return ir.Attribute.parse(f"[{', '.join(attrs)}]")


def _dualwave_lds_scope(kind, buf_id):
    return f"lds_{kind}{buf_id}"


def _dualwave_lds_alias_scopes(name):
    return _lds_alias_scope_array([name])


def _dualwave_lds_noalias_scopes(name, scope_names):
    return _lds_alias_scope_array([scope_name for scope_name in scope_names if scope_name != name])


def _cu_load(div, idx, cu_atom, cu_v1i32):
    v = fly.copy_atom_call_ssa([cu_v1i32], cu_atom, fx.slice(div, (None, fx.Int32(idx))))
    return fx.Index(Vec(v, (1,), fx.Int32)[0])


def _load_k_pack_aligned(traits, lds_kv_base_ptr, elem_idx, buf_id, kv_mfma_pack_type):
    scope_name = _dualwave_lds_scope("k", buf_id)
    byte_offset = elem_idx * traits.BF16_BYTES
    ptr = buffer_ops.get_element_ptr(lds_kv_base_ptr, byte_offset=byte_offset, elem_type=T.i8)
    return llvm.LoadOp(
        kv_mfma_pack_type,
        ptr,
        alignment=16,
        alias_scopes=_dualwave_lds_alias_scopes(scope_name),
        noalias_scopes=_dualwave_lds_noalias_scopes(scope_name, traits.LDS_SCOPE_NAMES),
    ).result


def _buffer_store_128(pack_i32_vec, elem_index, _o_store_reg_128, _store_atom_128, o_div):
    fx.memref_store_vec(pack_i32_vec, _o_store_reg_128)
    fx.copy(_store_atom_128, _o_store_reg_128, fx.slice(o_div, (None, fx.Int32(elem_index))))


# Mapping and address helpers


def _k_buf_base(traits, buf_id):
    if const_expr(isinstance(buf_id, int)):
        return traits.DUALWAVE_SWP_K_BUF_BASE[buf_id]
    return buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER


def _v_buf_base(traits, buf_id):
    if const_expr(isinstance(buf_id, int)):
        return traits.DUALWAVE_SWP_V_BUF_BASE[buf_id]
    return traits.SMEM_K_TILE_ELEMS + buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER


class DualwaveSwpTraits:
    """Pure compile-time tile/layout constants for gfx950 DUALWAVE_SWP."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def cache_tag(self):
        return (
            self.NUM_HEADS_Q,
            self.NUM_HEADS_KV,
            self.HEAD_DIM,
            self.BLOCK_M,
            self.Q_HEADS_PER_WG,
            self.CAUSAL,
            self.DTYPE_STR,
            self.WAVES_PER_EU,
            self.DAZ,
            self.DUALWAVE_SWP_FIXED_MAX,
            self.DUALWAVE_SWP_MFMA_ROWSUM,
            self.DUALWAVE_SWP_SETPRIO,
            self.DUALWAVE_SWP_ENABLE_STAGGER,
            self.EMIT_LSE,
            self.WINDOW_LEFT,
            self.VARLEN,
            self.CROSS_SEQLEN,
            self.SBHD,
            self.HAS_SINK,
            # MSA: a block-table build and a dense build of the same shape are
            # different kernels. Without these the cache returns whichever was
            # compiled first -- silently, and with wrong results.
            self.BLOCK_TABLE,
            self.TOPK_TILES,
        )


def _make_dualwave_swp_traits(
    num_heads,
    num_kv_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    waves_per_eu=2,
    daz=True,
    dualwave_swp_fixed_max=None,
    dualwave_swp_setprio=True,
    dualwave_swp_enable_stagger=True,
    varlen=False,
    cross_seqlen=False,
    emit_lse=False,
    window_left=-1,
    block_m=None,
    gqa_merge=None,
    sbhd=False,
    has_sink=False,
    block_table=False,
    topk_tiles=0,
):
    """Build gfx950 DUALWAVE_SWP compile-time layout traits."""
    rows_per_wave = 32
    warp_size = 64
    # Windowed default BLOCK_M=64 (fewer masked KV tiles; also avoids waves_per_eu=4).
    # Dense default stays 256. Callers can still pass block_m explicitly.
    if block_m is None:
        block_m = 64 if int(window_left) >= 0 else 256
    wave_row_groups = block_m // rows_per_wave
    block_n = 64
    k_sub_n = 32
    k_step_qk = 16
    d_chunk = 32
    gqa_group_size = num_heads // num_kv_heads

    # Global K/V DMA is 16B per lane; D_128B_SIZE is one 128B row in bf16 elements.
    bf16_bytes = 2
    d_128b_size = 64
    smem_linear_wave = warp_size * 16 // bf16_bytes
    smem_n_per_wave = smem_linear_wave // d_128b_size
    smem_n_rpt = block_n // smem_n_per_wave
    # Two q-heads of one kv-head share a CTA and thus one K/V LDS tile. Capped at SMEM_N_RPT
    # waves, one LDS line each, so the DMA still splits evenly.
    if gqa_merge is None:
        gqa_merge = (
            not dualwave_swp_enable_stagger and gqa_group_size % 2 == 0 and 2 * wave_row_groups <= smem_n_rpt
        )
    q_heads_per_wg = 2 if gqa_merge else 1
    num_waves = wave_row_groups * q_heads_per_wg
    smem_d_rpt = head_dim // d_128b_size
    # K/V LDS tiles are wave-linear rows with padding to avoid bank-aligned repeats.
    smem_k_line_stride = smem_linear_wave + 16 // bf16_bytes
    smem_v_line_stride = smem_linear_wave + 64 // bf16_bytes
    smem_k_tile_elems = smem_n_rpt * smem_d_rpt * smem_k_line_stride
    smem_v_tile_elems = smem_n_rpt * smem_d_rpt * smem_v_line_stride
    dualwave_swp_kv_per_buffer = smem_k_tile_elems + smem_v_tile_elems
    varlen = bool(varlen)
    cross_seqlen = bool(cross_seqlen)
    # Softmax is shift-invariant, so the main loop can run on a fixed zero reference max and let
    # the epilogue re-enter the online path.
    if dualwave_swp_fixed_max is None:
        dualwave_swp_fixed_max = causal
    # With a fixed reference max nothing rebases l_row mid-loop, so the running row sum
    # can live in an MFMA accumulator fed by a ones A operand instead of a VALU fold.
    dualwave_swp_mfma_rowsum = bool(dualwave_swp_fixed_max)
    # Splitting the K/V LDS reads across the memory/compute cluster boundary keeps the main
    # loop inside the 4-waves-per-SIMD register budget, so ask for that budget instead of the
    # caller's floor. Stagger spends the same scheduling slack, so it keeps it.
    if (
        not dualwave_swp_enable_stagger
        and dualwave_swp_mfma_rowsum
        and head_dim == 64
        and block_m in (128, 256)
    ):
        waves_per_eu = max(waves_per_eu, 4)

    return DualwaveSwpTraits(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        WARP_SIZE=warp_size,
        NUM_WAVES=num_waves,
        BLOCK_SIZE=num_waves * warp_size,
        WAVE_ROW_GROUPS=wave_row_groups,
        Q_HEADS_PER_WG=q_heads_per_wg,
        ROWS_PER_WAVE=rows_per_wave,
        HEAD_DIM=head_dim,
        K_STEP_QK=k_step_qk,
        K_STEPS_QK=head_dim // k_step_qk,
        D_CHUNK=d_chunk,
        D_CHUNKS=head_dim // d_chunk,
        PV_K_STEPS=k_sub_n // 16,
        MFMA_LANE_K=8,
        NUM_HEADS_Q=num_heads,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        CAUSAL=causal,
        DTYPE_STR=dtype_str,
        WAVES_PER_EU=waves_per_eu,
        DAZ=bool(daz),
        DUALWAVE_SWP_FIXED_MAX=bool(dualwave_swp_fixed_max),
        DUALWAVE_SWP_MFMA_ROWSUM=dualwave_swp_mfma_rowsum,
        DUALWAVE_SWP_SETPRIO=bool(dualwave_swp_setprio),
        DUALWAVE_SWP_ENABLE_STAGGER=bool(dualwave_swp_enable_stagger),
        EMIT_LSE=bool(emit_lse),
        WINDOW_LEFT=int(window_left),
        VARLEN=varlen,
        CROSS_SEQLEN=cross_seqlen,
        SBHD=bool(sbhd),
        HAS_SINK=bool(has_sink),
        # MSA: walk a per-query list of KV tiles instead of a contiguous span.
        BLOCK_TABLE=bool(block_table),
        TOPK_TILES=int(topk_tiles),
        DEFAULT_STRIDE_Q_N=num_heads * head_dim,
        DEFAULT_STRIDE_KV_N=num_kv_heads * head_dim,
        DMA_BYTES=16,
        BF16_BYTES=bf16_bytes,
        D_128B_SIZE=d_128b_size,
        VEC_KV=8,
        SMEM_N_RPT=smem_n_rpt,
        SMEM_D_RPT=smem_d_rpt,
        SMEM_K_LINE_STRIDE=smem_k_line_stride,
        SMEM_V_LINE_STRIDE=smem_v_line_stride,
        SMEM_K_TILE_ELEMS=smem_k_tile_elems,
        DUALWAVE_SWP_KV_PER_BUFFER=dualwave_swp_kv_per_buffer,
        LDS_KV_TOTAL_SIZE=2 * dualwave_swp_kv_per_buffer,
        DUALWAVE_SWP_K_BUF_BASE=(0, dualwave_swp_kv_per_buffer),
        DUALWAVE_SWP_V_BUF_BASE=(smem_k_tile_elems, smem_k_tile_elems + dualwave_swp_kv_per_buffer),
        # K LDS->VGPR reads: hi half jumps one N strip; ks uses inner then d_rpt stride.
        K_LDS_TO_REG_N_STRIP_STRIDE=256,
        K_LDS_TO_REG_KSTEP_INNER_STRIDE=16,
        K_LDS_TO_REG_KSTEP_OUTER_STRIDE=smem_n_rpt * smem_k_line_stride,
        # V LDS->VGPR base: half-wave, lane quad, N group, lane-in-quad; immediates step across
        # K substeps, D-chunk pairs, and transpose-load pairs.
        V_LDS_TO_REG_HALF_WAVE_STRIDE=2176,
        V_LDS_TO_REG_LANE_QUAD_STRIDE=smem_v_line_stride,
        V_LDS_TO_REG_N_GROUP_STRIDE=16,
        V_LDS_TO_REG_LANE_IN_QUAD_STRIDE=4,
        V_LDS_TO_REG_K_SUBSTEP_STRIDE=128,
        V_LDS_TO_REG_DCHUNK_PAIR_STRIDE=smem_n_rpt * smem_v_line_stride,
        V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE=32,
        V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE=d_128b_size,
        DUALWAVE_SWP_RESCALE_THRESHOLD=8.0,
        SCHED_MFMA_MASK=0x008,
        SCHED_VALU_MASK=0x002,
        SCHED_EXP_MASK=0x400,
        LDS_SCOPE_NAMES=("lds_k0", "lds_k1", "lds_v0", "lds_v1"),
        NEG_INF_F32_BITS=0xFF800000,
        LGKMCNT_0_ONLY=0xC07F,
    )


class DualwaveKernelContext:
    """Shared per-kernel state for the gfx950 dualwave attention helpers."""

    def __init__(
        self,
        traits,
        Q=None,
        K=None,
        V=None,
        O=None,
        DebugCounts=None,  # noqa: E741
        CuSeqQ=None,
        CuSeqKv=None,
        BlockTable=None,
        seq_len=None,
        seq_len_kv=None,
        stride_q_n=None,
        stride_kv_n=None,
        head_dim_runtime=None,
        block_table_stride=None,
        SINK=None,
    ):
        self.traits = traits
        self.SINK = SINK
        self.Q = Q
        self.K = K
        self.V = V
        self.O = O
        self.DebugCounts = DebugCounts
        self.CuSeqQ = CuSeqQ
        self.CuSeqKv = CuSeqKv
        self.BlockTable = BlockTable
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.stride_q_n = stride_q_n
        self.stride_kv_n = stride_kv_n
        self.head_dim_runtime = head_dim_runtime
        self.block_table_stride = block_table_stride

    def _setup(self, shared_storage):
        head_dim_runtime = self.head_dim_runtime
        traits = self.traits
        self.NUM_DMA_K = traits.SMEM_D_RPT
        self.NUM_DMA_V = traits.SMEM_D_RPT
        # vmcnt drains are counted in DMA instructions, but one K/V tile costs
        # DMA_WAVE_REPS of them per wave, so keep the drains at a fixed number of tiles
        # in flight however many waves share the tile.
        _dma_reps = traits.SMEM_N_RPT // traits.NUM_WAVES
        self.VM_DRAIN_KV = (self.NUM_DMA_K + self.NUM_DMA_V) * _dma_reps // 2
        self.VM_DRAIN_V = self.NUM_DMA_V * _dma_reps // 2

        self.fm_fast = fx.arith.FastMathFlags.fast
        self.elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)
        self.q_load_i32x4_type = Vec.make_type(4, fx.Int32)
        self.v_lds_read_vec4_type = Vec.make_type(4, self.elem_dtype)
        self.kv_mfma_pack_type = Vec.make_type(8, self.elem_dtype)
        self.mfma_acc_vec_type = Vec.make_type(16, fx.Float32)
        self.rowsum_acc_vec_type = Vec.make_type(4, fx.Float32)
        self.c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)

        self.c_neg_inf = fx.Float32(float("-inf"))
        self.c_neg_floor = fx.Float32(-3.0e38)
        self.c_zero_f = fx.Float32(0.0)
        self.c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        head_dim_f32 = fx.Float32(fx.Int32(head_dim_runtime))
        c_log2e_f = fx.Float32(LOG2E)
        self.c_sm_scale_log2e = fx.Float32(
            arith.mulf(
                as_mlir_value(fmath.rsqrt(head_dim_f32, fastmath=self.fm_fast)),
                as_mlir_value(c_log2e_f),
                fastmath=self.fm_fast,
            )
        )

        seq_len = self.seq_len
        seq_len_kv = self.seq_len_kv
        stride_q_n = self.stride_q_n
        stride_kv_n = self.stride_kv_n
        self.seq_len_v = fx.Index(seq_len)
        self.seq_len_kv_v = fx.Index(seq_len_kv)
        self.stride_q_n_v = fx.Index(stride_q_n)
        self.stride_kv_n_v = fx.Index(stride_kv_n)

        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.kv.ptr))
        self.lds_kv_base_ptr = buffer_ops.create_llvm_ptr(self.lds_kv_base_idx, address_space=3)

        self.h_idx = fx.Index(gpu.block_idx.x)
        if const_expr(traits.CAUSAL):
            # Dispatch order is the list-schedule order and causal work grows with q_block
            # (a window flattens but never inverts that), so walk q_block descending
            # (longest-processing-time-first). From seq_len, not grid_dim.y, to stay uniform.
            _nq = (self.seq_len_v + fx.Index(traits.BLOCK_M - 1)) // fx.Index(traits.BLOCK_M)
            self.q_block_idx = _nq - fx.Index(1) - fx.Index(gpu.block_idx.y)
        else:
            _nq = (self.seq_len_v + fx.Index(traits.BLOCK_M - 1)) // fx.Index(traits.BLOCK_M)
            self.q_block_idx = fx.Index(gpu.block_idx.y)
        # Row stride of the MSA block table; also the grid's q extent.
        self.num_q_blocks_v = _nq
        self.batch_idx = fx.Index(gpu.block_idx.z)
        self.tid = fx.Index(gpu.thread_idx.x)

        self.wave_id = self.tid // traits.WARP_SIZE
        self.lane = self.tid % traits.WARP_SIZE
        self.lane_mod_32 = self.lane % 32
        self.lane_div_32 = self.lane // 32

        _tid_i32 = as_mlir_value(fx.Int32(self.tid))
        _wave_id_uni_i32 = rocdl.readfirstlane(
            T.i32,
            arith.divsi(_tid_i32, as_mlir_value(fx.Int32(traits.WARP_SIZE))),
        )
        self.wave_id_uni = fx.Index(_wave_id_uni_i32)

        # Merged CTAs stack Q_HEADS_PER_WG sharers of one kv-head on top of the row groups:
        # the low wave bits pick the row group, the high bits pick the sharer.
        if const_expr(traits.Q_HEADS_PER_WG > 1):
            self.wave_row_group = self.wave_id % traits.WAVE_ROW_GROUPS
            self.wave_row_group_uni = self.wave_id_uni % traits.WAVE_ROW_GROUPS
        else:
            self.wave_row_group = self.wave_id
            self.wave_row_group_uni = self.wave_id_uni
        self.wave_q_offset = self.wave_row_group * traits.ROWS_PER_WAVE
        self.q_start = self.q_block_idx * traits.BLOCK_M

        self.h_kv_idx = self.h_idx % traits.NUM_HEADS_KV
        self.group_id = self.h_idx // traits.NUM_HEADS_KV
        if const_expr(traits.Q_HEADS_PER_WG > 1):
            self.q_head_idx = (
                self.h_kv_idx * traits.GQA_GROUP_SIZE
                + self.group_id * traits.Q_HEADS_PER_WG
                + self.wave_id_uni // traits.WAVE_ROW_GROUPS
            )
        else:
            self.q_head_idx = self.h_kv_idx * traits.GQA_GROUP_SIZE + self.group_id
        self.kv_head_idx = self.h_kv_idx

        if const_expr(traits.BLOCK_TABLE):
            # Row of the block table this work-group walks. One selection per
            # (batch, KV head, q block) -- the KV head axis is what makes the
            # selection per GQA group, which is what the reference indexer emits.
            # BLOCK_M is constrained to divide the indexer's block size, so every
            # query in this tile shares the row (see the builder's assert).
            self.block_table_row_base = (
                (self.batch_idx * fx.Index(traits.NUM_HEADS_KV) + self.kv_head_idx)
                * fx.Index(self.num_q_blocks_v)
                + self.q_block_idx
            ) * fx.Index(traits.TOPK_TILES)

        CuSeqQ = self.CuSeqQ
        CuSeqKv = self.CuSeqKv
        if const_expr(traits.VARLEN):
            _cuq_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(CuSeqQ), fx.make_layout(1, 1))
            _cuk_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(CuSeqKv), fx.make_layout(1, 1))
            _cu_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            _cu_v1i32 = Vec.make_type(1, fx.Int32)

            self.q_tok_base = _cu_load(_cuq_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.q_tok_end = _cu_load(_cuq_div, self.batch_idx + fx.Index(1), _cu_atom, _cu_v1i32)
            self.kv_tok_base = _cu_load(_cuk_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.kv_tok_end = _cu_load(_cuk_div, self.batch_idx + fx.Index(1), _cu_atom, _cu_v1i32)
            self.seqlen_q_v = self.q_tok_end - self.q_tok_base
            self.seqlen_kv_v = self.kv_tok_end - self.kv_tok_base
            self.seqlen_kv_i32 = fx.Int32(self.seqlen_kv_v)
        else:
            self.q_tok_base = self.batch_idx * self.seq_len_v
            self.kv_tok_base = self.batch_idx * self.seq_len_kv_v
            self.q_tok_end = (self.batch_idx + fx.Index(1)) * self.seq_len_v
            self.kv_tok_end = (self.batch_idx + fx.Index(1)) * self.seq_len_kv_v
            self.seqlen_q_v = self.seq_len_v
            self.seqlen_kv_v = self.seq_len_kv_v
            self.seqlen_kv_i32 = self.seq_len_kv

        q_tensor = self.Q
        k_tensor = self.K
        v_tensor = self.V
        o_tensor = self.O

        self.delta_i32 = fx.Int32(self.seqlen_kv_i32 - fx.Int32(self.seqlen_q_v))
        self.q_gmem_elem_offset = self.q_start * self.stride_q_n_v + self.q_head_idx * traits.HEAD_DIM
        self.kv_gmem_elem_offset = self.kv_head_idx * traits.HEAD_DIM

        self.buf_flags_i32 = fx.Int32(buffer_ops._get_buffer_flags())
        self.elem_ir = self.elem_dtype.ir_type

        def _view(tensor, byte_off, nrec, layout):
            base_iter = fx.get_iter(tensor)
            base_i64 = fx.Int64(fx.ptrtoint(base_iter))
            shifted = fx.inttoptr(base_iter.type, base_i64 + fx.Int64(byte_off))
            buf_ptr_ty = fx.PointerType.get(
                elem_ty=self.elem_ir,
                address_space=_TargetAddressSpace.BufferDesc,
                alignment=base_iter.alignment,
            )
            buf_ptr = fx.make_ptr(
                buf_ptr_ty,
                [shifted, fx.Int16(0).ir_value(), fx.Int64(nrec).ir_value(), self.buf_flags_i32.ir_value()],
            )
            return fx.logical_divide(fx.make_view(buf_ptr, layout), fx.make_layout(1, 1))

        def _batch_views(seqlen, stride, sbhd_heads, tok_base, ta, tb):
            per_batch = seqlen * stride
            nrec_bytes = per_batch * fx.Index(traits.BF16_BYTES)
            layout = fx.make_layout(fx.Int32(per_batch), fx.Int32(1))
            # SBHD [S,B,H,D]: per-batch base is only heads*D (the seq-step lives in the row stride).
            if const_expr(traits.SBHD):
                off = self.batch_idx * fx.Index(sbhd_heads * traits.HEAD_DIM) * fx.Index(traits.BF16_BYTES)
            else:
                off = tok_base * stride * fx.Index(traits.BF16_BYTES)
            return _view(ta, off, nrec_bytes, layout), _view(tb, off, nrec_bytes, layout)

        self.q_div, self.o_div = _batch_views(
            self.seqlen_q_v, self.stride_q_n_v, traits.NUM_HEADS_Q, self.q_tok_base, q_tensor, o_tensor
        )
        self.k_div, self.v_div = _batch_views(
            self.seqlen_kv_v, self.stride_kv_n_v, traits.NUM_HEADS_KV, self.kv_tok_base, k_tensor, v_tensor
        )
        # EMIT_LSE aliases the DebugCounts tensor slot as the fp32 LSE output
        # ([total_q, NUM_HEADS_Q] layout).
        if const_expr(traits.EMIT_LSE):
            _lse_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(self.DebugCounts)))
            _lse_addr_i64 = as_mlir_value(_lse_base_i64 + fx.Int64(fx.Int64(0)))
            self.lse_rsrc = buffer_ops.create_buffer_resource_from_addr(
                _lse_addr_i64, num_records_bytes=as_mlir_value(fx.Int64(fx.Int64(0xFFFFFFFF)))
            )
        else:
            self.lse_rsrc = None

        # Sink is loaded in the epilogue (`load_sink_h`), not here: a prologue load stays
        # live across the KV loop and spills ~112 B of scratch on the sink+LSE build.
        self.sink_h = None

        self.load_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.store_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, self.elem_dtype))
        if const_expr(self.traits.DUALWAVE_SWP_MFMA_ROWSUM):
            self.rowsum_mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, self.elem_dtype))
            # A operand of the 16x16x32 row-sum MFMA. Re-reading a packed P slice as 16x16x32
            # gives col = lane%16, k = 8*(lane//16)+i, so rows 0/8 tap the low q half and 4/12
            # the high one, leaving each lane's own q sum in D element 0 with no cross-lane move.
            _one_bits = 0x3F803F80 if const_expr(self.traits.DTYPE_STR == "bf16") else 0x3C003C00
            _is_tap_row = ArithValue(fx.Index(self.lane % 4) == fx.Index(0))
            _half_match = ArithValue(fx.Index((self.lane // 4) % 2) == fx.Index((self.lane // 16) % 2))
            _dword = _is_tap_row.select(_half_match.select(fx.Int32(_one_bits), fx.Int32(0)), fx.Int32(0))
            self.rowsum_ones_a = Vec.from_elements([_dword] * 4, fx.Int32).bitcast(self.elem_dtype).ir_value()
        self.o_store_reg_128 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        self.lds_ptr_ty = fx.PointerType.get(self.elem_dtype.ir_type, 2, self.traits.DMA_BYTES)

        self.lane_in_warp = self.tid % self.traits.WARP_SIZE
        self.n_in_warp = self.lane_in_warp // self.traits.VEC_KV
        self.d_bucket = self.lane_in_warp % self.traits.VEC_KV
        self.dma_wave_reps = self.traits.SMEM_N_RPT // self.traits.NUM_WAVES

        self.kv_tile_size = traits.BLOCK_N
        self.num_kv_tiles = (self.seqlen_kv_v + self.kv_tile_size - 1) // self.kv_tile_size
        if const_expr(traits.BLOCK_TABLE):
            # MSA: the block table already encodes causality (the indexer scores a
            # future block -inf, which top-k sorts to a -1 slot the host fills with
            # the query's own block), so there is no span to derive and no odd/even
            # rounding to do -- the walk is exactly TOPK_TILES entries long, and
            # that is a compile-time constant, so the pipeline loop below can be
            # fully unrolled.
            self.causal_end_raw_i32 = None
            self.max_num_tiles = fx.Index(traits.TOPK_TILES)
            self.split_t0 = 0
            self.split_t_end = fx.Index(traits.TOPK_TILES)
            self._block_table_row = None
        elif const_expr(traits.CAUSAL):
            self.causal_end_raw_i32 = fx.Int32(self.q_start + traits.BLOCK_M) + self.delta_i32
            causal_end_i32 = fx.Int32(
                ArithValue(self.causal_end_raw_i32 > fx.Int32(0)).select(self.causal_end_raw_i32, fx.Int32(0))
            )
            causal_num_tiles = (fx.Index(causal_end_i32) + self.kv_tile_size - 1) // self.kv_tile_size
            self.max_num_tiles = fx.Index(
                ArithValue(causal_num_tiles < self.num_kv_tiles).select(causal_num_tiles, self.num_kv_tiles)
            )
        else:
            self.causal_end_raw_i32 = None
            self.max_num_tiles = self.num_kv_tiles

        if const_expr(traits.BLOCK_TABLE):
            pass
        elif const_expr(traits.CAUSAL and traits.WINDOW_LEFT >= 0):
            # Keep an odd KV span: two-tile drain admits N in {3,5,7,...}; even-rounding
            # added a fourth tile that the window mask then threw away.
            self.max_num_tiles = fx.Index(
                ArithValue(self.max_num_tiles < fx.Index(3)).select(fx.Index(3), self.max_num_tiles)
            )
            # SWA: start at the first KV tile that can intersect the left window, leaving
            # >= 3 tiles for the pipeline.
            swa_lo_raw = fx.Int32(self.q_start) + self.delta_i32 - fx.Int32(traits.WINDOW_LEFT)
            swa_lo_raw = fx.Int32(ArithValue(swa_lo_raw > fx.Int32(0)).select(swa_lo_raw, fx.Int32(0)))
            swa_lo_tile = fx.Index(swa_lo_raw) // fx.Index(traits.BLOCK_N)
            max_lo = fx.Index(
                ArithValue(self.max_num_tiles > fx.Index(3)).select(
                    self.max_num_tiles - fx.Index(3), fx.Index(0)
                )
            )
            swa_lo_tile = fx.Index(ArithValue(swa_lo_tile < max_lo).select(swa_lo_tile, max_lo))
            # If the span is even, grow by one tile. Prefer downward (window-masked);
            # at tile 0 grow upward (causal-masked).
            span = self.max_num_tiles - swa_lo_tile
            span_odd = span - (span // fx.Index(2)) * fx.Index(2)
            grow = fx.Index(1) - span_odd
            down = fx.Index(ArithValue(swa_lo_tile > fx.Index(0)).select(grow, fx.Index(0)))
            self.split_t0 = swa_lo_tile - down
            self.split_t_end = self.max_num_tiles + (grow - down)
        else:
            self.max_num_tiles = ((self.max_num_tiles + fx.Index(1)) // fx.Index(2)) * fx.Index(2)
            self.max_num_tiles = fx.Index(
                ArithValue(self.max_num_tiles < fx.Index(4)).select(fx.Index(4), self.max_num_tiles)
            )
            self.split_t0 = 0
            self.split_t_end = self.max_num_tiles

        if const_expr(traits.VARLEN):
            if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
                self.active = ArithValue(self.q_start < self.seqlen_q_v) & (
                    self.causal_end_raw_i32 > fx.Int32(0)
                )
            else:
                self.active = ArithValue(self.q_start < self.seqlen_q_v)
        elif const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
            self.active = ArithValue(self.causal_end_raw_i32 > fx.Int32(0))
        else:
            self.active = None

        self.k_lds_read_base_per_lane = (
            (self.lane_mod_32 % 8) * traits.SMEM_K_LINE_STRIDE
            + (self.lane_mod_32 // 8) * traits.D_128B_SIZE
            + self.lane_div_32 * traits.VEC_KV
        )
        self.v_lds_read_base_per_lane = (
            self.lane_div_32 * traits.V_LDS_TO_REG_HALF_WAVE_STRIDE
            + ((self.lane % 16) // 4) * traits.V_LDS_TO_REG_LANE_QUAD_STRIDE
            + ((self.lane // 16) % 2) * traits.V_LDS_TO_REG_N_GROUP_STRIDE
            + (self.lane % 4) * traits.V_LDS_TO_REG_LANE_IN_QUAD_STRIDE
        )

        self.k_dma_m0 = self.dma_m0_table(
            lambda b, w, d: self._dma_base(b, w, d, _k_buf_base, traits.SMEM_K_LINE_STRIDE), self.NUM_DMA_K
        )
        self.v_dma_m0 = self.dma_m0_table(
            lambda b, w, d: self._dma_base(b, w, d, _v_buf_base, traits.SMEM_V_LINE_STRIDE), self.NUM_DMA_V
        )

    def _dma_base(self, buf_id, w_rep, d, buf_base_fn, line_stride):
        traits = self.traits
        wave_id_uni = self.wave_id_uni + fx.Index(w_rep * self.traits.NUM_WAVES)
        lds_byte_base = self.lds_kv_base_idx + buf_base_fn(traits, buf_id) * traits.BF16_BYTES
        lds_addr = (
            lds_byte_base
            + wave_id_uni * (line_stride * traits.BF16_BYTES)
            + (d * traits.SMEM_N_RPT * line_stride * traits.BF16_BYTES)
        )
        return rocdl.readfirstlane(T.i32, as_mlir_value(fx.Int32(lds_addr)))

    def dma_m0_table(self, base_fn, count):
        return tuple(
            tuple(tuple(base_fn(buf, wr, d) for d in range(count)) for wr in range(self.dma_wave_reps))
            for buf in range(2)
        )

    def split_tile(self, offset_tiles=0):
        return self.split_t0 + fx.Index(offset_tiles)

    def tile_start(self, tile_idx):
        """Physical KV element offset of the tile this step reads.

        Dense: tile_idx IS the tile number. MSA: tile_idx is a slot in the
        query's block table, and the tile number is whatever that slot holds --
        the one indirection that turns the dense walk into a sparse one. Every
        other user of a KV position goes through here, so nothing else changes.
        """
        if const_expr(self.traits.BLOCK_TABLE):
            return self.block_table_tile(tile_idx) * self.traits.BLOCK_N
        return tile_idx * self.traits.BLOCK_N

    def block_table_tile(self, slot):
        """Read one tile number out of this work-group's block-table row.

        The row is uniform across the work-group (one q block, one KV head), so
        the address is wave-uniform and the load scalarises.
        """
        div = fx.logical_divide(fx.rocdl.make_buffer_tensor(self.BlockTable), fx.make_layout(1, 1))
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        v1i32 = Vec.make_type(1, fx.Int32)
        return _cu_load(div, self.block_table_row_base + fx.Index(slot), atom, v1i32)

    # Pipeline helpers

    def load_all(self):
        traits = self.traits
        self.q_row_in_block = self.wave_q_offset + self.lane_mod_32
        self.q_start_pos_i32 = fx.Int32(self.q_start + self.wave_row_group_uni * traits.ROWS_PER_WAVE)
        self.q_row = self.q_start + self.q_row_in_block
        self.q_row_i32 = fx.Int32(self.q_row)

        q_raw_packs = []
        for ks in range_constexpr(traits.K_STEPS_QK):
            elem_index = self.q_gmem_elem_offset + (
                self.q_row_in_block * self.stride_q_n_v
                + (ks * self.traits.K_STEP_QK + self.lane_div_32 * self.traits.MFMA_LANE_K)
            )
            q_i32_pack = fly.copy_atom_call_ssa(
                [self.q_load_i32x4_type],
                self.load_atom_128,
                fx.slice(self.q_div, (None, fx.Int32(elem_index))),
            )
            q_raw_packs.append(Vec(q_i32_pack, (4,), fx.Int32).bitcast(self.elem_dtype).ir_value())
        q_16_packs = []
        for pair in range_constexpr(traits.K_STEPS_QK // 2):
            q_16_packs.append(_concat_vectors(q_raw_packs[pair * 2], q_raw_packs[pair * 2 + 1]))

        q_32_packs = []
        for pair in range_constexpr(traits.K_STEPS_QK // 4):
            q_32_packs.append(_concat_vectors(q_16_packs[pair * 2], q_16_packs[pair * 2 + 1]))

        q_all = (
            q_32_packs[0]
            if const_expr(traits.K_STEPS_QK == 4)
            else _concat_vectors(q_32_packs[0], q_32_packs[1])
        )
        return Vec(q_all, (traits.K_STEPS_QK * traits.MFMA_LANE_K,), self.elem_dtype)

    def scale_all(self, q_all_bf16):
        traits = self.traits
        fm_fast_attr = ir.Attribute.parse("#llvm.fastmath<fast>")
        v64bf16_type = Vec.make_type(traits.K_STEPS_QK * traits.MFMA_LANE_K, self.elem_dtype)
        v64f32_type = Vec.make_type(traits.K_STEPS_QK * traits.MFMA_LANE_K, fx.Float32)
        q_all_f32_op = llvm.FPExtOp(v64f32_type, as_mlir_value(q_all_bf16))
        q_all_f32_op.operation.attributes["fastmathFlags"] = fm_fast_attr
        q_all_f32 = q_all_f32_op.result
        scale_vec = Vec.from_elements([self.c_sm_scale_log2e], fx.Float32).broadcast_to(
            traits.K_STEPS_QK * traits.MFMA_LANE_K
        )
        q_all_scaled_f32 = arith.mulf(
            as_mlir_value(scale_vec),
            as_mlir_value(q_all_f32),
            fastmath=self.fm_fast,
        )
        q_all_scaled_bf16_op = llvm.FPTruncOp(v64bf16_type, q_all_scaled_f32)
        q_all_scaled_bf16_op.operation.attributes["fastmathFlags"] = fm_fast_attr
        q_all_scaled_bf16 = q_all_scaled_bf16_op.result
        return Vec(q_all_scaled_bf16, (traits.K_STEPS_QK * traits.MFMA_LANE_K,), self.elem_dtype)

    def qk(self, v_k, q_all_scaled_bf16, v_s=None, ks_range=None):
        k_lo, k_hi = v_k
        ks_lo, ks_hi = (0, self.traits.K_STEPS_QK) if ks_range is None else ks_range
        if v_s is None:
            v_s_lo = self.c_zero_v16f32
            v_s_hi = self.c_zero_v16f32
        else:
            v_s_lo, v_s_hi = v_s
        for ks in range_constexpr(ks_lo, ks_hi):
            _q_vec = Vec(q_all_scaled_bf16)
            _base = ks * self.traits.MFMA_LANE_K
            q_pack = _q_vec.shuffle(_q_vec, [_base + i for i in range(self.traits.MFMA_LANE_K)]).ir_value()
            v_s_lo = _mfma_acc(k_lo[ks], q_pack, v_s_lo, self.mma_atom, self.mfma_acc_vec_type)
            v_s_hi = _mfma_acc(k_hi[ks], q_pack, v_s_hi, self.mma_atom, self.mfma_acc_vec_type)
        return (v_s_lo, v_s_hi)

    # Skip a 32-col QK half that is wholly -inf for this wave (window/causal edge).
    # Predicates are wave-uniform; the mask still writes -inf over skipped accs.
    def chunk_live(self, tile_idx):
        traits = self.traits
        kv0 = fx.Int32(self.tile_start(tile_idx))
        q_lo = fx.Int32(self.q_start_pos_i32 + self.delta_i32)
        q_hi = fx.Int32(q_lo + fx.Int32(traits.ROWS_PER_WAVE - 1))
        live_hi = ArithValue(fx.Int32(kv0 + fx.Int32(32)) <= q_hi)
        if const_expr(traits.WINDOW_LEFT >= 0):
            win_lo = fx.Int32(q_lo - fx.Int32(traits.WINDOW_LEFT))
            live_lo = ArithValue(fx.Int32(kv0 + fx.Int32(31)) >= win_lo)
        else:
            # Dense causal: LO half is dead when kv0 already exceeds the wave's last q row.
            live_lo = ArithValue(kv0 <= q_hi)
        return live_lo, live_hi

    def qk_live(self, v_k, q_all_scaled_bf16, live_lo, live_hi, v_s=None, ks_range=None):
        k_lo, k_hi = v_k
        traits = self.traits
        ks_lo, ks_hi = (0, traits.K_STEPS_QK) if ks_range is None else ks_range
        if v_s is None:
            s_lo_in, s_hi_in = self.c_zero_v16f32, self.c_zero_v16f32
        else:
            s_lo_in, s_hi_in = v_s

        def _chain(k_part, acc):
            for ks in range_constexpr(ks_lo, ks_hi):
                _q_vec = Vec(q_all_scaled_bf16)
                _base = ks * traits.MFMA_LANE_K
                q_pack = _q_vec.shuffle(_q_vec, [_base + i for i in range(traits.MFMA_LANE_K)]).ir_value()
                acc = _mfma_acc(k_part[ks], q_pack, acc, self.mma_atom, self.mfma_acc_vec_type)
            return acc

        if live_lo is None:
            v_s_lo = _chain(k_lo, s_lo_in)
        else:

            @flyc.jit
            def _qk_lo(v_s_lo, live_lo):
                if live_lo:
                    v_s_lo = _chain(k_lo, v_s_lo)
                return v_s_lo

            v_s_lo = _qk_lo(s_lo_in, live_lo)

        @flyc.jit
        def _qk_hi(v_s_hi, live_hi):
            if live_hi:
                v_s_hi = _chain(k_hi, v_s_hi)
            return v_s_hi

        v_s_hi = _qk_hi(s_hi_in, live_hi)
        return (v_s_lo, v_s_hi)

    def pv_step_k(self, step, v_p, v_v, v_o):
        v_p_lo, v_p_hi = v_p
        v_pk = v_v[step]
        if const_expr(step < 2):
            p_pk = v_p_lo[step]
        else:
            p_pk = v_p_hi[step - 2]
        for dc in range_constexpr(self.traits.D_CHUNKS):
            v_o[dc] = _mfma_acc(v_pk[dc], p_pk, v_o[dc], self.mma_atom, self.mfma_acc_vec_type)
        return v_o

    def pv(self, v_p, v_v, v_o):
        for step in range_constexpr(4):
            v_o = self.pv_step_k(step, v_p, v_v, v_o)
        return v_o

    def reduce_max(self, v_s):
        return _lane_pair_reduce(
            _reduce_score_pair(v_s, self.c_neg_inf, _fmax, self.fm_fast), _fmax, self.fm_fast
        )

    def floor_masked_max(self, row_max):
        return _fmax(row_max, self.c_neg_floor, self.fm_fast)

    def exp2(self, v_s, start, length):
        if const_expr(start == 0):
            s_lo = [Vec(v_s[0])[r] for r in range_constexpr(16)]
            lo_partial = []
            for r in range_constexpr(16):
                lo_partial.append(rocdl.exp2(T.f32, as_mlir_value(s_lo[r])))
            return Vec.from_elements(lo_partial, fx.Float32).ir_value(), v_s[1]
        lo_partial = [Vec(v_s[0])[r] for r in range_constexpr(16)]
        hi_full = []
        for r in range_constexpr(16):
            hi_full.append(rocdl.exp2(T.f32, as_mlir_value(Vec(v_s[1])[r])))
        return lo_partial, hi_full

    def cast_p_and_sum(self, l_row, v_p):
        """Pack P to bf16 and fold the tile into the row sum; the MFMA path feeds the packs
        themselves to a ones-matrix MFMA so numerator and denominator see the same bf16 values.
        Each 16-kv pack is one 16x16x32 MFMA against the ones A operand, which also folds the
        half-wave partner, so no permlane pair reduce is left; the fence keeps the packs from
        being hoisted and the scored fptrunc supplies the wait states."""
        v_p = self.cast_p(v_p)
        p_lo_packs, p_hi_packs = v_p
        _sched_barrier(0)
        for pks in range_constexpr(self.traits.PV_K_STEPS):
            for pack in (p_lo_packs[pks], p_hi_packs[pks]):
                l_row = _mfma_acc(
                    self.rowsum_ones_a,
                    pack,
                    l_row,
                    self.rowsum_mma_atom,
                    self.rowsum_acc_vec_type,
                )
        return v_p, l_row

    def finish_row_sum(self, l_row):
        """Take the row sum out of the MFMA accumulator; D element 0 holds this lane's q."""
        return Vec(l_row)[0]

    def cast_p(self, v_p):
        traits = self.traits
        elem_dtype = self.elem_dtype

        def _pack_v8(f32_vals):
            if const_expr(traits.DTYPE_STR == "bf16"):
                # A vector fptrunc still selects v_cvt_pk_bf16_f32 pairwise, but as a scored op:
                # the backend places the pack-to-MFMA wait states itself, so consumers need no
                # hand fence (inline asm hides the VGPR def from GCNHazardRecognizer).
                f32_vec = Vec.from_elements([as_mlir_value(v) for v in f32_vals], fx.Float32)
                trunc_op = llvm.FPTruncOp(Vec.make_type(8, elem_dtype), as_mlir_value(f32_vec))
                trunc_op.operation.attributes["fastmathFlags"] = ir.Attribute.parse("#llvm.fastmath<fast>")
                return trunc_op.result
            f16_vals = []
            for i in range_constexpr(8):
                f16_vals.append(fx.Float32(f32_vals[i]).to(elem_dtype))
            return Vec.from_elements(f16_vals, elem_dtype).ir_value()

        lo_partial_list, hi_full = v_p
        p_lo_packs = []
        p_hi_packs = []
        for pks in range_constexpr(traits.PV_K_STEPS):
            p_base = pks * 8
            lo_slice = [lo_partial_list[p_base + s] for s in range_constexpr(8)]
            hi_slice = hi_full[p_base : p_base + 8]
            p_lo_packs.append(_pack_v8(lo_slice))
            p_hi_packs.append(_pack_v8(hi_slice))
        return p_lo_packs, p_hi_packs

    def safe_l_inv(self, l_row):
        l_inv = rocdl.rcp(T.f32, as_mlir_value(l_row))
        return ArithValue(fx.Float32(l_row) > self.c_zero_f).select(l_inv, self.c_zero_f)

    def load_sink_h(self):
        """Load the per-q-head attention sink (epilogue, not prologue)."""
        traits = self.traits
        _sink_rsrc = buffer_ops.create_buffer_resource(
            self.SINK,
            max_size=False,
            num_records_bytes=as_mlir_value(fx.Index(traits.NUM_HEADS_Q) * fx.Index(4)),
        )
        return fx.Float32(buffer_ops.buffer_load(_sink_rsrc, self.q_head_idx, vec_width=1, dtype=fx.Float32))

    def finalize_o_scale(self, m_row, l_row):
        """Return (o_scale, m_out, l_out): the normalizing scale applied to the unnormalized O
        and the (max, denom) pair to store as LSE. m_row/l_row are in the log2 domain
        (l_row = sum_j 2^(s_j - m_row)).

        With HAS_SINK, fold the learned per-head sink (a virtual key with logit=sink_h,
        value=0) into the denominator, mirroring sparse_mla_fwd._epi_scalars:
            mf = max(m_row, sink_log2);  af = 2^(m_row-mf);  st = 2^(sink_log2-mf)
            l_out = l_row*af + st;  O *= af/l_out  (sink's value=0 -> no O contribution)
        Without HAS_SINK, af==1 and mf==m_row, so this is byte-identical to the plain 1/l path."""
        if const_expr(self.traits.HAS_SINK):
            fm = self.fm_fast
            sink_log2 = _fmul(self.load_sink_h(), fx.Float32(LOG2E), fm)
            mf = fx.Float32(_fmax(m_row, sink_log2, fm))
            af = fx.Float32(rocdl.exp2(T.f32, _fsub(m_row, mf, fm)))
            st = fx.Float32(rocdl.exp2(T.f32, _fsub(sink_log2, mf, fm)))
            l_out = fx.Float32(_fadd(_fmul(l_row, af, fm), st, fm))
            o_scale = fx.Float32(_fmul(af, self.safe_l_inv(l_out), fm))
            return o_scale, mf, l_out
        return self.safe_l_inv(l_row), m_row, l_row

    def scale_o(self, v_o, scale_scalar):
        scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
        for dc in range_constexpr(self.traits.D_CHUNKS):
            v_o[dc] = _fmul(Vec(v_o[dc]), scale_vec, self.fm_fast)

    def zero_row_max(self):
        return self.c_zero_f

    def scores_for_softmax(self, v_s):
        return v_s

    def shift_scores(self, v_s, row_max):
        return _score_lists_to_vecs(v_s) if isinstance(v_s[0], list) else v_s

    def tile_rescale_o(self, v_o, m_row, l_row, v_s, v_p, sched_group):
        """With a fixed reference max the correction is identically 1, so the row-max reduction,
        the rescale and the m_row update all drop out."""
        return v_o, m_row, l_row, v_p

    def tile_row_max(self, m_row, v_s):
        """Returns None as the O/l correction when the reference max is fixed and nothing rebases."""
        return m_row, None

    def scale_o_by(self, v_o, rescale):
        if rescale is not None:
            self.scale_o(v_o, rescale)

    def scale_l_by(self, l_row, rescale):
        return l_row if rescale is None else _fmul(l_row, rescale, self.fm_fast)

    def v_s_vec_to_lists(self, v_s):
        s_lo, s_hi = v_s
        return (
            [Vec(s_lo)[r] for r in range_constexpr(16)],
            [Vec(s_hi)[r] for r in range_constexpr(16)],
        )

    def _causal_mask_inplace(self, v_s, tile_idx, q_row_i32=None, *, do_causal=True, do_window=True):
        # Python bools: each instantiation emits only the requested edge.
        if q_row_i32 is None:
            q_row_i32 = self.q_row_i32
        traits = self.traits
        s_lo, s_hi = v_s
        kv_tile_start = tile_idx * traits.BLOCK_N
        kv_start_i32 = fx.Int32(kv_tile_start)
        # lane>=32 has a larger n offset in the K-permuted P layout.
        lane_off_i32 = fx.Int32(self.lane_div_32) * fx.Int32(4)
        rel_lo_i32 = fx.Int32(q_row_i32 + self.delta_i32 - kv_start_i32 - lane_off_i32)
        rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(32))
        neg_inf_i32 = fx.Int32(traits.NEG_INF_F32_BITS)

        pair_thresholds = [(0, 1), (2, 3), (8, 9), (10, 11), (16, 17), (18, 19), (24, 25), (26, 27)]
        if const_expr(do_causal):
            _apply_dualwave_mask_pair(s_lo, rel_lo_i32, neg_inf_i32, pair_thresholds, "lt")
            _apply_dualwave_mask_pair(s_hi, rel_hi_i32, neg_inf_i32, pair_thresholds, "lt")
        # SWA left window: additionally mask columns below the window (phys_kv < q_row + delta - W).
        if const_expr(traits.WINDOW_LEFT >= 0 and do_window):
            w_i32 = fx.Int32(traits.WINDOW_LEFT)
            rel_win_lo_i32 = fx.Int32(rel_lo_i32 - w_i32)
            rel_win_hi_i32 = fx.Int32(rel_hi_i32 - w_i32)
            _apply_dualwave_mask_pair(s_lo, rel_win_lo_i32, neg_inf_i32, pair_thresholds, "gt")
            _apply_dualwave_mask_pair(s_hi, rel_win_hi_i32, neg_inf_i32, pair_thresholds, "gt")

    def causal_mask_prologue_if_needed(
        self, v_s, tile_idx=None, kv_end_pos=None, q_start_pos_i32=None, q_row_i32=None, *, kv_end_tile=None
    ):
        if tile_idx is None:
            tile_idx = fx.Index(0)
        if kv_end_pos is None:
            end_tile = tile_idx + fx.Index(1) if kv_end_tile is None else kv_end_tile
            kv_end_pos = self.tile_start(end_tile)
        if q_start_pos_i32 is None:
            q_start_pos_i32 = self.q_start_pos_i32
        if q_row_i32 is None:
            q_row_i32 = self.q_row_i32

        traits = self.traits
        # SWA lower-edge guard is deliberately conservative; _causal_mask_inplace is a no-op on tiles that need no masking.
        kv_start_pos_w = self.tile_start(tile_idx)

        @flyc.jit
        def _causal_mask_prologue_if_needed(v_s, tile_idx, kv_end_pos, q_start_pos_i32, q_row_i32):
            s_lo, s_hi = v_s
            _need = q_start_pos_i32 + self.delta_i32 < fx.Int32(kv_end_pos)
            if const_expr(traits.WINDOW_LEFT >= 0):
                # Window and causal edges never share a tile; apply each mask only when needed.
                _win_edge = (
                    q_start_pos_i32 + fx.Int32(traits.BLOCK_M) + self.delta_i32 - fx.Int32(traits.WINDOW_LEFT)
                )
                _need_win = ArithValue(fx.Int32(kv_start_pos_w) < _win_edge)
                if _need:
                    lo_list, hi_list = self.v_s_vec_to_lists((s_lo, s_hi))
                    self._causal_mask_inplace(
                        (lo_list, hi_list), tile_idx, q_row_i32=q_row_i32, do_window=False
                    )
                    s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
                if _need_win:
                    lo_list, hi_list = self.v_s_vec_to_lists((s_lo, s_hi))
                    self._causal_mask_inplace(
                        (lo_list, hi_list), tile_idx, q_row_i32=q_row_i32, do_causal=False
                    )
                    s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            else:
                if _need:
                    lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                    self._causal_mask_inplace((lo_list, hi_list), tile_idx, q_row_i32=q_row_i32)
                    s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _causal_mask_prologue_if_needed(v_s, tile_idx, kv_end_pos, q_start_pos_i32, q_row_i32)

    def causal_mask_split_prologue_if_needed(self, v_s, offset_tiles=0, end_offset_tiles=1):
        return self.causal_mask_prologue_if_needed(
            v_s,
            self.split_tile(offset_tiles),
            kv_end_tile=self.split_tile(end_offset_tiles),
        )

    def seq_pad_mask_if_needed(self, v_s, tile_idx):
        traits = self.traits
        seqlen_kv_i32 = self.seqlen_kv_i32

        @flyc.jit
        def _seq_pad_mask_if_needed(v_s, tile_idx):
            s_lo, s_hi = v_s
            kv_tile_end = (tile_idx + fx.Index(1)) * traits.BLOCK_N
            if fx.Int32(kv_tile_end) > seqlen_kv_i32:
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                col_base = fx.Int32(tile_idx * self.traits.BLOCK_N) + fx.Int32(self.lane_div_32) * fx.Int32(4)
                for r in range_constexpr(16):
                    col_lo = col_base + fx.Int32((r // 4) * 8 + (r % 4))
                    col_hi = col_lo + fx.Int32(32)
                    lo_list[r] = ArithValue(col_lo < self.seqlen_kv_i32).select(lo_list[r], self.c_neg_inf)
                    hi_list[r] = ArithValue(col_hi < self.seqlen_kv_i32).select(hi_list[r], self.c_neg_inf)
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _seq_pad_mask_if_needed(v_s, tile_idx)

    def _load_kv(self, tile_start, buf_id, dma_m0, div, num_dma):
        src_base = self.kv_gmem_elem_offset
        soffset = tile_start * self.stride_kv_n_v
        # w_rep: each wave fills DMA_WAVE_REPS LDS lines (1 for 8-wave, 2 for 4-wave).
        for wr in range_constexpr(self.dma_wave_reps):
            eff_wave = self.wave_id + fx.Index(wr * self.traits.NUM_WAVES)
            for d in range_constexpr(num_dma):
                # LDS line stride = SMEM_N_RPT (# lines, always 8), NOT NUM_WAVES: at the 8-wave
                # CTA they coincide, but a 4-wave CTA needs each wave to fill 2 lines, with
                # ``eff_wave`` carrying the effective line index (wave_id + w_rep*NUM_WAVES).
                n_in_tile = self.n_in_warp * self.traits.SMEM_N_RPT + eff_wave
                global_d = self.d_bucket * self.traits.VEC_KV + d * self.traits.D_128B_SIZE
                src_elem = src_base + n_in_tile * self.stride_kv_n_v + global_d
                # 128-bit global->LDS DMA; `src_elem` is voffset, `soffset` is scaled by the atom.
                lds_ptr = fx.inttoptr(self.lds_ptr_ty, fx.Int32(dma_m0[buf_id][wr][d]))
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                src = fx.slice(div, (None, fx.Int32(src_elem)))
                fx.copy(self.dma_atom, src, dst, soffset=fx.Int32(soffset))

    def load_k_tile(self, tile_idx, buf_id):
        self._load_kv(self.tile_start(tile_idx), buf_id, self.k_dma_m0, self.k_div, self.NUM_DMA_K)

    def load_k_split(self, offset_tiles, buf_id):
        self.load_k_tile(self.split_tile(offset_tiles), buf_id)

    def load_v_tile(self, tile_idx, buf_id):
        self._load_kv(self.tile_start(tile_idx), buf_id, self.v_dma_m0, self.v_div, self.NUM_DMA_V)

    def load_v_split(self, offset_tiles, buf_id):
        self.load_v_tile(self.split_tile(offset_tiles), buf_id)

    def lds_load_k(self, buf_id, urk_base=None, ks_range=None, k_regs=None):
        """Read this buffer's K packs for k-steps [ks_range). Splitting the range keeps only half the K tile resident in registers at once."""
        if urk_base is None:
            urk_base = self.k_lds_read_base_per_lane
        ks_lo, ks_hi = (0, self.traits.K_STEPS_QK) if ks_range is None else ks_range
        k_base = _k_buf_base(self.traits, buf_id)
        if k_regs is None:
            k_lo = [None] * self.traits.K_STEPS_QK
            k_hi = [None] * self.traits.K_STEPS_QK
        else:
            k_lo, k_hi = k_regs

        for ks in range_constexpr(ks_lo, ks_hi):
            ks_off = (ks // 4) * self.traits.K_LDS_TO_REG_KSTEP_OUTER_STRIDE + (
                ks % 4
            ) * self.traits.K_LDS_TO_REG_KSTEP_INNER_STRIDE
            _idx_lo = k_base + urk_base + ks_off
            k_lo[ks] = _load_k_pack_aligned(
                self.traits,
                self.lds_kv_base_ptr,
                _idx_lo,
                buf_id,
                self.kv_mfma_pack_type,
            )
            k_hi[ks] = _load_k_pack_aligned(
                self.traits,
                self.lds_kv_base_ptr,
                _idx_lo + self.traits.K_LDS_TO_REG_N_STRIP_STRIDE,
                buf_id,
                self.kv_mfma_pack_type,
            )
        return (k_lo, k_hi)

    def lds_load_v(self, buf_id, urv_base=None, substeps=None, packs=None):
        if urv_base is None:
            urv_base = self.v_lds_read_base_per_lane
        v_base = _v_buf_base(self.traits, buf_id)
        ks_lo, ks_hi = (0, 4) if substeps is None else substeps
        if packs is None:
            packs = [[None] * self.traits.D_CHUNKS for _ in range(4)]
        pair_off = self.traits.V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE * self.traits.BF16_BYTES
        lane_ptr = buffer_ops.get_element_ptr(
            self.lds_kv_base_ptr,
            byte_offset=as_mlir_value(fx.Int32((v_base + urv_base) * self.traits.BF16_BYTES)),
            elem_type=T.i8,
        )
        # k-substep major: the two reads of every P*V step land back to back, so the
        # backend's own lgkmcnt for step k does not also wait on the later substeps.
        for k_substep in range_constexpr(ks_lo, ks_hi):
            for dc in range_constexpr(self.traits.D_CHUNKS):
                dc_off = (dc // 2) * self.traits.V_LDS_TO_REG_DCHUNK_PAIR_STRIDE + (
                    dc % 2
                ) * self.traits.V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE
                imm_lo = (
                    k_substep * self.traits.V_LDS_TO_REG_K_SUBSTEP_STRIDE + dc_off
                ) * self.traits.BF16_BYTES
                a = _ds_read_tr16_b64(self.traits, self.v_lds_read_vec4_type, lane_ptr, imm_lo, buf_id)
                b = _ds_read_tr16_b64(
                    self.traits, self.v_lds_read_vec4_type, lane_ptr, imm_lo + pair_off, buf_id
                )
                packs[k_substep][dc] = Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
        return packs

    def _final_o_base(self, q_row):
        return q_row * self.stride_q_n_v + self.q_head_idx * self.traits.HEAD_DIM + self.lane_div_32 * 8

    def _final_o_global(self, o_base, dc, g):
        return o_base + (dc * self.traits.D_CHUNK + 2 * g * 8)

    def _store_o_128(self, o_base, pack_fn):
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for g in range_constexpr(2):
                _buffer_store_128(
                    pack_fn(dc, g),
                    self._final_o_global(o_base, dc, g),
                    _o_store_reg_128=self.o_store_reg_128,
                    _store_atom_128=self.store_atom_128,
                    o_div=self.o_div,
                )

    def zero_o_block_if_needed(self, causal_end_raw_i32=None):
        if causal_end_raw_i32 is None:
            causal_end_raw_i32 = self.causal_end_raw_i32
        q_start = self.q_start
        wave_q_offset = self.wave_q_offset
        lane_mod_32 = self.lane_mod_32
        seq_len_v = self.seq_len_v

        @flyc.jit
        def _zero_o_block_if_needed():
            if causal_end_raw_i32 <= fx.Int32(0):
                q_row_z = q_start + wave_q_offset + lane_mod_32
                c_zero_i = fx.Int32(0)
                zero_pack = Vec.from_elements([c_zero_i, c_zero_i, c_zero_i, c_zero_i], fx.Int32)
                if q_row_z < seq_len_v:
                    self._store_o_128(self._final_o_base(q_row_z), lambda dc, g: zero_pack)

        _zero_o_block_if_needed()

    def store_final_o(self, v_o, q_row):
        self._store_o_128(
            self._final_o_base(q_row),
            lambda dc, g: _packed_o_128_vec(self.traits, v_o, dc, g, self.lane_div_32, self.elem_dtype),
        )

    def store_lse(self, m_row, l_row, q_row):
        """LSE: m_row/l_row live in the log2 domain, hence LSE_natural =
        ln2 * (m_row + log2(l_row)); written as f32 per (row, head) into the DebugCounts slot."""
        traits = self.traits
        lse_rsrc = self.lse_rsrc
        q_tok_base = self.q_tok_base
        q_head_idx = self.q_head_idx
        seqlen_q_v = self.seqlen_q_v
        lane = self.lane
        fm_fast = self.fm_fast
        _ln2 = host_math.log(2.0)

        @flyc.jit
        def _store_lse():
            if q_row < seqlen_q_v:
                if lane < fx.Index(32):
                    log2_l = fx.Float32(fmath.log2(as_mlir_value(fx.Float32(l_row)), fastmath=fm_fast))
                    lse = _fmul(_fadd(m_row, log2_l, fm_fast), fx.Float32(_ln2), fm_fast)
                    elem_idx = (q_tok_base + q_row) * fx.Index(traits.NUM_HEADS_Q) + q_head_idx
                    _lse_f32_ir = as_mlir_value(fx.Float32(lse))
                    buffer_ops.buffer_store(_lse_f32_ir, lse_rsrc, as_mlir_value(fx.Int32(elem_idx)))

        _store_lse()


def _scale_sched_pairs(pairs, head_dim):
    """At head_dim 64, use 1/4 of the sched_group_barrier anchors (softmax work is D-independent)."""
    if head_dim != 64:
        return pairs
    return max(1, (pairs + 3) // 4)


def _sched_barrier_pairs(traits, pairs, valu_cnt, group, mask=None):
    # per (MFMA,second) pair: second op is VALU by default, EXP when ``mask`` is passed.
    mask = traits.SCHED_VALU_MASK if mask is None else mask
    pairs = _scale_sched_pairs(pairs, traits.HEAD_DIM)
    for _ in range_constexpr(pairs):
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, group)
        rocdl.sched_group_barrier(mask, valu_cnt, group)
