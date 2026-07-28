# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""dsa_fwd_v4_flydsl_kernel: DeepSeek-V4 sparse-MLA attention forward (FlyDSL MFMA).

Native FlyDSL MFMA forward for the fused single-latent (K == V) sparse-MLA form
used by the V4 attention adapter. Same public contract as the gluon / triton_v2
backends:

  * ``q``   : ``[T, H, D_QK]`` bf16   (D_QK = kv_lora_rank + rope; rope is a zero pad)
  * ``kv``  : ``[num_kv, 1, D_QK]`` bf16  (single MQA latent; V == K[:kv_lora_rank])
  * ``topk``: ``[T, TOPK]`` int32   (SWA window ++ sparse pool; -1 = invalid)
  * ``sink``: ``[H]`` fp32          (optional per-head softmax sink)
  * out ``o``   : ``[T, H, kv_lora_rank]`` bf16
  * out ``lse`` : ``[T, H]`` fp32   (sink-inclusive)

Design (adapted from the in-tree v0 SWA flash kernel v4_sla_fwd_kernel.py):
  * Grid: one workgroup per query TOKEN. The MFMA "M" axis is the HEAD axis
    (BLOCK_H heads, one head-group = all H), the "N" axis is the gathered key
    axis (BLOCK_N per tile), the contraction "K" axis is kv_lora_rank (=512).
  * Each outer tile gathers BLOCK_N latent rows kv[topk[t, tile]] into LDS
    (invalid topk == -1 -> row zeroed + column masked), runs the QK MFMA, an
    online (flash) softmax over the key axis, and the PV MFMA (V == the same
    latent, read transposed via ds_read_tr16_b64). K == V: the gathered tile is
    written to a K-LDS region (XOR swizzled, for the QK read) and a V-LDS region
    (row-major, for the transposed PV read).
  * Epilogue: fold the per-head sink into the denominator (V4), normalize, write
    O and sink-inclusive LSE = m*scale + ln(l).

Numerics mirror v0: raw-domain running max, exp2(scale*log2e*(s - m)) softmax,
bf16 truncation pack for P, finite NEG_INF (-1e30). One addition vs v0: a
running-max clamp (m<=-1e29 -> 0) so a fully-masked leading tile (common for the
SWA window of early tokens) contributes nothing instead of exp2(0)=1.

gfx950 / CDNA4 only (USE_HW_TR + K16 MFMA). Non-DMA cooperative-gather path first
(correctness); DMA / double-buffer / single-latent LDS sharing are follow-ups.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import math as math_dialect
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

_LOG2E = math.log2(math.e)  # 1.4426950408889634
_LLVM_GEP_DYNAMIC = -2147483648  # LLVM kDynamicIndex sentinel


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def build_dsa_bwd_dq_module(
    num_heads,
    kv_lora_rank,
    d_qk,
    topk,
    dtype_str="bf16",
    sm_scale=None,
    has_sink=True,
    block_n=64,
    block_h=None,
    single_latent=False,
    waves_per_eu=2,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
):
    """Build the sparse-MLA forward launcher (single-variant, gfx950).

    num_heads    : H (must be a multiple of 32).
    kv_lora_rank : D_V = 512 (contraction of QK, output dim of PV).
    d_qk         : row stride of q / kv (kv_lora_rank + rope pad, e.g. 576).
    topk         : TOPK (padded to a multiple of block_n by the launcher).
    """
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith("gfx950"), "dsa_fwd_v4_flydsl targets gfx950 (CDNA4)"
    assert dtype_str == "bf16", "bf16 only"

    HEAD_DIM = int(kv_lora_rank)  # D_V, the MFMA contraction / output dim
    D_QK = int(d_qk)  # q / kv row stride (includes rope pad)
    NUM_HEADS = int(num_heads)
    TOPK = int(topk)
    assert HEAD_DIM % 32 == 0 and HEAD_DIM >= 64
    assert NUM_HEADS % 32 == 0, f"num_heads ({NUM_HEADS}) must be a multiple of 32"

    BLOCK_N = int(block_n)
    assert BLOCK_N % 32 == 0
    assert TOPK % BLOCK_N == 0, f"TOPK ({TOPK}) must be a multiple of BLOCK_N ({BLOCK_N})"
    K_SUB_N = 32
    N_HALVES = BLOCK_N // 32  # halves of 32 columns per BLOCK_N

    WARP_SIZE = 64
    BLOCK_H = int(block_h) if block_h else NUM_HEADS  # heads per workgroup (M tile)
    assert BLOCK_H % 32 == 0 and NUM_HEADS % BLOCK_H == 0
    NUM_HEAD_GROUPS = NUM_HEADS // BLOCK_H
    NUM_WAVES = BLOCK_H // 32
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE
    ROWS_PER_WAVE = 32  # each wave owns 32 heads (MFMA M = 32)
    NUM_TILES = TOPK // BLOCK_N

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # ---- MFMA / K-step config (gfx950 CDNA4) ----
    K_STEP_QK = 16
    K_STEPS_QK = HEAD_DIM // K_STEP_QK  # 32 MFMA K-steps for the QK GEMM
    D_CHUNK = 32
    D_CHUNKS = HEAD_DIM // D_CHUNK  # 16 output chunks for the PV accumulator
    PV_K_STEP = 16
    PV_K_STEPS = K_SUB_N // PV_K_STEP  # 2 PV K-steps per 32-col sub-tile
    MFMA_LANE_K = 8

    # ---- LDS layout ----
    # SINGLE_LATENT (flash / small H, LDS-occupancy-bound): one row-major tile
    # (no swizzle, +4 pad) serves both QK (K) and PV (V via ds_read_tr). Halves LDS
    # (occ 1->2). DUAL (pro / large H, VGPR-bound): separate XOR-swizzled K tile
    # (conflict-free QK read) + row-major V tile; smaller LDS doesn't help pro
    # occupancy but the swizzle avoids QK bank conflicts.
    SINGLE_LATENT = bool(single_latent)
    if SINGLE_LATENT:
        K_STRIDE = HEAD_DIM + 4
        V_STRIDE = HEAD_DIM + 4
        LDS_V_BASE = 0
        LDS_KV_ELEMS = BLOCK_N * (HEAD_DIM + 4)
    else:
        K_STRIDE = HEAD_DIM
        V_STRIDE = HEAD_DIM + 4
        LDS_V_BASE = BLOCK_N * K_STRIDE
        LDS_KV_ELEMS = BLOCK_N * K_STRIDE + BLOCK_N * V_STRIDE

    # ---- Cooperative gather-load decomposition ----
    VEC_WIDTH = 16
    assert HEAD_DIM % VEC_WIDTH == 0
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH  # 32 threads per gathered row
    assert BLOCK_SIZE % THREADS_PER_ROW_LOAD == 0
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD
    assert BLOCK_N % ROWS_PER_BATCH_LOAD == 0
    NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"dsa_fwd_smem_H{BLOCK_H}_N{BLOCK_N}_K{TOPK}",
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    lds_valid_offset = allocator._align(lds_kv_offset + LDS_KV_ELEMS * 2, 16)  # f32 region (bytes)
    allocator.ptr = lds_valid_offset + BLOCK_N * 4

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def dsa_bwd_dq_kernel(
        Q: fx.Tensor,  # [T, H, D_QK] bf16 flat
        KV: fx.Tensor,  # [num_kv, D_QK] bf16 flat (single latent)
        dO: fx.Tensor,  # [T, H, HEAD_DIM] bf16 flat
        TopK: fx.Tensor,  # [T, TOPK] int32 flat
        LSE: fx.Tensor,  # [T, H] fp32 flat (sink-inclusive)
        Delta: fx.Tensor,  # [T, H] fp32 flat (rowsum(O*dO))
        dQ: fx.Tensor,  # [T, H, HEAD_DIM] bf16 flat (output; rope cols discarded)
        dS: fx.Tensor,  # [T, H, TOPK] bf16 flat (output for dKV kernel)
        Pout: fx.Tensor,  # [T, H, TOPK] bf16 flat (output for dKV kernel)
    ):
        elem_type = T.bf16
        compute_type = T.f32
        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        kv_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), KV)
        do_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), dO)
        dq_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), dQ)
        ds_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), dS)
        pout_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Pout)
        topk_rsrc = buffer_ops.create_buffer_resource(TopK, max_size=True)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        delta_rsrc = buffer_ops.create_buffer_resource(Delta, max_size=True)

        fm_fast = arith.FastMathFlags.fast
        v4f16_type = T.vec(4, elem_type)
        vxf16_type = T.vec(VEC_WIDTH, elem_type)
        v8f16_type = T.vec(8, elem_type)
        v16f32_type = T.vec(16, compute_type)
        mfma_pack_type = v8f16_type

        def mfma_acc(a, b, c):
            # bf16, K16 (gfx950): mfma_f32_32x32x16_bf16(result_type, [a, b, c])
            return rocdl.mfma_f32_32x32x16_bf16(v16f32_type, [a, b, c])

        # ---- LDS view ----
        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_kv_offset, elem_type, shape=(LDS_KV_ELEMS,)).get()
        lds_valid = SmemPtr(base_ptr, lds_valid_offset, compute_type, shape=(BLOCK_N,)).get()

        # ---- Thread / block indices ----
        block_id = arith.index_cast(T.index, gpu.block_idx.x)
        # block_id = token * NUM_HEAD_GROUPS + hg  (hg=0, hg_offset=0 when 1 group)
        token = block_id // arith.index(NUM_HEAD_GROUPS)
        hg_offset = (block_id % arith.index(NUM_HEAD_GROUPS)) * arith.index(BLOCK_H)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32  # 0/1

        # ds_read_b64_tr_b16 lane decomposition (hardware 4x4 transpose)
        tr_k_group = (lane % 16) // 4
        tr_col_sub = lane % 4
        tr_col_half = (lane % 32) // 16

        wave_h_offset = wave_id * ROWS_PER_WAVE  # this wave's head-row base

        # ---- ds_read_tr helper ----
        def ds_read_tr_v4f16(lds_elem_idx):
            byte_offset = lds_elem_idx * 2 + lds_kv_offset
            byte_i64 = arith.index_cast(T.i64, byte_offset)
            ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), byte_i64).result
            return rocdl.ds_read_tr16_b64(v4f16_type, ptr).result

        # ---- global index helpers (token-major sparse-MLA) ----
        HD_IDX = arith.index(HEAD_DIM)
        DQK_IDX = arith.index(D_QK)
        H_IDX = arith.index(NUM_HEADS)
        TOPK_IDX = arith.index(TOPK)

        def q_global_idx(head, col):
            # q[token, head, col] ; row stride = H * D_QK, head stride = D_QK
            return (token * H_IDX + head) * DQK_IDX + col

        def kv_global_idx(kv_row, col):
            # kv[kv_row, col] ; row stride = D_QK
            return kv_row * DQK_IDX + col

        def o_global_idx(head, col):
            # O[token, head, col] ; row stride = H * HEAD_DIM (no rope)
            return (token * H_IDX + head) * HD_IDX + col

        def _gep_load(bptr, elem_idx, vec_type):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(
                _llvm_ptr_ty(),
                bptr,
                [idx_i64],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=elem_type,
                noWrapFlags=0,
            )
            return _llvm.LoadOp(vec_type, gep.result).result

        def _gep_store(val, bptr, elem_idx):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(
                _llvm_ptr_ty(),
                bptr,
                [idx_i64],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=elem_type,
                noWrapFlags=0,
            )
            _llvm.StoreOp(val, gep.result)

        def load_global_mfma_pack(bptr, base_idx):
            return _gep_load(bptr, base_idx, mfma_pack_type)

        def load_global_f16xN(bptr, base_idx):
            return _gep_load(bptr, base_idx, vxf16_type)

        def bf16_trunc_pack_v8(f32_vals):
            _v4i32 = T.vec(4, T.i32)
            _c16 = arith.constant(16, type=T.i32)
            _cmask = arith.constant(0xFFFF0000, type=T.i32)
            pairs = []
            for j in range_constexpr(4):
                a = arith.ArithValue(f32_vals[j * 2]).bitcast(T.i32)
                b = arith.ArithValue(f32_vals[j * 2 + 1]).bitcast(T.i32)
                p = arith.OrIOp(arith.AndIOp(b, _cmask).result, arith.ShRUIOp(a, _c16).result).result
                pairs.append(p)
            return vector.bitcast(v8f16_type, vector.from_elements(_v4i32, pairs))

        # ---- cooperative decomposition ----
        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        c_neg_inf = arith.constant(-1.0e30, type=compute_type)
        c_zero_f = arith.constant(0.0, type=compute_type)
        c_one_f = arith.constant(1.0, type=compute_type)
        c_zero_v16f32 = arith.constant_vector(0.0, v16f32_type)
        c_zero_vxf16 = arith.constant_vector(0.0, vxf16_type)
        c_sm_scale_log2e = arith.constant(sm_scale * _LOG2E, type=compute_type)
        c_sm_scale_f = arith.constant(float(sm_scale), type=compute_type)

        width_i32 = arith.constant(WARP_SIZE, type=T.i32)
        shuf_32_i32 = arith.constant(32, type=T.i32)

        def reduction_peer(v_f32):
            return arith.ArithValue(v_f32).shuffle_xor(shuf_32_i32, width_i32)

        # ---- K XOR swizzle (col ^ ((row & 7) << 4)) ----
        def _k_swizzle(row_idx, col_idx):
            mask = (row_idx & arith.index(0x7)) << arith.index(4)
            return col_idx ^ mask

        def _swz_none(row_idx, col_idx):
            return col_idx

        # build-time layout selection (no traced `if`)
        _swz_k = _swz_none if SINGLE_LATENT else _k_swizzle

        def _store_row_single(lds_row, col, vec):
            vector.store(vec, lds, [lds_row * K_STRIDE + col])

        def _store_row_dual(lds_row, col, vec):
            vector.store(vec, lds, [lds_row * K_STRIDE + _k_swizzle(lds_row, col)])
            vector.store(vec, lds, [arith.index(LDS_V_BASE) + lds_row * V_STRIDE + col])

        _store_row = _store_row_single if SINGLE_LATENT else _store_row_dual

        # ---- Preload Q + dO B-operand packs and per-head lse/delta ----
        # head row = hg_offset + wave_h_offset + lane_mod_32 (MFMA M axis)
        head_row = hg_offset + wave_h_offset + lane_mod_32
        head_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, head_row, H_IDX)
        head_row_safe = arith.select(head_in_bounds, head_row, arith.index(0))
        c_zero_mfma_pack = arith.constant_vector(0.0, mfma_pack_type)
        q_b_packs = []
        do_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            q_b_packs.append(
                arith.select(
                    head_in_bounds,
                    load_global_mfma_pack(q_ptr, q_global_idx(head_row_safe, col)),
                    c_zero_mfma_pack,
                )
            )
            do_b_packs.append(
                arith.select(
                    head_in_bounds,
                    load_global_mfma_pack(do_ptr, o_global_idx(head_row_safe, col)),
                    c_zero_mfma_pack,
                )
            )
        c_log2e = arith.constant(_LOG2E, type=compute_type)
        head_flat_i32 = arith.index_cast(T.i32, token * H_IDX + head_row_safe)
        lse_val = buffer_ops.buffer_load(lse_rsrc, head_flat_i32, vec_width=1, dtype=T.f32)
        delta_val = buffer_ops.buffer_load(delta_rsrc, head_flat_i32, vec_width=1, dtype=T.f32)
        neg_lse_log2e = arith.MulFOp(
            lse_val, arith.SubFOp(c_zero_f, c_log2e, fastmath=fm_fast).result, fastmath=fm_fast
        ).result

        # ---- outer loop over TOPK tiles: dQ += dS @ K ----
        init_args = []
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v16f32)

        for tile_idx, inner_iter_args, loop_results in scf.for_(
            arith.index(0),
            arith.index(NUM_TILES),
            arith.index(1),
            iter_args=init_args,
        ):
            dq_accs = [inner_iter_args[i] for i in range_constexpr(D_CHUNKS)]

            tile_topk_start = tile_idx * arith.index(BLOCK_N)

            coop_gather_tile_dyn = tile_topk_start
            # gather this tile
            for batch in range_constexpr(NUM_BATCHES_KV):
                lds_row = load_row_in_batch + batch * ROWS_PER_BATCH_LOAD
                topk_pos = coop_gather_tile_dyn + lds_row
                topk_flat = token * TOPK_IDX + topk_pos
                topk_flat_i32 = arith.index_cast(T.i32, topk_flat)
                idx_raw = buffer_ops.buffer_load(topk_rsrc, topk_flat_i32, vec_width=1, dtype=T.i32)
                valid = arith.cmpi(arith.CmpIPredicate.sge, idx_raw, arith.constant(0, type=T.i32))
                safe_i32 = arith.select(valid, idx_raw, arith.constant(0, type=T.i32))
                kv_row = arith.index_cast(T.index, safe_i32)
                g_idx = kv_global_idx(kv_row, load_col_base)
                vec_raw = load_global_f16xN(kv_ptr, g_idx)
                vec = arith.select(valid, vec_raw, c_zero_vxf16)
                _store_row(lds_row, load_col_base, vec)
                is_col0 = arith.cmpi(arith.CmpIPredicate.eq, load_col_base, arith.index(0))
                _if_c0 = scf.IfOp(is_col0)
                with ir.InsertionPoint(_if_c0.then_block):
                    mask_add = arith.select(valid, c_zero_f, c_neg_inf)
                    vector.store(
                        vector.from_elements(T.vec(1, compute_type), [mask_add]),
                        lds_valid,
                        [lds_row],
                    )
                    scf.YieldOp([])
            gpu.barrier()

            # ==== GEMM1: QK. bulk-read K packs, pipelined MFMA ====
            k_hi_offset = K_SUB_N * K_STRIDE

            def _k_idx_lo(ks):
                col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                return lane_mod_32 * K_STRIDE + _swz_k(lane_mod_32, col)

            def _k_idx_hi(ks):
                col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                return k_hi_offset + lane_mod_32 * K_STRIDE + _swz_k(lane_mod_32, col)

            _QK_PREFETCH_DEPTH = 2
            k_packs_lo = [None] * K_STEPS_QK
            k_packs_hi = [None] * K_STEPS_QK
            for p in range_constexpr(_QK_PREFETCH_DEPTH):
                k_packs_lo[p] = vector.load_op(mfma_pack_type, lds, [_k_idx_lo(p)])
                if N_HALVES == 2:
                    k_packs_hi[p] = vector.load_op(mfma_pack_type, lds, [_k_idx_hi(p)])

            s_acc_lo = c_zero_v16f32
            s_acc_hi = c_zero_v16f32
            dp_acc_lo = c_zero_v16f32
            dp_acc_hi = c_zero_v16f32
            for ks in range_constexpr(K_STEPS_QK):
                s_acc_lo = mfma_acc(k_packs_lo[ks], q_b_packs[ks], s_acc_lo)
                dp_acc_lo = mfma_acc(k_packs_lo[ks], do_b_packs[ks], dp_acc_lo)
                if N_HALVES == 2:
                    s_acc_hi = mfma_acc(k_packs_hi[ks], q_b_packs[ks], s_acc_hi)
                    dp_acc_hi = mfma_acc(k_packs_hi[ks], do_b_packs[ks], dp_acc_hi)
                if ks + _QK_PREFETCH_DEPTH < K_STEPS_QK:
                    k_packs_lo[ks + _QK_PREFETCH_DEPTH] = vector.load_op(
                        mfma_pack_type, lds, [_k_idx_lo(ks + _QK_PREFETCH_DEPTH)]
                    )
                    if N_HALVES == 2:
                        k_packs_hi[ks + _QK_PREFETCH_DEPTH] = vector.load_op(
                            mfma_pack_type, lds, [_k_idx_hi(ks + _QK_PREFETCH_DEPTH)]
                        )

            # ==== P = exp(scale*S - lse); dS = P*(dP - delta)*scale (per element) ====
            # Column mapping (v0 32x32 C-layout): tile col = lane_div_32*4 +
            # (r//4)*8 + (r%4) (+ K_SUB_N for the hi half).
            lane_off = lane_div_32 * arith.index(4)

            def _ds_packs(s_acc, dp_acc, half):
                ds_vals = []
                for r in range_constexpr(16):
                    s_r = vector.extract(s_acc, static_position=[r], dynamic_position=[])
                    dp_r = vector.extract(dp_acc, static_position=[r], dynamic_position=[])
                    col = lane_off + arith.index((r % 4) + (r // 4) * 8 + half * K_SUB_N)
                    mv = vector.extract(
                        vector.load_op(T.vec(1, compute_type), lds_valid, [col]),
                        static_position=[0],
                        dynamic_position=[],
                    )
                    s_m = arith.AddFOp(s_r, mv, fastmath=fm_fast).result  # invalid -> -inf
                    # P = exp2(scale*log2e*s_m - lse*log2e)
                    p_arg = math_dialect.fma(s_m, c_sm_scale_log2e, neg_lse_log2e)
                    p_r = arith.ArithValue(p_arg).exp2(fastmath=fm_fast)
                    # dS = P * (dP - delta) * scale
                    dp_md = arith.SubFOp(dp_r, delta_val, fastmath=fm_fast).result
                    ds_r = arith.MulFOp(
                        arith.MulFOp(p_r, dp_md, fastmath=fm_fast).result, c_sm_scale_f, fastmath=fm_fast
                    ).result
                    ds_vals.append(ds_r)
                    # store dS and P to [T, H, TOPK] for the dKV kernel (kv_pos in [0,TOPK))
                    kv_pos = tile_topk_start + col
                    dsp_idx = (token * H_IDX + head_row) * TOPK_IDX + kv_pos
                    _gep_store(arith.trunc_f(elem_type, ds_r), ds_ptr, dsp_idx)
                    _gep_store(arith.trunc_f(elem_type, p_r), pout_ptr, dsp_idx)
                packs = []
                for pks in range_constexpr(PV_K_STEPS):
                    packs.append(bf16_trunc_pack_v8(ds_vals[pks * 8 : pks * 8 + 8]))
                return packs

            ds_packs_lo = _ds_packs(s_acc_lo, dp_acc_lo, 0)
            ds_packs_hi = _ds_packs(s_acc_hi, dp_acc_hi, 1) if N_HALVES == 2 else []

            # ==== GEMM2: dQ += dS @ K  (K == V, ds_read_tr) ====
            v_base = arith.index(LDS_V_BASE)
            _steps = [(dc, pks) for dc in range(D_CHUNKS) for pks in range(PV_K_STEPS)]
            TOTAL_PV = len(_steps)

            def _read_v_pack(step_idx):
                dc, pks = _steps[step_idx]
                d_col = arith.index(dc * D_CHUNK) + tr_col_half * 16 + tr_col_sub * 4
                k_row = arith.index(pks * PV_K_STEP) + lane_div_32 * 4 + tr_k_group
                lds_lo = v_base + k_row * V_STRIDE + d_col
                vl_a = ds_read_tr_v4f16(lds_lo)
                vl_b = ds_read_tr_v4f16(lds_lo + arith.index(8 * V_STRIDE))
                vl = vector.shuffle(vl_a, vl_b, [0, 1, 2, 3, 4, 5, 6, 7])
                vh = None
                if N_HALVES == 2:
                    lds_hi = lds_lo + arith.index(K_SUB_N * V_STRIDE)
                    vh_a = ds_read_tr_v4f16(lds_hi)
                    vh_b = ds_read_tr_v4f16(lds_hi + arith.index(8 * V_STRIDE))
                    vh = vector.shuffle(vh_a, vh_b, [0, 1, 2, 3, 4, 5, 6, 7])
                return vl, vh

            for si in range_constexpr(TOTAL_PV):
                dc, pks = _steps[si]
                v_lo_cur, v_hi_cur = _read_v_pack(si)
                dq_accs[dc] = mfma_acc(v_lo_cur, ds_packs_lo[pks], dq_accs[dc])
                if N_HALVES == 2:
                    dq_accs[dc] = mfma_acc(v_hi_cur, ds_packs_hi[pks], dq_accs[dc])

            # protect this tile's V/K LDS reads from the next tile's gather writes
            gpu.barrier()

            yield dq_accs

        # ---- epilogue: store dQ_lora [token, head, :HEAD_DIM] ----
        dq_finals = [loop_results[dc] for dc in range_constexpr(D_CHUNKS)]
        _o_guard = scf.IfOp(head_in_bounds, [], has_else=False)
        with ir.InsertionPoint(_o_guard.then_block):
            for dc in range_constexpr(D_CHUNKS):
                for r in range_constexpr(16):
                    dq_val = vector.extract(dq_finals[dc], static_position=[r], dynamic_position=[])
                    dq_f16 = arith.trunc_f(elem_type, dq_val)
                    d_row_rel = lane_div_32 * 4 + (r // 4) * 8 + (r % 4)
                    d_col = arith.index(dc * D_CHUNK) + d_row_rel
                    # dQ is [T, H, d_qk] (matches q); write the :HEAD_DIM lora cols
                    _gep_store(dq_f16, dq_ptr, q_global_idx(head_row, d_col))
            scf.YieldOp([])

    @flyc.jit
    def launch_dsa_bwd_dq(
        Q: fx.Tensor,
        KV: fx.Tensor,
        dO: fx.Tensor,
        TopK: fx.Tensor,
        LSE: fx.Tensor,
        Delta: fx.Tensor,
        dQ: fx.Tensor,
        dS: fx.Tensor,
        Pout: fx.Tensor,
        total_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        grid_x = arith.index_cast(T.index, total_tokens) * arith.index(NUM_HEAD_GROUPS)
        launcher = dsa_bwd_dq_kernel(Q, KV, dO, TopK, LSE, Delta, dQ, dS, Pout)

        if waves_per_eu is not None and int(waves_per_eu) >= 1:
            _wpe = int(waves_per_eu)
            for op in ctx.gpu_module_body.operations:
                if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, _wpe)

        _fwgs = int(BLOCK_SIZE)
        flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
        for op in ctx.gpu_module_body.operations:
            if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

        passthrough_entries = []
        if daz:
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("denormal-fp-math-f32"),
                        ir.StringAttr.get("preserve-sign,preserve-sign"),
                    ]
                )
            )
            passthrough_entries.append(
                ir.ArrayAttr.get([ir.StringAttr.get("no-nans-fp-math"), ir.StringAttr.get("true")])
            )
            passthrough_entries.append(
                ir.ArrayAttr.get([ir.StringAttr.get("unsafe-fp-math"), ir.StringAttr.get("true")])
            )
        for op in ctx.gpu_module_body.operations:
            if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    _compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_compile_hints):
            return launch_dsa_bwd_dq(*args, **kwargs)

    return _launch
