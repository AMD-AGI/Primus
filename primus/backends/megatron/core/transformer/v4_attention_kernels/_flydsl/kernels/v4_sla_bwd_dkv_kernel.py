# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""v4_swa_bwd_dkv: V4 SWA-causal attention backward dK/dV kernel for FlyDSL.

Strategy:
  * One workgroup per (batch, n_block). K and V loaded once into LDS
    and stay resident for the program's lifetime (no LUT, no atomics).
  * Inner loop iterates qhid in [0, HQ); for each head, iterates m-blocks
    bounded by the SWA window. dK and dV f32 accumulators live in VGPRs.
  * MQA head-loop accumulator: one dk/dv slice per (b, n_block) gets
    all HQ heads accumulated -- deterministic, no atomics.
  * SWA + causal mask per element. LSE is RAW-domain
    (qk*sm_scale + ln(l)); p = exp(qk*sm_scale - lse).
  * sm_scale applied to dK exactly once post-loop (Triton P57 cr=0).
  * dK / dV written as f32 (matches launcher's dk_fp32 / dv_fp32 buffers).

LDS budget at BLOCK_N=32, BLOCK_M2=32, D=512:
    K = 32 KB, V = 32 KB, DO = 32 KB, Q = 32 KB, pT = 2 KB
    Total = 130 KB <= 160 KB.
Auto-fallback: if predicted > 160 KB, drop DO/Q LDS scratches and
read Q/DO directly to register packs from HBM (mirrors STEP-1b dq).
"""

import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from kernels.kernels_common import dtype_to_elem_type
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import memref as _memref, scf, fly as _fly, llvm as _llvm, math as math_dialect

KERNEL_NAME = "v4_swa_bwd_dkv_kernel"

_LLVM_GEP_DYNAMIC = -2147483648


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def build_v4_swa_bwd_dkv_module(
    num_heads,
    head_dim,
    swa_window,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    flat_work_group_size=None,
    block_n=None,
    block_m2=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    layout_bhld=True,
    mqa_kv=True,
):
    """Build the V4 SWA backward dK/dV launcher (MQA only)."""
    gpu_arch = get_hip_arch()

    if block_n is None:
        # R6k-1: D-split wave remap default. BLOCK_N=16 halves per-lane
        # fp32 accumulator footprint (256 -> 128 fp32) by partitioning D
        # across waves instead of N. Both waves cover the same 16 KV
        # rows; each wave owns half of the D-chunks of dk/dv.
        BLOCK_N = 16
    else:
        BLOCK_N = int(block_n)
    if block_m2 is None:
        BLOCK_M2 = 32
    else:
        BLOCK_M2 = int(block_m2)
    WARP_SIZE = 64
    # R6k-1: D-split mode. When BLOCK_N == 16 (matches MFMA-N tile),
    # force NUM_WAVES=2 and partition D between the two waves instead
    # of partitioning N. Each wave then carries only half the dk/dv
    # accumulator footprint. For BLOCK_N > 16 we keep the legacy N-split.
    D_SPLIT_WAVES = (BLOCK_N == 16)
    if D_SPLIT_WAVES:
        NUM_WAVES = 2
        ROWS_PER_WAVE = 16  # both waves work on the same 16 rows
    else:
        # Legacy N-split: each wave gets BLOCK_N/NUM_WAVES rows.
        NUM_WAVES = max(1, BLOCK_N // 16)
        ROWS_PER_WAVE = BLOCK_N // NUM_WAVES
    if flat_work_group_size is None:
        flat_work_group_size = NUM_WAVES * WARP_SIZE
    BLOCK_SIZE = flat_work_group_size

    ENABLE_LDS_VEC16 = (os.getenv("FLYDSL_SLA_FWD_ENABLE_LDS_VEC16", "1") == "1")
    USE_K16 = gpu_arch.startswith("gfx950")
    # R6b: MFMA 16x16x32 K-step = 32; A/B-frag = 8 bf16 per lane.
    assert USE_K16, "R6b dkv requires gfx950 (MFMA 16x16x32 bf16)."
    K_STEP_QK = 32
    K_STEPS_QK = head_dim // K_STEP_QK
    # R6b: each MFMA 16x16x32 tile produces a 16-wide N-chunk; D_CHUNK = 16 cols.
    D_CHUNK = 16
    D_CHUNKS = head_dim // D_CHUNK
    # R6k-1: per-wave D-chunk count. In D-split mode each wave owns half.
    if D_SPLIT_WAVES:
        assert D_CHUNKS % NUM_WAVES == 0, (
            f"D_CHUNKS ({D_CHUNKS}) must be divisible by NUM_WAVES "
            f"({NUM_WAVES}) for D-split mode.")
        D_CHUNKS_LOCAL = D_CHUNKS // NUM_WAVES
    else:
        D_CHUNKS_LOCAL = D_CHUNKS
    K_STEPS_PT = BLOCK_M2 // K_STEP_QK
    # R6b: GEMM1/GEMM3 cover m_col [0..BLOCK_M2) with multiple 16-col MFMA-N tiles per ks.
    assert BLOCK_M2 % 16 == 0, f"BLOCK_M2 must be a multiple of 16 (MFMA-N), got {BLOCK_M2}"
    M_TILES = BLOCK_M2 // 16

    assert BLOCK_N % NUM_WAVES == 0
    assert ROWS_PER_WAVE == 16, f"ROWS_PER_WAVE must equal 16 for MFMA 16x16x32, got {ROWS_PER_WAVE}"
    assert head_dim % 32 == 0
    assert head_dim >= 64
    assert flat_work_group_size in (64, 128, 256, 512)
    assert dtype_str == "bf16", "R6b dkv currently only supports bf16."
    assert BLOCK_N % 16 == 0
    assert BLOCK_M2 % K_STEP_QK == 0, (
        f"BLOCK_M2 ({BLOCK_M2}) must be a multiple of MFMA K-step "
        f"({K_STEP_QK}) for the dV/dK GEMMs.")
    assert isinstance(swa_window, int) and swa_window > 0
    assert mqa_kv, "v4_swa_bwd_dkv currently only supports MQA (HK=1)."

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim

    K_STRIDE = HEAD_DIM
    K_SWZ_ROW_MASK = (K_STRIDE // 16) - 1
    assert K_SWZ_ROW_MASK >= 0
    assert (K_SWZ_ROW_MASK & (K_SWZ_ROW_MASK + 1)) == 0
    V_STRIDE = HEAD_DIM

    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8
    assert HEAD_DIM % VEC_WIDTH == 0
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    assert BLOCK_SIZE % THREADS_PER_ROW_LOAD == 0
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    LDS_DO_STRIDE = HEAD_DIM
    LDS_DO_ELEMS = BLOCK_M2 * LDS_DO_STRIDE
    LDS_Q_STRIDE = HEAD_DIM
    LDS_Q_ELEMS = BLOCK_M2 * LDS_Q_STRIDE
    LDS_PT_STRIDE = BLOCK_M2
    LDS_PT_ELEMS = BLOCK_N * LDS_PT_STRIDE

    _LDS_LIMIT_BYTES = 160 * 1024

    def _predict_lds_bytes(use_lds_for_q_do):
        b = (LDS_K_TILE_SIZE + LDS_V_TILE_SIZE + LDS_PT_ELEMS) * 2
        if use_lds_for_q_do:
            b += (LDS_DO_ELEMS + LDS_Q_ELEMS) * 2
        return b

    USE_LDS_FOR_Q_DO = True
    _pred_full = _predict_lds_bytes(True)
    if _pred_full > _LDS_LIMIT_BYTES:
        USE_LDS_FOR_Q_DO = False
        _pred_min = _predict_lds_bytes(False)
        if _pred_min > _LDS_LIMIT_BYTES:
            raise RuntimeError(
                f"v4_swa_bwd_dkv: minimal LDS {_pred_min}B > limit "
                f"{_LDS_LIMIT_BYTES}B at D={head_dim}, BLOCK_N={BLOCK_N}, "
                f"BLOCK_M2={BLOCK_M2}."
            )
        import sys as _sys
        print(f"[v4_swa_bwd_dkv] auto-fallback: LDS {_pred_full}B > "
              f"{_LDS_LIMIT_BYTES}B; disabling Q/DO LDS "
              f"({_pred_full} -> {_pred_min})",
              file=_sys.stderr, flush=True)
    else:
        import sys as _sys
        print(f"[v4_swa_bwd_dkv] LDS OK: predicted {_pred_full}B <= "
              f"{_LDS_LIMIT_BYTES}B at D={head_dim}, BLOCK_N={BLOCK_N}, "
              f"BLOCK_M2={BLOCK_M2}, USE_LDS_FOR_Q_DO=True",
              file=_sys.stderr, flush=True)

    allocator = SmemAllocator(
        None, arch=gpu_arch,
        global_sym_name=(
            f"v4_swa_bwd_dkv_smem_N{BLOCK_N}_M2_{BLOCK_M2}_W{swa_window}"
            f"_MQ{int(mqa_kv)}_QDO{int(USE_LDS_FOR_Q_DO)}"
            f"_DS{int(D_SPLIT_WAVES)}"),
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    LDS_V_BASE = LDS_K_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TILE_SIZE + LDS_V_TILE_SIZE
    allocator.ptr = lds_kv_offset + LDS_KV_TOTAL_SIZE * 2

    lds_pt_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_pt_offset + LDS_PT_ELEMS * 2

    if USE_LDS_FOR_Q_DO:
        lds_do_offset = allocator._align(allocator.ptr, 16)
        allocator.ptr = lds_do_offset + LDS_DO_ELEMS * 2
        lds_q_offset = allocator._align(allocator.ptr, 16)
        allocator.ptr = lds_q_offset + LDS_Q_ELEMS * 2
    else:
        lds_do_offset = None
        lds_q_offset = None

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def v4_swa_bwd_dkv_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DOS: fx.Tensor,
        LSE: fx.Tensor,
        DELTAS: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32
        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        k_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), K)
        v_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), V)
        do_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DOS)
        dk_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DK)
        dv_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DV)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        deltas_rsrc = buffer_ops.create_buffer_resource(DELTAS, max_size=True)

        fm_fast = arith.FastMathFlags.fast
        vxf16_type = T.vec(VEC_WIDTH, elem_type)
        v8f16_type = T.vec(8, elem_type)
        # R6M: tr16 returns 4 bf16 per lane (v4f16-typed). Two calls + shuffle
        # form the MFMA 16x16x32 B-frag (8 bf16) via HW transpose.
        v4f16_type = T.vec(4, elem_type)
        # R6b: MFMA 16x16x32 C-frag = 4 fp32 per lane.
        v4f32_type = T.vec(4, compute_type)
        mfma_pack_type = v8f16_type
        MFMA_LANE_K = 8
        v1_elem_type = T.vec(1, elem_type)
        _mfma_zero = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

        def mfma_acc(a, b, c):
            # rocdl.mfma_f32_16x16x32_bf16 is the wrapped form: takes
            # (result_type, [operands]) and returns the Value directly.
            return rocdl.mfma_f32_16x16x32_bf16(
                v4f32_type,
                [a, b, c, _mfma_zero, _mfma_zero, _mfma_zero],
            )

        seq_len_q_v = arith.index_cast(T.index, seq_len_q)
        seq_len_k_v = arith.index_cast(T.index, seq_len_k)

        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(base_ptr, lds_kv_offset, elem_type,
                         shape=(LDS_KV_TOTAL_SIZE,)).get()
        lds_pt = SmemPtr(base_ptr, lds_pt_offset, elem_type,
                         shape=(LDS_PT_ELEMS,)).get()
        if USE_LDS_FOR_Q_DO:
            lds_do = SmemPtr(base_ptr, lds_do_offset, elem_type,
                             shape=(LDS_DO_ELEMS,)).get()
            lds_q = SmemPtr(base_ptr, lds_q_offset, elem_type,
                            shape=(LDS_Q_ELEMS,)).get()

        block_id = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        # R6b MFMA 16x16x32 lane decomposition:
        #   A-frag: A[lane_mod_16, ks*32 + lane_div_16*8 + 0..7]
        #   B-frag: B[ks*32 + lane_div_16*8 + 0..7, lane_mod_16]
        #   C-frag: C[lane_div_16*4 + ii, lane_mod_16]  for ii in 0..3
        lane_mod_16 = lane % 16
        lane_div_16 = lane // 16
        # R6M: ds_read_tr16_b64 per-lane decomposition (matches proto).
        tr_k_group = lane_mod_16 // arith.index(4)
        tr_col_sub = lane_mod_16 % arith.index(4)

        # R6k-1: in D-split mode both waves cover the same 16 KV rows
        # (wave_n_offset == 0) and instead carry disjoint D-chunks of
        # dk/dv. wave_d_col_offset is the wave's D-column base offset
        # (0 for wave 0, D_CHUNKS_LOCAL*D_CHUNK for wave 1).
        if D_SPLIT_WAVES:
            wave_n_offset = arith.index(0)
            wave_d_col_offset = wave_id * arith.index(D_CHUNKS_LOCAL * D_CHUNK)
        else:
            wave_n_offset = wave_id * ROWS_PER_WAVE
            wave_d_col_offset = arith.index(0)

        # R6S: Hoist a SINGLE per-lane base byte-pointer for DO/Q LDS
        # tr16 reads. Each ds_read_tr16_b64 then uses
        # `base + constexpr byte imm` so LLVM ISel folds the
        # (m_step, dc, addr_0/1) deltas into the `offset:` immediate
        # of the LDS instruction. Removes the per-call IntToPtrOp ->
        # 32 AGPR spill chain that was throttling GEMM2/4 to 28.6%
        # MFMA duty (vs Triton's 53.7%).
        if USE_LDS_FOR_Q_DO:
            assert LDS_DO_STRIDE == LDS_Q_STRIDE, (
                'R6S tr16 base+imm assumes DO and Q share STRIDE')
            tr_lane_base_elem = (
                lane_div_16 * arith.index(MFMA_LANE_K * LDS_DO_STRIDE)
                + tr_k_group * arith.index(LDS_DO_STRIDE)
                + wave_d_col_offset
                + tr_col_sub * arith.index(4)
            )
            tr_lane_base_byte = tr_lane_base_elem * arith.index(2)
            tr_lane_base_i64 = arith.index_cast(T.i64, tr_lane_base_byte)
            tr_lane_base_do_byte_i64 = arith.AddIOp(
                tr_lane_base_i64,
                arith.constant(lds_do_offset, type=T.i64),
            ).result
            tr_lane_base_q_byte_i64 = arith.AddIOp(
                tr_lane_base_i64,
                arith.constant(lds_q_offset, type=T.i64),
            ).result
            tr_lane_base_do_ptr = _llvm.IntToPtrOp(
                _llvm_lds_ptr_ty(), tr_lane_base_do_byte_i64
            ).result
            tr_lane_base_q_ptr = _llvm.IntToPtrOp(
                _llvm_lds_ptr_ty(), tr_lane_base_q_byte_i64
            ).result

        # 6N-2 iter2: de-dup helper (Q/DO B-frag loads also gated). Each m-tile mt is owned by wave (mt % NUM_WAVES).
        # In D-split mode the owning wave runs GEMM1/softmax/pT-write/GEMM3/dsT
        # for that mt; the other wave skips. GEMM2/4 use the SHARED pT/dsT LDS.
        if D_SPLIT_WAVES:
            owns_mt_preds = [
                arith.cmpi(arith.CmpIPredicate.eq, wave_id,
                           arith.index(mt % NUM_WAVES))
                for mt in range(M_TILES)
            ]
        else:
            owns_mt_preds = None

        num_n_blocks = (seq_len_k_v + BLOCK_N - 1) // BLOCK_N
        n_block_idx = block_id % num_n_blocks
        batch_idx = block_id // num_n_blocks
        kv_start = n_block_idx * arith.index(BLOCK_N)

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        # MQA: KV stride drops head.
        bh_base_tokens_kv = batch_idx * seq_len_k_v

        def bh_base_tokens_q_of(qhid_index):
            return (batch_idx * NUM_HEADS + qhid_index) * seq_len_q_v

        def global_idx_q(qhid_index, token_idx, col):
            return ((bh_base_tokens_q_of(qhid_index) + token_idx)
                    * arith.index(HEAD_DIM) + col)

        def global_idx_kv(token_idx, col):
            return (bh_base_tokens_kv + token_idx) * arith.index(HEAD_DIM) + col

        def _gep_load(base_ptr_, elem_idx, vec_type, et=elem_type):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(_llvm_ptr_ty(), base_ptr_, [idx_i64],
                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                              elem_type=et, noWrapFlags=0)
            return _llvm.LoadOp(vec_type, gep.result).result

        def _gep_store_f32(val, base_ptr_, elem_idx):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(_llvm_ptr_ty(), base_ptr_, [idx_i64],
                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                              elem_type=T.f32, noWrapFlags=0)
            _llvm.StoreOp(val, gep.result)

        def load_global_mfma_pack(base_ptr_, base_idx):
            return _gep_load(base_ptr_, base_idx, mfma_pack_type)

        def load_global_f16xN(base_ptr_, base_idx):
            return _gep_load(base_ptr_, base_idx, vxf16_type)

        def _k_swizzle(row_idx, col_idx):
            mask = (row_idx & arith.index(K_SWZ_ROW_MASK)) << arith.index(4)
            return col_idx ^ mask

        if ROWS_PER_BATCH_LOAD >= BLOCK_N:
            NUM_BATCHES_KV = 1
            KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
        else:
            assert BLOCK_N % ROWS_PER_BATCH_LOAD == 0
            NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
            KV_NEEDS_GUARD = False

        if ROWS_PER_BATCH_LOAD >= BLOCK_M2:
            NUM_BATCHES_M = 1
            M_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_M2
        else:
            assert BLOCK_M2 % ROWS_PER_BATCH_LOAD == 0
            NUM_BATCHES_M = BLOCK_M2 // ROWS_PER_BATCH_LOAD
            M_NEEDS_GUARD = False

        c_zero_vxf16 = arith.constant_vector(0.0, vxf16_type)
        c_zero_mfma_pack = arith.constant_vector(0.0, mfma_pack_type)
        c_zero_elem = arith.constant(0.0, type=elem_type)
        c_neg_inf = arith.constant(-1.0e30, type=compute_type)
        c_sm_scale = arith.constant(sm_scale, type=compute_type)
        # R6b: 4-elem fp32 acc per MFMA 16x16x32 op.
        v4f32_zero = arith.constant_vector(0.0, v4f32_type)

        def coop_load_k(tile_start):
            k_base = arith.index(0)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                in_bounds = arith.cmpi(
                    arith.CmpIPredicate.slt, row_idx, seq_len_k_v)
                row_safe = arith.select(in_bounds, row_idx, arith.index(0))
                g_idx = global_idx_kv(row_safe, load_col_base)
                vec = load_global_f16xN(k_ptr, g_idx)
                vec_safe = arith.select(in_bounds, vec, c_zero_vxf16)
                lds_row = load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch + row_offset,
                        arith.index(BLOCK_N))
                    _if_k = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_k.then_block):
                        swz_col = _k_swizzle(lds_row, load_col_base)
                        lds_idx = k_base + lds_row * K_STRIDE + swz_col
                        vector.store(vec_safe, lds_kv, [lds_idx])
                        scf.YieldOp([])
                else:
                    swz_col = _k_swizzle(lds_row, load_col_base)
                    lds_idx = k_base + lds_row * K_STRIDE + swz_col
                    vector.store(vec_safe, lds_kv, [lds_idx])

        def coop_load_v(tile_start):
            v_base = arith.index(LDS_V_BASE)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                in_bounds = arith.cmpi(
                    arith.CmpIPredicate.slt, row_idx, seq_len_k_v)
                row_safe = arith.select(in_bounds, row_idx, arith.index(0))
                g_idx = global_idx_kv(row_safe, load_col_base)
                vec = load_global_f16xN(v_ptr, g_idx)
                vec_safe = arith.select(in_bounds, vec, c_zero_vxf16)
                lds_row = load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch + row_offset,
                        arith.index(BLOCK_N))
                    _if_v = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_v.then_block):
                        swz_col = _k_swizzle(lds_row, load_col_base)
                        lds_idx = v_base + lds_row * V_STRIDE + swz_col
                        vector.store(vec_safe, lds_kv, [lds_idx])
                        scf.YieldOp([])
                else:
                    swz_col = _k_swizzle(lds_row, load_col_base)
                    lds_idx = v_base + lds_row * V_STRIDE + swz_col
                    vector.store(vec_safe, lds_kv, [lds_idx])

        # ---- PROLOGUE: K, V into LDS (persistent for program lifetime) ----
        coop_load_k(kv_start)
        coop_load_v(kv_start)
        gpu.barrier()

        # R6b: K/V LDS A-frag load uses lane_mod_16 (16-row MFMA) and lane_div_16*8 for col.
        # Swizzle mask key is the per-wave KV row (wave_n_offset + lane_mod_16), but since
        # NUM_WAVES <= 2 and (wave_n_offset & K_SWZ_ROW_MASK) is 0 at D=64 (and bit-aligned
        # at D=512 once wave_n_offset is folded in below), we recompute the mask per call.
        def _k_idx_wave(ks):
            kv_row = wave_n_offset + lane_mod_16
            col = arith.index(ks * K_STEP_QK) + lane_div_16 * MFMA_LANE_K
            mask = (kv_row & arith.index(K_SWZ_ROW_MASK)) << arith.index(4)
            return (kv_row * arith.index(K_STRIDE) + (col ^ mask))

        def _v_idx_wave(ks):
            kv_row = wave_n_offset + lane_mod_16
            col = arith.index(ks * K_STEP_QK) + lane_div_16 * MFMA_LANE_K
            mask = (kv_row & arith.index(K_SWZ_ROW_MASK)) << arith.index(4)
            return (arith.index(LDS_V_BASE)
                    + kv_row * arith.index(V_STRIDE) + (col ^ mask))

        # ---- SWA + m-loop bounds (constant across head loop) ----
        SWA = arith.index(swa_window)
        BN_idx = arith.index(BLOCK_N)
        BM2_idx = arith.index(BLOCK_M2)
        _one_idx = arith.index(1)
        n_block_lo = kv_start
        n_block_hi = kv_start + BN_idx
        m_loop_start = (n_block_lo // BM2_idx) * BM2_idx
        _m_end_raw = n_block_hi + SWA - _one_idx
        _le_seq = arith.cmpi(arith.CmpIPredicate.sle, _m_end_raw, seq_len_q_v)
        _m_end_cl = arith.select(_le_seq, _m_end_raw, seq_len_q_v)
        m_loop_end = ((_m_end_cl + BM2_idx - _one_idx) // BM2_idx) * BM2_idx

        # R6k-1: each wave only carries D_CHUNKS_LOCAL chunks of dv and dk.
        outer_carry = ([v4f32_zero for _ in range(D_CHUNKS_LOCAL)]
                       + [v4f32_zero for _ in range(D_CHUNKS_LOCAL)])

        seq_len_k_i32 = arith.index_cast(T.i32, seq_len_k_v)
        seq_len_q_i32 = arith.index_cast(T.i32, seq_len_q_v)
        w_i32 = arith.constant(swa_window, type=T.i32)
        wave_n_off_i32 = arith.index_cast(T.i32, wave_n_offset)
        kv_start_i32 = arith.index_cast(T.i32, kv_start)
        lane_div_16_i32 = arith.index_cast(T.i32, lane_div_16)

        NUM_HEADS_idx = arith.index(NUM_HEADS)
        for qhid_constexpr_idx, h_carry, h_loop_results in scf.for_(
            arith.index(0), NUM_HEADS_idx, arith.index(1), iter_args=outer_carry,
        ):
            qhid = qhid_constexpr_idx

            m_init_args = [h_carry[i] for i in range(2 * D_CHUNKS_LOCAL)]
            for m_start, m_carry, m_loop_results in scf.for_(
                m_loop_start, m_loop_end, BM2_idx, iter_args=m_init_args,
            ):
                # R6k-1: each wave's dv/dk slice has D_CHUNKS_LOCAL chunks.
                dv_accs = [m_carry[dc] for dc in range_constexpr(D_CHUNKS_LOCAL)]
                dk_accs = [m_carry[D_CHUNKS_LOCAL + dc] for dc in range_constexpr(D_CHUNKS_LOCAL)]

                q_row_abs = m_start + lane_mod_16
                q_in_bounds = arith.cmpi(
                    arith.CmpIPredicate.slt, q_row_abs, seq_len_q_v)
                q_row_safe = arith.select(q_in_bounds, q_row_abs, arith.index(0))

                # ---- Q + DO loads ----
                if USE_LDS_FOR_Q_DO:
                    for batch in range_constexpr(NUM_BATCHES_M):
                        row_offset = batch * ROWS_PER_BATCH_LOAD
                        row_idx = m_start + load_row_in_batch + row_offset
                        in_bounds = arith.cmpi(
                            arith.CmpIPredicate.slt, row_idx, seq_len_q_v)
                        row_safe = arith.select(in_bounds, row_idx, arith.index(0))
                        g_idx_do = global_idx_q(qhid, row_safe, load_col_base)
                        g_idx_q = global_idx_q(qhid, row_safe, load_col_base)
                        vec_do = load_global_f16xN(do_ptr, g_idx_do)
                        vec_q = load_global_f16xN(q_ptr, g_idx_q)
                        vec_do_safe = arith.select(in_bounds, vec_do, c_zero_vxf16)
                        vec_q_safe = arith.select(in_bounds, vec_q, c_zero_vxf16)
                        lds_row = load_row_in_batch + row_offset
                        if M_NEEDS_GUARD:
                            row_valid = arith.cmpi(
                                arith.CmpIPredicate.ult,
                                load_row_in_batch + row_offset,
                                arith.index(BLOCK_M2))
                            _if_qd = scf.IfOp(row_valid)
                            with ir.InsertionPoint(_if_qd.then_block):
                                lds_idx_do = (lds_row * arith.index(LDS_DO_STRIDE)
                                              + load_col_base)
                                lds_idx_q = (lds_row * arith.index(LDS_Q_STRIDE)
                                             + load_col_base)
                                vector.store(vec_do_safe, lds_do, [lds_idx_do])
                                vector.store(vec_q_safe, lds_q, [lds_idx_q])
                                scf.YieldOp([])
                        else:
                            lds_idx_do = (lds_row * arith.index(LDS_DO_STRIDE)
                                          + load_col_base)
                            lds_idx_q = (lds_row * arith.index(LDS_Q_STRIDE)
                                         + load_col_base)
                            vector.store(vec_do_safe, lds_do, [lds_idx_do])
                            vector.store(vec_q_safe, lds_q, [lds_idx_q])
                    gpu.barrier()

                    # 6N-2 iter2: when D-split, q_b_packs / do_b_packs_gemm3
                    # are loaded INSIDE the per-mt scf.IfOp gates (see
                    # GEMM1 / GEMM3 below) so un-owned waves do not waste
                    # LDS read instructions. Build trivial sentinels.
                    if D_SPLIT_WAVES:
                        q_b_packs = None  # gated path: loads done in GEMM1 if-block
                        do_b_packs_gemm3 = None  # gated path: loads done in GEMM3 if-block
                    else:
                        q_b_packs = [[None] * K_STEPS_QK for _ in range(M_TILES)]
                        for mt in range_constexpr(M_TILES):
                            for ks in range_constexpr(K_STEPS_QK):
                                m_row = arith.index(mt * 16) + lane_mod_16
                                d_col = (arith.index(ks * K_STEP_QK)
                                         + lane_div_16 * MFMA_LANE_K)
                                lds_idx = m_row * arith.index(LDS_Q_STRIDE) + d_col
                                pack = vector.load_op(mfma_pack_type, lds_q, [lds_idx])
                                q_b_packs[mt][ks] = pack

                        do_b_packs_gemm3 = [[None] * K_STEPS_QK for _ in range(M_TILES)]
                        for mt in range_constexpr(M_TILES):
                            for ks in range_constexpr(K_STEPS_QK):
                                m_row = arith.index(mt * 16) + lane_mod_16
                                d_col = (arith.index(ks * K_STEP_QK)
                                         + lane_div_16 * MFMA_LANE_K)
                                lds_idx = m_row * arith.index(LDS_DO_STRIDE) + d_col
                                pack = vector.load_op(mfma_pack_type, lds_do, [lds_idx])
                                do_b_packs_gemm3[mt][ks] = pack
                else:
                    # R6b fallback: HBM-direct B-frag per m-tile. row = m_start + mt*16 + lane_mod_16.
                    q_b_packs = [[None] * K_STEPS_QK for _ in range(M_TILES)]
                    do_b_packs_gemm3 = [[None] * K_STEPS_QK for _ in range(M_TILES)]
                    for mt in range_constexpr(M_TILES):
                        row_abs = m_start + arith.index(mt * 16) + lane_mod_16
                        in_bnd = arith.cmpi(arith.CmpIPredicate.slt,
                                            row_abs, seq_len_q_v)
                        row_safe = arith.select(in_bnd, row_abs, arith.index(0))
                        for ks in range_constexpr(K_STEPS_QK):
                            col = (arith.index(ks * K_STEP_QK)
                                   + lane_div_16 * MFMA_LANE_K)
                            g_idx = global_idx_q(qhid, row_safe, col)
                            q_raw = load_global_mfma_pack(q_ptr, g_idx)
                            q_b_packs[mt][ks] = arith.select(
                                in_bnd, q_raw, c_zero_mfma_pack)
                            do_raw = load_global_mfma_pack(do_ptr, g_idx)
                            do_b_packs_gemm3[mt][ks] = arith.select(
                                in_bnd, do_raw, c_zero_mfma_pack)

                # 6N-2: de-dup GEMM1 + softmax + pT_write per mt-tile.
                # Each mt is owned by one wave (wave_id == mt % NUM_WAVES).
                # The owning wave runs everything; the other yields zeros.
                # delta_per_tile must be valid in BOTH waves (used after barrier
                # for the dsT compute), so we read it unconditionally.
                s_accs = [None for _ in range(M_TILES)]
                pT_vals_per_tile = [None for _ in range(M_TILES)]
                delta_per_tile = []
                for mt in range_constexpr(M_TILES):
                    # delta is needed by every wave (dsT compute happens after
                    # GEMM3 dp_accs[mt] is restored from LDS-broadcast / read).
                    # In de-dup mode dsT is owned by mt-owner only, so delta
                    # also only needs to be live in the owning wave. But the
                    # buffer_load is a cheap SMEM op; we leave it unconditional
                    # for simplicity. Same for lse below.
                    m_row_for_lse = m_start + arith.index(mt * 16) + lane_mod_16
                    m_row_in_bounds = arith.cmpi(
                        arith.CmpIPredicate.slt, m_row_for_lse, seq_len_q_v)
                    m_row_safe = arith.select(m_row_in_bounds, m_row_for_lse,
                                              arith.index(0))
                    lse_off_i32 = arith.index_cast(
                        T.i32, bh_base_tokens_q_of(qhid) + m_row_safe)
                    lse_v = buffer_ops.buffer_load(
                        lse_rsrc, lse_off_i32, vec_width=1, dtype=T.f32)
                    delta_v = buffer_ops.buffer_load(
                        deltas_rsrc, lse_off_i32, vec_width=1, dtype=T.f32)
                    delta_per_tile.append(delta_v)

                    if D_SPLIT_WAVES:
                        owns = owns_mt_preds[mt]
                        # ---- GATED block: GEMM1[mt] + softmax + pT LDS write ----
                        # Yields (s_acc, pT0, pT1, pT2, pT3).
                        if_g1sm = scf.IfOp(
                            owns,
                            results_=[v4f32_type, T.f32, T.f32, T.f32, T.f32],
                            has_else=True)
                        with ir.InsertionPoint(if_g1sm.then_block):
                            # GEMM1[mt]: accumulate over K_STEPS_QK.
                            # 6N-2 iter2: load Q B-frag inside the gate when LDS Q/DO.
                            acc = v4f32_zero
                            for ks in range_constexpr(K_STEPS_QK):
                                k_pack = vector.load_op(
                                    mfma_pack_type, lds_kv, [_k_idx_wave(ks)])
                                if USE_LDS_FOR_Q_DO:
                                    _m_row_b = arith.index(mt * 16) + lane_mod_16
                                    _d_col_b = (arith.index(ks * K_STEP_QK)
                                                + lane_div_16 * MFMA_LANE_K)
                                    _lds_idx_b = (_m_row_b * arith.index(LDS_Q_STRIDE)
                                                  + _d_col_b)
                                    q_pack = vector.load_op(
                                        mfma_pack_type, lds_q, [_lds_idx_b])
                                else:
                                    q_pack = q_b_packs[mt][ks]
                                acc = mfma_acc(k_pack, q_pack, acc)

                            m_row_abs_for_mask_i32 = arith.index_cast(
                                T.i32, m_row_for_lse)
                            m_oob = arith.cmpi(
                                arith.CmpIPredicate.sge,
                                m_row_abs_for_mask_i32, seq_len_q_i32)
                            pT_pieces = []
                            for ii in range_constexpr(4):
                                ii_i32 = arith.constant(ii, type=T.i32)
                                kv_row_rel_i32 = arith.AddIOp(
                                    arith.MulIOp(
                                        lane_div_16_i32,
                                        arith.constant(4, type=T.i32)).result,
                                    ii_i32).result
                                kv_abs_i32 = arith.AddIOp(
                                    arith.AddIOp(
                                        kv_start_i32, wave_n_off_i32).result,
                                    kv_row_rel_i32).result
                                kv_oob = arith.cmpi(
                                    arith.CmpIPredicate.sge,
                                    kv_abs_i32, seq_len_k_i32)
                                is_causal = arith.cmpi(
                                    arith.CmpIPredicate.sgt,
                                    kv_abs_i32, m_row_abs_for_mask_i32)
                                kv_plus_w = arith.AddIOp(
                                    kv_abs_i32, w_i32).result
                                is_swa = arith.cmpi(
                                    arith.CmpIPredicate.sle,
                                    kv_plus_w, m_row_abs_for_mask_i32)
                                bad = arith.OrIOp(
                                    arith.OrIOp(
                                        arith.OrIOp(is_causal, is_swa).result,
                                        kv_oob).result,
                                    m_oob).result
                                s_ii = vector.extract(
                                    acc, static_position=[ii],
                                    dynamic_position=[])
                                scaled = arith.MulFOp(
                                    s_ii, c_sm_scale,
                                    fastmath=fm_fast).result
                                scaled_m = arith.select(bad, c_neg_inf, scaled)
                                diff = arith.SubFOp(
                                    scaled_m, lse_v,
                                    fastmath=fm_fast).result
                                p = math_dialect.exp(diff, fastmath=fm_fast)
                                pT_pieces.append(p)
                                # Write pT to LDS for this mt.
                                kv_row_rel = (lane_div_16 * arith.index(4)
                                              + arith.index(ii))
                                kv_row = wave_n_offset + kv_row_rel
                                pt_bf16 = arith.trunc_f(elem_type, p)
                                lds_pt_idx = (
                                    kv_row * arith.index(LDS_PT_STRIDE)
                                    + arith.index(mt * 16) + lane_mod_16)
                                v1_pt = vector.from_elements(
                                    v1_elem_type, [pt_bf16])
                                vector.store(v1_pt, lds_pt, [lds_pt_idx])
                            scf.YieldOp([acc] + pT_pieces)
                        with ir.InsertionPoint(if_g1sm.else_block):
                            _zero_f32 = arith.constant(0.0, type=T.f32)
                            scf.YieldOp([v4f32_zero, _zero_f32, _zero_f32,
                                          _zero_f32, _zero_f32])
                        s_accs[mt] = if_g1sm.results[0]
                        pT_vals_per_tile[mt] = [
                            if_g1sm.results[1], if_g1sm.results[2],
                            if_g1sm.results[3], if_g1sm.results[4],
                        ]
                    else:
                        # Legacy path: every wave does GEMM1[mt] + softmax + pT_write.
                        acc = v4f32_zero
                        for ks in range_constexpr(K_STEPS_QK):
                            k_pack = vector.load_op(
                                mfma_pack_type, lds_kv, [_k_idx_wave(ks)])
                            acc = mfma_acc(k_pack, q_b_packs[mt][ks], acc)
                        s_accs[mt] = acc

                        m_row_abs_for_mask_i32 = arith.index_cast(
                            T.i32, m_row_for_lse)
                        m_oob = arith.cmpi(
                            arith.CmpIPredicate.sge,
                            m_row_abs_for_mask_i32, seq_len_q_i32)
                        pT_tile = []
                        for ii in range_constexpr(4):
                            ii_i32 = arith.constant(ii, type=T.i32)
                            kv_row_rel_i32 = arith.AddIOp(
                                arith.MulIOp(
                                    lane_div_16_i32,
                                    arith.constant(4, type=T.i32)).result,
                                ii_i32).result
                            kv_abs_i32 = arith.AddIOp(
                                arith.AddIOp(
                                    kv_start_i32, wave_n_off_i32).result,
                                kv_row_rel_i32).result
                            kv_oob = arith.cmpi(
                                arith.CmpIPredicate.sge,
                                kv_abs_i32, seq_len_k_i32)
                            is_causal = arith.cmpi(
                                arith.CmpIPredicate.sgt,
                                kv_abs_i32, m_row_abs_for_mask_i32)
                            kv_plus_w = arith.AddIOp(
                                kv_abs_i32, w_i32).result
                            is_swa = arith.cmpi(
                                arith.CmpIPredicate.sle,
                                kv_plus_w, m_row_abs_for_mask_i32)
                            bad = arith.OrIOp(
                                arith.OrIOp(
                                    arith.OrIOp(is_causal, is_swa).result,
                                    kv_oob).result,
                                m_oob).result
                            s_ii = vector.extract(
                                acc, static_position=[ii],
                                dynamic_position=[])
                            scaled = arith.MulFOp(
                                s_ii, c_sm_scale,
                                fastmath=fm_fast).result
                            scaled_m = arith.select(bad, c_neg_inf, scaled)
                            diff = arith.SubFOp(
                                scaled_m, lse_v, fastmath=fm_fast).result
                            p = math_dialect.exp(diff, fastmath=fm_fast)
                            pT_tile.append(p)
                            kv_row_rel = (lane_div_16 * arith.index(4)
                                          + arith.index(ii))
                            kv_row = wave_n_offset + kv_row_rel
                            pt_bf16 = arith.trunc_f(elem_type, p)
                            lds_pt_idx = (kv_row * arith.index(LDS_PT_STRIDE)
                                          + arith.index(mt * 16) + lane_mod_16)
                            v1_pt = vector.from_elements(v1_elem_type, [pt_bf16])
                            vector.store(v1_pt, lds_pt, [lds_pt_idx])
                        pT_vals_per_tile[mt] = pT_tile

                # R6k-1: D-split barrier. Both waves write the SAME pT
                # values (identical s_accs derived from full-D GEMM1) to
                # the SAME LDS rows (wave_n_offset=0 for both waves), but
                # the writer-lane != reader-lane mapping crosses wave
                # boundaries: lane L in wave 0 reads cells written by
                # lanes from wave 1 (and vice versa). Without a barrier
                # wave 1 could observe wave 0's stale prior-iteration pT.
                if D_SPLIT_WAVES:
                    gpu.barrier()

                # ---- pT A-frag for GEMM2 (dV += pT @ DO):
                #   A[kv_row=wave_n_offset+lane_mod_16, m_col=m_step*32+lane_div_16*8 + 0..7]
                pt_a_packs = []
                for m_step in range_constexpr(K_STEPS_PT):
                    kv_row_a = wave_n_offset + lane_mod_16
                    m_col_a = (arith.index(m_step * K_STEP_QK)
                               + lane_div_16 * MFMA_LANE_K)
                    lds_idx = (kv_row_a * arith.index(LDS_PT_STRIDE) + m_col_a)
                    pack = vector.load_op(mfma_pack_type, lds_pt, [lds_idx])
                    pt_a_packs.append(pack)

                # ---- GEMM2: dV += pT @ DO ----
                # R6b GEMM2: B-frag DO[m_step*32+lane_div_16*8+k, d_col_global+lane_mod_16].
                # R6k-1: dc_idx is wave-local; d_col_global = wave_d_col_offset + dc_idx*D_CHUNK.
                if USE_LDS_FOR_Q_DO:
                    # R6S: tr16 LDS reads via `base + constexpr byte imm`
                    # so LLVM ISel folds the per-call delta into the LDS
                    # instruction's `offset:` immediate. Replaces the
                    # per-call IntToPtrOp pattern that was spilling 32
                    # absolute addresses across AGPRs/scratch.
                    def _ds_read_tr_do_imm(byte_imm):
                        gep = _llvm.GEPOp(
                            _llvm_lds_ptr_ty(), tr_lane_base_do_ptr, [],
                            rawConstantIndices=[byte_imm],
                            elem_type=T.i8, noWrapFlags=0)
                        return rocdl.ds_read_tr16_b64(
                            v4f16_type, gep.result).result

                    def read_do_b_pack(m_step_idx, dc_idx):
                        # All offsets are Python ints (constexpr unrolled).
                        base_elem = (m_step_idx * K_STEP_QK * LDS_DO_STRIDE
                                     + dc_idx * D_CHUNK)
                        byte_imm_0 = base_elem * 2
                        byte_imm_1 = byte_imm_0 + 4 * LDS_DO_STRIDE * 2
                        v_lo = _ds_read_tr_do_imm(byte_imm_0)
                        v_hi = _ds_read_tr_do_imm(byte_imm_1)
                        v_full = vector.shuffle(
                            v_lo, v_hi, [0, 1, 2, 3, 4, 5, 6, 7])
                        return vector.bitcast(mfma_pack_type, v_full)
                else:
                    def read_do_b_pack(m_step_idx, dc_idx):
                        d_col = (wave_d_col_offset
                                 + arith.index(dc_idx * D_CHUNK)
                                 + lane_mod_16)
                        m_base = (arith.index(m_step_idx * K_STEP_QK)
                                  + lane_div_16 * MFMA_LANE_K)
                        vals = []
                        for rk in range_constexpr(MFMA_LANE_K):
                            m_row_rel = m_base + arith.index(rk)
                            m_row_abs = m_start + m_row_rel
                            in_b = arith.cmpi(
                                arith.CmpIPredicate.slt,
                                m_row_abs, seq_len_q_v)
                            m_row_safe2 = arith.select(
                                in_b, m_row_abs, arith.index(0))
                            g_idx = global_idx_q(qhid, m_row_safe2, d_col)
                            v1 = _gep_load(do_ptr, g_idx, T.vec(1, elem_type))
                            v_scalar = vector.extract(
                                v1, static_position=[0], dynamic_position=[])
                            v_safe = arith.select(in_b, v_scalar, c_zero_elem)
                            vals.append(v_safe)
                        return vector.from_elements(mfma_pack_type, vals)

                # R6k-1: dc loops wave-local (D_CHUNKS_LOCAL); reader adds
                # wave_d_col_offset internally.
                new_dv_accs = list(dv_accs)
                for dc in range_constexpr(D_CHUNKS_LOCAL):
                    for pks in range_constexpr(K_STEPS_PT):
                        b_pack = read_do_b_pack(pks, dc)
                        new_dv_accs[dc] = mfma_acc(
                            pt_a_packs[pks], b_pack, new_dv_accs[dc])

                # 6N-2: de-dup GEMM3 + dsT + dsT_write per mt-tile.
                # The owning wave for mt does GEMM3 MFMA, dsT compute, and LDS write.
                # The other wave skips entirely. GEMM4 readers see the full
                # dsT via LDS broadcast after the trailing barrier.
                if D_SPLIT_WAVES:
                    for mt in range_constexpr(M_TILES):
                        owns = owns_mt_preds[mt]
                        if_g3 = scf.IfOp(owns, results_=[], has_else=False)
                        with ir.InsertionPoint(if_g3.then_block):
                            # GEMM3[mt]
                            # 6N-2 iter2: load DO B-frag inside the gate when LDS Q/DO.
                            acc = v4f32_zero
                            for ks in range_constexpr(K_STEPS_QK):
                                v_pack = vector.load_op(
                                    mfma_pack_type, lds_kv, [_v_idx_wave(ks)])
                                if USE_LDS_FOR_Q_DO:
                                    _m_row_b = arith.index(mt * 16) + lane_mod_16
                                    _d_col_b = (arith.index(ks * K_STEP_QK)
                                                + lane_div_16 * MFMA_LANE_K)
                                    _lds_idx_b = (_m_row_b * arith.index(LDS_DO_STRIDE)
                                                  + _d_col_b)
                                    do_pack = vector.load_op(
                                        mfma_pack_type, lds_do, [_lds_idx_b])
                                else:
                                    do_pack = do_b_packs_gemm3[mt][ks]
                                acc = mfma_acc(v_pack, do_pack, acc)
                            # dsT + LDS write per ii.
                            for ii in range_constexpr(4):
                                dp_ii = vector.extract(
                                    acc, static_position=[ii],
                                    dynamic_position=[])
                                diff = arith.SubFOp(
                                    dp_ii, delta_per_tile[mt],
                                    fastmath=fm_fast).result
                                ds_ii = arith.MulFOp(
                                    pT_vals_per_tile[mt][ii], diff,
                                    fastmath=fm_fast).result
                                kv_row_rel = (lane_div_16 * arith.index(4)
                                              + arith.index(ii))
                                kv_row = wave_n_offset + kv_row_rel
                                ds_bf16 = arith.trunc_f(elem_type, ds_ii)
                                lds_pt_idx = (
                                    kv_row * arith.index(LDS_PT_STRIDE)
                                    + arith.index(mt * 16) + lane_mod_16)
                                v1_ds = vector.from_elements(
                                    v1_elem_type, [ds_bf16])
                                vector.store(v1_ds, lds_pt, [lds_pt_idx])
                            scf.YieldOp([])
                else:
                    # Legacy path
                    dp_accs = [v4f32_zero for _ in range(M_TILES)]
                    for ks in range_constexpr(K_STEPS_QK):
                        v_pack = vector.load_op(
                            mfma_pack_type, lds_kv, [_v_idx_wave(ks)])
                        for mt in range_constexpr(M_TILES):
                            dp_accs[mt] = mfma_acc(
                                v_pack, do_b_packs_gemm3[mt][ks], dp_accs[mt])
                    for mt in range_constexpr(M_TILES):
                        for ii in range_constexpr(4):
                            dp_ii = vector.extract(
                                dp_accs[mt], static_position=[ii],
                                dynamic_position=[])
                            diff = arith.SubFOp(
                                dp_ii, delta_per_tile[mt],
                                fastmath=fm_fast).result
                            ds_ii = arith.MulFOp(
                                pT_vals_per_tile[mt][ii], diff,
                                fastmath=fm_fast).result
                            kv_row_rel = (lane_div_16 * arith.index(4)
                                          + arith.index(ii))
                            kv_row = wave_n_offset + kv_row_rel
                            ds_bf16 = arith.trunc_f(elem_type, ds_ii)
                            lds_pt_idx = (kv_row * arith.index(LDS_PT_STRIDE)
                                          + arith.index(mt * 16) + lane_mod_16)
                            v1_ds = vector.from_elements(v1_elem_type, [ds_bf16])
                            vector.store(v1_ds, lds_pt, [lds_pt_idx])

                # R6k-1: D-split barrier (see GEMM2 pT barrier comment).
                if D_SPLIT_WAVES:
                    gpu.barrier()

                # ---- dsT A-frag for GEMM4 (dK += dsT @ Q):
                #   A[kv_row=wave_n_offset+lane_mod_16, m_col=m_step*32+lane_div_16*8 + 0..7]
                ds_a_packs = []
                for m_step in range_constexpr(K_STEPS_PT):
                    kv_row_a = wave_n_offset + lane_mod_16
                    m_col_a = (arith.index(m_step * K_STEP_QK)
                               + lane_div_16 * MFMA_LANE_K)
                    lds_idx = (kv_row_a * arith.index(LDS_PT_STRIDE) + m_col_a)
                    pack = vector.load_op(mfma_pack_type, lds_pt, [lds_idx])
                    ds_a_packs.append(pack)

                # ---- GEMM4: dK += dsT @ Q  (MFMA 16x16x32) ----
                # R6k-1: Q B-frag d_col = wave_d_col_offset + dc_idx*D_CHUNK + lane_mod_16.
                if USE_LDS_FOR_Q_DO:
                    # R6S: tr16 LDS reads via `base + constexpr byte imm`
                    # (same recipe as GEMM2). Uses tr_lane_base_q_ptr.
                    def _ds_read_tr_q_imm(byte_imm):
                        gep = _llvm.GEPOp(
                            _llvm_lds_ptr_ty(), tr_lane_base_q_ptr, [],
                            rawConstantIndices=[byte_imm],
                            elem_type=T.i8, noWrapFlags=0)
                        return rocdl.ds_read_tr16_b64(
                            v4f16_type, gep.result).result

                    def read_q_b_pack(m_step_idx, dc_idx):
                        base_elem = (m_step_idx * K_STEP_QK * LDS_Q_STRIDE
                                     + dc_idx * D_CHUNK)
                        byte_imm_0 = base_elem * 2
                        byte_imm_1 = byte_imm_0 + 4 * LDS_Q_STRIDE * 2
                        v_lo = _ds_read_tr_q_imm(byte_imm_0)
                        v_hi = _ds_read_tr_q_imm(byte_imm_1)
                        v_full = vector.shuffle(
                            v_lo, v_hi, [0, 1, 2, 3, 4, 5, 6, 7])
                        return vector.bitcast(mfma_pack_type, v_full)
                else:
                    def read_q_b_pack(m_step_idx, dc_idx):
                        d_col = (wave_d_col_offset
                                 + arith.index(dc_idx * D_CHUNK)
                                 + lane_mod_16)
                        m_base = (arith.index(m_step_idx * K_STEP_QK)
                                  + lane_div_16 * MFMA_LANE_K)
                        vals = []
                        for rk in range_constexpr(MFMA_LANE_K):
                            m_row_rel = m_base + arith.index(rk)
                            m_row_abs = m_start + m_row_rel
                            in_b = arith.cmpi(
                                arith.CmpIPredicate.slt,
                                m_row_abs, seq_len_q_v)
                            m_row_safe2 = arith.select(
                                in_b, m_row_abs, arith.index(0))
                            g_idx = global_idx_q(qhid, m_row_safe2, d_col)
                            v1 = _gep_load(q_ptr, g_idx, T.vec(1, elem_type))
                            v_scalar = vector.extract(
                                v1, static_position=[0], dynamic_position=[])
                            v_safe = arith.select(in_b, v_scalar, c_zero_elem)
                            vals.append(v_safe)
                        return vector.from_elements(mfma_pack_type, vals)

                # R6k-1: dc loops wave-local; reader adds wave_d_col_offset.
                new_dk_accs = list(dk_accs)
                for dc in range_constexpr(D_CHUNKS_LOCAL):
                    for pks in range_constexpr(K_STEPS_PT):
                        q_b_pack = read_q_b_pack(pks, dc)
                        new_dk_accs[dc] = mfma_acc(
                            ds_a_packs[pks], q_b_pack, new_dk_accs[dc])

                gpu.barrier()

                yield list(new_dv_accs) + list(new_dk_accs)

            yield list(m_loop_results)
        outer_carry = list(h_loop_results)

        # ---- Final: dK *= sm_scale, store ----
        # R6k-1: D-split mode -> each wave only owns D_CHUNKS_LOCAL chunks
        # and writes its own slice of dK/dV (no inter-wave reduction needed:
        # the D-axis is disjoint per wave, and the N-axis is shared so the
        # accumulators ARE complete for each wave's D-slice).
        dv_finals = [outer_carry[dc] for dc in range(D_CHUNKS_LOCAL)]
        dk_finals = [outer_carry[D_CHUNKS_LOCAL + dc] for dc in range(D_CHUNKS_LOCAL)]

        for dc in range_constexpr(D_CHUNKS_LOCAL):
            for ii in range_constexpr(4):
                kv_row_rel = (lane_div_16 * arith.index(4)
                              + arith.index(ii))
                kv_row_abs = kv_start + wave_n_offset + kv_row_rel
                d_col_abs = (wave_d_col_offset
                             + arith.index(dc * D_CHUNK)
                             + lane_mod_16)
                kv_in_bounds = arith.cmpi(
                    arith.CmpIPredicate.slt, kv_row_abs, seq_len_k_v)
                _if_kv = scf.IfOp(kv_in_bounds)
                with ir.InsertionPoint(_if_kv.then_block):
                    dv_val = vector.extract(
                        dv_finals[dc], static_position=[ii],
                        dynamic_position=[])
                    dk_val = vector.extract(
                        dk_finals[dc], static_position=[ii],
                        dynamic_position=[])
                    dk_scaled = arith.MulFOp(
                        dk_val, c_sm_scale, fastmath=fm_fast).result
                    g_idx = global_idx_kv(kv_row_abs, d_col_abs)
                    _gep_store_f32(dv_val, dv_ptr, g_idx)
                    _gep_store_f32(dk_scaled, dk_ptr, g_idx)
                    scf.YieldOp([])

    @flyc.jit
    def launch_v4_swa_bwd_dkv(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DOS: fx.Tensor,
        LSE: fx.Tensor,
        DELTAS: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        batch_size: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = arith.index_cast(T.index, batch_size)
        sk_idx = arith.index_cast(T.index, seq_len_k)
        num_n_blocks = (sk_idx + BLOCK_N - 1) // BLOCK_N
        grid_x = bs_idx * num_n_blocks

        launcher = v4_swa_bwd_dkv_kernel(
            Q, K, V, DOS, LSE, DELTAS, DK, DV, seq_len_q, seq_len_k,
        )

        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = (
                            ir.IntegerAttr.get(T.i32, _wpe))
        if flat_work_group_size is not None:
            _fwgs = int(flat_work_group_size)
            if _fwgs >= 1:
                flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
                for op in ctx.gpu_module_body.operations:
                    if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                        op.attributes["rocdl.flat_work_group_size"] = (
                            flat_wg_attr)

        passthrough_entries = []
        if daz:
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("denormal-fp-math-f32"),
                ir.StringAttr.get("preserve-sign,preserve-sign"),
            ]))
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("no-nans-fp-math"),
                ir.StringAttr.get("true"),
            ]))
            passthrough_entries.append(ir.ArrayAttr.get([
                ir.StringAttr.get("unsafe-fp-math"),
                ir.StringAttr.get("true"),
            ]))
        for op in ctx.gpu_module_body.operations:
            if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                op.attributes["passthrough"] = ir.ArrayAttr.get(
                    passthrough_entries)

        launcher.launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {
            "enable-post-misched": True,  # R6L iter5
            "lsr-drop-solution": True,
        },
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return launch_v4_swa_bwd_dkv(*args, **kwargs)

    def _compile(Q, K, V, DOS, LSE, DELTAS, DK, DV,
                 batch_size, seq_len_q, seq_len_k, stream=None):
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return flyc.compile(
                launch_v4_swa_bwd_dkv, Q, K, V, DOS, LSE, DELTAS, DK, DV,
                batch_size, seq_len_q, seq_len_k, fx.Stream(stream))

    _launch.compile = _compile

    return _launch


build_v4_swa_bwd_dkv_module_primary = build_v4_swa_bwd_dkv_module
