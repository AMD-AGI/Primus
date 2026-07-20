# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""v4_hca_bwd_dkv_pool: V4 HCA backward dK/dV POOL-stream kernel for FlyDSL.

Forked from v4_sla_bwd_dkv_kernel.py (the SLA MQA dKdV head-loop accumulator
template). Computes the POOL-stream contribution to dk/dv for an HCA
(Hybrid-Causal-Additive) attention backward and stores into the POOL slice
of dk_fp32 / dv_fp32 buffers (which are zero-initialised by the wrapper).

Key differences vs the SWA kernel:
  - KV range is fixed at [HCA_LOCAL_SEQLEN, HCA_LOCAL_SEQLEN+POOL_SIZE).
    POOL_SIZE is a build-time constexpr. For HCA shapes POOL_SIZE <= BLOCK_N
    (BLOCK_N=32 here, POOL_SIZE in {4,32}), so each program owns the entire
    pool slice for one batch.
  - No SWA-window mask, no causal mask. Two element-wise predicates only:
      (a) pool_n < POOL_SIZE      -> NEG_INF outside the pool
      (b) q_row < seq_len_q       -> NEG_INF for OOB q rows
  - qk + add_bias from ADD_MASK[Sq, POOL_SIZE]. Each element loaded as bf16/f16
    then cast to f32 inside the inner mask loop (matches dq_pool sibling).
  - LSE / DELTAS are JOINT (saved from HCA fwd) so the same q_row LSE governs
    both local and pool streams.
  - Grid: (B,) -- one program per batch, owning the entire pool slice.
  - Head loop is the SAME head-loop accumulator pattern: dynamic ``scf.for_``
    over HQ (NOT range_constexpr; constexpr-unroll at HQ=128 hangs MLIR).
  - sm_scale applied to dK once post head-loop (same as SWA dkv, P57 cr=0).
  - No atomics. Single store per (b, pool_n) slice at the end.

LDS budget at BLOCK_N=32, BLOCK_M2=32, D=512:
    K = 32 KB, V = 32 KB, DO = 32 KB, Q = 32 KB, pT = 2 KB
    Total = 130 KB <= 160 KB.
Auto-fallback: if predicted > 160 KB, drop DO/Q LDS scratches and read
Q/DO directly to register packs from HBM (mirrors SWA dkv).
"""

import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import math as math_dialect
from flydsl._mlir.dialects import memref as _memref
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.kernels_common import dtype_to_elem_type

KERNEL_NAME = "v4_hca_bwd_dkv_pool_kernel"

_LLVM_GEP_DYNAMIC = -2147483648


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def build_v4_hca_bwd_dkv_pool_module(
    num_heads,
    head_dim,
    pool_size,
    hca_local_seqlen,
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
    """Build the V4 HCA backward dK/dV POOL-stream launcher (MQA only)."""
    gpu_arch = get_hip_arch()

    if block_n is None:
        BLOCK_N = 32
    else:
        BLOCK_N = int(block_n)
    if block_m2 is None:
        BLOCK_M2 = 32
    else:
        BLOCK_M2 = int(block_m2)
    WARP_SIZE = 64
    # R6d-A: MFMA 16x16x32 -> each wave covers 16 KV rows.
    NUM_WAVES = max(1, BLOCK_N // 16)
    ROWS_PER_WAVE = BLOCK_N // NUM_WAVES
    if flat_work_group_size is None:
        flat_work_group_size = NUM_WAVES * WARP_SIZE
    BLOCK_SIZE = flat_work_group_size

    ENABLE_LDS_VEC16 = os.getenv("FLYDSL_SLA_FWD_ENABLE_LDS_VEC16", "1") == "1"
    USE_K16 = gpu_arch.startswith("gfx950")
    # R6d-A: MFMA 16x16x32 K-step = 32; A/B-frag = 8 bf16 per lane.
    assert USE_K16, "R6d-A dkv_pool requires gfx950 (MFMA 16x16x32 bf16)."
    K_STEP_QK = 32
    K_STEPS_QK = head_dim // K_STEP_QK
    # R6d-A: each MFMA 16x16x32 tile produces a 16-wide N-chunk; D_CHUNK = 16 cols.
    D_CHUNK = 16
    D_CHUNKS = head_dim // D_CHUNK
    K_STEPS_PT = BLOCK_M2 // K_STEP_QK
    # R6d-A: GEMM1/GEMM3 cover m_col [0..BLOCK_M2) with multiple 16-col MFMA-N tiles per ks.
    assert BLOCK_M2 % 16 == 0, f"BLOCK_M2 must be a multiple of 16 (MFMA-N), got {BLOCK_M2}"
    M_TILES = BLOCK_M2 // 16

    assert BLOCK_N % NUM_WAVES == 0
    assert ROWS_PER_WAVE == 16, f"ROWS_PER_WAVE must equal 16 for MFMA 16x16x32, got {ROWS_PER_WAVE}"
    assert head_dim % 32 == 0
    assert head_dim >= 64
    assert flat_work_group_size in (64, 128, 256, 512)
    assert dtype_str == "bf16", "R6d-A dkv_pool currently only supports bf16."
    assert BLOCK_N % 16 == 0
    assert BLOCK_M2 % K_STEP_QK == 0, (
        f"BLOCK_M2 ({BLOCK_M2}) must be a multiple of MFMA K-step " f"({K_STEP_QK}) for the dV/dK GEMMs."
    )
    assert mqa_kv, "v4_hca_bwd_dkv_pool currently only supports MQA (HK=1)."
    assert (
        isinstance(pool_size, int) and 0 < pool_size <= BLOCK_N
    ), f"pool_size must be int in (0, {BLOCK_N}], got {pool_size!r}"
    assert isinstance(hca_local_seqlen, int) and hca_local_seqlen >= 0
    assert (
        hca_local_seqlen % BLOCK_N == 0
    ), f"hca_local_seqlen must be multiple of BLOCK_N={BLOCK_N}, got {hca_local_seqlen}"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    POOL_SIZE = int(pool_size)
    HCA_LOCAL = int(hca_local_seqlen)

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
                f"v4_hca_bwd_dkv_pool: minimal LDS {_pred_min}B > limit "
                f"{_LDS_LIMIT_BYTES}B at D={head_dim}, BLOCK_N={BLOCK_N}, "
                f"BLOCK_M2={BLOCK_M2}."
            )
        import sys as _sys

        print(
            f"[v4_hca_bwd_dkv_pool] auto-fallback: LDS {_pred_full}B > "
            f"{_LDS_LIMIT_BYTES}B; disabling Q/DO LDS "
            f"({_pred_full} -> {_pred_min})",
            file=_sys.stderr,
            flush=True,
        )
    else:
        import sys as _sys

        print(
            f"[v4_hca_bwd_dkv_pool] LDS OK: predicted {_pred_full}B <= "
            f"{_LDS_LIMIT_BYTES}B at D={head_dim}, BLOCK_N={BLOCK_N}, "
            f"BLOCK_M2={BLOCK_M2}, USE_LDS_FOR_Q_DO=True",
            file=_sys.stderr,
            flush=True,
        )

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=(
            f"v4_hca_bwd_dkv_pool_smem_N{BLOCK_N}_M2_{BLOCK_M2}_P{POOL_SIZE}"
            f"_L{HCA_LOCAL}_MQ{int(mqa_kv)}_QDO{int(USE_LDS_FOR_Q_DO)}"
        ),
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
    def v4_hca_bwd_dkv_pool_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DOS: fx.Tensor,
        LSE: fx.Tensor,
        DELTAS: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        ADD_MASK: fx.Tensor,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32
        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        k_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), K)
        v_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), V)
        do_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DOS)
        _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DK)
        _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DV)
        add_mask_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), ADD_MASK)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        deltas_rsrc = buffer_ops.create_buffer_resource(DELTAS, max_size=True)
        # R6e: dk/dv pool-slice writes are atomic_fadd (multi-program collisions).
        dk_rsrc = buffer_ops.create_buffer_resource(DK, max_size=True)
        dv_rsrc = buffer_ops.create_buffer_resource(DV, max_size=True)

        fm_fast = arith.FastMathFlags.fast
        vxf16_type = T.vec(VEC_WIDTH, elem_type)
        v8f16_type = T.vec(8, elem_type)
        # R6d-A: MFMA 16x16x32 C-frag = 4 fp32 per lane.
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
        lds_kv = SmemPtr(base_ptr, lds_kv_offset, elem_type, shape=(LDS_KV_TOTAL_SIZE,)).get()
        lds_pt = SmemPtr(base_ptr, lds_pt_offset, elem_type, shape=(LDS_PT_ELEMS,)).get()
        if USE_LDS_FOR_Q_DO:
            lds_do = SmemPtr(base_ptr, lds_do_offset, elem_type, shape=(LDS_DO_ELEMS,)).get()
            lds_q = SmemPtr(base_ptr, lds_q_offset, elem_type, shape=(LDS_Q_ELEMS,)).get()

        block_id = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        # R6d-A MFMA 16x16x32 lane decomposition:
        #   A-frag: A[lane_mod_16, ks*32 + lane_div_16*8 + 0..7]
        #   B-frag: B[ks*32 + lane_div_16*8 + 0..7, lane_mod_16]
        #   C-frag: C[lane_div_16*4 + ii, lane_mod_16]  for ii in 0..3
        lane_mod_16 = lane % 16
        lane_div_16 = lane // 16

        wave_n_offset = wave_id * ROWS_PER_WAVE

        # R6e: grid is (B * num_m_blocks,). Decompose into (batch_idx, m_tile_idx).
        # Layout: m_tile fastest -> consecutive programs share Q/dO batch.
        BM2_idx_grid = arith.index(BLOCK_M2)
        _one_idx_grid = arith.index(1)
        num_m_blocks = (seq_len_q_v + BM2_idx_grid - _one_idx_grid) // BM2_idx_grid
        m_tile_idx = block_id % num_m_blocks
        batch_idx = block_id // num_m_blocks
        m_start = m_tile_idx * BM2_idx_grid
        # Pool slice starts at HCA_LOCAL_SEQLEN and runs for POOL_SIZE keys.
        kv_start = arith.index(HCA_LOCAL)

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        bh_base_tokens_kv = batch_idx * seq_len_k_v

        def bh_base_tokens_q_of(qhid_index):
            return (batch_idx * NUM_HEADS + qhid_index) * seq_len_q_v

        def global_idx_q(qhid_index, token_idx, col):
            return (bh_base_tokens_q_of(qhid_index) + token_idx) * arith.index(HEAD_DIM) + col

        def global_idx_kv(token_idx, col):
            return (bh_base_tokens_kv + token_idx) * arith.index(HEAD_DIM) + col

        def _gep_load(base_ptr_, elem_idx, vec_type, et=elem_type):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(
                _llvm_ptr_ty(),
                base_ptr_,
                [idx_i64],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=et,
                noWrapFlags=0,
            )
            return _llvm.LoadOp(vec_type, gep.result).result

        def _gep_store_f32(val, base_ptr_, elem_idx):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(
                _llvm_ptr_ty(),
                base_ptr_,
                [idx_i64],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=T.f32,
                noWrapFlags=0,
            )
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
        c_zero_f = arith.constant(0.0, type=compute_type)
        c_neg_inf = arith.constant(-1.0e30, type=compute_type)
        c_sm_scale = arith.constant(sm_scale, type=compute_type)
        # R6d-A: 4-elem fp32 acc per MFMA 16x16x32 op.
        v4f32_zero = arith.constant_vector(0.0, v4f32_type)

        # ---- PROLOGUE: cooperative load K, V into LDS ----
        pool_end = kv_start + arith.index(POOL_SIZE)

        def coop_load_k():
            k_base = arith.index(0)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = kv_start + load_row_in_batch + row_offset
                in_bounds = arith.cmpi(arith.CmpIPredicate.slt, row_idx, pool_end)
                row_safe = arith.select(in_bounds, row_idx, arith.index(0))
                g_idx = global_idx_kv(row_safe, load_col_base)
                vec = load_global_f16xN(k_ptr, g_idx)
                vec_safe = arith.select(in_bounds, vec, c_zero_vxf16)
                lds_row = load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult, load_row_in_batch + row_offset, arith.index(BLOCK_N)
                    )
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

        def coop_load_v():
            v_base = arith.index(LDS_V_BASE)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = kv_start + load_row_in_batch + row_offset
                in_bounds = arith.cmpi(arith.CmpIPredicate.slt, row_idx, pool_end)
                row_safe = arith.select(in_bounds, row_idx, arith.index(0))
                g_idx = global_idx_kv(row_safe, load_col_base)
                vec = load_global_f16xN(v_ptr, g_idx)
                vec_safe = arith.select(in_bounds, vec, c_zero_vxf16)
                lds_row = load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult, load_row_in_batch + row_offset, arith.index(BLOCK_N)
                    )
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

        coop_load_k()
        coop_load_v()
        gpu.barrier()

        # R6d-A: K/V LDS A-frag uses lane_mod_16 (16-row MFMA) and lane_div_16*8 for col.
        # Per-call swizzle mask (folds wave_n_offset bits at D=512).
        def _k_idx_wave(ks):
            kv_row = wave_n_offset + lane_mod_16
            col = arith.index(ks * K_STEP_QK) + lane_div_16 * MFMA_LANE_K
            mask = (kv_row & arith.index(K_SWZ_ROW_MASK)) << arith.index(4)
            return kv_row * arith.index(K_STRIDE) + (col ^ mask)

        def _v_idx_wave(ks):
            kv_row = wave_n_offset + lane_mod_16
            col = arith.index(ks * K_STEP_QK) + lane_div_16 * MFMA_LANE_K
            mask = (kv_row & arith.index(K_SWZ_ROW_MASK)) << arith.index(4)
            return arith.index(LDS_V_BASE) + kv_row * arith.index(V_STRIDE) + (col ^ mask)

        # R6e: m-loop replaced by per-program m-tile.
        arith.index(BLOCK_N)
        arith.index(BLOCK_M2)
        arith.index(1)

        outer_carry = [v4f32_zero for _ in range(D_CHUNKS)] + [v4f32_zero for _ in range(D_CHUNKS)]

        seq_len_q_i32 = arith.index_cast(T.i32, seq_len_q_v)
        wave_n_off_i32 = arith.index_cast(T.i32, wave_n_offset)
        lane_div_16_i32 = arith.index_cast(T.i32, lane_div_16)
        pool_size_i32 = arith.constant(POOL_SIZE, type=T.i32)

        # ADD_MASK row stride in elements (= POOL_SIZE; bf16/f16 contiguous).
        ADD_MASK_STRIDE_M = arith.index(POOL_SIZE)

        NUM_HEADS_idx = arith.index(NUM_HEADS)
        for qhid_constexpr_idx, h_carry, h_loop_results in scf.for_(
            arith.index(0),
            NUM_HEADS_idx,
            arith.index(1),
            iter_args=outer_carry,
        ):
            qhid = qhid_constexpr_idx

            # R6e: per-program m-tile (no inner m-loop). Accumulators are head-loop carry.
            dv_accs = [h_carry[dc] for dc in range_constexpr(D_CHUNKS)]
            dk_accs = [h_carry[D_CHUNKS + dc] for dc in range_constexpr(D_CHUNKS)]

            # R6d-A: lane_mod_16 is the per-tile q-row coord; full per-tile loop builds row below.
            q_row_abs = m_start + lane_mod_16
            q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, q_row_abs, seq_len_q_v)
            arith.select(q_in_bounds, q_row_abs, arith.index(0))

            if USE_LDS_FOR_Q_DO:
                for batch in range_constexpr(NUM_BATCHES_M):
                    row_offset = batch * ROWS_PER_BATCH_LOAD
                    row_idx = m_start + load_row_in_batch + row_offset
                    in_bounds = arith.cmpi(arith.CmpIPredicate.slt, row_idx, seq_len_q_v)
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
                            arith.CmpIPredicate.ult, load_row_in_batch + row_offset, arith.index(BLOCK_M2)
                        )
                        _if_qd = scf.IfOp(row_valid)
                        with ir.InsertionPoint(_if_qd.then_block):
                            lds_idx_do = lds_row * arith.index(LDS_DO_STRIDE) + load_col_base
                            lds_idx_q = lds_row * arith.index(LDS_Q_STRIDE) + load_col_base
                            vector.store(vec_do_safe, lds_do, [lds_idx_do])
                            vector.store(vec_q_safe, lds_q, [lds_idx_q])
                            scf.YieldOp([])
                    else:
                        lds_idx_do = lds_row * arith.index(LDS_DO_STRIDE) + load_col_base
                        lds_idx_q = lds_row * arith.index(LDS_Q_STRIDE) + load_col_base
                        vector.store(vec_do_safe, lds_do, [lds_idx_do])
                        vector.store(vec_q_safe, lds_q, [lds_idx_q])
                gpu.barrier()

                # R6d-A: B-frag for GEMM1 (qkT = K @ Q^T) per m-tile mt in [0, M_TILES).
                #   Q[mt*16 + lane_mod_16, ks*32 + lane_div_16*8 + 0..7]
                q_b_packs = [[None] * K_STEPS_QK for _ in range(M_TILES)]
                for mt in range_constexpr(M_TILES):
                    for ks in range_constexpr(K_STEPS_QK):
                        m_row = arith.index(mt * 16) + lane_mod_16
                        d_col = arith.index(ks * K_STEP_QK) + lane_div_16 * MFMA_LANE_K
                        lds_idx = m_row * arith.index(LDS_Q_STRIDE) + d_col
                        pack = vector.load_op(mfma_pack_type, lds_q, [lds_idx])
                        q_b_packs[mt][ks] = pack

                # R6d-A: B-frag for GEMM3 (dp = V @ DO^T) per m-tile mt.
                do_b_packs_gemm3 = [[None] * K_STEPS_QK for _ in range(M_TILES)]
                for mt in range_constexpr(M_TILES):
                    for ks in range_constexpr(K_STEPS_QK):
                        m_row = arith.index(mt * 16) + lane_mod_16
                        d_col = arith.index(ks * K_STEP_QK) + lane_div_16 * MFMA_LANE_K
                        lds_idx = m_row * arith.index(LDS_DO_STRIDE) + d_col
                        pack = vector.load_op(mfma_pack_type, lds_do, [lds_idx])
                        do_b_packs_gemm3[mt][ks] = pack
            else:
                # R6d-A fallback: HBM-direct B-frag per m-tile. row = m_start + mt*16 + lane_mod_16.
                q_b_packs = [[None] * K_STEPS_QK for _ in range(M_TILES)]
                do_b_packs_gemm3 = [[None] * K_STEPS_QK for _ in range(M_TILES)]
                for mt in range_constexpr(M_TILES):
                    row_abs = m_start + arith.index(mt * 16) + lane_mod_16
                    in_bnd = arith.cmpi(arith.CmpIPredicate.slt, row_abs, seq_len_q_v)
                    row_safe = arith.select(in_bnd, row_abs, arith.index(0))
                    for ks in range_constexpr(K_STEPS_QK):
                        col = arith.index(ks * K_STEP_QK) + lane_div_16 * MFMA_LANE_K
                        g_idx = global_idx_q(qhid, row_safe, col)
                        q_raw = load_global_mfma_pack(q_ptr, g_idx)
                        q_b_packs[mt][ks] = arith.select(in_bnd, q_raw, c_zero_mfma_pack)
                        do_raw = load_global_mfma_pack(do_ptr, g_idx)
                        do_b_packs_gemm3[mt][ks] = arith.select(in_bnd, do_raw, c_zero_mfma_pack)

            # ---- GEMM1: qkT = K @ Q^T (MFMA 16x16x32, one tile per m-tile mt) ----
            s_accs = [v4f32_zero for _ in range(M_TILES)]
            for ks in range_constexpr(K_STEPS_QK):
                k_pack = vector.load_op(mfma_pack_type, lds_kv, [_k_idx_wave(ks)])
                for mt in range_constexpr(M_TILES):
                    s_accs[mt] = mfma_acc(k_pack, q_b_packs[mt][ks], s_accs[mt])

            # R6d-A: per m-tile lse/delta, pool-only mask, ADD_MASK bias, softmax.
            # m_row for tile mt = m_start + mt*16 + lane_mod_16.
            # Resulting pT_vals_per_tile[mt][ii] for ii in 0..3.
            pT_vals_per_tile = []
            lse_per_tile = []
            delta_per_tile = []
            for mt in range_constexpr(M_TILES):
                m_row_for_lse = m_start + arith.index(mt * 16) + lane_mod_16
                m_row_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, m_row_for_lse, seq_len_q_v)
                m_row_safe = arith.select(m_row_in_bounds, m_row_for_lse, arith.index(0))
                lse_off_i32 = arith.index_cast(T.i32, bh_base_tokens_q_of(qhid) + m_row_safe)
                lse_v = buffer_ops.buffer_load(lse_rsrc, lse_off_i32, vec_width=1, dtype=T.f32)
                delta_v = buffer_ops.buffer_load(deltas_rsrc, lse_off_i32, vec_width=1, dtype=T.f32)
                lse_per_tile.append(lse_v)
                delta_per_tile.append(delta_v)

                m_row_abs_for_mask_i32 = arith.index_cast(T.i32, m_row_for_lse)
                m_oob = arith.cmpi(arith.CmpIPredicate.sge, m_row_abs_for_mask_i32, seq_len_q_i32)
                # ADD_MASK row base (clamp to row 0 if OOB; lane is masked).
                add_mask_row_base = m_row_safe * ADD_MASK_STRIDE_M

                pT_tile = []
                for ii in range_constexpr(4):
                    ii_i32 = arith.constant(ii, type=T.i32)
                    # R6d-A: 16x16x32 C-frag lane stride is 4 (NOT 8 like 32x32x16).
                    pool_n_rel_i32 = arith.AddIOp(
                        arith.MulIOp(lane_div_16_i32, arith.constant(4, type=T.i32)).result, ii_i32
                    ).result
                    pool_n_i32 = arith.AddIOp(wave_n_off_i32, pool_n_rel_i32).result
                    pool_n_oob = arith.cmpi(arith.CmpIPredicate.sge, pool_n_i32, pool_size_i32)
                    bad = arith.OrIOp(pool_n_oob, m_oob).result

                    # Load add_bias from ADD_MASK[m_row, pool_n].
                    pool_n_safe_i32 = arith.select(bad, arith.constant(0, type=T.i32), pool_n_i32)
                    pool_n_idx = arith.index_cast(T.index, pool_n_safe_i32)
                    add_elem_idx = add_mask_row_base + pool_n_idx
                    bias_raw_v1 = _gep_load(add_mask_ptr, add_elem_idx, T.vec(1, elem_type))
                    bias_raw = vector.extract(bias_raw_v1, static_position=[0], dynamic_position=[])
                    bias_f32 = arith.extf(compute_type, bias_raw)
                    bias_safe = arith.select(bad, c_zero_f, bias_f32)

                    s_ii = vector.extract(s_accs[mt], static_position=[ii], dynamic_position=[])
                    scaled = arith.MulFOp(s_ii, c_sm_scale, fastmath=fm_fast).result
                    scaled_plus_bias = arith.AddFOp(scaled, bias_safe, fastmath=fm_fast).result
                    scaled_m = arith.select(bad, c_neg_inf, scaled_plus_bias)
                    diff = arith.SubFOp(scaled_m, lse_v, fastmath=fm_fast).result
                    p = math_dialect.exp(diff, fastmath=fm_fast)
                    pT_tile.append(p)
                pT_vals_per_tile.append(pT_tile)

            # ---- pT register -> LDS (per m-tile mt, write 4 C-frag elems/lane @ col mt*16 + lane_mod_16) ----
            for mt in range_constexpr(M_TILES):
                for ii in range_constexpr(4):
                    kv_row_rel = lane_div_16 * arith.index(4) + arith.index(ii)
                    kv_row = wave_n_offset + kv_row_rel
                    pt_bf16 = arith.trunc_f(elem_type, pT_vals_per_tile[mt][ii])
                    lds_pt_idx = kv_row * arith.index(LDS_PT_STRIDE) + arith.index(mt * 16) + lane_mod_16
                    v1 = vector.from_elements(v1_elem_type, [pt_bf16])
                    vector.store(v1, lds_pt, [lds_pt_idx])

            # ---- pT A-frag for GEMM2 (dV += pT @ DO):
            #   A[kv_row=wave_n_offset+lane_mod_16, m_col=m_step*32+lane_div_16*8 + 0..7]
            pt_a_packs = []
            for m_step in range_constexpr(K_STEPS_PT):
                kv_row_a = wave_n_offset + lane_mod_16
                m_col_a = arith.index(m_step * K_STEP_QK) + lane_div_16 * MFMA_LANE_K
                lds_idx = kv_row_a * arith.index(LDS_PT_STRIDE) + m_col_a
                pack = vector.load_op(mfma_pack_type, lds_pt, [lds_idx])
                pt_a_packs.append(pack)

            # ---- GEMM2: dV += pT @ DO ----
            # R6d-A GEMM2 (dV += pT @ DO): B-frag DO[m_step*32+lane_div_16*8+k, dc*16+lane_mod_16].
            if USE_LDS_FOR_Q_DO:

                def read_do_b_pack(m_step_idx, dc_idx):
                    d_col = arith.index(dc_idx * D_CHUNK) + lane_mod_16
                    m_base = arith.index(m_step_idx * K_STEP_QK) + lane_div_16 * MFMA_LANE_K
                    vals = []
                    for rk in range_constexpr(MFMA_LANE_K):
                        m_row = m_base + arith.index(rk)
                        lds_idx = m_row * arith.index(LDS_DO_STRIDE) + d_col
                        val = _memref.load(lds_do, [lds_idx])
                        vals.append(val)
                    return vector.from_elements(mfma_pack_type, vals)

            else:

                def read_do_b_pack(m_step_idx, dc_idx):
                    d_col = arith.index(dc_idx * D_CHUNK) + lane_mod_16
                    m_base = arith.index(m_step_idx * K_STEP_QK) + lane_div_16 * MFMA_LANE_K
                    vals = []
                    for rk in range_constexpr(MFMA_LANE_K):
                        m_row_rel = m_base + arith.index(rk)
                        m_row_abs = m_start + m_row_rel
                        in_b = arith.cmpi(arith.CmpIPredicate.slt, m_row_abs, seq_len_q_v)
                        m_row_safe2 = arith.select(in_b, m_row_abs, arith.index(0))
                        g_idx = global_idx_q(qhid, m_row_safe2, d_col)
                        v1 = _gep_load(do_ptr, g_idx, T.vec(1, elem_type))
                        v_scalar = vector.extract(v1, static_position=[0], dynamic_position=[])
                        v_safe = arith.select(in_b, v_scalar, c_zero_elem)
                        vals.append(v_safe)
                    return vector.from_elements(mfma_pack_type, vals)

            new_dv_accs = list(dv_accs)
            for dc in range_constexpr(D_CHUNKS):
                for pks in range_constexpr(K_STEPS_PT):
                    b_pack = read_do_b_pack(pks, dc)
                    new_dv_accs[dc] = mfma_acc(pt_a_packs[pks], b_pack, new_dv_accs[dc])

            # ---- GEMM3: dp = V @ DO^T (MFMA 16x16x32, one tile per m-tile mt) ----
            dp_accs = [v4f32_zero for _ in range(M_TILES)]
            for ks in range_constexpr(K_STEPS_QK):
                v_pack = vector.load_op(mfma_pack_type, lds_kv, [_v_idx_wave(ks)])
                for mt in range_constexpr(M_TILES):
                    dp_accs[mt] = mfma_acc(v_pack, do_b_packs_gemm3[mt][ks], dp_accs[mt])

            # ---- dsT = pT * (dp - delta) per m-tile ----
            dsT_vals_per_tile = []
            for mt in range_constexpr(M_TILES):
                ds_tile = []
                for ii in range_constexpr(4):
                    dp_ii = vector.extract(dp_accs[mt], static_position=[ii], dynamic_position=[])
                    diff = arith.SubFOp(dp_ii, delta_per_tile[mt], fastmath=fm_fast).result
                    ds_ii = arith.MulFOp(pT_vals_per_tile[mt][ii], diff, fastmath=fm_fast).result
                    ds_tile.append(ds_ii)
                dsT_vals_per_tile.append(ds_tile)

            # ---- dsT register -> LDS (per m-tile, col = mt*16 + lane_mod_16) ----
            for mt in range_constexpr(M_TILES):
                for ii in range_constexpr(4):
                    kv_row_rel = lane_div_16 * arith.index(4) + arith.index(ii)
                    kv_row = wave_n_offset + kv_row_rel
                    ds_bf16 = arith.trunc_f(elem_type, dsT_vals_per_tile[mt][ii])
                    lds_pt_idx = kv_row * arith.index(LDS_PT_STRIDE) + arith.index(mt * 16) + lane_mod_16
                    v1_ds = vector.from_elements(v1_elem_type, [ds_bf16])
                    vector.store(v1_ds, lds_pt, [lds_pt_idx])

            # ---- dsT A-frag for GEMM4 (dK += dsT @ Q):
            #   A[kv_row=wave_n_offset+lane_mod_16, m_col=m_step*32+lane_div_16*8 + 0..7]
            ds_a_packs = []
            for m_step in range_constexpr(K_STEPS_PT):
                kv_row_a = wave_n_offset + lane_mod_16
                m_col_a = arith.index(m_step * K_STEP_QK) + lane_div_16 * MFMA_LANE_K
                lds_idx = kv_row_a * arith.index(LDS_PT_STRIDE) + m_col_a
                pack = vector.load_op(mfma_pack_type, lds_pt, [lds_idx])
                ds_a_packs.append(pack)

            # ---- GEMM4: dK += dsT @ Q  (MFMA 16x16x32) ----
            # B-frag: Q[m_step*32+lane_div_16*8+k, dc*16+lane_mod_16].
            if USE_LDS_FOR_Q_DO:

                def read_q_b_pack(m_step_idx, dc_idx):
                    d_col = arith.index(dc_idx * D_CHUNK) + lane_mod_16
                    m_base = arith.index(m_step_idx * K_STEP_QK) + lane_div_16 * MFMA_LANE_K
                    vals = []
                    for rk in range_constexpr(MFMA_LANE_K):
                        m_row = m_base + arith.index(rk)
                        lds_idx = m_row * arith.index(LDS_Q_STRIDE) + d_col
                        val = _memref.load(lds_q, [lds_idx])
                        vals.append(val)
                    return vector.from_elements(mfma_pack_type, vals)

            else:

                def read_q_b_pack(m_step_idx, dc_idx):
                    d_col = arith.index(dc_idx * D_CHUNK) + lane_mod_16
                    m_base = arith.index(m_step_idx * K_STEP_QK) + lane_div_16 * MFMA_LANE_K
                    vals = []
                    for rk in range_constexpr(MFMA_LANE_K):
                        m_row_rel = m_base + arith.index(rk)
                        m_row_abs = m_start + m_row_rel
                        in_b = arith.cmpi(arith.CmpIPredicate.slt, m_row_abs, seq_len_q_v)
                        m_row_safe2 = arith.select(in_b, m_row_abs, arith.index(0))
                        g_idx = global_idx_q(qhid, m_row_safe2, d_col)
                        v1 = _gep_load(q_ptr, g_idx, T.vec(1, elem_type))
                        v_scalar = vector.extract(v1, static_position=[0], dynamic_position=[])
                        v_safe = arith.select(in_b, v_scalar, c_zero_elem)
                        vals.append(v_safe)
                    return vector.from_elements(mfma_pack_type, vals)

            new_dk_accs = list(dk_accs)
            for dc in range_constexpr(D_CHUNKS):
                for pks in range_constexpr(K_STEPS_PT):
                    q_b_pack = read_q_b_pack(pks, dc)
                    new_dk_accs[dc] = mfma_acc(ds_a_packs[pks], q_b_pack, new_dk_accs[dc])

            gpu.barrier()

            # R6e: yield (dv, dk) partial directly as head-loop carry; no inner m-loop.
            yield list(new_dv_accs) + list(new_dk_accs)
        outer_carry = list(h_loop_results)

        # ---- Final: dK *= sm_scale, store ----
        dv_finals = [outer_carry[dc] for dc in range(D_CHUNKS)]
        dk_finals = [outer_carry[D_CHUNKS + dc] for dc in range(D_CHUNKS)]

        # R6e: atomic_fadd into shared pool slice of dk/dv.
        # Multiple m-tile programs collide on the same (b, pool_n, d) addresses.
        # DK/DV are fp32 contiguous; byte_offset = global_idx_kv * 4.
        _atom_zero_i32 = arith.constant(0, type=T.i32)
        _four_i32 = arith.constant(4, type=T.i32)
        for dc in range_constexpr(D_CHUNKS):
            for ii in range_constexpr(4):
                kv_row_rel = lane_div_16 * arith.index(4) + arith.index(ii)
                kv_row_abs = kv_start + wave_n_offset + kv_row_rel
                d_col_abs = arith.index(dc * D_CHUNK) + lane_mod_16
                kv_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, kv_row_abs, pool_end)
                _if_kv = scf.IfOp(kv_in_bounds)
                with ir.InsertionPoint(_if_kv.then_block):
                    dv_val = vector.extract(dv_finals[dc], static_position=[ii], dynamic_position=[])
                    dk_val = vector.extract(dk_finals[dc], static_position=[ii], dynamic_position=[])
                    dk_scaled = arith.MulFOp(dk_val, c_sm_scale, fastmath=fm_fast).result
                    g_elem_idx = global_idx_kv(kv_row_abs, d_col_abs)
                    g_elem_i32 = arith.index_cast(T.i32, g_elem_idx)
                    byte_off_i32 = arith.MulIOp(g_elem_i32, _four_i32).result
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        dv_val,
                        dv_rsrc,
                        byte_off_i32,
                        _atom_zero_i32,
                        _atom_zero_i32,
                    )
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        dk_scaled,
                        dk_rsrc,
                        byte_off_i32,
                        _atom_zero_i32,
                        _atom_zero_i32,
                    )
                    scf.YieldOp([])

    @flyc.jit
    def launch_v4_hca_bwd_dkv_pool(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DOS: fx.Tensor,
        LSE: fx.Tensor,
        DELTAS: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        ADD_MASK: fx.Tensor,
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
        # R6e: grid_x = B * num_m_blocks. Parallelize over m-tiles for B=1.
        sq_idx_h = arith.index_cast(T.index, seq_len_q)
        BM2_idx_h = arith.index(BLOCK_M2)
        _one_idx_h = arith.index(1)
        num_m_blocks_v = (sq_idx_h + BM2_idx_h - _one_idx_h) // BM2_idx_h
        grid_x = bs_idx * num_m_blocks_v

        launcher = v4_hca_bwd_dkv_pool_kernel(
            Q,
            K,
            V,
            DOS,
            LSE,
            DELTAS,
            DK,
            DV,
            ADD_MASK,
            seq_len_q,
            seq_len_k,
        )

        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, _wpe)
        if flat_work_group_size is not None:
            _fwgs = int(flat_work_group_size)
            if _fwgs >= 1:
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
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("no-nans-fp-math"),
                        ir.StringAttr.get("true"),
                    ]
                )
            )
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("unsafe-fp-math"),
                        ir.StringAttr.get("true"),
                    ]
                )
            )
        for op in ctx.gpu_module_body.operations:
            if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return launch_v4_hca_bwd_dkv_pool(*args, **kwargs)

    def _compile(Q, K, V, DOS, LSE, DELTAS, DK, DV, ADD_MASK, batch_size, seq_len_q, seq_len_k, stream=None):
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return flyc.compile(
                launch_v4_hca_bwd_dkv_pool,
                Q,
                K,
                V,
                DOS,
                LSE,
                DELTAS,
                DK,
                DV,
                ADD_MASK,
                batch_size,
                seq_len_q,
                seq_len_k,
                fx.Stream(stream),
            )

    _launch.compile = _compile

    return _launch


# Convenience alias.
build_v4_hca_bwd_dkv_pool_module_primary = build_v4_hca_bwd_dkv_pool_module
