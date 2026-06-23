# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""v4_hca_bwd_dq_pool: V4 HCA backward dQ POOL-stream kernel for FlyDSL.

Forked from v4_sla_bwd_dq_kernel.py. Computes the POOL stream contribution
to dq for an HCA (Hybrid-Causal-Additive) attention backward, then
ACCUMULATES into an existing dq_fp32 buffer (which already contains the
LOCAL stream dq from the SWA dq kernel).

Differences vs the SWA kernel:
  - KV range is fixed at [HCA_LOCAL_SEQLEN, HCA_LOCAL_SEQLEN+POOL_SIZE).
    POOL_SIZE is a build-time constexpr. For our shapes POOL_SIZE <= BLOCK_N
    so there is exactly ONE n-block iteration.
  - No SWA-window mask, no causal mask. Two element-wise predicates only:
      (a) pool_n < POOL_SIZE     -> NEG_INF outside the pool
      (b) q_row < seq_len_q      -> NEG_INF for OOB q rows
  - qk + add_bias from the ADD_MASK tensor (shape [Sq, POOL_SIZE]).
    Each element loaded as f32 (after cast from bf16/f16) and added to qk
    before -lse and exp.
  - No SINK / DSINK. Sink is handled by the LOCAL FlyDSL dq kernel; the
    pool stream does not touch the sink.
  - Final store ACCUMULATES into DQ (load + add + store). Race-free
    because each program owns a unique (b, qhid, m_block) slice and the
    pool kernel runs AFTER the SWA dq has finished writing dq for that
    same slice (sequential launches from the wrapper).
  - sm_scale is applied INSIDE the loop on qk (matches Triton ref); after
    the n-loop (which has only one iter), sm_scale is applied ONCE MORE
    on dq before accumulating into DQ (P57 cr=0 BWD).

Layout: BHLD. Q/DOUT/DQ flat from (B, HQ, Sq, D). K/V flat from
        (B, HK, Sk, D); MQA means HK=1, no head stride applied to KV.
LSE/DELTAS: (B, HQ, Sq) flat, fp32, raw-domain (JOINT lse from fwd).
ADD_MASK:   (Sq, POOL_SIZE) bf16/f16; loaded as f32 inside the kernel.
DQ:         (B, HQ, Sq, D) fp32; ACCUMULATED.
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

# ---- Module-level constants ----

KERNEL_NAME = "v4_hca_bwd_dq_pool_kernel"

_LOG2E = math.log2(math.e)  # 1.4426950408889634

_LLVM_GEP_DYNAMIC = -2147483648  # LLVM kDynamicIndex sentinel (0x80000000 as signed i32)


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (n & _VMCNT_LO_MASK) | _LGKMCNT_EXPCNT_BASE | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    rocdl.s_waitcnt(val)


def build_v4_hca_bwd_dq_pool_module(
    num_heads,
    head_dim,
    pool_size,
    hca_local_seqlen,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    flat_work_group_size=None,
    block_m=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    layout_bhld=True,
    mqa_kv=True,
):
    """Build the V4 HCA backward dQ POOL-stream launcher.

    Args:
        num_heads: HQ (number of Q heads).
        head_dim: D (head dimension), must be % 32 == 0 and >= 64.
        pool_size: int > 0; number of pool keys (<= BLOCK_N=64 in this
            kernel; one n-block per program iter). Build-time constexpr.
        hca_local_seqlen: int >= 0; offset into K/V at which the pool
            keys start (== Sq for HCA split-mask). Build-time constexpr;
            must be a multiple of BLOCK_N to keep the single-iter
            n-block aligned.
        sm_scale: defaults to 1/sqrt(head_dim).
        mqa_kv: if True, K/V indexing drops head_idx (HK=1, stride_h=0).
        layout_bhld: BHLD layout (True) or BLHD (False); V4 uses BHLD.

    Returns: launch(Q, K, V, DOUT, LSE, DELTAS, DQ_FP32, ADD_MASK,
                    batch_size, seq_len_q, seq_len_k, stream=None).
            DQ_FP32 is ACCUMULATED into (load + add + store).
    """
    gpu_arch = get_hip_arch()

    BLOCK_N = 64
    K_SUB_N = 32
    WARP_SIZE = 64

    if block_m is not None:
        BLOCK_M = block_m
    else:
        BLOCK_M = 128

    if flat_work_group_size is None:
        if BLOCK_M <= 128:
            flat_work_group_size = 256
        else:
            flat_work_group_size = 512
    NUM_WAVES = flat_work_group_size // WARP_SIZE
    BLOCK_SIZE = flat_work_group_size
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES
    # V4 SWA: dense path. One outer iter = one BLOCK_N block.
    BLOCK_N_OUT = BLOCK_N
    N_SUBTILES = 1
    ENABLE_PREFETCH_3BUF = (
        os.getenv("FLYDSL_SLA_FWD_ENABLE_PREFETCH3", "0") == "1"
    )
    _has_lds_load_b128 = not gpu_arch.startswith("gfx942")
    ENABLE_DMA = _has_lds_load_b128 and (
        os.getenv("FLYDSL_SLA_FWD_ENABLE_DMA", "1") == "1"
    )
    ENABLE_LDS_VEC16 = (
        os.getenv("FLYDSL_SLA_FWD_ENABLE_LDS_VEC16", "1") == "1"
    )
    REDUCE_MODE = os.getenv("FLYDSL_SLA_FWD_REDUCE_MODE", "xor").strip().lower()
    if REDUCE_MODE not in ("xor", "ds_bpermute"):
        REDUCE_MODE = "xor"
    NUM_PREFETCH_K = 3 if ENABLE_PREFETCH_3BUF else (2 if ENABLE_DMA else 1)
    NUM_PREFETCH_V = 3 if ENABLE_PREFETCH_3BUF else (2 if ENABLE_DMA else 1)
    CK_LDS_SEQ = (1, 2, 0, 1, 0, 1, 2, 0) if ENABLE_PREFETCH_3BUF else (0,)

    USE_HW_TR = gpu_arch.startswith("gfx950")
    USE_K16 = gpu_arch.startswith("gfx950")

    # Auto-fallback: gfx950 LDS limit is 160 KB. K+V double-buffering at D=512
    # easily exceeds that. If predicted LDS exceeds budget, force DMA off
    # (NUM_PREFETCH = 1) and warn. Keeps the kernel correct regardless of
    # the env knob.
    _LDS_LIMIT_BYTES = 160 * 1024
    def _predicted_lds_bytes(nk, nv, dma):
        # USE_HW_TR & DMA  -> V_STRIDE = head_dim
        # USE_HW_TR & !DMA -> V_STRIDE = head_dim + 4
        # !USE_HW_TR       -> V stored transposed: LDS V = head_dim * (BLOCK_N + 2)
        if USE_HW_TR:
            v_str = head_dim if dma else head_dim + 4
            return (nk * BLOCK_N * head_dim + nv * BLOCK_N * v_str) * 2
        vt = BLOCK_N + 2
        return (nk * BLOCK_N * head_dim + nv * head_dim * vt) * 2
    _pred = _predicted_lds_bytes(NUM_PREFETCH_K, NUM_PREFETCH_V, ENABLE_DMA)
    if _pred > _LDS_LIMIT_BYTES and ENABLE_DMA:
        # Try DMA off (single-buffered).
        ENABLE_DMA = False
        NUM_PREFETCH_K = 1
        NUM_PREFETCH_V = 1
        _pred2 = _predicted_lds_bytes(1, 1, False)
        if _pred2 > _LDS_LIMIT_BYTES:
            raise RuntimeError(
                f"v4_hca_bwd_dq_pool: predicted LDS {_pred2}B > limit {_LDS_LIMIT_BYTES}B "
                f"even single-buffered at D={head_dim}, BLOCK_N={BLOCK_N}"
            )
        import sys as _sys
        print(
            f"[v4_hca_bwd_dq_pool] LDS overflow at D={head_dim}, BLOCK_N={BLOCK_N}: "
            f"auto-disabled DMA (predicted {_pred} -> {_pred2} bytes)",
            file=_sys.stderr, flush=True,
        )
    K_STEP_QK = 16 if USE_K16 else 8
    K_STEPS_QK = head_dim // K_STEP_QK
    D_CHUNK = 32
    D_CHUNKS = head_dim // D_CHUNK
    PV_K_STEP = 16 if USE_K16 else 8
    PV_K_STEPS = K_SUB_N // PV_K_STEP  # 2 steps per sub-tile (K=16) or 4 (K=8)

    assert BLOCK_M % NUM_WAVES == 0
    assert head_dim % 32 == 0, f"head_dim ({head_dim}) must be divisible by 32"
    assert head_dim >= 64, f"head_dim ({head_dim}) must be >= 64"
    assert flat_work_group_size in (128, 256, 512), (
        f"flat_work_group_size must be 128, 256, or 512, got {flat_work_group_size}"
    )
    assert dtype_str in ("f16", "bf16"), "v4_hca_bwd_dq_pool only supports f16 and bf16"
    assert BLOCK_N % 32 == 0
    assert BLOCK_N_OUT == BLOCK_N
    assert isinstance(pool_size, int) and 0 < pool_size <= BLOCK_N, (
        f"pool_size must be int in (0, {BLOCK_N}], got {pool_size!r}")
    assert isinstance(hca_local_seqlen, int) and hca_local_seqlen >= 0, (
        f"hca_local_seqlen must be int >= 0, got {hca_local_seqlen!r}")
    assert hca_local_seqlen % BLOCK_N == 0, (
        f"hca_local_seqlen must be multiple of BLOCK_N={BLOCK_N}, got {hca_local_seqlen}")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    K_STRIDE = HEAD_DIM
    # XOR swizzle mask must fit within the row stride. The swizzle is
    # applied at 16-element granularity, so the maximum mask is
    # `(K_STRIDE // 16 - 1) << 4`. For D=128 that's 7 (=0x7), giving max
    # mask 112 < 128. For D=64 it's 3 (=0x3), giving max mask 48 < 64.
    # SLA test only covers D=128 (mask=7); using a hardcoded `& 7` for
    # D=64 wraps writes into adjacent rows -> silent corruption.
    K_SWZ_ROW_MASK = (K_STRIDE // 16) - 1
    assert K_SWZ_ROW_MASK >= 0
    assert (K_SWZ_ROW_MASK & (K_SWZ_ROW_MASK + 1)) == 0, (
        f"K_SWZ_ROW_MASK must be 2^n-1, got {K_SWZ_ROW_MASK} (K_STRIDE={K_STRIDE})"
    )
    if USE_HW_TR:
        V_STRIDE = HEAD_DIM if ENABLE_DMA else HEAD_DIM + 4
    else:
        VT_STRIDE = BLOCK_N + 2
        V_STRIDE = VT_STRIDE
    # V swizzle: similarly bounded by V_STRIDE / 16.
    V_SWZ_ROW_MASK = min(3, (V_STRIDE // 16) - 1)
    assert V_SWZ_ROW_MASK >= 0

    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8
    assert HEAD_DIM % VEC_WIDTH == 0
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    assert BLOCK_SIZE % THREADS_PER_ROW_LOAD == 0
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        assert BLOCK_N % ROWS_PER_BATCH_LOAD == 0
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False

    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    if USE_HW_TR:
        LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
    else:
        LDS_V_TILE_SIZE = HEAD_DIM * VT_STRIDE
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"v4_hca_bwd_dq_pool_smem_M{BLOCK_M}_P{pool_size}_L{hca_local_seqlen}_MQ{int(mqa_kv)}",
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_offset + LDS_KV_TOTAL_SIZE * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def v4_hca_bwd_dq_pool_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DOS: fx.Tensor,      # grad of output (input)
        LSE: fx.Tensor,      # log-sum-exp from fwd (input, f32, RAW domain, JOINT)
        DELTAS: fx.Tensor,   # (o * do).sum(-1) preprocess (input, f32)
        DQ: fx.Tensor,       # grad wrt Q (output, FP32, ACCUMULATED)
        ADD_MASK: fx.Tensor, # additive pool mask (input, bf16/f16, [Sq, POOL_SIZE])
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32
        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        k_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), K)
        v_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), V)
        do_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DOS)
        dq_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DQ)
        add_mask_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), ADD_MASK)
        # LSE / DELTAS: f32 scalar READS via buffer_load.
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        deltas_rsrc = buffer_ops.create_buffer_resource(DELTAS, max_size=True)

        # All FP operations use aggressive fast-math (no NaN/Inf checks, reassociation).
        fm_fast = arith.FastMathFlags.fast
        v4f16_type = T.vec(4, elem_type)
        vxf16_type = T.vec(VEC_WIDTH, elem_type)
        v8f16_type = T.vec(8, elem_type)
        v16f32_type = T.vec(16, compute_type)
        mfma_pack_type = v8f16_type if USE_K16 else v4f16_type
        MFMA_LANE_K = 8 if USE_K16 else 4
        _mfma_zero = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

        def _mfma(ods_fn, a, b, c):
            return ods_fn(v16f32_type, a, b, c, _mfma_zero, _mfma_zero, _mfma_zero).result

        def mfma_acc(a, b, c):
            if dtype_str == "bf16":
                if USE_K16:
                    return _mfma(rocdl.mfma_f32_32x32x16_bf16, a, b, c)
                a = vector.bitcast(T.i16x4, a)
                b = vector.bitcast(T.i16x4, b)
                return _mfma(rocdl.mfma_f32_32x32x8bf16_1k, a, b, c)
            if USE_K16:
                return _mfma(rocdl.mfma_f32_32x32x16_f16, a, b, c)
            return _mfma(rocdl.mfma_f32_32x32x8f16, a, b, c)

        seq_len_q_v = arith.index_cast(T.index, seq_len_q)
        seq_len_k_v = arith.index_cast(T.index, seq_len_k)

        # ---- LDS view ----
        base_ptr = allocator.get_base()
        lds_kv = SmemPtr(
            base_ptr,
            lds_kv_offset,
            elem_type,
            shape=(LDS_KV_TOTAL_SIZE,),
        ).get()

        # ---- Thread / block indices ----
        block_id = arith.index_cast(T.index, gpu.block_idx.x)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32  # 0/1

        # ds_read_b64_tr_b16 lane decomposition
        tr_k_group = (lane % 16) // 4
        tr_col_sub = lane % 4
        tr_col_half = (lane % 32) // 16

        def ds_read_tr_v4f16(lds_elem_idx):
            byte_offset = lds_elem_idx * 2 + lds_kv_offset
            byte_i64 = arith.index_cast(T.i64, byte_offset)
            ptr = _llvm.IntToPtrOp(_llvm_lds_ptr_ty(), byte_i64).result
            return rocdl.ds_read_tr16_b64(v4f16_type, ptr).result

        wave_q_offset = wave_id * ROWS_PER_WAVE

        # ---- Decompose block_id: (batch, q_tile, head) ----
        head_idx = block_id % NUM_HEADS
        batch_q_tile_id = block_id // NUM_HEADS
        num_q_tiles = (seq_len_q_v + BLOCK_M - 1) // BLOCK_M
        q_tile_idx = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        q_start = q_tile_idx * BLOCK_M

        # ---- V4 HCA POOL: fixed K-block range [HCA_LOCAL_SEQLEN, HCA_LOCAL_SEQLEN+POOL_SIZE) ----
        # POOL_SIZE <= BLOCK_N and HCA_LOCAL_SEQLEN % BLOCK_N == 0 (asserted
        # at build time), so this is exactly ONE n-block per program.
        BN = arith.index(BLOCK_N)
        BM = arith.index(BLOCK_M)
        _zero_idx = arith.index(0)
        _one_idx = arith.index(1)
        n_block_start = arith.index(hca_local_seqlen // BLOCK_N)
        n_block_end = n_block_start + _one_idx

        # ---- Cooperative load decomposition ----
        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        # ---- Helper: global flat index ----
        if layout_bhld:
            bh_base_tokens_q = (batch_idx * NUM_HEADS + head_idx) * seq_len_q_v
            if mqa_kv:
                bh_base_tokens_kv = batch_idx * seq_len_k_v
            else:
                bh_base_tokens_kv = (batch_idx * NUM_HEADS + head_idx) * seq_len_k_v

            def global_idx_q(token_idx, col):
                return (bh_base_tokens_q + token_idx) * arith.index(HEAD_DIM) + col

            def global_idx_kv(token_idx, col):
                return (bh_base_tokens_kv + token_idx) * arith.index(HEAD_DIM) + col
        else:
            # BLHD path: kept for symmetry but V4 uses BHLD.
            def global_idx_q(token_idx, col):
                token = batch_idx * seq_len_q_v + token_idx
                return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

            if mqa_kv:
                def global_idx_kv(token_idx, col):
                    token = batch_idx * seq_len_k_v + token_idx
                    return token * HEAD_DIM + col
            else:
                def global_idx_kv(token_idx, col):
                    token = batch_idx * seq_len_k_v + token_idx
                    return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

        def _gep_load(base_ptr_, elem_idx, vec_type, et=elem_type):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(_llvm_ptr_ty(), base_ptr_, [idx_i64],
                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                              elem_type=et,
                              noWrapFlags=0)
            return _llvm.LoadOp(vec_type, gep.result).result

        def _gep_store_f32(val, base_ptr_, elem_idx):
            """Store a single f32 value via GEP into an fp32 buffer."""
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(_llvm_ptr_ty(), base_ptr_, [idx_i64],
                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                              elem_type=T.f32,
                              noWrapFlags=0)
            _llvm.StoreOp(val, gep.result)

        def load_global_mfma_pack(base_ptr_, base_idx):
            return _gep_load(base_ptr_, base_idx, mfma_pack_type)

        def load_global_f16xN(base_ptr_, base_idx):
            return _gep_load(base_ptr_, base_idx, vxf16_type)

        def bf16_trunc_pack_v4(f32_vals):
            _v2i32 = T.vec(2, T.i32)
            _c16 = arith.constant(16, type=T.i32)
            _cmask = arith.constant(0xFFFF0000, type=T.i32)
            a0 = arith.ArithValue(f32_vals[0]).bitcast(T.i32)
            b0 = arith.ArithValue(f32_vals[1]).bitcast(T.i32)
            p0 = arith.OrIOp(arith.AndIOp(b0, _cmask).result,
                             arith.ShRUIOp(a0, _c16).result).result
            a1 = arith.ArithValue(f32_vals[2]).bitcast(T.i32)
            b1 = arith.ArithValue(f32_vals[3]).bitcast(T.i32)
            p1 = arith.OrIOp(arith.AndIOp(b1, _cmask).result,
                             arith.ShRUIOp(a1, _c16).result).result
            return vector.bitcast(v4f16_type, vector.from_elements(_v2i32, [p0, p1]))

        def bf16_trunc_pack_v8(f32_vals):
            _v4i32 = T.vec(4, T.i32)
            _c16 = arith.constant(16, type=T.i32)
            _cmask = arith.constant(0xFFFF0000, type=T.i32)
            pairs = []
            for j in range_constexpr(4):
                a = arith.ArithValue(f32_vals[j * 2]).bitcast(T.i32)
                b = arith.ArithValue(f32_vals[j * 2 + 1]).bitcast(T.i32)
                p = arith.OrIOp(arith.AndIOp(b, _cmask).result,
                                arith.ShRUIOp(a, _c16).result).result
                pairs.append(p)
            return vector.bitcast(v8f16_type, vector.from_elements(_v4i32, pairs))

        def k_buf_base(buf_id):
            if isinstance(buf_id, int):
                return arith.index(buf_id * LDS_K_TILE_SIZE)
            return buf_id * arith.index(LDS_K_TILE_SIZE)

        def v_buf_base(buf_id):
            if isinstance(buf_id, int):
                return arith.index(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)
            return arith.index(LDS_V_BASE) + buf_id * arith.index(LDS_V_TILE_SIZE)

        # ---- K XOR swizzle: col ^ ((row & K_SWZ_ROW_MASK) << 4) at 16-element granularity ----
        # K_SWZ_ROW_MASK derived from K_STRIDE to stay within the row.
        def _k_swizzle(row_idx, col_idx):
            mask = (row_idx & arith.index(K_SWZ_ROW_MASK)) << arith.index(4)
            return col_idx ^ mask

        # ---- Cooperative K load (row-major, XOR-swizzled) ----
        def coop_load_k(tile_start, buf_id=0):
            """K row-bounds-aware cooperative load.

            Pool kernel reads BLOCK_N rows starting at kv_start; for pool
            sizes < BLOCK_N the top rows are past seq_len_k and would read
            garbage. Gate each load with row_idx < seq_len_k_v and store
            zero into the LDS slot for OOB rows.
            """
            k_base = k_buf_base(buf_id)
            c_zero_vxf16 = arith.constant_vector(0.0, vxf16_type)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid_blk = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    row_valid_seq = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        row_idx, seq_len_k_v,
                    )
                    row_valid = arith.AndIOp(row_valid_blk, row_valid_seq).result
                    lds_row = load_row_in_batch + row_offset
                    swz_col = _k_swizzle(lds_row, load_col_base)
                    lds_idx = k_base + lds_row * K_STRIDE + swz_col
                    g_idx_safe_row = arith.select(row_valid_seq, row_idx, arith.index(0))
                    g_idx = global_idx_kv(g_idx_safe_row, load_col_base)
                    raw_vec = load_global_f16xN(k_ptr, g_idx)
                    vec = arith.select(row_valid, raw_vec, c_zero_vxf16)
                    _if_k = scf.IfOp(arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch, arith.index(BLOCK_N),
                    ))
                    with ir.InsertionPoint(_if_k.then_block):
                        vector.store(vec, lds_kv, [lds_idx])
                        scf.YieldOp([])
                else:
                    row_valid_seq = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        row_idx, seq_len_k_v,
                    )
                    lds_row = load_row_in_batch + row_offset
                    swz_col = _k_swizzle(lds_row, load_col_base)
                    lds_idx = k_base + lds_row * K_STRIDE + swz_col
                    g_idx_safe_row = arith.select(row_valid_seq, row_idx, arith.index(0))
                    g_idx = global_idx_kv(g_idx_safe_row, load_col_base)
                    raw_vec = load_global_f16xN(k_ptr, g_idx)
                    vec = arith.select(row_valid_seq, raw_vec, c_zero_vxf16)
                    vector.store(vec, lds_kv, [lds_idx])

        # ---- Cooperative V-into-K-LDS-slot load (K-style XOR swizzle) ----
        def coop_load_v_as_k(tile_start, buf_id=0):
            """V-into-K-slot with row bounds check (zero OOB rows)."""
            k_base = k_buf_base(buf_id)
            c_zero_vxf16 = arith.constant_vector(0.0, vxf16_type)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid_blk = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    row_valid_seq = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        row_idx, seq_len_k_v,
                    )
                    row_valid = arith.AndIOp(row_valid_blk, row_valid_seq).result
                    lds_row = load_row_in_batch + row_offset
                    swz_col = _k_swizzle(lds_row, load_col_base)
                    lds_idx = k_base + lds_row * K_STRIDE + swz_col
                    g_idx_safe_row = arith.select(row_valid_seq, row_idx, arith.index(0))
                    g_idx = global_idx_kv(g_idx_safe_row, load_col_base)
                    raw_vec = load_global_f16xN(v_ptr, g_idx)
                    vec = arith.select(row_valid, raw_vec, c_zero_vxf16)
                    _if_v = scf.IfOp(arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch, arith.index(BLOCK_N),
                    ))
                    with ir.InsertionPoint(_if_v.then_block):
                        vector.store(vec, lds_kv, [lds_idx])
                        scf.YieldOp([])
                else:
                    row_valid_seq = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        row_idx, seq_len_k_v,
                    )
                    lds_row = load_row_in_batch + row_offset
                    swz_col = _k_swizzle(lds_row, load_col_base)
                    lds_idx = k_base + lds_row * K_STRIDE + swz_col
                    g_idx_safe_row = arith.select(row_valid_seq, row_idx, arith.index(0))
                    g_idx = global_idx_kv(g_idx_safe_row, load_col_base)
                    raw_vec = load_global_f16xN(v_ptr, g_idx)
                    vec = arith.select(row_valid_seq, raw_vec, c_zero_vxf16)
                    vector.store(vec, lds_kv, [lds_idx])

        # ---- Cooperative V load (V LDS layout) ----
        def _v_store_row_major(v_base, lds_row, vec):
            lds_idx = v_base + lds_row * V_STRIDE + load_col_base
            vector.store(vec, lds_kv, [lds_idx])

        _v1_type = T.vec(1, elem_type) if not USE_HW_TR else None

        def _v_store_transposed(v_base, lds_row, vec):
            for _e in range_constexpr(VEC_WIDTH):
                elem = vector.extract(vec, static_position=[_e], dynamic_position=[])
                vt_d = load_col_base + _e
                vt_idx = v_base + vt_d * VT_STRIDE + lds_row
                v1 = vector.from_elements(_v1_type, [elem])
                vector.store(v1, lds_kv, [vt_idx])

        _v_store_to_lds = _v_store_row_major if USE_HW_TR else _v_store_transposed

        def coop_load_v(tile_start, buf_id=0):
            v_base = v_buf_base(buf_id)
            c_zero_vxf16 = arith.constant_vector(0.0, vxf16_type)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid_blk = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    row_valid_seq = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        row_idx, seq_len_k_v,
                    )
                    row_valid = arith.AndIOp(row_valid_blk, row_valid_seq).result
                    g_idx_safe_row = arith.select(row_valid_seq, row_idx, arith.index(0))
                    g_idx = global_idx_kv(g_idx_safe_row, load_col_base)
                    raw_vec = load_global_f16xN(v_ptr, g_idx)
                    vec = arith.select(row_valid, raw_vec, c_zero_vxf16)
                    _if_v = scf.IfOp(arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch, arith.index(BLOCK_N),
                    ))
                    with ir.InsertionPoint(_if_v.then_block):
                        lds_row = load_row_in_batch + row_offset
                        _v_store_to_lds(v_base, lds_row, vec)
                        scf.YieldOp([])
                else:
                    row_valid_seq = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        row_idx, seq_len_k_v,
                    )
                    g_idx_safe_row = arith.select(row_valid_seq, row_idx, arith.index(0))
                    g_idx = global_idx_kv(g_idx_safe_row, load_col_base)
                    raw_vec = load_global_f16xN(v_ptr, g_idx)
                    vec = arith.select(row_valid_seq, raw_vec, c_zero_vxf16)
                    lds_row = load_row_in_batch + row_offset
                    _v_store_to_lds(v_base, lds_row, vec)

        # ---- Cooperative K-into-V-LDS-slot load ----
        def coop_load_k_as_v(tile_start, buf_id=0):
            """K-into-V-slot with row bounds check (zero OOB rows)."""
            v_base = v_buf_base(buf_id)
            c_zero_vxf16 = arith.constant_vector(0.0, vxf16_type)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid_blk = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    row_valid_seq = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        row_idx, seq_len_k_v,
                    )
                    row_valid = arith.AndIOp(row_valid_blk, row_valid_seq).result
                    g_idx_safe_row = arith.select(row_valid_seq, row_idx, arith.index(0))
                    g_idx = global_idx_kv(g_idx_safe_row, load_col_base)
                    raw_vec = load_global_f16xN(k_ptr, g_idx)
                    vec = arith.select(row_valid, raw_vec, c_zero_vxf16)
                    _if_v = scf.IfOp(arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch, arith.index(BLOCK_N),
                    ))
                    with ir.InsertionPoint(_if_v.then_block):
                        lds_row = load_row_in_batch + row_offset
                        _v_store_to_lds(v_base, lds_row, vec)
                        scf.YieldOp([])
                else:
                    row_valid_seq = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        row_idx, seq_len_k_v,
                    )
                    g_idx_safe_row = arith.select(row_valid_seq, row_idx, arith.index(0))
                    g_idx = global_idx_kv(g_idx_safe_row, load_col_base)
                    raw_vec = load_global_f16xN(k_ptr, g_idx)
                    vec = arith.select(row_valid_seq, raw_vec, c_zero_vxf16)
                    lds_row = load_row_in_batch + row_offset
                    _v_store_to_lds(v_base, lds_row, vec)

        # ---- DMA loading for K (buffer_load_dwordx4 ... lds) ----
        if ENABLE_DMA:
            from flydsl._mlir.dialects import llvm
            k_rsrc = buffer_ops.create_buffer_resource(K, max_size=True)
            _lds_ptr_ty = _llvm_lds_ptr_ty()
            DMA_BYTES = 16
            DMA_BATCH_BYTES = BLOCK_SIZE * DMA_BYTES
            K_TILE_BYTES = BLOCK_N * K_STRIDE * 2
            NUM_DMA_K = K_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_K_ROW = HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH = DMA_BATCH_BYTES // (HEAD_DIM * 2)
            lds_kv_base_idx = _memref.extract_aligned_pointer_as_index(lds_kv)
            _dma_size = arith.constant(DMA_BYTES, type=T.i32)
            _dma_soff = arith.constant(0, type=T.i32)
            _dma_off = arith.constant(0, type=T.i32)
            _dma_aux = arith.constant(1, type=T.i32)

            def _kv_global_byte(tile_start, row_in_tile, col_byte):
                if layout_bhld:
                    row_within = tile_start + row_in_tile
                    return ((bh_base_tokens_kv + row_within)
                            * arith.index(HEAD_DIM * 2) + col_byte)
                else:
                    global_row = (batch_idx * seq_len_k_v + tile_start + row_in_tile)
                    if mqa_kv:
                        return (global_row * arith.index(HEAD_DIM * 2) + col_byte)
                    else:
                        return (global_row * arith.index(STRIDE_TOKEN * 2)
                                + head_idx * arith.index(HEAD_DIM * 2)
                                + col_byte)

            def coop_dma_k(tile_start, buf_id=0):
                if isinstance(buf_id, int):
                    k_lds_byte_base = lds_kv_base_idx + arith.index(buf_id * LDS_K_TILE_SIZE * 2)
                else:
                    k_lds_byte_base = lds_kv_base_idx + buf_id * arith.index(LDS_K_TILE_SIZE * 2)
                for d in range_constexpr(NUM_DMA_K):
                    lds_addr = (k_lds_byte_base
                                + wave_id * arith.index(WARP_SIZE * DMA_BYTES)
                                + arith.index(d * DMA_BATCH_BYTES))
                    lds_i64 = arith.index_cast(T.i64, lds_addr)
                    lds_lane0 = rocdl.readfirstlane(T.i64, lds_i64)
                    lds_ptr = llvm.IntToPtrOp(_lds_ptr_ty, lds_lane0).result

                    row_in_tile = (tid // LANES_PER_K_ROW
                                   + arith.index(d * ROWS_PER_DMA_BATCH))
                    swiz_col_f16 = (tid % LANES_PER_K_ROW) * (DMA_BYTES // 2)
                    xor_mask = (row_in_tile & arith.index(K_SWZ_ROW_MASK)) << arith.index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_byte = _kv_global_byte(tile_start, row_in_tile, col_byte)
                    voffset = arith.index_cast(T.i32, global_byte)
                    rocdl.raw_ptr_buffer_load_lds(
                        k_rsrc, lds_ptr, _dma_size, voffset,
                        _dma_soff, _dma_off, _dma_aux,
                    )

        def _v_swizzle(row_idx, col_idx):
            mask = (row_idx & arith.index(V_SWZ_ROW_MASK)) << arith.index(4)
            return col_idx ^ mask

        if ENABLE_DMA:
            v_rsrc = buffer_ops.create_buffer_resource(V, max_size=True)
            V_TILE_BYTES = BLOCK_N * V_STRIDE * 2
            NUM_DMA_V = V_TILE_BYTES // DMA_BATCH_BYTES
            LANES_PER_V_ROW = HEAD_DIM * 2 // DMA_BYTES
            ROWS_PER_DMA_BATCH_V = DMA_BATCH_BYTES // (HEAD_DIM * 2)

            def coop_dma_v(tile_start, buf_id=0):
                v_lds_byte_base = (lds_kv_base_idx
                                   + arith.index((LDS_V_BASE + buf_id * LDS_V_TILE_SIZE) * 2))
                for d in range_constexpr(NUM_DMA_V):
                    lds_addr = (v_lds_byte_base
                                + wave_id * arith.index(WARP_SIZE * DMA_BYTES)
                                + arith.index(d * DMA_BATCH_BYTES))
                    lds_i64 = arith.index_cast(T.i64, lds_addr)
                    lds_lane0 = rocdl.readfirstlane(T.i64, lds_i64)
                    lds_ptr = llvm.IntToPtrOp(_lds_ptr_ty, lds_lane0).result

                    row_in_tile = (tid // LANES_PER_V_ROW
                                   + arith.index(d * ROWS_PER_DMA_BATCH_V))
                    swiz_col_f16 = (tid % LANES_PER_V_ROW) * (DMA_BYTES // 2)
                    xor_mask = (row_in_tile & arith.index(V_SWZ_ROW_MASK)) << arith.index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_byte = _kv_global_byte(tile_start, row_in_tile, col_byte)
                    voffset = arith.index_cast(T.i32, global_byte)
                    rocdl.raw_ptr_buffer_load_lds(
                        v_rsrc, lds_ptr, _dma_size, voffset,
                        _dma_soff, _dma_off, _dma_aux,
                    )

            # ---- Bwd dQ DMA variants (cross-pointer, matching swizzle) ----
            def coop_dma_v_as_k(tile_start, buf_id=0):
                if isinstance(buf_id, int):
                    k_lds_byte_base = lds_kv_base_idx + arith.index(buf_id * LDS_K_TILE_SIZE * 2)
                else:
                    k_lds_byte_base = lds_kv_base_idx + buf_id * arith.index(LDS_K_TILE_SIZE * 2)
                for d in range_constexpr(NUM_DMA_K):
                    lds_addr = (k_lds_byte_base
                                + wave_id * arith.index(WARP_SIZE * DMA_BYTES)
                                + arith.index(d * DMA_BATCH_BYTES))
                    lds_i64 = arith.index_cast(T.i64, lds_addr)
                    lds_lane0 = rocdl.readfirstlane(T.i64, lds_i64)
                    lds_ptr = llvm.IntToPtrOp(_lds_ptr_ty, lds_lane0).result

                    row_in_tile = (tid // LANES_PER_K_ROW
                                   + arith.index(d * ROWS_PER_DMA_BATCH))
                    swiz_col_f16 = (tid % LANES_PER_K_ROW) * (DMA_BYTES // 2)
                    xor_mask = (row_in_tile & arith.index(K_SWZ_ROW_MASK)) << arith.index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_byte = _kv_global_byte(tile_start, row_in_tile, col_byte)
                    voffset = arith.index_cast(T.i32, global_byte)
                    rocdl.raw_ptr_buffer_load_lds(
                        v_rsrc, lds_ptr, _dma_size, voffset,
                        _dma_soff, _dma_off, _dma_aux,
                    )

            def coop_dma_k_as_v(tile_start, buf_id=0):
                if isinstance(buf_id, int):
                    v_lds_byte_base = (lds_kv_base_idx
                                       + arith.index((LDS_V_BASE + buf_id * LDS_V_TILE_SIZE) * 2))
                else:
                    v_lds_byte_base = (lds_kv_base_idx
                                       + arith.index(LDS_V_BASE * 2)
                                       + buf_id * arith.index(LDS_V_TILE_SIZE * 2))
                for d in range_constexpr(NUM_DMA_V):
                    lds_addr = (v_lds_byte_base
                                + wave_id * arith.index(WARP_SIZE * DMA_BYTES)
                                + arith.index(d * DMA_BATCH_BYTES))
                    lds_i64 = arith.index_cast(T.i64, lds_addr)
                    lds_lane0 = rocdl.readfirstlane(T.i64, lds_i64)
                    lds_ptr = llvm.IntToPtrOp(_lds_ptr_ty, lds_lane0).result

                    row_in_tile = (tid // LANES_PER_V_ROW
                                   + arith.index(d * ROWS_PER_DMA_BATCH_V))
                    swiz_col_f16 = (tid % LANES_PER_V_ROW) * (DMA_BYTES // 2)
                    xor_mask = (row_in_tile & arith.index(V_SWZ_ROW_MASK)) << arith.index(4)
                    unsw_col_f16 = swiz_col_f16 ^ xor_mask
                    col_byte = unsw_col_f16 * 2
                    global_byte = _kv_global_byte(tile_start, row_in_tile, col_byte)
                    voffset = arith.index_cast(T.i32, global_byte)
                    rocdl.raw_ptr_buffer_load_lds(
                        k_rsrc, lds_ptr, _dma_size, voffset,
                        _dma_soff, _dma_off, _dma_aux,
                    )

        # ---- Preload Q^T B-operand and DO^T B-operand packs ----
        q_row = q_start + wave_q_offset + lane_mod_32
        q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, q_row, seq_len_q_v)
        q_row_safe = arith.select(q_in_bounds, q_row, arith.index(0))
        c_zero_mfma_pack = arith.constant_vector(0.0, mfma_pack_type)
        q_b_packs = []
        do_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
            g_idx = global_idx_q(q_row_safe, col)
            q_raw = load_global_mfma_pack(q_ptr, g_idx)
            q_b_packs.append(arith.select(q_in_bounds, q_raw, c_zero_mfma_pack))
            do_raw = load_global_mfma_pack(do_ptr, g_idx)
            do_b_packs.append(arith.select(q_in_bounds, do_raw, c_zero_mfma_pack))

        # ---- Constants ----
        c_zero_f = arith.constant(0.0, type=compute_type)
        c_neg_inf = arith.constant(-1.0e30, type=compute_type)  # finite NEG_INF (matches Triton)
        c_zero_v16f32 = arith.constant_vector(0.0, v16f32_type)
        # V4 LSE is RAW-domain (qk*sm_scale + ln(l)). So use sm_scale, not sm_scale*log2e.
        c_sm_scale = arith.constant(sm_scale, type=compute_type)

        # ---- Per-q-row scalars (LSE, delta) ----
        # bh_base_tokens_q is the Q LSE/DELTAS base too (LSE shape [B,HQ,Sq]).
        lse_delta_off_i32 = arith.index_cast(
            T.i32, bh_base_tokens_q + q_row_safe)
        lse_val = buffer_ops.buffer_load(
            lse_rsrc, lse_delta_off_i32, vec_width=1, dtype=T.f32,
        )
        delta_val = buffer_ops.buffer_load(
            deltas_rsrc, lse_delta_off_i32, vec_width=1, dtype=T.f32,
        )

        # ---- (No SINK contribution: pool kernel does not touch sink.) ----

        # ---- POOL: single n-block; double-buffered LDS optional ----
        _use_dbuf = ENABLE_DMA

        init_args = []
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v16f32)
        if _use_dbuf:
            init_args.append(arith.index(0))  # cur_buf_id

            # PROLOGUE: prefetch iter 0's K into BOTH slots of buf 0.
            _init_kv_start = n_block_start * BN
            coop_dma_k(_init_kv_start, buf_id=0)
            coop_dma_k_as_v(_init_kv_start, buf_id=0)

        for block_idx, inner_iter_args, loop_results in scf.for_(
            n_block_start,
            n_block_end,
            arith.index(1),
            iter_args=init_args,
        ):
            dq_accs = [
                inner_iter_args[i] for i in range_constexpr(D_CHUNKS)
            ]
            if _use_dbuf:
                cur_buf = inner_iter_args[D_CHUNKS]
                next_buf = arith.index(1) - cur_buf

            # V4 SWA: block_idx is directly the K-block index.
            kv_block_start = block_idx * BN
            kv_start = kv_block_start

            if _use_dbuf:
                rocdl.s_waitcnt(0)
                gpu.barrier()
                k_base = k_buf_base(cur_buf)
            else:
                coop_load_k(kv_start, buf_id=0)
                coop_load_k_as_v(kv_start, buf_id=0)
                gpu.barrier()
                k_base = k_buf_base(0)

            # ==== GEMM1: s = Q @ K^T ====
            k_hi_offset = K_SUB_N * K_STRIDE
            k_swz_mask = (lane_mod_32 & arith.index(K_SWZ_ROW_MASK)) << arith.index(4)

            def _k_idx_lo(ks):
                col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                return k_base + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask)

            def _k_idx_hi(ks):
                col = arith.index(ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                return (k_base + k_hi_offset
                        + lane_mod_32 * K_STRIDE + (col ^ k_swz_mask))

            s_acc_lo = c_zero_v16f32
            s_acc_hi = c_zero_v16f32
            for ks in range_constexpr(K_STEPS_QK):
                k_pack_lo = vector.load_op(
                    mfma_pack_type, lds_kv, [_k_idx_lo(ks)])
                k_pack_hi = vector.load_op(
                    mfma_pack_type, lds_kv, [_k_idx_hi(ks)])
                s_acc_lo = mfma_acc(k_pack_lo, q_b_packs[ks], s_acc_lo)
                s_acc_hi = mfma_acc(k_pack_hi, q_b_packs[ks], s_acc_hi)

            # ==== Compute p[r] = exp((qk + add_bias) * sm_scale_inline - LSE) ====
            #
            # POOL-only mask. The s_acc registers map (per lane, per reg r) to
            # the S = Q @ K^T cell at row = q_row (= lane_mod_32 + wave_q_offset
            # + q_start) and column = kv_start + lane_div_32*4 + (r//4)*8 + r%4
            # (lo half), or +K_SUB_N (hi half). Translate to pool_n by
            # subtracting HCA_LOCAL_SEQLEN; gate with pool_n < POOL_SIZE and
            # q_row < seq_len_q.
            seq_len_q_i32 = arith.index_cast(T.i32, seq_len_q_v)
            q_row_i32_mask = arith.index_cast(T.i32, q_row)
            lane_div_32_i32 = arith.index_cast(T.i32, lane_div_32)
            lane_off_i32 = arith.MulIOp(
                lane_div_32_i32, arith.constant(4, type=T.i32)).result
            hca_local_i32 = arith.constant(hca_local_seqlen, type=T.i32)
            pool_size_i32 = arith.constant(pool_size, type=T.i32)
            kv_start_i32 = arith.index_cast(T.i32, kv_start)
            # pool_n base for this lane = kv_start + lane_off - hca_local_seqlen
            # (kv_start = hca_local_seqlen + 0 for the single pool block).
            pool_n_base_i32 = arith.SubIOp(
                arith.AddIOp(kv_start_i32, lane_off_i32).result,
                hca_local_i32,
            ).result

            # Is this q_row out of bounds? -> entire row masked.
            q_oob = arith.cmpi(
                arith.CmpIPredicate.sge, q_row_i32_mask, seq_len_q_i32,
            )

            # --- Load add_bias from ADD_MASK[q_row, pool_n] via GEP ---
            # ADD_MASK shape [Sq, POOL_SIZE] contiguous; element bytes = 2
            # (bf16 / f16). For each (lane, r), compute pool_n (lo or hi) and
            # load a single bf16/f16, then cast to f32.
            ADD_MASK_STRIDE_M = arith.index(pool_size)  # row stride in elements
            q_row_idx_for_mask = arith.select(q_oob, arith.index(0), q_row)
            # Precompute add-mask row base offset in elements.
            add_mask_row_base = q_row_idx_for_mask * ADD_MASK_STRIDE_M

            def _load_add_bias(pool_n_i32, bad_pred):
                """Load ADD_MASK[q_row, pool_n] as f32; return 0.0 if bad."""
                # Convert pool_n to index for GEP.
                pool_n_i32_safe = arith.select(
                    bad_pred, arith.constant(0, type=T.i32), pool_n_i32,
                )
                pool_n_idx = arith.index_cast(T.index, pool_n_i32_safe)
                elem_idx = add_mask_row_base + pool_n_idx
                bias_raw_v1 = _gep_load(
                    add_mask_ptr, elem_idx, T.vec(1, elem_type),
                )
                bias_raw = vector.extract(
                    bias_raw_v1, static_position=[0], dynamic_position=[],
                )
                bias_f32 = arith.extf(compute_type, bias_raw)
                # If bad (OOB pool_n or q_oob), zero out the bias (mask wins
                # anyway via NEG_INF, but keep numerics tidy).
                return arith.select(bad_pred, c_zero_f, bias_f32)

            p_vals_lo = []
            p_vals_hi = []
            for r in range_constexpr(16):
                r_off_i32 = arith.constant(
                    (r % 4) + (r // 4) * 8, type=T.i32)

                # --- lo half ---
                pool_n_lo_i32 = arith.AddIOp(
                    pool_n_base_i32, r_off_i32,
                ).result
                is_pool_oob_lo = arith.cmpi(
                    arith.CmpIPredicate.sge, pool_n_lo_i32, pool_size_i32,
                )
                bad_lo = arith.OrIOp(is_pool_oob_lo, q_oob).result

                s_lo_f32 = vector.extract(
                    s_acc_lo, static_position=[r], dynamic_position=[])
                bias_lo = _load_add_bias(pool_n_lo_i32, bad_lo)
                qk_plus_bias_lo = arith.AddFOp(
                    s_lo_f32, bias_lo, fastmath=fm_fast).result
                # Triton: qk = qk * sm_scale; qk = qk + add_bias.
                # We deferred sm_scale to here: (qk + bias) is wrong; need
                # qk*scale + bias. Re-do as: s_scaled + bias.
                scaled_lo = arith.MulFOp(
                    s_lo_f32, c_sm_scale, fastmath=fm_fast).result
                scaled_plus_bias_lo = arith.AddFOp(
                    scaled_lo, bias_lo, fastmath=fm_fast).result
                scaled_lo_masked = arith.select(
                    bad_lo, c_neg_inf, scaled_plus_bias_lo,
                )
                diff_lo = arith.SubFOp(
                    scaled_lo_masked, lse_val, fastmath=fm_fast).result
                p_lo = math_dialect.exp(diff_lo, fastmath=fm_fast)
                p_vals_lo.append(p_lo)

                # --- hi half: pool_n_hi = pool_n_lo + K_SUB_N ---
                pool_n_hi_i32 = arith.AddIOp(
                    pool_n_lo_i32,
                    arith.constant(K_SUB_N, type=T.i32),
                ).result
                is_pool_oob_hi = arith.cmpi(
                    arith.CmpIPredicate.sge, pool_n_hi_i32, pool_size_i32,
                )
                bad_hi = arith.OrIOp(is_pool_oob_hi, q_oob).result

                s_hi_f32 = vector.extract(
                    s_acc_hi, static_position=[r], dynamic_position=[])
                bias_hi = _load_add_bias(pool_n_hi_i32, bad_hi)
                scaled_hi = arith.MulFOp(
                    s_hi_f32, c_sm_scale, fastmath=fm_fast).result
                scaled_plus_bias_hi = arith.AddFOp(
                    scaled_hi, bias_hi, fastmath=fm_fast).result
                scaled_hi_masked = arith.select(
                    bad_hi, c_neg_inf, scaled_plus_bias_hi,
                )
                diff_hi = arith.SubFOp(
                    scaled_hi_masked, lse_val, fastmath=fm_fast).result
                p_hi = math_dialect.exp(diff_hi, fastmath=fm_fast)
                p_vals_hi.append(p_hi)

            # ==== Overwrite K LDS slot with V for GEMM2 ====
            gpu.barrier()
            if _use_dbuf:
                coop_dma_v_as_k(kv_start, buf_id=cur_buf)
                rocdl.s_waitcnt(0)
                gpu.barrier()
            elif ENABLE_DMA:
                coop_dma_v_as_k(kv_start, buf_id=0)
                rocdl.s_waitcnt(0)
                gpu.barrier()
            else:
                coop_load_v_as_k(kv_start, buf_id=0)
                gpu.barrier()

            # ==== GEMM2: dP = DO @ V^T ====
            dp_acc_lo = c_zero_v16f32
            dp_acc_hi = c_zero_v16f32
            for ks in range_constexpr(K_STEPS_QK):
                v_pack_lo = vector.load_op(
                    mfma_pack_type, lds_kv, [_k_idx_lo(ks)])
                v_pack_hi = vector.load_op(
                    mfma_pack_type, lds_kv, [_k_idx_hi(ks)])
                dp_acc_lo = mfma_acc(v_pack_lo, do_b_packs[ks], dp_acc_lo)
                dp_acc_hi = mfma_acc(v_pack_hi, do_b_packs[ks], dp_acc_hi)

            # ==== Prefetch iter+1's K DMAs (async) ====
            if _use_dbuf:
                _next_block_idx = block_idx + arith.index(1)
                _has_next = arith.cmpi(
                    arith.CmpIPredicate.slt,
                    _next_block_idx, n_block_end)
                _pre_if = scf.IfOp(_has_next)
                with ir.InsertionPoint(_pre_if.then_block):
                    _next_kv_start = _next_block_idx * BN
                    coop_dma_k(_next_kv_start, next_buf)
                    coop_dma_k_as_v(_next_kv_start, next_buf)
                    scf.YieldOp([])

            # ==== Compute dS[r] = p[r] * (dp[r] - delta) ====
            ds_vals_lo = []
            ds_vals_hi = []
            for r in range_constexpr(16):
                dp_lo = vector.extract(
                    dp_acc_lo, static_position=[r], dynamic_position=[])
                dp_hi = vector.extract(
                    dp_acc_hi, static_position=[r], dynamic_position=[])
                diff_lo = arith.SubFOp(
                    dp_lo, delta_val, fastmath=fm_fast).result
                diff_hi = arith.SubFOp(
                    dp_hi, delta_val, fastmath=fm_fast).result
                ds_lo = arith.MulFOp(
                    p_vals_lo[r], diff_lo, fastmath=fm_fast).result
                ds_hi = arith.MulFOp(
                    p_vals_hi[r], diff_hi, fastmath=fm_fast).result
                ds_vals_lo.append(ds_lo)
                ds_vals_hi.append(ds_hi)

            # ==== Pack dS f32 -> mfma_pack_type (bf16/f16) ====
            if dtype_str == "bf16" and USE_K16:
                ds_packs_lo = []
                ds_packs_hi = []
                for pks in range_constexpr(PV_K_STEPS):
                    base = pks * 8
                    ds_packs_lo.append(bf16_trunc_pack_v8(
                        ds_vals_lo[base:base+8]))
                    ds_packs_hi.append(bf16_trunc_pack_v8(
                        ds_vals_hi[base:base+8]))
            elif dtype_str == "bf16":
                ds_packs_lo = []
                ds_packs_hi = []
                for pks in range_constexpr(PV_K_STEPS):
                    base = pks * 4
                    ds_packs_lo.append(bf16_trunc_pack_v4(
                        ds_vals_lo[base:base+4]))
                    ds_packs_hi.append(bf16_trunc_pack_v4(
                        ds_vals_hi[base:base+4]))
            else:
                ds_f16_lo = [arith.trunc_f(elem_type, ds_vals_lo[r])
                             for r in range_constexpr(16)]
                ds_f16_hi = [arith.trunc_f(elem_type, ds_vals_hi[r])
                             for r in range_constexpr(16)]
                _pack_ty = v8f16_type if USE_K16 else v4f16_type
                _pw = 8 if USE_K16 else 4
                ds_packs_lo = []
                ds_packs_hi = []
                for pks in range_constexpr(PV_K_STEPS):
                    b = pks * _pw
                    ds_packs_lo.append(vector.from_elements(
                        _pack_ty, [ds_f16_lo[b + i] for i in range(_pw)]))
                    ds_packs_hi.append(vector.from_elements(
                        _pack_ty, [ds_f16_hi[b + i] for i in range(_pw)]))

            # ==== GEMM3: dQ += K @ dS  (uses fwd's V-read schedule with K substituted) ====
            _steps = [(dc, pks)
                      for dc in range(D_CHUNKS)
                      for pks in range(PV_K_STEPS)]
            TOTAL_PV = len(_steps)
            v_base = v_buf_base(cur_buf) if _use_dbuf else v_buf_base(0)

            def _read_k_as_v_pack(step_idx):
                dc, pks = _steps[step_idx]
                if USE_HW_TR:
                    d_col = (arith.index(dc * D_CHUNK)
                             + tr_col_half * 16 + tr_col_sub * 4)
                    k_row = (arith.index(pks * PV_K_STEP)
                             + lane_div_32 * 4 + tr_k_group)
                    _d_col_eff = _v_swizzle(k_row, d_col) if ENABLE_DMA else d_col
                    lds_lo = v_base + k_row * V_STRIDE + _d_col_eff
                    lds_hi = lds_lo + arith.index(K_SUB_N * V_STRIDE)
                    if USE_K16:
                        vl_a = ds_read_tr_v4f16(lds_lo)
                        vl_b = ds_read_tr_v4f16(
                            lds_lo + arith.index(8 * V_STRIDE))
                        vl = vector.shuffle(
                            vl_a, vl_b, [0, 1, 2, 3, 4, 5, 6, 7])
                        vh_a = ds_read_tr_v4f16(lds_hi)
                        vh_b = ds_read_tr_v4f16(
                            lds_hi + arith.index(8 * V_STRIDE))
                        vh = vector.shuffle(
                            vh_a, vh_b, [0, 1, 2, 3, 4, 5, 6, 7])
                    else:
                        vl = ds_read_tr_v4f16(lds_lo)
                        vh = ds_read_tr_v4f16(lds_hi)
                else:
                    d_pos = arith.index(dc * D_CHUNK) + lane_mod_32
                    kb = arith.index(pks * PV_K_STEP) + lane_div_32 * 4
                    v_lo_idx = v_base + d_pos * VT_STRIDE + kb
                    v_hi_idx = v_lo_idx + arith.index(K_SUB_N)
                    vl = vector.load(v4f16_type, lds_kv, [v_lo_idx])
                    vh = vector.load(v4f16_type, lds_kv, [v_hi_idx])
                return vl, vh

            k_lo_cur, k_hi_cur = _read_k_as_v_pack(0)
            for si in range_constexpr(TOTAL_PV):
                dc, pks = _steps[si]
                if si + 1 < TOTAL_PV:
                    k_lo_nxt, k_hi_nxt = _read_k_as_v_pack(si + 1)
                dq_accs[dc] = mfma_acc(
                    k_lo_cur, ds_packs_lo[pks], dq_accs[dc])
                dq_accs[dc] = mfma_acc(
                    k_hi_cur, ds_packs_hi[pks], dq_accs[dc])
                if si + 1 < TOTAL_PV:
                    k_lo_cur = k_lo_nxt
                    k_hi_cur = k_hi_nxt

            # End of iter: yield dq_accs and swapped cur_buf.
            gpu.barrier()
            _yield = list(dq_accs)
            if _use_dbuf:
                _yield.append(next_buf)
            yield _yield

        # ---- Final store: dQ_fp32 += dq_acc * sm_scale (load + add + store) ----
        # Pool kernel ACCUMULATES into the existing LOCAL-stream dq buffer.
        # Race-free because each program owns a unique (b, qhid, m_block)
        # slice, and the launcher runs this kernel sequentially AFTER the
        # SWA dq kernel has finished writing the LOCAL contribution.
        dq_finals = [
            loop_results[dc] for dc in range_constexpr(D_CHUNKS)
        ]
        sm_scale_vec = vector.broadcast(v16f32_type, c_sm_scale)

        def _gep_load_f32(base_ptr_, elem_idx):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(_llvm_ptr_ty(), base_ptr_, [idx_i64],
                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                              elem_type=T.f32,
                              noWrapFlags=0)
            return _llvm.LoadOp(T.f32, gep.result).result

        _o_guard = scf.IfOp(q_in_bounds, [], has_else=False)
        with ir.InsertionPoint(_o_guard.then_block):
            for dc in range_constexpr(D_CHUNKS):
                dq_scaled = arith.MulFOp(
                    dq_finals[dc], sm_scale_vec, fastmath=fm_fast,
                ).result
                for r in range_constexpr(16):
                    dq_val = vector.extract(
                        dq_scaled,
                        static_position=[r],
                        dynamic_position=[],
                    )
                    d_row_rel = lane_div_32 * 4 + (r // 4) * 8 + (r % 4)
                    d_col = arith.index(dc * D_CHUNK) + d_row_rel
                    dq_global = global_idx_q(q_row, d_col)
                    dq_prev = _gep_load_f32(dq_ptr, dq_global)
                    dq_sum = arith.AddFOp(
                        dq_val, dq_prev, fastmath=fm_fast,
                    ).result
                    _gep_store_f32(dq_sum, dq_ptr, dq_global)
            scf.YieldOp([])

    @flyc.jit
    def launch_v4_hca_bwd_dq_pool(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DOS: fx.Tensor,
        LSE: fx.Tensor,
        DELTAS: fx.Tensor,
        DQ: fx.Tensor,
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
        sl_idx = arith.index_cast(T.index, seq_len_q)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS

        launcher = v4_hca_bwd_dq_pool_kernel(
            Q, K, V, DOS, LSE, DELTAS, DQ, ADD_MASK, seq_len_q, seq_len_k,
        )

        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32,
                            _wpe,
                        )
        if flat_work_group_size is not None:
            _fwgs = int(flat_work_group_size)
            if _fwgs >= 1:
                flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
                for op in ctx.gpu_module_body.operations:
                    if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                        op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

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
            return launch_v4_hca_bwd_dq_pool(*args, **kwargs)

    def _compile(Q, K, V, DOS, LSE, DELTAS, DQ, ADD_MASK,
                 batch_size, seq_len_q, seq_len_k, stream=None):
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return flyc.compile(
                launch_v4_hca_bwd_dq_pool, Q, K, V, DOS, LSE, DELTAS, DQ, ADD_MASK,
                batch_size, seq_len_q, seq_len_k, fx.Stream(stream))

    _launch.compile = _compile

    return _launch


# Convenience alias.
build_v4_hca_bwd_dq_pool_module_primary = build_v4_hca_bwd_dq_pool_module
