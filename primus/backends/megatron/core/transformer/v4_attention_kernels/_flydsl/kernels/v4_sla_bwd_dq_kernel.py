# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""v4_swa_bwd_dq: V4 SWA-causal attention backward dQ kernel for FlyDSL.

Forked from kernels/sla_bwd_dq.py (FlyDSL SLA backward dQ). Differences:
  - Outer KV loop iterates a contiguous block range bounded by the SWA
    window for the current Q tile (no LUT).
  - Per-element SWA + causal mask + boundary mask (mirrors Triton ref).
  - LSE is RAW-domain (Triton: lse = m + ln(l), in domain qk*sm_scale),
    so p = exp(qk*sm_scale - lse) (NOT exp2).
  - sm_scale is applied TWICE: once inside the loop on qk (matches
    Triton: qk = qk * sm_scale), and once after the n-loop on dq
    (matches Triton: dq = dq * sm_scale). The two multiplies are NOT
    redundant: ds = p*(dp - dvec) carries the in-loop scale via P;
    the post-loop multiply is the outer chain-rule scale on dq.
  - MQA: K/V are [B, 1, Sk, D] - stride_kh = 0 - drop head_idx from
    KV indexing when mqa_kv=True.
  - Sink: optional SINK[HQ] fp32; dsink = -sum(p_sink * dvec) via
    atomic_fadd one scalar per row-owner-lane into DSINK[qhid]. Sink
    does NOT change dq (fwd already folded p_sink into lse).
  - dq written as fp32 (matches the launcher's fp32 dq_fp32 buffer).

Layout: BHLD. Q/DOUT/DQ flat from (B, HQ, Sq, D). K/V flat from
        (B, HK, Sk, D); MQA means HK=1, no head stride applied to KV.
LSE/DELTAS: (B, HQ, Sq) flat, fp32, raw-domain.
SINK:   (HQ,) fp32 - dummy buffer when has_sink=False.
DSINK:  (HQ,) fp32 - atomic accumulator - caller must zero-init.
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

KERNEL_NAME = "v4_swa_bwd_dq_kernel"

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


def build_v4_swa_bwd_dq_module(
    num_heads,
    head_dim,
    swa_window,
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
    has_sink=True,
):
    """Build the V4 SWA backward dQ launcher.

    Args:
        num_heads: HQ (number of Q heads).
        head_dim: D (head dimension), must be % 32 == 0 and >= 64.
        swa_window: int > 0; SWA window length.
        sm_scale: defaults to 1/sqrt(head_dim).
        mqa_kv: if True, K/V indexing drops head_idx (HK=1, stride_h=0).
        has_sink: if True, SINK/DSINK tensors are used. If False, both
            are dummy buffers (the kernel still takes them but doesn't
            touch them).
        layout_bhld: BHLD layout (True) or BLHD (False); V4 uses BHLD.

    Returns: launch(Q, K, V, DOUT, LSE, DELTAS, DQ_FP32, DSINK, SINK,
                    batch_size, seq_len_q, seq_len_k, stream=None).
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
                f"v4_swa_bwd_dq: predicted LDS {_pred2}B > limit {_LDS_LIMIT_BYTES}B "
                f"even single-buffered at D={head_dim}, BLOCK_N={BLOCK_N}"
            )
        import sys as _sys
        print(
            f"[v4_swa_bwd_dq] LDS overflow at D={head_dim}, BLOCK_N={BLOCK_N}: "
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
    assert dtype_str in ("f16", "bf16"), "v4_swa_bwd_dq only supports f16 and bf16"
    assert BLOCK_N % 32 == 0
    assert BLOCK_N_OUT == BLOCK_N
    assert isinstance(swa_window, int) and swa_window > 0, (
        f"swa_window must be int > 0, got {swa_window!r}")

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
        global_sym_name=f"v4_swa_bwd_dq_smem_M{BLOCK_M}_W{swa_window}_S{int(has_sink)}_MQ{int(mqa_kv)}",
    )
    lds_kv_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_offset + LDS_KV_TOTAL_SIZE * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def v4_swa_bwd_dq_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DOS: fx.Tensor,      # grad of output (input)
        LSE: fx.Tensor,      # log-sum-exp from fwd (input, f32, RAW domain)
        DELTAS: fx.Tensor,   # (o * do).sum(-1) preprocess (input, f32)
        DQ: fx.Tensor,       # grad wrt Q (output, FP32)
        DSINK: fx.Tensor,    # grad wrt sink (output, FP32, [HQ])
        SINK: fx.Tensor,     # sink param (input, FP32, [HQ]); dummy if !has_sink
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
        # LSE / DELTAS: f32 scalar READS via buffer_load.
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        deltas_rsrc = buffer_ops.create_buffer_resource(DELTAS, max_size=True)
        sink_rsrc = buffer_ops.create_buffer_resource(SINK, max_size=True)
        # DSINK: f32 atomic_fadd via buffer_atomic_fadd.
        dsink_rsrc = buffer_ops.create_buffer_resource(DSINK, max_size=True)

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

        # ---- V4 SWA: per-tile contiguous K-block range (wave-uniform) ----
        SWA = arith.index(swa_window)
        BN = arith.index(BLOCK_N)
        BM = arith.index(BLOCK_M)
        _zero_idx = arith.index(0)
        _one_idx = arith.index(1)
        # n_block_start = max(0, q_start - W + 1) // BLOCK_N
        # n_block_end   = ceil(min(q_start + BLOCK_M, seq_len_k), BLOCK_N)
        _q_plus_one = q_start + _one_idx
        _ge_w = arith.cmpi(arith.CmpIPredicate.sge, _q_plus_one, SWA)
        _n_start_row = arith.select(_ge_w, _q_plus_one - SWA, _zero_idx)
        n_block_start = _n_start_row // BN
        _n_end_row_uncl = q_start + BM
        _le_seq = arith.cmpi(arith.CmpIPredicate.sle, _n_end_row_uncl, seq_len_k_v)
        n_end_row_cl = arith.select(_le_seq, _n_end_row_uncl, seq_len_k_v)
        n_block_end = (n_end_row_cl + BN - _one_idx) // BN

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
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    _if_k = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_k.then_block):
                        g_idx = global_idx_kv(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        swz_col = _k_swizzle(lds_row, load_col_base)
                        lds_idx = k_base + lds_row * K_STRIDE + swz_col
                        vec = load_global_f16xN(k_ptr, g_idx)
                        vector.store(vec, lds_kv, [lds_idx])
                        scf.YieldOp([])
                else:
                    g_idx = global_idx_kv(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    swz_col = _k_swizzle(lds_row, load_col_base)
                    lds_idx = k_base + lds_row * K_STRIDE + swz_col
                    vec = load_global_f16xN(k_ptr, g_idx)
                    vector.store(vec, lds_kv, [lds_idx])

        # ---- Cooperative V-into-K-LDS-slot load (K-style XOR swizzle) ----
        def coop_load_v_as_k(tile_start, buf_id=0):
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    _if_v = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_v.then_block):
                        g_idx = global_idx_kv(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        swz_col = _k_swizzle(lds_row, load_col_base)
                        lds_idx = k_base + lds_row * K_STRIDE + swz_col
                        vec = load_global_f16xN(v_ptr, g_idx)
                        vector.store(vec, lds_kv, [lds_idx])
                        scf.YieldOp([])
                else:
                    g_idx = global_idx_kv(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    swz_col = _k_swizzle(lds_row, load_col_base)
                    lds_idx = k_base + lds_row * K_STRIDE + swz_col
                    vec = load_global_f16xN(v_ptr, g_idx)
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
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    _if_v = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_v.then_block):
                        g_idx = global_idx_kv(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        vec = load_global_f16xN(v_ptr, g_idx)
                        _v_store_to_lds(v_base, lds_row, vec)
                        scf.YieldOp([])
                else:
                    g_idx = global_idx_kv(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    vec = load_global_f16xN(v_ptr, g_idx)
                    _v_store_to_lds(v_base, lds_row, vec)

        # ---- Cooperative K-into-V-LDS-slot load ----
        def coop_load_k_as_v(tile_start, buf_id=0):
            v_base = v_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                row_idx = tile_start + load_row_in_batch + row_offset
                if KV_NEEDS_GUARD:
                    row_valid = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        load_row_in_batch,
                        arith.index(BLOCK_N),
                    )
                    _if_v = scf.IfOp(row_valid)
                    with ir.InsertionPoint(_if_v.then_block):
                        g_idx = global_idx_kv(row_idx, load_col_base)
                        lds_row = load_row_in_batch + row_offset
                        vec = load_global_f16xN(k_ptr, g_idx)
                        _v_store_to_lds(v_base, lds_row, vec)
                        scf.YieldOp([])
                else:
                    g_idx = global_idx_kv(row_idx, load_col_base)
                    lds_row = load_row_in_batch + row_offset
                    vec = load_global_f16xN(k_ptr, g_idx)
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

        # ---- SINK contribution to DSINK ----
        # Per Triton ref:
        #   sink_h = SINK[qhid]   (uniform across lanes of this program)
        #   p_sink = exp(sink_h - lse)
        #   dsink_contrib = sum_m -p_sink_masked * dvec_masked  (over BLOCK_M rows)
        #   atomic_fadd(DSINK + qhid, dsink_contrib)
        # We do the atomic per row-owner lane (lane_div_32==0 lane) to avoid
        # the cross-wave reduction. That gives BLOCK_M atomics per program -
        # slow but bulletproof. Each lane contributes -p_sink * dvec for its
        # owned row only when q_in_bounds.
        if has_sink:
            head_idx_i32 = arith.index_cast(T.i32, head_idx)
            sink_h_scalar = buffer_ops.buffer_load(
                sink_rsrc, head_idx_i32, vec_width=1, dtype=T.f32,
            )
            sink_h_uniform = rocdl.readfirstlane(T.f32, sink_h_scalar)
            sub_val = arith.SubFOp(
                sink_h_uniform, lse_val, fastmath=fm_fast,
            ).result
            p_sink = math_dialect.exp(sub_val, fastmath=fm_fast)
            neg_p_sink = arith.SubFOp(
                c_zero_f, p_sink, fastmath=fm_fast,
            ).result
            contrib = arith.MulFOp(
                neg_p_sink, delta_val, fastmath=fm_fast,
            ).result
            # Gate: only the row-owner lane (lane_div_32==0) and only when
            # q_row < seq_len_q, atomically adds.
            is_row_owner = arith.cmpi(
                arith.CmpIPredicate.eq, lane_div_32, arith.index(0),
            )
            do_sink_atomic = arith.AndIOp(is_row_owner, q_in_bounds).result
            _if_sink = scf.IfOp(do_sink_atomic, [], has_else=False)
            with ir.InsertionPoint(_if_sink.then_block):
                # DSINK byte offset = head_idx * 4 (fp32).
                _dsink_byte_off = arith.MulIOp(
                    head_idx_i32, arith.constant(4, type=T.i32),
                ).result
                _zero_i32_atom = arith.constant(0, type=T.i32)
                rocdl.raw_ptr_buffer_atomic_fadd(
                    contrib, dsink_rsrc, _dsink_byte_off,
                    _zero_i32_atom, _zero_i32_atom,
                )
                scf.YieldOp([])

        # ---- MILESTONE: dense SWA bwd dQ with double-buffered LDS ----
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

            # ==== Compute p[r] = exp(qk * sm_scale - LSE) with SWA + causal + boundary mask ====
            kv_start_i32 = arith.index_cast(T.i32, kv_start)
            seq_len_k_i32 = arith.index_cast(T.i32, seq_len_k_v)
            seq_len_q_i32 = arith.index_cast(T.i32, seq_len_q_v)
            q_row_i32_mask = arith.index_cast(T.i32, q_row)
            w_i32 = arith.constant(swa_window, type=T.i32)
            lane_div_32_i32 = arith.index_cast(T.i32, lane_div_32)
            lane_off_i32 = arith.MulIOp(
                lane_div_32_i32, arith.constant(4, type=T.i32)).result

            # Is this q_row out of bounds? In that case all mask elements = NEG_INF.
            q_oob = arith.cmpi(
                arith.CmpIPredicate.sge, q_row_i32_mask, seq_len_q_i32,
            )

            p_vals_lo = []
            p_vals_hi = []
            for r in range_constexpr(16):
                r_off_i32 = arith.constant(
                    (r % 4) + (r // 4) * 8, type=T.i32)

                # lo half: col = kv_start + lane_off + r_off
                kv_col_lo_i32 = arith.AddIOp(
                    arith.AddIOp(kv_start_i32, lane_off_i32).result,
                    r_off_i32).result
                # boundary
                is_oob_lo = arith.cmpi(
                    arith.CmpIPredicate.sge, kv_col_lo_i32, seq_len_k_i32)
                # causal: kv_col > q_row
                is_causal_lo = arith.cmpi(
                    arith.CmpIPredicate.sgt, kv_col_lo_i32, q_row_i32_mask)
                # SWA: kv_col + W <= q_row
                kv_plus_w_lo = arith.AddIOp(kv_col_lo_i32, w_i32).result
                is_swa_lo = arith.cmpi(
                    arith.CmpIPredicate.sle, kv_plus_w_lo, q_row_i32_mask)
                bad_lo = arith.OrIOp(
                    arith.OrIOp(
                        arith.OrIOp(is_causal_lo, is_swa_lo).result,
                        is_oob_lo,
                    ).result,
                    q_oob,
                ).result

                s_lo_f32 = vector.extract(
                    s_acc_lo, static_position=[r], dynamic_position=[])
                scaled_lo = arith.MulFOp(
                    s_lo_f32, c_sm_scale, fastmath=fm_fast).result
                scaled_lo_masked = arith.select(bad_lo, c_neg_inf, scaled_lo)
                diff_lo = arith.SubFOp(
                    scaled_lo_masked, lse_val, fastmath=fm_fast).result
                p_lo = math_dialect.exp(diff_lo, fastmath=fm_fast)
                p_vals_lo.append(p_lo)

                # hi half: col = lo_col + K_SUB_N
                kv_col_hi_i32 = arith.AddIOp(
                    kv_col_lo_i32,
                    arith.constant(K_SUB_N, type=T.i32)).result
                is_oob_hi = arith.cmpi(
                    arith.CmpIPredicate.sge, kv_col_hi_i32, seq_len_k_i32)
                is_causal_hi = arith.cmpi(
                    arith.CmpIPredicate.sgt, kv_col_hi_i32, q_row_i32_mask)
                kv_plus_w_hi = arith.AddIOp(kv_col_hi_i32, w_i32).result
                is_swa_hi = arith.cmpi(
                    arith.CmpIPredicate.sle, kv_plus_w_hi, q_row_i32_mask)
                bad_hi = arith.OrIOp(
                    arith.OrIOp(
                        arith.OrIOp(is_causal_hi, is_swa_hi).result,
                        is_oob_hi,
                    ).result,
                    q_oob,
                ).result

                s_hi_f32 = vector.extract(
                    s_acc_hi, static_position=[r], dynamic_position=[])
                scaled_hi = arith.MulFOp(
                    s_hi_f32, c_sm_scale, fastmath=fm_fast).result
                scaled_hi_masked = arith.select(bad_hi, c_neg_inf, scaled_hi)
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

        # ---- Final store: dQ_fp32 = dq_acc * sm_scale (NO trunc; fp32) ----
        dq_finals = [
            loop_results[dc] for dc in range_constexpr(D_CHUNKS)
        ]
        sm_scale_vec = vector.broadcast(v16f32_type, c_sm_scale)

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
                    _gep_store_f32(dq_val, dq_ptr, dq_global)
            scf.YieldOp([])

    @flyc.jit
    def launch_v4_swa_bwd_dq(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DOS: fx.Tensor,
        LSE: fx.Tensor,
        DELTAS: fx.Tensor,
        DQ: fx.Tensor,
        DSINK: fx.Tensor,
        SINK: fx.Tensor,
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

        launcher = v4_swa_bwd_dq_kernel(
            Q, K, V, DOS, LSE, DELTAS, DQ, DSINK, SINK, seq_len_q, seq_len_k,
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
            return launch_v4_swa_bwd_dq(*args, **kwargs)

    def _compile(Q, K, V, DOS, LSE, DELTAS, DQ, DSINK, SINK,
                 batch_size, seq_len_q, seq_len_k, stream=None):
        with CompilationContext.compile_hints(_fmha_compile_hints):
            return flyc.compile(
                launch_v4_swa_bwd_dq, Q, K, V, DOS, LSE, DELTAS, DQ, DSINK, SINK,
                batch_size, seq_len_q, seq_len_k, fx.Stream(stream))

    _launch.compile = _compile

    return _launch


# Convenience alias.
build_v4_swa_bwd_dq_module_primary = build_v4_swa_bwd_dq_module
