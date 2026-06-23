# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""v4_csa_fwd: V4 CSA forward (FlyDSL per-row design).

Round-3 Step 2b: ports the Triton monolithic CSA forward kernel to FlyDSL
in a 1:1 per-row design. The Triton CSA monolithic kernel uses one program
per (b, h, m), with per-key scalar dot-product. That structure ports cleanly:

  grid = (Sq, B * HQ)
  BLOCK_SIZE = 64 (one wave). Lane in 0..63 handles partial D=8.
  Online softmax accumulator (m, l, acc) lives in fp32 in-register and is
    distributed across lanes (each lane owns D/64 = 8 elements of the
    accumulator's D dimension).

This design does NOT use MFMA -- the per-row scalar dot-product matches
Triton's monolithic kernel exactly (Triton CSA monolithic also uses
``tl.sum(k * q, axis=1)`` not ``tl.dot``).

Layout: BHLD (Q/K_local/V_local/O all [B, H, Sq, D]).
  - Gathered: [B, Sq, K_topk, D] (no H dim -- shared across heads).
  - sparse_mask: [B, Sq, K_topk] (no H dim -- broadcasts over H).
  - sink: [H] fp32 or None.
  - LSE: [B, H, Sq] fp32 (raw-domain: m_final + ln(l_final), since m_final
    already includes sm_scale via the online softmax in raw-qk*sm_scale).

Forward computes the joint online softmax over local_SWA (block-causal,
window=swa_window, keys are k_local) + sparse (keys are gathered) + sink.

K_topk == 0 supported (kernel skips the sparse loop).
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


KERNEL_NAME = "v4_csa_fwd_kernel"

_LOG2E = math.log2(math.e)

_LLVM_GEP_DYNAMIC = -2147483648


def _waitcnt_lgkm_0():
    """s_waitcnt lgkmcnt(0): drain LDS/SMEM ops. vmcnt=63, expcnt=7."""
    val = 0xF | (0x7 << 4) | (0 << 8) | (0x3 << 14)
    rocdl.s_waitcnt(val)


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def build_v4_csa_fwd_module(
    num_heads,
    head_dim,
    swa_window,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    block_n=32,
    block_k=32,
    has_sink=False,
    has_sparse=True,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    mqa_kv=False,
    head_group=1,
):
    """Build the V4 CSA forward per-row launcher.

    Parameters:
      num_heads: int  -- H_Q
      head_dim: int   -- D (must be divisible by 64)
      swa_window: int -- SWA window (> 0)
      block_n: int    -- local-branch tile width (default 32)
      block_k: int    -- sparse-branch tile width (default 32)
      has_sink: bool  -- include sink epilogue
      has_sparse: bool -- include sparse branch loop (K_topk > 0)
    """
    gpu_arch = get_hip_arch()
    WARP_SIZE = 64
    BLOCK_SIZE = WARP_SIZE  # one wave per program
    BLOCK_N = int(block_n)
    BLOCK_K = int(block_k)
    NUM_HEADS = int(num_heads)
    HEAD_DIM = int(head_dim)
    assert HEAD_DIM % WARP_SIZE == 0, f"head_dim must be divisible by {WARP_SIZE}"
    D_PER_LANE = HEAD_DIM // WARP_SIZE  # 8 for D=512
    HEAD_GROUP = int(head_group)
    assert NUM_HEADS % HEAD_GROUP == 0, f"num_heads {NUM_HEADS} must be divisible by head_group {HEAD_GROUP}"
    NUM_HEAD_GROUPS = NUM_HEADS // HEAD_GROUP
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # ---- LDS cache for sparse-branch gathered K-tile ----
    ENABLE_LDS_CACHE = bool(has_sparse) and (os.environ.get("PRIMUS_V4_CSA_LDS_CACHE", "0") == "1")
    LDS_GATHER_TILE_ELEMS = BLOCK_K * HEAD_DIM
    LDS_GATHER_TILE_BYTES = LDS_GATHER_TILE_ELEMS * 2

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name=f"v4_csa_fwd_smem_N{BLOCK_N}_K{BLOCK_K}_C{int(ENABLE_LDS_CACHE)}_HG{HEAD_GROUP}",
    )
    if ENABLE_LDS_CACHE:
        lds_gather_offset = allocator._align(allocator.ptr, 16)
        allocator.ptr = lds_gather_offset + LDS_GATHER_TILE_BYTES
    else:
        lds_gather_offset = 0

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def v4_csa_fwd_kernel(
        Q: fx.Tensor,
        K_LOCAL: fx.Tensor,
        V_LOCAL: fx.Tensor,
        GATHERED: fx.Tensor,
        SPARSE_MASK: fx.Tensor,
        Sink: fx.Tensor,
        O: fx.Tensor,
        LSE: fx.Tensor,
        seq_len: fx.Int32,
        K_topk: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32
        fm_fast = arith.FastMathFlags.fast

        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        kl_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), K_LOCAL)
        vl_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), V_LOCAL)
        g_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), GATHERED)
        sm_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), SPARSE_MASK)
        o_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), O)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        if has_sink:
            sink_rsrc = buffer_ops.create_buffer_resource(Sink, max_size=True)

        f16_ty = elem_type
        f32_ty = T.f32

        # ---- Helpers ----
        def _gep_load_scalar(base_ptr, elem_idx, vec_type, elem_t):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(_llvm_ptr_ty(), base_ptr, [idx_i64],
                              rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                              elem_type=elem_t,
                              noWrapFlags=0)
            return _llvm.LoadOp(vec_type, gep.result).result

        def load_f16_v(base_ptr, elem_idx, n):
            vt = T.vec(n, f16_ty)
            return _gep_load_scalar(base_ptr, elem_idx, vt, f16_ty)

        def load_f32_scalar(base_ptr, elem_idx):
            return _gep_load_scalar(base_ptr, elem_idx, f32_ty, f32_ty)

        # ---- Thread / program ----
        pid_m = arith.index_cast(T.index, gpu.block_idx.x)
        pid_bh = arith.index_cast(T.index, gpu.block_idx.y)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)
        lane = tid

        seq_len_v = arith.index_cast(T.index, seq_len)
        K_topk_v = arith.index_cast(T.index, K_topk)

        bid = pid_bh // arith.index(NUM_HEAD_GROUPS)
        qhid_group = pid_bh % arith.index(NUM_HEAD_GROUPS)
        # qhid_base is the head index of the first head in this program's head group.
        qhid_base = qhid_group * arith.index(HEAD_GROUP)

        # ---- LDS view for sparse-branch gathered cache ----
        if ENABLE_LDS_CACHE:
            base_ptr = allocator.get_base()
            lds_gather = SmemPtr(
                base_ptr,
                lds_gather_offset,
                elem_type,
                shape=(LDS_GATHER_TILE_ELEMS,),
            ).get()

        # ---- Q row in-bounds guard ----
        q_active = arith.cmpi(arith.CmpIPredicate.slt, pid_m, seq_len_v)
        pid_m_safe = arith.select(q_active, pid_m, arith.index(0))

        # ---- Load Q for all HEAD_GROUP heads in this program ----
        zero_f32_vec = arith.constant_vector(0.0, T.vec(D_PER_LANE, f32_ty))
        q_f32_vecs = []
        for h_off in range_constexpr(HEAD_GROUP):
            qhid_h = qhid_base + arith.index(h_off)
            q_row_base = ((bid * arith.index(NUM_HEADS) + qhid_h) * seq_len_v + pid_m_safe) * arith.index(HEAD_DIM)
            q_lane_off = q_row_base + lane * arith.index(D_PER_LANE)
            q_vec = load_f16_v(q_ptr, q_lane_off, D_PER_LANE)
            q_f32_vec = arith.extf(T.vec(D_PER_LANE, f32_ty), q_vec)
            q_f32_vec = arith.select(q_active, q_f32_vec, zero_f32_vec)
            q_f32_vecs.append(q_f32_vec)

        # ---- Constants ----
        NEG_INF_F = -1.0e30
        c_neg_inf = arith.constant(NEG_INF_F, type=f32_ty)
        c_zero_f = arith.constant(0.0, type=f32_ty)
        c_one_f = arith.constant(1.0, type=f32_ty)
        c_sm_scale_f = arith.constant(float(sm_scale), type=f32_ty)
        c_log2e_f = arith.constant(_LOG2E, type=f32_ty)

        lane_i32 = arith.index_cast(T.i32, lane)
        width_i32 = arith.constant(WARP_SIZE, type=T.i32)

        def warp_reduce_sum_f32(v):
            cur = v
            for off in [32, 16, 8, 4, 2, 1]:
                xor_amt = arith.constant(off, type=T.i32)
                peer = arith.ArithValue(cur).shuffle_xor(xor_amt, width_i32)
                cur = arith.AddFOp(cur, peer, fastmath=fm_fast).result
            return cur

        def vec_dot_f32(a_vec, b_vec):
            s = c_zero_f
            for i in range_constexpr(D_PER_LANE):
                av = vector.extract(a_vec, static_position=[i], dynamic_position=[])
                bv = vector.extract(b_vec, static_position=[i], dynamic_position=[])
                p = arith.MulFOp(av, bv, fastmath=fm_fast).result
                s = arith.AddFOp(s, p, fastmath=fm_fast).result
            return s

        # ---- Local SWA loop bounds ----
        _pid_p1 = pid_m + arith.index(1)
        _le_seq = arith.cmpi(arith.CmpIPredicate.sle, _pid_p1, seq_len_v)
        n_loop_end = arith.select(_le_seq, _pid_p1, seq_len_v)
        SWA = arith.index(int(swa_window))
        _ge_w = arith.cmpi(arith.CmpIPredicate.sge, _pid_p1, SWA)
        _n_lo_raw = arith.select(_ge_w, _pid_p1 - SWA, arith.index(0))
        BN_idx = arith.index(BLOCK_N)
        n_loop_start = (_n_lo_raw // BN_idx) * BN_idx

        # Init state: HEAD_GROUP copies of (m_i, l_i, acc[D_PER_LANE]).
        # Layout: [m_0, l_0, acc_0_0..acc_0_{D-1}, m_1, l_1, acc_1_0..]
        STATE_PER_HEAD = 2 + D_PER_LANE
        init_args = []
        for _h in range_constexpr(HEAD_GROUP):
            init_args.append(c_neg_inf)  # m
            init_args.append(c_zero_f)   # l
            for _ in range_constexpr(D_PER_LANE):
                init_args.append(c_zero_f)

        # ==== LOCAL SWA loop ====
        for n_start, inner_args, loop_results_local in scf.for_(
            n_loop_start,
            n_loop_end,
            BN_idx,
            iter_args=init_args,
        ):
            # Unpack HEAD_GROUP states from inner_args
            m_is = [inner_args[h * STATE_PER_HEAD] for h in range_constexpr(HEAD_GROUP)]
            l_is = [inner_args[h * STATE_PER_HEAD + 1] for h in range_constexpr(HEAD_GROUP)]
            accs = [[inner_args[h * STATE_PER_HEAD + 2 + d] for d in range_constexpr(D_PER_LANE)] for h in range_constexpr(HEAD_GROUP)]

            n_start_i32 = arith.index_cast(T.i32, n_start)
            pid_m_i32 = arith.index_cast(T.i32, pid_m)
            seq_len_i32 = seq_len
            if hasattr(seq_len_i32, "ir_value"):
                seq_len_i32 = seq_len_i32.ir_value()
            w_i32 = arith.constant(int(swa_window), type=T.i32)

            # Per-head QK values: [HEAD_GROUP][BLOCK_N]
            qk_vals_per_head = [[] for _ in range_constexpr(HEAD_GROUP)]
            kl_f32_cache = []
            bad_lo_cache = []
            for n_off in range_constexpr(BLOCK_N):
                kv_col_i32 = arith.AddIOp(
                    n_start_i32,
                    arith.constant(n_off, type=T.i32),
                ).result
                _kv_plus_w_lo = arith.AddIOp(kv_col_i32, w_i32).result
                is_swa_lo = arith.cmpi(
                    arith.CmpIPredicate.sle, _kv_plus_w_lo, pid_m_i32)
                is_causal_lo = arith.cmpi(
                    arith.CmpIPredicate.sgt, kv_col_i32, pid_m_i32)
                is_oob = arith.cmpi(
                    arith.CmpIPredicate.sge, kv_col_i32, seq_len_i32)
                bad_lo = arith.OrIOp(
                    arith.OrIOp(is_causal_lo, is_swa_lo).result,
                    is_oob).result
                bad_lo_cache.append(bad_lo)

                kv_col_idx = arith.index_cast(T.index, kv_col_i32)
                kv_col_safe = arith.select(is_oob, arith.index(0), kv_col_idx)
                # When mqa_kv, K is shared across heads -> load once.
                # When mqa_kv=False, K differs per head, so we'd need per-head loads.
                # For now this kernel requires mqa_kv when head_group > 1.
                if mqa_kv:
                    kl_row_base = (bid * seq_len_v + kv_col_safe) * arith.index(HEAD_DIM)
                else:
                    # head_group > 1 with non-MQA: would need per-head load. Disallowed at setup.
                    kl_row_base = ((bid * arith.index(NUM_HEADS) + qhid_base) * seq_len_v + kv_col_safe) * arith.index(HEAD_DIM)
                kl_lane_off = kl_row_base + lane * arith.index(D_PER_LANE)
                kl_vec = load_f16_v(kl_ptr, kl_lane_off, D_PER_LANE)
                kl_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), kl_vec)
                kl_f32_cache.append(kl_f32)
                # Compute qk for each head using the shared K.
                for h_off in range_constexpr(HEAD_GROUP):
                    lane_dot = vec_dot_f32(kl_f32, q_f32_vecs[h_off])
                    qk_full = warp_reduce_sum_f32(lane_dot)
                    qk_scaled = arith.MulFOp(qk_full, c_sm_scale_f, fastmath=fm_fast).result
                    qk_masked = arith.select(bad_lo, c_neg_inf, qk_scaled)
                    qk_vals_per_head[h_off].append(qk_masked)

            # Per-head softmax update
            new_m_is = []
            new_l_is = []
            new_accs = []
            p_vals_per_head = []
            for h_off in range_constexpr(HEAD_GROUP):
                qk_vals_h = qk_vals_per_head[h_off]
                m_tile = qk_vals_h[0]
                for n_off in range_constexpr(BLOCK_N - 1):
                    m_tile = arith.MaxNumFOp(m_tile, qk_vals_h[n_off + 1], fastmath=fm_fast).result
                m_new = arith.MaxNumFOp(m_is[h_off], m_tile, fastmath=fm_fast).result

                diff_m = arith.SubFOp(m_is[h_off], m_new, fastmath=fm_fast).result
                diff_m_log2 = arith.MulFOp(diff_m, c_log2e_f, fastmath=fm_fast).result
                alpha = arith.ArithValue(diff_m_log2).exp2(fastmath=fm_fast)

                p_vals = []
                tile_sum = c_zero_f
                for n_off in range_constexpr(BLOCK_N):
                    d = arith.SubFOp(qk_vals_h[n_off], m_new, fastmath=fm_fast).result
                    dl = arith.MulFOp(d, c_log2e_f, fastmath=fm_fast).result
                    p = arith.ArithValue(dl).exp2(fastmath=fm_fast)
                    p_vals.append(p)
                    tile_sum = arith.AddFOp(tile_sum, p, fastmath=fm_fast).result
                p_vals_per_head.append(p_vals)

                l_alpha = arith.MulFOp(l_is[h_off], alpha, fastmath=fm_fast).result
                l_new = arith.AddFOp(l_alpha, tile_sum, fastmath=fm_fast).result

                acc_h = accs[h_off]
                new_acc_h = []
                for d_off in range_constexpr(D_PER_LANE):
                    new_acc_h.append(arith.MulFOp(acc_h[d_off], alpha, fastmath=fm_fast).result)
                new_m_is.append(m_new)
                new_l_is.append(l_new)
                new_accs.append(new_acc_h)

            # AV phase: load V once (MQA), reuse across HEAD_GROUP heads.
            for n_off in range_constexpr(BLOCK_N):
                kv_col_i32 = arith.AddIOp(
                    n_start_i32,
                    arith.constant(n_off, type=T.i32),
                ).result
                is_oob = arith.cmpi(
                    arith.CmpIPredicate.sge, kv_col_i32, seq_len_i32)
                kv_col_idx = arith.index_cast(T.index, kv_col_i32)
                kv_col_safe = arith.select(is_oob, arith.index(0), kv_col_idx)
                if mqa_kv:
                    vl_row_base = (bid * seq_len_v + kv_col_safe) * arith.index(HEAD_DIM)
                else:
                    vl_row_base = ((bid * arith.index(NUM_HEADS) + qhid_base) * seq_len_v + kv_col_safe) * arith.index(HEAD_DIM)
                vl_lane_off = vl_row_base + lane * arith.index(D_PER_LANE)
                vl_vec = load_f16_v(vl_ptr, vl_lane_off, D_PER_LANE)
                vl_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), vl_vec)
                for h_off in range_constexpr(HEAD_GROUP):
                    p_vals_h = p_vals_per_head[h_off]
                    new_acc_h = new_accs[h_off]
                    for d_off in range_constexpr(D_PER_LANE):
                        vv = vector.extract(vl_f32, static_position=[d_off], dynamic_position=[])
                        contrib = arith.MulFOp(p_vals_h[n_off], vv, fastmath=fm_fast).result
                        new_acc_h[d_off] = arith.AddFOp(new_acc_h[d_off], contrib, fastmath=fm_fast).result

            # Pack yield args
            yield_args = []
            for h_off in range_constexpr(HEAD_GROUP):
                yield_args.append(new_m_is[h_off])
                yield_args.append(new_l_is[h_off])
                for d in range_constexpr(D_PER_LANE):
                    yield_args.append(new_accs[h_off][d])
            yield yield_args

        # Unpack HEAD_GROUP states from local SWA results
        m_is = [loop_results_local[h * STATE_PER_HEAD] for h in range_constexpr(HEAD_GROUP)]
        l_is = [loop_results_local[h * STATE_PER_HEAD + 1] for h in range_constexpr(HEAD_GROUP)]
        accs = [[loop_results_local[h * STATE_PER_HEAD + 2 + d] for d in range_constexpr(D_PER_LANE)] for h in range_constexpr(HEAD_GROUP)]

        # ==== SPARSE branch ====
        if has_sparse:
            init_sparse = []
            for h_off in range_constexpr(HEAD_GROUP):
                init_sparse.append(m_is[h_off])
                init_sparse.append(l_is[h_off])
                for d in range_constexpr(D_PER_LANE):
                    init_sparse.append(accs[h_off][d])
            for k_start, inner_args, loop_results_sparse in scf.for_(
                arith.index(0),
                K_topk_v,
                arith.index(BLOCK_K),
                iter_args=init_sparse,
            ):
                m_is = [inner_args[h * STATE_PER_HEAD] for h in range_constexpr(HEAD_GROUP)]
                l_is = [inner_args[h * STATE_PER_HEAD + 1] for h in range_constexpr(HEAD_GROUP)]
                accs = [[inner_args[h * STATE_PER_HEAD + 2 + d] for d in range_constexpr(D_PER_LANE)] for h in range_constexpr(HEAD_GROUP)]

                k_start_i32 = arith.index_cast(T.i32, k_start)
                K_topk_i32 = K_topk
                if hasattr(K_topk_i32, "ir_value"):
                    K_topk_i32 = K_topk_i32.ir_value()

                qk_vals_sparse_per_head = [[] for _ in range_constexpr(HEAD_GROUP)]
                for k_off in range_constexpr(BLOCK_K):
                    k_pos_i32 = arith.AddIOp(
                        k_start_i32,
                        arith.constant(k_off, type=T.i32),
                    ).result
                    is_oob = arith.cmpi(
                        arith.CmpIPredicate.sge, k_pos_i32, K_topk_i32)
                    k_pos_idx = arith.index_cast(T.index, k_pos_i32)
                    k_pos_safe = arith.select(is_oob, arith.index(0), k_pos_idx)

                    g_row_base = (
                        (bid * seq_len_v + pid_m_safe) * K_topk_v + k_pos_safe
                    ) * arith.index(HEAD_DIM)
                    g_lane_off = g_row_base + lane * arith.index(D_PER_LANE)
                    g_vec = load_f16_v(g_ptr, g_lane_off, D_PER_LANE)
                    if ENABLE_LDS_CACHE:
                        lds_idx = arith.index(k_off * HEAD_DIM) + lane * arith.index(D_PER_LANE)
                        vector.store(g_vec, lds_gather, [lds_idx])
                    g_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), g_vec)

                    sm_off = (bid * seq_len_v + pid_m_safe) * K_topk_v + k_pos_safe
                    sm_val = load_f32_scalar(sm_ptr, sm_off)
                    # For each head, compute QK using shared g_f32.
                    for h_off in range_constexpr(HEAD_GROUP):
                        lane_dot = vec_dot_f32(g_f32, q_f32_vecs[h_off])
                        qk_full = warp_reduce_sum_f32(lane_dot)
                        qk_scaled = arith.MulFOp(qk_full, c_sm_scale_f, fastmath=fm_fast).result
                        qk_biased = arith.AddFOp(qk_scaled, sm_val, fastmath=fm_fast).result
                        qk_masked = arith.select(is_oob, c_neg_inf, qk_biased)
                        qk_vals_sparse_per_head[h_off].append(qk_masked)

                new_m_is = []
                new_l_is = []
                new_accs = []
                p_vals_per_head = []
                for h_off in range_constexpr(HEAD_GROUP):
                    qk_vals_h = qk_vals_sparse_per_head[h_off]
                    m_tile = qk_vals_h[0]
                    for k_off in range_constexpr(BLOCK_K - 1):
                        m_tile = arith.MaxNumFOp(m_tile, qk_vals_h[k_off + 1], fastmath=fm_fast).result
                    m_new = arith.MaxNumFOp(m_is[h_off], m_tile, fastmath=fm_fast).result

                    diff_m = arith.SubFOp(m_is[h_off], m_new, fastmath=fm_fast).result
                    diff_m_log2 = arith.MulFOp(diff_m, c_log2e_f, fastmath=fm_fast).result
                    alpha = arith.ArithValue(diff_m_log2).exp2(fastmath=fm_fast)

                    p_vals = []
                    tile_sum = c_zero_f
                    for k_off in range_constexpr(BLOCK_K):
                        d = arith.SubFOp(qk_vals_h[k_off], m_new, fastmath=fm_fast).result
                        dl = arith.MulFOp(d, c_log2e_f, fastmath=fm_fast).result
                        p = arith.ArithValue(dl).exp2(fastmath=fm_fast)
                        p_vals.append(p)
                        tile_sum = arith.AddFOp(tile_sum, p, fastmath=fm_fast).result
                    p_vals_per_head.append(p_vals)

                    l_alpha = arith.MulFOp(l_is[h_off], alpha, fastmath=fm_fast).result
                    l_new = arith.AddFOp(l_alpha, tile_sum, fastmath=fm_fast).result

                    acc_h = accs[h_off]
                    new_acc_h = []
                    for d_off in range_constexpr(D_PER_LANE):
                        new_acc_h.append(arith.MulFOp(acc_h[d_off], alpha, fastmath=fm_fast).result)
                    new_m_is.append(m_new)
                    new_l_is.append(l_new)
                    new_accs.append(new_acc_h)

                # ---- AV phase: re-read gathered K-block once per k_off, reuse for all heads ----
                if ENABLE_LDS_CACHE:
                    _waitcnt_lgkm_0()
                for k_off in range_constexpr(BLOCK_K):
                    if ENABLE_LDS_CACHE:
                        lds_idx = arith.index(k_off * HEAD_DIM) + lane * arith.index(D_PER_LANE)
                        g_vec = vector.load(T.vec(D_PER_LANE, f16_ty), lds_gather, [lds_idx])
                    else:
                        k_pos_i32 = arith.AddIOp(
                            k_start_i32,
                            arith.constant(k_off, type=T.i32),
                        ).result
                        is_oob = arith.cmpi(
                            arith.CmpIPredicate.sge, k_pos_i32, K_topk_i32)
                        k_pos_idx = arith.index_cast(T.index, k_pos_i32)
                        k_pos_safe = arith.select(is_oob, arith.index(0), k_pos_idx)
                        g_row_base = (
                            (bid * seq_len_v + pid_m_safe) * K_topk_v + k_pos_safe
                        ) * arith.index(HEAD_DIM)
                        g_lane_off = g_row_base + lane * arith.index(D_PER_LANE)
                        g_vec = load_f16_v(g_ptr, g_lane_off, D_PER_LANE)
                    g_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), g_vec)
                    for h_off in range_constexpr(HEAD_GROUP):
                        p_vals_h = p_vals_per_head[h_off]
                        new_acc_h = new_accs[h_off]
                        for d_off in range_constexpr(D_PER_LANE):
                            vv = vector.extract(g_f32, static_position=[d_off], dynamic_position=[])
                            contrib = arith.MulFOp(p_vals_h[k_off], vv, fastmath=fm_fast).result
                            new_acc_h[d_off] = arith.AddFOp(new_acc_h[d_off], contrib, fastmath=fm_fast).result

                # Pack yield args
                yield_args = []
                for h_off in range_constexpr(HEAD_GROUP):
                    yield_args.append(new_m_is[h_off])
                    yield_args.append(new_l_is[h_off])
                    for d in range_constexpr(D_PER_LANE):
                        yield_args.append(new_accs[h_off][d])
                yield yield_args

            m_is = [loop_results_sparse[h * STATE_PER_HEAD] for h in range_constexpr(HEAD_GROUP)]
            l_is = [loop_results_sparse[h * STATE_PER_HEAD + 1] for h in range_constexpr(HEAD_GROUP)]
            accs = [[loop_results_sparse[h * STATE_PER_HEAD + 2 + d] for d in range_constexpr(D_PER_LANE)] for h in range_constexpr(HEAD_GROUP)]

        # ==== Sink epilogue (per-head) ====
        if has_sink:
            for h_off in range_constexpr(HEAD_GROUP):
                qhid_h = qhid_base + arith.index(h_off)
                qhid_i32 = arith.index_cast(T.i32, qhid_h)
                sink_h_val = buffer_ops.buffer_load(
                    sink_rsrc, qhid_i32, vec_width=1, dtype=f32_ty,
                )
                m_i_h = m_is[h_off]
                l_i_h = l_is[h_off]
                acc_h = accs[h_off]
                m_new = arith.MaxNumFOp(m_i_h, sink_h_val, fastmath=fm_fast).result
                d_alpha = arith.SubFOp(m_i_h, m_new, fastmath=fm_fast).result
                d_alpha_log2 = arith.MulFOp(d_alpha, c_log2e_f, fastmath=fm_fast).result
                alpha_sink = arith.ArithValue(d_alpha_log2).exp2(fastmath=fm_fast)
                d_beta = arith.SubFOp(sink_h_val, m_new, fastmath=fm_fast).result
                d_beta_log2 = arith.MulFOp(d_beta, c_log2e_f, fastmath=fm_fast).result
                beta_sink = arith.ArithValue(d_beta_log2).exp2(fastmath=fm_fast)
                l_alpha = arith.MulFOp(l_i_h, alpha_sink, fastmath=fm_fast).result
                l_is[h_off] = arith.AddFOp(l_alpha, beta_sink, fastmath=fm_fast).result
                new_acc_h = []
                for d_off in range_constexpr(D_PER_LANE):
                    new_acc_h.append(arith.MulFOp(acc_h[d_off], alpha_sink, fastmath=fm_fast).result)
                accs[h_off] = new_acc_h
                m_is[h_off] = m_new

        # ==== Final divide and store (per-head) ====
        _o_guard = scf.IfOp(q_active, [], has_else=False)
        with ir.InsertionPoint(_o_guard.then_block):
            for h_off in range_constexpr(HEAD_GROUP):
                qhid_h = qhid_base + arith.index(h_off)
                l_i_h = l_is[h_off]
                m_i_h = m_is[h_off]
                acc_h = accs[h_off]
                inv_l = arith.DivFOp(c_one_f, l_i_h, fastmath=fm_fast).result
                o_row_base = ((bid * arith.index(NUM_HEADS) + qhid_h) * seq_len_v + pid_m_safe) * arith.index(HEAD_DIM)
                o_lane_off = o_row_base + lane * arith.index(D_PER_LANE)
                for d_off in range_constexpr(D_PER_LANE):
                    o_f32 = arith.MulFOp(acc_h[d_off], inv_l, fastmath=fm_fast).result
                    o_f16 = arith.trunc_f(elem_type, o_f32)
                    elem_off = o_lane_off + arith.index(d_off)
                    idx_i64 = arith.index_cast(T.i64, elem_off)
                    gep = _llvm.GEPOp(_llvm_ptr_ty(), o_ptr, [idx_i64],
                                      rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                                      elem_type=elem_type,
                                      noWrapFlags=0)
                    _llvm.StoreOp(o_f16, gep.result)

                is_lane0 = arith.cmpi(arith.CmpIPredicate.eq, lane, arith.index(0))
                _lse_if = scf.IfOp(is_lane0, [], has_else=False)
                with ir.InsertionPoint(_lse_if.then_block):
                    ln_l = math_dialect.log(l_i_h, fastmath=fm_fast)
                    lse_val = arith.AddFOp(m_i_h, ln_l, fastmath=fm_fast).result
                    lse_off = (bid * arith.index(NUM_HEADS) + qhid_h) * seq_len_v + pid_m_safe
                    lse_off_i32 = arith.index_cast(T.i32, lse_off)
                    buffer_ops.buffer_store(lse_val, lse_rsrc, lse_off_i32)
                    scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_v4_csa_fwd(
        Q: fx.Tensor,
        K_LOCAL: fx.Tensor,
        V_LOCAL: fx.Tensor,
        GATHERED: fx.Tensor,
        SPARSE_MASK: fx.Tensor,
        Sink: fx.Tensor,
        O: fx.Tensor,
        LSE: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        K_topk: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = arith.index_cast(T.index, batch_size)
        sl_idx = arith.index_cast(T.index, seq_len)
        grid_x = sl_idx
        grid_y = bs_idx * arith.index(NUM_HEAD_GROUPS)

        launcher = v4_csa_fwd_kernel(
            Q, K_LOCAL, V_LOCAL, GATHERED, SPARSE_MASK, Sink, O, LSE,
            seq_len, K_topk,
        )

        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32, _wpe)

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
            grid=(grid_x, grid_y, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(compile_hints):
            return launch_v4_csa_fwd(*args, **kwargs)

    def _compile(Q, K_LOCAL, V_LOCAL, GATHERED, SPARSE_MASK, Sink, O, LSE, batch_size, seq_len, K_topk, stream=None):
        with CompilationContext.compile_hints(compile_hints):
            return flyc.compile(
                launch_v4_csa_fwd, Q, K_LOCAL, V_LOCAL, GATHERED, SPARSE_MASK, Sink, O, LSE,
                batch_size, seq_len, K_topk, fx.Stream(stream))

    _launch.compile = _compile

    return _launch


build_v4_csa_fwd_module_primary = build_v4_csa_fwd_module
