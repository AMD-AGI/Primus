# SPDX-License-Identifier: Apache-2.0
"""v4_csa_bwd_full: V4 CSA backward kernel, full output set (STEP 3b).

Emits dq + dk_local + dv_local + dgathered + dsink in one launch.
Mirrors `_v4_csa_attention_bwd_kernel` (Triton, _triton/v4_csa_attention_bwd.py)
1:1: grid=(Sq, B*HQ); each program owns one query row. dq is direct-stored;
dk_local, dv_local, dgathered, dsink are accumulated via atomic_fadd.
"""
from __future__ import annotations

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from kernels.kernels_common import dtype_to_elem_type
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf, fly as _fly, llvm as _llvm, math as math_dialect


KERNEL_NAME = "v4_csa_bwd_full_kernel"
_LLVM_GEP_DYNAMIC = -2147483648


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def build_v4_csa_bwd_full_module(
    num_heads, head_dim, swa_window,
    dtype_str="bf16", sm_scale=None, waves_per_eu=2,
    block_n=32, block_k=32,
    has_sink=True, has_sparse=True,
    unsafe_fp_math=True, fast_fp_math=True, daz=True,
    mqa_kv=True,
):
    gpu_arch = get_hip_arch()
    WARP_SIZE = 64
    BLOCK_SIZE = WARP_SIZE
    BLOCK_N = int(block_n)
    BLOCK_K = int(block_k)
    NUM_HEADS = int(num_heads)
    HEAD_DIM = int(head_dim)
    assert HEAD_DIM % WARP_SIZE == 0, f"head_dim must be divisible by {WARP_SIZE}"
    D_PER_LANE = HEAD_DIM // WARP_SIZE
    assert mqa_kv, "v4_csa_bwd_full only supports MQA (HK=1)"
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    allocator = SmemAllocator(
        None, arch=gpu_arch,
        global_sym_name=f"v4_csa_bwd_full_smem_N{BLOCK_N}_K{BLOCK_K}_S{int(has_sink)}_HS{int(has_sparse)}",
    )

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def v4_csa_bwd_full_kernel(
        Q: fx.Tensor, K_LOCAL: fx.Tensor, V_LOCAL: fx.Tensor,
        GATHERED: fx.Tensor, SPARSE_MASK: fx.Tensor,
        DOUT: fx.Tensor, LSE: fx.Tensor, DELTAS: fx.Tensor, SINK: fx.Tensor,
        DQ: fx.Tensor, DK_LOCAL: fx.Tensor, DV_LOCAL: fx.Tensor,
        DGATHERED: fx.Tensor, DSINK: fx.Tensor,
        seq_len: fx.Int32, K_topk: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        f16_ty = elem_type
        f32_ty = T.f32
        fm_fast = arith.FastMathFlags.fast

        q_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), Q)
        kl_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), K_LOCAL)
        vl_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), V_LOCAL)
        g_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), GATHERED)
        sm_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), SPARSE_MASK)
        do_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DOUT)
        dq_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), DQ)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=True)
        deltas_rsrc = buffer_ops.create_buffer_resource(DELTAS, max_size=True)
        dkl_rsrc = buffer_ops.create_buffer_resource(DK_LOCAL, max_size=True)
        dvl_rsrc = buffer_ops.create_buffer_resource(DV_LOCAL, max_size=True)
        dg_rsrc = buffer_ops.create_buffer_resource(DGATHERED, max_size=True)
        if has_sink:
            sink_rsrc = buffer_ops.create_buffer_resource(SINK, max_size=True)
            dsink_rsrc = buffer_ops.create_buffer_resource(DSINK, max_size=True)

        def _gep_load(base_ptr, elem_idx, vec_type, elem_t):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(
                _llvm_ptr_ty(), base_ptr, [idx_i64],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=elem_t, noWrapFlags=0,
            )
            return _llvm.LoadOp(vec_type, gep.result).result

        def _gep_store_f32(val, base_ptr, elem_idx):
            idx_i64 = arith.index_cast(T.i64, elem_idx)
            gep = _llvm.GEPOp(
                _llvm_ptr_ty(), base_ptr, [idx_i64],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=T.f32, noWrapFlags=0,
            )
            _llvm.StoreOp(val, gep.result)

        def load_f16_v(base_ptr, elem_idx, n):
            vt = T.vec(n, f16_ty)
            return _gep_load(base_ptr, elem_idx, vt, f16_ty)

        pid_m = arith.index_cast(T.index, gpu.block_idx.x)
        pid_bh = arith.index_cast(T.index, gpu.block_idx.y)
        tid = arith.index_cast(T.index, gpu.thread_idx.x)
        lane = tid

        seq_len_v = arith.index_cast(T.index, seq_len)
        K_topk_v = arith.index_cast(T.index, K_topk)

        bid = pid_bh // arith.index(NUM_HEADS)
        qhid = pid_bh % arith.index(NUM_HEADS)

        q_active = arith.cmpi(arith.CmpIPredicate.slt, pid_m, seq_len_v)
        pid_m_safe = arith.select(q_active, pid_m, arith.index(0))

        NEG_INF_F = -1.0e30
        c_neg_inf = arith.constant(NEG_INF_F, type=f32_ty)
        c_zero_f = arith.constant(0.0, type=f32_ty)
        c_sm_scale = arith.constant(float(sm_scale), type=f32_ty)
        width_i32 = arith.constant(WARP_SIZE, type=T.i32)
        c_four_i32 = arith.constant(4, type=T.i32)
        c_zero_i32 = arith.constant(0, type=T.i32)

        zero_f32_vec = arith.constant_vector(0.0, T.vec(D_PER_LANE, f32_ty))
        q_row_base = (
            (bid * arith.index(NUM_HEADS) + qhid) * seq_len_v + pid_m_safe
        ) * arith.index(HEAD_DIM)
        q_lane_off = q_row_base + lane * arith.index(D_PER_LANE)
        q_vec_raw = load_f16_v(q_ptr, q_lane_off, D_PER_LANE)
        q_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), q_vec_raw)
        q_f32 = arith.select(q_active, q_f32, zero_f32_vec)

        do_lane_off = q_lane_off
        do_vec_raw = load_f16_v(do_ptr, do_lane_off, D_PER_LANE)
        do_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), do_vec_raw)
        do_f32 = arith.select(q_active, do_f32, zero_f32_vec)

        lse_delta_off = (bid * arith.index(NUM_HEADS) + qhid) * seq_len_v + pid_m_safe
        lse_delta_off_i32 = arith.index_cast(T.i32, lse_delta_off)
        lse_val = buffer_ops.buffer_load(
            lse_rsrc, lse_delta_off_i32, vec_width=1, dtype=f32_ty,
        )
        delta_val = buffer_ops.buffer_load(
            deltas_rsrc, lse_delta_off_i32, vec_width=1, dtype=f32_ty,
        )

        qhid_i32 = arith.index_cast(T.i32, qhid)
        bid_i32 = arith.index_cast(T.i32, bid)
        lane_i32 = arith.index_cast(T.i32, lane)
        pid_m_safe_i32 = arith.index_cast(T.i32, pid_m_safe)
        head_dim_i32 = arith.constant(HEAD_DIM, type=T.i32)
        d_per_lane_i32 = arith.constant(D_PER_LANE, type=T.i32)
        num_heads_i32 = arith.constant(NUM_HEADS, type=T.i32)

        if has_sink:
            sink_h = buffer_ops.buffer_load(
                sink_rsrc, qhid_i32, vec_width=1, dtype=f32_ty,
            )
            sink_h = rocdl.readfirstlane(f32_ty, sink_h)
            sub_sh = arith.SubFOp(sink_h, lse_val, fastmath=fm_fast).result
            p_sink = math_dialect.exp(sub_sh, fastmath=fm_fast)
            neg_p_sink = arith.SubFOp(c_zero_f, p_sink, fastmath=fm_fast).result
            dsink_contrib = arith.MulFOp(
                neg_p_sink, delta_val, fastmath=fm_fast,
            ).result
            is_lane0 = arith.cmpi(arith.CmpIPredicate.eq, lane, arith.index(0))
            do_sink = arith.AndIOp(is_lane0, q_active).result
            _if_sink = scf.IfOp(do_sink, [], has_else=False)
            with ir.InsertionPoint(_if_sink.then_block):
                _dsink_byte_off = arith.MulIOp(qhid_i32, c_four_i32).result
                rocdl.raw_ptr_buffer_atomic_fadd(
                    dsink_contrib, dsink_rsrc, _dsink_byte_off,
                    c_zero_i32, c_zero_i32,
                )
                scf.YieldOp([])

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

        _pid_p1 = pid_m + arith.index(1)
        _le_seq = arith.cmpi(arith.CmpIPredicate.sle, _pid_p1, seq_len_v)
        n_loop_end_row = arith.select(_le_seq, _pid_p1, seq_len_v)
        SWA = arith.index(int(swa_window))
        _ge_w = arith.cmpi(arith.CmpIPredicate.sge, _pid_p1, SWA)
        _n_lo_raw = arith.select(_ge_w, _pid_p1 - SWA, arith.index(0))
        BN_idx = arith.index(BLOCK_N)
        n_loop_start = (_n_lo_raw // BN_idx) * BN_idx
        n_loop_end_blk = (
            (n_loop_end_row + BN_idx - arith.index(1)) // BN_idx
        ) * BN_idx

        _PAD = 1 if D_PER_LANE == 1 else 0
        init_local = []
        for _ in range_constexpr(D_PER_LANE):
            init_local.append(c_zero_f)
        for _ in range_constexpr(_PAD):
            init_local.append(c_zero_f)

        pid_m_i32 = arith.index_cast(T.i32, pid_m)
        seq_len_i32 = arith.index_cast(T.i32, seq_len_v)
        w_i32 = arith.constant(int(swa_window), type=T.i32)
        K_topk_i32 = arith.index_cast(T.i32, K_topk_v)

        # ==== LOCAL SWA loop ====
        for n_start, inner_args, loop_results_local in scf.for_(
            n_loop_start, n_loop_end_blk, BN_idx, iter_args=init_local,
        ):
            dq_accs = [inner_args[d] for d in range_constexpr(D_PER_LANE)]
            n_start_i32 = arith.index_cast(T.i32, n_start)

            kl_f32_cache = []
            p_cache = []
            dp_cache = []
            kv_col_i32_cache = []

            for n_off in range_constexpr(BLOCK_N):
                kv_col_i32 = arith.AddIOp(
                    n_start_i32, arith.constant(n_off, type=T.i32),
                ).result
                kv_col_i32_cache.append(kv_col_i32)
                _kv_plus_w = arith.AddIOp(kv_col_i32, w_i32).result
                is_swa = arith.cmpi(arith.CmpIPredicate.sle, _kv_plus_w, pid_m_i32)
                is_causal = arith.cmpi(arith.CmpIPredicate.sgt, kv_col_i32, pid_m_i32)
                is_oob = arith.cmpi(arith.CmpIPredicate.sge, kv_col_i32, seq_len_i32)
                bad = arith.OrIOp(
                    arith.OrIOp(is_causal, is_swa).result,
                    is_oob,
                ).result

                kv_col_idx = arith.index_cast(T.index, kv_col_i32)
                kv_col_safe = arith.select(is_oob, arith.index(0), kv_col_idx)
                kl_row_base = (bid * seq_len_v + kv_col_safe) * arith.index(HEAD_DIM)
                kl_lane_off = kl_row_base + lane * arith.index(D_PER_LANE)
                kl_vec = load_f16_v(kl_ptr, kl_lane_off, D_PER_LANE)
                kl_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), kl_vec)
                kl_f32_cache.append(kl_f32)

                vl_vec = load_f16_v(vl_ptr, kl_lane_off, D_PER_LANE)
                vl_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), vl_vec)

                lane_dot_qk = vec_dot_f32(q_f32, kl_f32)
                qk_full = warp_reduce_sum_f32(lane_dot_qk)
                qk_scaled = arith.MulFOp(qk_full, c_sm_scale, fastmath=fm_fast).result
                qk_masked = arith.select(bad, c_neg_inf, qk_scaled)
                diff_qk = arith.SubFOp(qk_masked, lse_val, fastmath=fm_fast).result
                p = math_dialect.exp(diff_qk, fastmath=fm_fast)
                p_cache.append(p)

                lane_dot_dp = vec_dot_f32(do_f32, vl_f32)
                dp_full = warp_reduce_sum_f32(lane_dot_dp)
                dp_cache.append(dp_full)

            for n_off in range_constexpr(BLOCK_N):
                p = p_cache[n_off]
                dp = dp_cache[n_off]
                diff = arith.SubFOp(dp, delta_val, fastmath=fm_fast).result
                ds = arith.MulFOp(p, diff, fastmath=fm_fast).result
                ds_scaled = arith.MulFOp(ds, c_sm_scale, fastmath=fm_fast).result
                kl_f32 = kl_f32_cache[n_off]

                # dq accumulator
                for d_off in range_constexpr(D_PER_LANE):
                    klv = vector.extract(kl_f32, static_position=[d_off], dynamic_position=[])
                    contrib = arith.MulFOp(ds_scaled, klv, fastmath=fm_fast).result
                    dq_accs[d_off] = arith.AddFOp(dq_accs[d_off], contrib, fastmath=fm_fast).result

                # dk_local / dv_local atomic_add
                kv_col_i32 = kv_col_i32_cache[n_off]
                in_range = arith.cmpi(arith.CmpIPredicate.slt, kv_col_i32, seq_len_i32)
                do_atom = arith.AndIOp(in_range, q_active).result
                _if_dkv = scf.IfOp(do_atom, [], has_else=False)
                with ir.InsertionPoint(_if_dkv.then_block):
                    _bh = arith.AddIOp(
                        arith.MulIOp(bid_i32, num_heads_i32).result, qhid_i32,
                    ).result
                    _bh_n = arith.AddIOp(
                        arith.MulIOp(_bh, seq_len_i32).result, kv_col_i32,
                    ).result
                    _row_d = arith.AddIOp(
                        arith.MulIOp(_bh_n, head_dim_i32).result,
                        arith.MulIOp(lane_i32, d_per_lane_i32).result,
                    ).result
                    for d_off in range_constexpr(D_PER_LANE):
                        elem_i32 = arith.AddIOp(_row_d, arith.constant(d_off, type=T.i32)).result
                        byte_off = arith.MulIOp(elem_i32, c_four_i32).result
                        qv = vector.extract(q_f32, static_position=[d_off], dynamic_position=[])
                        dk_val = arith.MulFOp(ds_scaled, qv, fastmath=fm_fast).result
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            dk_val, dkl_rsrc, byte_off, c_zero_i32, c_zero_i32,
                        )
                        dov = vector.extract(do_f32, static_position=[d_off], dynamic_position=[])
                        dv_val = arith.MulFOp(p, dov, fastmath=fm_fast).result
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            dv_val, dvl_rsrc, byte_off, c_zero_i32, c_zero_i32,
                        )
                    scf.YieldOp([])

            _yield = list(dq_accs)
            for _ in range_constexpr(_PAD):
                _yield.append(c_zero_f)
            yield _yield

        dq_accs = [loop_results_local[d] for d in range_constexpr(D_PER_LANE)]

        # ==== GATHERED branch ====
        if has_sparse:
            init_sparse = list(dq_accs)
            for _ in range_constexpr(_PAD):
                init_sparse.append(c_zero_f)
            for k_start, inner_args_g, loop_results_g in scf.for_(
                arith.index(0), K_topk_v, arith.index(BLOCK_K),
                iter_args=init_sparse,
            ):
                dq_accs_g = [inner_args_g[d] for d in range_constexpr(D_PER_LANE)]
                k_start_i32 = arith.index_cast(T.i32, k_start)

                g_f32_cache = []
                p_cache_g = []
                dp_cache_g = []
                k_pos_i32_cache = []

                for k_off in range_constexpr(BLOCK_K):
                    k_pos_i32 = arith.AddIOp(
                        k_start_i32, arith.constant(k_off, type=T.i32),
                    ).result
                    k_pos_i32_cache.append(k_pos_i32)
                    is_oob = arith.cmpi(arith.CmpIPredicate.sge, k_pos_i32, K_topk_i32)
                    k_pos_idx = arith.index_cast(T.index, k_pos_i32)
                    k_pos_safe = arith.select(is_oob, arith.index(0), k_pos_idx)

                    g_row_base = (
                        (bid * seq_len_v + pid_m_safe) * K_topk_v + k_pos_safe
                    ) * arith.index(HEAD_DIM)
                    g_lane_off = g_row_base + lane * arith.index(D_PER_LANE)
                    g_vec = load_f16_v(g_ptr, g_lane_off, D_PER_LANE)
                    g_f32 = arith.extf(T.vec(D_PER_LANE, f32_ty), g_vec)
                    g_f32_cache.append(g_f32)

                    sm_off = (bid * seq_len_v + pid_m_safe) * K_topk_v + k_pos_safe
                    sm_raw_v1 = _gep_load(sm_ptr, sm_off, T.vec(1, elem_type), elem_type)
                    sm_raw = vector.extract(sm_raw_v1, static_position=[0], dynamic_position=[])
                    sm_val = arith.extf(f32_ty, sm_raw)
                    sm_val = arith.select(is_oob, c_zero_f, sm_val)

                    lane_dot_qk = vec_dot_f32(q_f32, g_f32)
                    qk_full = warp_reduce_sum_f32(lane_dot_qk)
                    qk_scaled = arith.MulFOp(qk_full, c_sm_scale, fastmath=fm_fast).result
                    qk_biased = arith.AddFOp(qk_scaled, sm_val, fastmath=fm_fast).result
                    bad = arith.OrIOp(
                        is_oob,
                        arith.cmpi(arith.CmpIPredicate.sge, pid_m_i32, seq_len_i32),
                    ).result
                    qk_masked = arith.select(bad, c_neg_inf, qk_biased)
                    diff_qk = arith.SubFOp(qk_masked, lse_val, fastmath=fm_fast).result
                    p = math_dialect.exp(diff_qk, fastmath=fm_fast)
                    p_cache_g.append(p)

                    lane_dot_dp = vec_dot_f32(do_f32, g_f32)
                    dp_full = warp_reduce_sum_f32(lane_dot_dp)
                    dp_cache_g.append(dp_full)

                for k_off in range_constexpr(BLOCK_K):
                    p = p_cache_g[k_off]
                    dp = dp_cache_g[k_off]
                    diff = arith.SubFOp(dp, delta_val, fastmath=fm_fast).result
                    ds = arith.MulFOp(p, diff, fastmath=fm_fast).result
                    ds_scaled = arith.MulFOp(ds, c_sm_scale, fastmath=fm_fast).result
                    g_f32 = g_f32_cache[k_off]

                    for d_off in range_constexpr(D_PER_LANE):
                        gv = vector.extract(g_f32, static_position=[d_off], dynamic_position=[])
                        contrib = arith.MulFOp(ds_scaled, gv, fastmath=fm_fast).result
                        dq_accs_g[d_off] = arith.AddFOp(dq_accs_g[d_off], contrib, fastmath=fm_fast).result

                    k_pos_i32 = k_pos_i32_cache[k_off]
                    in_range_k = arith.cmpi(arith.CmpIPredicate.slt, k_pos_i32, K_topk_i32)
                    do_atom_k = arith.AndIOp(in_range_k, q_active).result
                    _if_dg = scf.IfOp(do_atom_k, [], has_else=False)
                    with ir.InsertionPoint(_if_dg.then_block):
                        _bm = arith.AddIOp(
                            arith.MulIOp(bid_i32, seq_len_i32).result, pid_m_safe_i32,
                        ).result
                        _bm_k = arith.AddIOp(
                            arith.MulIOp(_bm, K_topk_i32).result, k_pos_i32,
                        ).result
                        _row_d = arith.AddIOp(
                            arith.MulIOp(_bm_k, head_dim_i32).result,
                            arith.MulIOp(lane_i32, d_per_lane_i32).result,
                        ).result
                        for d_off in range_constexpr(D_PER_LANE):
                            elem_i32 = arith.AddIOp(_row_d, arith.constant(d_off, type=T.i32)).result
                            byte_off = arith.MulIOp(elem_i32, c_four_i32).result
                            qv = vector.extract(q_f32, static_position=[d_off], dynamic_position=[])
                            dov = vector.extract(do_f32, static_position=[d_off], dynamic_position=[])
                            t1 = arith.MulFOp(ds_scaled, qv, fastmath=fm_fast).result
                            t2 = arith.MulFOp(p, dov, fastmath=fm_fast).result
                            dg_val = arith.AddFOp(t1, t2, fastmath=fm_fast).result
                            rocdl.raw_ptr_buffer_atomic_fadd(
                                dg_val, dg_rsrc, byte_off, c_zero_i32, c_zero_i32,
                            )
                        scf.YieldOp([])

                _yield_g = list(dq_accs_g)
                for _ in range_constexpr(_PAD):
                    _yield_g.append(c_zero_f)
                yield _yield_g

            dq_accs = [loop_results_g[d] for d in range_constexpr(D_PER_LANE)]

        # ==== Store dq direct ====
        _o_guard = scf.IfOp(q_active, [], has_else=False)
        with ir.InsertionPoint(_o_guard.then_block):
            dq_row_base = (
                (bid * arith.index(NUM_HEADS) + qhid) * seq_len_v + pid_m_safe
            ) * arith.index(HEAD_DIM)
            dq_lane_off = dq_row_base + lane * arith.index(D_PER_LANE)
            for d_off in range_constexpr(D_PER_LANE):
                elem_off = dq_lane_off + arith.index(d_off)
                _gep_store_f32(dq_accs[d_off], dq_ptr, elem_off)
            scf.YieldOp([])

    @flyc.jit
    def launch_v4_csa_bwd_full(
        Q: fx.Tensor, K_LOCAL: fx.Tensor, V_LOCAL: fx.Tensor,
        GATHERED: fx.Tensor, SPARSE_MASK: fx.Tensor,
        DOUT: fx.Tensor, LSE: fx.Tensor, DELTAS: fx.Tensor, SINK: fx.Tensor,
        DQ: fx.Tensor, DK_LOCAL: fx.Tensor, DV_LOCAL: fx.Tensor,
        DGATHERED: fx.Tensor, DSINK: fx.Tensor,
        batch_size: fx.Int32, seq_len: fx.Int32, K_topk: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = arith.index_cast(T.index, batch_size)
        sl_idx = arith.index_cast(T.index, seq_len)
        grid_x = sl_idx
        grid_y = bs_idx * arith.index(NUM_HEADS)

        launcher = v4_csa_bwd_full_kernel(
            Q, K_LOCAL, V_LOCAL, GATHERED, SPARSE_MASK,
            DOUT, LSE, DELTAS, SINK,
            DQ, DK_LOCAL, DV_LOCAL, DGATHERED, DSINK,
            seq_len, K_topk,
        )

        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, _wpe)

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

    compile_hints = {"fast_fp_math": fast_fp_math, "unsafe_fp_math": unsafe_fp_math}

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(compile_hints):
            return launch_v4_csa_bwd_full(*args, **kwargs)

    def _compile(
        Q, K_LOCAL, V_LOCAL, GATHERED, SPARSE_MASK,
        DOUT, LSE, DELTAS, SINK,
        DQ, DK_LOCAL, DV_LOCAL, DGATHERED, DSINK,
        batch_size, seq_len, K_topk, stream=None,
    ):
        with CompilationContext.compile_hints(compile_hints):
            return flyc.compile(
                launch_v4_csa_bwd_full,
                Q, K_LOCAL, V_LOCAL, GATHERED, SPARSE_MASK,
                DOUT, LSE, DELTAS, SINK,
                DQ, DK_LOCAL, DV_LOCAL, DGATHERED, DSINK,
                batch_size, seq_len, K_topk, fx.Stream(stream),
            )

    _launch.compile = _compile
    return _launch


build_v4_csa_bwd_full_module_primary = build_v4_csa_bwd_full_module
