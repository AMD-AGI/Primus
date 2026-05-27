###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""V4 CSA attention backward Triton kernel (plan-4 P26, ``compress_ratio == 4``).

Two-kernel design (mirroring :mod:`v4_attention_bwd`):

* Pre-pass: ``D[b, h, m] = sum_d (dout[b,h,m,d] * out[b,h,m,d])`` —
  reuses :func:`_v4_attention_bwd_preprocess_kernel` from the dense
  module since the contract is identical.
* Main pass: one program per ``(b, qhid, m)`` query row; re-materialises
  the joint softmax row from the saved LSE; emits

  ::
    dq         [B, H, Sq, D]      direct store (one program per row)
    dk_local   [B, H, Sq, D]      atomic-add (multiple m's hit same n)
    dv_local   [B, H, Sq, D]      atomic-add
    dgathered  [B, Sq, K_topk, D] atomic-add (no H dim — broadcast in fwd
                                  means all H heads contribute)
    dsink      [H]                atomic-add per query

dtype contract:

* All inputs loaded in input dtype; per-row dot products reduce in fp32
  via ``.to(tl.float32)`` upcast before the multiply (matches the FWD's
  bf16-tensor-core / fp32-accumulator semantics).
* The online ``P / dP / dS`` re-materialisation is fp32 (matches the
  FWD's softmax-in-fp32 contract).
* Output gradients are returned in input dtype (cast from fp32 buffers
  by the launcher).

Math derivation (per query (b, h, m), see plan-4 ``02-phase-details.md``
Phase 26 section):

  joint_logits = cat(qk_local, qk_sparse, sink_h)
  P_j = exp(joint_logits[j] - lse)
  out_d = sum_n P_local[n] * v_local[n,d] + sum_k P_sparse[k] * g[k,d]
  D = sum_d (dout[d] * out[d])
  dP_local[n]  = sum_d (dout[d] * v_local[n,d])
  dP_sparse[k] = sum_d (dout[d] * g[k,d])
  dS_local[n]  = P_local[n]  * (dP_local[n]  - D)
  dS_sparse[k] = P_sparse[k] * (dP_sparse[k] - D)
  dS_sink      = -P_sink * D                                    # sink val is 0

  dq[d]               = sum_n dS_local[n]  * scale * k_local[n,d]
                      + sum_k dS_sparse[k] * scale * g[k,d]
  dk_local[n,d]      += dS_local[n]  * scale * q[d]
  dv_local[n,d]      += P_local[n]   * dout[d]
  dgathered[k,d]     += dS_sparse[k] * scale * q[d]
                      + P_sparse[k]  * dout[d]                   # both branches
  dsink_h            += dS_sink

Edge cases:

* ``K_topk == 0`` — the wrapper short-circuits to the dense
  :func:`v4_attention` BWD before reaching this kernel.
* All-masked tile rows — uses ``NEG_INF = -1e30`` finite sentinel so
  ``exp(NEG_INF - lse) = exp(-large) ≈ 0`` for fully-masked positions
  (matches the FWD).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from primus.backends.megatron.core.transformer.v4_attention_kernels._triton.v4_attention_bwd import (
    _v4_attention_bwd_dkv_kernel,
    _v4_attention_bwd_dq_kernel,
    _v4_attention_bwd_kernel,
    _v4_attention_bwd_preprocess_kernel,
)

# ---------------------------------------------------------------------------
# Main BWD kernel
# ---------------------------------------------------------------------------


@triton.jit
def _v4_csa_attention_bwd_kernel(
    Q,
    K_LOCAL,
    V_LOCAL,
    GATHERED,
    SPARSE_MASK,
    DOUT,
    LSE,
    D,
    DQ,  # fp32 buffer [B, H, Sq, D]
    DK_LOCAL,  # fp32 buffer [B, H, Sq, D]
    DV_LOCAL,  # fp32 buffer [B, H, Sq, D]
    DGATHERED,  # fp32 buffer [B, Sq, K_topk, D]
    DSINK,  # fp32 buffer [H] or sentinel
    SINK,  # [H] or sentinel
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_klb,
    stride_klh,
    stride_kln,
    stride_kld,
    stride_vlb,
    stride_vlh,
    stride_vln,
    stride_vld,
    stride_gb,
    stride_gm,
    stride_gk,
    stride_gd,
    stride_smb,
    stride_smm,
    stride_smk,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_db,
    stride_dh,
    stride_dm,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    stride_dklb,
    stride_dklh,
    stride_dkln,
    stride_dkld,
    stride_dvlb,
    stride_dvlh,
    stride_dvln,
    stride_dvld,
    stride_dgb,
    stride_dgm,
    stride_dgk,
    stride_dgd,
    seqlen_q,
    K_topk,
    sm_scale,
    HEAD_Q: tl.constexpr,
    SWA_WINDOW: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """V4 CSA fused-attention BWD (one program per (b, qhid, m) query row)."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    bid = pid_bh // HEAD_Q
    qhid = pid_bh % HEAD_Q

    offs_d = tl.arange(0, BLOCK_DMODEL)

    NEG_INF: tl.constexpr = -1.0e30

    q_active = pid_m < seqlen_q

    # ---- Load Q row, dout row, lse, D scalar ------------------------------
    q_row_offset = bid * stride_qb + qhid * stride_qh + pid_m * stride_qm
    q = tl.load(Q + q_row_offset + offs_d * stride_qd, mask=q_active, other=0.0)

    do_row_offset = bid * stride_dob + qhid * stride_doh + pid_m * stride_dom
    dout = tl.load(DOUT + do_row_offset + offs_d * stride_dod, mask=q_active, other=0.0)
    q_f = q.to(tl.float32)
    dout.to(tl.float32)

    lse = tl.load(
        LSE + bid * stride_lb + qhid * stride_lh + pid_m * stride_lm,
        mask=q_active,
        other=0.0,
    )
    dvec = tl.load(
        D + bid * stride_db + qhid * stride_dh + pid_m * stride_dm,
        mask=q_active,
        other=0.0,
    )

    # ---- Sink contribution to dsink ---------------------------------------
    # dS_sink = -P_sink * D; logit_sink = sink_h, so dsink_h += dS_sink.
    if HAS_SINK:
        sink_h = tl.load(SINK + qhid).to(tl.float32)
        p_sink = tl.exp(sink_h - lse)
        # Mask boundary rows so they don't contribute.
        dsink_contrib = tl.where(q_active, -p_sink * dvec, 0.0)
        tl.atomic_add(DSINK + qhid, dsink_contrib)

    # dq accumulator (fp32, kept in registers across the n-loop and k-loop)
    dq = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    # ---- Local SWA branch -------------------------------------------------
    n_loop_end = pid_m + 1
    if n_loop_end > seqlen_q:
        n_loop_end = seqlen_q

    if SWA_WINDOW > 0:
        n_lo_raw = pid_m - SWA_WINDOW + 1
        if n_lo_raw < 0:
            n_lo_raw = 0
        n_loop_start = (n_lo_raw // BLOCK_N) * BLOCK_N
    else:
        n_loop_start = 0

    for n_start in range(n_loop_start, n_loop_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)

        kl_ptrs = (
            K_LOCAL
            + bid * stride_klb
            + qhid * stride_klh
            + offs_n[:, None] * stride_kln
            + offs_d[None, :] * stride_kld
        )
        kl_load_mask = offs_n[:, None] < seqlen_q
        kl = tl.load(kl_ptrs, mask=kl_load_mask, other=0.0)

        vl_ptrs = (
            V_LOCAL
            + bid * stride_vlb
            + qhid * stride_vlh
            + offs_n[:, None] * stride_vln
            + offs_d[None, :] * stride_vld
        )
        vl = tl.load(vl_ptrs, mask=kl_load_mask, other=0.0)

        # Re-materialise qk in fp32 (matches FWD).
        kl_f = kl.to(tl.float32)
        qk = tl.sum(kl_f * q_f[None, :], axis=1) * sm_scale

        if SWA_WINDOW > 0:
            in_window = (offs_n >= pid_m - SWA_WINDOW + 1) & (offs_n <= pid_m)
        else:
            in_window = offs_n <= pid_m
        qk = tl.where(in_window, qk, NEG_INF)
        qk = tl.where(offs_n < seqlen_q, qk, NEG_INF)
        # Boundary: off-grid m rows have lse=0 already, but we additionally
        # zero this whole tile's contribution by forcing qk to NEG_INF.
        qk = tl.where(q_active, qk, NEG_INF)

        # P = exp(qk - lse)  (joint softmax slice for the local branch)
        p = tl.exp(qk - lse)

        # dP[n] = sum_d (dout[d] * vl[n, d])
        dp = tl.sum(dout[None, :].to(tl.float32) * vl.to(tl.float32), axis=1)

        # dS[n] = P[n] * (dP[n] - D)
        ds = p * (dp - dvec)

        # dq += sum_n (ds[n] * scale * kl[n, d])
        dq += tl.sum(ds[:, None] * kl.to(tl.float32), axis=0) * sm_scale

        # dk_local[n, d] += ds[n] * scale * q[d]   — atomic-add into fp32 buf
        dk_contrib = ds[:, None] * sm_scale * q[None, :].to(tl.float32)
        dk_ptrs = (
            DK_LOCAL
            + bid * stride_dklb
            + qhid * stride_dklh
            + offs_n[:, None] * stride_dkln
            + offs_d[None, :] * stride_dkld
        )
        tl.atomic_add(dk_ptrs, dk_contrib, mask=kl_load_mask, sem="relaxed")

        # dv_local[n, d] += p[n] * dout[d]   — atomic-add into fp32 buf
        dv_contrib = p[:, None] * dout[None, :].to(tl.float32)
        dv_ptrs = (
            DV_LOCAL
            + bid * stride_dvlb
            + qhid * stride_dvlh
            + offs_n[:, None] * stride_dvln
            + offs_d[None, :] * stride_dvld
        )
        tl.atomic_add(dv_ptrs, dv_contrib, mask=kl_load_mask, sem="relaxed")

    # ---- Sparse branch ----------------------------------------------------
    for k_start in range(0, K_topk, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        g_ptrs = (
            GATHERED
            + bid * stride_gb
            + pid_m * stride_gm
            + offs_k[:, None] * stride_gk
            + offs_d[None, :] * stride_gd
        )
        g_load_mask = offs_k[:, None] < K_topk
        g = tl.load(g_ptrs, mask=g_load_mask, other=0.0)

        sm_ptrs = SPARSE_MASK + bid * stride_smb + pid_m * stride_smm + offs_k * stride_smk
        sm_load_mask = offs_k < K_topk
        sm = tl.load(sm_ptrs, mask=sm_load_mask, other=0.0).to(tl.float32)

        qk_sparse = tl.sum(g.to(tl.float32) * q[None, :].to(tl.float32), axis=1) * sm_scale + sm
        qk_sparse = tl.where(offs_k < K_topk, qk_sparse, NEG_INF)
        qk_sparse = tl.where(q_active, qk_sparse, NEG_INF)

        p = tl.exp(qk_sparse - lse)

        # dP[k] = sum_d (dout[d] * g[k, d])
        dp = tl.sum(dout[None, :].to(tl.float32) * g.to(tl.float32), axis=1)
        ds = p * (dp - dvec)

        # dq += sum_k (ds[k] * scale * g[k, d])
        dq += tl.sum(ds[:, None] * g.to(tl.float32), axis=0) * sm_scale

        # dgathered[k, d] += ds[k] * scale * q[d] + p[k] * dout[d]
        # gathered is broadcast across H in the FWD, so this atomic-add
        # accumulates contributions from every query head — matches the
        # eager autograd semantics of ``gathered.unsqueeze(1).expand(B, H,
        # Sq, K, D)``.
        dg_contrib = ds[:, None] * sm_scale * q[None, :].to(tl.float32) + p[:, None] * dout[None, :].to(
            tl.float32
        )
        dg_ptrs = (
            DGATHERED
            + bid * stride_dgb
            + pid_m * stride_dgm
            + offs_k[:, None] * stride_dgk
            + offs_d[None, :] * stride_dgd
        )
        tl.atomic_add(dg_ptrs, dg_contrib, mask=g_load_mask, sem="relaxed")

    # ---- Store dq (direct — no collisions across programs) ----------------
    dq_offset = bid * stride_dqb + qhid * stride_dqh + pid_m * stride_dqm
    tl.store(DQ + dq_offset + offs_d * stride_dqd, dq, mask=q_active)


@triton.jit
def _v4_csa_attention_pool_bwd_kernel(
    Q,
    K_LOCAL,
    V_LOCAL,
    POOL,
    TOPK_IDXS,
    DOUT,
    LSE,
    D,
    DQ,
    DK_LOCAL,
    DV_LOCAL,
    DPOOL,
    DSINK,
    SINK,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_klb,
    stride_klh,
    stride_kln,
    stride_kld,
    stride_vlb,
    stride_vlh,
    stride_vln,
    stride_vld,
    stride_pb,
    stride_pp,
    stride_pd,
    stride_tib,
    stride_tim,
    stride_tik,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_db,
    stride_dh,
    stride_dm,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    stride_dklb,
    stride_dklh,
    stride_dkln,
    stride_dkld,
    stride_dvlb,
    stride_dvlh,
    stride_dvln,
    stride_dvld,
    stride_dpb,
    stride_dpp,
    stride_dpd,
    seqlen_q,
    pool_size,
    K_topk,
    sm_scale,
    HEAD_Q: tl.constexpr,
    SWA_WINDOW: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    STORE_DPOOL: tl.constexpr,
    LOCAL_ONLY: tl.constexpr,
):
    """CSA BWD with in-kernel scatter-add into the compressed-pool gradient."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    bid = pid_bh // HEAD_Q
    qhid = pid_bh % HEAD_Q

    offs_d = tl.arange(0, BLOCK_DMODEL)
    NEG_INF: tl.constexpr = -1.0e30
    q_active = pid_m < seqlen_q

    q_row_offset = bid * stride_qb + qhid * stride_qh + pid_m * stride_qm
    q = tl.load(Q + q_row_offset + offs_d * stride_qd, mask=q_active, other=0.0)

    do_row_offset = bid * stride_dob + qhid * stride_doh + pid_m * stride_dom
    dout = tl.load(DOUT + do_row_offset + offs_d * stride_dod, mask=q_active, other=0.0)
    q_f = q.to(tl.float32)
    dout_f = dout.to(tl.float32)

    lse = tl.load(
        LSE + bid * stride_lb + qhid * stride_lh + pid_m * stride_lm,
        mask=q_active,
        other=0.0,
    )
    dvec = tl.load(
        D + bid * stride_db + qhid * stride_dh + pid_m * stride_dm,
        mask=q_active,
        other=0.0,
    )

    if HAS_SINK:
        sink_h = tl.load(SINK + qhid).to(tl.float32)
        p_sink = tl.exp(sink_h - lse)
        tl.atomic_add(DSINK + qhid, tl.where(q_active, -p_sink * dvec, 0.0))

    dq = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    n_loop_end = pid_m + 1
    if n_loop_end > seqlen_q:
        n_loop_end = seqlen_q
    if SWA_WINDOW > 0:
        n_lo_raw = pid_m - SWA_WINDOW + 1
        if n_lo_raw < 0:
            n_lo_raw = 0
        n_loop_start = (n_lo_raw // BLOCK_N) * BLOCK_N
    else:
        n_loop_start = 0

    for n_start in range(n_loop_start, n_loop_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)

        kl_ptrs = (
            K_LOCAL
            + bid * stride_klb
            + qhid * stride_klh
            + offs_n[:, None] * stride_kln
            + offs_d[None, :] * stride_kld
        )
        kl_load_mask = offs_n[:, None] < seqlen_q
        kl = tl.load(kl_ptrs, mask=kl_load_mask, other=0.0)

        vl_ptrs = (
            V_LOCAL
            + bid * stride_vlb
            + qhid * stride_vlh
            + offs_n[:, None] * stride_vln
            + offs_d[None, :] * stride_vld
        )
        vl = tl.load(vl_ptrs, mask=kl_load_mask, other=0.0)

        kl_f = kl.to(tl.float32)
        qk = tl.sum(kl_f * q_f[None, :], axis=1) * sm_scale
        if SWA_WINDOW > 0:
            in_window = (offs_n >= pid_m - SWA_WINDOW + 1) & (offs_n <= pid_m)
        else:
            in_window = offs_n <= pid_m
        qk = tl.where(in_window & (offs_n < seqlen_q) & q_active, qk, NEG_INF)

        p = tl.exp(qk - lse)
        vl_f = vl.to(tl.float32)
        dp = tl.sum(dout_f[None, :] * vl_f, axis=1)
        ds = p * (dp - dvec)

        dq += tl.sum(ds[:, None] * kl_f, axis=0) * sm_scale

        dk_contrib = ds[:, None] * sm_scale * q_f[None, :]
        dk_ptrs = (
            DK_LOCAL
            + bid * stride_dklb
            + qhid * stride_dklh
            + offs_n[:, None] * stride_dkln
            + offs_d[None, :] * stride_dkld
        )
        tl.atomic_add(dk_ptrs, dk_contrib, mask=kl_load_mask, sem="relaxed")

        dv_contrib = p[:, None] * dout_f[None, :]
        dv_ptrs = (
            DV_LOCAL
            + bid * stride_dvlb
            + qhid * stride_dvlh
            + offs_n[:, None] * stride_dvln
            + offs_d[None, :] * stride_dvld
        )
        tl.atomic_add(dv_ptrs, dv_contrib, mask=kl_load_mask, sem="relaxed")

    if not LOCAL_ONLY:
        for k_start in range(0, K_topk, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            topk_ptrs = TOPK_IDXS + bid * stride_tib + pid_m * stride_tim + offs_k * stride_tik
            topk = tl.load(topk_ptrs, mask=offs_k < K_topk, other=-1)
            valid = (offs_k < K_topk) & (topk >= 0) & (topk < pool_size)
            safe_topk = tl.where(valid, topk, 0)

            pool_ptrs = POOL + bid * stride_pb + safe_topk[:, None] * stride_pp + offs_d[None, :] * stride_pd
            pool = tl.load(pool_ptrs, mask=valid[:, None], other=0.0)
            pool_f = pool.to(tl.float32)

            qk_sparse = tl.sum(pool_f * q_f[None, :], axis=1) * sm_scale
            qk_sparse = tl.where(valid & q_active, qk_sparse, NEG_INF)

            p = tl.exp(qk_sparse - lse)
            dp = tl.sum(dout_f[None, :] * pool_f, axis=1)
            ds = p * (dp - dvec)

            dq += tl.sum(ds[:, None] * pool_f, axis=0) * sm_scale

            if STORE_DPOOL:
                dpool_contrib = ds[:, None] * sm_scale * q_f[None, :] + p[:, None] * dout_f[None, :]
                dpool_ptrs = (
                    DPOOL + bid * stride_dpb + safe_topk[:, None] * stride_dpp + offs_d[None, :] * stride_dpd
                )
                tl.atomic_add(dpool_ptrs, dpool_contrib, mask=valid[:, None], sem="relaxed")

    dq_offset = bid * stride_dqb + qhid * stride_dqh + pid_m * stride_dqm
    tl.store(DQ + dq_offset + offs_d * stride_dqd, dq, mask=q_active)


@triton.jit
def _v4_csa_attention_pool_sparse_bwd_kernel(
    Q,
    POOL,
    TOPK_IDXS,
    DOUT,
    LSE,
    D,
    DQ,
    DPOOL,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_pb,
    stride_pp,
    stride_pd,
    stride_tib,
    stride_tim,
    stride_tik,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_db,
    stride_dh,
    stride_dm,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    stride_dp_part,
    stride_dpb,
    stride_dpp,
    stride_dpd,
    partition_size,
    seqlen_q,
    pool_size,
    K_topk,
    sm_scale,
    HEAD_Q: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    STORE_DPOOL: tl.constexpr = True,
):
    """Sparse CSA BWD using a head block so pool work maps to tl.dot.

    Plan-5 P32: ``DPOOL`` is laid out as ``[N_PART, B, P, D]`` instead
    of ``[B, P, D]``. Each program writes its dpool contributions into
    ``DPOOL[pid_m // partition_size, bid, ...]``. With ``N_PART = Sq /
    partition_size``, the per-cache-line atomic contention drops by a
    factor of ``N_PART`` because writes from different m-partitions hit
    different DRAM rows. The launcher reduces the partial axis on the
    Python side, which is bandwidth-cheap (``2 * N_PART * P * D *
    sizeof(fp32)``).
    """
    pid_m = tl.program_id(0)
    pid_h_block = tl.program_id(1)
    bid = tl.program_id(2)
    partition_id = pid_m // partition_size

    offs_h = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    h_mask = offs_h < HEAD_Q
    q_active = pid_m < seqlen_q

    q_ptrs = (
        Q + bid * stride_qb + offs_h[:, None] * stride_qh + pid_m * stride_qm + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=h_mask[:, None] & q_active, other=0.0)

    dout_ptrs = (
        DOUT
        + bid * stride_dob
        + offs_h[:, None] * stride_doh
        + pid_m * stride_dom
        + offs_d[None, :] * stride_dod
    )
    dout = tl.load(dout_ptrs, mask=h_mask[:, None] & q_active, other=0.0)

    lse = tl.load(
        LSE + bid * stride_lb + offs_h * stride_lh + pid_m * stride_lm,
        mask=h_mask & q_active,
        other=0.0,
    )
    dvec = tl.load(
        D + bid * stride_db + offs_h * stride_dh + pid_m * stride_dm,
        mask=h_mask & q_active,
        other=0.0,
    )

    dq = tl.zeros([BLOCK_H, BLOCK_DMODEL], dtype=tl.float32)

    for k_start in range(0, K_topk, BLOCK_K):
        sparse_k = k_start + offs_k
        topk_ptrs = TOPK_IDXS + bid * stride_tib + pid_m * stride_tim + sparse_k * stride_tik
        topk = tl.load(topk_ptrs, mask=sparse_k < K_topk, other=-1)
        valid_k = (sparse_k < K_topk) & (topk >= 0) & (topk < pool_size)
        safe_topk = tl.where(valid_k, topk, 0)

        pool_ptrs = POOL + bid * stride_pb + safe_topk[:, None] * stride_pp + offs_d[None, :] * stride_pd
        pool = tl.load(pool_ptrs, mask=valid_k[:, None], other=0.0)

        q_bf16 = q.to(pool.dtype)
        dout_bf16 = dout.to(pool.dtype)
        qk = tl.dot(q_bf16, tl.trans(pool)) * sm_scale
        qk = tl.where((h_mask[:, None] & valid_k[None, :] & q_active), qk, -1.0e30)

        p = tl.exp(qk - lse[:, None])
        dp = tl.dot(dout_bf16, tl.trans(pool))
        ds = p * (dp - dvec[:, None])

        dq += tl.dot(ds.to(pool.dtype), pool) * sm_scale

        if STORE_DPOOL:
            dpool_contrib = tl.dot(tl.trans(ds.to(q.dtype)), q_bf16) * sm_scale
            dpool_contrib += tl.dot(tl.trans(p.to(dout.dtype)), dout_bf16)
            dpool_ptrs = (
                DPOOL
                + partition_id * stride_dp_part
                + bid * stride_dpb
                + safe_topk[:, None] * stride_dpp
                + offs_d[None, :] * stride_dpd
            )
            tl.atomic_add(dpool_ptrs, dpool_contrib, mask=valid_k[:, None], sem="relaxed")
        else:
            # Force the optimiser to keep this branch as a no-op so the
            # ``STORE_DPOOL=False`` variant truly drops the two trailing
            # ``tl.dot`` ops + atomic from the IR.
            pass

    # Plan-5 P32: each ``(pid_m, pid_h_block, bid)`` program writes to a
    # disjoint slice of ``DQ`` (different ``m`` rows + different head
    # blocks), so a plain ``tl.store`` is safe and ~2× cheaper than the
    # atomic on MI355.
    dq_ptrs = (
        DQ
        + bid * stride_dqb
        + offs_h[:, None] * stride_dqh
        + pid_m * stride_dqm
        + offs_d[None, :] * stride_dqd
    )
    tl.store(dq_ptrs, dq, mask=h_mask[:, None] & q_active)


@triton.jit
def _v4_csa_attention_pool_sparse_bwd_dq_only_kernel(
    Q,
    POOL,
    TOPK_IDXS,
    DOUT,
    LSE,
    D,
    DQ,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_pb,
    stride_pp,
    stride_pd,
    stride_tib,
    stride_tim,
    stride_tik,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_db,
    stride_dh,
    stride_dm,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    seqlen_q,
    pool_size,
    K_topk,
    sm_scale,
    HEAD_Q: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """P57: dq-only sparse CSA BWD (no dpool_partial write).

    Mirrors :func:`_v4_csa_attention_pool_sparse_bwd_partial_kernel`
    but drops the per-visit ``dpool_partial`` write and the two
    ``dpool_contrib`` matmuls. The freed register file lets us run
    with a larger ``BLOCK_K`` / more ``num_stages`` so the per-program
    latency drops sharply. The dpool partial is produced by a sibling
    ``_v4_csa_attention_pool_sparse_bwd_dpool_only_kernel`` that
    omits the ``dq`` accumulator instead, and the segreduce kernel
    folds the partial into ``dpool[B, P, D]``.

    The reason for splitting: the joint kernel's wall-clock at the
    proxy shape is ~3.7 ms, dominated by the 4 GiB ``dpool_partial``
    write *interleaved* with the ``[BLOCK_H, BLOCK_DMODEL]`` fp32
    ``dq`` accumulator (~64 KB live across the K-loop). Splitting
    drops the live-register footprint of each kernel ~2×, letting
    the compiler use higher ``num_stages`` for prefetching.
    """
    pid_m = tl.program_id(0)
    pid_h_block = tl.program_id(1)
    bid = tl.program_id(2)

    offs_h = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    h_mask = offs_h < HEAD_Q
    q_active = pid_m < seqlen_q

    q_ptrs = (
        Q + bid * stride_qb + offs_h[:, None] * stride_qh + pid_m * stride_qm + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=h_mask[:, None] & q_active, other=0.0)

    dout_ptrs = (
        DOUT
        + bid * stride_dob
        + offs_h[:, None] * stride_doh
        + pid_m * stride_dom
        + offs_d[None, :] * stride_dod
    )
    dout = tl.load(dout_ptrs, mask=h_mask[:, None] & q_active, other=0.0)

    lse = tl.load(
        LSE + bid * stride_lb + offs_h * stride_lh + pid_m * stride_lm,
        mask=h_mask & q_active,
        other=0.0,
    )
    dvec = tl.load(
        D + bid * stride_db + offs_h * stride_dh + pid_m * stride_dm,
        mask=h_mask & q_active,
        other=0.0,
    )

    dq = tl.zeros([BLOCK_H, BLOCK_DMODEL], dtype=tl.float32)

    # P57 R2: scale-defer + acc-form MFMA (mirror of the joint
    # partial kernel). See the comment in
    # ``_v4_csa_attention_pool_sparse_bwd_partial_kernel`` for the
    # math.
    pool_dtype = POOL.dtype.element_ty
    q_scaled = (q.to(tl.float32) * sm_scale).to(pool_dtype)
    dout_bf16 = dout.to(pool_dtype)

    for k_start in range(0, K_topk, BLOCK_K):
        sparse_k = k_start + offs_k
        topk_ptrs = TOPK_IDXS + bid * stride_tib + pid_m * stride_tim + sparse_k * stride_tik
        topk = tl.load(topk_ptrs, mask=sparse_k < K_topk, other=-1)
        valid_k = (sparse_k < K_topk) & (topk >= 0) & (topk < pool_size)
        safe_topk = tl.where(valid_k, topk, 0)

        pool_ptrs = POOL + bid * stride_pb + safe_topk[:, None] * stride_pp + offs_d[None, :] * stride_pd
        pool = tl.load(pool_ptrs, mask=valid_k[:, None], other=0.0)

        qk = tl.dot(q_scaled, tl.trans(pool))
        qk = tl.where((h_mask[:, None] & valid_k[None, :] & q_active), qk, -1.0e30)

        p = tl.exp(qk - lse[:, None])
        dp = tl.dot(dout_bf16, tl.trans(pool))
        ds = p * (dp - dvec[:, None])

        dq = tl.dot(ds.to(pool_dtype), pool, acc=dq)

    dq = dq * sm_scale

    dq_ptrs = (
        DQ
        + bid * stride_dqb
        + offs_h[:, None] * stride_dqh
        + pid_m * stride_dqm
        + offs_d[None, :] * stride_dqd
    )
    tl.store(dq_ptrs, dq, mask=h_mask[:, None] & q_active)


@triton.jit
def _v4_csa_attention_pool_sparse_bwd_dpool_only_kernel(
    Q,
    POOL,
    TOPK_IDXS,
    DOUT,
    LSE,
    D,
    DPOOL_PARTIAL,  # [B, M, K_topk, D] fp32 — NO atomics, each program owns its slice
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_pb,
    stride_pp,
    stride_pd,
    stride_tib,
    stride_tim,
    stride_tik,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_db,
    stride_dh,
    stride_dm,
    stride_dpb,
    stride_dpm,
    stride_dpk,
    stride_dpd,
    seqlen_q,
    pool_size,
    K_topk,
    sm_scale,
    HEAD_Q: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """P57: dpool-partial-only sparse CSA BWD (no dq accumulator).

    Mirror of :func:`_v4_csa_attention_pool_sparse_bwd_dq_only_kernel`
    that drops the ``dq`` accumulator (64 KB fp32 per program) so
    the per-iter ``dpool_contrib`` matmul + write loop runs with a
    tighter register footprint and can issue more in-flight HBM
    writes per warp.
    """
    pid_m = tl.program_id(0)
    pid_h_block = tl.program_id(1)
    bid = tl.program_id(2)

    offs_h = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    h_mask = offs_h < HEAD_Q
    q_active = pid_m < seqlen_q

    q_ptrs = (
        Q + bid * stride_qb + offs_h[:, None] * stride_qh + pid_m * stride_qm + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=h_mask[:, None] & q_active, other=0.0)

    dout_ptrs = (
        DOUT
        + bid * stride_dob
        + offs_h[:, None] * stride_doh
        + pid_m * stride_dom
        + offs_d[None, :] * stride_dod
    )
    dout = tl.load(dout_ptrs, mask=h_mask[:, None] & q_active, other=0.0)

    lse = tl.load(
        LSE + bid * stride_lb + offs_h * stride_lh + pid_m * stride_lm,
        mask=h_mask & q_active,
        other=0.0,
    )
    dvec = tl.load(
        D + bid * stride_db + offs_h * stride_dh + pid_m * stride_dm,
        mask=h_mask & q_active,
        other=0.0,
    )

    # P57 R2: scale-defer (q-fold) — see partial kernel for math.
    pool_dtype = POOL.dtype.element_ty
    q_scaled = (q.to(tl.float32) * sm_scale).to(pool_dtype)
    dout_bf16 = dout.to(pool_dtype)

    for k_start in range(0, K_topk, BLOCK_K):
        sparse_k = k_start + offs_k
        topk_ptrs = TOPK_IDXS + bid * stride_tib + pid_m * stride_tim + sparse_k * stride_tik
        topk = tl.load(topk_ptrs, mask=sparse_k < K_topk, other=-1)
        valid_k = (sparse_k < K_topk) & (topk >= 0) & (topk < pool_size)
        safe_topk = tl.where(valid_k, topk, 0)

        pool_ptrs = POOL + bid * stride_pb + safe_topk[:, None] * stride_pp + offs_d[None, :] * stride_pd
        pool = tl.load(pool_ptrs, mask=valid_k[:, None], other=0.0)

        qk = tl.dot(q_scaled, tl.trans(pool))
        qk = tl.where((h_mask[:, None] & valid_k[None, :] & q_active), qk, -1.0e30)

        p = tl.exp(qk - lse[:, None])
        dp = tl.dot(dout_bf16, tl.trans(pool))
        ds = p * (dp - dvec[:, None])

        dpool_contrib = tl.dot(tl.trans(ds.to(pool_dtype)), q_scaled)
        dpool_contrib = tl.dot(tl.trans(p.to(pool_dtype)), dout_bf16, acc=dpool_contrib)
        dpool_contrib = tl.where(valid_k[:, None], dpool_contrib, 0.0)
        dpool_partial_ptrs = (
            DPOOL_PARTIAL
            + bid * stride_dpb
            + pid_m * stride_dpm
            + sparse_k[:, None] * stride_dpk
            + offs_d[None, :] * stride_dpd
        )
        tl.store(
            dpool_partial_ptrs,
            dpool_contrib,
            mask=(sparse_k[:, None] < K_topk) & q_active,
        )


@triton.jit
def _v4_csa_attention_pool_sparse_bwd_partial_sorted_kernel(
    Q,
    POOL,
    TOPK_IDXS,
    INV_PERM,  # [B, MK] int32 — sorted_position = inv_perm[orig_flat]
    DOUT,
    LSE,
    D,
    DQ,
    DPOOL_PARTIAL,  # [B, MK_sorted, D] partial buffer (bf16 / fp32)
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_pb,
    stride_pp,
    stride_pd,
    stride_tib,
    stride_tim,
    stride_tik,
    stride_ipb,
    stride_ipi,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_db,
    stride_dh,
    stride_dm,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    stride_dpb,
    stride_dpi,
    stride_dpd,
    seqlen_q,
    pool_size,
    K_topk,
    sm_scale,
    HEAD_Q: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """P57 sorted variant of the sparse CSA BWD partial kernel.

    Writes the per-visit ``dpool_contrib`` tile into the sorted-order
    position of ``DPOOL_PARTIAL`` (a flat ``[B, MK, D]`` buffer)
    using a host-side ``INV_PERM`` (the inverse of the segreduce
    ``perm``). The downstream segreduce kernel can then read
    contiguous ``[bin_start:bin_end, :]`` slices instead of gathering
    via ``perm[i]`` — turning a random-access reduction into a
    streaming one (better L2 / Infinity Cache reuse on MI355).

    The partial kernel writes are now scattered (one row per sparse_k
    slot to a sorted position), but the HBM write bandwidth is
    similar because Triton's vector store still coalesces 128 B
    cache-line bursts as long as ``BLOCK_DMODEL`` is contiguous.
    """
    pid_m = tl.program_id(0)
    pid_h_block = tl.program_id(1)
    bid = tl.program_id(2)

    offs_h = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    h_mask = offs_h < HEAD_Q
    q_active = pid_m < seqlen_q

    q_ptrs = (
        Q + bid * stride_qb + offs_h[:, None] * stride_qh + pid_m * stride_qm + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=h_mask[:, None] & q_active, other=0.0)

    dout_ptrs = (
        DOUT
        + bid * stride_dob
        + offs_h[:, None] * stride_doh
        + pid_m * stride_dom
        + offs_d[None, :] * stride_dod
    )
    dout = tl.load(dout_ptrs, mask=h_mask[:, None] & q_active, other=0.0)

    lse = tl.load(
        LSE + bid * stride_lb + offs_h * stride_lh + pid_m * stride_lm,
        mask=h_mask & q_active,
        other=0.0,
    )
    dvec = tl.load(
        D + bid * stride_db + offs_h * stride_dh + pid_m * stride_dm,
        mask=h_mask & q_active,
        other=0.0,
    )

    dq = tl.zeros([BLOCK_H, BLOCK_DMODEL], dtype=tl.float32)
    flat_base = pid_m * K_topk

    # P57 R2: scale-defer (q-fold) — see partial kernel for math.
    pool_dtype = POOL.dtype.element_ty
    q_scaled = (q.to(tl.float32) * sm_scale).to(pool_dtype)
    dout_bf16 = dout.to(pool_dtype)

    for k_start in range(0, K_topk, BLOCK_K):
        sparse_k = k_start + offs_k
        topk_ptrs = TOPK_IDXS + bid * stride_tib + pid_m * stride_tim + sparse_k * stride_tik
        topk = tl.load(topk_ptrs, mask=sparse_k < K_topk, other=-1)
        valid_k = (sparse_k < K_topk) & (topk >= 0) & (topk < pool_size)
        safe_topk = tl.where(valid_k, topk, 0)

        pool_ptrs = POOL + bid * stride_pb + safe_topk[:, None] * stride_pp + offs_d[None, :] * stride_pd
        pool = tl.load(pool_ptrs, mask=valid_k[:, None], other=0.0)

        qk = tl.dot(q_scaled, tl.trans(pool))
        qk = tl.where((h_mask[:, None] & valid_k[None, :] & q_active), qk, -1.0e30)

        p = tl.exp(qk - lse[:, None])
        dp = tl.dot(dout_bf16, tl.trans(pool))
        ds = p * (dp - dvec[:, None])

        dq = tl.dot(ds.to(pool_dtype), pool, acc=dq)

        dpool_contrib = tl.dot(tl.trans(ds.to(pool_dtype)), q_scaled)
        dpool_contrib = tl.dot(tl.trans(p.to(pool_dtype)), dout_bf16, acc=dpool_contrib)
        dpool_contrib = tl.where(valid_k[:, None], dpool_contrib, 0.0)

        # P57: look up the SORTED position for each ``(m, sparse_k)``
        # visit. ``flat_base + sparse_k`` is the ORIGINAL flat index;
        # ``inv_perm[bid, orig_flat]`` is the sorted slot to write.
        orig_flat = flat_base + sparse_k
        inv_perm_ptrs = INV_PERM + bid * stride_ipb + orig_flat * stride_ipi
        sorted_idx = tl.load(inv_perm_ptrs, mask=sparse_k < K_topk, other=0)
        dpool_partial_ptrs = (
            DPOOL_PARTIAL + bid * stride_dpb + sorted_idx[:, None] * stride_dpi + offs_d[None, :] * stride_dpd
        )
        tl.store(
            dpool_partial_ptrs,
            dpool_contrib,
            mask=(sparse_k[:, None] < K_topk) & q_active,
        )

    dq = dq * sm_scale

    dq_ptrs = (
        DQ
        + bid * stride_dqb
        + offs_h[:, None] * stride_dqh
        + pid_m * stride_dqm
        + offs_d[None, :] * stride_dqd
    )
    tl.store(dq_ptrs, dq, mask=h_mask[:, None] & q_active)


@triton.jit
def _v4_csa_attention_pool_segreduce_sequential_kernel(
    DPOOL_PARTIAL,  # [B, MK_sorted, D] — written in sorted order by partial_sorted kernel
    BIN_PTR,  # [B, P+1] int32 — prefix sum of count per pool slot (same as non-sorted variant)
    DPOOL,  # [B, P, D]
    stride_dpb,
    stride_dpi,
    stride_dpd,
    stride_binb,
    stride_binp,
    stride_db,
    stride_dp,
    stride_dd,
    P,
    D_size,
    BLOCK_D: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """P57 sequential segreduce — reads dpool_partial[bid, i:i+BI, :]
    contiguously without a perm lookup since the partial kernel
    already wrote in sorted order.
    """
    pid_p = tl.program_id(0)
    pid_d_block = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_d = pid_d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < D_size

    bin_start = tl.load(BIN_PTR + pid_b * stride_binb + pid_p * stride_binp)
    bin_end = tl.load(BIN_PTR + pid_b * stride_binb + (pid_p + 1) * stride_binp)

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    i = bin_start
    while i < bin_end:
        offs_i = i + tl.arange(0, BLOCK_I)
        valid_i = offs_i < bin_end
        partial_ptrs = (
            DPOOL_PARTIAL + pid_b * stride_dpb + offs_i[:, None] * stride_dpi + offs_d[None, :] * stride_dpd
        )
        partial = tl.load(
            partial_ptrs,
            mask=valid_i[:, None] & d_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(partial, axis=0)
        i += BLOCK_I

    dpool_offset = pid_b * stride_db + pid_p * stride_dp + offs_d * stride_dd
    tl.store(DPOOL + dpool_offset, acc, mask=d_mask)


@triton.jit
def _v4_csa_attention_pool_sparse_bwd_partial_kernel(
    Q,
    POOL,
    TOPK_IDXS,
    DOUT,
    LSE,
    D,
    DQ,
    DPOOL_PARTIAL,  # [B, M, K_topk, D] fp32 — NO atomics, each program owns its slice
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_pb,
    stride_pp,
    stride_pd,
    stride_tib,
    stride_tim,
    stride_tik,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_db,
    stride_dh,
    stride_dm,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    stride_dpb,
    stride_dpm,
    stride_dpk,
    stride_dpd,
    seqlen_q,
    pool_size,
    K_topk,
    sm_scale,
    HEAD_Q: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Sparse CSA BWD that writes dpool contributions to a compact
    ``[B, M, K_topk, D]`` partial buffer **without** atomics.

    Plan-5 P32: the atomic_add to the shared ``dpool[B, P, D]`` buffer
    was the dominant cost (~15 ms out of ~24 ms — see
    ``PRIMUS_V4_CSA_BWD_SKIP_DPOOL_ATOMIC`` profile). By emitting the
    raw per-visit contributions into a compact partial buffer (4 GB
    for the proxy shape, indexed by ``(b, m, k_slot)``), a follow-up
    segmented reduction kernel can fold them into ``dpool[B, P, D]``
    using a sorted inverse index — atomics-free and bandwidth-bound.
    """
    pid_m = tl.program_id(0)
    pid_h_block = tl.program_id(1)
    bid = tl.program_id(2)

    offs_h = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    h_mask = offs_h < HEAD_Q
    q_active = pid_m < seqlen_q

    q_ptrs = (
        Q + bid * stride_qb + offs_h[:, None] * stride_qh + pid_m * stride_qm + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=h_mask[:, None] & q_active, other=0.0)

    dout_ptrs = (
        DOUT
        + bid * stride_dob
        + offs_h[:, None] * stride_doh
        + pid_m * stride_dom
        + offs_d[None, :] * stride_dod
    )
    dout = tl.load(dout_ptrs, mask=h_mask[:, None] & q_active, other=0.0)

    lse = tl.load(
        LSE + bid * stride_lb + offs_h * stride_lh + pid_m * stride_lm,
        mask=h_mask & q_active,
        other=0.0,
    )
    dvec = tl.load(
        D + bid * stride_db + offs_h * stride_dh + pid_m * stride_dm,
        mask=h_mask & q_active,
        other=0.0,
    )

    dq = tl.zeros([BLOCK_H, BLOCK_DMODEL], dtype=tl.float32)

    # P57 R2: hoist q/dout dtype-cast and SCALE-DEFER ``sm_scale`` out
    # of the K-loop. Pre-scaling ``q_bf16`` folds the ``sm_scale``
    # factor that the kernel previously applied to ``qk`` (line:
    # ``tl.dot(q_bf16, K^T) * sm_scale``) and to the first
    # ``dpool_contrib`` matmul (``tl.dot(ds^T, q_bf16) * sm_scale``)
    # — both are linear in q so factoring sm_scale into q is exact.
    # ``dq`` accumulates ``Σ_k ds @ pool`` without sm_scale; a single
    # ``dq *= sm_scale`` after the loop replaces the per-iter
    # multiply on the ``[BLOCK_H, BLOCK_DMODEL]`` fp32 accumulator.
    # Combined with ``tl.dot(..., acc=dq)`` (MFMA in-place
    # accumulator), this drops ~3 fp32 multiplies + 1 fp32 add per
    # K_topk/BLOCK_K=16 iteration on the proxy shape.
    pool_dtype = POOL.dtype.element_ty
    dout_bf16 = dout.to(pool_dtype)
    q_scaled = (q.to(tl.float32) * sm_scale).to(pool_dtype)

    for k_start in range(0, K_topk, BLOCK_K):
        sparse_k = k_start + offs_k
        topk_ptrs = TOPK_IDXS + bid * stride_tib + pid_m * stride_tim + sparse_k * stride_tik
        topk = tl.load(topk_ptrs, mask=sparse_k < K_topk, other=-1)
        valid_k = (sparse_k < K_topk) & (topk >= 0) & (topk < pool_size)
        safe_topk = tl.where(valid_k, topk, 0)

        pool_ptrs = POOL + bid * stride_pb + safe_topk[:, None] * stride_pp + offs_d[None, :] * stride_pd
        pool = tl.load(pool_ptrs, mask=valid_k[:, None], other=0.0)

        qk = tl.dot(q_scaled, tl.trans(pool))
        qk = tl.where((h_mask[:, None] & valid_k[None, :] & q_active), qk, -1.0e30)

        p = tl.exp(qk - lse[:, None])
        dp = tl.dot(dout_bf16, tl.trans(pool))
        ds = p * (dp - dvec[:, None])

        dq = tl.dot(ds.to(pool_dtype), pool, acc=dq)

        # Plan-5 P32: write the per-visit dpool contribution to its own
        # compact slot in ``DPOOL_PARTIAL[b, m, k_slot, :]``. No atomic
        # needed because each ``(b, m, k_slot)`` slot is owned by
        # exactly one program × iteration.
        # P57 R2: ``q_scaled`` already carries the ``sm_scale`` factor,
        # so the first matmul's ``* sm_scale`` is folded in. The
        # ``acc=`` form on the second matmul fuses the fp32 add.
        dpool_contrib = tl.dot(tl.trans(ds.to(pool_dtype)), q_scaled)
        dpool_contrib = tl.dot(tl.trans(p.to(pool_dtype)), dout_bf16, acc=dpool_contrib)
        # Zero out invalid k slots so the reduction can sum them in
        # without first checking validity (the inverse index will skip
        # invalid visits anyway via the sentinel sort key, but a clean
        # buffer is friendlier to debugging and to scalar fallbacks).
        dpool_contrib = tl.where(valid_k[:, None], dpool_contrib, 0.0)
        dpool_partial_ptrs = (
            DPOOL_PARTIAL
            + bid * stride_dpb
            + pid_m * stride_dpm
            + sparse_k[:, None] * stride_dpk
            + offs_d[None, :] * stride_dpd
        )
        tl.store(
            dpool_partial_ptrs,
            dpool_contrib,
            mask=(sparse_k[:, None] < K_topk) & q_active,
        )

    # P57 R2: deferred ``sm_scale`` on the dq accumulator. Single
    # ``[BLOCK_H, BLOCK_DMODEL]`` fp32 multiply replaces the per-iter
    # ``* sm_scale`` inside the loop.
    dq = dq * sm_scale

    dq_ptrs = (
        DQ
        + bid * stride_dqb
        + offs_h[:, None] * stride_dqh
        + pid_m * stride_dqm
        + offs_d[None, :] * stride_dqd
    )
    tl.store(dq_ptrs, dq, mask=h_mask[:, None] & q_active)


@triton.jit
def _v4_csa_attention_pool_segreduce_kernel(
    DPOOL_PARTIAL,  # [B, M, K_topk, D] fp32 — compact partial buffer
    SORTED_PERM,  # [B, M*K_topk] int32 — sorted (m*K + k) per pool slot
    BIN_PTR,  # [B, P+1] int32 — prefix sum of count per pool slot
    DPOOL,  # [B, P, D] fp32 — output (single tl.store per p, d)
    stride_dpb,
    stride_dpmk,
    stride_dpd,
    stride_permb,
    stride_permi,
    stride_binb,
    stride_binp,
    stride_db,
    stride_dp,
    stride_dd,
    P,
    D_size,
    BLOCK_D: tl.constexpr,
    BLOCK_I: tl.constexpr,  # tiles over the visit indices to expose ILP
):
    """Segmented reduction: ``DPOOL[b, p, :] = Σ_i DPOOL_PARTIAL[b, perm[i], :]``
    for ``i ∈ [bin_ptr[b, p], bin_ptr[b, p+1])``.

    Plan-5 P32: one program per ``(b, p, d_block)``. Different
    programs write disjoint output slices, so the writes are plain
    ``tl.store``\\ s — no atomics. The visit indices for slot ``p``
    are stored consecutively in ``SORTED_PERM`` (built once on the
    host by sorting ``topk_idxs``), so loads from
    ``DPOOL_PARTIAL[..., flat_idx, :]`` are reasonably coalescable
    after the sort. ``BLOCK_D`` may exceed the actual head dim
    ``D_size`` (the default proxy shape has ``D = 512`` but the unit
    tests use ``D = 32``), so all dpool / dpool_partial accesses are
    masked by ``offs_d < D_size``.
    """
    pid_p = tl.program_id(0)
    pid_d_block = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_d = pid_d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < D_size

    bin_start = tl.load(BIN_PTR + pid_b * stride_binb + pid_p * stride_binp)
    bin_end = tl.load(BIN_PTR + pid_b * stride_binb + (pid_p + 1) * stride_binp)

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    i = bin_start
    while i < bin_end:
        # Load up to BLOCK_I visit indices and prefetch their fp32
        # partial rows; the tl.dot-free vectorised path nets enough
        # ILP to saturate HBM bandwidth on MI355.
        offs_i = i + tl.arange(0, BLOCK_I)
        valid_i = offs_i < bin_end
        flat_idx = tl.load(
            SORTED_PERM + pid_b * stride_permb + offs_i * stride_permi,
            mask=valid_i,
            other=0,
        )
        partial_ptrs = (
            DPOOL_PARTIAL
            + pid_b * stride_dpb
            + flat_idx[:, None] * stride_dpmk
            + offs_d[None, :] * stride_dpd
        )
        partial = tl.load(
            partial_ptrs,
            mask=valid_i[:, None] & d_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(partial, axis=0)
        i += BLOCK_I

    dpool_offset = pid_b * stride_db + pid_p * stride_dp + offs_d * stride_dd
    tl.store(DPOOL + dpool_offset, acc, mask=d_mask)


# ---------------------------------------------------------------------------
# Python launcher
# ---------------------------------------------------------------------------


def _launch_v4_csa_attention_bwd(
    q: torch.Tensor,  # [B, H, Sq, D]
    k_local: torch.Tensor,  # [B, H, Sq, D]
    v_local: torch.Tensor,  # [B, H, Sq, D]
    gathered: torch.Tensor,  # [B, Sq, K_topk, D]
    sparse_mask: torch.Tensor,  # [B, Sq, K_topk]
    out: torch.Tensor,  # [B, H, Sq, D] (FWD output)
    dout: torch.Tensor,  # [B, H, Sq, D]
    lse: torch.Tensor,  # [B, H, Sq] fp32
    *,
    sink: Optional[torch.Tensor],  # [H] or None
    swa_window: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Launch the V4 CSA attention backward kernel.

    Returns ``(dq, dk_local, dv_local, dgathered, dsink)`` — gradients in
    the input dtype, with ``dsink`` returned only when ``sink is not
    None`` (else ``None``).
    """
    if not q.is_cuda:
        raise ValueError("v4_csa_attention BWD requires CUDA / HIP tensors.")
    if dout.shape != out.shape or out.shape != q.shape:
        raise ValueError(
            "v4_csa_attention BWD shape mismatch: "
            f"out={tuple(out.shape)}, dout={tuple(dout.shape)}, q={tuple(q.shape)}"
        )

    B, HQ, Sq, D = q.shape
    K_topk = gathered.shape[2]

    has_sink = sink is not None

    BLOCK_N = 32
    BLOCK_K = 32
    BLOCK_DMODEL = D

    # Allocate fp32 output buffers for atomic_add. Cast to input dtype
    # before returning.
    dq_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
    dk_local_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
    dv_local_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
    dgathered_fp32 = torch.zeros((B, Sq, K_topk, D), device=q.device, dtype=torch.float32)
    if has_sink:
        dsink_fp32 = torch.zeros((HQ,), device=q.device, dtype=torch.float32)
        sink_arg = sink.to(torch.float32) if sink.dtype != torch.float32 else sink
    else:
        dsink_fp32 = q  # sentinel; HAS_SINK=False inside kernel
        sink_arg = q

    # D scalar = (dout * out).sum(-1) — reuse the dense module's pre-pass
    d_buf = torch.empty((B, HQ, Sq), device=q.device, dtype=torch.float32)
    pre_grid = (triton.cdiv(Sq, BLOCK_N), B * HQ)
    _v4_attention_bwd_preprocess_kernel[pre_grid](
        out,
        dout,
        d_buf,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dout.stride(3),
        d_buf.stride(0),
        d_buf.stride(1),
        d_buf.stride(2),
        Sq,
        HEAD=HQ,
        BLOCK_M=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=1,
    )

    grid = (Sq, B * HQ)
    _v4_csa_attention_bwd_kernel[grid](
        q,
        k_local,
        v_local,
        gathered,
        sparse_mask,
        dout,
        lse,
        d_buf,
        dq_fp32,
        dk_local_fp32,
        dv_local_fp32,
        dgathered_fp32,
        dsink_fp32,
        sink_arg,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_local.stride(0),
        k_local.stride(1),
        k_local.stride(2),
        k_local.stride(3),
        v_local.stride(0),
        v_local.stride(1),
        v_local.stride(2),
        v_local.stride(3),
        gathered.stride(0),
        gathered.stride(1),
        gathered.stride(2),
        gathered.stride(3),
        sparse_mask.stride(0),
        sparse_mask.stride(1),
        sparse_mask.stride(2),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dout.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        d_buf.stride(0),
        d_buf.stride(1),
        d_buf.stride(2),
        dq_fp32.stride(0),
        dq_fp32.stride(1),
        dq_fp32.stride(2),
        dq_fp32.stride(3),
        dk_local_fp32.stride(0),
        dk_local_fp32.stride(1),
        dk_local_fp32.stride(2),
        dk_local_fp32.stride(3),
        dv_local_fp32.stride(0),
        dv_local_fp32.stride(1),
        dv_local_fp32.stride(2),
        dv_local_fp32.stride(3),
        dgathered_fp32.stride(0),
        dgathered_fp32.stride(1),
        dgathered_fp32.stride(2),
        dgathered_fp32.stride(3),
        Sq,
        K_topk,
        float(scale),
        HEAD_Q=HQ,
        SWA_WINDOW=int(swa_window) if swa_window > 0 else 0,
        HAS_SINK=has_sink,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=1,
    )

    dq_out = dq_fp32.to(q.dtype)
    dk_local_out = dk_local_fp32.to(k_local.dtype)
    dv_local_out = dv_local_fp32.to(v_local.dtype)
    dgathered_out = dgathered_fp32.to(gathered.dtype)
    dsink_out = dsink_fp32.to(sink.dtype) if has_sink else None
    return dq_out, dk_local_out, dv_local_out, dgathered_out, dsink_out


def _launch_v4_csa_attention_pool_bwd(
    q: torch.Tensor,  # [B, H, Sq, D]
    k_local: torch.Tensor,  # [B, H, Sq, D]
    v_local: torch.Tensor,  # [B, H, Sq, D]
    pool: torch.Tensor,  # [B, P, D]
    topk_idxs: torch.Tensor,  # [B, Sq, K_topk]
    out: torch.Tensor,  # [B, H, Sq, D]
    dout: torch.Tensor,  # [B, H, Sq, D]
    lse: torch.Tensor,  # [B, H, Sq]
    *,
    sink: Optional[torch.Tensor],
    swa_window: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Launch CSA backward with in-kernel scatter-add into ``pool.grad``."""
    if not q.is_cuda:
        raise ValueError("v4_csa_attention pool BWD requires CUDA / HIP tensors.")
    if dout.shape != out.shape or out.shape != q.shape:
        raise ValueError(
            "v4_csa_attention pool BWD shape mismatch: "
            f"out={tuple(out.shape)}, dout={tuple(dout.shape)}, q={tuple(q.shape)}"
        )
    if pool.dim() != 3:
        raise ValueError(f"v4_csa_attention pool BWD expects pool rank 3 [B, P, D], got {tuple(pool.shape)}.")
    if topk_idxs.dim() != 3:
        raise ValueError(
            f"v4_csa_attention pool BWD expects topk_idxs rank 3 [B, Sq, K], got {tuple(topk_idxs.shape)}."
        )

    B, HQ, Sq, D = q.shape
    Bp, P, Dp = pool.shape
    Bt, Sqt, K_topk = topk_idxs.shape
    if Bp != B or Dp != D or Bt != B or Sqt != Sq:
        raise ValueError(
            "v4_csa_attention pool BWD shape mismatch: "
            f"q={tuple(q.shape)}, pool={tuple(pool.shape)}, topk_idxs={tuple(topk_idxs.shape)}"
        )
    if topk_idxs.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"v4_csa_attention topk_idxs must be int32/int64, got {topk_idxs.dtype}.")

    has_sink = sink is not None

    BLOCK_N = 32
    # Plan-5 P32: sweeping ``BLOCK_K ∈ {32, 64, 128}`` and ``num_warps
    # ∈ {4, 8}`` on the EP8 microbench picked ``BLOCK_K=32`` /
    # ``num_warps=4`` as the cheapest configuration (24.9 ms vs 26.0 ms
    # at the previous defaults). Override via
    # ``PRIMUS_V4_CSA_BWD_BLOCK_K`` for shape-specific tuning.
    BLOCK_K = int(os.getenv("PRIMUS_V4_CSA_BWD_BLOCK_K", "32"))
    # Plan-5 P32: the segreduce / partial path is markedly more
    # register-pressured than the atomic-add gather kernel (the extra
    # ``dpool_partial`` write tile pins 64 × 512 fp32 in flight per
    # warp). A second sweep at ``BLOCK_K_PARTIAL ∈ {8, 16, 32, 64,
    # 128}`` × ``num_warps ∈ {4, 8, 16}`` showed ``BLOCK_K=16`` /
    # ``num_warps=8`` is ~2 ms faster than the gather path's
    # ``BLOCK_K=32`` choice (16.2 ms vs 18.6 ms). Keep the gather
    # path's ``BLOCK_K=32`` since it has its own register profile.
    BLOCK_K_PARTIAL = int(os.getenv("PRIMUS_V4_CSA_BWD_PARTIAL_BLOCK_K", "16"))
    BLOCK_DMODEL = D
    use_split_sparse = os.getenv("PRIMUS_V4_CSA_BWD_SPLIT_SPARSE", "1") != "0"

    # Plan-5 P32: enable stream-level overlap between the local SWA
    # BWD (dq + dk/dv kernels) and the sparse pool BWD kernel. The two
    # paths write to disjoint output buffers (``dq_local_fp32`` /
    # ``dk_local_fp32`` / ``dv_local_fp32`` / ``dsink_fp32`` vs
    # ``dq_sparse_fp32`` / ``dpool_partial``) so the streams can
    # progress concurrently. We sum ``dq_local + dq_sparse`` on the
    # default stream after the join.
    # Plan-5 P32: stream overlap defaults off — the segreduce path
    # already places ``dq`` writes on disjoint buffers and the local
    # kernel cost (~7.6 ms) is shorter than the sparse path (~3 ms in
    # the segreduce variant), so there is little compute to hide. Set
    # ``PRIMUS_V4_CSA_BWD_STREAM_OVERLAP=1`` to re-enable for
    # experimentation. P57 R2 confirmed that enabling overlap
    # actually doubles wall-clock to ~11 ms on the proxy shape — the
    # two paths both saturate HBM read bandwidth, so concurrent
    # execution serializes on memory traffic and adds stream
    # synchronization overhead.
    use_stream_overlap = use_split_sparse and os.getenv("PRIMUS_V4_CSA_BWD_STREAM_OVERLAP", "0") != "0"

    # P57: when the input is bf16/fp16 and we use the split local SWA
    # path (every slab overwritten by tl.store, no atomic_add), keep
    # ``dq / dk_local / dv_local`` in the INPUT dtype so the kernels
    # write directly at the final precision. Eliminates the trailing
    # ~256 MB fp32 → bf16 cast on each of these 3 buffers (~0.5 ms
    # total at the proxy shape) AND halves their HBM write traffic
    # in-kernel (~1 GB → 512 MB for the three buffers). The
    # monolithic local kernel and the gather sparse path still need
    # fp32 because they ``atomic_add`` into the buffer and Triton's
    # bf16 atomic_add is not supported on AMD MI355.
    use_local_split_alloc = use_split_sparse and os.getenv("PRIMUS_V4_ATTN_BWD_USE_SPLIT", "0") == "1"
    if use_local_split_alloc and q.dtype in (torch.bfloat16, torch.float16):
        local_dq_dtype = q.dtype
    else:
        local_dq_dtype = torch.float32
    if use_local_split_alloc:
        dq_fp32 = torch.empty((B, HQ, Sq, D), device=q.device, dtype=local_dq_dtype)
        dk_local_fp32 = torch.empty((B, HQ, Sq, D), device=q.device, dtype=local_dq_dtype)
        dv_local_fp32 = torch.empty((B, HQ, Sq, D), device=q.device, dtype=local_dq_dtype)
    else:
        dq_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
        dk_local_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
        dv_local_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
    # ``dpool_fp32`` is small (2 MB at proxy) so the zero-fill cost
    # is negligible (~20 us); keep ``zeros`` so the gather +
    # atomic_add fallback path stays safe.
    dpool_fp32 = torch.zeros((B, P, D), device=q.device, dtype=torch.float32)
    # The split-sparse paths (both segreduce and gather) always write
    # ``dq_sparse`` via ``tl.store`` from disjoint ``(pid_m,
    # pid_h_block)`` programs, so we use a dedicated buffer and sum
    # at the end. Aliasing to ``dq_fp32`` would clobber the
    # local-SWA contribution. P57: match input dtype (bf16) when
    # ``local_dq_dtype`` is bf16 so the final add and dtype-cast
    # become free.
    if use_split_sparse:
        dq_sparse_fp32 = torch.empty((B, HQ, Sq, D), device=q.device, dtype=local_dq_dtype)
    else:
        dq_sparse_fp32 = dq_fp32
    if has_sink:
        dsink_fp32 = torch.zeros((HQ,), device=q.device, dtype=torch.float32)
        sink_arg = sink.to(torch.float32) if sink.dtype != torch.float32 else sink
    else:
        dsink_fp32 = q
        sink_arg = q

    d_buf = torch.empty((B, HQ, Sq), device=q.device, dtype=torch.float32)
    pre_grid = (triton.cdiv(Sq, BLOCK_N), B * HQ)
    _v4_attention_bwd_preprocess_kernel[pre_grid](
        out,
        dout,
        d_buf,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dout.stride(3),
        d_buf.stride(0),
        d_buf.stride(1),
        d_buf.stride(2),
        Sq,
        HEAD=HQ,
        BLOCK_M=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=1,
    )

    # Plan-5 P32: pre-compute the segreduce inverse index on the
    # default stream BEFORE launching the local kernels. The sort +
    # searchsorted are CPU-light but launch ~0.2 ms of kernels, and
    # placing them BEFORE the local BWD launches lets the sparse
    # stream's wait_event fire as soon as the index + d_buf are
    # ready (instead of waiting until after all local kernels have
    # been queued).
    perm32 = None
    bin_ptr = None
    dpool_partial = None
    use_segreduce = use_split_sparse and os.getenv("PRIMUS_V4_CSA_BWD_SEGREDUCE", "1") == "1"
    if use_segreduce:
        # P57: ``dpool_partial`` defaults to the INPUT dtype when bf16
        # / fp16 — halves HBM write+read traffic on the 4 GiB partial
        # buffer (4 GB fp32 → 2 GB bf16) for the ~production-shape
        # bench, saving ~0.6 ms / step. fp32 inputs keep an fp32
        # partial so the parity tests' tight 1e-4 atol holds.
        # The legacy P32 attempt failed parity because it used bf16
        # unconditionally; gating on the input dtype keeps the
        # bf16-only speed win without regressing fp32 numerics.
        env_dtype = os.getenv("PRIMUS_V4_CSA_BWD_PARTIAL_DTYPE", "")
        if env_dtype == "bf16":
            partial_dtype = torch.bfloat16
        elif env_dtype == "fp16":
            partial_dtype = torch.float16
        elif env_dtype == "fp32":
            partial_dtype = torch.float32
        else:
            # Default: match input dtype for bf16 / fp16, fp32
            # otherwise. ``q.dtype`` is the most reliable proxy for
            # the gradient-tolerance class.
            if q.dtype in (torch.bfloat16, torch.float16):
                partial_dtype = q.dtype
            else:
                partial_dtype = torch.float32
        # P57: experimental ``use_sorted_partial`` flips the partial
        # buffer layout from ``[B, M, K_topk, D]`` to
        # ``[B, MK_sorted, D]``. The downstream segreduce kernel can
        # then read contiguous slices, but the partial kernel writes
        # become scattered (each ``sparse_k`` row goes to its
        # ``inv_perm[orig_flat]`` sorted position). On the EP8 proxy
        # the natural-order partial wins (~6.43 ms vs ~6.82 ms),
        # because the partial kernel is write-bound and contiguous
        # writes coalesce better than scattered ones. Keep the
        # sorted-partial kernels in tree for shapes where the
        # segreduce-read trade-off tips the other way (e.g., tiny
        # ``D`` or very-large ``P`` workloads).
        use_sorted_partial = os.getenv("PRIMUS_V4_CSA_BWD_SORTED_PARTIAL", "0") != "0"
        with torch.no_grad():
            MK = Sq * K_topk
            flat_topk = topk_idxs.contiguous().view(B, MK).to(torch.int32)
            sentinel = torch.full_like(flat_topk, P)
            masked = torch.where((flat_topk >= 0) & (flat_topk < P), flat_topk, sentinel)
            sorted_topk, perm = torch.sort(masked, dim=1, stable=True)
            perm32 = perm.to(torch.int32)
            queries = torch.arange(P + 1, device=q.device, dtype=torch.int32)
            queries = queries.unsqueeze(0).expand(B, -1).contiguous()
            bin_ptr = torch.searchsorted(sorted_topk, queries, right=False).to(torch.int32)
            inv_perm32 = None
            if use_sorted_partial:
                # inv_perm[bid, orig_flat] = sorted_position. Built by
                # scattering ``arange(MK)`` to the ``perm``-indexed
                # positions. ~30 us at the proxy shape.
                inv_perm32 = torch.empty_like(perm32)
                idx_range = torch.arange(MK, device=q.device, dtype=torch.int32).unsqueeze(0).expand(B, -1)
                inv_perm32.scatter_(1, perm.long(), idx_range)
        dpool_partial = torch.empty((B, Sq, K_topk, D), device=q.device, dtype=partial_dtype)

    # Lazily set up an extra CUDA stream for the sparse pool BWD so it
    # can overlap with the local BWD launches (default-stream serial).
    sparse_stream_ctx = None
    if use_stream_overlap:
        sparse_stream = torch.cuda.Stream(device=q.device)
        d_buf_done = torch.cuda.current_stream(q.device).record_event()
        sparse_stream.wait_event(d_buf_done)
        sparse_stream_ctx = torch.cuda.stream(sparse_stream)

    if use_split_sparse:
        # P57: expose local-SWA tuning knobs (BLOCK_M, num_warps,
        # num_stages). The local dq / dkv kernels live in
        # ``v4_attention_bwd.py`` (outside the P57 file scope), but the
        # LAUNCHER picks the block size + warp / stage count, and those
        # parameters dominate the local-path wall clock at the
        # production SWA=128 shape. The new defaults below are tuned
        # specifically for ``B=1, H=64, Sq=4096, D=512, K_topk=512,
        # swa_window=128`` on MI355 and cut the local dq+dkv from
        # ~6.9 ms (P32 defaults: BM=32, BN=32, w=8, s=1) to ~4.1 ms
        # (BM=64, BN=16 → fewer m programs and smaller n tiles to
        # match the small SWA window).
        local_block_m = int(os.getenv("PRIMUS_V4_CSA_BWD_LOCAL_BLOCK_M", "64"))
        local_block_n = int(os.getenv("PRIMUS_V4_CSA_BWD_LOCAL_BLOCK_N", "16"))
        # P57: the dq and dkv kernels prefer different per-axis block
        # sizes (dq iterates ``n`` over a small SWA window so larger
        # ``BLOCK_M`` packs more m-rows per program; dkv iterates
        # ``m`` so its program-axis ``BLOCK_N`` and the inner-axis
        # ``BLOCK_M`` can be tuned independently).
        local_block_m_dq = int(os.getenv("PRIMUS_V4_CSA_BWD_LOCAL_DQ_BLOCK_M", str(local_block_m)))
        local_block_n_dq = int(os.getenv("PRIMUS_V4_CSA_BWD_LOCAL_DQ_BLOCK_N", str(local_block_n)))
        # P57 R2: dkv-specific tuning — ``BLOCK_M_dkv=16 warps=2 stages=1``
        # wins post-R2 scale-defer (the dkv kernel's m-tile prefers
        # smaller block sizes when the sparse partial path runs faster
        # and exposes more concurrent CU slots). BM=16 is ~10 us faster
        # than BM=32 at the proxy shape.
        local_block_m_dkv = int(os.getenv("PRIMUS_V4_CSA_BWD_LOCAL_DKV_BLOCK_M", "16"))
        local_block_n_dkv = int(os.getenv("PRIMUS_V4_CSA_BWD_LOCAL_DKV_BLOCK_N", str(local_block_n)))
        # P57 R2: ``DQ_WARPS=4, DQ_STAGES=2`` wins ~140 us over the
        # R1 ``W=8, S=1`` default. The dq kernel iterates ``n`` over
        # a small SWA=128 window, so 4 warps fit the working set in
        # registers and 2 stages prefetch the next K/V tile while the
        # current MFMA chain executes.
        local_dq_warps = int(os.getenv("PRIMUS_V4_CSA_BWD_LOCAL_DQ_WARPS", "4"))
        local_dq_stages = int(os.getenv("PRIMUS_V4_CSA_BWD_LOCAL_DQ_STAGES", "2"))
        local_dkv_warps = int(os.getenv("PRIMUS_V4_CSA_BWD_LOCAL_DKV_WARPS", "2"))
        local_dkv_stages = int(os.getenv("PRIMUS_V4_CSA_BWD_LOCAL_DKV_STAGES", "1"))
        swa_local = int(swa_window) if swa_window > 0 else 0
        use_local_split = os.getenv("PRIMUS_V4_ATTN_BWD_USE_SPLIT", "0") == "1"
        if use_local_split:
            dq_grid = (triton.cdiv(Sq, local_block_m_dq), B * HQ)
            _v4_attention_bwd_dq_kernel[dq_grid](
                q,
                k_local,
                v_local,
                dout,
                lse,
                d_buf,
                dq_fp32,
                dsink_fp32,
                sink_arg,
                q,  # ADD_MASK sentinel
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k_local.stride(0),
                k_local.stride(1),
                k_local.stride(2),
                k_local.stride(3),
                v_local.stride(0),
                v_local.stride(1),
                v_local.stride(2),
                v_local.stride(3),
                dout.stride(0),
                dout.stride(1),
                dout.stride(2),
                dout.stride(3),
                lse.stride(0),
                lse.stride(1),
                lse.stride(2),
                d_buf.stride(0),
                d_buf.stride(1),
                d_buf.stride(2),
                dq_fp32.stride(0),
                dq_fp32.stride(1),
                dq_fp32.stride(2),
                dq_fp32.stride(3),
                0,
                0,
                Sq,
                Sq,
                float(scale),
                HEAD_Q=HQ,
                HEAD_K=HQ,
                SWA_WINDOW=swa_local,
                HAS_SINK=has_sink,
                HAS_ADD_MASK=False,
                HCA_LOCAL_SEQLEN=0,
                USE_CAUSAL=True,
                BLOCK_M=local_block_m_dq,
                BLOCK_N=local_block_n_dq,
                BLOCK_DMODEL=BLOCK_DMODEL,
                num_warps=local_dq_warps,
                num_stages=local_dq_stages,
            )
            dkv_grid = (triton.cdiv(Sq, local_block_n_dkv), B * HQ)
            _v4_attention_bwd_dkv_kernel[dkv_grid](
                q,
                k_local,
                v_local,
                dout,
                lse,
                d_buf,
                dk_local_fp32,
                dv_local_fp32,
                q,  # ADD_MASK sentinel
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k_local.stride(0),
                k_local.stride(1),
                k_local.stride(2),
                k_local.stride(3),
                v_local.stride(0),
                v_local.stride(1),
                v_local.stride(2),
                v_local.stride(3),
                dout.stride(0),
                dout.stride(1),
                dout.stride(2),
                dout.stride(3),
                lse.stride(0),
                lse.stride(1),
                lse.stride(2),
                d_buf.stride(0),
                d_buf.stride(1),
                d_buf.stride(2),
                dk_local_fp32.stride(0),
                dk_local_fp32.stride(1),
                dk_local_fp32.stride(2),
                dk_local_fp32.stride(3),
                dv_local_fp32.stride(0),
                dv_local_fp32.stride(1),
                dv_local_fp32.stride(2),
                dv_local_fp32.stride(3),
                0,
                0,
                Sq,
                Sq,
                float(scale),
                HEAD_Q=HQ,
                HEAD_K=HQ,
                SWA_WINDOW=swa_local,
                HAS_ADD_MASK=False,
                HCA_LOCAL_SEQLEN=0,
                USE_CAUSAL=True,
                BLOCK_M=local_block_m_dkv,
                BLOCK_N=local_block_n_dkv,
                BLOCK_DMODEL=BLOCK_DMODEL,
                num_warps=local_dkv_warps,
                num_stages=local_dkv_stages,
            )
        else:
            local_grid = (triton.cdiv(Sq, local_block_m), B * HQ)
            _v4_attention_bwd_kernel[local_grid](
                q,
                k_local,
                v_local,
                dout,
                lse,
                d_buf,
                dq_fp32,
                dk_local_fp32,
                dv_local_fp32,
                dsink_fp32,
                sink_arg,
                q,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k_local.stride(0),
                k_local.stride(1),
                k_local.stride(2),
                k_local.stride(3),
                v_local.stride(0),
                v_local.stride(1),
                v_local.stride(2),
                v_local.stride(3),
                dout.stride(0),
                dout.stride(1),
                dout.stride(2),
                dout.stride(3),
                lse.stride(0),
                lse.stride(1),
                lse.stride(2),
                d_buf.stride(0),
                d_buf.stride(1),
                d_buf.stride(2),
                dq_fp32.stride(0),
                dq_fp32.stride(1),
                dq_fp32.stride(2),
                dq_fp32.stride(3),
                dk_local_fp32.stride(0),
                dk_local_fp32.stride(1),
                dk_local_fp32.stride(2),
                dk_local_fp32.stride(3),
                dv_local_fp32.stride(0),
                dv_local_fp32.stride(1),
                dv_local_fp32.stride(2),
                dv_local_fp32.stride(3),
                0,
                0,
                Sq,
                Sq,
                float(scale),
                HEAD_Q=HQ,
                HEAD_K=HQ,
                SWA_WINDOW=swa_local,
                HAS_SINK=has_sink,
                HAS_ADD_MASK=False,
                HCA_LOCAL_SEQLEN=0,
                USE_CAUSAL=True,
                BLOCK_M=local_block_m,
                BLOCK_N=BLOCK_N,
                BLOCK_DMODEL=BLOCK_DMODEL,
                num_warps=8,
                num_stages=1,
            )
    else:
        grid = (Sq, B * HQ)
        _v4_csa_attention_pool_bwd_kernel[grid](
            q,
            k_local,
            v_local,
            pool,
            topk_idxs,
            dout,
            lse,
            d_buf,
            dq_fp32,
            dk_local_fp32,
            dv_local_fp32,
            dpool_fp32,
            dsink_fp32,
            sink_arg,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k_local.stride(0),
            k_local.stride(1),
            k_local.stride(2),
            k_local.stride(3),
            v_local.stride(0),
            v_local.stride(1),
            v_local.stride(2),
            v_local.stride(3),
            pool.stride(0),
            pool.stride(1),
            pool.stride(2),
            topk_idxs.stride(0),
            topk_idxs.stride(1),
            topk_idxs.stride(2),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dout.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            d_buf.stride(0),
            d_buf.stride(1),
            d_buf.stride(2),
            dq_fp32.stride(0),
            dq_fp32.stride(1),
            dq_fp32.stride(2),
            dq_fp32.stride(3),
            dk_local_fp32.stride(0),
            dk_local_fp32.stride(1),
            dk_local_fp32.stride(2),
            dk_local_fp32.stride(3),
            dv_local_fp32.stride(0),
            dv_local_fp32.stride(1),
            dv_local_fp32.stride(2),
            dv_local_fp32.stride(3),
            dpool_fp32.stride(0),
            dpool_fp32.stride(1),
            dpool_fp32.stride(2),
            Sq,
            P,
            K_topk,
            float(scale),
            HEAD_Q=HQ,
            SWA_WINDOW=int(swa_window) if swa_window > 0 else 0,
            HAS_SINK=has_sink,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            BLOCK_DMODEL=BLOCK_DMODEL,
            STORE_DPOOL=(os.getenv("PRIMUS_V4_CSA_BWD_SKIP_DPOOL", "0") != "1"),
            LOCAL_ONLY=False,
            num_warps=4,
            num_stages=1,
        )

    # Plan-5 P32: replace the gather-based sparse pool BWD with a
    # dense-pool BWD that re-uses the split dQ / dK_pool+dV_pool kernels
    # against a ``[Sq, P]`` additive mask synthesised from
    # ``topk_idxs``. The dense path eliminates the per-tile
    # ``tl.atomic_add(DPOOL, ...)`` collisions (``H_blocks × ~2k m's``
    # programs colliding on the shared ``[P, D]`` pool slots) at the
    # cost of iterating the full pool dimension per m-block. With
    # ``P=1024`` and ``K_topk=512`` (visibility ≈ 50%) the extra
    # compute is small but the wall time win is large because atomics
    # are the dominant cost at MI355 ``H=64``.
    #
    # Supported when ``B == 1`` (the proxy shape). For larger batches
    # the kernel mask is per-(b, m), which the legacy gather kernel
    # already handles natively — we fall back to it.
    # Plan-5 P32: an alternative dense-pool path re-uses the split
    # dQ + dK/dV kernels with a ``[Sq, P]`` ``log(count)`` additive
    # mask, swapping ~1 B fp32 atomic adds for ``H * P/BLOCK_N``
    # atomic_adds plus ~2× the compute (full ``P`` vs ``K_topk``).
    # At the proxy shape (``H=64, P=1024, K_topk=512``) the extra
    # compute outweighs the atomic savings (56 ms vs the gather
    # path's 25 ms), so the dense path is kept in tree but **off**
    # by default. Toggle on for shapes with ``K_topk / P`` near 1
    # via ``PRIMUS_V4_CSA_BWD_DENSE_POOL=1``.
    use_dense_pool_sparse = (
        use_split_sparse and B == 1 and os.getenv("PRIMUS_V4_CSA_BWD_DENSE_POOL", "0") == "1"
    )
    # Enter the sparse stream just before launching the sparse pool
    # kernels (default-stream kernels above have already been issued
    # async on the default stream — the GPU can now overlap their
    # execution with the sparse stream's work).
    sparse_stream_entered = False
    if sparse_stream_ctx is not None and use_split_sparse:
        sparse_stream_ctx.__enter__()
        sparse_stream_entered = True

    if use_dense_pool_sparse:
        # Build ``additive_mask[Sq, P]`` from ``topk_idxs[0]``. A finite
        # ``NEG_INF`` is required so the bf16 ``Q @ pool.T`` matmul does
        # not produce ``NaN`` after ``+ -inf``. Duplicate top-K slots
        # (k1, k2 both pointing to pool position p) make the gather BWD
        # accumulate ``2 *`` the contribution of a single key. We
        # collapse the gather visibility into a *count*-weighted mask:
        #
        #   mask[m, p] = log(count[m, p])  if count[m, p] > 0
        #                NEG_INF           otherwise
        #
        # With this mask the dense kernel's ``P[m, p] = exp(qk + log
        # count - LSE) = count * exp(qk - LSE)`` so the dense-pool
        # ``ds`` matches the gather BWD's ``sum_k ds[m, k]`` term-by-
        # term and the dense ``dq`` / ``dpool`` agree with the
        # reference. (Invalid ``-1`` slots contribute zero count.)
        NEG_INF = -1.0e30
        topk_b = topk_idxs[0]  # [Sq, K_topk]
        valid_per_topk = (topk_b >= 0) & (topk_b < P)
        safe_topk = torch.where(valid_per_topk, topk_b, torch.zeros_like(topk_b)).to(torch.int64)
        count = torch.zeros((Sq, P), device=q.device, dtype=torch.float32)
        count.scatter_add_(
            1,
            safe_topk,
            valid_per_topk.to(torch.float32),
        )
        sparse_mask = torch.where(
            count > 0,
            count.clamp_min(1.0).log(),
            torch.full((), NEG_INF, device=q.device, dtype=torch.float32),
        )

        # View pool as a per-head K/V with stride_kh=stride_vh=0 so the
        # split dq/dkv kernels see ``HEAD_K == HEAD_Q`` and parallelise
        # the dkv kernel over the full ``(P/BLOCK_N, B*HEAD_Q)`` grid
        # (2 048 programs at proxy) instead of ``HEAD_K=1`` MQA (32
        # programs) — that path was bandwidth-starved at ``D=512``.
        # Use ``stride_dk = stride_dv = 0`` so each program's ``dk /
        # dv`` write fans into the same shared ``dpool`` slice; the
        # accumulation is via ``tl.atomic_add`` which costs ~4 096
        # call ops total (single atomic per (n_block, head) plus dK +
        # dV) — orders of magnitude less than the gather kernel's
        # ~1 B fp32 atomic adds.
        pool_4d = pool.unsqueeze(1).expand(B, HQ, P, D)  # stride_kh=0

        sparse_block_m = 32
        sparse_block_n = 32
        dq_sparse_grid = (triton.cdiv(Sq, sparse_block_m), B * HQ)
        _v4_attention_bwd_dq_kernel[dq_sparse_grid](
            q,
            pool_4d,
            pool_4d,
            dout,
            lse,
            d_buf,
            dq_sparse_fp32,
            dsink_fp32,  # sentinel only; HAS_SINK=False
            q,
            sparse_mask,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            pool_4d.stride(0),
            pool_4d.stride(1),
            pool_4d.stride(2),
            pool_4d.stride(3),
            pool_4d.stride(0),
            pool_4d.stride(1),
            pool_4d.stride(2),
            pool_4d.stride(3),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dout.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            d_buf.stride(0),
            d_buf.stride(1),
            d_buf.stride(2),
            dq_sparse_fp32.stride(0),
            dq_sparse_fp32.stride(1),
            dq_sparse_fp32.stride(2),
            dq_sparse_fp32.stride(3),
            sparse_mask.stride(0),
            sparse_mask.stride(1),
            Sq,
            P,
            float(scale),
            HEAD_Q=HQ,
            HEAD_K=HQ,
            SWA_WINDOW=0,
            HAS_SINK=False,
            HAS_ADD_MASK=True,
            HCA_LOCAL_SEQLEN=0,
            USE_CAUSAL=False,
            BLOCK_M=sparse_block_m,
            BLOCK_N=sparse_block_n,
            BLOCK_DMODEL=BLOCK_DMODEL,
            ACCUMULATE=True,
            num_warps=8,
            num_stages=1,
        )
        # Single shared ``dpool`` slice that all ``(head, n_block)``
        # programs atomic_add into. Reshape with an extra head axis of
        # stride 0 so the dkv kernel can index it as ``[B, HQ, P, D]``
        # without per-head storage; the actual underlying tensor is
        # still ``[B, P, D]``.
        dk_pool_shared = dpool_fp32.unsqueeze(1).expand(B, HQ, P, D)
        dv_pool_shared = dpool_fp32.unsqueeze(1).expand(B, HQ, P, D)
        dkv_sparse_grid = (triton.cdiv(P, sparse_block_n), B * HQ)
        _v4_attention_bwd_dkv_kernel[dkv_sparse_grid](
            q,
            pool_4d,
            pool_4d,
            dout,
            lse,
            d_buf,
            dk_pool_shared,
            dv_pool_shared,
            sparse_mask,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            pool_4d.stride(0),
            pool_4d.stride(1),
            pool_4d.stride(2),
            pool_4d.stride(3),
            pool_4d.stride(0),
            pool_4d.stride(1),
            pool_4d.stride(2),
            pool_4d.stride(3),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dout.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            d_buf.stride(0),
            d_buf.stride(1),
            d_buf.stride(2),
            dk_pool_shared.stride(0),
            dk_pool_shared.stride(1),
            dk_pool_shared.stride(2),
            dk_pool_shared.stride(3),
            dv_pool_shared.stride(0),
            dv_pool_shared.stride(1),
            dv_pool_shared.stride(2),
            dv_pool_shared.stride(3),
            sparse_mask.stride(0),
            sparse_mask.stride(1),
            Sq,
            P,
            float(scale),
            HEAD_Q=HQ,
            HEAD_K=HQ,
            SWA_WINDOW=0,
            HAS_ADD_MASK=True,
            HCA_LOCAL_SEQLEN=0,
            USE_CAUSAL=False,
            BLOCK_M=sparse_block_m,
            BLOCK_N=sparse_block_n,
            BLOCK_DMODEL=BLOCK_DMODEL,
            ATOMIC_REDUCE=True,
            num_warps=8,
            num_stages=1,
        )
        # ``dpool_fp32`` already holds the summed dk+dv contributions
        # because both kernel views aliased the same ``[B, P, D]``
        # storage via ``stride_kh=stride_vh=0`` and the kernel does
        # both ``atomic_add(dk_ptrs, ...)`` and ``atomic_add(dv_ptrs,
        # ...)``.
    elif use_split_sparse:
        # P57: BLOCK_H=32, num_warps=4, num_stages=2 wins the proxy
        # sweep over the P32 default (BLOCK_H=64, num_warps=8, stages=1).
        # Doubling the head-axis grid (HQ=64 -> 2 h-blocks per m) lifts
        # MI355 occupancy, while fewer warps + 2 stages reduces register
        # pressure per warp (the dq[BLOCK_H, BLOCK_DMODEL]=fp32
        # accumulator dominates VGPR live range).
        BLOCK_H = int(os.getenv("PRIMUS_V4_CSA_BWD_SPARSE_BLOCK_H", "32"))
        # Plan-5 P32: ``PRIMUS_V4_CSA_BWD_SEGREDUCE=1`` is now the
        # default — wins both the standalone CSA BWD microbench
        # (16.31 ms vs 24.83 ms gather/atomic) and the EP8 proxy
        # (578 ms vs 665 ms / iter) after the P32 dual-RoPE bf16-cast
        # fix.  Pre-fix, ``apply_interleaved_partial_rope`` was
        # promoting Q/K to fp32 (cos/sin came from
        # ``position_ids.float()`` and bf16*fp32=fp32), which 2x'd
        # Q/K HBM traffic, inflated *every* attention kernel time
        # 1.8-7x in the proxy trace, and made the gather + atomic
        # path look faster end-to-end purely because the segreduce
        # 4 GiB partial buffer competed against artificially-bloated
        # attention traffic. Setting ``PRIMUS_V4_CSA_BWD_SEGREDUCE=0``
        # falls back to gather + atomic for kernel-tuning. The
        # segmented-reduction path writes per-visit dpool
        # contributions to a compact ``[B, M, K_topk, D]`` partial
        # buffer (no atomics) and then folds them into
        # ``dpool[B, P, D]`` via a sorted inverse index. ``perm32``,
        # ``bin_ptr`` and ``dpool_partial`` were built above on the
        # default stream BEFORE the local kernels were launched, so
        # they're ready by the time the sparse stream gets here.
        # P57: experimental split-kernel path (opt-in).
        # Sub-kernels:
        #
        #   _v4_csa_attention_pool_sparse_bwd_dq_only_kernel
        #   _v4_csa_attention_pool_sparse_bwd_dpool_only_kernel
        #
        # vs the joint kernel
        # ``_v4_csa_attention_pool_sparse_bwd_partial_kernel``. The
        # split path frees the live-VGPR footprint of each sub-kernel
        # but pays a 2× read on Q / dout / lse / dvec / pool /
        # topk_idxs. On the EP8 proxy shape the joint kernel still
        # wins (~3.7 ms vs ~4.5 ms split), so split is opt-in via
        # ``PRIMUS_V4_CSA_BWD_SPLIT_DQ_DPOOL=1`` and the joint kernel
        # remains the default. Keeping the split sub-kernels in tree
        # for future shapes where the read amplification is cheaper
        # (smaller H or D, or compute-bound regimes).
        use_split_dq_dpool = os.getenv("PRIMUS_V4_CSA_BWD_SPLIT_DQ_DPOOL", "0") != "0"
        if use_segreduce and use_split_dq_dpool:
            sparse_grid = (Sq, triton.cdiv(HQ, BLOCK_H), B)
            dq_warps = int(os.getenv("PRIMUS_V4_CSA_BWD_PARTIAL_DQ_WARPS", "4"))
            dq_stages = int(os.getenv("PRIMUS_V4_CSA_BWD_PARTIAL_DQ_STAGES", "2"))
            dpool_warps = int(os.getenv("PRIMUS_V4_CSA_BWD_PARTIAL_DPOOL_WARPS", "4"))
            dpool_stages = int(os.getenv("PRIMUS_V4_CSA_BWD_PARTIAL_DPOOL_STAGES", "2"))
            _v4_csa_attention_pool_sparse_bwd_dq_only_kernel[sparse_grid](
                q,
                pool,
                topk_idxs,
                dout,
                lse,
                d_buf,
                dq_sparse_fp32,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                pool.stride(0),
                pool.stride(1),
                pool.stride(2),
                topk_idxs.stride(0),
                topk_idxs.stride(1),
                topk_idxs.stride(2),
                dout.stride(0),
                dout.stride(1),
                dout.stride(2),
                dout.stride(3),
                lse.stride(0),
                lse.stride(1),
                lse.stride(2),
                d_buf.stride(0),
                d_buf.stride(1),
                d_buf.stride(2),
                dq_sparse_fp32.stride(0),
                dq_sparse_fp32.stride(1),
                dq_sparse_fp32.stride(2),
                dq_sparse_fp32.stride(3),
                Sq,
                P,
                K_topk,
                float(scale),
                HEAD_Q=HQ,
                BLOCK_H=BLOCK_H,
                BLOCK_K=BLOCK_K_PARTIAL,
                BLOCK_DMODEL=BLOCK_DMODEL,
                num_warps=dq_warps,
                num_stages=dq_stages,
            )
            _v4_csa_attention_pool_sparse_bwd_dpool_only_kernel[sparse_grid](
                q,
                pool,
                topk_idxs,
                dout,
                lse,
                d_buf,
                dpool_partial,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                pool.stride(0),
                pool.stride(1),
                pool.stride(2),
                topk_idxs.stride(0),
                topk_idxs.stride(1),
                topk_idxs.stride(2),
                dout.stride(0),
                dout.stride(1),
                dout.stride(2),
                dout.stride(3),
                lse.stride(0),
                lse.stride(1),
                lse.stride(2),
                d_buf.stride(0),
                d_buf.stride(1),
                d_buf.stride(2),
                dpool_partial.stride(0),
                dpool_partial.stride(1),
                dpool_partial.stride(2),
                dpool_partial.stride(3),
                Sq,
                P,
                K_topk,
                float(scale),
                HEAD_Q=HQ,
                BLOCK_H=BLOCK_H,
                BLOCK_K=BLOCK_K_PARTIAL,
                BLOCK_DMODEL=BLOCK_DMODEL,
                num_warps=dpool_warps,
                num_stages=dpool_stages,
            )
            # P57: segreduce reduction (shared between split and joint
            # partial paths). Reduces ``dpool_partial[B, MK, D]`` into
            # ``dpool[B, P, D]`` using the sorted inverse index.
            dpool_partial_flat = dpool_partial.view(B, Sq * K_topk, D)
            block_d_seg = int(os.getenv("PRIMUS_V4_CSA_BWD_SEGREDUCE_BLOCK_D", "512"))
            block_i_seg = int(os.getenv("PRIMUS_V4_CSA_BWD_SEGREDUCE_BLOCK_I", "64"))
            seg_grid = (P, triton.cdiv(D, block_d_seg), B)
            _v4_csa_attention_pool_segreduce_kernel[seg_grid](
                dpool_partial_flat,
                perm32,
                bin_ptr,
                dpool_fp32,
                dpool_partial_flat.stride(0),
                dpool_partial_flat.stride(1),
                dpool_partial_flat.stride(2),
                perm32.stride(0),
                perm32.stride(1),
                bin_ptr.stride(0),
                bin_ptr.stride(1),
                dpool_fp32.stride(0),
                dpool_fp32.stride(1),
                dpool_fp32.stride(2),
                P,
                D,
                BLOCK_D=block_d_seg,
                BLOCK_I=block_i_seg,
                num_warps=int(os.getenv("PRIMUS_V4_CSA_BWD_SEGREDUCE_WARPS", "4")),
                num_stages=int(os.getenv("PRIMUS_V4_CSA_BWD_SEGREDUCE_STAGES", "2")),
            )
        elif use_segreduce and use_sorted_partial:
            sparse_grid = (Sq, triton.cdiv(HQ, BLOCK_H), B)
            dpool_partial_flat = dpool_partial.view(B, Sq * K_topk, D)
            _v4_csa_attention_pool_sparse_bwd_partial_sorted_kernel[sparse_grid](
                q,
                pool,
                topk_idxs,
                inv_perm32,
                dout,
                lse,
                d_buf,
                dq_sparse_fp32,
                dpool_partial_flat,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                pool.stride(0),
                pool.stride(1),
                pool.stride(2),
                topk_idxs.stride(0),
                topk_idxs.stride(1),
                topk_idxs.stride(2),
                inv_perm32.stride(0),
                inv_perm32.stride(1),
                dout.stride(0),
                dout.stride(1),
                dout.stride(2),
                dout.stride(3),
                lse.stride(0),
                lse.stride(1),
                lse.stride(2),
                d_buf.stride(0),
                d_buf.stride(1),
                d_buf.stride(2),
                dq_sparse_fp32.stride(0),
                dq_sparse_fp32.stride(1),
                dq_sparse_fp32.stride(2),
                dq_sparse_fp32.stride(3),
                dpool_partial_flat.stride(0),
                dpool_partial_flat.stride(1),
                dpool_partial_flat.stride(2),
                Sq,
                P,
                K_topk,
                float(scale),
                HEAD_Q=HQ,
                BLOCK_H=BLOCK_H,
                BLOCK_K=BLOCK_K_PARTIAL,
                BLOCK_DMODEL=BLOCK_DMODEL,
                num_warps=int(os.getenv("PRIMUS_V4_CSA_BWD_PARTIAL_WARPS", "4")),
                num_stages=int(os.getenv("PRIMUS_V4_CSA_BWD_PARTIAL_STAGES", "2")),
            )
            # Sequential segreduce — no perm lookup.
            block_d_seg = int(os.getenv("PRIMUS_V4_CSA_BWD_SEGREDUCE_BLOCK_D", "512"))
            block_i_seg = int(os.getenv("PRIMUS_V4_CSA_BWD_SEGREDUCE_BLOCK_I", "64"))
            seg_grid = (P, triton.cdiv(D, block_d_seg), B)
            _v4_csa_attention_pool_segreduce_sequential_kernel[seg_grid](
                dpool_partial_flat,
                bin_ptr,
                dpool_fp32,
                dpool_partial_flat.stride(0),
                dpool_partial_flat.stride(1),
                dpool_partial_flat.stride(2),
                bin_ptr.stride(0),
                bin_ptr.stride(1),
                dpool_fp32.stride(0),
                dpool_fp32.stride(1),
                dpool_fp32.stride(2),
                P,
                D,
                BLOCK_D=block_d_seg,
                BLOCK_I=block_i_seg,
                num_warps=int(os.getenv("PRIMUS_V4_CSA_BWD_SEGREDUCE_WARPS", "4")),
                num_stages=int(os.getenv("PRIMUS_V4_CSA_BWD_SEGREDUCE_STAGES", "2")),
            )
        elif use_segreduce:
            sparse_grid = (Sq, triton.cdiv(HQ, BLOCK_H), B)
            _v4_csa_attention_pool_sparse_bwd_partial_kernel[sparse_grid](
                q,
                pool,
                topk_idxs,
                dout,
                lse,
                d_buf,
                dq_sparse_fp32,
                dpool_partial,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                pool.stride(0),
                pool.stride(1),
                pool.stride(2),
                topk_idxs.stride(0),
                topk_idxs.stride(1),
                topk_idxs.stride(2),
                dout.stride(0),
                dout.stride(1),
                dout.stride(2),
                dout.stride(3),
                lse.stride(0),
                lse.stride(1),
                lse.stride(2),
                d_buf.stride(0),
                d_buf.stride(1),
                d_buf.stride(2),
                dq_sparse_fp32.stride(0),
                dq_sparse_fp32.stride(1),
                dq_sparse_fp32.stride(2),
                dq_sparse_fp32.stride(3),
                dpool_partial.stride(0),
                dpool_partial.stride(1),
                dpool_partial.stride(2),
                dpool_partial.stride(3),
                Sq,
                P,
                K_topk,
                float(scale),
                HEAD_Q=HQ,
                BLOCK_H=BLOCK_H,
                BLOCK_K=BLOCK_K_PARTIAL,
                BLOCK_DMODEL=BLOCK_DMODEL,
                num_warps=int(os.getenv("PRIMUS_V4_CSA_BWD_PARTIAL_WARPS", "4")),
                num_stages=int(os.getenv("PRIMUS_V4_CSA_BWD_PARTIAL_STAGES", "3")),
            )
            # Same segreduce reduction as the split path above.
            dpool_partial_flat = dpool_partial.view(B, Sq * K_topk, D)
            block_d_seg = int(os.getenv("PRIMUS_V4_CSA_BWD_SEGREDUCE_BLOCK_D", "512"))
            block_i_seg = int(os.getenv("PRIMUS_V4_CSA_BWD_SEGREDUCE_BLOCK_I", "64"))
            seg_grid = (P, triton.cdiv(D, block_d_seg), B)
            _v4_csa_attention_pool_segreduce_kernel[seg_grid](
                dpool_partial_flat,
                perm32,
                bin_ptr,
                dpool_fp32,
                dpool_partial_flat.stride(0),
                dpool_partial_flat.stride(1),
                dpool_partial_flat.stride(2),
                perm32.stride(0),
                perm32.stride(1),
                bin_ptr.stride(0),
                bin_ptr.stride(1),
                dpool_fp32.stride(0),
                dpool_fp32.stride(1),
                dpool_fp32.stride(2),
                P,
                D,
                BLOCK_D=block_d_seg,
                BLOCK_I=block_i_seg,
                num_warps=int(os.getenv("PRIMUS_V4_CSA_BWD_SEGREDUCE_WARPS", "4")),
                num_stages=int(os.getenv("PRIMUS_V4_CSA_BWD_SEGREDUCE_STAGES", "2")),
            )
        else:
            # Plan-5 P32: legacy gather-sparse path. Keeps the atomic_add
            # to a shared ``dpool[B, P, D]`` buffer; this is the shipped
            # default. The segreduce path is opt-in via
            # ``PRIMUS_V4_CSA_BWD_SEGREDUCE=1`` for kernel-level perf
            # experiments on the CSA microbench.
            n_part_env = int(os.getenv("PRIMUS_V4_CSA_BWD_DPOOL_PARTITIONS", "1"))
            n_part = max(1, min(n_part_env, Sq))
            while Sq % n_part != 0 and n_part > 1:
                n_part -= 1
            partition_size = Sq // n_part
            if n_part > 1:
                dpool_partial = torch.zeros((n_part, B, P, D), device=q.device, dtype=torch.float32)
            else:
                dpool_partial = dpool_fp32.unsqueeze(0)
            sparse_grid = (Sq, triton.cdiv(HQ, BLOCK_H), B)
            _v4_csa_attention_pool_sparse_bwd_kernel[sparse_grid](
                q,
                pool,
                topk_idxs,
                dout,
                lse,
                d_buf,
                dq_sparse_fp32,
                dpool_partial,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                pool.stride(0),
                pool.stride(1),
                pool.stride(2),
                topk_idxs.stride(0),
                topk_idxs.stride(1),
                topk_idxs.stride(2),
                dout.stride(0),
                dout.stride(1),
                dout.stride(2),
                dout.stride(3),
                lse.stride(0),
                lse.stride(1),
                lse.stride(2),
                d_buf.stride(0),
                d_buf.stride(1),
                d_buf.stride(2),
                dq_sparse_fp32.stride(0),
                dq_sparse_fp32.stride(1),
                dq_sparse_fp32.stride(2),
                dq_sparse_fp32.stride(3),
                dpool_partial.stride(0),
                dpool_partial.stride(1),
                dpool_partial.stride(2),
                dpool_partial.stride(3),
                partition_size,
                Sq,
                P,
                K_topk,
                float(scale),
                HEAD_Q=HQ,
                BLOCK_H=BLOCK_H,
                BLOCK_K=BLOCK_K,
                BLOCK_DMODEL=BLOCK_DMODEL,
                STORE_DPOOL=(os.getenv("PRIMUS_V4_CSA_BWD_SKIP_DPOOL_ATOMIC", "0") != "1"),
                num_warps=int(os.getenv("PRIMUS_V4_CSA_BWD_SPARSE_WARPS", "4")),
                num_stages=int(os.getenv("PRIMUS_V4_CSA_BWD_SPARSE_STAGES", "1")),
            )
            if n_part > 1:
                dpool_fp32 = dpool_partial.sum(dim=0)

    # Exit the sparse stream context (if entered) and join it with the
    # default stream before the final dtype casts so the dq accumulator
    # below sees both the local and sparse contributions.
    if sparse_stream_entered:
        sparse_done = torch.cuda.current_stream(q.device).record_event()
        sparse_stream_ctx.__exit__(None, None, None)
        torch.cuda.current_stream(q.device).wait_event(sparse_done)

    # The split-sparse paths write the sparse contribution to a
    # dedicated ``dq_sparse_fp32`` buffer; combine it with the local
    # SWA contribution before the dtype cast. In-place add saves the
    # extra ~128 MB allocation that ``+`` would do.
    if use_split_sparse:
        dq_fp32.add_(dq_sparse_fp32)

    # P57: if the local buffers are already in input dtype, ``.to``
    # is a no-op view; only fp32 buffers actually do a cast.
    dq_out = dq_fp32 if dq_fp32.dtype == q.dtype else dq_fp32.to(q.dtype)
    dk_local_out = dk_local_fp32 if dk_local_fp32.dtype == k_local.dtype else dk_local_fp32.to(k_local.dtype)
    dv_local_out = dv_local_fp32 if dv_local_fp32.dtype == v_local.dtype else dv_local_fp32.to(v_local.dtype)
    dpool_out = dpool_fp32 if dpool_fp32.dtype == pool.dtype else dpool_fp32.to(pool.dtype)
    dsink_out = dsink_fp32.to(sink.dtype) if has_sink else None
    return dq_out, dk_local_out, dv_local_out, dpool_out, dsink_out


__all__ = [
    "_v4_csa_attention_bwd_kernel",
    "_v4_csa_attention_pool_bwd_kernel",
    "_v4_csa_attention_pool_sparse_bwd_kernel",
    "_launch_v4_csa_attention_bwd",
    "_launch_v4_csa_attention_pool_bwd",
]
