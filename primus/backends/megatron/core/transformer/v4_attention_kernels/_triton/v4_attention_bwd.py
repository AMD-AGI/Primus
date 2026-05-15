###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""V4 attention backward Triton kernel (plan-4 P25, ``compress_ratio in {0, 128}``).

Two-kernel design:

* :func:`_v4_attention_bwd_preprocess_kernel` — computes the per-query
  diagonal scalar ``D = sum_d (dout[..., d] * out[..., d])`` in fp32.
  This is the standard FlashAttention-2 BWD pre-pass; it lets the main
  kernel skip materialising ``dP @ V^T`` as a tile-wise sum.
* :func:`_v4_attention_bwd_kernel` — main BWD pass. Parallelizes over
  query blocks (one program per ``m_block × batch × head_q``) and:

    1. re-materialises ``P = exp(qk - LSE)`` in fp32 from saved
       ``Q / K / V / LSE`` (so the [Sq, Sk] ``P`` matrix is never
       stored),
    2. computes ``dS = P * (dP - D)`` where ``dP = dout @ V^T``,
    3. accumulates ``dQ`` in registers (no atomic — one program per
       m-block writes its dQ tile straight to global),
    4. atomic-adds ``dK = scale * dS^T @ Q`` and ``dV = P^T @ dout``
       into the global dK / dV buffers,
    5. atomic-adds the sink gradient
       ``dsink_h += -sum_t (P_sink_t * D_t)`` per query head.

The launcher allocates ``dQ / dK / dV / dsink`` as fp32 buffers (so
``tl.atomic_add`` works regardless of input dtype) and casts back to
the input dtype on return.

dtype contract:

* All matmuls run on tensor cores in input dtype with fp32 accumulator.
* The online ``P / dP / dS`` re-materialisation is fp32 (matches the
  FWD's softmax-in-fp32 contract).
* Output gradients are returned in input dtype (cast from fp32
  buffers).

Tile choice: ``BLOCK_M = BLOCK_N = 32`` at ``head_dim = 512`` so the
peak SMEM (Q + K + V + dout = 4 × 32 × 512 × 2 = 128 KiB) fits under
MI355's 160 KiB SMEM budget with some headroom for register pressure.
P25 perf follow-up may explore ``BLOCK_M = BLOCK_N = 64`` once the
correctness baseline is locked.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Pre-pass kernel (D scalar)
# ---------------------------------------------------------------------------


@triton.jit
def _v4_attention_bwd_preprocess_kernel(
    OUT,
    DOUT,
    D,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_db,
    stride_dh,
    stride_dm,
    seqlen_q,
    HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Compute ``D[b, h, m] = sum_d (dout[b, h, m, d] * out[b, h, m, d])``."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    bid = pid_bh // HEAD
    hid = pid_bh % HEAD

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    out_ptrs = (
        OUT + bid * stride_ob + hid * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    )
    dout_ptrs = (
        DOUT
        + bid * stride_dob
        + hid * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :] * stride_dod
    )
    mask = offs_m[:, None] < seqlen_q
    o = tl.load(out_ptrs, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(dout_ptrs, mask=mask, other=0.0).to(tl.float32)
    d = tl.sum(o * do, 1)

    d_ptrs = D + bid * stride_db + hid * stride_dh + offs_m * stride_dm
    tl.store(d_ptrs, d, mask=offs_m < seqlen_q)


# ---------------------------------------------------------------------------
# Main BWD kernel
# ---------------------------------------------------------------------------


@triton.jit
def _v4_attention_bwd_kernel(
    Q,
    K,
    V,
    DOUT,
    LSE,
    D,
    DQ,  # fp32 buffer [B, H, Sq, D]
    DK,  # fp32 buffer [B, K_H, Sk, D]
    DV,  # fp32 buffer [B, K_H, Sk, D]
    DSINK,  # fp32 buffer [H] or sentinel
    SINK,  # [H] fp32 or sentinel
    ADD_MASK,  # [Sq, Sk] or sentinel
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
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
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dvd,
    stride_ms,
    stride_mn,
    seqlen_q,
    seqlen_k,
    sm_scale,
    HEAD_Q: tl.constexpr,
    HEAD_K: tl.constexpr,
    SWA_WINDOW: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_ADD_MASK: tl.constexpr,
    HCA_LOCAL_SEQLEN: tl.constexpr,
    USE_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """V4 attention BWD (single kernel, parallelize over m-blocks)."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    bid = pid_bh // HEAD_Q
    qhid = pid_bh % HEAD_Q
    if HEAD_K == HEAD_Q:
        khid = qhid
    else:
        khid = 0

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    NEG_INF: tl.constexpr = -1.0e30

    # Load Q, dout, LSE, D for this m-block.
    q_ptrs = (
        Q + bid * stride_qb + qhid * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    )
    dout_ptrs = (
        DOUT
        + bid * stride_dob
        + qhid * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :] * stride_dod
    )
    lse_ptrs = LSE + bid * stride_lb + qhid * stride_lh + offs_m * stride_lm
    dvec_ptrs = D + bid * stride_db + qhid * stride_dh + offs_m * stride_dm

    q_load_mask = offs_m[:, None] < seqlen_q
    q = tl.load(q_ptrs, mask=q_load_mask, other=0.0)
    dout = tl.load(dout_ptrs, mask=q_load_mask, other=0.0)
    lse = tl.load(lse_ptrs, mask=offs_m < seqlen_q, other=0.0)
    dvec = tl.load(dvec_ptrs, mask=offs_m < seqlen_q, other=0.0)

    # dQ accumulator (fp32, kept in registers across the n-loop)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Sink gradient — accumulate per-query contribution then atomic_add
    # into DSINK[qhid] at the end. The sink contributes
    # ``dsink_h += sum_t -P_sink_t * D_t`` because dlogits_sink = P_sink * (0 - D).
    if HAS_SINK:
        sink_h = tl.load(SINK + qhid).to(tl.float32)
        # P_sink for each query in the m-block, masked for boundary rows
        p_sink = tl.exp(sink_h - lse)
        p_sink_masked = tl.where(offs_m < seqlen_q, p_sink, 0.0)
        dvec_masked = tl.where(offs_m < seqlen_q, dvec, 0.0)
        dsink_contrib = tl.sum(-p_sink_masked * dvec_masked)
        tl.atomic_add(DSINK + qhid, dsink_contrib)

    # Determine n-loop bounds (matches FWD's P30 SWA tile pruning).
    n_loop_start = 0
    if HAS_ADD_MASK and HCA_LOCAL_SEQLEN == 0:
        n_loop_end = seqlen_k
    elif SWA_WINDOW > 0:
        n_loop_start = pid_m * BLOCK_M - SWA_WINDOW + 1
        if n_loop_start < 0:
            n_loop_start = 0
        n_loop_start = (n_loop_start // BLOCK_N) * BLOCK_N
        n_loop_end = (pid_m + 1) * BLOCK_M
        local_end = HCA_LOCAL_SEQLEN if HCA_LOCAL_SEQLEN > 0 else seqlen_k
        if n_loop_end > local_end:
            n_loop_end = local_end
    elif USE_CAUSAL:
        n_loop_end = (pid_m + 1) * BLOCK_M
        if n_loop_end > seqlen_k:
            n_loop_end = seqlen_k
    else:
        n_loop_end = seqlen_k

    for n_start in range(n_loop_start, n_loop_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)

        k_ptrs = (
            K + bid * stride_kb + khid * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            V + bid * stride_vb + khid * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        )
        kv_load_mask = offs_n[:, None] < seqlen_k
        k = tl.load(k_ptrs, mask=kv_load_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_load_mask, other=0.0)

        # qk = Q @ K.T * scale + mask (re-materialise in fp32)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if HAS_ADD_MASK and HCA_LOCAL_SEQLEN == 0:
            mask_ptrs = ADD_MASK + offs_m[:, None] * stride_ms + offs_n[None, :] * stride_mn
            mask_load_mask = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
            add_bias = tl.load(mask_ptrs, mask=mask_load_mask, other=0.0).to(tl.float32)
            qk = qk + add_bias
        else:
            if SWA_WINDOW > 0:
                in_window = (offs_n[None, :] >= offs_m[:, None] - SWA_WINDOW + 1) & (
                    offs_n[None, :] <= offs_m[:, None]
                )
                qk = tl.where(in_window, qk, NEG_INF)
            elif USE_CAUSAL:
                qk = tl.where(offs_n[None, :] <= offs_m[:, None], qk, NEG_INF)
        qk = tl.where(offs_n[None, :] < seqlen_k, qk, NEG_INF)
        qk = tl.where(offs_m[:, None] < seqlen_q, qk, NEG_INF)

        # P = exp(qk - LSE) in fp32. For boundary rows / fully-masked
        # rows, lse is 0 (loaded with mask) and qk is NEG_INF, so
        # exp(NEG_INF - 0) = 0 — no contribution. ✓
        p = tl.exp(qk - lse[:, None])

        # dP = dout @ V.T (fp32 accumulator)
        dp = tl.dot(dout, tl.trans(v))

        # dS = P * (dP - D)
        ds = p * (dp - dvec[:, None])

        # dQ += dS @ K * scale
        dq += tl.dot(ds.to(k.dtype), k) * sm_scale

        # dK += scale * dS.T @ Q  (atomic_add into fp32 DK buffer)
        dk_contrib = tl.dot(tl.trans(ds.to(q.dtype)), q) * sm_scale
        dk_ptrs = (
            DK
            + bid * stride_dkb
            + khid * stride_dkh
            + offs_n[:, None] * stride_dkn
            + offs_d[None, :] * stride_dkd
        )
        dk_mask = offs_n[:, None] < seqlen_k
        tl.atomic_add(dk_ptrs, dk_contrib, mask=dk_mask, sem="relaxed")

        # dV += P.T @ dout  (atomic_add into fp32 DV buffer)
        dv_contrib = tl.dot(tl.trans(p.to(dout.dtype)), dout)
        dv_ptrs = (
            DV
            + bid * stride_dvb
            + khid * stride_dvh
            + offs_n[:, None] * stride_dvn
            + offs_d[None, :] * stride_dvd
        )
        dv_mask = offs_n[:, None] < seqlen_k
        tl.atomic_add(dv_ptrs, dv_contrib, mask=dv_mask, sem="relaxed")

    if HAS_ADD_MASK and HCA_LOCAL_SEQLEN > 0:
        for n_start in range(HCA_LOCAL_SEQLEN, seqlen_k, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            pool_n = offs_n - HCA_LOCAL_SEQLEN

            k_ptrs = (
                K
                + bid * stride_kb
                + khid * stride_kh
                + offs_n[:, None] * stride_kn
                + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                V
                + bid * stride_vb
                + khid * stride_vh
                + offs_n[:, None] * stride_vn
                + offs_d[None, :] * stride_vd
            )
            kv_load_mask = offs_n[:, None] < seqlen_k
            k = tl.load(k_ptrs, mask=kv_load_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_load_mask, other=0.0)

            qk = tl.dot(q, tl.trans(k)) * sm_scale
            mask_ptrs = ADD_MASK + offs_m[:, None] * stride_ms + pool_n[None, :] * stride_mn
            mask_load_mask = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
            add_bias = tl.load(mask_ptrs, mask=mask_load_mask, other=0.0).to(tl.float32)
            qk = qk + add_bias
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, NEG_INF)
            qk = tl.where(offs_m[:, None] < seqlen_q, qk, NEG_INF)

            p = tl.exp(qk - lse[:, None])
            dp = tl.dot(dout, tl.trans(v))
            ds = p * (dp - dvec[:, None])

            dq += tl.dot(ds.to(k.dtype), k) * sm_scale

            dk_contrib = tl.dot(tl.trans(ds.to(q.dtype)), q) * sm_scale
            dk_ptrs = (
                DK
                + bid * stride_dkb
                + khid * stride_dkh
                + offs_n[:, None] * stride_dkn
                + offs_d[None, :] * stride_dkd
            )
            dk_mask = offs_n[:, None] < seqlen_k
            tl.atomic_add(dk_ptrs, dk_contrib, mask=dk_mask, sem="relaxed")

            dv_contrib = tl.dot(tl.trans(p.to(dout.dtype)), dout)
            dv_ptrs = (
                DV
                + bid * stride_dvb
                + khid * stride_dvh
                + offs_n[:, None] * stride_dvn
                + offs_d[None, :] * stride_dvd
            )
            dv_mask = offs_n[:, None] < seqlen_k
            tl.atomic_add(dv_ptrs, dv_contrib, mask=dv_mask, sem="relaxed")

    # Store dQ (fp32 buffer; launcher casts back to input dtype on return).
    dq_ptrs = (
        DQ
        + bid * stride_dqb
        + qhid * stride_dqh
        + offs_m[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd
    )
    tl.store(dq_ptrs, dq, mask=offs_m[:, None] < seqlen_q)


# ---------------------------------------------------------------------------
# Plan-5 P32 split-kernel BWD — dQ kernel + dK/dV kernel (no atomics)
#
# The monolithic ``_v4_attention_bwd_kernel`` parallelises over m-blocks
# and ``tl.atomic_add``s into ``dK / dV``. With ``H=64`` heads ×
# ``SWA_WINDOW / BLOCK_M = 4`` m-blocks per K position, every K position
# is touched by ~256 concurrent atomics; even at "relaxed" semantics on
# MI355 these serialise through the L2 atomic engine.
#
# P32 splits the BWD into two kernels that each write their own output
# slice with NO atomics:
#
#  * ``_v4_attention_bwd_dq_kernel`` — one program per ``(b, qhid, m_block)``;
#    accumulates ``dQ`` in registers (same as the monolithic), but also
#    handles the per-head sink gradient (``dsink``) which only needs the
#    saved ``LSE`` and ``D``. Writes ``dQ`` straight to global; ``dsink``
#    is atomic-added (single counter per head — contention there is
#    negligible).
#  * ``_v4_attention_bwd_dkv_kernel`` — one program per
#    ``(b, khid, n_block)``; iterates the *m*-blocks that contribute to
#    this n-block (pruned by SWA / causal bounds), accumulates ``dK`` /
#    ``dV`` in registers and writes straight to global. No atomics.
#
# Total compute roughly doubles (``Q @ K.T`` re-materialised once per
# kernel instead of shared) but the atomic pressure goes to zero, which
# is the dominant cost on MI355 at ``H=64`` × ``BLOCK_M=BLOCK_N=32``.
# ---------------------------------------------------------------------------


@triton.jit
def _v4_attention_bwd_dq_kernel(
    Q,
    K,
    V,
    DOUT,
    LSE,
    D,
    DQ,
    DSINK,
    SINK,
    ADD_MASK,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
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
    stride_ms,
    stride_mn,
    seqlen_q,
    seqlen_k,
    sm_scale,
    HEAD_Q: tl.constexpr,
    HEAD_K: tl.constexpr,
    SWA_WINDOW: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_ADD_MASK: tl.constexpr,
    HCA_LOCAL_SEQLEN: tl.constexpr,
    USE_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACCUMULATE: tl.constexpr = False,
    EXACT_TILES_M: tl.constexpr = False,
    EXACT_TILES_N: tl.constexpr = False,
):
    """V4 attention BWD — dQ only (parallel over m-blocks, no atomics for dQ).

    When ``ACCUMULATE`` is True, the kernel performs ``dQ += dq`` instead
    of ``dQ = dq``. Each ``(b, qhid, m_block)`` program owns a unique
    output slice, so the implicit read-modify-write is race-free across
    programs within a single launch, and sequential launches of this
    kernel against the same buffer are also race-free.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    bid = pid_bh // HEAD_Q
    qhid = pid_bh % HEAD_Q
    if HEAD_K == HEAD_Q:
        khid = qhid
    else:
        khid = 0

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    NEG_INF: tl.constexpr = -1.0e30

    q_ptrs = (
        Q + bid * stride_qb + qhid * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    )
    dout_ptrs = (
        DOUT
        + bid * stride_dob
        + qhid * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :] * stride_dod
    )
    lse_ptrs = LSE + bid * stride_lb + qhid * stride_lh + offs_m * stride_lm
    dvec_ptrs = D + bid * stride_db + qhid * stride_dh + offs_m * stride_dm

    q_load_mask = offs_m[:, None] < seqlen_q
    q = tl.load(q_ptrs, mask=q_load_mask, other=0.0)
    dout = tl.load(dout_ptrs, mask=q_load_mask, other=0.0)
    lse = tl.load(lse_ptrs, mask=offs_m < seqlen_q, other=0.0)
    dvec = tl.load(dvec_ptrs, mask=offs_m < seqlen_q, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    if HAS_SINK:
        sink_h = tl.load(SINK + qhid).to(tl.float32)
        p_sink = tl.exp(sink_h - lse)
        p_sink_masked = tl.where(offs_m < seqlen_q, p_sink, 0.0)
        dvec_masked = tl.where(offs_m < seqlen_q, dvec, 0.0)
        dsink_contrib = tl.sum(-p_sink_masked * dvec_masked)
        tl.atomic_add(DSINK + qhid, dsink_contrib)

    n_loop_start = 0
    if HAS_ADD_MASK and HCA_LOCAL_SEQLEN == 0:
        n_loop_end = seqlen_k
    elif SWA_WINDOW > 0:
        n_loop_start = pid_m * BLOCK_M - SWA_WINDOW + 1
        if n_loop_start < 0:
            n_loop_start = 0
        n_loop_start = (n_loop_start // BLOCK_N) * BLOCK_N
        n_loop_end = (pid_m + 1) * BLOCK_M
        local_end = HCA_LOCAL_SEQLEN if HCA_LOCAL_SEQLEN > 0 else seqlen_k
        if n_loop_end > local_end:
            n_loop_end = local_end
    elif USE_CAUSAL:
        n_loop_end = (pid_m + 1) * BLOCK_M
        if n_loop_end > seqlen_k:
            n_loop_end = seqlen_k
    else:
        n_loop_end = seqlen_k

    for n_start in range(n_loop_start, n_loop_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)

        k_ptrs = (
            K + bid * stride_kb + khid * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            V + bid * stride_vb + khid * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        )
        kv_load_mask = offs_n[:, None] < seqlen_k
        k = tl.load(k_ptrs, mask=kv_load_mask, other=0.0)
        v = tl.load(v_ptrs, mask=kv_load_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if HAS_ADD_MASK and HCA_LOCAL_SEQLEN == 0:
            mask_ptrs = ADD_MASK + offs_m[:, None] * stride_ms + offs_n[None, :] * stride_mn
            mask_load_mask = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
            add_bias = tl.load(mask_ptrs, mask=mask_load_mask, other=0.0).to(tl.float32)
            qk = qk + add_bias
        else:
            if SWA_WINDOW > 0:
                in_window = (offs_n[None, :] >= offs_m[:, None] - SWA_WINDOW + 1) & (
                    offs_n[None, :] <= offs_m[:, None]
                )
                qk = tl.where(in_window, qk, NEG_INF)
            elif USE_CAUSAL:
                qk = tl.where(offs_n[None, :] <= offs_m[:, None], qk, NEG_INF)
        # Plan-8 P57: EXACT_TILES_* skip the boundary masks when the
        # launcher confirms ``seqlen_q % BLOCK_M == 0`` (resp. seqlen_k).
        if not EXACT_TILES_N:
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, NEG_INF)
        if not EXACT_TILES_M:
            qk = tl.where(offs_m[:, None] < seqlen_q, qk, NEG_INF)

        p = tl.exp(qk - lse[:, None])
        dp = tl.dot(dout, tl.trans(v))
        ds = p * (dp - dvec[:, None])
        # P57 cr=0 BWD: defer ``sm_scale`` to a single multiply after
        # the n-loop, and fuse the inner ``dq += dot(ds, k)`` into an
        # MFMA-acc form via ``tl.dot(..., acc=dq)``.  Numerics are
        # bit-identical modulo fp32 associativity.
        dq = tl.dot(ds.to(k.dtype), k, acc=dq)

    if HAS_ADD_MASK and HCA_LOCAL_SEQLEN > 0:
        for n_start in range(HCA_LOCAL_SEQLEN, seqlen_k, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            pool_n = offs_n - HCA_LOCAL_SEQLEN

            k_ptrs = (
                K
                + bid * stride_kb
                + khid * stride_kh
                + offs_n[:, None] * stride_kn
                + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                V
                + bid * stride_vb
                + khid * stride_vh
                + offs_n[:, None] * stride_vn
                + offs_d[None, :] * stride_vd
            )
            kv_load_mask = offs_n[:, None] < seqlen_k
            k = tl.load(k_ptrs, mask=kv_load_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_load_mask, other=0.0)

            qk = tl.dot(q, tl.trans(k)) * sm_scale
            mask_ptrs = ADD_MASK + offs_m[:, None] * stride_ms + pool_n[None, :] * stride_mn
            mask_load_mask = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
            add_bias = tl.load(mask_ptrs, mask=mask_load_mask, other=0.0).to(tl.float32)
            qk = qk + add_bias
            if not EXACT_TILES_N:
                qk = tl.where(offs_n[None, :] < seqlen_k, qk, NEG_INF)
            if not EXACT_TILES_M:
                qk = tl.where(offs_m[:, None] < seqlen_q, qk, NEG_INF)

            p = tl.exp(qk - lse[:, None])
            dp = tl.dot(dout, tl.trans(v))
            ds = p * (dp - dvec[:, None])
            dq = tl.dot(ds.to(k.dtype), k, acc=dq)

    # P57 cr=0 BWD: fold ``sm_scale`` once after both loops.
    dq = dq * sm_scale

    dq_ptrs = (
        DQ
        + bid * stride_dqb
        + qhid * stride_dqh
        + offs_m[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd
    )
    if ACCUMULATE:
        dq_prev = tl.load(dq_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        tl.store(dq_ptrs, dq + dq_prev, mask=offs_m[:, None] < seqlen_q)
    else:
        tl.store(dq_ptrs, dq, mask=offs_m[:, None] < seqlen_q)


@triton.jit
def _v4_attention_bwd_dkv_kernel(
    Q,
    K,
    V,
    DOUT,
    LSE,
    D,
    DK,  # fp32 buffer [B, K_H, Sk, D]
    DV,  # fp32 buffer [B, K_H, Sk, D]
    ADD_MASK,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
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
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dvd,
    stride_ms,
    stride_mn,
    seqlen_q,
    seqlen_k,
    sm_scale,
    HEAD_Q: tl.constexpr,
    HEAD_K: tl.constexpr,
    SWA_WINDOW: tl.constexpr,
    HAS_ADD_MASK: tl.constexpr,
    HCA_LOCAL_SEQLEN: tl.constexpr,
    USE_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    # Pool-suffix flag: when HCA split-mask, this program is one of
    # the (HEAD_Q-block over n in pool range). The pool branch iterates
    # all m's, so we run it inside the same kernel after the local
    # branch finishes.
    ATOMIC_REDUCE: tl.constexpr = False,
    # Plan-8 P57 — MQA head-split parallelism. When ``NUM_HEAD_GROUPS > 1``
    # and ``HEAD_K != HEAD_Q`` (MQA), the kernel adds an extra grid dim
    # ``pid_h_group = program_id(2)`` and each program owns
    # ``HEAD_Q / NUM_HEAD_GROUPS`` query heads. Multiple head_group
    # programs collide on the same MQA dK / dV slice, so writes are
    # ``tl.atomic_add`` instead of ``tl.store``.
    NUM_HEAD_GROUPS: tl.constexpr = 1,
    EXACT_TILES_M: tl.constexpr = False,
    EXACT_TILES_N: tl.constexpr = False,
):
    """V4 attention BWD — dK / dV only (parallel over n-blocks, no atomics for dK/dV).

    For MQA (``HEAD_K == 1``) every query head contributes to the same
    shared K / V, so this kernel must iterate ``H`` query heads per
    ``(b, n_block)``. We expose that via the ``HEAD_Q`` constexpr loop.

    HCA pool n-blocks (``n_block_start >= HCA_LOCAL_SEQLEN``) are handled
    by :func:`_v4_attention_bwd_dkv_pool_kernel`, which parallelises the
    pool work across ``(m_block, b * qhid)``. We early-return here for
    those blocks so we do not double-count the pool contribution.
    """
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_h_group = tl.program_id(2)
    bid = pid_bh // HEAD_K
    khid = pid_bh % HEAD_K

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    NEG_INF: tl.constexpr = -1.0e30

    # HCA mode: pool n-blocks are handled by the dedicated pool kernel.
    if HCA_LOCAL_SEQLEN > 0:
        if pid_n * BLOCK_N >= HCA_LOCAL_SEQLEN:
            return
        is_pool_block = False
    else:
        is_pool_block = False

    # Load K, V tiles for this n-block once.
    k_ptrs = (
        K + bid * stride_kb + khid * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    )
    v_ptrs = (
        V + bid * stride_vb + khid * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    )
    kv_load_mask = offs_n[:, None] < seqlen_k
    k = tl.load(k_ptrs, mask=kv_load_mask, other=0.0)
    v = tl.load(v_ptrs, mask=kv_load_mask, other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    # Determine m-loop bounds for this n-block. SWA / causal restricts the
    # range of m's that can see this n-block. For ``HAS_ADD_MASK`` with
    # arbitrary additive bias we iterate the full m axis.
    n_block_lo = pid_n * BLOCK_N
    n_block_hi = n_block_lo + BLOCK_N

    if HAS_ADD_MASK and HCA_LOCAL_SEQLEN == 0:
        m_loop_start = 0
        m_loop_end = seqlen_q
    elif is_pool_block:
        # Every m sees every visible pool slot; pool mask drives the rest.
        m_loop_start = 0
        m_loop_end = seqlen_q
    elif SWA_WINDOW > 0:
        # n is seen by m where n_block_lo <= m <= n_block_hi + SWA_WINDOW - 1.
        # Causal also requires m >= n_block_lo. Round to BLOCK_M tiles.
        m_loop_start = (n_block_lo // BLOCK_M) * BLOCK_M
        m_loop_end = n_block_hi + SWA_WINDOW - 1
        if m_loop_end > seqlen_q:
            m_loop_end = seqlen_q
        # Round up to BLOCK_M boundary so the m-loop iterates whole tiles.
        m_loop_end = ((m_loop_end + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    elif USE_CAUSAL:
        m_loop_start = (n_block_lo // BLOCK_M) * BLOCK_M
        m_loop_end = seqlen_q
        m_loop_end = ((m_loop_end + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    else:
        m_loop_start = 0
        m_loop_end = seqlen_q
        m_loop_end = ((m_loop_end + BLOCK_M - 1) // BLOCK_M) * BLOCK_M

    # MHA path (HEAD_K == HEAD_Q) — only one query head contributes to
    # this khid, so we use ``qhid = khid`` and skip the head loop. For
    # MQA (HEAD_K != HEAD_Q) we iterate every query head.
    if HEAD_K == HEAD_Q:
        qhid = khid
        for m_start in range(m_loop_start, m_loop_end, BLOCK_M):
            offs_m = m_start + tl.arange(0, BLOCK_M)

            q_ptrs = (
                Q
                + bid * stride_qb
                + qhid * stride_qh
                + offs_m[:, None] * stride_qm
                + offs_d[None, :] * stride_qd
            )
            dout_ptrs = (
                DOUT
                + bid * stride_dob
                + qhid * stride_doh
                + offs_m[:, None] * stride_dom
                + offs_d[None, :] * stride_dod
            )
            lse_ptrs = LSE + bid * stride_lb + qhid * stride_lh + offs_m * stride_lm
            dvec_ptrs = D + bid * stride_db + qhid * stride_dh + offs_m * stride_dm

            q_load_mask = offs_m[:, None] < seqlen_q
            q = tl.load(q_ptrs, mask=q_load_mask, other=0.0)
            dout = tl.load(dout_ptrs, mask=q_load_mask, other=0.0)
            lse = tl.load(lse_ptrs, mask=offs_m < seqlen_q, other=0.0)
            dvec = tl.load(dvec_ptrs, mask=offs_m < seqlen_q, other=0.0)

            qk = tl.dot(q, tl.trans(k)) * sm_scale

            if HAS_ADD_MASK and HCA_LOCAL_SEQLEN == 0:
                mask_ptrs = ADD_MASK + offs_m[:, None] * stride_ms + offs_n[None, :] * stride_mn
                mask_load_mask = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
                add_bias = tl.load(mask_ptrs, mask=mask_load_mask, other=0.0).to(tl.float32)
                qk = qk + add_bias
            elif is_pool_block:
                pool_n = offs_n - HCA_LOCAL_SEQLEN
                mask_ptrs = ADD_MASK + offs_m[:, None] * stride_ms + pool_n[None, :] * stride_mn
                mask_load_mask = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
                add_bias = tl.load(mask_ptrs, mask=mask_load_mask, other=0.0).to(tl.float32)
                qk = qk + add_bias
            else:
                if SWA_WINDOW > 0:
                    in_window = (offs_n[None, :] >= offs_m[:, None] - SWA_WINDOW + 1) & (
                        offs_n[None, :] <= offs_m[:, None]
                    )
                    qk = tl.where(in_window, qk, NEG_INF)
                elif USE_CAUSAL:
                    qk = tl.where(offs_n[None, :] <= offs_m[:, None], qk, NEG_INF)

            if not EXACT_TILES_N:
                qk = tl.where(offs_n[None, :] < seqlen_k, qk, NEG_INF)
            if not EXACT_TILES_M:
                qk = tl.where(offs_m[:, None] < seqlen_q, qk, NEG_INF)

            p = tl.exp(qk - lse[:, None])
            dp = tl.dot(dout, tl.trans(v))
            ds = p * (dp - dvec[:, None])

            # P57 cr=0 BWD: fuse the inner ``dv += dot(p, dout)`` and
            # ``dk += dot(ds, q)`` into MFMA-acc form; defer ``sm_scale``
            # on dk to a single multiply after the m-loop.
            dv = tl.dot(tl.trans(p.to(dout.dtype)), dout, acc=dv)
            dk = tl.dot(tl.trans(ds.to(q.dtype)), q, acc=dk)
    else:
        # MQA path. With NUM_HEAD_GROUPS > 1 the program owns a slice of
        # query heads; otherwise it iterates all HEAD_Q heads.
        head_per_group: tl.constexpr = HEAD_Q // NUM_HEAD_GROUPS
        h_start = pid_h_group * head_per_group
        h_end = h_start + head_per_group
        for h_iter in range(h_start, h_end):
            qhid = h_iter

            for m_start in range(m_loop_start, m_loop_end, BLOCK_M):
                offs_m = m_start + tl.arange(0, BLOCK_M)

                q_ptrs = (
                    Q
                    + bid * stride_qb
                    + qhid * stride_qh
                    + offs_m[:, None] * stride_qm
                    + offs_d[None, :] * stride_qd
                )
                dout_ptrs = (
                    DOUT
                    + bid * stride_dob
                    + qhid * stride_doh
                    + offs_m[:, None] * stride_dom
                    + offs_d[None, :] * stride_dod
                )
                lse_ptrs = LSE + bid * stride_lb + qhid * stride_lh + offs_m * stride_lm
                dvec_ptrs = D + bid * stride_db + qhid * stride_dh + offs_m * stride_dm

                q_load_mask = offs_m[:, None] < seqlen_q
                q = tl.load(q_ptrs, mask=q_load_mask, other=0.0)
                dout = tl.load(dout_ptrs, mask=q_load_mask, other=0.0)
                lse = tl.load(lse_ptrs, mask=offs_m < seqlen_q, other=0.0)
                dvec = tl.load(dvec_ptrs, mask=offs_m < seqlen_q, other=0.0)

                qk = tl.dot(q, tl.trans(k)) * sm_scale

                if HAS_ADD_MASK and HCA_LOCAL_SEQLEN == 0:
                    mask_ptrs = ADD_MASK + offs_m[:, None] * stride_ms + offs_n[None, :] * stride_mn
                    mask_load_mask = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
                    add_bias = tl.load(mask_ptrs, mask=mask_load_mask, other=0.0).to(tl.float32)
                    qk = qk + add_bias
                elif is_pool_block:
                    pool_n = offs_n - HCA_LOCAL_SEQLEN
                    mask_ptrs = ADD_MASK + offs_m[:, None] * stride_ms + pool_n[None, :] * stride_mn
                    mask_load_mask = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
                    add_bias = tl.load(mask_ptrs, mask=mask_load_mask, other=0.0).to(tl.float32)
                    qk = qk + add_bias
                else:
                    if SWA_WINDOW > 0:
                        in_window = (offs_n[None, :] >= offs_m[:, None] - SWA_WINDOW + 1) & (
                            offs_n[None, :] <= offs_m[:, None]
                        )
                        qk = tl.where(in_window, qk, NEG_INF)
                    elif USE_CAUSAL:
                        qk = tl.where(offs_n[None, :] <= offs_m[:, None], qk, NEG_INF)

                if not EXACT_TILES_N:
                    qk = tl.where(offs_n[None, :] < seqlen_k, qk, NEG_INF)
                if not EXACT_TILES_M:
                    qk = tl.where(offs_m[:, None] < seqlen_q, qk, NEG_INF)

                p = tl.exp(qk - lse[:, None])
                dp = tl.dot(dout, tl.trans(v))
                ds = p * (dp - dvec[:, None])

                # P57 cr=0 BWD: same scale-defer + tl.dot(acc=) trick
                # as the non-MQA branch above.
                dv = tl.dot(tl.trans(p.to(dout.dtype)), dout, acc=dv)
                dk = tl.dot(tl.trans(ds.to(q.dtype)), q, acc=dk)

    # P57 cr=0 BWD: fold ``sm_scale`` on dk once after the (head x m)
    # loop.  dv carries no scale.
    dk = dk * sm_scale

    dk_ptrs = (
        DK
        + bid * stride_dkb
        + khid * stride_dkh
        + offs_n[:, None] * stride_dkn
        + offs_d[None, :] * stride_dkd
    )
    dv_ptrs = (
        DV
        + bid * stride_dvb
        + khid * stride_dvh
        + offs_n[:, None] * stride_dvn
        + offs_d[None, :] * stride_dvd
    )
    if ATOMIC_REDUCE or NUM_HEAD_GROUPS > 1:
        # ``ATOMIC_REDUCE`` is set on the CSA pool BWD path (multiple
        # ``(b, head, n_block)`` programs collapse into a single
        # ``(b, n_block)`` global slice with ``stride_dkh = stride_dvh = 0``).
        # ``NUM_HEAD_GROUPS > 1`` is the MQA head-split path: multiple
        # head_group programs target the same MQA dK / dV slice. Both
        # require fp32 atomic_add to merge.
        tl.atomic_add(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k, sem="relaxed")
        tl.atomic_add(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k, sem="relaxed")
    else:
        tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)


# ---------------------------------------------------------------------------
# Plan-8 P57 — HCA pool dK / dV kernel
#
# The HCA split-mask BWD has a single "pool" n-block (P=Sk-HCA_LOCAL_SEQLEN
# keys; typically 32 at V4-Flash widths). Folding the pool into the main
# dK/dV kernel makes that single n-block program iterate every m-block ×
# every query head — at H=64 / Sq=4096 / BLOCK_M=32 that is 8192 inner
# iterations versus ~256 for a local-SWA n-block program (32× more work).
#
# This kernel parallelises pool work over ``m_block`` (one program per
# ``(b, m_block)``) and iterates ``HEAD_Q`` heads internally. Each program
# accumulates its (BLOCK_N, BLOCK_DMODEL) tile contribution in registers
# (no per-head atomic) and atomic-adds the final tile into the shared
# pool slice of dK / dV exactly twice (once for dk, once for dv) per
# program. At V4-Flash widths the contention is 128-way per pool cell
# instead of 8192-way for the m_block × qhid full grid, while still
# extracting 128× more parallelism than the original single-program pool
# branch.
# ---------------------------------------------------------------------------


@triton.jit
def _v4_attention_bwd_dkv_pool_kernel(
    Q,
    K,
    V,
    DOUT,
    LSE,
    D,
    DK,  # fp32 buffer [B, K_H, Sk, D]
    DV,  # fp32 buffer [B, K_H, Sk, D]
    ADD_MASK,  # [Sq, P] additive pool mask
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
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
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dvd,
    stride_ms,
    stride_mn,
    seqlen_q,
    seqlen_k,
    pool_size,
    sm_scale,
    HEAD_Q: tl.constexpr,
    HEAD_K: tl.constexpr,
    HCA_LOCAL_SEQLEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """V4 HCA BWD — dK / dV for the pool keys (parallel m-blocks, head-loop inside)."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    bid = pid_b

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    pool_n = tl.arange(0, BLOCK_N)
    offs_n = HCA_LOCAL_SEQLEN + pool_n

    NEG_INF: tl.constexpr = -1.0e30

    pool_n_mask = pool_n < pool_size
    q_load_mask = offs_m[:, None] < seqlen_q

    # Per-(m_block, b) program iterates all query heads, accumulating in
    # registers. For MQA (HK=1) all heads share the same (b, khid=0) K/V,
    # so we reload K/V once at the start. For MHA (HK=HQ) we reload K/V
    # per head inside the loop.
    if HEAD_K == 1:
        k_ptrs = K + bid * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + bid * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        kv_load_mask = pool_n_mask[:, None]
        k_shared = tl.load(k_ptrs, mask=kv_load_mask, other=0.0)
        v_shared = tl.load(v_ptrs, mask=kv_load_mask, other=0.0)
        kt_shared = tl.trans(k_shared)
        vt_shared = tl.trans(v_shared)

    dk_acc = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    for qhid in range(0, HEAD_Q):
        if HEAD_K == HEAD_Q:
            khid = qhid
        else:
            khid = 0

        q_ptrs = (
            Q + bid * stride_qb + qhid * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        )
        dout_ptrs = (
            DOUT
            + bid * stride_dob
            + qhid * stride_doh
            + offs_m[:, None] * stride_dom
            + offs_d[None, :] * stride_dod
        )
        lse_ptrs = LSE + bid * stride_lb + qhid * stride_lh + offs_m * stride_lm
        dvec_ptrs = D + bid * stride_db + qhid * stride_dh + offs_m * stride_dm

        q = tl.load(q_ptrs, mask=q_load_mask, other=0.0)
        dout = tl.load(dout_ptrs, mask=q_load_mask, other=0.0)
        lse = tl.load(lse_ptrs, mask=offs_m < seqlen_q, other=0.0)
        dvec = tl.load(dvec_ptrs, mask=offs_m < seqlen_q, other=0.0)

        if HEAD_K == 1:
            kt = kt_shared
            vt = vt_shared
        else:
            k_ptrs = (
                K
                + bid * stride_kb
                + khid * stride_kh
                + offs_n[:, None] * stride_kn
                + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                V
                + bid * stride_vb
                + khid * stride_vh
                + offs_n[:, None] * stride_vn
                + offs_d[None, :] * stride_vd
            )
            kv_load_mask = pool_n_mask[:, None]
            k = tl.load(k_ptrs, mask=kv_load_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_load_mask, other=0.0)
            kt = tl.trans(k)
            vt = tl.trans(v)

        qk = tl.dot(q, kt) * sm_scale
        mask_ptrs = ADD_MASK + offs_m[:, None] * stride_ms + pool_n[None, :] * stride_mn
        mask_load_mask = (offs_m[:, None] < seqlen_q) & pool_n_mask[None, :]
        add_bias = tl.load(mask_ptrs, mask=mask_load_mask, other=0.0).to(tl.float32)
        qk = qk + add_bias
        qk = tl.where(pool_n_mask[None, :], qk, NEG_INF)
        qk = tl.where(offs_m[:, None] < seqlen_q, qk, NEG_INF)

        p = tl.exp(qk - lse[:, None])
        dp = tl.dot(dout, vt)
        ds = p * (dp - dvec[:, None])

        if HEAD_K == 1:
            dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q) * sm_scale
            dv_acc += tl.dot(tl.trans(p.to(dout.dtype)), dout)
        else:
            # MHA path — each qhid maps to its own khid slice. Accumulator is
            # not shared across heads; flush per-head with atomic_add.
            dk_contrib = tl.dot(tl.trans(ds.to(q.dtype)), q) * sm_scale
            dv_contrib = tl.dot(tl.trans(p.to(dout.dtype)), dout)
            dk_ptrs_h = (
                DK
                + bid * stride_dkb
                + khid * stride_dkh
                + offs_n[:, None] * stride_dkn
                + offs_d[None, :] * stride_dkd
            )
            dv_ptrs_h = (
                DV
                + bid * stride_dvb
                + khid * stride_dvh
                + offs_n[:, None] * stride_dvn
                + offs_d[None, :] * stride_dvd
            )
            tl.atomic_add(dk_ptrs_h, dk_contrib, mask=pool_n_mask[:, None], sem="relaxed")
            tl.atomic_add(dv_ptrs_h, dv_contrib, mask=pool_n_mask[:, None], sem="relaxed")

    if HEAD_K == 1:
        khid_final = 0
        dk_ptrs = (
            DK
            + bid * stride_dkb
            + khid_final * stride_dkh
            + offs_n[:, None] * stride_dkn
            + offs_d[None, :] * stride_dkd
        )
        dv_ptrs = (
            DV
            + bid * stride_dvb
            + khid_final * stride_dvh
            + offs_n[:, None] * stride_dvn
            + offs_d[None, :] * stride_dvd
        )
        write_mask = pool_n_mask[:, None]
        tl.atomic_add(dk_ptrs, dk_acc, mask=write_mask, sem="relaxed")
        tl.atomic_add(dv_ptrs, dv_acc, mask=write_mask, sem="relaxed")


# ---------------------------------------------------------------------------
# Python launcher
# ---------------------------------------------------------------------------


def _launch_v4_attention_bwd(
    q: torch.Tensor,  # [B, H, Sq, D]
    k: torch.Tensor,  # [B, K_H, Sk, D]
    v: torch.Tensor,  # [B, K_H, Sk, D]
    out: torch.Tensor,  # [B, H, Sq, D] (FWD output)
    dout: torch.Tensor,  # [B, H, Sq, D]
    lse: torch.Tensor,  # [B, H, Sq] fp32
    *,
    sink: Optional[torch.Tensor],  # [H] or None
    swa_window: int,
    additive_mask: Optional[torch.Tensor],  # [Sq, Sk] or None
    scale: float,
    hca_local_seqlen: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Launch the V4 attention backward kernel.

    Returns ``(dq, dk, dv, dsink)`` — gradients in the input dtype, with
    ``dsink`` returned only when ``sink is not None`` (else ``None``).
    """
    if not q.is_cuda:
        raise ValueError("v4_attention BWD requires CUDA / HIP tensors.")
    if dout.shape != out.shape or out.shape != q.shape:
        raise ValueError(
            "v4_attention BWD shape mismatch: "
            f"out={tuple(out.shape)}, dout={tuple(dout.shape)}, q={tuple(q.shape)}"
        )

    B, HQ, Sq, D = q.shape
    HK = k.shape[1]
    Sk = k.shape[2]

    has_sink = sink is not None
    has_add_mask = additive_mask is not None
    hca_local_seqlen = int(hca_local_seqlen)
    if hca_local_seqlen:
        if not has_add_mask:
            raise ValueError("hca_local_seqlen requires a pool additive_mask.")
        if hca_local_seqlen <= 0 or hca_local_seqlen >= Sk:
            raise ValueError(
                "hca_local_seqlen must split local and pool keys "
                f"(got hca_local_seqlen={hca_local_seqlen}, Sk={Sk})."
            )
        expected_mask_shape = (Sq, Sk - hca_local_seqlen)
        if tuple(additive_mask.shape) != expected_mask_shape:
            raise ValueError(
                "HCA split-mask mode expects additive_mask shape "
                f"{expected_mask_shape}, got {tuple(additive_mask.shape)}."
            )
        if swa_window <= 0:
            raise ValueError("HCA split-mask mode requires swa_window > 0.")
    use_causal = (not has_add_mask) and (swa_window <= 0)
    swa_window_constexpr = (
        int(swa_window) if ((not has_add_mask or hca_local_seqlen) and swa_window > 0) else 0
    )

    # Plan-8 P57 sweep at V4-Flash widths (B=1 H=64 Sq=4096 D=512 P=32):
    # ``BLOCK_M=64 BLOCK_N=16`` lifts dKV grid from 128 -> 256 programs
    # (full 256-CU MI355 occupancy) and lets the Q tile (64x512) better
    # amortise the per-m-block load. The wider Q matches MFMA 32x32x16
    # output tile layout (two 32x16 tiles back-to-back) and gets the dKV
    # kernel from ~6.7 ms (BM=BN=32) down to ~3.6 ms in the dense bench.
    BLOCK_M = int(os.getenv("PRIMUS_V4_ATTN_BWD_BLOCK_M", "64"))
    BLOCK_N = int(os.getenv("PRIMUS_V4_ATTN_BWD_BLOCK_N", "16"))
    BLOCK_DMODEL = D

    # Allocate fp32 output buffers for atomic_add. Cast to input dtype
    # before returning.
    dq_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
    dk_fp32 = torch.zeros((B, HK, Sk, D), device=q.device, dtype=torch.float32)
    dv_fp32 = torch.zeros((B, HK, Sk, D), device=q.device, dtype=torch.float32)
    if has_sink:
        dsink_fp32 = torch.zeros((HQ,), device=q.device, dtype=torch.float32)
        sink_arg = sink.to(torch.float32) if sink.dtype != torch.float32 else sink
    else:
        dsink_fp32 = q  # sentinel; HAS_SINK=False inside kernel
        sink_arg = q

    # D scalar = (dout * out).sum(-1)
    d_buf = torch.empty((B, HQ, Sq), device=q.device, dtype=torch.float32)
    pre_grid = (triton.cdiv(Sq, BLOCK_M), B * HQ)
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
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=1,
    )

    # Mask sentinel ptr for HAS_ADD_MASK=False
    mask_arg = additive_mask if has_add_mask else q
    if has_add_mask:
        stride_ms = additive_mask.stride(0)
        stride_mn = additive_mask.stride(1)
    else:
        stride_ms = 0
        stride_mn = 0

    # Plan-8 P57: skip the boundary masks (``offs_m < seqlen_q`` /
    # ``offs_n < seqlen_k``) when the launcher confirms the seqlens are
    # exact multiples of BLOCK_M / BLOCK_N. Saves a per-inner-iter
    # broadcast + tl.where on production V4-Flash widths where Sq=4096
    # (BM=64) and Sk=4128 (BN=16) both divide cleanly.
    exact_tiles_m = (Sq % BLOCK_M) == 0
    exact_tiles_n = (Sk % BLOCK_N) == 0

    # Plan-5 P32: split BWD (dQ kernel + dK/dV kernel, no atomics for
    # dQ / dK / dV) is now the default — wins both the operator microbench
    # *and* the EP8 proxy after the dual-RoPE bf16-cast fix (P32 RoPE bug:
    # ``apply_interleaved_partial_rope`` was upcasting Q/K to fp32 because
    # cos/sin came from ``position_ids.float()`` and bf16 * fp32 = fp32,
    # which 2x'd Q/K HBM traffic, inflated the kernel time 1.8-7x in the
    # proxy and made the monolithic design *look* faster in A/B traces).
    # ``PRIMUS_V4_ATTN_BWD_USE_SPLIT=0`` falls back to the monolithic
    # design for kernel-level perf experiments / debugging.
    if os.getenv("PRIMUS_V4_ATTN_BWD_USE_SPLIT", "1") == "1":
        dq_grid = (triton.cdiv(Sq, BLOCK_M), B * HQ)
        _v4_attention_bwd_dq_kernel[dq_grid](
            q,
            k,
            v,
            dout,
            lse,
            d_buf,
            dq_fp32,
            dsink_fp32,
            sink_arg,
            mask_arg,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
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
            stride_ms,
            stride_mn,
            Sq,
            Sk,
            float(scale),
            HEAD_Q=HQ,
            HEAD_K=HK,
            SWA_WINDOW=swa_window_constexpr,
            HAS_SINK=has_sink,
            HAS_ADD_MASK=has_add_mask,
            HCA_LOCAL_SEQLEN=hca_local_seqlen,
            USE_CAUSAL=use_causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            EXACT_TILES_M=exact_tiles_m,
            EXACT_TILES_N=exact_tiles_n,
            # Plan-8 P57: ``num_warps=4 num_stages=2`` chosen by sweep at
            # V4-Flash widths. ``num_warps=8`` (the original default) over
            # -occupies VGPR/AGPR per program and starves the SIMDs;
            # ``num_warps=4`` with ``num_stages=2`` (double-buffer K/V loads)
            # is the dq sweet spot.
            num_warps=int(os.getenv("PRIMUS_V4_ATTN_BWD_DQ_NUM_WARPS", "4")),
            num_stages=int(os.getenv("PRIMUS_V4_ATTN_BWD_DQ_NUM_STAGES", "2")),
        )
        # Plan-8 P57: in HCA mode the pool n-block(s) are handled by the
        # dedicated pool kernel that parallelises over (m_block, b * qhid)
        # — see ``_v4_attention_bwd_dkv_pool_kernel``. Skip the pool grid
        # rows here so we don't double-count.
        if hca_local_seqlen > 0:
            dkv_n_blocks = triton.cdiv(hca_local_seqlen, BLOCK_N)
        else:
            dkv_n_blocks = triton.cdiv(Sk, BLOCK_N)
        # Plan-8 P57: ``NUM_HEAD_GROUPS`` controls MQA head-split parallelism
        # for the local dKV kernel. Default (1) keeps the original head loop;
        # >1 splits the head loop and uses ``tl.atomic_add`` for dKV.
        # The dense-bench sweep showed head-split is essentially neutral on
        # MI355 at H=64 (HBM/atomic-bound, not compute-bound), so default 1.
        num_head_groups = 1
        if HQ > HK:
            target = int(os.getenv("PRIMUS_V4_ATTN_BWD_DKV_HEAD_GROUPS", "1"))
            while target > 1 and HQ % target != 0:
                target //= 2
            num_head_groups = max(1, target)
        dkv_grid = (dkv_n_blocks, B * HK, num_head_groups)
        # Plan-8 P57: ``num_warps=4 num_stages=1`` chosen by sweep — the
        # dKV kernel keeps K/V in registers across the m-loop, which makes
        # the 8-warp / staged-pipeline variants thrash VGPR vs ``num_warps=4``.
        dkv_num_warps = int(os.getenv("PRIMUS_V4_ATTN_BWD_DKV_NUM_WARPS", "4"))
        dkv_num_stages = int(os.getenv("PRIMUS_V4_ATTN_BWD_DKV_NUM_STAGES", "1"))
        _v4_attention_bwd_dkv_kernel[dkv_grid](
            q,
            k,
            v,
            dout,
            lse,
            d_buf,
            dk_fp32,
            dv_fp32,
            mask_arg,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
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
            dk_fp32.stride(0),
            dk_fp32.stride(1),
            dk_fp32.stride(2),
            dk_fp32.stride(3),
            dv_fp32.stride(0),
            dv_fp32.stride(1),
            dv_fp32.stride(2),
            dv_fp32.stride(3),
            stride_ms,
            stride_mn,
            Sq,
            Sk,
            float(scale),
            HEAD_Q=HQ,
            HEAD_K=HK,
            SWA_WINDOW=swa_window_constexpr,
            HAS_ADD_MASK=has_add_mask,
            HCA_LOCAL_SEQLEN=hca_local_seqlen,
            USE_CAUSAL=use_causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            NUM_HEAD_GROUPS=num_head_groups,
            EXACT_TILES_M=exact_tiles_m,
            EXACT_TILES_N=exact_tiles_n,
            num_warps=dkv_num_warps,
            num_stages=dkv_num_stages,
        )
        if hca_local_seqlen > 0 and os.getenv("PRIMUS_V4_ATTN_BWD_HCA_POOL", "1") == "1":
            pool_size = Sk - hca_local_seqlen
            pool_block_n = max(16, triton.next_power_of_2(pool_size))
            # Plan-8 P57: pool kernel block_m is decoupled from the dKV
            # block_m. dKV benefits from BM=64 (fewer programs each loading
            # a wider Q tile); pool kernel benefits from a smaller BM (more
            # programs => more parallelism over the head loop).
            pool_block_m = int(os.getenv("PRIMUS_V4_ATTN_BWD_POOL_BLOCK_M", str(BLOCK_M)))
            pool_grid = (triton.cdiv(Sq, pool_block_m), B)
            _v4_attention_bwd_dkv_pool_kernel[pool_grid](
                q,
                k,
                v,
                dout,
                lse,
                d_buf,
                dk_fp32,
                dv_fp32,
                additive_mask,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
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
                dk_fp32.stride(0),
                dk_fp32.stride(1),
                dk_fp32.stride(2),
                dk_fp32.stride(3),
                dv_fp32.stride(0),
                dv_fp32.stride(1),
                dv_fp32.stride(2),
                dv_fp32.stride(3),
                stride_ms,
                stride_mn,
                Sq,
                Sk,
                pool_size,
                float(scale),
                HEAD_Q=HQ,
                HEAD_K=HK,
                HCA_LOCAL_SEQLEN=hca_local_seqlen,
                BLOCK_M=pool_block_m,
                BLOCK_N=pool_block_n,
                BLOCK_DMODEL=BLOCK_DMODEL,
                # Plan-8 P57 sweep with dKV BM=64 BN=16: pool kernel
                # ``num_warps=4 num_stages=1`` is the sweet spot. Higher
                # num_stages adds Q/dout double-buffering overhead that
                # doesn't pay off because the pool kernel keeps K/V in
                # registers (single load) and the head-loop dominates
                # the iteration count.
                num_warps=int(os.getenv("PRIMUS_V4_ATTN_BWD_POOL_NUM_WARPS", "4")),
                num_stages=int(os.getenv("PRIMUS_V4_ATTN_BWD_POOL_NUM_STAGES", "1")),
            )
        dq_out = dq_fp32.to(q.dtype)
        dk_out = dk_fp32.to(k.dtype)
        dv_out = dv_fp32.to(v.dtype)
        dsink_out = dsink_fp32.to(sink.dtype) if has_sink else None
        return dq_out, dk_out, dv_out, dsink_out

    grid = (triton.cdiv(Sq, BLOCK_M), B * HQ)
    _v4_attention_bwd_kernel[grid](
        q,
        k,
        v,
        dout,
        lse,
        d_buf,
        dq_fp32,
        dk_fp32,
        dv_fp32,
        dsink_fp32,
        sink_arg,
        mask_arg,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
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
        dk_fp32.stride(0),
        dk_fp32.stride(1),
        dk_fp32.stride(2),
        dk_fp32.stride(3),
        dv_fp32.stride(0),
        dv_fp32.stride(1),
        dv_fp32.stride(2),
        dv_fp32.stride(3),
        stride_ms,
        stride_mn,
        Sq,
        Sk,
        float(scale),
        HEAD_Q=HQ,
        HEAD_K=HK,
        SWA_WINDOW=swa_window_constexpr,
        HAS_SINK=has_sink,
        HAS_ADD_MASK=has_add_mask,
        HCA_LOCAL_SEQLEN=hca_local_seqlen,
        USE_CAUSAL=use_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=8,
        num_stages=1,
    )

    # Cast fp32 buffers back to input dtype.
    dq_out = dq_fp32.to(q.dtype)
    dk_out = dk_fp32.to(k.dtype)
    dv_out = dv_fp32.to(v.dtype)
    dsink_out = dsink_fp32.to(sink.dtype) if has_sink else None
    return dq_out, dk_out, dv_out, dsink_out


__all__ = [
    "_v4_attention_bwd_preprocess_kernel",
    "_v4_attention_bwd_kernel",
    "_launch_v4_attention_bwd",
]
