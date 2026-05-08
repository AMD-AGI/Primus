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

    # Determine n-loop range (matches FWD's logic).
    if HAS_ADD_MASK:
        n_loop_end = seqlen_k
    elif SWA_WINDOW > 0 or USE_CAUSAL:
        n_loop_end = (pid_m + 1) * BLOCK_M
        if n_loop_end > seqlen_k:
            n_loop_end = seqlen_k
    else:
        n_loop_end = seqlen_k

    for n_start in range(0, n_loop_end, BLOCK_N):
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
        if HAS_ADD_MASK:
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
    use_causal = (not has_add_mask) and (swa_window <= 0)
    swa_window_constexpr = int(swa_window) if (not has_add_mask and swa_window > 0) else 0

    BLOCK_M = 32
    BLOCK_N = 32
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
