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

from typing import Optional

import torch
import triton
import triton.language as tl

from primus.backends.megatron.core.transformer.v4_attention_kernels._triton.v4_attention_bwd import (
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
        qk = tl.sum(kl.to(tl.float32) * q[None, :].to(tl.float32), axis=1) * sm_scale

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


__all__ = [
    "_v4_csa_attention_bwd_kernel",
    "_launch_v4_csa_attention_bwd",
]
