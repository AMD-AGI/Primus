###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""V4 CSA attention forward Triton kernel (plan-4 P26, ``compress_ratio == 4``).

CSA fuses three branches into a single online softmax:

* **Local SWA**: ``q @ k_local^T`` with sliding-window-causal masking.
* **Sparse top-K**: ``q . gathered[m, :, :]`` where the wrapper has
  pre-gathered ``[B, Sq, K, D]`` rows from the compressed pool (the
  per-query top-K gather lives outside the kernel — see plan-4
  ``02-phase-details.md`` Phase 26 design notes).
* **Per-head learned sink**: a virtual key column with notional value
  zero, joined as the last softmax candidate so its probability mass
  is shared across local + sparse branches.

The kernel produces one ``[BLOCK_DMODEL]`` output row per program; the
grid is ``(seqlen_q, batch * head_q)`` so each program owns exactly one
``(b, qhid, m)`` query row. The per-row design keeps the sparse-branch
SMEM footprint inside the MI355 budget at ``head_dim=512``: the
gathered tile is only ``[BLOCK_K, head_dim] * 2 bytes ≈ 32 KiB`` per
program, while a multi-row tile would balloon to
``[BLOCK_M, BLOCK_K, head_dim] * 2 bytes ≈ 1 MiB``.

dtype contract (matches :func:`eager_v4_csa_attention`):

* Q / K / V / gathered are loaded in input dtype (bf16 in production);
  the per-row dot products (``sum(k * q[None, :], axis=-1)``) reduce in
  fp32 because we ``.to(tl.float32)`` before the multiply.
* The online-softmax accumulator (``m_running``, ``l_running``,
  ``acc``) lives in fp32 — the *only* fp32 step inside the kernel.
* Output is written back in input dtype; saved ``LSE`` is fp32 (BWD
  re-materialises ``P`` from it).

Edge cases handled:

* ``K_topk == 0`` — wrapper short-circuits to the dense
  :func:`v4_attention` kernel before reaching this file.
* ``topk_idx == -1`` — wrapper sets the corresponding ``sparse_mask``
  entry to ``-inf``; the kernel just adds the bias and the masked
  position contributes ~0 to the softmax denominator.
* All-masked tile rows — the running max and per-tile max are both
  the finite ``NEG_INF`` sentinel (``-1e30``), so
  ``exp(NEG_INF - NEG_INF) = exp(0) = 1`` algebraically but the
  contribution to ``acc`` and ``l_running`` is gated by the
  per-element ``exp(qk - m_new)`` which stays at exactly zero for
  every ``-inf``-masked entry. This avoids the ``exp(-inf - -inf) =
  exp(NaN)`` failure mode that ``-float("inf")`` would have.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from primus.backends.megatron.core.transformer.v4_attention_kernels._triton.v4_attention_fwd import (
    _launch_v4_attention_fwd,
)

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def _v4_csa_attention_fwd_kernel(
    Q,
    K_LOCAL,
    V_LOCAL,
    GATHERED,
    SPARSE_MASK,
    SINK,
    OUT,
    LSE,
    # Q strides: [B, H, Sq, D] row-major (contiguous on D)
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    # K_local strides: [B, H, Sq, D] row-major (CSA always has K_H == HQ —
    # the V4 forward broadcast-expanded MQA single-latent KV across the H
    # query heads before this call)
    stride_klb,
    stride_klh,
    stride_kln,
    stride_kld,
    # V_local strides: [B, H, Sq, D] row-major
    stride_vlb,
    stride_vlh,
    stride_vln,
    stride_vld,
    # gathered strides: [B, Sq, K_topk, D] row-major (no H dim — gather
    # is per-query but shared across heads)
    stride_gb,
    stride_gm,
    stride_gk,
    stride_gd,
    # sparse_mask strides: [B, Sq, K_topk] row-major (broadcasts over H)
    stride_smb,
    stride_smm,
    stride_smk,
    # OUT strides: [B, H, Sq, D] row-major
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    # LSE strides: [B, H, Sq] row-major
    stride_lb,
    stride_lh,
    stride_lm,
    seqlen_q,
    K_topk,
    sm_scale,
    HEAD_Q: tl.constexpr,
    SWA_WINDOW: tl.constexpr,  # > 0 for V4; 0 falls back to full causal
    HAS_SINK: tl.constexpr,
    BLOCK_N: tl.constexpr,  # local-key tile size
    BLOCK_K: tl.constexpr,  # sparse-key tile size
    BLOCK_DMODEL: tl.constexpr,  # head_dim — must be a power of 2
):
    """V4 CSA fused-attention FWD (one program per output row)."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    bid = pid_bh // HEAD_Q
    qhid = pid_bh % HEAD_Q

    offs_d = tl.arange(0, BLOCK_DMODEL)

    # ---- Load Q row [BLOCK_DMODEL] -----------------------------------------
    q_row_offset = bid * stride_qb + qhid * stride_qh + pid_m * stride_qm
    q_ptrs = Q + q_row_offset + offs_d * stride_qd
    q_active = pid_m < seqlen_q
    q = tl.load(q_ptrs, mask=q_active, other=0.0)

    # Online-softmax running state (fp32). NEG_INF is a finite sentinel
    # (-1e30) so all-masked tiles do not produce NaN through
    # ``exp(-inf - -inf) = exp(NaN)``.
    NEG_INF: tl.constexpr = -1.0e30
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.full((), value=NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)

    # ---- Local SWA branch --------------------------------------------------
    # Causal: keys n in [0, pid_m]. SWA: keys n in [pid_m - SWA_WINDOW + 1,
    # pid_m]. We walk from the SWA window's lower bound (rounded down to
    # BLOCK_N) up to (pid_m + 1). The in-kernel window check inside the
    # tile loop handles the boundary cases exactly so the result matches
    # ``sliding_window_causal_mask(...)``.
    n_loop_end = pid_m + 1
    if n_loop_end > seqlen_q:
        n_loop_end = seqlen_q

    # Lower bound of the SWA window (clamped to >= 0). When SWA_WINDOW <= 0
    # this collapses to a full causal walk from 0.
    if SWA_WINDOW > 0:
        n_lo_raw = pid_m - SWA_WINDOW + 1
        if n_lo_raw < 0:
            n_lo_raw = 0
        # Round down to a BLOCK_N multiple so tile-aligned loads stay aligned.
        n_loop_start = (n_lo_raw // BLOCK_N) * BLOCK_N
    else:
        n_loop_start = 0

    for n_start in range(n_loop_start, n_loop_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)

        # K_local tile: [BLOCK_N, BLOCK_DMODEL] in k_local.dtype
        kl_ptrs = (
            K_LOCAL
            + bid * stride_klb
            + qhid * stride_klh
            + offs_n[:, None] * stride_kln
            + offs_d[None, :] * stride_kld
        )
        kl_load_mask = offs_n[:, None] < seqlen_q
        kl = tl.load(kl_ptrs, mask=kl_load_mask, other=0.0)

        # qk = sum_d (kl[n, d] * q[d]) -> [BLOCK_N], computed in fp32 by
        # upcasting the operands. (Matches the eager reference's
        # bf16-tensor-core matmul w/ fp32 accumulator semantics; a 1xD
        # tl.dot is not portable on the HIP backend, see plan-4 P26 note.)
        qk = tl.sum(kl.to(tl.float32) * q[None, :].to(tl.float32), axis=1) * sm_scale

        # SWA-causal mask: keep n in [pid_m - SWA_WINDOW + 1, pid_m].
        # When SWA_WINDOW <= 0 fall back to full causal.
        if SWA_WINDOW > 0:
            in_window = (offs_n >= pid_m - SWA_WINDOW + 1) & (offs_n <= pid_m)
        else:
            in_window = offs_n <= pid_m
        qk = tl.where(in_window, qk, NEG_INF)
        # Boundary mask for keys past seqlen_q.
        qk = tl.where(offs_n < seqlen_q, qk, NEG_INF)

        # Online softmax update (shared with sparse branch + sink).
        m_tile = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, m_tile)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        # V_local tile: [BLOCK_N, BLOCK_DMODEL] in v_local.dtype
        vl_ptrs = (
            V_LOCAL
            + bid * stride_vlb
            + qhid * stride_vlh
            + offs_n[:, None] * stride_vln
            + offs_d[None, :] * stride_vld
        )
        vl = tl.load(vl_ptrs, mask=kl_load_mask, other=0.0)

        # acc += sum_n (p[n] * vl[n, :]) — fp32 accumulator.
        acc = acc * alpha + tl.sum(p[:, None] * vl.to(tl.float32), axis=0)
        m_i = m_new

    # ---- Sparse top-K branch ----------------------------------------------
    # gathered is per-query (no H dim — broadcast across heads in the
    # eager reference). We walk K_topk in BLOCK_K tiles.
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

        # qk_sparse = sum_d (g[k, d] * q[d]) -> [BLOCK_K]
        qk_sparse = tl.sum(g.to(tl.float32) * q[None, :].to(tl.float32), axis=1) * sm_scale

        # Caller-supplied sparse_mask: -inf for topk_idx == -1 entries.
        sm_ptrs = SPARSE_MASK + bid * stride_smb + pid_m * stride_smm + offs_k * stride_smk
        sm_load_mask = offs_k < K_topk
        sm = tl.load(sm_ptrs, mask=sm_load_mask, other=0.0).to(tl.float32)
        qk_sparse = qk_sparse + sm

        # Boundary mask for offs_k past K_topk.
        qk_sparse = tl.where(offs_k < K_topk, qk_sparse, NEG_INF)

        # Online softmax update — shares m_i / l_i with the local branch.
        m_tile = tl.max(qk_sparse, axis=0)
        m_new = tl.maximum(m_i, m_tile)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk_sparse - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        # acc += sum_k (p[k] * g[k, :]) — fp32 accumulator.
        acc = acc * alpha + tl.sum(p[:, None] * g.to(tl.float32), axis=0)
        m_i = m_new

    # ---- Sink (joint over both branches) ----------------------------------
    if HAS_SINK:
        sink_h = tl.load(SINK + qhid).to(tl.float32)
        m_new = tl.maximum(m_i, sink_h)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(sink_h - m_new)
        l_i = l_i * alpha + beta
        acc = acc * alpha
        m_i = m_new

    # ---- Final divide + cast back to output dtype -------------------------
    out = acc / l_i
    lse = m_i + tl.log(l_i)

    out_offset = bid * stride_ob + qhid * stride_oh + pid_m * stride_om
    out_ptrs = OUT + out_offset + offs_d * stride_od
    tl.store(out_ptrs, out.to(OUT.dtype.element_ty), mask=q_active)

    lse_ptr = LSE + bid * stride_lb + qhid * stride_lh + pid_m * stride_lm
    tl.store(lse_ptr, lse, mask=q_active)


@triton.jit
def _v4_csa_attention_pool_fwd_kernel(
    Q,
    K_LOCAL,
    V_LOCAL,
    POOL,
    TOPK_IDXS,
    SINK,
    OUT,
    LSE,
    # Q strides: [B, H, Sq, D]
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    # K_local strides: [B, H, Sq, D]
    stride_klb,
    stride_klh,
    stride_kln,
    stride_kld,
    # V_local strides: [B, H, Sq, D]
    stride_vlb,
    stride_vlh,
    stride_vln,
    stride_vld,
    # Pool strides: [B, P, D] (shared across heads)
    stride_pb,
    stride_pp,
    stride_pd,
    # topk_idxs strides: [B, Sq, K_topk]
    stride_tib,
    stride_tim,
    stride_tik,
    # OUT strides: [B, H, Sq, D]
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    # LSE strides: [B, H, Sq]
    stride_lb,
    stride_lh,
    stride_lm,
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
):
    """CSA FWD with in-kernel topk gather from the compressed pool."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    bid = pid_bh // HEAD_Q
    qhid = pid_bh % HEAD_Q

    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_active = pid_m < seqlen_q
    NEG_INF: tl.constexpr = -1.0e30

    q_row_offset = bid * stride_qb + qhid * stride_qh + pid_m * stride_qm
    q = tl.load(Q + q_row_offset + offs_d * stride_qd, mask=q_active, other=0.0)

    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.full((), value=NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)

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
        qk = tl.sum(kl.to(tl.float32) * q[None, :].to(tl.float32), axis=1) * sm_scale

        if SWA_WINDOW > 0:
            in_window = (offs_n >= pid_m - SWA_WINDOW + 1) & (offs_n <= pid_m)
        else:
            in_window = offs_n <= pid_m
        qk = tl.where(in_window & (offs_n < seqlen_q), qk, NEG_INF)

        m_tile = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, m_tile)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        vl_ptrs = (
            V_LOCAL
            + bid * stride_vlb
            + qhid * stride_vlh
            + offs_n[:, None] * stride_vln
            + offs_d[None, :] * stride_vld
        )
        vl = tl.load(vl_ptrs, mask=kl_load_mask, other=0.0)
        acc = acc * alpha + tl.sum(p[:, None] * vl.to(tl.float32), axis=0)
        m_i = m_new

    for k_start in range(0, K_topk, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        topk_ptrs = TOPK_IDXS + bid * stride_tib + pid_m * stride_tim + offs_k * stride_tik
        topk = tl.load(topk_ptrs, mask=offs_k < K_topk, other=-1)
        valid = (offs_k < K_topk) & (topk >= 0) & (topk < pool_size)
        safe_topk = tl.where(valid, topk, 0)

        pool_ptrs = POOL + bid * stride_pb + safe_topk[:, None] * stride_pp + offs_d[None, :] * stride_pd
        pool = tl.load(pool_ptrs, mask=valid[:, None], other=0.0)

        qk_sparse = tl.sum(pool.to(tl.float32) * q[None, :].to(tl.float32), axis=1) * sm_scale
        qk_sparse = tl.where(valid, qk_sparse, NEG_INF)

        m_tile = tl.max(qk_sparse, axis=0)
        m_new = tl.maximum(m_i, m_tile)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk_sparse - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        acc = acc * alpha + tl.sum(p[:, None] * pool.to(tl.float32), axis=0)
        m_i = m_new

    if HAS_SINK:
        sink_h = tl.load(SINK + qhid).to(tl.float32)
        m_new = tl.maximum(m_i, sink_h)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(sink_h - m_new)
        l_i = l_i * alpha + beta
        acc = acc * alpha
        m_i = m_new

    out = acc / l_i
    lse = m_i + tl.log(l_i)

    out_offset = bid * stride_ob + qhid * stride_oh + pid_m * stride_om
    tl.store(OUT + out_offset + offs_d * stride_od, out.to(OUT.dtype.element_ty), mask=q_active)

    lse_ptr = LSE + bid * stride_lb + qhid * stride_lh + pid_m * stride_lm
    tl.store(lse_ptr, lse, mask=q_active)


# ---------------------------------------------------------------------------
# Python launcher
# ---------------------------------------------------------------------------


def _launch_v4_csa_attention_fwd(
    q: torch.Tensor,  # [B, H, Sq, D]
    k_local: torch.Tensor,  # [B, H, Sq, D]
    v_local: torch.Tensor,  # [B, H, Sq, D]
    gathered: torch.Tensor,  # [B, Sq, K_topk, D]
    sparse_mask: torch.Tensor,  # [B, Sq, K_topk]
    *,
    sink: Optional[torch.Tensor],  # [H] or None
    swa_window: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the V4 CSA attention forward kernel.

    Returns ``(out, lse)`` where ``out`` matches ``v_local.dtype`` and
    ``lse`` is fp32. ``lse`` is what the BWD kernel needs to
    re-materialise the joint softmax without storing the
    ``[Sq, Sq + K_topk]`` joint ``P`` matrix.
    """
    if q.dim() != 4 or k_local.dim() != 4 or v_local.dim() != 4:
        raise ValueError(
            "v4_csa_attention forward expects q / k_local / v_local of rank 4 "
            f"(got {q.dim()} / {k_local.dim()} / {v_local.dim()})"
        )
    if gathered.dim() != 4:
        raise ValueError(
            f"v4_csa_attention forward expects gathered of rank 4 [B, Sq, K, D]; "
            f"got rank {gathered.dim()}, shape {tuple(gathered.shape)}"
        )
    if sparse_mask.dim() != 3:
        raise ValueError(
            f"v4_csa_attention forward expects sparse_mask of rank 3 [B, Sq, K]; "
            f"got rank {sparse_mask.dim()}, shape {tuple(sparse_mask.shape)}"
        )

    B, HQ, Sq, D = q.shape
    if k_local.shape != q.shape or v_local.shape != q.shape:
        raise ValueError(
            "v4_csa_attention requires k_local.shape == v_local.shape == q.shape "
            f"(got q={tuple(q.shape)}, k_local={tuple(k_local.shape)}, "
            f"v_local={tuple(v_local.shape)})."
        )

    Bg, Sqg, K_topk, Dg = gathered.shape
    if Bg != B or Sqg != Sq or Dg != D:
        raise ValueError(
            "v4_csa_attention gathered shape mismatch: expected "
            f"[B, Sq, K, D] = [{B}, {Sq}, *, {D}]; got {tuple(gathered.shape)}."
        )
    Bm, Sqm, Km = sparse_mask.shape
    if Bm != B or Sqm != Sq or Km != K_topk:
        raise ValueError(
            "v4_csa_attention sparse_mask shape mismatch: expected "
            f"[B, Sq, K] = [{B}, {Sq}, {K_topk}]; got {tuple(sparse_mask.shape)}."
        )

    if not q.is_cuda:
        raise ValueError("v4_csa_attention requires CUDA / HIP tensors.")
    if q.dtype != k_local.dtype or q.dtype != v_local.dtype or q.dtype != gathered.dtype:
        raise ValueError(
            "v4_csa_attention requires q.dtype == k_local.dtype == v_local.dtype "
            f"== gathered.dtype (got {q.dtype} / {k_local.dtype} / "
            f"{v_local.dtype} / {gathered.dtype})."
        )

    has_sink = sink is not None

    out = torch.empty_like(q)
    lse = torch.empty((B, HQ, Sq), device=q.device, dtype=torch.float32)

    # Tile sizes: BLOCK_N / BLOCK_K conservative for SMEM at head_dim=512.
    # Per-row design (one program per (b, qhid, m)) means the gathered
    # tile is [BLOCK_K, D] only; multi-row would balloon SMEM.
    BLOCK_N = 32
    BLOCK_K = 32
    BLOCK_DMODEL = D  # head_dim must be a power of 2 for tl.arange

    grid = (Sq, B * HQ)

    # Sentinel pointer when sink is absent. Triton requires a real tensor —
    # we pass q (any tensor) and gate via the constexpr.
    sink_ptr = sink if has_sink else q

    _v4_csa_attention_fwd_kernel[grid](
        q,
        k_local,
        v_local,
        gathered,
        sparse_mask,
        sink_ptr,
        out,
        lse,
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
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
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
    return out, lse


def _launch_v4_csa_attention_pool_fwd(
    q: torch.Tensor,  # [B, H, Sq, D]
    k_local: torch.Tensor,  # [B, H, Sq, D]
    v_local: torch.Tensor,  # [B, H, Sq, D]
    pool: torch.Tensor,  # [B, P, D]
    topk_idxs: torch.Tensor,  # [B, Sq, K_topk], -1 means masked
    *,
    sink: Optional[torch.Tensor],
    swa_window: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch CSA forward with in-kernel topk gather from ``pool``."""
    if q.dim() != 4 or k_local.dim() != 4 or v_local.dim() != 4:
        raise ValueError(
            "v4_csa_attention pool forward expects q / k_local / v_local of rank 4 "
            f"(got {q.dim()} / {k_local.dim()} / {v_local.dim()})"
        )
    if pool.dim() != 3:
        raise ValueError(
            f"v4_csa_attention pool forward expects pool of rank 3 [B, P, D]; "
            f"got rank {pool.dim()}, shape {tuple(pool.shape)}"
        )
    if topk_idxs.dim() != 3:
        raise ValueError(
            f"v4_csa_attention pool forward expects topk_idxs of rank 3 [B, Sq, K]; "
            f"got rank {topk_idxs.dim()}, shape {tuple(topk_idxs.shape)}"
        )

    B, HQ, Sq, D = q.shape
    if k_local.shape != q.shape or v_local.shape != q.shape:
        raise ValueError(
            "v4_csa_attention pool path requires k_local.shape == v_local.shape == q.shape "
            f"(got q={tuple(q.shape)}, k_local={tuple(k_local.shape)}, "
            f"v_local={tuple(v_local.shape)})."
        )
    Bp, P, Dp = pool.shape
    if Bp != B or Dp != D:
        raise ValueError(
            "v4_csa_attention pool shape mismatch: expected "
            f"[B, P, D] = [{B}, *, {D}]; got {tuple(pool.shape)}."
        )
    Bt, Sqt, K_topk = topk_idxs.shape
    if Bt != B or Sqt != Sq:
        raise ValueError(
            "v4_csa_attention topk_idxs shape mismatch: expected "
            f"[B, Sq, K] = [{B}, {Sq}, *]; got {tuple(topk_idxs.shape)}."
        )
    if topk_idxs.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"v4_csa_attention topk_idxs must be int32/int64, got {topk_idxs.dtype}.")
    if not q.is_cuda:
        raise ValueError("v4_csa_attention requires CUDA / HIP tensors.")
    if q.dtype != k_local.dtype or q.dtype != v_local.dtype or q.dtype != pool.dtype:
        raise ValueError(
            "v4_csa_attention pool path requires q.dtype == k_local.dtype == "
            f"v_local.dtype == pool.dtype (got {q.dtype} / {k_local.dtype} / "
            f"{v_local.dtype} / {pool.dtype})."
        )

    has_sink = sink is not None

    # Plan-5 P32: default to the split FWD (local SWA dense kernel +
    # head-block sparse kernel + LSE merge). The monolithic per-row
    # ``_v4_csa_attention_pool_fwd_kernel`` stays in tree as the
    # ``PRIMUS_V4_CSA_FWD_FORCE_MONOLITHIC=1`` fallback so we can A/B
    # the two designs from the proxy without rebuilding.
    if os.getenv("PRIMUS_V4_CSA_FWD_FORCE_MONOLITHIC", "0") != "1":
        return _launch_v4_csa_attention_pool_fwd_split(
            q,
            k_local,
            v_local,
            pool,
            topk_idxs,
            sink=sink,
            swa_window=int(swa_window),
            scale=float(scale),
        )

    out = torch.empty_like(q)
    lse = torch.empty((B, HQ, Sq), device=q.device, dtype=torch.float32)

    BLOCK_N = 32
    BLOCK_K = 32
    BLOCK_DMODEL = D
    grid = (Sq, B * HQ)
    sink_ptr = sink if has_sink else q

    _v4_csa_attention_pool_fwd_kernel[grid](
        q,
        k_local,
        v_local,
        pool,
        topk_idxs,
        sink_ptr,
        out,
        lse,
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
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
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
        num_warps=4,
        num_stages=1,
    )
    return out, lse


# ---------------------------------------------------------------------------
# Plan-5 P32 split CSA FWD — sparse head-block tile + LSE merge
#
# The per-row design in ``_v4_csa_attention_pool_fwd_kernel`` cannot reach
# tensor-core throughput because ``tl.sum(k * q, axis=1)`` is a per-row
# reduction (one program per ``(b, qhid, m)``). FlashAttention-style
# multi-row ``tl.dot`` tiles need ``BLOCK_M >= 16`` queries per program;
# the sparse branch's per-query ``topk_idxs`` gather blocks that — adjacent
# query rows have different sparse keys.
#
# P32 splits the FWD into three launches that ARE multi-row friendly:
#
#  1. ``_launch_v4_attention_fwd`` (already shipped) handles the local
#     SWA branch with ``BLOCK_M=BLOCK_N=32`` ``tl.dot`` tiles. We call it
#     with ``sink=None`` so the returned ``(out_local, lse_local)`` does
#     NOT include the per-head sink.
#  2. ``_v4_csa_attention_pool_sparse_fwd_kernel`` (new) handles the
#     sparse pool branch with a **head-block** tile. The per-query
#     ``topk_idxs`` gather is shared across all ``H`` query heads — the
#     pool tensor itself has no head dimension. One program owns one
#     ``(b, m, h_block)`` and runs ``tl.dot(Q[BLOCK_H, D],
#     tl.trans(pool[BLOCK_K, D]))`` per top-K tile, online-softmax-
#     updating per-head ``m_i / l_i / acc`` along the way. Output
#     ``(out_sparse, lse_sparse)`` does NOT include the sink.
#  3. ``_v4_csa_attention_lse_merge_kernel`` (new) combines the two
#     ``(out, lse)`` pairs with the per-head sink under one final online
#     softmax. Result: a joint ``out`` and joint ``lse`` mathematically
#     identical to the monolithic kernel (modulo fp32 reduction order),
#     plus the per-iteration BWD contract is unchanged because the joint
#     ``lse`` is what the BWD already re-materialises ``P`` from.
# ---------------------------------------------------------------------------


@triton.jit
def _v4_csa_attention_pool_sparse_fwd_kernel(
    Q,
    POOL,
    TOPK_IDXS,
    OUT,
    LSE,
    # Q strides: [B, H, Sq, D]
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    # Pool strides: [B, P, D]
    stride_pb,
    stride_pp,
    stride_pd,
    # topk_idxs strides: [B, Sq, K_topk]
    stride_tib,
    stride_tim,
    stride_tik,
    # OUT strides: [B, H, Sq, D]
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    # LSE strides: [B, H, Sq]
    stride_lb,
    stride_lh,
    stride_lm,
    seqlen_q,
    pool_size,
    K_topk,
    sm_scale,
    HEAD_Q: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Sparse-branch CSA FWD with head-block tile + ``tl.dot``.

    Grid: ``(seqlen_q, cdiv(HEAD_Q, BLOCK_H), B)``. Each program owns
    one ``(b, h_block, m)`` and computes the sparse branch's normalized
    output + LSE for ``BLOCK_H`` heads. The pool gather is shared
    across heads because ``pool`` and ``topk_idxs`` have no H axis.
    """
    pid_m = tl.program_id(0)
    pid_h_block = tl.program_id(1)
    bid = tl.program_id(2)

    offs_h = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    h_mask = offs_h < HEAD_Q
    q_active = pid_m < seqlen_q

    NEG_INF: tl.constexpr = -1.0e30

    # Q tile: [BLOCK_H, BLOCK_DMODEL]
    q_ptrs = (
        Q + bid * stride_qb + offs_h[:, None] * stride_qh + pid_m * stride_qm + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=h_mask[:, None] & q_active, other=0.0)

    # Online-softmax state per head.
    acc = tl.zeros([BLOCK_H, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.full([BLOCK_H], value=NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    for k_start in range(0, K_topk, BLOCK_K):
        sparse_k = k_start + offs_k
        topk_ptrs = TOPK_IDXS + bid * stride_tib + pid_m * stride_tim + sparse_k * stride_tik
        topk = tl.load(topk_ptrs, mask=sparse_k < K_topk, other=-1)
        valid_k = (sparse_k < K_topk) & (topk >= 0) & (topk < pool_size)
        safe_topk = tl.where(valid_k, topk, 0)

        pool_ptrs = POOL + bid * stride_pb + safe_topk[:, None] * stride_pp + offs_d[None, :] * stride_pd
        pool = tl.load(pool_ptrs, mask=valid_k[:, None], other=0.0)

        # qk = Q @ pool.T : [BLOCK_H, BLOCK_K] in fp32 accumulator
        qk = tl.dot(q.to(pool.dtype), tl.trans(pool)) * sm_scale
        qk = tl.where((h_mask[:, None] & valid_k[None, :] & q_active), qk, NEG_INF)

        # Online softmax update per head row.
        m_tile = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_tile)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(pool.dtype), pool)
        m_i = m_new

    # Detect "all invalid" rows (K_topk == 0 or every topk_idx == -1):
    # l_i stays zero; we leave acc as zero and write a NEG_INF lse so the
    # merge kernel treats the sparse branch as carrying zero softmax mass
    # for that (b, h, m).
    empty = l_i == 0.0
    safe_l = tl.where(empty, 1.0, l_i)
    out = acc / safe_l[:, None]
    lse = tl.where(empty, NEG_INF, m_i + tl.log(safe_l))

    out_ptrs = (
        OUT + bid * stride_ob + offs_h[:, None] * stride_oh + pid_m * stride_om + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out.to(OUT.dtype.element_ty), mask=h_mask[:, None] & q_active)

    lse_ptrs = LSE + bid * stride_lb + offs_h * stride_lh + pid_m * stride_lm
    tl.store(lse_ptrs, lse, mask=h_mask & q_active)


@triton.jit
def _v4_csa_attention_pool_sparse_merge_fwd_kernel(
    Q,
    POOL,
    TOPK_IDXS,
    OUT_LOCAL,  # local-branch result (no sink) — input
    LSE_LOCAL,  # local-branch lse (no sink) — input
    SINK,
    OUT,  # joint output
    LSE,  # joint lse
    # Q strides: [B, H, Sq, D]
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    # Pool strides: [B, P, D]
    stride_pb,
    stride_pp,
    stride_pd,
    # topk_idxs strides: [B, Sq, K_topk]
    stride_tib,
    stride_tim,
    stride_tik,
    # OUT_LOCAL strides: [B, H, Sq, D]
    stride_olb,
    stride_olh,
    stride_olm,
    stride_old,
    # LSE_LOCAL strides: [B, H, Sq]
    stride_llb,
    stride_llh,
    stride_llm,
    # OUT (joint) strides: [B, H, Sq, D]
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    # LSE (joint) strides: [B, H, Sq]
    stride_lb,
    stride_lh,
    stride_lm,
    seqlen_q,
    pool_size,
    K_topk,
    sm_scale,
    HEAD_Q: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    K_DIVISIBLE: tl.constexpr,  # True iff K_topk % BLOCK_K == 0
    H_DIVISIBLE: tl.constexpr,  # True iff HEAD_Q  % BLOCK_H == 0
):
    """Sparse-branch CSA FWD + joint merge fused.

    Plan-8 P57 cr=4 FWD speedup: eliminates the separate
    ``_v4_csa_attention_lse_merge_kernel`` launch by reading
    ``out_local`` / ``lse_local`` at the tail of this kernel and writing
    the joint output directly. Saves one kernel launch (~10 us) plus the
    intermediate ``[B, H, Sq, D]`` ``out_sparse`` HBM round-trip
    (~250 MiB read + 250 MiB write = ~100 us at MI355X HBM speed).

    Grid: ``(seqlen_q, cdiv(HEAD_Q, BLOCK_H), B)`` — same as the
    non-fused sparse kernel. Each program owns one ``(b, h_block, m)``
    and produces the joint ``out[BLOCK_H, BLOCK_DMODEL]`` row.

    Math (joint softmax over local + sparse + sink, all with
    ``-m_max`` rescaling):

        m_max = max(lse_local, lse_sparse, sink_h)
        denom = exp(lse_local - m_max)
              + exp(lse_sparse - m_max)
              + exp(sink_h    - m_max)     # 0 if no sink
        joint_out = (out_local * exp(lse_local  - m_max)
                  +  out_sparse * exp(lse_sparse - m_max)) / denom
        joint_lse = m_max + log(denom)

    where ``out_sparse = acc / l_sparse`` is the normalized sparse
    output (computed inside this kernel, never materialised to HBM).
    """
    pid_m = tl.program_id(0)
    pid_h_block = tl.program_id(1)
    bid = tl.program_id(2)

    offs_h = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    if H_DIVISIBLE:
        h_mask = tl.full([BLOCK_H], True, dtype=tl.int1)
    else:
        h_mask = offs_h < HEAD_Q

    NEG_INF: tl.constexpr = -1.0e30

    # Q tile: [BLOCK_H, BLOCK_DMODEL]
    q_ptrs = (
        Q + bid * stride_qb + offs_h[:, None] * stride_qh + pid_m * stride_qm + offs_d[None, :] * stride_qd
    )
    if H_DIVISIBLE:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)

    # Online-softmax state for the sparse branch.
    acc = tl.zeros([BLOCK_H, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.full([BLOCK_H], value=NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Hoist loop-invariant pointer offsets out of the K-loop. Triton
    # usually does this but spelling it out keeps the inner body lean.
    topk_base = TOPK_IDXS + bid * stride_tib + pid_m * stride_tim
    pool_base = POOL + bid * stride_pb
    pool_d_offs = offs_d[None, :] * stride_pd

    for k_start in range(0, K_topk, BLOCK_K):
        sparse_k = k_start + offs_k
        topk_ptrs = topk_base + sparse_k * stride_tik
        if K_DIVISIBLE:
            topk = tl.load(topk_ptrs)
            valid_k = (topk >= 0) & (topk < pool_size)
        else:
            topk = tl.load(topk_ptrs, mask=sparse_k < K_topk, other=-1)
            valid_k = (sparse_k < K_topk) & (topk >= 0) & (topk < pool_size)
        safe_topk = tl.where(valid_k, topk, 0)

        pool_ptrs = pool_base + safe_topk[:, None] * stride_pp + pool_d_offs
        pool = tl.load(pool_ptrs, mask=valid_k[:, None], other=0.0)

        qk = tl.dot(q.to(pool.dtype), tl.trans(pool), out_dtype=tl.float32) * sm_scale
        if H_DIVISIBLE:
            qk = tl.where(valid_k[None, :], qk, NEG_INF)
        else:
            qk = tl.where(h_mask[:, None] & valid_k[None, :], qk, NEG_INF)

        m_tile = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_tile)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(pool.dtype), pool, out_dtype=tl.float32)
        m_i = m_new

    # Detect "all invalid" sparse rows (every topk_idx == -1 or
    # K_topk == 0): l_sparse stays zero so the sparse branch carries
    # zero softmax mass for that (b, h, m).
    empty_sparse = l_i == 0.0
    safe_l_sparse = tl.where(empty_sparse, 1.0, l_i)
    out_sparse_norm = acc / safe_l_sparse[:, None]  # [BLOCK_H, BLOCK_DMODEL] fp32
    lse_sparse = tl.where(empty_sparse, NEG_INF, m_i + tl.log(safe_l_sparse))

    # ---- Load local-branch output + lse and merge with sink -------------
    out_local_ptrs = (
        OUT_LOCAL
        + bid * stride_olb
        + offs_h[:, None] * stride_olh
        + pid_m * stride_olm
        + offs_d[None, :] * stride_old
    )
    out_local = tl.load(out_local_ptrs, mask=h_mask[:, None], other=0.0).to(tl.float32)

    lse_local_ptrs = LSE_LOCAL + bid * stride_llb + offs_h * stride_llh + pid_m * stride_llm
    lse_local = tl.load(lse_local_ptrs, mask=h_mask, other=NEG_INF)

    if HAS_SINK:
        sink_h = tl.load(SINK + offs_h, mask=h_mask, other=NEG_INF).to(tl.float32)
    else:
        sink_h = tl.full([BLOCK_H], value=NEG_INF, dtype=tl.float32)

    m_max = tl.maximum(lse_local, lse_sparse)
    if HAS_SINK:
        m_max = tl.maximum(m_max, sink_h)

    alpha_local = tl.exp(lse_local - m_max)
    alpha_sparse = tl.exp(lse_sparse - m_max)
    if HAS_SINK:
        alpha_sink = tl.exp(sink_h - m_max)
    else:
        alpha_sink = tl.zeros([BLOCK_H], dtype=tl.float32)

    denom = alpha_local + alpha_sparse + alpha_sink
    # Empty-row safety: if every branch is NEG_INF, denom == 0. Use a
    # safe denom and rely on the alpha_* being zero so the numerator is
    # zero too — output stays at 0 / 1 = 0, lse remains NEG_INF.
    safe_denom = tl.where(denom == 0.0, 1.0, denom)

    joint_out = (out_local * alpha_local[:, None] + out_sparse_norm * alpha_sparse[:, None]) / safe_denom[
        :, None
    ]
    joint_lse = tl.where(denom == 0.0, NEG_INF, m_max + tl.log(safe_denom))

    out_ptrs = (
        OUT + bid * stride_ob + offs_h[:, None] * stride_oh + pid_m * stride_om + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, joint_out.to(OUT.dtype.element_ty), mask=h_mask[:, None])

    lse_ptrs = LSE + bid * stride_lb + offs_h * stride_lh + pid_m * stride_lm
    tl.store(lse_ptrs, joint_lse, mask=h_mask)


@triton.jit
def _v4_csa_attention_lse_merge_kernel(
    OUT_LOCAL,
    LSE_LOCAL,
    OUT_SPARSE,
    LSE_SPARSE,
    SINK,
    OUT,
    LSE,
    stride_olb,
    stride_olh,
    stride_olm,
    stride_old,
    stride_llb,
    stride_llh,
    stride_llm,
    stride_osb,
    stride_osh,
    stride_osm,
    stride_osd,
    stride_lsb,
    stride_lsh,
    stride_lsm,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lm,
    seqlen_q,
    HEAD_Q: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Merge ``(out_local, lse_local)`` and ``(out_sparse, lse_sparse)``
    with the per-head sink under one final online softmax.

    Grid: ``(cdiv(seqlen_q, BLOCK_M), B * HEAD_Q)``. Each program owns
    one ``[BLOCK_M, BLOCK_DMODEL]`` slice of the joint output.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    bid = pid_bh // HEAD_Q
    qhid = pid_bh % HEAD_Q

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    m_mask = offs_m < seqlen_q

    NEG_INF: tl.constexpr = -1.0e30

    lse_local = tl.load(
        LSE_LOCAL + bid * stride_llb + qhid * stride_llh + offs_m * stride_llm,
        mask=m_mask,
        other=NEG_INF,
    )
    lse_sparse = tl.load(
        LSE_SPARSE + bid * stride_lsb + qhid * stride_lsh + offs_m * stride_lsm,
        mask=m_mask,
        other=NEG_INF,
    )

    out_local = tl.load(
        OUT_LOCAL
        + bid * stride_olb
        + qhid * stride_olh
        + offs_m[:, None] * stride_olm
        + offs_d[None, :] * stride_old,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    out_sparse = tl.load(
        OUT_SPARSE
        + bid * stride_osb
        + qhid * stride_osh
        + offs_m[:, None] * stride_osm
        + offs_d[None, :] * stride_osd,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    if HAS_SINK:
        sink_h = tl.load(SINK + qhid).to(tl.float32)
    else:
        sink_h = NEG_INF

    m_max = tl.maximum(lse_local, lse_sparse)
    if HAS_SINK:
        m_max = tl.maximum(m_max, sink_h)

    alpha_local = tl.exp(lse_local - m_max)
    alpha_sparse = tl.exp(lse_sparse - m_max)
    if HAS_SINK:
        alpha_sink = tl.exp(sink_h - m_max)
    else:
        alpha_sink = tl.zeros_like(alpha_local)

    denom = alpha_local + alpha_sparse + alpha_sink
    # Empty-row safety: if every branch is NEG_INF, denom == 0. Use a
    # safe denom and rely on the alpha_* being zero so the numerator is
    # zero too — output stays at 0 / 1 = 0, lse remains NEG_INF.
    safe_denom = tl.where(denom == 0.0, 1.0, denom)

    out = (out_local * alpha_local[:, None] + out_sparse * alpha_sparse[:, None]) / safe_denom[:, None]
    lse = tl.where(denom == 0.0, NEG_INF, m_max + tl.log(safe_denom))

    out_ptrs = (
        OUT + bid * stride_ob + qhid * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out.to(OUT.dtype.element_ty), mask=m_mask[:, None])

    lse_ptrs = LSE + bid * stride_lb + qhid * stride_lh + offs_m * stride_lm
    tl.store(lse_ptrs, lse, mask=m_mask)


def _launch_v4_csa_attention_pool_fwd_split(
    q: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    pool: torch.Tensor,
    topk_idxs: torch.Tensor,
    *,
    sink: Optional[torch.Tensor],
    swa_window: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """P32 split CSA FWD: local SWA via dense kernel + sparse head-block + LSE merge."""
    B, HQ, Sq, D = q.shape
    P = pool.shape[1]
    K_topk = topk_idxs.shape[2]
    has_sink = sink is not None

    # Step 1: local SWA branch (no sink — applied in the merge).
    out_local, lse_local = _launch_v4_attention_fwd(
        q,
        k_local,
        v_local,
        sink=None,
        swa_window=int(swa_window) if swa_window > 0 else 0,
        additive_mask=None,
        scale=float(scale),
        hca_local_seqlen=0,
    )

    # P57 attempt-11: tile sweep on the fused sparse+merge kernel found
    # (BLOCK_H=64, BLOCK_K=16, num_warps=8, num_stages=3) wins on the
    # V4-Flash widths (H=64 D=512). The intuition:
    #
    # * BLOCK_H=64 keeps the full head axis in a single program, so the
    #   pool gather (which is per-(b, m) — no H dim) is read once and
    #   reused across all H query heads.
    # * BLOCK_K=16 halves the per-tile pool tile (16 × 512 × 2 B = 16 KiB)
    #   so the AMD Triton software pipeliner can keep 3 stages live in
    #   LDS (3 × 16 KiB = 48 KiB pool buffer) without crowding out the
    #   `acc` register accumulator. With BLOCK_K=32 the pool tile is
    #   2× larger and 3-stage pipelining ran out of resources.
    # * num_stages=3 overlaps pool gather → `tl.dot` → softmax update
    #   across 3 K-tile iterations, hiding the gather latency.
    #
    # P57 R2 re-confirmed the BLOCK_K/stages winner via a fresh BK x
    # ST x NW sweep (`p57/r2_sweep.sh`); BLOCK_K=32 ran out of LDS at
    # num_stages>=2 and regressed 2-3 x. The R2 win for cr=4 FWD comes
    # from the *local-SWA* kernel re-tile, not this kernel.
    BLOCK_H = 64 if HQ >= 64 else triton.next_power_of_2(HQ)
    if BLOCK_H > HQ:
        BLOCK_H = max(triton.next_power_of_2(HQ), 16)
    BLOCK_K = 16
    BLOCK_DMODEL = D
    sparse_grid = (Sq, triton.cdiv(HQ, BLOCK_H), B)
    sink_arg = sink if has_sink else q

    # P57 attempt-3: fuse sparse + merge into one kernel. The legacy
    # 2-kernel path (sparse → ``out_sparse``/``lse_sparse``; merge)
    # required a ~500 MiB intermediate HBM round-trip plus a ~135 us
    # merge launch. The fused kernel reads ``out_local`` / ``lse_local``
    # at the tail of the sparse loop and writes joint ``out`` / ``lse``
    # directly. Env-gated A/B fallback:
    # ``PRIMUS_V4_CSA_FWD_SEPARATE_MERGE=1`` keeps the split layout for
    # parity debugging.
    out = torch.empty_like(q)
    lse = torch.empty((B, HQ, Sq), device=q.device, dtype=torch.float32)

    if os.getenv("PRIMUS_V4_CSA_FWD_SEPARATE_MERGE", "0") != "1":
        _v4_csa_attention_pool_sparse_merge_fwd_kernel[sparse_grid](
            q,
            pool,
            topk_idxs,
            out_local,
            lse_local,
            sink_arg,
            out,
            lse,
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
            out_local.stride(0),
            out_local.stride(1),
            out_local.stride(2),
            out_local.stride(3),
            lse_local.stride(0),
            lse_local.stride(1),
            lse_local.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            Sq,
            P,
            K_topk,
            float(scale),
            HEAD_Q=HQ,
            HAS_SINK=has_sink,
            BLOCK_H=BLOCK_H,
            BLOCK_K=BLOCK_K,
            BLOCK_DMODEL=BLOCK_DMODEL,
            K_DIVISIBLE=(K_topk % BLOCK_K == 0),
            H_DIVISIBLE=(HQ % BLOCK_H == 0),
            num_warps=8,
            num_stages=3,
        )
        return out, lse

    # ---- legacy split-merge path (env-gated for A/B debugging) ----------
    out_sparse = torch.empty_like(q)
    lse_sparse = torch.empty((B, HQ, Sq), device=q.device, dtype=torch.float32)
    _v4_csa_attention_pool_sparse_fwd_kernel[sparse_grid](
        q,
        pool,
        topk_idxs,
        out_sparse,
        lse_sparse,
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
        out_sparse.stride(0),
        out_sparse.stride(1),
        out_sparse.stride(2),
        out_sparse.stride(3),
        lse_sparse.stride(0),
        lse_sparse.stride(1),
        lse_sparse.stride(2),
        Sq,
        P,
        K_topk,
        float(scale),
        HEAD_Q=HQ,
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=8,
        num_stages=1,
    )
    MERGE_BLOCK_M = 32
    merge_grid = (triton.cdiv(Sq, MERGE_BLOCK_M), B * HQ)
    _v4_csa_attention_lse_merge_kernel[merge_grid](
        out_local,
        lse_local,
        out_sparse,
        lse_sparse,
        sink_arg,
        out,
        lse,
        out_local.stride(0),
        out_local.stride(1),
        out_local.stride(2),
        out_local.stride(3),
        lse_local.stride(0),
        lse_local.stride(1),
        lse_local.stride(2),
        out_sparse.stride(0),
        out_sparse.stride(1),
        out_sparse.stride(2),
        out_sparse.stride(3),
        lse_sparse.stride(0),
        lse_sparse.stride(1),
        lse_sparse.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        Sq,
        HEAD_Q=HQ,
        HAS_SINK=has_sink,
        BLOCK_M=MERGE_BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=1,
    )
    return out, lse


__all__ = [
    "_v4_csa_attention_fwd_kernel",
    "_v4_csa_attention_pool_fwd_kernel",
    "_v4_csa_attention_pool_sparse_fwd_kernel",
    "_v4_csa_attention_pool_sparse_merge_fwd_kernel",
    "_v4_csa_attention_lse_merge_kernel",
    "_launch_v4_csa_attention_fwd",
    "_launch_v4_csa_attention_pool_fwd",
    "_launch_v4_csa_attention_pool_fwd_split",
]
