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

from typing import Optional

import torch
import triton
import triton.language as tl

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


__all__ = [
    "_v4_csa_attention_fwd_kernel",
    "_launch_v4_csa_attention_fwd",
]
