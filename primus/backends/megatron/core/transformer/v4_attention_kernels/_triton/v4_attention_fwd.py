###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""V4 attention forward Triton kernel (plan-4 P25, ``compress_ratio in {0, 128}``).

FlashAttention-style block-wise online softmax that handles V4's exact
shape envelope:

* ``head_dim = 512`` (single tile — no partial-RoPE / NOPE split because
  RoPE is applied outside the kernel);
* MQA single-latent KV (``K.shape[1] == 1``) broadcast across query
  heads, *or* full MHA (``K.shape[1] == H``);
* optional per-head learned softmax sink (``[H]``), joined as a virtual
  key column with notional value zero at the end of the K-loop;
* optional sliding-window-causal mask (``swa_window > 0``) applied
  in-kernel;
* optional caller-supplied additive bias. The generic path accepts
  ``[Sq, Sk]`` and ignores ``swa_window``. The HCA split-mask path
  accepts pool-only ``[Sq, P]`` with ``HCA_LOCAL_SEQLEN=Sq`` and keeps
  the local branch on kernel-native SWA.

dtype contract (matches :func:`eager_v4_attention`):

* Q / K / V matmuls run in input dtype on tensor cores; accumulators
  inside the matmul are fp32.
* The online-softmax accumulator (``m_running``, ``l_running``,
  ``acc``) lives in fp32 — this is the *only* fp32 step inside the
  kernel.
* Output is written back in input dtype; saved ``LSE`` is fp32 (BWD
  re-materialises ``P`` from it).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def _v4_attention_fwd_kernel(
    Q,
    K,
    V,
    OUT,
    LSE,
    SINK,
    ADD_MASK,
    # Q strides: [B, H, Sq, D] row-major (contiguous on D)
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    # K strides: [B, K_H, Sk, D] row-major (K_H == 1 for MQA, == H for MHA)
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    # V strides: [B, K_H, Sk, D] row-major
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    # OUT strides: [B, H, Sq, D] row-major
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    # LSE strides: [B, H, Sq] row-major
    stride_lb,
    stride_lh,
    stride_lm,
    # ADD_MASK strides: [Sq, Sk] row-major (broadcasts over B, H)
    stride_ms,
    stride_mn,
    seqlen_q,
    seqlen_k,
    sm_scale,
    HEAD_Q: tl.constexpr,
    HEAD_K: tl.constexpr,  # 1 for MQA, == HEAD_Q for MHA
    SWA_WINDOW: tl.constexpr,  # 0 = off, > 0 = SWA window
    HAS_SINK: tl.constexpr,
    HAS_ADD_MASK: tl.constexpr,
    HCA_LOCAL_SEQLEN: tl.constexpr,  # 0 = generic mask; >0 = [local SWA keys | pool keys]
    USE_CAUSAL: tl.constexpr,  # only meaningful when HAS_ADD_MASK = False
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """V4 attention FWD.

    Grid layout: ``(cdiv(seqlen_q, BLOCK_M), batch * HEAD_Q)``. Each
    program (program-id) computes one ``[BLOCK_M, BLOCK_DMODEL]`` slice
    of ``OUT`` and the matching ``[BLOCK_M]`` slice of ``LSE``.

    Mask precedence (must match :func:`eager_v4_attention`):

    1. ``HAS_ADD_MASK`` — load the ``[Sq, Sk]`` additive bias and add
       to ``qk``. SWA / causal masks are NOT applied in-kernel (the
       caller has embedded all structure in the bias).
    2. ``SWA_WINDOW > 0`` — sliding-window causal:
       keep ``offs_n in [offs_m - SWA_WINDOW + 1, offs_m]``.
    3. ``USE_CAUSAL`` — full causal: keep ``offs_n <= offs_m``.

    In all branches, keys at index ``>= seqlen_k`` are masked to ``-inf``
    so the boundary rows of the K-loop tile do not contaminate the
    softmax denominator.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    bid = pid_bh // HEAD_Q
    qhid = pid_bh % HEAD_Q
    if HEAD_K == HEAD_Q:
        khid = qhid
    else:
        khid = 0  # MQA: single shared K / V head

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Q tile: [BLOCK_M, BLOCK_DMODEL] in q.dtype
    q_ptrs = (
        Q + bid * stride_qb + qhid * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    )
    q_load_mask = offs_m[:, None] < seqlen_q
    q = tl.load(q_ptrs, mask=q_load_mask, other=0.0)

    # Online-softmax running state (fp32). We use the finite sentinel
    # NEG_INF (-1e30) instead of -float("inf") so that the all-masked-
    # tile corner case (m_running == m_tile == NEG_INF, e.g. for early
    # queries under SWA when n_start is entirely outside the window)
    # does NOT produce NaN through ``exp(-inf - -inf) = exp(NaN)``.
    # NEG_INF is far enough below any plausible logit that
    # ``exp(NEG_INF) ≈ 0`` and the algebra is identical to using -inf.
    NEG_INF: tl.constexpr = -1.0e30
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Determine k-loop bounds.
    #
    # Plan-5 P30: for SWA, skip K tiles that are guaranteed outside the
    # sliding window for *every* row in this M block.  The old P25 loop
    # started at zero and relied on the mask below, which means late
    # rows at S=4096 spent most of their time multiplying all-masked
    # tiles.  Rounding down to the BLOCK_N boundary preserves exact
    # masking semantics while cutting the steady-state SWA tile count
    # from O(S / BLOCK_N) to O(window / BLOCK_N).
    n_loop_start = 0
    if HAS_ADD_MASK and HCA_LOCAL_SEQLEN == 0:
        # Caller's additive_mask handles all masking; iterate the full
        # key axis. (We still apply boundary mask for keys past seqlen_k.)
        n_loop_end = seqlen_k
    elif SWA_WINDOW > 0:
        # This block's earliest row is pid_m * BLOCK_M.  Any key before
        # earliest_row - SWA_WINDOW + 1 is invisible for every row in the
        # block, so skip those tiles entirely.
        n_loop_start = pid_m * BLOCK_M - SWA_WINDOW + 1
        if n_loop_start < 0:
            n_loop_start = 0
        n_loop_start = (n_loop_start // BLOCK_N) * BLOCK_N
        # Keys with n > max(offs_m) are causal-masked for every row.
        n_loop_end = (pid_m + 1) * BLOCK_M
        local_end = HCA_LOCAL_SEQLEN if HCA_LOCAL_SEQLEN > 0 else seqlen_k
        if n_loop_end > local_end:
            n_loop_end = local_end
    elif USE_CAUSAL:
        # Full causal: keys with n > max(offs_m) are -inf, so the loop
        # only needs to cover keys up to (pid_m + 1) * BLOCK_M.
        n_loop_end = (pid_m + 1) * BLOCK_M
        if n_loop_end > seqlen_k:
            n_loop_end = seqlen_k
    else:
        n_loop_end = seqlen_k

    # K-loop: local branch. For HCA split-mask this covers only the
    # pruned local SWA prefix; the pool suffix is handled by the second
    # loop below.
    for n_start in range(n_loop_start, n_loop_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)

        # K tile: [BLOCK_N, BLOCK_DMODEL] in k.dtype
        k_ptrs = (
            K + bid * stride_kb + khid * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        )
        k_load_mask = offs_n[:, None] < seqlen_k
        k = tl.load(k_ptrs, mask=k_load_mask, other=0.0)

        # qk = q @ k.T : [BLOCK_M, BLOCK_N] in fp32 (tl.dot accumulator)
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        # Mask: additive_mask OR SWA / causal (mutually exclusive — see
        # the docstring's precedence rule).
        if HAS_ADD_MASK and HCA_LOCAL_SEQLEN == 0:
            mask_ptrs = ADD_MASK + offs_m[:, None] * stride_ms + offs_n[None, :] * stride_mn
            mask_load_mask = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
            add_bias = tl.load(mask_ptrs, mask=mask_load_mask, other=0.0).to(tl.float32)
            qk = qk + add_bias
        else:
            if SWA_WINDOW > 0:
                # offs_n in [offs_m - SWA_WINDOW + 1, offs_m]
                in_window = (offs_n[None, :] >= offs_m[:, None] - SWA_WINDOW + 1) & (
                    offs_n[None, :] <= offs_m[:, None]
                )
                qk = tl.where(in_window, qk, NEG_INF)
            elif USE_CAUSAL:
                qk = tl.where(offs_n[None, :] <= offs_m[:, None], qk, NEG_INF)

        # Boundary: keys past seqlen_k were loaded as 0; mask them to
        # NEG_INF (finite sentinel) so they do not contribute to the
        # softmax denominator and do NOT produce NaN through
        # ``exp(-inf - -inf)``.
        qk = tl.where(offs_n[None, :] < seqlen_k, qk, NEG_INF)

        # Online softmax update
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        # V tile: [BLOCK_N, BLOCK_DMODEL] in v.dtype
        v_ptrs = (
            V + bid * stride_vb + khid * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        )
        v_load_mask = offs_n[:, None] < seqlen_k
        v = tl.load(v_ptrs, mask=v_load_mask, other=0.0)

        # P57 R2: fuse ``acc * alpha`` into the V-dot MFMA-acc input.
        # AMD's Triton lowering threads ``acc`` directly into the MFMA
        # C-tile, eliminating one fp32 register round-trip per K-tile.
        # Same pattern as the cr=0 BWD's ``dq = tl.dot(ds, k, acc=dq)``.
        acc = tl.dot(p.to(v.dtype), v, acc=acc * alpha[:, None])
        m_i = m_new

    # HCA split-mask pool branch. The local branch above prunes the SWA
    # prefix; the pool suffix is short (P = S / 128 for V4-Flash) and
    # uses the caller-provided pool-only visibility mask [Sq, P].
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
            k_load_mask = offs_n[:, None] < seqlen_k
            k = tl.load(k_ptrs, mask=k_load_mask, other=0.0)

            qk = tl.dot(q, tl.trans(k)) * sm_scale

            mask_ptrs = ADD_MASK + offs_m[:, None] * stride_ms + pool_n[None, :] * stride_mn
            mask_load_mask = (offs_m[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
            add_bias = tl.load(mask_ptrs, mask=mask_load_mask, other=0.0).to(tl.float32)
            qk = qk + add_bias
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, NEG_INF)

            m_ij = tl.max(qk, 1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)

            v_ptrs = (
                V
                + bid * stride_vb
                + khid * stride_vh
                + offs_n[:, None] * stride_vn
                + offs_d[None, :] * stride_vd
            )
            v_load_mask = offs_n[:, None] < seqlen_k
            v = tl.load(v_ptrs, mask=v_load_mask, other=0.0)

            # P57 R2: MFMA-acc fusion -- see local-branch comment above.
            acc = tl.dot(p.to(v.dtype), v, acc=acc * alpha[:, None])
            m_i = m_new

    # Sink: virtual key column with value 0. The max-subtract trick
    # uses sink as a candidate row maximum; sink contributes to
    # l_running but NOT to acc (its notional value is zero).
    if HAS_SINK:
        sink_h = tl.load(SINK + qhid).to(tl.float32)
        m_new = tl.maximum(m_i, sink_h)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(sink_h - m_new)
        l_i = l_i * alpha + beta
        acc = acc * alpha[:, None]
        m_i = m_new

    # Final divide + cast back to output dtype.
    out = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)

    out_ptrs = (
        OUT + bid * stride_ob + qhid * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out.to(OUT.dtype.element_ty), mask=offs_m[:, None] < seqlen_q)

    lse_ptrs = LSE + bid * stride_lb + qhid * stride_lh + offs_m * stride_lm
    tl.store(lse_ptrs, lse, mask=offs_m < seqlen_q)


# ---------------------------------------------------------------------------
# Python launcher
# ---------------------------------------------------------------------------


def _launch_v4_attention_fwd(
    q: torch.Tensor,  # [B, H, Sq, D]
    k: torch.Tensor,  # [B, K_H, Sk, D]   K_H ∈ {1, H}
    v: torch.Tensor,  # [B, K_H, Sk, D]
    *,
    sink: Optional[torch.Tensor],  # [H] or None
    swa_window: int,
    additive_mask: Optional[torch.Tensor],  # [Sq, Sk] or None
    scale: float,
    hca_local_seqlen: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the V4 attention forward kernel.

    Returns ``(out, lse)`` where ``out`` matches ``v.dtype`` and
    ``lse`` is fp32. ``lse`` is what the BWD kernel needs to
    re-materialise the softmax without storing the ``[Sq, Sk]`` ``P``
    matrix.

    The launcher does NO autograd bookkeeping — it's a thin wrapper
    around the kernel suitable for the
    :class:`V4AttentionFn` autograd Function and for unit tests.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            "v4_attention forward expects q / k / v of rank 4 "
            f"(got q.dim={q.dim()}, k.dim={k.dim()}, v.dim={v.dim()})"
        )
    B, HQ, Sq, D = q.shape
    Bk, HK, Sk, Dk = k.shape
    Bv, HKv, Skv, Dv = v.shape
    if (Bk, Sk, Dk) != (B, Sk, D) or (Bv, HKv, Skv, Dv) != (Bk, HK, Sk, D):
        raise ValueError(
            "v4_attention shape mismatch: " f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if HK != 1 and HK != HQ:
        raise ValueError(f"v4_attention requires K_H ∈ {{1 (MQA), {HQ} (MHA)}}; got K_H={HK}.")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("v4_attention requires CUDA / HIP tensors.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            "v4_attention requires q.dtype == k.dtype == v.dtype " f"(got {q.dtype} / {k.dtype} / {v.dtype})."
        )

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

    # Mask precedence: additive_mask wins over swa_window. When neither
    # is set, USE_CAUSAL = True (eager / V4 default).
    use_causal = (not has_add_mask) and (swa_window <= 0)
    swa_window_constexpr = (
        int(swa_window) if ((not has_add_mask or hca_local_seqlen) and swa_window > 0) else 0
    )

    out = torch.empty_like(q)
    lse = torch.empty((B, HQ, Sq), device=q.device, dtype=torch.float32)

    # P57 R2: tile sweep on V4-Flash widths (B=1, H=64, S=4096, D=512,
    # SWA=128) finds (BLOCK_M=64, BLOCK_N=16, num_warps=8, num_stages=2)
    # wins by **~17 %** on cr=4 FWD (1.69 -> 1.43 ms) and **~38 %** on
    # cr=0 FWD (0.79 -> 0.49 ms). The intuition:
    #
    # * BLOCK_M=64 halves the program grid (vs the R1 32x32 layout) so
    #   the SWA K-tile prefix is shared across twice as many query rows.
    #   Q is loaded once per program -- bigger M tiles amortise the
    #   Q-load over more K-tile iterations.
    # * BLOCK_N=16 keeps the per-stage K/V tile at 16x512x2 = 16 KiB so
    #   num_stages=2 double-buffering fits the LDS budget:
    #   64 (Q) + 16*2 (K) + 16*2 (V) = 128 KiB << 160 KiB MI355X limit.
    #   The R1 32x32 layout already needed 96 KiB at num_stages=1 and
    #   would have hit 160 KiB at num_stages=2 -- no headroom.
    # * num_stages=2 software-pipelines K/V loads against the
    #   ``tl.dot`` + softmax update, hiding the HBM gather latency
    #   that previously serialised the inner loop.
    #
    # Env-overridable so future shape regressions can fall back without
    # rebuilding. The defaults are the per-shape winner from the R2
    # sweep (`p57/r2_sweep_local.sh`).
    BLOCK_M = int(os.getenv("PRIMUS_V4_ATTN_FWD_BLOCK_M", "64"))
    BLOCK_N = int(os.getenv("PRIMUS_V4_ATTN_FWD_BLOCK_N", "16"))
    NUM_WARPS_FWD = int(os.getenv("PRIMUS_V4_ATTN_FWD_WARPS", "8"))
    NUM_STAGES_FWD = int(os.getenv("PRIMUS_V4_ATTN_FWD_STAGES", "2"))
    BLOCK_DMODEL = D  # head_dim must be a power of 2 for tl.dot

    grid = (triton.cdiv(Sq, BLOCK_M), B * HQ)

    # Sentinel pointers when sink / mask are absent. Triton requires a
    # real tensor — we pass q (any tensor) and gate via the constexpr.
    sink_ptr = sink if has_sink else q
    mask_ptr = additive_mask if has_add_mask else q
    if has_add_mask:
        stride_ms = additive_mask.stride(0)
        stride_mn = additive_mask.stride(1)
    else:
        stride_ms = 0
        stride_mn = 0

    _v4_attention_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        sink_ptr,
        mask_ptr,
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
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
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
        num_warps=NUM_WARPS_FWD,
        num_stages=NUM_STAGES_FWD,
    )
    return out, lse


__all__ = [
    "_v4_attention_fwd_kernel",
    "_launch_v4_attention_fwd",
]
