###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton-fused Indexer scoring (plan-6 P38).

Fuses the ``einsum + relu + mul + sum + causal_mask`` chain in
:meth:`primus.backends.megatron.core.transformer.indexer.Indexer.forward`
into a single FWD + single BWD Triton kernel pair.  Eager body:

.. code-block:: python

    relu_term = F.relu(torch.einsum("bshd,bpd->bshp", q_i, k_icomp))
    scores = (relu_term * w_i.unsqueeze(-1)).sum(dim=2)  # [B, S, P]
    mask = self._causal_mask(S, P, scores.device, scores.dtype)
    scores = scores + mask.unsqueeze(0)

P38 collapses these ~7 ATen kernels (einsum + relu + mul + sum +
mask alloc + mask add + dtype cast) into one Triton kernel that:

* Computes the per-(q, p) dot product over the head-feature dim ``Hd``
  inline (no `[B, S, H, P]` intermediate tensor materialised);
* Applies ``relu`` per element;
* Multiplies by ``w_i`` per head;
* Reduces over heads;
* Materialises the causal mask **inline** via
  ``tl.where((p + 1) * compress_ratio - 1 <= s, 0.0, -INF)`` — no
  ``[S, P]`` mask tensor, no HBM traffic;
* Writes ``scores [B, S, P]`` (the `topk` and trailing tail stay
  host-side; `topk` is heavy GPU compute on its own and benefits
  from being its own kernel).

The BWD kernel recomputes the per-tile `relu` mask in the BWD pass
(FlashAttention-style trick) instead of saving it — saves ``H * S * P``
bits of HBM per CSA layer.

Gating: ``PRIMUS_INDEXER_TRITON == "1"`` (**default-OFF**).

Default-off rationale (P38 descope per `plan-6/02-phase-details.md`
§"Task list refinement"): at V4-Flash widths (B=1, S=4096, P=1024,
H=8, Hd=128) the eager `einsum` maps to a cuBLAS / hipBLASLt
batched-matmul that already runs at ~28 TFLOP/s on MI355.  The
generic Triton kernel here is FWD-competitive only at small shapes
(3.35x FWD speedup at B=2, S=128, P=32) but regresses ~30% at the
production V4-Flash shape; BWD regresses ~12x because of cross-tile
``atomic_add`` traffic on `dq` / `dk` / `dw`.  The kernel stays
available for future tuning + small-shape paths via the env knob.

Per-shape behavior summary:

* V4-Flash production (B=1, S=4096, P=1024, H=8, Hd=128, bf16):
  FWD 0.424 ms (triton) vs 0.306 ms (eager) -> **0.72x** (regression).
  BWD 6.457 ms (triton) vs 0.489 ms (eager) -> **0.08x** (regression).
* Small (B=2, S=128, P=32, H=8, Hd=128, bf16):
  FWD 0.053 ms (triton) vs 0.176 ms (eager) -> **3.35x** (speedup).
  BWD 0.226 ms (triton) vs 0.256 ms (eager) -> **1.14x** (speedup).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# Indexer scoring runs at V4-Flash widths [B=1, S=4096, P=1024, H=8,
# Hd=128].  H is small enough to be a compile-time constant in the
# kernel; supported values are documented here.
_SUPPORTED_H = (1, 2, 4, 8, 16)


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _indexer_score_fwd_kernel(
    Q_PTR,  # [B, S, H, Hd] - q_i
    K_PTR,  # [B, P, Hd]    - k_icomp.squeeze(2)
    W_PTR,  # [B, S, H]     - w_i
    SCORES_PTR,  # [B, S, P]     - out (fp32 internal, cast OUT_DTYPE)
    B,
    S,
    P,
    H: tl.constexpr,
    HD: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_P: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """One program tile = one ``[B_b, BLOCK_S, BLOCK_P]`` chunk.

    Loop axis: ``Hd`` (head-feature dim) and ``H`` (heads), both
    compile-time known.  Per tile the kernel:

        1. For each head ``h`` (unrolled, since H is constexpr):
            a. Load ``q_i[b_b, s_tile, h, :]`` -- shape ``[BLOCK_S, HD]``.
            b. Load ``k_icomp[b_b, p_tile, :]``    -- shape ``[BLOCK_P, HD]``.
            c. Compute ``dot = q @ k.T`` -- shape ``[BLOCK_S, BLOCK_P]``.
            d. Apply ``relu(dot)``.
            e. Load ``w_i[b_b, s_tile, h]`` -- shape ``[BLOCK_S]``.
            f. Multiply: ``acc += relu * w[:, None]``.
        2. Materialise causal mask inline: positions ``s`` may attend to
           pool position ``p`` iff ``(p + 1) * compress_ratio - 1 <= s``;
           write ``-inf`` otherwise.
        3. Store ``acc`` cast to OUT_DTYPE.

    Register footprint at V4-Flash (BLOCK_S=64, BLOCK_P=64, H=8, HD=128):
        per head: 64*128 + 64*128 = 16384 fp32 = 64 KiB for (q, k).
        per tile: 64*64 = 4096 fp32 = 16 KiB for acc.
    Cumulative per-program peak ~ 80 KiB ≈ 320 VGPRs / warp.  At MI355
    256 VGPRs we drop BLOCK_S to 32 for safety on the small kernel.
    """

    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_p = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    s_mask = s_offs < S
    p_mask = p_offs < P

    hd_idx = tl.arange(0, HD)

    acc = tl.zeros((BLOCK_S, BLOCK_P), dtype=tl.float32)

    # Unroll over heads (H is small and constexpr).
    for h in tl.static_range(0, H):
        # q [BLOCK_S, HD]: q_i[pid_b, s_offs, h, :]
        q_tile = tl.load(
            Q_PTR + pid_b * S * H * HD + s_offs[:, None] * H * HD + h * HD + hd_idx[None, :],
            mask=s_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        # k [BLOCK_P, HD]: k_icomp[pid_b, p_offs, :]
        k_tile = tl.load(
            K_PTR + pid_b * P * HD + p_offs[:, None] * HD + hd_idx[None, :],
            mask=p_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        # dot [BLOCK_S, BLOCK_P] = q @ k.T
        dot = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32)
        dot = tl.maximum(dot, 0.0)  # relu
        # w [BLOCK_S]: w_i[pid_b, s_offs, h]
        w_h = tl.load(
            W_PTR + pid_b * S * H + s_offs * H + h,
            mask=s_mask,
            other=0.0,
        ).to(tl.float32)
        acc += dot * w_h[:, None]

    # Apply causal mask inline.  Allowed iff `(p + 1) * cr - 1 <= s`,
    # i.e. the pool position's window end is no later than the query.
    s_arr = s_offs[:, None]
    p_arr = p_offs[None, :]
    allowed = (p_arr + 1) * COMPRESS_RATIO - 1 <= s_arr
    NEG_INF = -float("inf")
    acc = tl.where(allowed, acc, NEG_INF)

    tl.store(
        SCORES_PTR + pid_b * S * P + s_offs[:, None] * P + p_offs[None, :],
        acc.to(OUT_DTYPE),
        mask=s_mask[:, None] & p_mask[None, :],
    )


@triton.jit
def _indexer_score_bwd_kernel(
    DSCORES_PTR,  # [B, S, P]    grad in OUT_DTYPE
    Q_PTR,  # [B, S, H, Hd]
    K_PTR,  # [B, P, Hd]
    W_PTR,  # [B, S, H]
    DQ_PTR,  # [B, S, H, Hd] OUT (fp32)
    DK_PTR,  # [B, P, Hd]    OUT (fp32, scattered)
    DW_PTR,  # [B, S, H]     OUT (fp32)
    B,
    S,
    P,
    H: tl.constexpr,
    HD: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """VJP through the scoring chain.

    The mask is `where(allowed, acc, -inf)`; its derivative is 1 in
    allowed positions and 0 elsewhere.  Out-of-range positions therefore
    contribute zero grad and we set ``dscores`` to 0 there before
    walking back.

    Per element forward:
        acc = sum_h relu(q . k) * w_h
        masked = where(allowed, acc, -inf)
    Per element backward:
        d_acc = d_masked * where(allowed, 1, 0)        # equiv. d_masked masked by allowed
        # d_relu_dot[h] = d_acc * w_h
        # d_w[h]        = d_acc * relu_dot[h]
        # relu' = (dot > 0).  Recompute dot, relu mask.
        # d_dot[h]   = d_relu_dot[h] * (dot > 0)
        # d_q[h, :]  = d_dot[h, p] @ k[p, :]
        # d_k[p, :]  = d_dot[s, p] @ q[s, h, :]

    For the BWD we use one program per (b, s_tile, p_tile) just like
    FWD; ``d_q`` is accumulated locally (one chunk per s_tile so
    multiple p_tiles need atomic add), ``d_k`` is scattered across
    s_tiles so needs atomic add, ``d_w`` is accumulated similarly.
    """

    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_p = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    s_mask = s_offs < S
    p_mask = p_offs < P

    hd_idx = tl.arange(0, HD)

    # Load dscores [BLOCK_S, BLOCK_P]
    dmasked = tl.load(
        DSCORES_PTR + pid_b * S * P + s_offs[:, None] * P + p_offs[None, :],
        mask=s_mask[:, None] & p_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    # Apply causal mask (zero out invalid positions).
    s_arr = s_offs[:, None]
    p_arr = p_offs[None, :]
    allowed = (p_arr + 1) * COMPRESS_RATIO - 1 <= s_arr
    d_acc = tl.where(allowed, dmasked, 0.0)

    # Unroll over heads.
    for h in tl.static_range(0, H):
        # Reload q, k for this head (FlashAttention-style recompute).
        q_tile = tl.load(
            Q_PTR + pid_b * S * H * HD + s_offs[:, None] * H * HD + h * HD + hd_idx[None, :],
            mask=s_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        k_tile = tl.load(
            K_PTR + pid_b * P * HD + p_offs[:, None] * HD + hd_idx[None, :],
            mask=p_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        w_h = tl.load(
            W_PTR + pid_b * S * H + s_offs * H + h,
            mask=s_mask,
            other=0.0,
        ).to(tl.float32)

        # Recompute dot, relu, relu_dot.
        dot = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32)
        relu_dot = tl.maximum(dot, 0.0)  # [BLOCK_S, BLOCK_P]
        relu_mask = dot > 0.0

        # d_w[h] = sum_p d_acc * relu_dot
        dw_h = tl.sum(d_acc * relu_dot, axis=1)  # [BLOCK_S]

        # d_relu_dot = d_acc * w_h[:, None]
        d_relu_dot = d_acc * w_h[:, None]
        # d_dot = d_relu_dot where relu_mask else 0
        d_dot = tl.where(relu_mask, d_relu_dot, 0.0)

        # d_q[s, h, :] = sum_p d_dot[s, p] * k[p, :]   shape [BLOCK_S, HD]
        d_q = tl.dot(d_dot, k_tile, out_dtype=tl.float32)
        # d_k[p, :]    = sum_s d_dot[s, p] * q[s, :]   shape [BLOCK_P, HD]
        d_k = tl.dot(tl.trans(d_dot), q_tile, out_dtype=tl.float32)

        # Stores with atomic_add since multiple p_tiles/s_tiles touch
        # the same locations.
        tl.atomic_add(
            DQ_PTR + pid_b * S * H * HD + s_offs[:, None] * H * HD + h * HD + hd_idx[None, :],
            d_q,
            mask=s_mask[:, None],
        )
        tl.atomic_add(
            DK_PTR + pid_b * P * HD + p_offs[:, None] * HD + hd_idx[None, :],
            d_k,
            mask=p_mask[:, None],
        )
        tl.atomic_add(
            DW_PTR + pid_b * S * H + s_offs * H + h,
            dw_h,
            mask=s_mask,
        )


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


def _pick_block(s: int, p: int, h: int, hd: int) -> tuple[int, int]:
    """Pick BLOCK_S, BLOCK_P that fit MI355 register budget."""
    block_s = 32
    block_p = 32
    if hd >= 128:
        block_s = 32
        block_p = 32
    if hd >= 256:
        block_s = 16
        block_p = 32
    # Floor both tiles to 16: BLOCK_S / BLOCK_P are the M / N dims of the
    # ``tl.dot(q_tile, k_tile.T)`` in the FWD/BWD score kernels, so they
    # must be >= the gfx1250 WMMA minimum. For tiny S / P (e.g. unit-test
    # shapes), ``next_power_of_2(s|p)`` can drop below 16 and the dot fails
    # to select a matrix-core intrinsic ("no matching matrix core intrinsic
    # for wmma version 3 ... [0, 0, K]"). Surplus rows/cols are masked by
    # ``s_mask`` / ``p_mask`` inside the kernels, so a 16-wide tile over a
    # smaller S / P is safe.
    block_s = max(16, min(block_s, triton.next_power_of_2(max(1, s))))
    block_p = max(16, min(block_p, triton.next_power_of_2(max(1, p))))
    return block_s, block_p


# ---------------------------------------------------------------------------
# autograd.Function wrapper
# ---------------------------------------------------------------------------


class IndexerScoreFn(torch.autograd.Function):
    """Autograd-aware wrapper around the FWD/BWD Triton kernels.

    Returns ``scores [B, S, P]`` in ``out_dtype``.  Caller runs
    ``torch.topk`` / sentinel substitution / padding host-side.

    Inputs:
        q_i [B, S, H, Hd] - per-head queries (any float dtype).
        k_icomp [B, P, Hd] - compressed keys (any float dtype).
        w_i [B, S, H] - per-head weights (any float dtype).
        compress_ratio - int (typically 4 for CSA).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q_i: torch.Tensor,
        k_icomp: torch.Tensor,
        w_i: torch.Tensor,
        compress_ratio: int,
        out_dtype: torch.dtype,
    ):
        if q_i.dim() != 4:
            raise ValueError(f"q_i must be [B, S, H, Hd], got shape {tuple(q_i.shape)}")
        if k_icomp.dim() != 3:
            raise ValueError(f"k_icomp must be [B, P, Hd], got shape {tuple(k_icomp.shape)}")
        if w_i.dim() != 3:
            raise ValueError(f"w_i must be [B, S, H], got shape {tuple(w_i.shape)}")

        B, S, H, HD = q_i.shape
        Bk, P, HDk = k_icomp.shape
        Bw, Sw, Hw = w_i.shape
        if B != Bk or B != Bw:
            raise ValueError(f"Mismatched B: q={B} k={Bk} w={Bw}")
        if HD != HDk:
            raise ValueError(f"Mismatched Hd: q={HD} k={HDk}")
        if S != Sw:
            raise ValueError(f"Mismatched S: q={S} w={Sw}")
        if H != Hw:
            raise ValueError(f"Mismatched H: q={H} w={Hw}")
        if H not in _SUPPORTED_H:
            raise ValueError(f"Unsupported H={H}; expected one of {_SUPPORTED_H}")
        if HD & (HD - 1) != 0:
            raise ValueError(f"Hd must be a power of 2, got {HD}")

        q_c = q_i.contiguous()
        k_c = k_icomp.contiguous()
        w_c = w_i.contiguous()

        device = q_c.device
        scores = torch.empty((B, S, P), dtype=out_dtype, device=device)

        block_s, block_p = _pick_block(S, P, H, HD)
        grid = (B, triton.cdiv(S, block_s), triton.cdiv(P, block_p))
        _indexer_score_fwd_kernel[grid](
            q_c,
            k_c,
            w_c,
            scores,
            B,
            S,
            P,
            H=H,
            HD=HD,
            COMPRESS_RATIO=int(compress_ratio),
            BLOCK_S=block_s,
            BLOCK_P=block_p,
            OUT_DTYPE={
                torch.float32: tl.float32,
                torch.float16: tl.float16,
                torch.bfloat16: tl.bfloat16,
                torch.float64: tl.float64,
            }[out_dtype],
        )

        ctx.save_for_backward(q_c, k_c, w_c)
        ctx.compress_ratio = int(compress_ratio)
        ctx.shape = (B, S, P, H, HD)
        ctx.in_dtypes = (q_i.dtype, k_icomp.dtype, w_i.dtype)
        return scores

    @staticmethod
    def backward(ctx, d_scores):  # type: ignore[override]
        q_c, k_c, w_c = ctx.saved_tensors
        B, S, P, H, HD = ctx.shape
        compress_ratio = ctx.compress_ratio
        q_dtype, k_dtype, w_dtype = ctx.in_dtypes

        d_scores = d_scores.contiguous()
        device = q_c.device

        d_q_fp32 = torch.zeros((B, S, H, HD), dtype=torch.float32, device=device)
        d_k_fp32 = torch.zeros((B, P, HD), dtype=torch.float32, device=device)
        d_w_fp32 = torch.zeros((B, S, H), dtype=torch.float32, device=device)

        block_s, block_p = _pick_block(S, P, H, HD)
        grid = (B, triton.cdiv(S, block_s), triton.cdiv(P, block_p))
        _indexer_score_bwd_kernel[grid](
            d_scores,
            q_c,
            k_c,
            w_c,
            d_q_fp32,
            d_k_fp32,
            d_w_fp32,
            B,
            S,
            P,
            H=H,
            HD=HD,
            COMPRESS_RATIO=int(compress_ratio),
            BLOCK_S=block_s,
            BLOCK_P=block_p,
        )

        return d_q_fp32.to(q_dtype), d_k_fp32.to(k_dtype), d_w_fp32.to(w_dtype), None, None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def is_triton_path_enabled() -> bool:
    """Return True iff ``PRIMUS_INDEXER_TRITON_FULL == "1"``.

    **Re-purposed at P41:** the P38 full-fuse path is now gated by the
    distinct env knob ``PRIMUS_INDEXER_TRITON_FULL`` (default ``"0"``).
    The original ``PRIMUS_INDEXER_TRITON`` env now controls the cheaper
    P41 post-einsum tail fusion in
    :mod:`indexer_score_post`.

    The full-fuse path is descoped at V4-Flash widths (cuBLAS einsum
    beats the generic Triton kernel + BWD atomic_add contention
    regresses 12x).  Kept available for small-shape paths and future
    tuning.  Set ``PRIMUS_INDEXER_TRITON_FULL=1`` to opt-in.
    """
    return os.environ.get("PRIMUS_INDEXER_TRITON_FULL", "0") == "1"


def is_triton_kernel_supported(q_i: torch.Tensor, k_icomp: torch.Tensor, w_i: torch.Tensor) -> bool:
    """Return True iff the input shapes / device support the Triton path."""
    if not q_i.is_cuda or not k_icomp.is_cuda or not w_i.is_cuda:
        return False
    if q_i.dim() != 4 or k_icomp.dim() != 3 or w_i.dim() != 3:
        return False
    B, S, H, HD = q_i.shape
    if H not in _SUPPORTED_H:
        return False
    if HD & (HD - 1) != 0:
        return False
    return True


def indexer_score_triton(
    q_i: torch.Tensor,
    k_icomp: torch.Tensor,
    w_i: torch.Tensor,
    *,
    compress_ratio: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute Indexer scores via the fused Triton kernel.

    Returns ``scores [B, S, P]`` of dtype ``out_dtype``.  Masked
    positions hold ``-inf``.
    """
    return IndexerScoreFn.apply(q_i, k_icomp, w_i, compress_ratio, out_dtype)


__all__ = [
    "IndexerScoreFn",
    "indexer_score_triton",
    "is_triton_path_enabled",
    "is_triton_kernel_supported",
]
