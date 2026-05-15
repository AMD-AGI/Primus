###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton-fused Indexer scoring -- post-einsum tail (plan-6 P41).

Companion to :mod:`indexer_score` (P38).  P38 fused the entire
``einsum + relu + mul + sum + causal_mask`` chain into one kernel and
lost to cuBLAS / hipBLASLt on the matmul half (~28 TFLOP/s eager vs
~20 TFLOP/s Triton at V4-Flash widths).  P41 keeps the einsum eager
and fuses **only** the post-matmul tail
(``relu -> mul(w_i) -> sum(H) -> + causal_mask``).  The matmul is
compute-bound and stays on cuBLAS peak; the tail is bandwidth-bound
and Triton wins because every elementwise op costs a full HBM
round-trip.

Eager body (with einsum kept):

.. code-block:: python

    dot = torch.einsum("bshd,bpd->bshp", q_i, k_icomp)   # stays eager
    relu_term = F.relu(dot)
    scores = (relu_term * w_i.unsqueeze(-1)).sum(dim=2)  # [B, S, P]
    mask = self._causal_mask(S, P, scores.device, scores.dtype)
    scores = scores + mask.unsqueeze(0)

P41 collapses the tail (``relu + mul + sum + mask_alloc + mask_add +
dtype cast``, ~5 ATen launches) into one Triton kernel that:

* Reads ``dot [B, S, H, P]`` once;
* Applies ``relu`` per element;
* Multiplies by ``w_i[B, S, H, 1]`` (broadcast over P);
* Reduces over heads ``H``;
* Materialises the causal mask **inline** via
  ``tl.where((p + 1) * compress_ratio - 1 <= s, acc, -inf)``;
* Writes ``scores [B, S, P]`` cast to ``OUT_DTYPE``.

BWD takes ``d_scores [B, S, P]`` + saved ``dot [B, S, H, P]`` +
``w_i [B, S, H]``, emits:

* ``d_dot[b, s, h, p] = d_scores[b, s, p] * w_i[b, s, h]`` where
  ``dot[b, s, h, p] > 0`` else 0  (one HBM write per element);
* ``d_w_i[b, s, h] = sum_p(d_scores[b, s, p] * relu(dot[b, s, h, p]))``
  (one reduction per (b, s, h) — no cross-block atomic_add).

The BWD has no atomic_add traffic (the P38 BWD's killer).  Both FWD
and BWD are bandwidth-bound and should win at V4-Flash widths.

Gating: ``PRIMUS_INDEXER_TRITON == "1"`` (the re-purposed env knob;
default initially ``"0"`` then ``"1"`` after the proxy A/B confirms).
The legacy P38 full-fuse path lives behind ``PRIMUS_INDEXER_TRITON_FULL``
(see :mod:`indexer_score`).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# H is small and known per call site -- V4-Flash uses H=8.  Same
# supported set as P38.
_SUPPORTED_H = (1, 2, 4, 8, 16)


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _indexer_score_post_fwd_kernel(
    DOT_PTR,  # [B, S, H, P] - eager einsum output, pre-relu
    W_PTR,  # [B, S, H]    - per-head weights
    SCORES_PTR,  # [B, S, P]    - out
    B,
    S,
    P,
    H: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_P: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """One program tile = ``[B_b, BLOCK_S, BLOCK_P]``.

    Per-tile workflow:

        1. Initialise ``acc [BLOCK_S, BLOCK_P]`` fp32 to 0.
        2. For each head ``h`` (constexpr unroll):
           a. Load ``dot[b, s_tile, h, p_tile]``  -> [BLOCK_S, BLOCK_P].
           b. Apply ``relu``.
           c. Load ``w[b, s_tile, h]``            -> [BLOCK_S].
           d. ``acc += relu * w[:, None]``.
        3. Apply causal mask inline.
        4. Store ``acc`` cast to ``OUT_DTYPE``.
    """

    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_p = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    s_mask = s_offs < S
    p_mask = p_offs < P

    acc = tl.zeros((BLOCK_S, BLOCK_P), dtype=tl.float32)

    for h in tl.static_range(0, H):
        # dot[pid_b, s_offs, h, p_offs] -> [BLOCK_S, BLOCK_P]
        dot_tile = tl.load(
            DOT_PTR + pid_b * S * H * P + s_offs[:, None] * H * P + h * P + p_offs[None, :],
            mask=s_mask[:, None] & p_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        relu_tile = tl.maximum(dot_tile, 0.0)

        # w[pid_b, s_offs, h] -> [BLOCK_S]
        w_h = tl.load(
            W_PTR + pid_b * S * H + s_offs * H + h,
            mask=s_mask,
            other=0.0,
        ).to(tl.float32)

        acc += relu_tile * w_h[:, None]

    # Causal mask inline.
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
def _indexer_score_post_bwd_dw_kernel(
    DSCORES_PTR,  # [B, S, P]    - grad in (any float dtype)
    DOT_PTR,  # [B, S, H, P] - saved pre-relu activations
    DW_PTR,  # [B, S, H]    - OUT  (fp32)
    B,
    S,
    P,
    H: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_P_INNER: tl.constexpr,
):
    """Per ``(b, s_tile, h)`` program: ``d_w[h] = sum_p(d_acc * relu(dot[h]))``.

    Loops over P internally in chunks of ``BLOCK_P_INNER``; ``d_w``
    is fully local to each program (no cross-block ``atomic_add``).
    """

    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    h = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S
    s_arr = s_offs[:, None]

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    n_p_chunks = tl.cdiv(P, BLOCK_P_INNER)
    for chunk in range(n_p_chunks):
        p_offs = chunk * BLOCK_P_INNER + tl.arange(0, BLOCK_P_INNER)
        p_mask = p_offs < P
        p_arr = p_offs[None, :]

        dmasked = tl.load(
            DSCORES_PTR + pid_b * S * P + s_offs[:, None] * P + p_offs[None, :],
            mask=s_mask[:, None] & p_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        allowed = (p_arr + 1) * COMPRESS_RATIO - 1 <= s_arr
        d_acc = tl.where(allowed, dmasked, 0.0)

        dot_tile = tl.load(
            DOT_PTR + pid_b * S * H * P + s_offs[:, None] * H * P + h * P + p_offs[None, :],
            mask=s_mask[:, None] & p_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        relu_tile = tl.maximum(dot_tile, 0.0)

        acc += tl.sum(d_acc * relu_tile, axis=1)

    tl.store(
        DW_PTR + pid_b * S * H + s_offs * H + h,
        acc,
        mask=s_mask,
    )


@triton.jit
def _indexer_score_post_bwd_ddot_kernel(
    DSCORES_PTR,  # [B, S, P]    - grad in (any float dtype)
    DOT_PTR,  # [B, S, H, P] - saved pre-relu activations
    W_PTR,  # [B, S, H]    - per-head weights
    DDOT_PTR,  # [B, S, H, P] - OUT  (fp32)
    B,
    S,
    P,
    H: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """VJP through the post-einsum tail — ``d_dot`` only.

    Per-element forward (with mask):
        d_acc = where(allowed, d_scores, 0)
    Per-element backward:
        d_dot[h] = where(dot[h] > 0, d_acc * w_h, 0)

    ``d_dot`` is one HBM store per element (the grid covers each
    output element exactly once); ``d_w`` is computed via eager
    PyTorch in the autograd wrapper as a single ATen reduce (cheap
    + avoids the cross-block ``atomic_add`` traffic that hurt the
    P38 BWD).
    """

    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_p = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    s_mask = s_offs < S
    p_mask = p_offs < P

    dmasked = tl.load(
        DSCORES_PTR + pid_b * S * P + s_offs[:, None] * P + p_offs[None, :],
        mask=s_mask[:, None] & p_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    s_arr = s_offs[:, None]
    p_arr = p_offs[None, :]
    allowed = (p_arr + 1) * COMPRESS_RATIO - 1 <= s_arr
    d_acc = tl.where(allowed, dmasked, 0.0)

    for h in tl.static_range(0, H):
        dot_tile = tl.load(
            DOT_PTR + pid_b * S * H * P + s_offs[:, None] * H * P + h * P + p_offs[None, :],
            mask=s_mask[:, None] & p_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        relu_mask = dot_tile > 0.0

        w_h = tl.load(
            W_PTR + pid_b * S * H + s_offs * H + h,
            mask=s_mask,
            other=0.0,
        ).to(tl.float32)

        d_relu = d_acc * w_h[:, None]
        d_dot = tl.where(relu_mask, d_relu, 0.0)
        tl.store(
            DDOT_PTR + pid_b * S * H * P + s_offs[:, None] * H * P + h * P + p_offs[None, :],
            d_dot,
            mask=s_mask[:, None] & p_mask[None, :],
        )


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


def _pick_block(s: int, p: int, h: int) -> tuple[int, int]:
    """Pick BLOCK_S, BLOCK_P that fit MI355 register budget.

    Tail kernel is much cheaper than the P38 full-fuse one (no dot
    product, only elementwise + reduce) so we can use larger blocks.
    """
    block_s = 64
    block_p = 64
    return min(block_s, triton.next_power_of_2(max(1, s))), min(block_p, triton.next_power_of_2(max(1, p)))


# ---------------------------------------------------------------------------
# autograd.Function wrapper
# ---------------------------------------------------------------------------


class IndexerScorePostFn(torch.autograd.Function):
    """Autograd-aware wrapper around the post-einsum tail kernels.

    Inputs:
        dot [B, S, H, P] - eager einsum output (pre-relu); any float dtype.
        w_i [B, S, H]    - per-head weights; any float dtype.
        compress_ratio   - int (typically 4 for CSA).
        out_dtype        - dtype of the returned ``scores``.

    Returns:
        scores [B, S, P] of dtype ``out_dtype``.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        dot: torch.Tensor,
        w_i: torch.Tensor,
        compress_ratio: int,
        out_dtype: torch.dtype,
    ):
        if dot.dim() != 4:
            raise ValueError(f"dot must be [B, S, H, P], got shape {tuple(dot.shape)}")
        if w_i.dim() != 3:
            raise ValueError(f"w_i must be [B, S, H], got shape {tuple(w_i.shape)}")

        B, S, H, P = dot.shape
        Bw, Sw, Hw = w_i.shape
        if B != Bw:
            raise ValueError(f"Mismatched B: dot={B} w={Bw}")
        if S != Sw:
            raise ValueError(f"Mismatched S: dot={S} w={Sw}")
        if H != Hw:
            raise ValueError(f"Mismatched H: dot={H} w={Hw}")
        if H not in _SUPPORTED_H:
            raise ValueError(f"Unsupported H={H}; expected one of {_SUPPORTED_H}")

        dot_c = dot.contiguous()
        w_c = w_i.contiguous()
        device = dot_c.device
        scores = torch.empty((B, S, P), dtype=out_dtype, device=device)

        block_s, block_p = _pick_block(S, P, H)
        grid = (B, triton.cdiv(S, block_s), triton.cdiv(P, block_p))
        _indexer_score_post_fwd_kernel[grid](
            dot_c,
            w_c,
            scores,
            B,
            S,
            P,
            H=H,
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

        ctx.save_for_backward(dot_c, w_c)
        ctx.compress_ratio = int(compress_ratio)
        ctx.shape = (B, S, H, P)
        ctx.in_dtypes = (dot.dtype, w_i.dtype)
        return scores

    @staticmethod
    def backward(ctx, d_scores):  # type: ignore[override]
        dot_c, w_c = ctx.saved_tensors
        B, S, H, P = ctx.shape
        compress_ratio = ctx.compress_ratio
        dot_dtype, w_dtype = ctx.in_dtypes

        d_scores = d_scores.contiguous()
        device = dot_c.device

        d_dot_fp32 = torch.empty((B, S, H, P), dtype=torch.float32, device=device)
        d_w_fp32 = torch.empty((B, S, H), dtype=torch.float32, device=device)

        block_s, block_p = _pick_block(S, P, H)
        grid_ddot = (B, triton.cdiv(S, block_s), triton.cdiv(P, block_p))
        _indexer_score_post_bwd_ddot_kernel[grid_ddot](
            d_scores,
            dot_c,
            w_c,
            d_dot_fp32,
            B,
            S,
            P,
            H=H,
            COMPRESS_RATIO=int(compress_ratio),
            BLOCK_S=block_s,
            BLOCK_P=block_p,
        )

        # d_w[b, s, h] = sum_p (d_acc * relu(dot[h])) — one program per
        # (b, s_tile, h), loops over P internally so the reduce is fully
        # local (no cross-block atomic_add).
        block_p_inner = min(128, triton.next_power_of_2(max(1, P)))
        grid_dw = (B, triton.cdiv(S, block_s), H)
        _indexer_score_post_bwd_dw_kernel[grid_dw](
            d_scores,
            dot_c,
            d_w_fp32,
            B,
            S,
            P,
            H=H,
            COMPRESS_RATIO=int(compress_ratio),
            BLOCK_S=block_s,
            BLOCK_P_INNER=block_p_inner,
        )

        return d_dot_fp32.to(dot_dtype), d_w_fp32.to(w_dtype), None, None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def is_triton_path_enabled() -> bool:
    """Return True iff ``PRIMUS_INDEXER_TRITON == "1"``.

    The env knob is **re-purposed** at P41 to mean "post-einsum tail
    fusion" (cheap, bandwidth-bound).  Legacy P38 full-fuse path
    lives behind ``PRIMUS_INDEXER_TRITON_FULL``.
    """
    return os.environ.get("PRIMUS_INDEXER_TRITON", "0") == "1"


def is_triton_kernel_supported(dot: torch.Tensor, w_i: torch.Tensor) -> bool:
    """Return True iff the input shapes / device support the Triton path."""
    if not dot.is_cuda or not w_i.is_cuda:
        return False
    if dot.dim() != 4 or w_i.dim() != 3:
        return False
    _, _, H, _ = dot.shape
    if H not in _SUPPORTED_H:
        return False
    return True


def indexer_score_post_triton(
    dot: torch.Tensor,
    w_i: torch.Tensor,
    *,
    compress_ratio: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute Indexer scores from the eager einsum output via Triton.

    Returns ``scores [B, S, P]`` of dtype ``out_dtype``.  Masked
    positions hold ``-inf``.
    """
    return IndexerScorePostFn.apply(dot, w_i, compress_ratio, out_dtype)


__all__ = [
    "IndexerScorePostFn",
    "indexer_score_post_triton",
    "is_triton_path_enabled",
    "is_triton_kernel_supported",
]
