###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton-fused KL target for the indexer distillation loss.

Fuses the body of
:func:`primus.backends.megatron.core.transformer.indexer_distill_loss._target_distribution`
-- gather + einsum + scale + two masked_fill + softmax + mask-mul + head-sum --
into a single forward kernel. No backward is needed: the target is built under
``torch.no_grad``.

``noncompressed_lse`` makes the per-head softmax match the attention the target
imitates. A CSA layer takes one softmax jointly over the sliding window, the
sparse compressed entries and the sink, so the share of a head's attention that
lands on the compressed branch is ``exp(compressed_lse - full_lse)`` -- below 1
and different per head. Normalising each head over the compressed entries alone
throws that away and over-weights the heads that are almost entirely local;
passing the log mass of the non-compressed part puts it back in the denominator.

Why this one is worth fusing when the P38 indexer-score fusion was not: the cost
here is not the GEMM, it is the gather in front of it. The eager body does

.. code-block:: python

    gathered = kv[batch_idx, idx.clamp_min(0)]          # [B, chunk, K, D]
    logits = torch.einsum("bhsd,bskd->bhsk", q, gathered)

and at V4-Flash widths (B=1, S=4096, K=512, D=512) ``gathered`` is 2.1 GB per
CSA layer per microbatch -- written to HBM and immediately read back by the
GEMM. A profiler diff of the proxy with the loss on vs off attributes 9.16 ms
of the 24.3 ms delta (2 iterations, 3 CSA layers) to that one
``index_elementwise_kernel``, versus 1.4 ms for the GEMM it feeds.

The pool it gathers from is only ``[B, P, D]`` = 1 MB, so indexing it inside the
kernel turns 4.2 GB of HBM traffic into repeated reads that land in cache. The
per-head softmax and the head sum then happen in registers, which also removes
the ~7.8 ms of masked_fill / mul / bitwise_not / softmax elementwise kernels the
eager chain launches around the GEMM.

Measured at V4-Flash CSA widths (B=1, H=64, S=4096, D=512, P=1024, K=512, bf16),
137.4 GFLOP per call:

    eager (chunked)   5.35 ms
    fused             0.354 ms   -> 15.1x, 385 TFLOP/s

End to end on the 8-layer EP=8 proxy (3 CSA layers, GBS=8), the distillation
loss costs +22.3 ms/iter eager and +7.6 ms/iter with this kernel plus the fused
KL tail. The kernel is also *more* accurate than the path it replaces: the eager
einsum lands in bf16 before being promoted, while ``tl.dot`` accumulates in
fp32 (max error against an fp64 reference 1.8e-07 vs 2.5e-03).

Gating: ``PRIMUS_V4_DISTILL_TARGET_TRITON`` (default ON; set to 0 for the eager
body). :func:`can_use_triton_target` refuses shapes the kernel does not cover
(non-power-of-two ``K``, ``D``/``H`` not divisible by the block sizes), so the
caller always has a working path.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

__all__ = ["can_use_triton_target", "target_distribution_triton"]

_ENABLE_ENV = "PRIMUS_V4_DISTILL_TARGET_TRITON"
_BLOCK_H_ENV = "PRIMUS_V4_DISTILL_TARGET_BLOCK_H"
_BLOCK_D_ENV = "PRIMUS_V4_DISTILL_TARGET_BLOCK_D"
_WARPS_ENV = "PRIMUS_V4_DISTILL_TARGET_WARPS"

# BLOCK_H defaults to "every head this layer has, up to 64". The pool rows a
# query selects are shared by all its heads, so a bigger head block loads them
# fewer times: at V4-Flash widths (H=64) going from 16 to 64 is 1.113 -> 0.369 ms
# (123 -> 373 TFLOP/s). BLOCK_D is the feature-axis tile; smaller is better here
# because it keeps the [K, BLOCK_D] pool tile in registers (32 beats 64/128).
_MAX_BLOCK_H = 64
_DEFAULT_BLOCK_D = 32
_DEFAULT_WARPS = 8


@triton.jit
def _distill_target_fwd_kernel(
    Q_PTR,  # [B, H, S, D]   post-RoPE queries (detached)
    POOL_PTR,  # [B, P, D]      compressed KV pool (detached)
    IDX_PTR,  # [B, S, K]      selected pool rows, -1 = invalid
    OUT_PTR,  # [B, S, K]      head-summed probabilities (fp32)
    NCLSE_PTR,  # [B, H, S]      log mass outside the compressed branch (fp32)
    q_sb,
    q_sh,
    q_ss,
    q_sd,
    p_sb,
    p_sp,
    p_sd,
    i_sb,
    i_ss,
    i_sk,
    o_sb,
    o_ss,
    o_sk,
    n_sb,
    n_sh,
    n_ss,
    scale,
    eps,
    H: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NORMALIZE: tl.constexpr,
    HAS_NCLSE: tl.constexpr,
):
    """One program = one query row ``(b, s)``, all heads, all K entries.

    Per head block: accumulate ``q @ pool[idx].T`` over the feature axis,
    softmax across K in registers, and add the result into the running
    head sum. ``pool`` is read through ``idx`` directly, so the
    ``[B, S, K, D]`` gather never exists.

    With ``NORMALIZE`` the row is turned into a distribution before it is
    stored, which saves a full read-modify-write of the ``[B, S, K]`` fp32
    target. It has to stay off when the head sum is still incomplete, i.e.
    when the heads are sharded and an all-reduce has to happen first.

    With ``HAS_NCLSE`` the softmax denominator also carries the window and
    sink mass, so each head contributes in proportion to how much of its
    attention actually reaches the compressed entries.
    """
    b = tl.program_id(0)
    s = tl.program_id(1)

    offs_k = tl.arange(0, K)
    idx = tl.load(IDX_PTR + b * i_sb + s * i_ss + offs_k * i_sk)
    valid = idx >= 0
    # Clamp the -1 sentinels to a legal row; the loads are masked out again by
    # `valid` before they can affect the softmax.
    idx_safe = tl.where(valid, idx, 0)

    acc = tl.zeros([K], dtype=tl.float32)

    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        logits = tl.zeros([BLOCK_H, K], dtype=tl.float32)

        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            q = tl.load(
                Q_PTR + b * q_sb + offs_h[:, None] * q_sh + s * q_ss + offs_d[None, :] * q_sd
            )  # [BLOCK_H, BLOCK_D]
            kv = tl.load(
                POOL_PTR + b * p_sb + idx_safe[:, None] * p_sp + offs_d[None, :] * p_sd
            )  # [K, BLOCK_D]
            logits += tl.dot(q, tl.trans(kv))

        logits = logits * scale
        logits = tl.where(valid[None, :], logits, float("-inf"))

        # A row with no legal entry would have max == -inf and produce NaN;
        # neutralise it here and let the zeroed numerator carry the 0 result.
        m = tl.max(logits, axis=1)
        m = tl.where(m == float("-inf"), 0.0, m)
        e = tl.exp(logits - m[:, None])
        e = tl.where(valid[None, :], e, 0.0)
        denom = tl.sum(e, axis=1)

        if HAS_NCLSE:
            nclse = tl.load(NCLSE_PTR + b * n_sb + offs_h * n_sh + s * n_ss)
            # A head whose attention is almost entirely local makes
            # nclse - m large; the clamp keeps the exponential finite so the
            # ratio stays a well-defined 0 instead of inf/inf. exp(80) is
            # ~5.5e34, still 4 orders below the fp32 ceiling.
            denom += tl.exp(tl.minimum(nclse - m, 80.0))

        p = e / tl.where(denom > 0.0, denom, 1.0)[:, None]

        acc += tl.sum(p, axis=0)

    if NORMALIZE:
        total = tl.sum(acc)
        acc = acc / tl.where(total > eps, total, eps)

    tl.store(OUT_PTR + b * o_sb + s * o_ss + offs_k * o_sk, acc)


def _read_env_int(env: str, default: int) -> int:
    raw = os.environ.get(env)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _auto_block_h(H: int) -> int:
    """Largest head block <= 64 that divides ``H``; 0 when none does.

    tl.dot needs at least 16 on the M axis, so anything below that is left to
    the eager path rather than padded.
    """
    for cand in (64, 32, 16):
        if cand <= _MAX_BLOCK_H and H % cand == 0:
            return cand
    return 0


def _block_sizes(H: int) -> tuple[int, int, int]:
    block_h = _read_env_int(_BLOCK_H_ENV, 0) or _auto_block_h(H)
    block_d = _read_env_int(_BLOCK_D_ENV, _DEFAULT_BLOCK_D)
    warps = _read_env_int(_WARPS_ENV, _DEFAULT_WARPS)
    return block_h, block_d, warps


def can_use_triton_target(*, query: torch.Tensor, topk_idxs: torch.Tensor) -> bool:
    """Whether the fused kernel covers this shape / configuration."""
    if os.environ.get(_ENABLE_ENV, "1") != "1":
        return False
    if not query.is_cuda:
        return False

    _, H, _, D = query.shape
    K = topk_idxs.shape[-1]
    block_h, block_d, _ = _block_sizes(H)

    # tl.arange over K needs a power of two; tl.dot needs >= 16 on every axis.
    if K < 16 or (K & (K - 1)) != 0:
        return False
    if block_h < 16 or block_d < 16:
        return False
    if H % block_h or D % block_d:
        return False
    return True


def target_distribution_triton(
    *,
    query: torch.Tensor,
    pool: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    normalize: bool = False,
    eps: float = 1e-10,
    noncompressed_lse: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Head-summed attention distribution over the selected entries: ``[B, S, K]``.

    Numerically equivalent to the eager ``_target_distribution``: per-head
    softmax over the K selected entries with invalid slots masked out, summed
    over heads, and zero on rows that have no legal entry.

    ``normalize`` folds the caller's ``target /= target.sum(-1)`` into the
    kernel. Pass ``False`` when the head sum still needs an all-reduce.

    ``noncompressed_lse`` is ``[B, H, S]`` fp32: the log of the attention mass
    the layer places outside the compressed entries (sliding window + sink).
    Supplying it makes the per-head softmax the conditional the joint CSA
    softmax actually produces rather than one renormalised over the compressed
    entries alone.
    """
    B, H, S, D = query.shape
    K = topk_idxs.shape[-1]
    block_h, block_d, warps = _block_sizes(H)

    q = query if query.stride(-1) == 1 else query.contiguous()
    kv = pool if pool.stride(-1) == 1 else pool.contiguous()
    idx = topk_idxs if topk_idxs.stride(-1) == 1 else topk_idxs.contiguous()

    has_nclse = noncompressed_lse is not None
    if has_nclse:
        nclse = noncompressed_lse
        if nclse.shape != (B, H, S):
            raise ValueError(f"noncompressed_lse must be [B, H, S] = {(B, H, S)}, got {tuple(nclse.shape)}")
        nclse = nclse.to(torch.float32)
        if nclse.stride(-1) != 1:
            nclse = nclse.contiguous()
        n_strides = (nclse.stride(0), nclse.stride(1), nclse.stride(2))
    else:
        # Triton still needs a real pointer and three strides for the unused
        # argument; the query tensor is already resident, so alias it.
        nclse = q
        n_strides = (0, 0, 0)

    out = torch.empty((B, S, K), device=query.device, dtype=torch.float32)

    _distill_target_fwd_kernel[(B, S)](
        q,
        kv,
        idx,
        out,
        nclse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        idx.stride(0),
        idx.stride(1),
        idx.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        n_strides[0],
        n_strides[1],
        n_strides[2],
        float(softmax_scale),
        float(eps),
        H=H,
        K=K,
        D=D,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        NORMALIZE=bool(normalize),
        HAS_NCLSE=has_nclse,
        num_warps=warps,
    )
    return out
