###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton-fused sliding-window log-sum-exp for the indexer distillation target.

:func:`primus.backends.megatron.core.transformer.indexer_distill_loss.noncompressed_lse`
needs, per query and per head, the log attention mass outside the compressed
entries: the sliding window keys plus the sink. That is the whole cost of making
the distillation target a conditional of the layer's joint softmax -- the extra
denominator term inside the target kernel itself is one load and one exp.

The eager version scores a chunk of queries against every key any of them can
see, which at V4-Flash widths (S=4096, window=128, H=64, head_dim=512) means a
``[1, 64, 512, 639]`` fp32 logit tensor per chunk, of which the sliding-window
mask throws away four fifths. Measured on MI355X it costs 2.00 ms against the
0.37 ms the target kernel spends on 4x the arithmetic.

This keeps the scores in registers and never writes logits to HBM. A program
owns ``BLOCK_S`` queries of one head and loads the ``BLOCK_S + window - 1`` keys
they collectively span, so the only waste left is the triangle inside that band:

    B=1, H=64, S=4096, head_dim=512, window=128, bf16

        eager   2.00 ms
        fused   0.17 ms   -> 12x

Gating: ``PRIMUS_V4_DISTILL_WINDOW_TRITON`` (default ON).
:func:`can_use_triton_window_lse` declines windows too wide for the key band to
fit in registers, so the caller always has the eager path to fall back to.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

__all__ = ["can_use_triton_window_lse", "window_lse_triton"]

_ENABLE_ENV = "PRIMUS_V4_DISTILL_WINDOW_TRITON"
_BLOCK_S_ENV = "PRIMUS_V4_DISTILL_WINDOW_BLOCK_S"
_BLOCK_D_ENV = "PRIMUS_V4_DISTILL_WINDOW_BLOCK_D"

_DEFAULT_BLOCK_S = 64
_DEFAULT_BLOCK_D = 64
# The key band is BLOCK_S + window - 1 rounded up to a power of two. Past this
# the tile stops fitting and the eager path is the better answer.
_MAX_BLOCK_KV = 1024


@triton.jit
def _window_lse_kernel(
    Q_PTR,  # [B, H, S, D]  post-RoPE queries
    K_PTR,  # [B, H, S, D]  sliding-window keys
    SINK_PTR,  # [H]           per-head sink logits (fp32)
    OUT_PTR,  # [B, H, S]     log mass (fp32)
    q_sb,
    q_sh,
    q_ss,
    q_sd,
    k_sb,
    k_sh,
    k_ss,
    k_sd,
    o_sb,
    o_sh,
    o_ss,
    scale,
    S,
    WINDOW,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    """One program = ``BLOCK_S`` queries of one ``(b, h)``.

    Queries ``[s0, s0 + BLOCK_S)`` can only see keys
    ``[s0 - WINDOW + 1, s0 + BLOCK_S)``, so one band of ``BLOCK_KV`` keys covers
    the whole block and the scores never leave registers.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    s0 = tl.program_id(2) * BLOCK_S

    offs_s = s0 + tl.arange(0, BLOCK_S)
    offs_kv = (s0 - WINDOW + 1) + tl.arange(0, BLOCK_KV)

    q_row = offs_s < S
    k_row = (offs_kv >= 0) & (offs_kv < S)

    q_base = Q_PTR + b * q_sb + h * q_sh
    k_base = K_PTR + b * k_sb + h * k_sh

    logits = tl.zeros([BLOCK_S, BLOCK_KV], dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        q = tl.load(
            q_base + offs_s[:, None] * q_ss + offs_d[None, :] * q_sd,
            mask=q_row[:, None],
            other=0.0,
        )
        k = tl.load(
            k_base + offs_kv[:, None] * k_ss + offs_d[None, :] * k_sd,
            mask=k_row[:, None],
            other=0.0,
        )
        logits += tl.dot(q, tl.trans(k))

    logits *= scale

    # Same predicate as ``sliding_window_causal_mask``: causal and inside the
    # window. Every in-range query keeps at least its own key, so no row of the
    # reduction below is entirely -inf.
    dist = offs_s[:, None] - offs_kv[None, :]
    valid = (dist >= 0) & (dist < WINDOW) & k_row[None, :] & q_row[:, None]
    logits = tl.where(valid, logits, float("-inf"))

    m = tl.max(logits, axis=1)
    m = tl.where(m == float("-inf"), 0.0, m)
    e = tl.where(valid, tl.exp(logits - m[:, None]), 0.0)
    total = tl.sum(e, axis=1)
    lse = m + tl.log(tl.where(total > 0.0, total, 1.0))

    if HAS_SINK:
        sink = tl.load(SINK_PTR + h).to(tl.float32)
        hi = tl.maximum(lse, sink)
        lse = hi + tl.log(tl.exp(lse - hi) + tl.exp(sink - hi))

    tl.store(OUT_PTR + b * o_sb + h * o_sh + offs_s * o_ss, lse, mask=q_row)


def _read_env_int(env: str, default: int) -> int:
    raw = os.environ.get(env)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _block_sizes(window: int) -> tuple[int, int, int]:
    block_s = _read_env_int(_BLOCK_S_ENV, _DEFAULT_BLOCK_S)
    block_d = _read_env_int(_BLOCK_D_ENV, _DEFAULT_BLOCK_D)
    block_kv = triton.next_power_of_2(block_s + window - 1)
    return block_s, block_kv, block_d


def can_use_triton_window_lse(*, query: torch.Tensor, k_local: torch.Tensor, swa_window: int) -> bool:
    """Whether the fused window log-sum-exp covers this configuration."""
    if os.environ.get(_ENABLE_ENV, "1") != "1":
        return False
    if not (query.is_cuda and k_local.is_cuda):
        return False
    if swa_window <= 0:
        # Degenerate "full causal" -- the band would be the whole sequence.
        return False
    if query.shape != k_local.shape:
        return False

    D = query.shape[-1]
    block_s, block_kv, block_d = _block_sizes(swa_window)
    if block_kv > _MAX_BLOCK_KV:
        return False
    # tl.dot needs at least 16 on every axis and a clean feature split.
    if block_s < 16 or block_d < 16 or block_kv < 16:
        return False
    return D % block_d == 0


def window_lse_triton(
    *,
    query: torch.Tensor,
    k_local: torch.Tensor,
    sink: Optional[torch.Tensor],
    swa_window: int,
    softmax_scale: float,
) -> torch.Tensor:
    """``[B, H, S]`` fp32 ``logaddexp(logsumexp_window(q . k * scale), sink)``."""
    B, H, S, D = query.shape
    block_s, block_kv, block_d = _block_sizes(swa_window)

    q = query if query.stride(-1) == 1 else query.contiguous()
    k = k_local if k_local.stride(-1) == 1 else k_local.contiguous()

    has_sink = sink is not None
    if has_sink:
        sink_t = sink.detach().to(torch.float32)
        if sink_t.numel() != H:
            raise ValueError(f"sink must hold {H} head values, got {sink_t.numel()}")
        sink_t = sink_t if sink_t.stride(-1) == 1 else sink_t.contiguous()
    else:
        # Triton needs a real pointer for the unused argument.
        sink_t = q

    out = torch.empty((B, H, S), device=query.device, dtype=torch.float32)

    _window_lse_kernel[(B, H, triton.cdiv(S, block_s))](
        q,
        k,
        sink_t,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        float(softmax_scale),
        S,
        int(swa_window),
        D=D,
        BLOCK_S=block_s,
        BLOCK_KV=block_kv,
        BLOCK_D=block_d,
        HAS_SINK=has_sink,
    )
    return out
