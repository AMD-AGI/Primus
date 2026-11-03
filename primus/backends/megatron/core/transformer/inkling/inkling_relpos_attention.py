###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused relative-bias attention for Inkling (``inkling_attention_impl: fused``).

A Triton flash-attention forward and backward that add Inkling's banded relative bias inside the
kernel: ``score[q, k] = scale * <q_q, k_k> + rel_logits[q, q - k]`` for ``0 <= q - k < E`` (else
no bias), then the causal mask, the sliding window (keys with ``q - k >= window`` are dropped) and
an optional boolean ``attention_mask`` (True = masked out). No ``[s, s]`` score or bias tensor is
stored: memory is ``O(s * (d + E))`` per head instead of the eager path's ``O(s * s)``.

``rel_logits [b, h, s, E]`` (``r @ rel_proj``, already multiplied by tau on the global layers) is
an input of the op, so autograd reaches ``r``, ``rel_proj`` and tau outside the kernel. Its
gradient is written by the query-block backward program: every ``(q, d)`` entry comes from
exactly one key (``k = q - d``), so no atomics are needed.

GQA: query head ``h`` reads K / V head ``h // (h_q / h_kv)``; K / V are never repeated. The K / V
gradients are accumulated per query head in fp32 and summed over each group.

Numbers: the dot products accumulate in fp32 (fp32 inputs use IEEE fp32), the softmax is fp32
(the log-sum-exp is saved for backward), as in the eager reference
(``InklingAttention._core_attention``).
"""

import torch
import triton
import triton.language as tl

_NEG = tl.constexpr(
    -1.0e30
)  # finite "minus infinity": a fully masked row stays finite (its probabilities are 0)


@triton.jit
def _band_bounds(start_m, S, WINDOW, BLOCK_M: tl.constexpr, HAS_WINDOW: tl.constexpr):
    hi = tl.minimum(start_m + BLOCK_M, S)
    if HAS_WINDOW:
        lo = tl.maximum(start_m - WINDOW + 1, 0)
    else:
        lo = 0
    return lo, hi


@triton.jit
def _scores(
    q,
    k,
    start_m,
    start_n,
    offs_m,
    offs_n,
    S,
    E,
    WINDOW,
    scale,
    rel_ptr,
    s_rel_q,
    mask_ptr,
    s_mask_q,
    s_mask_k,
    HAS_WINDOW: tl.constexpr,
    HAS_MASK: tl.constexpr,
    IEEE: tl.constexpr,
):
    """Biased, masked scores of one [BLOCK_M, BLOCK_N] tile (fp32) and its keep mask."""
    if IEEE:
        qk = tl.dot(q, tl.trans(k), input_precision="ieee")
    else:
        qk = tl.dot(q, tl.trans(k))
    qi = start_m + offs_m[:, None]
    ki = start_n + offs_n[None, :]
    dist = qi - ki
    keep = (dist >= 0) & (qi < S) & (ki < S)
    if HAS_WINDOW:
        keep = keep & (dist < WINDOW)
    in_band = keep & (dist < E)
    bias = tl.load(rel_ptr + qi.to(tl.int64) * s_rel_q + dist, mask=in_band, other=0.0).to(tl.float32)
    if HAS_MASK:
        m = tl.load(mask_ptr + qi.to(tl.int64) * s_mask_q + ki.to(tl.int64) * s_mask_k, mask=keep, other=0)
        keep = keep & (m == 0)
    s = qk * scale + bias
    s = tl.where(keep, s, _NEG)
    return s, keep, in_band, dist, qi


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    REL,
    MASK,
    O,
    LSE,
    sqb,
    sqh,
    sqs,
    skb,
    skh,
    sks,
    svb,
    svh,
    svs,
    srb,
    srh,
    srs,
    smb,
    smh,
    smq,
    smk,
    sob,
    soh,
    sos,
    H,
    REP,
    S,
    E,
    WINDOW,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    HAS_MASK: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)  # int64 offsets: b*stride, bh*S*D can pass 2**31
    b = bh // H
    h = bh % H
    hk = h // REP
    start_m = pid_m * BLOCK_M
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    qmask = (start_m + offs_m) < S
    q = tl.load(
        Q + b * sqb + h * sqh + (start_m + offs_m).to(tl.int64)[:, None] * sqs + offs_d[None, :],
        mask=qmask[:, None],
        other=0.0,
    )
    rel_ptr = REL + b * srb + h * srh
    mask_ptr = MASK + b * smb + h * smh
    m_i = tl.full([BLOCK_M], _NEG, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    lo, hi = _band_bounds(start_m, S, WINDOW, BLOCK_M, HAS_WINDOW)
    lo = (lo // BLOCK_N) * BLOCK_N
    for start_n in range(lo, hi, BLOCK_N):
        kmask = (start_n + offs_n) < S
        k = tl.load(
            K + b * skb + hk * skh + (start_n + offs_n).to(tl.int64)[:, None] * sks + offs_d[None, :],
            mask=kmask[:, None],
            other=0.0,
        )
        v = tl.load(
            V + b * svb + hk * svh + (start_n + offs_n).to(tl.int64)[:, None] * svs + offs_d[None, :],
            mask=kmask[:, None],
            other=0.0,
        )
        s, keep, _, _, _ = _scores(
            q,
            k,
            start_m,
            start_n,
            offs_m,
            offs_n,
            S,
            E,
            WINDOW,
            scale,
            rel_ptr,
            srs,
            mask_ptr,
            smq,
            smk,
            HAS_WINDOW,
            HAS_MASK,
            IEEE,
        )
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.where(keep, tl.exp(s - m_new[:, None]), 0.0)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        if IEEE:
            acc += tl.dot(p.to(v.dtype), v, input_precision="ieee")
        else:
            acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]
    lse = tl.where(l_i > 0, m_i + tl.log(l_safe), float("inf"))  # inf: a fully masked row (p = 0)
    tl.store(
        O + b * sob + h * soh + (start_m + offs_m).to(tl.int64)[:, None] * sos + offs_d[None, :],
        out.to(O.dtype.element_ty),
        mask=qmask[:, None],
    )
    tl.store(LSE + bh * S + start_m + offs_m, lse, mask=qmask)


@triton.jit
def _bwd_dq_kernel(
    Q,
    K,
    V,
    REL,
    MASK,
    DO,
    LSE,
    DELTA,
    DQ,
    DREL,
    sqb,
    sqh,
    sqs,
    skb,
    skh,
    sks,
    svb,
    svh,
    svs,
    srb,
    srh,
    srs,
    smb,
    smh,
    smq,
    smk,
    sdob,
    sdoh,
    sdos,
    H,
    REP,
    S,
    E,
    WINDOW,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    HAS_MASK: tl.constexpr,
    IEEE: tl.constexpr,
):
    """dQ and dREL of one query block (dREL[q, d] = dS[q, q - d]: one key per entry)."""
    pid_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)  # int64 offsets: b*stride, bh*S*D can pass 2**31
    b = bh // H
    h = bh % H
    hk = h // REP
    start_m = pid_m * BLOCK_M
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    qmask = (start_m + offs_m) < S
    rows = (start_m + offs_m).to(tl.int64)[:, None]
    q = tl.load(Q + b * sqb + h * sqh + rows * sqs + offs_d[None, :], mask=qmask[:, None], other=0.0)
    do = tl.load(DO + b * sdob + h * sdoh + rows * sdos + offs_d[None, :], mask=qmask[:, None], other=0.0)
    lse = tl.load(LSE + bh * S + start_m + offs_m, mask=qmask, other=float("inf"))
    delta = tl.load(DELTA + bh * S + start_m + offs_m, mask=qmask, other=0.0)
    rel_ptr = REL + b * srb + h * srh
    drel_ptr = DREL + bh * S * E
    mask_ptr = MASK + b * smb + h * smh
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    lo, hi = _band_bounds(start_m, S, WINDOW, BLOCK_M, HAS_WINDOW)
    lo = (lo // BLOCK_N) * BLOCK_N
    for start_n in range(lo, hi, BLOCK_N):
        kmask = (start_n + offs_n) < S
        k = tl.load(
            K + b * skb + hk * skh + (start_n + offs_n).to(tl.int64)[:, None] * sks + offs_d[None, :],
            mask=kmask[:, None],
            other=0.0,
        )
        v = tl.load(
            V + b * svb + hk * svh + (start_n + offs_n).to(tl.int64)[:, None] * svs + offs_d[None, :],
            mask=kmask[:, None],
            other=0.0,
        )
        s, keep, in_band, dist, qi = _scores(
            q,
            k,
            start_m,
            start_n,
            offs_m,
            offs_n,
            S,
            E,
            WINDOW,
            scale,
            rel_ptr,
            srs,
            mask_ptr,
            smq,
            smk,
            HAS_WINDOW,
            HAS_MASK,
            IEEE,
        )
        p = tl.where(keep, tl.exp(s - lse[:, None]), 0.0)
        if IEEE:
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        else:
            dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        tl.store(drel_ptr + qi * E + dist, ds, mask=in_band)
        if IEEE:
            dq += tl.dot(ds.to(k.dtype), k, input_precision="ieee")
        else:
            dq += tl.dot(ds.to(k.dtype), k)
    tl.store(DQ + bh * S * D + rows * D + offs_d[None, :], dq * scale, mask=qmask[:, None])


@triton.jit
def _bwd_dkv_kernel(
    Q,
    K,
    V,
    REL,
    MASK,
    DO,
    LSE,
    DELTA,
    DK,
    DV,
    sqb,
    sqh,
    sqs,
    skb,
    skh,
    sks,
    svb,
    svh,
    svs,
    srb,
    srh,
    srs,
    smb,
    smh,
    smq,
    smk,
    sdob,
    sdoh,
    sdos,
    H,
    REP,
    S,
    E,
    WINDOW,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    HAS_MASK: tl.constexpr,
    IEEE: tl.constexpr,
):
    """dK and dV (per query head, fp32) of one key block."""
    pid_n = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)  # int64 offsets: b*stride, bh*S*D can pass 2**31
    b = bh // H
    h = bh % H
    hk = h // REP
    start_n = pid_n * BLOCK_N
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    kmask = (start_n + offs_n) < S
    cols = (start_n + offs_n).to(tl.int64)[:, None]
    k = tl.load(K + b * skb + hk * skh + cols * sks + offs_d[None, :], mask=kmask[:, None], other=0.0)
    v = tl.load(V + b * svb + hk * svh + cols * svs + offs_d[None, :], mask=kmask[:, None], other=0.0)
    rel_ptr = REL + b * srb + h * srh
    mask_ptr = MASK + b * smb + h * smh
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    # queries that can see this key block: from its first key to (window) its last key + window - 1
    m_lo = (start_n // BLOCK_M) * BLOCK_M
    if HAS_WINDOW:
        m_hi = tl.minimum(start_n + BLOCK_N + WINDOW - 1, S)
    else:
        m_hi = S
    for start_m in range(m_lo, m_hi, BLOCK_M):
        qmask = (start_m + offs_m) < S
        rows = (start_m + offs_m).to(tl.int64)[:, None]
        q = tl.load(Q + b * sqb + h * sqh + rows * sqs + offs_d[None, :], mask=qmask[:, None], other=0.0)
        do = tl.load(DO + b * sdob + h * sdoh + rows * sdos + offs_d[None, :], mask=qmask[:, None], other=0.0)
        lse = tl.load(LSE + bh * S + start_m + offs_m, mask=qmask, other=float("inf"))
        delta = tl.load(DELTA + bh * S + start_m + offs_m, mask=qmask, other=0.0)
        s, keep, _, _, _ = _scores(
            q,
            k,
            start_m,
            start_n,
            offs_m,
            offs_n,
            S,
            E,
            WINDOW,
            scale,
            rel_ptr,
            srs,
            mask_ptr,
            smq,
            smk,
            HAS_WINDOW,
            HAS_MASK,
            IEEE,
        )
        p = tl.where(keep, tl.exp(s - lse[:, None]), 0.0)
        if IEEE:
            dv += tl.dot(tl.trans(p.to(do.dtype)), do, input_precision="ieee")
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        else:
            dv += tl.dot(tl.trans(p.to(do.dtype)), do)
            dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        if IEEE:
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision="ieee")
        else:
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
    tl.store(DK + bh * S * D + cols * D + offs_d[None, :], dk * scale, mask=kmask[:, None])
    tl.store(DV + bh * S * D + cols * D + offs_d[None, :], dv, mask=kmask[:, None])


_LARGE_TILE_MAX_HEAD_DIM = 128  # the largest head dim that gets the 64 x 64 tile


def _blocks(D, dtype):
    """(BLOCK_M, BLOCK_N): query / key tile sizes.

    The backward kernel holds a Q, K, V and dO tile, the relative-bias rows and fp32 accumulators
    per program: fp32 inputs (twice the bytes of bf16) and head dims above 128 use 32 x 32 tiles,
    so the K / V and bias tiles fit in LDS without spilling registers; 64 x 64 is the larger tile
    that fits for bf16 / fp16 with D <= 128. Not autotuned: a safe default on MI300 / MI355
    (the Inkling head dims are 128 and below; the GPU test compares fused and eager attention,
    forward and gradients, on a small config)."""
    if dtype == torch.float32:
        return 32, 32
    return (64, 64) if D <= _LARGE_TILE_MAX_HEAD_DIM else (32, 32)


def _mask_args(mask, b, h, s, device):
    """A bool mask broadcast to [b, h, s, s] (stride 0 on broadcast axes), or a dummy.

    A bool mask is reinterpreted as uint8 in place (``view``: no copy per layer, and the tensor
    saved for backward is the caller's mask, not a new [b, 1, s, s] buffer)."""
    if mask is None:
        return torch.zeros(1, dtype=torch.uint8, device=device), (0, 0, 0, 0), False
    m = mask.view(torch.uint8) if mask.dtype == torch.bool else (mask != 0).view(torch.uint8)
    while m.dim() < 4:
        m = m.unsqueeze(0)
    m = m.expand(b, h, s, s)
    return m, tuple(m.stride()), True


class _RelposAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, rel, mask, scale, window):
        B, H, S, D = q.shape
        KV, E = k.shape[1], rel.shape[-1]
        assert H % KV == 0 and D in (16, 32, 64, 128, 256), (H, KV, D)
        q, k, v, rel = (t if t.stride(-1) == 1 else t.contiguous() for t in (q, k, v, rel))
        o = torch.empty_like(q, memory_format=torch.contiguous_format)
        lse = torch.empty(B * H, S, dtype=torch.float32, device=q.device)
        m, ms, has_mask = _mask_args(mask, B, H, S, q.device)
        bm, bn = _blocks(D, q.dtype)
        ieee = q.dtype == torch.float32
        W = int(window) if window else 0
        _fwd_kernel[(triton.cdiv(S, bm), B * H)](
            q,
            k,
            v,
            rel,
            m,
            o,
            lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            rel.stride(0),
            rel.stride(1),
            rel.stride(2),
            *ms,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            H,
            H // KV,
            S,
            E,
            W,
            float(scale),
            D=D,
            BLOCK_M=bm,
            BLOCK_N=bn,
            HAS_WINDOW=bool(W),
            HAS_MASK=has_mask,
            IEEE=ieee,
        )
        ctx.save_for_backward(q, k, v, rel, m, o, lse)
        ctx.meta = (scale, W, has_mask, ms)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, rel, m, o, lse = ctx.saved_tensors
        scale, W, has_mask, ms = ctx.meta
        B, H, S, D = q.shape
        KV, E = k.shape[1], rel.shape[-1]
        do = do if do.stride(-1) == 1 else do.contiguous()
        delta = (do.float() * o.float()).sum(-1).reshape(B * H, S).contiguous()
        dq = torch.empty(B, H, S, D, dtype=torch.float32, device=q.device)
        dk = torch.empty(B, H, S, D, dtype=torch.float32, device=q.device)
        dv = torch.empty(B, H, S, D, dtype=torch.float32, device=q.device)
        drel = torch.zeros(B, H, S, E, dtype=torch.float32, device=q.device)
        bm, bn = _blocks(D, q.dtype)
        ieee = q.dtype == torch.float32
        common = (
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            rel.stride(0),
            rel.stride(1),
            rel.stride(2),
            *ms,
            do.stride(0),
            do.stride(1),
            do.stride(2),
            H,
            H // KV,
            S,
            E,
            W,
            float(scale),
        )
        flags = dict(D=D, BLOCK_M=bm, BLOCK_N=bn, HAS_WINDOW=bool(W), HAS_MASK=has_mask, IEEE=ieee)
        _bwd_dq_kernel[(triton.cdiv(S, bm), B * H)](
            q, k, v, rel, m, do, lse, delta, dq, drel, *common, **flags
        )
        _bwd_dkv_kernel[(triton.cdiv(S, bn), B * H)](
            q, k, v, rel, m, do, lse, delta, dk, dv, *common, **flags
        )
        rep = H // KV
        dk = dk.view(B, KV, rep, S, D).sum(2)
        dv = dv.view(B, KV, rep, S, D).sum(2)
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), drel.to(rel.dtype), None, None, None)


def relpos_flash_attention(q, k, v, rel_logits, scale, window=None, attention_mask=None):
    """``softmax(scale * q k^T + band(rel_logits) + mask) v`` without an ``[s, s]`` buffer.

    q ``[b, h, s, d]``, k / v ``[b, h_kv, s, d]`` (``h % h_kv == 0``), rel_logits ``[b, h, s, E]``
    (``rel_logits[.., q, dist]`` is the bias of key ``q - dist``), ``window`` (None = full causal)
    keeps keys with ``q - k < window``; ``attention_mask`` (bool, True = masked, broadcastable to
    ``[b, h, s, s]``). Returns ``[b, h, s, d]`` in q's dtype. A row with no visible key is 0."""
    return _RelposAttention.apply(q, k, v, rel_logits, attention_mask, scale, window)
