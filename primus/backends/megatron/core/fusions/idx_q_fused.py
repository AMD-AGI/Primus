###############################################################################
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""bf16g2: fused indexer query path (one Triton kernel instead of 4-6).

Reference chain (Indexer.forward -> _rotate_queries, fwd only in training: the indexer params are frozen and its
scores only feed a top-k):
    q_i  = F.linear(q_q, W_iuq)                       # [T, H*Hd] bf16       (hipBLASLt, K = dq_rank = 128)
    q_i  = apply_interleaved_partial_rope(q_i, cos, sin, rotary_dim)   # Triton _apply_rope_fwd_kernel, bf16 math
    q_i  = torch.matmul(q_i, Hadamard) * Hd**-0.5     # hipBLASLt (K = Hd = 128) + an elementwise mul
Fused: per (token block of 128, 8 heads; the next head's W rows are prefetched before the current head's GEMMs): the same two K=128 GEMMs (fp32 accumulate, bf16 rounding of each result), the same
bf16 rope expressions on the rounded GEMM output, the bf16 rounding of the Hadamard product before the fp32 scale.
Meant to be byte-identical to the chain (checked by tools/bench_idx.py on the real shapes).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _one_head(q, c, s, hm, rot, W_PTR, OUT_PTR, t, tm, k, n, h, norm, H: tl.constexpr, HD: tl.constexpr,
              DQ: tl.constexpr, BT: tl.constexpr, HAS_W: tl.constexpr, w=None):
    if HAS_W:
        if w is None:
            w = tl.load(W_PTR + (h * HD + n)[:, None] * DQ + k[None, :])  # [HD, DQ] rows of W_iuq (K contiguous)
        x = tl.dot(q, tl.trans(w)).to(tl.bfloat16)  # [BT, HD]
    else:
        x = q
    # interleaved partial rope on channels NOPE.. (pairs (2j, 2j+1)), bf16 math as in _apply_rope_fwd_kernel
    e, o = tl.split(tl.reshape(x, (BT, HD // 2, 2)))
    re = e * c - o * s
    ro = e * s + o * c
    e2 = tl.where(rot[None, :], re, e)
    o2 = tl.where(rot[None, :], ro, o)
    x2 = tl.reshape(tl.join(e2, o2), (BT, HD))
    y = tl.dot(x2, hm).to(tl.bfloat16).to(tl.float32) * norm
    tl.store(OUT_PTR + (t[:, None].to(tl.int64) * H + h) * HD + n[None, :], y.to(tl.bfloat16), mask=tm[:, None])


@triton.jit
def _idx_q_fused_kernel(QQ_PTR, stride_qq, W_PTR, COS_PTR, SIN_PTR, HM_PTR, OUT_PTR, T, norm,
                        H: tl.constexpr, HD: tl.constexpr, DQ: tl.constexpr, NOPE: tl.constexpr,
                        RDH: tl.constexpr, BT: tl.constexpr, HB: tl.constexpr, NSTAGE: tl.constexpr,
                        HAS_W: tl.constexpr = True, UNROLL: tl.constexpr = False, PF: tl.constexpr = False):
    # one program: BT tokens x HB heads (the q tile, cos / sin and the Hadamard are loaded once)
    # HAS_W=False (keys path): QQ is already the [T, HD] activation (H = 1); only rope + Hadamard + scale
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    t = pid_t * BT + tl.arange(0, BT)
    tm = t < T
    k = tl.arange(0, DQ)
    n = tl.arange(0, HD)
    q = tl.load(QQ_PTR + t[:, None].to(tl.int64) * stride_qq + k[None, :], mask=tm[:, None], other=0.0)
    j = tl.arange(0, HD // 2)
    rot = j >= NOPE // 2
    ci = tl.where(rot, j - NOPE // 2, 0)
    cm = tm[:, None] & rot[None, :]
    c = tl.load(COS_PTR + t[:, None] * RDH + ci[None, :], mask=cm, other=1.0).to(tl.bfloat16)
    s = tl.load(SIN_PTR + t[:, None] * RDH + ci[None, :], mask=cm, other=0.0).to(tl.bfloat16)
    hm = tl.load(HM_PTR + n[:, None] * HD + n[None, :])  # [HD(K), HD(N)]
    if PF:
        # software prefetch: issue the next head's W load before this head's two GEMMs
        w = tl.load(W_PTR + (pid_h * HB * HD + n)[:, None] * DQ + k[None, :])
        for hh in tl.range(0, HB, num_stages=1):
            h = pid_h * HB + hh
            hn = tl.minimum(h + 1, pid_h * HB + HB - 1)
            w_next = tl.load(W_PTR + (hn * HD + n)[:, None] * DQ + k[None, :])
            _one_head(q, c, s, hm, rot, W_PTR, OUT_PTR, t, tm, k, n, h, norm, H, HD, DQ, BT, HAS_W, w)
            w = w_next
    elif UNROLL:
        for hh in tl.static_range(HB):
            _one_head(q, c, s, hm, rot, W_PTR, OUT_PTR, t, tm, k, n, pid_h * HB + hh, norm, H, HD, DQ, BT, HAS_W)
    else:
        for hh in tl.range(0, HB, num_stages=NSTAGE):
            _one_head(q, c, s, hm, rot, W_PTR, OUT_PTR, t, tm, k, n, pid_h * HB + hh, norm, H, HD, DQ, BT, HAS_W)


def idx_q_fused(q_q, w, cos, sin, hm, rotary_dim, norm, bt=128, num_warps=8, hb=8, nstage=1, unroll=False, pf=True):
    """q_q [T, DQ] (unit inner stride), w [H*HD, DQ], cos/sin [T, rotary_dim//2] (any float dtype; rounded to bf16
    like RoPEInterleavedPartialFn), hm [HD, HD] bf16, norm (python float). Returns [T, H, HD] bf16."""
    T, DQ = q_q.shape
    HD = hm.shape[0]
    H = w.shape[0] // HD
    assert q_q.stride(1) == 1 and w.is_contiguous() and hm.is_contiguous() and w.shape[1] == DQ
    rdh = rotary_dim // 2
    cos = cos.reshape(T, rdh).contiguous()
    sin = sin.reshape(T, rdh).contiguous()
    out = torch.empty(T, H, HD, device=q_q.device, dtype=torch.bfloat16)
    if H % hb:
        hb = 1
    grid = (triton.cdiv(T, bt), H // hb)
    _idx_q_fused_kernel[grid](q_q, q_q.stride(0), w, cos, sin, hm, out, T, norm, H=H, HD=HD, DQ=DQ,
                              NOPE=HD - rotary_dim, RDH=rdh, BT=bt, HB=hb, NSTAGE=nstage, UNROLL=unroll, PF=pf,
                              num_warps=num_warps)
    return out


def rope_hadamard_fused(x, cos, sin, hm, rotary_dim, norm, bt=32, num_warps=4):
    """Keys path: x [T, HD] bf16 (unit inner stride) -> rotate_activation(apply_interleaved_partial_rope(x)) as one
    kernel (same bf16 rope expressions, same K=HD Hadamard GEMM, bf16 rounding before the fp32 scale)."""
    T, HD = x.shape
    assert x.stride(1) == 1 and hm.is_contiguous() and hm.shape == (HD, HD)
    rdh = rotary_dim // 2
    cos = cos.reshape(T, rdh).contiguous()
    sin = sin.reshape(T, rdh).contiguous()
    out = torch.empty(T, 1, HD, device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(T, bt), 1)
    _idx_q_fused_kernel[grid](x, x.stride(0), x, cos, sin, hm, out, T, norm, H=1, HD=HD, DQ=HD,
                              NOPE=HD - rotary_dim, RDH=rdh, BT=bt, HB=1, NSTAGE=1, HAS_W=False,
                              num_warps=num_warps)
    return out.view(T, HD)
