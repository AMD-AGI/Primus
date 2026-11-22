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

import os

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


# ---------------------------------------------------------------------------------------------------------------------
# Indexer glue (PRIMUS_BF16G2_IDXQ=1; called from Indexer.forward). Takes precedence over PRIMUS_FUSE2_IDX_ROT: with
# both set, the queries / keys never reach Indexer._rotate_queries / _rotate_keys (logged once).
# ---------------------------------------------------------------------------------------------------------------------
_LOGGED = {}


def _log_once(key, msg):
    if not _LOGGED.get(key):
        _LOGGED[key] = True
        print(f"[bf16g2] {msg}", flush=True)
        if os.environ.get("PRIMUS_FUSE2_IDX_ROT", "0") == "1" and not _LOGGED.get("fuse2"):
            _LOGGED["fuse2"] = True
            print("[bf16g2] PRIMUS_BF16G2_IDXQ=1 takes precedence over PRIMUS_FUSE2_IDX_ROT for the indexer "
                  "rotations (IDX_ROT skipped)", flush=True)


class _IdxQFusedFn(torch.autograd.Function):
    """Fused forward; the backward (never reached while the indexer is frozen) recomputes the reference chain."""

    @staticmethod
    def forward(ctx, q2, w, cos, sin, hm, rotary_dim, norm, shape, ref_fn):
        ctx.save_for_backward(q2, w)
        ctx.ref_fn = ref_fn
        return idx_q_fused(q2, w, cos, sin, hm, rotary_dim, norm).view(shape)

    @staticmethod
    def backward(ctx, g):
        q2, w = ctx.saved_tensors
        with torch.enable_grad():
            q2r = q2.detach().requires_grad_(ctx.needs_input_grad[0])
            wr = w.detach().requires_grad_(ctx.needs_input_grad[1])
            out = ctx.ref_fn(q2r, wr)
            ins = [t for t, need in ((q2r, ctx.needs_input_grad[0]), (wr, ctx.needs_input_grad[1])) if need]
            grads = list(torch.autograd.grad(out, ins, g)) if ins else []
        gq = grads.pop(0) if ctx.needs_input_grad[0] else None
        gw = grads.pop(0) if ctx.needs_input_grad[1] else None
        return gq, gw, None, None, None, None, None, None, None


def indexer_queries_bf16g2(indexer, q_q, position_ids, B, S, H, Hd):
    """Indexer queries: rotate(w_iuq(q_q)) as one kernel; None when not applicable (caller runs the chain)."""
    import torch.nn.functional as F

    from primus.backends.megatron.core.transformer import hadamard_rotation as _hr
    from primus.backends.megatron.core.transformer.dual_rope import apply_interleaved_partial_rope

    rope, lin = indexer.rope, indexer.w_iuq
    w = lin.weight
    if (rope is None or indexer.rotary_dim == 0 or getattr(lin, "bias", None) is not None
            or _hr._fast_hadamard_transform is not None or not q_q.is_cuda or q_q.dtype != torch.bfloat16
            or w.dtype != torch.bfloat16 or not w.is_contiguous() or w.shape != (H * Hd, q_q.shape[-1])
            or Hd & (Hd - 1) or q_q.shape[-1] & (q_q.shape[-1] - 1) or q_q.stride(-1) != 1):
        return None
    if position_ids is None:
        position_ids = torch.arange(S, device=q_q.device)
    from primus.backends.megatron.core.transformer import dual_rope as _dr

    if getattr(_dr, "mb_rope_tbl", lambda: False)() and hasattr(rope, "forward_pos_cast"):
        cos, sin = rope.forward_pos_cast(position_ids, q_q.dtype)  # membound memo (bf16 tables; same bytes)
    else:
        cos, sin = rope(position_ids)
    if tuple(cos.shape[:-1]) != (B, S):
        cos = cos.expand(B, S, -1)
        sin = sin.expand(B, S, -1)
    hm = _hr._sylvester(Hd, q_q.device, q_q.dtype)
    rd = indexer.rotary_dim

    def ref_fn(q2r, wr):
        qi = F.linear(q2r.view(B, S, -1), wr).view(B, S, H, Hd)
        return _hr.rotate_activation(apply_interleaved_partial_rope(qi, cos, sin, rotary_dim=rd))

    _log_once("q", f"indexer query path fused (T={B * S}, H={H}, Hd={Hd}, rotary_dim={rd})")
    return _IdxQFusedFn.apply(q_q.reshape(B * S, q_q.shape[-1]), w, cos, sin, hm, rd, Hd**-0.5, (B, S, H, Hd), ref_fn)


def indexer_keys_bf16g2(indexer, k_icomp):
    """Indexer keys [B, P, Hd]: rotate (partial RoPE at arange(P) * ratio + Hadamard + scale) as one kernel; None when
    not applicable (no grad needed, power-of-two Hd, bf16, contiguous)."""
    from primus.backends.megatron.core.transformer import hadamard_rotation as _hr

    rope = indexer.rope
    if rope is None or indexer.rotary_dim == 0:
        return None
    B, P, Hd = k_icomp.shape
    if ((torch.is_grad_enabled() and k_icomp.requires_grad) or _hr._fast_hadamard_transform is not None
            or not k_icomp.is_cuda or k_icomp.dtype != torch.bfloat16 or Hd & (Hd - 1) or not k_icomp.is_contiguous()):
        return None
    from primus.backends.megatron.core.transformer import dual_rope as _dr

    if getattr(_dr, "mb_rope_tbl", lambda: False)() and hasattr(rope, "forward_arange_cast"):
        cos, sin = rope.forward_arange_cast(P, k_icomp.device, k_icomp.dtype, indexer.rotary_dim // 2,
                                            stride=indexer.compress_ratio)  # membound memo (same bytes)
    else:
        cos, sin = rope.forward_arange(P, k_icomp.device, stride=indexer.compress_ratio)
    _log_once("k", f"indexer key rope+Hadamard fused (P={P}, Hd={Hd})")
    return rope_hadamard_fused(k_icomp.reshape(B * P, Hd), cos.unsqueeze(0).expand(B, -1, -1),
                               sin.unsqueeze(0).expand(B, -1, -1), _hr._sylvester(Hd, k_icomp.device, k_icomp.dtype),
                               indexer.rotary_dim, Hd**-0.5).view(B, P, Hd)
