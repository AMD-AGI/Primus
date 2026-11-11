###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""FLUX attention's q/k RMSNorm, RoPE and stream concat as one kernel each way.

WHY:
  Between the fused QKV GEMM and flash attention, every FLUX block runs RMSNorm on q and k
  (separate weights for the text stream in dual-stream blocks), concatenates text and image
  along the sequence (dual-stream), and applies RoPE. Inductor splits that into many kernels
  that move the tensors well below the GPU's copy bandwidth: the forward re-reads q and k with a
  permuted index for RoPE's pair swap and reloads the cos/sin tables for every (token, head)
  row, and the backward runs RoPE's backward as separate elementwise passes before the RMSNorm
  backward.

WHAT:
  ``primus_flux::qk_norm_rope(qkv, txt_qkv, w_q, w_k, w_txt_q, w_txt_k, cos, sin, eps, heads)``
  takes the QKV GEMM output(s) ``[B, S, 3 * H * D]`` and returns q, k, v as ``[B, S_txt + S, H, D]``
  -- q and k normalized and rotated, v copied -- exactly as diffusers' ``FluxAttnProcessor``
  computes them (RMSNorm in FP32 with one rounding to BF16, then RoPE in FP32). One Triton
  kernel reads each token's q, k and v once, loads its cos/sin row once for all heads, and
  writes attention's layout directly. The backward kernel undoes the rotation, computes the
  RMSNorm input gradient and writes the gradient of the QKV GEMM output in place (v's slice
  included), so autograd needs no concat; the weight gradients are per-program partial sums
  reduced once, so they are deterministic.

  ``install()`` swaps ``FluxAttnProcessor.__call__`` for the same forward with the op in the
  middle, pinned to diffusers 0.40.0 by a source hash. Calls the op cannot serve (unfused
  projections, no RoPE, other dtypes, head_dim != 128, weightless norms) take diffusers' path.

  With ``flux_fused_qk_norm_rope_bias_grad`` the backward kernel also sums the QKV GEMM output's
  gradient over tokens, which is the QKV bias gradient: the GEMM then adds a detached bias and
  the op returns the bias gradient, so autograd's separate column sum, which re-reads the whole
  gradient, goes. Same forward; the bias gradient sums the same BF16 values in another order.
  Only for stock ``nn.Linear`` projections with a bias and no hooks.

  With ``flux_fused_qk_norm_rope_flydsl_bwd`` the backward runs the FlyDSL kernel of
  ``qk_norm_rope_flydsl.py`` instead, which moves the same bytes at the GPU's copy speed; calls it
  does not serve fall back to the Triton kernel.

Settings (module config, ``primus_perf:`` section):
    flux_fused_qk_norm_rope: false
    flux_fused_qk_norm_rope_bias_grad: false
    flux_fused_qk_norm_rope_flydsl_bwd: false
"""
from __future__ import annotations

import hashlib
import inspect
import logging
import threading
from typing import Optional

import torch
import torch.nn.functional as F

from primus.backends.automodel import options

logger = logging.getLogger(__name__)

_SECTION = "primus_perf"
_FLAG = f"{_SECTION}.flux_fused_qk_norm_rope"
_BIAS_FLAG = f"{_SECTION}.flux_fused_qk_norm_rope_bias_grad"
_FLYDSL_FLAG = f"{_SECTION}.flux_fused_qk_norm_rope_flydsl_bwd"
_PATCHED_ATTR = "_primus_fused_qk_norm_rope"
_HEAD_DIM = 128
_BLOCK_H = 8
_FWD_TOKENS = 4
_BWD_PROGRAMS = 2048
# Backward programs per stream when the kernel also sums the bias gradient: each writes one FP32
# row of partial sums for its heads' QKV columns, so fewer programs mean less to reduce.
_BWD_PROGRAMS_BIAS = 512
_COLSUM_BLOCK_R = 32
_COLSUM_BLOCK_C = 64
# sha256 of diffusers 0.40.0's FluxAttnProcessor.__call__, which _processor_call re-implements.
_PROCESSOR_SHA256 = "d5b36134887ebe450b6c15f200bf33f781ba8258201d25caceb16bb9bc07d2c9"
_lock = threading.Lock()
_state = {"bias_grad": False, "flydsl_bwd": False}


def is_enabled() -> bool:
    return options.flag(_FLAG)


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _rope_pair(x, c, s, BLOCK_H: tl.constexpr, D: tl.constexpr):
        # x [BLOCK_H, D] FP32; pairs (2i, 2i+1) rotate by the same angle, c/s are [D // 2].
        xe, xo = tl.split(tl.reshape(x, (BLOCK_H, D // 2, 2)))
        ye = xe * c[None, :] - xo * s[None, :]
        yo = xo * c[None, :] + xe * s[None, :]
        return tl.reshape(tl.join(ye, yo), (BLOCK_H, D))

    @triton.jit
    def _rope_pair_t(g, c, s, BLOCK_H: tl.constexpr, D: tl.constexpr):
        # Transpose of _rope_pair (the rotation by -angle).
        ge, go = tl.split(tl.reshape(g, (BLOCK_H, D // 2, 2)))
        ye = ge * c[None, :] + go * s[None, :]
        yo = go * c[None, :] - ge * s[None, :]
        return tl.reshape(tl.join(ye, yo), (BLOCK_H, D))

    @triton.jit
    def _norm_rope(x_ptrs, out_ptrs, w, c, s, eps, hmask, BLOCK_H: tl.constexpr, D: tl.constexpr):
        x = tl.load(x_ptrs, mask=hmask, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
        # RMSNorm's output is BF16 in diffusers; RoPE then works on it in FP32.
        y = (x * rstd[:, None] * w[None, :]).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptrs, _rope_pair(y, c, s, BLOCK_H, D).to(tl.bfloat16), mask=hmask)

    @triton.jit
    def fwd_kernel(
        src_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        wq_ptr,
        wk_ptr,
        cos_ptr,
        sin_ptr,
        n_tok,
        s_src,
        s_out,
        seq_off,
        eps,
        src_stride,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_H: tl.constexpr,
        TOKENS: tl.constexpr,
    ):
        pid_t, pid_h = tl.program_id(0), tl.program_id(1)
        h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        d = tl.arange(0, D)
        hmask = (h < H)[:, None]
        hd = h[:, None] * D + d[None, :]
        half = tl.arange(0, D // 2) * 2
        wq = tl.load(wq_ptr + d).to(tl.float32)
        wk = tl.load(wk_ptr + d).to(tl.float32)
        for i in tl.static_range(TOKENS):
            tok = pid_t * TOKENS + i
            if tok < n_tok:
                b = tok // s_src
                pos = tok % s_src + seq_off
                src = src_ptr + tok.to(tl.int64) * src_stride
                dst = (b.to(tl.int64) * s_out + pos) * (H * D)
                c = tl.load(cos_ptr + pos * D + half).to(tl.float32)
                s = tl.load(sin_ptr + pos * D + half).to(tl.float32)
                _norm_rope(src + hd, q_ptr + dst + hd, wq, c, s, eps, hmask, BLOCK_H, D)
                _norm_rope(src + H * D + hd, k_ptr + dst + hd, wk, c, s, eps, hmask, BLOCK_H, D)
                vv = tl.load(src + 2 * H * D + hd, mask=hmask)
                tl.store(v_ptr + dst + hd, vv, mask=hmask)

    @triton.jit
    def _norm_rope_bwd(x_ptrs, go_ptrs, dx_ptrs, w, c, s, eps, hmask, BLOCK_H: tl.constexpr, D: tl.constexpr):
        x = tl.load(x_ptrs, mask=hmask, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
        xhat = x * rstd[:, None]
        go = tl.load(go_ptrs, mask=hmask, other=0.0).to(tl.float32)
        # Gradient of the BF16 norm output, rounded as autograd rounds it.
        dy = _rope_pair_t(go, c, s, BLOCK_H, D).to(tl.bfloat16).to(tl.float32)
        g = dy * w[None, :]
        dx = (rstd[:, None] * (g - xhat * (tl.sum(g * xhat, axis=1) / D)[:, None])).to(tl.bfloat16)
        tl.store(dx_ptrs, dx, mask=hmask)
        return tl.sum(dy * xhat, axis=0), dx.to(tl.float32)

    @triton.jit
    def bwd_kernel(
        src_ptr,
        dsrc_ptr,
        dq_ptr,
        dk_ptr,
        dv_ptr,
        wq_ptr,
        wk_ptr,
        cos_ptr,
        sin_ptr,
        part_ptr,
        bpart_ptr,
        n_tok,
        s_src,
        s_out,
        seq_off,
        eps,
        tok_per_prog,
        src_stride,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BIAS: tl.constexpr,
    ):
        pid_t, pid_h = tl.program_id(0), tl.program_id(1)
        h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        d = tl.arange(0, D)
        hmask = (h < H)[:, None]
        hd = h[:, None] * D + d[None, :]
        half = tl.arange(0, D // 2) * 2
        wq = tl.load(wq_ptr + d).to(tl.float32)
        wk = tl.load(wk_ptr + d).to(tl.float32)
        acc_q = tl.zeros([D], dtype=tl.float32)
        acc_k = tl.zeros([D], dtype=tl.float32)
        # Column sums of the QKV GEMM output's gradient (its bias gradient), as written in BF16.
        acc_bq = tl.zeros([BLOCK_H, D], dtype=tl.float32)
        acc_bk = tl.zeros([BLOCK_H, D], dtype=tl.float32)
        acc_bv = tl.zeros([BLOCK_H, D], dtype=tl.float32)
        t0 = pid_t * tok_per_prog
        t1 = tl.minimum(t0 + tok_per_prog, n_tok)
        for tok in range(t0, t1):
            b = tok // s_src
            pos = tok % s_src + seq_off
            row = tok.to(tl.int64) * src_stride
            dst = (b.to(tl.int64) * s_out + pos) * (H * D)
            c = tl.load(cos_ptr + pos * D + half).to(tl.float32)
            s = tl.load(sin_ptr + pos * D + half).to(tl.float32)
            dw_q, dx_q = _norm_rope_bwd(
                src_ptr + row + hd, dq_ptr + dst + hd, dsrc_ptr + row + hd, wq, c, s, eps, hmask, BLOCK_H, D
            )
            dw_k, dx_k = _norm_rope_bwd(
                src_ptr + row + H * D + hd,
                dk_ptr + dst + hd,
                dsrc_ptr + row + H * D + hd,
                wk,
                c,
                s,
                eps,
                hmask,
                BLOCK_H,
                D,
            )
            acc_q += dw_q
            acc_k += dw_k
            dv = tl.load(dv_ptr + dst + hd, mask=hmask)
            tl.store(dsrc_ptr + row + 2 * H * D + hd, dv, mask=hmask)
            if BIAS:
                acc_bq += dx_q
                acc_bk += dx_k
                acc_bv += dv.to(tl.float32)
        base = part_ptr + (pid_t * tl.num_programs(1) + pid_h) * (2 * D)
        tl.store(base + d, acc_q)
        tl.store(base + D + d, acc_k)
        if BIAS:
            bbase = bpart_ptr + pid_t.to(tl.int64) * (3 * H * D)
            tl.store(bbase + hd, acc_bq, mask=hmask)
            tl.store(bbase + H * D + hd, acc_bk, mask=hmask)
            tl.store(bbase + 2 * H * D + hd, acc_bv, mask=hmask)

    @triton.jit
    def colsum_kernel(src_ptr, out_ptr, n_rows, n_cols, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
        # out[c] = sum_r src[r, c] for the FP32 partials; faster here than torch's column sum.
        cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        col_mask = cols < n_cols
        acc = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)
        for r0 in range(0, n_rows, BLOCK_R):
            rows = (r0 + tl.arange(0, BLOCK_R)).to(tl.int64)
            acc += tl.load(
                src_ptr + rows[:, None] * n_cols + cols[None, :],
                mask=(rows < n_rows)[:, None] & col_mask[None, :],
                other=0.0,
            )
        tl.store(out_ptr + cols, tl.sum(acc, axis=0).to(out_ptr.dtype.element_ty), mask=col_mask)

    return fwd_kernel, bwd_kernel, colsum_kernel


_KERNELS = None


def _get_kernels():
    global _KERNELS
    if _KERNELS is None:
        with _lock:
            if _KERNELS is None:
                _KERNELS = _kernels()
    return _KERNELS


def _segments(qkv, txt_qkv):
    """(source, sequence offset) for each stream, in output order: text first."""
    if txt_qkv is None:
        return [(qkv, 0)]
    return [(txt_qkv, 0), (qkv, txt_qkv.shape[1])]


@torch.library.custom_op("primus_flux::qk_norm_rope", mutates_args=(), device_types="cuda")
def qk_norm_rope(
    qkv: torch.Tensor,
    txt_qkv: Optional[torch.Tensor],
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_txt_q: Optional[torch.Tensor],
    w_txt_k: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    heads: int,
    bias: Optional[torch.Tensor] = None,
    txt_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``bias``/``txt_bias``: the QKV GEMMs' biases, already added into ``qkv``/``txt_qkv`` by a GEMM that
    does not differentiate them. Unused here; the backward returns their gradients."""
    fwd = _get_kernels()[0]
    qkv = qkv.contiguous()
    txt_qkv = None if txt_qkv is None else txt_qkv.contiguous()
    cos, sin = cos.contiguous(), sin.contiguous()
    b, s_img = qkv.shape[0], qkv.shape[1]
    s_out = s_img + (0 if txt_qkv is None else txt_qkv.shape[1])
    shape = (b, s_out, heads, _HEAD_DIM)
    q, k, v = (torch.empty(shape, device=qkv.device, dtype=qkv.dtype) for _ in range(3))
    weights = [(w_q, w_k)] if txt_qkv is None else [(w_txt_q, w_txt_k), (w_q, w_k)]
    for (src, off), (wq, wk) in zip(_segments(qkv, txt_qkv), weights):
        n_tok = src.shape[0] * src.shape[1]
        grid = (triton_cdiv(n_tok, _FWD_TOKENS), triton_cdiv(heads, _BLOCK_H))
        fwd[grid](
            src,
            q,
            k,
            v,
            wq.contiguous(),
            wk.contiguous(),
            cos,
            sin,
            n_tok,
            src.shape[1],
            s_out,
            off,
            eps,
            src.stride(1),
            H=heads,
            D=_HEAD_DIM,
            BLOCK_H=_BLOCK_H,
            TOKENS=_FWD_TOKENS,
            num_warps=4,
        )
    return q, k, v


@qk_norm_rope.register_fake
def _(qkv, txt_qkv, w_q, w_k, w_txt_q, w_txt_k, cos, sin, eps, heads, bias=None, txt_bias=None):
    s_out = qkv.shape[1] + (0 if txt_qkv is None else txt_qkv.shape[1])
    shape = (qkv.shape[0], s_out, heads, _HEAD_DIM)
    return qkv.new_empty(shape), qkv.new_empty(shape), qkv.new_empty(shape)


@torch.library.custom_op("primus_flux::qk_norm_rope_backward", mutates_args=(), device_types="cuda")
def qk_norm_rope_backward(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    qkv: torch.Tensor,
    txt_qkv: Optional[torch.Tensor],
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_txt_q: Optional[torch.Tensor],
    w_txt_k: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    heads: int,
    bias: bool = False,
) -> list[torch.Tensor]:
    """[d_qkv, (d_txt_qkv,) dw_q, dw_k, (dw_txt_q, dw_txt_k)] and, with ``bias``, the column sums of
    d_qkv (and d_txt_qkv) at the end."""
    _, bwd, colsum = _get_kernels()
    if _state["flydsl_bwd"]:
        from primus.backends.automodel.models.flux import qk_norm_rope_flydsl

        if qk_norm_rope_flydsl.supported(qkv, txt_qkv, cos, heads):
            return qk_norm_rope_flydsl.backward(
                dq, dk, dv, qkv, txt_qkv, w_q, w_k, w_txt_q, w_txt_k, cos, sin, eps, heads, bias, colsum
            )
    dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()
    qkv = qkv.contiguous()
    txt_qkv = None if txt_qkv is None else txt_qkv.contiguous()
    cos, sin = cos.contiguous(), sin.contiguous()
    s_out = dq.shape[1]
    weights = [(w_q, w_k)] if txt_qkv is None else [(w_txt_q, w_txt_k), (w_q, w_k)]
    grads, wgrads, bgrads = [], [], []
    n_h = triton_cdiv(heads, _BLOCK_H)
    for (src, off), (wq, wk) in zip(_segments(qkv, txt_qkv), weights):
        n_tok = src.shape[0] * src.shape[1]
        per = triton_cdiv(n_tok, min(_BWD_PROGRAMS_BIAS if bias else _BWD_PROGRAMS, n_tok))
        n_prog = triton_cdiv(n_tok, per)
        dsrc = torch.empty_like(src)
        part = torch.empty((n_prog * n_h, 2, _HEAD_DIM), device=src.device, dtype=torch.float32)
        bpart = torch.empty((n_prog, src.shape[-1]) if bias else (1,), device=src.device, dtype=torch.float32)
        bwd[(n_prog, n_h)](
            src,
            dsrc,
            dq,
            dk,
            dv,
            wq.contiguous(),
            wk.contiguous(),
            cos,
            sin,
            part,
            bpart,
            n_tok,
            src.shape[1],
            s_out,
            off,
            eps,
            per,
            src.stride(1),
            H=heads,
            D=_HEAD_DIM,
            BLOCK_H=_BLOCK_H,
            BIAS=bias,
            num_warps=4,
        )
        dw = part.sum(0)
        grads.append(dsrc)
        wgrads.extend([dw[0].to(wq.dtype), dw[1].to(wk.dtype)])
        if bias:
            n_cols = src.shape[-1]
            db = torch.empty(n_cols, device=src.device, dtype=src.dtype)
            colsum[(triton_cdiv(n_cols, _COLSUM_BLOCK_C),)](
                bpart, db, n_prog, n_cols, BLOCK_R=_COLSUM_BLOCK_R, BLOCK_C=_COLSUM_BLOCK_C, num_warps=4
            )
            bgrads.append(db)
    if txt_qkv is None:
        return [grads[0], wgrads[0], wgrads[1]] + bgrads
    return [grads[1], grads[0], wgrads[2], wgrads[3], wgrads[0], wgrads[1]] + bgrads[::-1]


@qk_norm_rope_backward.register_fake
def _(dq, dk, dv, qkv, txt_qkv, w_q, w_k, w_txt_q, w_txt_k, cos, sin, eps, heads, bias=False):
    out = [torch.empty_like(qkv)]
    if txt_qkv is not None:
        out.append(torch.empty_like(txt_qkv))
    out += [torch.empty_like(w_q), torch.empty_like(w_k)]
    if txt_qkv is not None:
        out += [torch.empty_like(w_txt_q), torch.empty_like(w_txt_k)]
    if bias:
        out.append(qkv.new_empty((qkv.shape[-1],)))
        if txt_qkv is not None:
            out.append(txt_qkv.new_empty((txt_qkv.shape[-1],)))
    return out


def _setup_context(ctx, inputs, output):
    qkv, txt_qkv, w_q, w_k, w_txt_q, w_txt_k, cos, sin, eps, heads, bias, txt_bias = inputs
    ctx.has_txt = txt_qkv is not None
    ctx.has_bias = bias is not None
    if ctx.has_txt and (txt_bias is not None) != ctx.has_bias:
        raise ValueError(
            "qk_norm_rope: a dual-stream call takes both stream biases or neither, as the "
            "processor fuses their gradients together"
        )
    ctx.bias_dtypes = (bias.dtype, None if txt_bias is None else txt_bias.dtype) if ctx.has_bias else None
    ctx.eps, ctx.heads = eps, heads
    saved = [qkv, w_q, w_k, cos, sin]
    if ctx.has_txt:
        saved += [txt_qkv, w_txt_q, w_txt_k]
    ctx.save_for_backward(*saved)


def _backward(ctx, dq, dk, dv):
    if ctx.has_txt:
        qkv, w_q, w_k, cos, sin, txt_qkv, w_txt_q, w_txt_k = ctx.saved_tensors
        out = qk_norm_rope_backward(
            dq, dk, dv, qkv, txt_qkv, w_q, w_k, w_txt_q, w_txt_k, cos, sin, ctx.eps, ctx.heads, ctx.has_bias
        )
        d_qkv, d_txt, dwq, dwk, dwtq, dwtk = out[:6]
        db, db_txt = (
            (out[6].to(ctx.bias_dtypes[0]), out[7].to(ctx.bias_dtypes[1])) if ctx.has_bias else (None, None)
        )
        return d_qkv, d_txt, dwq, dwk, dwtq, dwtk, None, None, None, None, db, db_txt
    qkv, w_q, w_k, cos, sin = ctx.saved_tensors
    out = qk_norm_rope_backward(
        dq, dk, dv, qkv, None, w_q, w_k, None, None, cos, sin, ctx.eps, ctx.heads, ctx.has_bias
    )
    db = out[3].to(ctx.bias_dtypes[0]) if ctx.has_bias else None
    return out[0], None, out[1], out[2], None, None, None, None, None, None, db, None


qk_norm_rope.register_autograd(_backward, setup_context=_setup_context)


def triton_cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _norm_eps(norms) -> Optional[float]:
    """The common eps of affine torch.nn.RMSNorm modules over head_dim, else None."""
    eps = set()
    for n in norms:
        if (
            not isinstance(n, torch.nn.RMSNorm)
            or n.weight is None
            or tuple(n.normalized_shape) != (_HEAD_DIM,)
        ):
            return None
        eps.add(n.eps)
    return eps.pop() if len(eps) == 1 and None not in eps else None


def _plain_linear(m) -> bool:
    """A stock torch.nn.Linear with a bias and no hooks, which F.linear computes exactly."""
    return (
        type(m) is torch.nn.Linear
        and m.bias is not None
        and not m._forward_hooks
        and not m._forward_pre_hooks
    )


def _fusable(attn, hidden_states, encoder_hidden_states, image_rotary_emb) -> Optional[float]:
    """eps for the op if this call is one it computes exactly as diffusers does, else None."""
    if not getattr(attn, "fused_projections", False) or attn.head_dim != _HEAD_DIM:
        return None
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_cuda:
        return None
    if not isinstance(image_rotary_emb, (tuple, list)) or len(image_rotary_emb) != 2:
        return None
    cos, sin = image_rotary_emb
    # FSDP2's mixed precision casts the block inputs, these tables included, to BF16.
    if (
        cos.dtype not in (torch.float32, torch.bfloat16)
        or sin.dtype != cos.dtype
        or cos.shape != sin.shape
        or cos.shape[-1] != _HEAD_DIM
    ):
        return None
    dual = attn.added_kv_proj_dim is not None
    if dual != (encoder_hidden_states is not None) or (dual and not hasattr(attn, "to_added_qkv")):
        return None
    norms = [attn.norm_q, attn.norm_k] + ([attn.norm_added_q, attn.norm_added_k] if dual else [])
    return _norm_eps(norms)


def _install_processor() -> bool:
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor

    orig = FluxAttnProcessor.__call__
    digest = hashlib.sha256(inspect.getsource(orig).encode()).hexdigest()
    if digest != _PROCESSOR_SHA256:
        logger.warning(
            "[PrimusFluxQkNormRope] FluxAttnProcessor.__call__ is not diffusers 0.40.0's; not applied."
        )
        return False

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None
    ):
        eps = _fusable(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
        if eps is None:
            return orig(self, attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
        # diffusers 0.40.0's forward with the q/k norms, the stream concat and RoPE in one op.
        cos, sin = image_rotary_emb
        dual = encoder_hidden_states is not None
        projections = [attn.to_qkv] + ([attn.to_added_qkv] if dual else [])
        if _state["bias_grad"] and all(_plain_linear(p) for p in projections):
            # The op returns the bias gradients from its backward kernel, so the GEMMs add the
            # biases without differentiating them.
            qkv = F.linear(hidden_states, attn.to_qkv.weight, attn.to_qkv.bias.detach())
            txt_qkv = None
            if dual:
                txt_qkv = F.linear(
                    encoder_hidden_states, attn.to_added_qkv.weight, attn.to_added_qkv.bias.detach()
                )
            biases = (attn.to_qkv.bias, attn.to_added_qkv.bias if dual else None)
        else:
            qkv = attn.to_qkv(hidden_states)
            txt_qkv = attn.to_added_qkv(encoder_hidden_states) if dual else None
            biases = (None, None)
        heads = qkv.shape[-1] // (3 * attn.head_dim)
        if dual:
            query, key, value = qk_norm_rope(
                qkv,
                txt_qkv,
                attn.norm_q.weight,
                attn.norm_k.weight,
                attn.norm_added_q.weight,
                attn.norm_added_k.weight,
                cos,
                sin,
                eps,
                heads,
                *biases,
            )
        else:
            query, key, value = qk_norm_rope(
                qkv, None, attn.norm_q.weight, attn.norm_k.weight, None, None, cos, sin, eps, heads, *biases
            )

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

    __call__.__wrapped__ = orig
    FluxAttnProcessor.__call__ = __call__
    return True


def install() -> bool:
    """Route FLUX attention's q/k norm + RoPE through the fused op. Idempotent; edits no diffusers source."""
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor

    if getattr(FluxAttnProcessor, _PATCHED_ATTR, False):
        return True
    if not _install_processor():
        return False
    _state["bias_grad"] = options.flag(_BIAS_FLAG)
    _state["flydsl_bwd"] = options.flag(_FLYDSL_FLAG)
    setattr(FluxAttnProcessor, _PATCHED_ATTR, True)
    logger.info(
        "[PrimusFluxQkNormRope] FLUX attention's q/k RMSNorm, stream concat and RoPE run as one op "
        "each way (fused QKV projections only)"
        f"{'; QKV bias gradients in its backward kernel' if _state['bias_grad'] else ''}"
        f"{'; FlyDSL backward kernel' if _state['flydsl_bwd'] else ''}."
    )
    return True
