###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused Gluon indexer forward score for gfx1250 (opt3 / design X2).

Replaces, in the forward pass only, the current default chain

.. code-block:: python

    dot = torch.einsum("bshd,bpd->bshp", q_i, k_icomp)        # hipBLASLt, writes [B,S,H,P]
    scores = indexer_score_post_triton(dot, w_i, ...)         # relu * w, sum_h, causal -inf

with one kernel that never materialises ``dot``:

* grid = (cdiv(S, BM) * cdiv(P, BN), B), 8 waves per CTA, BM = BN = 128;
* the k tile ``[BN, D]`` is TDM-loaded once into a partitioned LDS buffer and
  then kept in registers as the WMMA B operand for all heads;
* the per-head q tile ``[BM, D]`` (rows s, columns h*D:(h+1)*D of the
  ``[B*S, H*D]`` view) streams through an ``NQ``-slot TDM ring (prefetch
  distance NQ-1);
* per head: ``acc = q_h @ k^T`` (WMMA, fp32), rounded to bf16 exactly like
  the einsum output that the post kernel reads, then
  ``out += w[s, h] * relu(acc)`` in fp32;
* epilogue: ``out = where((p + 1) * ratio - 1 <= s + q_offset, out, -inf)``
  stored as ``out_dtype``. Tiles that are fully masked skip the head loop
  (no TDM traffic, no WMMA) and store -inf only.

Backward stays on the existing path: the existing ``indexer_score_post``
backward kernels produce ``d_dot`` / ``d_w`` from the bf16 ``dot`` and the two
einsum-backward GEMMs produce ``dq`` / ``dk``. By default
(``PRIMUS_INDEXER_GLUON_SAVE_DOT=1``) the kernel itself writes that ``dot``
(zeros in dead tiles), so no forward GEMM runs anywhere; with ``=0`` nothing
is saved and the backward recomputes ``dot`` with the einsum.

Gating: ``PRIMUS_INDEXER_GLUON=1`` (default OFF). Needs a Triton with the
CDNA5 Gluon API (``triton.experimental.gluon.language.amd.cdna5`` with
``tdm`` and ``make_partitioned_dot_layouts``) and a gfx1250 device.
"""

from __future__ import annotations

import functools
import os
import types
from typing import Optional

import torch

try:  # Gluon CDNA5 API is only in recent Triton (main / 3.9+).
    import triton
    from triton.experimental import gluon
    from triton.experimental import gluon as _gluon_mod  # noqa: F401
    import triton.experimental.gluon.language as ttgl
    from triton.experimental.gluon.language.amd.cdna5 import tdm as _tdm  # noqa: F401

    _HAS_GLUON_CDNA5 = hasattr(ttgl.amd.cdna5, "make_partitioned_dot_layouts")
    # The image's Triton 3.8 exposes the same API but did not finish compiling this kernel
    # for gfx1250 in >20 CPU-min (Triton main: ~1 min); require 3.9+ unless overridden.
    _TRITON_OK = tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 9) or         os.environ.get("PRIMUS_INDEXER_GLUON_ALLOW_OLD_TRITON", "0") == "1"
except Exception:  # pragma: no cover - depends on the installed Triton
    _HAS_GLUON_CDNA5 = False
    _TRITON_OK = False


NUM_WARPS = 8
_DEF_BM = 128
_DEF_BN = 128
_DEF_NQ = 3


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

if _HAS_GLUON_CDNA5:

    @gluon.jit
    def _xcd_remap(pid, total, NUM_XCDS: ttgl.constexpr):
        # Hardware dispatches consecutive pids round-robin over XCDs; give each
        # XCD a contiguous range of logical tiles so the p-tiles that share one
        # q row block (same s tile) land on the same L2.
        xcd = pid % NUM_XCDS
        pos = pid // NUM_XCDS
        base = total // NUM_XCDS
        extra = total % NUM_XCDS
        return xcd * base + min(xcd, extra) + pos

    @gluon.jit
    def _indexer_score_fwd_gluon_kernel(
        q_ptr,  # [B*S, H*D]  bf16/fp16, row-major contiguous
        k_ptr,  # [B*P, D]
        w_ptr,  # [B*S, H]    any float dtype (already scaled by score_scale)
        out_ptr,  # [B*S, P]  out dtype
        dot_ptr,  # [B*S, H, P] q.k in q dtype (only written when WRITE_DOT)
        S,
        P,
        QROWS,  # B*S
        KROWS,  # B*P
        q_offset,
        n_tiles_p,
        n_tiles,
        H: ttgl.constexpr,
        D: ttgl.constexpr,
        RATIO: ttgl.constexpr,
        BM: ttgl.constexpr,
        BN: ttgl.constexpr,
        NQ: ttgl.constexpr,
        ROUND_DOT: ttgl.constexpr,
        NUM_XCDS: ttgl.constexpr,
        WRITE_DOT: ttgl.constexpr,
        SHARED_Q: ttgl.constexpr,
        SHARED_K: ttgl.constexpr,
        WMMA: ttgl.constexpr,
    ):
        ttgl.static_assert(NQ >= 2, "NQ must be >= 2")
        ttgl.static_assert(H >= NQ - 1, "need H >= NQ - 1 heads")
        DOT_A: ttgl.constexpr = ttgl.DotOperandLayout(0, WMMA, 8)
        DOT_B: ttgl.constexpr = ttgl.DotOperandLayout(1, WMMA, 8)
        ROW_L: ttgl.constexpr = ttgl.SliceLayout(1, WMMA)
        COL_L: ttgl.constexpr = ttgl.SliceLayout(0, WMMA)
        # 16-byte row-contiguous store layout for the optional dot write.
        TPW_N: ttgl.constexpr = BN // 8
        DOT_ST: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [32 // TPW_N, TPW_N], [8, 1], [1, 0])

        pid = ttgl.program_id(0)
        b = ttgl.program_id(1)
        if NUM_XCDS > 1:
            pid = _xcd_remap(pid, n_tiles, NUM_XCDS)
        pid_s = pid // n_tiles_p
        pid_p = pid % n_tiles_p
        s0 = pid_s * BM
        p0 = pid_p * BN

        # Causal tile skip: the tile has a live entry iff its smallest p is
        # visible to its largest (valid) s.
        s_last = min(s0 + BM, S) - 1
        live = (p0 + 1) * RATIO - 1 <= s_last + q_offset
        n_iter = live.to(ttgl.int32) * H

        q_desc = _tdm.make_tensor_descriptor(base=q_ptr, shape=(QROWS, H * D), strides=(H * D, 1),
                                             block_shape=(BM, D), layout=SHARED_Q)
        k_desc = _tdm.make_tensor_descriptor(base=k_ptr, shape=(KROWS, D), strides=(D, 1), block_shape=(BN, D),
                                             layout=SHARED_K)
        q_buf = ttgl.allocate_shared_memory(q_ptr.type.element_ty, [NQ, BM, D], SHARED_Q)
        k_buf = ttgl.allocate_shared_memory(k_ptr.type.element_ty, [BN, D], SHARED_K)

        q_row = b * S + s0
        k_row = b * P + p0
        # Prologue: k tile once + q heads 0 .. NQ-2 (dead tiles issue nothing).
        _tdm.async_load(k_desc, [k_row, 0], k_buf, pred=live)
        for i in ttgl.static_range(NQ - 1):
            _tdm.async_load(q_desc, [q_row, i * D], q_buf.index(i), pred=live)
        _tdm.async_wait(NQ - 1)  # k landed (this wave's part) ...
        ttgl.barrier()  # ... and every other wave's part
        kb = k_buf.permute([1, 0]).load(layout=DOT_B)  # [D, BN], lives in VGPRs for all heads

        rows = s0 + ttgl.arange(0, BM, layout=ROW_L)
        cols = p0 + ttgl.arange(0, BN, layout=COL_L)
        row_ok = rows < S
        # Rows past S (last s tile) read row S-1's weights instead of masking: no
        # exec-mask branches in the loop, and those rows are never stored.
        w_row_ptr = w_ptr + (b * S + ttgl.minimum(rows, S - 1)).to(ttgl.int64) * H

        out = ttgl.zeros((BM, BN), dtype=ttgl.float32, layout=WMMA)
        zero = ttgl.zeros((BM, BN), dtype=ttgl.float32, layout=WMMA)
        w_nxt = ttgl.load(w_row_ptr)  # head 0; head h+1 is prefetched during head h
        if WRITE_DOT:
            d_rows = s0 + ttgl.arange(0, BM, layout=ttgl.SliceLayout(1, DOT_ST))
            d_cols = p0 + ttgl.arange(0, BN, layout=ttgl.SliceLayout(0, DOT_ST))
            d_row_off = (b * S + d_rows).to(ttgl.int64)[:, None] * (H * P) + d_cols[None, :]
            d_mask = (d_rows < S)[:, None] & (d_cols < P)[None, :]
        for h in range(n_iter):
            _tdm.async_wait(NQ - 2)  # q_h landed for this wave
            ttgl.barrier()  # all waves: q_h visible, slot (h-1)%NQ no longer read
            hn = min(h + NQ - 1, H - 1)  # clamped refill keeps the wait count uniform
            _tdm.async_load(q_desc, [q_row, hn * D], q_buf.index((h + NQ - 1) % NQ))
            w_h = w_nxt.to(ttgl.float32)
            w_nxt = ttgl.load(w_row_ptr + min(h + 1, H - 1))
            a = q_buf.index(h % NQ).load(layout=DOT_A)
            acc = ttgl.amd.cdna5.wmma(a, kb, zero)
            if ROUND_DOT or WRITE_DOT:  # the einsum hands the post kernel a bf16/fp16 dot
                acc_lp = acc.to(q_ptr.type.element_ty)
                if WRITE_DOT:  # saved for the existing tail backward (replaces the einsum output)
                    ttgl.store(dot_ptr + (d_row_off + h * P), ttgl.convert_layout(acc_lp, DOT_ST), mask=d_mask)
                if ROUND_DOT:
                    acc = acc_lp.to(ttgl.float32)
            out = ttgl.fma(ttgl.maximum(acc, 0.0), w_h[:, None], out)
        if WRITE_DOT:
            # Dead tiles: the tail backward reads relu(dot) * 0 here, so the bytes
            # must be finite -> write zeros (store only, no GEMM).
            zlp = ttgl.zeros((BM, BN), dtype=q_ptr.type.element_ty, layout=DOT_ST)
            for h in range(H - n_iter):
                ttgl.store(dot_ptr + (d_row_off + h * P), zlp, mask=d_mask)
        _tdm.async_wait(0)

        allowed = (cols[None, :] + 1) * RATIO - 1 <= rows[:, None] + q_offset
        res = ttgl.where(allowed, out, float("-inf"))
        offs = (b * S + rows).to(ttgl.int64)[:, None] * P + cols[None, :]
        ttgl.store(out_ptr + offs, res.to(out_ptr.type.element_ty), mask=row_ok[:, None] & (cols < P)[None, :])

    @functools.lru_cache(maxsize=None)
    def _layouts(BM: int, BN: int, D: int):
        pad_q = ttgl.PaddedSharedLayout.with_identity_for([[D, 8]], [BM, D], [1, 0])
        pad_k = ttgl.PaddedSharedLayout.with_identity_for([[D, 8]], [BN, D], [1, 0])
        mk = ttgl.amd.cdna5.make_partitioned_dot_layouts
        try:
            return mk(BM, BN, pad_q, pad_k, NUM_WARPS, [16, 16, 32], a_transposed=False, b_transposed=True,
                      transposed=True)
        except TypeError:  # Triton 3.8 (image): no `transposed` kwarg
            return mk(BM, BN, pad_q, pad_k, NUM_WARPS, [16, 16, 32], a_transposed=False, b_transposed=True)


def _num_xcds() -> int:
    return max(1, int(os.environ.get("PRIMUS_INDEXER_GLUON_XCDS", "8")))


def launch_args(B, S, P, H, D, compress_ratio, q_offset, BM=_DEF_BM, BN=_DEF_BN, NQ=_DEF_NQ, round_dot=True,
                num_xcds=None, write_dot=False):
    """Grid + scalar/constexpr kwargs (shared by the runtime launch and compile-only checks)."""
    shared_q, shared_k, wmma = _layouts(BM, BN, D)
    n_tiles_p = -(-P // BN)
    n_tiles = -(-S // BM) * n_tiles_p
    grid = (n_tiles, B)
    kw = dict(S=S, P=P, QROWS=B * S, KROWS=B * P, q_offset=int(q_offset), n_tiles_p=n_tiles_p, n_tiles=n_tiles, H=H,
              D=D, RATIO=int(compress_ratio), BM=BM, BN=BN, NQ=NQ, ROUND_DOT=bool(round_dot),
              NUM_XCDS=_num_xcds() if num_xcds is None else num_xcds, WRITE_DOT=bool(write_dot),
              SHARED_Q=shared_q, SHARED_K=shared_k, WMMA=wmma)
    return grid, kw


def _tile_cfg():
    bm = int(os.environ.get("PRIMUS_INDEXER_GLUON_BM", _DEF_BM))
    bn = int(os.environ.get("PRIMUS_INDEXER_GLUON_BN", _DEF_BN))
    nq = int(os.environ.get("PRIMUS_INDEXER_GLUON_NQ", _DEF_NQ))
    return bm, bn, nq


def indexer_score_fwd_gluon(q_i, k_icomp, w_i, *, compress_ratio: int, out_dtype: torch.dtype, q_offset: int = 0,
                            round_dot: bool = True, write_dot: bool = False):
    """Launch the fused kernel. q_i [B,S,H,D], k_icomp [B,P,D], w_i [B,S,H] -> scores [B,S,P].

    write_dot=True also returns ``dot [B,S,H,P]`` (q dtype, zeros in causally dead tiles),
    i.e. the tensor the einsum would have produced, for the existing tail backward.
    """
    B, S, H, D = q_i.shape
    P = k_icomp.shape[1]
    q = q_i.contiguous()
    k = k_icomp.contiguous()
    w = w_i.contiguous()
    out = torch.empty((B, S, P), dtype=out_dtype, device=q.device)
    dot = torch.empty((B, S, H, P), dtype=q.dtype, device=q.device) if write_dot else out
    BM, BN, NQ = _tile_cfg()
    grid, kw = launch_args(B, S, P, H, D, compress_ratio, q_offset, BM=BM, BN=BN, NQ=NQ, round_dot=round_dot,
                           write_dot=write_dot)
    _indexer_score_fwd_gluon_kernel[grid](q, k, w, out, dot, **kw, num_warps=NUM_WARPS, waves_per_eu=2)
    return (out, dot) if write_dot else out


# ---------------------------------------------------------------------------
# Torch emulation of the kernel's tiling (CPU reference / fallback)
# ---------------------------------------------------------------------------


def indexer_score_fwd_tiled_reference(q_i, k_icomp, w_i, *, compress_ratio: int, out_dtype: torch.dtype,
                                      q_offset: int = 0, BM: int = _DEF_BM, BN: int = _DEF_BN,
                                      round_dot: bool = True, return_live: bool = False, write_dot: bool = False):
    """Same tile loop, dead-tile skip, bf16 rounding, head order and epilogue as the kernel."""
    B, S, H, D = q_i.shape
    P = k_icomp.shape[1]
    out = torch.empty((B, S, P), dtype=out_dtype, device=q_i.device)
    dot = torch.zeros((B, S, H, P), dtype=q_i.dtype, device=q_i.device) if write_dot else None
    n_live = 0
    for b in range(B):
        for s0 in range(0, S, BM):
            s1 = min(s0 + BM, S)
            rows = torch.arange(s0, s1, device=q_i.device)
            for p0 in range(0, P, BN):
                p1 = min(p0 + BN, P)
                cols = torch.arange(p0, p1, device=q_i.device)
                acc_out = torch.zeros((s1 - s0, p1 - p0), dtype=torch.float32, device=q_i.device)
                live = (p0 + 1) * compress_ratio - 1 <= (s1 - 1) + q_offset
                if live:
                    n_live += 1
                    kt = k_icomp[b, p0:p1].float()
                    for h in range(H):
                        acc = q_i[b, s0:s1, h].float() @ kt.t()
                        if write_dot:
                            dot[b, s0:s1, h, p0:p1] = acc.to(q_i.dtype)
                        if round_dot:
                            acc = acc.to(q_i.dtype).float()
                        acc_out += acc.clamp_min(0.0) * w_i[b, s0:s1, h].float()[:, None]
                allowed = (cols[None, :] + 1) * compress_ratio - 1 <= rows[:, None] + q_offset
                out[b, s0:s1, p0:p1] = torch.where(allowed, acc_out, torch.full_like(acc_out,
                                                                                     float("-inf"))).to(out_dtype)
    res = (out, dot) if write_dot else out
    return (res, n_live) if return_live else res


# ---------------------------------------------------------------------------
# Backward helpers (existing path: recompute einsum, post bwd, einsum bwd)
# ---------------------------------------------------------------------------


def _post_bwd(dot, w, d_scores, compress_ratio: int, q_offset: int):
    """d_dot, d_w of ``where(mask, sum_h w * relu(dot), -inf)``; masked entries get no gradient."""
    if dot.is_cuda and q_offset == 0:
        from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_score_post import (
            IndexerScorePostFn,
            is_triton_kernel_supported,
        )

        if is_triton_kernel_supported(dot, w):
            ctx = types.SimpleNamespace(saved_tensors=(dot.contiguous(), w.contiguous()),
                                        compress_ratio=int(compress_ratio), shape=tuple(dot.shape),
                                        in_dtypes=(dot.dtype, w.dtype))
            d_dot, d_w, _, _ = IndexerScorePostFn.backward(ctx, d_scores)
            return d_dot, d_w
    B, S, H, P = dot.shape
    s = torch.arange(S, device=dot.device)[:, None] + q_offset
    p = torch.arange(P, device=dot.device)[None, :]
    allowed = (p + 1) * compress_ratio - 1 <= s  # [S, P]
    d_acc = torch.where(allowed, d_scores.float(), torch.zeros((), device=dot.device))  # [B, S, P]
    dotf = dot.float()
    d_dot = torch.where(dotf > 0, d_acc[:, :, None, :] * w.float()[..., None], torch.zeros((), device=dot.device))
    d_w = (d_acc[:, :, None, :] * dotf.clamp_min(0.0)).sum(-1)
    return d_dot.to(dot.dtype), d_w.to(w.dtype)


def _save_dot_enabled() -> bool:
    """PRIMUS_INDEXER_GLUON_SAVE_DOT (default 1): the kernel also writes the bf16 dot for the
    backward (same memory as today's einsum output, no extra GEMM). 0: save nothing and recompute
    the einsum in the backward (saves B*S*H*P*2 bytes of activation memory per layer)."""
    return os.environ.get("PRIMUS_INDEXER_GLUON_SAVE_DOT", "1") != "0"


class IndexerScoreGluonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_i, k_icomp, w_i, compress_ratio: int, out_dtype: torch.dtype, q_offset: int):  # type: ignore
        q = q_i.contiguous()
        k = k_icomp.contiguous()
        w = w_i.contiguous()
        need_grad = any(ctx.needs_input_grad[:3])
        save_dot = need_grad and _save_dot_enabled()
        if q.is_cuda:
            res = indexer_score_fwd_gluon(q, k, w, compress_ratio=compress_ratio, out_dtype=out_dtype,
                                          q_offset=q_offset, write_dot=save_dot)
        else:
            res = indexer_score_fwd_tiled_reference(q, k, w, compress_ratio=compress_ratio, out_dtype=out_dtype,
                                                    q_offset=q_offset, write_dot=save_dot)
        scores, dot = res if save_dot else (res, None)
        if need_grad:
            if save_dot:
                ctx.save_for_backward(q, k, w, dot)
            else:
                ctx.save_for_backward(q, k, w)
        ctx.save_dot = save_dot
        ctx.compress_ratio = int(compress_ratio)
        ctx.q_offset = int(q_offset)
        return scores

    @staticmethod
    def backward(ctx, d_scores):  # type: ignore[override]
        need_q, need_k, need_w = ctx.needs_input_grad[:3]
        if ctx.save_dot:
            # Existing path without the forward einsum: tail bwd on the kernel-written dot,
            # then the two einsum-backward GEMMs.
            q, k, w, dot = ctx.saved_tensors
            d_dot, d_w = _post_bwd(dot, w, d_scores.contiguous(), ctx.compress_ratio, ctx.q_offset)
            dq = torch.einsum("bshp,bpd->bshd", d_dot, k) if need_q else None
            dk = torch.einsum("bshp,bshd->bpd", d_dot, q) if need_k else None
            return dq, dk, (d_w if need_w else None), None, None, None
        q, k, w = ctx.saved_tensors
        with torch.enable_grad():
            q_ = q.detach().requires_grad_(need_q)
            k_ = k.detach().requires_grad_(need_k)
            dot = torch.einsum("bshd,bpd->bshp", q_, k_)  # recompute (same op as the existing forward)
        d_dot, d_w = _post_bwd(dot.detach(), w, d_scores.contiguous(), ctx.compress_ratio, ctx.q_offset)
        del_inputs = [t for t, n in ((q_, need_q), (k_, need_k)) if n]
        grads = torch.autograd.grad(dot, del_inputs, d_dot) if del_inputs else ()
        it = iter(grads)
        dq = next(it) if need_q else None
        dk = next(it) if need_k else None
        return dq, dk, (d_w if need_w else None), None, None, None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def is_gluon_path_enabled() -> bool:
    """``PRIMUS_INDEXER_GLUON=1`` turns the fused Gluon forward on (default off)."""
    return os.environ.get("PRIMUS_INDEXER_GLUON", "0") == "1"


@functools.lru_cache(maxsize=None)
def _device_is_gfx1250(index: int) -> bool:
    try:
        return "gfx1250" in torch.cuda.get_device_properties(index).gcnArchName
    except Exception:
        return False


def is_gluon_kernel_supported(q_i: torch.Tensor, k_icomp: torch.Tensor, w_i: torch.Tensor) -> bool:
    if not _HAS_GLUON_CDNA5 or not _TRITON_OK or not q_i.is_cuda:
        return False
    if q_i.dim() != 4 or k_icomp.dim() != 3 or w_i.dim() != 3:
        return False
    B, S, H, D = q_i.shape
    if D not in (64, 128) or H < _DEF_NQ - 1:
        return False
    if q_i.dtype not in (torch.bfloat16, torch.float16) or k_icomp.dtype != q_i.dtype:
        return False
    if B * S * H * D >= 2**31 or B * k_icomp.shape[1] * D >= 2**31:
        return False
    return _device_is_gfx1250(q_i.device.index or 0)


def indexer_score_gluon(q_i, k_icomp, w_i, *, compress_ratio: int, out_dtype: torch.dtype,
                        q_offset: Optional[int] = 0) -> torch.Tensor:
    """Masked indexer scores ``[B, S, P]`` (``-inf`` where causally masked)."""
    return IndexerScoreGluonFn.apply(q_i, k_icomp, w_i, int(compress_ratio), out_dtype, int(q_offset or 0))


__all__ = [
    "IndexerScoreGluonFn",
    "indexer_score_gluon",
    "indexer_score_fwd_tiled_reference",
    "is_gluon_path_enabled",
    "is_gluon_kernel_supported",
    "launch_args",
]
