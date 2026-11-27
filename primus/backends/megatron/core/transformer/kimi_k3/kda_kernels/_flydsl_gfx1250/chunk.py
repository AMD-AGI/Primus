###############################################################################
# Some parts of this code are adapted from flash-linear-attention
# (https://github.com/fla-org/flash-linear-attention, 0.5.2): fla/ops/kda/chunk.py, chunk_fwd.py
# and chunk_bwd.py (ChunkKDAFunction, chunk_kda_fwd, chunk_kda_bwd): the order of the steps and the
# tensors each one hands to the next.
# The steps here are FlyDSL kernels for gfx1250; nothing is imported from fla.
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li. Licensed under the MIT License.
#
# Modification Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""The ``flydsl`` KDA chunk kernel on gfx1250.

The chunked delta rule as fla's ``chunk_kda`` evaluates it with ``safe_gate`` and ``disable_recompute``, every
step on a gfx1250 FlyDSL kernel:

forward
    ``gc = chunk-local cumsum(g) / ln 2`` (:func:`chunk_local_cumsum`); :func:`kda_fwd_intra` (``Aqk``, ``Akk = (I + L)^-1``, ``w``, ``u``,
    ``qg``, ``kg``); :func:`kda_fwd_h` (chunk states ``h`` and ``v_new``); :func:`kda_fwd_o`.
backward (nothing recomputed: ``w``, ``qg``, ``kg``, ``v_new``, ``h`` are kept from the forward)
    :func:`kda_bwd_dav` (``dAqk``, ``dv``); :func:`kda_bwd_dhu` (``dh``, ``dv``); :func:`wy_dqkg_fused`;
    :func:`kda_bwd_intra`; ``dg`` = reverse chunk-local cumsum.

``safe_gate`` means every ``g`` lies in ``[lower_bound, 0)`` with ``-5 <= lower_bound < 0`` (Kimi K3's bounded
gate); the caller checks the bound. Shapes: ``q, k, v, g`` ``[B, T, H, 128]`` with ``T % 64 == 0``, ``beta``
``[B, T, H]``; ``q / k / v`` bf16, ``g / beta`` fp32. No initial or final state, no packed sequences, no context
parallelism.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from .bwd_dav_kernel import kda_bwd_dav
from .bwd_dhu_kernel import kda_bwd_dhu
from .bwd_intra_kernel import kda_bwd_intra
from .bwd_wy_dqkg_kernel import wy_dqkg_fused
from .cumsum_kernel import chunk_local_cumsum
from .fwd_h_kernel import kda_fwd_h
from .fwd_intra_kernel import kda_fwd_intra
from .fwd_o_kernel import kda_fwd_o

__all__ = ["flydsl_chunk_kda", "chunk_kda_unsupported_reason"]

_BT = 64
_D = 128
_RCP_LN2 = 1.0 / math.log(2.0)
# One buffer view addresses at most 2 GiB; g is the largest tensor a kernel views (fp32 [B, T, H, 128]).
_MAX_VIEW_BYTES = (1 << 31) - 4096


def chunk_kda_unsupported_reason(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor) -> Optional[str]:
    """``None`` when :func:`flydsl_chunk_kda` runs these inputs, else why not."""
    if q.dim() != 4 or q.shape != k.shape or v.shape != k.shape or g.shape != k.shape:
        return f"q / k / v / g must share one [B, T, H, {_D}] shape; got {[tuple(t.shape) for t in (q, k, v, g)]}"
    B, T, H, K = k.shape
    if K != _D or T % _BT:
        return f"needs head dim {_D} and T a multiple of {_BT}; got K={K}, T={T}"
    if beta.shape != (B, T, H):
        return f"beta must be [B, T, H] = {(B, T, H)}; got {tuple(beta.shape)}"
    if (
        any(t.dtype != torch.bfloat16 for t in (q, k, v))
        or g.dtype != torch.float32
        or beta.dtype != torch.float32
    ):
        return "dtypes must be bf16 q / k / v and fp32 g / beta"
    if g.numel() * 4 > _MAX_VIEW_BYTES:
        return "g is larger than one 2 GiB buffer view"
    if not k.is_cuda:
        return "CPU tensors"
    arch = str(getattr(torch.cuda.get_device_properties(k.device), "gcnArchName", ""))
    if not arch.startswith("gfx1250"):
        return f"device arch is {arch!r}, not gfx1250"
    return None


class _ChunkKDAFlydsl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, scale: float):
        gc = chunk_local_cumsum(g, scale=_RCP_LN2)
        w, u, qg, kg, Aqk, Akk = kda_fwd_intra(q, k, v, gc, beta, scale)
        h, v_new, _ = kda_fwd_h(kg, w, u, gc)
        del u  # the backward needs v_new, not u
        o = kda_fwd_o(q, v_new, gc, Aqk, h, scale)
        ctx.save_for_backward(q, k, v, gc, beta, Aqk, Akk, w, qg, kg, v_new, h)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do: Tensor):
        q, k, v, gc, beta, Aqk, Akk, w, qg, kg, v_new, h = ctx.saved_tensors
        scale = ctx.scale
        do = do.contiguous()
        dAqk, dv = kda_bwd_dav(v_new, do, Aqk, scale)
        dh, _, dv = kda_bwd_dhu(qg, kg, w, do, dv, gc, scale)
        dq, dk, dv, db, dg, dAkk = wy_dqkg_fused(q, k, v, v_new, gc, beta, Akk, h, do, dh, dv, scale)
        dq, dk, db, dg = kda_bwd_intra(q, k, gc, beta, dAqk, dAkk, dq, dk, db, dg)
        dg = chunk_local_cumsum(dg, reverse=True)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg, db, None


def flydsl_chunk_kda(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
    cu_seqlens: Optional[Tensor] = None,
    cp_context: Optional[object] = None,
) -> Tuple[Tensor, None]:
    """KDA through the gfx1250 FlyDSL chunk kernels; signature-compatible with ``eager_chunk_kda``.

    ``g`` must already be the bounded log-decay (every value in ``[lower_bound, 0)``, ``-5 <= lower_bound < 0``)
    and ``beta`` already sigmoid-activated. Returns ``(o, None)``.
    """
    unsupported = []
    if initial_state is not None or output_final_state:
        unsupported.append("an initial or final state")
    if use_qk_l2norm_in_kernel:
        unsupported.append("use_qk_l2norm_in_kernel")
    if chunk_size != _BT:
        unsupported.append(f"chunk_size {chunk_size} (it tiles by {_BT})")
    if cu_seqlens is not None or cp_context is not None:
        unsupported.append("packed sequences / context parallelism")
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    why = chunk_kda_unsupported_reason(q, k, v, g, beta)
    if why is not None:
        unsupported.append(why)
    if unsupported:
        raise ValueError(
            "the gfx1250 FlyDSL KDA chunk kernel does not take "
            + "; ".join(unsupported)
            + ". Select kda_backend: fla."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return _ChunkKDAFlydsl.apply(q, k, v, g, beta, float(scale)), None
