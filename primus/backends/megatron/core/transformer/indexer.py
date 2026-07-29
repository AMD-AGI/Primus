###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

r"""
DeepSeek-V4 Indexer (sparse position selector for CSA).

Reference: techblog §1.4 ("Indexer: CSA's Sparse Selector").

The Indexer is **only** used by CSA layers (``compress_ratio == 4``). For
each query position ``t`` it picks ``index_topk`` compressed-KV positions
``s`` (out of all compressed positions ``[0, P)`` where ``P = S // ratio``).

Math (from the techblog):

.. math::

    q^Q_t = h_t W^{DQ},\quad q^I_{t,h} = q^Q_t W^{IUQ}_h,\quad
    w^I_{t,h} = h_t W^w_h

.. math::

    I_{t,s} = \\sum_h w^I_{t,h}\\cdot \\mathrm{ReLU}(q^I_{t,h}\\cdot K^{IComp}_s)

.. math::

    \\mathrm{topk\\_idxs}_t = \\mathrm{argTopK}_s\\,I_{t,s}

The Indexer carries its **own** mini-Compressor (``index_head_dim``,
``index_n_heads``); the ``K^{IComp}`` it produces is independent of the
main attention's compressed KV pool. It is only used to **select** top-k
positions; the actual values fetched into main attention come from the
main Compressor in the surrounding CSA layer.

Both scoring operands are prepared the way the reference prepares them
before the dot product:

* **Partial RoPE**, at the compressed-branch base -- queries at their own
  token positions, compressed keys at ``s * compress_ratio`` (the window's
  first token), so the two live in one coordinate system.
* **Hadamard rotation** (``rotate_activation``) on top. Orthogonal, so it
  leaves the inner product alone in exact arithmetic; what it buys is
  spreading each coordinate's energy across the vector so no single channel
  dominates the low-precision QK product. Note the reference rotates *only*
  here -- the main compressed KV pool is built with ``rotate=False``.

Phase 4 contract:
* Plain ``nn.Linear`` projections (TP integration in P6).
* Causal masking: positions ``s`` whose start raw-token index exceeds the
  query's raw-token index get a value of ``-inf`` so they cannot be
  selected. Out-of-range positions are returned as ``-1`` in the output
  ``topk_idxs`` so the caller can treat them as "no key".
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _is_rank0() -> bool:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        pass
    return True


from primus.backends.megatron.core.transformer.compressor import Compressor
from primus.backends.megatron.core.transformer.dual_rope import (
    apply_interleaved_partial_rope,
)
from primus.backends.megatron.core.transformer.hadamard_rotation import rotate_activation

# E4M3 finite max magnitude (float8_e4m3fn): largest representable value.
_FP8_E4M3_MAX = 448.0


def fake_quantize_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor dynamic FP8 (E4M3) fake-quantization.

    Scales ``x`` so its max magnitude maps to the E4M3 finite range, rounds
    through ``torch.float8_e4m3fn``, then dequantizes back to ``x.dtype``. This
    simulates the precision of an FP8 QK GEMM input while keeping the matmul
    itself in the activation dtype (QAT-style "simulated FP8" path). Returns
    ``x`` unchanged when the platform/torch build lacks ``float8_e4m3fn`` or
    when ``x`` is all-zero (degenerate scale).
    """
    if not hasattr(torch, "float8_e4m3fn"):
        return x
    orig_dtype = x.dtype
    amax = x.detach().abs().amax()
    if not torch.isfinite(amax) or float(amax) <= 0.0:
        return x
    scale = (_FP8_E4M3_MAX / amax).to(x.dtype)
    x_scaled = torch.clamp(x * scale, -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    return x_fp8.to(orig_dtype) / scale


from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_score import (
    indexer_score_triton,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_score import (
    is_triton_kernel_supported as _indexer_triton_full_supported,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_score import (
    is_triton_path_enabled as _indexer_triton_full_enabled,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_score_post import (
    indexer_score_post_triton,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_score_post import (
    is_triton_kernel_supported as _indexer_tail_triton_supported,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_score_post import (
    is_triton_path_enabled as _indexer_tail_triton_enabled,
)

# MXFP4 block size (E2M1 data + E8M0 per-32 block scales).
_MXFP4_BLOCK = 32


def _indexer_fp4_enabled() -> bool:
    """True iff PRIMUS_INDEXER_FP4 == "1" (default off): run the CSA-indexer QK in MXFP4."""
    return os.environ.get("PRIMUS_INDEXER_FP4", "0") == "1"


def _fp4_qk_gemm(q_i: torch.Tensor, k_icomp: torch.Tensor) -> torch.Tensor:
    """Real MXFP4 indexer QK: per-batch [S*H,Hd] @ [P,Hd]^T (NT, trans_b) -> [B,S,H,P].

    hipBLASLt FP4 needs K=Hd%128, M,N%16; force PRIMUS_TURBO_GEMM_BACKEND=FP4:HIPBLASLT.
    """
    import primus_turbo.pytorch as pt
    from primus_turbo.pytorch.core.low_precision import (
        Float4QuantConfig,
        Format,
        ScaleDtype,
        ScalingGranularity,
    )

    cfg = Float4QuantConfig(
        format=Format.E2M1_X2,
        granularity=ScalingGranularity.MX_BLOCKWISE,
        block_size=_MXFP4_BLOCK,
        scale_dtype=ScaleDtype.E8M0,
    )
    B, S, H, Hd = q_i.shape
    P = k_icomp.shape[1]
    outs = []
    for b in range(B):
        a = q_i[b].reshape(S * H, Hd).contiguous()  # [S*H, Hd]
        bk = k_icomp[b].contiguous()  # [P, Hd]
        o = pt.ops.gemm_fp4(a, bk, trans_b=True, config=cfg)  # [S*H, P]
        outs.append(o.view(1, S, H, P))
    return torch.cat(outs, dim=0)


def _indexer_fp8_proj_enabled() -> bool:
    """Run the indexer projections (w_dq/w_iuq/w_w) in MXFP8 (default off).

    Gated by its own knob, deliberately **not** by the attention-projection flag
    ``PRIMUS_V4_FP8_ATTN_PROJ``. The open-source reference wraps the indexer's
    weight projection in an fp8-disabled context so it stays high precision even
    when the enclosing layer runs FP8: the indexer decides *which* compressed KV
    entries each query reads, so error there changes the selection instead of
    perturbing a value. Quantizing the attention projections and quantizing the
    selector are separate decisions, so one no longer implies the other.

    Only fires inside turbo-fp8. The linears are duplicated (no TP shard), so
    fp8 is safe at any TP.
    """
    if os.environ.get("PRIMUS_V4_FP8_INDEXER_PROJ", "0") != "1":
        return False
    try:
        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboLowPrecisionGlobalStateManager as _M,
        )

        return _M.is_turbo_fp8_enabled()
    except Exception:
        return False


def _fp8_linear(lin: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """MXFP8 apply of an ``nn.Linear`` (weight [out,in], no bias): y = x @ Wᵀ."""
    import primus_turbo.pytorch as pt

    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboLowPrecisionGlobalStateManager as _M,
    )

    cfg = _M.get_turbo_quant_config().data()
    orig = x.shape
    x2 = x.reshape(-1, orig[-1]).contiguous()
    out = pt.ops.gemm_fp8(x2, lin.weight, trans_b=True, config=cfg)  # [*, out]
    return out.reshape(*orig[:-1], out.shape[-1])


class Indexer(nn.Module):
    """Sparse position selector for CSA.

    Args:
        hidden_size: input feature dim ``D`` (same as main attention).
        index_head_dim: head dim used by the mini-Compressor and the
            low-rank query projection.
        index_n_heads: number of indexer "heads".
        index_topk: number of compressed positions to select per query.
        compress_ratio: ratio ``m`` of the mini-Compressor (matches the
            main Compressor of the surrounding CSA layer; usually ``4``).
        dq_rank: rank of the shared low-rank query projection ``W^{DQ}``.
            Defaults to ``index_head_dim`` (the V4 reference doesn't expose
            a separate setting; ``W^{IUQ}_h`` then projects from ``dq_rank``
            to ``index_head_dim``).
        rope: the surrounding CSA layer's compressed-branch
            :class:`~primus.backends.megatron.core.transformer.dual_rope.RoPECache`.
            Queries and compressed indexer keys are rotated with it before
            scoring, exactly as in the reference. ``None`` disables the
            rotation entirely and is only meant for isolated unit tests --
            :meth:`DeepseekV4Attention._build_indexer` always supplies it.
        rotary_dim: partial-RoPE width (``qk_pos_emb_head_dim``, V4 = 64),
            applied to the trailing channels of the ``index_head_dim``-wide
            queries and keys.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        index_head_dim: int,
        index_n_heads: int,
        index_topk: int,
        compress_ratio: int = 4,
        dq_rank: int = None,
        use_fp8_qk: bool = False,
        rope: Optional[nn.Module] = None,
        rotary_dim: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.compress_ratio = compress_ratio
        self.dq_rank = dq_rank if dq_rank is not None else index_head_dim

        # Held by reference (list-wrapped) so the shared RoPE cache is not
        # re-registered as a submodule of every CSA layer's indexer.
        self._rope = [rope] if rope is not None else []
        self.rotary_dim = int(rotary_dim) if rope is not None else 0
        if self.rotary_dim > self.index_head_dim:
            raise ValueError(
                f"rotary_dim ({self.rotary_dim}) must be <= index_head_dim ({self.index_head_dim})"
            )
        if self.rotary_dim and index_head_dim & (index_head_dim - 1) != 0:
            # The Hadamard rotation spans the whole head, so fail here with the
            # config name rather than deep inside the transform.
            raise ValueError(
                f"index_head_dim must be a power of two for the Hadamard rotation, "
                f"got {index_head_dim}"
            )
        # FP8 (E4M3) fake-quant of the QK scoring inputs (V4 low-precision
        # indexer QK path). See ``fake_quantize_fp8_e4m3`` / config flag
        # ``use_v4_fp8_indexer``.
        self.use_fp8_qk = bool(use_fp8_qk)
        if self.use_fp8_qk and _is_rank0():
            logger.info(
                "[V4-Indexer] FP8 (E4M3) QK scoring path ENABLED "
                "(query/compressed-key activations fake-quantized; "
                "BF16 index-score + top-k preserved)."
            )

        # W^{DQ} (hidden->dq_rank) and W^w (hidden->n_heads) both consume `hidden`,
        # so fuse them into ONE GEMM (default-on); split the output. W^{IUQ} stays
        # separate (it consumes q_q, sequentially). PRIMUS_INDEXER_FUSE_PROJ=0 keeps
        # the two separate linears.
        self._fuse_qw_proj = os.environ.get("PRIMUS_INDEXER_FUSE_PROJ", "1") != "0"
        if self._fuse_qw_proj:
            self.w_dq_w = nn.Linear(hidden_size, self.dq_rank + index_n_heads, bias=False)
        else:
            # W^{DQ}: low-rank query down-projection.
            self.w_dq = nn.Linear(hidden_size, self.dq_rank, bias=False)
            # W^w_h: per-head scalar weight.
            self.w_w = nn.Linear(hidden_size, index_n_heads, bias=False)
        # W^{IUQ}_h: per-head up-projection from dq_rank → index_head_dim.
        self.w_iuq = nn.Linear(self.dq_rank, index_n_heads * index_head_dim, bias=False)

        # Temperature of the index score I_{t,s}. Folded into the per-head
        # weights ``w_i`` so the head sum lands on the same scale the reference
        # uses: it applies ``index_n_heads ** -0.5`` when building the per-head
        # weights and ``index_head_dim ** -0.5`` (the indexer's own softmax
        # scale) on the way into the loss.
        #
        # top-k is invariant under a positive constant, so this only matters
        # once the scores enter a softmax -- which is exactly what the indexer
        # distillation loss does. Without it the indexer distribution is
        # sqrt(index_n_heads * index_head_dim) times too sharp (~90x at the V4
        # widths) and the KL gradient is unusable.
        self.score_scale: float = (index_n_heads**-0.5) * (index_head_dim**-0.5)

        # Mini-Compressor producing K^{IComp}.
        self.indexer_compressor = Compressor(
            hidden_size=hidden_size,
            head_dim=index_head_dim,
            ratio=compress_ratio,
        )

    @property
    def rope(self) -> Optional[nn.Module]:
        """The compressed-branch RoPE cache, or ``None`` when not supplied."""
        return self._rope[0] if self._rope else None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Bridge checkpoints across the fused/unfused (w_dq, w_w) projection.

        Old checkpoints store ``w_dq.weight`` + ``w_w.weight``; the fused path wants
        ``w_dq_w.weight`` = ``cat([w_dq, w_w])`` (and vice-versa). Remap in-place so
        either layout loads under either runtime setting.
        """
        dq_k, w_k, fused_k = prefix + "w_dq.weight", prefix + "w_w.weight", prefix + "w_dq_w.weight"
        if self._fuse_qw_proj and dq_k in state_dict and fused_k not in state_dict:
            state_dict[fused_k] = torch.cat([state_dict.pop(dq_k), state_dict.pop(w_k)], dim=0)
        elif (not self._fuse_qw_proj) and fused_k in state_dict and dq_k not in state_dict:
            w = state_dict.pop(fused_k)
            state_dict[dq_k], state_dict[w_k] = w[: self.dq_rank], w[self.dq_rank :]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # ------------------------------------------------------------------

    def _causal_mask(
        self,
        n_queries: int,
        n_pool: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``[n_queries, n_pool]`` mask: 0.0 if pool position ``s``
        is allowed for query ``t``, ``-inf`` otherwise.

        A compressed position ``s`` covers raw tokens ``[s*ratio, (s+1)*ratio)``;
        a query at raw token ``t`` may attend to ``s`` iff its window
        end ``(s+1)*ratio - 1 <= t``.
        """
        # The mask depends only on (n_queries, n_pool, compress_ratio, dtype) — all
        # fixed per run — so cache it instead of rebuilding arange + where every
        # call. PRIMUS_INDEXER_MASK_CACHE=0 forces the eager rebuild.
        use_cache = os.environ.get("PRIMUS_INDEXER_MASK_CACHE", "1") != "0"
        if use_cache:
            cache = getattr(self, "_causal_mask_cache", None)
            if cache is None:
                cache = self._causal_mask_cache = {}
            key = (n_queries, n_pool, device, dtype)
            cached = cache.get(key)
            if cached is not None:
                return cached
        t_idx = torch.arange(n_queries, device=device).unsqueeze(1)  # [t, 1]
        s_end = (torch.arange(n_pool, device=device).unsqueeze(0) + 1) * self.compress_ratio - 1  # [1, s]
        allowed = s_end <= t_idx  # [t, s] bool
        mask = torch.where(allowed, 0.0, float("-inf")).to(dtype)
        if use_cache:
            cache[key] = mask
        return mask

    # ------------------------------------------------------------------

    def _rotate_keys(self, k_icomp: torch.Tensor) -> torch.Tensor:
        """Partial RoPE + Hadamard on the compressed indexer keys.

        ``k_icomp`` is ``[B, P, Hd]``. Compressed entry ``s`` covers the window
        starting at original token ``s * compress_ratio``, so it is rotated
        there -- the same coordinate system the queries use, and the same
        stride sampling the main compressed pool uses.
        """
        rope = self.rope
        if rope is None or self.rotary_dim == 0:
            return k_icomp

        P = k_icomp.shape[1]
        cos, sin = rope.forward_arange(P, k_icomp.device, stride=self.compress_ratio)
        # ``apply_interleaved_partial_rope`` inserts a singleton head axis into
        # cos / sin, so give it a ``[B, P, 1, Hd]`` view to broadcast against.
        rotated = apply_interleaved_partial_rope(
            k_icomp.unsqueeze(2),
            cos.unsqueeze(0).expand(k_icomp.shape[0], -1, -1),
            sin.unsqueeze(0).expand(k_icomp.shape[0], -1, -1),
            rotary_dim=self.rotary_dim,
        )
        return rotate_activation(rotated.squeeze(2))

    def _rotate_queries(self, q_i: torch.Tensor, position_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Partial RoPE + Hadamard on the indexer queries (``[B, S, H, Hd]``)."""
        rope = self.rope
        if rope is None or self.rotary_dim == 0:
            return q_i

        B, S = q_i.shape[0], q_i.shape[1]
        if position_ids is None:
            position_ids = torch.arange(S, device=q_i.device)
        cos, sin = rope(position_ids)
        if tuple(cos.shape[:-1]) != (B, S):
            cos = cos.expand(B, S, -1)
            sin = sin.expand(B, S, -1)
        return rotate_activation(
            apply_interleaved_partial_rope(q_i, cos, sin, rotary_dim=self.rotary_dim)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-k compressed positions for each query.

        Args:
            hidden: ``[B, S, D]``.
            position_ids: ``[B, S]`` or ``[S]`` absolute token positions, used
                to rotate the indexer queries. Defaults to ``arange(S)``.

        Returns:
            ``(topk_idxs, topk_scores)`` where:
            * ``topk_idxs`` ``[B, S, K]`` (long): selected pool positions
              in ``[0, P)`` for valid slots, ``-1`` for masked / invalid.
            * ``topk_scores`` ``[B, S, K]``: the selection scores ``I_{t,s}``
              (``-inf`` for masked positions).
        """
        B, S, D = hidden.shape
        assert S % self.compress_ratio == 0, (
            f"Indexer: sequence length {S} not divisible by compress_ratio " f"{self.compress_ratio}"
        )
        K = self.index_topk
        H = self.index_n_heads
        Hd = self.index_head_dim

        # 1) K^{IComp}: pool hidden via the mini-Compressor → [B, P, Hd], then
        # rotate. The reference builds the indexer keys with a compressor
        # configured as ``rotate=True`` (unlike the main one), and applies the
        # compressed-position RoPE inside it.
        k_icomp = self._rotate_keys(self.indexer_compressor(hidden))  # [B, P, Hd]
        P = k_icomp.shape[1]
        k_icomp = k_icomp.unsqueeze(2)  # [B, P, 1, Hd]

        # 2) Per-head query and per-head weight.
        # High precision by default, matching the reference: only the opt-in
        # PRIMUS_V4_FP8_INDEXER_PROJ knob routes these through MXFP8. No TP
        # gather/scatter (duplicated linears).
        proj = _fp8_linear if _indexer_fp8_proj_enabled() else (lambda lin, x: lin(x))
        if self._fuse_qw_proj:
            dqw = proj(self.w_dq_w, hidden)  # [B, S, dq_rank + H] in one GEMM
            q_q = dqw[..., : self.dq_rank]  # [B, S, dq_rank]
            w_i = dqw[..., self.dq_rank :]  # [B, S, H]
        else:
            q_q = proj(self.w_dq, hidden)  # [B, S, dq_rank]
            w_i = proj(self.w_w, hidden)  # [B, S, H]
        q_i = proj(self.w_iuq, q_q).view(B, S, H, Hd)  # [B, S, H, Hd]
        # Queries are rotated at their own token positions, then Hadamard-rotated
        # like the keys, so the ReLU'd dot product sees both operands in the same
        # basis the reference scores in.
        q_i = self._rotate_queries(q_i, position_ids)

        # Score temperature (see ``self.score_scale``). Applied on ``w_i``
        # because every scoring branch below multiplies by it before summing
        # over heads, so all of them inherit the scale from one place.
        w_i = w_i * self.score_scale

        # 3) Score I_{t,s} = Σ_h w_i[t,h] * ReLU(q_i[t,h] · k_icomp[s])
        #    q_i [B,S,H,Hd] · k_icomp[B,P,Hd] → relu[B,S,H,P]; w_i[B,S,H,1] → sum over H
        # 4) Causal mask + (effective topk capped at P).
        #
        # Dispatch precedence (P41 re-routing):
        #   PRIMUS_INDEXER_TRITON=1       → post-einsum tail fused
        #                                    (einsum stays eager / cuBLAS).
        #   PRIMUS_INDEXER_TRITON_FULL=1  → legacy P38 full-fuse path
        #                                    (einsum + tail in one kernel).
        #   else                          → fully eager.
        k_icomp_2d = k_icomp.squeeze(2)

        # Indexer QK precision (both default OFF -> BF16 QK). FP8 (E4M3) fake-
        # quantizes the operands before the normal score dispatch; FP4 (below)
        # is a dedicated real-GEMM branch and takes precedence when both are set.
        # The ReLU + per-head weight (``w_i``) + sum + causal mask + top-k stay
        # in the activation dtype — only the QK operands are quantized.
        if self.use_fp8_qk and not _indexer_fp4_enabled():
            q_i = fake_quantize_fp8_e4m3(q_i)
            k_icomp_2d = fake_quantize_fp8_e4m3(k_icomp_2d)

        # Phase 5: FP4 CSA-indexer QK. Real MXFP4 GEMM for the QK product (paper
        # §2.3.4/§5.2.1: "QK multiplied entirely in FP4"), then the eager
        # ReLU/weight/sum tail (w_i + tail stay BF16/FP32 — only the QK is FP4).
        if _indexer_fp4_enabled():
            dot = _fp4_qk_gemm(q_i, k_icomp_2d)  # [B, S, H, P], real FP4 matmul
            relu = F.relu(dot)
            scores = (relu * w_i.unsqueeze(-1)).sum(dim=2)  # [B, S, P]
            mask = self._causal_mask(S, P, scores.device, scores.dtype)  # [S, P]
            scores = scores + mask.unsqueeze(0)  # [B, S, P]
        elif _indexer_triton_full_enabled() and _indexer_triton_full_supported(q_i, k_icomp_2d, w_i):
            scores = indexer_score_triton(
                q_i,
                k_icomp_2d,
                w_i,
                compress_ratio=self.compress_ratio,
                out_dtype=hidden.dtype,
            )
        else:
            dot = torch.einsum("bshd,bpd->bshp", q_i, k_icomp_2d)
            if _indexer_tail_triton_enabled() and _indexer_tail_triton_supported(dot, w_i):
                scores = indexer_score_post_triton(
                    dot,
                    w_i,
                    compress_ratio=self.compress_ratio,
                    out_dtype=hidden.dtype,
                )
            else:
                relu = F.relu(dot)
                scores = (relu * w_i.unsqueeze(-1)).sum(dim=2)  # [B, S, P]
                mask = self._causal_mask(S, P, scores.device, scores.dtype)  # [S, P]
                scores = scores + mask.unsqueeze(0)  # [B, S, P]

        topk_eff = min(K, P)
        topk_scores, topk_idxs = scores.topk(topk_eff, dim=-1)  # [B, S, topk_eff]

        # 5) Replace selections that are still -inf (i.e. fewer than K valid
        #    pool positions for very early queries) with sentinel ``-1`` so
        #    callers can drop them.
        invalid = torch.isneginf(topk_scores)
        topk_idxs = torch.where(invalid, torch.full_like(topk_idxs, -1), topk_idxs)

        # 6) Pad with -1 to exactly K columns if topk_eff < K (S smaller than
        #    K * ratio in unit tests).
        if topk_eff < K:
            pad_idxs = torch.full((B, S, K - topk_eff), -1, dtype=topk_idxs.dtype, device=topk_idxs.device)
            pad_scores = torch.full(
                (B, S, K - topk_eff), float("-inf"), dtype=topk_scores.dtype, device=topk_scores.device
            )
            topk_idxs = torch.cat([topk_idxs, pad_idxs], dim=-1)
            topk_scores = torch.cat([topk_scores, pad_scores], dim=-1)

        return topk_idxs, topk_scores


__all__ = ["Indexer"]
