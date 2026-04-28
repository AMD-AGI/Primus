###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek-V4 transformer block (multi-stream HC + per-layer attention dispatch).

Reference:
* techblog §1 ("Hybrid Attention") — per-layer attention selected by
  ``compress_ratios[layer_id]``.
* techblog §2 ("mHC: Manifold-Constrained Hyper-Connections") — ``hc_mult``
  parallel hidden streams, mixed via per-layer ``HyperMixer`` and final
  ``HyperHead`` collapse.

Phase 4 contract:
* This is a **standalone** ``nn.Module``. It does not inherit from
  ``megatron.core.transformer.transformer_block.TransformerBlock``; instead
  it exposes the same call signature so :class:`DeepseekV4Model` can
  ``self.decoder = DeepseekV4TransformerBlock(...)`` and Megatron's
  ``GPTModel.forward`` keeps working.
* PP / recompute / sequence-parallel etc. are **not** wired up here; that
  is Phase 6's job (deferred). For Phase 4 the block always runs the full
  ``num_layers`` on the local rank.
* The FFN sub-block is a vanilla SwiGLU MLP. The V4 MoE / hash-routed
  experts / clamped SwiGLU swap in during Phase 5.

Forward shape contract:
* Input ``hidden_states`` is ``[S, B, D]`` (Megatron's sequence-first
  convention). The block transposes to ``[B, S, D]`` internally, runs the
  K-stream HC loop, and transposes back before return.
* Output is ``[S, B, D]`` of the same dtype/device.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from primus.backends.megatron.core.transformer.csa_attention import CSAAttention
from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
    DeepseekV4Attention,
)
from primus.backends.megatron.core.transformer.dual_rope import DualRoPE
from primus.backends.megatron.core.transformer.hca_attention import HCAAttention
from primus.backends.megatron.core.transformer.hyper_connection import (
    HyperHead,
    HyperMixer,
)
from primus.backends.megatron.core.transformer.moe.v4_moe import DeepseekV4MoE

# ---------------------------------------------------------------------------
# Pieces used by every layer
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """Standalone RMSNorm so the block has no hard dep on TE."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        rsqrt = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * rsqrt).to(in_dtype) * self.weight


class _DenseSwiGLUMLP(nn.Module):
    """Plain dense SwiGLU FFN.

    Used for non-MoE layers (or as a fallback when ``num_routed_experts``
    is 0). V4-Flash has a tiny number of dense head/tail layers; the bulk
    of layers are MoE (see :class:`DeepseekV4MoE`).
    """

    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.w_up = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.w_down = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# Per-layer attention factory
# ---------------------------------------------------------------------------


def _build_attention(
    *,
    compress_ratio: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    rope: DualRoPE,
    attn_sliding_window: int,
    attn_sink_enabled: bool,
    q_lora_rank: Optional[int],
    index_topk: int,
    index_head_dim: int,
    index_n_heads: int,
) -> DeepseekV4Attention:
    """Pick the right attention class for ``compress_ratio``.

    * ``0``   → :class:`DeepseekV4Attention` (dense + SWA + sink)
    * ``128`` (or any value with the HCA convention) → :class:`HCAAttention`
    * ``4``   → :class:`CSAAttention` (overlap compressor + Indexer)

    The CSA / HCA distinction is determined by the ratio itself: by V4
    convention ratio ``4`` means CSA (sparse with Indexer) and any larger
    ratio (e.g. ``128``) means HCA (full compressed pool, no Indexer).
    """
    common = dict(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rope=rope,
        attn_sliding_window=attn_sliding_window,
        attn_sink_enabled=attn_sink_enabled,
        q_lora_rank=q_lora_rank,
    )

    if compress_ratio == 0:
        return DeepseekV4Attention(compress_ratio=0, **common)
    if compress_ratio == 4:
        return CSAAttention(
            compress_ratio=compress_ratio,
            index_topk=index_topk,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
            compressor_overlap=True,
            **common,
        )
    return HCAAttention(
        compress_ratio=compress_ratio,
        compressor_overlap=False,
        **common,
    )


# ---------------------------------------------------------------------------
# A single V4 block layer (attention sub-block + FFN sub-block, both wrapped
# by HyperMixer for the K-stream residual)
# ---------------------------------------------------------------------------


class DeepseekV4HybridLayer(nn.Module):
    """One layer of the V4 decoder.

    Holds:
      * pre-attention RMSNorm
      * attention sub-block (Dense / HCA / CSA, picked from ``compress_ratio``)
      * pre-FFN RMSNorm
      * FFN sub-block: MoE (:class:`DeepseekV4MoE`) when ``num_routed_experts > 0``,
        otherwise a plain dense SwiGLU. The MoE handles its own router dispatch
        (hash for the first ``num_hash_layers`` layers, sqrtsoftplus / sigmoid
        / softmax for the rest).
      * Two :class:`HyperMixer` instances (one per sub-block) when ``hc_mult > 1``
    """

    def __init__(
        self,
        *,
        layer_idx: int,
        compress_ratio: int,
        hidden_size: int,
        ffn_hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_dim: int,
        rope: DualRoPE,
        attn_sliding_window: int,
        attn_sink_enabled: bool,
        q_lora_rank: Optional[int],
        index_topk: int,
        index_head_dim: int,
        index_n_heads: int,
        hc_mult: int,
        hc_eps: float = 1e-6,
        hc_sinkhorn_iters: int = 20,
        norm_eps: float = 1e-6,
        # MoE config (set num_routed_experts=0 to use the plain dense FFN).
        num_routed_experts: int = 0,
        moe_router_topk: int = 1,
        moe_intermediate_size: Optional[int] = None,
        num_shared_experts: int = 1,
        num_hash_layers: int = 0,
        hash_vocab_size: Optional[int] = None,
        hash_seed: int = 0,
        moe_score_function: str = "sqrtsoftplus",
        moe_enable_expert_bias: bool = True,
        clamp_alpha: float = 7.0,
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.compress_ratio = int(compress_ratio)
        self.hc_mult = int(hc_mult)

        self.attn_norm = _RMSNorm(hidden_size, eps=norm_eps)
        self.attn = _build_attention(
            compress_ratio=compress_ratio,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            rope=rope,
            attn_sliding_window=attn_sliding_window,
            attn_sink_enabled=attn_sink_enabled,
            q_lora_rank=q_lora_rank,
            index_topk=index_topk,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
        )

        self.ffn_norm = _RMSNorm(hidden_size, eps=norm_eps)
        self.is_moe = num_routed_experts > 0
        if self.is_moe:
            moe_inner = moe_intermediate_size or ffn_hidden_size
            self.ffn = DeepseekV4MoE(
                hidden_size=hidden_size,
                moe_intermediate_size=moe_inner,
                num_routed_experts=num_routed_experts,
                moe_router_topk=moe_router_topk,
                num_shared_experts=num_shared_experts,
                layer_idx=self.layer_idx,
                num_hash_layers=num_hash_layers,
                hash_vocab_size=hash_vocab_size,
                hash_seed=hash_seed,
                score_function=moe_score_function,
                enable_expert_bias=moe_enable_expert_bias,
                clamp_alpha=clamp_alpha,
            )
        else:
            self.ffn = _DenseSwiGLUMLP(hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size)

        if hc_mult > 1:
            self.attn_hc = HyperMixer(
                hidden_size=hidden_size,
                hc_mult=hc_mult,
                eps=hc_eps,
                sinkhorn_iters=hc_sinkhorn_iters,
            )
            self.ffn_hc = HyperMixer(
                hidden_size=hidden_size,
                hc_mult=hc_mult,
                eps=hc_eps,
                sinkhorn_iters=hc_sinkhorn_iters,
            )
        else:
            self.attn_hc = None
            self.ffn_hc = None

    # ------------------------------------------------------------------

    def _hc_apply(self, mixer: Optional[HyperMixer], x: torch.Tensor, sub_block, *args):
        """Run a sub-block under HC.

        ``x`` shape: ``[B, S, K, D]`` if ``hc_mult > 1``, else ``[B, S, D]``.
        ``sub_block`` is ``Callable[[Tensor, *Any], Tensor]`` whose first
        positional arg is the (collapsed) hidden in ``[B, S, D]``.
        """
        if mixer is None:
            # Single-stream: classic residual; x already has shape [B, S, D].
            out = sub_block(x, *args)
            return x + out

        pre, post, comb = mixer.compute_weights(x)  # [..., K], [..., K], [..., K, K]
        collapsed = HyperMixer.collapse(x, pre)  # [B, S, D]
        out = sub_block(collapsed, *args)  # sub-block first positional = collapsed
        return HyperMixer.expand(x, out, post, comb)  # [B, S, K, D]

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one V4 layer.

        ``x``: ``[B, S, K, D]`` (multi-stream) or ``[B, S, D]`` (single).
        ``position_ids``: ``[B, S]`` or ``[S]``.
        ``token_ids``: ``[B, S]`` integer tensor; required when this is a
            hash-routed MoE layer (``layer_idx < num_hash_layers``). Ignored
            for non-MoE / non-hash layers.
        """

        # Attention sub-block. The collapse passes a [B, S, D] hidden, then
        # the attention runs and returns [B, S, D]; HC expand writes back.
        def _attn_sub(collapsed: torch.Tensor) -> torch.Tensor:
            return self.attn(self.attn_norm(collapsed), position_ids)

        x = self._hc_apply(self.attn_hc, x, _attn_sub)

        # FFN sub-block. MoE FFN needs token_ids when the layer is hash-routed;
        # plain SwiGLU ignores it.
        if self.is_moe:

            def _ffn_sub(collapsed: torch.Tensor) -> torch.Tensor:
                return self.ffn(self.ffn_norm(collapsed), token_ids=token_ids)

        else:

            def _ffn_sub(collapsed: torch.Tensor) -> torch.Tensor:
                return self.ffn(self.ffn_norm(collapsed))

        x = self._hc_apply(self.ffn_hc, x, _ffn_sub)
        return x


# ---------------------------------------------------------------------------
# Top-level V4 transformer block
# ---------------------------------------------------------------------------


class DeepseekV4TransformerBlock(nn.Module):
    """Multi-stream HC decoder for DeepSeek-V4.

    Replaces Megatron's ``TransformerBlock`` for V4. The ``__init__``
    signature accepts a TransformerConfig-like object so it can be
    constructed exactly the way ``GPTModel.__init__`` constructs the
    standard block. Most fields are read off ``config`` with ``getattr``
    fallbacks (so this works whether the V4 fields land via Primus's
    ``merge_namespace`` mechanism or are set explicitly).

    In Phase 4 the block always runs the full ``num_layers`` on the local
    rank — no PP slicing yet (P6 deferred).
    """

    def __init__(
        self,
        config,
        spec=None,
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection=None,
        vp_stage=None,
    ) -> None:
        super().__init__()
        # Save arguments matching the parent's interface for compatibility.
        self.config = config
        self.spec = spec
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process

        # ---- shape / model fields ----
        hidden_size = config.hidden_size
        ffn_hidden_size = getattr(config, "ffn_hidden_size", None) or 4 * hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_query_groups", None) or num_heads
        head_dim = getattr(config, "kv_channels", None) or hidden_size // num_heads
        rotary_dim = getattr(config, "qk_pos_emb_head_dim", 64)
        num_layers = config.num_layers
        norm_eps = getattr(config, "norm_epsilon", 1.0e-6)

        # ---- V4-specific fields ----
        hc_mult = getattr(config, "hc_mult", 1) or 1
        hc_eps = getattr(config, "hc_eps", 1.0e-6) or 1.0e-6
        hc_sinkhorn_iters = getattr(config, "hc_sinkhorn_iters", 20) or 20
        compress_ratios: Optional[Sequence[int]] = getattr(config, "compress_ratios", None)
        if compress_ratios is None:
            compress_ratios = [0] * num_layers
        compress_ratios = list(compress_ratios)
        if len(compress_ratios) != num_layers:
            raise ValueError(f"compress_ratios length {len(compress_ratios)} != num_layers {num_layers}")
        self.compress_ratios: List[int] = compress_ratios

        attn_sliding_window = getattr(config, "attn_sliding_window", 128) or 0
        attn_sink_enabled = bool(getattr(config, "attn_sink", False))
        q_lora_rank = getattr(config, "q_lora_rank", None) or None
        index_topk = getattr(config, "index_topk", 512) or 512
        index_head_dim = getattr(config, "index_head_dim", 128) or 128
        index_n_heads = getattr(config, "index_n_heads", 64) or 64

        rope_theta = getattr(config, "rotary_base", 10000.0)
        compress_rope_theta = getattr(config, "compress_rope_theta", 160000.0)
        yarn_factor = getattr(config, "rotary_scaling_factor", 1.0) or 1.0
        original_max_pos = getattr(config, "original_max_position_embeddings", 0) or 0

        # ---- V4 MoE fields ----
        # ``num_routed_experts == 0`` keeps the dense SwiGLU FFN; this lets us
        # stand up small unit tests without instantiating MoE state.
        num_routed_experts = int(getattr(config, "num_moe_experts", 0) or 0)
        moe_router_topk = int(getattr(config, "moe_router_topk", 1) or 1)
        moe_intermediate_size = getattr(config, "moe_ffn_hidden_size", None) or getattr(
            config, "moe_intermediate_size", None
        )
        num_shared_experts = int(getattr(config, "moe_shared_expert_intermediate_size", 0) > 0) or int(
            getattr(config, "num_shared_experts", 1)
        )
        num_hash_layers = int(getattr(config, "num_hash_layers", 0) or 0)
        hash_vocab_size = getattr(config, "padded_vocab_size", None) or getattr(config, "vocab_size", None)
        hash_seed = int(getattr(config, "hash_routing_seed", 0) or 0)
        moe_score_function = getattr(config, "moe_router_score_function", "sqrtsoftplus")
        moe_enable_expert_bias = bool(getattr(config, "moe_router_enable_expert_bias", True))
        clamp_alpha = float(getattr(config, "swiglu_limit", 7.0) or 7.0)
        self.num_hash_layers = num_hash_layers

        # ---- shared dual-RoPE for the whole stack ----
        self.rope = DualRoPE(
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            compress_rope_theta=compress_rope_theta,
            yarn_factor=yarn_factor,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
            original_max_position_embeddings=original_max_pos,
        )

        # ---- layers ----
        self.layers = nn.ModuleList()
        for i, ratio in enumerate(compress_ratios):
            layer = DeepseekV4HybridLayer(
                layer_idx=i,
                compress_ratio=int(ratio),
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                rope=self.rope,
                attn_sliding_window=attn_sliding_window,
                attn_sink_enabled=attn_sink_enabled,
                q_lora_rank=q_lora_rank,
                index_topk=index_topk,
                index_head_dim=index_head_dim,
                index_n_heads=index_n_heads,
                hc_mult=hc_mult,
                hc_eps=hc_eps,
                hc_sinkhorn_iters=hc_sinkhorn_iters,
                norm_eps=norm_eps,
                num_routed_experts=num_routed_experts,
                moe_router_topk=moe_router_topk,
                moe_intermediate_size=moe_intermediate_size,
                num_shared_experts=num_shared_experts,
                num_hash_layers=num_hash_layers,
                hash_vocab_size=hash_vocab_size,
                hash_seed=hash_seed,
                moe_score_function=moe_score_function,
                moe_enable_expert_bias=moe_enable_expert_bias,
                clamp_alpha=clamp_alpha,
            )
            self.layers.append(layer)
        self.hc_mult = hc_mult

        # Final HC collapse (only if multi-stream).
        if hc_mult > 1:
            self.hyper_head = HyperHead(hidden_size=hidden_size, hc_mult=hc_mult, eps=hc_eps)
        else:
            self.hyper_head = None

        # Final RMSNorm if post_layer_norm.
        if post_layer_norm:
            self.final_layernorm = _RMSNorm(hidden_size, eps=norm_eps)
        else:
            self.final_layernorm = None

    # ------------------------------------------------------------------

    @property
    def num_layers_per_pipeline_rank(self) -> int:
        """Compatibility shim for upstream debug printing — we run all layers
        on every rank since PP isn't wired up yet."""
        return len(self.layers)

    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the V4 decoder.

        Megatron passes ``hidden_states`` as ``[S, B, D]`` (sequence-first).
        We transpose to ``[B, S, D]`` for HC math and back for the return.
        ``attention_mask`` and the various ``rotary_pos_*`` kwargs are
        ignored — V4 manages its own dual-RoPE and SWA mask internally.

        ``token_ids`` (``[B, S]`` long tensor) is required only when one
        or more layers run hash routing (``layer_idx < num_hash_layers``).
        :class:`DeepseekV4Model` forwards ``input_ids`` here for that
        purpose; non-V4 callers (e.g. probes / unit tests of layers
        without hash routing) can omit it.
        """
        # If the model stashed ``input_ids`` on the block (see
        # :class:`DeepseekV4Model.forward`), pick it up; explicit kwarg wins.
        if token_ids is None:
            token_ids = getattr(self, "_v4_token_ids", None)

        # [S, B, D] -> [B, S, D]
        x = hidden_states.transpose(0, 1).contiguous()
        B, S, D = x.shape

        # Position ids (one per token).
        position_ids = torch.arange(S, device=x.device)

        # Expand to K streams.
        if self.hc_mult > 1:
            x = x.unsqueeze(2).expand(B, S, self.hc_mult, D).contiguous()

        # Run the layers.
        for layer in self.layers:
            x = layer(x, position_ids, token_ids=token_ids)

        # Final HC collapse.
        if self.hc_mult > 1 and self.hyper_head is not None:
            x = self.hyper_head(x)  # [B, S, D]

        if self.final_layernorm is not None:
            x = self.final_layernorm(x)

        # Back to [S, B, D] for downstream Megatron code.
        return x.transpose(0, 1).contiguous()


__all__ = [
    "DeepseekV4HybridLayer",
    "DeepseekV4TransformerBlock",
]
