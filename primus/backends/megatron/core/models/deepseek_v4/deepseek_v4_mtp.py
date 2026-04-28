###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 Multi-Token Prediction (MTP) block.

Reference: techblog §7 ("MTP V4 head") and
``DeepSeek-V4-Flash/inference/model.py:MTPBlock``.

V4's MTP differs from V3 in one key way: each MTP layer owns its **own**
:class:`HyperHead` for the final K-stream collapse. Embedding and the
final ``lm_head`` are shared with the main decoder, but the HC head is
**not**. This keeps gradients clean — the main loss and the auxiliary
MTP loss don't fight over a single set of HyperHead parameters.

Phase 5 contract (this file):
* Standalone module ``DeepseekV4MTPBlock`` that takes the post-decoder
  hidden state ``[B, S, D]`` and produces ``mtp_num_layers`` shifted
  hidden states, each ``[B, S, D]``, ready to be fed through the shared
  ``lm_head`` by the caller.
* Each MTP layer is a :class:`DeepseekV4HybridLayer` with the same
  attention / FFN configuration as the main decoder, plus its own
  per-layer HyperHead at the output.
* Wiring this into the model's loss path (``forward`` + auxiliary MTP
  loss term) is intentionally deferred to **Phase 6**, when the model
  integrates with Megatron's training loop and the upstream
  ``mtp_num_layers`` argument flows end-to-end.

The fact that this file exists as a P5 deliverable (instead of waiting
for P6) lets us land + unit-test the V4-specific MTP shape contract in
isolation, parallel to the rest of P5.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_block import (
    DeepseekV4HybridLayer,
    _RMSNorm,
)
from primus.backends.megatron.core.transformer.dual_rope import DualRoPE
from primus.backends.megatron.core.transformer.hyper_connection import HyperHead


class DeepseekV4MTPBlock(nn.Module):
    """A stack of ``mtp_num_layers`` V4 layers + per-layer :class:`HyperHead`.

    Args:
        config: TransformerConfig-like object. Reads the same V4 fields as
            :class:`DeepseekV4TransformerBlock` (``hidden_size``,
            ``hc_mult``, ``compress_ratios`` first MTP entry, etc.).
        rope: shared :class:`DualRoPE` instance — must be the same object
            the main decoder uses, so embedding / position layout stay
            aligned.
        mtp_num_layers: number of MTP layers (V4-Flash = 1).
        mtp_compress_ratios: per-MTP-layer compression ratio. Defaults to
            all-zero (dense), which matches the V4 reference. The list
            length must equal ``mtp_num_layers``.
    """

    def __init__(
        self,
        config,
        *,
        rope: DualRoPE,
        mtp_num_layers: int,
        mtp_compress_ratios: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if mtp_num_layers <= 0:
            raise ValueError(f"mtp_num_layers must be > 0, got {mtp_num_layers}")

        if mtp_compress_ratios is None:
            mtp_compress_ratios = [0] * mtp_num_layers
        mtp_compress_ratios = list(mtp_compress_ratios)
        if len(mtp_compress_ratios) != mtp_num_layers:
            raise ValueError(
                "mtp_compress_ratios length " f"{len(mtp_compress_ratios)} != mtp_num_layers {mtp_num_layers}"
            )

        # ---- shape / model fields (mirror DeepseekV4TransformerBlock) ----
        hidden_size = config.hidden_size
        ffn_hidden_size = getattr(config, "ffn_hidden_size", None) or 4 * hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_query_groups", None) or num_heads
        head_dim = getattr(config, "kv_channels", None) or hidden_size // num_heads
        rotary_dim = getattr(config, "qk_pos_emb_head_dim", 64)
        norm_eps = getattr(config, "norm_epsilon", 1.0e-6)

        hc_mult = int(getattr(config, "hc_mult", 1) or 1)
        hc_eps = float(getattr(config, "hc_eps", 1.0e-6) or 1.0e-6)
        hc_sinkhorn_iters = int(getattr(config, "hc_sinkhorn_iters", 20) or 20)

        attn_sliding_window = int(getattr(config, "attn_sliding_window", 128) or 0)
        attn_sink_enabled = bool(getattr(config, "attn_sink", False))
        q_lora_rank = getattr(config, "q_lora_rank", None) or None
        index_topk = int(getattr(config, "index_topk", 512) or 512)
        index_head_dim = int(getattr(config, "index_head_dim", 128) or 128)
        index_n_heads = int(getattr(config, "index_n_heads", 64) or 64)

        num_routed_experts = int(getattr(config, "num_moe_experts", 0) or 0)
        moe_router_topk = int(getattr(config, "moe_router_topk", 1) or 1)
        moe_intermediate_size = getattr(config, "moe_ffn_hidden_size", None) or getattr(
            config, "moe_intermediate_size", None
        )
        num_shared_experts = int(getattr(config, "num_shared_experts", 1) or 1)
        # MTP runs at the top of the stack — past the hash-routed prefix —
        # so its routing always uses the learned router.
        num_hash_layers = 0
        hash_vocab_size = getattr(config, "padded_vocab_size", None) or getattr(config, "vocab_size", None)
        moe_score_function = getattr(config, "moe_router_score_function", "sqrtsoftplus")
        moe_enable_expert_bias = bool(getattr(config, "moe_router_enable_expert_bias", True))
        clamp_alpha = float(getattr(config, "swiglu_limit", 7.0) or 7.0)

        self.mtp_num_layers = int(mtp_num_layers)
        self.hc_mult = hc_mult

        # ---- pre-MTP norm (matches main decoder's final_layernorm role) ----
        # The main decoder's output already passed through final_layernorm
        # before reaching us; we add a small re-norm so MTP's first attention
        # sees a well-conditioned input. This matches the reference.
        self.input_norm = _RMSNorm(hidden_size, eps=norm_eps)

        # ---- MTP layers + per-layer HyperHead ----
        self.layers = nn.ModuleList()
        self.heads: List[HyperHead] = nn.ModuleList()  # type: ignore[assignment]
        for i, ratio in enumerate(mtp_compress_ratios):
            layer = DeepseekV4HybridLayer(
                # ``layer_idx`` is purposely left at the post-prefix range so
                # the layer's MoE picks the learned router — see num_hash_layers=0.
                layer_idx=10**6 + i,
                compress_ratio=int(ratio),
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
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
                hash_seed=0,
                moe_score_function=moe_score_function,
                moe_enable_expert_bias=moe_enable_expert_bias,
                clamp_alpha=clamp_alpha,
            )
            self.layers.append(layer)

            # Each MTP layer owns its own HyperHead so its gradient path is
            # independent of the main HyperHead. Single-stream MTP (hc_mult=1)
            # skips this head — there's only one stream to collapse.
            if hc_mult > 1:
                self.heads.append(HyperHead(hidden_size=hidden_size, hc_mult=hc_mult, eps=hc_eps))
            else:
                self.heads.append(nn.Identity())  # type: ignore[arg-type]

        # ---- final norm (per-MTP, applied after collapse) ----
        self.final_layernorm = _RMSNorm(hidden_size, eps=norm_eps)

    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        token_ids: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Run the MTP stack.

        Args:
            hidden_states: ``[B, S, D]`` post-decoder hidden state. (Note:
                **not** the Megatron sequence-first ``[S, B, D]`` — the
                caller is :class:`DeepseekV4Model` which already
                transposes back. This makes MTP shape contract internal
                and uniform with the rest of V4.)
            token_ids: ``[B, S]`` long tensor. Forwarded to each MTP
                layer's MoE FFN. Optional even when MoE is on, since
                ``num_hash_layers=0`` here means the learned router is
                always used.

        Returns:
            A list of ``mtp_num_layers`` tensors, each ``[B, S, D]``,
            ready to be fed through the shared ``lm_head`` (and shifted
            by ``i+1`` positions for loss computation).
        """
        outputs: List[torch.Tensor] = []
        x = self.input_norm(hidden_states)
        B, S, D = x.shape

        # MTP layers operate on a single stream that we expand to K
        # streams per layer (independent of the main decoder's collapsed
        # output, which is already single-stream after the main HyperHead).
        position_ids = torch.arange(S, device=x.device)

        for i, (layer, head) in enumerate(zip(self.layers, self.heads)):
            stream = x
            if self.hc_mult > 1:
                stream = stream.unsqueeze(2).expand(B, S, self.hc_mult, D).contiguous()
            stream = layer(stream, position_ids, token_ids=token_ids)
            if self.hc_mult > 1:
                stream = head(stream)
            outputs.append(self.final_layernorm(stream))
            # The reference uses each MTP layer's output as the seed for
            # the next; we do the same so MTP layer ``i+1`` extends layer
            # ``i``'s prediction.
            x = stream

        return outputs


__all__ = ["DeepseekV4MTPBlock"]
