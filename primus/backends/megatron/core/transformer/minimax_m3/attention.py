###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax Sparse Attention (MSA).

MSA is MiniMax-M3's attention: an index branch scores every
``sparse_block_size``-token KV block, the top ``sparse_topk_blocks`` blocks are
kept per GQA group (plus the init and local blocks), and the main branch then
runs exact attention over that subset only.

This class occupies the ``self_attention`` slot of upstream's
``TransformerLayer`` on the layers where
:attr:`MSATransformerConfig.sparse_layer_pattern` is 1, so M3 keeps running on
upstream ``GPTModel`` / ``TransformerLayer``. The swap is installed by
``primus/backends/megatron/patches/minimax_m3_patches.py``.

``forward`` is overridden rather than delegating to ``SelfAttention.forward``
because the eager backend computes attention itself and must not reach
Transformer Engine's ``core_attention``. The projections and the per-head q/k
norms are still upstream's -- only the attention core is replaced. (The unused
``core_attention`` module is still built by ``Attention.__init__``; it holds no
parameters, so leaving it costs nothing.)

Mirrors ``MiniMaxM3VLAttention`` in transformers'
``models/minimax_m3_vl/modeling_minimax_m3_vl.py``.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
    MSATransformerConfig,
)
from primus.backends.megatron.core.transformer.minimax_m3.eager import (
    build_block_keep,
    eager_block_sparse_attention,
)
from primus.backends.megatron.core.transformer.minimax_m3.indexer import (
    MinimaxM3Indexer,
    MinimaxM3IndexerSubmodules,
)
from primus.backends.megatron.core.transformer.minimax_m3.indexer_loss import (
    MSAIndexerLossAutoScaler,
    MSAIndexerLossTracker,
    compute_indexer_loss,
)

# Backends that compute the block-sparse attention itself. `eager` is the
# reference; `flydsl` is declared so presets and the config validation can name
# it before it exists.
MSA_BACKENDS = ("eager", "flydsl")


@dataclass
class MinimaxSparseAttentionSubmodules(SelfAttentionSubmodules):
    """``SelfAttentionSubmodules`` plus the indexer branch's four modules."""

    linear_index_q: Union[ModuleSpec, type] = None
    linear_index_k: Union[ModuleSpec, type] = None
    index_q_layernorm: Union[ModuleSpec, type] = None
    index_k_layernorm: Union[ModuleSpec, type] = None


class MinimaxSparseAttention(SelfAttention):
    """Self-attention layer for MiniMax-M3's MSA layers."""

    def __init__(
        self,
        config: MSATransformerConfig,
        submodules: MinimaxSparseAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: Optional[int] = None,
    ):
        if not isinstance(config, MSATransformerConfig):
            raise TypeError(
                "MinimaxSparseAttention requires an MSATransformerConfig, got "
                f"{type(config).__name__}. Set `minimax_sparse_attention: true` in the model "
                "preset so the config-class patch selects it."
            )
        if config.msa_backend != "eager":
            raise NotImplementedError(
                f"msa_backend={config.msa_backend!r} is not implemented yet; only 'eager' is. "
                "The eager backend is the numerical reference a faster backend aligns against."
            )

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
        )

        self.block_size = config.sparse_block_size
        self.indexer_loss_coeff = config.sparse_indexer_loss_coeff
        # TEDotProductAttention keeps no `softmax_scale` attribute, so resolve it
        # the same way Attention does when it hands one to the kernel: the config
        # override if set (mu-P uses it), else 1/sqrt(head_dim) -- the
        # reference's `self.scaling = self.head_dim**-0.5`.
        self.softmax_scale = config.softmax_scale or self.hidden_size_per_attention_head**-0.5
        self.indexer = build_module(
            ModuleSpec(
                module=MinimaxM3Indexer,
                submodules=MinimaxM3IndexerSubmodules(
                    linear_index_q=submodules.linear_index_q,
                    linear_index_k=submodules.linear_index_k,
                    index_q_layernorm=submodules.index_q_layernorm,
                    index_k_layernorm=submodules.index_k_layernorm,
                ),
            ),
            config=config,
            pg_collection=self.pg_collection,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[object] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        rotary_pos_cos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[object] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[object] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Training-only forward: qkv -> rope -> indexer -> block-sparse attention -> proj."""
        unsupported = {
            "key_value_states": key_value_states,
            "inference_context": inference_context or inference_params,
            "attention_bias": attention_bias,
            "packed_seq_params": packed_seq_params,
        }
        for name, value in unsupported.items():
            if value is not None:
                raise NotImplementedError(f"MinimaxSparseAttention does not support {name} yet.")

        # attention_mask, when the dataloader builds one, is [b, 1, sq, sk] bool
        # with True meaning "do not attend". The eager kernel applies causality
        # itself and composes this on top, so padding is honoured. The block
        # selection is left alone: an extra allowed block is harmless, and the
        # composed mask is what decides the final weights.

        # [sq, b, np, hn] / [sq, b, ng, hn] -- upstream's projections and q/k norms.
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = (
                rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else (rotary_pos_emb,) * 2
            )
            # Unfused on purpose; see the note in indexer.py.
            query = _apply_rotary_pos_emb_bshd(
                query, q_pos_emb, rotary_interleaved=self.config.rotary_interleaved
            )
            key = _apply_rotary_pos_emb_bshd(
                key, k_pos_emb, rotary_interleaved=self.config.rotary_interleaved
            )
            index_pos_emb = q_pos_emb
        else:
            index_pos_emb = None

        block_indices, block_scores = self.indexer(hidden_states, index_pos_emb)

        # sbhd -> bhsd for the eager kernel, matching the reference's layout.
        query_bhsd = query.permute(1, 2, 0, 3)
        key_bhsd = key.permute(1, 2, 0, 3)
        value_bhsd = value.permute(1, 2, 0, 3)

        block_keep = build_block_keep(block_indices, key_bhsd.shape[2], self.block_size)
        core_attn_out, dense_scores = eager_block_sparse_attention(
            query_bhsd,
            key_bhsd,
            value_bhsd,
            block_keep,
            self.softmax_scale,
            attention_mask=attention_mask,
        )

        if self.training and self.indexer_loss_coeff > 0.0:
            indexer_loss = compute_indexer_loss(
                dense_scores, block_scores, self.block_size, self.indexer_loss_coeff
            )
            MSAIndexerLossTracker.record(indexer_loss, self.layer_number, self.config.num_layers)
            core_attn_out = MSAIndexerLossAutoScaler.apply(core_attn_out, indexer_loss)

        # bhsd -> sbhd -> [sq, b, hp] for linear_proj.
        sq, b = hidden_states.shape[0], hidden_states.shape[1]
        core_attn_out = core_attn_out.permute(2, 0, 1, 3).reshape(sq, b, -1)

        return self.linear_proj(core_attn_out)
