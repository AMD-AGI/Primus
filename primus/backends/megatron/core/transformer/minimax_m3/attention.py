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
because the attention core is MSA's own and must not reach Transformer Engine's
``core_attention``. The projections and the per-head q/k norms are still
upstream's -- only the attention core is replaced. (The unused
``core_attention`` module is still built by ``Attention.__init__``; it holds no
parameters, so leaving it costs nothing.)

``msa_backend`` picks the core: ``eager``, the plain-PyTorch reference that
materialises the full score matrix, or ``flydsl``, the gfx950 kernels in
``minimax_m3/flydsl`` that never do.

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
    compute_indexer_loss,
    compute_sparse_indexer_loss,
    record_indexer_loss,
    slot_mass_from_probs,
    slot_mass_from_slot_lse,
)

# Backends that compute the block-sparse attention itself.
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

        if self._fused_input_norm_weight() is not None and config.normalization != "RMSNorm":
            raise NotImplementedError(
                "MinimaxSparseAttention recomputes linear_qkv's fused input norm for the indexer "
                f"and supports RMSNorm only; got normalization={config.normalization!r}."
            )

        self.msa_attention = None
        if config.msa_backend == "flydsl":
            # Imported here, not at module level: flydsl exists only on gfx950 builds.
            from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_function import (
                msa_attention,
            )

            heads_per_group = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
            if self.hidden_size_per_attention_head != 128 or heads_per_group != 16:
                raise NotImplementedError(
                    "msa_backend='flydsl' is built for head_dim 128 and 16 query heads per KV head "
                    f"(M3's shape); got head_dim={self.hidden_size_per_attention_head}, "
                    f"{heads_per_group} heads per KV head."
                )
            self.msa_attention = msa_attention

    def _fused_input_norm_weight(self) -> Optional[torch.Tensor]:
        """The input-norm weight fused into ``linear_qkv``, or None when the norm is separate."""
        return getattr(self.linear_qkv, "layer_norm_weight", None)

    def _indexer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``input_layernorm(hidden_states)``, detached: what the reference indexer reads.

        The TE specs fuse the input norm into ``linear_qkv``, so ``hidden_states``
        arrives here as the raw residual stream; the norm is recomputed from the
        fused weight. Weight and input are both detached because the distillation
        loss must train the indexer alone, not the layers that feed it.
        """
        x = hidden_states.detach()
        norm_weight = self._fused_input_norm_weight()
        if norm_weight is None:
            return x
        weight = norm_weight.detach().float()
        if self.config.layernorm_zero_centered_gamma:
            weight = weight + 1.0
        normed = torch.nn.functional.rms_norm(
            x.float(), (x.shape[-1],), weight, self.config.layernorm_epsilon
        )
        return normed.to(x.dtype)

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

        block_indices, block_scores, block_plan = self.indexer(
            self._indexer_input(hidden_states), index_pos_emb
        )

        # Full recompute runs this forward twice, first under no_grad; gating on
        # grad mode computes and records the loss once, in the pass that backprops.
        want_loss = self.training and torch.is_grad_enabled() and self.indexer_loss_coeff > 0.0
        sparse_loss = self.config.sparse_indexer_loss_type == "sparse"
        sq, b = hidden_states.shape[0], hidden_states.shape[1]

        if self.msa_attention is not None:
            if attention_mask is not None:
                raise NotImplementedError(
                    "msa_backend='flydsl' applies causality itself and has no padding mask; set "
                    "`create_attention_mask_in_dataloader: false`."
                )
            if block_indices.shape[1] != key.shape[2]:
                raise NotImplementedError(
                    f"the indexer selected for {block_indices.shape[1]} KV heads but this rank holds "
                    f"{key.shape[2]}; msa_backend='flydsl' needs one selection per local KV head."
                )
            # [sq, b, heads, hn] in and out: o needs no permute before linear_proj.
            outs = self.msa_attention(
                query,
                key,
                value,
                block_indices,
                self.softmax_scale,
                self.block_size,
                return_slot_lse=want_loss,
                plan=block_plan,
            )
            core_attn_out = outs[0]
            if want_loss:
                slot_mass = slot_mass_from_slot_lse(outs[2], outs[1], block_indices.shape[1])
                indexer_loss = compute_sparse_indexer_loss(
                    slot_mass, block_scores, block_indices, self.indexer_loss_coeff
                )
        else:
            # sbhd -> bhsd for the eager kernel, matching the reference's layout.
            block_keep = build_block_keep(block_indices, key.shape[0], self.block_size)
            need_probs = want_loss and sparse_loss
            outs = eager_block_sparse_attention(
                query.permute(1, 2, 0, 3),
                key.permute(1, 2, 0, 3),
                value.permute(1, 2, 0, 3),
                block_keep,
                self.softmax_scale,
                attention_mask=attention_mask,
                return_probs=need_probs,
            )
            core_attn_out = outs[0]
            if need_probs:
                slot_mass = slot_mass_from_probs(outs[2], block_indices, self.block_size)
                indexer_loss = compute_sparse_indexer_loss(
                    slot_mass, block_scores, block_indices, self.indexer_loss_coeff
                )
            elif want_loss:
                indexer_loss = compute_indexer_loss(
                    outs[1], block_scores, self.block_size, self.indexer_loss_coeff
                )
            # bhsd -> sbhd
            core_attn_out = core_attn_out.permute(2, 0, 1, 3)

        if want_loss:
            mtp_layers = getattr(self.config, "mtp_num_layers", None) or 0
            record_indexer_loss(indexer_loss, self.layer_number, self.config.num_layers + mtp_layers)
            core_attn_out = MSAIndexerLossAutoScaler.apply(core_attn_out, indexer_loss)

        return self.linear_proj(core_attn_out.reshape(sq, b, -1))
