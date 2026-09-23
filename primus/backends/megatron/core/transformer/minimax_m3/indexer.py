###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax-M3's lightning indexer: the branch that decides which KV blocks to read.

A small ``sparse_num_index_heads``-head dot-product branch scores every query
against every key, max-pools those per-key scores into blocks of
``sparse_block_size`` keys, and keeps the top ``sparse_topk_blocks`` blocks per
query -- plus the local and init blocks, which are always visible. Selection is
therefore per *block of keys*, and there is one independent selection per GQA
group (``sparse_num_index_heads == num_query_groups``).

Like DeepSeek-V4's and DSA's indexers this is purely a *selection* branch: no
value projection, no residual output of its own. Its parameters get no gradient
from the main attention path, because top-k is not differentiable -- see
``indexer_loss.py`` for what actually trains them.

Mirrors ``MiniMaxM3VLIndexer`` in transformers'
``models/minimax_m3_vl/modeling_minimax_m3_vl.py``. Megatron works in ``sbhd``
while the reference works in ``bhsd``, so the projections here run on
``[sq, b, h]`` and the scores are permuted into ``[b, n_index, sq, sk]`` at the
matmul.
"""

import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
    MSATransformerConfig,
)


@dataclass
class MinimaxM3IndexerSubmodules:
    """Submodules of :class:`MinimaxM3Indexer`.

    Shaped after ``DSAIndexerSubmodules`` in Megatron's
    ``transformer/experimental_attention_variant/dsa.py``.
    """

    linear_index_q: Union[ModuleSpec, type] = None
    linear_index_k: Union[ModuleSpec, type] = None
    index_q_layernorm: Union[ModuleSpec, type] = None
    index_k_layernorm: Union[ModuleSpec, type] = None


class MinimaxM3Indexer(MegatronModule):
    """Scores KV blocks and returns the per-query top-k block ids."""

    def __init__(
        self,
        config: MSATransformerConfig,
        submodules: MinimaxM3IndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)

        self.index_n_heads = config.sparse_num_index_heads
        self.index_head_dim = config.sparse_index_dim
        self.block_size = config.sparse_block_size
        self.topk_blocks = config.sparse_topk_blocks
        self.init_blocks = config.sparse_init_block
        self.local_blocks = config.sparse_local_block
        self.pg_collection = pg_collection

        # Replicated across TP ranks rather than sharded, the way DSA builds its
        # indexer: the branch is tiny and every rank needs the full selection.
        self.linear_index_q = build_module(
            submodules.linear_index_q,
            config.hidden_size,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.linear_index_k = build_module(
            submodules.linear_index_k,
            config.hidden_size,
            self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        # The reference normalises both index q and k with the same Gemma-style
        # RMSNorm the rest of the model uses, over index_head_dim.
        norm_config = copy.copy(config)
        norm_config.normalization = "RMSNorm"
        self.index_q_layernorm = build_module(
            submodules.index_q_layernorm,
            config=norm_config,
            hidden_size=self.index_head_dim,
            eps=config.layernorm_epsilon,
        )
        self.index_k_layernorm = build_module(
            submodules.index_k_layernorm,
            config=norm_config,
            hidden_size=self.index_head_dim,
            eps=config.layernorm_epsilon,
        )

    def _project(self, hidden_states: torch.Tensor, linear, norm, num_heads: int) -> torch.Tensor:
        """``[sq, b, h]`` -> normalised ``[sq, b, num_heads, index_head_dim]``."""
        out = linear(hidden_states)
        if isinstance(out, tuple):  # Megatron linears return (output, bias)
            out = out[0]
        sq, b = out.shape[0], out.shape[1]
        return norm(out.view(sq, b, num_heads, self.index_head_dim))

    def forward(
        self, hidden_states: torch.Tensor, rotary_pos_emb: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select KV blocks for every query.

        Args:
            hidden_states: ``[sq, b, h]``, the attention module's input.
            rotary_pos_emb: rope freqs ``[sq, 1, 1, rotary_dim]``, or None.

        Returns:
            ``block_indices`` ``[b, n_index, sq, topk]`` (int64, ``-1`` padding
            in the unused right-hand slots) and ``block_scores``
            ``[b, n_index, sq, n_blocks]`` (float32), which the distillation
            loss consumes.
        """
        index_q = self._project(
            hidden_states, self.linear_index_q, self.index_q_layernorm, self.index_n_heads
        )
        index_k = self._project(hidden_states, self.linear_index_k, self.index_k_layernorm, 1)

        if rotary_pos_emb is not None:
            # Partial rope: freqs cover the first rotary_dim of index_head_dim and
            # the tail passes through, which is the reference's
            # `cos[..., :head_dim]` slice. Called unfused on purpose -- a
            # reference implementation should not depend on apply_rope_fusion.
            index_q = _apply_rotary_pos_emb_bshd(
                index_q, rotary_pos_emb, rotary_interleaved=self.config.rotary_interleaved
            )
            index_k = _apply_rotary_pos_emb_bshd(
                index_k, rotary_pos_emb, rotary_interleaved=self.config.rotary_interleaved
            )

        # [sq, b, n, d] -> [b, n, sq, d], and score in fp32 like the reference.
        index_q = index_q.permute(1, 2, 0, 3).float()
        index_k = index_k.permute(1, 2, 0, 3).float()
        scores = torch.matmul(index_q, index_k.transpose(-1, -2))  # [b, n_index, sq, sk]

        return self.select_blocks(scores)

    def select_blocks(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-key scores ``[b, n, sq, sk]`` -> (top-k block ids, block scores)."""
        b, n_index, sq, sk = scores.shape
        device = scores.device
        neg_inf = float("-inf")

        positions = torch.arange(sk, device=device)
        # Self-attention here, so query i sits at key position i.
        future = positions.view(1, 1, 1, sk) > positions[:sq].view(1, 1, sq, 1)
        scores = scores.masked_fill(future, neg_inf)

        n_blocks = -(-sk // self.block_size)  # ceil-div
        pad = n_blocks * self.block_size - sk
        if pad:
            scores = torch.nn.functional.pad(scores, (0, pad), value=neg_inf)

        # Max-pool the keys of each block; a block nobody may attend stays -inf,
        # which sorts to the end of top-k and becomes a -1 slot below.
        block_scores = scores.view(b, n_index, sq, n_blocks, self.block_size).amax(dim=-1)

        block_scores = self._boost_always_visible(block_scores, sq, n_blocks, device)

        topk = min(self.topk_blocks, n_blocks)
        topk_scores, block_indices = block_scores.topk(topk, dim=-1)
        block_indices = block_indices.masked_fill(topk_scores == neg_inf, -1)
        return block_indices, block_scores

    def _boost_always_visible(
        self, block_scores: torch.Tensor, sq: int, n_blocks: int, device: torch.device
    ) -> torch.Tensor:
        """Force the local and init blocks to win top-k slots.

        The reference does this by scattering ``+inf`` so the blocks always
        survive the top-k, which also keeps the selection free of duplicates --
        the block-sparse kernel reads the returned slots sequentially and would
        double-count a repeat.
        """
        q_block = torch.arange(sq, device=device) // self.block_size  # [sq]

        forced = []
        if self.local_blocks > 0:
            offsets = torch.arange(self.local_blocks, device=device)
            forced.append((q_block.view(sq, 1) - offsets.view(1, -1)).clamp(min=0))
        if self.init_blocks > 0:
            leading = torch.arange(min(self.init_blocks, n_blocks), device=device)
            # Never force a block the query cannot see anyway.
            forced.append(torch.minimum(leading.view(1, -1).expand(sq, -1), q_block.view(sq, 1)))
        if not forced:
            return block_scores

        index = torch.cat(forced, dim=-1)  # [sq, forced]
        index = index.view(1, 1, sq, -1).expand(block_scores.shape[0], block_scores.shape[1], -1, -1)
        return block_scores.scatter(-1, index, float("inf"))
