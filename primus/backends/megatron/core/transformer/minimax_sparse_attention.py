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
``primus/backends/megatron/patches/minimax_sparse_attention_patches.py``.

STATUS: skeleton. The config is read and validated here, but the block-sparse
path is not implemented yet and ``forward`` is inherited unchanged, so these
layers currently compute *dense* attention. See the TODO in
:class:`MinimaxSparseAttention` for what is missing.
"""

from typing import Optional

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType

from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
    MSATransformerConfig,
)
from primus.core.utils.module_utils import warning_rank_0

# The dense-fallback warning says the same thing for all 57 MSA layers; one
# line per run is enough to make the state of the model obvious in the log.
_warned_dense_fallback = False


class MinimaxSparseAttention(SelfAttention):
    """Self-attention layer for MiniMax-M3's MSA layers.

    Signature matches upstream ``SelfAttention`` exactly so it can be dropped
    into a ``SelfAttentionSubmodules`` spec in its place.

    TODO: implement the actual block-sparse path. What is missing:
      - the index branch: ``sparse_num_index_heads`` x ``sparse_index_dim``
        q/k projections, partial RoPE, and the per-token index scores;
      - reduction of each ``sparse_block_size``-token block to one score with
        ``sparse_score_type`` (``max``);
      - per-GQA-group top-``sparse_topk_blocks`` selection, unioned with the
        leading ``sparse_init_block`` and trailing ``sparse_local_block``
        blocks;
      - a block mask (or a block-sparse kernel) for the main branch, plus the
        training signal for the non-differentiable top-k;
      - a context-parallel guard: block selection spans the whole KV sequence,
        which a sharded CP sequence does not have locally.
    """

    def __init__(
        self,
        config: MSATransformerConfig,
        submodules: SelfAttentionSubmodules,
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

        self.index_n_heads = config.sparse_num_index_heads
        self.index_head_dim = config.sparse_index_dim
        self.block_size = config.sparse_block_size
        self.topk_blocks = config.sparse_topk_blocks
        self.score_type = config.sparse_score_type
        self.init_block = config.sparse_init_block
        self.local_block = config.sparse_local_block

        global _warned_dense_fallback
        if not _warned_dense_fallback:
            _warned_dense_fallback = True
            warning_rank_0(
                "[MinimaxSparseAttention] MSA is configured "
                f"({self.index_n_heads} index heads x dim {self.index_head_dim}, "
                f"block {self.block_size}, top-{self.topk_blocks} blocks, "
                f"score={self.score_type}, init={self.init_block}, local={self.local_block}) "
                "but the block-sparse path is not implemented yet: these layers run DENSE "
                "attention. Compute and memory will match a dense model, not M3."
            )
