###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax Sparse Attention (MSA).

    attention.py      MinimaxSparseAttention -- the self_attention slot
    indexer.py        the branch that picks which KV blocks to read
    eager.py          the reference block-sparse attention kernel
    indexer_loss.py   the distillation loss that trains the indexer

``MSA_BACKENDS`` names the kernels: ``eager`` (this reference) and ``flydsl``
(declared, not implemented).
"""

from primus.backends.megatron.core.transformer.minimax_m3.attention import (
    MSA_BACKENDS,
    MinimaxSparseAttention,
    MinimaxSparseAttentionSubmodules,
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

__all__ = [
    "MSA_BACKENDS",
    "MinimaxSparseAttention",
    "MinimaxSparseAttentionSubmodules",
    "MinimaxM3Indexer",
    "MinimaxM3IndexerSubmodules",
    "MSAIndexerLossAutoScaler",
    "MSAIndexerLossTracker",
    "compute_indexer_loss",
]
