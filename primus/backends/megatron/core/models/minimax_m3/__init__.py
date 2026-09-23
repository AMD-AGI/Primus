###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax-M3 model package (text tower).

M3 reuses upstream ``GPTModel``; the only Primus-owned pieces are the config
class carrying the MSA (MiniMax Sparse Attention) fields and, via the patches
under ``primus/backends/megatron/patches``, the per-layer swap of
``SelfAttention`` for
:class:`~primus.backends.megatron.core.transformer.minimax_m3.MinimaxSparseAttention`.

The attention module is deliberately not re-exported here: importing it pulls
in transformer_engine, which the config-level tests avoid.
"""

from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
    MSATransformerConfig,
    normalize_sparse_layer_pattern,
)

__all__ = ["MSATransformerConfig", "normalize_sparse_layer_pattern"]
