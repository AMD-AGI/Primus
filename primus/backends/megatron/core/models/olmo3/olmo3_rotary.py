###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two RoPE tables for OLMo-3: plain RoPE (sliding layers) and YaRN (full layers).

``forward`` returns ``torch.stack([rope, yarn])`` of shape ``[2, s, 1, 1, d]``;
:class:`Olmo3SelfAttention` selects index 0 or 1 for its layer. The YaRN
attention (concentration) factor is *not* applied here: ``get_emb`` calls only
``YarnRotaryEmbedding.get_emb`` (never its ``forward``), keeps the frequency table
and drops the concentration factor it returns (with ``mscale=1``, ``mscale_all_dim=0`` that
value is ``0.1 * ln(factor) + 1``, about 1.208 for factor 8, not 1; ``mscale``
does not change the frequencies). The attention applies the factor once, through
``softmax_scale *= m**2``; do not also scale by the returned value.

``get_emb`` returns the same stacked table (before CP slicing). ``get_cos_sin``
(used only by Megatron's flash-decode / FlashInfer inference paths) has no
per-layer form and raises.
"""

from functools import lru_cache

import torch
from megatron.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
)
from torch import Tensor

__all__ = ["Olmo3DualRotaryEmbedding"]


class Olmo3DualRotaryEmbedding(RotaryEmbedding):
    def __init__(self, config, *, rotary_base: float, rotary_percent: float = 1.0, cp_group=None):
        # HF OLMo-3 rotates the full head dim (partial_rotary_factor 1); a partial rotary would be
        # a different model, so refuse it instead of ignoring it.
        if rotary_percent is not None and float(rotary_percent) != 1.0:
            raise ValueError(
                f"OLMo-3 RoPE covers the full head dim; rotary_percent must be 1.0, got {rotary_percent}"
            )
        super().__init__(
            kv_channels=config.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=False,
            rotary_base=rotary_base,
            use_cpu_initialization=config.use_cpu_initialization,
            cp_group=cp_group,
        )
        self.olmo3_yarn = None
        if config.olmo3_yarn_enabled:
            self.olmo3_yarn = YarnRotaryEmbedding(
                kv_channels=config.kv_channels,
                rotary_percent=1.0,
                rotary_interleaved=False,
                rotary_base=rotary_base,
                use_cpu_initialization=config.use_cpu_initialization,
                scaling_factor=float(config.olmo3_yarn_rotary_scaling_factor),
                original_max_position_embeddings=int(config.olmo3_yarn_original_max_position_embeddings),
                beta_fast=float(config.olmo3_yarn_beta_fast),
                beta_slow=float(config.olmo3_yarn_beta_slow),
                mscale=1.0,
                mscale_all_dim=0.0,
                correction_range_round_to_int=True,
                cp_group=cp_group,
            )

    def get_emb(self, max_seq_len: int, offset: int = 0) -> Tensor:
        """Stacked ``[2, s, 1, 1, d]`` table (index 0 plain RoPE, index 1 YaRN), before CP slicing."""
        rope = super().get_emb(max_seq_len, offset)
        if self.olmo3_yarn is not None:
            yarn, _ = self.olmo3_yarn.get_emb(max_seq_len, offset)
            yarn = yarn.to(rope.dtype)
        else:
            yarn = rope
        return torch.stack([rope, yarn], dim=0)

    def get_cos_sin(self, max_seq_len: int, offset: int = 0):
        raise NotImplementedError(
            "OLMo-3 uses a different RoPE table per layer type; the single cos/sin cache used by "
            "flash_decode / FlashInfer fused RoPE is not supported. Disable flash_decode."
        )

    @lru_cache(maxsize=32)
    def forward(self, max_seq_len: int, offset: int = 0, packed_seq: bool = False, cp_group=None) -> Tensor:
        if cp_group is None:
            cp_group = self.cp_group
        emb = self.get_emb(max_seq_len, offset)
        if cp_group is not None and cp_group.size() > 1 and not packed_seq:
            emb = get_pos_emb_on_this_cp_rank(emb, 1, cp_group)
        return emb
