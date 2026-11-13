###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 GPT model: upstream ``GPTModel`` with a per-layer (RoPE | YaRN) table."""

from megatron.core.models.gpt.gpt_model import GPTModel

from primus.backends.megatron.core.models.olmo3.olmo3_rotary import (
    Olmo3DualRotaryEmbedding,
)

__all__ = ["Olmo3Model"]


class Olmo3Model(GPTModel):
    def __init__(self, config, *args, rotary_base: float = 500000.0, rotary_percent: float = 1.0, **kwargs):
        kwargs.setdefault("position_embedding_type", "rope")
        if kwargs["position_embedding_type"] != "rope":
            raise ValueError(
                f"OLMo-3 uses position_embedding_type=rope, got {kwargs['position_embedding_type']!r}"
            )
        # Build as a plain "rope" GPTModel (so every "rope"-gated piece of GPTModel.__init__ runs as
        # upstream), then replace its single RotaryEmbedding (only an inv_freq buffer, no weights or
        # checkpoint keys) with the stacked per-layer-type (RoPE | YaRN) table that forward uses.
        super().__init__(config, *args, rotary_base=rotary_base, rotary_percent=rotary_percent, **kwargs)
        self.rotary_pos_emb = Olmo3DualRotaryEmbedding(
            self.config,
            rotary_base=rotary_base,
            rotary_percent=rotary_percent,
            cp_group=self.pg_collection.cp,
        )
