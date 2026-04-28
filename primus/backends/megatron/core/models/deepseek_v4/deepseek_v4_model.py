###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek-V4 top-level model.

Phase 3 layout:

* :class:`DeepseekV4Model` subclasses :class:`megatron.core.models.gpt.GPTModel`
  so we keep the embedding / output-head / position-embedding / RoPE
  plumbing for free.
* After ``super().__init__`` runs (which builds a stock ``TransformerBlock``
  as ``self.decoder``), we **replace** ``self.decoder`` with
  :class:`DeepseekV4TransformerBlock`. In Phase 3 the V4 block is a
  transparent subclass of ``TransformerBlock`` -- it stashes V4 config
  fields (``hc_mult``, ``compress_ratios``, ``attn_sink``, ...) for the
  Phase-4 patches to consume.

Phase 4 will plug HC + per-layer hybrid attention into the V4 block.
Phase 5 will add the V4 MoE / hash routing / clamped SwiGLU and the V4 MTP
block.
"""

from typing import Optional

from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_block import (
    DeepseekV4TransformerBlock,
)


class DeepseekV4Model(GPTModel):
    """V4 model class. See module docstring for the rollout plan."""

    def __init__(
        self,
        *args,
        transformer_layer_spec: ModuleSpec,
        vp_stage: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            transformer_layer_spec=transformer_layer_spec,
            vp_stage=vp_stage,
            **kwargs,
        )

        # GPTModel.__init__ already built a stock ``TransformerBlock`` and
        # stored it on ``self.decoder``. Swap in the V4 block -- it has the
        # same call signature so ``GPTModel.forward`` keeps working.
        self.decoder = DeepseekV4TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=self.pg_collection,
            vp_stage=vp_stage,
        )


__all__ = ["DeepseekV4Model"]
