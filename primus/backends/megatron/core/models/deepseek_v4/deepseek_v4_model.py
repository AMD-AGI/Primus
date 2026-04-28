###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek-V4 top-level model.

Phase 4 layout:

* ``DeepseekV4Model`` subclasses ``GPTModel`` so we keep the embedding /
  output-head / position-embedding / RoPE plumbing for free.
* After ``super().__init__`` runs (which builds a stock ``TransformerBlock``
  as ``self.decoder``), we **replace** ``self.decoder`` with
  :class:`DeepseekV4TransformerBlock` — a standalone module that owns the
  V4-specific hybrid attention dispatch and ``hc_mult`` parallel hidden
  streams.
* The V4 config fields (``hc_mult``, ``compress_ratios``, ``attn_sink``,
  ``num_hash_layers``, ``swiglu_limit``, ...) flow from yaml → ``args`` →
  ``backend_args`` via Primus's ``merge_namespace`` step (see
  ``primus/core/runtime/train_runtime.py:_initialize_trainer``), then onto
  ``config`` because the V4 builder calls ``core_transformer_config_from_args``
  which picks up unknown attrs. Both the model class and the V4 block read
  these via ``getattr(config, ..., default)``.

Phase 5 will:
* Override ``__init__`` further to instantiate a V4-specific MTP block
  (separate ``HyperHead`` per MTP layer).
* Plug in V4's hash-routed MoE for the first ``num_hash_layers`` MoE layers.

Phase 6 (deferred) will:
* Skip the intermediate stock ``TransformerBlock`` allocation entirely
  (saves init memory at large scale).
* Wire PP / TP / EP integration so the V4 block participates in Megatron's
  distributed framework.
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

        # GPTModel.__init__ has already built a stock ``TransformerBlock`` and
        # stored it on ``self.decoder``. Swap in the V4 block — it has the
        # same call signature so ``GPTModel.forward`` keeps working.
        self.decoder = DeepseekV4TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=getattr(self, "pg_collection", None),
            vp_stage=vp_stage,
        )


__all__ = ["DeepseekV4Model"]
