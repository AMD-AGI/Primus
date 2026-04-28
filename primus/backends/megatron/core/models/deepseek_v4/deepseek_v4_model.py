###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek-V4 top-level model.

Phase 5 layout:

* ``DeepseekV4Model`` subclasses ``GPTModel`` so we keep the embedding /
  output-head / position-embedding / RoPE plumbing for free.
* After ``super().__init__`` runs (which builds a stock ``TransformerBlock``
  as ``self.decoder``), we **replace** ``self.decoder`` with
  :class:`DeepseekV4TransformerBlock` — a standalone module that owns the
  V4-specific hybrid attention dispatch, ``hc_mult`` parallel hidden
  streams, and (Phase 5) per-layer V4 MoE FFN.
* If ``mtp_num_layers > 0`` (V4-Flash = 1), we additionally build a
  :class:`DeepseekV4MTPBlock` and store it on ``self.mtp_block``. The
  MTP block owns its **own** HyperHead per layer (V4 design), but
  shares ``rope`` with the main decoder. Wiring the MTP outputs into
  the loss path is a Phase 6 concern; Phase 5 only stands the module up
  so it can be unit-tested standalone.
* The V4 config fields (``hc_mult``, ``compress_ratios``, ``attn_sink``,
  ``num_hash_layers``, ``swiglu_limit``, MoE / hash-router knobs ...)
  flow from yaml → ``args`` → ``backend_args`` via Primus's
  ``merge_namespace`` step (see
  ``primus/core/runtime/train_runtime.py:_initialize_trainer``), then onto
  ``config`` because the V4 builder calls ``core_transformer_config_from_args``
  which picks up unknown attrs. Both the model class and the V4 block read
  these via ``getattr(config, ..., default)``.
* ``forward`` is overridden so we can stash ``input_ids`` on the V4 block
  before delegating to ``GPTModel.forward``. The V4 block needs token ids
  for its hash-routed MoE layers (``layer_idx < num_hash_layers``).

Phase 6 (deferred) will:
* Skip the intermediate stock ``TransformerBlock`` allocation entirely
  (saves init memory at large scale).
* Wire MTP outputs into the loss path.
* Wire PP / TP / EP integration so the V4 block participates in Megatron's
  distributed framework.
"""

from typing import Optional

from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_block import (
    DeepseekV4TransformerBlock,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_mtp import (
    DeepseekV4MTPBlock,
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

        # ---- V4 MTP block (P5) ----
        mtp_num_layers = int(getattr(self.config, "mtp_num_layers", 0) or 0)
        if mtp_num_layers > 0 and self.post_process:
            mtp_compress_ratios = getattr(self.config, "mtp_compress_ratios", None)
            self.mtp_block = DeepseekV4MTPBlock(
                config=self.config,
                rope=self.decoder.rope,
                mtp_num_layers=mtp_num_layers,
                mtp_compress_ratios=mtp_compress_ratios,
            )
        else:
            self.mtp_block = None

    def forward(self, input_ids, *args, **kwargs):
        """Cache ``input_ids`` on the V4 block for hash-routed MoE layers.

        ``GPTModel.forward`` consumes ``input_ids`` itself for the embedding
        on the first PP stage; on later stages it is ``None``. We stash a
        local copy on ``self.decoder`` before super-forward so the V4 block
        can pick it up via ``getattr(self, "_v4_token_ids", None)``.

        Cross-PP propagation of ``input_ids`` is a Phase 6 concern.
        """
        decoder = getattr(self, "decoder", None)
        if decoder is not None:
            decoder._v4_token_ids = input_ids
        try:
            return super().forward(input_ids, *args, **kwargs)
        finally:
            if decoder is not None:
                decoder._v4_token_ids = None


__all__ = ["DeepseekV4Model"]
