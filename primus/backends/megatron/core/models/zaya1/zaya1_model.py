###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ZAYA1 language model.

The decoder is the SGLang stack in ``zaya1_modules`` (80 alternating CCA and
MoE stages for the released shape). Partial RoPE lives inside CCA, so this
module does not build a Megatron ``RotaryEmbedding``. Embeddings stay tied.
"""

from typing import Optional

import torch
from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.enums import ModelType
from torch import Tensor

from primus.backends.megatron.core.models.zaya1.zaya1_modules import ZayaStack
from primus.backends.megatron.core.models.zaya1.zaya1_transformer_config import (
    Zaya1TransformerConfig,
)

__all__ = ["Zaya1Model"]


class Zaya1Model(LanguageModule):
    """ZAYA1 pretraining model rooted on :class:`LanguageModule`."""

    def __init__(
        self,
        config: Zaya1TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = True,
        position_embedding_type: str = "rope",
        scatter_embedding_sequence_parallel: bool = False,
        pg_collection=None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)

        if position_embedding_type != "rope":
            raise ValueError(
                "ZAYA1 applies partial RoPE inside CCA and requires "
                f"position_embedding_type='rope', got {position_embedding_type!r}."
            )

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.vp_stage = vp_stage
        self.model_type = ModelType.encoder_or_decoder
        self._input_tensor = None

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                tp_group=self.pg_collection.tp,
            )

        self.decoder = ZayaStack(config)

        if self.post_process:
            self.embedding_activation_buffer = None
            self.grad_output_buffer = None
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                self.config.hidden_size,
                self.vocab_size,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
                tp_group=self.pg_collection.tp,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Pipeline hook. This port only runs with pipeline size 1."""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for decoder-only models"
        self._input_tensor = input_tensor[0]

    def forward(
        self,
        input_ids: Optional[Tensor],
        position_ids: Optional[Tensor],
        attention_mask: Optional[Tensor],
        decoder_input: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        loss_mask: Optional[Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params=None,
        **_kwargs,
    ):
        """Match ``pretrain_gpt.forward_step``: tokens, positions, mask, labels, loss_mask.

        ``loss_mask`` is applied by the caller. Packed sequences are rejected:
        the conv and the value shift run along each row and do not know document
        boundaries.
        """
        del loss_mask
        if packed_seq_params is not None:
            raise NotImplementedError(
                "ZAYA1 CCA conv and value shift run along the sequence axis of each "
                "batch row. Packed document boundaries inside a row are not masked."
            )

        if decoder_input is None and self.pre_process:
            if input_ids is None:
                raise ValueError("input_ids must be provided when pre_process=True.")
            if position_ids is None:
                batch, seq = input_ids.shape
                position_ids = torch.arange(seq, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(batch, -1)
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        elif decoder_input is None:
            decoder_input = self._input_tensor

        hidden_states = self.decoder(
            hidden_states=decoder_input,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        if not self.post_process:
            return hidden_states

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        logits, _ = self.output_layer(
            hidden_states,
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        )
        logits = self._scale_logits(logits)

        if labels is None:
            return logits.transpose(0, 1).contiguous()
        return self.compute_language_model_loss(labels, logits)
