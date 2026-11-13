###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mistral Large 4 (``mistral_large4_text``) language model.

A :class:`LanguageModule` like Primus's GLM-5.3 / DeepSeek-V4 models: word embedding (rope
is applied inside the attention from the block's YaRN + LeRoPE table, so the embedding has
no position term), the Mistral Large 4 decoder block, a vocab-parallel output layer (untied in the release),
and the final logit softcap ``cap * tanh(logits / cap)`` applied (with autograd) before the
loss. The vision tower, projector and DSpark MTP head of the release are not built.
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

from primus.backends.megatron.core.models.mistral_large4.mistral_large4_block import (
    MistralLarge4TransformerBlock,
)

__all__ = ["MistralLarge4Model", "softcap"]


def softcap(logits: Tensor, cap: Optional[float]) -> Tensor:
    """``cap * tanh(logits / cap)`` (differentiable); identity when ``cap`` is falsy. Computed in
    fp32 (bf16 resolution near the cap is ~0.125) and returned in fp32 for the cross-entropy, which
    upcasts the logits to fp32 anyway; no extra bf16 copy is kept."""
    if not cap:
        return logits
    return torch.tanh(logits.float() / cap) * cap


class MistralLarge4Model(LanguageModule):
    """Mistral Large 4 text decoder (see the module docstring)."""

    def __init__(
        self,
        config,
        transformer_layer_spec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        pg_collection=None,
        vp_stage: Optional[int] = None,
        **_unused,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)
        self.transformer_layer_spec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.vp_stage = vp_stage
        self.model_type = ModelType.encoder_or_decoder
        self.position_embedding_type = "none"
        if int(getattr(config, "mtp_num_layers", 0) or 0):
            raise NotImplementedError("Mistral Large 4: the DSpark draft head (MTP) is not ported yet")

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type="none",
                tp_group=self.pg_collection.tp,
            )
        self.decoder = MistralLarge4TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=self.pg_collection,
            vp_stage=vp_stage,
        )
        if self.post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                self.config.hidden_size,
                self.vocab_size,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process and self.share_embeddings_and_output_weights,
                tp_group=self.pg_collection.tp,
            )
        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for decoder-only models"
        self.decoder.set_input_tensor(input_tensor[0])

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
        padding_mask: Optional[Tensor] = None,
        **kwargs,
    ):
        if decoder_input is None and self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=None)
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
            position_ids=position_ids,  # checked against the block's own rope / SSMax positions
        )
        if not self.post_process:
            return hidden_states
        output_weight = (
            self.shared_embedding_or_output_weight() if self.share_embeddings_and_output_weights else None
        )
        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )
        logits = softcap(logits, getattr(self.config, "final_logit_softcapping", None))
        if labels is None:
            return logits.transpose(0, 1).contiguous()
        return self.compute_language_model_loss(labels, logits)
