###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling top-level model, rooted on :class:`LanguageModule`.

Module tree (names are what the HF importer writes)::

    embedding.word_embeddings     token embedding, no positional embedding
    embed_norm                    RMSNorm on the embeddings
    decoder.layers.{i}            InklingTransformerLayer
    decoder.final_layernorm       RMSNorm
    output_layer                  untied LM head (hidden / logits_mup_width_multiplier)

Building with ``config.init_model_with_meta_device`` allocates nothing:
TE modules honour the flag directly, and the embedding and LM head (plain
Megatron layers, which pin the CUDA device unless ``use_cpu_initialization``
is set) are built from a copy of the config with CPU initialisation on, so
their ``torch.empty`` follows the surrounding ``torch.device("meta")``.

The multi-token-prediction layers are a later addition (P4); the importer
skips them.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_block import (
    TransformerBlock,
    TransformerBlockSubmodules,
)
from torch import Tensor

from primus.backends.megatron.core.models.inkling.inkling_transformer_config import (
    InklingTransformerConfig,
)

__all__ = ["InklingModel"]


class InklingModel(LanguageModule):
    def __init__(
        self,
        config: InklingTransformerConfig,
        decoder_spec: TransformerBlockSubmodules,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        scatter_embedding_sequence_parallel: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        **_kwargs,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)
        if share_embeddings_and_output_weights:
            raise ValueError("Inkling's embedding and LM head are separate tensors; keep them untied")
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = False
        self.vp_stage = vp_stage
        self.model_type = ModelType.encoder_or_decoder
        self.position_embedding_type = "none"

        vocab_config = config
        if config.init_model_with_meta_device:
            vocab_config = copy.copy(config)
            vocab_config.use_cpu_initialization = True

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=vocab_config,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                position_embedding_type="none",
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                tp_group=self.pg_collection.tp,
            )
            self.embed_norm = (
                TENorm(config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon)
                if config.use_embed_norm
                else None
            )

        self.decoder = TransformerBlock(
            config=config,
            spec=decoder_spec,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=self.pg_collection,
            vp_stage=vp_stage,
        )

        if self.post_process:
            if config.defer_embedding_wgrad_compute:
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                vocab_size,
                config=vocab_config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not parallel_output,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
                tp_group=self.pg_collection.tp,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for decoder-only models"
        self.decoder.set_input_tensor(input_tensor[0])

    def _mask_padded_vocab(self, logits: Tensor) -> Tensor:
        """Push the LM-head rows past ``unpadded_vocab_size`` out of the softmax.

        The reference slices them off; masking keeps Megatron's padded,
        vocab-parallel logits shape.
        """
        unpadded = self.config.unpadded_vocab_size
        if unpadded is None or unpadded >= self.vocab_size:
            return logits
        local = logits.shape[-1]
        start = 0 if local == self.vocab_size else self.pg_collection.tp.rank() * local
        if start + local <= unpadded:
            return logits
        token = torch.arange(start, start + local, device=logits.device)
        return logits.masked_fill(token >= unpadded, torch.finfo(logits.dtype).min)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor],
        attention_mask: Optional[Tensor],
        decoder_input: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        inference_context=None,
        packed_seq_params=None,
        extra_block_kwargs: Optional[dict] = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params=None,
        loss_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Logits ``[b, s, vocab]`` without ``labels``, else the per-token loss ``[b, s]``.

        ``position_ids`` is unused: position enters only through the
        relative-logit bias, which is a function of the sequence index.
        """
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError("Inkling has no KV-cache inference path")
        if decoder_input is None and self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=None)
            if self.embed_norm is not None:
                decoder_input = self.embed_norm(decoder_input)

        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
            **(extra_block_kwargs or {}),
        )
        if not self.post_process:
            return hidden_states

        hidden_states = hidden_states / self.config.logits_mup_width_multiplier
        logits, _ = self.output_layer(hidden_states, runtime_gather_output=runtime_gather_output)
        logits = self._mask_padded_vocab(logits)
        if labels is None:
            return logits.transpose(0, 1).contiguous()
        return self.compute_language_model_loss(labels, logits)
