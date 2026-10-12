###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GLM-5.3 (``glm5_next``) top-level language model.

Rooted on :class:`LanguageModule` like ``DeepseekV4Model``: word embedding (no
position embedding -- the model is NoPE), the mHC decoder block, and an untied
vocab-parallel output layer. The released checkpoint's single MTP layer is not
built.
"""

from typing import Literal, Optional, Union

from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from torch import Tensor

from primus.backends.megatron.core.models.glm5_next.glm5_next_transformer_config import (
    Glm5NextTransformerConfig,
)


class Glm5NextModel(LanguageModule):
    """GLM-5.3 language model."""

    def __init__(
        self,
        config: Glm5NextTransformerConfig,
        transformer_layer_spec: Union[ModuleSpec, type],
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal["none"] = "none",
        scatter_embedding_sequence_parallel: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        **_kwargs,
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
        assert not int(
            getattr(config, "mtp_num_layers", 0) or 0
        ), "GLM-5.3 MTP is not supported in Primus yet."

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type="none",
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                tp_group=self.pg_collection.tp,
            )

        self.decoder = build_module(
            transformer_layer_spec,
            config=self.config,
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
        **kwargs,
    ):
        if decoder_input is None and self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=None)

        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
        )
        if not self.post_process:
            return hidden_states

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )
        if labels is None:
            return logits.transpose(0, 1).contiguous()
        return self.compute_language_model_loss(labels, logits)


__all__ = ["Glm5NextModel"]
