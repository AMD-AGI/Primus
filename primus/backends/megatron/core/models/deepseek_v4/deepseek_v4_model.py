###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek-V4 top-level model.

This model intentionally subclasses :class:`LanguageModule` (not GPTModel)
so DeepSeek-V4 no longer depends on GPT's internal TransformerBlock
construction path.
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

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_mtp import (
    DeepseekV4MTPBlock,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)


class DeepseekV4Model(LanguageModule):
    """DeepSeek-V4 language model rooted on LanguageModule."""

    def __init__(
        self,
        config: DeepSeekV4TransformerConfig,
        transformer_layer_spec: Union[ModuleSpec, type],
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            "learned_absolute",
            "rope",
            "mrope",
            "yarn",
            "none",
        ] = "none",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        scatter_embedding_sequence_parallel: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        **_kwargs,
    ) -> None:
        del rotary_percent, rotary_base, rope_scaling
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

        if hasattr(self.config, "position_embedding_type"):
            self.position_embedding_type = self.config.position_embedding_type
        else:
            self.position_embedding_type = position_embedding_type

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=self.position_embedding_type,
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

        # Optional V4 custom MTP block. This path is experimental and currently
        # orthogonal to Megatron's native MTP loss pipeline.
        mtp_num_layers = int(getattr(self.config, "mtp_num_layers", 0) or 0)
        use_custom_v4_mtp = bool(getattr(self.config, "v4_use_custom_mtp_block", False))
        if use_custom_v4_mtp and mtp_num_layers > 0 and self.post_process:
            mtp_compress_ratios = getattr(self.config, "mtp_compress_ratios", None)
            decoder_rope = getattr(self.decoder, "rope", None)
            if decoder_rope is None:
                raise ValueError(
                    "v4_use_custom_mtp_block requires decoder.rope. "
                    "Please ensure transformer_layer_spec builds a decoder with DualRoPE support."
                )
            self.mtp_block = DeepseekV4MTPBlock(
                config=self.config,
                rope=decoder_rope,
                mtp_num_layers=mtp_num_layers,
                mtp_compress_ratios=mtp_compress_ratios,
            )
        else:
            self.mtp_block = None

        if self.post_process:
            if getattr(self.config, "defer_embedding_wgrad_compute", False):
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                self.config.hidden_size,
                self.vocab_size,
                config=self.config,
                init_method=(
                    self.config.embedding_init_method
                    if getattr(self.config, "use_mup", False) and not self.share_embeddings_and_output_weights
                    else self.config.init_method
                ),
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
        """Pipeline-parallel hook to set decoder input tensor."""
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
        runtime_gather_output: Optional[bool] = None,
        **kwargs,
    ):
        """Forward pass for DeepSeek-V4.

        ``input_ids`` are additionally stashed on the decoder for hash-routed
        MoE layers.
        """
        if decoder_input is None:
            if self.pre_process:
                if input_ids is None:
                    raise ValueError("input_ids must be provided when pre_process=True.")
                if position_ids is None:
                    batch, seq = input_ids.shape
                    position_ids = (
                        input_ids.new_arange(seq, dtype=input_ids.dtype).unsqueeze(0).expand(batch, -1)
                    )
                decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
            else:
                decoder_input = None

        decoder = getattr(self, "decoder", None)
        if decoder is not None:
            decoder._v4_token_ids = input_ids
        try:
            hidden_states = self.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                **kwargs,
            )
        finally:
            if decoder is not None:
                decoder._v4_token_ids = None

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


__all__ = ["DeepseekV4Model"]
