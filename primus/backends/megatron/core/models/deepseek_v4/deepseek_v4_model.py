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
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    mtp_on_this_rank,
    process_mtp_loss,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from torch import Tensor

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_mtp_specs import (
    get_v4_mtp_block_spec,
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

        # ----- MTP block ---------------------------------------------------
        # Plan-2 P16/P17: V4 wires multi-token prediction exclusively via
        # the spec-based upstream :class:`MultiTokenPredictionBlock`,
        # built from :func:`get_v4_mtp_block_spec`. The legacy
        # primus-owned ``DeepseekV4MTPBlock`` (gated by
        # ``v4_use_custom_mtp_block`` in plan-2 P16) was retired in plan-2
        # P17; only the spec-based path remains.
        mtp_num_layers = int(getattr(self.config, "mtp_num_layers", 0) or 0)
        self.mtp_process = False
        self.mtp = None

        if mtp_num_layers > 0:
            self.mtp_block_spec = get_v4_mtp_block_spec(
                self.config,
                transformer_layer_spec=transformer_layer_spec,
                vp_stage=vp_stage,
            )
            # ``mtp_on_this_rank`` reads ``parallel_state`` and
            # :class:`MultiTokenPredictionBlock` walks ``pg_collection.cp``;
            # both require a real distributed init. On CPU smokes (no
            # ``torch.distributed``) we leave ``self.mtp`` as ``None`` and
            # surface the spec via ``self.mtp_block_spec`` so callers can
            # still inspect the MTP wiring (the spec helper itself is fully
            # CPU-testable).
            try:
                self.mtp_process = mtp_on_this_rank(self.config, ignore_virtual=False, vp_stage=vp_stage)
            except (AssertionError, RuntimeError, AttributeError):
                self.mtp_process = False
            if self.mtp_process:
                self.mtp = MultiTokenPredictionBlock(
                    config=self.config,
                    spec=self.mtp_block_spec,
                    vp_stage=vp_stage,
                    pg_collection=self.pg_collection,
                )

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
        loss_mask: Optional[Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params=None,
        **kwargs,
    ):
        """Forward pass for DeepSeek-V4.

        Plan-2 P15: ``input_ids`` are passed to the decoder as the
        ``token_ids`` forward kwarg (replacing the ``decoder._v4_token_ids``
        attribute stash). Hash-routed MoE layers consume them directly via
        the standard kwargs propagation chain
        ``model.forward -> decoder.forward -> layer.forward -> mlp.forward
        -> hash_router.forward``.

        Plan-2 P16: when ``mtp_num_layers > 0`` and the spec-based MTP
        path is enabled (default), this method runs the upstream
        :class:`MultiTokenPredictionBlock` on the post_process stage and
        feeds its concatenated output through :func:`process_mtp_loss`,
        which adds the auxiliary MTP loss term to the main LM loss.
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

        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_ids=input_ids,
            packed_seq_params=packed_seq_params,
            **kwargs,
        )

        # Run the spec-based MTP block on stages that own MTP layers.
        # Mirrors GPTModel's mtp_in_postprocess gating.
        if self.mtp_process and self.mtp is not None:
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
                embedding=self.embedding if self.pre_process else None,
            )

        if not self.post_process:
            return hidden_states

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        # Plan-2 P16: when MTP is on, ``hidden_states`` arrives as the
        # concatenation of the main-decoder hidden state plus
        # ``mtp_num_layers`` shifted MTP hidden states (along the
        # sequence axis). :func:`process_mtp_loss` splits the chunks,
        # computes the per-depth MTP loss, and returns the main hidden
        # state for the standard LM-head path below.
        mtp_num_layers = int(getattr(self.config, "mtp_num_layers", 0) or 0)
        if mtp_num_layers > 0 and self.mtp is not None:
            cp_group = getattr(self.pg_collection, "cp", None)
            hidden_states = process_mtp_loss(
                hidden_states=hidden_states,
                labels=labels,
                loss_mask=loss_mask,
                output_layer=self.output_layer,
                output_weight=output_weight,
                runtime_gather_output=runtime_gather_output,
                is_training=self.training,
                compute_language_model_loss=self.compute_language_model_loss,
                config=self.config,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
                scale_logits_fn=self._scale_logits if getattr(self.config, "use_mup", False) else None,
            )

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
