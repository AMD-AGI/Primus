###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi K3 top-level model.

Rooted on :class:`LanguageModule` rather than ``GPTModel`` for the same
reason DeepSeek-V4 is (``deepseek_v4_model.py:10-12``): the decoder is
built from a Kimi-K3-owned spec tree, so there is nothing to gain from
GPT's ``TransformerBlock`` construction path.

Multi-Token Prediction is wired through upstream's
:class:`MultiTokenPredictionBlock` and :func:`process_mtp_loss`, the same
way ``DeepseekV4Model`` does it, and is gated on
``num_nextn_predict_layers`` / ``mtp_num_layers``. The released
``config.json`` ships ``num_nextn_predict_layers: 0`` and the released
modelling code has no MTP module, but tech-report Table 1 lists one MTP
layer, so the release was stripped rather than trained without one. See
``kimi_k3_mtp_specs.py`` for what the report specifies and what had to be
chosen.

``transformer_layer_spec`` is the whole decoder tree — a
:class:`KimiK3TransformerBlock` spec from
``kimi_k3_layer_specs.get_kimi_k3_runtime_decoder_spec`` — not a single layer
spec, matching how ``DeepseekV4Model`` is built (``deepseek_v4_model.py:120``).
"""

from typing import Literal, Optional, Union

import torch
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

from primus.backends.megatron.core.models.kimi_k3.kimi_k3_mtp_specs import (
    get_kimi_k3_mtp_block_spec,
)
from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
    KimiK3TransformerConfig,
)

__all__ = ["KimiK3Model"]


class KimiK3Model(LanguageModule):
    """Kimi K3 language model rooted on :class:`LanguageModule`."""

    def __init__(
        self,
        config: KimiK3TransformerConfig,
        transformer_layer_spec: Union[ModuleSpec, type],
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal["learned_absolute", "rope", "none"] = "none",
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

        # Kimi K3 is NoPE everywhere: the full-attention layers carry a
        # qk_pos_emb_head_dim-wide positional head but never rotate it (the
        # attention module swaps in a zero-width frequency table), and the KDA
        # layers carry position implicitly through the causal recurrence and the
        # short causal convolution. No rotary module in the stack ever rotates.
        self.position_embedding_type = getattr(
            self.config, "position_embedding_type", position_embedding_type
        )

        # Resolved before the embedding, as upstream ``GPTModel`` does
        # (``gpt_model.py:145-149``): a pipeline stage that owns MTP layers but
        # is not ``pre_process`` still needs a tied copy of the input
        # embedding, because :class:`MultiTokenPredictionLayer` embeds the
        # rolled ``input_ids`` itself. Without it
        # ``setup_embeddings_and_output_layer`` asserts on a missing
        # ``self.embedding``.
        mtp_num_layers = int(getattr(self.config, "mtp_num_layers", 0) or 0)
        self.mtp_process = False
        self.mtp_block_spec = None
        if mtp_num_layers > 0:
            self.mtp_block_spec = get_kimi_k3_mtp_block_spec(self.config, vp_stage=vp_stage)
            # ``mtp_on_this_rank`` reads ``parallel_state`` and
            # ``MultiTokenPredictionBlock`` requires a ``cp`` process group, so
            # both need a real distributed init. On a CPU spec smoke we leave
            # ``self.mtp`` unset and expose the spec through
            # ``self.mtp_block_spec``, so the wiring stays inspectable —
            # ``deepseek_v4_model.py:110-120`` does the same.
            try:
                self.mtp_process = mtp_on_this_rank(self.config, ignore_virtual=False, vp_stage=vp_stage)
            except (AssertionError, RuntimeError, AttributeError):
                self.mtp_process = False

        if self.pre_process or self.mtp_process:
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

        # Do **not** pre-assign ``self.mtp = None``. Megatron's cudagraph
        # ``set_current_microbatch`` probes ``hasattr(model, 'mtp')`` and then
        # iterates ``model.mtp.layers`` unconditionally, so the attribute must
        # exist only when MTP is live. Upstream ``GPTModel`` has the same
        # asymmetry (``gpt_model.py:218-221``).
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

        if self.pre_process or self.post_process or self.mtp_process:
            self.setup_embeddings_and_output_layer()

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Pipeline-parallel hook to set the decoder input tensor."""
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
        """Forward pass for Kimi K3.

        With MTP enabled the tail of this method mirrors
        ``GPTModel._postprocess`` (``gpt_model.py:592-629``): the decoder's
        output feeds :class:`MultiTokenPredictionBlock`, whose return value is
        the sequence-axis concatenation of the main hidden state and one
        shifted hidden state per MTP depth;
        :func:`process_mtp_loss` splits that back apart, computes each depth's
        shifted-label loss and folds it into the gradient through
        ``MTPLossAutoScaler``, and returns the main chunk for the ordinary LM
        head below.

        The MTP block therefore reads the decoder's output **after**
        ``attn_res_head`` and ``final_layernorm`` — which is what report
        §4.1.4 calls the high-level feature "the input on which the MTP layer
        was pre-trained", and what §2.2's "the final output layer then
        aggregates all N block representations" makes of the checkpoint set.
        No ``block_residual`` reaches the MTP layer, by construction: the
        decoder block returns the bare ``[s, b, h]`` tensor on
        ``post_process`` (``kimi_k3_block.py:249-250``).
        """
        if decoder_input is None and self.pre_process:
            if input_ids is None:
                raise ValueError("input_ids must be provided when pre_process=True.")
            if position_ids is None:
                batch, seq = input_ids.shape
                position_ids = (
                    torch.arange(seq, dtype=torch.long, device=input_ids.device)
                    .unsqueeze(0)
                    .expand(batch, -1)
                )
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)

        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
            **kwargs,
        )

        if self.mtp_process and getattr(self, "mtp", None) is not None:
            if input_ids is None:
                raise ValueError(
                    "Multi-Token Prediction needs input_ids on the MTP stage: each depth "
                    "embeds the ids rolled one position left "
                    "(multi_token_prediction.py:874-889)."
                )
            if position_ids is None:
                batch, seq = input_ids.shape
                position_ids = (
                    torch.arange(seq, dtype=torch.long, device=input_ids.device)
                    .unsqueeze(0)
                    .expand(batch, -1)
                )
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
                # The tied embedding: on a dedicated MTP pipeline stage
                # pre_process is False but __init__ still built one, gated on
                # ``pre_process or mtp_process``.
                embedding=getattr(self, "embedding", None),
            )

        if not self.post_process:
            return hidden_states

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if self.mtp_process and getattr(self, "mtp", None) is not None:
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
                cp_group=getattr(self.pg_collection, "cp", None),
                packed_seq_params=packed_seq_params,
                scale_logits_fn=(self._scale_logits if getattr(self.config, "use_mup", False) else None),
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
