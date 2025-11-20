###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch.nn.functional as F
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import deprecate_inference_params


class PaddedMLASelfAttention(MLASelfAttention):
    """
    PaddedMLASelfAttention class

    This custom attention module is designed to address a compatibility issue observed in DeepSeek models,
    where the head dimension of QK is 192 and that of V is 128. This asymmetry prevents the use of AMD
    TransformerEngine's fused attention kernel, which requires Q, K, and V to have matching head dimensions.

    To enable fused attention and reduce memory usage, this module pads the V tensor so that all Q, K, and V
    have a uniform head dimension of 192. After padding, AMD TE's fused attention can be invoked, resulting
    in more efficient memory usage and improved performance.

    Note:
    - Padding is only applied to the V tensor when head dimension mismatch is detected.
    - This optimization is particularly important for large models like DeepSeek variants where memory savings
    have a meaningful impact on batch size and throughput.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        if self.q_head_dim > self.config.v_head_dim:
            self.core_attention = build_module(
                submodules.core_attention,
                config=self.config,
                layer_number=self.layer_number,
                attn_mask_type=self.attn_mask_type,
                attention_type=self.attention_type,
                softmax_scale=self.softmax_scale,
                k_channels=self.q_head_dim,
                v_channels=self.q_head_dim,  # pad self.config.v_head_dim,
                cp_comm_type=cp_comm_type,
                pg_collection=self.pg_collection,
            )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        """Forward pass for multi-latent attention"""
        if self.q_head_dim <= self.config.v_head_dim:
            super().forward(
                hidden_states,
                attention_mask,
                key_value_states=key_value_states,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                position_ids=position_ids,
                sequence_len_offset=sequence_len_offset,
                inference_params=inference_params,
            )

        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert rotary_pos_cos is None and rotary_pos_sin is None, "MLA does not support Flash Decoding"

        # hidden_states: [sq, b, h]

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        query, key, value = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )

        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        query, key, value, _, attn_mask_type, _ = self._adjust_key_value_for_inference(
            inference_context, query, key, value, rotary_pos_emb=None
        )

        seq_length = query.shape[0]
        batch_size = query.shape[1]
        num_heads = query.shape[2]

        # Pad value head dim
        assert self.q_head_dim > self.config.v_head_dim
        padded_dim = self.q_head_dim - self.config.v_head_dim
        # pad value to q_head_dim
        value = F.pad(value, (0, padded_dim))

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=attn_mask_type,
            )

        # unpad value head dim
        assert packed_seq_params is None
        # [s, b, n * dim] -> [s, b, n, dim=192] -> [s, b, n, dim=128] -> [s, b, n*dim]
        core_attn_out = core_attn_out.reshape(seq_length, batch_size, num_heads, self.q_head_dim)[
            ..., : self.config.v_head_dim
        ]
        core_attn_out = core_attn_out.reshape(seq_length, batch_size, num_heads * self.config.v_head_dim)

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)

        return output, bias
