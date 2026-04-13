###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    _yarn_get_concentration_factor_from_config,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.utils import (
    deprecate_inference_params,
    is_fa_min_version,
    nvtx_range_pop,
    nvtx_range_push,
)

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from flashattn_hopper.flash_attn_interface import _flash_attn_forward  # noqa: F401

    HAVE_FA3 = True
except Exception:
    HAVE_FA3 = False

try:
    from transformer_engine.pytorch.attention.rope import apply_fused_qkv_rotary_pos_emb

    HAVE_FUSED_QKV_ROPE = True
except ImportError:
    HAVE_FUSED_QKV_ROPE = False


def mha_forward_part_a(
    self,
    hidden_states: Tensor,
    attention_mask: Tensor,
    key_value_states: Optional[Tensor] = None,
    inference_context: Optional[BaseInferenceContext] = None,
    rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    rotary_pos_cos: Optional[Tensor] = None,
    rotary_pos_sin: Optional[Tensor] = None,
    rotary_pos_cos_sin: Optional[Tensor] = None,
    attention_bias: Optional[Tensor] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    sequence_len_offset: Optional[int] = None,
    *,
    inference_params: Optional[BaseInferenceContext] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Perform a forward pass through the attention module.

    Args:
        hidden_states (Tensor): Hidden states.
        attention_mask (Tensor): Attention mask.
        key_value_states (Optional[Tensor]): Key/value states (for cross attention).
        inference_context (Optional[BaseInferenceContext]): Inference context that manages
            KV cache.
        rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
            embedding tensor(s).
        rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
        rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
        rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
        Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
        attention_bias (Optional[Tensor]): Attention bias.
        packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
        sequence_len_offset (Optional[int]): Sequence length offset used for
            inference CUDA graphs.

    Return:
        (Tuple[Tensor, Tensor]) Attention output and bias.

    """
    # Check if we need to skip RoPE
    # no_rope is 0-indexed array and self.layer_number is 1-indexed
    no_rope = (
        self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
    )
    if no_rope:
        rotary_pos_emb = None

    inference_context = deprecate_inference_params(inference_context, inference_params)

    if inference_context and inference_context.is_dynamic_batching():
        assert HAVE_FA3 or is_fa_min_version(
            "2.7.3"
        ), "flash attn verion v2.7.3 and above is required for dynamic batching."

    # hidden_states: [sq, b, h]
    is_inference_mode = inference_context is not None and not self.training
    # is_using_flash_decode - True is we are using the static inference engine with flash decode
    is_using_flash_decode = is_inference_mode and self.config.flash_decode
    # is_using_flashinfer_rope - True if we are using the dynamic inference engine
    # with flashinfer fused rope
    is_using_flashinfer_rope = is_inference_mode and (
        not inference_context.is_static_batching()
        and inference_context.use_flashinfer_fused_rope
    )
    if is_using_flash_decode or is_using_flashinfer_rope:
        # flash decode and flash-infer fused rope use rotary_pos_cos and rotary_pos_sin
        rotary_pos_emb = None
    else:
        assert rotary_pos_cos is None and rotary_pos_sin is None

    # For self attention we just duplicate the rotary_pos_emb if it isn't already
    if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
        rotary_pos_emb = (rotary_pos_emb,) * 2

    # =====================
    # Query, Key, and Value
    # =====================
    # Get the query, key and value tensors based on the type of attention -
    # self or cross attn.
    nvtx_range_push(suffix="qkv")
    split_qkv = (self.attention_type == "cross") or not all(
        [
            not self.config.test_mode,
            self.config.fused_single_qkv_rope,
            inference_context is None,
            packed_seq_params is None,
            (
                rotary_pos_emb is not None
                and rotary_pos_emb[0] is not None
                and rotary_pos_emb[1] is not None
            ),
            not self.config.flash_decode,
            HAVE_FUSED_QKV_ROPE,
            self.q_layernorm is None or isinstance(self.q_layernorm, IdentityOp),
            self.k_layernorm is None or isinstance(self.k_layernorm, IdentityOp),
        ]
    )
    # Check if fused_single_qkv_rope is requested but either unavailable or not
    # supported for the current use case.
    if self.attention_type != "cross":
        assert not (
            self.config.fused_single_qkv_rope and split_qkv
        ), "fused_single_qkv_rope requested but not available/supported for the config."

    qkv_output = self.get_query_key_value_tensors(
        hidden_states, key_value_states, split_qkv=split_qkv
    )
    attn_mask_type = self.attn_mask_type
    block_table = None
    if split_qkv:
        query, key, value = qkv_output
    else:
        mixed_qkv, qkv_split_arg_list = qkv_output
    nvtx_range_pop(suffix="qkv")

    # ===================================================
    # Adjust key, value, and rotary_pos_emb for inference
    # ===================================================

    in_decode_mode = (
        inference_context is not None
        and inference_context.is_decode_only()
        and not self.training
    )

    # This branch only runs in the decode phase of flash decoding and returns after the linear
    # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
    nvtx_range_push(suffix="adjust_key_value")
    if in_decode_mode and self.config.flash_decode:
        assert self.layer_number in inference_context.key_value_memory_dict
        assert inference_context.sequence_len_offset is not None
        inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[
            self.layer_number
        ]
        output = self.flash_decode(
            sequence_len_offset=sequence_len_offset,
            query_layer=query,
            key_layer=key,
            value_layer=value,
            inference_key_memory=inference_key_memory,
            inference_value_memory=inference_value_memory,
            rotary_cos=rotary_pos_cos,
            rotary_sin=rotary_pos_sin,
            rotary_interleaved=self.config.rotary_interleaved,
        )
        out = output.transpose(0, 1).contiguous()
        context_layer = out.view(out.size(0), out.size(1), -1)
        output, bias = self.linear_proj(context_layer)
        return output, bias

    if (
        in_decode_mode
        and self.config.enable_cuda_graph
        and self.config.cuda_graph_scope != "full_iteration"
        and inference_context.is_static_batching()
    ):
        raise ValueError(f"CUDA graphs must use flash decode with static batching!")

    if split_qkv:
        query, key, value, rotary_pos_emb, attn_mask_type, block_table = (
            self._adjust_key_value_for_inference(
                inference_context,
                query,
                key,
                value,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                rotary_pos_cos_sin,
                sequence_len_offset,
            )
        )

    if packed_seq_params is not None:
        query = query.squeeze(1)
        key = key.squeeze(1)
        value = value.squeeze(1)
    nvtx_range_pop(suffix="adjust_key_value")

    # ================================================
    # relative positional embedding (rotary embedding)
    # ================================================
    nvtx_range_push(suffix="rotary_pos_emb")
    if rotary_pos_emb is not None and (
        not self.config.flash_decode or inference_context is None
    ):
        q_pos_emb, k_pos_emb = rotary_pos_emb

        if packed_seq_params is not None:
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        if split_qkv:
            if q_pos_emb is not None:
                # TODO VIJAY: simplify
                if inference_context is None or inference_context.is_static_batching():
                    query = apply_rotary_pos_emb(
                        query,
                        q_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        mscale=_yarn_get_concentration_factor_from_config(self.config),
                        cp_group=self.pg_collection.cp,
                    )
                else:
                    query = inference_context.apply_rotary_emb_query(
                        query, q_pos_emb, self.config, cu_seqlens_q, self.pg_collection.cp
                    )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key,
                    k_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    mscale=_yarn_get_concentration_factor_from_config(self.config),
                    cp_group=self.pg_collection.cp,
                )
        else:
            query, key, value = apply_fused_qkv_rotary_pos_emb(
                mixed_qkv, q_pos_emb, k_pos_emb, qkv_split_arg_list
            )

        # TODO, can apply positional embedding to value_layer so it has
        # absolute positional embedding.
        # otherwise, only relative positional embedding takes effect
        # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
    nvtx_range_pop(suffix="rotary_pos_emb")

    # ==================================
    # core attention computation (split here to forward_b)
    # ==================================
    return query, key, value, attn_mask_type, block_table, inference_context





def mha_forward_part_b(
    self,
    query,
    key,
    value,
    attn_mask_type,
    block_table,
    attention_mask: Tensor,
    key_value_states: Optional[Tensor] = None,
    inference_context: Optional[BaseInferenceContext] = None,
    rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    rotary_pos_cos: Optional[Tensor] = None,
    rotary_pos_sin: Optional[Tensor] = None,
    rotary_pos_cos_sin: Optional[Tensor] = None,
    attention_bias: Optional[Tensor] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    sequence_len_offset: Optional[int] = None,
    *,
    inference_params: Optional[BaseInferenceContext] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Perform a forward pass through the attention module.

    Args:
        hidden_states (Tensor): Hidden states.
        attention_mask (Tensor): Attention mask.
        key_value_states (Optional[Tensor]): Key/value states (for cross attention).
        inference_context (Optional[BaseInferenceContext]): Inference context that manages
            KV cache.
        rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
            embedding tensor(s).
        rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
        rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
        rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
        Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
        attention_bias (Optional[Tensor]): Attention bias.
        packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
        sequence_len_offset (Optional[int]): Sequence length offset used for
            inference CUDA graphs.

    Return:
        (Tuple[Tensor, Tensor]) Attention output and bias.

    """
    # ==================================
    # core attention computation
    # ==================================

    nvtx_range_push(suffix="core_attention")
    if self.checkpoint_core_attention and self.training:
        core_attn_out = self._checkpointed_attention_forward(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
    else:
        if inference_context is None or inference_context.is_static_batching():
            # Static batching attention kernel.
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        else:
            # Dynamic batching attention kernel.
            q, k, v = (query, key, value)
            cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
            cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

            core_attn_out = self.flash_decode_and_prefill(
                q,
                k,
                v,
                max_seqlen_q,
                max_seqlen_k,
                cu_query_lengths,
                cu_kv_lengths,
                kv_lengths,
                block_table,
            )
            core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

    if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
        # reshape to same output shape as unpacked case
        # (t, np, hn) -> (t, b=1, h=np*hn)
        # t is the pack size = sum (sq_i)
        # note that batch is a dummy dimension in the packed case
        core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
    nvtx_range_pop(suffix="core_attention")

    # =================
    # Output. [sq, b, h]
    # =================

    nvtx_range_push(suffix="linear_proj")
    output, bias = self.linear_proj(core_attn_out)
    nvtx_range_pop(suffix="linear_proj")

    return output, bias




def mla_forward_part_a(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
):
    """Forward pass for multi-latent attention"""
    assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
    assert attention_bias is None, "Attention bias should not be passed into MLA."
    assert (
            rotary_pos_cos is None and rotary_pos_sin is None
    ), "MLA does not support Flash Decoding"
    assert not (
            self.training and self.cache_mla_latents
    ), "cache_mla_latents conflicts with training."

    # hidden_states: [sq, b, h]

    inference_context = deprecate_inference_params(inference_context, inference_params)
    if inference_context and not inference_context.is_static_batching():
        assert (
            self.config.cache_mla_latents
        ), "currently to use dynamic backend for MLA cache mla latents must be true"

    if self.config.cache_mla_latents:
        self.prepare_for_absorption()

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
    query, key, value, _, attn_mask_type, block_table = self._adjust_key_value_for_inference(
        inference_context, query, key, value, rotary_pos_emb=None
    )

    # TODO: Currently, TE can only accept contiguous tensors for MLA
    query = query.contiguous()
    key = key.contiguous()

    # Value is none during decode for absorption
    if value is not None:
        value = value.contiguous()
    return query, key, value, attn_mask_type, block_table, inference_context


def mla_forward_part_b(
    self,
    query,
    key,
    value,
    attn_mask_type,
    block_table,
    attention_mask,
    key_value_states=None,
    inference_context=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    rotary_pos_cos_sin=None,
    attention_bias=None,
    packed_seq_params=None,
    position_ids=None,
    sequence_len_offset=None,
    *,
    inference_params=None,
):
    """Forward pass for multi-latent attention"""
    assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
    assert attention_bias is None, "Attention bias should not be passed into MLA."
    assert (
        rotary_pos_cos is None and rotary_pos_sin is None
    ), "MLA does not support Flash Decoding"
    assert not (
        self.training and self.cache_mla_latents
    ), "cache_mla_latents conflicts with training."
    inference_context = deprecate_inference_params(inference_context, inference_params)
    if inference_context and not inference_context.is_static_batching():
        assert (
            self.config.cache_mla_latents
        ), "currently to use dynamic backend for MLA cache mla latents must be true"

    # ==================================
    # core attention computation
    # ==================================
    # Need corresponding TE change
    if self.checkpoint_core_attention and self.training:
        core_attn_out = self._checkpointed_attention_forward(
            query, key, value, attention_mask, packed_seq_params=packed_seq_params
        )
    else:
        if inference_context is None or inference_context.is_static_batching():
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=attn_mask_type,
            )
        elif self.cache_mla_latents:
            # Dynamic batching attention kernel.
            q, k, v = (query, key, value)
            cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
            cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()

            core_attn_out = self.flash_decode_and_prefill(
                q,
                k,
                v,
                max_seqlen_q,
                max_seqlen_k,
                cu_query_lengths,
                cu_kv_lengths,
                kv_lengths,
                block_table,
            )
            # Only rearrange if not in absorption mode (Flash MLA handles format correctly)
            if not inference_context.is_decode_only():
                core_attn_out = rearrange(core_attn_out, 's b h d -> s b (h d)')

    # We are doing absorption with cache mla latents and decode mode.
    if self.cache_mla_latents and inference_context.is_decode_only():
        # core_attn_out = self.self.up_v_layer(core_attn_out)
        core_attn_out = torch.einsum("sbhc,hdc->sbhd", core_attn_out, self.up_v_weight)
        core_attn_out = core_attn_out.contiguous()

        # Flatten back: [seq, batch, num_heads * v_head_dim]
        core_attn_out = core_attn_out.view(core_attn_out.size(0), core_attn_out.size(1), -1)

    if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
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


def mla_forward_combined(self, *args, **kwargs):
    query, key, value, attn_mask_type, block_table, inference_context = self.forward_a(*args, **kwargs)
    kwargs['inference_context'] = inference_context
    args_forward_b = [query, key, value, attn_mask_type, block_table] + list(args[1:])
    forward_b_outputs = self.forward_b(*args_forward_b, **kwargs)
    return forward_b_outputs