###############################################################################
# Copyright 2023–2025 Google LLC. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

from flax import nnx
from jax.sharding import Mesh

from MaxText import max_utils
from MaxText.common_types import Config, MODEL_MODE_PREFILL
from MaxText.layers import quantizations
from MaxText.layers.attentions import Attention, AttentionType
from MaxText.layers.linears import MlpBlock, Dropout
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.quantizations import AqtQuantization as Quant

from MaxText.layers.gemma2 import Gemma2DecoderLayer


class PrimusGemma2DecoderLayer(Gemma2DecoderLayer):
    def __init__(
        self,
        config: Config,
        mesh: Mesh,
        model_mode: str,
        quant: Optional[Quant] = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.mesh = mesh
        self.model_mode = model_mode
        self.quant = quant
        self.rngs = rngs

        batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
        dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

        self.pre_self_attention_norm_local = RMSNorm(
            num_features=config.emb_dim,
            dtype=config.dtype,
            weight_dtype=config.weight_dtype,
            kernel_axes=("norm",),
            rngs=self.rngs,
        )

        self.self_attention_local = Attention(
            config=config,
            num_query_heads=config.num_query_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            max_target_length=config.max_target_length,
            max_prefill_predict_length=config.max_prefill_predict_length,
            attention_kernel=config.attention,
            inputs_q_shape=dummy_inputs_shape,
            inputs_kv_shape=dummy_inputs_shape,
            mesh=self.mesh,
            dtype=config.dtype,
            weight_dtype=config.weight_dtype,
            dropout_rate=config.dropout_rate,
            float32_qk_product=config.float32_qk_product,
            float32_logits=config.float32_logits,
            quant=self.quant,
            kv_quant=quantizations.configure_kv_quant(config),
            attention_type=AttentionType.LOCAL_SLIDING,
            sliding_window_size=config.sliding_window_size,
            attn_logits_soft_cap=config.attn_logits_soft_cap,
            query_pre_attn_scalar=(config.head_dim**-0.5),
            model_mode=self.model_mode,
            rngs=self.rngs,
        )

        if config.use_post_attn_norm:
            self.post_self_attention_norm_local = RMSNorm(
                num_features=config.emb_dim,
                dtype=config.dtype,
                weight_dtype=config.weight_dtype,
                kernel_axes=("norm",),
                rngs=self.rngs,
            )

        self.pre_ffw_norm_local = RMSNorm(
            num_features=config.emb_dim,
            dtype=config.dtype,
            weight_dtype=config.weight_dtype,
            kernel_axes=("norm",),
            rngs=self.rngs,
        )

        self.mlp_local = MlpBlock(
            config=config,
            mesh=self.mesh,
            in_features=config.emb_dim,
            intermediate_dim=config.mlp_dim,
            activations=config.mlp_activations,
            intermediate_dropout_rate=config.dropout_rate,
            dtype=config.dtype,
            weight_dtype=config.weight_dtype,
            quant=self.quant,
            model_mode=self.model_mode,
            rngs=self.rngs,
        )

        if config.use_post_ffw_norm:
            self.post_ffw_norm_local = RMSNorm(
                num_features=config.emb_dim,
                dtype=config.dtype,
                weight_dtype=config.weight_dtype,
                kernel_axes=("norm",),
                rngs=self.rngs,
            )

        self.dropout = Dropout(rate=config.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)

        self.pre_self_attention_norm_global = RMSNorm(
            num_features=config.emb_dim,
            dtype=config.dtype,
            weight_dtype=config.weight_dtype,
            kernel_axes=("norm",),
            rngs=self.rngs,
        )

        self.self_attention_global = Attention(
            config=config,
            num_query_heads=config.num_query_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            max_target_length=config.max_target_length,
            max_prefill_predict_length=config.max_prefill_predict_length,
            attention_kernel=config.attention,
            inputs_q_shape=dummy_inputs_shape,
            inputs_kv_shape=dummy_inputs_shape,
            mesh=self.mesh,
            dtype=config.dtype,
            weight_dtype=config.weight_dtype,
            dropout_rate=config.dropout_rate,
            float32_qk_product=True,
            float32_logits=True,
            quant=self.quant,
            kv_quant=quantizations.configure_kv_quant(config),
            attention_type=AttentionType.GLOBAL,
            attn_logits_soft_cap=config.attn_logits_soft_cap,
            query_pre_attn_scalar=(config.head_dim**-0.5),
            model_mode=model_mode,
            rngs=self.rngs,
        )

        if config.use_post_attn_norm:
            self.post_self_attention_norm_global = RMSNorm(
                num_features=config.emb_dim,
                dtype=config.dtype,
                weight_dtype=config.weight_dtype,
                kernel_axes=("norm",),
                rngs=self.rngs,
            )

        self.pre_ffw_norm_global = RMSNorm(
            num_features=config.emb_dim,
            dtype=config.dtype,
            weight_dtype=config.weight_dtype,
            kernel_axes=("norm",),
            rngs=self.rngs,
        )

        self.mlp_global = MlpBlock(
            config=config,
            mesh=self.mesh,
            in_features=config.emb_dim,
            intermediate_dim=config.mlp_dim,
            activations=config.mlp_activations,
            intermediate_dropout_rate=config.dropout_rate,
            dtype=config.dtype,
            weight_dtype=config.weight_dtype,
            quant=self.quant,
            model_mode=self.model_mode,
            rngs=self.rngs,
        )

        if config.use_post_ffw_norm:
            self.post_ffw_norm_global = RMSNorm(
                num_features=config.emb_dim,
                dtype=config.dtype,
                weight_dtype=config.weight_dtype,
                kernel_axes=("norm",),
                rngs=self.rngs,
            )

        if model_mode == MODEL_MODE_PREFILL:
            self.activation_axis_names = ("activation_batch", "prefill_activation_norm_length", "activation_embed")
        else:
            self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")
