###############################################################################
# Copyright 2023–2025 Google LLC. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from flax import linen as nn
from flax import nnx
from jax.sharding import Mesh
from MaxText import max_utils
from MaxText.common_types import Config
from MaxText.layers import initializers, moe, quantizations
from MaxText.layers.attentions import Attention
from MaxText.layers.linears import Dropout
from MaxText.layers.mixtral import MixtralDecoderLayer
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.quantizations import AqtQuantization as Quant


class PrimusMixtralDecoderLayer(MixtralDecoderLayer):
    @nn.compact
    def __init__(
        self,
        config: Config,
        mesh: Mesh,
        model_mode: str,
        quant: None | Quant = None,
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

        self.pre_self_attention_layer_norm = RMSNorm(
            num_features=config.emb_dim,
            dtype=config.dtype,
            weight_dtype=config.weight_dtype,
            kernel_axes=("norm",),
            epsilon=config.normalization_layer_epsilon,
            rngs=self.rngs,
        )

        self.self_attention = Attention(
            config=config,
            num_query_heads=config.num_query_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            max_target_length=config.max_target_length,
            max_prefill_predict_length=config.max_prefill_predict_length,
            attention_kernel=config.attention,
            inputs_q_shape=dummy_inputs_shape,
            inputs_kv_shape=dummy_inputs_shape,
            mesh=mesh,
            dtype=config.dtype,
            weight_dtype=config.weight_dtype,
            dropout_rate=config.dropout_rate,
            float32_qk_product=config.float32_qk_product,
            float32_logits=config.float32_logits,
            quant=self.quant,
            kv_quant=quantizations.configure_kv_quant(config),
            prefill_cache_axis_order=tuple(map(int, config.prefill_cache_axis_order.split(","))),
            ar_cache_axis_order=tuple(map(int, config.ar_cache_axis_order.split(","))),
            compute_axis_order=tuple(map(int, config.compute_axis_order.split(","))),
            reshape_q=config.reshape_q,
            use_ragged_attention=config.use_ragged_attention,
            ragged_block_size=config.ragged_block_size,
            query_pre_attn_scalar=(config.head_dim**-0.5),
            model_mode=model_mode,
            rngs=self.rngs,
        )

        self.post_self_attention_layer_norm = RMSNorm(
            num_features=config.emb_dim,
            dtype=config.dtype,
            weight_dtype=config.weight_dtype,
            kernel_axes=("norm",),
            epsilon=config.normalization_layer_epsilon,
            rngs=self.rngs,
        )

        self.MoeBlock_0 = moe.RoutedMoE(
            config=config,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            mesh=mesh,
            kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
            kernel_axes=("embed", None),
            intermediate_dim=config.mlp_dim,
            dtype=config.dtype,
            weight_dtype=config.weight_dtype,
            quant=self.quant,
            rngs=self.rngs,
        )

        self.dropout = Dropout(rate=config.dropout_rate, broadcast_dims=(-2,), rngs=rngs)

        self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")
