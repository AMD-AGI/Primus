###############################################################################
# Copyright 2023–2025 Google LLC. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import jax.numpy as jnp
from MaxText.common_types import (
    DEFAULT_MASK_VALUE,
    MODEL_MODE_TRAIN,
    Array,
    AttentionType,
)
from MaxText.layers import nnx_wrappers
from MaxText.layers.attention_op import AttentionOp


class PrimusAttentionOp(AttentionOp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def cudnn_flash_attention(
        self,
        query: Array,
        key: Array,
        value: Array,
        decoder_segment_ids: Array | None,
        model_mode: str = MODEL_MODE_TRAIN,
    ) -> Array:
        """CUDNN Flash Attention with Transformer Engine.
        1. Stable API, supports MHA, GQA, SWA, Packing and Context Parallelism
        2. Context Parallelism currently only supports causal masking and no packing
        """
        # These imports are only meant to work in a GPU build.
        # pylint: disable=import-outside-toplevel
        from transformer_engine.jax.attention import (
            SequenceDescriptor,  # pytype: disable=import-error
        )
        from transformer_engine.jax.flax.transformer import (
            DotProductAttention,  # pytype: disable=import-error
        )

        _, _, _, head_dim = query.shape  # pylint: disable=unused-variable

        using_context_parallelism = self.mesh.shape["context"] > 1

        # Initialize default attention configuration
        sliding_window_size = None
        mask_type = "padding_causal"
        qkv_layout = "BSHD_BSHD_BSHD"  # Non-packed format: 'BS3HD', 'BSHD_BS2HD' or 'BSHD_BSHD_BSHD'
        max_segments_per_seq = 1  # max number of segments per sequence; for non-packed its 1

        # Handle local sliding window attention if configured
        if self.attention_type == AttentionType.LOCAL_SLIDING:
            sliding_window_size = [self.sliding_window_size, 0]

        # Handle packing configurations
        if self.config.packing and self.config.dataset_type != "synthetic":
            qkv_layout = "THD_THD_THD"  # Packed format: 'T3HD', 'THD_T2HD' or 'THD_THD_THD'
            if decoder_segment_ids is None:
                decoder_segment_ids = jnp.ones(shape=query.shape[:2], dtype=jnp.int32)
            attn_mask = SequenceDescriptor.from_segment_ids_and_pos(
                segment_ids=decoder_segment_ids, segment_pos=None
            )
            # Create dummy SequenceDescriptor for lazy_init
            dummy_segment_ids = jnp.ones(shape=query.shape[:2], dtype=jnp.int32)
            dummy_attn_mask = SequenceDescriptor.from_segment_ids_and_pos(
                segment_ids=dummy_segment_ids, segment_pos=None
            )
            max_segments_per_seq = self.config.max_segments_per_seq
        elif using_context_parallelism or self.config.dataset_type == "synthetic":
            if self.attention_type == AttentionType.LOCAL_SLIDING:
                raise AssertionError("Sliding window attention is not supported for context parallelism")
            # Context parallelism without packing: only supports causal masking
            attn_mask = None
            dummy_attn_mask = None
            mask_type = "causal"
        else:
            # Default case: no packing, no context parallelism
            dummy_attn_mask = jnp.zeros(
                (1, 1, 1, self.max_target_length, self.max_target_length), dtype=jnp.uint8
            )
            attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, model_mode)
            attn_mask = jnp.where((attn_mask >= DEFAULT_MASK_VALUE * 0.5), 0, 1).astype(jnp.uint8)

        dpa_layer = DotProductAttention(
            head_dim=head_dim,
            num_attention_heads=self.num_query_heads,
            num_gqa_groups=self.num_kv_heads,
            attn_mask_type=mask_type,  # 'no_mask', 'padding', 'causal', or 'padding_causal'
            attn_bias_type="no_bias",  # 'no_bias', 'pre_scale_bias' or 'post_scale_bias'
            attention_dropout=self.dropout_rate,
            dropout_rng_name="aqt",
            dtype=self.dtype,
            float32_logits=self.float32_logits,
            qkv_layout=qkv_layout,
            scale_factor=1.0,
            transpose_batch_sequence=False,
            window_size=sliding_window_size,
            context_parallel_causal_load_balanced=self.config.context_parallel_load_balance,
            context_parallel_axis="context",
            # context_parallel_strategy=self.config.context_parallel_strategy,
            max_segments_per_seq=max_segments_per_seq,
        )

        dpa_layer = nnx_wrappers.ToNNX(dpa_layer, rngs=self.rngs)
        dummy_query_prefill = jnp.zeros(
            (1, self.max_target_length, self.num_query_heads, self.config.head_dim), dtype=self.dtype
        )
        dummy_key_prefill = jnp.zeros(
            (1, self.max_target_length, self.num_kv_heads, self.config.head_dim), dtype=self.dtype
        )
        dummy_value_prefill = jnp.zeros(
            (1, self.max_target_length, self.num_kv_heads, self.config.head_dim), dtype=self.dtype
        )

        dpa_layer.lazy_init(
            dummy_query_prefill, dummy_key_prefill, dummy_value_prefill, sequence_descriptor=dummy_attn_mask
        )
        return dpa_layer(query, key, value, sequence_descriptor=attn_mask)
