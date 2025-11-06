
import jax.numpy as jnp

from MaxText.common_types import (
    Array,
    AttentionType,
    MODEL_MODE_TRAIN,
)
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
        1. Stable API, supports GQA, SWA (only with causal masking)
        2. Head_dim = 256 is also supported from TE-1.12 stable release with CUDNN 12.6
        """
        # These imports are only meant to work in a GPU build.
        # pylint: disable=import-outside-toplevel
        from transformer_engine.jax.flax.transformer import DotProductAttention  # pytype: disable=import-error

        _, _, _, head_dim = query.shape  # pylint: disable=unused-variable

        using_context_parallelism = self.mesh.shape["context"] > 1

        if self.attention_type == AttentionType.LOCAL_SLIDING and using_context_parallelism:
            raise AssertionError("Sliding window attention is not supported when context parallelism is enabled")

        sliding_window_size = None
        mask_type = "padding_causal"
        qkv_layout = "BSHD_BSHD_BSHD"  # 'BS3HD', 'BSHD_BS2HD' or 'BSHD_BSHD_BSHD'
        max_segments_per_seq = 1  # max number of segments per sequence; for non-packed its 1

        if self.attention_type == AttentionType.LOCAL_SLIDING:
            sliding_window_size = [self.sliding_window_size, 0]

        if self.config.packing and self.config.dataset_type != "synthetic":
            if decoder_segment_ids is None:
                decoder_segment_ids = jnp.ones(shape=query.shape[:2], dtype=jnp.int32)
            attn_mask = SequenceDescriptor.from_segment_ids_and_pos(segment_ids=decoder_segment_ids, segment_pos=None)
            qkv_layout = "THD_THD_THD"  # 'T3HD', 'THD_T2HD' or 'THD_THD_THD'
            max_segments_per_seq = 32
        elif using_context_parallelism or self.config.dataset_type == "synthetic":  # context parallelism currently only supports causal masking and no packing
            attn_mask = None
            mask_type = "causal"
        else:
            attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, model_mode)

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
            # scale_factor=1.0,
            transpose_batch_sequence=False,
            window_size=sliding_window_size,
            context_parallel_causal_load_balanced=self.config.context_parallel_load_balance,
            context_parallel_axis="context",
            max_segments_per_seq=max_segments_per_seq,
        )
        return dpa_layer(query, key, value, mask=attn_mask)
