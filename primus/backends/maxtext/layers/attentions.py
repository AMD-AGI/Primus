###############################################################################
# Copyright 2023–2025 Google LLC. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Tuple

from flax import nnx
from MaxText.layers.attentions import Attention
from MaxText.layers.linears import DenseGeneral


class PrimusAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_query_w(self, inputs_q_shape: Tuple) -> nnx.Module:
        """Query projection initialization."""

        # NOTE: T5 does not explicitly rescale the attention logits by
        #       1/sqrt(depth_kq)!  This is folded into the initializers of the
        #       linear transformations, which is equivalent under Adafactor.
        # depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
        depth_scaling = 1.0

        def query_init(*args):
            # pylint: disable=no-value-for-parameter
            return self.kernel_init(*args) / depth_scaling

        kernel_axes = (
            (None, None, None)
            if self.config.ici_context_autoregressive_parallelism > 1
            else ("embed", "q_heads", "kv")
        )
        return DenseGeneral(
            in_features_shape=self.convert_dense_general_inputs_shape(inputs_q_shape),
            out_features_shape=(self.num_query_heads, self.head_dim),
            axis=-1,
            kernel_init=query_init,
            kernel_axes=kernel_axes,
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            quant=self.quant,
            matmul_precision=self.config.matmul_precision,
            use_bias=self.use_bias_in_projections,
            rngs=self.rngs,
        )
