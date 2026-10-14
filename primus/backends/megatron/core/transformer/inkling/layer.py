###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling decoder layer: Megatron's :class:`TransformerLayer` plus two short convs.

``InklingDecoderLayer`` in ``modeling_inkling.py``::

    h = x + attn_sconv(attn(input_layernorm(x)))
    y = h + mlp_sconv(mlp(pre_mlp_layernorm(h)))

Both convs act on the full hidden size before the residual add, so they are
replicated across TP and run on the reduced attention / MLP output. They mix
neighbouring tokens, so a sequence-parallel shard would need its neighbour's
last ``kernel - 1`` tokens; sequence parallelism is rejected until that
exchange exists.
"""

from __future__ import annotations

import torch.nn.functional as F
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import make_viewless_tensor

from primus.backends.megatron.core.transformer.inkling.short_conv import (
    InklingShortConv,
)

__all__ = ["InklingTransformerLayer"]


class InklingTransformerLayer(TransformerLayer):
    def __init__(self, config, submodules, layer_number: int = 1, **kwargs):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, **kwargs)
        if config.sequence_parallel:
            raise NotImplementedError("Inkling's short convolutions do not support sequence parallelism yet")
        k = config.short_conv_kernel_size
        self.attn_sconv = InklingShortConv(config, config.hidden_size, k)
        self.mlp_sconv = InklingShortConv(config, config.hidden_size, k)

    def _residual_add(self, residual, output_with_bias, sconv):
        output, bias = output_with_bias
        if bias is not None:
            output = output + bias
        output = sconv(output)
        if self.hidden_dropout > 0:
            output = F.dropout(output, p=self.hidden_dropout, training=self.training)
        return residual + output

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        padding_mask=None,
        *,
        inference_params=None,
    ):
        residual = hidden_states.float() if self.config.fp32_residual_connection else hidden_states
        attn = self.self_attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            inference_context=inference_context or inference_params,
            rotary_pos_emb=rotary_pos_emb,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
        hidden_states = self._residual_add(residual, attn, self.attn_sconv)

        residual = hidden_states.float() if self.config.fp32_residual_connection else hidden_states
        mlp = self.mlp(self.pre_mlp_layernorm(hidden_states), padding_mask=padding_mask)
        output = self._residual_add(residual, mlp, self.mlp_sconv)
        output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)
        return output, context
