###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 decoder layer: norms after each sublayer (``h = x + norm(f(x))``)."""

from __future__ import annotations

from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import make_viewless_tensor


class Olmo3TransformerLayer(TransformerLayer):
    """``TransformerLayer`` with OLMo-3's post-sublayer RMSNorms.

    The spec leaves ``input_layernorm`` / ``pre_mlp_layernorm`` as ``IdentityOp``;
    this class adds ``post_attention_layernorm`` and ``post_feedforward_layernorm``
    (built with ``post_norm``, e.g. ``TENorm``) and applies them to the sublayer
    output before the residual add.
    """

    def __init__(self, config, submodules, layer_number=1, hidden_dropout=None, post_norm=None, **kwargs):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number,
                         hidden_dropout=hidden_dropout, **kwargs)
        assert post_norm is not None, "Olmo3TransformerLayer needs a post_norm builder"
        self.post_attention_layernorm = post_norm(
            config=self.config, hidden_size=self.config.hidden_size, eps=self.config.layernorm_epsilon
        )
        self.post_feedforward_layernorm = post_norm(
            config=self.config, hidden_size=self.config.hidden_size, eps=self.config.layernorm_epsilon
        )

    def _dropout_add(self, out_with_bias, residual):
        out, bias = out_with_bias
        assert bias is None, "OLMo-3 linears are bias-free"
        return out, residual

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
        **kwargs,
    ):
        import torch

        if inference_context is None:
            inference_context = inference_params
        residual = hidden_states
        attn_out, attn_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        assert attn_bias is None, "OLMo-3 attention output projection is bias-free"
        attn_out = self.post_attention_layernorm(attn_out)
        attn_out = torch.nn.functional.dropout(attn_out, p=self.hidden_dropout, training=self.training)
        hidden_states = residual + attn_out

        residual = hidden_states
        mlp_out, mlp_bias = self.mlp(hidden_states)
        assert mlp_bias is None, "OLMo-3 MLP is bias-free"
        mlp_out = self.post_feedforward_layernorm(mlp_out)
        mlp_out = torch.nn.functional.dropout(mlp_out, p=self.hidden_dropout, training=self.training)
        output = residual + mlp_out
        output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)
        return output, context
