###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Gated attention output (Qwen3-Next / Qwen3.5 full-attention layers), training.

``q_proj`` emits a second ``heads x head_dim`` block, the gate ``g``, and the
attention output is ``o * sigmoid(g)`` before ``o_proj`` (Qiu et al., Gated
Attention for Large Language Models, the head-specific elementwise SDPA gate,
G1).  On top of standard attention that is one more ``hidden -> heads*head_dim``
projection and an elementwise gate over the attention output.

When the model interleaves linear attention, only the full-attention layers
carry the gate.
"""

from __future__ import annotations

from primus.core.projection.op_models.base import (
    SLOT_ATTENTION_EXTRA,
    OpContext,
    OpCost,
    OpModel,
    cfg_value,
    interleave_fraction,
)
from primus.core.projection.op_models.registry import register_op_model

_ACT = 2.0
_LINEAR_VARIANTS = ("gated_delta_net", "kda")


@register_op_model
class AttentionOutputGate(OpModel):
    name = "attention_output_gate"
    slot = SLOT_ATTENTION_EXTRA
    claims = {"attention_output_gate": (True, "true")}
    consumes = ("attention_output_gate",)
    references = (
        "Qiu et al. Gated Attention for Large Language Models: Non-linearity, Sparsity, and "
        "Attention-Sink-Free. arXiv:2505.06708 (G1 head-specific sigmoid gate)",
        "HF transformers Qwen3NextAttention (q_proj -> query, gate; attn_output * sigmoid(gate))",
    )

    def layer_fraction(self, model_config) -> float:
        variant = str(cfg_value(model_config, "experimental_attention_variant", "") or "").lower()
        if variant in _LINEAR_VARIANTS:
            every = int(cfg_value(model_config, "linear_attention_freq", 4) or 4)
            return 1.0 - interleave_fraction(model_config, every)
        return 1.0

    def _width(self, ctx: OpContext) -> int:
        mc = ctx.mc
        return int(mc.num_attention_heads) * int(mc.kv_channels or mc.hidden_size // mc.num_attention_heads)

    def cost(self, ctx: OpContext) -> OpCost:
        hidden = int(ctx.mc.hidden_size)
        width = self._width(ctx)
        m = ctx.tokens
        bpe_w = 1 if ctx.gemm_dtype in ("fp8", "mx8") else 2

        proj_f, proj_b = ctx.linear(m, width, hidden)
        proj_flops = 2.0 * m * width * hidden
        proj_bytes = (m * hidden + m * width) * _ACT + width * hidden * bpe_w / ctx.tp

        # sigmoid(g) * o: read g and o, write the gated output; backward reads
        # g, o and the incoming grad, writes dg and do.
        gate_f = 3.0 * m * width * _ACT
        gate_b = 5.0 * m * width * _ACT
        parts = {
            "gate_proj": (proj_f, proj_b),
            "sigmoid_mul": (ctx.stream_ms(gate_f), ctx.stream_ms(gate_b)),
        }
        return OpCost(
            fwd_ms=proj_f + parts["sigmoid_mul"][0],
            bwd_ms=proj_b + parts["sigmoid_mul"][1],
            fwd_flops=proj_flops + 5.0 * m * width,
            bwd_flops=2 * proj_flops + 8.0 * m * width,
            fwd_bytes=proj_bytes + gate_f,
            bwd_bytes=2 * proj_bytes + gate_b,
            activation_bytes=m * width * _ACT,
            num_params=hidden * width,
            breakdown=parts,
        )

    def reference_flops(self, ctx: OpContext):
        m, hidden, width = ctx.tokens, int(ctx.mc.hidden_size), self._width(ctx)
        proj = 2.0 * m * hidden * width
        return proj + 4.0 * m * width, 2 * proj + 8.0 * m * width
