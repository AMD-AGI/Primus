###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""LFM2 gated short convolution (Liquid AI LFM2 ``conv`` layers), training.

The conv operator replaces attention on the layers ``lfm_layer_types`` marks
``conv`` (HF ``Lfm2ShortConv``):

    B, C, x = split(x W_in, 3)            W_in: hidden -> 3 * hidden
    y = C * causal_dwconv_L(B * x)        depthwise, kernel L = conv_L_cache
    out = y W_out                         W_out: hidden -> hidden

The two projections are GEMMs; the gates and the depthwise conv (L = 3 taps)
are memory bound and priced as HBM streams.  Channels shard over TP like the
attention heads they replace.
"""

from __future__ import annotations

from primus.core.projection.op_models.base import (
    SLOT_ATTENTION,
    OpContext,
    OpCost,
    OpModel,
    cfg_value,
)
from primus.core.projection.op_models.registry import register_op_model

_ACT = 2.0


@register_op_model
class Lfm2ShortConv(OpModel):
    name = "lfm2_short_conv"
    slot = SLOT_ATTENTION
    claims = {"lfm_layer_types": ("conv",)}
    consumes = ("lfm_layer_types", "conv_L_cache", "conv_bias")
    references = (
        "Liquid AI. LFM2 Technical Report (gated short convolution blocks). arXiv:2511.23404",
        "HF transformers Lfm2ShortConv (in_proj -> B, C, x; conv1d groups=hidden; out_proj)",
    )

    @classmethod
    def matches(cls, model_config) -> bool:
        types = cfg_value(model_config, "lfm_layer_types") or []
        return any(str(t).lower() == "conv" for t in types)

    def layer_fraction(self, model_config) -> float:
        types = cfg_value(model_config, "lfm_layer_types") or []
        return sum(str(t).lower() == "conv" for t in types) / len(types) if types else 0.0

    def cost(self, ctx: OpContext) -> OpCost:
        h = int(ctx.mc.hidden_size)
        L = int(ctx.value("conv_L_cache", 3) or 3)
        m = ctx.tokens
        tok_r = ctx.batch_size * ctx.seq_per_cp
        ch_r = max(1, h // ctx.tp)
        bpe_w = 1 if ctx.gemm_dtype in ("fp8", "mx8") else 2

        parts, flops_f, bytes_f = {}, 0.0, 0.0
        for tag, n, k in (("in_proj", 3 * h, h), ("out_proj", h, h)):
            parts[tag] = ctx.linear(m, n, k)
            flops_f += 2.0 * m * n * k
            bytes_f += (m * k + m * n) * _ACT + n * k * bpe_w / ctx.tp
        proj_flops, proj_bytes = flops_f, bytes_f

        # B * x (read B, x; write), depthwise conv (read, write), C * y (read C, y; write).
        elem_bytes = (3 + 2 + 3) * tok_r * ch_r * _ACT
        elem_flops = tok_r * ch_r * (2.0 * L + 2.0)
        parts["gates_conv"] = (ctx.stream_ms(elem_bytes), ctx.stream_ms(2 * elem_bytes))
        flops_f += elem_flops
        bytes_f += elem_bytes

        return OpCost(
            fwd_ms=sum(v[0] for v in parts.values()),
            bwd_ms=sum(v[1] for v in parts.values()),
            fwd_flops=flops_f,
            bwd_flops=2 * proj_flops + 2 * elem_flops,
            fwd_bytes=bytes_f,
            bwd_bytes=2 * proj_bytes + 2 * elem_bytes,
            activation_bytes=m * 6 * h * _ACT,  # x, B|C|x, B*x, conv output
            num_params=4 * h * h + h * L,
            breakdown=parts,
        )

    def reference_flops(self, ctx: OpContext):
        h, L = int(ctx.mc.hidden_size), int(ctx.value("conv_L_cache", 3) or 3)
        m = ctx.tokens
        tok_ch = ctx.batch_size * ctx.seq_per_cp * max(1, h // ctx.tp)
        proj = 2.0 * m * h * (3 * h + h)
        elem = tok_ch * (2.0 * L + 2.0)
        return proj + elem, 2 * proj + 2 * elem
