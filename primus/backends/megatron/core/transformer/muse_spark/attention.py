###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark gated self-attention.

Megatron's ``SelfAttention`` with ``attention_output_gate=True`` already packs
the sigmoid gate per query group in ``linear_qkv`` and applies
``x * sigmoid(gate)`` after core attention. Muse Spark adds a weightless
per-head RMSNorm (``o_norm``) on the core-attention output before the gate.
The weightless QK-norm comes in through ``q_layernorm`` / ``k_layernorm`` in
the layer spec (:class:`MuseSparkQKNorm`); the query scale
``qk_scale_factor`` is folded into ``softmax_scale`` by the config.
"""

from megatron.core.transformer.attention import SelfAttention

from primus.backends.megatron.core.models.muse_spark.model_types import config_flag
from primus.backends.megatron.core.transformer.muse_spark.compiled_norms import (
    compiled,
    o_norm_gate_math,
)
from primus.backends.megatron.core.transformer.muse_spark.norms import (
    MuseSparkRMSNorm,
    weightless_rms,
)

__all__ = ["MuseSparkQKNorm", "MuseSparkSelfAttention"]


class MuseSparkQKNorm(MuseSparkRMSNorm):
    """Weightless per-head RMSNorm for the q / k heads: ``MuseSparkRMSNorm(with_scale=False)``.

    Only fixes ``with_scale=False`` (Megatron builds q / k layernorms with
    ``(config=, hidden_size=, eps=)``); the fp32 math is MuseSparkRMSNorm's.
    """

    def __init__(self, config=None, hidden_size: int = None, eps: float = 1e-5, **kw):
        kw.pop("with_scale", None)
        super().__init__(config, hidden_size, eps=eps, with_scale=False, **kw)


class MuseSparkSelfAttention(SelfAttention):
    """SelfAttention with a weightless per-head ``o_norm`` before the sigmoid output gate."""

    def __init__(self, config, *args, **kwargs):
        assert config.attention_output_gate, "Muse Spark attention needs attention_output_gate=True"
        super().__init__(config, *args, **kwargs)
        self.o_norm_eps = float(config.layernorm_epsilon)
        self.o_norm_fused = config_flag(config, "muse_spark_fused_attn_norm", True)
        self.o_norm_compile = config_flag(config, "muse_spark_compile_norms", False)

    def _apply_output_gate(self, x, gate):
        # x: [sq, b, np * hn] (or [t, 1, np * hn] for thd); normalize each head.
        hn = self.hidden_size_per_attention_head
        if self.o_norm_compile and x.is_cuda:
            return compiled(o_norm_gate_math)(x, gate, hn, self.o_norm_eps)
        shape = x.shape
        # fp32 per-head norm, rounded to x.dtype; the fused kernel on GPU (weightless_rms)
        x = weightless_rms(x.view(*shape[:-1], -1, hn), self.o_norm_eps, self.o_norm_fused).view(shape)
        return super()._apply_output_gate(x, gate)
