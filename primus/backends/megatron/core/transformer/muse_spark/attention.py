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

import torch
from megatron.core.transformer.attention import SelfAttention

from primus.backends.megatron.core.transformer.muse_spark.norms import _rms

__all__ = ["MuseSparkQKNorm", "MuseSparkSelfAttention"]


class MuseSparkQKNorm(torch.nn.Module):
    """Weightless per-head RMSNorm (fp32 math) for the q / k heads."""

    def __init__(self, config=None, hidden_size: int = None, eps: float = 1e-5, **_kw):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms(x.float(), self.eps).type_as(x)


class MuseSparkSelfAttention(SelfAttention):
    """SelfAttention with a weightless per-head ``o_norm`` before the sigmoid output gate."""

    def __init__(self, config, *args, **kwargs):
        assert config.attention_output_gate, "Muse Spark attention needs attention_output_gate=True"
        super().__init__(config, *args, **kwargs)
        self.o_norm_eps = float(config.layernorm_epsilon)

    def _apply_output_gate(self, x, gate):
        # x: [sq, b, np * hn] (or [t, 1, np * hn] for thd); normalize each head.
        hn = self.hidden_size_per_attention_head
        shape = x.shape
        x = _rms(x.float().view(*shape[:-1], -1, hn), self.o_norm_eps).to(x.dtype).view(shape)
        return super()._apply_output_gate(x, gate)
