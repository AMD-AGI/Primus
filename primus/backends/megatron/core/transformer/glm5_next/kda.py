###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GLM-5.3 KDA layer: Kimi Delta Attention with the low-rank output gate.

The math is :class:`KimiDeltaAttention` with ``kda_use_full_rank_gate=False``
(SGLang ``Glm5NextLinearAttention``: ``g_b_proj(g_a_proj(x))`` feeds the
sigmoid-gated ``o_norm``). The only addition is holding ``A_log`` / ``dt_bias``
at FP32 through the model-wide bf16 cast, as the checkpoint stores them.
"""

from __future__ import annotations

from primus.backends.megatron.core.transformer.glm5_next.fp32_params import (
    KeepFp32ParamsMixin,
)
from primus.backends.megatron.core.transformer.kimi_k3.kimi_delta_attention import (
    KimiDeltaAttention,
)

__all__ = ["Glm5NextKimiDeltaAttention"]


class Glm5NextKimiDeltaAttention(KeepFp32ParamsMixin, KimiDeltaAttention):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self._keep_fp32_enabled = bool(getattr(config, "glm5_keep_fp32_params", True))
        self._fp32_param_names = ("A_log", "dt_bias")
