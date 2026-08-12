###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Eager attention-residual mixer: the permanent numerical oracle."""

from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._eager.reference import (
    accum_dtype,
    eager_attn_res_mix,
    fused_score_weight,
)

__all__ = ["eager_attn_res_mix", "fused_score_weight", "accum_dtype"]
