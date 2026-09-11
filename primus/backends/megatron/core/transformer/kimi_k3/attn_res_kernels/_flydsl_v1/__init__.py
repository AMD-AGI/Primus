###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused FlyDSL attention-residual backend, gfx950 / CDNA4 only.

Importing this package pulls in ``flydsl``, so it must only ever be reached
through :func:`...attn_res_kernels.load_flydsl_attn_res_backend`, never from
module scope of anything on the default import path. Same lazy-loader rationale
as :mod:`...kda_kernels._flydsl_v1`.
"""

from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._flydsl_v1.attn_res_mixer_bwd_kernel import (
    BWD_INJECTIONS,
)
from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._flydsl_v1.attn_res_mixer_kernel import (
    FWD_INJECTIONS,
    FWD_NEUTRAL_VARIANTS,
)
from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._flydsl_v1.mixer import (
    flydsl_attn_res_mix,
    flydsl_mix_kernel,
    inject_defect,
    supports_mixer_inputs,
)

__all__ = [
    "flydsl_attn_res_mix",
    "flydsl_mix_kernel",
    "supports_mixer_inputs",
    # test-only: proving the parity assertions have discrimination power
    "inject_defect",
    "FWD_INJECTIONS",
    "FWD_NEUTRAL_VARIANTS",
    "BWD_INJECTIONS",
]
