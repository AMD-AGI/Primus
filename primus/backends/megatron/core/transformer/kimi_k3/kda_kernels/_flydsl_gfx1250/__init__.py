###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native FlyDSL KDA backend, gfx1250 only; does not import fla.

The chunk kernel (:mod:`.chunk`) and the short causal conv + SiLU (:mod:`.conv_kernel`). Importing this package
pulls in ``flydsl``, so it must only ever be reached through :func:`...kda_kernels.load_flydsl_kda_backend`, as
:mod:`.._flydsl_v1` (the gfx950 backend) is.
"""

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_gfx1250.chunk import (
    chunk_kda_unsupported_reason,
    flydsl_chunk_kda,
)
from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_gfx1250.conv_kernel import (
    flydsl_causal_conv1d_silu,
    supports_conv_inputs,
)

__all__ = [
    "chunk_kda_unsupported_reason",
    "flydsl_causal_conv1d_silu",
    "flydsl_chunk_kda",
    "supports_conv_inputs",
]
