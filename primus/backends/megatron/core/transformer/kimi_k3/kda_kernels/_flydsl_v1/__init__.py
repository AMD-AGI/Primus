###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native FlyDSL KDA backend, gfx950 / CDNA4 only.

Importing this package pulls in ``flydsl``, so it must only ever be reached
through :func:`...kda_kernels.load_flydsl_kda_backend`, never from module
scope of anything on the default import path. See the lazy-loader rationale
in :mod:`...kda_kernels` and its DeepSeek-V4 precedent
(``v4_attention_kernels/__init__.py:137-161``).
"""

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.chunk import (
    flydsl_chunk_kda,
    flydsl_chunk_kda_bwd,
    flydsl_chunk_kda_fwd,
)

__all__ = ["flydsl_chunk_kda", "flydsl_chunk_kda_fwd", "flydsl_chunk_kda_bwd"]
