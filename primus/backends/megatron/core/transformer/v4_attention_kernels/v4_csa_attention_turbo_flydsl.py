###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""V4 attention via the Primus-Turbo native-FlyDSL sparse-MLA backend ("turbo").

Thin binding of the kernel-agnostic V4 adapters (:mod:`v4_sparse_mla_adapter`) to
the ``sparse_mla_{fwd,bwd}_v4_turbo_flydsl`` kernel pair (:mod:`_turbo_flydsl`),
which re-exports the installed ``primus_turbo`` flydsl sparse-MLA v2 kernels. Same
fused single-latent (K == V) sparse-MLA-with-sink math as the ``gluon_v2`` /
``triton_v2`` / ``flydsl_v1`` backends, so it is numerically equivalent to the
eager V4 references (validated in
``tests/.../test_v4_turbo_flydsl_attention.py``).

Requires the installed ``primus_turbo`` (with the flydsl sparse-MLA attention) and
the ``flydsl`` pip package (gfx950 / CDNA4); the import raises a clear hint
otherwise (see :func:`..load_turbo_attention_backends`).
"""

from __future__ import annotations

from primus.backends.megatron.core.transformer.v4_attention_kernels._turbo_flydsl import (
    sparse_mla_bwd_v4_turbo_flydsl,
    sparse_mla_fwd_v4_turbo_flydsl,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels.v4_sparse_mla_adapter import (
    make_attention,
    make_csa_from_pool,
)

v4_csa_attention_turbo = make_csa_from_pool(sparse_mla_fwd_v4_turbo_flydsl, sparse_mla_bwd_v4_turbo_flydsl)
v4_attention_turbo = make_attention(sparse_mla_fwd_v4_turbo_flydsl, sparse_mla_bwd_v4_turbo_flydsl)

__all__ = [
    "v4_csa_attention_turbo",
    "v4_attention_turbo",
]
