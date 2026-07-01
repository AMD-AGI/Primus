###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Eager-Python reference ops for DeepSeek-V4 attention.

The single source of "eager truth" shared by ``DeepseekV4Attention``, every
kernel backend (triton v0/v1/v2, gluon, flydsl_v2, tilelang) and the unit tests:

* :func:`eager_v4_attention`     — dense (cr=0) / HCA (cr=128)
* :func:`eager_v4_csa_attention` — CSA (cr=4)
"""

from .reference import eager_v4_attention, eager_v4_csa_attention

__all__ = ["eager_v4_attention", "eager_v4_csa_attention"]
