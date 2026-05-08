###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton kernels backing :mod:`primus...transformer.v4_attention_kernels`.

Submodules:

* :mod:`v4_attention_fwd` — forward kernel for the dense
  (``compress_ratio == 0``) and HCA (``compress_ratio == 128``) paths.
* :mod:`v4_attention_bwd` — backward kernel matching the FWD's saved
  LSE.
"""
