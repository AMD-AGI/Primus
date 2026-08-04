###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Eager-PyTorch reference ops for Kimi Delta Attention.

The single source of "eager truth" shared by ``KimiDeltaAttention``, every
KDA kernel backend (``fla`` today, FlyDSL in WP9) and the unit tests:

* :func:`eager_chunk_kda`     — chunkwise-parallel form (the training path)
* :func:`eager_recurrent_kda` — ``O(T)`` sequential recurrence (the oracle)
* :func:`kda_gate`            — per-channel log-decay gate
* :func:`kda_l2norm`          — the ``use_qk_l2norm_in_kernel`` transform
"""

from .reference import eager_chunk_kda, eager_recurrent_kda, kda_gate, kda_l2norm

__all__ = ["eager_chunk_kda", "eager_recurrent_kda", "kda_gate", "kda_l2norm"]
