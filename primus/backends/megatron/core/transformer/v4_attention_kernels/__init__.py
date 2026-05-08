###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek-V4 attention kernels (Plan-4).

Plan-4 P24 lands two pure-function eager-Python references for V4
attention — :func:`eager_v4_attention` (dense ``compress_ratio == 0`` /
HCA ``compress_ratio == 128``) and :func:`eager_v4_csa_attention`
(``compress_ratio == 4``). They are extracted from the math that
previously lived inline in :meth:`DeepseekV4Attention._attention_forward`
and :meth:`DeepseekV4Attention._csa_forward` so that:

* there is exactly one definition of "the eager truth" that
  ``DeepseekV4Attention``, the plan-4 Triton kernels (P25 / P26), and
  the plan-4 unit-test harness all share;
* the existing checkpoint-reproduction baseline does not move (the
  refactor is bit-identical at the existing call sites — see G22
  in ``tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p24_reference_op.py``);
* the reference op signatures match the future Triton kernel signatures
  ``v4_attention`` (P25) and ``v4_csa_attention`` (P26) so the test
  harness can plug reference ↔ candidate interchangeably.

The Triton kernels themselves land in P25 / P26 alongside this package.
"""

from primus.backends.megatron.core.transformer.v4_attention_kernels.reference import (
    eager_v4_attention,
    eager_v4_csa_attention,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels.v4_attention import (
    V4AttentionFn,
    v4_attention,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels.v4_csa_attention import (
    V4CSAAttentionFn,
    v4_csa_attention,
)

__all__ = [
    "eager_v4_attention",
    "eager_v4_csa_attention",
    "v4_attention",
    "V4AttentionFn",
    "v4_csa_attention",
    "V4CSAAttentionFn",
]
