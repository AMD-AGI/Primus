###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MLA key/value up-projection with fewer key/value heads than query heads.

Megatron's MLA gives every query head its own key/value block. Zebra-Llama's
MLA (``use_full_kv_head: false``) keeps the teacher's GQA key/value heads and
shares each one with consecutive query heads, like Llama's ``repeat_kv``.
"""

from __future__ import annotations

from typing import Optional

import torch
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear


def repeat_kv_heads(kv: torch.Tensor, head_dim: int, repeats: int) -> torch.Tensor:
    """Repeat each ``head_dim`` block of the last dimension ``repeats`` times in place."""
    if repeats == 1:
        return kv
    return kv.unflatten(-1, (-1, head_dim)).repeat_interleave(repeats, dim=-2).flatten(-2)


class GroupedKVUpProjection(TEColumnParallelLinear):
    """``linear_kv_up_proj`` with one weight block per key/value head.

    The weight has ``kv_heads * (qk_head_dim + v_head_dim)`` rows, like the
    reference ``kv_b_proj``. The output repeats each block for its
    ``num_attention_heads // kv_heads`` query heads, the per-head layout
    Megatron's MLA reads. ``kv_heads`` defaults to the ``hylo_mla_kv_heads``
    argument.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config,
        kv_heads: Optional[int] = None,
        **kwargs,
    ):
        if kv_heads is None:
            from megatron.training import get_args

            kv_heads = get_args().hylo_mla_kv_heads
        heads = config.num_attention_heads
        if not kv_heads or heads % int(kv_heads) or output_size % heads:
            raise ValueError(
                f"GroupedKVUpProjection needs hylo_mla_kv_heads dividing the {heads} "
                f"attention heads, got {kv_heads}"
            )
        repeats = heads // int(kv_heads)
        super().__init__(input_size, output_size // repeats, config=config, **kwargs)
        self.kv_head_dim = output_size // heads
        self.kv_repeats = repeats

    def forward(self, x: torch.Tensor):
        out, bias = super().forward(x)
        return repeat_kv_heads(out, self.kv_head_dim, self.kv_repeats), bias
