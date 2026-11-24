###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unfused RoPE whose cos/sin tables are shared by the calls of one layer.

An MSA layer applies the same freqs four times: to its query and key, and to the
indexer's query and key. Megatron's ``_apply_rotary_pos_emb_bshd`` recomputes
``cos(freqs)`` and ``sin(freqs)`` on every call, six kernel launches each time,
and the MSA forward is bound by kernel launches on the CPU. :class:`RopeTables`
computes them once per dtype and then runs the same elementwise ops in the same
order, so its output is bitwise that of the upstream function.
"""

from typing import Dict, Tuple

import torch
from megatron.core.models.common.embeddings.rope_utils import _rotate_half


class RopeTables:
    """``_apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved)`` for many ``t``.

    Args:
        freqs: ``[sq, 1, 1, rotary_dim]`` rope freqs. ``rotary_dim`` may be less
            than the head dim; the tail of every head then passes through.
    """

    def __init__(self, freqs: torch.Tensor):
        self.freqs = freqs
        self._tables: Dict[torch.dtype, Tuple[torch.Tensor, torch.Tensor]] = {}

    def cos_sin(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        tables = self._tables.get(dtype)
        if tables is None:
            # upstream's (cos(freqs) * mscale).to(dtype) at mscale 1.0, which is exact
            tables = (torch.cos(self.freqs).to(dtype), torch.sin(self.freqs).to(dtype))
            self._tables[dtype] = tables
        return tables

    def apply(self, t: torch.Tensor, rotary_interleaved: bool = False) -> torch.Tensor:
        cos_, sin_ = self.cos_sin(t.dtype)
        rot_dim = cos_.shape[-1]
        x, x_pass = t[..., :rot_dim], t[..., rot_dim:]
        x = (x * cos_) + (_rotate_half(x, rotary_interleaved) * sin_)
        return torch.cat((x, x_pass), dim=-1)
