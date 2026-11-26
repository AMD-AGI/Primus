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

With ``shared=True`` (config switch ``msa_shared_rope_tables``) the tables are also kept per
``freqs`` tensor: Megatron's ``RotaryEmbedding.forward`` is ``lru_cache``d, so every layer,
micro-batch and step passes the same tensor object, and the tables are then computed once per
run instead of once per layer call (four launches each). The tables are a function of ``freqs``
alone, whatever the rank, micro-batch or data. The first reuse of a ``freqs`` tensor's tables
compares them with freshly computed ones (``bitwise_check``) and keeps recomputing on a mismatch.
"""

import weakref
from collections import OrderedDict
from typing import Dict, Tuple

import torch
from megatron.core.models.common.embeddings.rope_utils import _rotate_half


class RopeTables:
    """``_apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved)`` for many ``t``.

    Args:
        freqs: ``[sq, 1, 1, rotary_dim]`` rope freqs. ``rotary_dim`` may be less
            than the head dim; the tail of every head then passes through.
        shared: reuse the tables of an earlier ``RopeTables`` of the same ``freqs`` tensor object.
    """

    # Tables of the last few ``freqs`` tensors, keyed by identity. The weak reference makes the
    # id() of a freed tensor a miss, and the saved version counter makes a ``freqs`` written in
    # place since then a miss too.
    _shared: "OrderedDict[int, Tuple[weakref.ref, int, Dict]]" = OrderedDict()
    _SHARED_MAX = 8

    def __init__(self, freqs: torch.Tensor, shared: bool = False):
        self.freqs = freqs
        self._tables: Dict[torch.dtype, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Eligibility before any work: a plain table, not something autograd follows, and not
        # one computed inside a CUDA graph capture (it would be reused outside the graph).
        if (
            not shared
            or freqs.requires_grad
            or torch.is_inference_mode_enabled()
            or (freqs.is_cuda and torch.cuda.is_current_stream_capturing())
        ):
            return
        hit = RopeTables._shared.get(id(freqs))
        if hit is not None and hit[0]() is freqs and hit[1] == freqs._version:
            if self._reuse_checked(hit[2]):
                RopeTables._shared.move_to_end(id(freqs))
                self._tables = hit[2]
            return
        RopeTables._shared[id(freqs)] = (weakref.ref(freqs), freqs._version, self._tables)
        while len(RopeTables._shared) > RopeTables._SHARED_MAX:
            RopeTables._shared.popitem(last=False)

    def _reuse_checked(self, tables: Dict) -> bool:
        """Whether kept tables may stand in for recomputed ones: decided once per freqs shape and
        dtype, on the first reuse, by comparing the kept tables with freshly computed ones."""
        from primus.backends.megatron.core.fusions import bitwise_check

        if not tables:
            return True  # nothing computed yet: this object fills the shared dict itself
        return bitwise_check.passed(
            "msa_shared_rope_tables",
            lambda: [t for d in sorted(tables, key=str) for t in tables[d]],
            lambda: [t for d in sorted(tables, key=str) for t in self._compute(d)],
            key=(tuple(self.freqs.shape), str(self.freqs.dtype), tuple(sorted(str(d) for d in tables))),
        )

    def _compute(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # upstream's (cos(freqs) * mscale).to(dtype) at mscale 1.0, which is exact
        return torch.cos(self.freqs).to(dtype), torch.sin(self.freqs).to(dtype)

    def cos_sin(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        tables = self._tables.get(dtype)
        if tables is None:
            tables = self._compute(dtype)
            self._tables[dtype] = tables
        return tables

    def apply(self, t: torch.Tensor, rotary_interleaved: bool = False) -> torch.Tensor:
        cos_, sin_ = self.cos_sin(t.dtype)
        rot_dim = cos_.shape[-1]
        x, x_pass = t[..., :rot_dim], t[..., rot_dim:]
        x = (x * cos_) + (_rotate_half(x, rotary_interleaved) * sin_)
        return torch.cat((x, x_pass), dim=-1)
