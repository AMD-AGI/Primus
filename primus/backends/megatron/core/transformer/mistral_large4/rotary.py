###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mistral Large 4 rotary embedding (YaRN + LeRoPE, learned in pretraining, frozen here) and SSMax query scaling.

* YaRN inverse frequencies exactly as Megatron's ``YarnRotaryEmbedding`` builds them
  (same correction range and ramp helpers), with no concentration factor (the checkpoint's
  ``attention_factor`` is 1.0), times ``exp(alpha)`` per frequency (LeRoPE). ``alpha`` is
  one checkpoint tensor shared by every layer; like the transformers reference it is not
  trained (a persistent buffer) and it stays fp32 through the bf16 model cast.
* The rotary table is computed in fp32 and returned in Megatron's ``[s, 1, 1, dim]``
  layout (``cat(freqs, freqs)``); MLA applies it with ``mla_rotary_interleaved=True``.
* SSMax: ``q * (s_h * log(pos + 1) + b_h)`` per head, ``log`` in fp32, then ``log``, ``s`` and
  ``b`` cast to the query dtype and the scale computed there (as the reference).
"""

from __future__ import annotations

import torch
from megatron.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank,
)
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    _yarn_find_correction_range,
    _yarn_linear_ramp_mask,
)
from torch import nn

__all__ = ["MistralLarge4RotaryEmbedding", "apply_ssmax", "positions_for"]


def yarn_inv_freq(
    dim: int, base: float, factor: float, original_max: int, beta_fast: float, beta_slow: float
):
    """YaRN inverse frequencies (fp32, CPU), as ``YarnRotaryEmbedding.get_emb`` computes them."""
    if dim == 0:
        return torch.zeros(0, dtype=torch.float32)
    exps = torch.arange(0, dim, 2, dtype=torch.float32) / dim
    pos_freqs = base**exps
    extra = 1.0 / pos_freqs
    inter = 1.0 / (factor * pos_freqs)
    low, high = _yarn_find_correction_range(beta_fast, beta_slow, dim, base, original_max, True)
    mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2, device=torch.device("cpu"))
    return inter * (1 - mask) + extra * mask


class MistralLarge4RotaryEmbedding(nn.Module):
    """YaRN + LeRoPE rotary table shared by every layer of a decoder block."""

    def __init__(self, config) -> None:
        super().__init__()
        self.dim = int(config.qk_pos_emb_head_dim)
        inv = yarn_inv_freq(
            self.dim,
            float(config.rotary_base),
            float(config.rotary_scaling_factor),
            int(config.original_max_position_embeddings),
            float(config.beta_fast),
            float(config.beta_slow),
        )
        device = None if config.use_cpu_initialization else torch.cuda.current_device()
        # a plain fp32 tensor (not a buffer): Float16Module's bf16 cast must not round it
        self._inv_freq = inv
        # LeRoPE per-frequency log-multiplier, learned in pretraining and loaded from the checkpoint
        # (lerope.alpha); frozen here (persistent buffer, not trained; outside the optimizer and FSDP
        # sharding). 0 = plain YaRN.
        self.register_buffer("alpha", torch.zeros(self.dim // 2, dtype=torch.float32, device=device))

    def _apply(self, fn, *args, **kwargs):  # keep alpha fp32 (unrounded) through Float16Module's cast
        before = self.alpha
        out = super()._apply(fn, *args, **kwargs)
        if self.alpha.dtype != torch.float32:
            self.alpha = before.to(device=self.alpha.device, dtype=torch.float32)
        return out

    def scaled_inv_freq(self) -> torch.Tensor:
        if self._inv_freq.device != self.alpha.device:
            self._inv_freq = self._inv_freq.to(self.alpha.device)
        return self._inv_freq * self.alpha.float().exp()

    def freqs(self, positions: torch.Tensor, inv_freq: torch.Tensor = None) -> torch.Tensor:
        """[n, 1, 1, dim] angles for integer ``positions`` [n] (fp32). ``inv_freq``: the caller's
        ``scaled_inv_freq()`` when it has it already (computed here otherwise)."""
        if inv_freq is None:
            inv_freq = self.scaled_inv_freq()
        f = positions.float()[:, None] * inv_freq.to(positions.device)[None, :]
        return torch.cat((f, f), dim=-1)[:, None, None, :]


def positions_for(sq: int, device, cp_group=None, packed_seq_params=None) -> torch.Tensor:
    """Absolute positions [sq] of this rank's query rows: arange, CP zigzag slice, or per document (THD)."""
    cp = cp_group.size() if cp_group is not None else 1
    if packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", None) == "thd":
        cu = packed_seq_params.cu_seqlens_q_padded
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        cu = cu.to(device=device, dtype=torch.long)
        if cp > 1:
            raise NotImplementedError(
                f"Mistral Large 4 SSMax does not support packed (THD) sequences with context parallelism (cp={cp})"
            )
        idx = torch.arange(sq, device=device)
        doc = torch.searchsorted(cu[1:], idx, right=True)
        return idx - cu[doc]
    pos = torch.arange(sq * cp, device=device)
    if cp > 1:
        pos = get_pos_emb_on_this_cp_rank(pos, 0, cp_group)
    return pos


def apply_ssmax(
    query: torch.Tensor,
    positions: torch.Tensor,
    s: torch.Tensor,
    b: torch.Tensor,
    log_pos: torch.Tensor = None,
) -> torch.Tensor:
    """The query scaled by SSMax: returns ``query * (s_h * log(pos + 1) + b_h)``; query
    [sq, ..., heads, d], positions [sq].
    ``log_pos`` (fp32 ``log(pos + 1)``, computed once per forward by the decoder block) is used
    when given.

    Intentional, as the transformers reference (``modeling_mistral_large4.py``): ``log`` is fp32,
    then ``log``, ``s`` and ``b`` are cast to the query dtype and the scale is computed in that
    dtype (bf16 in training). Computing it in fp32 would round less, but would not be the
    released model's computation, against which parity is measured."""
    dt = query.dtype
    if log_pos is None:
        log_pos = (positions.float() + 1.0).log()
    log_pos = log_pos.to(device=query.device, dtype=dt)
    factor = s.to(dt)[None, :] * log_pos[:, None] + b.to(dt)[None, :]  # [sq, heads]
    shape = [factor.size(0)] + [1] * (query.dim() - 3) + [factor.size(1), 1]
    return query * factor.view(shape)
