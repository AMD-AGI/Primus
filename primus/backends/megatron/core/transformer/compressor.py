###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek-V4 Compressor.

Reference: techblog §1.3 ("Compressor: the Long-Range Compression Branch") and
the diagrams in ``deepseek-v4/develop/techblog/diagrams/csa.png`` /
``hca.png``.

Two configurations:

* ``ratio == 4`` (CSA branch) — overlap mode, ``coff == 2``: each compressed
  token sees an effective window of ``2*ratio`` raw tokens (current window
  plus the previous window's "leftover-half" channels). This smooths
  boundary effects between adjacent compressed positions.
* ``ratio == 128`` (HCA branch) — non-overlap mode, ``coff == 1``: each
  compressed token covers exactly ``ratio`` raw tokens.

Compressor returns the pooled KV after a final ``kv_norm`` (RMSNorm). RoPE
at the compress branch theta is applied **outside** this module by the
caller (the dual-RoPE module produced in P4.3 + the CSA / HCA modules in
P4.4 will consume the output).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RMSNorm(nn.Module):
    """Tiny RMSNorm (TE / Megatron's RMSNorm requires extra deps; we keep it
    local here because Compressor is independent of the surrounding stack)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * rms).to(in_dtype) * self.weight


class Compressor(nn.Module):
    """V4 Compressor block.

    Args:
        hidden_size: input feature dim ``D``.
        head_dim: output channel dim per compressed position.
        ratio: compression ratio ``m``. Must be a divisor of the runtime
            sequence length (``S % ratio == 0``).
        overlap: whether to use the overlap-stitched mode. If ``None``,
            defaults to ``ratio == 4`` (the V4 convention).
        rmsnorm_eps: RMSNorm stability eps.

    Shapes:
        Forward input  ``hidden``: ``[B, S, D]``.
        Forward output ``pooled``: ``[B, S // ratio, head_dim]``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        ratio: int,
        overlap: Optional[bool] = None,
        rmsnorm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if ratio < 1:
            raise ValueError(f"ratio must be >= 1, got {ratio}")

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.ratio = ratio
        self.overlap = bool(ratio == 4 if overlap is None else overlap)
        # coff is the projection multiplier — overlap mode needs 2x the
        # channels because half goes to the "current window" half and half to
        # the "previous window" half.
        self.coff = 2 if self.overlap else 1

        proj_out = self.coff * head_dim
        self.wkv = nn.Linear(hidden_size, proj_out, bias=False)
        self.wgate = nn.Linear(hidden_size, proj_out, bias=False)

        # Learnable absolute position embedding (APE) added on top of the
        # softmax score. After overlap, the effective window length is
        # ``2*ratio`` slots of size ``head_dim``; in non-overlap mode it's
        # ``ratio`` slots of size ``head_dim``.
        ape_len = 2 * ratio if self.overlap else ratio
        self.ape = nn.Parameter(torch.zeros(ape_len, head_dim))
        nn.init.normal_(self.ape, std=0.02)

        self.kv_norm = _RMSNorm(head_dim, eps=rmsnorm_eps)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _reshape_into_windows(self, t: torch.Tensor) -> torch.Tensor:
        """``[B, S, coff*head_dim]`` → ``[B, N, ratio, coff*head_dim]``,
        where ``N = S // ratio``.
        """
        B, S, C = t.shape
        assert S % self.ratio == 0, f"Compressor: sequence length {S} not divisible by ratio {self.ratio}"
        N = S // self.ratio
        return t.reshape(B, N, self.ratio, C)

    def _overlap_transform(self, t: torch.Tensor) -> torch.Tensor:
        """``[B, N, ratio, 2*head_dim]`` → ``[B, N, 2*ratio, head_dim]``.

        For window ``i``, the augmented sequence is
        ``[half_a[i], half_b[i-1]]`` concatenated along the per-window
        axis. Window 0's "previous half" is filled with zeros (causal
        padding).
        """
        # Split channels.
        half_a, half_b = torch.chunk(t, 2, dim=-1)  # each [B, N, ratio, head_dim]
        # Roll along the window dim so half_b[i] becomes "previous-window's b" of i+1.
        half_b_prev = torch.cat(
            [torch.zeros_like(half_b[:, :1]), half_b[:, :-1]],
            dim=1,
        )
        # Concat along the per-window token axis.
        return torch.cat([half_a, half_b_prev], dim=2)  # [B, N, 2*ratio, head_dim]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Pool ``hidden[B, S, D]`` to ``[B, S/ratio, head_dim]``."""
        kv_proj = self.wkv(hidden)  # [B, S, coff*head_dim]
        score_proj = self.wgate(hidden)  # [B, S, coff*head_dim]

        kv = self._reshape_into_windows(kv_proj)  # [B, N, ratio, coff*head_dim]
        score = self._reshape_into_windows(score_proj)  # [B, N, ratio, coff*head_dim]

        if self.overlap:
            kv = self._overlap_transform(kv)  # [B, N, 2*ratio, head_dim]
            score = self._overlap_transform(score)  # [B, N, 2*ratio, head_dim]
        # else: kv / score already at [B, N, ratio, head_dim]

        # APE adds a per-window-slot bias to the score (broadcast over B, N).
        score = score + self.ape  # [B, N, win, head_dim]

        # Softmax over the per-window axis (dim=2). Each compressed token is
        # a weighted average of its window members.
        weights = F.softmax(score.float(), dim=2).to(kv.dtype)
        pooled = (kv * weights).sum(dim=2)  # [B, N, head_dim]

        pooled = self.kv_norm(pooled)
        return pooled


__all__ = ["Compressor"]
