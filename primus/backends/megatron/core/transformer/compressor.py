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

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from primus.backends.megatron.core.transformer.local_rmsnorm import LocalRMSNorm


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
        self._proj_out = proj_out
        # Fuse the kv + gate projections into ONE [hidden -> 2*proj_out] GEMM
        # (default-on): ~1.5x on the projection and one launch instead of two.
        # PRIMUS_COMPRESS_FUSE_PROJ=0 restores the two separate linears.
        self._fuse_proj = os.environ.get("PRIMUS_COMPRESS_FUSE_PROJ", "1") != "0"
        if self._fuse_proj:
            self.wkv_gate = nn.Linear(hidden_size, 2 * proj_out, bias=False)
        else:
            self.wkv = nn.Linear(hidden_size, proj_out, bias=False)
            self.wgate = nn.Linear(hidden_size, proj_out, bias=False)

        # Learnable absolute position embedding (APE) added on top of the
        # softmax score. After overlap, the effective window length is
        # ``2*ratio`` slots of size ``head_dim``; in non-overlap mode it's
        # ``ratio`` slots of size ``head_dim``.
        ape_len = 2 * ratio if self.overlap else ratio
        self.ape = nn.Parameter(torch.zeros(ape_len, head_dim))
        nn.init.normal_(self.ape, std=0.02)

        self.kv_norm = LocalRMSNorm(head_dim, eps=rmsnorm_eps)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Bridge checkpoints across the fused/unfused projection layouts.

        Old checkpoints store ``wkv.weight`` + ``wgate.weight``; the fused path
        wants ``wkv_gate.weight`` = ``cat([wkv, wgate])`` (and vice-versa). Remap
        in-place so either layout loads under either runtime setting.
        """
        wkv_k, wgate_k, fused_k = prefix + "wkv.weight", prefix + "wgate.weight", prefix + "wkv_gate.weight"
        if self._fuse_proj and wkv_k in state_dict and fused_k not in state_dict:
            state_dict[fused_k] = torch.cat([state_dict.pop(wkv_k), state_dict.pop(wgate_k)], dim=0)
        elif (not self._fuse_proj) and fused_k in state_dict and wkv_k not in state_dict:
            w = state_dict.pop(fused_k)
            state_dict[wkv_k], state_dict[wgate_k] = w[: self._proj_out], w[self._proj_out :]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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
        if self._fuse_proj:
            kv_proj, score_proj = self.wkv_gate(hidden).split(self._proj_out, dim=-1)
        else:
            kv_proj = self.wkv(hidden)  # [B, S, coff*head_dim]
            score_proj = self.wgate(hidden)  # [B, S, coff*head_dim]

        kv = self._reshape_into_windows(kv_proj)  # [B, N, ratio, coff*head_dim]
        score = self._reshape_into_windows(score_proj)  # [B, N, ratio, coff*head_dim]

        if self.overlap:
            kv = self._overlap_transform(kv)  # [B, N, 2*ratio, head_dim]
            score = self._overlap_transform(score)  # [B, N, 2*ratio, head_dim]
        # else: kv / score already at [B, N, ratio, head_dim]

        # Per-window-softmax pool: APE bias + softmax over the window axis (dim=2)
        # + weighted sum -- each compressed token is a softmax-weighted average of
        # its window members. The forward burst (add + cast + softmax + cast + mul
        # + reduce) is fused into one Triton launch on CUDA fp16/bf16/fp32 inputs;
        # PRIMUS_COMPRESS_POOL_TRITON=0 (or non-CUDA / unsupported dtype) falls back
        # to eager.
        if (
            os.environ.get("PRIMUS_COMPRESS_POOL_TRITON", "1") != "0"
            and kv.is_cuda
            and kv.dtype
            in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            )
        ):
            from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.compressor_pool import (
                fused_softmax_weighted_pool,
            )

            pooled = fused_softmax_weighted_pool(kv, score, self.ape)  # [B, N, head_dim]
        else:
            score = score + self.ape  # [B, N, win, head_dim]
            weights = F.softmax(score.float(), dim=2).to(kv.dtype)
            pooled = (kv * weights).sum(dim=2)  # [B, N, head_dim]

        pooled = self.kv_norm(pooled)
        return pooled


__all__ = ["Compressor"]
