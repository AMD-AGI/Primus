# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
WAN building blocks: 3D rotary position embedding and condition embedders.

Flux's counterpart is ``flux/layers.py`` (``EmbedND``), but none of it is
reusable here: Flux embeds 2D image positions plus a text stream, whereas WAN
embeds a 3D ``(frame, height, width)`` patch grid.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..common.embeddings import Timesteps


class WanRotaryPosEmbed(nn.Module):
    """WAN 3D RoPE producing mcore ``apply_rotary_pos_emb`` frequencies.

    Builds per-axis angle tables, concatenates the (frame, height, width)
    slices for the patch grid, then doubles the last dim as
    ``(a0, a0, a1, a1, ...)`` so mcore's interleaved rotary matches diffusers'
    ``WanAttnProcessor`` interleaving.

    Output: ``[S, 1, 1, head_dim]`` angles, pre cos/sin.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 1024, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        # The temporal axis takes the remainder so t + h + w == head_dim / 2.
        h = head_dim // 6
        freqs = torch.cat(
            [
                self._rope_params(max_seq_len, head_dim - 4 * h, theta),
                self._rope_params(max_seq_len, 2 * h, theta),
                self._rope_params(max_seq_len, 2 * h, theta),
            ],
            dim=1,
        )
        self.register_buffer("freqs", freqs, persistent=False)

    @staticmethod
    def _rope_params(max_seq_len: int, dim: int, theta: float) -> Tensor:
        if dim % 2 != 0:
            raise ValueError(f"RoPE axis dim must be even, got {dim}")
        return torch.outer(
            torch.arange(max_seq_len, dtype=torch.float32),
            1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float32) / dim),
        )

    def forward(self, num_frames: int, height: int, width: int, device: torch.device) -> Tensor:
        c = self.head_dim // 2
        freqs = self.freqs.to(device).split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        f, hgt, wid = num_frames, height, width
        seq_len = f * hgt * wid
        fi = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, hgt, wid, -1),
                freqs[1][:hgt].view(1, hgt, 1, -1).expand(f, hgt, wid, -1),
                freqs[2][:wid].view(1, 1, wid, -1).expand(f, hgt, wid, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, 1, -1)
        fi = fi.unsqueeze(-1).expand(-1, -1, -1, -1, 2).reshape(seq_len, 1, 1, self.head_dim)
        return fi.contiguous()


class WanTimestepEmbedding(nn.Module):
    """diffusers ``TimestepEmbedding``: linear_1 -> SiLU -> linear_2."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


class WanTextProjection(nn.Module):
    """diffusers ``PixArtAlphaTextProjection(act_fn='gelu_tanh')``."""

    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(self.act_1(self.linear_1(x)))


class WanConditionEmbedder(nn.Module):
    """diffusers ``WanTimeTextImageEmbedding`` for T2V (no image branch).

    Returns ``(temb, timestep_proj, text_ctx)`` where ``temb`` drives the final
    AdaLN and ``timestep_proj`` (``dim * 6``) drives per-block AdaLN modulation.
    """

    def __init__(self, dim: int, time_freq_dim: int, time_proj_dim: int, text_embed_dim: int):
        super().__init__()
        self.timesteps_proj = Timesteps(
            embedding_dim=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = WanTimestepEmbedding(time_freq_dim, dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = WanTextProjection(text_embed_dim, dim)

    def forward(self, timestep: Tensor, encoder_hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # ``Timesteps`` casts back to the input dtype, so feed it a float to
        # avoid truncating the sinusoid when ``timestep`` is integer.
        if not torch.is_floating_point(timestep):
            timestep = timestep.float()
        timestep = self.timesteps_proj(timestep)

        weight_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != weight_dtype:
            timestep = timestep.to(weight_dtype)

        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        text_ctx = self.text_embedder(encoder_hidden_states)
        return temb, timestep_proj, text_ctx


def thd_cu_seqlens(seqlen: int, batch: int, device: torch.device) -> Tensor:
    """Cumulative sequence lengths for ``batch`` sequences of ``seqlen`` tokens.

    ``[0, seqlen, 2*seqlen, ..., batch*seqlen]`` as int32, the layout
    ``TEDotProductAttention`` and the fused THD RoPE expect for packed
    sequences.
    """
    return torch.arange(0, batch + 1, dtype=torch.int32, device=device) * int(seqlen)


def unfused_fp32_attention(query, key, value, packed_seq_params) -> Tensor:
    """Scaled dot-product attention over packed THD sequences in plain fp32.

    ``query``/``key``/``value`` are ``[T, H, D]`` packed in THD layout, with
    ``cu_seqlens_{q,kv}`` delimiting the individual sequences so
    cross-attention's differing q/kv lengths work. Returns ``[T, H*D]``.

    Deliberately bare ``matmul`` + ``softmax``: there is no fp32 fused
    attention backend, and this path exists so a parity run can be bit-exact
    against a reference stack running the same routine.
    """
    heads, head_dim = query.shape[1], query.shape[2]
    scale = 1.0 / math.sqrt(head_dim)
    cu_q = packed_seq_params.cu_seqlens_q
    cu_kv = packed_seq_params.cu_seqlens_kv
    outs = []
    for i in range(cu_q.numel() - 1):
        qs, qe = int(cu_q[i]), int(cu_q[i + 1])
        ks, ke = int(cu_kv[i]), int(cu_kv[i + 1])
        q = query[qs:qe].transpose(0, 1)
        k = key[ks:ke].transpose(0, 1)
        v = value[ks:ke].transpose(0, 1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)
        outs.append(ctx.transpose(0, 1).reshape(qe - qs, heads * head_dim))
    return torch.cat(outs, dim=0)


__all__ = [
    "WanRotaryPosEmbed",
    "WanTimestepEmbedding",
    "WanTextProjection",
    "WanConditionEmbedder",
    "thd_cu_seqlens",
    "unfused_fp32_attention",
]
