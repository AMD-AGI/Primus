###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# Vendored from `Wan2.2/wan/modules/model.py` with minimal changes:
# - remove diffusers dependencies (ModelMixin/ConfigMixin/register_to_config)
# - keep pure PyTorch modeling as close as possible
# - provide a `DiTBlock` alias class so existing Wan configs can keep
#   `fsdp_transformer_layer_cls_to_wrap: "DiTBlock"`

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from primus.backends.diffusion.attention import attention
from primus.backends.diffusion.distributed.ulysses import (
    distributed_attention,
    sp_gather,
    sp_split,
    sp_unpad,
)

__all__ = ["WanModel", "DiTBlock"]


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs, sp_group=None):
    """
    Apply 3-D RoPE.  When *sp_group* is given the input is assumed to hold
    only this rank's local token chunk (S/P) and the correct positional
    frequencies are selected automatically (following Wan2.2 official).
    """
    n, c = x.size(2), x.size(3) // 2
    s = x.size(1)  # local token count (= full seq when no SP)

    if sp_group is not None:
        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
    else:
        sp_size, sp_rank = 1, 0

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # tokens to process: local chunk when SP, valid tokens when single-GPU
        t = s if sp_size > 1 else seq_len

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :t].to(torch.float64).reshape(t, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # SP: pad freqs to padded-full-length, select this rank's range
        if sp_size > 1:
            full_len = s * sp_size
            if seq_len < full_len:
                freqs_i = torch.cat(
                    [
                        freqs_i,
                        torch.ones(
                            full_len - seq_len,
                            1,
                            freqs_i.size(-1),
                            dtype=freqs_i.dtype,
                            device=freqs_i.device,
                        ),
                    ],
                    dim=0,
                )
            freqs_i = freqs_i[sp_rank * s : (sp_rank + 1) * s]

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, t:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # Keep numerical stability by normalizing in fp32, but do NOT rely on
        # module parameters being fp32 (FSDP/bf16 can cast weights/bias).
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        y = F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps)
        return y.type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, sp_group: Optional[dist.ProcessGroup] = None):
        r"""
        Args:
            x(Tensor): Shape [B, L, C] (L = S/P when SP enabled)
            seq_lens(Tensor): Shape [B] — valid lengths in the *full* sequence
            grid_sizes(Tensor): Shape [B, 3], (F, H, W) — full spatial grid
            freqs(Tensor): Rope freqs
            sp_group: Ulysses SP process group (None = no SP)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        x_ = x.to(dtype=self.q.weight.dtype)
        q = self.norm_q(self.q(x_)).view(b, s, n, d)
        k = self.norm_k(self.k(x_)).view(b, s, n, d)
        v = self.v(x_).view(b, s, n, d)

        # RoPE on local tokens (SP-aware: uses rank-offset freqs when sp_group is set)
        q = rope_apply(q, grid_sizes, freqs, sp_group=sp_group)
        k = rope_apply(k, grid_sizes, freqs, sp_group=sp_group)

        attn_kwargs = {
            "q_lens": seq_lens,
            "k_lens": seq_lens,
            "window_size": self.window_size,
            "dtype": self.q.weight.dtype,
        }
        if sp_group is not None:
            x = distributed_attention(q, k, v, group=sp_group, attention_fn=attention, **attn_kwargs)
        else:
            x = attention(q=q, k=k, v=v, **attn_kwargs)

        x = x.to(dtype=self.o.weight.dtype)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        x = x.to(dtype=self.q.weight.dtype)
        context = context.to(dtype=self.k.weight.dtype)

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        x = attention(q, k, v, k_lens=context_lens, dtype=self.q.weight.dtype)

        x = x.to(dtype=self.o.weight.dtype)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        sp_group: Optional[dist.ProcessGroup] = None,
    ):
        # Memory-critical:
        # `e` can be very large ([B, seq_len, 6, dim]). Keeping it in fp32 and
        # doing broadcast add in fp32 easily OOMs for Wan2.1 (more tokens).
        # Align to x.dtype (bf16/fp16) like `wan_new` implementation.
        e = e.to(dtype=x.dtype)
        modulation = self.modulation.to(dtype=x.dtype)
        e = (modulation.unsqueeze(0) + e).chunk(6, dim=2)

        # self-attention (Ulysses all-to-all happens inside self_attn)
        y = self.self_attn(
            self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens,
            grid_sizes,
            freqs,
            sp_group=sp_group,
        )
        x = x + y * e[2].squeeze(2)

        # cross-attention + FFN (no SP communication needed:
        # each rank's visual queries attend to full text keys independently)
        def cross_attn_ffn(x_, context_, context_lens_, e_):
            x_ = x_ + self.cross_attn(self.norm3(x_), context_, context_lens_)
            ffn_in = self.norm2(x_) * (1 + e_[4].squeeze(2)) + e_[3].squeeze(2)
            y_ = self.ffn(ffn_in)
            x_ = x_ + y_ * e_[5].squeeze(2)
            return x_

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


# Wan config compatibility: FSDP auto-wrap often uses "DiTBlock".
class DiTBlock(WanAttentionBlock):
    pass


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        # Align modulation embedding dtype to x (saves memory vs fp32 broadcast).
        e = e.to(dtype=x.dtype)
        modulation = self.modulation.to(dtype=x.dtype)
        e = (modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


class WanModel(nn.Module):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        super().__init__()

        assert model_type in ["t2v", "i2v", "ti2v", "s2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        # Used by the Wan trainer: it sets `model.dit.gradient_checkpointing = True`
        # when `trainer_args.gradient_checkpointing: true`.
        self.gradient_checkpointing = False

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks (use DiTBlock for YAML compatibility)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # rope buffers (avoid register_buffer to keep dtype stable under .to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        self.init_weights()

    def forward(self, x, t, context, seq_len, y=None, sp_group: Optional[dist.ProcessGroup] = None):
        if self.model_type == "i2v":
            assert y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # time embeddings
        if t.dim() == 1:
            # [B] -> [B, seq_len]: per-sample timestep broadcast across the sequence.
            # Must unsqueeze before expand, otherwise the batch dim (B) is wrongly
            # aligned with seq_len and breaks for batch_size > 1.
            t = t.unsqueeze(1).expand(t.size(0), seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = t.size(0)
            t_flat = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float()
            )
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # Cast large modulation tensors back to model dtype early to reduce peak memory.
        e = e.to(dtype=x.dtype)
        e0 = e0.to(dtype=x.dtype)

        # --- Ulysses SP: split sequence-dim tensors across SP ranks ---
        original_seq_len = None
        if sp_group is not None:
            (x, e, e0), original_seq_len = sp_split([x, e, e0], dim=1, group=sp_group)

        # context (text embeddings — NOT sliced, full text on every SP rank)
        context_lens = None
        context_in = torch.stack(
            [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
        )
        context_in = context_in.to(dtype=self.text_embedding[0].weight.dtype)
        context = self.text_embedding(context_in)

        freqs = self.freqs
        seq_lens_ = seq_lens
        grid_sizes_ = grid_sizes
        e0_ = e0
        context_ = context
        context_lens_ = context_lens

        for block in self.blocks:
            if self.training and getattr(self, "gradient_checkpointing", False):

                def create_custom_forward(module, _sp_group):
                    def custom_forward(x_in, e_in, ctx_in):
                        return module(
                            x_in,
                            e=e_in,
                            seq_lens=seq_lens_,
                            grid_sizes=grid_sizes_,
                            freqs=freqs,
                            context=ctx_in,
                            context_lens=context_lens_,
                            sp_group=_sp_group,
                        )

                    return custom_forward

                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, sp_group),
                    x,
                    e0_,
                    context_,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    e=e0_,
                    seq_lens=seq_lens_,
                    grid_sizes=grid_sizes_,
                    freqs=freqs,
                    context=context_,
                    context_lens=context_lens_,
                    sp_group=sp_group,
                )

        # head operates on local tokens (like Wan2.2 official)
        x = self.head(x, e)

        # --- Ulysses SP: gather output back to full sequence ---
        if sp_group is not None:
            x = sp_unpad(sp_gather(x, dim=1, group=sp_group), dim=1, original_size=original_seq_len)

        x = self.unpatchify(x, grid_sizes)
        # IMPORTANT:
        # Returning fp32 here dramatically increases activation/grad memory and slows
        # training (DiffSynth returns bf16 here). Keep dtype consistent with model.
        return x

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
