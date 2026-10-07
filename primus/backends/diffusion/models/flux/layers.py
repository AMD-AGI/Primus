###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# Adapted from Black Forest Labs FLUX official implementation.

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from primus.backends.diffusion.models.flux.fused_norm_rope import fused_qkv_norm_rope
from primus.backends.diffusion.models.flux.math import attention, rope

# Read once at import so it is a Python constant when Dynamo traces a block.
FUSED_NORM_ROPE = os.environ.get("FLUX_FUSED_NORM_ROPE", "0").lower() in (
    "1",
    "true",
    "on",
)

FUSED_DGELU_LEGPACK = os.getenv("FLUX_FUSED_DGELU_LEGPACK", "0") == "1"
FUSED_DGELU_MLP = os.getenv("FLUX_FUSED_DGELU_MLP", "0") == "1"
FWD_GELU_PACK = os.getenv("FLUX_FWD_GELU_PACK", "0") == "1"
GATE_DGATE = os.getenv("FLUX_GATE_DGATE", "0") == "1"
ATTN_REGION = os.getenv("FLUX_ATTN_REGION", "0") == "1"
if FWD_GELU_PACK and not (FUSED_DGELU_LEGPACK and FUSED_DGELU_MLP):
    raise ValueError("FLUX_FWD_GELU_PACK=1 needs FLUX_FUSED_DGELU_LEGPACK=1 and FLUX_FUSED_DGELU_MLP=1")
if GATE_DGATE and (
    not FWD_GELU_PACK
    or os.getenv("FLUX_FP4_PASSES", "off") in ("", "off")
    or os.getenv("FLUX_FP4_H16_STOCK", "0") == "1"
):
    raise ValueError(
        "FLUX_GATE_DGATE=1 needs FLUX_FWD_GELU_PACK=1, FLUX_FP4_PASSES on and FLUX_FP4_H16_STOCK=0"
    )

if ATTN_REGION and FUSED_NORM_ROPE:
    from primus.backends.diffusion.models.flux import attn_region as _attn_region


def _fused_attention(qkv, qk_norm, heads, cos, sin):
    """rearrange + QKNorm + apply_rope in one kernel, then attention."""
    if ATTN_REGION:
        return _attn_region.single_attention(
            qkv, qk_norm.query_norm.weight, qk_norm.key_norm.weight, cos, sin, heads
        )
    from primus.backends.diffusion.models.flux.math import backend_attention

    q, k, v = fused_qkv_norm_rope(
        qkv,
        qk_norm.query_norm.weight,
        qk_norm.key_norm.weight,
        cos,
        sin,
        heads,
    )
    x = backend_attention(q=q, k=k, v=v, dtype=q.dtype)
    return rearrange(x, "B L H D -> B L (H D)")


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(2)


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> Tensor:
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def init_weights(self, init_std: float = 0.02) -> None:
        nn.init.normal_(self.in_layer.weight, std=init_std)
        nn.init.constant_(self.in_layer.bias, 0)
        nn.init.normal_(self.out_layer.weight, std=init_std)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.RMSNorm):
    def __init__(self, dim: int):
        # Match TorchTitan's nn.RMSNorm(dim): eps remains None and PyTorch
        # selects the epsilon for the BF16 input dtype at runtime.
        super().__init__(dim)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def init_weights(self) -> None:
        self.query_norm.reset_parameters()
        self.key_norm.reset_parameters()

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def init_weights(self) -> None:
        for layer in (self.qkv, self.proj):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        self.norm.init_weights()

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        if FUSED_NORM_ROPE:
            return self.proj(_fused_attention(qkv, self.norm, self.num_heads, pe.all_cos, pe.all_sin))
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        return self.proj(x)


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def init_weights(self) -> None:
        nn.init.constant_(self.lin.weight, 0)
        nn.init.constant_(self.lin.bias, 0)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return ModulationOut(*out[:3]), ModulationOut(*out[3:]) if self.is_double else None


def _gated_linear(lin: nn.Module, x: Tensor, gate: Tensor) -> Tensor:
    """``gate * lin(x)``, through MXFP4Linear.forward_gated when that exists."""
    if GATE_DGATE and hasattr(lin, "forward_gated"):
        return lin.forward_gated(x, gate)
    return gate * lin(x)


def _mlp_gelu_fused(mlp: nn.Sequential, x: Tensor, gate: Tensor) -> Tensor:
    """``gate * mlp(x)`` for Sequential(Linear, GELU(tanh), Linear), with the first Linear and
    the GELU as one MXFP4Linear.forward_gelu call when that exists."""
    if FUSED_DGELU_MLP and hasattr(mlp[0], "forward_gelu"):
        if FWD_GELU_PACK and mlp[0].lazy_gelu_into(mlp[2], x, (mlp[2].in_features,)):
            act = mlp[0].forward_gelu(x, lazy=True)
            if GATE_DGATE:
                return mlp[2].forward_gelu_input(None, act, gate=gate)
            return gate * mlp[2].forward_gelu_input(None, act)
        return _gated_linear(mlp[2], mlp[0].forward_gelu(x), gate)
    return gate * mlp(x)


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def init_weights(self) -> None:
        for layer in (self.img_mlp[0], self.img_mlp[2], self.txt_mlp[0], self.txt_mlp[2]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        for layer in (self.img_attn, self.img_mod, self.txt_attn, self.txt_mod):
            layer.init_weights()
        for norm in (self.txt_norm1, self.txt_norm2, self.img_norm1, self.img_norm2):
            norm.reset_parameters()

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        img_modulated = (1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        txt_modulated = (1 + txt_mod1.scale) * self.txt_norm1(txt) + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)

        if FUSED_NORM_ROPE and ATTN_REGION:
            txt_attn, img_attn = _attn_region.double_attention(
                txt_qkv,
                img_qkv,
                self.txt_attn.norm.query_norm.weight,
                self.txt_attn.norm.key_norm.weight,
                self.img_attn.norm.query_norm.weight,
                self.img_attn.norm.key_norm.weight,
                pe,
                self.num_heads,
            )
        elif FUSED_NORM_ROPE:
            # Rotating each side with its own table rows is identical to
            # rotating the concatenation, because RoPE is positional.
            img_q, img_k, img_v = fused_qkv_norm_rope(
                img_qkv,
                self.img_attn.norm.query_norm.weight,
                self.img_attn.norm.key_norm.weight,
                pe.img_cos,
                pe.img_sin,
                self.num_heads,
                round_grad=True,
            )
            txt_q, txt_k, txt_v = fused_qkv_norm_rope(
                txt_qkv,
                self.txt_attn.norm.query_norm.weight,
                self.txt_attn.norm.key_norm.weight,
                pe.txt_cos,
                pe.txt_sin,
                self.num_heads,
                round_grad=True,
            )
            from primus.backends.diffusion.models.flux.math import backend_attention

            q = torch.cat((txt_q, img_q), dim=1)
            k = torch.cat((txt_k, img_k), dim=1)
            v = torch.cat((txt_v, img_v), dim=1)
            attn = rearrange(backend_attention(q=q, k=k, v=v, dtype=q.dtype), "B L H D -> B L (H D)")
        else:
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
            img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
            txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
            q = torch.cat((txt_q, img_q), dim=1)
            k = torch.cat((txt_k, img_k), dim=1)
            v = torch.cat((txt_v, img_v), dim=1)
            attn = attention(q, k, v, pe=pe)
        if not (FUSED_NORM_ROPE and ATTN_REGION):
            txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        img = img + _gated_linear(self.img_attn.proj, img_attn, img_mod1.gate)
        img = img + _mlp_gelu_fused(
            self.img_mlp, (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift, img_mod2.gate
        )
        txt = txt + _gated_linear(self.txt_attn.proj, txt_attn, txt_mod1.gate)
        txt = txt + _mlp_gelu_fused(
            self.txt_mlp, (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift, txt_mod2.gate
        )
        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, qk_scale: float | None = None
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def init_weights(self) -> None:
        for layer in (self.linear1, self.linear2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        self.norm.init_weights()
        self.pre_norm.reset_parameters()
        self.modulation.init_weights()

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        lazy_act = False
        if FUSED_DGELU_LEGPACK and hasattr(self.linear1, "forward_split_gelu"):
            lazy_act = FWD_GELU_PACK and self.linear1.lazy_gelu_into(
                self.linear2, x_mod, (self.hidden_size, self.mlp_hidden_dim)
            )
            qkv, mlp_act = self.linear1.forward_split_gelu(x_mod, 3 * self.hidden_size, lazy=lazy_act)
        else:
            qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
            mlp_act = self.mlp_act(mlp)
        if FUSED_NORM_ROPE:
            attn = _fused_attention(qkv, self.norm, self.num_heads, pe.all_cos, pe.all_sin)
        else:
            q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
            q, k = self.norm(q, k, v)
            attn = attention(q, k, v, pe=pe)
        if lazy_act:
            if GATE_DGATE:
                return x + self.linear2.forward_gelu_input(attn, mlp_act, gate=mod.gate)
            return x + mod.gate * self.linear2.forward_gelu_input(attn, mlp_act)
        return x + _gated_linear(self.linear2, torch.cat((attn, mlp_act), 2), mod.gate)


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def init_weights(self) -> None:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        self.norm_final.reset_parameters()

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        return self.linear(x)
