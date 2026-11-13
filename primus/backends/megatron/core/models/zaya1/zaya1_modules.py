###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ZAYA1 mixer modules.

Per-sublayer residual scaling has the form ``(x + bias) * scale``. CCA uses a
raw key temperature, and the router is a vector EDA router with optional
mixture-of-depths. Tensors use Megatron layout ``[sequence, batch, hidden]``.

When ``config.fp8`` is set, CCA projections and expert SwiGLU GEMMs are
Transformer Engine linears under one FP8 autocast. The router, the depthwise
convolutions, RMSNorm, and attention stay in BF16 or FP32.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from primus.backends.megatron.core.utils import maybe_compile

# zaya_attn_flash (default on): drop a purely causal dense mask once per forward
# and run CCA attention through flash_attn_func with native GQA, or through
# SDPA enable_gqa when flash_attn is unavailable. Off restores the dense-mask +
# repeat_interleave path.
try:
    from flash_attn import flash_attn_func as _flash_attn_func
except Exception:  # pragma: no cover - depends on the image
    _flash_attn_func = None


# zaya_fuse_norm (default on): run each stage's fp32 elementwise chain
# (ResidualScaling affine, residual merge, RMSNorm) as one torch.compile'd
# region instead of ~12 eager kernels over [tokens, hidden] fp32 tensors, fwd
# and bwd, in all 80 stages. zaya_fuse_cca (default on) does the same for the
# CCA query/key post-processing (grouped means, qk-norm, temperature, partial
# RoPE). Both are free functions with tensor arguments, so every layer shares
# one graph (no per-module recompiles). Off runs the same functions uncompiled.

# zaya_router_defer_ar (default off): accumulate each router's token counts
# during the forward and apply all balancing-bias updates after the decoder's
# backward (or, failing that, at the start of the next forward) with one
# all-reduce for all MoE layers instead of one per layer. The bias only feeds
# topk before its own update, so with one microbatch per step (or
# non-pipelined microbatches) the bias each forward sees is bit-identical to
# the per-layer path.


# By default the experts run split: per-expert nn.Linear GEMMs over host-sized
# chunks (three host syncs per MoE layer: bincount reads its input's min and
# max, then counts.tolist()). With Primus Turbo's enable_primus_turbo and
# use_turbo_grouped_gemm, the 16 experts' weights live in two stacked
# [E, out, in] parameters and run as two Primus Turbo grouped GEMMs whose
# group sizes stay on device, so a MoE layer never syncs with the host;
# turbo_grouped_gemm_backend picks the kernel. Only Turbo's BF16 grouped GEMM
# is wired up here; Turbo's FP8 grouped GEMM isn't yet, so under FP8 the
# experts stay split. Checkpoint keys stay local_experts.N.linear_fc{1,2}.weight
# either way; the distributed-optimizer state is shaped per parameter, so it
# does not carry across a switch (load with --no-load-optim when switching).
def _megatron_args():
    """Megatron's parsed arguments, which hold the Primus Turbo settings, or None.

    The Turbo settings are Megatron arguments, not TransformerConfig fields.
    get_args raises before Megatron is initialized, as in the CPU unit tests,
    which reads as every Turbo setting off.
    """
    try:
        from megatron.training import get_args

        return get_args()
    except Exception:
        return None


def _turbo_experts(config) -> Tuple[bool, str]:
    """Whether the experts run as Primus Turbo grouped GEMMs, and their path as the build log names it."""
    args = _megatron_args()
    if not (getattr(args, "enable_primus_turbo", False) and getattr(args, "use_turbo_grouped_gemm", False)):
        return False, "split"
    if _fp8_requested(config):
        return False, "split (Turbo's FP8 grouped GEMM isn't wired up for ZAYA1 yet)"
    backend = getattr(args, "turbo_grouped_gemm_backend", "default")
    return True, f"Primus Turbo grouped GEMM (backend: {backend})"


def _publish_turbo_grouped_gemm_backend() -> None:
    """Hand turbo_grouped_gemm_backend to Turbo the way Primus's own grouped linear does."""
    from primus.backends.megatron.core.extensions.primus_turbo import (
        _apply_turbo_grouped_gemm_backend_env,
    )

    _apply_turbo_grouped_gemm_backend_env()


_PATH_DEFAULTS = {
    "zaya_attn_flash": True,
    "zaya_fuse_norm": True,
    "zaya_fuse_cca": True,
    "zaya_conv_gemm": True,
    "zaya_fuse_moe": True,
    "zaya_router_defer_ar": False,
}


def _path_on(config, name: str) -> bool:
    """A kernel-path field of ``config``; configs without it, like the unit tests' stand-ins, get its default."""
    return bool(getattr(config, name, _PATH_DEFAULTS[name]))


def switch_summary(config) -> str:
    """The kernel-path fields and the experts' path in effect for ``config``, as one line for the build log."""

    def on(name: str) -> str:
        return "on" if _path_on(config, name) else "off"

    if not _path_on(config, "zaya_attn_flash"):
        attention = "dense mask"
    else:
        attention = "flash_attn" if _flash_attn_func is not None else "SDPA, flash_attn not importable"
    return (
        f"zaya_attn_flash={on('zaya_attn_flash')} ({attention}), zaya_fuse_norm={on('zaya_fuse_norm')}, "
        f"zaya_fuse_cca={on('zaya_fuse_cca')}, zaya_conv_gemm={on('zaya_conv_gemm')}, "
        f"zaya_fuse_moe={on('zaya_fuse_moe')}, zaya_router_defer_ar={on('zaya_router_defer_ar')}; "
        f"experts: {_turbo_experts(config)[1]}"
    )


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv * weight.float()).to(dtype=x.dtype)


def _scale_merge_norm_res(hidden, residual, h_bias, h_scale, r_bias, r_scale, norm_w, eps: float):
    """ResidualScaling (with residual) + merge_residual + RMSNorm, same op order as eager."""
    target_dtype = hidden.dtype
    h = (hidden.float() + h_bias.float()) * h_scale.float()
    r = (residual.float() + r_bias.float()) * r_scale.float()
    merged = r + h
    return merged, _rms(merged.to(dtype=target_dtype), norm_w, eps)


def _scale_merge_norm_first(hidden, h_bias, h_scale, norm_w, eps: float):
    """Stage 0: no incoming residual, so merged is the scaled mixer input."""
    target_dtype = hidden.dtype
    merged = (hidden.float() + h_bias.float()) * h_scale.float()
    return merged, _rms(merged.to(dtype=target_dtype), norm_w, eps)


_fused_smn_res = maybe_compile(_scale_merge_norm_res)
_fused_smn_first = maybe_compile(_scale_merge_norm_first)


def layer_kinds(num_layers: int, zaya_layers: Optional[Sequence] = None) -> List[str]:
    """Return ``"a"`` for CCA and ``"m"`` for MoE, one entry per stage.

    Absent an explicit list, even stages are CCA and odd stages are MoE, which
    is the ``zaya_layers`` pattern of ``Zyphra/ZAYA1-base`` (40 + 40).
    """
    if zaya_layers:
        kinds = []
        for entry in zaya_layers:
            if entry == "a":
                kinds.append("a")
            else:
                kinds.append("m")
        if len(kinds) != num_layers:
            raise ValueError(f"zaya_layers has length {len(kinds)}, num_layers is {num_layers}.")
        return kinds
    return ["a" if i % 2 == 0 else "m" for i in range(num_layers)]


def _init_weight(tensor: torch.Tensor, config) -> None:
    init = getattr(config, "init_method", None)
    if callable(init):
        init(tensor)
    else:
        nn.init.normal_(tensor, mean=0.0, std=0.02)


def _fp8_requested(config) -> bool:
    return bool(getattr(config, "fp8", None))


def _align_fp8_leading(hidden: torch.Tensor):
    """Pad the token axis so the leading-dimension product is a multiple of 8."""
    leading = 1
    for size in hidden.shape[:-1]:
        leading *= int(size)
    pad = (8 - leading % 8) % 8
    if pad == 0:
        return hidden, None
    flat = hidden.reshape(leading, hidden.shape[-1])
    return F.pad(flat, (0, 0, 0, pad)), (hidden.shape, leading)


def _unpad_fp8_leading(output: torch.Tensor, meta) -> torch.Tensor:
    if meta is None:
        return output
    shape, leading = meta
    return output[:leading].reshape(*shape[:-1], output.shape[-1])


class _Fp8Linear(nn.Module):
    """TE linear whose forward returns a tensor, matching ``nn.Linear``.

    Megatron's ``TELinear`` always returns ``(output, bias)``. CCA and the
    SwiGLU experts call the projection as a single tensor. ``parallel_mode``
    is ``duplicated`` because this port is data-parallel only.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool, config) -> None:
        super().__init__()
        from megatron.core.extensions.transformer_engine import TELinear

        init = getattr(config, "init_method", None)
        if not callable(init):

            def init(tensor):
                nn.init.normal_(tensor, mean=0.0, std=getattr(config, "init_method_std", 0.02))

        self.linear = TELinear(
            in_features,
            out_features,
            parallel_mode="duplicated",
            config=config,
            init_method=init,
            bias=bias,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # TE FP8 requires the product of every dimension except the last to be
        # divisible by 8. Expert slices often are not: a top-1 route of 128
        # tokens across 16 experts can hand one expert 4 tokens.
        flat, meta = _align_fp8_leading(hidden)
        output, _ = self.linear(flat)
        return _unpad_fp8_leading(output, meta)


def _linear(in_features: int, out_features: int, bias: bool, config, *, fp8: bool = True) -> nn.Module:
    """BF16 ``nn.Linear``, or a Transformer Engine linear when FP8 training is on.

    The router passes ``fp8=False``. Its last projection scores 17 classes
    (16 experts plus the mixture-of-depths skip), which is not a multiple of
    16, and the router softmax already runs in fp32.
    """
    if fp8 and _fp8_requested(config):
        if in_features % 16 or out_features % 16:
            raise ValueError(
                "ZAYA1 FP8 GEMMs require in and out features divisible by 16, "
                f"got {in_features} -> {out_features}. The ZAYA1-8B shape "
                "(hidden 2048) satisfies this. The router stays in BF16."
            )
        return _Fp8Linear(in_features, out_features, bias, config)
    layer = nn.Linear(in_features, out_features, bias=bias)
    _init_weight(layer.weight, config)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class RMSNorm(nn.Module):
    """RMSNorm with a learnable scale. Statistics are computed in fp32."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms(x, self.weight, self.eps)


class ResidualScaling(nn.Module):
    """Parameters of the affine ``(x + bias) * scale`` on the mixer output and the residual.

    ``_scale_merge_norm_res`` and ``_scale_merge_norm_first`` apply it. Stage 0
    has no incoming residual, so it stores only the mixer-output affine.
    """

    def __init__(self, hidden_size: int, has_residual: bool) -> None:
        super().__init__()
        self.has_residual = has_residual
        self.hidden_states_scale = nn.Parameter(torch.ones(hidden_size))
        self.hidden_states_bias = nn.Parameter(torch.zeros(hidden_size))
        if has_residual:
            self.residual_scale = nn.Parameter(torch.ones(hidden_size))
            self.residual_bias = nn.Parameter(torch.zeros(hidden_size))


def merge_residual(residual: Optional[torch.Tensor], hidden_states: torch.Tensor) -> torch.Tensor:
    if residual is None:
        return hidden_states.float()
    return residual.float() + hidden_states.float()


def _scale_merge_norm(rs: Optional[ResidualScaling], norm: RMSNorm, hidden, residual, compiled: bool):
    """Residual affine, merge and RMSNorm ahead of a mixer or as the final norm: ``(merged, normed)``.

    ``compiled`` (zaya_fuse_norm) runs the compiled function instead of the plain one.
    """
    if rs is None:
        merged = merge_residual(residual, hidden)
        return merged, norm(merged.to(dtype=hidden.dtype))
    if residual is None:
        first = _fused_smn_first if compiled else _scale_merge_norm_first
        return first(hidden, rs.hidden_states_bias, rs.hidden_states_scale, norm.weight, norm.eps)
    res = _fused_smn_res if compiled else _scale_merge_norm_res
    return res(
        hidden,
        residual,
        rs.hidden_states_bias,
        rs.hidden_states_scale,
        rs.residual_bias,
        rs.residual_scale,
        norm.weight,
        norm.eps,
    )


def _cca_qk_post(
    query_conv,
    key_conv,
    query_pre,
    key_base,
    temp,
    position_ids,
    num_q_heads: int,
    num_k_heads: int,
    gqa_groups: int,
    head_dim: int,
    clamp_temp: bool,
    rotary_dim: int,
    rope_base: float,
):
    """CCA's query/key post-processing: grouped means, qk-norm and temperature, partial RoPE.

    A function of tensors, so one torch.compile of it is shared by all 40 CCA
    stages. Output dtype is ``query_pre.dtype``. ``position_ids`` is
    ``[batch, seq]``; RoPE uses the NeoX layout, where the two halves of the
    rotary slice share each angle.
    """
    dtype = query_pre.dtype
    seq, batch = query_pre.shape[0], query_pre.shape[1]
    # Grouped means.
    key_base_f = key_base.float()
    query_pre_g = query_pre.view(seq, batch, num_k_heads, gqa_groups, head_dim).float()
    query_conv_g = query_conv.view_as(query_pre_g).float()
    query = query_conv_g + 0.5 * query_pre_g + 0.5 * key_base_f.unsqueeze(-2)
    query = query.reshape(seq, batch, num_q_heads, head_dim)
    query_mean = query_pre_g.mean(dim=-2)
    key = key_conv.float() + 0.5 * query_mean + 0.5 * key_base_f
    # qk-norm and temperature.
    eps = 1e-12
    scale = math.sqrt(head_dim)
    query = query.float() * (torch.rsqrt(query.float().pow(2).sum(-1, keepdim=True) + eps) * scale)
    key = key.float() * (torch.rsqrt(key.float().pow(2).sum(-1, keepdim=True) + eps) * scale)
    t = temp.float().view(1, 1, num_k_heads, 1)
    if clamp_temp:
        t = torch.exp(torch.clamp(t, 1e-7, 2.0))
    key = key * t
    query = query.to(dtype)
    key = key.to(dtype)
    # Partial RoPE.
    if rotary_dim == 0:
        return query, key
    if position_ids is None:
        pos = torch.arange(seq, device=query.device)
    else:
        pos = position_ids.transpose(0, 1).to(device=query.device)
    freq_idx = torch.arange(0, rotary_dim, 2, device=query.device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_base ** (freq_idx / rotary_dim))
    if pos.dim() == 1:
        freqs = torch.outer(pos.float(), inv_freq)
        cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)[:, None, None, :]
        sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)[:, None, None, :]
    else:
        freqs = torch.einsum("sb,d->sbd", pos.float(), inv_freq)
        cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)[:, :, None, :]
        sin = torch.sin(freqs)
        sin = torch.cat((sin, sin), dim=-1)[:, :, None, :]

    def _apply(x):
        rot, rest = x[..., :rotary_dim], x[..., rotary_dim:]
        rot_f = rot.float()
        half = rot_f.shape[-1] // 2
        x1, x2 = rot_f[..., :half], rot_f[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)
        out = rot_f * cos + rotated * sin
        return torch.cat((out, rest.float()), dim=-1).to(dtype=x.dtype)

    return _apply(query), _apply(key)


_fused_cca_qk = maybe_compile(_cca_qk_post)

# zaya_conv_gemm (default on): run CCA's two causal Conv1d layers without MIOpen.
# The depthwise kernel-2 conv and the per-head grouped kernel-2 conv otherwise
# hit naive_conv_ab_nonpacked fwd/bwd kernels (double accumulation) plus NCL
# transposes in all 40 CCA stages. Here the depthwise conv is a fused
# elementwise two-tap filter emitted directly in the [groups, tokens, 2*cg]
# layout, and the grouped conv is one baddbmm per stage. Same parameters, same
# math (fp32 accumulation, bf16 output). Set it off for nn.Conv1d.


def _cca_conv_taps(qk, w0, b0, groups: int):
    """Depthwise causal conv (kernel 2, left pad 2) and the grouped conv's two taps.

    ``qk`` is ``[s, b, C]``. With ``xp`` the input left-padded by two zeros, the
    depthwise output is ``y[u] = w0[:,0,0]*xp[u] + w0[:,0,1]*xp[u+1] + b0`` for
    ``u in 0..s``, so ``y[0] = b0`` and ``y[t+1]`` reads ``x[t-1], x[t]``. The
    grouped conv output at ``t`` reads ``y[t]`` (tap 0) and ``y[t+1]`` (tap 1).
    Returns ``[groups, s*b, 2*cg]`` holding ``(y[t], y[t+1])`` per group.
    """
    seq, batch, ch = qk.shape
    cg = ch // groups
    dtype = qk.dtype
    xf = qk.float()
    x_prev = F.pad(xf[:-1], (0, 0, 0, 0, 1, 0))
    w = w0.float()
    bias = b0.float()
    # Round to the conv's output dtype exactly where nn.Conv1d does.
    y_cur = (x_prev * w[:, 0, 0] + xf * w[:, 0, 1] + bias).to(dtype)
    y_first = bias.to(dtype).expand(1, batch, ch)
    y_prev = torch.cat((y_first, y_cur[:-1]), dim=0)
    taps = torch.cat(
        (y_prev.reshape(seq * batch, groups, cg), y_cur.reshape(seq * batch, groups, cg)), dim=-1
    )
    return taps.transpose(0, 1).contiguous()


_fused_conv_taps = maybe_compile(_cca_conv_taps)


def cca_causal_conv(qk: torch.Tensor, conv0: nn.Conv1d, conv1: nn.Conv1d) -> torch.Tensor:
    """``conv1(conv0(pad(qk)))`` for CCA's kernel-2 convs, in ``[s, b, C]`` layout."""
    seq, batch, ch = qk.shape
    groups = conv1.groups
    cg = ch // groups
    taps = _fused_conv_taps(qk, conv0.weight, conv0.bias, groups)  # [G, N, 2cg]
    # conv1.weight is [C_out, cg, 2]; output channel g*cg+o reads its own group.
    w = conv1.weight.view(groups, cg, cg, 2).permute(0, 3, 2, 1).reshape(groups, 2 * cg, cg)
    out = torch.baddbmm(conv1.bias.view(groups, 1, cg).to(taps.dtype), taps, w.to(taps.dtype))
    return out.transpose(0, 1).reshape(seq, batch, ch)


class CCA(nn.Module):
    """Compressed convolutional attention projection plus partial RoPE and GQA."""

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_q_heads = int(config.num_attention_heads)
        self.num_k_heads = int(getattr(config, "num_query_groups", None) or config.num_attention_heads)
        self.head_dim = int(config.kv_channels)
        if self.num_q_heads % self.num_k_heads != 0:
            raise ValueError("num_attention_heads must be a multiple of num_query_groups.")
        self.gqa_groups = self.num_q_heads // self.num_k_heads
        self.latent_q = self.num_q_heads * self.head_dim
        self.latent_k = self.num_k_heads * self.head_dim
        self.cca_time0 = int(getattr(config, "cca_time0", 2))
        self.cca_time1 = int(getattr(config, "cca_time1", 2))
        self.total_padding = (self.cca_time0 - 1) + (self.cca_time1 - 1)
        frac = getattr(config, "partial_rotary_factor", None)
        if frac is None:
            frac = getattr(config, "rotary_percent", 0.5)
        self.partial_rotary_factor = float(frac)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if self.rotary_dim % 2 != 0:
            raise ValueError(f"partial RoPE dim must be even, got {self.rotary_dim}.")
        self.rope_base = float(getattr(config, "rotary_base", 10000.0))
        self.clamp_temp = bool(getattr(config, "clamp_temp", False))
        bias = bool(getattr(config, "add_qkv_bias", False) or getattr(config, "attention_bias", False))

        self.linear_q = _linear(self.hidden_size, self.latent_q, bias, config)
        self.linear_k = _linear(self.hidden_size, self.latent_k, bias, config)
        self.val_proj1 = _linear(self.hidden_size, self.latent_k // 2, bias, config)
        self.val_proj2 = _linear(self.hidden_size, self.latent_k // 2, bias, config)
        in_ch = self.latent_q + self.latent_k
        self.conv_qk = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, self.cca_time0, groups=in_ch, bias=True),
            nn.Conv1d(
                in_ch,
                in_ch,
                self.cca_time1,
                groups=self.num_q_heads + self.num_k_heads,
                bias=True,
            ),
        )
        for conv in self.conv_qk:
            _init_weight(conv.weight, config)
            nn.init.zeros_(conv.bias)
        # The GEMM conv path covers the released shape: kernel 2 for both convs
        # (left pad 2) and conv1 groups that split channels evenly.
        self._conv_gemm_ok = (
            _path_on(config, "zaya_conv_gemm")
            and self.cca_time0 == 2
            and self.cca_time1 == 2
            and in_ch % (self.num_q_heads + self.num_k_heads) == 0
        )
        self.fuse_cca = _path_on(config, "zaya_fuse_cca")
        self.attn_flash = _path_on(config, "zaya_attn_flash")
        # Ones, not zeros: a zero default is the pre-checkpoint initialization, and a
        # raw temperature of zero would wipe the keys at step 0.
        self.temp = nn.Parameter(torch.ones(self.num_k_heads))
        self.o_proj = _linear(self.latent_q, self.hidden_size, bias, config)

    def project(self, hidden_states: torch.Tensor, position_ids: Optional[torch.Tensor]):
        seq, batch, _ = hidden_states.shape
        q_raw = self.linear_q(hidden_states)
        k_raw = self.linear_k(hidden_states)
        qk = torch.cat((q_raw, k_raw), dim=-1)
        if self._conv_gemm_ok:
            qk_out = cca_causal_conv(qk, self.conv_qk[0], self.conv_qk[1])
        else:
            qk_ncl = F.pad(qk.permute(1, 2, 0), (self.total_padding, 0))
            qk_out = self.conv_qk(qk_ncl).permute(2, 0, 1)
        query_pre = q_raw.view(seq, batch, self.num_q_heads, self.head_dim)
        key_base = k_raw.view(seq, batch, self.num_k_heads, self.head_dim)
        query_conv = qk_out[..., : self.latent_q].view(seq, batch, self.num_q_heads, self.head_dim)
        key_conv = qk_out[..., self.latent_q :].view(seq, batch, self.num_k_heads, self.head_dim)
        dtype = hidden_states.dtype
        qk_post = _fused_cca_qk if self.fuse_cca else _cca_qk_post
        query, key = qk_post(
            query_conv,
            key_conv,
            query_pre,
            key_base,
            self.temp,
            position_ids,
            self.num_q_heads,
            self.num_k_heads,
            self.gqa_groups,
            self.head_dim,
            self.clamp_temp,
            self.rotary_dim,
            self.rope_base,
        )

        shifted = F.pad(hidden_states[:-1], (0, 0, 0, 0, 1, 0))
        v1 = self.val_proj1(hidden_states)
        v2 = self.val_proj2(shifted)
        value = torch.cat((v1, v2), dim=-1).view(seq, batch, self.num_k_heads, self.head_dim)
        return query, key, value.to(dtype)

    def forward(self, hidden_states: torch.Tensor, position_ids: Optional[torch.Tensor], attention_mask):
        query, key, value = self.project(hidden_states, position_ids)
        attn = attention(query, key, value, attention_mask, flash=self.attn_flash)
        seq, batch = hidden_states.shape[:2]
        return self.o_proj(attn.reshape(seq, batch, self.latent_q))


def is_causal_mask(mask: Optional[torch.Tensor]) -> bool:
    """True when ``mask`` (Megatron convention, True blocks a key) is exactly causal.

    Accepts any shape broadcastable to ``[..., s, s]``. One device-to-host sync.
    """
    if mask is None:
        return True
    if mask.dim() < 2 or mask.shape[-1] != mask.shape[-2]:
        return False
    s = mask.shape[-1]
    ref = torch.ones(s, s, dtype=torch.bool, device=mask.device).triu_(1)
    m = mask if mask.dtype == torch.bool else mask.bool()
    return bool(torch.all(m == ref).item())


def _attention_causal_gqa(query, key, value) -> torch.Tensor:
    """Pure causal GQA without a dense mask or repeat_interleave. ``[s, b, h, d]`` in and out."""
    if (
        _flash_attn_func is not None
        and query.is_cuda
        and query.dtype in (torch.bfloat16, torch.float16)
        and query.dtype == key.dtype == value.dtype
        and query.shape[-1] <= 256
    ):
        # flash_attn layout is [b, s, h, d]; it handles h_q % h_kv == 0 natively.
        # softmax_scale=None means 1/sqrt(d), the same as the SDPA default.
        q = query.transpose(0, 1).contiguous()
        k = key.transpose(0, 1).contiguous()
        v = value.transpose(0, 1).contiguous()
        out = _flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=True)
        return out.transpose(0, 1)
    q = query.permute(1, 2, 0, 3)
    k = key.permute(1, 2, 0, 3)
    v = value.permute(1, 2, 0, 3)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=q.shape[1] != k.shape[1])
    return out.permute(2, 0, 1, 3)


def attention(query, key, value, attention_mask: Optional[torch.Tensor], flash: bool = False) -> torch.Tensor:
    """Causal GQA. ``attention_mask`` uses Megatron's convention: True blocks a key.

    ``flash`` is zaya_attn_flash: with no mask, run the causal kernel instead of a dense causal mask.
    """
    if flash and attention_mask is None:
        return _attention_causal_gqa(query, key, value)
    # [s, b, h, d] -> [b, h, s, d]
    q = query.permute(1, 2, 0, 3)
    k = key.permute(1, 2, 0, 3)
    v = value.permute(1, 2, 0, 3)
    if q.shape[1] != k.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    attn_mask = None
    is_causal = True
    if attention_mask is not None:
        is_causal = False
        mask = attention_mask
        if mask.dtype != torch.bool:
            mask = mask.bool()
        blocked = torch.finfo(q.dtype).min
        attn_mask = torch.zeros(mask.shape, dtype=q.dtype, device=q.device).masked_fill(mask, blocked)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    return out.permute(2, 0, 1, 3)


# zaya_fuse_moe (default on): compile the MoE-stage elementwise chains that
# zaya_fuse_norm/zaya_fuse_cca do not cover: the expert SwiGLU (silu * up, one
# kernel fwd and bwd, dynamic token count), the router EDA add + RMSNorm and
# the fp32 softmax/bias/top-k/gather/cast, and the gate multiply + unpermute
# (one gather*prob kernel replacing the per-expert multiplies). Off runs the
# same functions uncompiled.


def _swiglu(fc1_out: torch.Tensor) -> torch.Tensor:
    gate, up = fc1_out.chunk(2, dim=-1)
    return F.silu(gate) * up


def _eda_norm(routed, prev, scale, norm_w, eps: float):
    """Router EDA (``routed + prev * scale``) followed by the router RMSNorm."""
    routed = routed + prev * scale
    return routed, _rms(routed, norm_w, eps)


def _route_select(logits, balancing_biases, topk: int, high_prec: bool, out_dtype: torch.dtype):
    """Softmax, biased top-k (no grad through the bias), gather, cast."""
    if high_prec:
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    else:
        probs = torch.softmax(logits, dim=-1)
    biased = probs.detach().float() + balancing_biases.float()
    _, index = torch.topk(biased, k=topk, dim=-1)
    route_prob = torch.gather(probs, dim=-1, index=index)
    return route_prob.to(dtype=out_dtype), index


def _unpermute_scale_fwd(out_sorted, inv_perm, prob):
    return out_sorted.index_select(0, inv_perm) * prob


def _unpermute_scale_bwd(grad, out_sorted, perm, inv_perm, prob):
    d_sorted = (grad * prob).index_select(0, perm)
    d_prob = (grad * out_sorted.index_select(0, inv_perm)).sum(dim=-1, keepdim=True)
    return d_sorted, d_prob


def _unpermute_scale_bwd_noprob(grad, perm, prob):
    return (grad * prob).index_select(0, perm)


# Expert token counts change every layer and step: compile SwiGLU with dynamic
# shapes so it is one graph instead of a recompile per count.
_fused_swiglu = maybe_compile(_swiglu, dynamic=True)
_fused_eda_norm = maybe_compile(_eda_norm)
_fused_route_select = maybe_compile(_route_select)
_fused_unpermute_fwd = maybe_compile(_unpermute_scale_fwd)
_fused_unpermute_bwd = maybe_compile(_unpermute_scale_bwd)
_fused_unpermute_bwd_noprob = maybe_compile(_unpermute_scale_bwd_noprob)


class _PermuteRows(torch.autograd.Function):
    """``x.index_select(0, perm)`` for a permutation; backward gathers by ``inv_perm``.

    Same values as the eager index_select backward (zero fill + index_add with
    unique indices), but one gather instead of a fill and an atomic scatter.
    """

    @staticmethod
    def forward(ctx, x, perm, inv_perm):
        ctx.save_for_backward(inv_perm)
        return x.index_select(0, perm)

    @staticmethod
    def backward(ctx, grad):
        (inv_perm,) = ctx.saved_tensors
        return grad.index_select(0, inv_perm), None, None


class _UnpermuteScale(torch.autograd.Function):
    """``out[t] = out_sorted[inv_perm[t]] * prob[t]`` as one gather*scale, compiled when ``compiled``."""

    @staticmethod
    def forward(ctx, out_sorted, perm, inv_perm, prob, compiled: bool):
        ctx.save_for_backward(out_sorted, perm, inv_perm, prob)
        ctx.compiled = compiled
        unpermute = _fused_unpermute_fwd if compiled else _unpermute_scale_fwd
        return unpermute(out_sorted, inv_perm, prob)

    @staticmethod
    def backward(ctx, grad):
        out_sorted, perm, inv_perm, prob = ctx.saved_tensors
        grad = grad.contiguous()
        if ctx.needs_input_grad[3]:
            unpermute_bwd = _fused_unpermute_bwd if ctx.compiled else _unpermute_scale_bwd
            d_sorted, d_prob = unpermute_bwd(grad, out_sorted, perm, inv_perm, prob)
        else:
            unpermute_bwd = _fused_unpermute_bwd_noprob if ctx.compiled else _unpermute_scale_bwd_noprob
            d_sorted, d_prob = unpermute_bwd(grad, perm, prob), None
        return d_sorted, None, None, d_prob, None


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, config) -> None:
        super().__init__()
        self.linear_fc1 = _linear(hidden_size, 2 * intermediate_size, False, config)
        self.linear_fc2 = _linear(intermediate_size, hidden_size, False, config)
        self.fuse_moe = _path_on(config, "zaya_fuse_moe")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        swiglu = _fused_swiglu if self.fuse_moe else _swiglu
        return self.linear_fc2(swiglu(self.linear_fc1(hidden)))


class ZayaRouter(nn.Module):
    """3-layer GELU router with vector EDA and a detached balancing bias."""

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.width = int(getattr(config, "zaya_mlp_expansion", 256))
        self.num_real_experts = int(config.num_moe_experts)
        self.use_mod = bool(getattr(config, "zaya_use_mod", True))
        self.use_eda = bool(getattr(config, "zaya_use_eda", True))
        self.high_prec = bool(getattr(config, "zaya_high_prec", True))
        self.topk = int(getattr(config, "moe_router_topk", 1))
        self.num_scored = self.num_real_experts + (1 if self.use_mod else 0)
        eps = float(getattr(config, "layernorm_epsilon", 1e-5))
        self.balance_lr = float(getattr(config, "zaya_balance_lr", 1e-3))
        self.balance_beta1 = float(getattr(config, "zaya_balance_beta1", 0.9))
        self.balance_beta2 = float(getattr(config, "zaya_balance_beta2", 0.999))
        self.balance_wd = float(getattr(config, "zaya_balance_wd", 0.0))

        self.down_proj = _linear(self.hidden_size, self.width, True, config, fp8=False)
        self.rmsnorm_eda = RMSNorm(self.width, eps)
        if self.use_eda:
            self.router_states_scale = nn.Parameter(torch.ones(self.width))
        self.router_mlp = nn.Sequential(
            _linear(self.width, self.width, True, config, fp8=False),
            nn.GELU(),
            _linear(self.width, self.width, True, config, fp8=False),
            nn.GELU(),
            _linear(self.width, self.num_scored, False, config, fp8=False),
        )
        self.register_buffer("balancing_biases", torch.zeros(self.num_scored), persistent=True)
        self.register_buffer("balance_m", torch.zeros(self.num_scored), persistent=True)
        self.register_buffer("balance_v", torch.zeros(self.num_scored), persistent=True)
        self.register_buffer("balance_step", torch.zeros((), dtype=torch.long), persistent=True)
        if self.use_mod:
            with torch.no_grad():
                self.balancing_biases[-1] = -1.0
        self.fuse_moe = _path_on(config, "zaya_fuse_moe")
        # Deferred-update state (zaya_router_defer_ar). Not a buffer: it is
        # always flushed before the next forward reads balancing_biases.
        self.defer_bias_update = _path_on(config, "zaya_router_defer_ar")
        self._pending_counts: Optional[torch.Tensor] = None

    def _apply(self, fn, recurse=True):
        # Float16Module casts the whole module to bf16. The PID moments are
        # buffers, not parameters, so the main optimizer never sees them; keep
        # them in fp32 or the Adam step collapses.
        super()._apply(fn, recurse=recurse)
        for name in ("balancing_biases", "balance_m", "balance_v"):
            buf = getattr(self, name)
            if buf.is_floating_point():
                setattr(self, name, buf.float())
        return self

    def forward(self, hidden_states: torch.Tensor, prev_router: Optional[torch.Tensor]):
        seq, batch, _ = hidden_states.shape
        routed = self.down_proj(hidden_states.reshape(seq * batch, self.hidden_size))
        # EDA state is the post-add, pre-norm router hidden. RMSNorm and the
        # 3-layer MLP both see that state.
        if self.use_eda and prev_router is not None:
            eda_norm = _fused_eda_norm if self.fuse_moe else _eda_norm
            routed, logits = eda_norm(
                routed,
                prev_router.reshape(seq * batch, self.width),
                self.router_states_scale,
                self.rmsnorm_eda.weight,
                self.rmsnorm_eda.eps,
            )
        else:
            logits = self.rmsnorm_eda(routed)
        next_router = routed.view(seq, batch, self.width)
        for stage in self.router_mlp:
            logits = stage(logits)
        # The bias is a buffer updated in place below; the top-k keeps it out of the graph.
        route_select = _fused_route_select if self.fuse_moe else _route_select
        route_prob, index = route_select(
            logits, self.balancing_biases, self.topk, self.high_prec, hidden_states.dtype
        )
        if self.training and self.balance_lr != 0.0 and self.topk == 1:
            self._record_bias_update(index.reshape(-1))
        return route_prob.view(seq, batch, self.topk), index.view(seq, batch, self.topk), next_router

    def _record_bias_update(self, index: torch.Tensor) -> None:
        if self.defer_bias_update:
            counts = self._local_counts(index)
            self._pending_counts = counts if self._pending_counts is None else self._pending_counts + counts
        else:
            self._update_biases(index)

    def _local_counts(self, index: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            counts = torch.zeros(self.num_scored, device=index.device, dtype=torch.float32)
            counts.scatter_add_(0, index, torch.ones_like(index, dtype=torch.float32))
        return counts

    def _update_biases(self, index: torch.Tensor) -> None:
        counts = self._local_counts(index)
        _allreduce_dp(counts)
        self._apply_bias_update(counts)

    def _apply_bias_update(self, counts: torch.Tensor) -> None:
        """PID/Adam step on the balancing bias from DP-summed token counts."""
        with torch.no_grad():
            total = counts.sum().clamp_min(1.0)
            grad = counts / total - (1.0 / self.num_scored)
            self.balance_step += 1
            # Bias corrections stay on device (fp64 pow, then cast to fp32) so
            # this per-MoE-layer update does not force a host sync via .item().
            step = self.balance_step.double()
            b1, b2 = self.balance_beta1, self.balance_beta2
            self.balance_m.mul_(b1).add_(grad, alpha=1.0 - b1)
            self.balance_v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
            mhat = self.balance_m / (1.0 - torch.pow(b1, step)).float()
            vhat = self.balance_v / (1.0 - torch.pow(b2, step)).float()
            if self.balance_wd != 0.0:
                self.balancing_biases.mul_(1.0 - self.balance_lr * self.balance_wd)
            self.balancing_biases.addcdiv_(mhat, vhat.sqrt().add_(1e-8), value=-self.balance_lr)


def flush_router_bias_updates(routers: Sequence["ZayaRouter"]) -> None:
    """Apply every pending deferred bias update with a single DP all-reduce.

    Counts are small integers held in fp32, so summing them in one
    concatenated all-reduce is exact and matches the per-layer reduces.
    """
    pending = [r for r in routers if r._pending_counts is not None]
    if not pending:
        return
    with torch.no_grad():
        flat = torch.cat([r._pending_counts for r in pending])
        _allreduce_dp(flat)
        for r, counts in zip(pending, flat.split([r.num_scored for r in pending])):
            r._pending_counts = None
            r._apply_bias_update(counts)


def _allreduce_dp(tensor: torch.Tensor) -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    group = None
    try:
        from megatron.core import parallel_state

        group = parallel_state.get_data_parallel_group(check_initialized=False)
    except Exception:
        group = None
    torch.distributed.all_reduce(tensor, group=group)


class _ZeroGradFor(torch.autograd.Function):
    """Identity on ``x``; backward passes ``x``'s grad through and gives ``param`` zeros."""

    @staticmethod
    def forward(ctx, x, param):
        ctx.param_meta = (param.shape, param.dtype, param.device)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        shape, dtype, device = ctx.param_meta
        return grad, torch.zeros(shape, dtype=dtype, device=device)


def _turbo_grouped_gemm(a, b, group_lens, group_offs):
    """``a[rows of group g] @ b[g].T`` with device-resident group sizes.

    Rows past ``group_offs[-1]`` (the MoD skip slot) come back as zeros, and
    so does their input gradient. Tests swap this for a CPU reference.
    """
    from primus_turbo.pytorch.ops import grouped_gemm

    return grouped_gemm(a, b, group_lens, group_offs, trans_b=True)


def _expert_key(prefix: str, expert_id: int, fc: str) -> str:
    return f"{prefix}local_experts.{expert_id}.linear_{fc}.weight"


class ZayaMoE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_real_experts = int(config.num_moe_experts)
        intermediate = int(getattr(config, "moe_ffn_hidden_size", None) or config.ffn_hidden_size)
        self.fuse_moe = _path_on(config, "zaya_fuse_moe")
        self.router = ZayaRouter(config)
        experts = [
            SwiGLUExpert(int(config.hidden_size), intermediate, config) for _ in range(self.num_real_experts)
        ]
        self.use_turbo_gemm, _ = _turbo_experts(config)
        if self.use_turbo_gemm:
            _publish_turbo_grouped_gemm_backend()
            # Built from the same per-expert modules (same init RNG stream),
            # then stacked into [E, out, in] (nn.Linear layout per expert, so
            # the grouped GEMM runs with trans_b=True). The per-expert copies
            # are dropped; state_dict hooks keep the per-expert keys.
            with torch.no_grad():
                self.fc1_weight = nn.Parameter(torch.stack([e.linear_fc1.weight for e in experts]))
                self.fc2_weight = nn.Parameter(torch.stack([e.linear_fc2.weight for e in experts]))
            del experts
            self.local_experts = None
            self._register_state_dict_hook(ZayaMoE._unstack_state_dict)
            self._register_load_state_dict_pre_hook(self._stack_state_dict)
        else:
            self.local_experts = nn.ModuleList(experts)

    @staticmethod
    def _unstack_state_dict(module, state_dict, prefix, local_metadata):
        # Views share the stacked storage, so torch.save writes it once.
        for fc in ("fc1", "fc2"):
            stacked = state_dict.pop(f"{prefix}{fc}_weight").detach()
            for expert_id in range(module.num_real_experts):
                state_dict[_expert_key(prefix, expert_id, fc)] = stacked[expert_id]
        return state_dict

    def _stack_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for fc in ("fc1", "fc2"):
            keys = [_expert_key(prefix, i, fc) for i in range(self.num_real_experts)]
            if all(k in state_dict for k in keys):
                state_dict[f"{prefix}{fc}_weight"] = torch.stack([state_dict.pop(k) for k in keys])
            elif any(k in state_dict for k in keys):
                # Load the experts that are there, as separate per-expert
                # weights would, and report the others by their own keys.
                stacked = getattr(self, f"{fc}_weight").detach().clone()
                for expert_id, key in enumerate(keys):
                    if key in state_dict:
                        stacked[expert_id].copy_(state_dict.pop(key))
                    else:
                        missing_keys.append(key)
                state_dict[f"{prefix}{fc}_weight"] = stacked

    def _forward_turbo(self, flat, flat_idx, flat_prob):
        """Grouped-GEMM experts with no host sync.

        Tokens are stably sorted by routed group; the MoD skip slot has the
        largest index, so its rows form the tail past the last real expert's
        offset. The grouped GEMMs leave that tail zero, and the skip rows add
        their identity input there instead, which reproduces the split path's
        per-group results (``expert(x) * p`` or ``x * p``) row for row.
        """
        num_groups = self.router.num_scored
        E = self.num_real_experts
        sorted_idx, perm = torch.sort(flat_idx, stable=True)
        # scatter_add instead of bincount: CUDA bincount reads max() on host.
        counts = torch.zeros(num_groups, dtype=torch.int64, device=flat.device)
        counts.scatter_add_(0, sorted_idx, torch.ones_like(sorted_idx, dtype=torch.int64))
        group_lens = counts[:E].contiguous()
        group_offs = F.pad(torch.cumsum(group_lens, dim=0), (1, 0))
        # The split path's permute, SwiGLU and gather*prob unpermute, with the
        # per-expert loop replaced by two grouped GEMMs.
        inv_perm = torch.empty_like(perm)
        inv_perm.scatter_(0, perm, torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype))
        x_sorted = _PermuteRows.apply(flat, perm, inv_perm)
        swiglu = _fused_swiglu if self.fuse_moe else _swiglu
        gate_up = _turbo_grouped_gemm(x_sorted, self.fc1_weight, group_lens, group_offs)
        y = _turbo_grouped_gemm(swiglu(gate_up), self.fc2_weight, group_lens, group_offs)
        if self.router.use_mod:
            # Skip rows take the identity input; the GEMM left them zero.
            y = torch.where((sorted_idx == E).unsqueeze(-1), x_sorted, y)
        return _UnpermuteScale.apply(y, perm, inv_perm, flat_prob, self.fuse_moe)

    def forward(self, hidden_states: torch.Tensor, prev_router: Optional[torch.Tensor]):
        if (
            self.use_turbo_gemm
            and self.router.use_eda
            and prev_router is None
            and torch.is_grad_enabled()
            and self.router.router_states_scale.requires_grad
        ):
            # First MoE layer: no previous router state, so the EDA scale is
            # unused. Give it an exact-zero grad (identity forward) so every
            # parameter gets a grad each step, which Megatron DDP's
            # --overlap_grad_reduce per-bucket hook count requires. Together
            # with the stacked experts (every expert weight is in each grouped
            # GEMM, empty groups get a zero wgrad) the Turbo path leaves no
            # parameter without a grad.
            hidden_states = _ZeroGradFor.apply(hidden_states, self.router.router_states_scale)
        probs, index, next_router = self.router(hidden_states, prev_router)
        # Top-1 is the released configuration. The mix weight is the unbiased
        # gathered probability.
        prob = probs[..., 0]
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_idx = index[..., 0].reshape(-1)
        flat_prob = prob.reshape(-1, 1).to(dtype=flat.dtype)
        if self.use_turbo_gemm:
            return self._forward_turbo(flat, flat_idx, flat_prob).view_as(hidden_states), next_router
        # Sort-based dispatch. A stable sort keeps the original token order
        # inside each expert group, so every expert sees exactly the rows (in
        # the same order) that the old boolean-mask loop gave it. The forward
        # syncs three times per layer (bincount reads its input's min and max,
        # then counts.tolist()) instead of one torch.any plus two nonzero syncs
        # per expert in forward and more in backward. Only the first waits for
        # queued work, so scatter_add_ counts as in _forward_turbo save little
        # here. The permute and unpermute backwards are gathers by the inverse
        # permutation, with no host sync.
        num_groups = self.router.num_scored
        sorted_idx, perm = torch.sort(flat_idx, stable=True)
        counts = torch.bincount(sorted_idx, minlength=num_groups).tolist()
        inv_perm = torch.empty_like(perm)
        inv_perm.scatter_(0, perm, torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype))
        x_chunks = torch.split(_PermuteRows.apply(flat, perm, inv_perm), counts, dim=0)
        pieces = []
        for expert_id, expert in enumerate(self.local_experts):
            # Experts with no tokens are skipped: their params get no autograd
            # grad and their main_grad stays zero.
            if counts[expert_id] == 0:
                continue
            pieces.append(expert(x_chunks[expert_id]))
        if self.router.use_mod and counts[self.num_real_experts] > 0:
            # Mixture-of-depths skip slot: the identity.
            pieces.append(x_chunks[self.num_real_experts])
        # Every token lands in exactly one group (real expert or MoD skip), so
        # the concatenation covers all rows in sorted order. The gate multiply
        # comes after the unpermute, as one gather*prob over all tokens.
        out_sorted = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        out = _UnpermuteScale.apply(out_sorted, perm, inv_perm, flat_prob, self.fuse_moe)
        return out.view_as(hidden_states), next_router


class ZayaStage(nn.Module):
    """One CCA or MoE stage: scale, add, RMSNorm, mixer."""

    def __init__(self, config, kind: str, layer_number: int) -> None:
        super().__init__()
        self.kind = kind
        self.layer_number = layer_number
        eps = float(getattr(config, "layernorm_epsilon", 1e-5))
        self.input_norm = RMSNorm(int(config.hidden_size), eps)
        use_scale = bool(getattr(config, "scale_residual_merge", True))
        self.res_scale = ResidualScaling(int(config.hidden_size), layer_number != 0) if use_scale else None
        self.fuse_norm = _path_on(config, "zaya_fuse_norm")
        if kind == "a":
            self.self_attn = CCA(config)
            self.zaya_block = None
        else:
            self.self_attn = None
            self.zaya_block = ZayaMoE(config)

    def forward(self, hidden_states, residual, prev_router, position_ids, attention_mask):
        residual, normed = _scale_merge_norm(
            self.res_scale, self.input_norm, hidden_states, residual, self.fuse_norm
        )
        if self.kind == "a":
            hidden_states = self.self_attn(normed, position_ids, attention_mask)
        else:
            hidden_states, prev_router = self.zaya_block(normed, prev_router)
        return hidden_states, residual, prev_router


class ZayaStack(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        kinds = layer_kinds(int(config.num_layers), getattr(config, "zaya_layers", None))
        self.layers = nn.ModuleList(ZayaStage(config, kind, i) for i, kind in enumerate(kinds))
        eps = float(getattr(config, "layernorm_epsilon", 1e-5))
        self.final_norm = RMSNorm(int(config.hidden_size), eps)
        use_scale = bool(getattr(config, "scale_residual_merge", True))
        self.res_scale = ResidualScaling(int(config.hidden_size), True) if use_scale else None
        self.fuse_norm = _path_on(config, "zaya_fuse_norm")
        self.attn_flash = _path_on(config, "zaya_attn_flash")

    def _fp8_context(self):
        """One FP8 autocast for the whole stack.

        The recipe comes from ``config.fp8_recipe``. Delayed scaling must not
        enter and exit the context per stage, or the amax history is reduced
        incorrectly, so the context wraps every stage. This port does not keep
        the first or last stage in BF16. Import at call time so Primus's FP8
        patch, which replaces ``megatron.core.fp8_utils.get_fp8_context``
        before training, is the function that runs.
        """
        from contextlib import nullcontext

        if not _fp8_requested(self.config):
            return nullcontext()
        from megatron.core.fp8_utils import get_fp8_context

        return get_fp8_context(self.config)

    def _routers(self) -> List[ZayaRouter]:
        return [layer.zaya_block.router for layer in self.layers if layer.zaya_block is not None]

    def flush_router_bias_updates(self) -> None:
        flush_router_bias_updates(self._routers())

    def forward(self, hidden_states, position_ids, attention_mask):
        # Deferred router updates from a forward whose backward never ran (or
        # had no grad path to the stack input) land before any topk reads the
        # bias. A no-op when nothing is pending; no host sync either way.
        self.flush_router_bias_updates()
        if (
            self.training
            and torch.is_grad_enabled()
            and hidden_states.requires_grad
            and any(r.defer_bias_update for r in self._routers())
        ):
            # Fires once the stack input's gradient is known, i.e. after every
            # stage's backward: one all-reduce for all routers, applied before
            # the optimizer step and any checkpoint save.
            def _flush_after_backward(grad, stack=self):
                stack.flush_router_bias_updates()
                return None

            hidden_states.register_hook(_flush_after_backward)
        if self.attn_flash and attention_mask is not None and is_causal_mask(attention_mask):
            # The dataloader's tril mask (reset_attention_mask=false) is plain
            # causal: check it once here instead of materializing an additive
            # [b, h, s, s] mask in each of the 40 CCA stages.
            attention_mask = None
        with self._fp8_context():
            residual = None
            prev_router = None
            for layer in self.layers:
                hidden_states, residual, prev_router = layer(
                    hidden_states, residual, prev_router, position_ids, attention_mask
                )
            _, normed = _scale_merge_norm(
                self.res_scale, self.final_norm, hidden_states, residual, self.fuse_norm
            )
            return normed
