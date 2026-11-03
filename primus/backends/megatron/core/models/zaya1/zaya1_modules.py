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

On GPU, the elementwise chains between GEMMs (residual scaling, merge and
RMSNorm at each stage boundary, and CCA's grouped means, QK norm and partial
RoPE) run through Megatron's ``jit_fuser``
(``torch.compile``), which fuses them into a few kernels and keeps only their
inputs for the backward. Fused kernels hold intermediates in fp32, so they skip
the bf16 rounding that eager mode applies between those steps. CPU runs, like
the unit tests, stay eager.
"""

from __future__ import annotations

import functools
import math
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _compiled_on_gpu(fn):
    """Run ``fn`` through Megatron's ``jit_fuser`` for GPU inputs and eagerly otherwise."""
    compiled = None

    @functools.wraps(fn)
    def run(*args):
        nonlocal compiled
        if not args[0].is_cuda:
            return fn(*args)
        if compiled is None:
            from megatron.core.jit import jit_fuser

            compiled = jit_fuser(fn)
        return compiled(*args)

    return run


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
    """Pad the token axis so the leading-dimension product is a nonzero multiple of 8."""
    leading = 1
    for size in hidden.shape[:-1]:
        leading *= int(size)
    pad = (8 - leading % 8) % 8 if leading else 8
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


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv * weight.float()).to(dtype=x.dtype)


class RMSNorm(nn.Module):
    """RMSNorm with a learnable scale. Statistics are computed in fp32."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.weight, self.eps)


class ResidualScaling(nn.Module):
    """Affine ``(x + bias) * scale`` on the mixer output and the residual.

    Stage 0 has no incoming residual, so it stores only the mixer-output affine.
    """

    def __init__(self, hidden_size: int, has_residual: bool) -> None:
        super().__init__()
        self.has_residual = has_residual
        self.hidden_states_scale = nn.Parameter(torch.ones(hidden_size))
        self.hidden_states_bias = nn.Parameter(torch.zeros(hidden_size))
        if has_residual:
            self.residual_scale = nn.Parameter(torch.ones(hidden_size))
            self.residual_bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, residual: Optional[torch.Tensor], hidden_states: torch.Tensor):
        hidden_states = hidden_states.float() + self.hidden_states_bias.float()
        hidden_states = hidden_states * self.hidden_states_scale.float()
        if self.has_residual and residual is not None:
            residual = residual.float() + self.residual_bias.float()
            residual = residual * self.residual_scale.float()
        return residual, hidden_states


def _scale_merge_norm(hidden, residual, h_scale, h_bias, r_scale, r_bias, norm_weight, eps: float):
    """``ResidualScaling``, the fp32 residual add and the following ``RMSNorm`` in one function.

    Returns the fp32 residual and the norm of it in ``hidden``'s dtype. An
    affine whose parameters are ``None`` is skipped, as is a ``None`` residual.
    """
    merged = hidden.float()
    if h_scale is not None:
        merged = (merged + h_bias.float()) * h_scale.float()
    if residual is not None:
        res = residual.float()
        if r_scale is not None:
            res = (res + r_bias.float()) * r_scale.float()
        merged = res + merged
    return merged, _rms_norm(merged.to(dtype=hidden.dtype), norm_weight, eps)


_fused_scale_merge_norm = _compiled_on_gpu(_scale_merge_norm)


def _residual_affine(res_scale: Optional[ResidualScaling]):
    if res_scale is None:
        return None, None, None, None
    if not res_scale.has_residual:
        return res_scale.hidden_states_scale, res_scale.hidden_states_bias, None, None
    return (
        res_scale.hidden_states_scale,
        res_scale.hidden_states_bias,
        res_scale.residual_scale,
        res_scale.residual_bias,
    )


def _grouped_means(query_conv, key_conv, query_pre, key_base):
    seq, batch, num_q_heads, head_dim = query_pre.shape
    num_k_heads = key_base.shape[2]
    key_base_f = key_base.float()
    query_pre_g = query_pre.reshape(seq, batch, num_k_heads, num_q_heads // num_k_heads, head_dim).float()
    query_conv_g = query_conv.reshape(query_pre_g.shape).float()
    query = query_conv_g + 0.5 * query_pre_g + 0.5 * key_base_f.unsqueeze(-2)
    query = query.reshape(seq, batch, num_q_heads, head_dim)
    query_mean = query_pre_g.mean(dim=-2)
    key = key_conv.float() + 0.5 * query_mean + 0.5 * key_base_f
    return query, key


def _qk_norm_and_temp(query, key, temp, clamp_temp: bool):
    eps = 1e-12
    scale = math.sqrt(query.shape[-1])
    query = query.float() * (torch.rsqrt(query.float().pow(2).sum(-1, keepdim=True) + eps) * scale)
    key = key.float() * (torch.rsqrt(key.float().pow(2).sum(-1, keepdim=True) + eps) * scale)
    temp = temp.float().view(1, 1, -1, 1)
    if clamp_temp:
        temp = torch.exp(torch.clamp(temp, 1e-7, 2.0))
    return query, key * temp


def _rope_cos_sin(position_ids: Optional[torch.Tensor], seq: int, rotary_dim: int, base: float, device):
    if position_ids is None:
        pos = torch.arange(seq, device=device)
    else:
        # position_ids is [batch, seq]. Transpose to [seq, batch] so each
        # row can carry its own positions.
        pos = position_ids.transpose(0, 1).to(device=device)
    freq_idx = torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (freq_idx / rotary_dim))
    # NeoX layout: the same angle is shared by the two halves of the rotary slice.
    if pos.dim() == 1:
        freqs = torch.outer(pos.float(), inv_freq)
        cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)[:, None, None, :]
        sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)[:, None, None, :]
    else:
        freqs = torch.einsum("sb,d->sbd", pos.float(), inv_freq)
        cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)[:, :, None, :]
        sin = torch.sin(freqs)
        sin = torch.cat((sin, sin), dim=-1)[:, :, None, :]
    return cos, sin


def _apply_rope(x, cos, sin):
    rotary_dim = cos.shape[-1]
    rot, rest = x[..., :rotary_dim], x[..., rotary_dim:]
    rot_f = rot.float()
    half = rotary_dim // 2
    x1, x2 = rot_f[..., :half], rot_f[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    out = rot_f * cos + rotated * sin
    return torch.cat((out, rest.float()), dim=-1).to(dtype=x.dtype)


def _cca_query_key(query_conv, key_conv, query_pre, key_base, temp, clamp_temp: bool, rope):
    """Grouped means, QK norm with the key temperature, then partial RoPE, in the projections' dtype."""
    query, key = _grouped_means(query_conv, key_conv, query_pre, key_base)
    query, key = _qk_norm_and_temp(query, key, temp, clamp_temp)
    query, key = query.to(dtype=query_pre.dtype), key.to(dtype=query_pre.dtype)
    if rope is None:
        return query, key
    return _apply_rope(query, *rope), _apply_rope(key, *rope)


_fused_cca_query_key = _compiled_on_gpu(_cca_query_key)


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
        # Ones, not zeros: a zero default is the pre-checkpoint initialization, and a
        # raw temperature of zero would wipe the keys at step 0.
        self.temp = nn.Parameter(torch.ones(self.num_k_heads))
        self.o_proj = _linear(self.latent_q, self.hidden_size, bias, config)

    def _qk_norm_and_temp(self, query: torch.Tensor, key: torch.Tensor):
        return _qk_norm_and_temp(query, key, self.temp, self.clamp_temp)

    def _grouped_means(self, query_conv, key_conv, query_pre, key_base):
        return _grouped_means(query_conv, key_conv, query_pre, key_base)

    def rope_cos_sin(self, position_ids: Optional[torch.Tensor], seq: int, device):
        """Partial-RoPE ``(cos, sin)`` for ``[seq, batch, heads, rotary_dim]``, or ``None`` without RoPE."""
        if self.rotary_dim == 0:
            return None
        return _rope_cos_sin(position_ids, seq, self.rotary_dim, self.rope_base, device)

    def project(self, hidden_states: torch.Tensor, position_ids: Optional[torch.Tensor], rope=None):
        """Query, key and value; ``rope`` is ``rope_cos_sin(position_ids, ...)`` when the caller has it."""
        seq, batch, _ = hidden_states.shape
        q_raw = self.linear_q(hidden_states)
        k_raw = self.linear_k(hidden_states)
        qk = torch.cat((q_raw, k_raw), dim=-1)
        qk_ncl = F.pad(qk.permute(1, 2, 0), (self.total_padding, 0))
        qk_out = self.conv_qk(qk_ncl).permute(2, 0, 1)
        query_pre = q_raw.view(seq, batch, self.num_q_heads, self.head_dim)
        key_base = k_raw.view(seq, batch, self.num_k_heads, self.head_dim)
        query_conv = qk_out[..., : self.latent_q].view(seq, batch, self.num_q_heads, self.head_dim)
        key_conv = qk_out[..., self.latent_q :].view(seq, batch, self.num_k_heads, self.head_dim)
        if rope is None:
            rope = self.rope_cos_sin(position_ids, seq, hidden_states.device)
        query, key = _fused_cca_query_key(
            query_conv, key_conv, query_pre, key_base, self.temp, self.clamp_temp, rope
        )

        shifted = F.pad(hidden_states[:-1], (0, 0, 0, 0, 1, 0))
        v1 = self.val_proj1(hidden_states)
        v2 = self.val_proj2(shifted)
        value = torch.cat((v1, v2), dim=-1).view(seq, batch, self.num_k_heads, self.head_dim)
        return query, key, value.to(hidden_states.dtype)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: Optional[torch.Tensor], attention_mask, rope=None
    ):
        query, key, value = self.project(hidden_states, position_ids, rope)
        attn = attention(query, key, value, attention_mask)
        seq, batch = hidden_states.shape[:2]
        return self.o_proj(attn.reshape(seq, batch, self.latent_q))


def attention(query, key, value, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Causal GQA. ``attention_mask`` uses Megatron's convention: True blocks a key."""
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


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, config) -> None:
        super().__init__()
        self.linear_fc1 = _linear(hidden_size, 2 * intermediate_size, False, config)
        self.linear_fc2 = _linear(intermediate_size, hidden_size, False, config)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, up = self.linear_fc1(hidden).chunk(2, dim=-1)
        return self.linear_fc2(F.silu(gate) * up)


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
        if self.use_eda and prev_router is not None:
            routed = routed + prev_router.reshape(seq * batch, self.width) * self.router_states_scale
        # EDA state is the post-add, pre-norm router hidden. RMSNorm and the
        # 3-layer MLP both see that state.
        next_router = routed.view(seq, batch, self.width)
        logits = self.rmsnorm_eda(routed)
        for stage in self.router_mlp:
            logits = stage(logits)
        if self.high_prec:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        else:
            probs = torch.softmax(logits, dim=-1)
        # The bias is a buffer updated in place below. Keep it out of the graph.
        with torch.no_grad():
            biased = probs.detach().float() + self.balancing_biases.float()
            _, index = torch.topk(biased, k=self.topk, dim=-1)
        route_prob = torch.gather(probs, dim=-1, index=index)
        if self.training and self.balance_lr != 0.0 and self.topk == 1:
            self._update_biases(index.reshape(-1))
        route_prob = route_prob.to(dtype=hidden_states.dtype).view(seq, batch, self.topk)
        index = index.view(seq, batch, self.topk)
        return route_prob, index, next_router

    def _update_biases(self, index: torch.Tensor) -> None:
        with torch.no_grad():
            counts = torch.zeros(self.num_scored, device=index.device, dtype=torch.float32)
            counts.scatter_add_(0, index, torch.ones_like(index, dtype=torch.float32))
            _allreduce_dp(counts)
            total = counts.sum().clamp_min(1.0)
            grad = counts / total - (1.0 / self.num_scored)
            self.balance_step += 1
            step = int(self.balance_step.item())
            b1, b2 = self.balance_beta1, self.balance_beta2
            self.balance_m.mul_(b1).add_(grad, alpha=1.0 - b1)
            self.balance_v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
            mhat = self.balance_m / (1.0 - b1**step)
            vhat = self.balance_v / (1.0 - b2**step)
            if self.balance_wd != 0.0:
                self.balancing_biases.mul_(1.0 - self.balance_lr * self.balance_wd)
            self.balancing_biases.addcdiv_(mhat, vhat.sqrt().add_(1e-8), value=-self.balance_lr)


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


class _PermuteRows(torch.autograd.Function):
    """``x.index_select(0, index)`` for a permutation ``index`` whose inverse is ``inverse``.

    The backward gathers with the inverse instead of scatter-adding into zeros,
    so neither direction needs atomics.
    """

    @staticmethod
    def forward(ctx, x, index, inverse):
        ctx.save_for_backward(inverse)
        return x.index_select(0, index)

    @staticmethod
    def backward(ctx, grad):
        (inverse,) = ctx.saved_tensors
        return grad.index_select(0, inverse), None, None


class ZayaMoE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_real_experts = int(config.num_moe_experts)
        intermediate = int(getattr(config, "moe_ffn_hidden_size", None) or config.ffn_hidden_size)
        self.router = ZayaRouter(config)
        self.local_experts = nn.ModuleList(
            [
                SwiGLUExpert(int(config.hidden_size), intermediate, config)
                for _ in range(self.num_real_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor, prev_router: Optional[torch.Tensor]):
        probs, index, next_router = self.router(hidden_states, prev_router)
        # Top-1 is the released configuration. The mix weight is the unbiased
        # gathered probability.
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_idx = index[..., 0].reshape(-1)
        flat_prob = probs[..., 0].reshape(-1, 1).to(dtype=flat.dtype)
        # Group the tokens by route, keeping their order within a route, so each
        # expert reads one contiguous slice. The slice sizes are the stage's only
        # device-to-host copy. The skip class sorts last and passes through.
        order = torch.argsort(flat_idx, stable=True)
        inverse = torch.argsort(order)
        sizes = torch.bincount(flat_idx, minlength=self.router.num_scored).tolist()
        routed = _PermuteRows.apply(flat, order, inverse).split(sizes)
        # Every expert runs, on an empty slice too, so every expert weight gets a
        # gradient each step; overlap_grad_reduce waits for all of them.
        outputs = [expert(tokens) for expert, tokens in zip(self.local_experts, routed)]
        outputs.extend(routed[self.num_real_experts :])
        out = _PermuteRows.apply(torch.cat(outputs), inverse, order) * flat_prob
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
        if kind == "a":
            self.self_attn = CCA(config)
            self.zaya_block = None
        else:
            self.self_attn = None
            self.zaya_block = ZayaMoE(config)

    def forward(self, hidden_states, residual, prev_router, position_ids, attention_mask, rope=None):
        residual, normed = _fused_scale_merge_norm(
            hidden_states,
            residual,
            *_residual_affine(self.res_scale),
            self.input_norm.weight,
            self.input_norm.eps,
        )
        if self.kind == "a":
            hidden_states = self.self_attn(normed, position_ids, attention_mask, rope)
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

    def forward(self, hidden_states, position_ids, attention_mask):
        # Every CCA stage rotates the same positions, so build the tables once.
        cca = next((layer.self_attn for layer in self.layers if layer.self_attn is not None), None)
        rope = None
        if cca is not None:
            rope = cca.rope_cos_sin(position_ids, hidden_states.shape[0], hidden_states.device)
        with self._fp8_context():
            residual = None
            prev_router = None
            for layer in self.layers:
                hidden_states, residual, prev_router = layer(
                    hidden_states, residual, prev_router, position_ids, attention_mask, rope
                )
            _, normed = _fused_scale_merge_norm(
                hidden_states,
                residual,
                *_residual_affine(self.res_scale),
                self.final_norm.weight,
                self.final_norm.eps,
            )
            return normed
