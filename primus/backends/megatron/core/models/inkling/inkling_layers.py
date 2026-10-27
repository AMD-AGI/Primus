###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling decoder layer modules (tensor-parallel aware, sequence-first [T, B, H]).

Tensor parallelism: q/k/v/r projections are column-parallel (split by head),
the attention output projection is row-parallel; the k/v short convolutions
are split with their channels. Everything that acts on the full hidden state
(norms, hidden-size short convolutions, router, routed and shared experts) is
replicated and computed redundantly on every TP rank, so its gradients agree
across ranks without extra reduction. The two small parameters shared by all
heads (q/k norm weights, relative-bias profile bank) pass through
``copy_to_tensor_model_parallel_region`` so their gradient is summed over the
heads of every rank.
"""

import math

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.parallel_state import get_tensor_model_parallel_world_size
from megatron.core.transformer.module import MegatronModule

from primus.backends.megatron.core.models.inkling.inkling_transformer_config import (
    InklingTransformerConfig,
)


def _device(config):
    return torch.device("cpu") if config.use_cpu_initialization else torch.cuda.current_device()


def _param(config, *shape, init=None):
    p = torch.nn.Parameter(torch.empty(*shape, dtype=config.params_dtype, device=_device(config)))
    if config.perform_initialization:
        with torch.no_grad():
            if init is None:
                p.fill_(1.0)
            elif init == "zeros":
                p.zero_()
            else:
                init(p)
    return p


def rms_norm(x, weight, eps):
    """HF InklingRMSNorm: fp32 normalisation, cast back, times the weight."""
    dtype = x.dtype
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(dtype)


class InklingRMSNorm(MegatronModule):
    def __init__(self, config, size):
        super().__init__(config=config)
        self.eps = config.layernorm_epsilon
        self.weight = _param(config, size)

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class InklingShortConv(MegatronModule):
    """y = x + depthwise_causal_conv(x) in fp32 over the sequence dim of [T, B, C]."""

    def __init__(self, config, channels, tensor_parallel_split=False):
        super().__init__(config=config)
        k = config.inkling_sconv_kernel_size
        self.kernel_size = k
        self.weight = _param(config, channels, 1, k, init=config.init_method)
        if tensor_parallel_split:
            tensor_parallel.set_tensor_model_parallel_attributes(self.weight, True, 0, 1)

    def forward(self, x):
        T = x.shape[0]
        dtype = x.dtype
        xf = x.float()
        y = F.conv1d(xf.permute(1, 2, 0), self.weight.float(), padding=self.kernel_size - 1,
                     groups=xf.shape[-1])[..., :T]
        return (y.permute(2, 0, 1) + xf).to(dtype)


class InklingAttention(MegatronModule):
    def __init__(self, config: InklingTransformerConfig, layer_number: int):
        super().__init__(config=config)
        tp = get_tensor_model_parallel_world_size()
        self.is_local = layer_number % config.inkling_local_layer_period == 0
        self.head_dim = config.kv_channels
        self.num_heads = config.num_attention_heads
        self.num_kv = config.inkling_swa_num_query_groups if self.is_local else config.num_query_groups
        assert self.num_heads % tp == 0 and self.num_kv % tp == 0, (
            f"Inkling attention: heads {self.num_heads} / kv heads {self.num_kv} must divide by TP {tp}")
        self.heads_local = self.num_heads // tp
        self.kv_local = self.num_kv // tp
        self.rel_dim = config.inkling_rel_dim
        self.window = config.inkling_sliding_window if self.is_local else None
        self.rel_extent = config.inkling_sliding_window if self.is_local else config.inkling_rel_extent
        self.scaling = 1.0 / self.head_dim
        self.eps = config.layernorm_epsilon
        H = config.hidden_size

        def col(out):
            return tensor_parallel.ColumnParallelLinear(H, out, config=config, init_method=config.init_method,
                                                        bias=False, gather_output=False)

        self.linear_q = col(self.num_heads * self.head_dim)
        self.linear_k = col(self.num_kv * self.head_dim)
        self.linear_v = col(self.num_kv * self.head_dim)
        self.linear_r = col(self.num_heads * self.rel_dim)
        self.linear_proj = tensor_parallel.RowParallelLinear(
            self.num_heads * self.head_dim, H, config=config, init_method=config.output_layer_init_method,
            bias=False, input_is_parallel=True, skip_bias_add=True)
        self.k_sconv = InklingShortConv(config, self.kv_local * self.head_dim, tensor_parallel_split=True)
        self.v_sconv = InklingShortConv(config, self.kv_local * self.head_dim, tensor_parallel_split=True)
        self.q_norm_weight = _param(config, self.head_dim)
        self.k_norm_weight = _param(config, self.head_dim)
        self.rel_proj = _param(config, self.rel_dim, self.rel_extent, init=config.init_method)

    def forward(self, hidden_states):
        T, B, _ = hidden_states.shape
        q, _ = self.linear_q(hidden_states)
        k, _ = self.linear_k(hidden_states)
        v, _ = self.linear_v(hidden_states)
        r, _ = self.linear_r(hidden_states)
        k = self.k_sconv(k)
        v = self.v_sconv(v)

        q_w = tensor_parallel.copy_to_tensor_model_parallel_region(self.q_norm_weight)
        k_w = tensor_parallel.copy_to_tensor_model_parallel_region(self.k_norm_weight)
        rel = tensor_parallel.copy_to_tensor_model_parallel_region(self.rel_proj)

        # [T, B, n, d] -> [B, n, T, d]
        q = rms_norm(q.view(T, B, self.heads_local, self.head_dim), q_w, self.eps).permute(1, 2, 0, 3)
        k = rms_norm(k.view(T, B, self.kv_local, self.head_dim), k_w, self.eps).permute(1, 2, 0, 3)
        v = v.view(T, B, self.kv_local, self.head_dim).permute(1, 2, 0, 3)
        r = r.view(T, B, self.heads_local, self.rel_dim).permute(1, 2, 0, 3)  # [B, n, T, d_rel]

        pos = torch.arange(T, device=hidden_states.device)
        distance = pos[:, None] - pos[None, :]  # q - k
        rel_logits = torch.matmul(r, rel)  # [B, n, T, extent]
        idx = distance.clamp(0, self.rel_extent - 1).expand(B, self.heads_local, T, T)
        bias = rel_logits.gather(-1, idx)
        bias = bias.masked_fill((distance < 0) | (distance >= self.rel_extent), 0.0)

        if not self.is_local and self.config.inkling_log_scaling_n_floor is not None:
            tau = 1.0 + self.config.inkling_log_scaling_alpha * torch.log(
                ((pos + 1).float() / self.config.inkling_log_scaling_n_floor).clamp(min=1.0))
            tau = tau.view(1, 1, -1, 1)
            q = (q.float() * tau).to(q.dtype)
            bias = (bias.float() * tau).to(bias.dtype)

        rep = self.heads_local // self.kv_local
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling + bias
        allowed = distance >= 0
        if self.window is not None:
            allowed = allowed & (distance < self.window)
        scores = scores.masked_fill(~allowed, float("-inf"))
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(probs, v)  # [B, n, T, d]
        out = out.permute(2, 0, 1, 3).reshape(T, B, self.heads_local * self.head_dim)
        out, _ = self.linear_proj(out)
        return out


class InklingMoE(MegatronModule):
    """Sigmoid top-k router with shared-expert sink, routed experts and shared experts (replicated over TP)."""

    def __init__(self, config: InklingTransformerConfig):
        super().__init__(config=config)
        H, I = config.hidden_size, config.ffn_hidden_size
        self.num_experts = config.inkling_num_routed_experts
        self.num_shared = config.inkling_num_shared_experts
        self.topk = config.inkling_moe_router_topk
        self.route_scale = config.inkling_route_scale
        self.router_weight = _param(config, self.num_experts + self.num_shared, H, init=config.init_method)
        self.router_global_scale = _param(config, 1)
        self.register_buffer("router_expert_bias",
                             torch.zeros(self.num_experts, dtype=torch.float32, device=_device(config)))
        # gate rows first, then up rows (HF gate_up_proj layout)
        self.experts_fc1 = _param(config, self.num_experts, 2 * I, H, init=config.init_method)
        self.experts_fc2 = _param(config, self.num_experts, H, I, init=config.output_layer_init_method)
        self.shared_fc1 = _param(config, self.num_shared, 2 * I, H, init=config.init_method)
        self.shared_fc2 = _param(config, self.num_shared, H, I, init=config.output_layer_init_method)

    def route(self, x):
        logits = F.linear(x, self.router_weight)
        n = self.num_shared
        routed_logits, shared_logits = logits[..., :-n], logits[..., -n:]
        with torch.no_grad():
            choice = routed_logits.sigmoid() + self.router_expert_bias.to(logits.dtype)
            idx = torch.topk(choice, self.topk, dim=-1, sorted=False)[1]
        top = torch.cat([routed_logits.gather(-1, idx), shared_logits], dim=-1)
        lp = F.logsigmoid(top)
        w = torch.exp(lp - torch.logsumexp(lp, dim=-1, keepdim=True))
        w = w * self.route_scale * self.router_global_scale
        return w[..., : self.topk], idx, w[..., self.topk:]

    def forward(self, hidden_states):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        weights, idx, gammas = self.route(x)
        out = torch.zeros_like(x)
        flat_idx = idx.reshape(-1)
        order = torch.argsort(flat_idx, stable=True)
        counts = torch.bincount(flat_idx, minlength=self.num_experts).tolist()
        tok = order // self.topk
        wsel = weights.reshape(-1)[order]
        start = 0
        for e, c in enumerate(counts):
            if c == 0:
                continue
            t = tok[start:start + c]
            gate, up = F.linear(x[t], self.experts_fc1[e]).chunk(2, dim=-1)
            h = F.linear(F.silu(gate) * up, self.experts_fc2[e]) * wsel[start:start + c, None]
            out = out.index_add(0, t, h.to(out.dtype))
            start += c
        # shared experts, weighted by their sink gammas
        xs = x.unsqueeze(0).expand(self.num_shared, -1, -1)
        gu = torch.bmm(xs, self.shared_fc1.transpose(1, 2))
        gate, up = gu.chunk(2, dim=-1)
        act = F.silu(gate) * up * gammas.transpose(0, 1).unsqueeze(-1)
        shared = torch.bmm(act, self.shared_fc2.transpose(1, 2)).float().sum(0).to(x.dtype)
        return (out + shared).view(shape)


class InklingDecoderLayer(MegatronModule):
    def __init__(self, config: InklingTransformerConfig, layer_number: int):
        super().__init__(config=config)
        self.layer_number = layer_number  # 0-based global index
        H = config.hidden_size
        self.input_layernorm = InklingRMSNorm(config, H)
        self.self_attention = InklingAttention(config, layer_number)
        self.attn_sconv = InklingShortConv(config, H)
        self.pre_mlp_layernorm = InklingRMSNorm(config, H)
        self.mlp = InklingMoE(config)
        self.mlp_sconv = InklingShortConv(config, H)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        h = self.self_attention(self.input_layernorm(hidden_states))
        hidden_states = residual + self.attn_sconv(h)
        residual = hidden_states
        h = self.mlp(self.pre_mlp_layernorm(hidden_states))
        return residual + self.mlp_sconv(h)
