###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling self-attention (``InklingAttention`` in ``modeling_inkling.py``).

What one layer holds:

* ``linear_qkv`` -- q, k, v in Megatron's own fused layout: rows grouped per
  KV head (``[q heads of g | k_g | v_g]``), so a contiguous TP split hands each
  rank whole groups and Megatron tooling (e.g. Muon's ``qkv_split_shapes``)
  reads it unchanged. ``fuse_qkv`` / ``split_qkv`` in ``inkling_hf_mapping.py``
  own the layout.
* ``linear_rel`` -- the relative states r (``d_rel`` per q head), a separate
  column-parallel GEMM on the same input; its contiguous TP split lines up with
  the rank's q heads.
* ``q_layernorm`` / ``k_layernorm`` -- per-head RMSNorm; no RoPE.
* ``k_sconv`` / ``v_sconv`` -- short convs on K and V (TP-sharded channels),
  applied before the K norm.
* ``rel_proj`` -- ``[d_rel, rel_extent]`` bank of bias-vs-distance profiles;
  ``rel_extent`` is ``sliding_window_size`` on sliding layers.
* ``linear_proj`` -- the output projection.

The sliding and global layers can have different head geometry
(``swa_*``), so each layer builds from
:meth:`InklingTransformerConfig.attention_config`.

The score of query ``t`` on key ``u`` (``d = t - u``) is::

    q_t . k_u / head_dim + [0 <= d < rel_extent] * (r_t @ rel_proj)[h, d]

masked to ``0 <= d`` (global) or ``0 <= d < window`` (sliding). Global
layers multiply q and the bias by ``1 + alpha * log(max(1, (t + 1) / floor))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.transformer.attention import Attention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec

from primus.backends.megatron.core.transformer.inkling.short_conv import (
    InklingShortConv,
)

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # torch < 2.5
    create_block_mask = flex_attention = None

__all__ = [
    "InklingSelfAttention",
    "InklingSelfAttentionSubmodules",
    "InklingCoreAttention",
    "inkling_log_scale",
]

_FLEX = None
_BLOCK_MASKS = {}


def _compiled_flex_attention():
    global _FLEX
    if _FLEX is None:
        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX


def _inkling_block_mask(seq_len: int, batch: int, window: Optional[int], device, attention_mask=None):
    """Causal (``window=None``) or causal + ``window`` block mask, cached per shape.

    A Megatron ``attention_mask`` (``True`` = masked, broadcastable to
    ``[b, 1, s, s]``) is folded in per batch row; that mask is rebuilt every call.
    """
    key = (seq_len, window, str(device))
    if attention_mask is None and key in _BLOCK_MASKS:
        return _BLOCK_MASKS[key]
    extra = None if attention_mask is None else attention_mask.bool().expand(batch, 1, seq_len, seq_len)

    def mask_mod(b_idx, h_idx, q_idx, kv_idx):
        allowed = q_idx >= kv_idx
        if window is not None:
            allowed = allowed & (q_idx - kv_idx < window)
        if extra is not None:
            allowed = allowed & ~extra[b_idx, 0, q_idx, kv_idx]
        return allowed

    mask = create_block_mask(
        mask_mod, None if extra is None else batch, None, seq_len, seq_len, device=device
    )
    if extra is None:
        _BLOCK_MASKS[key] = mask
    return mask


def inkling_log_scale(seq_len: int, n_floor: int, alpha: float, device) -> torch.Tensor:
    """``tau[t] = 1 + alpha * log(max(1, (t + 1) / n_floor))`` in fp32, shape ``[seq_len]``."""
    n = torch.arange(1, seq_len + 1, device=device, dtype=torch.float32)
    return 1.0 + alpha * torch.log((n / n_floor).clamp(min=1.0))


class InklingCoreAttention(MegatronModule):
    """Softmax attention with the relative-logit bias. Holds no parameters.

    ``eager`` materialises the ``[b, heads, s, s]`` bias and scores; it is the
    numerical reference for the FlexAttention backend. ``flex`` adds the bias
    in a ``score_mod`` that reads ``rel_logits[b, h, t, t - u]`` and skips
    masked tiles through a block mask, so no ``[s, s]`` tensor is stored.
    """

    def __init__(
        self,
        config,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str = "self",
        cp_comm_type: Optional[str] = None,
        softmax_scale: Optional[float] = None,
        pg_collection=None,
        **kwargs,
    ):
        super().__init__(config=config)
        geo = config.attention_geometry(layer_number - 1)
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / geo["kv_channels"]
        self.window = geo["window"]
        self.rel_extent = geo["rel_extent"]
        self.is_sliding = geo["sliding"]
        self.log_scaling = (not self.is_sliding) and config.log_scaling_n_floor is not None
        self.backend = config.inkling_attention_backend
        self.dropout_p = config.attention_dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rel_states: torch.Tensor,
        rel_proj: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``query`` ``[s, b, h, d]``, ``key``/``value`` ``[s, b, g, d]``,
        ``rel_states`` ``[s, b, h, d_rel]``; returns ``[s, b, h * d]``.

        ``attention_mask`` is Megatron's boolean mask (``True`` = masked),
        broadcastable to ``[b, 1, s, s]``; it is combined with the causal and
        window masks.
        """
        if self.backend == "flex":
            return self._flex(query, key, value, rel_states, rel_proj, attention_mask)
        return self._eager(query, key, value, rel_states, rel_proj, attention_mask)

    def _rel_logits_and_query(self, query, rel_states, rel_proj):
        """``q`` ``[b, h, s, d]`` and ``rel_logits`` ``[b, h, s, rel_extent]``, log-scaled on global layers."""
        s = query.shape[0]
        q = query.permute(1, 2, 0, 3)
        rel_logits = torch.matmul(rel_states, rel_proj).permute(1, 2, 0, 3)
        if self.log_scaling:
            tau = inkling_log_scale(
                s, self.config.log_scaling_n_floor, self.config.log_scaling_alpha, query.device
            ).view(1, 1, s, 1)
            q = (q.float() * tau).to(q.dtype)
            rel_logits = (rel_logits.float() * tau).to(rel_logits.dtype)
        return q, rel_logits

    def _flex(self, query, key, value, rel_states, rel_proj, attention_mask):
        if flex_attention is None:
            raise RuntimeError("inkling_attention_backend: flex needs torch.nn.attention.flex_attention")
        if self.training and self.dropout_p > 0:
            raise NotImplementedError("Inkling's flex attention does not support attention dropout")
        s, b, h, d = query.shape
        g = key.shape[2]
        q, rel_logits = self._rel_logits_and_query(query, rel_states, rel_proj)
        rel_extent = self.rel_extent

        def score_mod(score, b_idx, h_idx, q_idx, kv_idx):
            distance = q_idx - kv_idx
            bias = rel_logits[b_idx, h_idx, q_idx, distance.clamp(0, rel_extent - 1)]
            return torch.where((distance >= 0) & (distance < rel_extent), score + bias.to(score.dtype), score)

        out = _compiled_flex_attention()(
            q,
            key.permute(1, 2, 0, 3),
            value.permute(1, 2, 0, 3),
            score_mod=score_mod,
            block_mask=_inkling_block_mask(s, b, self.window, query.device, attention_mask),
            scale=self.softmax_scale,
            enable_gqa=h != g,
        )
        return out.permute(2, 0, 1, 3).reshape(s, b, h * d)

    def _eager(self, query, key, value, rel_states, rel_proj, attention_mask):
        s, b, h, d = query.shape
        g = key.shape[2]
        q, rel_logits = self._rel_logits_and_query(query, rel_states, rel_proj)
        k = key.permute(1, 2, 0, 3).repeat_interleave(h // g, dim=1)
        v = value.permute(1, 2, 0, 3).repeat_interleave(h // g, dim=1)

        pos = torch.arange(s, device=query.device)
        distance = pos[:, None] - pos[None, :]
        index = distance.clamp(0, self.rel_extent - 1).expand(b, h, s, s)
        bias = rel_logits.gather(-1, index)
        bias = bias.masked_fill((distance < 0) | (distance >= self.rel_extent), 0.0)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale + bias
        allowed = distance >= 0
        if self.window is not None:
            allowed = allowed & (distance < self.window)
        if attention_mask is not None:
            allowed = allowed & ~attention_mask.bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

        # A query with no allowed key attends to nothing (as in flex), not uniformly to every key.
        probs = F.softmax(scores, dim=-1, dtype=torch.float32)
        probs = probs.masked_fill(~allowed.any(dim=-1, keepdim=True), 0.0).to(q.dtype)
        if self.training and self.dropout_p > 0:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                probs = F.dropout(probs, p=self.dropout_p)
        out = torch.matmul(probs, v)  # [b, h, s, d]
        return out.permute(2, 0, 1, 3).reshape(s, b, h * d)


@dataclass
class InklingSelfAttentionSubmodules(SelfAttentionSubmodules):
    linear_rel: Union[ModuleSpec, type] = None


class InklingSelfAttention(Attention):
    def __init__(
        self,
        config,
        submodules: InklingSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        cp_comm_type: Optional[str] = None,
        pg_collection=None,
        pp_layer_offset: Optional[int] = None,
        name: Optional[str] = None,  # passed by Megatron-core 0.19+
    ):
        layer_idx = layer_number - 1
        geo = config.attention_geometry(layer_idx)
        super().__init__(
            config=config.attention_config(layer_idx),
            submodules=submodules,
            layer_number=layer_number,
            attention_type="self",
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            **({"name": name} if name is not None else {}),
        )
        cfg = self.config
        if cfg.num_query_groups % self.world_size:
            raise ValueError(
                f"Inkling layer {layer_idx}: {cfg.num_query_groups} KV groups do not split over TP={self.world_size}"
            )
        if cfg.context_parallel_size > 1:
            raise NotImplementedError("Inkling attention does not support context parallelism")

        self.is_sliding = geo["sliding"]
        self.rel_extent = geo["rel_extent"]
        self.d_rel = cfg.d_rel
        self.linear_qkv_out_dim = self.query_projection_size + 2 * self.kv_projection_size

        self.linear_qkv = submodules.linear_qkv(
            cfg.hidden_size,
            self.linear_qkv_out_dim,
            config=cfg,
            init_method=cfg.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkv",
            tp_group=self.pg_collection.tp,
        )
        self.linear_rel = submodules.linear_rel(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.d_rel,
            config=cfg,
            init_method=cfg.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="rel",
            tp_group=self.pg_collection.tp,
        )
        self.q_layernorm = submodules.q_layernorm(
            hidden_size=self.hidden_size_per_attention_head, config=cfg, eps=cfg.layernorm_epsilon
        )
        self.k_layernorm = submodules.k_layernorm(
            hidden_size=self.hidden_size_per_attention_head, config=cfg, eps=cfg.layernorm_epsilon
        )

        kv_channels_local = self.num_query_groups_per_partition * self.hidden_size_per_attention_head
        k = cfg.short_conv_kernel_size
        self.k_sconv = InklingShortConv(cfg, kv_channels_local, k, tensor_parallel=True)
        self.v_sconv = InklingShortConv(cfg, kv_channels_local, k, tensor_parallel=True)

        self.rel_proj = torch.nn.Parameter(torch.empty(cfg.d_rel, self.rel_extent, dtype=cfg.params_dtype))
        setattr(self.rel_proj, "sequence_parallel", cfg.sequence_parallel)
        if cfg.perform_initialization and self.rel_proj.device.type != "meta":
            cfg.init_method(self.rel_proj)

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, *args, **kwargs):
        """``[s, b, hidden]`` -> q ``[s, b, h, d]``, k/v ``[s, b, g, d]``, r ``[s, b, h, d_rel]``.

        q and k come out RMS-normalised; k and v went through their short
        convs first.
        """
        mixed, _ = self.linear_qkv(hidden_states)
        s, b = mixed.shape[:2]
        groups = self.num_query_groups_per_partition
        heads_per_group = self.num_attention_heads_per_partition // groups
        d = self.hidden_size_per_attention_head
        mixed = mixed.view(s, b, groups, -1)
        q, k, v = torch.split(mixed, [heads_per_group * d, d, d], dim=-1)
        q = q.reshape(s, b, groups * heads_per_group, d)
        r, _ = self.linear_rel(hidden_states)
        r = r.view(s, b, groups * heads_per_group, self.d_rel)
        k = self.k_sconv(k.reshape(s, b, groups * d)).view(s, b, groups, d)
        v = self.v_sconv(v.reshape(s, b, groups * d)).view(s, b, groups, d)
        return self.q_layernorm(q), self.k_layernorm(k), v, r

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError("Inkling attention has no KV-cache inference path")
        if packed_seq_params is not None:
            raise NotImplementedError("Packed sequences (per-document masking) are optional item O8")
        if rotary_pos_emb is not None or attention_bias is not None:
            raise ValueError("Inkling attention takes no rotary embedding or external attention bias")
        query, key, value, rel_states = self.get_query_key_value_tensors(hidden_states)
        if self.checkpoint_core_attention and self.training:
            core = tensor_parallel.checkpoint(
                self.core_attention, False, query, key, value, rel_states, self.rel_proj, attention_mask
            )
        else:
            core = self.core_attention(
                query, key, value, rel_states, self.rel_proj, attention_mask=attention_mask
            )
        return self.linear_proj(core)
