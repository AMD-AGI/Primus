# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
WAN attention modules, built from backend-resolved submodule specs.

WAN needs its own submodule dataclasses rather than reusing Flux's
:class:`JointSelfAttentionSubmodules`: Flux's joint attention carries a second
``added_*`` QKV stream for the text tokens, whereas WAN conditions on text
through a separate cross-attention with an independent ``linear_q`` and a fused
``linear_kv``.

Both modules run sequence-major ``[S, B, dim]`` end to end, matching
Megatron-Bridge, so AdaLN modulation, the LayerNorms, residuals, and the
attention core all operate on the same layout and dispatch the same kernels.

TE path: q/k/v are packed into the ``thd`` layout with ``cu_seqlens`` and rotated
with the fused RoPE kernel, matching Megatron-Bridge's WAN attention exactly.
Local path: q/k/v stay in SBHD and use unfused interleaved RoPE plus
``PrimusTurboLocalAttention``.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from torch import Tensor

from .layers import thd_cu_seqlens, unfused_fp32_attention


@dataclass
class WanSelfAttentionSubmodules:
    """Submodules of WAN self-attention (fused QKV, RoPE-rotated q/k)."""

    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


@dataclass
class WanCrossAttentionSubmodules:
    """Submodules of WAN cross-attention (split q, fused kv, no RoPE)."""

    linear_q: Union[ModuleSpec, type] = None
    linear_kv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class WanAttentionBase(MegatronModule):
    """Shared plumbing for WAN self- and cross-attention.

    ``config`` here is the attention config produced by
    ``layer_spec._attention_config`` -- kv_channels, num_query_groups, biases,
    and the RoPE/qkv-format knobs are already pinned on it. ``norm_config`` is
    the RMSNorm clone used for the q/k norms only.
    """

    def __init__(
        self,
        config,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        norm_config=None,
    ):
        super().__init__(config)
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.norm_config = norm_config if norm_config is not None else config
        self.local = config.transformer_impl == "local"
        self.use_fp32_attention = getattr(config, "use_fp32_attention", False)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.kv_channels
        self.inner_dim = self.num_heads * self.head_dim
        self.across_heads = bool(getattr(config, "layernorm_across_heads", True))
        self.norm_size = self.inner_dim if self.across_heads else self.head_dim

    def _build_norm(self, spec, hidden_size: int):
        return build_module(
            spec,
            config=self.norm_config,
            hidden_size=hidden_size,
            eps=self.config.layernorm_epsilon,
        )

    def _apply_qk_norm(self, tensor: Tensor, norm, seq_len: int, batch: int) -> Tensor:
        """Normalize q or k, folding heads into the feature dim when required.

        The norm holds an fp32 weight, so a bf16 input comes back as fp32 while
        ``value`` stays bf16. Flash attention rejects that mismatch, so cast
        back to the input dtype -- computing the norm in fp32 and casting down
        is what diffusers and Megatron-Bridge do.
        """
        dtype = tensor.dtype
        if self.across_heads:
            flat = tensor.reshape(seq_len, batch, -1)
            if not self.local:
                flat = flat.contiguous()
            normed = norm(flat).view(seq_len, batch, self.num_heads, self.head_dim)
        else:
            normed = norm(tensor if self.local else tensor.contiguous())
        return normed.to(dtype)

    def _core(self, query, key, value, packed_seq_params=None) -> Tensor:
        """Run the attention core, honouring the fp32 parity path."""
        if self.use_fp32_attention and query.dtype == torch.float32:
            return unfused_fp32_attention(query, key, value, packed_seq_params).to(query.dtype)
        if packed_seq_params is None:
            return self.core_attention(query, key, value, None, self.attn_mask_type)
        return self.core_attention(
            query,
            key,
            value,
            None,
            attn_mask_type=self.attn_mask_type,
            packed_seq_params=packed_seq_params,
        )


class WanSelfAttention(WanAttentionBase):
    """WAN self-attention: fused ``linear_qkv``, RMSNorm-ed and RoPE-rotated q/k."""

    def __init__(
        self,
        config,
        submodules: WanSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        rope_config=None,
        norm_config=None,
    ):
        super().__init__(config, layer_number, attn_mask_type, norm_config)
        self.rope_config = rope_config if rope_config is not None else config

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            config.hidden_size,
            3 * self.inner_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkv",
        )
        self.q_layernorm = self._build_norm(submodules.q_layernorm, self.norm_size)
        self.k_layernorm = self._build_norm(submodules.k_layernorm, self.norm_size)
        self.core_attention = build_module(
            submodules.core_attention,
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
        )
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.inner_dim,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="proj",
        )

    def _split_qkv(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor, int, int]:
        s, b, _ = hidden_states.shape
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        mixed_qkv = mixed_qkv.view(s, b, self.num_heads, 3 * self.head_dim)
        query, key, value = torch.split(mixed_qkv, [self.head_dim, self.head_dim, self.head_dim], dim=3)
        query = self._apply_qk_norm(query, self.q_layernorm, s, b)
        key = self._apply_qk_norm(key, self.k_layernorm, s, b)
        return query, key, value, s, b

    def forward(
        self, hidden_states: Tensor, rotary_freqs: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        query, key, value, s, b = self._split_qkv(hidden_states)

        if self.local:
            if rotary_freqs is not None:
                query = apply_rotary_pos_emb(query, rotary_freqs, config=self.rope_config)
                key = apply_rotary_pos_emb(key, rotary_freqs, config=self.rope_config)
            core_out = self._core(query, key, value)
            return self.linear_proj(core_out)

        # Pack the batch into the token dim BEFORE RoPE so the fused thd rotary
        # and the fused attention kernel see Megatron-Bridge's exact layout.
        cu_seqlens = thd_cu_seqlens(s, b, hidden_states.device)
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
        )
        query = query.transpose(0, 1).reshape(b * s, self.num_heads, self.head_dim).contiguous()
        key = key.transpose(0, 1).reshape(b * s, self.num_heads, self.head_dim).contiguous()
        value = value.transpose(0, 1).reshape(b * s, self.num_heads, self.head_dim).contiguous()

        if rotary_freqs is not None:
            query = apply_rotary_pos_emb(query, rotary_freqs, config=self.config, cu_seqlens=cu_seqlens)
            key = apply_rotary_pos_emb(key, rotary_freqs, config=self.config, cu_seqlens=cu_seqlens)

        core_out = self._core(query, key, value, packed_seq_params)
        core_out = core_out.reshape(b, s, -1).transpose(0, 1).contiguous()
        return self.linear_proj(core_out)


class WanCrossAttention(WanAttentionBase):
    """WAN cross-attention: query from video tokens, key/value from text tokens."""

    def __init__(
        self,
        config,
        submodules: WanCrossAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        norm_config=None,
    ):
        super().__init__(config, layer_number, attn_mask_type, norm_config)

        self.linear_q = build_module(
            submodules.linear_q,
            config.hidden_size,
            self.inner_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="q",
        )
        self.linear_kv = build_module(
            submodules.linear_kv,
            config.hidden_size,
            2 * self.inner_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv",
        )
        self.q_layernorm = self._build_norm(submodules.q_layernorm, self.norm_size)
        self.k_layernorm = self._build_norm(submodules.k_layernorm, self.norm_size)
        self.core_attention = build_module(
            submodules.core_attention,
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="cross",
        )
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.inner_dim,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="proj",
        )

    def forward(self, hidden_states: Tensor, context: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        sq, b, _ = hidden_states.shape
        skv = context.shape[0]

        query, _ = self.linear_q(hidden_states)
        query = query.view(sq, b, self.num_heads, self.head_dim)

        mixed_kv, _ = self.linear_kv(context)
        mixed_kv = mixed_kv.view(skv, b, self.num_heads, 2 * self.head_dim)
        key, value = torch.split(mixed_kv, [self.head_dim, self.head_dim], dim=3)

        query = self._apply_qk_norm(query, self.q_layernorm, sq, b)
        key = self._apply_qk_norm(key, self.k_layernorm, skv, b)

        if self.local:
            core_out = self._core(query, key, value)
            return self.linear_proj(core_out)

        # Query (video) and key/value (text) carry independent cu_seqlens.
        cu_seqlens_q = thd_cu_seqlens(sq, b, hidden_states.device)
        cu_seqlens_kv = thd_cu_seqlens(skv, b, hidden_states.device)
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q,
            cu_seqlens_kv_padded=cu_seqlens_kv,
        )
        query = query.transpose(0, 1).reshape(b * sq, self.num_heads, self.head_dim).contiguous()
        key = key.transpose(0, 1).reshape(b * skv, self.num_heads, self.head_dim).contiguous()
        value = value.transpose(0, 1).reshape(b * skv, self.num_heads, self.head_dim).contiguous()

        core_out = self._core(query, key, value, packed_seq_params)
        core_out = core_out.reshape(b, sq, -1).transpose(0, 1).contiguous()
        return self.linear_proj(core_out)


__all__ = [
    "WanSelfAttention",
    "WanCrossAttention",
    "WanSelfAttentionSubmodules",
    "WanCrossAttentionSubmodules",
]
