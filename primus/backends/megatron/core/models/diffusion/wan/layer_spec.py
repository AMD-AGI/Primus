# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
WAN layer specifications for Megatron-Core integration.

This module is WAN's equivalent of ``flux/layer_spec.py`` and plays the same
role: one place where the backend (TransformerEngine, local, FP8, MXFP4) is
resolved, and one place where a DiT block declares its submodules as
``ModuleSpec``s. A WAN block is :class:`WanLayer`, a Megatron
``TransformerLayer``, and :func:`get_wan_layer_spec` returns the
``TransformerBlockSubmodules`` the backbone's ``DiffusionTransformerBlock`` is
built from, as Flux does.

Three ``TransformerConfig`` clones are produced here, because WAN's attention,
q/k norms, and FFN each need knobs pinned that must not leak into the rest of
the backbone:

    - the attention config pins kv_channels, MHA query groups, biases,
      interleaved RoPE, and the ``thd`` qkv format,
    - the norm config forces RMSNorm so the backend's norm builder produces an
      RMSNorm for q/k,
    - the FFN config pins GELU with fused bias-activation and no gated linear
      unit, which is what Megatron-Bridge's WAN FFN uses.
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from torch import Tensor

from ..common.backend_resolution import resolve_diffusion_backend, sensitive_layer_range
from .attention import WanCrossAttentionSubmodules, WanSelfAttentionSubmodules


def _attention_config(config):
    """Config clone tuned for the WAN attention modules.

    ``qkv_format='thd'`` matches Megatron-Bridge's WAN provider, which
    dispatches attention through the packed thd path. The SBHD/no-mask path
    selects a different fused-attention variant and leaves a residual per-layer
    gap, so the packed layout is not optional for parity. ``apply_rope_fusion``
    selects TE's fused rotary, which the TE path applies in SBHD before packing:
    the fused thd rotary ignores ``rotary_interleaved`` (see
    ``WanSelfAttention.forward``).
    """
    acfg = copy.copy(config)
    head_dim = config.hidden_size // config.num_attention_heads
    acfg.kv_channels = head_dim
    acfg.num_query_groups = config.num_attention_heads
    acfg.attention_dropout = 0.0
    acfg.add_bias_linear = True
    acfg.add_qkv_bias = True
    acfg.rotary_interleaved = True
    acfg.apply_rope_fusion = True
    acfg.qkv_format = "thd"
    return acfg


def _norm_config(config):
    """Config clone forcing RMSNorm, used only to build the q/k norms.

    WAN's block norms are LayerNorm but its q/k norms are RMSNorm, and the norm
    modules pick between the two from ``config.normalization`` rather than from
    a constructor flag. Without this clone the q/k norms come out as LayerNorms,
    which both changes the math (LayerNorm re-centers) and adds ``.bias``
    entries that no WAN checkpoint contains.
    """
    ncfg = copy.copy(config)
    ncfg.normalization = "RMSNorm"
    return ncfg


def _rope_config(acfg):
    """Attention-config clone for the local (unfused) RoPE path.

    The local path keeps q/k in SBHD and calls the unfused interleaved rotary,
    so fusion is off while ``rotary_interleaved`` stays on.
    """
    rcfg = copy.copy(acfg)
    rcfg.apply_rope_fusion = False
    return rcfg


def _ffn_config(config):
    """Config clone for the WAN feed-forward ``MLP``, matching Megatron-Bridge.

    With ``activation_func=F.gelu`` + ``bias_activation_fusion=True`` +
    ``gated_linear_unit=False`` the MLP dispatches the fused ``bias_gelu_impl``,
    which is the tanh approximation of GELU -- exactly diffusers'
    ``gelu-approximate``.
    """
    fcfg = copy.copy(config)
    fcfg.add_bias_linear = True
    fcfg.gated_linear_unit = False
    fcfg.activation_func = F.gelu
    fcfg.bias_activation_fusion = True
    return fcfg


class WanLayer(TransformerLayer):
    """One WAN DiT block: gated self-attention, cross-attention, gated MLP.

    The block is Megatron's decoder layer with cross-attention plus WAN's AdaLN:
    ``input_layernorm`` / ``pre_mlp_layernorm`` are the modulated (affine-free)
    norms, ``pre_cross_attn_layernorm`` is the optional affine cross-attention
    norm, and ``scale_shift_table`` holds the six per-block modulation offsets.

    AdaLN modulation, the LayerNorms, and the residual adds run in the tensor's
    native dtype with plain ``nn.LayerNorm`` -- Megatron-Bridge's
    normalize/modulate/scale-add math, not the diffusers explicit-fp32 path.

    The attention and MLP modules need the config clones above, which
    ``TransformerLayer.__init__`` would not pass, so the base class builds only
    identity placeholders and this class builds the spec's ``self_attention``,
    ``cross_attention`` and ``mlp`` itself.

    Input/Output, matching ``DiffusionTransformerBlock``'s layer contract:
        (hidden_states ``[S, B, D]``, context ``[S_txt, B, D]``, rotary_pos_emb
        ``[S, 1, 1, head_dim]``, timestep_emb ``[B, 6, D]``)
        -> (hidden_states, context), with context returned unchanged.
    """

    def __init__(
        self,
        config,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        **kwargs,
    ):
        super().__init__(
            config=config, submodules=TransformerLayerSubmodules(), layer_number=layer_number, **kwargs
        )
        dim = config.hidden_size
        eps = config.layernorm_epsilon

        acfg = _attention_config(config)
        ncfg = _norm_config(acfg)
        rope_config = _rope_config(acfg) if config.transformer_impl == "local" else acfg

        self.input_layernorm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attention = build_module(
            submodules.self_attention,
            config=acfg,
            layer_number=self.layer_number,
            rope_config=rope_config,
            norm_config=ncfg,
        )
        self.pre_cross_attn_layernorm = (
            nn.LayerNorm(dim, eps=eps, elementwise_affine=True) if config.cross_attn_norm else nn.Identity()
        )
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=acfg,
            layer_number=self.layer_number,
            norm_config=ncfg,
        )
        self.pre_mlp_layernorm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.mlp = build_module(
            submodules.mlp, config=_ffn_config(config), ffn_hidden_size=config.ffn_hidden_size
        )
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        timestep_emb: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        if timestep_emb is None:
            raise ValueError("WanLayer requires timestep_emb for AdaLN modulation")

        # timestep_emb is [B, 6, dim]; transposing gives modulation terms of
        # [1, B, dim] so they broadcast over the sequence-major activations.
        mod = (self.scale_shift_table + timestep_emb).transpose(0, 1)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = mod.chunk(6, dim=0)

        norm_h = self.input_layernorm(hidden_states) * (1 + scale_msa) + shift_msa
        attn_out, attn_bias = self.self_attention(norm_h, rotary_pos_emb)
        if attn_bias is not None:
            attn_out = attn_out + attn_bias
        hidden_states = hidden_states + gate_msa * attn_out

        norm_h = self.pre_cross_attn_layernorm(hidden_states)
        attn_out, attn_bias = self.cross_attention(norm_h, context)
        if attn_bias is not None:
            attn_out = attn_out + attn_bias
        hidden_states = hidden_states + attn_out

        norm_h = self.pre_mlp_layernorm(hidden_states) * (1 + c_scale_msa) + c_shift_msa
        # The mcore MLP returns (output, fc2_bias) under skip_bias_add; diffusers
        # folds that bias in before the gate scaling, so add it here.
        ff_out, ff_bias = self.mlp(norm_h)
        if ff_bias is not None:
            ff_out = ff_out + ff_bias
        return hidden_states + c_gate_msa * ff_out, context


def get_wan_transformer_spec_for_backend(config, backend) -> ModuleSpec:
    """Build one WAN DiT block's spec from a single backend provider."""
    from .attention import WanCrossAttention, WanSelfAttention

    # The local path uses full non-causal flash attention with no mask; the TE
    # path uses the packed/padding kernel to match Megatron-Bridge.
    attn_mask_type = AttnMaskType.no_mask if config.transformer_impl == "local" else AttnMaskType.padding

    qk_norm = backend.layer_norm(rms_norm=True, for_qk=True)
    column_linear = backend.column_parallel_linear()
    row_linear = backend.row_parallel_linear()
    core_attention = backend.core_attention()

    self_attention = ModuleSpec(
        module=WanSelfAttention,
        params={"attn_mask_type": attn_mask_type},
        submodules=WanSelfAttentionSubmodules(
            linear_qkv=column_linear,
            core_attention=core_attention,
            q_layernorm=qk_norm,
            k_layernorm=qk_norm,
            linear_proj=row_linear,
        ),
    )

    cross_attention = ModuleSpec(
        module=WanCrossAttention,
        params={"attn_mask_type": attn_mask_type},
        submodules=WanCrossAttentionSubmodules(
            linear_q=column_linear,
            linear_kv=column_linear,
            core_attention=core_attention,
            q_layernorm=qk_norm,
            k_layernorm=qk_norm,
            linear_proj=row_linear,
        ),
    )

    return ModuleSpec(
        module=WanLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=self_attention,
            cross_attention=cross_attention,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(linear_fc1=column_linear, linear_fc2=row_linear),
            ),
        ),
    )


def get_wan_layer_spec(
    config,
    backend: Optional[object] = None,
) -> TransformerBlockSubmodules:
    """Block spec for a WAN backbone.

    Returns one :class:`WanLayer` spec per DiT block and no final norm (WAN's
    output norm is modulated, so the backbone applies it). All layers are
    identical unless ``sensitive_layers_enabled`` is set, in which case the
    first and last N blocks are built from the higher-precision sensitive
    backend -- the same heterogeneity Flux gets from ``get_flux_layer_spec``.
    WAN requires ``pipeline_model_parallel_size == 1``, so the list is never
    sliced per stage.

    Args:
        config: ``WanConfig``.
        backend: Explicit ``BackendSpecProvider``; resolved from the config when
            ``None``. Passing one disables sensitive-layer heterogeneity, since
            the caller has taken over backend choice.

    Example:
        >>> config = WanConfig.wan2_1_t2v_1_3b()
        >>> spec = get_wan_layer_spec(config)
        >>> len(spec.layer_specs) == config.num_dit_layers
        True
    """
    if backend is None:
        backend, sensitive_backend = resolve_diffusion_backend(config)
    else:
        sensitive_backend = None

    total = config.num_dit_layers
    num_start, num_end = sensitive_layer_range(config, total) if sensitive_backend is not None else (0, 0)

    specs = []
    for i in range(total):
        is_sensitive = (i < num_start) or (i >= total - num_end)
        layer_backend = sensitive_backend if is_sensitive else backend
        specs.append(get_wan_transformer_spec_for_backend(config, layer_backend))
    return TransformerBlockSubmodules(layer_specs=specs, layer_norm=None)


__all__ = [
    "WanLayer",
    "get_wan_layer_spec",
    "get_wan_transformer_spec_for_backend",
]
