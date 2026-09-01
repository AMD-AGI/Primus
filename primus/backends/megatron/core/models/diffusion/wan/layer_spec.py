# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
WAN layer specifications for Megatron-Core integration.

This module is WAN's equivalent of ``flux/layer_spec.py`` and plays the same
role: one place where the backend (TransformerEngine, local, FP8, MXFP4) is
resolved, and one place where a DiT block declares its submodules as
``ModuleSpec``s. Before this existed, WAN re-derived the backend independently
inside every attention module, every q/k norm, and every FFN.

Three ``TransformerConfig`` clones are produced here, because WAN's attention,
q/k norms, and FFN each need knobs pinned that must not leak into the rest of
the backbone:

    - the attention config pins kv_channels, MHA query groups, biases,
      interleaved RoPE, and the ``thd`` qkv format,
    - the norm config forces RMSNorm so the backend's norm builder produces an
      RMSNorm for q/k,
    - the FFN config pins GELU with fused bias-activation and no gated linear
      unit, which is what Megatron-Bridge's WAN FFN uses.

Unlike Flux, WAN does not yet return ``TransformerBlockSubmodules``: its
backbone is an ``nn.ModuleList`` of blocks, and moving to a Megatron
``TransformerBlock`` would change the checkpoint key layout. The spec surface
here is deliberately shaped so that migration is a later, separable change.
"""

import copy
from dataclasses import dataclass
from typing import List, Optional

import torch.nn.functional as F
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec

from ..common.backend_resolution import resolve_diffusion_backend, sensitive_layer_range
from .attention import WanCrossAttentionSubmodules, WanSelfAttentionSubmodules


@dataclass
class WanLayerSpec:
    """Everything one WAN DiT block needs to build itself.

    ``self_attention`` / ``cross_attention`` / ``mlp`` are ``ModuleSpec``s whose
    submodules were resolved from a single backend provider, so a block never
    inspects ``config.transformer_impl`` itself.
    """

    self_attention: ModuleSpec
    cross_attention: ModuleSpec
    mlp: MLPSubmodules
    attention_config: object
    ffn_config: object
    rope_config: object
    norm_config: object


def _attention_config(config):
    """Config clone tuned for the WAN attention modules.

    ``apply_rope_fusion`` and ``qkv_format='thd'`` match Megatron-Bridge's WAN
    provider: it rotates q/k with the fused kernel and dispatches attention
    through the packed thd path. The SBHD/no-mask path selects a different
    fused-attention variant and leaves a residual per-layer gap, so the packed
    layout is not optional for parity.
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


def get_wan_transformer_spec_for_backend(config, backend) -> WanLayerSpec:
    """Build one WAN DiT block's spec from a single backend provider."""
    from .attention import WanCrossAttention, WanSelfAttention

    acfg = _attention_config(config)
    ncfg = _norm_config(acfg)

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

    return WanLayerSpec(
        self_attention=self_attention,
        cross_attention=cross_attention,
        mlp=MLPSubmodules(linear_fc1=column_linear, linear_fc2=row_linear),
        attention_config=acfg,
        ffn_config=_ffn_config(config),
        rope_config=_rope_config(acfg),
        norm_config=ncfg,
    )


def get_wan_layer_spec(
    config,
    backend: Optional[object] = None,
) -> List[WanLayerSpec]:
    """Per-layer specs for a WAN backbone.

    Returns one :class:`WanLayerSpec` per DiT block. All layers are identical
    unless ``sensitive_layers_enabled`` is set, in which case the first and last
    N blocks are built from the higher-precision sensitive backend -- the same
    heterogeneity Flux gets from ``get_flux_layer_spec``.

    Args:
        config: ``WanConfig``.
        backend: Explicit ``BackendSpecProvider``; resolved from the config when
            ``None``. Passing one disables sensitive-layer heterogeneity, since
            the caller has taken over backend choice.

    Example:
        >>> config = WanConfig.wan2_1_t2v_1_3b()
        >>> specs = get_wan_layer_spec(config)
        >>> len(specs) == config.num_dit_layers
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
    return specs


__all__ = [
    "WanLayerSpec",
    "get_wan_layer_spec",
    "get_wan_transformer_spec_for_backend",
]
