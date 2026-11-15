# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
A conversion that cannot be loaded has to fail, not report success.

``strict=True`` checks the source keys that were dropped and the target keys
that are missing, so a key the converter writes but the backbone does not have,
or one backbone written over another, would pass it unnoticed.
"""

import re

import pytest
import torch

from primus.backends.megatron.core.models.diffusion.wan.checkpoint_converter import (
    convert_hf_to_primus,
    expected_diffusers_keys,
    expected_mcore_keys,
)
from primus.backends.megatron.core.models.diffusion.wan.config import WanConfig

TINY = dict(hidden_size=64, num_attention_heads=2, num_dit_layers=2, ffn_hidden_size=128, text_embed_dim=32)
_FUSED = re.compile(r"attn[12]\.to_[qkv]\.(weight|bias)$")


def _config(**kwargs):
    return WanConfig.wan2_1_t2v_1_3b(**TINY, **kwargs)


def _diffusers_state_dict(config):
    """Zeros under every diffusers key; the per-head fused projections get ``[inner, ...]`` shapes."""
    inner = config.hidden_size
    state_dict = {}
    for key in expected_diffusers_keys(config):
        if not _FUSED.search(key):
            state_dict[key] = torch.zeros(1)
        else:
            state_dict[key] = torch.zeros((inner, 4) if key.endswith("weight") else (inner,))
    return state_dict


def test_cross_attn_norm_maps_onto_the_pre_cross_attention_layernorm():
    config = _config()

    out, _ = convert_hf_to_primus(_diffusers_state_dict(config), config, strict=True)

    assert set(out) == expected_mcore_keys(config)
    assert "decoder.layers.0.pre_cross_attn_layernorm.weight" in out


def test_norm2_without_cross_attn_norm_is_dropped_not_written():
    """The backbone has an Identity there, so a written key would fail the load."""
    config = _config(cross_attn_norm=False)
    source = _diffusers_state_dict(_config())

    out, report = convert_hf_to_primus(source, config)

    assert not [k for k in out if "pre_cross_attn_layernorm" in k]
    assert "blocks.0.norm2.weight" in report.dropped
    with pytest.raises(RuntimeError, match="strict=True forbids lossy conversion"):
        convert_hf_to_primus(source, config, strict=True)


def test_both_experts_keep_their_own_prefixes():
    config = _config()
    bare = _diffusers_state_dict(config)
    source = {f"{p}{k}": v for p in ("transformer.", "transformer_2.") for k, v in bare.items()}

    out, _ = convert_hf_to_primus(source, config, strict=True)

    assert len(out) == 2 * len(expected_mcore_keys(config))


def test_one_prefix_for_both_experts_is_rejected():
    """Both converted backbones would land under it, the second over the first."""
    config = _config()
    bare = _diffusers_state_dict(config)
    source = {f"{p}{k}": v for p in ("transformer.", "transformer_2.") for k, v in bare.items()}

    with pytest.raises(ValueError, match="would write both backbones"):
        convert_hf_to_primus(source, config, prefix="transformer.")
