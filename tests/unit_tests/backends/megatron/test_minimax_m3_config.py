###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for MiniMax-M3's MSATransformerConfig and its selection patch.

Verifies:
  1. ``sparse_attention_freq`` accepts the spellings the preset uses and
     normalizes to a per-layer 0/1 pattern.
  2. Invalid MSA settings are rejected rather than silently accepted.
  3. The patch makes ``core_transformer_config_from_args`` resolve
     ``MSATransformerConfig``, and leaves non-MSA runs (and explicit
     ``config_class=`` callers) alone.
"""

import pytest

pytest.importorskip("megatron")

from megatron.core.transformer.transformer_config import TransformerConfig

import primus.backends.megatron.patches.minimax_m3_config_patches as patch_mod
from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
    MSATransformerConfig,
)

# The preset's own values; kwargs a bare TransformerConfig needs to construct.
_M3_FREQ = "([0]*3+[1]*57)"
_BASE = dict(num_layers=60, hidden_size=6144, num_attention_heads=64)


def _config(**overrides):
    return MSATransformerConfig(**{**_BASE, **overrides})


def test_layer_pattern_from_list_expression():
    config = _config(sparse_attention_freq=_M3_FREQ)

    assert config.sparse_layer_pattern == (0,) * 3 + (1,) * 57
    # __post_init__ normalizes the stored field too, so consumers never see the string.
    assert config.sparse_attention_freq == (0,) * 3 + (1,) * 57


def test_layer_pattern_defaults_to_every_layer():
    assert _config().sparse_layer_pattern == (1,) * 60


def test_released_defaults_match_config_json():
    config = _config(sparse_attention_freq=_M3_FREQ)

    assert config.sparse_num_index_heads == 4
    assert config.sparse_index_dim == 128
    assert config.sparse_block_size == 128
    assert config.sparse_topk_blocks == 16
    assert config.sparse_score_type == "max"
    assert config.sparse_init_block == 0
    assert config.sparse_local_block == 1


def test_disabled_msa_reports_no_sparse_layers():
    config = _config(minimax_sparse_attention=False, sparse_attention_freq=_M3_FREQ)

    assert config.sparse_layer_pattern == (0,) * 60
    # Validation is skipped entirely when the family switch is off.
    assert config.sparse_attention_freq == _M3_FREQ


@pytest.mark.parametrize(
    "overrides, exc",
    [
        ({"sparse_score_type": "sum"}, NotImplementedError),
        ({"sparse_attention_freq": "([0]*3+[1]*5)"}, ValueError),
        ({"sparse_attention_freq": [2] * 60}, ValueError),
        ({"sparse_topk_blocks": 0}, ValueError),
        ({"sparse_local_block": -1}, ValueError),
        ({"multi_latent_attention": True}, ValueError),
    ],
)
def test_invalid_settings_are_rejected(overrides, exc):
    with pytest.raises(exc):
        _config(**overrides)


@pytest.fixture
def patched_config_selection(monkeypatch):
    """Install the config-class patch, stubbing the function it wraps.

    ``core_transformer_config_from_args`` reads dozens of attributes off a real
    Megatron ``args``; the patch only adds the config-class branch, so the stub
    records which class the wrapper resolved and lets the test assert on that.
    """
    import sys
    from types import SimpleNamespace

    import megatron.training.arguments as arguments_module

    resolved = {}

    def fake_original(args, config_class=None):
        resolved["config_class"] = config_class
        return config_class

    monkeypatch.setattr(arguments_module, "_primus_applied_patch_keys", set(), raising=False)
    monkeypatch.setattr(arguments_module, "core_transformer_config_from_args", fake_original)
    # The patch rebinds gpt_builders' local name; a stub keeps the test from
    # importing the real module (and transformer_engine with it).
    monkeypatch.setitem(sys.modules, "gpt_builders", SimpleNamespace())

    patch_mod.patch_minimax_m3_config(ctx=None)
    return arguments_module.core_transformer_config_from_args, resolved


def test_patch_selects_msa_config(patched_config_selection):
    from types import SimpleNamespace

    wrapper, resolved = patched_config_selection

    wrapper(SimpleNamespace(minimax_sparse_attention=True))

    assert resolved["config_class"] is MSATransformerConfig


def test_patch_leaves_non_msa_runs_alone(patched_config_selection):
    from types import SimpleNamespace

    wrapper, resolved = patched_config_selection

    wrapper(SimpleNamespace(minimax_sparse_attention=False))

    # None means upstream keeps its own default (TransformerConfig / MLA).
    assert resolved["config_class"] is None


def test_patch_respects_an_explicit_config_class(patched_config_selection):
    from types import SimpleNamespace

    wrapper, resolved = patched_config_selection

    wrapper(SimpleNamespace(minimax_sparse_attention=True), config_class=TransformerConfig)

    assert resolved["config_class"] is TransformerConfig


def test_patch_is_idempotent(patched_config_selection):
    import megatron.training.arguments as arguments_module

    wrapper, _ = patched_config_selection

    patch_mod.patch_minimax_m3_config(ctx=None)

    assert arguments_module.core_transformer_config_from_args is wrapper
