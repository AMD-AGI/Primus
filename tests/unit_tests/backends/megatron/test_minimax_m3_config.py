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
  4. Setting MSA and MLA together raises instead of silently downgrading to
     ``MLATransformerConfig``, and the wrapper reaches modules that imported
     the symbol by name.
  5. ``swigluoai`` and ``use_gemma_norm`` are derived onto the upstream fields
     that already express them.
"""

import pytest

pytest.importorskip("megatron")

from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.transformer.transformer_config import TransformerConfig

import primus.backends.megatron.patches.minimax_m3_config_patches as patch_mod
from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
    MSATransformerConfig,
)

# The preset's own values; kwargs a bare TransformerConfig needs to construct.
# `activation_func` stands in for what `core_transformer_config_from_args` sets
# from the preset's `quick_geglu: true`.
_M3_FREQ = "([0]*3+[1]*57)"
_BASE = dict(
    num_layers=60,
    hidden_size=6144,
    num_attention_heads=64,
    activation_func=quick_gelu,
    gated_linear_unit=True,
    normalization="RMSNorm",
    # The indexer emits one block selection per GQA group, so these must agree;
    # the preset pairs num_query_groups 4 with sparse_num_index_heads 4.
    num_query_groups=4,
)


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


def test_swigluoai_derives_quick_geglu_fields():
    """HF swigluoai is quick_gelu(gate.clamp(max=limit)) * (up.clamp(+-limit) + 1),
    which is Megatron's GLU path with these two fields set."""
    config = _config(sparse_attention_freq=_M3_FREQ)

    assert config.activation_func is quick_gelu
    assert config.gated_linear_unit is True
    assert config.glu_linear_offset == 1.0
    assert config.activation_func_clamp_value == 7.0


def test_swiglu_limit_flows_into_the_clamp():
    assert _config(swiglu_limit=5.0).activation_func_clamp_value == 5.0


def test_non_quick_gelu_activation_is_rejected():
    import torch.nn.functional as F

    with pytest.raises(ValueError, match="quick_geglu"):
        _config(activation_func=F.silu)


def test_unexpressible_swiglu_alpha_is_rejected():
    """quick_gelu hardcodes 1.702, so any other alpha would silently train a
    different activation than the preset asked for."""
    with pytest.raises(NotImplementedError, match="swiglu_alpha"):
        _config(swiglu_alpha=1.5)


def test_use_gemma_norm_sets_zero_centered_gamma():
    assert _config(normalization="RMSNorm").layernorm_zero_centered_gamma is True


def test_gemma_norm_off_leaves_the_flag_alone():
    assert _config(use_gemma_norm=False).layernorm_zero_centered_gamma is False


def test_gemma_norm_requires_rmsnorm():
    with pytest.raises(ValueError, match="RMSNorm"):
        _config(normalization="LayerNorm")


def test_activation_and_norm_are_skipped_when_msa_is_off():
    import torch.nn.functional as F

    config = _config(minimax_sparse_attention=False, activation_func=F.silu, use_gemma_norm=True)

    assert config.glu_linear_offset == 0.0
    assert config.layernorm_zero_centered_gamma is False


@pytest.fixture
def patched_config_selection(monkeypatch):
    """Install the config-class patch, stubbing the function it wraps.

    ``core_transformer_config_from_args`` reads dozens of attributes off a real
    Megatron ``args``; the patch only adds the config-class branch, so the stub
    records which class the wrapper resolved and lets the test assert on that.

    ``importer`` stands in for ``gpt_builders`` -- a module holding the symbol
    by name rather than going through ``megatron.training.arguments``.
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

    importer = SimpleNamespace(core_transformer_config_from_args=fake_original)
    monkeypatch.setitem(sys.modules, "_minimax_m3_fake_importer", importer)

    patch_mod.patch_minimax_m3_config(ctx=None)
    return SimpleNamespace(
        wrapper=arguments_module.core_transformer_config_from_args,
        resolved=resolved,
        importer=importer,
    )


def _args(**overrides):
    from types import SimpleNamespace

    return SimpleNamespace(**{"minimax_sparse_attention": True, "multi_latent_attention": False, **overrides})


def test_patch_selects_msa_config(patched_config_selection):
    patched_config_selection.wrapper(_args())

    assert patched_config_selection.resolved["config_class"] is MSATransformerConfig


def test_patch_leaves_non_msa_runs_alone(patched_config_selection):
    patched_config_selection.wrapper(_args(minimax_sparse_attention=False))

    # None means upstream keeps its own default (TransformerConfig / MLA).
    assert patched_config_selection.resolved["config_class"] is None


def test_patch_respects_an_explicit_config_class(patched_config_selection):
    patched_config_selection.wrapper(_args(), config_class=TransformerConfig)

    assert patched_config_selection.resolved["config_class"] is TransformerConfig


def test_msa_plus_mla_raises_instead_of_silently_downgrading(patched_config_selection):
    """Upstream's MLA branch has no `config_class is None` guard, so it would
    replace MSATransformerConfig with MLATransformerConfig and drop every
    sparse_* field without a word."""
    with pytest.raises(ValueError, match="cannot both be set"):
        patched_config_selection.wrapper(_args(multi_latent_attention=True))

    assert "config_class" not in patched_config_selection.resolved


def test_patch_reaches_modules_that_imported_the_symbol_by_name(patched_config_selection):
    importer = patched_config_selection.importer

    assert importer.core_transformer_config_from_args is patched_config_selection.wrapper


def test_patch_is_idempotent(patched_config_selection):
    import megatron.training.arguments as arguments_module

    patch_mod.patch_minimax_m3_config(ctx=None)

    assert arguments_module.core_transformer_config_from_args is patched_config_selection.wrapper


def test_preset_builds_the_real_config(monkeypatch):
    """End-to-end over the shipped preset: YAML -> args -> MSATransformerConfig.

    The stubbed tests above pin the patch's dispatch; this one pins that the
    preset's own key spellings survive MegatronArgBuilder (which drops keys
    Megatron's argparse does not know) and land on the upstream fields.
    """
    import sys
    from types import SimpleNamespace

    import torch

    from primus.backends.megatron.argument_builder import MegatronArgBuilder
    from primus.core.config.preset_loader import PresetLoader

    preset = PresetLoader.load("minimax_m3.yaml", "megatron", "models")

    builder = MegatronArgBuilder()
    builder.update(preset)
    args = SimpleNamespace(**vars(builder.to_namespace()))
    # What `merge_namespace(backend_args, params, allow_override=False)` does in
    # train_runtime: keys Megatron never declared come back as raw YAML values.
    for key, value in preset.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.params_dtype = torch.bfloat16
    # The preset assumes TP > 1; this test builds the config on one rank.
    args.sequence_parallel = False

    import megatron.training.arguments as arguments_module

    monkeypatch.setattr(arguments_module, "_primus_applied_patch_keys", set(), raising=False)
    monkeypatch.setattr(
        arguments_module,
        "core_transformer_config_from_args",
        arguments_module.core_transformer_config_from_args,
    )
    monkeypatch.setitem(sys.modules, "_minimax_m3_preset_importer", SimpleNamespace())
    patch_mod.patch_minimax_m3_config(ctx=None)

    config = arguments_module.core_transformer_config_from_args(args)

    assert isinstance(config, MSATransformerConfig)
    # swigluoai
    assert config.activation_func is quick_gelu
    assert config.gated_linear_unit is True
    assert config.glu_linear_offset == 1.0
    assert config.activation_func_clamp_value == 7.0
    # use_gemma_norm
    assert config.normalization == "RMSNorm"
    assert config.layernorm_zero_centered_gamma is True
    # MSA
    assert config.sparse_layer_pattern == (0,) * 3 + (1,) * 57
    assert config.sparse_topk_blocks == 16


def test_bias_activation_fusion_is_rejected():
    """mlp.py's fused branch covers only gelu and swiglu; quick_gelu without a
    per-token scale falls through to a ValueError at the first forward."""
    with pytest.raises(ValueError, match="bias_activation_fusion"):
        _config(bias_activation_fusion=True)


def test_swigluoai_matches_the_reference_implementation():
    """Numerical parity against transformers' MiniMaxM3VLDenseMLP.forward /
    MiniMaxM3VLExperts._apply_gate -- the official M3 implementation."""
    import torch

    config = _config()
    gate_up = torch.randn(16, 2 * 32)

    # Reference: transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=config.swiglu_limit)
    up = up.clamp(min=-config.swiglu_limit, max=config.swiglu_limit)
    reference = (up + 1.0) * gate * torch.sigmoid(gate * config.swiglu_alpha)

    # Megatron: the glu() closure in mlp.py / moe/experts.py, fed the fields
    # MSATransformerConfig derives.
    x_glu, x_linear = torch.chunk(gate_up, 2, dim=-1)
    clamp = config.activation_func_clamp_value
    x_glu = x_glu.clamp(min=None, max=clamp)
    x_linear = x_linear.clamp(min=-clamp, max=clamp)
    megatron = config.activation_func(x_glu) * (x_linear + config.glu_linear_offset)

    torch.testing.assert_close(megatron, reference)


def test_gemma_norm_matches_the_reference_implementation():
    """Numerical parity against transformers' MiniMaxM3VLRMSNorm.forward."""
    import torch

    te = pytest.importorskip("transformer_engine.pytorch")
    if not torch.cuda.is_available():
        pytest.skip("TE RMSNorm needs a GPU")

    dim, eps = 64, 1.0e-6
    norm = te.RMSNorm(dim, eps=eps, zero_centered_gamma=True).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(dim, device="cuda") * 0.1)
    x = torch.randn(8, dim, device="cuda")

    # Reference: weight inits at zeros and is applied as (1 + w) in fp32.
    x_fp32 = x.float()
    reference = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    reference = (reference * (1.0 + norm.weight.float())).type_as(x)

    torch.testing.assert_close(norm(x), reference, rtol=1e-5, atol=1e-5)
