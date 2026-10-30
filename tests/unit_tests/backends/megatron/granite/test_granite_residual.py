###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the IBM Granite port: scaled residual add, layer spec / layer wrapping,
config class, converter fields and builder argument checks. The SFT checkpoint hook's
tests are in test_granite_sft_hook.py, which needs neither torch nor Megatron.

Tests that need ``megatron.training`` or Megatron parallel state skip without a GPU (the
ROCm image probes the device at import); run this file on a GPU node for full coverage."""

import ast
import dataclasses
import importlib.util
import inspect
import pathlib
import re
import sys
import types
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("megatron")

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

from primus.backends.megatron.checkpoint import native_loader_granite as conv
from primus.backends.megatron.core.models.granite.granite_layer_specs import (
    granite_layer_spec_from,
)
from primus.backends.megatron.core.models.granite.granite_transformer_config import (
    GraniteTransformerConfig,
)
from primus.backends.megatron.core.transformer.granite import (
    GraniteTransformerLayer,
    scaled_bias_dropout_add,
)

GRANITE_HF = {
    "architectures": ["GraniteForCausalLM"],
    "attention_bias": False,
    "attention_multiplier": 0.015625,
    "embedding_multiplier": 12.0,
    "hidden_size": 2560,
    "intermediate_size": 8192,
    "logits_scaling": 10.0,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "granite",
    "num_attention_heads": 40,
    "num_hidden_layers": 40,
    "num_key_value_heads": 8,
    "residual_multiplier": 0.22,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 10000000,
    "tie_word_embeddings": True,
    "vocab_size": 100352,
}


def _reference(x, bias, residual, multiplier):
    return residual + multiplier * (x + (bias if bias is not None else 0.0))


# Megatron's unfused bda runs in place when training=False and nothing needs grad,
# so every call gets its own copies of the inputs.
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_scaled_bda_matches_granite_residual(training, with_bias, fused):
    torch.manual_seed(0)
    x = torch.randn(4, 2, 8)
    residual = torch.randn(4, 2, 8)
    bias = torch.randn(8) if with_bias else None
    expected = _reference(x, bias, residual, 0.22)
    fn = scaled_bias_dropout_add(get_bias_dropout_add, 0.22)(training, fused)
    out = fn((x.clone(), None if bias is None else bias.clone()), residual.clone(), 0.0)
    torch.testing.assert_close(out, expected)


def test_scaled_bda_does_not_modify_inputs():
    x, residual = torch.randn(3, 1, 4), torch.randn(3, 1, 4)
    x0, r0 = x.clone(), residual.clone()
    scaled_bias_dropout_add(get_bias_dropout_add, 0.22)(False, False)((x, None), residual, 0.0)
    torch.testing.assert_close(x, x0)
    torch.testing.assert_close(residual, r0)


@pytest.mark.parametrize("fused", [False, True])
def test_scaled_bda_unit_multiplier_is_identity(fused):
    torch.manual_seed(1)
    x, residual = torch.randn(3, 1, 4), torch.randn(3, 1, 4)
    base = get_bias_dropout_add(False, fused)((x.clone(), None), residual.clone(), 0.0)
    scaled = scaled_bias_dropout_add(get_bias_dropout_add, 1.0)(False, fused)(
        (x.clone(), None), residual.clone(), 0.0
    )
    torch.testing.assert_close(base, scaled)


def test_scaled_bda_dropout_scales_branch_only():
    """With dropout, the kept elements are residual + multiplier * x / (1 - p)."""
    torch.manual_seed(2)
    x = torch.randn(64, 2, 16).abs() + 1.0
    residual = torch.randn(64, 2, 16)
    p = 0.5
    out = scaled_bias_dropout_add(get_bias_dropout_add, 0.22)(True, False)(
        (x.clone(), None), residual.clone(), p
    )
    branch = out - residual
    kept = branch != 0
    assert 0 < kept.sum() < kept.numel()
    torch.testing.assert_close(branch[kept], (0.22 * x / (1 - p))[kept])


def test_scaled_bda_gradient_is_scaled():
    x = torch.randn(3, 1, 4, requires_grad=True)
    residual = torch.randn(3, 1, 4, requires_grad=True)
    out = scaled_bias_dropout_add(get_bias_dropout_add, 0.22)(True, False)((x, None), residual, 0.0)
    out.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 0.22))
    torch.testing.assert_close(residual.grad, torch.ones_like(residual))


def test_scaled_bda_wraps_other_factories_generically():
    # A factory other than upstream get_bias_dropout_add is wrapped (scale, then call it).
    seen = []

    def factory(training, fused):
        def fn(x_with_bias, residual, prob):
            seen.append((training, fused))
            x, bias = x_with_bias
            return residual + x + (0.0 if bias is None else bias)

        return fn

    x, bias, residual = torch.randn(3, 1, 4), torch.randn(4), torch.randn(3, 1, 4)
    wrapped = scaled_bias_dropout_add(factory, 0.22)
    assert wrapped.residual_multiplier == pytest.approx(0.22)
    out = wrapped(True, True)((x, bias), residual, 0.0)
    torch.testing.assert_close(out, _reference(x, bias, residual, 0.22))
    assert seen == [(True, True)]


# ---------------------------------------------------------------------------
# Layer spec and layer
# ---------------------------------------------------------------------------
def _local_spec():
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    return get_gpt_layer_local_spec(normalization="RMSNorm")


def test_layer_spec_uses_granite_layer():
    spec = granite_layer_spec_from(_local_spec())
    assert spec.module is GraniteTransformerLayer
    base = _local_spec()
    assert spec.submodules is base.submodules or spec.submodules == base.submodules
    assert granite_layer_spec_from(spec) is spec  # idempotent


def test_layer_spec_rejects_non_transformer_layer():
    from megatron.core.transformer.spec_utils import ModuleSpec

    with pytest.raises(TypeError):
        granite_layer_spec_from(ModuleSpec(module=torch.nn.Identity))


@pytest.fixture
def parallel_state_tp1(request):
    """Single-rank Megatron parallel state from the shared conftest ``init_parallel_state``
    fixture (ephemeral tcp:// port, TP=PP=1, RNG tracker; it destroys what it creates).

    That fixture imports ``megatron.training``, which needs a GPU in the ROCm image
    (primus_turbo / fla probe the device at import), so these tests skip without one.
    """
    if not torch.cuda.is_available():
        pytest.skip("Megatron parallel state (megatron.training) needs a GPU")
    request.getfixturevalue("init_parallel_state")
    yield


def _cfg(**kw):
    base = dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=2,
        kv_channels=64,
        use_cpu_initialization=True,
        # Neither 1/64 (MuP 1/d fallback) nor 1/8 (1/sqrt(d)): the tests can tell them apart.
        attention_multiplier=0.05,
        residual_multiplier=1.0,
    )
    base.update(kw)
    return GraniteTransformerConfig(**base)


def _layer(multiplier, fused_bda=False, **overrides):
    kw = dict(
        normalization="RMSNorm",
        gated_linear_unit=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        residual_multiplier=multiplier,
        bias_dropout_fusion=fused_bda,
    )
    kw.update(overrides)
    return GraniteTransformerLayer(
        _cfg(**kw), granite_layer_spec_from(_local_spec()).submodules, layer_number=1
    )


@pytest.mark.parametrize("name", ["self_attn_bda", "mlp_bda"])
def test_layer_refuses_identity_bda(parallel_state_tp1, name):
    # A spec without attention or MLP has an IdentityFuncOp module bda: not a Granite
    # layer. Refused with a clear error (not a TypeError on module assignment).

    from megatron.core.transformer.identity_op import IdentityFuncOp

    sub = dataclasses.replace(granite_layer_spec_from(_local_spec()).submodules, **{name: IdentityFuncOp})
    config = _cfg(residual_multiplier=0.22, normalization="RMSNorm", gated_linear_unit=True)
    with pytest.raises(ValueError, match=f"{name} is a module \\(IdentityFuncOp\\)"):
        GraniteTransformerLayer(config, sub, layer_number=1)
    # Not a Granite layer at multiplier 1.0 either.
    plain = _cfg(residual_multiplier=1.0, normalization="RMSNorm", gated_linear_unit=True)
    with pytest.raises(ValueError, match=f"{name} is a module \\(IdentityFuncOp\\)"):
        GraniteTransformerLayer(plain, sub, layer_number=1)


@pytest.mark.parametrize("value", [None, True, float("nan"), -0.5, 0.0, "0.5"])
def test_layer_rejects_bad_residual_multiplier(parallel_state_tp1, value):
    # Bypass GraniteTransformerConfig's own check: the layer must check too.
    config = _cfg(residual_multiplier=0.22, normalization="RMSNorm", gated_linear_unit=True)
    config.residual_multiplier = value
    with pytest.raises(ValueError, match="residual_multiplier"):
        GraniteTransformerLayer(config, granite_layer_spec_from(_local_spec()).submodules, layer_number=1)


def test_layer_rejects_plain_transformer_config(parallel_state_tp1):
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=2, hidden_size=128, num_attention_heads=2, use_cpu_initialization=True
    )
    with pytest.raises(TypeError, match="residual_multiplier"):
        GraniteTransformerLayer(config, granite_layer_spec_from(_local_spec()).submodules, layer_number=1)


@pytest.mark.parametrize("multiplier", [0.22, 1.0])
def test_layer_wraps_both_bda(parallel_state_tp1, multiplier):
    layer = _layer(multiplier)
    assert layer.residual_multiplier == pytest.approx(multiplier)
    for name in ("self_attn_bda", "mlp_bda"):
        wrapped = getattr(getattr(layer, name), "residual_multiplier", None)
        if multiplier == 1.0:
            assert wrapped is None, f"{name} is wrapped at multiplier 1.0"
        else:
            assert wrapped == pytest.approx(multiplier), f"{name} is not scaled"


def test_layer_rejects_fused_tp_inference(parallel_state_tp1):
    # Upstream TransformerLayer skips both bda calls on this path; the scale would be lost.
    # The config refuses the flag itself; this covers a config changed after construction.
    layer = _layer(0.22)
    layer.config.inference_fuse_tp_communication = True
    # A config-only check: refused before the attention and MLP modules are built.
    from megatron.core.transformer.transformer_layer import TransformerLayer

    def _no_build(*_a, **_k):
        pytest.fail("TransformerLayer.__init__ ran before the config check")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TransformerLayer, "__init__", _no_build)
        with pytest.raises(ValueError, match="inference_fuse_tp_communication"):
            GraniteTransformerLayer(
                layer.config, granite_layer_spec_from(_local_spec()).submodules, layer_number=1
            )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Megatron's local DotProductAttention allocates its buffer on the current GPU",
)
@pytest.mark.parametrize("fused_bda", [False, True])
def test_layer_forward_matches_reference(parallel_state_tp1, fused_bda):
    """Granite layer, with RoPE, against the residual formula of HF GraniteDecoderLayer::

    h1  = h  + m * attn(n1(h))
    out = h1 + m * mlp(n2(h1))

    attn / mlp / norms are the layer's own submodules, so this checks the residual
    wiring (both scaled adds, unfused and jit-fused bda), not the attention kernels.
    No HF model is used here; parity with HF is covered by the logprob-parity gate.
    The counterfactual is a real unscaled layer (residual_multiplier 1.0) with the same
    weights, which must give a different output.
    """
    from megatron.core.models.common.embeddings.rotary_pos_embedding import (
        RotaryEmbedding,
    )

    torch.manual_seed(3)
    layer = _layer(0.22, fused_bda=fused_bda).eval().cuda()
    # The core attention holds the explicit attention_multiplier, not 1/sqrt(d) or MuP's 1/d.
    scale = layer.self_attention.core_attention.softmax_scale
    assert scale == pytest.approx(0.05) and scale != pytest.approx(1 / 8) and scale != pytest.approx(1 / 64)
    s, b, h = 5, 2, 128
    hidden = torch.randn(s, b, h, device="cuda")
    mask = torch.triu(torch.ones(s, s, dtype=torch.bool, device="cuda"), diagonal=1).view(1, 1, s, s)
    rope = RotaryEmbedding(kv_channels=64, rotary_percent=1.0, rotary_base=10000000)(s).cuda()
    with torch.no_grad():
        out = layer(hidden.clone(), attention_mask=mask, rotary_pos_emb=rope)
        out = out[0] if isinstance(out, tuple) else out
        attn, attn_bias = layer.self_attention(
            layer.input_layernorm(hidden), attention_mask=mask, rotary_pos_emb=rope
        )
        assert attn_bias is None
        h1 = hidden + 0.22 * attn
        mlp, mlp_bias = layer.mlp(layer.pre_mlp_layernorm(h1))
        assert mlp_bias is None
        expected = h1 + 0.22 * mlp
        unscaled = _layer(1.0, fused_bda=fused_bda).eval().cuda()
        unscaled.load_state_dict(layer.state_dict())
        plain = unscaled(hidden.clone(), attention_mask=mask, rotary_pos_emb=rope)
        plain = plain[0] if isinstance(plain, tuple) else plain
        # Unscaled reference with the same weights: h1' = h + attn, out' = h1' + mlp(n2(h1')).
        h1_plain = hidden + attn
        expected_plain = h1_plain + layer.mlp(layer.pre_mlp_layernorm(h1_plain))[0]
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(plain, expected_plain, rtol=1e-5, atol=1e-5)
    # The scale matters: the same weights without it give a clearly different output.
    assert (out - plain).abs().max() > 1e-2


# ---------------------------------------------------------------------------
# Fused residual + RMSNorm guard (Primus Turbo)
# ---------------------------------------------------------------------------
class _FakeTurboRMSNorm(torch.nn.Module):
    """Stand-in for PrimusTurboRMSNorm (a Module, so it can replace a real layer's norm)."""


@pytest.fixture
def fused_rmsnorm_guard(monkeypatch):
    """``fused_residual_rmsnorm._can_fuse`` with a stub ``PrimusTurboRMSNorm`` (the real
    primus_turbo import needs a GPU), plus a factory for a layer that may fuse."""

    from primus.backends.megatron.core.extensions import fused_residual_rmsnorm as frr

    stub = types.ModuleType("primus.backends.megatron.core.extensions.primus_turbo")
    stub.PrimusTurboRMSNorm = _FakeTurboRMSNorm
    monkeypatch.setitem(sys.modules, stub.__name__, stub)

    def make(**attrs):
        cfg = SimpleNamespace(fp32_residual_connection=False, inference_fuse_tp_communication=False)
        layer = SimpleNamespace(
            config=cfg, pre_mlp_layernorm=_FakeTurboRMSNorm(), cross_attention=None, hidden_dropout=0.0
        )
        for key, value in attrs.items():
            setattr(layer, key, value)
        return layer

    return frr._can_fuse, make


def test_fused_residual_rmsnorm_skips_scaled_residual(fused_rmsnorm_guard):
    can_fuse, make = fused_rmsnorm_guard
    assert can_fuse(make())  # baseline: a stock layer may fuse
    assert can_fuse(make(residual_multiplier=1.0))  # Granite does not wrap at exactly 1.0
    assert not can_fuse(make(residual_multiplier=0.22))
    layer = make()
    layer.config.residual_multiplier = 0.22
    assert not can_fuse(layer)
    # Non-numeric value: refuse fusion instead of raising from forward.
    assert not can_fuse(make(residual_multiplier="abc"))


def test_fused_residual_rmsnorm_guard_is_not_stale(fused_rmsnorm_guard):
    # The guard re-reads the layer on every call: a later change is seen.
    can_fuse, make = fused_rmsnorm_guard
    layer = make(residual_multiplier=0.22)
    assert not can_fuse(layer)
    layer.residual_multiplier = 1.0
    assert can_fuse(layer)
    layer.self_attn_bda = scaled_bias_dropout_add(get_bias_dropout_add, 0.5)
    assert not can_fuse(layer)


def test_fused_residual_rmsnorm_guard_logs_once(fused_rmsnorm_guard, monkeypatch):
    # The fallback is logged once per process (as the module's other path logs), not per
    # layer or per forward.

    from primus.backends.megatron.core.extensions import fused_residual_rmsnorm as frr

    can_fuse, make = fused_rmsnorm_guard
    logged = []
    monkeypatch.setattr(frr, "_log", logged.append)
    monkeypatch.setattr(frr, "_LOGGED_SCALED_RESIDUAL", False)
    layers = [make(residual_multiplier=0.22), make(residual_multiplier=0.5)]
    for layer in layers * 3:
        assert not can_fuse(layer)
    assert len(logged) == 1 and "unfused forward" in logged[0], logged


def test_fused_residual_rmsnorm_guard_refuses_bool_multiplier(fused_rmsnorm_guard):
    # True is not a multiplier (float(True) == 1.0 would wrongly allow fusion).
    can_fuse, make = fused_rmsnorm_guard
    assert not can_fuse(make(residual_multiplier=True))


def test_fused_residual_rmsnorm_skips_wrapped_bda(fused_rmsnorm_guard):
    # A bda wrapped by scaled_bias_dropout_add is detected even without the attribute.
    can_fuse, make = fused_rmsnorm_guard
    wrapped = scaled_bias_dropout_add(get_bias_dropout_add, 0.5)
    assert not can_fuse(make(self_attn_bda=wrapped))
    assert not can_fuse(make(mlp_bda=wrapped))
    assert can_fuse(make(self_attn_bda=get_bias_dropout_add, mlp_bda=get_bias_dropout_add))


def test_fused_residual_rmsnorm_skips_real_granite_layer(parallel_state_tp1, fused_rmsnorm_guard):
    can_fuse, _ = fused_rmsnorm_guard
    layer = _layer(0.22)
    layer.pre_mlp_layernorm = _FakeTurboRMSNorm()  # otherwise fusable
    assert not can_fuse(layer)
    plain = _layer(1.0)
    plain.pre_mlp_layernorm = _FakeTurboRMSNorm()
    assert can_fuse(plain)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("attention_multiplier", [1 / 64, 0.02])
def test_config_mup_matches_granite_multipliers(attention_multiplier):
    # kv_channels=128: MuP's own fallback would be 1/128, so only the explicit
    # attention_multiplier gives the expected scale (kv_channels=64 cannot tell them apart).
    cfg = _cfg(
        kv_channels=128,
        use_mup=True,
        mup_embedding_mult=12.0,
        mup_output_mult=0.1,
        residual_multiplier=0.22,
        attention_multiplier=attention_multiplier,
    )
    assert cfg.residual_multiplier == pytest.approx(0.22)
    assert cfg.softmax_scale == pytest.approx(attention_multiplier)
    assert cfg.mup_width_mult == 1.0  # no init / LR / logits rescaling
    assert cfg.mup_output_mult == pytest.approx(0.1)


def test_config_requires_attention_multiplier():
    # Without it MuP would silently use 1/kv_channels, which matches neither HF nor the converter.
    with pytest.raises(ValueError, match="attention_multiplier"):
        _cfg(kv_channels=128, use_mup=True, attention_multiplier=None)


def test_config_requires_residual_multiplier():
    with pytest.raises(ValueError, match="residual_multiplier"):
        _cfg(residual_multiplier=None)


def test_config_rejects_conflicting_softmax_scale():
    with pytest.raises(ValueError, match="softmax_scale"):
        _cfg(softmax_scale=0.125, attention_multiplier=1 / 64)


@pytest.mark.parametrize("base", [64, 256])
def test_config_rejects_mup_width_scaling(base):
    with pytest.raises(ValueError, match="mup_base_hidden_size"):
        _cfg(use_mup=True, mup_base_hidden_size=base)


@pytest.mark.parametrize("mults", [{"mup_embedding_mult": 12.0}, {"mup_output_mult": 0.1}])
def test_config_requires_use_mup_for_mup_multipliers(mults):
    # Upstream applies both multipliers only with use_mup; without it they would be dropped.
    with pytest.raises(ValueError, match="use_mup"):
        _cfg(use_mup=False, **mults)


def test_config_accepts_use_mup_false_without_mup_multipliers():
    assert _cfg(use_mup=False).use_mup is False


def test_config_accepts_base_width_equal_to_hidden_size():
    assert _cfg(use_mup=True, mup_base_hidden_size=128).mup_width_mult == 1.0


def _preset():
    """The Granite model preset merged over its ``extends`` chain (plain YAML)."""

    root = __import__("pathlib").Path(__file__).resolve().parents[5] / "primus/configs/models/megatron"

    def load(name):
        data = yaml.safe_load((root / name).read_text()) or {}
        merged = {}
        for parent in data.pop("extends", []) or []:
            merged.update(load(parent))
        merged.update(data)
        return merged

    return load("granite_4_1_3b.yaml")


def _config_from(values, **overrides):
    """GraniteTransformerConfig from a flat arg dict, the way core_transformer_config_from_args
    picks dataclass fields by name (enough for the scalar fields checked here)."""

    names = {f.name for f in dataclasses.fields(GraniteTransformerConfig)}
    kw = {k: v for k, v in values.items() if k in names}
    kw.update(num_layers=2, use_cpu_initialization=True, apply_rope_fusion=False, sequence_parallel=False)
    kw.update(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    kw.update(overrides)
    return GraniteTransformerConfig(**kw)


def test_preset_builds_granite_config():
    preset = _preset()
    assert preset["model_type"] == "granite"
    cfg = _config_from(preset)
    assert cfg.softmax_scale == pytest.approx(1 / 64)
    assert cfg.residual_multiplier == pytest.approx(0.22)
    assert cfg.mup_embedding_mult == 12.0 and cfg.mup_output_mult == pytest.approx(0.1)
    assert cfg.mup_width_mult == 1.0
    # The preset's attention scale is explicit: still 1/64 with another head dim.
    assert _config_from(preset, kv_channels=128).softmax_scale == pytest.approx(1 / 64)


def test_preset_without_attention_multiplier_is_refused():
    preset = {k: v for k, v in _preset().items() if k != "attention_multiplier"}
    with pytest.raises(ValueError, match="attention_multiplier"):
        _config_from(preset)


def test_converter_fields_build_granite_config():
    # converter -> GraniteTransformerConfig: the link the trained/converted model depends on.
    f = conv.derive_megatron_fields(GRANITE_HF)
    cfg = _config_from(f)
    assert cfg.softmax_scale == pytest.approx(1 / 64)
    assert cfg.residual_multiplier == pytest.approx(0.22)
    assert cfg.mup_embedding_mult == 12.0 and cfg.mup_output_mult == pytest.approx(0.1)
    assert cfg.mup_width_mult == 1.0
    # HF-default checkpoint: attention_multiplier 1.0 is used as is, not MuP's 1/kv_channels.
    plain = {k: v for k, v in GRANITE_HF.items() if not k.endswith("multiplier") and k != "logits_scaling"}
    assert _config_from(conv.derive_megatron_fields(plain)).softmax_scale == 1.0


@pytest.mark.parametrize("key", ["residual_multiplier", "attention_multiplier"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan"), True, "0.5", "x", None])
def test_config_rejects_bad_multipliers(key, value):
    with pytest.raises(ValueError, match=key):
        _cfg(**{key: value})


def test_config_rejects_fused_tp_inference_with_scaled_residual():
    with pytest.raises(ValueError, match="inference_fuse_tp_communication"):
        _cfg(residual_multiplier=0.22, inference_fuse_tp_communication=True)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------
def test_converter_fields():
    f = conv.derive_megatron_fields(GRANITE_HF)
    assert f["untie_embeddings_and_output_weights"] is False
    assert f["kv_channels"] == 64 and f["num_query_groups"] == 8
    assert f["rotary_base"] == 10000000
    assert f["use_mup"] is True
    assert f["mup_embedding_mult"] == 12.0
    assert f["mup_output_mult"] == pytest.approx(0.1)
    assert f["mup_base_head_dim"] is None
    assert f["mup_base_hidden_size"] is None
    assert not f.get("add_bias_linear") and not f.get("add_qkv_bias")
    assert f["attention_multiplier"] == pytest.approx(1 / 64)
    assert f["residual_multiplier"] == pytest.approx(0.22)


def test_converter_missing_multipliers_use_hf_defaults():
    cfg = {k: v for k, v in GRANITE_HF.items() if not k.endswith("multiplier") and k != "logits_scaling"}
    f = conv.derive_megatron_fields(cfg)
    # transformers GraniteConfig defaults are all 1.0 (attention_multiplier too).
    assert f["attention_multiplier"] == 1.0
    assert f["mup_embedding_mult"] == 1.0
    assert f["mup_output_mult"] == 1.0
    assert f["residual_multiplier"] == 1.0


@pytest.mark.parametrize(
    "key", ["attention_multiplier", "embedding_multiplier", "logits_scaling", "residual_multiplier"]
)
@pytest.mark.parametrize("value", [None, 0, -1.0, "x", "0.5", True, float("inf")])
def test_converter_rejects_bad_multipliers(key, value):
    with pytest.raises(ValueError, match=key):
        conv.derive_megatron_fields({**GRANITE_HF, key: value})


@pytest.mark.parametrize(
    "rope",
    [None, {}, {"rope_type": "default"}, {"type": "default"}, {"rope_type": None, "type": "default"}],
)
def test_converter_accepts_plain_rope(rope):
    f = conv.derive_megatron_fields({**GRANITE_HF, "rope_scaling": rope})
    assert f["rotary_base"] == 10000000


@pytest.mark.parametrize("key", ["rope_parameters", "rope_scaling"])
@pytest.mark.parametrize("top_level", ["missing", None, 5000000])
def test_converter_reads_rope_theta_from_rope_dict(key, top_level):
    cfg = {k: v for k, v in GRANITE_HF.items() if k != "rope_theta"}
    if top_level != "missing":
        cfg["rope_theta"] = top_level
    cfg[key] = {"rope_type": "default", "rope_theta": 5000000}
    assert conv.derive_megatron_fields(cfg)["rotary_base"] == 5000000


def test_converter_rope_theta_default_and_errors():
    cfg = {k: v for k, v in GRANITE_HF.items() if k != "rope_theta"}
    assert conv.derive_megatron_fields(cfg)["rotary_base"] == 10000.0  # HF GraniteConfig default
    with pytest.raises(ValueError, match="rope_theta is null"):
        conv.derive_megatron_fields({**cfg, "rope_theta": None})
    with pytest.raises(ValueError, match="conflicting"):
        conv.derive_megatron_fields({**GRANITE_HF, "rope_parameters": {"rope_theta": 5000000}})
    for bad in ("1e7", 0, -1.0, True):
        with pytest.raises(ValueError):
            conv.derive_megatron_fields({**cfg, "rope_theta": bad})


@pytest.mark.parametrize("key", ["attention_bias", "mlp_bias"])
def test_converter_rejects_biases_early(key):
    with pytest.raises(NotImplementedError, match=key):
        conv.derive_megatron_fields({**GRANITE_HF, key: True})
    assert "add_bias_linear" not in conv.GRANITE_KEYS


@pytest.mark.parametrize("key", ["attention_bias", "mlp_bias"])
@pytest.mark.parametrize("value", ["false", "true", None, 0, 1])
def test_converter_requires_bool_biases(key, value):
    # Strict bool, like tie_word_embeddings: no truthiness ("false" is not False).
    with pytest.raises(ValueError, match=f"{key}=.* must be a bool"):
        conv.derive_megatron_fields({**GRANITE_HF, key: value})


@pytest.mark.parametrize("value", ["missing", None])
def test_converter_null_kv_heads_is_mha(value):
    # HF GraniteConfig: num_key_value_heads absent or None -> num_attention_heads (MHA).
    cfg = {**GRANITE_HF}
    if value == "missing":
        del cfg["num_key_value_heads"]
    else:
        cfg["num_key_value_heads"] = None
    f = conv.derive_megatron_fields(cfg)
    assert f["num_query_groups"] == GRANITE_HF["num_attention_heads"]


@pytest.mark.parametrize("key", ["rope_scaling", "rope_parameters"])
@pytest.mark.parametrize(
    "rope",
    [
        {"rope_type": "linear", "factor": 2.0},
        # Each type key is checked on its own: a plain or null rope_type does not hide
        # real scaling named by the legacy "type" key.
        {"rope_type": "default", "type": "linear"},
        {"rope_type": None, "type": "yarn"},
        {"rope_type": None, "type": "dynamic"},
        {"type": "default", "rope_type": "llama3"},
        # Only null and {} mean "no scaling"; other falsy values are not RoPE configs.
        False,
        [],
        0,
        "",
    ],
)
def test_converter_rejects_rope_scaling(key, rope):
    with pytest.raises(NotImplementedError, match=key):
        conv.derive_megatron_fields({**GRANITE_HF, key: rope})


@pytest.mark.parametrize(
    "override",
    [
        {"model_type": "granitemoe", "architectures": ["GraniteMoeForCausalLM"]},
        {"model_type": "granitemoehybrid"},
        {"architectures": ["GraniteMoeHybridForCausalLM"]},
    ],
)
def test_converter_rejects_other_granite_families(override):
    with pytest.raises(ValueError, match="dense Granite"):
        conv.derive_megatron_fields({**GRANITE_HF, **override})


def _convert_args():
    return SimpleNamespace(
        hf_dir="/nonexistent",
        save_dir="/tmp/x",
        dtype="fp32",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        seq_length=128,
        make_vocab_size_divisible_by=128,
    )


def _needs_megatron_training():
    if not torch.cuda.is_available():
        pytest.skip("megatron.training (validate_args) needs a GPU in this image")


def test_converter_build_margs_builds_granite_config():
    _needs_megatron_training()
    from megatron.training.arguments import core_transformer_config_from_args

    small = {**GRANITE_HF, "num_hidden_layers": 2, "hidden_size": 256, "num_attention_heads": 2}
    small.update(num_key_value_heads=2, intermediate_size=512, head_dim=128)
    m = conv.build_margs(conv.derive_megatron_fields(small), _convert_args())
    assert m.model_type == "granite"
    assert m.add_bias_linear is False
    # Same config path as the builder; head dim 128 so MuP's 1/kv_channels would differ.
    cfg = core_transformer_config_from_args(m, config_class=GraniteTransformerConfig)
    assert cfg.kv_channels == 128
    assert cfg.softmax_scale == pytest.approx(1 / 64)
    assert cfg.residual_multiplier == pytest.approx(0.22)
    assert cfg.mup_width_mult == 1.0
    assert cfg.mup_embedding_mult == 12.0 and cfg.mup_output_mult == pytest.approx(0.1)


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"qk_layernorm": True}, "which build_margs sets"),
        ({"num_experts": 8}, "which build_margs sets"),
        ({"save": "/elsewhere"}, "which build_margs sets"),
        ({"iteration": 5}, "which build_margs sets"),
        ({"perform_initialization": True}, "which build_margs sets"),
        ({"make_vocab_size_divisible_by": 1}, "which build_margs sets"),
        ({"rotary_interleaved": True}, "not in allowed_fields"),
        ({"apply_layernorm_1p": True}, "not in allowed_fields"),
        ({"residul_multiplier": 0.2}, "not in allowed_fields"),
    ],
)
def test_build_margs_restricts_extra_fields(extra, match):
    _needs_megatron_training()
    from primus.backends.megatron.checkpoint import native_loader_llama as llama

    f = conv.derive_megatron_fields(GRANITE_HF)
    with pytest.raises(ValueError, match=match):
        llama.build_margs(f, _convert_args(), extra_fields=extra, allowed_fields=conv.GRANITE_ALLOWED_FIELDS)


def test_build_margs_refuses_owned_field_even_if_allowed():
    _needs_megatron_training()
    from primus.backends.megatron.checkpoint import native_loader_llama as llama

    f = conv.derive_megatron_fields(GRANITE_HF)
    with pytest.raises(ValueError, match="which build_margs sets"):
        llama.build_margs(f, _convert_args(), extra_fields={"iteration": 5}, allowed_fields={"iteration"})


def test_build_margs_returns_plain_namespace():
    _needs_megatron_training()
    m = conv.build_margs(conv.derive_megatron_fields(GRANITE_HF), _convert_args())
    assert type(m) is __import__("argparse").Namespace


def test_build_margs_owned_matches_assignments():
    # Runs on CPU: BUILD_MARGS_OWNED is exactly the set of m.<name> = ... in build_margs.

    from primus.backends.megatron.checkpoint import native_loader_llama as llama

    tree = ast.parse(inspect.getsource(llama.build_margs))
    assigned = {
        n.targets[0].attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Attribute)
        and isinstance(n.targets[0].value, ast.Name)
        and n.targets[0].value.id == "m"
    }
    assert assigned == set(llama.BUILD_MARGS_OWNED)


def test_build_margs_unchanged_for_llama_callers():
    # Without extra_fields the Llama/Qwen3/DeepSeek args are those of upstream build_margs:
    # bf16 sets only bf16 (fp16 keeps its argparse default) and no extra key appears.
    _needs_megatron_training()
    from primus.backends.megatron.checkpoint import native_loader_llama as llama

    llama_cfg = {k: v for k, v in GRANITE_HF.items() if "multiplier" not in k and k != "logits_scaling"}
    llama_cfg.update(architectures=["LlamaForCausalLM"], model_type="llama")
    convert_args = _convert_args()
    convert_args.dtype = "bf16"
    m = llama.build_margs(llama.derive_megatron_fields(llama_cfg), convert_args)
    assert m.bf16 is True and m.fp16 is False
    assert not hasattr(m, "residual_multiplier") and getattr(m, "model_type", None) != "granite"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
_BUILDER_BASE = dict(
    use_legacy_models=False,
    yaml_cfg=None,
    experimental_attention_variant=None,
    heterogeneous_layers_config_path=None,
    transformer_impl="transformer_engine",
    num_experts=None,
    mtp_num_layers=None,
    multi_latent_attention=False,
    use_rope_scaling=False,
    position_embedding_type="rope",
)


@pytest.mark.parametrize(
    "key, value, name",
    [
        ("yaml_cfg", "x.yaml", "yaml_cfg"),
        ("experimental_attention_variant", "dsa", "experimental_attention_variant"),
        ("heterogeneous_layers_config_path", "h.json", "heterogeneous_layers_config_path"),
        ("transformer_impl", "inference_optimized", "transformer_impl=inference_optimized"),
        ("num_experts", 8, "num_experts"),
        ("mtp_num_layers", 1, "mtp_num_layers"),
        ("use_legacy_models", True, "use_legacy_models"),
        ("multi_latent_attention", True, "multi_latent_attention"),
        ("use_rope_scaling", True, "use_rope_scaling"),
        ("position_embedding_type", "learned_absolute", "position_embedding_type!=rope"),
        ("position_embedding_type", "none", "position_embedding_type!=rope"),
    ],
)
def test_builder_rejects_unsupported_upstream_options(key, value, name):
    from primus.backends.megatron.core.models.granite.granite_args import (
        reject_unsupported_args,
    )

    reject_unsupported_args(SimpleNamespace(**_BUILDER_BASE))  # baseline passes
    reject_unsupported_args(SimpleNamespace())  # missing attributes count as unset
    with pytest.raises(ValueError, match=re.escape(name)):
        reject_unsupported_args(SimpleNamespace(**{**_BUILDER_BASE, key: value}))


def test_builder_rejects_all_unsupported_options_at_once():
    from primus.backends.megatron.core.models.granite.granite_args import (
        reject_unsupported_args,
    )

    with pytest.raises(ValueError, match="yaml_cfg, .*num_experts"):
        reject_unsupported_args(SimpleNamespace(**{**_BUILDER_BASE, "yaml_cfg": "x", "num_experts": 4}))


def test_model_provider_builds_granite_model(parallel_state_tp1, monkeypatch):
    """get_model_provider('granite') -> granite_builder -> GPTModel with Granite layers whose
    core attention holds attention_multiplier (head dim 128, so MuP's 1/128 would differ)."""
    from megatron.training import global_vars

    from primus.core.utils.import_utils import get_model_provider

    small = {**GRANITE_HF, "num_hidden_layers": 2, "hidden_size": 256, "num_attention_heads": 2}
    small.update(num_key_value_heads=2, intermediate_size=512, head_dim=128)
    m = conv.build_margs(conv.derive_megatron_fields(small), _convert_args())
    m.padded_vocab_size = 256
    m.params_dtype = torch.float32
    # Megatron's global args are process-wide: restore them after the test.
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", m)
    for var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(var, raising=False)
    model = get_model_provider(model_type="granite")(pre_process=True, post_process=True)
    layers = list(model.decoder.layers)
    assert len(layers) == 2
    assert all(isinstance(layer, GraniteTransformerLayer) for layer in layers)
    for layer in layers:
        assert layer.self_attn_bda.residual_multiplier == pytest.approx(0.22)
    # Read back from the built attention modules (TE keeps it on its backend submodules).
    from primus.backends.megatron.core.models.granite.granite_checks import (
        _core_attention_scales,
    )

    assert _core_attention_scales(layers) == pytest.approx([1 / 64, 1 / 64])
    assert model.config.mup_width_mult == 1.0


# ---------------------------------------------------------------------------
# granite_builder guards (GPTModel replaced by a fake: no GPU or parallel state needed
# for the model itself; importing the builder needs megatron.training)
# ---------------------------------------------------------------------------
class _FakeAttention:
    def __init__(self, scale):
        if scale is not None:
            self.softmax_scale = scale


class _FakeLayer:
    def __init__(self, scale):
        self.self_attention = SimpleNamespace(core_attention=_FakeAttention(scale))


def _fake_layers(case):
    """The built decoder layers for a post-build check case (shared by the builder test and
    the megatron.training-free check test, so their cases cannot drift apart)."""
    if case == "disagree":
        split = _FakeLayer(None)
        split.self_attention.core_attention.flash_attention = _FakeAttention(0.05)
        split.self_attention.core_attention.fused_attention = _FakeAttention(0.1)
        return [split]
    good = GraniteTransformerLayer.__new__(GraniteTransformerLayer)  # isinstance only
    good.self_attention = _FakeLayer(0.05).self_attention
    return {
        "no_scale": [_FakeLayer(None)],
        "wrong_scale": [_FakeLayer(1 / 8)],
        "plain_layer": [_FakeLayer(0.05)],
        "no_layers": [],
    }.get(case, [good])


# Post-build check cases shared by both tests: (case, error, match).
_CHECK_CASES = [
    ("no_scale", ValueError, "cannot read the softmax scale"),
    ("wrong_scale", ValueError, "do not use attention_multiplier"),
    ("plain_layer", TypeError, "needs GraniteTransformerLayer"),
    ("disagree", ValueError, "hold different softmax scales"),
    ("no_layers_attr", TypeError, "no 'layers' attribute"),
    ("no_layers", None, None),
    ("ok", None, None),
]


def _builder_args(spec="granite_test.spec"):
    return SimpleNamespace(
        **_BUILDER_BASE,
        spec=spec,
        padded_vocab_size=256,
        max_position_embeddings=64,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=False,
        rotary_percent=1.0,
        rotary_base=10000,
    )


@pytest.mark.parametrize(
    "case, error, match",
    [("plain_config", TypeError, "GraniteTransformerConfig")] + _CHECK_CASES,
)
def test_granite_builder_guards(monkeypatch, case, error, match):
    _needs_megatron_training()

    from primus.backends.megatron.core.models.granite import granite_builders as gb

    config = _cfg(residual_multiplier=0.22, normalization="RMSNorm", gated_linear_unit=True)
    layers = _fake_layers(case)
    built = {}

    def fake_gpt_model(**kw):
        built.update(kw)
        if case == "no_layers_attr":
            return SimpleNamespace(decoder=SimpleNamespace())
        return SimpleNamespace(decoder=SimpleNamespace(layers=layers))

    imported = []
    monkeypatch.setattr(gb, "GPTModel", fake_gpt_model)
    monkeypatch.setattr(gb, "import_module", lambda path: imported.append(path) or _local_spec())
    if case == "plain_config":
        config = SimpleNamespace(residual_multiplier=0.22)
    if error is None:
        gb.granite_builder(_builder_args(), True, True, config=config)
        # args.spec path: the user spec is imported and its layer becomes the Granite layer.
        assert imported == ["granite_test.spec"]
        assert built["transformer_layer_spec"].module is GraniteTransformerLayer
        assert built["config"] is config
    else:
        with pytest.raises(error, match=match):
            gb.granite_builder(_builder_args(), True, True, config=config)


@pytest.mark.parametrize(
    "case, error, match",
    _CHECK_CASES
    + [
        ("no_decoder", TypeError, "has no 'decoder'"),
        ("none_decoder", TypeError, "has no 'decoder'"),
    ],
)
def test_granite_build_checks_without_megatron_training(case, error, match, monkeypatch):
    """The post-build checks (granite_checks) need neither a GPU nor megatron.training,
    so they run on every CPU-only per-PR job (the builder test above may skip)."""

    from primus.backends.megatron.core.models.granite import granite_checks
    from primus.backends.megatron.core.models.granite.granite_checks import (
        check_built_granite_model,
    )

    logged = []
    monkeypatch.setattr(granite_checks, "log_rank_0", logged.append)

    config = _cfg(residual_multiplier=0.22, normalization="RMSNorm", gated_linear_unit=True)
    layers = _fake_layers(case)
    decoder = SimpleNamespace() if case == "no_layers_attr" else SimpleNamespace(layers=layers)
    model = SimpleNamespace(decoder=decoder)
    if case == "no_decoder":  # e.g. a wrapper (Float16Module / DDP) or a list of VPP chunks
        model = SimpleNamespace(module=model)
    elif case == "none_decoder":
        model = SimpleNamespace(decoder=None)
    if error is None:
        check_built_granite_model(model, config)
        # An empty stage says so and skips the scale check; a stage with layers checks it.
        empty = case == "no_layers"
        assert any("builds no decoder layers" in m for m in logged) == empty
        assert any("attention softmax_scale" in m for m in logged) == (not empty)
    else:
        with pytest.raises(error, match=match):
            check_built_granite_model(model, config)


def test_build_margs_rejects_str_allowed_fields():
    # Checked before megatron.training is imported, so this runs without a GPU.
    from primus.backends.megatron.checkpoint import native_loader_llama as llama

    with pytest.raises(TypeError, match="not a str"):
        llama.build_margs({}, _convert_args(), extra_fields={}, allowed_fields="model_type")


# ---------------------------------------------------------------------------
# SFT checkpoint hook: family detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override, match",
    [
        ({"hidden_act": "gelu"}, "hidden_act"),
        ({"num_key_value_heads": 0}, "num_key_value_heads"),
        ({"num_key_value_heads": 3}, "does not divide"),
        ({"partial_rotary_factor": 0.5}, "partial_rotary_factor"),
        ({"head_dim": 0}, "head_dim"),
        ({"text_config": {}}, "text_config"),
        ({"text_config": "x"}, "text_config"),
        ({"architectures": ["GraniteForCausalLM", "Other"]}, "architectures"),
        ({"model_type": None}, "model_type"),
    ],
)
def test_converter_refuses_non_llama_layout(override, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        conv.derive_megatron_fields({**GRANITE_HF, **override})


def test_converter_refuses_vision_wrapper():
    wrapper = {
        "architectures": ["LlavaNextForConditionalGeneration"],
        "model_type": "llava_next",
        "text_config": dict(GRANITE_HF),
    }
    with pytest.raises(NotImplementedError, match="text_config"):
        conv.derive_megatron_fields(wrapper)


# ---------------------------------------------------------------------------
# Review follow-ups (CPU-only)
# ---------------------------------------------------------------------------
def test_converter_uses_explicit_attention_multiplier_not_head_dim():
    # 0.02 differs from 1/head_dim (1/64) and from 1/sqrt(64), so only the explicit value fits.
    f = conv.derive_megatron_fields({**GRANITE_HF, "attention_multiplier": 0.02})
    assert f["attention_multiplier"] == pytest.approx(0.02)
    assert f["kv_channels"] == 64


def test_converter_rms_norm_eps_granite_default_and_errors():
    cfg = {k: v for k, v in GRANITE_HF.items() if k != "rms_norm_eps"}
    assert conv.derive_megatron_fields(cfg)["norm_epsilon"] == pytest.approx(1e-6)
    assert conv.derive_megatron_fields(GRANITE_HF)["norm_epsilon"] == pytest.approx(1e-5)
    for bad in (None, "1e-5", True, 0, -1e-5, float("inf")):
        with pytest.raises(ValueError, match="rms_norm_eps"):
            conv.derive_megatron_fields({**GRANITE_HF, "rms_norm_eps": bad})


@pytest.mark.parametrize(
    "override, match",
    [
        ({"tie_word_embeddings": 1}, "tie_word_embeddings"),
        ({"tie_word_embeddings": 0}, "tie_word_embeddings"),
        ({"partial_rotary_factor": True}, "partial_rotary_factor"),
    ],
)
def test_converter_strict_types(override, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        conv.derive_megatron_fields({**GRANITE_HF, **override})


def test_converter_explicit_head_dim_used_as_given():
    # HF GraniteAttention uses head_dim as given; so does the converter (documented).
    f = conv.derive_megatron_fields({**GRANITE_HF, "head_dim": 128})
    assert f["kv_channels"] == 128


def test_builder_rejects_query_key_layer_scaling():
    from primus.backends.megatron.core.models.granite.granite_args import (
        reject_unsupported_args,
    )

    args = SimpleNamespace(**{**_BUILDER_BASE, "apply_query_key_layer_scaling": True})
    with pytest.raises(ValueError, match="apply_query_key_layer_scaling"):
        reject_unsupported_args(args)


def test_converter_null_head_dim_is_derived():
    # head_dim: null means "not given": derive hidden/heads like HF, never kv_channels=None.
    from primus.backends.megatron.checkpoint import native_loader_granite as conv

    f = conv.derive_megatron_fields({**GRANITE_HF, "head_dim": None})
    assert f["kv_channels"] == GRANITE_HF["hidden_size"] // GRANITE_HF["num_attention_heads"]


def test_check_supported_returns_normalised_config():
    from primus.backends.megatron.checkpoint import native_loader_granite as conv

    src = {k: v for k, v in GRANITE_HF.items() if k != "residual_multiplier"}
    cfg = conv.check_supported({**src, "rope_theta": 5e5, "head_dim": None, "embedding_multiplier": 3})
    assert cfg["rope_theta"] == 5e5
    assert cfg["head_dim"] == GRANITE_HF["hidden_size"] // GRANITE_HF["num_attention_heads"]
    # The multipliers are resolved once here (missing -> HF default 1.0, ints -> float).
    assert cfg["residual_multiplier"] == 1.0
    assert cfg["embedding_multiplier"] == 3.0 and isinstance(cfg["embedding_multiplier"], float)


# ---------------------------------------------------------------------------
# SFT checkpoint hook: hf_path given as a HuggingFace Hub repo id
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SFT hook and native converters share one config.json reader (needs torch)
# ---------------------------------------------------------------------------


def _hook():
    path = (
        pathlib.Path(__file__).resolve().parents[5]
        / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"
    )
    spec = importlib.util.spec_from_file_location("granite_convert_hook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hook_and_converters_share_the_config_reader(tmp_path):
    # One config.json reader: the hook and the native converters accept/refuse the same files.
    from primus.backends.megatron.checkpoint import native_convert_common as common
    from primus.core.utils.hf_config import HFConfigError

    hook = _hook()
    for text in ("{bad", "[1, 2]"):
        (tmp_path / "config.json").write_text(text)
        with pytest.raises(HFConfigError):
            common.read_hf_config(str(tmp_path))
        with pytest.raises(hook.HFConfigError):
            hook._load_hf_config(str(tmp_path))
    (tmp_path / "config.json").write_text('{"model_type": "granite"}')
    assert (
        common.read_hf_config(str(tmp_path))
        == hook._load_hf_config(str(tmp_path))
        == {"model_type": "granite"}
    )
    assert hook._load_hf_config(str(tmp_path / "none")) is None
    with pytest.raises(FileNotFoundError):
        common.read_hf_config(str(tmp_path / "none"))


def test_granite_layer_spec_accepts_block_spec_and_refuses_custom_layers():
    pytest.importorskip("megatron.core")
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from primus.backends.megatron.core.models.granite.granite_layer_specs import (
        granite_layer_spec_from,
    )
    from primus.backends.megatron.core.transformer.granite import (
        GraniteTransformerLayer,
    )

    base = get_gpt_layer_local_spec(normalization="RMSNorm")
    assert granite_layer_spec_from(base).module is GraniteTransformerLayer
    block = TransformerBlockSubmodules(layer_specs=[base, base], layer_norm="norm")
    out = granite_layer_spec_from(block)
    assert [s.module for s in out.layer_specs] == [GraniteTransformerLayer] * 2
    assert out.layer_norm == "norm"

    class CustomLayer(TransformerLayer):
        pass

    with pytest.raises(TypeError, match="CustomLayer"):
        granite_layer_spec_from(dataclasses.replace(base, module=CustomLayer))

    class GraniteChild(GraniteTransformerLayer):
        pass

    kept = dataclasses.replace(base, module=GraniteChild)
    assert granite_layer_spec_from(kept).module is GraniteChild


# ---------------------------------------------------------------------------
# SFT hook: the install step and main() take the same path (one _route)
# ---------------------------------------------------------------------------
