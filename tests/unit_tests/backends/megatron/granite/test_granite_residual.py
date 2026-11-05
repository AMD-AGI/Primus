###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the IBM Granite port: scaled residual add, layer spec bda replacement,
config class, converter fields and builder argument checks.

Tests that need ``megatron.training`` or Megatron parallel state skip without a GPU (the
ROCm image probes the device at import); run this file on a GPU node for full coverage."""

import dataclasses
import functools
import re
import sys
import types
from types import SimpleNamespace

import pytest

from tests.unit_tests.backends.megatron.granite import _granite_common as granite_common

torch = pytest.importorskip("torch")
pytest.importorskip("megatron")

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

from primus.backends.megatron.checkpoint import native_loader_granite as conv
from primus.backends.megatron.core.fusions.scaled_bias_dropout_add import (
    scaled_bias_dropout_add,
)
from primus.backends.megatron.core.models.granite.granite_layer_specs import (
    granite_layer_spec_from,
)
from primus.backends.megatron.core.models.granite.granite_transformer_config import (
    GraniteTransformerConfig,
)

GRANITE_HF = granite_common.GRANITE_HF


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


@pytest.mark.parametrize("generic", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_scaled_bda_fp32_residual_scales_after_the_upcast(fused, generic):
    # fp32_residual_connection: a bf16 sub-block output is cast to fp32 first (as upstream
    # does), then scaled, so the scale is not rounded to bf16. The generic path (a factory
    # that is not upstream get_bias_dropout_add) casts to the residual's dtype too.
    torch.manual_seed(3)
    x = torch.randn(64, 1, 64).to(torch.bfloat16)
    residual = torch.randn(64, 1, 64)

    def other_factory(training, fused):  # not upstream's factory: takes the generic path
        return get_bias_dropout_add(training, fused)

    factory = other_factory if generic else get_bias_dropout_add
    out = scaled_bias_dropout_add(factory, 0.22)(False, fused)((x.clone(), None), residual.clone(), 0.0)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, residual + 0.22 * x.float(), rtol=0, atol=1e-6)


@pytest.mark.parametrize("bad", [0, -1.0, float("inf"), float("nan"), True, "0.22", None])
def test_scaled_bda_rejects_bad_multiplier(bad):
    with pytest.raises(ValueError, match="residual_multiplier must be a positive finite number"):
        scaled_bias_dropout_add(get_bias_dropout_add, bad)


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


def test_scaled_bda_refuses_an_already_scaled_factory():
    # Wrapping twice would scale by m_outer * m_inner while the tag reports only m_outer.
    once = scaled_bias_dropout_add(get_bias_dropout_add, 0.5)
    with pytest.raises(ValueError, match=r"already scaled \(residual_multiplier=0.5\)"):
        scaled_bias_dropout_add(once, 0.5)
    # functools.partial drops the function attribute; the tag is still found through it.
    with pytest.raises(ValueError, match="already scaled"):
        scaled_bias_dropout_add(functools.partial(once), 0.5)


def _capture_log(monkeypatch):
    """The messages scaled_bias_dropout_add logs (through Primus' log_rank_0)."""
    import importlib

    module = importlib.import_module(scaled_bias_dropout_add.__module__)
    messages = []
    monkeypatch.setattr(module, "log_rank_0", lambda msg, *a, **k: messages.append(msg))
    return messages


def test_scaled_bda_partial_layers_are_wrapped_generically(monkeypatch):
    from primus.backends.megatron.core.fusions.scaled_bias_dropout_add import (
        residual_multiplier_of,
    )

    assert (
        residual_multiplier_of(functools.partial(scaled_bias_dropout_add(get_bias_dropout_add, 0.5))) == 0.5
    )
    assert residual_multiplier_of(get_bias_dropout_add) is None
    messages = _capture_log(monkeypatch)
    # Only upstream get_bias_dropout_add itself takes the fused fast path; a partial of it
    # is called as TransformerLayer calls it, through the generic wrapper.
    factory = scaled_bias_dropout_add(functools.partial(get_bias_dropout_add), 0.22)
    assert any("generic" in m for m in messages)
    torch.manual_seed(0)
    x, residual, bias = torch.randn(4, 2, 8), torch.randn(4, 2, 8), torch.randn(8)
    out = factory(True, False)((x.clone(), bias.clone()), residual.clone(), 0.0)
    torch.testing.assert_close(out, _reference(x, bias, residual, 0.22))


@pytest.mark.parametrize(
    "bda",
    [
        functools.partial(get_bias_dropout_add, True),  # binds training positionally
        functools.partial(get_bias_dropout_add, fused=False),  # fused would be given twice
        functools.partial(get_bias_dropout_add, training=True),
        lambda training: None,  # takes one argument
    ],
)
def test_scaled_bda_refuses_a_factory_transformer_layer_cannot_call(bda):
    # TransformerLayer calls a bda factory as bda(training, fused): a factory that fails
    # that call is refused when the spec is built, not on the first forward.
    with pytest.raises(TypeError, match=r"bda_factory\(training, fused\)"):
        scaled_bias_dropout_add(bda, 0.5)


def test_scaled_bda_partial_binding_another_keyword_runs():
    # A partial that binds a keyword other than training / fused is called positionally,
    # as TransformerLayer calls it.
    calls = []

    def inner_factory(training, fused, scale=1.0):
        calls.append((training, fused, scale))
        return lambda x_with_bias, residual, prob: residual + scale * x_with_bias[0]

    x, residual = torch.ones(2, 1, 4), torch.zeros(2, 1, 4)
    fn = scaled_bias_dropout_add(functools.partial(inner_factory, scale=2.0), 0.5)(True, False)
    torch.testing.assert_close(fn((x, None), residual, 0.0), x)
    assert calls == [(True, False, 2.0)]


def test_scaled_bda_logs_the_generic_path(monkeypatch):
    messages = _capture_log(monkeypatch)
    scaled_bias_dropout_add(lambda training, fused: None, 0.5)
    assert any("generic scaled wrapper" in m for m in messages)
    messages.clear()
    scaled_bias_dropout_add(get_bias_dropout_add, 0.5)
    assert not any("generic" in m for m in messages)


# ---------------------------------------------------------------------------
# Layer spec and layer
# ---------------------------------------------------------------------------
def _local_spec():
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    return get_gpt_layer_local_spec(normalization="RMSNorm")


def _bda_tags(spec):
    return [
        getattr(getattr(spec.submodules, n), "residual_multiplier", None)
        for n in ("self_attn_bda", "mlp_bda")
    ]


def test_layer_spec_keeps_transformer_layer_and_scales_bda():
    from megatron.core.transformer.transformer_layer import TransformerLayer

    base = _local_spec()
    spec = granite_layer_spec_from(base, 0.22)
    assert spec.module is TransformerLayer  # upstream layer, no subclass
    assert _bda_tags(spec) == [pytest.approx(0.22)] * 2
    # Only the two bdas change.
    for f in dataclasses.fields(base.submodules):
        if f.name not in ("self_attn_bda", "mlp_bda"):
            assert getattr(spec.submodules, f.name) is getattr(base.submodules, f.name), f.name
    assert _bda_tags(base) == [None, None]  # the base spec is not mutated
    assert granite_layer_spec_from(spec, 0.22) == spec  # idempotent (already scaled)
    with pytest.raises(ValueError, match="already scales"):
        granite_layer_spec_from(spec, 0.5)


def test_layer_spec_unit_multiplier_keeps_upstream_bda():
    base = _local_spec()
    assert granite_layer_spec_from(base, 1.0) is base


def test_layer_spec_rejects_non_transformer_layer():
    from megatron.core.transformer.spec_utils import ModuleSpec

    with pytest.raises(TypeError):
        granite_layer_spec_from(ModuleSpec(module=torch.nn.Identity), 0.22)


@pytest.mark.parametrize("name", ["self_attn_bda", "mlp_bda"])
@pytest.mark.parametrize("multiplier", [0.22, 1.0])
def test_layer_spec_refuses_identity_bda(name, multiplier):
    # A spec without attention or MLP has an IdentityFuncOp module bda: not a Granite layer.
    from megatron.core.transformer.identity_op import IdentityFuncOp

    base = _local_spec()
    spec = dataclasses.replace(
        base, submodules=dataclasses.replace(base.submodules, **{name: IdentityFuncOp})
    )
    with pytest.raises(ValueError, match=f"{name} is"):
        granite_layer_spec_from(spec, multiplier)


@pytest.mark.parametrize("value", [None, True, float("nan"), -0.5, 0.0, "0.5"])
def test_layer_spec_rejects_bad_residual_multiplier(value):
    with pytest.raises(ValueError, match="residual_multiplier"):
        granite_layer_spec_from(_local_spec(), value)


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
        use_mup=True,
        # MuP softmax scale sqrt(10.24) / 64 = 0.05: neither 1/64 (MuP default) nor 1/8
        # (1/sqrt(d)), so the tests can tell them apart.
        mup_base_head_dim=10.24,
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
    from megatron.core.transformer.transformer_layer import TransformerLayer

    config = _cfg(**kw)
    spec = granite_layer_spec_from(_local_spec(), config.residual_multiplier)
    return TransformerLayer(config, spec.submodules, layer_number=1)


@pytest.mark.parametrize("multiplier", [0.22, 1.0])
def test_layer_wraps_both_bda(parallel_state_tp1, multiplier):
    layer = _layer(multiplier)
    for name in ("self_attn_bda", "mlp_bda"):
        wrapped = getattr(getattr(layer, name), "residual_multiplier", None)
        if multiplier == 1.0:
            assert wrapped is None, f"{name} is wrapped at multiplier 1.0"
        else:
            assert wrapped == pytest.approx(multiplier), f"{name} is not scaled"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Megatron's local DotProductAttention allocates its buffer on the current GPU",
)
@pytest.mark.parametrize("fused_bda", [False, True])
def test_layer_forward_matches_reference(parallel_state_tp1, fused_bda):
    """Upstream TransformerLayer with the Granite spec, with RoPE, against the residual
    formula of HF GraniteDecoderLayer::

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
    # The core attention holds the MuP scale from the config, not 1/sqrt(d) or MuP's default 1/d.
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


def test_fused_residual_rmsnorm_fuses_a_consistent_scaled_residual(fused_rmsnorm_guard):
    # A layer whose two bda are scaled by the same factor as its residual_multiplier fuses:
    # the fused path applies that factor itself (see test_fused_forward_matches_unfused).
    from primus.backends.megatron.core.extensions import fused_residual_rmsnorm as frr

    can_fuse, make = fused_rmsnorm_guard
    scaled = scaled_bias_dropout_add(get_bias_dropout_add, 0.22)
    assert can_fuse(make())  # baseline: a stock layer may fuse
    assert can_fuse(make(residual_multiplier=1.0))  # the Granite spec does not wrap at exactly 1.0
    layer = make(self_attn_bda=scaled, mlp_bda=scaled, residual_multiplier=0.22)
    layer.config.residual_multiplier = 0.22
    assert can_fuse(layer)
    assert frr._residual_scale(layer, layer.config) == pytest.approx(0.22)
    assert frr._residual_scale(make(), make().config) == 1.0


@pytest.mark.parametrize(
    "attrs, cfg_multiplier",
    [
        ({"residual_multiplier": 0.22}, None),  # a multiplier but unscaled bda
        ({}, 0.22),  # the same, from the config
        ({"self_attn_bda": "scaled"}, None),  # only one bda scaled
        ({"mlp_bda": "scaled"}, None),
        ({"self_attn_bda": "scaled", "mlp_bda": "scaled", "residual_multiplier": 0.5}, None),
        ({"residual_multiplier": "abc"}, None),  # not a number
        ({"residual_multiplier": True}, None),  # a bool is not a multiplier
        ({"residual_multiplier": float("nan")}, None),
    ],
)
def test_fused_residual_rmsnorm_skips_an_inconsistent_scale(fused_rmsnorm_guard, attrs, cfg_multiplier):
    can_fuse, make = fused_rmsnorm_guard
    scaled = scaled_bias_dropout_add(get_bias_dropout_add, 0.22)
    attrs = {k: (scaled if v == "scaled" else v) for k, v in attrs.items()}
    layer = make(**attrs)
    if cfg_multiplier is not None:
        layer.config.residual_multiplier = cfg_multiplier
    assert not can_fuse(layer)
    # The fc1 (TE fused LN+Linear) branch uses the same check.
    layer = _fc1_layer(make, **attrs)
    if cfg_multiplier is not None:
        layer.config.residual_multiplier = cfg_multiplier
    assert not can_fuse(layer)


def test_fused_residual_rmsnorm_untagged_bda_equals_unit_scale(fused_rmsnorm_guard):
    # An untagged bda (None) and a bda scaled by 1.0 add the same: the layer still fuses.
    can_fuse, make = fused_rmsnorm_guard
    layer = make(
        self_attn_bda=scaled_bias_dropout_add(get_bias_dropout_add, 1.0), mlp_bda=get_bias_dropout_add
    )
    assert can_fuse(layer)


def test_fused_residual_rmsnorm_scale_is_cached_per_layer(fused_rmsnorm_guard, monkeypatch):
    # The scale is fixed once the model is built: computed on the first call for a layer,
    # then read from the layer (no recomputation per forward).
    from primus.backends.megatron.core.extensions import fused_residual_rmsnorm as frr

    can_fuse, make = fused_rmsnorm_guard
    calls = []
    real = frr._compute_residual_scale
    monkeypatch.setattr(frr, "_compute_residual_scale", lambda *a: calls.append(1) or real(*a))
    scaled = scaled_bias_dropout_add(get_bias_dropout_add, 0.22)
    layer = make(self_attn_bda=scaled, mlp_bda=scaled, residual_multiplier=0.22)
    bad = make(residual_multiplier=0.22)
    for _ in range(3):
        assert can_fuse(layer)
        assert frr._residual_scale(layer, layer.config) == pytest.approx(0.22)
        assert not can_fuse(bad)
    assert len(calls) == 2  # once per layer
    keys = []
    real_key = frr._residual_scale_key
    monkeypatch.setattr(frr, "_residual_scale_key", lambda *a: keys.append(1) or real_key(*a))
    assert frr._stored_residual_scale(layer) == pytest.approx(0.22)  # what the fused forward reads
    assert len(calls) == 2 and keys == []  # a plain attribute read: no key, no recomputation
    # A changed residual_multiplier is noticed: recomputed, here inconsistent -> unfused.
    layer.residual_multiplier = 0.5
    assert not can_fuse(layer) and len(calls) == 3
    with pytest.raises(RuntimeError, match="inconsistent residual scale"):
        frr._stored_residual_scale(layer)
    # A changed bda too: consistent again with the new scale.
    rescaled = scaled_bias_dropout_add(get_bias_dropout_add, 0.5)
    layer.self_attn_bda = layer.mlp_bda = rescaled
    assert can_fuse(layer) and frr._stored_residual_scale(layer) == pytest.approx(0.5)


def test_fused_forward_checks_the_scale_before_changing_layer_state(fused_rmsnorm_guard):
    # An inconsistent scale must raise before the V2 carry is consumed and before attention
    # runs, so the fallback to the original forward sees the layer unchanged (no lost carry,
    # no second attention call).
    pytest.importorskip("megatron.core.utils")
    from primus.backends.megatron.core.extensions import fused_residual_rmsnorm as frr

    _, make = fused_rmsnorm_guard
    calls = []
    carry = ("mlp_out", "residual")
    layer = make(
        residual_multiplier=0.22,  # unscaled bda: inconsistent
        _v2_carry=carry,
        self_attention=lambda *a, **k: calls.append("attention"),
        input_layernorm=lambda x: calls.append("input_layernorm") or x,
    )
    with pytest.raises(RuntimeError, match="inconsistent residual scale"):
        frr._do_fused_forward(layer, hidden_states=torch.zeros(2, 1, 4))
    assert calls == []
    assert layer._v2_carry is carry


def test_fused_residual_rmsnorm_bda_fields_match():
    from primus.backends.megatron.core.extensions import fused_residual_rmsnorm as frr
    from primus.backends.megatron.core.fusions.scaled_bias_dropout_add import BDA_FIELDS

    assert frr._BDA_FIELDS == BDA_FIELDS


def test_fused_residual_rmsnorm_stored_scale_without_can_fuse(fused_rmsnorm_guard):
    # A path that reaches the fused forward before _can_fuse computes the scale itself.
    from primus.backends.megatron.core.extensions import fused_residual_rmsnorm as frr

    _, make = fused_rmsnorm_guard
    scaled = scaled_bias_dropout_add(get_bias_dropout_add, 0.22)
    layer = make(self_attn_bda=scaled, mlp_bda=scaled, residual_multiplier=0.22)
    assert frr._stored_residual_scale(layer) == pytest.approx(0.22)


def test_fused_residual_rmsnorm_accepts_what_positive_finite_accepts(fused_rmsnorm_guard):
    # The guard uses the same check as the Granite config (positive_finite): a numpy scalar
    # equal to the bda scale is valid.
    np = pytest.importorskip("numpy")
    can_fuse, make = fused_rmsnorm_guard
    scaled = scaled_bias_dropout_add(get_bias_dropout_add, 0.5)
    assert can_fuse(make(self_attn_bda=scaled, mlp_bda=scaled, residual_multiplier=np.float32(0.5)))
    assert can_fuse(make(residual_multiplier=np.float64(1.0)))
    assert not can_fuse(make(residual_multiplier=10**400))  # too large for a float


def test_fused_residual_rmsnorm_guard_logs_once(fused_rmsnorm_guard, monkeypatch):
    # The fallback is logged once per distinct inconsistency (the first layer that has it),
    # not per layer or per forward.

    from primus.backends.megatron.core.extensions import fused_residual_rmsnorm as frr

    can_fuse, make = fused_rmsnorm_guard
    logged = []
    monkeypatch.setattr(frr, "_log", logged.append)
    monkeypatch.setattr(frr, "_LOGGED_SCALED_RESIDUAL", set())
    # A layer that cannot fuse for another reason (cross-attention) is not reported.
    assert not can_fuse(make(residual_multiplier=0.22, cross_attention=object()))
    assert logged == []
    # Once per distinct inconsistency: layers 3 and 4 repeat one problem, layer 5 has another.
    layers = [
        make(residual_multiplier=0.22, layer_number=3),
        make(residual_multiplier=0.22, layer_number=4),
        make(residual_multiplier=0.5, layer_number=5),
    ]
    for layer in layers * 3:
        assert not can_fuse(layer)
    assert len(logged) == 2 and all("unfused forward" in m for m in logged), logged
    assert "layer 3" in logged[0] and "layer 5" in logged[1] and "0.5" in logged[1]


def _fc1_layer(make, **attrs):
    """A layer shaped like the TE dense spec: IdentityOp pre-MLP norm, RMSNorm fused into
    ``mlp.linear_fc1`` (``layer_norm_weight``). This is the branch Granite's TE spec takes."""
    from megatron.core.transformer.identity_op import IdentityOp

    fc1 = SimpleNamespace(layer_norm_weight=torch.ones(4), eps=1e-5)
    return make(pre_mlp_layernorm=IdentityOp(), mlp=SimpleNamespace(linear_fc1=fc1), **attrs)


def test_fused_residual_rmsnorm_fc1_branch(fused_rmsnorm_guard, monkeypatch):
    from primus.backends.megatron.core.extensions import fused_residual_rmsnorm as frr

    can_fuse, make = fused_rmsnorm_guard
    monkeypatch.setattr(frr, "_log", lambda msg: None)
    monkeypatch.setattr(frr, "_LOGGED_FC1_PATH", False)
    scaled = scaled_bias_dropout_add(get_bias_dropout_add, 0.22)
    assert can_fuse(_fc1_layer(make))  # baseline: the fc1 branch fuses a stock layer
    assert can_fuse(_fc1_layer(make, residual_multiplier=1.0))
    assert can_fuse(_fc1_layer(make, self_attn_bda=scaled, mlp_bda=scaled, residual_multiplier=0.22))
    # Without fc1's layer_norm_weight the layer is not fusable at all (no branch matches).
    no_ln = _fc1_layer(make)
    no_ln.mlp.linear_fc1 = SimpleNamespace()
    assert not can_fuse(no_ln)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Transformer Engine layers need a GPU")
def test_fused_residual_rmsnorm_real_granite_te_layer(parallel_state_tp1, fused_rmsnorm_guard):
    # A real Granite TE layer (upstream TE dense spec, Granite bda) has an
    # IdentityOp pre-MLP norm and RMSNorm fused into fc1, so the guard runs on the fc1 branch.
    pytest.importorskip("transformer_engine")
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from primus.backends.megatron.core.extensions import fused_residual_rmsnorm as frr

    can_fuse, _ = fused_rmsnorm_guard

    def te_layer(multiplier):
        config = _cfg(
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            residual_multiplier=multiplier,
        )
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )

        spec = granite_layer_spec_from(get_gpt_layer_with_transformer_engine_spec(), multiplier)
        return TransformerLayer(config, spec.submodules, layer_number=1).cuda()

    scaled, plain = te_layer(0.22), te_layer(1.0)
    for layer in (scaled, plain):
        assert frr._is_identity(layer.pre_mlp_layernorm)
        assert frr._fc1_fused_ln_params(layer) is not None
        assert can_fuse(layer)
    assert frr._residual_scale(scaled, scaled.config) == pytest.approx(0.22)
    assert frr._residual_scale(plain, plain.config) == 1.0


class _ResidualNorm(_FakeTurboRMSNorm):
    """The fused path's ``PrimusTurboRMSNorm(x, residual=r)`` contract,
    ``(norm(x + r), x + r)``, computed with the layer's own norm (no primus_turbo)."""

    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, x, residual=None):
        if residual is None:
            return self.norm(x)
        total = x + residual
        return self.norm(total), total


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Megatron's local DotProductAttention allocates its buffer on the current GPU",
)
@pytest.mark.parametrize("multiplier", [0.22, 1.0])
@pytest.mark.parametrize("carry", [False, True])
def test_fused_forward_matches_unfused(
    parallel_state_tp1, fused_rmsnorm_guard, monkeypatch, multiplier, carry
):
    """The fused residual+RMSNorm forward of a real Granite layer gives the unfused output:
    the attention output is scaled before the fused add, and the MLP output is scaled in
    the V2 carry (``carry``) or by mlp_bda (V1)."""
    from megatron.core.models.common.embeddings.rotary_pos_embedding import (
        RotaryEmbedding,
    )

    from primus.backends.megatron.core.extensions import fused_residual_rmsnorm as frr

    can_fuse, _ = fused_rmsnorm_guard
    torch.manual_seed(5)
    layer = _layer(multiplier).eval().cuda()
    s, b, h = 5, 2, 128
    hidden = torch.randn(s, b, h, device="cuda")
    mask = torch.triu(torch.ones(s, s, dtype=torch.bool, device="cuda"), diagonal=1).view(1, 1, s, s)
    rope = RotaryEmbedding(kv_channels=64, rotary_percent=1.0, rotary_base=10000000)(s).cuda()
    with torch.no_grad():
        expected = layer(hidden.clone(), attention_mask=mask, rotary_pos_emb=rope)
        expected = expected[0] if isinstance(expected, tuple) else expected
        layer.pre_mlp_layernorm = _ResidualNorm(layer.pre_mlp_layernorm)
        assert can_fuse(layer)
        monkeypatch.setattr(frr, "_v2_enabled", lambda: carry)
        monkeypatch.setattr(frr, "_v2_next_can_consume_carry", lambda _layer: carry)
        nxt = SimpleNamespace(_v2_carry=None)
        object.__setattr__(layer, "_v2_is_last_layer", False)
        object.__setattr__(layer, "_v2_next_layer", nxt)
        object.__setattr__(layer, "_v2_carry", None)
        out, _ = frr._do_fused_forward(layer, hidden.clone(), attention_mask=mask, rotary_pos_emb=rope)
        if carry:
            mlp_out, residual = nxt._v2_carry
            out = mlp_out + residual  # what the next layer's fused norm adds
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("base_head_dim, scale", [(None, 1 / 128), (5.12**2, 0.04)])
def test_config_mup_matches_granite_multipliers(base_head_dim, scale):
    # softmax_scale is upstream's: MuP sets sqrt(mup_base_head_dim or 1) / kv_channels.
    cfg = _cfg(
        kv_channels=128,
        use_mup=True,
        mup_embedding_mult=12.0,
        mup_output_mult=0.1,
        residual_multiplier=0.22,
        mup_base_head_dim=base_head_dim,
    )
    assert cfg.residual_multiplier == pytest.approx(0.22)
    assert cfg.softmax_scale == pytest.approx(scale)
    assert cfg.mup_width_mult == 1.0  # no init / LR / logits rescaling
    assert cfg.mup_output_mult == pytest.approx(0.1)


def test_config_has_no_attention_multiplier_field():
    # One source for the attention scale: upstream softmax_scale.
    names = {f.name for f in dataclasses.fields(GraniteTransformerConfig)}
    assert "attention_multiplier" not in names and "softmax_scale" in names


def test_config_explicit_softmax_scale_is_kept():
    # Upstream MuP only fills softmax_scale when it is None.
    assert _cfg(softmax_scale=0.125).softmax_scale == 0.125


@pytest.mark.parametrize("base", [64, 256])
def test_config_rejects_mup_width_scaling(base):
    with pytest.raises(ValueError, match="mup_base_hidden_size"):
        _cfg(use_mup=True, mup_base_hidden_size=base)


def test_config_requires_use_mup():
    # Upstream applies the embedding / logits multipliers and the 1/d scale only with use_mup.
    with pytest.raises(ValueError, match="use_mup"):
        _cfg(use_mup=False)


def test_config_accepts_base_width_equal_to_hidden_size():
    assert _cfg(use_mup=True, mup_base_hidden_size=128).mup_width_mult == 1.0


def _preset():
    """The Granite model preset resolved by Primus's own preset loader."""
    return granite_common.preset()


def _config_from(values, **overrides):
    """GraniteTransformerConfig built the runtime way: Megatron's argparse defaults, then the
    Megatron keys of ``values`` (a preset or converter field dict) plus ``residual_multiplier``
    (a Primus key, merged back by the runtime), small shapes, ``validate_args``, then the
    builder's ``core_transformer_config_from_args(args, config_class=GraniteTransformerConfig)``."""
    _needs_megatron_training()
    import argparse

    from megatron.training.arguments import (
        add_megatron_arguments,
        core_transformer_config_from_args,
        validate_args,
    )

    m = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False)).parse_args([])
    for key, value in values.items():
        if hasattr(m, key) or key == "residual_multiplier":
            setattr(m, key, value)
    small = dict(
        num_layers=2, hidden_size=256, num_attention_heads=4, num_query_groups=2, ffn_hidden_size=512
    )
    small.update(seq_length=128, max_position_embeddings=128, micro_batch_size=1, global_batch_size=1)
    small.update(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, world_size=1, rank=0)
    small.update(apply_rope_fusion=False, use_cpu_initialization=True, gradient_accumulation_fusion=False)
    small.update(group_query_attention=True, tokenizer_type="NullTokenizer", vocab_size=256)
    for key, value in {**small, **overrides}.items():
        setattr(m, key, value)
    m = validate_args(m)
    return core_transformer_config_from_args(m, config_class=GraniteTransformerConfig)


def test_preset_builds_granite_config():
    preset = _preset()
    assert preset["model_type"] == "granite"
    cfg = _config_from(preset)
    assert cfg.softmax_scale == pytest.approx(1 / 64)
    assert cfg.residual_multiplier == pytest.approx(0.22)
    assert cfg.mup_embedding_mult == 12.0 and cfg.mup_output_mult == pytest.approx(0.1)
    assert cfg.mup_width_mult == 1.0
    assert "attention_multiplier" not in preset


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
    assert _config_from(conv.derive_megatron_fields(plain)).softmax_scale == pytest.approx(1.0)
    # Any attention_multiplier and head dim: the MuP fields reproduce it.
    for mult, head_dim in ((0.02, 64), (1 / 64, 128), (0.1, 128), (0.0884, 128)):
        f = conv.derive_megatron_fields({**GRANITE_HF, "attention_multiplier": mult, "head_dim": head_dim})
        assert _config_from(f).softmax_scale == pytest.approx(mult, rel=1e-9), (mult, head_dim)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan"), True, "0.5", "x", None])
def test_config_rejects_bad_residual_multiplier(value):
    with pytest.raises(ValueError, match="residual_multiplier"):
        _cfg(residual_multiplier=value)


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
    assert "attention_multiplier" not in f  # 1/64 = 1/kv_channels: upstream MuP default
    assert f["residual_multiplier"] == pytest.approx(0.22)


def test_converter_missing_multipliers_use_hf_defaults():
    cfg = {k: v for k, v in GRANITE_HF.items() if not k.endswith("multiplier") and k != "logits_scaling"}
    f = conv.derive_megatron_fields(cfg)
    # transformers GraniteConfig defaults are all 1.0 (attention_multiplier too):
    # sqrt(mup_base_head_dim) / 64 == 1.0.
    assert f["mup_base_head_dim"] == pytest.approx(64.0**2)
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
    assert "add_bias_linear" not in conv.GRANITE_EXTRA_FIELDS


def test_converters_keep_their_explicit_options():
    # Granite's convert() has the Llama converter's keyword options, and maps weights like Llama.
    import inspect

    from primus.backends.megatron.checkpoint import native_loader_llama as llama

    for module in (llama, conv):
        params = inspect.signature(module.convert).parameters
        assert list(params)[:2] == ["hf_dir", "save_dir"]
        assert params["dtype"].default == "bf16" and params["log"].default is None
        assert "name" not in params and "build_margs" not in params
    assert conv.build_mcore_state_dict is llama.build_mcore_state_dict


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
    # Same config path as the builder; head dim 128 so MuP's default 1/kv_channels would differ.
    cfg = core_transformer_config_from_args(m, config_class=GraniteTransformerConfig)
    assert cfg.kv_channels == 128
    assert cfg.softmax_scale == pytest.approx(1 / 64)
    assert cfg.residual_multiplier == pytest.approx(0.22)
    assert cfg.mup_width_mult == 1.0
    assert cfg.mup_embedding_mult == 12.0 and cfg.mup_output_mult == pytest.approx(0.1)


def test_build_margs_returns_plain_namespace():
    _needs_megatron_training()
    m = conv.build_margs(conv.derive_megatron_fields(GRANITE_HF), _convert_args())
    assert type(m) is __import__("argparse").Namespace


def test_build_margs_unchanged_for_llama_callers():
    # The Llama converter's args get no Granite field: bf16 sets only bf16 (fp16 keeps its
    # argparse default) and no extra key appears.
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
    rotary_percent=1.0,
    rotary_interleaved=False,
    rotary_seq_len_interpolation_factor=None,
    normalization="RMSNorm",
    swiglu=True,
    add_bias_linear=False,
    add_qkv_bias=False,
    qk_layernorm=False,
    apply_layernorm_1p=False,
    apply_residual_connection_post_layernorm=False,
    qk_l2_norm=False,
    use_kitchen=False,
    softmax_scale=None,
    window_size=None,
    attention_output_gate=False,
    no_rope_freq=None,
    apply_query_key_layer_scaling=False,
    use_sink_attention=False,
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
        ("modelopt_enabled", True, "modelopt_enabled"),
        ("multi_latent_attention", True, "multi_latent_attention"),
        ("use_rope_scaling", True, "use_rope_scaling"),
        ("position_embedding_type", "learned_absolute", "position_embedding_type!=rope"),
        ("position_embedding_type", "none", "position_embedding_type!=rope"),
        # None also builds no positional encoding (no RotaryEmbedding, no learned one).
        ("position_embedding_type", None, "position_embedding_type!=rope"),
        ("rotary_percent", 0.5, "rotary_percent!=1"),
        # None (``rotary_percent: null``) is refused with the ValueError, not a TypeError.
        ("rotary_percent", None, "rotary_percent!=1"),
        # Only a real number passes: the builder forwards the raw value to GPTModel.
        ("rotary_percent", "1", "rotary_percent!=1"),
        ("rotary_percent", True, "rotary_percent!=1"),
        ("rotary_interleaved", True, "rotary_interleaved"),
        ("rotary_seq_len_interpolation_factor", 4.0, "rotary_seq_len_interpolation_factor"),
        ("normalization", "LayerNorm", "normalization!=RMSNorm"),
        ("swiglu", False, "swiglu!=True"),
        ("add_bias_linear", True, "add_bias_linear"),
        ("add_qkv_bias", True, "add_qkv_bias"),
        ("qk_layernorm", True, "qk_layernorm"),
        ("apply_layernorm_1p", True, "apply_layernorm_1p"),
        ("apply_residual_connection_post_layernorm", True, "apply_residual_connection_post_layernorm"),
        ("qk_l2_norm", True, "qk_l2_norm"),
        ("use_kitchen", True, "use_kitchen"),
        ("softmax_scale", 0.125, "softmax_scale"),
        ("window_size", (128, 0), "window_size"),
        ("attention_output_gate", True, "attention_output_gate"),
        ("no_rope_freq", 4, "no_rope_freq"),
        ("no_rope_freq", [0, 1], "no_rope_freq"),
        ("apply_query_key_layer_scaling", True, "apply_query_key_layer_scaling"),
        ("use_sink_attention", True, "use_sink_attention"),
    ],
)
def test_builder_rejects_unsupported_upstream_options(key, value, name):
    from primus.backends.megatron.core.models.granite.granite_args import (
        reject_unsupported_args,
    )

    reject_unsupported_args(SimpleNamespace(**_BUILDER_BASE))  # baseline passes
    reject_unsupported_args(SimpleNamespace(**{**_BUILDER_BASE, "mtp_num_layers": 0}))  # 0: no MTP
    with pytest.raises(ValueError, match=re.escape(name)):
        reject_unsupported_args(SimpleNamespace(**{**_BUILDER_BASE, key: value}))


def test_builder_rejects_missing_position_embedding_type():
    # Upstream's argparse default is learned_absolute, so a missing attribute is not rope.
    from primus.backends.megatron.core.models.granite.granite_args import (
        reject_unsupported_args,
    )

    base = {k: v for k, v in _BUILDER_BASE.items() if k != "position_embedding_type"}
    with pytest.raises(ValueError, match=re.escape("position_embedding_type!=rope")):
        reject_unsupported_args(SimpleNamespace(**base))


def test_builder_rejects_all_unsupported_options_at_once():
    from primus.backends.megatron.core.models.granite.granite_args import (
        reject_unsupported_args,
    )

    with pytest.raises(ValueError, match="yaml_cfg, .*num_experts"):
        reject_unsupported_args(SimpleNamespace(**{**_BUILDER_BASE, "yaml_cfg": "x", "num_experts": 4}))


def test_model_provider_builds_granite_model(parallel_state_tp1, monkeypatch):
    """get_model_provider('granite') -> upstream model_provider -> granite_builder -> GPTModel
    with upstream TransformerLayers that scale the residual add, whose core attention holds
    attention_multiplier (head dim 128, so MuP's default 1/128 would differ)."""
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
    from megatron.core.transformer.transformer_layer import TransformerLayer

    provider = get_model_provider(model_type="granite")
    model = provider(pre_process=True, post_process=True)
    layers = list(model.decoder.layers)
    assert len(layers) == 2
    assert all(type(layer) is TransformerLayer for layer in layers)
    for layer in layers:
        assert layer.self_attn_bda.residual_multiplier == pytest.approx(0.22)
        assert layer.mlp_bda.residual_multiplier == pytest.approx(0.22)
    assert model.config.softmax_scale == pytest.approx(1 / 64)
    assert model.config.mup_width_mult == 1.0
    # The converter's entry point (native_loader_granite.build_model, the Granite model
    # provider) builds the
    # same Granite model, not a plain GPT one without the scaled residual add.
    converted = conv.build_model(m)
    assert all(type(layer) is TransformerLayer for layer in converted.decoder.layers)
    for layer in converted.decoder.layers:
        assert layer.self_attn_bda.residual_multiplier == pytest.approx(0.22)
        assert layer.mlp_bda.residual_multiplier == pytest.approx(0.22)
    assert converted.config.softmax_scale == pytest.approx(1 / 64)
    # The default model_type stays "gpt": a plain GPTModel with unscaled residual adds.
    from primus.backends.megatron.checkpoint import native_convert_common as common

    plain = common.build_mcore_gpt_model(m)
    assert all(
        getattr(layer.self_attn_bda, "residual_multiplier", None) is None for layer in plain.decoder.layers
    )
    # ModelOpt would swap the Granite builder for its own: refused.
    m.modelopt_enabled = True
    with pytest.raises(ValueError, match="ModelOpt"):
        provider(pre_process=True, post_process=True)


# ---------------------------------------------------------------------------
# Model preset vs HF config.json (native_loader_granite.check_preset)
# ---------------------------------------------------------------------------
def _shipped_preset():
    """The shipped Granite model preset (resolved by Primus's preset loader) as a dict."""
    return granite_common.shipped_preset()


def test_shipped_granite_preset_matches_the_hf_config():
    assert conv.preset_mismatches(GRANITE_HF, _shipped_preset()) == []


@pytest.mark.parametrize("value", [4096, 262144, None])
def test_preset_max_position_embeddings_is_not_compared(value):
    # RoPE: max_position_embeddings only sizes the rotary cache (no learned position table),
    # and the converter builds with max(config value, seq_length); SFT runs set their own.
    assert conv.preset_mismatches(GRANITE_HF, {**_shipped_preset(), "max_position_embeddings": value}) == []


@pytest.mark.parametrize(
    "change, needle",
    [
        ({"norm_epsilon": 1e-6}, "norm_epsilon: preset 1e-06"),
        ({"rotary_base": 10000}, "rotary_base"),
        ({"mup_embedding_mult": None}, "mup_embedding_mult"),
        ({"mup_output_mult": 1.0}, "mup_output_mult"),
        ({"residual_multiplier": 0.5}, "residual_multiplier"),
        ({"residual_multiplier": None}, "residual_multiplier: not set"),
        ({"mup_base_head_dim": 4.0}, "softmax scale"),
        ({"mup_attn_scale_power": 0.5}, "softmax scale"),
        ({"untie_embeddings_and_output_weights": True}, "untie_embeddings_and_output_weights"),
        ({"num_query_groups": 40}, "num_query_groups"),
        ({"use_mup": False}, "use_mup"),
        ({"group_query_attention": None}, "group_query_attention is not true"),
        ({"group_query_attention": False}, "group_query_attention is not true"),
        ({"swiglu": False}, "swiglu"),
        ({"normalization": None}, "normalization: preset 'LayerNorm'"),
        ({"add_bias_linear": True}, "add_bias_linear"),
        ({"position_embedding_type": "learned_absolute"}, "position_embedding_type"),
        ({"rotary_percent": 0.5}, "rotary_percent"),
        ({"qk_layernorm": True}, "qk_layernorm"),
        ({"apply_layernorm_1p": True}, "apply_layernorm_1p"),
        ({"apply_residual_connection_post_layernorm": True}, "apply_residual_connection_post_layernorm"),
        ({"qk_l2_norm": True}, "qk_l2_norm"),
        ({"softmax_scale": 0.125}, "softmax_scale: preset 0.125"),
        ({"softmax_scale": 0.015625}, "softmax_scale"),  # even the HF value: one source (MuP)
        ({"window_size": [128, 0]}, "window_size"),
        ({"attention_output_gate": True}, "attention_output_gate"),
        ({"no_rope_freq": 4}, "no_rope_freq"),
        ({"apply_query_key_layer_scaling": True}, "apply_query_key_layer_scaling"),
        ({"use_sink_attention": True}, "use_sink_attention"),
        # Everything else the builder refuses (granite_args.unsupported_problems).
        ({"mup_base_hidden_size": 1280}, "mup_base_hidden_size != hidden_size"),
        ({"use_rope_scaling": True}, "use_rope_scaling"),
        ({"rotary_interleaved": True}, "rotary_interleaved"),
        ({"rotary_seq_len_interpolation_factor": 2}, "rotary_seq_len_interpolation_factor"),
        ({"num_experts": 8}, "num_experts"),
        ({"mtp_num_layers": 1}, "mtp_num_layers"),
        ({"multi_latent_attention": True}, "multi_latent_attention"),
        ({"add_qkv_bias": True}, "add_qkv_bias"),
        ({"vocab_size": 32000}, "vocab_size"),
    ],
)
def test_granite_preset_mismatch_is_reported(change, needle):
    preset = _shipped_preset()
    for key, value in change.items():
        if value is None:
            preset.pop(key, None)  # missing: takes the Megatron default (or is required)
        else:
            preset[key] = value
    bad = conv.preset_mismatches(GRANITE_HF, preset)
    assert any(needle in b for b in bad), bad
    with pytest.raises(ValueError, match="does not match the Granite checkpoint"):
        conv.check_preset(GRANITE_HF, preset)


@pytest.mark.parametrize("missing", ["hidden_size", "num_attention_heads"])
def test_granite_partial_preset_is_listed_not_raised(missing):
    # kv_channels unset and a shape key missing: the mismatch list, not a KeyError.
    preset = {k: v for k, v in _shipped_preset().items() if k not in (missing, "kv_channels")}
    bad = conv.preset_mismatches(GRANITE_HF, preset)
    assert any(f"{missing}: not set in the preset" in b for b in bad), bad
    assert any(b.startswith("kv_channels: preset None") for b in bad), bad
    with pytest.raises(ValueError, match="does not match the Granite checkpoint"):
        conv.check_preset(GRANITE_HF, preset)


def test_granite_mha_preset_without_num_query_groups_matches():
    # Dense multi-head attention (num_key_value_heads absent): Megatron builds one group per
    # head without group_query_attention, so an unset num_query_groups matches config.json.
    hf = {k: v for k, v in GRANITE_HF.items() if k != "num_key_value_heads"}
    preset = {
        k: v for k, v in _shipped_preset().items() if k not in ("num_query_groups", "group_query_attention")
    }
    assert conv.preset_mismatches(hf, preset) == []
    # The GQA checkpoint with that preset is still reported (40 groups built, 8 needed).
    assert any("group_query_attention is not true" in b for b in conv.preset_mismatches(GRANITE_HF, preset))


@pytest.mark.parametrize(
    "key, value",
    [("rotary_percent", "1"), ("residual_multiplier", "0.22"), ("group_query_attention", "false")],
)
def test_granite_preset_string_values_are_reported(key, value):
    # A quoted number or boolean (YAML string, CLI override) is refused by the builder, so
    # the hook's preset check must report it too instead of reading it through float().
    bad = conv.preset_mismatches(GRANITE_HF, {**_shipped_preset(), key: value})
    assert any(b.startswith(f"{key}: preset {value!r}") for b in bad), bad


@pytest.mark.parametrize(
    "key",
    [
        "normalization",
        "swiglu",
        "add_bias_linear",
        "add_qkv_bias",
        "qk_layernorm",
        "position_embedding_type",
        "rotary_percent",
    ],
)
def test_builder_and_converter_share_the_layout_rules(key):
    # One table (granite_args.GRANITE_LAYOUT): a layout value the converter's preset check
    # reports is the one the builder refuses, and the other way round.
    from primus.backends.megatron.core.models.granite.granite_args import (
        GRANITE_LAYOUT,
        reject_unsupported_args,
    )

    want, _ = GRANITE_LAYOUT[key]
    wrong = "x" if isinstance(want, str) else (0.5 if isinstance(want, float) else not want)
    bad = conv.preset_mismatches(GRANITE_HF, {**_shipped_preset(), key: wrong})
    assert any(b.startswith(f"{key}: preset") for b in bad), bad
    with pytest.raises(ValueError, match=key):
        reject_unsupported_args(SimpleNamespace(**{**_BUILDER_BASE, key: wrong}))


def test_granite_preset_softmax_scale_follows_attention_multiplier():
    # A checkpoint with another attention_multiplier needs mup_base_head_dim to match it.
    hf = {**GRANITE_HF, "attention_multiplier": 0.125}
    preset = {**_shipped_preset(), "mup_base_head_dim": 64.0}
    assert conv.preset_mismatches(hf, preset) == []
    assert any("softmax scale" in b for b in conv.preset_mismatches(hf, _shipped_preset()))


# ---------------------------------------------------------------------------
# granite_builder guards (GPTModel replaced by a fake: no GPU or parallel state needed
# for the model itself; importing the builder needs megatron.training)
# ---------------------------------------------------------------------------
def _builder_args(spec="granite_test.spec"):
    return SimpleNamespace(
        **_BUILDER_BASE,
        spec=spec,
        padded_vocab_size=256,
        max_position_embeddings=64,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=False,
        rotary_base=10000,
    )


class _CoreAttention:
    """Stand-in for the core attention ``built_model_summary`` reads (it keeps no own
    ``softmax_scale``, like TE's DotProductAttention, so the config's is reported)."""

    def __init__(self):
        self.config = SimpleNamespace(softmax_scale=0.015625)


class _Layer:
    def __init__(self, scaled):
        bda = scaled_bias_dropout_add(get_bias_dropout_add, 0.22) if scaled else get_bias_dropout_add
        self.self_attn_bda = self.mlp_bda = bda
        self.self_attention = SimpleNamespace(core_attention=_CoreAttention())


@pytest.mark.parametrize(
    "case, error, match",
    [
        ("plain_config", TypeError, "GraniteTransformerConfig"),
        ("unscaled_layer", ValueError, "do not scale the residual add"),
        ("no_layers", None, None),
        ("ok", None, None),
    ],
)
def test_granite_builder_guards(monkeypatch, case, error, match):
    _needs_megatron_training()

    from megatron.core.transformer.transformer_layer import TransformerLayer

    from primus.backends.megatron.core.models.granite import granite_builders as gb
    from primus.backends.megatron.core.models.granite import granite_layer_specs

    config = _cfg(residual_multiplier=0.22, normalization="RMSNorm", gated_linear_unit=True)
    layers = {"unscaled_layer": [_Layer(False)], "no_layers": []}.get(case, [_Layer(True)])
    built, logged, imported = {}, [], []

    def fake_gpt_model(**kw):
        built.update(kw)
        return SimpleNamespace(decoder=SimpleNamespace(layers=layers))

    monkeypatch.setattr(gb, "GPTModel", fake_gpt_model)
    monkeypatch.setattr(gb, "import_module", lambda path: imported.append(path) or _local_spec())
    monkeypatch.setattr(granite_layer_specs, "log_rank_0", logged.append)
    monkeypatch.setattr(gb, "log_rank_0", logged.append)
    if case == "plain_config":
        config = SimpleNamespace(residual_multiplier=0.22)
    if error is None:
        gb.granite_builder(_builder_args(), True, True, config=config)
        # args.spec path: the user spec is imported and gets the scaled bdas.
        assert imported == ["granite_test.spec"]
        spec = built["transformer_layer_spec"]
        assert spec.module is TransformerLayer
        assert _bda_tags(spec) == [pytest.approx(0.22)] * 2
        assert built["config"] is config
        assert any(f"built {len(layers)} decoder layers with scaled residual add" in m for m in logged)
        # The summary line the trainer test checks, read from the built modules.
        summary = [m for m in logged if m.startswith("[Primus:Granite] built model:")]
        assert len(summary) == 1, logged
        assert f"decoder_layers={len(layers)}" in summary[0]
        if layers:
            assert "core_attention=_CoreAttention" in summary[0]
            assert "core_attention.softmax_scale=0.015625" in summary[0]
            assert "core_attention.softmax_scale_from=config" in summary[0]
            assert "layer_bda_scale=0.22" in summary[0]
    else:
        with pytest.raises(error, match=match):
            gb.granite_builder(_builder_args(), True, True, config=config)


def test_check_scaled_residual_layers_multiplier_one_and_wrapped_model(monkeypatch):
    from primus.backends.megatron.core.models.granite import granite_layer_specs as ls

    logged = []
    monkeypatch.setattr(ls, "log_rank_0", logged.append)
    plain = SimpleNamespace(self_attn_bda=get_bias_dropout_add, mlp_bda=get_bias_dropout_add)
    ls.check_scaled_residual_layers(SimpleNamespace(decoder=SimpleNamespace(layers=[plain])), 1.0)
    assert logged == [
        "[Primus:Granite] built 1 decoder layers; residual_multiplier=1.0, so the residual add is not scaled"
    ]
    # A wrapped model (DDP / Float16Module) has no decoder: refused, not silently passed.
    with pytest.raises(TypeError, match="bare GPTModel"):
        ls.check_scaled_residual_layers(SimpleNamespace(module=SimpleNamespace()), 0.22)
    # A decoder without `layers` (renamed / wrapped) is refused too, not read as zero layers.
    logged.clear()
    with pytest.raises(TypeError, match="without it"):
        ls.check_scaled_residual_layers(SimpleNamespace(decoder=SimpleNamespace(blocks=[1, 2])), 0.22)
    assert logged == []
    # residual_multiplier 1.0 with a scaled layer: refused, not skipped.
    scaled = _Layer(True)
    with pytest.raises(ValueError, match=r"1 of 2 decoder layers do not leave the residual add unscaled"):
        ls.check_scaled_residual_layers(SimpleNamespace(decoder=SimpleNamespace(layers=[plain, scaled])), 1)
    # A --spec bda already scaled by 1.0 passes granite_layer_spec_from at 1.0, so the
    # built-model check accepts it too (one rule in both places).
    one = scaled_bias_dropout_add(get_bias_dropout_add, 1.0)
    assert ls._check_bda("self_attn_bda", one, 1.0) is True
    tagged = SimpleNamespace(self_attn_bda=one, mlp_bda=one)
    ls.check_scaled_residual_layers(SimpleNamespace(decoder=SimpleNamespace(layers=[plain, tagged])), 1.0)
    # The multiplier is validated first, and the error counts every bad layer.
    for bad_value in (0, float("nan"), True, "0.22", None):
        with pytest.raises((TypeError, ValueError)):
            ls.check_scaled_residual_layers(SimpleNamespace(decoder=SimpleNamespace(layers=[])), bad_value)
    with pytest.raises(
        ValueError, match=r"40 of 40 decoder layers do not scale .* \(first: \[0, 1, 2, 3, 4\]\)"
    ):
        ls.check_scaled_residual_layers(SimpleNamespace(decoder=SimpleNamespace(layers=[plain] * 40)), 0.22)
    logged.clear()
    # A layer without the bda attributes was not inspected: refused for any multiplier.
    for multiplier in (1.0, 0.22):
        with pytest.raises(TypeError, match=r"1 of 2 decoder layers have no self_attn_bda / mlp_bda"):
            ls.check_scaled_residual_layers(
                SimpleNamespace(decoder=SimpleNamespace(layers=[plain, SimpleNamespace(foo=1)])), multiplier
            )
    assert logged == []
    # A pipeline stage with an empty `layers` passes.
    ls.check_scaled_residual_layers(SimpleNamespace(decoder=SimpleNamespace(layers=[])), 0.22)
    assert logged == [
        "[Primus:Granite] built 0 decoder layers with scaled residual add (residual_multiplier=0.22)"
    ]


def test_granite_build_model_uses_granite_provider(monkeypatch):
    # build_model is the converter entry point the parity tools call: it builds through
    # Primus' model provider with model_type="granite" and casts to params_dtype.
    import primus.core.utils.import_utils as import_utils
    from primus.backends.megatron.checkpoint import native_loader_granite as gr

    calls = []

    class _Model:
        def to(self, dtype):
            calls.append(("to", dtype))
            return self

    def fake_get_model_provider(model_type="gpt"):
        calls.append(("provider", model_type))
        return lambda **kw: calls.append(("build", kw)) or _Model()

    monkeypatch.setattr(import_utils, "get_model_provider", fake_get_model_provider)
    gr.build_model(SimpleNamespace(params_dtype="bf16"))
    assert calls == [
        ("provider", "granite"),
        ("build", {"pre_process": True, "post_process": True}),
        ("to", "bf16"),
    ]


# ---------------------------------------------------------------------------
# Converter: refused layouts
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
        # Strict types: 1 / 0 / True are not the bool / number HF documents.
        ({"tie_word_embeddings": 1}, "tie_word_embeddings"),
        ({"tie_word_embeddings": 0}, "tie_word_embeddings"),
        ({"partial_rotary_factor": True}, "partial_rotary_factor"),
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
# Converter field details
# ---------------------------------------------------------------------------
def test_converter_maps_attention_multiplier_to_mup_fields():
    # 0.02 differs from 1/head_dim (1/64) and from 1/sqrt(64): mup_base_head_dim carries it.
    f = conv.derive_megatron_fields({**GRANITE_HF, "attention_multiplier": 0.02})
    assert f["kv_channels"] == 64
    assert f["mup_base_head_dim"] == pytest.approx((0.02 * 64) ** 2)
    # The softmax scale the upstream config derives from the converter's fields.
    built = _cfg(mup_base_head_dim=f["mup_base_head_dim"], mup_attn_scale_power=f["mup_attn_scale_power"])
    assert built.softmax_scale == pytest.approx(0.02, rel=1e-9)
    assert conv.mup_base_head_dim_for(1 / 64, 64) is None


def test_converter_rms_norm_eps_granite_default_and_errors():
    cfg = {k: v for k, v in GRANITE_HF.items() if k != "rms_norm_eps"}
    assert conv.derive_megatron_fields(cfg)["norm_epsilon"] == pytest.approx(1e-6)
    assert conv.derive_megatron_fields(GRANITE_HF)["norm_epsilon"] == pytest.approx(1e-5)
    for bad in (None, "1e-5", True, 0, -1e-5, float("inf")):
        with pytest.raises(ValueError, match="rms_norm_eps"):
            conv.derive_megatron_fields({**GRANITE_HF, "rms_norm_eps": bad})


@pytest.mark.parametrize(
    "head_dim, expected",
    [
        # HF GraniteAttention uses head_dim as given; so does the converter (documented).
        (128, 128),
        # head_dim: null means "not given": derive hidden/heads like HF, never kv_channels=None.
        (None, GRANITE_HF["hidden_size"] // GRANITE_HF["num_attention_heads"]),
    ],
)
def test_converter_head_dim(head_dim, expected):
    f = conv.derive_megatron_fields({**GRANITE_HF, "head_dim": head_dim})
    assert f["kv_channels"] == expected


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
# Layer spec: block specs and TransformerLayer subclasses
# ---------------------------------------------------------------------------


def test_granite_layer_spec_accepts_block_spec_and_custom_layers():
    pytest.importorskip("megatron.core")
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
    from megatron.core.transformer.transformer_layer import TransformerLayer

    base = get_gpt_layer_local_spec(normalization="RMSNorm")
    block = TransformerBlockSubmodules(layer_specs=[base, base], layer_norm="norm")
    out = granite_layer_spec_from(block, 0.22)
    assert [s.module for s in out.layer_specs] == [TransformerLayer] * 2
    assert [_bda_tags(s) for s in out.layer_specs] == [[pytest.approx(0.22)] * 2] * 2
    assert out.layer_norm == "norm"

    class CustomLayer(TransformerLayer):
        pass

    # A TransformerLayer subclass keeps its module; only the bdas change.
    kept = granite_layer_spec_from(dataclasses.replace(base, module=CustomLayer), 0.22)
    assert kept.module is CustomLayer and _bda_tags(kept) == [pytest.approx(0.22)] * 2


def test_preset_check_refuses_what_the_builder_refuses():
    # The builder (unsupported_problems) and the preset check (builder_problems plus the
    # attention and layout tables, each run once) refuse the same options: every option the
    # builder refuses is also a preset mismatch, so the hook stops before converting.
    from primus.backends.megatron.core.models.granite.granite_args import (
        reject_unsupported_args,
        unsupported_problems,
    )

    preset = _shipped_preset()
    assert unsupported_problems(preset.get) == []
    preset.update(mup_base_hidden_size=1280, use_rope_scaling=True, rotary_interleaved=True)
    labels = unsupported_problems(preset.get)
    assert len(labels) == 3, labels
    bad = conv.preset_mismatches(GRANITE_HF, preset)
    for label in labels:
        assert any(b.startswith(label) for b in bad), (label, bad)
    with pytest.raises(ValueError, match="use_rope_scaling"):
        reject_unsupported_args(SimpleNamespace(**preset))
    # mup_base_hidden_size equal to hidden_size is not width scaling.
    assert unsupported_problems({**_shipped_preset(), "mup_base_hidden_size": 2560}.get) == []
    # Each option is reported once (no table is run twice and filtered).
    preset.update(softmax_scale=0.125, add_bias_linear=True)
    bad = conv.preset_mismatches(GRANITE_HF, preset)
    for key in ("softmax_scale", "add_bias_linear", "use_rope_scaling"):
        assert sum(b.startswith(key) for b in bad) == 1, (key, bad)


# ---------------------------------------------------------------------------
# "Off" values of the attention options and of the boolean layout keys (granite_args)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "key, value",
    [
        ("window_size", [-1, -1]),  # TE / flash-attn: no sliding window
        ("window_size", (-1, -1)),
        ("window_size", [-1, 0]),  # TE / flash-attn: causal, no sliding window
        ("no_rope_freq", []),  # no NoPE layer
        ("no_rope_freq", [0, 0, 0]),
        ("no_rope_freq", 0),
        ("attention_output_gate", False),
        ("softmax_scale", None),
    ],
)
def test_granite_attention_off_values_are_accepted(key, value):
    from primus.backends.megatron.core.models.granite.granite_args import (
        attention_problems,
    )

    assert attention_problems({key: value}.get) == []
    assert not any(
        b.startswith(f"{key}:") for b in conv.preset_mismatches(GRANITE_HF, {**_shipped_preset(), key: value})
    )


@pytest.mark.parametrize(
    "key, value",
    [
        ("window_size", [128, 0]),
        ("window_size", [-7, -3]),  # only -1 means "no window" to TE / flash-attn
        ("window_size", [-1, 3]),
        ("softmax_scale", 0),  # only None leaves the MuP scale: 0 would be uniform attention
        ("softmax_scale", 0.0),
        ("softmax_scale", False),
        ("no_rope_freq", [0, 1]),
        ("attention_output_gate", "false"),  # a quoted string is true to Megatron
        ("use_sink_attention", "0"),
    ],
)
def test_granite_attention_on_values_are_refused(key, value):
    from primus.backends.megatron.core.models.granite.granite_args import (
        attention_problems,
    )

    assert [k for k, _, _ in attention_problems({key: value}.get)] == [key]


@pytest.mark.parametrize("key", ["qk_layernorm", "add_qkv_bias", "apply_layernorm_1p", "qk_l2_norm"])
def test_granite_layout_null_counts_as_false(key):
    # Megatron reads an unset (None) flag as off, so `key: null` is the required False.
    from primus.backends.megatron.core.models.granite.granite_args import (
        layout_problems,
    )

    assert [k for k, _, _, _ in layout_problems({key: None}.get) if k == key] == []
    assert not any(
        b.startswith(f"{key}:") for b in conv.preset_mismatches(GRANITE_HF, {**_shipped_preset(), key: None})
    )


def test_preset_null_takes_the_megatron_default():
    # An explicit null is read as the Megatron default, as granite_args._layout_matches does:
    # untie_embeddings_and_output_weights null is False (tied, as config.json).
    preset = {**_shipped_preset(), "untie_embeddings_and_output_weights": None}
    assert conv.preset_mismatches(GRANITE_HF, preset) == []
    # norm_epsilon null is Megatron's 1e-5 (granite-4.1-3b's rms_norm_eps), not "preset None".
    assert conv.preset_mismatches(GRANITE_HF, {**_shipped_preset(), "norm_epsilon": None}) == []
    # rotary_base null is Megatron's 10000, not config.json's 1e7.
    bad = conv.preset_mismatches(GRANITE_HF, {**_shipped_preset(), "rotary_base": None})
    assert [b.split(",")[0] for b in bad] == ["rotary_base: preset 10000"]


def test_built_model_summary_reads_the_built_modules():
    # The e2e trainer test checks this line: the values come from the modules of the model.
    from primus.backends.megatron.core.models.granite.granite_layer_specs import (
        built_model_summary,
    )

    scaled = scaled_bias_dropout_add(get_bias_dropout_add, 0.22)
    cfg = SimpleNamespace(mup_embedding_mult=12.0, mup_output_mult=0.1, softmax_scale=1.0 / 64)

    def model_with(core):
        layer = SimpleNamespace(self_attention=SimpleNamespace(core_attention=core), self_attn_bda=scaled)
        return SimpleNamespace(
            decoder=SimpleNamespace(layers=[layer] * 4),
            embedding=SimpleNamespace(config=cfg),
            post_process=True,
            config=cfg,
        )

    # TE keeps no softmax_scale attribute: the config's, which Megatron passes to it.
    te = type("TEDotProductAttention", (), {"config": cfg})()
    assert built_model_summary(model_with(te)) == (
        "decoder_layers=4 core_attention=TEDotProductAttention core_attention.softmax_scale=0.015625 "
        "core_attention.softmax_scale_from=config "
        "layer_bda_scale=0.22 embedding.mup_embedding_mult=12.0 output.mup_output_mult=0.1"
    )
    # PrimusTurboAttention keeps the scale it passes to the kernel: that one is reported,
    # so a module that dropped it (None) or changed it shows in the line.
    turbo = type("PrimusTurboAttention", (), {"config": cfg, "softmax_scale": 0.5})()
    summary = built_model_summary(model_with(turbo))
    assert "core_attention=PrimusTurboAttention core_attention.softmax_scale=0.5 " in summary
    assert "core_attention.softmax_scale_from=module" in summary


def test_fused_residual_rmsnorm_logs_the_branch_scale_it_applies(monkeypatch):
    # The fused forward logs the factor it multiplies the residual branch by (once per
    # factor), so the trainer test can check the fused path applied residual_multiplier.
    from primus.backends.megatron.core.extensions import fused_residual_rmsnorm as frr

    logged = []
    monkeypatch.setattr(frr, "_log", logged.append)
    monkeypatch.setattr(frr, "_LOGGED_BRANCH_SCALES", set())
    x = torch.ones(2, 3)
    assert frr._scale_branch(x, 1.0) is x
    assert logged == []
    for _ in range(3):
        torch.testing.assert_close(frr._scale_branch(x, 0.22), x * 0.22)
    assert logged == ["fused forward scales the residual branch by 0.22 (scaled residual add)"]


# One option per check in granite_args.UNSUPPORTED_CHECKS, keyed by the check's name.
_ONE_REFUSED_OPTION = {
    "builder_problems": {"use_rope_scaling": True},
    "attention_problems": {"window_size": [128, 0]},
    "layout_problems": {"qk_layernorm": True},
}


def test_every_unsupported_check_has_a_refused_option_case():
    # A check added to UNSUPPORTED_CHECKS (what unsupported_problems runs) needs a case here.
    from primus.backends.megatron.core.models.granite import granite_args

    assert set(_ONE_REFUSED_OPTION) == set(granite_args.UNSUPPORTED_CHECKS)


@pytest.mark.parametrize("check", sorted(_ONE_REFUSED_OPTION))
def test_converter_preset_check_reports_every_option_the_builder_refuses(check):
    # The check's option: that check reports it, the builder refuses it, and the converter's
    # preset check reports it too (behaviour, not source text).
    from primus.backends.megatron.core.models.granite import granite_args

    option = _ONE_REFUSED_OPTION[check]
    preset = {**_shipped_preset(), **option}
    (key,) = option
    assert any(label.startswith(key) for label in granite_args.UNSUPPORTED_CHECKS[check](preset.get))
    assert any(label.startswith(key) for label in granite_args.unsupported_problems(preset.get))
    assert any(b.startswith(key) for b in conv.preset_mismatches(GRANITE_HF, preset))
    with pytest.raises(ValueError, match=key):
        granite_args.reject_unsupported_args(SimpleNamespace(**preset))
