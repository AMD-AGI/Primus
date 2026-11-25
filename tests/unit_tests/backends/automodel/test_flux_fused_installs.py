###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for when the FLUX fused-op patches install and when they decline.

WHAT THESE ARE DEFENDING:
  The block patches replace a diffusers forward only when the installed source is the one they
  re-implement, and the q/k patch makes its processor diffusers' default only while that default is
  diffusers' own ``FluxAttnProcessor``. Each also needs its ops in the installed Primus-Turbo (the
  adaLN patches also gfx950, the one GPU where their ops are faster). Otherwise a patch must return
  False and leave the classes as they were, so the run keeps diffusers' code. Installing twice must
  not wrap a forward twice. The q/k processor must reach every attention module built afterwards and
  compute exactly what diffusers' processor does on the calls its op does not take. The block tests
  that run the ops need a GPU; these run anywhere, with stand-ins for Primus-Turbo, which needs a GPU
  to import.
"""

import hashlib
import inspect
import sys
import types

import pytest

torch = pytest.importorskip("torch")
transformer_flux = pytest.importorskip("diffusers.models.transformers.transformer_flux")

from diffusers.models.attention import FeedForward  # noqa: E402

from primus.backends.automodel.models.flux import (  # noqa: E402
    adaln_dual_flydsl,
    adaln_flydsl,
    mlp_gelu,
    qk_norm_rope,
)
from primus.backends.automodel.models.flux.attn_processor import (  # noqa: E402
    PrimusFluxAttnProcessor,
)
from primus.backends.automodel.models.flux.turbo_ops import turbo_ops  # noqa: E402

SINGLE, DUAL = transformer_flux.FluxSingleTransformerBlock, transformer_flux.FluxTransformerBlock
ATTENTION = transformer_flux.FluxAttention


def _digest(fn):
    return hashlib.sha256(inspect.getsource(inspect.unwrap(fn)).encode()).hexdigest()


known_diffusers = pytest.mark.skipif(
    not (
        _digest(SINGLE.forward) == mlp_gelu.SINGLE_FORWARD_SHA256
        and _digest(DUAL.forward) == adaln_dual_flydsl._DUAL_FORWARD_SHA256
    ),
    reason="the block patches re-implement diffusers 0.40.0's to 0.43.0's forwards",
)


def _fake_turbo(monkeypatch, *names, gfx950=True):
    """Stand-ins for ``primus_turbo.pytorch.ops`` with the ops ``names`` and for ``is_gfx950``."""
    modules = {
        name: types.ModuleType(name)
        for name in (
            "primus_turbo",
            "primus_turbo.pytorch",
            "primus_turbo.pytorch.ops",
            "primus_turbo.pytorch.core",
            "primus_turbo.pytorch.core.utils",
        )
    }
    for name, module in modules.items():
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)
        monkeypatch.setitem(sys.modules, name, module)
    ops = modules["primus_turbo.pytorch.ops"]
    for name in names:
        setattr(ops, name, lambda *args, **kwargs: None)
    modules["primus_turbo.pytorch.core.utils"].is_gfx950 = lambda: gfx950
    return ops


def _not_diffusers(self, *args, **kwargs):
    """A forward whose source is not diffusers'."""


class _AnotherProcessor(transformer_flux.FluxAttnProcessor):
    """A default processor another patch might have installed."""


def _unchanged(**expected):
    """Whether the FLUX classes still have these forwards and default processor and no patch marker."""
    current = {
        "feed_forward": FeedForward.forward,
        "single": SINGLE.forward,
        "dual": DUAL.forward,
        "processor": ATTENTION._default_processor_cls,
    }
    markers = (
        (FeedForward, mlp_gelu._PATCHED_ATTR),
        (ATTENTION, qk_norm_rope._PATCHED_ATTR),
        (SINGLE, adaln_flydsl._PATCHED_ATTR),
        (DUAL, adaln_dual_flydsl._PATCHED_ATTR),
    )
    return all(current[k] is v for k, v in expected.items()) and not any(
        getattr(cls, attr, False) for cls, attr in markers
    )


class TestTurboOps:
    def test_returns_the_ops(self, monkeypatch):
        ops = _fake_turbo(monkeypatch, "linear_gelu")
        assert turbo_ops("Test", "linear_gelu") is ops

    def test_declines_without_an_op(self, monkeypatch):
        _fake_turbo(monkeypatch, "linear_gelu")
        assert turbo_ops("Test", "linear_gelu", "adaln_modulate") is None

    def test_declines_when_not_importable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "primus_turbo", None)
        assert turbo_ops("Test", "linear_gelu") is None

    def test_gfx950_only_declines_elsewhere(self, monkeypatch):
        ops = _fake_turbo(monkeypatch, "adaln_modulate", gfx950=False)
        assert turbo_ops("Test", "adaln_modulate", gain_on_gfx950_only=True) is None
        assert turbo_ops("Test", "adaln_modulate") is ops


class TestDeclines:
    def test_fused_gelu_without_linear_gelu(self, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch)
        before = dict(feed_forward=FeedForward.forward, single=SINGLE.forward)
        assert flux_fused_patches.mlp_gelu.install() is False
        assert _unchanged(**before)

    def test_fused_gelu_keeps_a_foreign_single_stream_forward(self, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "linear_gelu")
        monkeypatch.setattr(SINGLE, "forward", _not_diffusers)
        assert flux_fused_patches.mlp_gelu.install() is True
        assert SINGLE.forward is _not_diffusers

    def test_qk_norm_rope_over_another_default_processor(self, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "dit_qk_norm_rope")
        monkeypatch.setattr(ATTENTION, "_default_processor_cls", _AnotherProcessor)
        assert flux_fused_patches.qk_norm_rope.install() is False
        assert _unchanged(processor=_AnotherProcessor)

    def test_qk_norm_rope_without_the_op(self, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch)
        before = ATTENTION._default_processor_cls
        assert flux_fused_patches.qk_norm_rope.install() is False
        assert _unchanged(processor=before)

    def test_adaln_without_fused_gelu(self, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "adaln_modulate", "linear_gelu")
        before = SINGLE.forward
        assert flux_fused_patches.adaln.install() is False
        assert _unchanged(single=before)

    def test_adaln_dual_on_a_foreign_forward(self, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "gated_residual_adaln")
        monkeypatch.setattr(DUAL, "forward", _not_diffusers)
        assert flux_fused_patches.adaln_dual.install() is False
        assert _unchanged(dual=_not_diffusers)

    @known_diffusers
    def test_adaln_off_gfx950(self, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "adaln_modulate", "linear_gelu", gfx950=False)
        assert flux_fused_patches.mlp_gelu.install() is True
        fused_gelu = SINGLE.forward
        assert flux_fused_patches.adaln.install() is False
        assert SINGLE.forward is fused_gelu
        assert not getattr(SINGLE, adaln_flydsl._PATCHED_ATTR, False)

    @known_diffusers
    def test_adaln_dual_off_gfx950(self, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "gated_residual_adaln", gfx950=False)
        before = DUAL.forward
        assert flux_fused_patches.adaln_dual.install() is False
        assert _unchanged(dual=before)


class TestQkNormRopeProcessor:
    def test_attention_built_afterwards_takes_it(self, flux_fused_patches, monkeypatch):
        ops = _fake_turbo(monkeypatch, "dit_qk_norm_rope")
        assert flux_fused_patches.qk_norm_rope.install() is True
        assert PrimusFluxAttnProcessor.dit_qk_norm_rope is ops.dit_qk_norm_rope
        assert type(ATTENTION(query_dim=64, heads=2, dim_head=32).processor) is PrimusFluxAttnProcessor

    @pytest.mark.parametrize("dual", [False, True])
    def test_calls_off_the_op_compute_what_diffusers_does(self, dual, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "dit_qk_norm_rope")
        assert flux_fused_patches.qk_norm_rope.install() is True
        torch.manual_seed(0)
        attn = ATTENTION(
            query_dim=64, heads=2, dim_head=32, added_kv_proj_dim=64 if dual else None, bias=True
        )
        hidden_states = torch.randn(2, 8, 64)
        encoder_hidden_states = torch.randn(2, 4, 64) if dual else None
        # CPU tensors, which the op does not take.
        args = (attn, hidden_states, encoder_hidden_states)
        got = PrimusFluxAttnProcessor()(*args)
        expected = transformer_flux.FluxAttnProcessor()(*args)
        if not dual:
            got, expected = (got,), (expected,)
        assert all(torch.equal(g, e) for g, e in zip(got, expected, strict=True))


class TestIdempotent:
    @known_diffusers
    def test_fused_gelu(self, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "linear_gelu")
        diffusers_single = SINGLE.forward
        assert flux_fused_patches.mlp_gelu.install() is True
        patched = FeedForward.forward, SINGLE.forward
        assert SINGLE.forward is not diffusers_single
        assert flux_fused_patches.mlp_gelu.install() is True
        assert (FeedForward.forward, SINGLE.forward) == patched

    def test_qk_norm_rope(self, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "dit_qk_norm_rope")
        assert flux_fused_patches.qk_norm_rope.install() is True
        assert flux_fused_patches.qk_norm_rope.install() is True
        assert ATTENTION._default_processor_cls is PrimusFluxAttnProcessor
        assert ATTENTION._available_processors.count(PrimusFluxAttnProcessor) == 1
