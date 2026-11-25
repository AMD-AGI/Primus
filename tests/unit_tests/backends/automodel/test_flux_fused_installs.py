###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for when the FLUX fused-op patches install and when they decline.

WHAT THESE ARE DEFENDING:
  The block patches replace a diffusers forward only when the installed code is the one they
  re-implement and no other patch has wrapped it (``pinned_source``, which also ignores docstrings,
  annotations and formatting), and the q/k patch makes its processor diffusers' default only while that default is
  diffusers' own ``FluxAttnProcessor``. Each also needs its ops in the installed Primus-Turbo (the
  adaLN patches also gfx950, the one GPU where their ops are faster). Otherwise a patch must return
  False and leave the classes as they were, so the run keeps diffusers' code. Installing twice must
  not wrap a forward twice. The q/k processor must reach every attention module built afterwards and
  compute exactly what diffusers' processor does on the calls its op does not take. The block tests
  that run the ops need a GPU; these run anywhere, with stand-ins for Primus-Turbo, which needs a GPU
  to import.
"""

import functools
import importlib.util
import sys
import types

import pytest

torch = pytest.importorskip("torch")
transformer_flux = pytest.importorskip("diffusers.models.transformers.transformer_flux")

from diffusers.models.activations import GELU  # noqa: E402
from diffusers.models.attention import FeedForward  # noqa: E402
from diffusers.models.normalization import AdaLayerNormZeroSingle  # noqa: E402

from primus.backends.automodel.models.flux import (  # noqa: E402
    adaln,
    adaln_dual,
    mlp_gelu,
    pinned_source,
    qk_norm_rope,
)
from primus.backends.automodel.models.flux.attn_processor import (  # noqa: E402
    PrimusFluxAttnProcessor,
)
from primus.backends.automodel.models.flux.turbo_ops import turbo_ops  # noqa: E402

SINGLE, DUAL = transformer_flux.FluxSingleTransformerBlock, transformer_flux.FluxTransformerBlock
ATTENTION = transformer_flux.FluxAttention

known_diffusers = pytest.mark.skipif(
    not all(
        pinned_source.digest(fn) == expected
        for fn, expected in (
            (SINGLE.forward, mlp_gelu.SINGLE_FORWARD_DIGEST),
            (FeedForward.forward, mlp_gelu.FEED_FORWARD_DIGEST),
            (GELU.forward, mlp_gelu.GELU_FORWARD_DIGEST),
            (AdaLayerNormZeroSingle.forward, adaln.ADA_NORM_FORWARD_DIGEST),
            (DUAL.forward, adaln_dual.DUAL_FORWARD_DIGEST),
        )
    ),
    reason="the block patches re-implement diffusers 0.38.0's to 0.43.0's forwards",
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


def _wrapped(fn):
    """``fn`` as another patch might wrap it."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


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
        (SINGLE, mlp_gelu._PATCHED_ATTR),
        (ATTENTION, qk_norm_rope._PATCHED_ATTR),
        (SINGLE, adaln._PATCHED_ATTR),
        (DUAL, adaln_dual._PATCHED_ATTR),
    )
    return all(current[k] is v for k, v in expected.items()) and not any(
        getattr(cls, attr, False) for cls, attr in markers
    )


_PINNED = """
def forward(self, x):
    y, z = x.chunk(2)
    return y + z
"""


def _forward(tmp_path, source):
    """The ``forward`` a module with this source defines; ``inspect.getsource`` needs it in a file."""
    path = tmp_path / f"source_{len(list(tmp_path.iterdir()))}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.forward


class TestPinnedSource:
    @pytest.mark.parametrize(
        "source",
        [
            'def forward(self, x):\n    """A docstring."""\n    y, z = x.chunk(2)\n    return y + z\n',
            "def forward(self, x: int) -> int:\n    y, z = x.chunk(2)\n    return y + z\n",
            "def forward(self, x):\n    # A comment.\n    (y, z) = x.chunk(\n        2,\n    )\n    return (y + z)\n",
            "class Block:\n    def forward(self, x):\n        y, z = x.chunk(2)\n        return y + z\nforward = Block.forward\n",
        ],
        ids=["docstring", "annotations", "formatting", "indented"],
    )
    def test_matches_the_same_code(self, source, tmp_path):
        expected = pinned_source.digest(_forward(tmp_path, _PINNED))
        assert expected is not None
        assert pinned_source.digest(_forward(tmp_path, source)) == expected

    def test_a_changed_statement_differs(self, tmp_path):
        changed = "def forward(self, x):\n    y, z = x.chunk(2)\n    return y - z\n"
        assert pinned_source.digest(_forward(tmp_path, changed)) != pinned_source.digest(
            _forward(tmp_path, _PINNED)
        )

    def test_a_wrapped_function_matches_nothing(self, tmp_path):
        assert pinned_source.digest(_wrapped(_forward(tmp_path, _PINNED))) is None


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

    @known_diffusers
    def test_fused_gelu_keeps_a_foreign_single_stream_forward(self, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "linear_gelu")
        monkeypatch.setattr(SINGLE, "forward", _not_diffusers)
        assert flux_fused_patches.mlp_gelu.install() is True
        assert SINGLE.forward is _not_diffusers
        assert flux_fused_patches.mlp_gelu.single_forward() is None
        assert getattr(FeedForward, mlp_gelu._PATCHED_ATTR, False)

    @known_diffusers
    @pytest.mark.parametrize("foreign", ["feed_forward", "gelu"])
    def test_fused_gelu_keeps_a_foreign_feed_forward(self, foreign, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "linear_gelu")
        monkeypatch.setattr(FeedForward if foreign == "feed_forward" else GELU, "forward", _not_diffusers)
        feed_forward = FeedForward.forward
        assert flux_fused_patches.mlp_gelu.install() is True
        assert FeedForward.forward is feed_forward
        assert not getattr(FeedForward, mlp_gelu._PATCHED_ATTR, False)
        assert SINGLE.forward is flux_fused_patches.mlp_gelu.single_forward()

    def test_fused_gelu_on_no_known_forward(self, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "linear_gelu")
        monkeypatch.setattr(FeedForward, "forward", _not_diffusers)
        monkeypatch.setattr(SINGLE, "forward", _wrapped(SINGLE.forward))
        before = dict(feed_forward=FeedForward.forward, single=SINGLE.forward)
        assert flux_fused_patches.mlp_gelu.install() is False
        assert _unchanged(**before)

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

    @known_diffusers
    def test_adaln_over_a_wrapped_fused_gelu(self, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "adaln_modulate", "linear_gelu")
        assert flux_fused_patches.mlp_gelu.install() is True
        monkeypatch.setattr(SINGLE, "forward", _wrapped(SINGLE.forward))
        wrapped = SINGLE.forward
        assert flux_fused_patches.adaln.install() is False
        assert SINGLE.forward is wrapped
        assert not getattr(SINGLE, adaln._PATCHED_ATTR, False)

    @known_diffusers
    def test_adaln_on_a_foreign_ada_norm(self, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "adaln_modulate", "linear_gelu")
        assert flux_fused_patches.mlp_gelu.install() is True
        fused_gelu = SINGLE.forward
        monkeypatch.setattr(AdaLayerNormZeroSingle, "forward", _not_diffusers)
        assert flux_fused_patches.adaln.install() is False
        assert SINGLE.forward is fused_gelu
        assert not getattr(SINGLE, adaln._PATCHED_ATTR, False)

    @pytest.mark.parametrize(
        "make_forward",
        [lambda: _not_diffusers, lambda: _wrapped(DUAL.forward)],
        ids=["foreign", "wrapped"],
    )
    def test_adaln_dual_on_another_forward(self, make_forward, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "gated_residual_adaln")
        dual = make_forward()
        monkeypatch.setattr(DUAL, "forward", dual)
        assert flux_fused_patches.adaln_dual.install() is False
        assert _unchanged(dual=dual)

    @known_diffusers
    def test_adaln_off_gfx950(self, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "adaln_modulate", "linear_gelu", gfx950=False)
        assert flux_fused_patches.mlp_gelu.install() is True
        fused_gelu = SINGLE.forward
        assert flux_fused_patches.adaln.install() is False
        assert SINGLE.forward is fused_gelu
        assert not getattr(SINGLE, adaln._PATCHED_ATTR, False)

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
        assert SINGLE.forward is flux_fused_patches.mlp_gelu.single_forward()
        assert flux_fused_patches.mlp_gelu.install() is True
        assert (FeedForward.forward, SINGLE.forward) == patched

    def test_qk_norm_rope(self, flux_fused_patches, monkeypatch):
        _fake_turbo(monkeypatch, "dit_qk_norm_rope")
        assert flux_fused_patches.qk_norm_rope.install() is True
        assert flux_fused_patches.qk_norm_rope.install() is True
        assert ATTENTION._default_processor_cls is PrimusFluxAttnProcessor
        assert ATTENTION._available_processors.count(PrimusFluxAttnProcessor) == 1
