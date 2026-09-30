###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the Gemma 4 MoE post-attention norm patch.

The behaviour worth pinning down is that the patch is inert unless explicitly
asked for, and that when it does fire it neither double-normalises nor touches
the TransformerEngine path -- which already supplies this norm through
``TERowParallelLinearLayerNorm``. A silent mistake in either direction changes
the model's mathematics while still producing a plausible loss curve, so these
are the cases that no downstream assertion would catch.

No GPU, no megatron-core and no real layer spec are needed: the patch takes a
config container and a callable spec, so both are substitutable.
"""

from functools import partial
from types import SimpleNamespace

import pytest


class _RowParallelLinear:
    """Stand-in for the plain local projection, which applies no norm."""


class _TERowParallelLinearLayerNorm:
    """Stand-in for TE's variant, which already carries a post-norm."""


class _Norm(_RowParallelLinear):
    """Stand-in for the norm-carrying replacement the patch installs."""


@pytest.fixture(autouse=True)
def logged(monkeypatch):
    """Capture log_rank_0; Primus's logger is None outside a real run."""
    from primus.backends.megatron_bridge.patches.gemma4 import (
        gemma4_moe_post_attn_norm as mod,
    )

    messages: list[str] = []
    monkeypatch.setattr(mod, "log_rank_0", lambda msg, *a, **k: messages.append(str(msg)))
    return messages


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("PRIMUS_GEMMA4_MOE_POST_ATTN_NORM", raising=False)


@pytest.fixture
def mod():
    from primus.backends.megatron_bridge.patches.gemma4 import (
        gemma4_moe_post_attn_norm as module,
    )

    return module


def _block_spec(n_layers=3, linear_proj=_RowParallelLinear, shared=True):
    """Build a minimal block spec.

    ``shared`` mirrors the real thing, where one self-attention submodules object
    is reused across layer specs, so a single assignment covers every layer.
    """
    if shared:
        submodules = SimpleNamespace(linear_proj=linear_proj)
        attn = [SimpleNamespace(submodules=submodules) for _ in range(n_layers)]
    else:
        attn = [SimpleNamespace(submodules=SimpleNamespace(linear_proj=linear_proj)) for _ in range(n_layers)]

    return SimpleNamespace(
        layer_specs=[SimpleNamespace(submodules=SimpleNamespace(self_attention=a)) for a in attn]
    )


def _container(impl="local", use_te=False, n_layers=3, linear_proj=_RowParallelLinear, shared=True):
    spec = partial(
        lambda config, **kw: _block_spec(n_layers, linear_proj, shared), use_transformer_engine=use_te
    )
    model = SimpleNamespace(transformer_impl=impl, transformer_layer_spec=spec)
    return SimpleNamespace(model=model), spec


def _projections(container):
    built = container.model.transformer_layer_spec(container.model)
    return [ls.submodules.self_attention.submodules.linear_proj for ls in built.layer_specs]


# --- the flag gates everything -------------------------------------------------


def test_no_op_when_flag_unset(mod):
    container, original = _container()
    mod._apply(container)
    assert container.model.transformer_layer_spec is original
    assert mod.enabled() is False


@pytest.mark.parametrize("value", ["0", "", "true", "yes", "2"])
def test_only_exactly_1_enables(mod, monkeypatch, value):
    """Anything other than "1" leaves the model's mathematics alone."""
    monkeypatch.setenv("PRIMUS_GEMMA4_MOE_POST_ATTN_NORM", value)
    container, original = _container()
    mod._apply(container)
    assert mod.enabled() is False
    assert container.model.transformer_layer_spec is original


# --- the local path gets the norm ----------------------------------------------


def test_local_spec_gets_norm_on_every_layer(mod, monkeypatch):
    monkeypatch.setenv("PRIMUS_GEMMA4_MOE_POST_ATTN_NORM", "1")
    monkeypatch.setattr(mod, "_build_norm_carrying_row_parallel_linear", lambda: _Norm)

    container, original = _container(n_layers=4)
    mod._apply(container)

    assert container.model.transformer_layer_spec is not original
    assert _projections(container) == [_Norm] * 4


def test_reports_all_layers_even_when_submodules_are_shared(mod, monkeypatch, logged):
    """One assignment can cover every layer; the log must not read as a near-miss."""
    monkeypatch.setenv("PRIMUS_GEMMA4_MOE_POST_ATTN_NORM", "1")
    monkeypatch.setattr(mod, "_build_norm_carrying_row_parallel_linear", lambda: _Norm)

    container, _ = _container(n_layers=3, shared=True)
    mod._apply(container)
    _projections(container)

    assert any("3/3 layer spec(s)" in m for m in logged)


def test_unshared_submodules_each_get_patched(mod, monkeypatch):
    monkeypatch.setenv("PRIMUS_GEMMA4_MOE_POST_ATTN_NORM", "1")
    monkeypatch.setattr(mod, "_build_norm_carrying_row_parallel_linear", lambda: _Norm)

    container, _ = _container(n_layers=3, shared=False)
    mod._apply(container)

    assert _projections(container) == [_Norm] * 3


# --- and nothing else does -----------------------------------------------------


def test_declines_when_impl_is_not_local(mod, monkeypatch, logged):
    """TE supplies this norm itself; adding another would double-normalise."""
    monkeypatch.setenv("PRIMUS_GEMMA4_MOE_POST_ATTN_NORM", "1")
    container, original = _container(impl="transformer_engine", use_te=True)

    mod._apply(container)

    assert container.model.transformer_layer_spec is original
    assert any("not 'local'" in m for m in logged)


def test_declines_loudly_when_spec_still_te_bound(mod, monkeypatch, logged):
    """Means gemma4.local_spec has not run; a wrapper installed now is discarded."""
    monkeypatch.setenv("PRIMUS_GEMMA4_MOE_POST_ATTN_NORM", "1")
    container, original = _container(impl="local", use_te=True)

    mod._apply(container)

    assert container.model.transformer_layer_spec is original
    assert any("check patch order" in m for m in logged)


def test_does_not_stack_a_second_norm(mod, monkeypatch):
    """A projection already carrying a post-norm is left alone."""
    monkeypatch.setenv("PRIMUS_GEMMA4_MOE_POST_ATTN_NORM", "1")
    monkeypatch.setattr(mod, "_build_norm_carrying_row_parallel_linear", lambda: _Norm)

    container, _ = _container(linear_proj=_TERowParallelLinearLayerNorm)
    mod._apply(container)

    assert _projections(container) == [_TERowParallelLinearLayerNorm] * 3


def test_skips_layers_that_carry_the_norm_as_a_submodule_field(mod, monkeypatch):
    """The Dense layer supplies this norm itself, so its projection is left alone.

    Dense's spec pairs a plain ``RowParallelLinear`` with a live
    ``post_self_attn_layernorm`` that its own forward applies. Replacing the
    projection there would normalise twice, which no loss assertion downstream
    would flag -- it would simply train a different model.
    """
    monkeypatch.setenv("PRIMUS_GEMMA4_MOE_POST_ATTN_NORM", "1")
    monkeypatch.setattr(mod, "_build_norm_carrying_row_parallel_linear", lambda: _Norm)

    def dense_like(config, **kw):
        spec = _block_spec()
        for layer_spec in spec.layer_specs:
            layer_spec.submodules.post_self_attn_layernorm = _Norm
        return spec

    container, _ = _container()
    container.model.transformer_layer_spec = partial(dense_like, use_transformer_engine=False)
    mod._apply(container)

    assert _projections(container) == [_RowParallelLinear] * 3


def test_identity_post_attn_norm_does_not_count_as_carrying(mod, monkeypatch):
    """An IdentityOp in that field is a placeholder, so the norm is still needed."""
    monkeypatch.setenv("PRIMUS_GEMMA4_MOE_POST_ATTN_NORM", "1")
    monkeypatch.setattr(mod, "_build_norm_carrying_row_parallel_linear", lambda: _Norm)

    class IdentityOp:
        pass

    def moe_like(config, **kw):
        spec = _block_spec()
        for layer_spec in spec.layer_specs:
            layer_spec.submodules.post_self_attn_layernorm = IdentityOp
        return spec

    container, _ = _container()
    container.model.transformer_layer_spec = partial(moe_like, use_transformer_engine=False)
    mod._apply(container)

    assert _projections(container) == [_Norm] * 3


def test_applying_twice_is_idempotent(mod, monkeypatch):
    monkeypatch.setenv("PRIMUS_GEMMA4_MOE_POST_ATTN_NORM", "1")
    monkeypatch.setattr(mod, "_build_norm_carrying_row_parallel_linear", lambda: _Norm)

    container, _ = _container()
    mod._apply(container)
    wrapped_once = container.model.transformer_layer_spec
    mod._apply(container)

    assert container.model.transformer_layer_spec is wrapped_once
    assert _projections(container) == [_Norm] * 3


# --- degraded inputs -----------------------------------------------------------


def test_missing_model_is_tolerated(mod, monkeypatch):
    monkeypatch.setenv("PRIMUS_GEMMA4_MOE_POST_ATTN_NORM", "1")
    mod._apply(SimpleNamespace(model=None))


def test_non_callable_spec_is_left_alone(mod, monkeypatch, logged):
    monkeypatch.setenv("PRIMUS_GEMMA4_MOE_POST_ATTN_NORM", "1")
    container = SimpleNamespace(
        model=SimpleNamespace(transformer_impl="local", transformer_layer_spec="nope")
    )

    mod._apply(container)

    assert container.model.transformer_layer_spec == "nope"
    assert any("not callable" in m for m in logged)


def test_unimportable_norm_leaves_spec_alone(mod, monkeypatch):
    """If the pieces cannot be imported, do nothing rather than half-apply."""
    monkeypatch.setenv("PRIMUS_GEMMA4_MOE_POST_ATTN_NORM", "1")
    monkeypatch.setattr(mod, "_build_norm_carrying_row_parallel_linear", lambda: None)

    container, original = _container()
    mod._apply(container)

    assert container.model.transformer_layer_spec is original
