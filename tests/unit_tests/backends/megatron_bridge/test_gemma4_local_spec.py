###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for primus.backends.megatron_bridge.patches.gemma4.gemma4_local_spec

Guards the bug the patch exists for: ``Gemma4ModelProvider`` binds its layer
spec at dataclass-construction time from the module-level ``HAVE_TE``, and never
consults ``transformer_impl``. Setting ``transformer_impl="local"`` therefore
leaves a config that contradicts itself -- the field says local, the spec partial
still carries ``use_transformer_engine=True`` -- and the spec wins. The failure
is silent, which is what makes it worth a test rather than a comment.

The tests drive ``_rebind_spec`` directly with stub containers. It reads plain
attributes and looks megatron up through ``sys.modules`` (absent here, so those
helpers no-op), which keeps the whole file free of megatron-core and fast.

Two properties are asserted beyond the happy path, both of which a naive fix
would get wrong:
  * a config that did *not* ask for local layers must be left alone;
  * the optimizer switch must work independently of the layer spec, because the
    runs that need it most are exactly the TE runs this function otherwise
    declines to touch.
"""

from functools import partial
from types import SimpleNamespace

import pytest


def _block_spec(*, use_transformer_engine: bool):
    """Stand-in for megatron-bridge's _gemma4_block_spec."""
    return ("spec", use_transformer_engine)


def _make_container(impl, *, spec=None, full_layer_spec=False, precision_aware=False):
    """A config container shaped like the part of ConfigContainer that matters."""
    if spec is None:
        spec = partial(_block_spec, use_transformer_engine=True)
    return SimpleNamespace(
        model=SimpleNamespace(
            transformer_impl=impl,
            transformer_layer_spec=spec,
            use_transformer_engine_full_layer_spec=full_layer_spec,
        ),
        optimizer=SimpleNamespace(use_precision_aware_optimizer=precision_aware),
    )


@pytest.fixture(autouse=True)
def logged(monkeypatch):
    """Capture log_rank_0; Primus's logger is None outside a real run."""
    from primus.backends.megatron_bridge.patches.gemma4 import gemma4_local_spec as mod

    messages: list[str] = []
    monkeypatch.setattr(mod, "log_rank_0", lambda msg, *a, **k: messages.append(str(msg)))
    return messages


@pytest.fixture(autouse=True)
def _no_torch_optim_env(monkeypatch):
    """Keep the optimizer override out of the layer-spec tests unless asked for."""
    monkeypatch.delenv("PRIMUS_GEMMA4_TORCH_OPTIM", raising=False)


# -----------------------------------------------------------------------------
# transformer_impl="local" -- the fix
# -----------------------------------------------------------------------------


def test_local_impl_rebinds_spec_off_transformer_engine():
    """The whole point: local must actually turn the TE layers off."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_local_spec import (
        _rebind_spec,
    )

    container = _make_container("local")
    _rebind_spec(container)

    spec = container.model.transformer_layer_spec
    assert spec.keywords["use_transformer_engine"] is False
    # The rest of the partial must survive untouched.
    assert spec.func is _block_spec
    assert spec() == ("spec", False)


def test_local_impl_clears_full_layer_spec_switch():
    """use_transformer_engine_full_layer_spec is a second, independent switch."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_local_spec import (
        _rebind_spec,
    )

    container = _make_container("local", full_layer_spec=True)
    _rebind_spec(container)

    assert container.model.use_transformer_engine_full_layer_spec is False


def test_local_impl_clears_precision_aware_optimizer():
    """Precision-aware optimisation is a FusedAdam feature and cannot follow."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_local_spec import (
        _rebind_spec,
    )

    container = _make_container("local", precision_aware=True)
    _rebind_spec(container)

    assert container.optimizer.use_precision_aware_optimizer is False


# -----------------------------------------------------------------------------
# Everything else must be left alone
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("impl", ["transformer_engine", None, "te"])
def test_non_local_impl_leaves_the_spec_untouched(impl):
    """A config that never asked for local layers must not be rewritten."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_local_spec import (
        _rebind_spec,
    )

    container = _make_container(impl, full_layer_spec=True, precision_aware=True)
    original = container.model.transformer_layer_spec

    _rebind_spec(container)

    assert container.model.transformer_layer_spec is original
    assert container.model.transformer_layer_spec.keywords["use_transformer_engine"] is True
    assert container.model.use_transformer_engine_full_layer_spec is True
    assert container.optimizer.use_precision_aware_optimizer is True


def test_already_local_spec_is_not_rewrapped():
    """No second partial layer when there is nothing to change."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_local_spec import (
        _rebind_spec,
    )

    spec = partial(_block_spec, use_transformer_engine=False)
    container = _make_container("local", spec=spec)

    _rebind_spec(container)

    assert container.model.transformer_layer_spec is spec


def test_hand_supplied_spec_is_declined_loudly(logged):
    """An unrecognised spec is reported, not guessed at.

    Rewriting a spec whose shape we do not understand could silently build a
    different model, so refusing is correct -- but it has to be visible, since
    the user asked for local layers and will not get them.
    """
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_local_spec import (
        _rebind_spec,
    )

    sentinel = object()
    container = _make_container("local", spec=sentinel)

    _rebind_spec(container)

    assert container.model.transformer_layer_spec is sentinel
    assert any("not a functools.partial" in m for m in logged)


def test_missing_model_or_spec_is_harmless():
    """Patches run against configs they do not fully control."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_local_spec import (
        _rebind_spec,
    )

    _rebind_spec(SimpleNamespace())  # no model at all
    _rebind_spec(SimpleNamespace(model=None))

    container = _make_container("local", spec=None)
    container.model.transformer_layer_spec = None
    _rebind_spec(container)  # must not raise


# -----------------------------------------------------------------------------
# PRIMUS_GEMMA4_TORCH_OPTIM -- deliberately independent of the layer spec
# -----------------------------------------------------------------------------


def test_torch_optim_env_applies_even_when_layers_stay_on_te(monkeypatch):
    """The TE runs are the ones that need this most.

    TE's FusedAdam does not return on gfx1250, so a run that keeps TE layers
    still needs a way off it. If this were gated on transformer_impl=local it
    would be unreachable in exactly the case it was written for.
    """
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_local_spec import (
        _rebind_spec,
    )

    monkeypatch.setenv("PRIMUS_GEMMA4_TORCH_OPTIM", "1")
    container = _make_container("transformer_engine", precision_aware=True)
    original = container.model.transformer_layer_spec

    _rebind_spec(container)

    # Optimizer switched...
    assert container.optimizer.use_precision_aware_optimizer is False
    # ...while the TE layer spec is left exactly as it was.
    assert container.model.transformer_layer_spec is original
    assert original.keywords["use_transformer_engine"] is True


def test_torch_optim_env_must_be_exactly_one(monkeypatch):
    """Truthy-looking values are not accepted, so the knob stays predictable."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_local_spec import (
        _rebind_spec,
    )

    monkeypatch.setenv("PRIMUS_GEMMA4_TORCH_OPTIM", "true")
    container = _make_container("transformer_engine", precision_aware=True)

    _rebind_spec(container)

    assert container.optimizer.use_precision_aware_optimizer is True
