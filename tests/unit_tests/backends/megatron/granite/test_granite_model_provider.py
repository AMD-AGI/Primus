###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The Granite model provider: ``get_model_provider("granite")``
(``primus/core/utils/import_utils.py``, which returns ``granite_builders.model_provider``,
re-exported from ``granite_model_provider``),
without a GPU or Megatron: upstream ``model_provider``, ``megatron.training.get_args`` and
the Granite builder module are replaced by fakes, so these run on CPU-only hosts too."""

import inspect
import sys
import types
from functools import partial

import pytest

from primus.backends.megatron.core.models.granite import granite_model_provider as gmp
from primus.core.utils import import_utils


def _upstream(calls):
    def model_provider(
        model_builder, pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None
    ):
        calls.append((model_builder, pre_process, post_process, vp_stage))
        return "model"

    return model_provider


@pytest.fixture
def fake_megatron(monkeypatch):
    """A ``megatron.training`` module whose ``get_args`` returns ``state.args``, and a stub
    ``granite_builders`` module (the real one imports Megatron, which needs a GPU here)."""
    state = types.SimpleNamespace(args=types.SimpleNamespace())
    training = types.ModuleType("megatron.training")
    training.get_args = lambda: state.args
    megatron = sys.modules.get("megatron") or types.ModuleType("megatron")
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    builders = types.ModuleType("primus.backends.megatron.core.models.granite.granite_builders")
    builders.granite_builder = object()
    builders.model_provider = gmp.model_provider  # granite_builders re-exports it
    monkeypatch.setitem(sys.modules, builders.__name__, builders)
    return state, builders.granite_builder


def test_granite_routes_to_upstream_model_provider_with_the_granite_builder(monkeypatch, fake_megatron):
    state, granite_builder = fake_megatron
    calls = []
    monkeypatch.setattr(gmp, "lazy_import", lambda paths, symbol, log_prefix="": _upstream(calls))
    provider = import_utils.get_model_provider(model_type="granite")
    # As for the other Primus-owned models: partial(<module>.model_provider, <builder>).
    assert isinstance(provider, partial)
    assert provider.func is gmp.model_provider and provider.args == (granite_builder,)
    # The upstream parameters stay visible to signature inspection.
    assert list(inspect.signature(provider).parameters) == [
        "pre_process",
        "post_process",
        "vp_stage",
        "config",
        "pg_collection",
    ]
    assert provider(True, False, vp_stage=1) == "model"
    assert calls == [(granite_builder, True, False, 1)]
    # Called without a builder, it uses the Granite builder.
    assert gmp.model_provider() == "model" and calls[-1][0] is granite_builder

    # ModelOpt would swap in upstream's ModelOpt builder and drop residual_multiplier: refused.
    state.args = types.SimpleNamespace(modelopt_enabled=True)
    with pytest.raises(ValueError, match="ModelOpt"):
        provider(True, True)
    assert len(calls) == 2


def test_granite_needs_the_model_builder_layout(monkeypatch, fake_megatron):
    # Old layout (no model_provider.py): a readable ImportError, not a bare
    # "model_provider not found".
    def fail(paths, symbol, log_prefix=""):
        raise ImportError(f"{symbol} not found in any of: {paths}")

    monkeypatch.setattr(gmp, "lazy_import", fail)
    with pytest.raises(ImportError, match="Granite needs Megatron-LM's model_provider.py"):
        import_utils.get_model_provider(model_type="granite")(True, True)

    # Mid layout (model_provider without model_builder): the module imports, its API is the
    # wrong version, so RuntimeError (not an ImportError a fallback would swallow).
    def mid(pre_process=True, post_process=True, vp_stage=None):
        return None

    monkeypatch.setattr(gmp, "lazy_import", lambda paths, symbol, log_prefix="": mid)
    with pytest.raises(RuntimeError, match=r"model_provider\(model_builder, \.\.\.\)"):
        import_utils.get_model_provider(model_type="granite")(True, True)
