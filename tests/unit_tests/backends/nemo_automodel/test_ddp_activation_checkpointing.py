###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the ``ddp.activation_checkpointing`` forwarding repair.

The stub tests pin the wrapper's own behaviour, including that it does nothing
once upstream forwards the flag itself. The contract tests run the real
AutoModel parser and dispatch, and are skipped when it is not installed.
"""

import importlib.util
import sys
import types

import pytest

from primus.backends.nemo_automodel.distributed import (
    ddp_activation_checkpointing as shim,
)

MODULE_PATH = "nemo_automodel.components.distributed.model_parallelizer"

requires_automodel = pytest.mark.skipif(
    importlib.util.find_spec("nemo_automodel") is None, reason="nemo_automodel is not installed"
)


def _context(requested, configured=False):
    return types.SimpleNamespace(
        activation_checkpointing=requested,
        strategy_config=types.SimpleNamespace(activation_checkpointing=configured),
    )


@pytest.fixture
def stub_dispatch(monkeypatch):
    """A stub ``model_parallelizer`` whose DDP dispatch records the config it saw."""
    module = types.ModuleType(MODULE_PATH)
    seen = []

    def _parallelize_ddp(model, mesh_context):
        seen.append(mesh_context.strategy_config.activation_checkpointing)
        return model

    module._parallelize_ddp = _parallelize_ddp
    parts = MODULE_PATH.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, MODULE_PATH, module)
    monkeypatch.setattr(sys.modules[".".join(parts[:-1])], parts[-1], module, raising=False)
    return module, seen


class TestForwarding:
    def test_dropped_flag_reaches_ddp(self, stub_dispatch):
        module, seen = stub_dispatch
        assert shim.install() is True
        module._parallelize_ddp(object(), _context(True))
        assert seen == [True]

    def test_selective_is_forwarded_verbatim(self, stub_dispatch):
        module, seen = stub_dispatch
        shim.install()
        module._parallelize_ddp(object(), _context("selective"))
        assert seen == ["selective"]

    def test_nothing_requested_changes_nothing(self, stub_dispatch):
        module, seen = stub_dispatch
        shim.install()
        module._parallelize_ddp(object(), _context(False))
        assert seen == [False]

    def test_upstream_value_wins(self, stub_dispatch):
        """Once upstream fills DDPConfig itself, the wrapper must not touch it."""
        module, seen = stub_dispatch
        shim.install()
        module._parallelize_ddp(object(), _context(True, configured="selective"))
        assert seen == ["selective"]

    def test_install_is_idempotent(self, stub_dispatch):
        module, _ = stub_dispatch
        shim.install()
        first = module._parallelize_ddp
        assert shim.install() is True
        assert module._parallelize_ddp is first

    def test_missing_hook_declines_without_raising(self, stub_dispatch):
        module, _ = stub_dispatch
        del module._parallelize_ddp
        assert shim.install() is False


@requires_automodel
class TestUpstreamContract:
    """Real parser and real dispatch, with only the final DDP wrap recorded."""

    def _run(self, monkeypatch, ddp_section):
        from nemo_automodel.components.distributed import ddp
        from nemo_automodel.components.distributed.mesh import MeshContext
        from nemo_automodel.components.distributed.model_parallelizer import (
            _DEFAULT_PARALLELIZER,
            _apply_model_parallelizer,
        )
        from nemo_automodel.recipes._dist_utils import parse_distributed_section

        seen = []
        monkeypatch.setattr(
            ddp, "parallelize_ddp", lambda model, config, **kw: seen.append(config.activation_checkpointing)
        )
        parsed = parse_distributed_section({"strategy": "ddp", **ddp_section})
        mesh_context = MeshContext(
            strategy_config=parsed["strategy_config"],
            activation_checkpointing=parsed["activation_checkpointing"],
        )
        _apply_model_parallelizer(_DEFAULT_PARALLELIZER, object(), mesh_context)
        return seen

    def test_flag_reaches_parallelize_ddp(self, monkeypatch):
        from nemo_automodel.components.distributed import model_parallelizer as mp

        monkeypatch.setattr(mp, "_parallelize_ddp", mp._parallelize_ddp)
        assert shim.install() is True
        assert self._run(monkeypatch, {"activation_checkpointing": True}) == [True]

    def test_unset_flag_stays_off(self, monkeypatch):
        from nemo_automodel.components.distributed import model_parallelizer as mp

        monkeypatch.setattr(mp, "_parallelize_ddp", mp._parallelize_ddp)
        shim.install()
        assert self._run(monkeypatch, {}) == [False]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
