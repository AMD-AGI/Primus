###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the DDP + ZeRO-1 hook.

WHY THIS TEST EXISTS:
  The hook installed cleanly, logged "Installed optional hook", checkpoint-wrapped all 34
  layers, and trained to a falling loss -- while the ZeRO-1 half never ran at all. YAML
  ``_target_: torch.optim.AdamW`` is not an OptimizerConfig subclass, so Automodel wraps it
  in ``OptimizerFromFactoryConfig``, which overrides ``build`` and never chains to
  ``super()``. The patch was on the base class, so it was simply never called. Nothing
  raised, nothing warned, and the optimizer state stayed fully replicated: a pure-DDP run
  wearing a ZeRO-1 label.

  So these tests assert the *interception*, not that training "works" -- the ZeRO-1 run
  looked perfectly healthy while being broken. The load-bearing one is
  ``test_factory_config_build_is_intercepted``: it fails against the base-class-only patch.

  No GPU and no process group needed; the ZeRO construction itself is stubbed out.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nemo_automodel")

from nemo_automodel.components.optim.optimizer import (  # noqa: E402
    AdamWConfig,
    OptimizerConfig,
    OptimizerFromFactoryConfig,
    build_optimizer_config,
)

from primus.backends.nemo_automodel.models.ideogram4 import zero1  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_builds():
    """The patch mutates shared Automodel classes; undo it so tests cannot leak."""
    saved = [(cls, vars(cls)["build"]) for cls in zero1._optimizer_config_classes()]
    yield
    for cls, fn in saved:
        cls.build = fn
    for cls, _ in saved:
        if getattr(vars(cls).get("build"), "_primus_zero1_patched", False):  # pragma: no cover
            del cls.build


@pytest.fixture
def zero1_on(monkeypatch):
    monkeypatch.setenv("PRIMUS_IDEOGRAM_ZERO1", "1")


def _tiny_model():
    return torch.nn.Linear(4, 4)


def _factory_config():
    """What the DDP preset's ``_target_: torch.optim.AdamW`` actually normalizes to."""
    return build_optimizer_config(torch.optim.AdamW, {"lr": 1e-4})


# --------------------------------------------------------------------------------------
# The hole itself
# --------------------------------------------------------------------------------------


def test_ddp_preset_target_normalizes_to_the_factory_config():
    """Pins the premise: the preset never goes through the base class's ``build``."""
    cfg = _factory_config()
    assert isinstance(cfg, OptimizerFromFactoryConfig)
    assert "build" in vars(type(cfg)), "override is what made the base-class patch dead"


def test_factory_config_does_not_chain_to_super():
    """If it ever did, patching the base alone would have been enough."""
    import inspect

    assert "super()" not in inspect.getsource(OptimizerFromFactoryConfig.build)


def test_hierarchy_walk_covers_every_build_override():
    covered = {c.__name__ for c in zero1._optimizer_config_classes()}
    expected = {
        cls.__name__
        for cls in _all_subclasses(OptimizerConfig) | {OptimizerConfig}
        if "build" in vars(cls)
    }
    assert covered == expected
    assert "OptimizerFromFactoryConfig" in covered


def _all_subclasses(cls):
    out = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _all_subclasses(sub)
    return out


# --------------------------------------------------------------------------------------
# Interception (the regression pin)
# --------------------------------------------------------------------------------------


def test_factory_config_build_is_intercepted(zero1_on, monkeypatch):
    """THE load-bearing test: fails if the patch only covers ``OptimizerConfig``."""
    seen = []
    monkeypatch.setattr(zero1, "_wrap_in_zero1", lambda opt: seen.append(opt) or opt)

    zero1._install_zero1_optimizer_patch()
    optimizers = _factory_config().build(_tiny_model())

    assert len(seen) == 1, "ZeRO-1 wrap never ran for the config the DDP preset builds"
    assert seen[0] is optimizers[0]


def test_typed_config_build_is_intercepted(zero1_on, monkeypatch):
    """``AdamWConfig`` inherits ``build``, so the base-class patch must still cover it."""
    seen = []
    monkeypatch.setattr(zero1, "_wrap_in_zero1", lambda opt: seen.append(opt) or opt)

    zero1._install_zero1_optimizer_patch()
    AdamWConfig(lr=1e-4).build(_tiny_model())

    assert len(seen) == 1


def test_build_is_untouched_when_zero1_is_off(monkeypatch):
    monkeypatch.delenv("PRIMUS_IDEOGRAM_ZERO1", raising=False)
    seen = []
    monkeypatch.setattr(zero1, "_wrap_in_zero1", lambda opt: seen.append(opt) or opt)

    zero1._install_zero1_optimizer_patch()
    optimizers = _factory_config().build(_tiny_model())

    assert seen == [], "the gate is read at call time, not install time"
    assert isinstance(optimizers[0], torch.optim.AdamW)


def test_install_is_idempotent(zero1_on, monkeypatch):
    """Re-installing must not nest wrappers (which would double-wrap the optimizer)."""
    seen = []
    monkeypatch.setattr(zero1, "_wrap_in_zero1", lambda opt: seen.append(opt) or opt)

    zero1._install_zero1_optimizer_patch()
    zero1._install_zero1_optimizer_patch()
    _factory_config().build(_tiny_model())

    assert len(seen) == 1


def test_patched_build_keeps_its_signature(zero1_on):
    """The CP plan hit exactly this: a ``*args`` wrapper broke signature introspection."""
    import inspect

    before = inspect.signature(OptimizerFromFactoryConfig.build)
    zero1._install_zero1_optimizer_patch()
    assert inspect.signature(OptimizerFromFactoryConfig.build) == before


# --------------------------------------------------------------------------------------
# Degradation paths: they must be loud, never silent
# --------------------------------------------------------------------------------------


def test_wrap_is_a_noop_on_an_already_wrapped_optimizer():
    class ZeroRedundancyOptimizer:  # name-matched, as the guard checks the type name
        pass

    already = ZeroRedundancyOptimizer()
    assert zero1._wrap_in_zero1(already) is already


def test_wrap_skips_dtensor_params_with_a_warning(caplog):
    """FSDP2 mode: ZeRO-1 does not apply, and the run must say so rather than pretend."""

    class _FakeDTensor:
        _local_tensor = None

    base = type("Opt", (), {"param_groups": [{"params": [_FakeDTensor()]}]})()
    with caplog.at_level("WARNING"):
        assert zero1._wrap_in_zero1(base) is base
    assert "DTensor" in caplog.text


def test_wrap_skips_without_a_process_group(caplog):
    base = torch.optim.AdamW(_tiny_model().parameters(), lr=1e-4)
    with caplog.at_level("WARNING"):
        assert zero1._wrap_in_zero1(base) is base
    assert "process group" in caplog.text


def test_install_is_a_noop_unless_a_gate_is_set(monkeypatch):
    monkeypatch.delenv("PRIMUS_IDEOGRAM_ZERO1", raising=False)
    monkeypatch.delenv("PRIMUS_IDEOGRAM_DDP", raising=False)
    assert zero1.install() is False
