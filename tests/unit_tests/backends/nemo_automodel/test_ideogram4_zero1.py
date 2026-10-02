###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the ZeRO-1 optimizer patch.

WHAT IS BEING DEFENDED:

  1. WHICH OPTIMIZER-CONFIG CLASSES GET PATCHED. A config naming a plain torch
     optimizer does not resolve to an optimizer-config subclass, so the recipe
     wraps it in a factory config -- and that class overrides ``build`` without
     chaining to ``super()``. Patch only the base class and the patch is never
     called: the optimizer state stays replicated, nothing warns, and the run
     looks correct. ``TestFindsNonChainingOverrides`` builds a hierarchy with a
     non-chaining override and asserts the patch still reaches it.

  2. WHERE ZeRO-1 DOES NOT APPLY it must warn instead of pretending: already-
     sharded parameters, and a single rank. Those return the plain optimizer,
     whereas a genuine failure to construct one RAISES, pinned in
     ``TestDoesNotSilentlyDegrade``: falling back would let training start with
     the replicated optimizer state this was turned on to avoid.

No GPU and no process group: the optimizer hierarchy is stubbed, and torch's ZeRO
class and rank queries are replaced.
"""

import pytest

from tests.unit_tests.backends.nemo_automodel.parallelize._support import (
    install_stub_module,
)

OPTIM_PATH = "nemo_automodel.components.optim.optimizer"


@pytest.fixture(autouse=True)
def restore_checkpoint_guard(monkeypatch):
    """install() records checkpoint.enabled in module state."""
    from primus.backends.nemo_automodel.models.ideogram4 import zero1

    monkeypatch.setattr(zero1, "_checkpointing_enabled", False)


# --------------------------------------------------------------------------- #
# The optimizer-config hierarchy walk                                         #
# --------------------------------------------------------------------------- #
class FakeOptimizer:
    """Enough of an optimizer for the wrapper: param groups and defaults."""

    def __init__(self, params, **defaults):
        self.param_groups = [{"params": list(params)}]
        self.defaults = dict(defaults)


def install_stub_optimizer_hierarchy(monkeypatch):
    """Build a config hierarchy shaped like the real one, including the trap.

    ``FactoryConfig`` overrides ``build`` and does NOT chain to ``super()``, which
    is exactly how the real factory wrapper behaves. ``ChainingConfig`` overrides
    and does chain, so the double-wrap guard gets exercised too.
    """

    class OptimizerConfig:
        def build(self, params):
            return [FakeOptimizer(params, lr=0.1)]

    class FactoryConfig(OptimizerConfig):
        # The trap: overrides build, never calls super().
        def build(self, params):
            return [FakeOptimizer(params, lr=0.2)]

    class ChainingConfig(OptimizerConfig):
        def build(self, params):
            return super().build(params)

    class InheritingConfig(OptimizerConfig):
        # No build of its own, so it inherits the patched base.
        pass

    return install_stub_module(
        monkeypatch,
        OPTIM_PATH,
        OptimizerConfig=OptimizerConfig,
        FactoryConfig=FactoryConfig,
        ChainingConfig=ChainingConfig,
        InheritingConfig=InheritingConfig,
    )


def install_fake_zero(monkeypatch, world_size=8, fail=False):
    """Replace torch's ZeRO class and rank queries. Returns the recorded builds."""
    builds = []

    class FakeZero:
        def __init__(self, params, optimizer_class=None, **kwargs):
            if fail:
                raise RuntimeError("could not build")
            self.params = list(params)
            self.optimizer_class = optimizer_class
            self.kwargs = kwargs
            builds.append(self)

    FakeZero.__name__ = "ZeroRedundancyOptimizer"

    import torch.distributed as dist
    import torch.distributed.optim as dist_optim

    monkeypatch.setattr(dist_optim, "ZeroRedundancyOptimizer", FakeZero)
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda *a, **kw: world_size)
    return builds


@pytest.fixture
def zero1_installed(monkeypatch, set_option):
    """Install the patch with ZeRO-1 on and checkpointing off, and hand back the
    config module."""
    set_option("primus_ideogram4.zero1", True)
    module = install_stub_optimizer_hierarchy(monkeypatch)

    from primus.backends.nemo_automodel.models.ideogram4 import zero1

    assert zero1.install(checkpoint_enabled=False) is True
    return module


class TestFindsNonChainingOverrides:
    """The headline regression. Patching only the base class is a silent no-op."""

    def test_the_non_chaining_subclass_is_patched(self, zero1_installed, monkeypatch):
        """This is the class a plain-torch-optimizer config actually goes through.
        If the walk regressed to patching the base only, this build returns an
        unwrapped optimizer and nothing anywhere says so."""
        builds = install_fake_zero(monkeypatch)

        result = zero1_installed.FactoryConfig().build([1, 2, 3])

        assert len(builds) == 1, "the non-chaining override was never patched"
        assert type(result[0]).__name__ == "ZeroRedundancyOptimizer"

    def test_the_base_class_is_patched(self, zero1_installed, monkeypatch):
        builds = install_fake_zero(monkeypatch)

        zero1_installed.OptimizerConfig().build([1, 2])

        assert len(builds) == 1

    def test_a_subclass_without_its_own_build_is_covered(self, zero1_installed, monkeypatch):
        """It inherits the patched base, so it must not be double-patched either."""
        builds = install_fake_zero(monkeypatch)

        zero1_installed.InheritingConfig().build([1, 2])

        assert len(builds) == 1

    def test_a_chaining_subclass_does_not_double_wrap(self, zero1_installed, monkeypatch):
        """Both its own build and the base's are patched, so the value passes
        through the wrapper twice. The guard is the class-name check."""
        builds = install_fake_zero(monkeypatch)

        result = zero1_installed.ChainingConfig().build([1, 2])

        assert len(builds) == 1, "wrapped twice"
        assert type(result[0]).__name__ == "ZeroRedundancyOptimizer"


class TestOptimizerConstruction:
    def test_learning_rate_survives_and_is_not_duplicated(self, zero1_installed, monkeypatch):
        """ZeRO takes lr as its own parameter and the rest as defaults. Leaving lr
        in the defaults dict as well makes the call raise "multiple values for lr",
        so the build succeeding at the right rate covers both halves."""
        builds = install_fake_zero(monkeypatch)

        zero1_installed.OptimizerConfig().build([1])

        assert builds[0].kwargs["lr"] == 0.1

    def test_overlap_is_off(self, zero1_installed, monkeypatch):
        """The overlapping mode ties the step to DDP's gradient buckets and cannot
        take a learning-rate change after construction, which every schedule here
        does."""
        builds = install_fake_zero(monkeypatch)

        zero1_installed.OptimizerConfig().build([1])

        assert builds[0].kwargs["overlap_with_ddp"] is False

    def test_defaults_the_constructor_rejects_are_dropped(self, monkeypatch):
        """The real case: AdamW carries ``decoupled_weight_decay``, set by its
        parent, which AdamW's own signature has no parameter for. Forwarding it is a
        TypeError from inside ZeRO that reads as ZeRO being broken."""
        from primus.backends.nemo_automodel.models.ideogram4 import zero1

        class Narrow:
            def __init__(self, params, lr=0.0, weight_decay=0.0):
                pass

        kept = zero1._constructor_defaults(Narrow, {"weight_decay": 0.1, "decoupled_weight_decay": True})

        assert kept == {"weight_decay": 0.1}

    def test_kwargs_constructors_are_left_alone(self, monkeypatch):
        """A ``**kwargs`` constructor may accept keys its signature does not name,
        so filtering against the signature would drop valid settings."""
        from primus.backends.nemo_automodel.models.ideogram4 import zero1

        class Open:
            def __init__(self, params, lr=0.0, **kwargs):
                pass

        defaults = {"weight_decay": 0.1, "fused": True}
        assert zero1._constructor_defaults(Open, defaults) == defaults


class TestDoesNotSilentlyDegrade:
    """Where ZeRO-1 does not apply it warns; where it fails it raises."""

    def test_already_sharded_params_keep_the_plain_optimizer(self, zero1_installed, monkeypatch, caplog):
        """FSDP shards the optimizer state itself, so there is nothing to do and
        nothing is lost. Warn rather than raise: this combination is a
        misconfiguration, not a broken run."""
        builds = install_fake_zero(monkeypatch)

        class FakeDTensor:
            _local_tensor = None

        FakeDTensor.__name__ = "DTensor"

        with caplog.at_level("WARNING"):
            result = zero1_installed.OptimizerConfig().build([FakeDTensor()])

        assert builds == [], "should not have built a ZeRO optimizer"
        assert isinstance(result[0], FakeOptimizer)
        assert "already sharded" in caplog.text

    def test_single_rank_keeps_the_plain_optimizer(self, zero1_installed, monkeypatch, caplog):
        builds = install_fake_zero(monkeypatch, world_size=1)

        with caplog.at_level("WARNING"):
            result = zero1_installed.OptimizerConfig().build([1, 2])

        assert builds == []
        assert isinstance(result[0], FakeOptimizer)
        assert "one rank" in caplog.text

    def test_param_groups_reach_zero_intact(self, zero1_installed, monkeypatch):
        """AutoModel's param_group_overrides put lr_mult / wd_mult in the groups;
        flattening them into one list would silently train every group alike."""
        builds = install_fake_zero(monkeypatch)
        base = FakeOptimizer([1], lr=0.1)
        base.param_groups = [{"params": [1], "lr_mult": 1.0}, {"params": [2], "lr_mult": 0.1}]

        from primus.backends.nemo_automodel.models.ideogram4 import zero1

        zero1._wrap_in_zero1(base)

        assert [g["lr_mult"] for g in builds[0].params] == [1.0, 0.1]
        assert [g["params"] for g in builds[0].params] == [[1], [2]]

    def test_checkpointing_is_refused_where_zero1_applies(self, monkeypatch, set_option):
        """The first save would otherwise crash after training had started."""
        set_option("primus_ideogram4.zero1", True)
        module = install_stub_optimizer_hierarchy(monkeypatch)
        from primus.backends.nemo_automodel.models.ideogram4 import zero1

        assert zero1.install(checkpoint_enabled=True) is True
        builds = install_fake_zero(monkeypatch)

        with pytest.raises(RuntimeError, match="checkpoint.enabled"):
            module.OptimizerConfig().build([1, 2])
        assert builds == []

    def test_checkpointing_is_assumed_on_when_not_said(self, monkeypatch, set_option):
        """AutoModel's checkpoint.enabled defaults to true, so a config that
        leaves it out still saves."""
        set_option("primus_ideogram4.zero1", True)
        module = install_stub_optimizer_hierarchy(monkeypatch)
        from primus.backends.nemo_automodel.models.ideogram4 import zero1

        zero1.install()
        install_fake_zero(monkeypatch)

        with pytest.raises(RuntimeError, match="checkpoint.enabled"):
            module.OptimizerConfig().build([1, 2])

    def test_checkpointing_is_not_refused_where_zero1_declines(self, monkeypatch, caplog, set_option):
        set_option("primus_ideogram4.zero1", True)
        module = install_stub_optimizer_hierarchy(monkeypatch)
        from primus.backends.nemo_automodel.models.ideogram4 import zero1

        zero1.install(checkpoint_enabled=True)
        install_fake_zero(monkeypatch, world_size=1)

        with caplog.at_level("WARNING"):
            result = module.OptimizerConfig().build([1, 2])
        assert isinstance(result[0], FakeOptimizer)

    def test_a_build_failure_raises(self, zero1_installed, monkeypatch):
        """The asymmetry that matters. Nothing else in the run needs ZeRO-1 to be
        there, so a fallback here would start training with the replicated optimizer
        state this was turned on to avoid -- discovered later as an
        out-of-memory, or not at all."""
        install_fake_zero(monkeypatch, fail=True)

        with pytest.raises(RuntimeError):
            zero1_installed.OptimizerConfig().build([1, 2])


class TestGating:
    def test_off_by_default(self, monkeypatch):
        """Not set: install does nothing and reports it."""
        module = install_stub_optimizer_hierarchy(monkeypatch)
        before = vars(module.FactoryConfig)["build"]

        from primus.backends.nemo_automodel.models.ideogram4 import zero1

        assert zero1.install() is False
        assert vars(module.FactoryConfig)["build"] is before

    def test_switching_off_after_install_is_inert(self, zero1_installed, monkeypatch, set_option):
        """The switch is re-read at call time, not captured at install time, so the
        installed wrapper cannot outlive the request for it."""
        builds = install_fake_zero(monkeypatch)
        set_option("primus_ideogram4.zero1", False)

        result = zero1_installed.OptimizerConfig().build([1, 2])

        assert builds == []
        assert isinstance(result[0], FakeOptimizer)

    @pytest.mark.parametrize("value", [True, 1, "true", "True", "yes", "on"])
    def test_accepted_spellings(self, set_option, value):
        set_option("primus_ideogram4.zero1", value)
        from primus.backends.nemo_automodel.models.ideogram4 import zero1

        assert zero1.is_zero1_enabled() is True

    @pytest.mark.parametrize("value", [False, 0, "false", "off", "no"])
    def test_rejected_spellings(self, set_option, value):
        set_option("primus_ideogram4.zero1", value)
        from primus.backends.nemo_automodel.models.ideogram4 import zero1

        assert zero1.is_zero1_enabled() is False

    def test_the_patch_passes_checkpoint_enabled_through(self, monkeypatch, set_option):
        """The patch reads checkpoint.enabled from the module config, defaulting to
        AutoModel's own default of true when the key is absent."""
        import types

        import primus.backends.nemo_automodel.patches  # noqa: F401
        from primus.backends.nemo_automodel.models.ideogram4 import zero1
        from primus.core.patches import PatchContext
        from primus.core.patches.patch_registry import PatchRegistry

        patch = next(
            p
            for p in PatchRegistry.iter_patches(backend="nemo_automodel", phase="before_train")
            if p.id.endswith("zero1")
        )
        seen = []
        monkeypatch.setattr(zero1, "install", lambda checkpoint_enabled=True: seen.append(checkpoint_enabled))
        set_option("primus_ideogram4.zero1", True)

        def ctx(**params):
            return PatchContext(
                backend="nemo_automodel",
                phase="before_train",
                extra={"module_config": types.SimpleNamespace(params=types.SimpleNamespace(**params))},
            )

        patch.handler(ctx())
        patch.handler(ctx(checkpoint=types.SimpleNamespace(enabled=False)))
        assert seen == [True, False]
