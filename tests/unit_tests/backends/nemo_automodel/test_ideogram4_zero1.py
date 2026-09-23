###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the Ideogram-4 DDP and ZeRO-1 patches.

WHAT IS BEING DEFENDED. Both patches fix a silent no-op, and the tests exist
because both are easy to reintroduce:

  1. WHICH OPTIMIZER-CONFIG CLASSES GET PATCHED. This is the subtle one. A config
     naming a plain torch optimizer does not resolve to an optimizer-config
     subclass, so the recipe wraps it in a factory config -- and that class
     overrides ``build`` without chaining to ``super()``. Patch only the base
     class and the patch is never called: the optimizer state stays replicated,
     nothing warns, and the run looks correct. The only cheap way to catch that
     regression is to build a hierarchy containing a non-chaining override and
     assert the patch still reaches it, which is what
     ``TestFindsNonChainingOverrides`` does.

  2. ACTIVATION CHECKPOINTING ON THE DDP PATH. The DDP manager wraps by submodule
     attribute name, which finds nothing on a single-stream diffusion block, so
     checkpointing silently does not happen and the model runs out of memory. The
     assertion is that blocks were wrapped, and that it happened before the DDP
     wrap -- the module structure has to be final before anything indexes the
     parameters.

  A third group covers the cases where ZeRO-1 does not apply and must warn
  instead of pretending: already-sharded parameters, and a single rank. Those
  return the plain optimizer, whereas a genuine failure to construct one RAISES.
  That asymmetry is deliberate and pinned in ``TestDoesNotSilentlyDegrade``:
  nothing else in a run needs ZeRO-1 to be present, so falling back would let
  training start and then use the replicated optimizer state this was turned on to
  avoid.

No GPU and no process group: the parallelizer, the DDP manager and the optimizer
hierarchy are stubbed, and torch's ZeRO class and rank queries are replaced.
"""

import pytest

from tests.unit_tests.backends.nemo_automodel.parallelize._reshard_contract import (
    install_stub_module,
    install_stub_parallelizer,
    model_with_blocks,
)

BLOCK_ATTR = "layers"
MODEL_NAME = "Ideogram4Transformer2DModel"
DDP_PATH = "nemo_automodel.components.distributed.ddp"
OPTIM_PATH = "nemo_automodel.components.optim.optimizer"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "PRIMUS_IDEOGRAM_ZERO1",
        "PRIMUS_IDEOGRAM_DDP",
        "PRIMUS_IDEOGRAM_AC_EVERY",
    ):
        monkeypatch.delenv(name, raising=False)


def ideogram_model(blocks=6, with_block_list=True):
    """A stand-in matched the way the patch matches: by class name.

    The patch compares ``type(model).__name__`` rather than using isinstance, so
    that it never has to import diffusers. A test model therefore has to carry the
    real class name, which means building the type here rather than reaching for
    the shared ``model_with_blocks`` helper.
    """
    instance = type(MODEL_NAME, (), {})()
    if with_block_list:
        setattr(instance, BLOCK_ATTR, list(range(blocks)))
    return instance


# --------------------------------------------------------------------------- #
# DDP-path stubs                                                              #
# --------------------------------------------------------------------------- #
def install_stub_ddp(monkeypatch):
    """Stub the DDP manager and record the order of what happens to the model.

    Returns ``(module, events)``. ``events`` receives ``("wrapped", index)`` for
    each checkpoint wrap and ``("ddp", ac_flag)`` when the original parallelize
    runs, so a test can assert on ORDER and not merely on counts.
    """
    events = []

    class DDPManager:
        def __init__(self, activation_checkpointing=False, should_fail=False):
            self.activation_checkpointing = activation_checkpointing
            self.should_fail = should_fail

        def parallelize(self, model):
            # Records what the manager's own checkpointing flag looked like at the
            # moment it ran, which is how the suppression is checked.
            events.append(("ddp", self.activation_checkpointing))
            if self.should_fail:
                raise RuntimeError("the DDP wrap failed")
            return model

    module = install_stub_module(monkeypatch, DDP_PATH, DDPManager=DDPManager)

    import nemo_automodel.components.distributed.parallelizer as P

    monkeypatch.setattr(
        P,
        "checkpoint_wrapper",
        lambda block, **kw: events.append(("wrapped", block)) or ("w", block),
    )
    return module, events


@pytest.fixture
def ddp(monkeypatch):
    """Install the DDP checkpointing patch and hand back the manager and events."""
    monkeypatch.setenv("PRIMUS_IDEOGRAM_DDP", "1")
    install_stub_parallelizer(monkeypatch)
    module, events = install_stub_ddp(monkeypatch)

    from primus.backends.nemo_automodel.models.ideogram4 import zero1

    assert zero1.install() is True
    return module, events


class TestDdpCheckpointingHappens:
    def test_blocks_are_wrapped(self, ddp):
        """The whole reason the patch exists. Unpatched this wraps nothing and says
        nothing about it."""
        module, events = ddp
        model = ideogram_model(blocks=6)

        module.DDPManager(activation_checkpointing=True).parallelize(model)

        wrapped = [e for e in events if e[0] == "wrapped"]
        assert len(wrapped) == 6
        assert all(isinstance(b, tuple) for b in getattr(model, BLOCK_ATTR))

    def test_wrapping_precedes_the_ddp_wrap(self, ddp):
        """Order, not just occurrence: DDP flattens parameters, so the module
        structure has to be final before it runs."""
        module, events = ddp

        module.DDPManager(activation_checkpointing=True).parallelize(ideogram_model(3))

        kinds = [e[0] for e in events]
        assert kinds.count("ddp") == 1
        assert kinds.index("ddp") == len(kinds) - 1, f"the DDP wrap must come last, got {kinds}"

    def test_managers_own_checkpointing_is_suppressed_then_restored(self, ddp):
        """The manager's traversal finds nothing here, and its "wrapped 0" would be
        indistinguishable from the bug this patch fixes -- so it is turned off for
        the inner call. It has to come back on afterwards: the manager is not
        necessarily used only once."""
        module, events = ddp
        manager = module.DDPManager(activation_checkpointing=True)

        manager.parallelize(ideogram_model(2))

        assert ("ddp", False) in events, "the manager's own AC was left enabled"
        assert manager.activation_checkpointing is True, "the flag was not restored"

    def test_restored_even_when_the_ddp_wrap_raises(self, ddp):
        """Restoration sits in a finally block, so a failure downstream does not
        leave the manager permanently altered -- which would turn one failure into a
        second, differently-shaped one on any retry."""
        module, _events = ddp
        manager = module.DDPManager(activation_checkpointing=True, should_fail=True)

        with pytest.raises(RuntimeError):
            manager.parallelize(ideogram_model(2))

        assert manager.activation_checkpointing is True


class TestDdpCheckpointingScope:
    def test_other_models_are_untouched(self, ddp):
        """The patch is on a shared manager, so it has to be a no-op for anything
        that is not this model."""
        module, events = ddp
        other = model_with_blocks(BLOCK_ATTR, blocks=range(4))

        module.DDPManager(activation_checkpointing=True).parallelize(other)

        assert [e for e in events if e[0] == "wrapped"] == []
        assert events == [("ddp", True)], "the manager's own AC should stay enabled"

    @pytest.mark.parametrize("value", [False, "false", "0", "off", "no", "none", ""])
    def test_off_wraps_nothing(self, ddp, value):
        """The false-like strings matter because some config paths forward the flag
        as a raw string, where a bare truthiness test enables checkpointing on a run
        configured to have none."""
        module, events = ddp

        module.DDPManager(activation_checkpointing=value).parallelize(ideogram_model(4))

        assert [e for e in events if e[0] == "wrapped"] == []

    def test_missing_block_list_does_not_raise(self, ddp):
        """A model shape this does not recognize should not take down a run that was
        otherwise fine."""
        module, events = ddp
        model = ideogram_model(with_block_list=False)

        module.DDPManager(activation_checkpointing=True).parallelize(model)

        assert [e for e in events if e[0] == "wrapped"] == []

    def test_stride_applies_on_this_path_too(self, ddp, monkeypatch):
        """The stride describes the model's memory profile, not the sharding
        strategy, so a run that set it and chose DDP should get it. Forwarding it on
        one path only is the kind of divergence that is invisible until two runs
        that should match do not."""
        monkeypatch.setenv("PRIMUS_IDEOGRAM_AC_EVERY", "3")
        module, events = ddp

        module.DDPManager(activation_checkpointing=True).parallelize(ideogram_model(9))

        wrapped = [e[1] for e in events if e[0] == "wrapped"]
        assert wrapped == [0, 3, 6], f"expected every third block, got {wrapped}"

    def test_install_is_idempotent(self, ddp):
        """Patch installation can be reached more than once; wrapping the wrapper
        would double-checkpoint every block."""
        module, events = ddp
        from primus.backends.nemo_automodel.models.ideogram4 import zero1

        assert zero1.install() is True
        module.DDPManager(activation_checkpointing=True).parallelize(ideogram_model(5))

        assert len([e for e in events if e[0] == "wrapped"]) == 5


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
def zero1_installed(monkeypatch):
    """Install both patches with ZeRO-1 on, and hand back the config module."""
    monkeypatch.setenv("PRIMUS_IDEOGRAM_ZERO1", "1")
    install_stub_parallelizer(monkeypatch)
    install_stub_ddp(monkeypatch)
    module = install_stub_optimizer_hierarchy(monkeypatch)

    from primus.backends.nemo_automodel.models.ideogram4 import zero1

    assert zero1.install() is True
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
        """Neither switch set: install does nothing and reports it."""
        install_stub_parallelizer(monkeypatch)
        install_stub_ddp(monkeypatch)

        from primus.backends.nemo_automodel.models.ideogram4 import zero1

        assert zero1.install() is False

    def test_ddp_alone_does_not_touch_the_optimizer(self, monkeypatch):
        """The pure-DDP baseline the ZeRO-1 saving is measured against: real
        checkpointing, replicated optimizer state."""
        monkeypatch.setenv("PRIMUS_IDEOGRAM_DDP", "1")
        install_stub_parallelizer(monkeypatch)
        install_stub_ddp(monkeypatch)
        module = install_stub_optimizer_hierarchy(monkeypatch)

        from primus.backends.nemo_automodel.models.ideogram4 import zero1

        assert zero1.install() is True
        builds = install_fake_zero(monkeypatch)
        result = module.OptimizerConfig().build([1, 2])

        assert builds == []
        assert isinstance(result[0], FakeOptimizer)

    def test_switching_off_after_install_is_inert(self, zero1_installed, monkeypatch):
        """The switch is re-read at call time, not captured at install time, so the
        installed wrapper cannot outlive the request for it."""
        builds = install_fake_zero(monkeypatch)
        monkeypatch.delenv("PRIMUS_IDEOGRAM_ZERO1")

        result = zero1_installed.OptimizerConfig().build([1, 2])

        assert builds == []
        assert isinstance(result[0], FakeOptimizer)

    @pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on"])
    def test_accepted_spellings(self, monkeypatch, value):
        monkeypatch.setenv("PRIMUS_IDEOGRAM_ZERO1", value)
        from primus.backends.nemo_automodel.models.ideogram4 import zero1

        assert zero1.is_zero1_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "off", "no", ""])
    def test_rejected_spellings(self, monkeypatch, value):
        monkeypatch.setenv("PRIMUS_IDEOGRAM_ZERO1", value)
        from primus.backends.nemo_automodel.models.ideogram4 import zero1

        assert zero1.is_zero1_enabled() is False
