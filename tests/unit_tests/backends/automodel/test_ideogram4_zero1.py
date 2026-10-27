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

  3. checkpoint.enabled IS READ AS AUTOMODEL READS IT, so the refusal matches
     whether AutoModel will actually save.

No GPU. Most tests stub the optimizer hierarchy and replace torch's ZeRO class and
rank queries; ``TestRealTorch`` builds a real ZeroRedundancyOptimizer over a real
AdamW on two CPU ranks, and checks the real AutoModel hierarchy where it is
installed.
"""

import json
import os
import socket

import pytest

from tests.unit_tests.backends.automodel.parallelize._support import install_stub_module

OPTIM_PATH = "nemo_automodel.components.optim.optimizer"


@pytest.fixture(autouse=True)
def restore_checkpoint_guard(monkeypatch):
    """install() records checkpoint.enabled in module state."""
    from primus.backends.automodel.models.ideogram4 import zero1

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

    from primus.backends.automodel.models.ideogram4 import zero1

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
        assert result[0] is builds[0]

    def test_the_base_class_is_patched(self, zero1_installed, monkeypatch):
        builds = install_fake_zero(monkeypatch)

        zero1_installed.OptimizerConfig().build([1, 2])

        assert len(builds) == 1

    def test_a_subclass_without_its_own_build_is_covered(self, zero1_installed, monkeypatch):
        """It inherits the patched base, so it must not be double-patched either."""
        builds = install_fake_zero(monkeypatch)

        zero1_installed.InheritingConfig().build([1, 2])

        assert "build" not in vars(zero1_installed.InheritingConfig), "patched a build it does not own"
        assert len(builds) == 1

    def test_a_chaining_subclass_does_not_double_wrap(self, zero1_installed, monkeypatch):
        """Both its own build and the base's are patched, so the value passes
        through the wrapper twice. The guard is the isinstance check."""
        builds = install_fake_zero(monkeypatch)

        result = zero1_installed.ChainingConfig().build([1, 2])

        assert len(builds) == 1, "wrapped twice"
        assert result[0] is builds[0]

    def test_a_subclass_defined_after_install_is_patched(self, zero1_installed, monkeypatch):
        """The walk sees only the classes that exist at install time. A module
        imported later in setup can still add an override, which must not keep a
        build that never shards."""
        builds = install_fake_zero(monkeypatch)

        class LateConfig(zero1_installed.OptimizerConfig):
            def build(self, params):
                return [FakeOptimizer(params, lr=0.3)]

        result = LateConfig().build([1, 2])

        assert len(builds) == 1, "the later subclass was never patched"
        assert result[0] is builds[0]

    def test_reinstalling_does_not_stack_the_subclass_hook(self, zero1_installed, monkeypatch):
        from primus.backends.automodel.models.ideogram4 import zero1

        assert zero1.install(checkpoint_enabled=False) is True
        builds = install_fake_zero(monkeypatch)

        class LateConfig(zero1_installed.OptimizerConfig):
            def build(self, params):
                return [FakeOptimizer(params, lr=0.3)]

        LateConfig().build([1])

        assert len(builds) == 1


class TestOptimizerConstruction:
    def test_the_learning_rate_is_forwarded_with_the_defaults(self, zero1_installed, monkeypatch):
        """ZeRO has no lr parameter of its own; it reaches the per-rank optimizer
        through the forwarded defaults."""
        builds = install_fake_zero(monkeypatch)

        zero1_installed.OptimizerConfig().build([1])

        assert builds[0].kwargs["lr"] == 0.1

    def test_an_absent_learning_rate_is_not_invented(self, monkeypatch):
        """An optimizer whose defaults have no lr must not be handed lr=None."""
        builds = install_fake_zero(monkeypatch)
        from primus.backends.automodel.models.ideogram4 import zero1

        zero1._wrap_in_zero1(FakeOptimizer([1], momentum=0.9))

        assert "lr" not in builds[0].kwargs
        assert builds[0].kwargs["momentum"] == 0.9

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
        from primus.backends.automodel.models.ideogram4 import zero1

        class Narrow:
            def __init__(self, params, lr=0.0, weight_decay=0.0):
                pass

        kept = zero1._constructor_defaults(Narrow, {"weight_decay": 0.1, "decoupled_weight_decay": True})

        assert kept == {"weight_decay": 0.1}

    def test_kwargs_constructors_are_left_alone(self, monkeypatch):
        """A ``**kwargs`` constructor may accept keys its signature does not name,
        so filtering against the signature would drop valid settings."""
        from primus.backends.automodel.models.ideogram4 import zero1

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
        from primus.backends.automodel.models.ideogram4 import zero1

        class FakeDTensor:
            pass

        monkeypatch.setattr(zero1, "_dtensor_class", lambda: FakeDTensor)

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

        from primus.backends.automodel.models.ideogram4 import zero1

        zero1._wrap_in_zero1(base)

        assert [g["lr_mult"] for g in builds[0].params] == [1.0, 0.1]
        assert [g["params"] for g in builds[0].params] == [[1], [2]]

    def test_checkpointing_is_refused_where_zero1_applies(self, monkeypatch, set_option):
        """The first save would otherwise crash after training had started."""
        set_option("primus_ideogram4.zero1", True)
        module = install_stub_optimizer_hierarchy(monkeypatch)
        from primus.backends.automodel.models.ideogram4 import zero1

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
        from primus.backends.automodel.models.ideogram4 import zero1

        zero1.install()
        install_fake_zero(monkeypatch)

        with pytest.raises(RuntimeError, match="checkpoint.enabled"):
            module.OptimizerConfig().build([1, 2])

    def test_checkpointing_is_not_refused_where_zero1_declines(self, monkeypatch, caplog, set_option):
        set_option("primus_ideogram4.zero1", True)
        module = install_stub_optimizer_hierarchy(monkeypatch)
        from primus.backends.automodel.models.ideogram4 import zero1

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

        with pytest.raises(RuntimeError, match="could not build"):
            zero1_installed.OptimizerConfig().build([1, 2])

    def test_an_install_with_zero1_off_clears_the_checkpoint_guard(self, monkeypatch, set_option):
        """The guard is module state; a later run in the same process must not
        inherit an earlier run's checkpoint.enabled."""
        from primus.backends.automodel.models.ideogram4 import zero1

        set_option("primus_ideogram4.zero1", True)
        install_stub_optimizer_hierarchy(monkeypatch)
        zero1.install(checkpoint_enabled=True)
        set_option("primus_ideogram4.zero1", False)

        assert zero1.install(checkpoint_enabled=False) is False
        assert zero1._checkpointing_enabled is False


class TestGating:
    def test_off_by_default(self, monkeypatch):
        """Not set: install does nothing and reports it."""
        module = install_stub_optimizer_hierarchy(monkeypatch)
        before = vars(module.FactoryConfig)["build"]

        from primus.backends.automodel.models.ideogram4 import zero1

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
        from primus.backends.automodel.models.ideogram4 import zero1

        assert zero1.is_zero1_enabled() is True

    @pytest.mark.parametrize("value", [False, 0, "false", "off", "no"])
    def test_rejected_spellings(self, set_option, value):
        set_option("primus_ideogram4.zero1", value)
        from primus.backends.automodel.models.ideogram4 import zero1

        assert zero1.is_zero1_enabled() is False

    # (value, whether AutoModel saves): absent is its default of true; a string is
    # read as its config loader reads it, so "off" and "no" stay truthy strings.
    CHECKPOINT_VALUES = [
        ("<absent>", True),
        (False, False),
        (True, True),
        ("false", False),
        ("False", False),
        ("0", False),
        (0, False),
        ("None", False),
        ("true", True),
        ("1", True),
        ("off", True),
        ("no", True),
    ]

    @pytest.mark.parametrize("value, saves", CHECKPOINT_VALUES)
    def test_the_patch_reads_checkpoint_enabled_as_automodel_does(
        self, monkeypatch, set_option, value, saves
    ):
        """bool() alone would turn the string "false" into true and refuse a run
        that does not save; a Primus-style parse would turn "off" into false and let
        through a run that AutoModel does save, which fails at the first save."""
        import types

        import primus.backends.automodel.patches  # noqa: F401
        from primus.backends.automodel.models.ideogram4 import zero1
        from primus.core.patches import PatchContext
        from primus.core.patches.patch_registry import PatchRegistry

        patch = next(
            p
            for p in PatchRegistry.iter_patches(backend="automodel", phase="before_train")
            if p.id.endswith("zero1")
        )
        seen = []
        monkeypatch.setattr(zero1, "install", lambda checkpoint_enabled=True: seen.append(checkpoint_enabled))
        set_option("primus_ideogram4.zero1", True)

        params = {} if value == "<absent>" else {"checkpoint": types.SimpleNamespace(enabled=value)}
        ctx = PatchContext(
            backend="automodel",
            phase="before_train",
            extra={"module_config": types.SimpleNamespace(params=types.SimpleNamespace(**params))},
        )
        patch.handler(ctx)

        assert seen == [saves]

    @pytest.mark.parametrize(
        "value", [v for v, _ in CHECKPOINT_VALUES if v != "<absent>"] + [" false", "FALSE"]
    )
    def test_the_reading_matches_automodels_loader(self, value):
        """Pins the mirror against the real loader wherever AutoModel is installed."""
        loader = pytest.importorskip("nemo_automodel.components.config.loader")
        from primus.backends.automodel.patches import ideogram4_zero1_patches

        assert ideogram4_zero1_patches.as_automodel_reads_it(value) == loader.translate_value(value)

    @pytest.mark.parametrize(
        "transformer_cls, flag, applies",
        [
            ("Ideogram4Transformer2DModel", True, True),
            (None, True, True),
            ("FluxTransformer2DModel", True, False),
            ("WanTransformer3DModel", True, False),
            ("Ideogram4Transformer2DModel", False, False),
        ],
    )
    def test_the_patch_applies_only_to_ideogram4(self, set_option, transformer_cls, flag, applies):
        """The optimizer patch is process-wide, so the setting copied into another
        model's config must not shard that model's optimizer."""
        import types

        import primus.backends.automodel.patches  # noqa: F401
        from primus.core.patches import PatchContext
        from primus.core.patches.patch_registry import PatchRegistry

        patch = next(
            p
            for p in PatchRegistry.iter_patches(backend="automodel", phase="before_train")
            if p.id.endswith("zero1")
        )
        set_option("primus_ideogram4.zero1", flag)
        spec = (
            types.SimpleNamespace(transformer_cls=transformer_cls)
            if transformer_cls
            else types.SimpleNamespace()
        )
        params = types.SimpleNamespace(model=types.SimpleNamespace(pipeline_spec=spec))
        ctx = PatchContext(
            backend="automodel",
            phase="before_train",
            extra={"module_config": types.SimpleNamespace(params=params)},
        )
        assert patch.condition(ctx) is applies


# --------------------------------------------------------------------------- #
# Real torch: ZeroRedundancyOptimizer over AdamW on two CPU ranks             #
# --------------------------------------------------------------------------- #
def _real_zero_worker(rank, port, out_dir):
    """One rank: wrap a real AdamW, step it, and record what came out."""
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # Both ranks are local; without this gloo picks the interface the hostname
    # resolves to, which a sandboxed runner may not have.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        from primus.backends.automodel.models.ideogram4 import zero1

        def make_params():
            generator = torch.Generator().manual_seed(0)
            shapes = [(8, 4), (4,), (3, 3)]
            return [torch.nn.Parameter(torch.randn(*shape, generator=generator)) for shape in shapes]

        def groups(params):
            return [
                {"params": params[:2], "lr_mult": 1.0},
                {"params": params[2:], "lr_mult": 0.1, "lr": 1e-2},
            ]

        settings = {"lr": 1e-3, "betas": (0.8, 0.9), "weight_decay": 0.01}
        sharded, reference = make_params(), make_params()
        optimizer = zero1._wrap_in_zero1(torch.optim.AdamW(groups(sharded), **settings))
        plain = torch.optim.AdamW(groups(reference), **settings)

        for step in range(3):
            for a, b in zip(sharded, reference):
                grad = torch.sin(torch.arange(a.numel(), dtype=a.dtype) + step).view_as(a)
                a.grad, b.grad = grad.clone(), grad.clone()
            optimizer.step()
            plain.step()

        try:
            optimizer.state_dict()
            refused = False
        except RuntimeError:
            refused = True

        inner = getattr(optimizer, "optim", None)
        result = {
            "is_zero": isinstance(optimizer, ZeroRedundancyOptimizer),
            "inner": type(inner).__name__,
            "lr_mult": [group["lr_mult"] for group in optimizer.param_groups],
            "lr": [group["lr"] for group in optimizer.param_groups],
            "inner_defaults": {
                "lr": inner.defaults["lr"],
                "betas": list(inner.defaults["betas"]),
                "weight_decay": inner.defaults["weight_decay"],
            },
            "matches_plain_adamw": all(torch.allclose(a, b) for a, b in zip(sharded, reference)),
            "state_dict_refused": refused,
        }
        with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as handle:
            json.dump(result, handle)
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestRealTorch:
    """The stubs above pin the wiring; these pin it against the real classes."""

    def test_real_adamw_defaults_rebuild_an_adamw(self):
        """AdamW's defaults carry a key its own constructor rejects; the filtered
        defaults must construct one, at the same settings."""
        import torch

        from primus.backends.automodel.models.ideogram4 import zero1

        base = torch.optim.AdamW(
            [torch.nn.Parameter(torch.zeros(2))], lr=1e-3, betas=(0.8, 0.9), weight_decay=0.01
        )
        kept = zero1._constructor_defaults(torch.optim.AdamW, dict(base.defaults))
        rebuilt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], **kept)

        for key in ("lr", "betas", "weight_decay", "eps"):
            assert rebuilt.defaults[key] == base.defaults[key]

    def test_a_real_zero_optimizer_steps_like_adamw(self, tmp_path):
        """Two gloo ranks on CPU: the wrapper builds a real ZeroRedundancyOptimizer
        that keeps the lr_mult groups, forwards AdamW's settings, steps to the same
        parameters as a plain AdamW, and refuses an unconsolidated state_dict -- the
        reason checkpointing is refused."""
        import torch.distributed as dist
        import torch.multiprocessing as mp

        if not dist.is_available() or not dist.is_gloo_available():
            pytest.skip("needs torch.distributed with gloo")

        mp.spawn(_real_zero_worker, args=(_free_port(), str(tmp_path)), nprocs=2, join=True)

        for rank in range(2):
            result = json.loads((tmp_path / f"rank{rank}.json").read_text())
            assert result["is_zero"] and result["inner"] == "AdamW"
            assert result["lr_mult"] == [1.0, 0.1]
            assert result["lr"] == [1e-3, 1e-2]
            assert result["inner_defaults"] == {"lr": 1e-3, "betas": [0.8, 0.9], "weight_decay": 0.01}
            assert result["matches_plain_adamw"], f"rank {rank} diverged from plain AdamW"
            assert result["state_dict_refused"]

    def test_the_real_hierarchy_includes_the_factory_config(self):
        """The class a plain-torch-optimizer config goes through, found by the walk
        on the installed AutoModel rather than on a stub of it."""
        pytest.importorskip(OPTIM_PATH)
        from primus.backends.automodel.models.ideogram4 import zero1

        names = {cls.__name__ for cls in zero1._optimizer_config_classes()}

        assert {"OptimizerConfig", "OptimizerFromFactoryConfig"} <= names
