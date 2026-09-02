###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the repaired Wan 2.2 parallelization strategy.

Three separate failures are covered, because the module fixes three things and
each fails silently on its own:

  1. ``reshard_after_forward`` reaching the sharding helper -- the shared
     contract in ``_reshard_contract.py``.
  2. ``activation_checkpointing: "selective"`` taking the selective branch. In
     the in-tree strategy this is a bare truthiness test, so the non-empty string
     "selective" runs the *full* AC branch and the two settings are
     byte-identical in effect. The same test also enables AC for the string
     "false".
  3. The registry entry surviving ``_init_parallelizer``, which unconditionally
     resets it to the in-tree instance every time the pipeline parallelizes. A
     registration that does not survive it is never read, so this is the
     difference between the repair working and doing nothing at all.

No GPU and no real AutoModel: the parallelizer and pipeline modules are stubbed.
"""

import importlib
import sys
import types

import pytest

from tests.unit_tests.backends.nemo_automodel.parallelize import (
    _reshard_contract as contract,
)

MODULE_PATH = "primus.backends.nemo_automodel.models.wan.parallelize"
PIPELINE_PATH = "nemo_automodel._diffusers.auto_diffusion_pipeline"
MODEL_NAME = "WanTransformer3DModel"
BLOCK_ATTR = "blocks"


def _add_in_tree_wan_strategy(parallelizer):
    """Model the in-tree strategy the repair subclasses, including its bug.

    Faithfulness matters in one specific way: it must call
    ``apply_fsdp2_sharding_recursively`` with seven positional arguments, because
    that is exactly what the repair works around. A stub that passed keywords
    would let a broken repair pass.
    """

    class WanParallelizationStrategy(parallelizer.ParallelizationStrategy):
        def parallelize(
            self,
            model,
            device_mesh,
            mp_policy=None,
            offload_policy=None,
            sequence_parallel=False,
            activation_checkpointing=False,
            tp_shard_plan=None,
            **kwargs,
        ):
            if activation_checkpointing and hasattr(model, BLOCK_ATTR):
                blocks = getattr(model, BLOCK_ATTR)
                for idx in range(len(blocks)):
                    blocks[idx] = parallelizer.checkpoint_wrapper(
                        blocks[idx],
                        checkpoint_impl=parallelizer.CheckpointImpl.NO_REENTRANT,
                    )
            # Seven positional arguments, one short of reshard_after_forward.
            parallelizer.apply_fsdp2_sharding_recursively(
                model,
                device_mesh,
                mp_policy,
                offload_policy,
                enable_fsdp2_prefetch=True,
                fsdp2_backward_prefetch_depth=1,
                fsdp2_forward_prefetch_depth=1,
            )
            parallelizer.fully_shard(model, reshard_after_forward=False)
            return model

    parallelizer.WanParallelizationStrategy = WanParallelizationStrategy
    parallelizer.PARALLELIZATION_STRATEGIES[MODEL_NAME] = WanParallelizationStrategy()
    return WanParallelizationStrategy


@pytest.fixture
def wan(monkeypatch):
    """Install the repaired strategy against stubs and hand back the pieces."""
    monkeypatch.setenv("PRIMUS_WAN_PARALLELIZE_FIX", "1")

    parallelizer, calls = contract.install_stub_parallelizer(monkeypatch)
    in_tree_cls = _add_in_tree_wan_strategy(parallelizer)

    def _init_parallelizer():
        # The behaviour that makes a one-shot registration useless.
        parallelizer.PARALLELIZATION_STRATEGIES[MODEL_NAME] = in_tree_cls()

    pipeline = contract.install_stub_module(monkeypatch, PIPELINE_PATH, _init_parallelizer=_init_parallelizer)

    module = importlib.import_module(MODULE_PATH)
    # The module caches the built subclass in a global; clear it so each test
    # builds against its own stub rather than the previous test's.
    monkeypatch.setattr(module, "_STRATEGY_CLS", None, raising=False)

    assert module.install() is True
    strategy = parallelizer.PARALLELIZATION_STRATEGIES[MODEL_NAME]
    return types.SimpleNamespace(
        module=module,
        parallelizer=parallelizer,
        pipeline=pipeline,
        strategy=strategy,
        calls=calls,
        in_tree_cls=in_tree_cls,
    )


@pytest.fixture
def reshard_installed(monkeypatch):
    """Pretend the sibling FSDP2 repair is in place, so no error path is taken."""
    from primus.backends.nemo_automodel.distributed import fsdp2_reshard

    monkeypatch.setattr(fsdp2_reshard, "patch_installed", True, raising=False)
    monkeypatch.setattr(fsdp2_reshard, "applied_reshard_after_forward", False, raising=False)


pytestmark = pytest.mark.usefixtures("reshard_installed")


class TestTheSharedContract:
    @pytest.mark.parametrize("value", [False, True, None])
    def test_reshard_after_forward_reaches_the_sharding_helper(self, wan, value):
        contract.assert_forwards_reshard(wan.strategy, wan.calls, BLOCK_ATTR, value)

    def test_root_unit_still_keeps_its_params_unsharded(self, wan):
        contract.assert_root_unit_stays_unsharded(wan.strategy, wan.calls, BLOCK_ATTR)

    def test_the_in_tree_parent_would_fail_the_contract(self, wan):
        """Guards the guard: if the stub parent were fixed, the tests above would
        pass with no repair at all and would stop meaning anything."""
        wan.parallelizer.PARALLELIZATION_STRATEGIES[MODEL_NAME] = wan.in_tree_cls()
        with pytest.raises(AssertionError, match="dropped before the sharding helper"):
            contract.assert_forwards_reshard(
                wan.parallelizer.PARALLELIZATION_STRATEGIES[MODEL_NAME],
                wan.calls,
                BLOCK_ATTR,
                False,
            )


class TestActivationCheckpointing:
    def _parallelize_with_ac(self, wan, monkeypatch, ac_value, num_blocks=3):
        selective, full = [], []
        monkeypatch.setattr(
            wan.parallelizer,
            "apply_selective_checkpointing_to_layers",
            lambda *a, **kw: selective.append((a, kw)),
        )
        monkeypatch.setattr(
            wan.parallelizer,
            "checkpoint_wrapper",
            lambda block, **kw: full.append(block) or block,
        )
        model = contract.model_with_blocks(BLOCK_ATTR, [object() for _ in range(num_blocks)])
        wan.strategy.parallelize(
            model=model,
            device_mesh=contract.FakeMesh(),
            activation_checkpointing=ac_value,
            reshard_after_forward=False,
        )
        return selective, full

    def test_selective_takes_the_selective_branch(self, wan, monkeypatch):
        """The bug: "selective" is truthy, so in-tree it ran FULL AC instead."""
        selective, full = self._parallelize_with_ac(wan, monkeypatch, "selective")
        assert len(selective) == 1, "selective AC machinery was not used"
        assert not full, "blocks were also wrapped with full AC"

    def test_full_wraps_every_block(self, wan, monkeypatch):
        selective, full = self._parallelize_with_ac(wan, monkeypatch, "full", num_blocks=3)
        assert not selective
        assert len(full) == 3

    @pytest.mark.parametrize("ac_value", ["false", "False", "off", "no", "none", "", False])
    def test_false_like_strings_do_not_enable_ac(self, wan, monkeypatch, ac_value):
        """A non-empty "false" is truthy in Python; treating it as on is the same
        class of bug as treating "selective" as full."""
        selective, full = self._parallelize_with_ac(wan, monkeypatch, ac_value)
        assert not selective and not full

    def test_the_parent_is_never_asked_to_checkpoint_again(self, wan, monkeypatch):
        """AC is resolved before delegating, so the parent must see a falsy value
        or every block gets wrapped twice."""
        seen = {}
        original = wan.in_tree_cls.parallelize

        def spy(self, model, device_mesh, **kwargs):
            seen.update(kwargs)
            return original(self, model, device_mesh, **kwargs)

        monkeypatch.setattr(wan.in_tree_cls, "parallelize", spy)
        self._parallelize_with_ac(wan, monkeypatch, "full")
        assert not seen.get("activation_checkpointing")


class TestSurvivingTheRegistryReset:
    def test_the_repaired_strategy_is_installed(self, wan):
        assert isinstance(wan.strategy, wan.module._STRATEGY_CLS)

    def test_it_survives_init_parallelizer(self, wan):
        """_init_parallelizer resets the entry every time the pipeline runs; if the
        repair does not come back afterwards it is never read."""
        wan.pipeline._init_parallelizer()
        restored = wan.parallelizer.PARALLELIZATION_STRATEGIES[MODEL_NAME]
        assert isinstance(restored, wan.module._STRATEGY_CLS)

    def test_reinstalling_does_not_nest_pipeline_wrappers(self, wan):
        first = wan.pipeline._init_parallelizer
        wan.module.install()
        assert wan.pipeline._init_parallelizer is first

    def test_the_subclass_is_built_once(self, wan):
        """Rebuilding per call would eventually subclass our own subclass, since
        _init_parallelizer keeps putting the in-tree instance back."""
        cls = wan.module._STRATEGY_CLS
        wan.pipeline._init_parallelizer()
        wan.module.install()
        assert wan.module._STRATEGY_CLS is cls

    def test_the_in_tree_tp_plan_is_still_authoritative(self, wan):
        """Subclassing rather than reimplementing is the whole forward-compat
        argument; assert the inheritance actually holds."""
        assert issubclass(wan.module._STRATEGY_CLS, wan.in_tree_cls)


class TestMissingSiblingRepair:
    def test_it_logs_an_error_but_still_runs(self, wan, monkeypatch, caplog):
        """Fixing either half alone changes nothing, so say so loudly -- but do not
        take down a training run that is already under way."""
        from primus.backends.nemo_automodel.distributed import fsdp2_reshard

        monkeypatch.setattr(fsdp2_reshard, "patch_installed", False, raising=False)
        with caplog.at_level("ERROR"):
            wan.strategy.parallelize(
                model=contract.model_with_blocks(BLOCK_ATTR),
                device_mesh=contract.FakeMesh(),
                reshard_after_forward=False,
            )
        assert any("reshard repair" in r.message for r in caplog.records)


class TestPatchRegistration:
    def test_the_patch_is_registered_and_gated(self, monkeypatch):
        import primus.backends.nemo_automodel.patches  # noqa: F401
        from primus.core.patches.patch_registry import PatchRegistry

        patch = next(
            (
                p
                for p in PatchRegistry.iter_patches(backend="nemo_automodel", phase="before_train")
                if p.id == "nemo_automodel.models.wan.parallelize"
            ),
            None,
        )
        assert patch is not None, "the Wan parallelize patch was not discovered"

        monkeypatch.delenv("PRIMUS_WAN_PARALLELIZE_FIX", raising=False)
        assert patch.condition(None) is False
        monkeypatch.setenv("PRIMUS_WAN_PARALLELIZE_FIX", "1")
        assert patch.condition(None) is True

    def test_it_runs_after_the_reshard_repair(self):
        """It depends on that repair, and logs an error if it ran first and found
        it absent -- so the ordering is part of the design, not incidental."""
        import primus.backends.nemo_automodel.patches  # noqa: F401
        from primus.core.patches.patch_registry import PatchRegistry

        by_id = {p.id: p for p in PatchRegistry.iter_patches(backend="nemo_automodel", phase="before_train")}
        assert (
            by_id["nemo_automodel.distributed.fsdp2_reshard"].priority
            < by_id["nemo_automodel.models.wan.parallelize"].priority
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
