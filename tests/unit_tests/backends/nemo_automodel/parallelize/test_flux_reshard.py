###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the real-AC FLUX parallelization strategy.

The FLUX bug is different in kind from the Wan one: FluxTransformer2DModel is not
in AutoModel's strategy registry at all, so it falls back to the default strategy,
which checkpoints by wrapping `self_attn` and `mlp` submodules that a FLUX block
does not have. `activation_checkpointing: true` is therefore accepted and does
nothing. What is worth asserting is that blocks are genuinely wrapped and that
both of FLUX's two block lists are covered -- the failure mode is a plausible
looking run with a silently unchanged memory ceiling.

The reshard obligation is identical to every other strategy's, so it comes from
the shared contract rather than being restated here. This is the contract's
second consumer, and unlike Wan this strategy is written from scratch, so it is
also held to the keyword-passing rule that the delegating strategies are not.

No GPU and no real AutoModel: the parallelizer module is stubbed.
"""

import importlib
import sys
import types

import pytest

from tests.unit_tests.backends.nemo_automodel.parallelize import (
    _reshard_contract as contract,
)

MODULE_PATH = "primus.backends.nemo_automodel.models.flux.parallelize"
MODEL_NAME = "FluxTransformer2DModel"
DUAL, SINGLE = "transformer_blocks", "single_transformer_blocks"


@pytest.fixture
def flux(monkeypatch):
    monkeypatch.setenv("PRIMUS_FLUX_REAL_AC", "1")
    parallelizer, calls = contract.install_stub_parallelizer(monkeypatch)

    # The strategy imports torch only for MixedPrecisionPolicy dtypes; stub it so
    # the suite runs without torch installed.
    contract.install_stub_module(monkeypatch, "torch", bfloat16="bfloat16", float32="float32")

    module = importlib.import_module(MODULE_PATH)
    monkeypatch.setattr(module, "P", parallelizer, raising=False)

    assert module.install() is True
    strategy = parallelizer.PARALLELIZATION_STRATEGIES[MODEL_NAME]()
    return types.SimpleNamespace(module=module, parallelizer=parallelizer, strategy=strategy, calls=calls)


@pytest.fixture(autouse=True)
def reshard_installed(monkeypatch):
    from primus.backends.nemo_automodel.distributed import fsdp2_reshard

    monkeypatch.setattr(fsdp2_reshard, "patch_installed", True, raising=False)
    monkeypatch.setattr(fsdp2_reshard, "applied_reshard_after_forward", False, raising=False)


class TestTheSharedContract:
    """Second consumer of the contract; the first was Wan."""

    @pytest.mark.parametrize("value", [False, True, None])
    def test_reshard_after_forward_reaches_the_sharding_helper(self, flux, value):
        contract.assert_forwards_reshard(flux.strategy, flux.calls, DUAL, value)

    def test_root_unit_still_keeps_its_params_unsharded(self, flux):
        contract.assert_root_unit_stays_unsharded(flux.strategy, flux.calls, DUAL)

    def test_sharding_arguments_are_passed_by_keyword(self, flux):
        """Applies here but not to Wan: this strategy owns its call site, so
        positional drift is preventable rather than inherited."""
        contract.assert_sharding_args_are_keyword(flux.strategy, flux.calls, DUAL)


class TestRegistration:
    def test_the_strategy_is_registered_for_flux(self, flux):
        assert MODEL_NAME in flux.parallelizer.PARALLELIZATION_STRATEGIES

    def test_installing_twice_is_idempotent(self, flux):
        before = flux.parallelizer.PARALLELIZATION_STRATEGIES[MODEL_NAME]
        assert flux.module.install() is True
        assert flux.parallelizer.PARALLELIZATION_STRATEGIES[MODEL_NAME] is before

    def test_no_reregistration_wrapper_is_needed(self, flux):
        """FLUX is not in the list _init_parallelizer resets, so unlike Wan this
        module must not be patching the pipeline. Pin it, so nobody adds one by
        analogy."""
        assert not hasattr(flux.module, "_INIT_PATCH_SENTINEL")


class TestActivationCheckpointing:
    def _run(self, flux, monkeypatch, ac_value, dual_n=2, single_n=3):
        selective, wrapped = [], []
        monkeypatch.setattr(
            flux.parallelizer,
            "apply_selective_checkpointing_to_layers",
            lambda model, layers, kv, **kw: selective.append(list(layers)),
        )
        monkeypatch.setattr(flux.parallelizer, "checkpoint_wrapper", lambda b, **kw: wrapped.append(b) or b)
        model = types.SimpleNamespace(
            **{
                DUAL: [f"d{i}" for i in range(dual_n)],
                SINGLE: [f"s{i}" for i in range(single_n)],
            }
        )
        flux.strategy.parallelize(
            model=model,
            device_mesh=contract.FakeMesh(),
            activation_checkpointing=ac_value,
            reshard_after_forward=False,
        )
        return selective, wrapped

    def test_full_wraps_both_block_lists(self, flux, monkeypatch):
        """The whole point: the default strategy wrapped neither."""
        selective, wrapped = self._run(flux, monkeypatch, "full", dual_n=2, single_n=3)
        assert not selective
        assert wrapped == ["d0", "d1", "s0", "s1", "s2"]

    def test_selective_covers_both_lists_in_one_call(self, flux, monkeypatch):
        selective, wrapped = self._run(flux, monkeypatch, "selective", dual_n=2, single_n=3)
        assert len(selective) == 1
        assert selective[0] == ["d0", "d1", "s0", "s1", "s2"]
        assert not wrapped

    @pytest.mark.parametrize("ac_value", ["false", "off", "none", "", False])
    def test_false_like_values_checkpoint_nothing(self, flux, monkeypatch, ac_value):
        selective, wrapped = self._run(flux, monkeypatch, ac_value)
        assert not selective and not wrapped


class TestTensorParallelWarning:
    def test_a_tp_mesh_warns_rather_than_pretending(self, flux, caplog):
        """There is no FLUX TP plan here; proceeding silently would look like TP
        was applied."""

        class TPMesh(contract.FakeMesh):
            mesh_dim_names = ("tp",)

            def __getitem__(self, name):
                return types.SimpleNamespace(size=lambda: 8)

        with caplog.at_level("WARNING"):
            flux.strategy.parallelize(
                model=types.SimpleNamespace(**{DUAL: [], SINGLE: []}),
                device_mesh=TPMesh(),
                reshard_after_forward=False,
            )
        assert any("TP plan" in r.message for r in caplog.records)


class TestPatchRegistration:
    def test_registered_and_gated(self, monkeypatch):
        import primus.backends.nemo_automodel.patches  # noqa: F401
        from primus.core.patches.patch_registry import PatchRegistry

        patch = next(
            (
                p
                for p in PatchRegistry.iter_patches(backend="nemo_automodel", phase="before_train")
                if p.id == "nemo_automodel.models.flux.parallelize"
            ),
            None,
        )
        assert patch is not None
        monkeypatch.delenv("PRIMUS_FLUX_REAL_AC", raising=False)
        assert patch.condition(None) is False
        monkeypatch.setenv("PRIMUS_FLUX_REAL_AC", "1")
        assert patch.condition(None) is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
