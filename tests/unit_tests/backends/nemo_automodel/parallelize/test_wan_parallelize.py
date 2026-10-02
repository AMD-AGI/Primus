###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit and contract tests for the Wan sidecar repair."""

import types

import pytest

from primus.backends.nemo_automodel.models.wan import parallelize as wan
from tests.unit_tests.backends.nemo_automodel.parallelize._support import (
    install_stub_automodel,
    is_checkpoint_wrapped,
    requires_automodel,
    tiny_wan,
)


@pytest.fixture
def automodel(monkeypatch):
    rec = install_stub_automodel(monkeypatch)
    rec.registry["WanTransformer3DModel"] = rec.WanModelParallelizer()
    wan.install()
    rec.sidecar = rec.registry["WanTransformer3DModel"]
    return rec


def _model():
    return types.SimpleNamespace(blocks=["b0", "b1"])


class TestInstall:
    def test_upstream_wan_sidecar_is_the_base(self, automodel):
        assert isinstance(automodel.sidecar, automodel.WanModelParallelizer)

    def test_install_is_idempotent(self, automodel):
        wan.install()
        assert automodel.registry["WanTransformer3DModel"] is automodel.sidecar


class TestActivationCheckpointing:
    def test_selective_takes_the_selective_branch(self, automodel):
        automodel.sidecar._apply(_model(), activation_checkpointing="selective")
        assert automodel.selective == [["b0", "b1"]]
        assert automodel.apply[-1]["activation_checkpointing"] is False, "parent must not wrap again"
        assert automodel.full == []

    def test_full_is_left_to_the_parent(self, automodel):
        automodel.sidecar._apply(_model(), activation_checkpointing=True)
        assert automodel.full == [["b0", "b1"]]
        assert automodel.selective == []

    @pytest.mark.parametrize("raw", ["false", "off", "0"])
    def test_false_like_strings_do_not_enable_ac(self, automodel, raw):
        automodel.sidecar._apply(_model(), activation_checkpointing=raw)
        assert automodel.apply[-1]["activation_checkpointing"] is False
        assert automodel.full == [] and automodel.selective == []


class TestReshardAfterForward:
    @pytest.mark.parametrize("value", [False, True])
    def test_value_reaches_the_sharding_call(self, automodel, value):
        automodel.sidecar._apply(_model(), reshard_after_forward=value)
        _args, kwargs = automodel.sharding[-1]
        assert kwargs["reshard_after_forward"] is value

    def test_none_leaves_the_call_untouched(self, automodel):
        automodel.sidecar._apply(_model(), reshard_after_forward=None)
        _args, kwargs = automodel.sharding[-1]
        assert "reshard_after_forward" not in kwargs

    def test_sharding_helper_is_restored(self, automodel):
        original = automodel.registry_module.apply_fsdp2_sharding_recursively
        automodel.sidecar._apply(_model(), reshard_after_forward=False)
        assert automodel.registry_module.apply_fsdp2_sharding_recursively is original

    def test_an_upstream_fix_wins(self, automodel, monkeypatch, caplog):
        """Once the parent passes the value itself, the repair must not override it."""
        module = automodel.registry_module

        def fixed_apply(self, model, reshard_after_forward=None, **kwargs):
            module.apply_fsdp2_sharding_recursively(
                model, None, None, None, True, 2, 1, reshard_after_forward
            )
            return model

        monkeypatch.setattr(automodel.WanModelParallelizer, "_apply", fixed_apply)
        fixed_apply.__module__ = module.__name__
        with caplog.at_level("INFO"):
            automodel.sidecar._apply(_model(), reshard_after_forward=True)
        args, kwargs = automodel.sharding[-1]
        assert args[7] is True and "reshard_after_forward" not in kwargs
        assert "nothing to forward" in caplog.text

    def test_a_parent_that_stops_calling_the_helper_is_reported(self, automodel, monkeypatch, caplog):
        def no_sharding(self, model, **kwargs):
            return model

        no_sharding.__module__ = automodel.registry_module.__name__
        monkeypatch.setattr(automodel.WanModelParallelizer, "_apply", no_sharding)
        with caplog.at_level("WARNING"):
            automodel.sidecar._apply(_model(), reshard_after_forward=False)
        assert "re-check this repair" in caplog.text


class TestPatchRegistration:
    def test_registered_and_gated_on_the_model(self):
        import primus.backends.nemo_automodel.patches  # noqa: F401
        from primus.backends.nemo_automodel.patches._conditions import transformer_is
        from primus.core.patches.patch_registry import PatchRegistry

        patch = next(
            p
            for p in PatchRegistry.iter_patches(backend="nemo_automodel", phase="before_train")
            if p.id == "nemo_automodel.models.wan.parallelize"
        )
        assert patch.condition.__name__ == transformer_is("WanTransformer3DModel").__name__


@requires_automodel
class TestUpstreamContract:
    """Upstream's real WanModelParallelizer with only fully_shard recorded."""

    @pytest.fixture
    def sidecar(self, monkeypatch):
        from nemo_automodel._diffusers import parallelization

        monkeypatch.setattr(parallelization, "_PARALLELIZERS", dict(parallelization._PARALLELIZERS))
        assert wan.install() is True
        instance = parallelization._PARALLELIZERS["WanTransformer3DModel"]
        assert isinstance(instance, parallelization.WanModelParallelizer)

        calls = []
        monkeypatch.setattr(
            parallelization,
            "apply_fsdp2_sharding_recursively",
            lambda *a, **kw: calls.append((a, kw)),
        )
        monkeypatch.setattr(parallelization, "fully_shard", lambda model, **kw: model)
        monkeypatch.setattr(parallelization, "get_fsdp_dp_mesh", lambda *a, **kw: None)
        instance.calls = calls
        return instance

    @staticmethod
    def mesh():
        return {"tp": types.SimpleNamespace(size=lambda: 1)}

    def test_selective_and_reshard_both_take_effect(self, sidecar):
        from nemo_automodel.components.distributed.activation_checkpointing import (
            SELECTIVE_AC_WRAPPER_FLAG,
        )

        model = tiny_wan()
        sidecar._apply(
            model,
            self.mesh(),
            activation_checkpointing="selective",
            reshard_after_forward=False,
            enable_compile=False,
        )
        assert all(getattr(b, SELECTIVE_AC_WRAPPER_FLAG, False) for b in model.blocks)
        (_args, kwargs) = sidecar.calls[-1]
        assert kwargs["reshard_after_forward"] is False

    def test_full_stays_upstreams(self, sidecar):
        from nemo_automodel.components.distributed.activation_checkpointing import (
            SELECTIVE_AC_WRAPPER_FLAG,
        )

        model = tiny_wan()
        sidecar._apply(model, self.mesh(), activation_checkpointing=True)
        assert all(is_checkpoint_wrapped(b) for b in model.blocks)
        assert not any(getattr(b, SELECTIVE_AC_WRAPPER_FLAG, False) for b in model.blocks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
