###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit and contract tests for the FLUX block-checkpointing sidecar."""

import types

import pytest

from primus.backends.nemo_automodel.models.flux import parallelize as flux
from tests.unit_tests.backends.nemo_automodel.parallelize._support import (
    FakeMeshContext,
    attach_and_parallelize,
    install_stub_automodel,
    is_checkpoint_wrapped,
    real_mesh_context,
    record_strategy_dispatch,
    requires_automodel,
    tiny_flux,
)


class TestStubbed:
    @pytest.fixture
    def automodel(self, monkeypatch):
        return install_stub_automodel(monkeypatch)

    def test_registered_under_the_flux_class_name(self, automodel):
        assert flux.install() is True
        assert isinstance(automodel.registry["FluxTransformer2DModel"], automodel.ModelParallelizer)

    def test_both_block_lists_are_checkpointed_in_order(self, automodel):
        flux.install()
        model = types.SimpleNamespace(transformer_blocks=["d0", "d1"], single_transformer_blocks=["s0"])
        automodel.registry["FluxTransformer2DModel"].parallelize(
            model, FakeMeshContext(activation_checkpointing=True)
        )
        assert automodel.full == [["d0", "d1", "s0"]]


class TestPatchRegistration:
    def test_registered_and_gated_on_the_model(self):
        import primus.backends.nemo_automodel.patches  # noqa: F401
        from primus.backends.nemo_automodel.patches._conditions import transformer_is
        from primus.core.patches.patch_registry import PatchRegistry

        patch = next(
            p
            for p in PatchRegistry.iter_patches(backend="nemo_automodel", phase="before_train")
            if p.id == "nemo_automodel.models.flux.parallelize"
        )
        assert patch.condition.__name__ == transformer_is("FluxTransformer2DModel").__name__


@requires_automodel
class TestUpstreamContract:
    """The real attach-then-parallelize sequence on a tiny CPU FLUX."""

    @pytest.fixture(autouse=True)
    def isolated_registry(self, monkeypatch):
        from diffusers import FluxTransformer2DModel
        from nemo_automodel._diffusers import parallelization

        monkeypatch.setattr(parallelization, "_PARALLELIZERS", dict(parallelization._PARALLELIZERS))
        monkeypatch.setattr(FluxTransformer2DModel, "parallelizer", None, raising=False)

    @pytest.mark.parametrize("strategy", ["fsdp2", "ddp"])
    def test_every_block_is_wrapped_and_upstream_sees_ac_off(self, monkeypatch, strategy):
        seen = record_strategy_dispatch(monkeypatch)
        assert flux.install() is True
        model = tiny_flux()
        attach_and_parallelize(model, real_mesh_context(strategy, True))

        blocks = [*model.transformer_blocks, *model.single_transformer_blocks]
        assert len(blocks) == 5 and all(is_checkpoint_wrapped(b) for b in blocks)
        ((dispatched, ctx),) = seen
        assert dispatched == strategy
        assert ctx.activation_checkpointing is False
        assert not getattr(ctx.strategy_config, "activation_checkpointing", False)

    def test_ac_off_leaves_the_model_alone(self, monkeypatch):
        seen = record_strategy_dispatch(monkeypatch)
        flux.install()
        model = tiny_flux()
        attach_and_parallelize(model, real_mesh_context("fsdp2", False))
        assert not any(is_checkpoint_wrapped(b) for b in model.transformer_blocks)
        assert len(seen) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
