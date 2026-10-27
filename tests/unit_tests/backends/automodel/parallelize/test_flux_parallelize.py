###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit and contract tests for the FLUX block-checkpointing sidecar."""

import types

import pytest

from primus.backends.automodel.models.flux import parallelize as flux
from tests.unit_tests.backends.automodel.parallelize._support import (
    FakeMeshContext,
    attach_and_parallelize,
    install_stub_automodel,
    is_checkpoint_wrapped,
    patch_context,
    real_mesh_context,
    record_strategy_dispatch,
    registered_patch,
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

    def test_it_steps_aside_under_tensor_parallelism(self, automodel, caplog):
        """Wrapping before upstream's TP plan runs would break the plan's paths."""
        flux.install()
        model = types.SimpleNamespace(transformer_blocks=["d0"], single_transformer_blocks=["s0"])
        context = FakeMeshContext(activation_checkpointing=True, tp_size=2)
        with caplog.at_level("WARNING"):
            automodel.registry["FluxTransformer2DModel"].parallelize(model, context)
        assert automodel.full == []
        assert automodel.parallelize[-1] is context, "upstream gets the request unchanged"
        assert "tp_size=2" in caplog.text


class TestPatchRegistration:
    @pytest.mark.parametrize(
        "transformer_cls, runs",
        [("FluxTransformer2DModel", True), ("WanTransformer3DModel", False), (None, True)],
    )
    def test_it_runs_unless_another_model_is_configured(self, transformer_cls, runs):
        patch = registered_patch("automodel.models.flux.parallelize")
        assert patch.condition(patch_context(transformer_cls)) is runs


class TestGuidancePatchRegistration:
    """Here rather than in test_flux_guidance.py, which needs diffusers and AutoModel."""

    @pytest.mark.parametrize(
        "transformer_cls, runs",
        [("FluxTransformer2DModel", True), ("WanTransformer3DModel", False), (None, True)],
    )
    def test_it_runs_unless_another_model_is_configured(self, transformer_cls, runs):
        patch = registered_patch("automodel.models.flux.guidance")
        assert patch.condition(patch_context(transformer_cls)) is runs


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
