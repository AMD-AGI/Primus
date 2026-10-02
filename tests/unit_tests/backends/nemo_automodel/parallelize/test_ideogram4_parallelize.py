###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit and contract tests for the Ideogram-4 block-checkpointing sidecar."""

import types

import pytest

from primus.backends.nemo_automodel.models.ideogram4 import parallelize as ideogram
from tests.unit_tests.backends.nemo_automodel.parallelize._support import (
    FakeMeshContext,
    attach_and_parallelize,
    install_stub_automodel,
    is_checkpoint_wrapped,
    real_mesh_context,
    record_strategy_dispatch,
    requires_automodel,
)

MODEL_NAME = "Ideogram4Transformer2DModel"
PATCH_ID = "nemo_automodel.models.ideogram4.parallelize"


STRIDE = "primus_ideogram4.ac_every"


def _patch():
    import primus.backends.nemo_automodel.patches  # noqa: F401
    from primus.core.patches.patch_registry import PatchRegistry

    return next(
        p
        for p in PatchRegistry.iter_patches(backend="nemo_automodel", phase="before_train")
        if p.id == PATCH_ID
    )


class TestStubbed:
    @pytest.fixture
    def automodel(self, monkeypatch):
        return install_stub_automodel(monkeypatch)

    def test_registered_under_the_model_class_name(self, automodel):
        assert ideogram.install() is True
        assert isinstance(automodel.registry[MODEL_NAME], automodel.ModelParallelizer)

    def test_every_block_is_checkpointed(self, automodel):
        ideogram.install()
        model = types.SimpleNamespace(layers=["b0", "b1", "b2"])
        automodel.registry[MODEL_NAME].parallelize(model, FakeMeshContext(activation_checkpointing=True))
        assert automodel.full == [["b0", "b1", "b2"]]
        (ctx,) = automodel.parallelize
        assert ctx.activation_checkpointing is False

    def test_the_stride_is_read_at_install(self, automodel, set_option):
        set_option(STRIDE, 2)
        ideogram.install()
        set_option(STRIDE, 3)
        model = types.SimpleNamespace(layers=["b0", "b1", "b2", "b3"])
        automodel.registry[MODEL_NAME].parallelize(model, FakeMeshContext(activation_checkpointing=True))
        assert automodel.full == [["b0", "b2"]]

    def test_ac_off_leaves_the_model_alone(self, automodel):
        ideogram.install()
        model = types.SimpleNamespace(layers=["b0"])
        automodel.registry[MODEL_NAME].parallelize(model, FakeMeshContext())
        assert automodel.full == [] and automodel.selective == []

    def test_a_bad_stride_fails_at_install(self, automodel, set_option):
        set_option(STRIDE, "nonsense")
        with pytest.raises(ValueError, match=STRIDE):
            ideogram.install()
        assert MODEL_NAME not in automodel.registry


class TestStride:
    @pytest.mark.parametrize("raw,expected", [(None, 0), ("", 0), (1, 0), (2, 2), ("12", 12)])
    def test_parsing(self, set_option, raw, expected):
        """A stride of 1 means the same as none, so it normalizes to 0."""
        set_option(STRIDE, raw)
        assert ideogram.ac_stride() == expected

    @pytest.mark.parametrize("raw", ["nonsense", "2.5", 2.5, "two", 0, -3, True])
    def test_a_bad_value_is_refused_rather_than_defaulted(self, set_option, raw):
        set_option(STRIDE, raw)
        with pytest.raises(ValueError, match=STRIDE):
            ideogram.ac_stride()


class TestPatchRegistration:
    def test_applies_without_any_opt_in(self):
        assert _patch().condition(None) is True

    def test_a_bad_stride_stops_the_run_at_filtering(self, set_option):
        """Conditions run outside the patch runner's error isolation; apply() does not."""
        set_option(STRIDE, "nonsense")
        with pytest.raises(ValueError, match=STRIDE):
            _patch().condition(None)

    def test_another_model_ignores_the_stride(self, set_option):
        """The setting belongs to Ideogram-4, so a bad value must not stop a
        different model's run."""
        from primus.core.patches import PatchContext

        params = types.SimpleNamespace(
            model=types.SimpleNamespace(
                pipeline_spec=types.SimpleNamespace(transformer_cls="WanTransformer3DModel")
            )
        )
        ctx = PatchContext(
            backend="nemo_automodel",
            phase="before_train",
            extra={"module_config": types.SimpleNamespace(params=params)},
        )
        set_option(STRIDE, "nonsense")
        assert _patch().condition(ctx) is False


def _tiny_ideogram(num_layers=4):
    from diffusers.models.transformers.transformer_ideogram4 import (
        Ideogram4Transformer2DModel,
    )

    return Ideogram4Transformer2DModel(
        in_channels=128,
        num_layers=num_layers,
        attention_head_dim=32,
        num_attention_heads=4,
        intermediate_size=64,
        adaln_dim=32,
        llm_features_dim=16,
        mrope_section=(4, 2, 2),
    )


@requires_automodel
class TestUpstreamContract:
    """The real attach-then-parallelize sequence on a tiny CPU Ideogram-4."""

    @pytest.fixture(autouse=True)
    def isolated_registry(self, monkeypatch):
        from diffusers.models.transformers.transformer_ideogram4 import (
            Ideogram4Transformer2DModel,
        )
        from nemo_automodel._diffusers import parallelization

        monkeypatch.setattr(parallelization, "_PARALLELIZERS", dict(parallelization._PARALLELIZERS))
        monkeypatch.setattr(Ideogram4Transformer2DModel, "parallelizer", None, raising=False)

    def test_upstream_still_has_no_ideogram_sidecar(self):
        """If this fails, upstream now ships one and becomes the sidecar's base;
        re-check whether this module is still needed."""
        from nemo_automodel._diffusers import parallelization

        assert MODEL_NAME not in parallelization._PARALLELIZERS

    @pytest.mark.parametrize("strategy", ["fsdp2", "ddp"])
    def test_whole_blocks_are_wrapped_and_upstream_sees_ac_off(self, monkeypatch, strategy):
        seen = record_strategy_dispatch(monkeypatch)
        assert ideogram.install() is True
        model = _tiny_ideogram()
        attach_and_parallelize(model, real_mesh_context(strategy, True))

        assert all(is_checkpoint_wrapped(b) for b in model.layers)
        ((dispatched, ctx),) = seen
        assert dispatched == strategy
        assert ctx.activation_checkpointing is False
        assert not getattr(ctx.strategy_config, "activation_checkpointing", False)

    def test_the_stride_wraps_every_nth_block(self, monkeypatch, set_option):
        record_strategy_dispatch(monkeypatch)
        set_option(STRIDE, 2)
        ideogram.install()
        model = _tiny_ideogram()
        attach_and_parallelize(model, real_mesh_context("fsdp2", True))
        assert [is_checkpoint_wrapped(b) for b in model.layers] == [True, False, True, False]

    def test_ac_off_leaves_the_model_alone(self, monkeypatch):
        seen = record_strategy_dispatch(monkeypatch)
        ideogram.install()
        model = _tiny_ideogram()
        attach_and_parallelize(model, real_mesh_context("fsdp2", False))
        assert not any(is_checkpoint_wrapped(b) for b in model.layers)
        assert len(seen) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
