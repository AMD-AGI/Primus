###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the Ideogram-4 flow-matching adapter.

THE THREE CLAIMS WORTH DEFENDING, in order of how expensive they are to get wrong:

  1. THE SIGN. The transformer predicts ``x0 - eps`` while AutoModel's target is
     ``eps - x0``, so the adapter negates. Getting this wrong yields a run that
     trains, converges, and optimizes the exact opposite of the objective. No
     shape check catches it, so it is pinned directly.

  2. THE RESERVED PAD COLUMN. One always-pad position, added only when the packing
     is actually going to be built. It is what makes every row contribute exactly
     two segments, and therefore what makes the packing's shape independent of the
     data. Drop it and a full-width caption produces a one-segment row, the shape
     varies per batch, and the compiled graph is rebuilt continuously.

  3. PUBLISH BEFORE FORWARD. The processor reads the packing off the module during
     the forward, and again during the backward recompute. Publishing after the
     model call would leave the first step reading nothing. The ordering is
     asserted by recording it, because nothing about the code's appearance makes
     the requirement visible.

Also covered: the refusal of a ragged batch under assume_dense, which is the only
cheap place to catch a flag combination that would otherwise let padding attend;
and that install is additive, since it wraps a factory every other model shares.

AutoModel and diffusers are faked. The adapter is built through a factory
specifically so this is possible without the full stack.
"""

import sys
import types

import pytest

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------- #
# Fake AutoModel                                                              #
# --------------------------------------------------------------------------- #
class FakeModelAdapter:
    """Stand-in for AutoModel's ModelAdapter base."""

    def post_process_prediction(self, out):
        return out[0] if isinstance(out, tuple) else out


@pytest.fixture
def fake_automodel(monkeypatch):
    """Install a fake AutoModel adapter base and adapter factory."""
    created = []

    def create_adapter(adapter_type, **kwargs):
        created.append((adapter_type, kwargs))
        return f"stock:{adapter_type}"

    base_mod = types.ModuleType("nemo_automodel.components.flow_matching.adapters.base")
    base_mod.ModelAdapter = FakeModelAdapter
    pipeline_mod = types.ModuleType("nemo_automodel.components.flow_matching.pipeline")
    pipeline_mod.create_adapter = create_adapter

    for name in (
        "nemo_automodel",
        "nemo_automodel.components",
        "nemo_automodel.components.flow_matching",
        "nemo_automodel.components.flow_matching.adapters",
        "nemo_automodel.components.flow_matching.adapters.base",
        "nemo_automodel.components.flow_matching.pipeline",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.flow_matching.adapters.base", base_mod)
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.flow_matching.pipeline", pipeline_mod)

    from primus.backends.nemo_automodel.models.ideogram4 import adapter as adapter_mod

    # The class is cached for identity stability, so reset it between tests.
    monkeypatch.setattr(adapter_mod, "_ADAPTER_CLS", None)
    return types.SimpleNamespace(pipeline=pipeline_mod, created=created, module=adapter_mod)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "PRIMUS_IDEOGRAM_VARLEN_ATTN",
        "PRIMUS_IDEOGRAM_PRECOMPUTE_CU_SEQLENS",
        "PRIMUS_IDEOGRAM_ATTN_ASSUME_DENSE",
    ):
        monkeypatch.delenv(var, raising=False)


CHANNELS = 128
FEATURES = 32


def make_context(text_lengths, grid=(2, 3), cfg_dropout_prob=0.0, features=FEATURES):
    batch = len(text_lengths)
    grid_h, grid_w = grid
    max_text = max(text_lengths)
    llm = torch.zeros(batch, max_text, features)
    for row, length in enumerate(text_lengths):
        # Left-padded, as the dataloader produces.
        llm[row, max_text - length :] = 1.0
    return types.SimpleNamespace(
        batch={"llm_features": llm, "text_lengths": text_lengths},
        device=torch.device("cpu"),
        dtype=torch.float32,
        noisy_latents=torch.randn(batch, CHANNELS, grid_h, grid_w),
        sigma=torch.full((batch,), 0.25),
        cfg_dropout_prob=cfg_dropout_prob,
    )


class RecordingModel(torch.nn.Module):
    """A model that records what it was called with, and what the packing buffer
    held at the moment it was called."""

    def __init__(self, seq_len, prediction=None):
        super().__init__()
        self.seq_len = seq_len
        self.prediction = prediction
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        batch = kwargs["hidden_states"].shape[0]
        if self.prediction is not None:
            return (self.prediction,)
        return (torch.zeros(batch, self.seq_len, CHANNELS),)


# --------------------------------------------------------------------------- #
# 1. The sign                                                                 #
# --------------------------------------------------------------------------- #
class TestVelocitySign:
    def test_the_prediction_is_negated(self, fake_automodel):
        """The transformer predicts x0 - eps; AutoModel's target is eps - x0. A run
        without this negation trains happily toward the opposite objective."""
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([4, 4], grid=(2, 3))
        inputs = adapter.prepare_inputs(context)

        seq_len = inputs["hidden_states"].shape[1]
        prediction = torch.ones(2, seq_len, CHANNELS)
        model = RecordingModel(seq_len, prediction=prediction)

        out = adapter.forward(model, inputs)
        assert torch.all(out == -1.0)

    def test_the_negation_can_be_turned_off(self, fake_automodel):
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS, predict_negative_velocity=False)
        context = make_context([4, 4], grid=(2, 3))
        inputs = adapter.prepare_inputs(context)
        seq_len = inputs["hidden_states"].shape[1]
        model = RecordingModel(seq_len, prediction=torch.ones(2, seq_len, CHANNELS))
        assert torch.all(adapter.forward(model, inputs) == 1.0)

    def test_only_the_image_tokens_are_returned(self, fake_automodel):
        """The text region carries no velocity the loss reads, so the slice has to
        start exactly at the end of it."""
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([4, 4], grid=(2, 3))
        inputs = adapter.prepare_inputs(context)
        max_text = inputs["_max_text"]
        seq_len = inputs["hidden_states"].shape[1]

        # Mark text positions 0 and image positions 1, then check nothing leaked.
        prediction = torch.zeros(2, seq_len, CHANNELS)
        prediction[:, max_text:] = 1.0
        model = RecordingModel(seq_len, prediction=prediction)

        out = adapter.forward(model, inputs)
        assert out.shape == (2, CHANNELS, 2, 3)
        assert torch.all(out == -1.0)


# --------------------------------------------------------------------------- #
# 2. The reserved pad column                                                  #
# --------------------------------------------------------------------------- #
class TestReservedPadColumn:
    def test_it_is_added_when_the_packing_will_be_built(self, fake_automodel, monkeypatch):
        monkeypatch.setenv("PRIMUS_IDEOGRAM_VARLEN_ATTN", "1")
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([4, 4], grid=(2, 3))
        inputs = adapter.prepare_inputs(context)
        assert inputs["_max_text"] == 5, "one column reserved on top of the 4 produced"

    def test_it_is_not_added_when_nothing_would_read_the_packing(self, fake_automodel):
        """It costs a token position, so it is only paid for when it buys the
        data-independent shape it exists for."""
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([4, 4], grid=(2, 3))
        inputs = adapter.prepare_inputs(context)
        assert inputs["_max_text"] == 4
        assert inputs["_cu_seqlens"] is None

    def test_a_full_width_caption_still_gets_two_segments(self, fake_automodel, monkeypatch):
        """This is the case the column exists for: without it, a caption filling
        the full width leaves no padding, the row is one segment instead of two,
        and the packing's shape becomes data-dependent."""
        monkeypatch.setenv("PRIMUS_IDEOGRAM_VARLEN_ATTN", "1")
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        # Every caption fills the dataloader's full width.
        context = make_context([4, 4], grid=(2, 3))
        inputs = adapter.prepare_inputs(context)
        assert inputs["_cu_seqlens"].numel() == 2 * 2 + 1

    def test_the_packing_shape_is_the_same_for_every_length_pattern(self, fake_automodel, monkeypatch):
        """The property that makes a compiled graph reusable."""
        monkeypatch.setenv("PRIMUS_IDEOGRAM_VARLEN_ATTN", "1")
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        shapes = set()
        for lengths in ([1, 4], [4, 4], [2, 3], [1, 1]):
            context = make_context(lengths, grid=(2, 3))
            # Pad every case to the same dataloader width, as a real one would.
            context.batch["llm_features"] = torch.zeros(2, 4, FEATURES)
            inputs = adapter.prepare_inputs(context)
            shapes.add(tuple(inputs["_cu_seqlens"].shape))
        assert len(shapes) == 1, f"the packing shape varied with the data: {shapes}"


# --------------------------------------------------------------------------- #
# 3. Publish before forward                                                   #
# --------------------------------------------------------------------------- #
class TestPublishOrdering:
    def test_the_packing_is_published_before_the_model_runs(self, fake_automodel, monkeypatch):
        """The processor reads it during the forward, and again during the backward
        recompute. Publishing afterwards would leave the first read empty."""
        monkeypatch.setenv("PRIMUS_IDEOGRAM_VARLEN_ATTN", "1")
        order = []

        from primus.backends.nemo_automodel.models.ideogram4 import (
            adapter as adapter_mod,
        )

        def fake_publish(model, cu, max_seqlen, device=None, required=False):
            order.append("publish")
            return 1

        monkeypatch.setattr(adapter_mod, "publish_packing", fake_publish)

        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([4, 4], grid=(2, 3))
        inputs = adapter.prepare_inputs(context)
        seq_len = inputs["hidden_states"].shape[1]

        class OrderedModel(RecordingModel):
            def forward(self, **kwargs):
                order.append("forward")
                return super().forward(**kwargs)

        adapter.forward(OrderedModel(seq_len), inputs)
        assert order == ["publish", "forward"]

    def test_publishing_is_required_once_a_packing_was_built(self, fake_automodel, monkeypatch):
        """A model that cannot read the packing is a misconfiguration, not a
        fallback: the layers would each derive their own from the mask, and on a
        subset of ranks that silently averages two attention paths into one
        gradient."""
        monkeypatch.setenv("PRIMUS_IDEOGRAM_VARLEN_ATTN", "1")
        seen = {}

        from primus.backends.nemo_automodel.models.ideogram4 import (
            adapter as adapter_mod,
        )

        def fake_publish(model, cu, max_seqlen, device=None, required=False):
            seen["required"] = required
            return 1

        monkeypatch.setattr(adapter_mod, "publish_packing", fake_publish)

        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        inputs = adapter.prepare_inputs(make_context([4, 4], grid=(2, 3)))
        adapter.forward(RecordingModel(inputs["hidden_states"].shape[1]), inputs)
        assert seen["required"] is True

    def test_nothing_is_published_when_no_packing_was_built(self, fake_automodel, monkeypatch):
        published = []

        from primus.backends.nemo_automodel.models.ideogram4 import (
            adapter as adapter_mod,
        )

        monkeypatch.setattr(
            adapter_mod,
            "publish_packing",
            lambda *a, **kw: published.append(1),
        )

        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        inputs = adapter.prepare_inputs(make_context([4, 4], grid=(2, 3)))
        adapter.forward(RecordingModel(inputs["hidden_states"].shape[1]), inputs)
        assert published == []


# --------------------------------------------------------------------------- #
# Ragged batch under assume_dense                                             #
# --------------------------------------------------------------------------- #
class TestAssumeDenseRefusal:
    def test_a_ragged_batch_is_refused(self, fake_automodel, monkeypatch):
        """Dense flash over a padded row lets padding attend to real tokens. The
        lengths are already on the host here, so this is the one place the
        combination can be caught for free."""
        monkeypatch.setenv("PRIMUS_IDEOGRAM_ATTN_ASSUME_DENSE", "1")
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        with pytest.raises(ValueError, match="same text length"):
            adapter.prepare_inputs(make_context([2, 4], grid=(2, 3)))

    def test_an_equal_length_batch_is_allowed(self, fake_automodel, monkeypatch):
        monkeypatch.setenv("PRIMUS_IDEOGRAM_ATTN_ASSUME_DENSE", "1")
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        inputs = adapter.prepare_inputs(make_context([4, 4], grid=(2, 3)))
        assert inputs["hidden_states"].shape[0] == 2

    def test_the_error_names_the_offending_lengths(self, fake_automodel, monkeypatch):
        monkeypatch.setenv("PRIMUS_IDEOGRAM_ATTN_ASSUME_DENSE", "1")
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        with pytest.raises(ValueError) as excinfo:
            adapter.prepare_inputs(make_context([2, 4], grid=(2, 3)))
        assert "[2, 4]" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Input validation and layout                                                 #
# --------------------------------------------------------------------------- #
class TestInputValidation:
    def test_non_4d_latents_are_refused(self, fake_automodel):
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([4, 4], grid=(2, 3))
        context.noisy_latents = torch.randn(2, CHANNELS, 6)
        with pytest.raises(ValueError, match="4-D"):
            adapter.prepare_inputs(context)

    def test_a_channel_mismatch_is_refused(self, fake_automodel):
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([4, 4], grid=(2, 3))
        context.noisy_latents = torch.randn(2, 64, 2, 3)
        with pytest.raises(ValueError, match="packed channels"):
            adapter.prepare_inputs(context)

    def test_a_caption_longer_than_the_feature_width_is_refused(self, fake_automodel):
        """The dataloader has to left-pad to at least the longest caption;
        otherwise the features for the overflow simply are not there."""
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([4, 4], grid=(2, 3))
        context.batch["text_lengths"] = [4, 9]
        with pytest.raises(ValueError, match="left-pad"):
            adapter.prepare_inputs(context)

    def test_missing_lengths_default_to_the_full_width(self, fake_automodel):
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([4, 4], grid=(2, 3))
        del context.batch["text_lengths"]
        inputs = adapter.prepare_inputs(context)
        assert inputs["_max_text"] == 4

    def test_tensor_lengths_are_accepted(self, fake_automodel):
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([2, 4], grid=(2, 3))
        context.batch["text_lengths"] = torch.tensor([2, 4])
        inputs = adapter.prepare_inputs(context)
        assert inputs["_max_text"] == 4


class TestLayout:
    def test_the_time_convention_is_inverted(self, fake_automodel):
        """The model's time runs from noise to data, the opposite way from the
        noise level the trainer hands over."""
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([4, 4], grid=(2, 3))
        inputs = adapter.prepare_inputs(context)
        assert torch.allclose(inputs["timestep"], torch.full((2,), 0.75))

    def test_the_sequence_is_text_then_image(self, fake_automodel):
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([4, 4], grid=(2, 3))
        inputs = adapter.prepare_inputs(context)
        max_text = inputs["_max_text"]
        assert inputs["hidden_states"].shape[1] == max_text + 6
        # The text region of hidden_states is zeroed: the latents only occupy the
        # image positions.
        assert torch.all(inputs["hidden_states"][:, :max_text] == 0)

    def test_the_encoder_features_are_zero_over_the_image_region(self, fake_automodel):
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([4, 4], grid=(2, 3))
        inputs = adapter.prepare_inputs(context)
        max_text = inputs["_max_text"]
        assert torch.all(inputs["encoder_hidden_states"][:, max_text:] == 0)

    def test_the_indicator_marks_padding_text_and_image_regions(self, fake_automodel):
        """The per-token role is the layout contract with the model. A row with a
        short caption must have its leading positions left as padding."""
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([2, 4], grid=(2, 3))
        inputs = adapter.prepare_inputs(context)
        indicator = inputs["indicator"]
        max_text = inputs["_max_text"]
        # Row 0 has a 2-token caption in a 4-wide region, so two leading pads.
        assert torch.all(indicator[0, :2] == 0)
        assert torch.all(indicator[0, 2:max_text] != 0)
        # Row 1 fills the region, so no leading pad.
        assert torch.all(indicator[1, :max_text] != 0)

    def test_the_segment_ids_join_text_and_image(self, fake_automodel):
        """One segment covering both, because they attend jointly; the leading
        padding keeps the padding id and so forms a segment of its own. This is the
        structure the packing encodes."""
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([2, 4], grid=(2, 3))
        inputs = adapter.prepare_inputs(context)
        segment_ids = inputs["segment_ids"]
        assert torch.all(segment_ids[0, :2] == -1), "leading padding is its own segment"
        assert torch.all(segment_ids[0, 2:] == 1), "text and image share one segment"
        assert torch.all(segment_ids[1] == 1), "a full row is entirely one segment"

    def test_the_derived_mask_agrees_with_the_built_packing(self, fake_automodel, monkeypatch):
        """THE EQUIVALENCE, end to end through the adapter: the packing the adapter
        builds on the host is the same one the processor's mask analysis would
        derive from the segment ids the adapter also built. If these ever diverge,
        the fast path and the reference path are computing different attention."""
        monkeypatch.setenv("PRIMUS_IDEOGRAM_VARLEN_ATTN", "1")
        from primus.backends.nemo_automodel.attention import varlen_utils

        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)

        for lengths in ([1, 4], [2, 3], [4, 4], [1, 1], [3, 2]):
            context = make_context(lengths, grid=(2, 3))
            context.batch["llm_features"] = torch.zeros(2, 4, FEATURES)
            inputs = adapter.prepare_inputs(context)

            segment_ids = inputs["segment_ids"]
            mask = (segment_ids[:, :, None] == segment_ids[:, None, :]).unsqueeze(1)
            derived, _, is_trivial = varlen_utils.blockdiag_bool_mask_to_cu_seqlens(mask)

            assert not is_trivial, f"lengths {lengths} should have padding to pack"
            assert (
                derived.tolist() == inputs["_cu_seqlens"].tolist()
            ), f"host-built and mask-derived packings disagree for {lengths}"


class TestCfgDropout:
    def test_dropout_zeroes_whole_samples(self, fake_automodel):
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([4, 4], grid=(2, 3), cfg_dropout_prob=1.0)
        inputs = adapter.prepare_inputs(context)
        max_text = inputs["_max_text"]
        assert torch.all(inputs["encoder_hidden_states"][:, :max_text] == 0)

    def test_no_dropout_leaves_the_features_alone(self, fake_automodel):
        cls = fake_automodel.module.get_ideogram4_adapter_class()
        adapter = cls(in_channels=CHANNELS)
        context = make_context([4, 4], grid=(2, 3), cfg_dropout_prob=0.0)
        inputs = adapter.prepare_inputs(context)
        assert inputs["encoder_hidden_states"].abs().sum() > 0


# --------------------------------------------------------------------------- #
# Registration                                                               #
# --------------------------------------------------------------------------- #
class TestInstall:
    def test_the_ideogram4_route_is_added(self, fake_automodel):
        assert fake_automodel.module.install() is True
        adapter = fake_automodel.pipeline.create_adapter("ideogram4")
        assert isinstance(adapter, fake_automodel.module.get_ideogram4_adapter_class())

    def test_every_other_adapter_type_is_untouched(self, fake_automodel):
        """It wraps a factory every diffusion model shares, so a FLUX or Wan run
        must be unaffected."""
        fake_automodel.module.install()
        assert fake_automodel.pipeline.create_adapter("flux") == "stock:flux"
        assert fake_automodel.pipeline.create_adapter("wan") == "stock:wan"
        assert [t for t, _ in fake_automodel.created] == ["flux", "wan"]

    def test_kwargs_reach_the_adapter(self, fake_automodel):
        fake_automodel.module.install()
        adapter = fake_automodel.pipeline.create_adapter("ideogram4", predict_negative_velocity=False)
        assert adapter.predict_negative_velocity is False

    def test_it_is_idempotent(self, fake_automodel):
        fake_automodel.module.install()
        wrapped = fake_automodel.pipeline.create_adapter
        fake_automodel.module.install()
        assert fake_automodel.pipeline.create_adapter is wrapped, "double-wrapped"

    def test_the_recipe_namespace_is_patched_too(self, fake_automodel, monkeypatch):
        """The recipe does a from-import, which binds by value, so patching only
        the pipeline module would leave the recipe holding the original."""
        original = fake_automodel.pipeline.create_adapter
        recipe = types.ModuleType("nemo_automodel.recipes.diffusion.train")
        recipe.create_adapter = original
        for name in (
            "nemo_automodel.recipes",
            "nemo_automodel.recipes.diffusion",
        ):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "nemo_automodel.recipes.diffusion.train", recipe)

        fake_automodel.module.install()
        assert recipe.create_adapter is not original
        assert isinstance(
            recipe.create_adapter("ideogram4"),
            fake_automodel.module.get_ideogram4_adapter_class(),
        )


class TestClassIdentity:
    def test_the_class_is_cached(self, fake_automodel):
        """Identity has to be stable, because isinstance checks and anything keyed
        off the type would otherwise break between calls."""
        first = fake_automodel.module.get_ideogram4_adapter_class()
        assert fake_automodel.module.get_ideogram4_adapter_class() is first


class TestPatchGating:
    """The patch conditions, which decide whether any of this runs at all."""

    @staticmethod
    def _conditions():
        import primus.backends.nemo_automodel.patches  # noqa: F401
        from primus.core.patches.patch_registry import PatchRegistry

        return {p.id: p for p in PatchRegistry.iter_patches(backend="nemo_automodel", phase="before_train")}

    @staticmethod
    def _ctx(adapter_type):
        return types.SimpleNamespace(
            extra={
                "config": types.SimpleNamespace(
                    flow_matching=types.SimpleNamespace(adapter_type=adapter_type)
                )
            }
        )

    def test_an_ideogram4_run_registers_the_adapter(self):
        patch = self._conditions()["nemo_automodel.models.ideogram4.adapter"]
        assert patch.condition(self._ctx("ideogram4")) is True

    @pytest.mark.parametrize("other", ["flux", "flux2", "wan", "hunyuan", "qwen_image", "simple"])
    def test_another_model_does_not(self, other):
        """The wrapper is additive, so this is hygiene rather than correctness --
        but a patch that only appears in the runs that use it is a much easier
        thing to reason about when something goes wrong."""
        patch = self._conditions()["nemo_automodel.models.ideogram4.adapter"]
        assert patch.condition(self._ctx(other)) is False

    def test_an_unreadable_config_defaults_to_registering(self):
        """The two ways to be wrong are not symmetric. Registering for a run that
        does not want it changes nothing; skipping for a run that does want it
        fails later at adapter resolution, with an error about an unknown adapter
        type that says nothing about this patch."""
        patch = self._conditions()["nemo_automodel.models.ideogram4.adapter"]
        assert patch.condition(types.SimpleNamespace(extra={})) is True
        assert patch.condition(None) is True

    def test_the_processor_needs_both_the_flag_and_the_config(self, monkeypatch):
        patch = self._conditions()["nemo_automodel.models.ideogram4.varlen_attn"]
        assert patch.condition(self._ctx("ideogram4")) is False
        monkeypatch.setenv("PRIMUS_IDEOGRAM_VARLEN_ATTN", "1")
        assert patch.condition(self._ctx("ideogram4")) is True
        assert patch.condition(self._ctx("flux")) is False

    def test_the_conditions_do_not_import_the_implementation(self):
        """Discovery runs before anything has decided a run needs torch or
        diffusers. A condition that reached for the processor would make an
        unrelated diffusion job fail at startup on an import it never needed, so
        the gates live in a policy module with no heavy imports."""
        import subprocess

        code = (
            "import sys\n"
            "from primus.backends.nemo_automodel.models.ideogram4 import _varlen_common\n"
            "assert not _varlen_common.is_varlen_attn_enabled()\n"
            "assert 'torch' not in sys.modules\n"
            "assert 'diffusers' not in sys.modules\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    def test_both_run_before_the_model_is_built(self):
        """The adapter has to be in place before the recipe resolves adapter_type,
        and the processor before any attention module is constructed -- afterwards
        they all hold the stock processor."""
        patches = self._conditions()
        adapter_priority = patches["nemo_automodel.models.ideogram4.adapter"].priority
        varlen_priority = patches["nemo_automodel.models.ideogram4.varlen_attn"].priority
        assert adapter_priority < varlen_priority
        for strategy in (
            "nemo_automodel.models.wan.parallelize",
            "nemo_automodel.models.flux.parallelize",
        ):
            assert varlen_priority < patches[strategy].priority


class TestImportSafety:
    def test_importing_the_module_needs_neither_automodel_nor_diffusers(self):
        """The factory and the constant fallback exist for this. Without it, patch
        discovery would need a full model stack on every machine, and a FLUX job
        would fail at startup on an import it never needed."""
        import subprocess

        code = (
            "import sys\n"
            "from primus.backends.nemo_automodel.models.ideogram4 import adapter\n"
            "assert 'nemo_automodel.components' not in sys.modules\n"
            "assert 'diffusers' not in sys.modules\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
