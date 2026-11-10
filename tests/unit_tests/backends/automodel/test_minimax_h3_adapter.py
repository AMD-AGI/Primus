###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the MiniMax-H3 flow-matching adapter.

THE CLAIMS WORTH DEFENDING, in order of how expensive they are to get wrong:

  1. THE SIGN. The transformer predicts the velocity toward the data, ``x0 - eps``,
     while AutoModel's target is ``eps - x0``, so the adapter negates both streams.
     Getting it wrong yields a run that trains, converges, and optimizes the exact
     opposite of the objective. Random weights would never show it, so it is pinned
     with an oracle model that returns the exact data-ward velocity.

  2. THE ASSEMBLY. The transformer reads one packed sequence and is told, per row,
     the modality, the rotary position and the timestep. The adapter's call is
     compared with one built from diffusers' own inference helpers, so a wrong
     index, a wrong timestep for the text rows or a transposed patch fails here.

  3. THE AUDIO NOISE LEVEL pairs with the video's the way inference pairs the two
     schedules (shift 12 and shift 3 on one base grid).

Also covered: that install is additive, since it wraps a factory every other model shares.

AutoModel is faked, and the adapter is built through a factory so this is possible without
the full stack. The tests that need the H3 transformer use diffusers' real one at a tiny size.
"""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from primus.backends.automodel.models.minimax_h3 import (  # noqa: E402
    adapter as adapter_mod,
)
from primus.backends.automodel.models.minimax_h3 import layout as L  # noqa: E402

# A transformer small enough to run on the CPU in milliseconds, with the real channel widths and patch.
TINY = dict(
    num_attention_heads=2,
    attention_head_dim=96,  # the rotary embedding rotates 96 channels, so a head cannot be narrower
    hidden_size=64,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=128,
    text_dim=48,
    freq_dim=32,
    time_embed_hidden_dim=64,
    time_embed_dim=32,
    rope_freq_dim=16,
)
FRAMES, HEIGHT, WIDTH, AUDIO_LEN, TEXT_TOKENS = 3, 4, 4, 3, 5


class FakeModelAdapter:
    """Stand-in for AutoModel's ModelAdapter base."""


@pytest.fixture
def fake_automodel(monkeypatch):
    """Install a fake AutoModel adapter base and adapter factory."""

    def create_adapter(adapter_type, **kwargs):
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
        "nemo_automodel.recipes",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.flow_matching.adapters.base", base_mod)
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.flow_matching.pipeline", pipeline_mod)
    # The class is cached for identity stability, so reset it between tests.
    monkeypatch.setattr(adapter_mod, "_ADAPTER_CLS", None)
    return types.SimpleNamespace(pipeline=pipeline_mod, original=create_adapter)


@pytest.fixture(autouse=True)
def _single_thread():
    """Tiny CPU models spend nearly all their time in thread synchronization: one thread is about
    100x faster than sixteen. The setting is process-global, so it is restored."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def _fresh_layout_cache():
    L.clear_layout_cache()
    yield
    L.clear_layout_cache()


@pytest.fixture
def reference():
    return pytest.importorskip("diffusers.modular_pipelines.minimax_h3.before_denoise")


@pytest.fixture
def tiny_model():
    module = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
    torch.manual_seed(0)
    return module.MiniMaxH3Transformer3DModel(**TINY)


def make_context(sigma=0.37, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(1, 24, FRAMES, HEIGHT, WIDTH, generator=g)
    eps = torch.randn(1, 24, FRAMES, HEIGHT, WIDTH, generator=g)
    s = torch.tensor([sigma])
    noisy = (1 - s.view(1, 1, 1, 1, 1)) * x0 + s.view(1, 1, 1, 1, 1) * eps
    batch = {
        "audio_latents": torch.randn(1, 2, 32, AUDIO_LEN, generator=g),
        "text_embeddings": torch.randn(1, TEXT_TOKENS, TINY["text_dim"], generator=g),
    }
    ctx = types.SimpleNamespace(
        batch=batch,
        device=torch.device("cpu"),
        dtype=torch.float32,
        noisy_latents=noisy,
        latents=x0,
        sigma=s,
        cfg_dropout_prob=0.0,
    )
    return ctx, x0, eps


def new_adapter(fake_automodel, **kwargs):
    return adapter_mod.get_adapter_class()(**kwargs)


def shifted(u, shift):
    return shift * u / (1 + (shift - 1) * u)


class TestAudioSigma:
    def test_the_endpoints_are_fixed(self):
        for sigma in (0.0, 1.0):
            out = adapter_mod.audio_sigma_from_video(torch.tensor([sigma]), 12.0, 3.0)
            assert out.item() == pytest.approx(sigma)

    def test_it_pairs_the_schedules_the_way_inference_does(self):
        """Both modalities step one base grid, each through its own shift."""
        u = torch.linspace(0, 1, 21)
        video = shifted(u, 12.0)
        audio = adapter_mod.audio_sigma_from_video(video, 12.0, 3.0)
        assert torch.allclose(audio, shifted(u, 3.0), atol=1e-6)

    def test_equal_shifts_are_the_identity(self):
        sigma = torch.linspace(0, 1, 11)
        assert torch.allclose(adapter_mod.audio_sigma_from_video(sigma, 7.0, 7.0), sigma, atol=1e-6)

    def test_the_audio_is_noisier_at_a_given_video_level_when_its_shift_is_smaller(self):
        """Shift 3 spends less of the grid at high noise than shift 12, so mid-grid the audio is cleaner."""
        video = torch.tensor([0.5])
        assert adapter_mod.audio_sigma_from_video(video, 12.0, 3.0).item() < video.item()


class TestConstruction:
    def test_an_unknown_audio_sigma_mode_is_refused(self, fake_automodel):
        with pytest.raises(ValueError, match="audio_sigma"):
            new_adapter(fake_automodel, audio_sigma="independent")

    def test_a_non_positive_shift_is_refused(self, fake_automodel):
        with pytest.raises(ValueError, match="positive"):
            new_adapter(fake_automodel, video_flow_shift=0.0)

    def test_the_class_is_cached(self, fake_automodel):
        first = adapter_mod.get_adapter_class()
        assert adapter_mod.get_adapter_class() is first


class TestPrepareInputs:
    @pytest.fixture(autouse=True)
    def _needs_diffusers(self, reference):
        pass

    def test_shapes(self, fake_automodel):
        ctx, _, _ = make_context()
        inputs = new_adapter(fake_automodel).prepare_inputs(ctx)
        assert inputs["hidden_states"].shape == (
            1,
            FRAMES * 4,
            24 * 4,
        )  # 12 patches of 24 channels x (1, 2, 2)
        assert inputs["audio_hidden_states"].shape == (1, 2 * AUDIO_LEN, 32)
        assert inputs["encoder_hidden_states"].shape == (1, TEXT_TOKENS, TINY["text_dim"])
        rows = FRAMES * 4 + 2 * AUDIO_LEN + TEXT_TOKENS
        for key in ("timestep_indices", "token_tags"):
            assert inputs[key].shape == (rows,)
        assert inputs["position_ids"].shape == (rows, 3)

    def test_the_timestep_table_is_one_minus_sigma(self, fake_automodel):
        """H3's t = 1 - sigma, where 1 means clean; AutoModel's own `timesteps` run the other way on a 1000x scale."""
        ctx, _, _ = make_context(sigma=0.37)
        inputs = new_adapter(fake_automodel).prepare_inputs(ctx)
        expected_audio_sigma = adapter_mod.audio_sigma_from_video(ctx.sigma, 12.0, 3.0)
        assert inputs["timestep"][0].item() == pytest.approx(1 - 0.37)
        assert inputs["timestep"][1].item() == pytest.approx(1 - expected_audio_sigma.item())

    def test_the_same_mode_uses_the_video_level_for_both(self, fake_automodel):
        ctx, _, _ = make_context(sigma=0.37)
        inputs = new_adapter(fake_automodel, audio_sigma="same").prepare_inputs(ctx)
        assert inputs["timestep"][0].item() == inputs["timestep"][1].item()

    def test_every_row_gets_the_timestep_the_reference_gives_it(self, fake_automodel, reference):
        """Text rows inherit the video timestep; audio rows have their own."""
        ctx, _, _ = make_context()
        inputs = new_adapter(fake_automodel).prepare_inputs(ctx)
        table, index = reference.MiniMaxH3SetTimestepsStep.build_row_timesteps(
            inputs["video_indices"],
            inputs["audio_indices"],
            0,
            0,
            TEXT_TOKENS,
            float(inputs["timestep"][0]),
            float(inputs["timestep"][1]),
            0.0,
            1.0,
        )
        mine = inputs["timestep"][inputs["timestep_indices"]]
        assert torch.equal(table[index], mine)

    def test_the_audio_input_and_target_are_consistent(self, fake_automodel):
        """x_t = (1 - s) x0 + s eps and target = eps - x0, so x_t = x0 + s * target."""
        ctx, _, _ = make_context(sigma=0.5)
        inputs = new_adapter(fake_automodel).prepare_inputs(ctx)
        s = adapter_mod.audio_sigma_from_video(ctx.sigma, 12.0, 3.0).item()
        noisy = L.unpack_audio(inputs["audio_hidden_states"], 2, AUDIO_LEN)
        assert torch.allclose(noisy, ctx.batch["audio_latents"] + s * inputs["_audio_target"], atol=1e-5)

    def test_the_audio_noise_is_fresh_each_call(self, fake_automodel):
        ctx, _, _ = make_context()
        adapter = new_adapter(fake_automodel)
        assert not torch.equal(
            adapter.prepare_inputs(ctx)["_audio_target"], adapter.prepare_inputs(ctx)["_audio_target"]
        )

    def test_the_layout_is_built_once_per_shape(self, fake_automodel):
        ctx, _, _ = make_context()
        adapter = new_adapter(fake_automodel)
        first = adapter.prepare_inputs(ctx)
        second = adapter.prepare_inputs(ctx)
        assert first["position_ids"] is second["position_ids"]

    def test_micro_batch_above_one_is_refused(self, fake_automodel):
        ctx, _, _ = make_context()
        ctx.noisy_latents = ctx.noisy_latents.repeat(2, 1, 1, 1, 1)
        with pytest.raises(ValueError, match="micro-batch 1"):
            new_adapter(fake_automodel).prepare_inputs(ctx)

    @pytest.mark.parametrize("missing", ["audio_latents", "text_embeddings"])
    def test_a_missing_batch_key_is_named(self, fake_automodel, missing):
        ctx, _, _ = make_context()
        del ctx.batch[missing]
        with pytest.raises(KeyError, match=missing):
            new_adapter(fake_automodel).prepare_inputs(ctx)

    def test_audio_that_is_not_stereo_is_refused(self, fake_automodel):
        ctx, _, _ = make_context()
        ctx.batch["audio_latents"] = torch.randn(1, 1, 32, AUDIO_LEN)
        with pytest.raises(ValueError, match="audio_latents must be"):
            new_adapter(fake_automodel).prepare_inputs(ctx)


class TestForward:
    @pytest.fixture(autouse=True)
    def _needs_diffusers(self, reference):
        pass

    def _oracle(self, ctx, x0, eps, inputs):
        """A model that returns the exact data-ward velocity for both streams."""
        audio_x0 = ctx.batch["audio_latents"]
        audio_eps = inputs["_audio_target"] + audio_x0
        video_rows = L.patchify_video(x0 - eps, (1, 2, 2))
        audio_rows = L.pack_audio(audio_x0 - audio_eps)
        return lambda **kwargs: (video_rows, audio_rows)

    def test_the_video_prediction_is_eps_minus_x0(self, fake_automodel):
        """THE SIGN. An oracle that knows x0 - eps must come out as AutoModel's eps - x0."""
        ctx, x0, eps = make_context()
        adapter = new_adapter(fake_automodel)
        inputs = adapter.prepare_inputs(ctx)
        prediction = adapter.forward(self._oracle(ctx, x0, eps, inputs), inputs)
        assert prediction.shape == x0.shape
        assert torch.allclose(prediction, eps - x0, atol=1e-6)

    def test_the_audio_loss_is_zero_for_the_oracle(self, fake_automodel):
        ctx, x0, eps = make_context()
        adapter = new_adapter(fake_automodel)
        inputs = adapter.prepare_inputs(ctx)
        adapter.forward(self._oracle(ctx, x0, eps, inputs), inputs)
        assert adapter.auxiliary_losses(inputs)["audio_loss"].item() == pytest.approx(0.0, abs=1e-10)

    def test_the_audio_loss_is_not_zero_for_the_wrong_sign(self, fake_automodel):
        """The test above could pass if the audio loss ignored its inputs."""
        ctx, x0, eps = make_context()
        adapter = new_adapter(fake_automodel)
        inputs = adapter.prepare_inputs(ctx)
        audio_x0 = ctx.batch["audio_latents"]
        wrong = L.pack_audio(
            inputs["_audio_target"] + audio_x0 - audio_x0 - inputs["_audio_target"] - audio_x0
        )
        adapter.forward(lambda **kw: (L.patchify_video(x0 - eps, (1, 2, 2)), wrong), inputs)
        assert adapter.auxiliary_losses(inputs)["audio_loss"].item() > 0.1

    def test_the_audio_loss_weight_scales_it(self, fake_automodel):
        ctx, x0, _ = make_context()
        losses = {}
        for weight in (1.0, 3.0):
            adapter = new_adapter(fake_automodel, audio_loss_weight=weight)
            torch.manual_seed(5)
            inputs = adapter.prepare_inputs(ctx)
            adapter.forward(
                lambda **kw: (L.patchify_video(x0, (1, 2, 2)), torch.zeros(1, 2 * AUDIO_LEN, 32)), inputs
            )
            losses[weight] = adapter.auxiliary_losses(inputs)["audio_loss"].item()
        assert losses[3.0] == pytest.approx(3 * losses[1.0])

    def test_private_keys_never_reach_the_model(self, fake_automodel):
        ctx, x0, eps = make_context()
        adapter = new_adapter(fake_automodel)
        inputs = adapter.prepare_inputs(ctx)
        seen = {}

        def model(**kwargs):
            seen.update(kwargs)
            return L.patchify_video(x0, (1, 2, 2)), torch.zeros(1, 2 * AUDIO_LEN, 32)

        adapter.forward(model, inputs)
        assert not [k for k in seen if k.startswith("_")]
        assert seen["return_dict"] is False

    def test_an_output_object_is_handled_like_a_tuple(self, fake_automodel):
        ctx, x0, eps = make_context()
        adapter = new_adapter(fake_automodel)
        inputs = adapter.prepare_inputs(ctx)
        video_rows = L.patchify_video(x0 - eps, (1, 2, 2))
        audio_rows = torch.zeros(1, 2 * AUDIO_LEN, 32)
        out = types.SimpleNamespace(sample=video_rows, audio_sample=audio_rows)
        prediction = adapter.forward(lambda **kw: out, inputs)
        assert torch.allclose(prediction, eps - x0, atol=1e-6)


class TestAudioLossLogging:
    """The recipe never reports auxiliary losses, so the adapter logs the audio loss itself."""

    @pytest.fixture(autouse=True)
    def _needs_diffusers(self, reference):
        pass

    def _micro_step(self, adapter, ctx, x0, audio_rows):
        inputs = adapter.prepare_inputs(ctx)
        adapter.forward(lambda **kw: (L.patchify_video(x0, (1, 2, 2)), audio_rows), inputs)
        return adapter.auxiliary_losses(inputs)["audio_loss"]

    def _messages(self, caplog):
        return [r.getMessage() for r in caplog.records if "audio_loss=" in r.getMessage()]

    def test_it_logs_the_unweighted_mean_every_interval(self, fake_automodel, caplog):
        ctx, x0, _ = make_context()
        adapter = new_adapter(fake_automodel, audio_loss_weight=3.0, audio_loss_log_interval=2)
        zeros = torch.zeros(1, 2 * AUDIO_LEN, 32)
        with caplog.at_level("INFO", logger=adapter_mod.logger.name):
            first = self._micro_step(adapter, ctx, x0, zeros)
            assert self._messages(caplog) == []  # not yet at the interval
            self._micro_step(adapter, ctx, x0, zeros)
        (message,) = self._messages(caplog)
        # Unweighted: the returned (weighted) loss is 3x what is logged, for a fixed audio noise seed.
        logged = float(message.split("audio_loss=")[1].split()[0])
        assert logged > 0
        assert first.item() == pytest.approx(3 * logged, rel=0.5)  # same distribution, fresh noise

    def test_the_window_restarts_after_each_log(self, fake_automodel, caplog):
        ctx, x0, _ = make_context()
        adapter = new_adapter(fake_automodel, audio_loss_log_interval=1)
        zeros = torch.zeros(1, 2 * AUDIO_LEN, 32)
        with caplog.at_level("INFO", logger=adapter_mod.logger.name):
            for _ in range(3):
                self._micro_step(adapter, ctx, x0, zeros)
        assert len(self._messages(caplog)) == 3

    def test_zero_turns_it_off(self, fake_automodel, caplog):
        ctx, x0, _ = make_context()
        adapter = new_adapter(fake_automodel, audio_loss_log_interval=0)
        with caplog.at_level("INFO", logger=adapter_mod.logger.name):
            for _ in range(12):
                self._micro_step(adapter, ctx, x0, torch.zeros(1, 2 * AUDIO_LEN, 32))
        assert self._messages(caplog) == []

    def test_a_negative_interval_is_refused(self, fake_automodel):
        with pytest.raises(ValueError, match="audio_loss_log_interval"):
            new_adapter(fake_automodel, audio_loss_log_interval=-1)

    def test_logging_does_not_change_the_returned_loss(self, fake_automodel):
        ctx, x0, _ = make_context()
        values = []
        for interval in (0, 1):
            adapter = new_adapter(fake_automodel, audio_loss_log_interval=interval)
            torch.manual_seed(5)
            values.append(self._micro_step(adapter, ctx, x0, torch.zeros(1, 2 * AUDIO_LEN, 32)).item())
        assert values[0] == values[1]


class TestWithTheRealTransformer:
    @pytest.fixture(autouse=True)
    def _needs_diffusers(self, reference):
        pass

    def test_the_call_equals_one_built_from_diffusers_helpers(self, fake_automodel, reference, tiny_model):
        """THE ASSEMBLY. Layout, video rows and timesteps from diffusers' own inference code."""
        ctx, _, _ = make_context()
        adapter = new_adapter(fake_automodel)
        inputs = adapter.prepare_inputs(ctx)
        c = L.constants()

        text_tags = torch.full((TEXT_TOKENS,), c.text_tag, dtype=torch.long)
        pos, tags, vid_idx, aud_idx, txt_idx, _, _ = (
            reference.MiniMaxH3PrepareLayoutStep.build_packed_sequence(
                text_tags,
                FRAMES,
                HEIGHT,
                WIDTH,
                AUDIO_LEN,
                c.patch_size,
                c.audio_channels,
                c.audio_tag,
                c.video_tag,
            )
        )
        timestep, timestep_indices = reference.MiniMaxH3SetTimestepsStep.build_row_timesteps(
            vid_idx,
            aud_idx,
            0,
            0,
            TEXT_TOKENS,
            float(inputs["timestep"][0]),
            float(inputs["timestep"][1]),
            0.0,
            1.0,
        )
        expected = tiny_model(
            hidden_states=reference.patchify_video_latents(ctx.noisy_latents, c.patch_size)[None],
            audio_hidden_states=inputs["audio_hidden_states"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            timestep=timestep,
            timestep_indices=timestep_indices,
            token_tags=tags,
            position_ids=pos,
            video_indices=vid_idx,
            audio_indices=aud_idx,
            text_indices=txt_idx,
            return_dict=False,
        )
        actual = tiny_model(**{k: v for k, v in inputs.items() if not k.startswith("_")}, return_dict=False)
        assert torch.allclose(actual[0], expected[0], atol=1e-6)
        assert torch.allclose(actual[1], expected[1], atol=1e-6)

    def test_forward_and_backward_reach_every_parameter(self, fake_automodel, tiny_model):
        ctx, x0, eps = make_context()
        adapter = new_adapter(fake_automodel)
        inputs = adapter.prepare_inputs(ctx)
        prediction = adapter.forward(tiny_model, inputs)
        assert prediction.shape == x0.shape and torch.isfinite(prediction).all()
        loss = (
            torch.nn.functional.mse_loss(prediction.float(), eps - x0)
            + adapter.auxiliary_losses(inputs)["audio_loss"]
        )
        loss.backward()
        params = dict(tiny_model.named_parameters())
        missing = [n for n, p in params.items() if p.grad is None]
        assert not missing, missing
        assert all(torch.isfinite(p.grad).all() for p in params.values())

    def test_a_step_on_a_fixed_draw_lowers_the_loss(self, fake_automodel, tiny_model):
        """Same noise every step: the loss must go down, which it cannot if the gradient points the wrong way."""
        ctx, x0, eps = make_context()
        adapter = new_adapter(fake_automodel)
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-2)
        losses = []
        for _ in range(6):
            torch.manual_seed(11)
            inputs = adapter.prepare_inputs(ctx)
            prediction = adapter.forward(tiny_model, inputs)
            loss = (
                torch.nn.functional.mse_loss(prediction.float(), eps - x0)
                + adapter.auxiliary_losses(inputs)["audio_loss"]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]


class TestInstall:
    def test_the_route_is_added_and_the_rest_fall_through(self, fake_automodel):
        assert adapter_mod.install() is True
        created = fake_automodel.pipeline.create_adapter("minimax_h3", audio_loss_weight=2.0)
        assert type(created).__name__ == "MiniMaxH3Adapter" and created.audio_loss_weight == 2.0
        assert fake_automodel.pipeline.create_adapter("flux") == "stock:flux"

    def test_it_is_idempotent(self, fake_automodel):
        adapter_mod.install()
        once = fake_automodel.pipeline.create_adapter
        adapter_mod.install()
        assert fake_automodel.pipeline.create_adapter is once

    def test_it_composes_with_another_route_in_either_order(self, fake_automodel):
        """Ideogram-4 wraps the same factory; each only handles its own type and forwards the rest."""
        original = fake_automodel.pipeline.create_adapter

        def other_route(adapter_type, **kwargs):
            return "other" if adapter_type == "other" else original(adapter_type, **kwargs)

        fake_automodel.pipeline.create_adapter = other_route
        adapter_mod.install()
        assert fake_automodel.pipeline.create_adapter("other") == "other"
        assert type(fake_automodel.pipeline.create_adapter("minimax_h3")).__name__ == "MiniMaxH3Adapter"
        assert fake_automodel.pipeline.create_adapter("flux") == "stock:flux"

    def _recipes(self, monkeypatch, bound):
        recipe = types.ModuleType("nemo_automodel.recipes.diffusion.train")
        recipe.create_adapter = bound
        diffusion = types.ModuleType("nemo_automodel.recipes.diffusion")
        diffusion.train = recipe
        monkeypatch.setitem(sys.modules, "nemo_automodel.recipes.diffusion", diffusion)
        monkeypatch.setitem(sys.modules, "nemo_automodel.recipes.diffusion.train", recipe)
        sys.modules["nemo_automodel.recipes"].diffusion = diffusion
        return recipe

    def test_a_recipe_holding_the_original_is_rebound(self, fake_automodel, monkeypatch):
        """The recipe binds create_adapter by value when it is imported."""
        recipe = self._recipes(monkeypatch, fake_automodel.original)
        adapter_mod.install()
        assert recipe.create_adapter is fake_automodel.pipeline.create_adapter

    def test_someone_elses_wrapper_is_left_alone_and_reported(self, fake_automodel, monkeypatch, caplog):
        def someone_elses(adapter_type, **kwargs):
            return "theirs"

        recipe = self._recipes(monkeypatch, someone_elses)
        with caplog.at_level("WARNING"):
            adapter_mod.install()
        assert recipe.create_adapter is someone_elses
        assert any("left alone" in r.getMessage() for r in caplog.records)

    def test_a_recipe_that_cannot_be_imported_is_reported(self, fake_automodel, caplog):
        with caplog.at_level("WARNING"):
            adapter_mod.install()
        assert any("could not import the diffusion recipe" in r.getMessage() for r in caplog.records)


class TestAudioLossLogging:
    """AutoModel never reports auxiliary losses, so the adapter logs the audio loss itself."""

    @staticmethod
    def _inputs(error):
        target = torch.zeros(1, 2, 32, 4)
        return {"_audio_pred": torch.full_like(target, error), "_audio_target": target}

    def test_it_logs_the_unweighted_mean_once_per_interval(self, fake_automodel, caplog):
        adapter = new_adapter(fake_automodel, audio_loss_weight=0.5, audio_loss_log_interval=3)
        with caplog.at_level("INFO", logger=adapter_mod.logger.name):
            for error in (1.0, 2.0):
                adapter.auxiliary_losses(self._inputs(error))
            assert "audio_loss=" not in caplog.text
            out = adapter.auxiliary_losses(self._inputs(3.0))
        # MSE of a constant error e is e**2: mean of 1, 4 and 9, unweighted.
        assert "audio_loss=4.666667" in caplog.text
        assert "last 3 micro-steps" in caplog.text
        # The returned loss is still the weighted one.
        assert out["audio_loss"].item() == pytest.approx(0.5 * 9.0)

    def test_the_window_restarts_after_each_log(self, fake_automodel, caplog):
        adapter = new_adapter(fake_automodel, audio_loss_log_interval=2)
        with caplog.at_level("INFO", logger=adapter_mod.logger.name):
            for error in (1.0, 1.0, 3.0, 3.0):
                adapter.auxiliary_losses(self._inputs(error))
        assert [r.getMessage().split()[1] for r in caplog.records] == ["audio_loss=1.000000", "audio_loss=9.000000"]

    def test_zero_turns_it_off(self, fake_automodel, caplog):
        adapter = new_adapter(fake_automodel, audio_loss_log_interval=0)
        with caplog.at_level("INFO", logger=adapter_mod.logger.name):
            for _ in range(20):
                adapter.auxiliary_losses(self._inputs(1.0))
        assert "audio_loss=" not in caplog.text

    def test_a_negative_interval_is_refused(self, fake_automodel):
        with pytest.raises(ValueError, match="audio_loss_log_interval"):
            new_adapter(fake_automodel, audio_loss_log_interval=-1)

    def test_it_does_not_keep_the_graph(self, fake_automodel):
        adapter = new_adapter(fake_automodel, audio_loss_log_interval=5)
        inputs = self._inputs(1.0)
        inputs["_audio_pred"].requires_grad_(True)
        adapter.auxiliary_losses(inputs)
        assert not adapter._audio_loss_sum.requires_grad
