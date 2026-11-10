###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the MiniMax-H3 packed layout and latent packing.

THE CLAIMS WORTH DEFENDING:

  1. THE LAYOUT IS DIFFUSERS'. Training calls ``MiniMaxH3PrepareLayoutStep.build_packed_sequence``
     rather than copying it, so a drift between the two would show here as a failure
     rather than as a mislabeled row somewhere in a 38K-row sequence.

  2. THE PACKING INVERTS. A prediction that is unpacked in the wrong order trains,
     converges, and learns the wrong thing. No shape check catches a transposed
     patch, so the round trip is pinned exactly.

  3. THE SHAPES MATCH THE PLAN. The row counts of the 5.2 s and 15 s clips are what the
     memory and compute estimates in the implementation plan were derived from.
"""

import sys

import pytest

torch = pytest.importorskip("torch")

from primus.backends.automodel.models.minimax_h3 import layout as L  # noqa: E402


@pytest.fixture
def reference():
    """Diffusers' own H3 inference code, which needs the optional packages its modular pipelines import."""
    return pytest.importorskip("diffusers.modular_pipelines.minimax_h3.before_denoise")


@pytest.fixture(autouse=True)
def _fresh_cache():
    L.clear_layout_cache()
    yield
    L.clear_layout_cache()


class TestConstants:
    def test_the_defaults_match_diffusers(self):
        """The defaults are what a run without the H3-capable diffusers uses, so they have to equal
        the real values wherever both exist."""
        ref = pytest.importorskip("diffusers.modular_pipelines.minimax_h3.modular_pipeline")
        c = L.H3Constants()
        assert c.video_tag == ref.MINIMAX_H3_VIDEO_TAG
        assert c.text_tag == ref.MINIMAX_H3_TEXT_TAG
        assert c.audio_tag == ref.MINIMAX_H3_AUDIO_TAG
        assert c.audio_channels == ref.MINIMAX_H3_AUDIO_CHANNELS
        assert c.fps == ref.MINIMAX_H3_FPS
        assert c.audio_latents_per_second == ref.MINIMAX_H3_AUDIO_LATENTS_PER_SECOND

    def test_the_transformers_patch_and_channels_match_its_defaults(self):
        module = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
        with torch.device("meta"):
            model = module.MiniMaxH3Transformer3DModel(num_layers=1)
        c = L.H3Constants()
        assert tuple(model.config.patch_size) == c.patch_size
        assert model.config.in_channels == c.video_latent_channels
        assert model.config.audio_in_channels == c.audio_latent_channels

    def test_a_diffusers_without_h3_falls_back_to_the_defaults(self, monkeypatch):
        monkeypatch.setattr(L, "_CONSTANTS", None)
        monkeypatch.setitem(sys.modules, "diffusers.modular_pipelines.minimax_h3", None)
        assert L.constants() == L.H3Constants()


class TestGeometry:
    @pytest.fixture(autouse=True)
    def _needs_diffusers(self, reference):
        pass

    def test_the_shortest_clip(self):
        """5.2 s at the default canvas: the numbers the plan's tables are built on."""
        g = L.latent_geometry(768, 1344, 124)
        assert (g.latent_frames, g.latent_height, g.latent_width, g.num_audio_latents) == (37, 48, 84, 207)

    def test_a_ten_second_clip(self):
        g = L.latent_geometry(768, 1344, 243)
        assert (g.latent_frames, g.num_audio_latents) == (72, 405)

    def test_the_longest_clip(self):
        g = L.latent_geometry(768, 1344, 362)
        assert (g.latent_frames, g.num_audio_latents) == (107, 603)

    def test_a_frame_count_the_vae_cannot_encode_is_refused(self):
        with pytest.raises(ValueError, match="17 \\* n \\+ 5"):
            L.latent_geometry(768, 1344, 125)

    def test_a_side_that_is_not_a_multiple_of_32_is_refused(self):
        with pytest.raises(ValueError, match="multiples of 32"):
            L.latent_geometry(770, 1344, 124)


class TestLayout:
    @pytest.fixture(autouse=True)
    def _needs_diffusers(self, reference):
        pass

    def test_it_equals_diffusers_build_packed_sequence(self, reference):
        c = L.H3Constants()
        text_tags = torch.full((5,), c.text_tag, dtype=torch.long)
        expected = reference.MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_tags, 3, 4, 6, 7, c.patch_size, c.audio_channels, c.audio_tag, c.video_tag
        )
        layout = L.build_layout(5, 3, 4, 6, 7)
        assert torch.equal(layout.position_ids, expected[0])
        assert torch.equal(layout.token_tags, expected[1])
        assert torch.equal(layout.video_indices, expected[2])
        assert torch.equal(layout.audio_indices, expected[3])
        assert torch.equal(layout.text_indices, expected[4])

    def test_the_rows_add_up(self):
        g = L.latent_geometry(768, 1344, 124)
        layout = L.build_layout(300, g.latent_frames, g.latent_height, g.latent_width, g.num_audio_latents)
        # 37 frames of 24 x 42 patches, 2 stereo channels of 207, and the text.
        assert layout.num_video_rows == 37 * 24 * 42 == 37_296
        assert layout.num_audio_rows == 2 * 207 == 414
        assert layout.token_tags.numel() == 37_296 + 414 + 300

    def test_the_longest_clip_is_about_108k_video_rows(self):
        """The plan's figure: the master plan's 92K assumed 4x temporal compression."""
        g = L.latent_geometry(768, 1344, 362)
        layout = L.build_layout(300, g.latent_frames, g.latent_height, g.latent_width, g.num_audio_latents)
        assert layout.num_video_rows == 107 * 24 * 42 == 107_856

    def test_every_row_has_the_modality_its_indices_say(self):
        c = L.H3Constants()
        layout = L.build_layout(5, 3, 4, 6, 7)
        assert (layout.token_tags[layout.video_indices] == c.video_tag).all()
        assert (layout.token_tags[layout.audio_indices] == c.audio_tag).all()
        assert (layout.token_tags[layout.text_indices] == c.text_tag).all()

    def test_only_audio_rows_use_the_second_timestep(self):
        """Text rows inherit the video timestep, as in the reference; the table is [video, audio]."""
        layout = L.build_layout(5, 3, 4, 6, 7)
        assert (layout.timestep_indices[layout.audio_indices] == 1).all()
        assert (layout.timestep_indices[layout.video_indices] == 0).all()
        assert (layout.timestep_indices[layout.text_indices] == 0).all()

    def test_it_matches_the_references_row_timesteps(self, reference):
        layout = L.build_layout(5, 3, 4, 6, 7)
        table, index = reference.MiniMaxH3SetTimestepsStep.build_row_timesteps(
            layout.video_indices, layout.audio_indices, 0, 0, 5, 0.6, 0.8, 0.0, 1.0
        )
        mine = torch.tensor([0.6, 0.8])[layout.timestep_indices]
        assert torch.equal(table[index], mine)

    def test_a_temporal_patch_above_one_is_refused(self):
        with pytest.raises(ValueError, match="temporal patch of 1"):
            L.build_layout(5, 3, 4, 6, 7, c=L.H3Constants(patch_size=(2, 2, 2)))

    def test_model_kwargs_name_the_transformers_structural_arguments(self):
        layout = L.build_layout(5, 3, 4, 6, 7)
        assert set(layout.model_kwargs()) == {
            "timestep_indices",
            "token_tags",
            "position_ids",
            "video_indices",
            "audio_indices",
            "text_indices",
        }


class TestLayoutCache:
    @pytest.fixture(autouse=True)
    def _needs_diffusers(self, reference):
        pass

    def test_a_repeated_shape_is_built_once(self):
        first = L.get_layout(5, 3, 4, 6, 7, "cpu")
        assert L.get_layout(5, 3, 4, 6, 7, "cpu") is first

    def test_another_shape_is_another_entry(self):
        assert L.get_layout(5, 3, 4, 6, 7, "cpu") is not L.get_layout(6, 3, 4, 6, 7, "cpu")

    def test_the_cache_is_bounded(self, monkeypatch):
        """A loader that varied the text length per sample would otherwise grow it without limit."""
        monkeypatch.setattr(L, "_CACHE_MAX_ENTRIES", 3)
        for text_tokens in range(4, 10):
            L.get_layout(text_tokens, 3, 4, 6, 7, "cpu")
        assert len(L._layout_cache) == 3

    def test_the_least_recently_used_entry_goes_first(self, monkeypatch):
        monkeypatch.setattr(L, "_CACHE_MAX_ENTRIES", 2)
        first = L.get_layout(4, 3, 4, 6, 7, "cpu")
        L.get_layout(5, 3, 4, 6, 7, "cpu")
        L.get_layout(4, 3, 4, 6, 7, "cpu")  # touch the first
        L.get_layout(6, 3, 4, 6, 7, "cpu")  # evicts the second, not the first
        assert L.get_layout(4, 3, 4, 6, 7, "cpu") is first


class TestVideoPacking:
    def test_it_equals_diffusers_patchify(self, reference):
        c = L.H3Constants()
        x = torch.randn(1, 24, 7, 8, 12)
        assert torch.equal(
            L.patchify_video(x, c.patch_size)[0], reference.patchify_video_latents(x, c.patch_size)
        )

    def test_the_round_trip_is_exact(self):
        x = torch.randn(2, 24, 7, 8, 12)
        rows = L.patchify_video(x, (1, 2, 2))
        assert rows.shape == (2, 7 * 4 * 6, 24 * 4)
        assert torch.equal(L.unpatchify_video(rows, (1, 2, 2), 24, 7, 8, 12), x)

    def test_a_shape_the_patch_does_not_divide_is_refused(self):
        with pytest.raises(ValueError, match="not divisible"):
            L.patchify_video(torch.randn(1, 24, 7, 9, 12), (1, 2, 2))

    def test_the_patch_is_channel_major(self):
        """Each row holds one patch, its channels outermost: the order the transformer's input projection reads."""
        x = torch.arange(2 * 1 * 2 * 2, dtype=torch.float32).reshape(1, 2, 1, 2, 2)
        row = L.patchify_video(x, (1, 2, 2))[0, 0]
        assert row.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


class TestAudioPacking:
    def test_it_is_channel_major_over_the_stereo_channels(self):
        """The reference packs the soundtrack with one stereo channel after the other."""
        left, right = torch.zeros(1, 1, 3, 4), torch.ones(1, 1, 3, 4)
        rows = L.pack_audio(torch.cat([left, right], dim=1))
        assert rows.shape == (1, 8, 3)
        assert (rows[0, :4] == 0).all() and (rows[0, 4:] == 1).all()

    def test_the_round_trip_is_exact(self):
        x = torch.randn(1, 2, 32, 9)
        assert torch.equal(L.unpack_audio(L.pack_audio(x), 2, 9), x)

    def test_it_matches_the_references_row_layout(self):
        """The reference turns a [stereo, channels, length] latent into rows with permute(0, 2, 1)."""
        x = torch.randn(2, 32, 9)
        reference_rows = x.permute(0, 2, 1).reshape(-1, 32)
        assert torch.equal(L.pack_audio(x[None])[0], reference_rows)
