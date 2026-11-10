###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the synthetic MiniMax-H3 dataloader.

THE CLAIMS WORTH DEFENDING:

  1. THE DATASET IS FIXED, NOT RANDOM PER STEP. Sample i is the same tensor on every
     rank and in every run, with its own distinct conditioning. That is what lets an
     overfit run show a loss trend: a dataset random per step would sit flat at the
     variance of the noise, and a working model and a broken one would look alike.

  2. A BAD CLIP SHAPE FAILS AT CONSTRUCTION. A side that is not a multiple of 32, or a
     frame count the video VAE cannot encode, is a configuration error that must not
     survive until the first forward.

  3. MICRO-BATCH 1 IS ENFORCED, because the batch shares one packed layout.
"""

import sys
import types

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def synthetic(monkeypatch):
    """The module under test, with AutoModel's dataloader result type faked."""
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3.modular_pipeline")

    class DiffusionDataloaderBuild:
        def __init__(self, dataloader, sampler):
            self.dataloader, self.sampler = dataloader, sampler

    loader_mod = types.ModuleType("nemo_automodel.components.datasets.diffusion.loader")
    loader_mod.DiffusionDataloaderBuild = DiffusionDataloaderBuild
    for name in (
        "nemo_automodel",
        "nemo_automodel.components",
        "nemo_automodel.components.datasets",
        "nemo_automodel.components.datasets.diffusion",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.datasets.diffusion.loader", loader_mod)
    monkeypatch.delitem(
        sys.modules, "primus.backends.automodel.models.minimax_h3.data.synthetic", raising=False
    )
    from primus.backends.automodel.models.minimax_h3.data import synthetic

    return synthetic


SMALL = {"buckets": "256x448x39", "text_tokens": 8, "text_dim": 16}


class TestBuckets:
    def test_a_list_is_parsed_in_order(self, synthetic):
        assert synthetic.parse_buckets("768x1344x124, 768x1344x243") == [(768, 1344, 124), (768, 1344, 243)]

    @pytest.mark.parametrize("bad", ["768x1344", "abc", "768x1344x12x3", "", "768x-1x124"])
    def test_a_malformed_shape_is_refused(self, synthetic, bad):
        with pytest.raises(ValueError):
            synthetic.parse_buckets(bad)

    def test_a_frame_count_the_vae_cannot_encode_is_refused(self, synthetic):
        with pytest.raises(ValueError, match="17 \\* n \\+ 5"):
            synthetic.parse_buckets("768x1344x125")

    def test_a_side_that_is_not_a_multiple_of_32_is_refused(self, synthetic):
        with pytest.raises(ValueError, match="multiples of 32"):
            synthetic.parse_buckets("770x1344x124")


class TestDataset:
    def test_sample_shapes_and_dtypes(self, synthetic):
        sample = synthetic.SyntheticMiniMaxH3Dataset(num_samples=2, **SMALL)[0]
        # 256x448 at 39 frames: 16x spatial compression, (39 - 5) / 17 * 5 + 2 latent frames.
        assert sample["video_latents"].shape == (1, 24, 12, 16, 28)
        assert sample["video_latents"].dtype == torch.float16
        assert sample["audio_latents"].shape == (1, 2, 32, 65)
        assert sample["audio_latents"].dtype == torch.float16
        assert sample["text_embeddings"].shape == (1, 8, 16)
        assert sample["text_embeddings"].dtype == torch.bfloat16

    def test_a_sample_is_the_same_every_time(self, synthetic):
        ds = synthetic.SyntheticMiniMaxH3Dataset(num_samples=3, **SMALL)
        again = synthetic.SyntheticMiniMaxH3Dataset(num_samples=3, **SMALL)
        for key in ("video_latents", "audio_latents", "text_embeddings"):
            assert torch.equal(ds[1][key], ds[1][key])
            assert torch.equal(ds[1][key], again[1][key])

    def test_samples_differ_so_there_is_something_to_memorize(self, synthetic):
        ds = synthetic.SyntheticMiniMaxH3Dataset(num_samples=3, **SMALL)
        assert not torch.equal(ds[0]["text_embeddings"], ds[1]["text_embeddings"])
        assert not torch.equal(ds[0]["video_latents"], ds[1]["video_latents"])

    def test_the_seed_changes_the_data(self, synthetic):
        a = synthetic.SyntheticMiniMaxH3Dataset(num_samples=1, seed=1, **SMALL)[0]
        b = synthetic.SyntheticMiniMaxH3Dataset(num_samples=1, seed=2, **SMALL)[0]
        assert not torch.equal(a["video_latents"], b["video_latents"])

    def test_the_shapes_are_cycled_by_index(self, synthetic):
        ds = synthetic.SyntheticMiniMaxH3Dataset(
            num_samples=4, buckets="256x448x39,256x448x56", text_tokens=8, text_dim=16
        )
        frames = [ds[i]["video_latents"].shape[2] for i in range(4)]
        assert frames == [12, 17, 12, 17]

    @pytest.mark.parametrize("field", ["num_samples", "text_tokens", "text_dim"])
    def test_a_non_positive_size_is_refused(self, synthetic, field):
        kwargs = {"num_samples": 1, **SMALL, field: 0}
        with pytest.raises(ValueError, match=field):
            synthetic.SyntheticMiniMaxH3Dataset(**kwargs)


class TestDataloader:
    def _config(self, synthetic, **overrides):
        kwargs = {"num_samples": 4, "num_workers": 0, "pin_memory": False, **SMALL, **overrides}
        return synthetic.SyntheticMiniMaxH3DataloaderConfig(**kwargs)

    def test_a_batch_has_the_keys_the_adapter_consumes(self, synthetic):
        build = self._config(synthetic).build(dp_rank=0, dp_world_size=1, batch_size=1)
        batch = next(iter(build.dataloader))
        assert set(batch) == {"video_latents", "audio_latents", "text_embeddings", "data_type"}
        assert batch["data_type"] == "video"
        assert batch["video_latents"].shape[0] == 1
        assert build.sampler is None

    def test_micro_batch_above_one_is_refused(self, synthetic):
        """The batch shares one packed layout and one timestep table."""
        with pytest.raises(ValueError, match="micro-batch 1"):
            self._config(synthetic).build(dp_rank=0, dp_world_size=1, batch_size=2)

    def test_data_parallel_ranks_get_disjoint_shards(self, synthetic):
        cfg = self._config(synthetic, shuffle=False)
        seen = []
        for rank in range(2):
            build = cfg.build(dp_rank=rank, dp_world_size=2, batch_size=1)
            assert build.sampler is not None
            seen.append(list(build.sampler))
        assert sorted(seen[0] + seen[1]) == [0, 1, 2, 3]
        assert not set(seen[0]) & set(seen[1])

    def test_it_is_a_dataloader_automodel_checkpoints(self, synthetic):
        """AutoModel saves a dataloader's position only for a StatefulDataLoader
        (``base_recipe.is_dataloader``: that type, with state_dict and load_state_dict)."""
        from torchdata.stateful_dataloader import StatefulDataLoader

        build = self._config(synthetic).build(dp_rank=0, dp_world_size=1, batch_size=1)
        assert isinstance(build.dataloader, StatefulDataLoader)
        assert callable(build.dataloader.state_dict) and callable(build.dataloader.load_state_dict)

    @pytest.mark.parametrize("world", [1, 2])
    def test_a_resume_mid_epoch_continues_where_it_stopped(self, synthetic, world):
        """Without a saved position, a resumed run would replay the interrupted epoch from its start."""
        cfg = self._config(synthetic, num_samples=8, shuffle=True)

        def key(batch):
            return batch["video_latents"].float().sum().item()

        build = cfg.build(dp_rank=0, dp_world_size=world, batch_size=1)
        if build.sampler is not None:
            build.sampler.set_epoch(3)
        it = iter(build.dataloader)
        next(it)
        state = build.dataloader.state_dict()
        rest = [key(b) for b in it]
        assert rest  # something is left to resume into

        resumed = cfg.build(dp_rank=0, dp_world_size=world, batch_size=1)
        if resumed.sampler is not None:
            resumed.sampler.set_epoch(3)
        resumed.dataloader.load_state_dict(state)
        assert [key(b) for b in resumed.dataloader] == rest

    def test_too_few_samples_for_the_ranks_is_refused(self, synthetic):
        cfg = self._config(synthetic, num_samples=1)
        with pytest.raises(ValueError, match="no batch"):
            cfg.build(dp_rank=0, dp_world_size=2, batch_size=1)

    def test_every_field_is_a_yaml_scalar(self, synthetic):
        """The recipe validates the config keys against the dataclass fields, so a list or dict field
        could not be written in YAML."""
        import dataclasses

        for field in dataclasses.fields(synthetic.SyntheticMiniMaxH3DataloaderConfig):
            assert field.type in (int, float, str, bool, "int", "float", "str", "bool"), field.name
