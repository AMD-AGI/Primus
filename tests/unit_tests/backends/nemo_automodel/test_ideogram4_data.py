###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the two Ideogram-4 dataloaders.

THE TWO CLAIMS THAT ARE SILENT WHEN WRONG, and so get the most attention here:

  1. LEFT-PADDING ORIENTATION. The adapter computes its text offset as
     ``width - n``, so the real tokens must occupy the LAST n rows. Padding on the
     right would put real features where the adapter expects padding and padding
     where it expects text. Nothing errors -- the model simply trains on
     conditioning that does not line up with its own position ids. The test below
     writes distinguishable features and checks which rows they land in.

  2. THE TEXT WIDTH IS CONSTANT ACROSS BATCHES. torch.compile keys its graphs on
     input shapes, so a width that follows the per-batch longest caption
     recompiles on most batches and blocks compilation outright. The test feeds
     batches with deliberately different longest captions and asserts the shape
     does not move.

Also checked: that the two loaders agree on the batch contract, since a config is
supposed to be able to swap one for the other; that the synthetic dataset is
genuinely fixed per index, which is the whole basis of the overfit signal it
exists to produce; and that a cache entry cannot reach outside its own directory.

The cache tests build a real cache on disk in a throwaway directory. Nothing here
needs a GPU or any encoder weights.
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "nemo_automodel.components.datasets.diffusion.loader",
    reason="the loaders subclass AutoModel's dataloader build type",
)

from primus.backends.nemo_automodel.models.ideogram4.data import (  # noqa: E402
    cache,
    synthetic,
)

FEATURE_DIM = 8
CHANNELS = 4


# --------------------------------------------------------------------------- #
# Cache fixtures                                                              #
# --------------------------------------------------------------------------- #
def write_cache(root, text_lengths, grid=(2, 3), feature_dim=FEATURE_DIM):
    """Write a cache whose features are identifiable, so padding can be located.

    Sample i, token j gets the value ``(i + 1) * 100 + j``, which is non-zero
    everywhere and unique per position. Padding is zero, so a test can tell
    exactly which rows hold real tokens.
    """
    samples_dir = root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, length in enumerate(text_lengths):
        features = torch.zeros(length, feature_dim)
        for j in range(length):
            features[j] = (i + 1) * 100 + j
        torch.save(
            {
                "image_latents": torch.full((CHANNELS, grid[0], grid[1]), float(i)),
                "llm_features": features,
                "text_length": length,
            },
            samples_dir / f"{i}.pt",
        )
        entries.append({"cache_file": f"samples/{i}.pt", "text_length": length})

    (root / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "ideogram4",
                "grid_h": grid[0],
                "grid_w": grid[1],
                "in_channels": CHANNELS,
                "llm_features_dim": feature_dim,
                "num_samples": len(entries),
                "samples": entries,
            }
        )
    )
    return root


@pytest.fixture
def cache_dir(tmp_path):
    return write_cache(tmp_path / "cache", [2, 4, 3, 4])


# --------------------------------------------------------------------------- #
# 1. Left-padding orientation                                                 #
# --------------------------------------------------------------------------- #
class TestLeftPadding:
    def test_the_real_tokens_land_in_the_last_rows(self, cache_dir):
        """The claim the adapter depends on. Its text offset is width - n, so the
        real tokens have to be at the END of the text region."""
        dataset = cache.Ideogram4CacheDataset(str(cache_dir))
        batch = cache._collate([dataset[0], dataset[1]], text_width=4)

        features = batch["llm_features"]
        assert features.shape == (2, 4, FEATURE_DIM)

        # Sample 0 has 2 real tokens in a width of 4, so rows 0 and 1 are padding.
        assert torch.all(features[0, :2] == 0), "the padding is not at the front"
        assert torch.all(features[0, 2] == 100), "the first real token is misplaced"
        assert torch.all(features[0, 3] == 101), "the second real token is misplaced"

        # Sample 1 fills the width, so there is no padding.
        assert torch.all(features[1, 0] == 200)
        assert torch.all(features[1, 3] == 203)

    def test_the_token_order_within_the_region_is_preserved(self, cache_dir):
        """Left-padding shifts the block; it must not reverse or rotate it."""
        dataset = cache.Ideogram4CacheDataset(str(cache_dir))
        batch = cache._collate([dataset[2]], text_width=6)
        features = batch["llm_features"]
        assert torch.all(features[0, :3] == 0)
        for offset, expected in enumerate((300, 301, 302)):
            assert torch.all(features[0, 3 + offset] == expected)

    def test_the_reported_lengths_match_the_real_token_counts(self, cache_dir):
        """The adapter reconstructs the offset from these, so a length that
        disagreed with the padding would misplace the whole text region."""
        dataset = cache.Ideogram4CacheDataset(str(cache_dir))
        batch = cache._collate([dataset[i] for i in range(4)], text_width=4)
        assert batch["text_lengths"].tolist() == [2, 4, 3, 4]

        features = batch["llm_features"]
        for row, length in enumerate(batch["text_lengths"].tolist()):
            pad_rows = 4 - length
            assert torch.all(features[row, :pad_rows] == 0)
            if pad_rows < 4:
                assert torch.all(features[row, pad_rows] != 0)


# --------------------------------------------------------------------------- #
# 2. A constant text width                                                    #
# --------------------------------------------------------------------------- #
class TestConstantTextWidth:
    def test_the_derived_width_is_the_longest_caption_in_the_cache(self, cache_dir):
        dataset = cache.Ideogram4CacheDataset(str(cache_dir))
        assert dataset.max_text_length == 4

    def test_it_is_read_from_metadata_without_loading_samples(self, tmp_path, monkeypatch):
        """A dataset-level constant is only cheap if getting it does not mean
        reading every sample, which for a real cache would be minutes of I/O
        before the first step."""
        root = write_cache(tmp_path / "cache", [2, 7, 3])

        def refuse(*args, **kwargs):
            raise AssertionError("a sample was loaded while deriving the text width")

        monkeypatch.setattr(torch, "load", refuse)
        dataset = cache.Ideogram4CacheDataset(str(root))
        assert dataset.max_text_length == 7

    def test_the_shape_does_not_move_between_batches(self, cache_dir):
        """THE COMPILE REQUIREMENT. Batches with different longest captions must
        still produce the same shape."""
        dataset = cache.Ideogram4CacheDataset(str(cache_dir))
        width = dataset.max_text_length

        shapes = set()
        for indices in ([0, 2], [1, 3], [0, 1], [2, 2], [0, 0]):
            batch = cache._collate([dataset[i] for i in indices], text_width=width)
            shapes.add(tuple(batch["llm_features"].shape))
        assert len(shapes) == 1, f"the sequence length moved between batches: {shapes}"

    def test_the_per_batch_width_does_move(self, cache_dir):
        """The behaviour the default exists to avoid, asserted so the difference is
        not theoretical: this is what -1 buys and what it costs."""
        dataset = cache.Ideogram4CacheDataset(str(cache_dir))
        shapes = set()
        for indices in ([0, 2], [1, 3]):
            batch = cache._collate([dataset[i] for i in indices], text_width=0)
            shapes.add(tuple(batch["llm_features"].shape))
        assert len(shapes) == 2, "expected the per-batch width to vary with the data"

    def test_the_derived_width_never_truncates(self, cache_dir, caplog):
        dataset = cache.Ideogram4CacheDataset(str(cache_dir))
        with caplog.at_level("WARNING"):
            batch = cache._collate([dataset[i] for i in range(4)], text_width=dataset.max_text_length)
        assert batch["text_lengths"].tolist() == [2, 4, 3, 4]
        assert not [r for r in caplog.records if "truncated" in r.getMessage()]


class TestTruncation:
    def test_an_explicit_width_that_is_too_small_truncates_and_warns(self, cache_dir, monkeypatch, caplog):
        monkeypatch.setattr(cache, "_WARNED", set())
        dataset = cache.Ideogram4CacheDataset(str(cache_dir))
        with caplog.at_level("WARNING"):
            batch = cache._collate([dataset[1]], text_width=2)

        assert batch["text_lengths"].tolist() == [2]
        # The FIRST tokens are kept, so the caption is cut at its tail.
        assert torch.all(batch["llm_features"][0, 0] == 200)
        assert torch.all(batch["llm_features"][0, 1] == 201)

        warnings = [r for r in caplog.records if "truncated" in r.getMessage()]
        assert len(warnings) == 1
        assert "max_text_tokens" in warnings[0].getMessage()

    def test_it_warns_only_once(self, cache_dir, monkeypatch, caplog):
        monkeypatch.setattr(cache, "_WARNED", set())
        dataset = cache.Ideogram4CacheDataset(str(cache_dir))
        with caplog.at_level("WARNING"):
            cache._collate([dataset[1]], text_width=2)
            cache._collate([dataset[3]], text_width=2)
        assert len([r for r in caplog.records if "truncated" in r.getMessage()]) == 1


# --------------------------------------------------------------------------- #
# The width tri-state, resolved by build()                                    #
# --------------------------------------------------------------------------- #
class TestWidthResolution:
    @staticmethod
    def widths_used(config):
        """Build a loader and recover the width its collate was bound to."""
        built = config.build(dp_rank=0, dp_world_size=1, batch_size=2)
        return built.dataloader.collate_fn.keywords["text_width"]

    def test_zero_derives_from_the_cache(self, cache_dir):
        config = cache.Ideogram4CacheDataloaderConfig(
            cache_dir=str(cache_dir), num_workers=0, max_text_tokens=0
        )
        assert self.widths_used(config) == 4

    def test_a_positive_value_is_used_verbatim(self, cache_dir):
        config = cache.Ideogram4CacheDataloaderConfig(
            cache_dir=str(cache_dir), num_workers=0, max_text_tokens=16
        )
        assert self.widths_used(config) == 16

    def test_minus_one_selects_the_per_batch_width(self, cache_dir):
        config = cache.Ideogram4CacheDataloaderConfig(
            cache_dir=str(cache_dir), num_workers=0, max_text_tokens=-1
        )
        assert self.widths_used(config) == 0

    def test_the_per_batch_choice_is_warned_about(self, cache_dir, caplog):
        """It is invisible until compilation is switched on, and then only shows up
        as a run that never stops recompiling."""
        config = cache.Ideogram4CacheDataloaderConfig(
            cache_dir=str(cache_dir), num_workers=0, max_text_tokens=-1
        )
        with caplog.at_level("WARNING"):
            config.build(dp_rank=0, dp_world_size=1, batch_size=2)
        assert [r for r in caplog.records if "cannot be compiled" in r.getMessage()]

    def test_the_collate_is_picklable_for_the_workers(self, cache_dir):
        """A closure here would break num_workers > 0, and only there."""
        import pickle

        config = cache.Ideogram4CacheDataloaderConfig(cache_dir=str(cache_dir), num_workers=0)
        built = config.build(dp_rank=0, dp_world_size=1, batch_size=2)
        pickle.loads(pickle.dumps(built.dataloader.collate_fn))


# --------------------------------------------------------------------------- #
# Cache integrity                                                             #
# --------------------------------------------------------------------------- #
class TestCacheIntegrity:
    def test_a_missing_cache_says_how_to_build_one(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="automodel-cache"):
            cache.Ideogram4CacheDataset(str(tmp_path))

    def test_an_empty_cache_is_refused(self, tmp_path):
        root = tmp_path / "cache"
        root.mkdir()
        (root / "metadata.json").write_text(json.dumps({"samples": []}))
        with pytest.raises(ValueError, match="no samples"):
            cache.Ideogram4CacheDataset(str(root))

    def test_an_entry_cannot_reach_outside_the_cache(self, cache_dir):
        """The metadata is data, and a cache directory can come from anywhere."""
        metadata = json.loads((cache_dir / "metadata.json").read_text())
        metadata["samples"][0]["cache_file"] = "../../etc/passwd"
        (cache_dir / "metadata.json").write_text(json.dumps(metadata))

        dataset = cache.Ideogram4CacheDataset(str(cache_dir))
        with pytest.raises(ValueError, match="outside"):
            dataset[0]

    def test_latents_are_promoted_to_float32(self, tmp_path):
        """The cache stores them narrow to keep it small; the pipeline noises them
        in float32."""
        root = tmp_path / "cache"
        samples = root / "samples"
        samples.mkdir(parents=True)
        torch.save(
            {
                "image_latents": torch.zeros(CHANNELS, 2, 3, dtype=torch.float16),
                "llm_features": torch.zeros(2, FEATURE_DIM),
                "text_length": 2,
            },
            samples / "0.pt",
        )
        (root / "metadata.json").write_text(
            json.dumps({"samples": [{"cache_file": "samples/0.pt", "text_length": 2}]})
        )
        dataset = cache.Ideogram4CacheDataset(str(root))
        assert dataset[0]["image_latents"].dtype == torch.float32


# --------------------------------------------------------------------------- #
# The synthetic loader                                                        #
# --------------------------------------------------------------------------- #
class TestSyntheticDataset:
    @staticmethod
    def make(**overrides):
        kwargs = dict(
            num_samples=8,
            in_channels=CHANNELS,
            grid_h=2,
            grid_w=3,
            max_text_tokens=6,
            min_text_tokens=4,
            llm_features_dim=FEATURE_DIM,
            feature_scale=0.1,
            latent_scale=1.0,
            seed=7,
        )
        kwargs.update(overrides)
        return synthetic.SyntheticIdeogram4Dataset(**kwargs)

    def test_an_index_yields_the_same_sample_every_time(self):
        """The basis of the overfit signal. A dataset that were random per read
        would give a loss that sits at the variance of the noise, and a working
        model would look exactly like a broken one."""
        first = self.make()
        second = self.make()
        for idx in range(8):
            assert torch.equal(first[idx]["image_latents"], second[idx]["image_latents"])
            assert torch.equal(first[idx]["llm_features"], second[idx]["llm_features"])

    def test_different_indices_get_different_conditioning(self):
        """Also load-bearing for the signal: identical conditioning across samples
        makes the targets contradictory and leaves nothing to memorize."""
        dataset = self.make()
        assert not torch.equal(dataset[0]["llm_features"], dataset[1]["llm_features"])
        assert not torch.equal(dataset[0]["image_latents"], dataset[1]["image_latents"])

    def test_the_lengths_are_ragged_within_the_configured_range(self):
        dataset = self.make()
        lengths = {int(dataset[i]["text_lengths"]) for i in range(8)}
        assert lengths == {4, 5, 6}

    def test_a_degenerate_range_is_clamped(self):
        dataset = self.make(min_text_tokens=99, max_text_tokens=6)
        assert dataset.min_text_tokens == 6
        assert {int(dataset[i]["text_lengths"]) for i in range(8)} == {6}

    def test_share_text_features_hands_back_one_buffer(self):
        dataset = self.make(share_text_features=True)
        assert dataset[0]["llm_features"] is dataset[3]["llm_features"]

    def test_but_the_lengths_still_vary(self):
        """Otherwise it would change the shapes it is meant to leave alone."""
        dataset = self.make(share_text_features=True)
        assert len({int(dataset[i]["text_lengths"]) for i in range(8)}) > 1

    def test_the_in_memory_cache_returns_the_same_object(self):
        dataset = self.make(cache_in_memory=True)
        assert dataset[2] is dataset[2]

    def test_the_cache_does_not_change_the_values(self):
        cached = self.make(cache_in_memory=True)
        plain = self.make()
        assert torch.equal(cached[5]["llm_features"], plain[5]["llm_features"])

    def test_at_least_one_sample_is_produced(self):
        assert len(self.make(num_samples=0)) == 1


class TestSyntheticLoaderWarning:
    def test_sharing_features_is_warned_about(self, caplog):
        """It silently destroys the loss signal the loader exists to produce."""
        config = synthetic.SyntheticIdeogram4DataloaderConfig(
            num_samples=4,
            in_channels=CHANNELS,
            grid_h=2,
            grid_w=3,
            max_text_tokens=6,
            min_text_tokens=4,
            llm_features_dim=FEATURE_DIM,
            num_workers=0,
            share_text_features=True,
        )
        with caplog.at_level("WARNING"):
            config.build(dp_rank=0, dp_world_size=1, batch_size=2)
        assert [r for r in caplog.records if "not" in r.getMessage() and "meaningful" in r.getMessage()]


# --------------------------------------------------------------------------- #
# The shared contract                                                         #
# --------------------------------------------------------------------------- #
class TestSharedBatchContract:
    """A config is supposed to be able to swap one loader for the other, which is
    only true if the batches are interchangeable."""

    EXPECTED = {"image_latents", "llm_features", "text_lengths", "data_type"}

    def synthetic_batch(self):
        config = synthetic.SyntheticIdeogram4DataloaderConfig(
            num_samples=4,
            in_channels=CHANNELS,
            grid_h=2,
            grid_w=3,
            max_text_tokens=4,
            min_text_tokens=2,
            llm_features_dim=FEATURE_DIM,
            num_workers=0,
            shuffle=False,
        )
        built = config.build(dp_rank=0, dp_world_size=1, batch_size=2)
        return next(iter(built.dataloader))

    def cache_batch(self, cache_dir):
        config = cache.Ideogram4CacheDataloaderConfig(
            cache_dir=str(cache_dir), num_workers=0, shuffle=False, drop_last=False
        )
        built = config.build(dp_rank=0, dp_world_size=1, batch_size=2)
        return next(iter(built.dataloader))

    def test_the_keys_match(self, cache_dir):
        assert set(self.synthetic_batch()) == self.EXPECTED
        assert set(self.cache_batch(cache_dir)) == self.EXPECTED

    def test_the_ranks_and_dtypes_match(self, cache_dir):
        for batch in (self.synthetic_batch(), self.cache_batch(cache_dir)):
            assert batch["image_latents"].ndim == 4
            assert batch["image_latents"].dtype == torch.float32
            assert batch["llm_features"].ndim == 3
            assert batch["text_lengths"].dtype == torch.long
            assert batch["text_lengths"].ndim == 1
            assert batch["data_type"] == "image"

    def test_the_batch_dimension_agrees_across_every_key(self, cache_dir):
        for batch in (self.synthetic_batch(), self.cache_batch(cache_dir)):
            size = batch["image_latents"].shape[0]
            assert batch["llm_features"].shape[0] == size
            assert batch["text_lengths"].shape[0] == size

    def test_no_reported_length_exceeds_the_feature_width(self, cache_dir):
        """The adapter refuses this, so a loader producing it would fail the run at
        the first step."""
        for batch in (self.synthetic_batch(), self.cache_batch(cache_dir)):
            assert int(batch["text_lengths"].max()) <= batch["llm_features"].shape[1]


class TestDistributedSampling:
    def test_a_sampler_is_only_created_when_there_is_more_than_one_rank(self, cache_dir):
        single = cache.Ideogram4CacheDataloaderConfig(cache_dir=str(cache_dir), num_workers=0).build(
            dp_rank=0, dp_world_size=1, batch_size=2
        )
        assert single.sampler is None

    def test_the_ranks_get_disjoint_samples(self, cache_dir):
        """Two ranks training on the same samples would be silently wasted work."""
        built = [
            cache.Ideogram4CacheDataloaderConfig(
                cache_dir=str(cache_dir), num_workers=0, shuffle=False, drop_last=False
            ).build(dp_rank=rank, dp_world_size=2, batch_size=1)
            for rank in range(2)
        ]
        indices = [set(b.sampler) for b in built]
        assert indices[0].isdisjoint(indices[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
