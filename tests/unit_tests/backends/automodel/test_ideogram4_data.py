###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the Ideogram-4 cache dataloader, and its batch contract with the
synthetic one (whose own tests are in ``test_ideogram4_synthetic_data.py``).

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
supposed to be able to swap one for the other, and that a cache entry cannot
reach outside its own directory.

The cache tests build a real cache on disk in a throwaway directory. Nothing here
needs a GPU or any encoder weights, and AutoModel is needed only for the type
``build()`` returns: without it, that one call is replaced for each test.
"""

import importlib.util
import json
import types

import pytest

torch = pytest.importorskip("torch")

from primus.backends.automodel.models.ideogram4.data import (  # noqa: E402
    cache,
    synthetic,
)

FEATURE_DIM = 8
CHANNELS = 4


@pytest.fixture(autouse=True)
def automodel_build_type(monkeypatch):
    """The real build type when AutoModel is importable, else a stand-in with the
    same two fields, patched onto the two loader modules only."""
    try:
        found = importlib.util.find_spec("nemo_automodel.components.datasets.diffusion.loader")
    except (ModuleNotFoundError, ValueError):
        found = None
    if found is None:
        for module in (cache, synthetic):
            monkeypatch.setattr(module, "_dataloader_build", lambda **kw: types.SimpleNamespace(**kw))


# --------------------------------------------------------------------------- #
# Cache fixtures                                                              #
# --------------------------------------------------------------------------- #
def write_cache(root, text_lengths, grid=(2, 3), feature_dim=FEATURE_DIM):
    """Write a cache whose features are identifiable, so padding can be located.

    Sample i, token j gets the value ``(i + 1) * 100 + j``, which is non-zero
    everywhere and unique per position. Padding is zero, so a test can tell
    exactly which rows hold real tokens. Stored as the builder stores them:
    fp16 latents and fp16 features.
    """
    samples_dir = root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, length in enumerate(text_lengths):
        features = torch.zeros(length, feature_dim, dtype=torch.float16)
        for j in range(length):
            features[j] = (i + 1) * 100 + j
        torch.save(
            {
                "image_latents": torch.full((CHANNELS, grid[0], grid[1]), float(i), dtype=torch.float16),
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

    def test_the_derived_width_fits_every_caption(self, cache_dir):
        dataset = cache.Ideogram4CacheDataset(str(cache_dir))
        batch = cache._collate([dataset[i] for i in range(4)], text_width=dataset.max_text_length)
        assert batch["text_lengths"].tolist() == [2, 4, 3, 4]


class TestNoTruncation:
    """Truncating would keep the head of the chat-templated sequence and drop the
    end of the prompt and the generation-prompt tokens every sample ends with."""

    def test_an_explicit_width_below_the_longest_caption_is_refused(self, cache_dir):
        config = cache.Ideogram4CacheDataloaderConfig(
            cache_dir=str(cache_dir), num_workers=0, max_text_tokens=3
        )
        with pytest.raises(ValueError, match="below the longest caption in the cache"):
            config.build(dp_rank=0, dp_world_size=1, batch_size=2)

    def test_a_sample_longer_than_the_width_is_an_error_not_a_cut(self, cache_dir):
        dataset = cache.Ideogram4CacheDataset(str(cache_dir))
        with pytest.raises(ValueError, match="disagree"):
            cache._collate([dataset[1]], text_width=2)

    @pytest.mark.parametrize("value", [-2, -5, -128])
    def test_a_width_below_minus_one_is_refused(self, cache_dir, value):
        """Rather than silently meaning per-batch padding."""
        config = cache.Ideogram4CacheDataloaderConfig(
            cache_dir=str(cache_dir), num_workers=0, max_text_tokens=value
        )
        with pytest.raises(ValueError, match="-1 \\(per batch\\)"):
            config.build(dp_rank=0, dp_world_size=1, batch_size=2)


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

    def test_latents_are_promoted_to_float32(self, cache_dir):
        """The cache stores them narrow to keep it small; the pipeline noises them
        in float32."""
        dataset = cache.Ideogram4CacheDataset(str(cache_dir))
        assert dataset[0]["image_latents"].dtype == torch.float32

    def test_metadata_without_a_sample_list_is_refused(self, tmp_path):
        root = tmp_path / "cache"
        root.mkdir()
        (root / "metadata.json").write_text("{}")
        with pytest.raises(ValueError, match="no 'samples' list"):
            cache.Ideogram4CacheDataset(str(root))

    @pytest.mark.parametrize("key", ["grid_h", "grid_w", "in_channels", "llm_features_dim"])
    def test_metadata_missing_a_dimension_is_refused(self, cache_dir, key):
        """Rather than defaulting a grid to 0."""
        metadata = json.loads((cache_dir / "metadata.json").read_text())
        del metadata[key]
        (cache_dir / "metadata.json").write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match=key):
            cache.Ideogram4CacheDataset(str(cache_dir))

    def test_a_cache_for_another_model_is_refused(self, cache_dir):
        metadata = json.loads((cache_dir / "metadata.json").read_text())
        metadata["model_type"] = "flux"
        (cache_dir / "metadata.json").write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="built for 'flux'"):
            cache.Ideogram4CacheDataset(str(cache_dir))

    def test_a_sample_that_disagrees_with_its_entry_is_refused(self, cache_dir):
        """The reviewer's probe: an index claiming one token for a longer sample
        used to be truncated silently."""
        metadata = json.loads((cache_dir / "metadata.json").read_text())
        metadata["samples"][1]["text_length"] = 1
        (cache_dir / "metadata.json").write_text(json.dumps(metadata))
        dataset = cache.Ideogram4CacheDataset(str(cache_dir))
        with pytest.raises(ValueError, match="text rows is 4, the index says 1"):
            dataset[1]

    def test_a_sample_with_another_grid_is_refused(self, tmp_path):
        root = write_cache(tmp_path / "cache", [2, 3])
        metadata = json.loads((root / "metadata.json").read_text())
        metadata["grid_w"] = 4
        (root / "metadata.json").write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="latent shape"):
            cache.Ideogram4CacheDataset(str(root))[0]

    def test_a_cache_too_small_for_one_batch_per_rank_is_refused(self, cache_dir):
        config = cache.Ideogram4CacheDataloaderConfig(cache_dir=str(cache_dir), num_workers=0)
        with pytest.raises(ValueError, match="no batch"):
            config.build(dp_rank=0, dp_world_size=8, batch_size=2)


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
            assert batch["llm_features"].is_floating_point()
            assert batch["text_lengths"].dtype == torch.long
            assert batch["text_lengths"].ndim == 1
            assert batch["data_type"] == "image"

    def test_the_feature_dtype_is_each_producers_own(self, cache_dir):
        """The contract is floating point in the producer's dtype, cast by the
        adapter: the cache keeps its stored fp16, which halves the host-to-device
        copy, and the synthetic loader generates fp32."""
        assert self.cache_batch(cache_dir)["llm_features"].dtype == torch.float16
        assert self.synthetic_batch()["llm_features"].dtype == torch.float32

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


# --------------------------------------------------------------------------- #
# Cache builder                                                               #
# --------------------------------------------------------------------------- #
class TestCacheBuilder:
    @pytest.fixture
    def sources(self, tmp_path):
        from PIL import Image

        images, captions = tmp_path / "images", tmp_path / "captions"
        images.mkdir()
        captions.mkdir()
        for i in range(3):
            Image.new("RGB", (8, 8)).save(images / f"{i}.png")
            (captions / f"{i}.txt").write_text(f"caption {i}")
        return images, captions

    @staticmethod
    def fake_processor(monkeypatch, grid, interrupt_after=None, tokens=2, fail_load=False, fail_encode=False):
        """The real processor with its models replaced: ``tokens`` is what every
        caption counts as, and each encode is recorded."""
        from primus.backends.automodel.models.ideogram4 import processor

        calls = types.SimpleNamespace(images=[], texts=[])

        class FakeProcessor(processor.Ideogram4Processor):
            def load_models(self):
                if fail_load:
                    raise OSError("weights not found")

            def count_text_tokens(self, prompt):
                return tokens

            @staticmethod
            def preprocess_image(image, resolution):
                return torch.zeros(1, 3, 8, 8)

            def encode_image(self, image_tensor):
                if interrupt_after is not None and len(calls.images) == interrupt_after:
                    raise KeyboardInterrupt
                if fail_encode:
                    raise RuntimeError("HIP error: invalid device function")
                calls.images.append(1)
                return torch.zeros(CHANNELS, *grid, dtype=torch.float16)

            def encode_text(self, prompt, max_text_tokens):
                calls.texts.append(prompt)
                return {
                    "llm_features": torch.ones(tokens, FEATURE_DIM, dtype=torch.float16),
                    "text_length": tokens,
                }

        monkeypatch.setattr(processor, "Ideogram4Processor", FakeProcessor)
        return calls

    @staticmethod
    def build(images, captions, out, **kwargs):
        from primus.backends.automodel.models.ideogram4.data import build

        return build.build_cache(
            image_dir=str(images), caption_dir=str(captions), output_dir=str(out), **kwargs
        )

    def test_a_build_writes_an_index_the_loader_reads(self, tmp_path, sources, monkeypatch):
        images, captions = sources
        out = tmp_path / "cache"
        self.fake_processor(monkeypatch, grid=(2, 2))
        metadata = self.build(images, captions, out, num_samples=3)

        assert metadata["num_samples"] == 3
        assert metadata["skipped"] == {"no_caption": 0, "too_long": 0, "failed": 0}
        assert json.loads((out / "metadata.json").read_text()) == metadata
        assert not (out / "metadata.json.partial").exists()
        dataset = cache.Ideogram4CacheDataset(str(out))
        assert len(dataset) == 3
        assert dataset[0]["llm_features"].shape == (2, FEATURE_DIM)

    def test_a_smaller_rebuild_leaves_no_stale_samples(self, tmp_path, sources, monkeypatch):
        images, captions = sources
        out = tmp_path / "cache"
        self.fake_processor(monkeypatch, grid=(2, 2))
        self.build(images, captions, out, num_samples=3)
        self.build(images, captions, out, num_samples=1)
        assert sorted(p.name for p in (out / "samples").iterdir()) == ["0.pt"]

    def test_a_failed_load_leaves_the_previous_cache_intact(self, tmp_path, sources, monkeypatch):
        images, captions = sources
        out = tmp_path / "cache"
        self.fake_processor(monkeypatch, grid=(2, 2))
        self.build(images, captions, out, num_samples=3)

        self.fake_processor(monkeypatch, grid=(2, 2), fail_load=True)
        with pytest.raises(OSError):
            self.build(images, captions, out, num_samples=3)
        assert len(cache.Ideogram4CacheDataset(str(out))) == 3

    def test_an_over_long_caption_is_dropped_before_any_encoding(self, tmp_path, sources, monkeypatch):
        images, captions = sources
        calls = self.fake_processor(monkeypatch, grid=(2, 2), tokens=200)
        with pytest.raises(RuntimeError, match="3 with a caption over 128 tokens"):
            self.build(images, captions, tmp_path / "cache", num_samples=3)
        assert calls.images == [] and calls.texts == []

    def test_the_images_are_closed(self, tmp_path, sources, monkeypatch):
        """A large build would otherwise hold a file handle per image until the
        garbage collector got to it."""
        from PIL import Image

        real_open, closed = Image.open, []

        class Tracked:
            def __init__(self, path):
                self.image = real_open(path)

            def __enter__(self):
                return self.image

            def __exit__(self, *exc):
                closed.append(True)
                self.image.close()

        monkeypatch.setattr(Image, "open", Tracked)
        images, captions = sources
        self.fake_processor(monkeypatch, grid=(2, 2))
        self.build(images, captions, tmp_path / "cache", num_samples=3)
        assert len(closed) == 3

    def test_a_systemic_failure_stops_the_build(self, tmp_path, monkeypatch):
        from PIL import Image

        images, captions = tmp_path / "images", tmp_path / "captions"
        images.mkdir()
        captions.mkdir()
        for i in range(12):
            Image.new("RGB", (8, 8)).save(images / f"{i:02d}.png")
            (captions / f"{i:02d}.txt").write_text(f"caption {i}")
        self.fake_processor(monkeypatch, grid=(2, 2), fail_encode=True)
        with pytest.raises(RuntimeError, match="8 images in a row failed"):
            self.build(images, captions, tmp_path / "cache", num_samples=12)

    def test_an_undecodable_caption_is_skipped_not_fatal(self, tmp_path, sources, monkeypatch):
        images, captions = sources
        (captions / "1.txt").write_bytes(b"\xff\xfe caption")
        self.fake_processor(monkeypatch, grid=(2, 2))
        metadata = self.build(images, captions, tmp_path / "cache", num_samples=3)
        assert metadata["num_samples"] == 2
        assert metadata["skipped"]["failed"] == 1

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"max_text_tokens": 0}, "max_text_tokens must be at least 1"),
            ({"max_text_tokens": -4}, "max_text_tokens must be at least 1"),
            ({"resolution": 0}, "multiple of 16"),
            ({"resolution": 250}, "multiple of 16"),
        ],
    )
    def test_bad_arguments_are_refused_before_the_encoders_load(
        self, tmp_path, sources, monkeypatch, kwargs, match
    ):
        images, captions = sources
        self.fake_processor(monkeypatch, grid=(2, 2), fail_load=True)
        with pytest.raises(ValueError, match=match):
            self.build(images, captions, tmp_path / "cache", num_samples=3, **kwargs)
        assert not (tmp_path / "cache").exists()

    def test_an_interrupted_rebuild_does_not_keep_the_old_index(self, tmp_path, sources, monkeypatch):
        """The old metadata would describe samples the rebuild has already overwritten."""
        from primus.backends.automodel.models.ideogram4.data import build

        images, captions = sources
        out = tmp_path / "cache"
        self.fake_processor(monkeypatch, grid=(2, 2))
        build.build_cache(
            image_dir=str(images), caption_dir=str(captions), output_dir=str(out), num_samples=3
        )
        assert (out / "metadata.json").exists()

        self.fake_processor(monkeypatch, grid=(4, 4), interrupt_after=1)
        with pytest.raises(KeyboardInterrupt):
            build.build_cache(
                image_dir=str(images), caption_dir=str(captions), output_dir=str(out), num_samples=3
            )
        assert not (out / "metadata.json").exists()
        with pytest.raises(FileNotFoundError, match="automodel-cache"):
            cache.Ideogram4CacheDataset(str(out))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
