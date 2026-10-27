###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the synthetic Ideogram-4 dataloader.

The synthetic set exists to produce an overfit signal, so the property that
matters is that it is genuinely fixed per index while still varying across
indices. Nothing here needs a GPU or any encoder weights, and AutoModel is
needed only for the type ``build()`` returns: without it, that one call is
replaced for the duration of each test.
"""

import importlib.util
import types

import pytest

torch = pytest.importorskip("torch")

from primus.backends.automodel.models.ideogram4.data import synthetic  # noqa: E402

FEATURE_DIM = 8
CHANNELS = 4


@pytest.fixture(autouse=True)
def automodel_build_type(monkeypatch):
    """The real build type when AutoModel is importable, else a stand-in with the
    same two fields, patched onto this module only."""
    try:
        found = importlib.util.find_spec("nemo_automodel.components.datasets.diffusion.loader")
    except (ModuleNotFoundError, ValueError):
        found = None
    if found is None:
        monkeypatch.setattr(synthetic, "_dataloader_build", lambda **kw: types.SimpleNamespace(**kw))


def loader_config(**overrides):
    kwargs = dict(
        num_samples=4,
        in_channels=CHANNELS,
        grid_h=2,
        grid_w=3,
        max_text_tokens=6,
        min_text_tokens=4,
        llm_features_dim=FEATURE_DIM,
        num_workers=0,
        pin_memory=False,
    )
    kwargs.update(overrides)
    return synthetic.SyntheticIdeogram4DataloaderConfig(**kwargs)


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

    def test_an_inverted_range_is_refused(self):
        """Clamping it would give every sample one length and lose the ragged batches."""
        with pytest.raises(ValueError, match="min_text_tokens"):
            self.make(min_text_tokens=28, max_text_tokens=16)

    @pytest.mark.parametrize(
        "field",
        [
            "num_samples",
            "in_channels",
            "grid_h",
            "grid_w",
            "max_text_tokens",
            "min_text_tokens",
            "llm_features_dim",
        ],
    )
    def test_a_non_positive_size_is_refused(self, field):
        with pytest.raises(ValueError, match=f"{field} must be at least 1"):
            self.make(**{field: 0})

    def test_the_left_pad_region_is_zero(self):
        """As the cache loader leaves it: the real tokens are the last n rows."""
        dataset = self.make()
        for idx in range(8):
            sample = dataset[idx]
            pad = 6 - int(sample["text_lengths"])
            assert torch.all(sample["llm_features"][:pad] == 0)
            assert torch.all(sample["llm_features"][pad:].abs().sum(dim=-1) > 0)

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


class TestSyntheticLoader:
    def test_sharing_features_is_warned_about(self, caplog):
        """It silently destroys the loss signal the loader exists to produce."""
        with caplog.at_level("WARNING"):
            loader_config(share_text_features=True).build(dp_rank=0, dp_world_size=1, batch_size=2)
        assert [r for r in caplog.records if "not" in r.getMessage() and "meaningful" in r.getMessage()]

    def test_a_loader_with_no_batch_is_refused(self):
        """Otherwise training would run zero steps per epoch without saying so."""
        with pytest.raises(ValueError, match="no batch"):
            loader_config(num_samples=4).build(dp_rank=0, dp_world_size=8, batch_size=2)

    def test_drop_last_defaults_on(self):
        """A short final batch changes the packing's shape and recompiles."""
        assert synthetic.SyntheticIdeogram4DataloaderConfig().drop_last is True

    def test_a_batch_meets_the_adapter_contract(self):
        built = loader_config().build(dp_rank=0, dp_world_size=1, batch_size=2)
        batch = next(iter(built.dataloader))
        assert batch["image_latents"].shape == (2, CHANNELS, 2, 3)
        assert batch["llm_features"].shape == (2, 6, FEATURE_DIM)
        assert batch["text_lengths"].shape == (2,)
        assert built.sampler is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
