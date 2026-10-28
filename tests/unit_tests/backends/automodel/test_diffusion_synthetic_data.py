###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the synthetic FLUX and Wan dataloaders."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nemo_automodel")

from primus.backends.automodel.models.flux.data.synthetic import (  # noqa: E402
    SyntheticFluxDataloaderConfig,
)
from primus.backends.automodel.models.wan.data.synthetic import (  # noqa: E402
    SyntheticWanDataloaderConfig,
    SyntheticWanDataset,
)

_SMALL_WAN = dict(in_channels=2, latent_t=1, latent_h=4, latent_w=6, text_seq_len=3, text_dim=8)
_SMALL_FLUX = dict(in_channels=2, latent_h=4, latent_w=4, text_seq_len=3, t5_features_dim=8, pooled_dim=4)
_SINGLE_PROCESS = dict(num_workers=0, pin_memory=False)


def _wan(**overrides):
    return SyntheticWanDataset(**{"num_samples": 4, **_SMALL_WAN, **overrides})


class TestWanDataset:
    def test_shared_text_does_not_depend_on_the_first_index_read(self):
        first_read_late = _wan(share_text_features=True)
        late = first_read_late[3]["text_embeddings"]
        early = _wan(share_text_features=True)[0]["text_embeddings"]
        assert torch.equal(late, early)
        assert late.dtype == torch.bfloat16

    def test_unshared_text_differs_per_index(self):
        ds = _wan(share_text_features=False)
        assert not torch.equal(ds[0]["text_embeddings"], ds[1]["text_embeddings"])

    def test_bucket_metadata_follows_the_latent_size(self):
        """The real cache records (width, height) in pixels and width / height."""
        sample = _wan(latent_h=4, latent_w=6)[0]
        assert sample["bucket_resolution"].tolist() == [48, 32]
        assert sample["aspect_ratio"] == pytest.approx(1.5)

    @pytest.mark.parametrize("field", ["latent_h", "latent_w"])
    def test_an_odd_latent_size_is_refused(self, field):
        with pytest.raises(ValueError, match="must be even"):
            _wan(**{field: 5})

    @pytest.mark.parametrize("field", ["num_samples", "latent_t", "text_seq_len", "text_dim"])
    def test_a_non_positive_size_is_refused(self, field):
        with pytest.raises(ValueError, match="must be positive"):
            _wan(**{field: 0})


class TestWanDataloader:
    def test_a_batch_has_the_collated_shapes(self):
        config = SyntheticWanDataloaderConfig(num_samples=4, **_SMALL_WAN, **_SINGLE_PROCESS)
        batch = next(iter(config.build(dp_rank=0, dp_world_size=1, batch_size=2).dataloader))
        assert batch["video_latents"].shape == (2, 2, 1, 4, 6)
        assert batch["text_embeddings"].shape == (2, 3, 8)

    def test_a_loader_with_no_batch_is_refused(self):
        config = SyntheticWanDataloaderConfig(num_samples=2, drop_last=True, **_SMALL_WAN, **_SINGLE_PROCESS)
        with pytest.raises(ValueError, match="gives no batch"):
            config.build(dp_rank=0, dp_world_size=1, batch_size=4)


class TestFluxDataloader:
    def _config(self, **overrides):
        return SyntheticFluxDataloaderConfig(
            **{"num_samples": 4, **_SMALL_FLUX, **_SINGLE_PROCESS, **overrides}
        )

    def test_a_batch_has_the_collated_shapes(self):
        batch = next(iter(self._config().build(dp_rank=0, dp_world_size=1, batch_size=2).dataloader))
        assert batch["image_latents"].shape == (2, 2, 4, 4)
        assert batch["text_embeddings"].shape == (2, 3, 8)

    @pytest.mark.parametrize("field", ["latent_h", "latent_w"])
    def test_an_odd_latent_size_is_refused(self, field):
        with pytest.raises(ValueError, match="must be even"):
            self._config(**{field: 5}).build(dp_rank=0, dp_world_size=1, batch_size=1)

    @pytest.mark.parametrize("field", ["num_samples", "text_seq_len", "pooled_dim"])
    def test_a_non_positive_size_is_refused(self, field):
        with pytest.raises(ValueError, match="must be positive"):
            self._config(**{field: 0}).build(dp_rank=0, dp_world_size=1, batch_size=1)

    def test_a_loader_with_no_batch_is_refused(self):
        with pytest.raises(ValueError, match="gives no batch"):
            self._config(num_samples=2, drop_last=True).build(dp_rank=0, dp_world_size=1, batch_size=4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
