###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Synthetic Ideogram-4 dataloader: the ``share_text_features`` perf lever.

The lever exists because a realistic caption width makes per-sample generation of the
[width, 53248] feature buffer cost more CPU than the training step costs GPU, which turns a
throughput run into a measurement of the dataloader. These tests pin the two properties that
make sharing safe to use for that purpose -- shapes and per-sample text lengths are unchanged
-- and the one property that makes it unsafe for the overfit smoke, namely that the samples
stop being distinct. They run on CPU with a tiny feature dim.
"""
import torch

from primus.backends.nemo_automodel.models.ideogram4.data.synthetic import (
    SyntheticIdeogram4Dataset,
)

DIM = 8


def _dataset(share: bool, **kwargs) -> SyntheticIdeogram4Dataset:
    params = dict(
        num_samples=16,
        in_channels=4,
        grid_h=2,
        grid_w=2,
        max_text_tokens=12,
        min_text_tokens=4,
        llm_features_dim=DIM,
        feature_scale=0.1,
        latent_scale=1.0,
        seed=1234,
        share_text_features=share,
    )
    params.update(kwargs)
    return SyntheticIdeogram4Dataset(**params)


def test_default_is_unshared():
    """Off by default: the overfit smoke needs distinct samples for its loss signal."""
    assert _dataset(share=False)[0]["llm_features"].data_ptr() != _dataset(share=False)[1]["llm_features"].data_ptr()

    unshared = _dataset(share=False)
    assert not torch.equal(unshared[0]["llm_features"], unshared[1]["llm_features"])


def test_shared_features_are_one_buffer():
    """The whole point: one allocation, handed back for every index."""
    shared = _dataset(share=True)
    first = shared[0]["llm_features"]
    assert all(shared[i]["llm_features"].data_ptr() == first.data_ptr() for i in range(1, 16))


def test_shape_is_identical_to_unshared():
    """Sharing must not change what the model sees, or throughput is not comparable."""
    shared, unshared = _dataset(share=True), _dataset(share=False)
    assert shared[5]["llm_features"].shape == unshared[5]["llm_features"].shape
    assert shared[5]["llm_features"].dtype == unshared[5]["llm_features"].dtype


def test_text_lengths_still_vary_per_sample():
    """Raggedness is the property the var-len benchmarks turn on; sharing must preserve it."""
    shared = _dataset(share=True)
    lengths = {int(shared[i]["text_lengths"]) for i in range(16)}
    assert len(lengths) > 1
    assert lengths == {int(_dataset(share=False)[i]["text_lengths"]) for i in range(16)}


def test_latents_still_distinct_per_sample():
    """Only the text buffer is shared; the images are cheap and stay per-index."""
    shared = _dataset(share=True)
    assert not torch.equal(shared[0]["image_latents"], shared[1]["image_latents"])


def test_shared_buffer_is_deterministic_across_instances():
    """Same seed, same buffer -- reruns of a benchmark must feed identical data."""
    assert torch.equal(_dataset(share=True)[0]["llm_features"], _dataset(share=True)[7]["llm_features"])


def test_composes_with_cache_in_memory():
    """The two perf levers are meant to be used together and must not fight."""
    both = _dataset(share=True, cache_in_memory=True)
    first = both[3]
    assert both[3]["llm_features"].data_ptr() == first["llm_features"].data_ptr()
    assert both[4]["llm_features"].data_ptr() == first["llm_features"].data_ptr()
    assert torch.equal(both[3]["image_latents"], first["image_latents"])
