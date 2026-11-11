# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
WAN's pre-encoded shards reach the trainer through this cooker and batcher.

Every WAN recipe reads them, so a regression here shows up only once a real
training run loads its first sample.
"""

import io

import pytest
import torch

from primus.backends.megatron.data.diffusion.task_encoders.video import (
    EncodedWanTaskEncoder,
    WanSample,
    cook_wan_preencoded,
    decode_wan_caption,
    load_wan_tensor,
)

LATENTS = (16, 3, 8, 8)
TEXT = (12, 32)


def _bytes(tensor):
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def _sample(key="sample_001", **fields):
    sample = {
        "__key__": key,
        "__restore_key__": lambda: key,
        "__subflavors__": {"encoding": "wan_preencoded"},
        "latents.pth": _bytes(torch.randn(LATENTS)),
        "encoder_hidden_states.pth": _bytes(torch.randn(TEXT)),
        "caption.txt": b"a red fox in the snow",
    }
    sample.update(fields)
    return sample


def test_a_shard_sample_cooks_into_a_wan_sample():
    latents, text = torch.randn(LATENTS), torch.randn(TEXT)

    cooked = cook_wan_preencoded(
        _sample(**{"latents.pth": _bytes(latents), "encoder_hidden_states.pth": _bytes(text)})
    )

    assert isinstance(cooked, WanSample)
    assert cooked.__key__ == "sample_001"
    assert torch.equal(cooked.latents, latents)
    assert torch.equal(cooked.encoder_hidden_states, text)
    assert cooked.caption == "a red fox in the snow"


@pytest.mark.parametrize("missing", ["latents.pth", "encoder_hidden_states.pth"])
def test_a_sample_without_its_features_is_rejected(missing):
    sample = _sample()
    del sample[missing]

    with pytest.raises(ValueError, match=f"missing '{missing}'"):
        cook_wan_preencoded(sample)


def test_tensors_load_from_bytes_a_path_or_a_tensor(tmp_path):
    tensor = torch.arange(6.0)
    torch.save(tensor, tmp_path / "t.pth")

    for data in (_bytes(tensor), tmp_path / "t.pth", str(tmp_path / "t.pth"), tensor):
        assert torch.equal(load_wan_tensor(data), tensor)
    assert load_wan_tensor(None) is None


def test_captions_decode_from_bytes_a_sidecar_path_or_text(tmp_path):
    (tmp_path / "caption.txt").write_text("from a file\n")

    assert decode_wan_caption(b"inline") == "inline"
    assert decode_wan_caption(tmp_path / "caption.txt") == "from a file"
    assert decode_wan_caption("not a path, just text") == "not a path, just text"
    assert decode_wan_caption(b"\xff\xfe") == ""
    assert decode_wan_caption(None) == ""


def test_the_batch_stacks_latents_and_text_features():
    samples = [cook_wan_preencoded(_sample(key=f"sample_{i}")) for i in range(3)]

    batch = EncodedWanTaskEncoder().batch(samples)

    assert set(batch) == {"latents", "encoder_hidden_states"}
    assert batch["latents"].shape == (3, *LATENTS)
    assert batch["encoder_hidden_states"].shape == (3, *TEXT)
    assert torch.equal(batch["latents"][1], samples[1].latents)
