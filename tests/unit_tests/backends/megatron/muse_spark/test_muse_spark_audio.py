###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU tests of the Muse Spark waveform audio encoder (small config)."""

import pytest
import torch

from primus.backends.megatron.core.models.muse_spark.audio_encoder import (
    MuseSparkAudioEncoder,
    audio_config_from_hf,
    audio_features,
    merge_audio,
    merge_padded_audio,
    padded_audio_features,
    samples_per_token,
)

SMALL = {
    "hidden_sizes": [8, 16],
    "depths": [1, 2],
    "patch_sizes": [4, 3],
    "conv_kernel_size": 3,
    "mlp_ratio": 2,
    "hidden_act": "gelu",
    "rms_norm_eps": 1e-6,
    "out_hidden_size": 12,
}


def _enc():
    torch.manual_seed(0)
    return MuseSparkAudioEncoder(audio_config_from_hf(SMALL)).eval()


def test_config_defaults_and_checks():
    cfg = audio_config_from_hf({"num_hidden_layers": 4})
    assert cfg["patch_sizes"] == [16, 6, 5, 4] and samples_per_token(cfg) == 1920
    assert audio_config_from_hf(None) is None
    with pytest.raises(ValueError):
        audio_config_from_hf({**SMALL, "depths": [1]})
    with pytest.raises(ValueError):
        audio_config_from_hf({**SMALL, "hidden_act": "nope"})


def test_parameter_names_match_hf():
    names = set(_enc().state_dict())
    assert "layers.0.patch_embedder.proj.weight" in names
    assert "layers.1.conv_blocks.1.conv.weight" in names
    assert "layers.1.conv_blocks.1.mlp.fc2.bias" in names
    assert {"norm.weight", "proj.weight", "proj.bias", "post_proj_norm.weight"} <= names
    assert _enc().layers[1].conv_blocks[0].conv.weight.shape == (16, 1, 3)  # depthwise


def test_shape_and_causality():
    enc = _enc()
    x = torch.randn(1, 12 * 5)
    with torch.no_grad():
        y = enc(x)
        assert y.shape == (1, 5, 12)
        x2 = x.clone()
        x2[:, 12 * 3 :] = torch.randn(1, 12 * 2)
        y2 = enc(x2)
    torch.testing.assert_close(y[:, :3], y2[:, :3])
    assert not torch.allclose(y[:, 3:], y2[:, 3:])
    with pytest.raises(ValueError):
        enc(torch.randn(1, 13))


def test_features_drop_padding_and_embed_norm():
    enc = _enc()
    x = torch.randn(2, 36)
    mask = torch.zeros(2, 36, dtype=torch.long)
    mask[0] = 1
    mask[1, :13] = 1  # 13 samples -> ceil(13 / 12) = 2 tokens
    with torch.no_grad():
        f = audio_features(enc, x, mask, 1e-5)
        full = enc(x)
    assert f.shape == (3 + 2, 12)
    ref = torch.cat([full[0], full[1, :2]])
    ref = ref * torch.pow(ref.pow(2).mean(-1, keepdim=True) + 1e-5, -0.5)
    torch.testing.assert_close(f, ref)
    assert audio_features(enc, x, None, 1e-5).shape == (6, 12)


def test_merge_audio():
    torch.manual_seed(0)
    emb = torch.randn(5, 2, 3)  # [S, B, H]
    orig = emb.clone()
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[0, 1] = mask[1, 3] = mask[1, 4] = True
    feats = torch.arange(9.0).reshape(3, 3)
    out = merge_audio(emb, mask, feats)
    torch.testing.assert_close(out[1, 0], feats[0])
    torch.testing.assert_close(out[3, 1], feats[1])
    torch.testing.assert_close(out[4, 1], feats[2])
    # every non-audio row keeps its text embedding
    keep = ~mask.t()  # [S, B]
    torch.testing.assert_close(out[keep], orig[keep])
    with pytest.raises(ValueError):
        merge_audio(emb, mask, feats[:2])


@pytest.mark.parametrize("masked", [True, False])
def test_merge_padded_audio_matches_the_boolean_index_path(masked):
    # the model's sync-free path (padded features, on-device gather) == HF's
    # get_audio_features + masked_scatter, including padded clips and token order
    enc = _enc()
    x = torch.randn(3, 36)  # 3 tokens per clip
    mask = torch.zeros(3, 36, dtype=torch.long)
    mask[0] = 1
    mask[1, :13] = 1  # 2 tokens
    mask[2, :5] = 1  # 1 token
    m = mask if masked else None
    with torch.no_grad():
        ref_feats = audio_features(enc, x, m, 1e-5)
        feats, n_tok = padded_audio_features(enc, x, m, 1e-5)
    n = ref_feats.shape[0]
    torch.manual_seed(1)
    emb = torch.randn(7, 2, 12)  # [S, B, H]
    amask = torch.zeros(2, 7, dtype=torch.bool)
    slots = [(b, s) for b in range(2) for s in range(7)]
    pos = (slots[::2] + slots[1::2])[:n]  # spread over both rows, batch-major order kept by the mask
    for b, s in pos:
        amask[b, s] = True
    torch.testing.assert_close(
        merge_padded_audio(emb, amask, feats, n_tok), merge_audio(emb, amask, ref_feats)
    )
    with pytest.raises(ValueError, match="do not match"):
        merge_padded_audio(emb, amask, feats[:2], None if n_tok is None else n_tok[:2])
