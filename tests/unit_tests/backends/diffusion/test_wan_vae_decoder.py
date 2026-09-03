##########################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###########################################################################

import torch

from primus.backends.diffusion.models.wan.vae2_2 import Decoder3d, count_conv3d


def _make_decoder():
    # A tiny decoder with a single temporal-upsample stage: enough to exercise
    # cache indexing, the "Rep" sentinel, and first_chunk propagation without
    # a channel-changing residual shortcut (which vae2_2's ResidualBlock does
    # not route through the feat_cache, and would desync feat_idx from
    # count_conv3d(decoder)).
    torch.manual_seed(0)
    decoder = Decoder3d(
        dim=8,
        z_dim=4,
        dim_mult=[1, 1],
        num_res_blocks=1,
        attn_scales=[],
        temperal_upsample=[True],
        dropout=0.0,
    )
    decoder.eval()
    return decoder


def _decode_streaming(decoder, x, conv_num):
    """Replay WanVAE_.decode's per-frame chunked-decode protocol: one latent
    frame per call, a single feat_cache list reused across calls, feat_idx
    reset to [0] for every frame, and first_chunk=True only on the first
    call."""
    feat_map = [None] * conv_num
    outputs = []
    for i in range(x.shape[2]):
        feat_idx = [0]
        frame_out = decoder(
            x[:, :, i : i + 1, :, :],
            feat_cache=feat_map,
            feat_idx=feat_idx,
            first_chunk=(i == 0),
        )
        # Every CausalConv3d on the cached path consumes exactly one slot.
        assert feat_idx[0] == conv_num
        outputs.append(frame_out)
    return outputs, feat_map


def test_forward_streaming_decode_matches_wan22_chunked_protocol():
    # Direct Decoder3d output is 12 channels (patchified latent space); the
    # conversion to 3 RGB channels happens later, in unpatchify.
    decoder = _make_decoder()
    conv_num = count_conv3d(decoder)
    x = torch.randn(1, 4, 3, 4, 4)
    feat_map = [None] * conv_num

    outputs = []
    with torch.no_grad():
        for i in range(x.shape[2]):
            feat_idx = [0]
            frame_out = decoder(
                x[:, :, i : i + 1, :, :],
                feat_cache=feat_map,
                feat_idx=feat_idx,
                first_chunk=(i == 0),
            )
            # Every CausalConv3d on the cached path consumes exactly one slot.
            assert feat_idx[0] == conv_num

            if i == 0:
                # After the first chunk every cache slot is populated; the
                # temporal-upsample stage's slot holds the "Rep" sentinel
                # until a second chunk gives it real history to work with.
                assert all(slot is not None for slot in feat_map)
                assert any(isinstance(slot, str) and slot == "Rep" for slot in feat_map)
            else:
                # Once real history is available, "Rep" must have been
                # replaced by an actual cached tensor.
                assert not any(isinstance(slot, str) and slot == "Rep" for slot in feat_map)

            outputs.append(frame_out)

    # First chunk has no cached history yet, so its upsample3d stage can only
    # emit its own frame; later chunks have history and double their
    # temporal contribution.
    assert outputs[0].shape == (1, 12, 1, 8, 8)
    assert outputs[1].shape == (1, 12, 2, 8, 8)
    assert outputs[2].shape == (1, 12, 2, 8, 8)

    out = torch.cat(outputs, dim=2)
    assert out.shape == (1, 12, 5, 8, 8)
    assert torch.isfinite(out).all()


def test_forward_streaming_decode_is_deterministic_after_cache_reset():
    decoder = _make_decoder()
    conv_num = count_conv3d(decoder)
    x = torch.randn(1, 4, 3, 4, 4)

    with torch.no_grad():
        outputs_1, _ = _decode_streaming(decoder, x, conv_num)
        outputs_2, _ = _decode_streaming(decoder, x, conv_num)

    assert torch.equal(torch.cat(outputs_1, dim=2), torch.cat(outputs_2, dim=2))
