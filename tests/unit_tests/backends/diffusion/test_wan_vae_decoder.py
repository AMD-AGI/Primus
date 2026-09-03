###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

import torch

from primus.backends.diffusion.models.wan.vae2_1 import Decoder3d, count_conv3d


def _make_decoder():
    torch.manual_seed(0)
    decoder = Decoder3d(
        dim=8,
        z_dim=4,
        dim_mult=[1, 2],
        num_res_blocks=1,
        attn_scales=[],
        temperal_upsample=[True],
        dropout=0.0,
    )
    decoder.eval()
    return decoder


def test_forward_without_cache_upsamples_spatially_and_projects_to_rgb():
    decoder = _make_decoder()
    x = torch.randn(1, 4, 2, 4, 4)

    with torch.no_grad():
        out = decoder(x)

    # Without a feature cache, temporal upsampling is skipped (it only kicks
    # in once cached history is available), but the spatial upsample3d/2d
    # stages still double H and W, and the head projects back to 3 channels.
    assert out.shape == (1, 3, 2, 8, 8)
    assert torch.isfinite(out).all()


def test_forward_is_deterministic_in_eval_mode():
    decoder = _make_decoder()
    x = torch.randn(1, 4, 2, 4, 4)

    with torch.no_grad():
        out1 = decoder(x)
        out2 = decoder(x)

    assert torch.equal(out1, out2)


def test_forward_with_feat_cache_grows_temporal_dimension_across_chunks():
    decoder = _make_decoder()
    x = torch.randn(1, 4, 2, 4, 4)

    conv_num = count_conv3d(decoder)
    feat_map = [None] * conv_num

    outputs = []
    with torch.no_grad():
        for i in range(x.shape[2]):
            feat_idx = [0]
            frame_out = decoder(x[:, :, i : i + 1, :, :], feat_cache=feat_map, feat_idx=feat_idx)
            # Every CausalConv3d call consumes one cache slot.
            assert feat_idx[0] == conv_num
            outputs.append(frame_out)

    cached_out = torch.cat(outputs, dim=2)

    # The first chunk is a "Rep" placeholder (no temporal doubling yet); every
    # subsequent chunk doubles its own temporal contribution once the
    # upsample3d time_conv has cached history to work with.
    assert cached_out.shape == (1, 3, 3, 8, 8)
    assert torch.isfinite(cached_out).all()
    # All cache slots should be populated after a full pass.
    assert all(slot is not None for slot in feat_map)


def test_forward_defaults_feat_idx_when_not_provided():
    decoder = _make_decoder()
    x = torch.randn(1, 4, 2, 4, 4)

    conv_num = count_conv3d(decoder)
    feat_map = [None] * conv_num

    with torch.no_grad():
        # feat_idx=None should be treated the same as passing [0].
        out = decoder(x[:, :, :1, :, :], feat_cache=feat_map, feat_idx=None)

    assert out.shape[0] == 1
    assert out.shape[1] == 3
    assert torch.isfinite(out).all()
