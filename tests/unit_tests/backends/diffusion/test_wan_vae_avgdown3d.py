###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus.backends.diffusion.models.wan.vae2_2 import AvgDown3D


def test_forward_averages_grouped_channels_without_padding():
    # in_channels=1, factor=2*1*1=2 -> out_channels=1 needs group_size=2
    module = AvgDown3D(in_channels=1, out_channels=1, factor_t=2, factor_s=1)

    x = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1, 1)
    out = module(x)

    assert out.shape == (1, 1, 2, 1, 1)
    # consecutive pairs of frames along t are averaged together
    expected = x.view(1, 1, 2, 2, 1, 1).mean(dim=3)
    torch.testing.assert_close(out, expected)


def test_forward_pads_time_dimension_to_multiple_of_factor_t():
    module = AvgDown3D(in_channels=1, out_channels=1, factor_t=2, factor_s=1)

    # t=3 is not a multiple of factor_t=2, so one zero frame is left-padded.
    x = torch.arange(1 * 1 * 3 * 1 * 1, dtype=torch.float32).view(1, 1, 3, 1, 1) + 1.0
    out = module(x)

    assert out.shape == (1, 1, 2, 1, 1)
    padded = torch.nn.functional.pad(x, (0, 0, 0, 0, 1, 0))
    expected = padded.view(1, 1, 2, 2, 1, 1).mean(dim=3)
    torch.testing.assert_close(out, expected)


def test_forward_downsamples_spatial_and_channel_dims():
    # factor_s=2 halves height/width; in_channels != out_channels exercises the
    # group_size channel averaging (group_size = in_channels*factor // out_channels).
    factor_t, factor_s = 1, 2
    in_channels, out_channels = 2, 4
    module = AvgDown3D(in_channels=in_channels, out_channels=out_channels, factor_t=factor_t, factor_s=factor_s)

    b, t, h, w = 1, 2, 4, 4
    # Every element has a distinct value, so an incorrect permute order or
    # group/channel mapping would average the wrong elements together and be
    # caught by the full-tensor comparison below (a shape-only check would not).
    x = torch.arange(b * in_channels * t * h * w, dtype=torch.float32).view(b, in_channels, t, h, w)
    out = module(x)

    factor = factor_t * factor_s * factor_s
    group_size = in_channels * factor // out_channels
    t_out, h_out, w_out = t // factor_t, h // factor_s, w // factor_s

    # Independent reference: for every input element, work out which
    # (output_channel, spatial_block) slot it contributes to and average by
    # hand. This does not call AvgDown3D's own pad/view/permute/view/view/mean
    # sequence, so it is a real cross-check of the spatial block grouping and
    # the contiguous channel/group mapping, not a restatement of it.
    expected_sum = torch.zeros(b, out_channels, t_out, h_out, w_out)
    counts = torch.zeros(out_channels, t_out, h_out, w_out)
    for bi in range(b):
        for c in range(in_channels):
            for ti in range(t):
                for hi in range(h):
                    for wi in range(w):
                        to, ft = divmod(ti, factor_t)
                        ho, fh = divmod(hi, factor_s)
                        wo, fw = divmod(wi, factor_s)
                        # Matches the module's own reshape order: channels are
                        # split into (c, factor_t, factor_s, factor_s) with c
                        # slowest and the trailing factor_s (width) fastest,
                        # then that merged axis is grouped in chunks of
                        # group_size to form each output channel.
                        merged = ((c * factor_t + ft) * factor_s + fh) * factor_s + fw
                        oc, _ = divmod(merged, group_size)
                        expected_sum[bi, oc, to, ho, wo] += x[bi, c, ti, hi, wi]
                        counts[oc, to, ho, wo] += 1

    assert torch.all(counts == group_size)
    expected = expected_sum / group_size

    assert out.shape == (b, out_channels, t_out, h_out, w_out)
    torch.testing.assert_close(out, expected)


def test_init_rejects_incompatible_channel_factor():
    # factor = factor_t * factor_s * factor_s = 2; 3 * 2 = 6 is not divisible by 4.
    with pytest.raises(AssertionError):
        AvgDown3D(in_channels=3, out_channels=4, factor_t=2, factor_s=1)
