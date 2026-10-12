###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Parameterized CausalConv3d.forward streaming-cache tests (PRPUNDIT-24).

Wan 2.1 (``vae2_1.py``) and Wan 2.2 (``vae2_2.py``) ship the same causal 3D
conv primitive. One Jira covers both implementations; this suite runs the
same cache/padding oracle against each module rather than duplicating a
per-file test.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import primus.backends.diffusion.models.wan.vae2_1 as wan21
import primus.backends.diffusion.models.wan.vae2_2 as wan22

_VAE_MODULES = (
    pytest.param(wan21, id="wan2_1"),
    pytest.param(wan22, id="wan2_2"),
)


def _make_conv(vae_mod, *, kernel_size=(3, 3, 3), padding=(1, 1, 1), in_ch=2, out_ch=3, seed=0):
    torch.manual_seed(seed)
    return vae_mod.CausalConv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)


def _manual_conv(conv, x, cache_x=None):
    """Independent F.pad + Conv3d of CausalConv3d.forward's cache/padding rule."""
    padding = list(conv._padding)
    if cache_x is not None and padding[4] > 0:
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    padded = F.pad(x, padding)
    return F.conv3d(padded, conv.weight, conv.bias, stride=conv.stride, dilation=conv.dilation)


def _chunked_streaming(conv, x, cache_t: int):
    """Production streaming protocol: last CACHE_T frames of the previous chunk."""
    mid = x.shape[2] // 2
    chunks = (x[:, :, :mid], x[:, :, mid:])
    outputs = []
    cache_x = None
    for chunk in chunks:
        outputs.append(conv(chunk, cache_x=cache_x))
        cache_x = chunk[:, :, -cache_t:].clone()
    return torch.cat(outputs, dim=2)


@pytest.mark.parametrize("vae_mod", _VAE_MODULES)
def test_full_sequence_matches_manual_causal_pad_oracle(vae_mod):
    conv = _make_conv(vae_mod)
    x = torch.randn(1, 2, 6, 5, 5)

    out = conv(x)
    expected = _manual_conv(conv, x)

    # Causal padding is 2*pad_t on the front and 0 on the back, so T is preserved.
    assert out.shape == (1, 3, 6, 5, 5)
    torch.testing.assert_close(out, expected)
    assert list(conv._padding) == [1, 1, 1, 1, 2, 0]


@pytest.mark.parametrize("vae_mod", _VAE_MODULES)
def test_chunked_cache_matches_full_sequence(vae_mod):
    conv = _make_conv(vae_mod)
    cache_t = vae_mod.CACHE_T
    x = torch.randn(2, 2, 7, 4, 4)

    full = conv(x)
    chunked = _chunked_streaming(conv, x, cache_t)

    torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-5)
    assert chunked.shape == full.shape


@pytest.mark.parametrize("vae_mod", _VAE_MODULES)
def test_cache_x_reduces_front_padding_by_cached_frames(vae_mod):
    conv = _make_conv(vae_mod)
    cache_x = torch.randn(1, 2, vae_mod.CACHE_T, 5, 5)
    x = torch.randn(1, 2, 4, 5, 5)

    out = conv(x, cache_x=cache_x)
    expected = _manual_conv(conv, x, cache_x=cache_x)

    # CACHE_T fully covers the 2 frames of causal front padding, so output T
    # equals the new chunk length rather than growing by the cached history.
    assert out.shape == (1, 3, 4, 5, 5)
    torch.testing.assert_close(out, expected)
    remaining_front = max(0, conv._padding[4] - cache_x.shape[2])
    assert remaining_front == 0


@pytest.mark.parametrize("vae_mod", _VAE_MODULES)
def test_partial_cache_x_still_pads_remaining_front_frames(vae_mod):
    conv = _make_conv(vae_mod)
    cache_x = torch.randn(1, 2, 1, 5, 5)
    x = torch.randn(1, 2, 4, 5, 5)

    out = conv(x, cache_x=cache_x)
    expected = _manual_conv(conv, x, cache_x=cache_x)

    torch.testing.assert_close(out, expected)
    remaining_front = conv._padding[4] - cache_x.shape[2]
    assert remaining_front == 1
    assert out.shape[2] == x.shape[2]


@pytest.mark.parametrize("vae_mod", _VAE_MODULES)
def test_cache_x_is_ignored_when_temporal_padding_is_zero(vae_mod):
    conv = _make_conv(vae_mod, kernel_size=(1, 3, 3), padding=(0, 1, 1))
    x = torch.randn(1, 2, 4, 5, 5)
    cache_x = torch.randn(1, 2, 2, 5, 5)

    torch.testing.assert_close(conv(x, cache_x=cache_x), conv(x))


@pytest.mark.parametrize("vae_mod", _VAE_MODULES)
def test_first_output_frame_is_causal_without_cache(vae_mod):
    conv = _make_conv(vae_mod)
    x = torch.randn(1, 2, 4, 5, 5)
    out_full = conv(x)

    x_truncated = x.clone()
    x_truncated[:, :, 1:] = torch.randn_like(x_truncated[:, :, 1:])
    out_truncated = conv(x_truncated)

    torch.testing.assert_close(out_full[:, :, 0], out_truncated[:, :, 0])
