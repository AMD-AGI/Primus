###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU checks of Inkling's short-convolution switch (``inkling_sconv_impl``): the eager reference
with ``seq_idx`` (packed documents) equals running every document on its own, and without
``seq_idx`` it is the plain causal ``F.conv1d``. The fused kernels are checked on a GPU by the L2
op spec beside this file (``l2_inkling_ops.py``)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")

import torch.nn.functional as F  # noqa: E402

from primus.backends.megatron.core.transformer.inkling.inkling_layers import (  # noqa: E402
    InklingShortConv,
    eager_causal_conv1d,
)


def _plain(x, w):
    k = w.shape[-1]
    return F.conv1d(x, w.unsqueeze(1), padding=k - 1, groups=x.shape[1])[..., : x.shape[-1]]


def test_eager_without_seq_idx_is_the_causal_conv():
    g = torch.Generator().manual_seed(0)
    x, w = torch.randn(2, 8, 37, generator=g), torch.randn(8, 4, generator=g)
    torch.testing.assert_close(eager_causal_conv1d(x, w), _plain(x, w), rtol=0, atol=1e-6)
    one = torch.zeros(2, 37, dtype=torch.int32)
    torch.testing.assert_close(eager_causal_conv1d(x, w, one), _plain(x, w), rtol=1e-6, atol=1e-6)


def test_eager_seq_idx_resets_at_document_boundaries():
    g = torch.Generator().manual_seed(1)
    lengths = [[5, 1, 20, 11], [37]]
    x, w = torch.randn(2, 8, 37, generator=g), torch.randn(8, 4, generator=g)
    seq = torch.stack([torch.repeat_interleave(torch.arange(len(ls)), torch.tensor(ls)) for ls in lengths])
    got = eager_causal_conv1d(x, w, seq.to(torch.int32))
    for b, ls in enumerate(lengths):
        start = 0
        for n in ls:
            ref = _plain(x[b : b + 1, :, start : start + n], w)
            torch.testing.assert_close(got[b : b + 1, :, start : start + n], ref, rtol=1e-6, atol=1e-6)
            start += n


def test_short_conv_impls_and_cpu_fallback():
    with pytest.raises(ValueError, match="inkling_sconv_impl"):
        InklingShortConv(8, 4, impl="fast")
    conv = InklingShortConv(8, 4, impl="auto")  # on the CPU: always the eager path
    with torch.no_grad():
        conv.weight.normal_()
    x = torch.randn(13, 2, 8)
    ref = _plain(x.permute(1, 2, 0), conv.weight).permute(2, 0, 1) + x
    torch.testing.assert_close(conv(x), ref, rtol=1e-6, atol=1e-6)
