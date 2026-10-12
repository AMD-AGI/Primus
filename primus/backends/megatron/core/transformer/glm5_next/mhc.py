###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GLM-5.3 manifold-constrained hyper-connections (mHC).

Transcribes SGLang ``_mhc_pre_torch`` / ``_mhc_post_torch``
(``sglang/kernels/ops/layernorm/mhc.py``) on a sequence-first stream tensor
``[s, b, n, h]``::

    mixes = (x_flat @ fn^T) * rsqrt(mean(x_flat^2) + rms_eps)   # [s, b, (2+n)n]
    pre   = sigmoid(mixes[:n]   * scale[0] + base[:n]) + hc_eps
    post  = post_mult * sigmoid(mixes[n:2n] * scale[1] + base[n:2n])
    comb  = sinkhorn(softmax(mixes[2n:].view(n, n) * scale[2] + base[2n:]))
    layer_input = sum_i pre_i * x_i
    out_j = post_j * f(layer_input) + sum_i comb[i, j] * x_i

Note the ``comb`` contraction runs over its **first** index (``comb^T @ x``);
DeepSeek-V4's ``HyperMixer.expand`` contracts the second, so it is not reused.

The math is per token, so it runs unchanged on a sequence-parallel shard.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["mhc_pre", "mhc_post", "mhc_expand", "mhc_contract"]


def _sinkhorn(comb: Tensor, iters: int, eps: float) -> Tensor:
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return comb


def mhc_pre(
    streams: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    post_mult: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """``streams [s, b, n, h]`` -> ``(layer_input [s, b, h], post [s, b, n], comb [s, b, n, n])``."""
    s, b, n, h = streams.shape
    x32 = streams.float()
    x_flat = x32.reshape(s, b, n * h)
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(x_flat, fn.float()) * rsqrt
    scale = scale.float()
    base = base.float()

    pre = torch.sigmoid(mixes[..., :n] * scale[0] + base[:n]) + hc_eps
    post = post_mult * torch.sigmoid(mixes[..., n : 2 * n] * scale[1] + base[n : 2 * n])
    comb = mixes[..., 2 * n :].reshape(s, b, n, n) * scale[2] + base[2 * n :].view(n, n)
    comb = _sinkhorn(comb, sinkhorn_iters, hc_eps)

    layer_input = (pre.unsqueeze(-1) * x32).sum(dim=2).to(streams.dtype)
    return layer_input, post, comb


def mhc_post(x: Tensor, streams: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    """``out[..., j, :] = post_j * x + sum_i comb[i, j] * streams[..., i, :]``; dtype of ``x``."""
    out = post.unsqueeze(-1) * x.float().unsqueeze(-2) + torch.einsum(
        "sbij,sbih->sbjh", comb, streams.float()
    )
    return out.to(x.dtype)


def mhc_expand(x: Tensor, n: int) -> Tensor:
    """``[s, b, h]`` -> ``[s, b, n, h]`` (SGLang ``hc_expand``: ``x.repeat(1, n)``)."""
    return x.unsqueeze(-2).expand(*x.shape[:-1], n, x.shape[-1]).contiguous()


def mhc_contract(streams: Tensor) -> Tensor:
    """``[s, b, n, h]`` -> ``[s, b, h]`` mean over streams (SGLang ``hc_contract``)."""
    return streams.mean(dim=-2)
