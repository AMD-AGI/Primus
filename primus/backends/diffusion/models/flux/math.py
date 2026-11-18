###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# Adapted from Black Forest Labs FLUX official implementation.

from __future__ import annotations

import os

import torch
from einops import rearrange
from torch import Tensor

from primus.backends.diffusion.attention import attention as backend_attention
from primus.backends.diffusion.models.flux.rope_triton import (
    fused_apply_rope,
    fused_rope_supported,
)


def _fused_rope_enabled() -> bool:
    """Read ``PRIMUS_FLUX_TRITON_ROPE`` at call time.

    The benchmark configs export it from the YAML ``env`` block, which can
    happen after this module is imported. flux.1-dev and flux.1-schnell stay
    on the broadcast multiply unless the variable is ``1``.
    """
    return os.getenv("PRIMUS_FLUX_TRITON_ROPE", "0") == "1"


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    x = backend_attention(
        q=q,
        k=k,
        v=v,
        dtype=q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16,
    )
    return rearrange(x, "B L H D -> B L (H D)")


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2).float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    if _fused_rope_enabled() and xq.shape == xk.shape and fused_rope_supported(xq, freqs_cis):
        return fused_apply_rope(xq, xk, freqs_cis)
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
