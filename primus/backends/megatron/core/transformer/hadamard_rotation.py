###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Hadamard rotation for the DeepSeek-V4 lightning indexer.

The indexer scores ``ReLU(q_h . k_s)`` over compressed keys, and the reference
rotates both operands by a normalised Hadamard matrix first (DeepSeek-V3.2's
``inference/model.py``, and the open-source training reference's
``rotate_activation``). The rotation is orthogonal, so it preserves the inner
product exactly in infinite precision -- what it buys is spreading each
coordinate's energy across the whole vector, so no single channel dominates the
low-precision QK product the indexer is designed to run in.

The reference calls into the ``fast_hadamard_transform`` extension and
hard-requires BF16. That extension is not guaranteed present here, so this
module prefers it when available and otherwise multiplies by a cached Sylvester
matrix: the same transform, one GEMM, and it works on CPU and in FP32 so the
behaviour is testable without a GPU.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

try:  # pragma: no cover - depends on the image
    from fast_hadamard_transform import hadamard_transform as _fast_hadamard_transform
except Exception:  # pragma: no cover
    _fast_hadamard_transform = None

__all__ = ["rotate_activation"]

# Sylvester matrices keyed by (n, device, dtype). Tiny (128x128 at the V4
# indexer width) and reused every layer, every step.
_MATRIX_CACHE: Dict[Tuple[int, str, torch.dtype], torch.Tensor] = {}

# The extension only accepts half precision.
_FAST_PATH_DTYPES = (torch.float16, torch.bfloat16)


def _sylvester(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Unnormalised ``n x n`` Sylvester Hadamard matrix (entries +-1).

    Symmetric, so ``x @ H`` and ``x @ H.T`` agree and the caller does not have
    to care which side it multiplies on.
    """
    key = (n, str(device), dtype)
    cached = _MATRIX_CACHE.get(key)
    if cached is not None:
        return cached

    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < n:
        top = torch.cat([h, h], dim=1)
        bottom = torch.cat([h, -h], dim=1)
        h = torch.cat([top, bottom], dim=0)
    if h.shape[0] != n:
        raise ValueError(f"Hadamard rotation needs a power-of-two width, got {n}")

    _MATRIX_CACHE[key] = h
    return h


def rotate_activation(x: torch.Tensor, *, scale: Optional[float] = None) -> torch.Tensor:
    """Apply the normalised Hadamard rotation over ``x``'s last dimension.

    ``scale`` defaults to ``n ** -0.5``, which makes the transform orthonormal
    (and matches ``rotate_activation`` in the reference).
    """
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard rotation needs a power-of-two width, got {n}")
    norm = n**-0.5 if scale is None else scale

    if _fast_hadamard_transform is not None and x.is_cuda and x.dtype in _FAST_PATH_DTYPES:
        return _fast_hadamard_transform(x, scale=norm)

    h = _sylvester(n, x.device, x.dtype)
    return torch.matmul(x, h) * norm
