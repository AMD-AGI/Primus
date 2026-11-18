###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
# See LICENSE for license information.
###############################################################################

"""Deterministic Hadamard rotation shared by the FLUX MXFP4 linear paths."""

from __future__ import annotations

import math

import torch

_HADAMARD_CACHE: dict[tuple, torch.Tensor] = {}


def hadamard_matrix(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Sylvester Hadamard of order ``n``, scaled by 1/sqrt(n) to be orthogonal."""
    key = (n, dtype, str(device))
    cached = _HADAMARD_CACHE.get(key)
    if cached is None:
        if n <= 0 or n & (n - 1):
            raise ValueError(f"Hadamard order must be a power of two, got {n}")
        h = torch.ones(1, 1, dtype=torch.float64, device=device)
        while h.shape[0] < n:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        cached = (h / math.sqrt(n)).to(dtype)
        _HADAMARD_CACHE[key] = cached
    return cached


def rotate_last_dim(x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
    """Apply the block-diagonal Hadamard ``h`` along ``x``'s last dim.

    Both operands of a GEMM are rotated along the dim they contract over, so
    the rotations cancel -- (XH)(WH)^T = X H H^T W^T = X W^T -- leaving the
    result unchanged while spreading per-channel outliers across each
    32-element micro-scaling block.

    ``h`` is passed in rather than looked up so nothing in the traced path
    touches a module-level cache.
    """
    if h is None:
        return x
    n = h.shape[0]
    *lead, d = x.shape
    if d % n:
        raise ValueError(f"last dim {d} is not divisible by Hadamard order {n}")
    return torch.matmul(x.reshape(*lead, d // n, n), h.to(x.dtype)).reshape(*lead, d)
