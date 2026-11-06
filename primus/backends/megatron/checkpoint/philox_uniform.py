###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""NumPy model of the values PyTorch's GPU ``uniform_`` writes.

On a CUDA or ROCm GPU, PyTorch fills a 32-bit float tensor through
``distribution_elementwise_grid_stride_kernel`` with Philox4x32-10 draws. The kernel launches
256-thread blocks on ``min(ceil(numel / 256), compute_units * threads_per_cu / 256)`` blocks.
Thread ``t`` draws Philox blocks with counter ``(offset / 4 + k, t)`` under key ``(seed, 0)``,
and element ``i`` takes component ``(i // S) % 4`` of draw ``i // (4 S)``, where ``S`` is the
number of threads in the grid. The generator offset then advances by
``((numel - 1) // (4 S) + 1) * 4``.

The grid size depends on the compute-unit count of the GPU, so one seed gives different values
on GPUs with different counts: large tensors map their elements to other draws, and every
later tensor starts from another offset. Tensors with at most one element per thread take the
same values on every GPU that launches enough threads for them.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

_M0, _M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
_W0, _W1 = 0x9E3779B9, 0xBB67AE85
_LOW32 = np.uint64(0xFFFFFFFF)
_TWO_POW_M32 = np.float32(2.3283064e-10)

#: Float conversion of the random library behind ``curand_uniform4``: rocRAND maps a 32-bit
#: draw ``v`` to ``(v + 1) / 2**32``, cuRAND to ``v / 2**32 + 2**-33``.
FLAVORS = ("rocrand", "curand")


def philox4x32_10(counter: Sequence, key: Tuple[int, int]) -> Tuple[np.ndarray, ...]:
    """Philox4x32 with 10 rounds on arrays of counter words; returns four ``uint32`` arrays."""
    c0, c1, c2, c3 = (np.asarray(word, dtype=np.uint32) for word in counter)
    k0, k1 = int(key[0]) & 0xFFFFFFFF, int(key[1]) & 0xFFFFFFFF
    for round_index in range(10):
        if round_index:
            k0, k1 = (k0 + _W0) & 0xFFFFFFFF, (k1 + _W1) & 0xFFFFFFFF
        p0 = _M0 * c0.astype(np.uint64)
        p1 = _M1 * c2.astype(np.uint64)
        hi0, lo0 = (p0 >> np.uint64(32)).astype(np.uint32), (p0 & _LOW32).astype(np.uint32)
        hi1, lo1 = (p1 >> np.uint64(32)).astype(np.uint32), (p1 & _LOW32).astype(np.uint32)
        c0, c1, c2, c3 = hi1 ^ c1 ^ np.uint32(k0), lo1, hi0 ^ c3 ^ np.uint32(k1), lo0
    return c0, c1, c2, c3


def grid_threads(numel: int, compute_units: int, threads_per_cu: int = 2048) -> int:
    """Threads the kernel launches for a tensor of ``numel`` elements."""
    blocks = min((int(numel) + 255) // 256, int(compute_units) * (int(threads_per_cu) // 256))
    return blocks * 256


def offset_increment(numel: int, compute_units: int, threads_per_cu: int = 2048) -> int:
    """How far one ``uniform_`` call on ``numel`` elements advances the generator offset."""
    threads = grid_threads(numel, compute_units, threads_per_cu)
    return ((int(numel) - 1) // (threads * 4) + 1) * 4


def uniform_unit(
    index,
    numel: int,
    offset: int,
    compute_units: int,
    seed: int,
    flavor: str = "rocrand",
    threads_per_cu: int = 2048,
) -> np.ndarray:
    """Values in (0, 1] that ``uniform_`` draws for flat element ``index`` of an ``numel`` tensor.

    ``offset`` is the generator offset when the call starts; ``index`` may be any subset of
    ``range(numel)``, so sampled elements of a large tensor cost only their own draws.
    """
    if flavor not in FLAVORS:
        raise ValueError(f"flavor must be one of {FLAVORS}, got {flavor!r}")
    threads = grid_threads(numel, compute_units, threads_per_cu)
    index = np.asarray(index, dtype=np.int64)
    thread = index % threads
    draw = index // (4 * threads)
    component = (index // threads) % 4
    counter = int(offset) // 4 + draw
    words = philox4x32_10(
        (counter & 0xFFFFFFFF, counter >> 32, thread & 0xFFFFFFFF, thread >> 32),
        (int(seed) & 0xFFFFFFFF, int(seed) >> 32),
    )
    value = np.choose(component, words)
    if flavor == "curand":
        return value.astype(np.float32) * _TWO_POW_M32 + _TWO_POW_M32 / np.float32(2.0)
    return _TWO_POW_M32 + value.astype(np.float32) * _TWO_POW_M32


def uniform_like_torch(
    index,
    numel: int,
    offset: int,
    compute_units: int,
    low: float,
    high: float,
    dtype,
    seed: int,
    flavor: str = "rocrand",
    threads_per_cu: int = 2048,
):
    """``torch.empty(numel, dtype=dtype, device=gpu).uniform_(low, high)`` at ``index``.

    The bounds are cast to ``dtype`` first and the range is their difference in ``dtype``, as
    PyTorch's kernel does; a value that rounds to ``high`` is replaced by ``low``.
    """
    import torch

    unit = torch.from_numpy(uniform_unit(index, numel, offset, compute_units, seed, flavor, threads_per_cu))
    low_t, high_t = torch.tensor(low, dtype=dtype), torch.tensor(high, dtype=dtype)
    span = (high_t - low_t).float()
    value = (unit * span + low_t.float()).to(dtype)
    return torch.where(value == high_t, low_t, value)
