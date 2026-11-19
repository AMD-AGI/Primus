###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Sort-free inverted top-k index for the sparse-MLA backward.

``_build_inverted_topk_slice`` needs, for a ``[T, R]`` slice of top-k KV indices, the flat
positions ``q * R + r`` grouped by KV token in ascending ``q`` order (what a STABLE argsort
by key gives) plus CSR row pointers. ``torch.argsort(stable=True)`` on 0.5-1M int64 keys runs
rocprim's merge sort (~21 launches, 190-275 us per call on MI455X with the bincount/cumsum/
casts around it, 14 calls per step on the 1-GPU proxy).

The keys are small integers and every valid KV index appears at most once per query row
(top-k / sliding-window index sets), so the index is a transpose, not a sort:

1. ``_scatter_kernel``: ``dense[k, q] = r`` for every valid entry (``dense`` is a
   ``[n_kv, T]`` int16 scratch pre-filled with -1; no write conflicts by the uniqueness above).
2. ``_count_kernel``: per KV row, the number of valid entries -> ``inv_ptr`` (cumsum,
   with the ``-1`` count in front, exactly as before).
3. ``_compact_kernel``: per KV row, stream-compact ``q * R + dense[k, q]`` in ascending ``q``
   into ``inv_data[inv_ptr[k] ...]``.

The valid part of ``inv_data`` and ``inv_ptr`` are bit-identical to the argsort version, so
the dKV accumulation order (and result) is unchanged. The leading ``inv_ptr[0]`` slots that
the argsort filled with the invalid entries are never read by the consumer and are left
uninitialised here.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_kernel(KEYS_PTR, stride_t, DENSE_PTR, T, n_kv, R: tl.constexpr, BLOCK_R: tl.constexpr):
    q = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    k = tl.load(KEYS_PTR + q.to(tl.int64) * stride_t + r, mask=r < R, other=-1)
    ok = (k >= 0) & (k < n_kv)
    tl.store(DENSE_PTR + k.to(tl.int64) * T + q, r.to(tl.int16), mask=ok)


@triton.jit
def _count_kernel(DENSE_PTR, COUNT_PTR, T, BLOCK_T: tl.constexpr):
    k = tl.program_id(0)
    row = DENSE_PTR + k.to(tl.int64) * T
    cnt = tl.zeros((), dtype=tl.int32)
    for t0 in range(0, T, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        v = tl.load(row + t, mask=t < T, other=-1)
        cnt += tl.sum((v >= 0).to(tl.int32), axis=0)
    tl.store(COUNT_PTR + k, cnt)


@triton.jit
def _compact_kernel(DENSE_PTR, PTR_PTR, OUT_PTR, T, R: tl.constexpr, BLOCK_T: tl.constexpr):
    k = tl.program_id(0)
    row = DENSE_PTR + k.to(tl.int64) * T
    base = tl.load(PTR_PTR + k)  # inv_ptr[k] = start of KV token k's entries
    for t0 in range(0, T, BLOCK_T):
        t = t0 + tl.arange(0, BLOCK_T)
        v = tl.load(row + t, mask=t < T, other=-1).to(tl.int32)
        valid = v >= 0
        vi = valid.to(tl.int32)
        pos = tl.cumsum(vi, axis=0) - vi  # exclusive prefix -> ascending q order
        tl.store(OUT_PTR + base + pos, t * R + v, mask=valid)
        base += tl.sum(vi, axis=0)


@triton.jit
def _ptr_kernel(COUNT_PTR, PTR_PTR, n_kv, total, BLOCK: tl.constexpr):
    """inv_ptr = cumsum([total - sum(counts), counts...]) in one program (n_kv <= BLOCK)."""
    i = tl.arange(0, BLOCK)
    c = tl.load(COUNT_PTR + i, mask=i < n_kv, other=0)
    invalid = total - tl.sum(c, axis=0)
    tl.store(PTR_PTR, invalid)
    tl.store(PTR_PTR + 1 + i, invalid + tl.cumsum(c, axis=0), mask=i < n_kv)


def counting_sort_enabled() -> bool:
    return os.environ.get("PRIMUS_V4_INV_COUNTING_SORT", "1") != "0"


_MAX_DENSE_BYTES = int(os.environ.get("PRIMUS_V4_INV_MAX_DENSE_MB", "512")) << 20


def counting_sort_applicable(T: int, R: int, n_kv: int) -> bool:
    return R <= 32767 and T * n_kv * 2 <= _MAX_DENSE_BYTES


def build_inverted_topk_counting(topk_slice: torch.Tensor, n_kv: int):
    """Same contract as ``_build_inverted_topk_slice``: ``inv_ptr`` int32 ``[n_kv+1]``
    (``inv_ptr[0]`` = number of invalid entries) and ``inv_data`` int32 ``[T*R]``."""
    T, R = topk_slice.shape
    keys = topk_slice
    if keys.stride(1) != 1:
        keys = keys.contiguous()
    dev = keys.device
    n_kv = int(n_kv)
    dense = torch.full((n_kv, T), -1, dtype=torch.int16, device=dev)
    _scatter_kernel[(T,)](keys, keys.stride(0), dense, T, n_kv, R=R, BLOCK_R=triton.next_power_of_2(R))
    block_t = min(triton.next_power_of_2(T), 4096)
    counts = torch.empty((n_kv,), dtype=torch.int32, device=dev)
    _count_kernel[(n_kv,)](dense, counts, T, BLOCK_T=block_t)
    # invalid entries sort first (bin 0), then one bin per KV token
    if n_kv <= 65536:
        inv_ptr = torch.empty((n_kv + 1,), dtype=torch.int32, device=dev)
        _ptr_kernel[(1,)](counts, inv_ptr, n_kv, T * R, BLOCK=triton.next_power_of_2(n_kv))
    else:
        inv_ptr = torch.cumsum(torch.cat([(T * R - counts.sum()).view(1).to(torch.int32), counts]), 0,
                               dtype=torch.int32)
    inv_data = torch.empty((T * R,), dtype=torch.int32, device=dev)
    _compact_kernel[(n_kv,)](dense, inv_ptr, inv_data, T, R=R, BLOCK_T=block_t)
    return inv_ptr, inv_data


__all__ = ["build_inverted_topk_counting", "counting_sort_enabled", "counting_sort_applicable"]
