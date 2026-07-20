# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""Parity + caching test for ``RoPECache.forward_arange``.

The compressed-branch RoPE is evaluated at the deterministic positions
``arange(P)`` every forward; ``forward_arange`` caches that table. This asserts
the cached table is bit-identical to recomputing ``forward(arange(n))`` and that
the cache actually memoises (and that ``PRIMUS_COMPRESS_ROPE_CACHE=0`` bypasses it).
"""
from __future__ import annotations

import pytest
import torch

from primus.backends.megatron.core.transformer.dual_rope import RoPECache


def _rope() -> RoPECache:
    return RoPECache(rotary_dim=64, theta=10000.0)


@pytest.mark.parametrize("n", [32, 1024])
def test_forward_arange_matches_forward(n):
    """Cached table is bit-identical to the eager arange->outer->cos/sin path."""
    rc = _rope()
    cos_a, sin_a = rc.forward_arange(n, "cpu")
    cos_e, sin_e = rc.forward(torch.arange(n, device="cpu"))
    torch.testing.assert_close(cos_a, cos_e, rtol=0, atol=0)
    torch.testing.assert_close(sin_a, sin_e, rtol=0, atol=0)


def test_forward_arange_memoises(monkeypatch):
    """A repeat call returns the SAME cached tensors (no recompute)."""
    monkeypatch.setenv("PRIMUS_COMPRESS_ROPE_CACHE", "1")
    rc = _rope()
    a = rc.forward_arange(128, "cpu")
    b = rc.forward_arange(128, "cpu")
    assert a[0] is b[0] and a[1] is b[1]


def test_cache_disabled_recomputes(monkeypatch):
    """PRIMUS_COMPRESS_ROPE_CACHE=0 bypasses the cache (fresh, equal tensors)."""
    monkeypatch.setenv("PRIMUS_COMPRESS_ROPE_CACHE", "0")
    rc = _rope()
    a = rc.forward_arange(128, "cpu")
    b = rc.forward_arange(128, "cpu")
    assert a[0] is not b[0]
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
