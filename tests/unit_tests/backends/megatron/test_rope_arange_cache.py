# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""Parity + caching test for ``RoPECache.forward_arange``.

The compressed-branch RoPE is evaluated at deterministic positions every
forward (``arange(P) * compress_ratio``); ``forward_arange`` caches that table.
This asserts the cached table is bit-identical to recomputing
``forward(arange(n) * stride)``, that ``stride`` is honoured and part of the
cache key, and that the cache actually memoises (with
``PRIMUS_COMPRESS_ROPE_CACHE=0`` bypassing it).
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


@pytest.mark.parametrize("stride", [4, 128])
def test_forward_arange_stride_uses_original_positions(stride):
    """``stride`` evaluates at ``arange(n) * stride``, not at block indices.

    Compressed KV entry ``s`` covers the window starting at original token
    ``s * compress_ratio`` and must be rotated at that position so it shares
    the query coordinate system.
    """
    n = 16
    rc = _rope()
    cos_a, sin_a = rc.forward_arange(n, "cpu", stride=stride)
    cos_e, sin_e = rc.forward(torch.arange(n, device="cpu") * stride)
    torch.testing.assert_close(cos_a, cos_e, rtol=0, atol=0)
    torch.testing.assert_close(sin_a, sin_e, rtol=0, atol=0)

    # Guard against a silently-ignored stride.
    cos_blocks, _ = rc.forward_arange(n, "cpu", stride=1)
    assert not torch.allclose(cos_a, cos_blocks)


def test_forward_arange_stride_is_part_of_cache_key():
    """Different strides must not collide in the memo table."""
    rc = _rope()
    cos_s1, _ = rc.forward_arange(32, "cpu", stride=1)
    cos_s4, _ = rc.forward_arange(32, "cpu", stride=4)
    torch.testing.assert_close(cos_s4, rc.forward(torch.arange(32) * 4)[0], rtol=0, atol=0)
    torch.testing.assert_close(cos_s1, rc.forward(torch.arange(32))[0], rtol=0, atol=0)
    # And each stride still memoises independently.
    assert rc.forward_arange(32, "cpu", stride=4)[0] is cos_s4
