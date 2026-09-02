###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Bitwise checks on the two context-parallel collectives, forward AND backward.

The CP equivalence script compares whole-attention outputs with a tolerance, which is the
right tool for bf16 numerics but the wrong one here: it cannot distinguish "delivered the
wrong rows" from "delivered the right rows, rounded differently". These tests use
integer-valued inputs so every expected value is exact, and assert with ``torch.equal``.

Both backward passes are the interesting half. They move data *against* the forward
direction, they are hand-written rather than derived by autograd, and an error in either
leaves the forward perfectly correct while training a different model:

* :class:`LeftBoundaryExchange` -- the neighbour's tokens contribute to this rank's
  compressed keys, so the gradient for them has to travel back to the rank that owns those
  parameters. Dropping it silently trains the neighbour on less signal than it produced.
* :class:`_AllGatherPool` -- rank r's pool rows are read by the queries of *every* rank at
  or after r, so each of those holds a partial gradient. Slicing this rank's block out of
  its own ``grad_out`` keeps only its own contribution and drops the rest.

Structure follows Megatron-LM PR #5087's
``test_thd_cp_left_boundary_exchange_forward_backward``, adapted from THD to the BSHD
entry point this repo uses for the contiguous CP path.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))

from tests.unit_tests.megatron.transformer.deepseek_v4._dist_harness import (  # noqa: E402
    arange_block,
    run_dist,
)

WORLD_SIZES = [2, 4]


# --------------------------------------------------------------------------------------
# 1. Left-boundary exchange, BSHD
# --------------------------------------------------------------------------------------


def _boundary_body(rank, world, group, d_window, s_local, heads, head_dim):
    import torch.distributed as dist  # noqa: F401

    from primus.backends.megatron.core.transformer.deepseek_v4_cp import (
        exchange_boundary_kv,
    )

    width = heads * head_dim
    flat = arange_block(rank, s_local, width)
    kv = flat.reshape(1, s_local, heads, head_dim).clone().requires_grad_(True)

    boundary = exchange_boundary_kv(kv, d_window, group)
    assert boundary.shape == (1, d_window, heads, head_dim), boundary.shape

    # Forward: rank 0 has no left neighbour and must get zeros -- the adapter's global
    # position mask excludes those rows, so zeros are the contract, not a placeholder.
    if rank == 0:
        expected = torch.zeros_like(boundary)
    else:
        left = arange_block(rank - 1, s_local, width)
        expected = left[-d_window:].reshape(1, d_window, heads, head_dim)
    assert torch.equal(boundary, expected), (
        f"rank {rank} forward boundary mismatch\n got {boundary.flatten()[:8]}\n"
        f" want {expected.flatten()[:8]}"
    )

    # Backward: an all-ones upstream gradient makes the expected gradient exactly 0 or 1.
    # This rank's tail rows earned gradient iff the NEXT rank consumed them as its left
    # boundary; the last rank sent nothing, so its gradient must be entirely zero.
    boundary.sum().backward()
    expected_grad = torch.zeros_like(kv)
    if rank + 1 < world:
        expected_grad[:, -d_window:] = 1.0
    assert torch.equal(kv.grad, expected_grad), (
        f"rank {rank} backward gradient mismatch: the neighbour's share of the update is "
        f"what this catches\n got nonzero rows "
        f"{torch.nonzero(kv.grad.reshape(s_local, -1).sum(-1)).flatten().tolist()}\n"
        f" want {torch.nonzero(expected_grad.reshape(s_local, -1).sum(-1)).flatten().tolist()}"
    )


@pytest.mark.parametrize("world", WORLD_SIZES)
@pytest.mark.parametrize("d_window", [1, 2, 4])
def test_bshd_left_boundary_exchange_forward_backward(world, d_window):
    run_dist(_boundary_body, world, d_window, 4, 2, 3)


def _boundary_too_short_body(rank, world, group):
    from primus.backends.megatron.core.transformer.deepseek_v4_cp import (
        exchange_boundary_kv,
    )

    # d_window larger than the local shard cannot be served by a single-hop ring exchange.
    # It must raise, not silently return a short or zero-padded window.
    kv = torch.zeros(1, 2, 1, 3, requires_grad=True)
    with pytest.raises(RuntimeError, match="local rows >= d_window"):
        exchange_boundary_kv(kv, 3, group)


@pytest.mark.parametrize("world", [2])
def test_boundary_exchange_rejects_window_longer_than_shard(world):
    run_dist(_boundary_too_short_body, world)


# --------------------------------------------------------------------------------------
# 2. All-gather pool backward
# --------------------------------------------------------------------------------------


def _pool_ones_body(rank, world, group, p_local, dim):
    from primus.backends.megatron.core.transformer.deepseek_v4_cp import (
        build_global_pool,
    )

    pool = arange_block(rank, p_local, dim).reshape(1, p_local, dim).clone().requires_grad_(True)
    out = build_global_pool(pool, group)

    # Forward is rank-major concatenation, which IS sequence order for this path.
    assert out.shape == (1, p_local * world, dim), out.shape
    expected_fwd = torch.cat(
        [arange_block(r, p_local, dim).reshape(1, p_local, dim) for r in range(world)], dim=1
    )
    assert torch.equal(out, expected_fwd), f"rank {rank}: gathered pool is not in rank order"

    # Every rank computes the same scalar from the same global pool, so the total loss is
    # cp_size copies of it and d(total)/d(pool_local) is exactly cp_size -- an integer, so
    # this is bitwise-checkable. A backward that sliced its own grad_out instead of
    # reducing first would give 1 here, not cp_size.
    out.sum().backward()
    expected_grad = torch.full_like(pool, float(world))
    assert torch.equal(pool.grad, expected_grad), (
        f"rank {rank}: expected every entry to be {world} (one unit from each rank's "
        f"queries), got {pool.grad.unique().tolist()}"
    )


@pytest.mark.parametrize("world", WORLD_SIZES)
def test_all_gather_pool_backward_is_exactly_cp_size(world):
    run_dist(_pool_ones_body, world, 3, 2)


def _pool_reduce_scatter_body(rank, world, group, p_local, dim):
    import torch.distributed as dist

    from primus.backends.megatron.core.transformer.deepseek_v4_cp import (
        build_global_pool,
    )

    pool = arange_block(rank, p_local, dim).reshape(1, p_local, dim).clone().requires_grad_(True)
    out = build_global_pool(pool, group)

    # A non-uniform upstream gradient, still integer-valued so the two reduction orders
    # cannot differ by rounding. Each rank contributes a DIFFERENT gradient, which is the
    # case that distinguishes reduce-scatter from "slice my own".
    g = (torch.arange(out.numel(), dtype=out.dtype).reshape(out.shape) + 1) * (rank + 1)
    out.backward(g)

    # Independent reference: reduce-scatter the same upstream gradient. all_reduce followed
    # by slicing this rank's block is the same operation expressed differently, and with
    # integer values it must agree bit for bit.
    #
    # `g` is reused here, which only works because backward must not have modified it --
    # see test_backward_does_not_mutate_grad_output. When it did, this reference silently
    # reduced already-reduced values and came out cp_size times too large.
    ref = torch.empty(1, p_local, dim, dtype=g.dtype)
    dist.reduce_scatter_tensor(ref, g.reshape(world, p_local, dim).contiguous(), group=group)

    assert torch.equal(pool.grad, ref), (
        f"rank {rank}: all_reduce+slice disagrees with reduce_scatter\n"
        f" got  {pool.grad.flatten()[:6].tolist()}\n ref  {ref.flatten()[:6].tolist()}"
    )


@pytest.mark.parametrize("world", WORLD_SIZES)
def test_all_gather_pool_backward_equals_reduce_scatter(world):
    run_dist(_pool_reduce_scatter_body, world, 3, 2)


def _pool_no_mutate_body(rank, world, group, p_local, dim):
    from primus.backends.megatron.core.transformer.deepseek_v4_cp import (
        build_global_pool,
    )

    pool = arange_block(rank, p_local, dim).reshape(1, p_local, dim).clone().requires_grad_(True)
    out = build_global_pool(pool, group)

    g = (torch.arange(out.numel(), dtype=out.dtype).reshape(out.shape) + 1) * (rank + 1)
    before = g.clone()
    out.backward(g)

    assert torch.equal(g, before), (
        f"rank {rank}: backward modified its grad_output in place. all_reduce is an "
        f"in-place op and .contiguous() is a no-op on an already-contiguous tensor, so "
        f"reducing into it rewrites a buffer autograd may share with another consumer of "
        f"this output -- corrupting a gradient somewhere else entirely.\n"
        f" before {before.flatten()[:4].tolist()}\n after  {g.flatten()[:4].tolist()}"
    )


@pytest.mark.parametrize("world", WORLD_SIZES)
def test_all_gather_pool_backward_does_not_mutate_grad_output(world):
    run_dist(_pool_no_mutate_body, world, 3, 2)
