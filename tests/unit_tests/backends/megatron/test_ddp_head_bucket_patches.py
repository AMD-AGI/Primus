###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ddp_head_bucket: the ramp is built by Megatron's own bucketing loop (a real ``_ParamAndGradBuffer``), every parameter
lands in exactly one bucket, the first-forward buckets follow the ramp, and the rest keep Megatron's rule."""

import os

import pytest
import torch

from primus.backends.megatron.patches.ddp_head_bucket_patches import (
    _HeadRampBucketSize,
    close_after,
    head_cuts,
)


def test_head_cuts_ramp():
    assert head_cuts(10_000, 64, 2.0, 1000) == [64, 192, 448, 960]
    assert head_cuts(100, 64, 2.0, 1000) == [64]  # stops inside the total
    assert head_cuts(10_000, 2000, 2.0, 1000) == []  # a head at or above bucket_size: no ramp


def test_close_after_ramp_then_full_buckets_remainder_at_tail():
    numels = [10] * 100  # forward order, 1000 elements
    decide, cuts = close_after(numels, head=20, ramp=2.0, bucket_size=200)
    assert cuts == [20, 60, 140, 300, 500, 700, 900]  # ramp (160 < bucket_size still ramps), then every 200
    # Walk order is the reverse; the first 70 walk steps are outside the ramp (more than 300 elements remain after).
    fill, closes = 0, []
    for i in range(100):
        fill += 10
        if decide(i, fill):
            closes.append(i)
            fill = 0
    sizes, prev = [], -1
    for c in closes:
        sizes.append((c - prev) * 10)
        prev = c
    if prev != 99:
        sizes.append((99 - prev) * 10)
    assert sum(sizes) == 1000
    # Walk order: the 100 remainder at the tail first, then full buckets, then the ramp 160, 80, 40, 20 ending at the
    # first-forward parameter.
    assert sizes == [100, 200, 200, 200, 160, 80, 40, 20]


def test_stand_in_counts_calls():
    s = _HeadRampBucketSize(lambda i, f: f >= 5, 2)
    assert (7 >= s) is True and (1 >= s) is False
    with pytest.raises(AssertionError):
        _ = 9 >= s


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Megatron's buffer allocates on cuda)")
def test_real_megatron_buffer_ramp():
    import torch.distributed as dist
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.distributed import param_and_grad_buffer as pgb

    from primus.backends.megatron.patches import ddp_head_bucket_patches as P

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29633")
        dist.init_process_group("nccl", rank=0, world_size=1)
    g = torch.Generator().manual_seed(0)
    sizes = [int(x) for x in torch.randint(1_000, 50_000, (300,), generator=g)]
    params = [torch.nn.Parameter(torch.empty(n, device="cuda", dtype=torch.bfloat16)) for n in sizes]
    cfg = DistributedDataParallelConfig(
        use_distributed_optimizer=True, overlap_grad_reduce=True, overlap_param_gather=True
    )
    bucket = 1_000_000

    from types import SimpleNamespace

    pgs = SimpleNamespace(
        tp=dist.group.WORLD, dp_cp=dist.group.WORLD
    )  # what DDP passes; avoids parallel_state

    def build(bucket_size):
        return pgb._ParamAndGradBuffer(
            cfg,
            torch.bfloat16,
            torch.float32,
            params,
            dist.group.WORLD,
            bucket_size,
            {p: f"p{i}" for i, p in enumerate(params)},
            1.0,
            list(range(len(params))),
            False,
            pgs,
        )

    ref = build(bucket)
    decide, cuts = close_after(sizes, 60_000, 1.5, bucket)
    stand_in = P._HeadRampBucketSize(decide, len(params))
    got = build(stand_in)
    assert stand_in.calls == len(params)
    # Every parameter in exactly one bucket, buckets contiguous in the buffer.
    owners = {}
    for p, (s, e, b) in got.param_index_map.items():
        owners.setdefault(b, []).append(p)
    assert sum(len(v) for v in owners.values()) == len(params)
    # The first-forward parameter sits in the last bucket, which is small; Megatron's own head bucket is a remainder.
    head_b = got.param_index_map[params[0]][2]
    head_numel = sum(p.numel() for p in owners[head_b])
    assert head_b == len(got.bucket_indices) - 1
    assert head_numel <= cuts[0] + max(sizes)
    assert len(got.bucket_indices) > len(ref.bucket_indices)
