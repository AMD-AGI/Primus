###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU-only planning for complete 32-row MXFP4 strips in flat optimizer shards."""

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class WeightShardRegion:
    rank: int
    start: int
    end: int
    complete_tiles: bool


def plan_weight_regions(shape, param_start, bucket_numel, world_size):
    """Partition one weight into safe strips and fragments requiring exchange.

    Offsets are in elements relative to the bucket. A safe strip contains 32
    complete rows, so neither direction's 32x32 scale tile crosses a rank.
    This conservative planner may label a partial row as a boundary even when
    some tiles within that row are locally complete. It never pads fragments
    with zero: quantizing those would change the training recipe.
    """
    if len(shape) != 3 or any(d <= 0 for d in shape) or any(d % 32 for d in shape[-2:]):
        raise ValueError("expected [experts, rows, columns] with rows/columns divisible by 32")
    if world_size <= 0 or bucket_numel <= 0 or bucket_numel % world_size:
        raise ValueError("bucket must divide evenly across a positive world size")
    param_end = param_start + prod(shape)
    if param_start < 0 or param_end > bucket_numel:
        raise ValueError("weight must be contained in its bucket")
    shard_size = bucket_numel // world_size
    strip_size = 32 * shape[-1]
    regions = []
    for rank in range(world_size):
        start = max(param_start, rank * shard_size)
        end = min(param_end, (rank + 1) * shard_size)
        if start >= end:
            continue
        aligned_start = param_start + (start - param_start + strip_size - 1) // strip_size * strip_size
        aligned_end = param_start + (end - param_start) // strip_size * strip_size
        if aligned_start >= aligned_end:
            regions.append(WeightShardRegion(rank, start, end, False))
            continue
        if start < aligned_start:
            regions.append(WeightShardRegion(rank, start, aligned_start, False))
        regions.append(WeightShardRegion(rank, aligned_start, aligned_end, True))
        if aligned_end < end:
            regions.append(WeightShardRegion(rank, aligned_end, end, False))
    return tuple(regions)


def audit_buffer(buffer, param_to_name):
    """Inspect expert weights without copying tensors or changing buffer layout."""
    reports = []
    for param, (start, end, bucket_id) in buffer.param_index_map.items():
        name = param_to_name.get(param, "")
        if "experts" not in name or param.ndim != 3:
            continue
        shape = tuple(param.shape)
        if any(d % 32 for d in shape[-2:]):
            reports.append(dict(name=name, shape=shape, unsupported="not divisible by 32"))
            continue
        bucket = buffer.buckets[bucket_id]
        regions = plan_weight_regions(
            shape, start - bucket.offset, bucket.param_data.numel(), buffer.data_parallel_world_size
        )
        complete = sum(region.end - region.start for region in regions if region.complete_tiles)
        reports.append(
            dict(
                name=name,
                shape=shape,
                complete_tile_elements=complete,
                boundary_elements=end - start - complete,
            )
        )
    return reports
