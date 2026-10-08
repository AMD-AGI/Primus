###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in MXFP4 expert parameter synchronization for Megatron TP=PP=EP=1.

Only optimizer-owned BF16 elements are authoritative after an update. First
gather ordinary parameters and the small set of 32-row strips crossing shard
boundaries. The owner of each strip's first element can then quantize complete
tiles. A second byte AllGather carries both quantized orientations and scales.
Remote BF16 expert interiors remain stale until explicit checkpoint/fallback
materialization; compute MUST consume the refreshed quantized cache instead.
"""

from dataclasses import dataclass
from math import prod

import torch
import torch.distributed as dist


def strip_ownership(shape, start, bucket_numel, world):
    """Return (first_strip, count) per rank and cross-rank BF16 strip ranges."""
    if len(shape) != 3 or min(shape) <= 0 or any(d % 32 for d in shape[-2:]):
        raise ValueError("MXFP4 communication needs 3D weights with dimensions divisible by 32")
    if world <= 0 or bucket_numel % world or start < 0 or start + prod(shape) > bucket_numel:
        raise ValueError("invalid bucket/shard layout")
    size, strip, total = bucket_numel // world, 32 * shape[-1], shape[0] * shape[1] // 32

    def first_at(offset):
        return max(0, min(total, (offset - start + strip - 1) // strip))

    owners = [(first_at(r * size), first_at((r + 1) * size) - first_at(r * size)) for r in range(world)]
    boundaries = set()
    for rank in range(1, world):
        cut = rank * size
        if start < cut < start + prod(shape) and (cut - start) % strip:
            first = start + (cut - start) // strip * strip
            boundaries.add((first, first + strip))
    return owners, sorted(boundaries)


def merge_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


class ByteAllGather:
    """Persistent symmetric byte workspace with explicit stream dependencies."""

    def __init__(self, width, group, device, use_sdma):
        self.group, self.device = group, device
        self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        self.width = max(256, (width + 255) // 256 * 256)
        self.pool = self.symmetric_memory = None
        if use_sdma:
            import torch.distributed._symmetric_memory as symm_mem

            from .rccl_sdma_param_gather import (
                prepare_direct_param_buffer_pool,
                rendezvous_direct_param_buffer,
            )

            self.group, _ = prepare_direct_param_buffer_pool(group, device)
            self.pool = torch.cuda.MemPool(symm_mem.get_mempool_allocator(device), no_split=True)
            with torch.cuda.use_mem_pool(self.pool):
                self.storage = torch.zeros(self.world * self.width, dtype=torch.uint8, device=device)
            self.symmetric_memory = rendezvous_direct_param_buffer(self.storage, self.group)
        else:
            self.storage = torch.zeros(self.world * self.width, dtype=torch.uint8, device=device)
        self.rows = self.storage.view(self.world, self.width)
        self.local = self.rows[self.rank]

    def launch(self):
        return dist.all_gather_into_tensor(self.storage, self.local, group=self.group, async_op=True)

    def close(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.local = self.rows = self.storage = None
        self.symmetric_memory = self.pool = None


@dataclass
class ExpertWeight:
    owner: object
    param: torch.Tensor
    start: int
    layout: object
    ownership: list
    offsets: list
    pair: object = None
    consumed_generation: int = -1

    def get_pair(self, scale_rounding_mode):
        if scale_rounding_mode != self.layout.scale_rounding_mode:
            raise RuntimeError("MXFP4 communication and compute scale rounding modes differ")
        self.owner.wait()
        if self.owner.generation == 0:
            raise RuntimeError("MXFP4 expert weight consumed before parameter synchronization")
        if self.pair is None:
            from primus_turbo.pytorch.core.mxfp4_comm import MXFP4WireLayout

            pieces = []
            for rank, (first, count) in enumerate(self.ownership):
                if count:
                    size = MXFP4WireLayout((count, 32, self.layout.shape[-1])).nbytes
                    pieces.append(
                        (
                            first,
                            count,
                            self.owner.packed.rows[rank, self.offsets[rank] : self.offsets[rank] + size],
                        )
                    )
            self.pair = self.layout.wrap_components(self.layout.assemble_strip_shards(pieces))
        if self.consumed_generation < 0 and self.owner.rank == 0:
            print(
                f"[MXFP4-COMM] consumed expert cache shape={tuple(self.param.shape)} generation={self.owner.generation}",
                flush=True,
            )
        self.consumed_generation = self.owner.generation
        return self.pair

    def materialize(self):
        self.owner.materialize_bf16()


class PackedExpertBucket:
    """State owned by a Megatron bucket, refreshed at every start_param_sync."""

    def __init__(self, bucket, group, *, use_sdma=True, scale_rounding_mode=2):
        from primus_turbo.pytorch.core.mxfp4_comm import MXFP4WireLayout

        self.bucket, self.group = bucket, group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.generation = 0
        self.work = None
        self.weights = []
        self.device = bucket.param_data.device
        if bucket.param_data.dtype != torch.bfloat16:
            raise ValueError("MXFP4 communication requires BF16 model parameter storage")
        numel = bucket.param_data.numel()
        if numel % self.world:
            raise ValueError("parameter bucket is not evenly sharded")
        self.shard_size = numel // self.world
        fallback, packed_sizes = [], [0] * self.world
        for param, (start, end) in sorted(bucket.param_to_index.items(), key=lambda item: item[1][0]):
            if not getattr(param, "_primus_mxfp4_comm_candidate", False):
                fallback.append((start, end))
                continue
            shape = tuple(param.shape)
            layout = MXFP4WireLayout(shape, scale_rounding_mode)
            owners, boundaries = strip_ownership(shape, start, numel, self.world)
            state = ExpertWeight(self, param, start, layout, owners, packed_sizes.copy())
            self.weights.append(state)
            param._primus_mxfp4_comm_state = state
            for rank, (_first, count) in enumerate(owners):
                if count:
                    packed_sizes[rank] += MXFP4WireLayout((count, 32, shape[-1])).nbytes
            fallback.extend(boundaries)
        if not self.weights:
            raise ValueError("packed bucket requires at least one marked expert weight")
        self.fallback_ranges = []
        merged = merge_ranges(fallback)
        for rank in range(self.world):
            self.fallback_ranges.append(
                [
                    (max(start, rank * self.shard_size), min(end, (rank + 1) * self.shard_size))
                    for start, end in merged
                    if max(start, rank * self.shard_size) < min(end, (rank + 1) * self.shard_size)
                ]
            )
        fallback_sizes = [sum(end - start for start, end in ranges) * 2 for ranges in self.fallback_ranges]
        self.fallback = (
            ByteAllGather(max(fallback_sizes), group, self.device, use_sdma) if any(fallback_sizes) else None
        )
        self.packed = ByteAllGather(max(packed_sizes), group, self.device, use_sdma)
        if self.rank == 0:
            wire_bytes = self.world * (self.packed.width + (self.fallback.width if self.fallback else 0))
            print(
                f"[MXFP4-COMM] enabled experts={len(self.weights)} bf16_bytes={numel * 2} wire_bytes={wire_bytes} scale_rounding={scale_rounding_mode}",
                flush=True,
            )

    @torch.no_grad()
    def dispatch(self):
        from primus_turbo.pytorch.core.mxfp4_comm import MXFP4WireLayout

        if self.work is not None:
            raise RuntimeError("previous MXFP4 parameter gather is still outstanding")
        for weight in self.weights:
            weight.pair = None
        if self.fallback is not None:
            offset = 0
            for start, end in self.fallback_ranges[self.rank]:
                size = (end - start) * 2
                self.fallback.local[offset : offset + size].copy_(
                    self.bucket.param_data[start:end].view(torch.uint8)
                )
                offset += size
            self.fallback.launch().wait()
            for rank, ranges in enumerate(self.fallback_ranges):
                offset = 0
                for start, end in ranges:
                    size = (end - start) * 2
                    self.bucket.param_data[start:end].copy_(
                        self.fallback.rows[rank, offset : offset + size].view(torch.bfloat16)
                    )
                    offset += size
        for weight in self.weights:
            first, count = weight.ownership[self.rank]
            if not count:
                continue
            k = weight.layout.shape[-1]
            source = weight.param.detach().view(-1, 32, k)[first : first + count]
            layout = MXFP4WireLayout((count, 32, k), weight.layout.scale_rounding_mode)
            payload = layout.quantize(source)
            offset = weight.offsets[self.rank]
            self.packed.local[offset : offset + layout.nbytes].copy_(payload)
        self.work = self.packed.launch()
        self.generation += 1
        return self

    def wait(self):
        if self.work is not None:
            self.work.wait()
            self.work = None

    @torch.no_grad()
    def materialize_bf16(self):
        """Collect exact optimizer-owned BF16 shards for checkpoint/non-FP4 use.

        Always gather: an optimizer may have written new local BF16 values
        since the last materialization, before the next packed dispatch.
        """
        self.wait()
        local = self.bucket.param_data[self.rank * self.shard_size : (self.rank + 1) * self.shard_size]
        dist.all_gather_into_tensor(
            self.bucket.param_data, local, group=self.packed.group, async_op=True
        ).wait()

    def close(self):
        self.wait()
        for weight in self.weights:
            weight.pair = None
        self.packed.close()
        if self.fallback is not None:
            self.fallback.close()


class GatherGroupWork:
    def __init__(self, handles):
        self.handles = handles

    def wait(self):
        for handle in self.handles:
            handle.wait()
