###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Reference packed-weight collective; not yet a Megatron bucket replacement."""

from contextlib import nullcontext

import torch
import torch.distributed as dist


class MXFP4ParamGather:
    """Reusable equal-row-shard gather workspace for Turbo MXFP4WireLayout.

    Both compute orientations and scales travel as uint8. RCCL performs no
    quantization or reduction. Construction is collective. The caller must
    provide complete 32-row tiles, identical layouts and the same call order
    on every rank. Megatron's arbitrary flat parameter shards do not satisfy
    this contract without a separate boundary exchange/repartitioning step.
    """

    def __init__(self, layout, group, device, *, use_sdma=False):
        self.layout = layout
        self.device = torch.device(device)
        self.group = group
        self.pool = self.symmetric_memory = None
        if use_sdma:
            if self.device.type != "cuda":
                raise ValueError("RCCL SDMA requires a GPU device")
            import torch.distributed._symmetric_memory as symm_mem

            from .rccl_sdma_param_gather import prepare_direct_param_buffer_pool

            self.group, _shared_pool = prepare_direct_param_buffer_pool(group, self.device)
            # Own the allocation lifetime. A process-global pool can keep RCCL
            # window registrations alive until after communicator destruction.
            self.pool = torch.cuda.MemPool(
                symm_mem.get_mempool_allocator(self.device), use_on_oom=False, no_split=True
            )

        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        metadata = torch.tensor(
            (*layout.shape, layout.scale_rounding_mode, layout.nbytes), dtype=torch.int64, device=self.device
        )
        metadata_by_rank = [torch.empty_like(metadata) for _ in range(self.world_size)]
        dist.all_gather(metadata_by_rank, metadata, group=group)
        if any(not torch.equal(other, metadata) for other in metadata_by_rank):
            raise ValueError("packed all-gather requires identical layouts on all ranks")

        allocation = torch.cuda.use_mem_pool(self.pool) if self.pool is not None else nullcontext()
        with allocation:
            self.storage = torch.empty(self.world_size * layout.nbytes, dtype=torch.uint8, device=self.device)
        if use_sdma:
            from .rccl_sdma_param_gather import rendezvous_direct_param_buffer

            self.symmetric_memory = rendezvous_direct_param_buffer(self.storage, self.group)
        self.gathered = self.storage.view(self.world_size, layout.nbytes)
        self.local = self.gathered[self.rank]

    def gather(self, payload):
        """Return rank-major bytes, valid until the next call on this workspace.

        Launch asynchronously even for this synchronous reference API: RCCL's
        null-stream fallback is incompatible with imported ROCr VMM pointers.
        Work.wait establishes dependency on the caller's current GPU stream.
        """
        if self.storage is None:
            raise RuntimeError("packed gather workspace is closed")
        self.layout.views(payload)
        if payload.device != self.storage.device:
            raise ValueError("payload and collective workspace must be on the same device")
        self.local.copy_(payload)
        work = dist.all_gather_into_tensor(self.storage, self.local, group=self.group, async_op=True)
        work.wait()
        return self.gathered

    def close(self):
        """Release storage before process-group teardown, after consumers finish.

        The caller must drop all tensors returned by gather() first. Reconstructed
        Turbo pairs have independent storage and may outlive this workspace.
        """
        if self.storage is None:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.local = self.gathered = self.storage = None
        self.symmetric_memory = None
        self.pool = None
