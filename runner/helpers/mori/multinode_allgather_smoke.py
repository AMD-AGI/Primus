#!/usr/bin/env python3
"""Two-node correctness smoke for Primus's MORI FSDP all-gather adapter."""

import argparse
import os

import torch
import torch.distributed as dist

from primus.backends.common.mori_allgather import MoriAllGather


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Keep RCCL lazy: this smoke validates MORI's RDMA path, and no torch
    # collective is needed before MORI initializes from the default c10d store.
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ranks_per_node = int(os.environ["LOCAL_WORLD_SIZE"])

    input_tensor = torch.full(
        (args.numel,),
        rank + 1,
        dtype=torch.bfloat16,
        device=device,
    )
    output_tensor = torch.empty(
        args.numel * world_size,
        dtype=input_tensor.dtype,
        device=device,
    )

    work = MoriAllGather(ranks_per_node=ranks_per_node)(
        output_tensor,
        input_tensor,
        dist.group.WORLD,
        async_op=True,
    )
    if work is not None:
        work.wait()
    torch.cuda.synchronize(device)

    expected = torch.repeat_interleave(
        torch.arange(1, world_size + 1, dtype=input_tensor.dtype, device=device),
        args.numel,
    )
    if not torch.equal(output_tensor, expected):
        raise RuntimeError(f"MORI all-gather mismatch on rank {rank}")

    if rank == 0:
        size_mb = args.numel * input_tensor.element_size() / (1 << 20)
        print(
            f"MORI_MULTINODE_ALLGATHER_PASS world={world_size} "
            f"ranks_per_node={ranks_per_node} per_rank_mb={size_mb:.1f}",
            flush=True,
        )

    import mori.shmem as shmem

    shmem.shmem_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
