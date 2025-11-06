###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os
import time
from datetime import timedelta
from typing import List, Tuple

import torch
import torch.distributed as dist


def add_arguments(parser: argparse.ArgumentParser):
    """
    Register Strided Allgather benchmark arguments to the CLI parser.
    """
    parser.add_argument(
        "--sizes-mb",
        type=str,
        default="64,128,256",
        help="Comma-separated message sizes in MB (per-rank send size)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Rank stride for the group, default is 8. For example, "
        "if the world size is 32, and the stride is 8, then the groups will be "
        "[0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26], [3, 11, 19, 27], "
        "[4, 12, 20, 28], [5, 13, 21, 29], [6, 14, 22, 30], [7, 15, 23, 31]."
        "[4, 12, 20, 28], [5, 13, 21, 29], [6, 14, 22, 30], [7, 15, 23, 31].",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run the allgather of multiple groups in parallel, default is False. "
        "If True, the allgather of multiple groups will be run in parallel for each rank.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Timed iterations per size",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations per size (not timed)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Tensor dtype",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
        help="Distributed backend",
    )

    return parser


def get_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def bytes_to_gib_per_s(num_bytes: float, seconds: float) -> float:
    if seconds <= 0:
        return float("inf")
    return (num_bytes / seconds) / (1024**3)


def format_size_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.0f}MB"


def barrier(group=None) -> None:
    try:
        dist.barrier(group=group)
    except Exception:
        # Some backends may not support subgroup barrier; ignore if unsupported
        pass


@torch.no_grad()
def benchmark_strided_allgather(
    group: dist.ProcessGroup,
    group_id: int,
    group_ranks: List[int],
    num_groups: int,
    device: torch.device,
    sizes_bytes: List[int],
    parallel: bool,
    iters: int,
    warmup: int,
    dtype: torch.dtype,
    print_rank: int,
) -> None:
    """Run an all_gather micro-benchmark for a given process group.

    Args:
        group: The process group to run collectives on.
        group_id: The id of the group.
        group_ranks: The ranks in the group.
        num_groups: The total number of groups.
        device: Torch device for tensor allocation and synchronization.
        sizes_bytes: List of per-rank input sizes in bytes.
        parallel: Whether to run the allgather of multiple groups in parallel.
        iters: Number of timed iterations per message size.
        warmup: Number of warmup iterations per message size (not timed).
        dtype: Tensor dtype used for the benchmark tensors.
        print_rank: Global rank that prints results for this group.

    Returns:
        None. Prints timing and aggregate bandwidth per message size.
    """
    group_size = dist.get_world_size(group=group)
    rank = dist.get_rank()

    for size_bytes in sizes_bytes:
        num_elements = size_bytes // torch.tensor([], dtype=dtype).element_size()
        if num_elements <= 0:
            raise ValueError(
                f"Message size ({size_bytes} bytes) is too small for dtype {dtype}, got {num_elements} elements."
            )

        inp = torch.randn(int(num_elements), dtype=dtype, device=device)
        out_tensor = torch.empty((group_size,) + inp.shape, dtype=dtype, device=device)

        for _ in range(warmup):
            dist.all_gather_into_tensor(out_tensor, inp, group=group)
        barrier(group)
        if device.type == "cuda":
            torch.cuda.synchronize()

        if parallel:
            t0 = time.perf_counter()
            for _ in range(iters):
                dist.all_gather_into_tensor(out_tensor, inp, group=group)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            total_s = t1 - t0
            avg_s = total_s / iters
            # Each iteration moves group_size * size_bytes worth of data to each rank
            total_bytes = size_bytes * group_size
            algobw = bytes_to_gib_per_s(total_bytes, avg_s)
            busbw = algobw * (group_size - 1) / group_size

            if rank == print_rank:
                print(
                    f"[RANK-{rank}][StridedAllGather][Parallel][Group-{group_id}][Ranks-{group_ranks}] size={format_size_mb(size_bytes)} "
                    f"avg={avg_s*1e3:.2f} ms algobw={algobw:.2f} GiB/s busbw={busbw:.2f} GiB/s"
                )
        else:
            for g_id in range(num_groups):
                if g_id == group_id:
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        dist.all_gather_into_tensor(out_tensor, inp, group=group)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()

                    total_s = t1 - t0
                    avg_s = total_s / iters
                    total_bytes = size_bytes * group_size
                    algobw = bytes_to_gib_per_s(total_bytes, avg_s)
                    busbw = algobw * (group_size - 1) / group_size
                    if rank == print_rank:
                        print(
                            f"[RANK-{rank}][StridedAllGather][Single][Group-{g_id}][Ranks-{group_ranks}] size={format_size_mb(size_bytes)} "
                            f"avg={avg_s*1e3:.2f} ms algobw={algobw:.2f} GiB/s busbw={busbw:.2f} GiB/s"
                        )
                barrier()
                torch.cuda.synchronize()


def init_distributed(backend: str) -> Tuple[int, int, int, int]:
    """Initialize torch.distributed via torchrun env.

    Returns:
        (rank, world_size, local_rank, local_world_size)
    """

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))

    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{local_rank}"),
            init_method="env://",
            timeout=timedelta(minutes=5),
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank, local_world_size


def run_strided_allgather_benchmark(args):
    # init pytorch distributed
    rank, world_size, local_rank, _ = init_distributed(args.backend)

    # validate arguments
    assert args.stride > 0, "Stride must be greater than 0"
    assert world_size % args.stride == 0, "World size must be divisible by stride"
    num_groups = args.stride
    num_ranks_per_group = world_size // args.stride

    dtype = get_dtype(args.dtype)
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    # Sizes to bytes
    sizes_mb = [int(s.strip()) for s in args.sizes_mb.split(",") if s.strip()]
    sizes_bytes = [s * 1024 * 1024 for s in sizes_mb]

    # > if stride is 8, world_size is 32, then the groups will be:
    # [0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26], [3, 11, 19, 27],
    # [4, 12, 20, 28], [5, 13, 21, 29], [6, 14, 22, 30], [7, 15, 23, 31]
    # > if stride is 4, world_size is 16, then the groups will be:
    # [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]
    # > if stride is 2, world_size is 8, then the groups will be:
    # [0, 2, 4, 6], [1, 3, 5, 7]
    # > if stride is 1, world_size is 8, then the groups will be:
    # [0, 1, 2, 3, 4, 5, 6, 7]
    pg_by_gpu = None
    group_leader = None
    group_id = None
    group_ranks = None
    for g_id in range(num_groups):
        ranks = [
            local_rank_in_group * args.stride + g_id for local_rank_in_group in range(num_ranks_per_group)
        ]
        leader = min(ranks)
        # All ranks call new_group for every GPU id
        pg = dist.new_group(ranks=ranks)
        if rank in ranks:
            pg_by_gpu = pg
            group_leader = leader
            group_id = g_id
            group_ranks = ranks
            if rank == group_leader:
                print(f"[benchmark][strided-allgather][SETUP-PG] group_id={g_id}, ranks={ranks}")
    assert pg_by_gpu is not None, f"RANK[{rank}]: Failed to create process group"

    benchmark_strided_allgather(
        group=pg_by_gpu,
        group_id=group_id,
        group_ranks=group_ranks,
        num_groups=num_groups,
        device=device,
        sizes_bytes=sizes_bytes,
        parallel=args.parallel,
        iters=args.iters,
        warmup=args.warmup,
        dtype=dtype,
        print_rank=group_leader,
    )

    # Ensure subgroup completes before world test
    barrier()
    torch.cuda.synchronize()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    """
    Run the strided allgather benchmark.
    """
    parser = argparse.ArgumentParser(description="Strided Allgather Benchmark")
    add_arguments(parser)
    args = parser.parse_args()
    run_strided_allgather_benchmark(args)
