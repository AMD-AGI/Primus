###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import csv
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

# [Seq, HiddenSize]
MODEL_PARAMS_TABLE = {
    "llama-2-7B": (4096, 4096),
    "llama-2-70B": (4096, 8192),
    "llama-3-8B": (8192, 4096),
    "llama-3-70B": (8192, 8192),
    "deepseek-v2-lite": (4096, 2048),
    "deepseek-v2": (4096, 5120),
    "deepseek-v3": (8192, 7168),
    "mitral-8x22B": (8192, 6144),
}
MBS_LIST = [1, 2, 3, 4, 5, 6, 7, 8]
ITERS = 100


def test_allreduce(mbs, seq, hidden, dtype, rank, local_rank, world_size, dry_run=False):
    if local_rank == 0:
        size = mbs * seq * hidden * torch.tensor([], dtype=dtype).element_size()
        print("AllReduce with input size(Byte): ", size)
        print(
            f"Rccl-test command: \n$ mpirun -np {world_size} -N $NNODES ./build/all_reduce_perf -b {size} -e {size} -g 1"
        )
    if dry_run:
        return 0, 0
    shape = (mbs, seq, hidden)
    device = torch.device(f"cuda:{local_rank}")
    tensor = torch.ones(shape, dtype=dtype, device=device)
    if local_rank == 0:
        print("AllReduce with input size(Byte): ", tensor.nelement() * tensor.element_size())
    # Warm-up
    for _ in range(5):
        dist.all_reduce(tensor)
    dist.barrier()
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(ITERS):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    end = time.time()

    size_bytes = tensor.numel() * tensor.element_size()
    total_bytes = 2 * size_bytes * (world_size - 1) / world_size
    avg_time = (end - start) / ITERS
    bandwidth = total_bytes / avg_time / 1e9  # GB/s

    return avg_time, bandwidth


def test_allgather(mbs, seq, hidden, dtype, rank, local_rank, world_size, dry_run=False):
    local_seq = seq // world_size

    if local_rank == 0:
        element_size = torch.tensor([], dtype=dtype).element_size()
        nelement = mbs * local_seq * hidden
        print(
            "AllGather with input size(Byte): ",
            nelement * element_size,
            " Output size ",
            world_size * nelement * element_size,
        )
    if dry_run:
        return 0, 0

    shape = (mbs, local_seq, hidden)
    device = torch.device(f"cuda:{local_rank}")
    tensor = torch.randn(shape, dtype=dtype, device=device)

    # Gather buffer
    output = [torch.randn_like(tensor) for _ in range(world_size)]
    if local_rank == 0:
        print(
            "AllGather with input size(Byte): ",
            tensor.nelement() * tensor.element_size(),
            " Output size ",
            world_size * tensor.nelement() * tensor.element_size(),
        )
    for _ in range(5):
        dist.all_gather(output, tensor)
    dist.barrier()
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(ITERS):
        dist.all_gather(output, tensor)
    torch.cuda.synchronize()
    end = time.time()

    send_bytes = tensor.numel() * tensor.element_size() * world_size * (world_size - 1) / world_size
    avg_time = (end - start) / ITERS
    bandwidth = send_bytes / avg_time / 1e9

    return avg_time, bandwidth


def test_reducescatter(mbs, seq, hidden, dtype, rank, local_rank, world_size, dry_run=False):
    full_shape = (mbs, seq, hidden)
    chunk_seq = seq // world_size
    chunk_shape = (mbs, chunk_seq, hidden)

    if local_rank == 0:
        print(
            "ReduceScatter with each output chunk size(Byte): ",
            mbs * chunk_seq * hidden * torch.tensor([], dtype=dtype).element_size(),
        )

    if dry_run:
        return 0, 0

    device = torch.device(f"cuda:{local_rank}")
    tensor = torch.ones(full_shape, dtype=dtype, device=device)
    output = torch.empty(chunk_shape, dtype=dtype, device=device)
    if local_rank == 0:
        print("ReduceScatter with each output chunk size(Byte): ", output.nelement() * output.element_size())
    for _ in range(5):
        dist.reduce_scatter(output, list(tensor.chunk(world_size, dim=1)))
    dist.barrier()
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(ITERS):
        dist.reduce_scatter(output, list(tensor.chunk(world_size, dim=1)))
    torch.cuda.synchronize()
    end = time.time()

    send_bytes = tensor.numel() * tensor.element_size() * (world_size - 1) / world_size
    avg_time = (end - start) / ITERS
    bandwidth = send_bytes / avg_time / 1e9

    return avg_time, bandwidth


def benchmark(test_func, output_csv_path, rank, local_rank, world_size, dry_run=False):
    benchmark_results = []

    for model_name, (seq, hidden) in MODEL_PARAMS_TABLE.items():
        for mbs in MBS_LIST:
            print(f"\nModel Name {model_name}, mbs {mbs}")
            for dtype in [torch.float16]:
                avg_time, bandwidth = test_func(
                    mbs, seq, hidden, dtype, rank, local_rank, world_size, dry_run
                )
                if rank == 0:
                    result = {
                        "Model": model_name,
                        "MBS": mbs,
                        "Seq": seq,
                        "HiddenSize": hidden,
                        "DataType": dtype,
                        "WorldSize": world_size,
                        "Time(s)": avg_time,
                        "Bandwidth(GB/s)": bandwidth,
                    }
                    benchmark_results.append(result)

    if rank == 0 and not dry_run:
        fieldnames = list(benchmark_results[0].keys())
        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in benchmark_results:
                writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allreduce-report-csv-path", type=str)
    parser.add_argument("--allgather-report-csv-path", type=str)
    parser.add_argument(
        "-dry",
        "--dry-run",
        action="store_true",
        help="Testing run to generate message size and rccl-test command.",
    )
    parser.add_argument("--reducescatter-report-csv-path", type=str)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    assert world_size >= 2, "This script requires at least 2 processes."

    if args.dry_run:
        benchmark(test_allreduce, args.allreduce_report_csv_path, 0, 0, world_size, True)
        # benchmark(test_allgather, args.allgather_report_csv_path, 0, 0, world_size, True)
        # benchmark(test_reducescatter, args.reducescatter_report_csv_path, 0, 0, world_size, True)
        exit(0)

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=5),
    )
    dist.barrier()
    torch.manual_seed(42 + rank)

    benchmark(test_allreduce, args.allreduce_report_csv_path, rank, local_rank, world_size)
    benchmark(test_allgather, args.allgather_report_csv_path, rank, local_rank, world_size)
    benchmark(test_reducescatter, args.reducescatter_report_csv_path, rank, local_rank, world_size)

    dist.barrier()
    dist.destroy_process_group()
