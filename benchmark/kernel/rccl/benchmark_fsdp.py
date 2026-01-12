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

# [d_model, d_ff, n_heads, n_kv_heads, d_qkv]
# parameter calculation reference: https://jax-ml.github.io/scaling-book/applied-training/#counting-parameters-and-flops
MODEL_PARAMS_TABLE = {
    "llama3-70B": (8192, 8192 * 3.5, 64, 8, 128),
    "llama3-405B": (16384, 53248, 128, 16, 128),
}
ITERS = 20


def test_allgather(size, dtype, rank, local_rank, world_size, dry_run=False):
    (size)
    chunk_shape = size // world_size

    if local_rank == 0:
        element_size = torch.tensor([], dtype=dtype).element_size()
        nelement = size // world_size
        byte_size = nelement * element_size
        print(
            "AllGather with input size(Byte): ",
            byte_size,
            " Output size ",
            world_size * byte_size,
        )
        print(
            "HSA_NO_SCRATCH_RECLAIM=1 ./build/all_gather_perf -b ",
            byte_size,
            " -e ",
            byte_size,
            " -g ",
            world_size,
            " -d half",
        )
        print(
            "HSA_NO_SCRATCH_RECLAIM=1 mpirun --allow-run-as-root -np ",
            world_size,
            " ./build/all_gather_perf -b ",
            byte_size,
            " -e ",
            byte_size,
            " -g 1 -d half",
        )
    if dry_run:
        return 0, 0

    device = torch.device(f"cuda:{local_rank}")
    tensor = torch.randn(chunk_shape, dtype=dtype, device=device)

    # Gather buffer
    output = [torch.randn_like(tensor) for _ in range(world_size)]

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


def test_reducescatter(size, dtype, rank, local_rank, world_size, dry_run=False):
    full_shape = size
    chunk_shape = size // world_size

    if local_rank == 0:
        byte_size = size // world_size * torch.tensor([], dtype=dtype).element_size()
        print("ReduceScatter with total size(Byte): ", byte_size)
        print(
            "HSA_NO_SCRATCH_RECLAIM=1 ./build/reduce_scatter_perf -b ",
            byte_size,
            " -e ",
            byte_size,
            " -g ",
            world_size,
            " -d half # assuming dtype float16",
        )
        print(
            "HSA_NO_SCRATCH_RECLAIM=1 mpirun --allow-run-as-root -np ",
            world_size,
            " ./build/reduce_scatter_perf -b ",
            byte_size,
            " -e ",
            byte_size,
            " -g 1 -d half",
        )
    if dry_run:
        return 0, 0

    device = torch.device(f"cuda:{local_rank}")
    tensor = torch.ones(full_shape, dtype=dtype, device=device)
    output = torch.empty(chunk_shape, dtype=dtype, device=device)

    for _ in range(5):
        dist.reduce_scatter(output, list(tensor.chunk(world_size, dim=0)))
    dist.barrier()
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(ITERS):
        dist.reduce_scatter(output, list(tensor.chunk(world_size, dim=0)))
    torch.cuda.synchronize()
    end = time.time()

    send_bytes = tensor.numel() * tensor.element_size() * (world_size - 1) / world_size
    avg_time = (end - start) / ITERS
    bandwidth = send_bytes / avg_time / 1e9

    return avg_time, bandwidth


def benchmark(test_func, output_csv_path, rank, local_rank, world_size, dry_run=False):
    benchmark_results = []

    for model_name, (d_model, d_ff, n_heads, n_kv_heads, d_qkv) in MODEL_PARAMS_TABLE.items():
        size = d_model * d_ff * 3 + 2 * d_model * n_heads * d_qkv + 2 * d_model * n_kv_heads * d_qkv
        size = int(size)
        if rank == 0:
            print(f"\nModel Name {model_name} with size {size}")
        for dtype in [torch.float16]:
            avg_time, bandwidth = test_func(size, dtype, rank, local_rank, world_size, dry_run)
            if rank == 0:
                result = {
                    "Model": model_name,
                    "Layer Weight": size,
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
        benchmark(test_allgather, args.allgather_report_csv_path, 0, 0, world_size, True)
        benchmark(test_reducescatter, args.reducescatter_report_csv_path, 0, 0, world_size, True)
        exit(0)

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=5),
    )
    dist.barrier()
    torch.manual_seed(42 + rank)

    benchmark(test_allgather, args.allgather_report_csv_path, rank, local_rank, world_size)
    benchmark(test_reducescatter, args.reducescatter_report_csv_path, rank, local_rank, world_size)

    dist.barrier()
    dist.destroy_process_group()
