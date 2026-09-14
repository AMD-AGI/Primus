#!/usr/bin/env python3
"""Cross-node MORI/RCCL all-gather bandwidth at FSDP-scale message sizes."""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

from primus.backends.common.mori_allgather import MoriAllGather


def _time_collective(fn, sync_group, reps, warmup):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    local_times = []
    for _ in range(reps):
        dist.barrier(group=sync_group)
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        local_times.append((time.perf_counter() - start) * 1e3)

    per_rank_times = [None] * dist.get_world_size()
    dist.all_gather_object(per_rank_times, local_times, group=sync_group)
    return [max(rank_times[index] for rank_times in per_rank_times) for index in range(reps)]


def _check_samples(output, numel, world_size):
    sample_indices = sorted({0, numel // 2, numel - 1})
    for rank in range(world_size):
        values = output[rank * numel + torch.tensor(sample_indices, device=output.device)]
        expected = torch.full_like(values, rank + 1)
        if not torch.equal(values, expected):
            raise RuntimeError(f"all-gather sample mismatch for source rank {rank}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes-mib",
        type=float,
        nargs="+",
        default=[8, 32, 64, 128, 256, 380.00390625],
    )
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("cpu:gloo,cuda:nccl")
    sync_group = dist.new_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ranks_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
    sizes_numel = [int(size * (1 << 20) / 2) for size in args.sizes_mib]
    max_numel = max(sizes_numel)

    input_tensor = torch.full(
        (max_numel,),
        rank + 1,
        dtype=torch.bfloat16,
        device=device,
    )
    output_tensor = torch.empty(
        max_numel * world_size,
        dtype=torch.bfloat16,
        device=device,
    )

    mori = MoriAllGather(ranks_per_node=ranks_per_node)

    def run_mori(numel):
        work = mori(
            output_tensor[: numel * world_size],
            input_tensor[:numel],
            dist.group.WORLD,
            async_op=True,
        )
        if work is not None:
            work.wait()

    def run_rccl(numel):
        work = dist.all_gather_into_tensor(
            output_tensor[: numel * world_size],
            input_tensor[:numel],
            group=dist.group.WORLD,
            async_op=True,
        )
        work.wait()

    # Construct MORI once at the maximum capacity so every measured size reuses
    # the same buffers and does not include setup or resize time.
    run_mori(max_numel)
    torch.cuda.synchronize()
    _check_samples(output_tensor, max_numel, world_size)

    rows = []
    for mode, fn in (("mori", run_mori), ("rccl", run_rccl)):
        for requested_mib, numel in zip(args.sizes_mib, sizes_numel):
            times_ms = _time_collective(
                lambda numel=numel: fn(numel),
                sync_group,
                args.reps,
                args.warmup,
            )
            fn(numel)
            torch.cuda.synchronize()
            _check_samples(output_tensor, numel, world_size)

            median_ms = statistics.median(times_ms)
            per_rank_bytes = numel * 2
            algorithm_bytes = per_rank_bytes * (world_size - 1)
            output_bytes = per_rank_bytes * world_size
            row = {
                "mode": mode,
                "requested_mib": requested_mib,
                "per_rank_mib": per_rank_bytes / (1 << 20),
                "median_ms": median_ms,
                "min_ms": min(times_ms),
                "max_ms": max(times_ms),
                "algorithm_gbps": algorithm_bytes / (median_ms * 1e6),
                "output_gbps": output_bytes / (median_ms * 1e6),
            }
            rows.append(row)
            if rank == 0:
                print(
                    f"AG_BENCH mode={mode} per_rank_mib={row['per_rank_mib']:.6f} "
                    f"median_ms={median_ms:.3f} min_ms={row['min_ms']:.3f} "
                    f"algo_GBps={row['algorithm_gbps']:.2f} "
                    f"output_GBps={row['output_gbps']:.2f}",
                    flush=True,
                )

    if rank == 0 and args.output_json:
        with open(args.output_json, "w") as output_file:
            json.dump(rows, output_file, indent=2)
            output_file.write("\n")

    import mori.shmem as shmem

    dist.barrier(group=sync_group)
    shmem.shmem_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
