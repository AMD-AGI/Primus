#!/usr/bin/env python3
"""Standalone two-node reproducer for GEMM interference from all-gather.

Dependencies are limited to PyTorch and MORI. Launch with 8 ranks per node.
The default shapes are taken from the Llama 405B-width batch-size-2 trace.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

import mori.shmem as shmem
from mori.ccl import HierAllGather


SHAPES = {
    # name: (M, N, K, transpose A storage, transpose B storage)
    "mlp_up": (4096, 53248, 16384, False, True),
    "mlp_down": (4096, 16384, 53248, False, False),
    "mlp_wgrad": (53248, 16384, 4096, True, False),
    "attention_proj": (4096, 16384, 16384, False, True),
}


def _matrix(rows, cols, transposed, device):
    if transposed:
        base = torch.empty((cols, rows), dtype=torch.bfloat16, device=device)
        base.normal_(mean=0.0, std=0.01)
        return base.t(), base
    tensor = torch.empty((rows, cols), dtype=torch.bfloat16, device=device)
    tensor.normal_(mean=0.0, std=0.01)
    return tensor, tensor


def _check_all_gather(output, numel, world_size):
    indices = torch.tensor([0, numel // 2, numel - 1], device=output.device)
    for source_rank in range(world_size):
        values = output[source_rank * numel + indices]
        if not torch.equal(values, torch.full_like(values, source_rank + 1)):
            raise RuntimeError(f"all-gather sample mismatch for source rank {source_rank}")


def _percentile(values, percentile):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile / 100)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", choices=["all", *SHAPES], default=["all"])
    parser.add_argument("--modes", nargs="+", choices=["baseline", "copy", "mori", "rccl"], default=None)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--fsdp-numel", type=int, default=199_231_488)
    parser.add_argument("--output-json")
    parser.add_argument("--trace-dir")
    parser.add_argument("--trace-shape", choices=SHAPES, default="mlp_down")
    parser.add_argument("--trace-mode", choices=["copy", "mori", "rccl"], default="mori")
    args = parser.parse_args()

    shape_names = list(SHAPES) if "all" in args.shapes else args.shapes
    modes = args.modes or ["baseline", "copy", "mori", "rccl"]
    if "baseline" not in modes:
        modes = ["baseline", *modes]

    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "2G")
    os.environ.setdefault("MORI_HIER_CUDA_GRAPH", "0")
    os.environ.setdefault("MORI_HIER_DEBUG_SYNC", "0")
    os.environ.setdefault("MORI_HIER_FUSE_LOCAL", "1")
    os.environ.setdefault("MORI_HIER_FUSE_REMOTE", "1")
    os.environ.setdefault("MORI_HIER_LOCAL_PUSHONLY", "1")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("cpu:gloo,cuda:nccl")
    sync_group = dist.new_group(backend="gloo")
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ranks_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
    num_nodes = world_size // ranks_per_node
    shmem.shmem_torch_process_group_init("default")

    per_rank_bytes = args.fsdp_numel * 2
    slice_min_bytes = 8 << 20
    output_workspace_bytes = max(
        num_nodes * per_rank_bytes,
        world_size * min(per_rank_bytes, slice_min_bytes),
    )
    mori_all_gather = HierAllGather(
        my_pe=rank,
        npes=world_size,
        input_buffer_size=min(per_rank_bytes, slice_min_bytes),
        output_buffer_size=output_workspace_bytes,
        copy_output_to_user=True,
        ranks_per_node=ranks_per_node,
        slice_min_bytes=slice_min_bytes,
        slice_direct=True,
    )

    ag_input = torch.full(
        (args.fsdp_numel,),
        rank + 1,
        dtype=torch.bfloat16,
        device=device,
    )
    ag_output = torch.empty(
        args.fsdp_numel * world_size,
        dtype=torch.bfloat16,
        device=device,
    )
    copy_output = torch.empty_like(ag_input)
    compute_stream = torch.cuda.current_stream(device)
    comm_stream = torch.cuda.Stream(device=device, priority=-1)

    def launch_comm(mode):
        if mode == "baseline":
            return None
        with torch.cuda.stream(comm_stream):
            if mode == "copy":
                copy_output.copy_(ag_input, non_blocking=True)
                return None
            if mode == "mori":
                if not mori_all_gather(
                    ag_input,
                    ag_output,
                    args.fsdp_numel,
                    stream=comm_stream,
                ):
                    raise RuntimeError("MORI all-gather failed")
                return None
            return dist.all_gather_into_tensor(
                ag_output,
                ag_input,
                group=dist.group.WORLD,
                async_op=True,
            )

    # Eagerly initialize both communication paths outside measured regions.
    launch_comm("mori")
    comm_stream.synchronize()
    _check_all_gather(ag_output, args.fsdp_numel, world_size)
    launch_comm("rccl")
    comm_stream.synchronize()
    _check_all_gather(ag_output, args.fsdp_numel, world_size)

    local_results = {}
    for shape_name in shape_names:
        m, n, k, transpose_a, transpose_b = SHAPES[shape_name]
        a, a_storage = _matrix(m, k, transpose_a, device)
        b, b_storage = _matrix(k, n, transpose_b, device)
        output = torch.empty((m, n), dtype=torch.bfloat16, device=device)

        def gemm():
            torch.mm(a, b, out=output)

        for _ in range(args.warmup):
            gemm()
        torch.cuda.synchronize()

        for mode in modes:
            for _ in range(args.warmup):
                launch_comm(mode)
                gemm()
                torch.cuda.synchronize()

            gemm_times = []
            comm_times = []
            wall_times = []
            for _ in range(args.reps):
                dist.barrier(group=sync_group)
                torch.cuda.synchronize()
                gemm_start = torch.cuda.Event(enable_timing=True)
                gemm_end = torch.cuda.Event(enable_timing=True)
                comm_start = torch.cuda.Event(enable_timing=True)
                comm_end = torch.cuda.Event(enable_timing=True)
                start = time.perf_counter()
                with torch.cuda.stream(comm_stream):
                    comm_start.record()
                    work = launch_comm(mode)
                    if work is not None:
                        work.wait()
                    comm_end.record()
                with torch.cuda.stream(compute_stream):
                    gemm_start.record()
                    gemm()
                    gemm_end.record()
                gemm_end.synchronize()
                comm_end.synchronize()
                torch.cuda.synchronize()
                wall_times.append((time.perf_counter() - start) * 1e3)
                gemm_times.append(gemm_start.elapsed_time(gemm_end))
                comm_times.append(comm_start.elapsed_time(comm_end))

            local_results[(shape_name, mode)] = {
                "gemm_ms": gemm_times,
                "comm_ms": comm_times,
                "wall_ms": wall_times,
            }

        del a, b, a_storage, b_storage, output
        torch.cuda.empty_cache()

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, local_results, group=sync_group)
    rows = []
    if rank == 0:
        for shape_name in shape_names:
            baseline_values = [
                value
                for rank_results in gathered_results
                for value in rank_results[(shape_name, "baseline")]["gemm_ms"]
            ]
            baseline_median = statistics.median(baseline_values)
            for mode in modes:
                gemm_values = [
                    value
                    for rank_results in gathered_results
                    for value in rank_results[(shape_name, mode)]["gemm_ms"]
                ]
                comm_values = [
                    value
                    for rank_results in gathered_results
                    for value in rank_results[(shape_name, mode)]["comm_ms"]
                ]
                wall_values = [
                    value
                    for rank_results in gathered_results
                    for value in rank_results[(shape_name, mode)]["wall_ms"]
                ]
                row = {
                    "shape": shape_name,
                    "mode": mode,
                    "m": SHAPES[shape_name][0],
                    "n": SHAPES[shape_name][1],
                    "k": SHAPES[shape_name][2],
                    "gemm_median_ms": statistics.median(gemm_values),
                    "gemm_p95_ms": _percentile(gemm_values, 95),
                    "gemm_slowdown_pct": (
                        100 * (statistics.median(gemm_values) / baseline_median - 1)
                    ),
                    "comm_median_ms": statistics.median(comm_values),
                    "wall_median_ms": statistics.median(wall_values),
                }
                rows.append(row)
                print(
                    f"GEMM_AG_REPRO shape={shape_name} mode={mode} "
                    f"gemm_ms={row['gemm_median_ms']:.3f} "
                    f"slowdown_pct={row['gemm_slowdown_pct']:.2f} "
                    f"comm_ms={row['comm_median_ms']:.3f} "
                    f"wall_ms={row['wall_median_ms']:.3f}",
                    flush=True,
                )

        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(rows, indent=2) + "\n")

    if args.trace_dir:
        shape_name = args.trace_shape
        if shape_name not in shape_names:
            raise ValueError("--trace-shape must also be included in --shapes")
        m, n, k, transpose_a, transpose_b = SHAPES[shape_name]
        a, a_storage = _matrix(m, k, transpose_a, device)
        b, b_storage = _matrix(k, n, transpose_b, device)
        output = torch.empty((m, n), dtype=torch.bfloat16, device=device)
        dist.barrier(group=sync_group)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as profiler:
            with record_function(f"gemm_overlap_{args.trace_mode}_{shape_name}"):
                launch_comm(args.trace_mode)
                torch.mm(a, b, out=output)
                torch.cuda.synchronize()
        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(
            str(trace_dir / f"rank{rank}_{args.trace_mode}_{shape_name}.json")
        )
        del a, b, a_storage, b_storage, output

    dist.barrier(group=sync_group)
    del mori_all_gather
    shmem.shmem_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
