"""Multi-process preflight workers.

Spawned via ``torchrun --nproc-per-node=N`` from ``pilot.tools.preflight``.

Each worker process:
  - Joins a torch.distributed process group (NCCL/RCCL backend).
  - Performs collective benchmarks at the requested sizes.
  - Reports its measurements via stdout (only RANK=0 emits JSON).

This module does **not** define a CLI; it is meant to be invoked through
``python -m torch.distributed.run`` so torchrun handles rendezvous, ranks,
and process IDs.

Collective coverage (5 primitives important for training):
  - **AllReduce**: DP gradient sync.
  - **AllGather**: FSDP/ZeRO parameter gather, TP weight gather.
  - **ReduceScatter**: FSDP/ZeRO gradient scatter.
  - **Broadcast**: parameter init broadcast, sanity probes.
  - **AllToAll**: MoE token routing, expert parallelism.

`size_mb` semantics follow NCCL/RCCL convention:
  - AllReduce, Broadcast, AllToAll: tensor size (bytes per rank).
  - AllGather: total **output** size (= per-rank input * world_size).
  - ReduceScatter: total **input** size (= per-rank output * world_size).

Bus bandwidth (`bw_gbs`) is derived from algorithmic bandwidth using the
standard ring-collective correction factors:
  - AllReduce:     2*(N-1)/N
  - AllGather, ReduceScatter, AllToAll: (N-1)/N
  - Broadcast:     1                          (one-to-all)

Output (rank 0 only) goes to stdout as a single JSON object:

    {
      "world_size": <int>,
      "device_count": <int>,
      "intra_node_bw_gbs": <float>,            # AllReduce 256MB sustained
      "rccl_baseline": {
        "allreduce":      [{"size_mb": <int>, "bw_gbs": <float>, "latency_us": <float>}, ...],
        "allgather":      [...],
        "reduce_scatter": [...],
        "broadcast":      [...],
        "alltoall":       [...]
      },
      "t1_connectivity": {"pass": <bool>, "msg": <str>}
    }

All other ranks emit nothing on stdout (they may log to stderr).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any


def _is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _allreduce_bw(group_size: int, dtype, n_elements: int, *, warmup: int = 3, runs: int = 8):
    """Return (bw_gbs, latency_us) for one AllReduce size.

    Bus bandwidth (algbw_factor) for ring AllReduce is 2*(N-1)/N * size; we
    approximate algbw = size_bytes / time and bus_bw = algbw * 2*(N-1)/N.
    Many production reports use bus_bw, so we return that.
    """
    import torch
    import torch.distributed as dist

    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    t = torch.ones(n_elements, dtype=dtype, device=dev)

    # warmup
    for _ in range(warmup):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    dist.barrier()

    times: list[float] = []
    for _ in range(runs):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        dist.all_reduce(t)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    size_bytes = n_elements * t.element_size()
    algbw = size_bytes / median  # bytes/sec
    busbw = algbw * 2 * (group_size - 1) / group_size
    return (busbw / 1e9, median * 1e6)


def _alltoall_bw(group_size: int, dtype, n_elements: int, *, warmup: int = 2, runs: int = 6):
    """AllToAll (each rank sends N-1/N of its data). bus_bw = algbw * (N-1)/N.

    `n_elements` is the per-rank tensor size; auto-truncated to multiple of N.
    """
    import torch
    import torch.distributed as dist

    if n_elements % group_size != 0:
        n_elements = (n_elements // group_size) * group_size
    if n_elements == 0:
        return (0.0, 0.0)

    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    src = torch.ones(n_elements, dtype=dtype, device=dev)
    dst = torch.empty_like(src)

    for _ in range(warmup):
        dist.all_to_all_single(dst, src)
    torch.cuda.synchronize()
    dist.barrier()

    times: list[float] = []
    for _ in range(runs):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        dist.all_to_all_single(dst, src)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    size_bytes = n_elements * src.element_size()
    algbw = size_bytes / median
    busbw = algbw * (group_size - 1) / group_size
    return (busbw / 1e9, median * 1e6)


def _allgather_bw(
    group_size: int, dtype, n_elements_per_rank: int, *, warmup: int = 2, runs: int = 6,
):
    """AllGather: each rank contributes n_per_rank, output is N*n_per_rank.

    bus_bw = algbw * (N-1)/N where algbw = output_bytes / time.
    """
    import torch
    import torch.distributed as dist

    if n_elements_per_rank == 0:
        return (0.0, 0.0)

    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    src = torch.ones(n_elements_per_rank, dtype=dtype, device=dev)
    dst = torch.empty(n_elements_per_rank * group_size, dtype=dtype, device=dev)

    for _ in range(warmup):
        dist.all_gather_into_tensor(dst, src)
    torch.cuda.synchronize()
    dist.barrier()

    times: list[float] = []
    for _ in range(runs):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        dist.all_gather_into_tensor(dst, src)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    output_bytes = n_elements_per_rank * group_size * src.element_size()
    algbw = output_bytes / median
    busbw = algbw * (group_size - 1) / group_size
    return (busbw / 1e9, median * 1e6)


def _reduce_scatter_bw(
    group_size: int, dtype, n_elements_per_rank: int, *, warmup: int = 2, runs: int = 6,
):
    """ReduceScatter: input N*n_per_rank, output n_per_rank per rank.

    bus_bw = algbw * (N-1)/N where algbw = input_bytes / time.
    """
    import torch
    import torch.distributed as dist

    if n_elements_per_rank == 0:
        return (0.0, 0.0)

    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    src = torch.ones(n_elements_per_rank * group_size, dtype=dtype, device=dev)
    dst = torch.empty(n_elements_per_rank, dtype=dtype, device=dev)

    for _ in range(warmup):
        dist.reduce_scatter_tensor(dst, src)
    torch.cuda.synchronize()
    dist.barrier()

    times: list[float] = []
    for _ in range(runs):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        dist.reduce_scatter_tensor(dst, src)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    input_bytes = n_elements_per_rank * group_size * src.element_size()
    algbw = input_bytes / median
    busbw = algbw * (group_size - 1) / group_size
    return (busbw / 1e9, median * 1e6)


def _broadcast_bw(
    group_size: int, dtype, n_elements: int, *, src_rank: int = 0, warmup: int = 2, runs: int = 6,
):
    """Broadcast (one-to-all). bus_bw = algbw (no ring factor)."""
    import torch
    import torch.distributed as dist

    if n_elements == 0:
        return (0.0, 0.0)

    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    t = torch.ones(n_elements, dtype=dtype, device=dev)

    for _ in range(warmup):
        dist.broadcast(t, src=src_rank)
    torch.cuda.synchronize()
    dist.barrier()

    times: list[float] = []
    for _ in range(runs):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        dist.broadcast(t, src=src_rank)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    size_bytes = n_elements * t.element_size()
    algbw = size_bytes / median
    busbw = algbw  # one-to-all: no ring correction
    return (busbw / 1e9, median * 1e6)


def _curve_entry(mb: int, bw: float, lat: float, err: str | None = None) -> dict[str, Any]:
    e: dict[str, Any] = {"size_mb": mb, "bw_gbs": round(bw, 2), "latency_us": round(lat, 1)}
    if err:
        e["error"] = err[:120]
    return e


def _ar_curve(sizes_mb: list[int], world_size: int) -> list[dict[str, Any]]:
    import torch
    out = []
    for mb in sizes_mb:
        n_elem = (mb * (1 << 20)) // 4  # float32, total tensor size
        try:
            bw, lat = _allreduce_bw(world_size, torch.float32, n_elem)
            out.append(_curve_entry(mb, bw, lat))
        except Exception as exc:  # noqa: BLE001
            out.append(_curve_entry(mb, 0.0, -1.0, str(exc)))
    return out


def _a2a_curve(sizes_mb: list[int], world_size: int) -> list[dict[str, Any]]:
    import torch
    out = []
    for mb in sizes_mb:
        n_elem = (mb * (1 << 20)) // 4
        try:
            bw, lat = _alltoall_bw(world_size, torch.float32, n_elem)
            out.append(_curve_entry(mb, bw, lat))
        except Exception as exc:  # noqa: BLE001
            out.append(_curve_entry(mb, 0.0, -1.0, str(exc)))
    return out


def _ag_curve(sizes_mb: list[int], world_size: int) -> list[dict[str, Any]]:
    """AllGather. `size_mb` = total output size; per-rank input = total / N."""
    import torch
    out = []
    for mb in sizes_mb:
        n_elem_per_rank = ((mb * (1 << 20)) // 4) // max(world_size, 1)
        if n_elem_per_rank == 0:
            out.append(_curve_entry(mb, 0.0, -1.0, "size below per-rank threshold"))
            continue
        try:
            bw, lat = _allgather_bw(world_size, torch.float32, n_elem_per_rank)
            out.append(_curve_entry(mb, bw, lat))
        except Exception as exc:  # noqa: BLE001
            out.append(_curve_entry(mb, 0.0, -1.0, str(exc)))
    return out


def _rs_curve(sizes_mb: list[int], world_size: int) -> list[dict[str, Any]]:
    """ReduceScatter. `size_mb` = total input size; per-rank output = total / N."""
    import torch
    out = []
    for mb in sizes_mb:
        n_elem_per_rank = ((mb * (1 << 20)) // 4) // max(world_size, 1)
        if n_elem_per_rank == 0:
            out.append(_curve_entry(mb, 0.0, -1.0, "size below per-rank threshold"))
            continue
        try:
            bw, lat = _reduce_scatter_bw(world_size, torch.float32, n_elem_per_rank)
            out.append(_curve_entry(mb, bw, lat))
        except Exception as exc:  # noqa: BLE001
            out.append(_curve_entry(mb, 0.0, -1.0, str(exc)))
    return out


def _bcast_curve(sizes_mb: list[int], world_size: int) -> list[dict[str, Any]]:
    import torch
    out = []
    for mb in sizes_mb:
        n_elem = (mb * (1 << 20)) // 4
        try:
            bw, lat = _broadcast_bw(world_size, torch.float32, n_elem)
            out.append(_curve_entry(mb, bw, lat))
        except Exception as exc:  # noqa: BLE001
            out.append(_curve_entry(mb, 0.0, -1.0, str(exc)))
    return out


def main() -> int:
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    backend = "nccl"  # rccl on rocm presents as the nccl backend
    try:
        dist.init_process_group(backend=backend, init_method="env://")
    except Exception as exc:  # noqa: BLE001
        if _is_rank0():
            print(json.dumps({"error": f"init_process_group failed: {exc}"}))
        return 1

    t1_pass = True
    t1_msg = "ok"
    try:
        import torch
        x = torch.ones(1, device=f"cuda:{local_rank}")
        dist.all_reduce(x)
        torch.cuda.synchronize()
        expected = float(world_size)
        if abs(x.item() - expected) > 1e-3:
            t1_pass = False
            t1_msg = f"sanity allreduce returned {x.item()}, expected {expected}"
    except Exception as exc:  # noqa: BLE001
        t1_pass = False
        t1_msg = f"sanity allreduce failed: {exc}"

    sizes_mb = [int(x) for x in os.environ.get("PILOT_AR_SIZES_MB", "1,16,64,256").split(",")]
    a2a_sizes_mb = [int(x) for x in os.environ.get("PILOT_A2A_SIZES_MB", "1,16,64").split(",")]
    ag_sizes_mb = [int(x) for x in os.environ.get("PILOT_AG_SIZES_MB", "1,16,64,256").split(",")]
    rs_sizes_mb = [int(x) for x in os.environ.get("PILOT_RS_SIZES_MB", "1,16,64,256").split(",")]
    bcast_sizes_mb = [int(x) for x in os.environ.get("PILOT_BCAST_SIZES_MB", "1,16,64,256").split(",")]

    if t1_pass:
        ar = _ar_curve(sizes_mb, world_size)
        ag = _ag_curve(ag_sizes_mb, world_size)
        rs = _rs_curve(rs_sizes_mb, world_size)
        bcast = _bcast_curve(bcast_sizes_mb, world_size)
        a2a = _a2a_curve(a2a_sizes_mb, world_size)
    else:
        ar, ag, rs, bcast, a2a = [], [], [], [], []

    intra_bw = 0.0
    for entry in ar:
        if entry.get("size_mb") == 256:
            intra_bw = entry.get("bw_gbs", 0.0)
            break
    if not intra_bw and ar:
        intra_bw = max(e.get("bw_gbs", 0.0) for e in ar)

    if _is_rank0():
        payload: dict[str, Any] = {
            "world_size": world_size,
            "device_count": torch.cuda.device_count(),
            "intra_node_bw_gbs": round(intra_bw, 2),
            "rccl_baseline": {
                "allreduce": ar,
                "allgather": ag,
                "reduce_scatter": rs,
                "broadcast": bcast,
                "alltoall": a2a,
            },
            "t1_connectivity": {"pass": t1_pass, "msg": t1_msg},
        }
        print(json.dumps(payload))

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
