###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tier 2 -- node-local RCCL all-reduce (optional).

Tier 2 also includes a node-local RCCL all-reduce as a steady-state
intra-node bandwidth check. We use ``torch.multiprocessing.spawn`` to
launch one worker per local GPU on a process group bound to
``tcp://127.0.0.1:<free port>``. No cross-node communication.
"""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Any, Dict, List


def _rccl_worker(
    local_rank: int,
    world_size: int,
    port: int,
    size_mb: int,
    out_path: str,
) -> None:
    """Subprocess body for ``torch.multiprocessing.spawn``.

    Runs ``warmup`` warmup + ``iters`` timed all-reduces of ``size_mb`` MB on a
    local-only NCCL/RCCL process group bound to ``tcp://127.0.0.1:port``.
    Local rank 0 writes the resulting GB/s to ``out_path``.

    Iteration counts are intentionally aligned with the preflight `--quick`
    preset (`intra_node_comm.py` with WARMUP=5, ITERATION=20) so smoke and
    preflight report comparable steady-state bandwidth.
    """
    import torch  # type: ignore
    import torch.distributed as dist  # type: ignore

    warmup = 5
    iters = 20

    try:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            world_size=world_size,
            rank=local_rank,
        )
        nbytes = size_mb * 1024 * 1024
        n_elem = nbytes // 2  # bf16
        t = torch.ones(n_elem, dtype=torch.bfloat16, device=f"cuda:{local_rank}")

        for _ in range(warmup):
            dist.all_reduce(t)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            dist.all_reduce(t)
        end.record()
        end.synchronize()

        elapsed_s = start.elapsed_time(end) / 1000.0 / iters
        # NCCL all-reduce effective bandwidth: 2*S*(P-1)/P bytes per rank.
        comm_bytes = 2.0 * nbytes * (world_size - 1) / world_size
        gbs = comm_bytes / elapsed_s / 1e9

        if local_rank == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"status": "PASS", "gbs": round(gbs, 1)}, f)
        dist.barrier()
        dist.destroy_process_group()
    except Exception as e:
        if local_rank == 0:
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"status": "FAIL", "error": str(e)}, f)
            except Exception:
                pass


def _run_local_rccl(*, local_world_size: int, size_mb: int, timeout_sec: int) -> Dict[str, Any]:
    """Spawn local-only RCCL workers to measure intra-node all-reduce bandwidth.

    Returns ``{"status": "PASS"|"FAIL"|"TIMEOUT", ...}``.
    """
    import tempfile

    if local_world_size <= 1:
        return {"status": "PASS", "gbs": None, "skipped": "local_world_size<=1"}

    try:
        import torch  # type: ignore
        import torch.multiprocessing as mp  # type: ignore
    except Exception as e:
        return {"status": "FAIL", "error": f"torch import failed: {e}"}

    # Pick a free local TCP port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    with tempfile.NamedTemporaryFile(prefix="node_smoke_rccl_", suffix=".json", delete=False) as tf:
        out_path = tf.name

    ctx = mp.get_context("spawn")
    procs: List[Any] = []
    try:
        for r in range(local_world_size):
            p = ctx.Process(
                target=_rccl_worker,
                args=(r, local_world_size, port, size_mb, out_path),
            )
            p.start()
            procs.append(p)

        deadline = time.time() + timeout_sec
        for p in procs:
            remaining = max(0.0, deadline - time.time())
            p.join(timeout=remaining)
            if p.is_alive():
                # One worker stuck -> kill all and report TIMEOUT.
                for q in procs:
                    if q.is_alive():
                        q.terminate()
                for q in procs:
                    q.join(timeout=5)
                    if q.is_alive():
                        q.kill()
                return {
                    "status": "TIMEOUT",
                    "error": f"local RCCL all-reduce did not finish in {timeout_sec}s",
                }

        if not os.path.exists(out_path):
            return {"status": "FAIL", "error": "no result file produced"}
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    finally:
        try:
            os.unlink(out_path)
        except Exception:
            pass
