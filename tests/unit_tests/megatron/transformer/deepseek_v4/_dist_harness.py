###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tiny multi-rank harness for the context-parallel collectives.

Runs on **gloo over CPU** on purpose. The properties these tests pin -- who receives
whose rows, where a gradient lands, whether a reduction is equivalent to a
reduce-scatter -- are properties of the exchange pattern, not of the device, so
requiring GPUs would only make them skip on the machines most likely to run them.

Assertions execute inside the child processes; the failure text is shipped back so a
`torch.equal` mismatch on rank 3 shows up as a normal pytest failure rather than a
child process that quietly exited non-zero.
"""

import os
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Port is fixed rather than random so a leaked process is obvious instead of
# intermittently colliding with the next run.
_PORT = os.environ.get("PRIMUS_TEST_DIST_PORT", "29593")


def _child(rank, world, port, fn, args, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world)
        group = dist.new_group(list(range(world)))
        fn(rank, world, group, *args)
        q.put((rank, None))
    except Exception:  # noqa: BLE001
        q.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_dist(fn, world_size, *args, timeout=180):
    """Run ``fn(rank, world, group, *args)`` on ``world_size`` gloo ranks."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_child, args=(r, world_size, _PORT, fn, args, q)) for r in range(world_size)]
    for p in procs:
        p.start()
    try:
        results = [q.get(timeout=timeout) for _ in range(world_size)]
    except Exception as exc:  # noqa: BLE001
        for p in procs:
            p.terminate()
        pytest.fail(f"distributed test did not report back within {timeout}s: {exc}")
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

    failures = [(r, tb) for r, tb in sorted(results) if tb is not None]
    if failures:
        pytest.fail(
            "\n\n".join(f"--- rank {r} ---\n{tb}" for r, tb in failures),
            pytrace=False,
        )


def arange_block(rank, rows, width, dtype=torch.float32):
    """Distinct exact integers per rank: ``[rank*rows*width, ...)`` shaped ``[rows, width]``.

    Integers matter. Every expected value below is then exactly representable, so the
    assertions can be ``torch.equal`` rather than a tolerance -- which is the whole point:
    a boundary exchange that delivers the *wrong rows* and one that delivers the right rows
    with a rounding difference are not the same bug, and a tolerance hides only the second.
    """
    start = rank * rows * width
    return torch.arange(start, start + rows * width, dtype=dtype).reshape(rows, width)
