###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""NcclTransport on 2 GPUs: weights go only into the slots whose expert changed, gradients come back into the
owners and leave the returned slots cleared."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from primus.backends.megatron.core.transformer.moe.replica import (
    LayerwiseStorage,
    NcclTransport,
)

WORLD, L, B, SHAPE = 2, 2, 2, (3, 5)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD, reason=f"needs {WORLD} GPUs"
)


def _worker(rank: int, init_file: str):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
    try:
        dev = torch.device("cuda", rank)
        transport = NcclTransport(dist.group.WORLD, L)
        storage = LayerwiseStorage(transport, B, SHAPE, torch.float32, torch.float32, dev)
        # expert e (owned by rank e // L) is filled with the value e
        own = torch.stack([torch.full(SHAPE, float(rank * L + i), device=dev) for i in range(L)])
        empty = torch.full((WORLD, B), -1, dtype=torch.int64)

        table = torch.tensor([[2, -1], [0, 1]])
        transport.push_weights(own, storage, empty, table).wait()
        want = {0: [2.0, 0.0], 1: [0.0, 1.0]}[rank]
        assert [float(storage.w[c, 0, 0]) for c in range(B)] == want

        # only rank 0's empty second slot gets an expert (3); no other slot may be written
        storage.w.fill_(-7.0)
        new = torch.tensor([[2, 3], [0, 1]])
        transport.push_weights(own, storage, table, new).wait()
        want = {0: [-7.0, 3.0], 1: [-7.0, -7.0]}[rank]
        assert [float(storage.w[c, 0, 0]) for c in range(B)] == want

        # slot c of rank d returns (d + 1) * (c + 1) to the owner of its expert
        for c in range(B):
            storage.main_grad[c] = float((rank + 1) * (c + 1))
        owner = torch.ones(L, *SHAPE, device=dev)
        transport.return_grads(storage, new, lambda: owner).wait()
        # rank 0 owns experts 0 and 1, held by rank 1's slots 0 and 1; rank 1 owns 2 and 3, held by rank 0's slots
        want = {0: [1.0 + 2.0, 1.0 + 4.0], 1: [1.0 + 1.0, 1.0 + 2.0]}[rank]
        assert [float(owner[i, 0, 0]) for i in range(L)] == want
        assert not bool(storage.main_grad.any()), "returned slots are cleared"
    finally:
        dist.destroy_process_group()


def test_nccl_transport(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "pg_init"),), nprocs=WORLD, join=True)
