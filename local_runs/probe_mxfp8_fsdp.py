#!/usr/bin/env python3
"""Two-rank composable FSDP2 smoke test for TorchAO MXFP8Linear."""

import os

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.device_mesh import init_device_mesh
from torchao.prototype.moe_training.mxfp8_linear import MXFP8Linear


def main() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = torch.nn.Sequential(
        MXFP8Linear(3072, 3072, bias=True, device=device),
        torch.nn.GELU(),
        MXFP8Linear(3072, 3072, bias=True, device=device),
    )
    fully_shard(
        model,
        mesh=init_device_mesh("cuda", (dist.get_world_size(),)),
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    x = torch.randn(256, 3072, device=device, dtype=torch.bfloat16)
    loss = model(x).float().square().mean()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    print(f"rank={dist.get_rank()} loss={loss.item():.8f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
