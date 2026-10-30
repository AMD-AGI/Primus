# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Wan2_2 on one data-parallel rank, with steps that skip an expert.

``Wan2_2`` runs only the experts its micro-batch routes to, so a step whose
samples are all high noise leaves the low-noise expert without gradients. DDP's
``overlap_grad_reduce`` and Megatron-FSDP both assert on such a step, so the
trainer only allows plain DDP without overlap here
(``check_dual_expert_grad_sync``). This runs that configuration through steps
that use both, one, and the other expert.

GPU-only, and needs the Megatron / TE / Primus-Turbo stack of the ROCm training
container.
"""

import argparse

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from megatron.core.distributed import (  # noqa: E402
    DistributedDataParallel,
    DistributedDataParallelConfig,
)
from megatron.training.global_vars import set_args  # noqa: E402

from primus.backends.megatron.core.models.diffusion.wan.config import (  # noqa: E402
    WanConfig,
)
from primus.backends.megatron.core.models.diffusion.wan.model import (  # noqa: E402
    Wan2_2,
)

TINY = dict(
    hidden_size=256,
    num_attention_heads=2,
    num_dit_layers=2,
    ffn_hidden_size=512,
    text_embed_dim=64,
    adaln_zero_init=False,
    bf16=True,
)
LATENTS = (2, 16, 2, 16, 16)
TEXT = (2, 24, TINY["text_embed_dim"])

# The boundary is 0.875 * 1000 = 875 on the conditioning axis.
MIXED = (150.0, 950.0)
HIGH = (900.0, 950.0)
LOW = (150.0, 400.0)
ROUTES = [MIXED, HIGH, LOW, MIXED, LOW, HIGH]


@pytest.fixture(autouse=True)
def _parallel(init_parallel_state):
    set_args(argparse.Namespace(enable_turbo_attention_float8=False))


def _expert_grad_norms(model):
    norms = {"transformer": 0.0, "transformer_2": 0.0}
    for name, param in model.named_parameters():
        expert = "transformer_2" if ".transformer_2." in f".{name}" else "transformer"
        norms[expert] += param.main_grad.float().square().sum().item()
    return norms


def test_steps_that_skip_an_expert_sync_and_leave_it_a_zero_gradient():
    torch.manual_seed(0)
    config = WanConfig.wan2_2_t2v_a14b(**TINY)
    ddp_config = DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False)
    module = Wan2_2(config).to(device="cuda", dtype=torch.bfloat16)
    model = DistributedDataParallel(config=config, ddp_config=ddp_config, module=module).train()

    latents = torch.randn(LATENTS, device="cuda", dtype=torch.bfloat16)
    text = torch.randn(TEXT, device="cuda", dtype=torch.bfloat16)
    for step, route in enumerate(ROUTES):
        model.zero_grad_buffer()
        out = model(latents, torch.tensor(route, device="cuda"), text)
        out.float().square().mean().backward()
        model.finish_grad_sync()

        norms = _expert_grad_norms(model)
        assert (norms["transformer"] > 0) == (max(route) >= 875), (step, norms)
        assert (norms["transformer_2"] > 0) == (min(route) < 875), (step, norms)
