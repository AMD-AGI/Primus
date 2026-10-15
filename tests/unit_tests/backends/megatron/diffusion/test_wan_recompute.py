# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Activation recompute in the WAN backbone changes memory, not math.

The backbone's ``DiffusionTransformerBlock`` recomputes layers through Megatron's
``recompute_method`` / ``recompute_num_layers``, and its checkpointed path
returns only hidden states where the plain path also returns the text context.
This runs one training step with and without recompute on the same weights and
requires the same output and gradients, and counts the checkpointed calls so a
config that silently skips recompute fails too.

GPU-only, and needs the Megatron / TE / Primus-Turbo stack of the ROCm training
container.
"""

import argparse

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from megatron.core import tensor_parallel  # noqa: E402
from megatron.training.global_vars import set_args  # noqa: E402

from primus.backends.megatron.core.models.diffusion.wan.config import (  # noqa: E402
    WanConfig,
)
from primus.backends.megatron.core.models.diffusion.wan.model import (  # noqa: E402
    WanTransformer3D,
)

TINY = dict(
    hidden_size=256,
    num_attention_heads=2,
    num_dit_layers=4,
    ffn_hidden_size=512,
    text_embed_dim=64,
    adaln_zero_init=False,
)
LATENTS = (2, 16, 2, 16, 16)
TEXT = (2, 24, TINY["text_embed_dim"])

# (recompute_method, recompute_num_layers, checkpointed calls for 4 layers)
RECOMPUTE = [
    pytest.param("uniform", 1, 4, id="uniform_1"),
    pytest.param("block", 1, 1, id="block_1"),
]


@pytest.fixture(autouse=True)
def _parallel(init_parallel_state):
    set_args(argparse.Namespace(enable_turbo_attention_float8=False))


def _train_step(model, inputs):
    latents, timestep, text = inputs
    out = model(latents, timestep, text)
    out.float().square().mean().backward()
    return out.detach(), {n: p.grad.detach().float() for n, p in model.named_parameters()}


@pytest.mark.parametrize("transformer_impl", ["local", "transformer_engine"])
@pytest.mark.parametrize("method,num_layers,expected_calls", RECOMPUTE)
def test_recompute_matches_plain_forward(transformer_impl, method, num_layers, expected_calls, monkeypatch):
    torch.manual_seed(0)
    plain_config = WanConfig.wan2_1_t2v_1_3b(**TINY, transformer_impl=transformer_impl)
    recompute_config = WanConfig.wan2_1_t2v_1_3b(
        **TINY,
        transformer_impl=transformer_impl,
        recompute_granularity="full",
        recompute_method=method,
        recompute_num_layers=num_layers,
    )
    plain = WanTransformer3D(plain_config).to(device="cuda", dtype=torch.bfloat16).train()
    recompute = WanTransformer3D(recompute_config).to(device="cuda", dtype=torch.bfloat16).train()
    recompute.load_state_dict(plain.state_dict())

    inputs = (
        torch.randn(LATENTS, device="cuda", dtype=torch.bfloat16),
        torch.tensor([150.0, 850.0], device="cuda"),
        torch.randn(TEXT, device="cuda", dtype=torch.bfloat16),
    )

    calls = []
    checkpoint = tensor_parallel.checkpoint

    def counting_checkpoint(*args, **kwargs):
        calls.append(1)
        return checkpoint(*args, **kwargs)

    monkeypatch.setattr(tensor_parallel, "checkpoint", counting_checkpoint)

    expected_out, expected_grads = _train_step(plain, inputs)
    assert not calls
    out, grads = _train_step(recompute, inputs)
    assert len(calls) == expected_calls

    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
    assert grads.keys() == expected_grads.keys()
    for name, grad in grads.items():
        reference = expected_grads[name]
        error = ((grad - reference).norm() / reference.norm().clamp_min(1e-12)).item()
        assert error < 1e-2, f"{name}: relative gradient error {error:.2e}"
