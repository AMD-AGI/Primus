# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Flux scales the timestep by 1000 as the MLPerf reference does: t at the bf16 compute dtype's
resolution (the reference samples it in that dtype), multiplied in fp32. A bf16 multiply rounds
t * 1000 a second time, e.g. the evaluation timestep 3/8 becomes 376 instead of 375.
"""

from types import SimpleNamespace

import torch

from primus.backends.megatron.core.models.diffusion.flux.model import Flux


def _embedded_timesteps(t):
    seen = {}

    def timestep_embedding(x):
        seen["t"] = x
        return torch.zeros(x.shape[0], 8, dtype=torch.bfloat16)

    def ident(x):
        return x

    fake = SimpleNamespace(
        img_embed=ident,
        txt_embed=ident,
        timestep_embedding=timestep_embedding,
        vector_embedding=lambda y: torch.zeros(y.shape[0], 8, dtype=torch.bfloat16),
        pos_embed=lambda ids: ids,
    )
    img = torch.zeros(4, 2, 8, dtype=torch.bfloat16)  # [S, B, H]: no transpose
    txt = torch.zeros(4, 2, 8, dtype=torch.bfloat16)
    ids = torch.zeros(2, 4, 3)
    Flux._compute_embeddings(fake, img, txt, t, ids, ids, None, torch.zeros(t.shape[0], 8))
    return seen["t"]


def test_eval_timesteps_scale_exactly():
    t = torch.tensor([k / 8 for k in range(8)], dtype=torch.float32)
    out = _embedded_timesteps(t)
    assert out.dtype == torch.float32
    assert out.tolist() == [125.0 * k for k in range(8)]


def test_training_timesteps_keep_bf16_resolution_only():
    t = torch.rand(4096)
    out = _embedded_timesteps(t)
    expected = t.to(torch.bfloat16).float() * 1000.0
    assert torch.equal(out, expected)
