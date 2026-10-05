# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Flux scales the timestep by 1000 as the MLPerf reference (torchtitan) and NeMo do: t is cast to the
compute dtype and multiplied by 1000 in that dtype, so in bf16 the product is rounded again. The
reference's evaluation timesteps k/8 therefore enter the embedding as 0, 125, 250, 376, 500, 624,
752, 876.
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


def _reference_scaled(t, dtype=torch.bfloat16):
    """The reference: ``timesteps.to(model_dtype)``, then ``t = time_factor * t`` in that dtype."""
    return 1000.0 * t.to(dtype)


def test_eval_timesteps_match_reference():
    t = torch.tensor([k / 8 for k in range(8)], dtype=torch.float32)
    out = _embedded_timesteps(t)
    assert out.dtype == torch.bfloat16
    assert out.float().tolist() == [0.0, 125.0, 250.0, 376.0, 500.0, 624.0, 752.0, 876.0]
    assert torch.equal(out, _reference_scaled(t))


def test_training_timesteps_match_reference():
    t = torch.rand(4096)
    out = _embedded_timesteps(t)
    assert torch.equal(out, _reference_scaled(t))
