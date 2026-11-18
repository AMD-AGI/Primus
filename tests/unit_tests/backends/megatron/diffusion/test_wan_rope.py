# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
WAN self-attention rotates q/k exactly as an interleaved pairwise rotation.

The diffusers parity test cannot see a wrong rotation: with random weights and
a small grid, attention is close to uniform and nearly position-blind, so a
wrong RoPE stays inside its bf16 bound. This test instead records every rotary
call self-attention makes, on the real PusaV1 latent grid, and compares each
output with the exact rotation of the same input by the same angles.

GPU-only, and needs the Megatron / TE / Primus-Turbo stack of the ROCm training
container.
"""

import argparse

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from megatron.training.global_vars import set_args  # noqa: E402

import primus.backends.megatron.core.models.diffusion.wan.attention as wan_attention  # noqa: E402
from primus.backends.megatron.core.models.diffusion.wan.config import (  # noqa: E402
    WanConfig,
)
from primus.backends.megatron.core.models.diffusion.wan.model import (  # noqa: E402
    WanTransformer3D,
)

TINY = dict(
    hidden_size=256,
    num_attention_heads=2,
    num_dit_layers=1,
    ffn_hidden_size=512,
    text_embed_dim=64,
)
# PusaV1 at 480x832, 25 frames: [16, 7, 60, 104] latents, 7 x 30 x 52 = 10,920 tokens.
LATENTS = (2, 16, 7, 60, 104)
TEXT = (2, 24, TINY["text_embed_dim"])

PATHS = [
    pytest.param("transformer_engine", False, id="te_thd"),
    pytest.param("local", False, id="local_bshd"),
    pytest.param("local", True, id="local_thd"),
]


@pytest.fixture(autouse=True)
def _parallel(init_parallel_state):
    set_args(argparse.Namespace(enable_turbo_attention_float8=False))


def _exact_interleaved_rotation(t, freqs, cu_seqlens):
    """Rotate pairs (x[2i], x[2i+1]) by angle freqs[..., 2i], in float64.

    ``t`` is ``[S, B, H, D]`` with ``freqs`` ``[S, 1, 1, D]``, or packed
    ``[T, H, D]`` with ``cu_seqlens``, where each sequence restarts at
    position 0.
    """
    if cu_seqlens is None:
        angles = freqs
    else:
        bounds = cu_seqlens.tolist()
        angles = torch.cat([freqs[: end - start] for start, end in zip(bounds[:-1], bounds[1:])]).squeeze(1)
    x = t.double()
    cos, sin = angles.double().cos(), angles.double().sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    partner = torch.stack((-odd, even), dim=-1).flatten(-2)
    return x * cos + partner * sin


@pytest.mark.parametrize("transformer_impl,local_thd", PATHS)
def test_self_attention_rotation_is_exact(transformer_impl, local_thd, monkeypatch):
    torch.manual_seed(0)
    config = WanConfig.wan2_1_t2v_1_3b(
        **TINY, transformer_impl=transformer_impl, local_thd_attention=local_thd
    )
    model = WanTransformer3D(config).to(device="cuda", dtype=torch.bfloat16).eval()

    calls = []
    original = wan_attention.apply_rotary_pos_emb

    def recording_rotary(t, freqs, *args, **kwargs):
        out = original(t, freqs, *args, **kwargs)
        calls.append(
            (t.detach().clone(), freqs.detach().clone(), kwargs.get("cu_seqlens"), out.detach().clone())
        )
        return out

    monkeypatch.setattr(wan_attention, "apply_rotary_pos_emb", recording_rotary)

    latents = torch.randn(LATENTS, device="cuda", dtype=torch.bfloat16)
    text = torch.randn(TEXT, device="cuda", dtype=torch.bfloat16)
    timestep = torch.tensor([150.0, 850.0], device="cuda")
    with torch.no_grad():
        model(latents, timestep, text)

    assert len(calls) == 2, f"expected one rotary call each for q and k, got {len(calls)}"
    for t, freqs, cu_seqlens, out in calls:
        expected = _exact_interleaved_rotation(t, freqs, cu_seqlens)
        error = ((out.double() - expected).norm() / expected.norm()).item()
        assert error < 1e-2, f"{transformer_impl} (thd={local_thd}): rotary error {error:.2e} vs exact"


def test_rope_angles_stay_fp32_under_a_bf16_cast():
    """Float16Module calls ``.bfloat16()``, which casts floating buffers too.

    In bf16 the angles on the 52-wide axis are up to 0.11 rad off, so the table
    is kept out of the module's buffers.
    """
    from primus.backends.megatron.core.models.diffusion.wan.layers import (
        WanRotaryPosEmbed,
    )

    rope = WanRotaryPosEmbed(128).bfloat16().cuda()
    angles = rope(7, 30, 52, torch.device("cuda"))

    assert angles.dtype == torch.float32
    assert torch.equal(angles, WanRotaryPosEmbed(128)(7, 30, 52, torch.device("cuda")))
