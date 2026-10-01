# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Wan2_2 splits a mixed batch between its experts and reassembles it in input order.

The experts are stubbed so the routing and reassembly are checked without
building two 14B backbones: each stub returns its own marker value at the
configured output width, which may differ from the input width.
"""

import torch
from torch import nn

from primus.backends.megatron.core.models.diffusion.wan.model import Wan2_2


class _StubExpert(nn.Module):
    def __init__(self, out_channels: int, marker: float):
        super().__init__()
        self.out_channels = out_channels
        self.marker = marker

    def forward(self, hidden_states, timestep, encoder_hidden_states, **kwargs):
        b, _, t, h, w = hidden_states.shape
        out = hidden_states.new_full((b, self.out_channels, t, h, w), self.marker)
        return out + timestep.view(b, 1, 1, 1, 1)


def _dual_expert(out_channels: int, boundary: float) -> Wan2_2:
    model = Wan2_2.__new__(Wan2_2)
    nn.Module.__init__(model)
    model.transformer = _StubExpert(out_channels, marker=1000.0)
    model.transformer_2 = _StubExpert(out_channels, marker=2000.0)
    model.default_boundary_timestep = boundary
    return model


def _forward(model, timestep, in_channels=4):
    latents = torch.zeros(len(timestep), in_channels, 2, 3, 3)
    text = torch.zeros(len(timestep), 5, 8)
    return Wan2_2.forward(model, latents, timestep, text)


def test_mixed_batch_is_reassembled_in_input_order():
    timestep = torch.tensor([900.0, 100.0, 950.0, 50.0])

    out = _forward(_dual_expert(out_channels=4, boundary=500.0), timestep)

    expected = torch.tensor([1900.0, 2100.0, 1950.0, 2050.0])
    assert torch.equal(out[:, 0, 0, 0, 0], expected)


def test_output_takes_the_experts_channel_width():
    timestep = torch.tensor([900.0, 100.0])

    out = _forward(_dual_expert(out_channels=6, boundary=500.0), timestep, in_channels=4)

    assert out.shape == (2, 6, 2, 3, 3)


def test_a_batch_on_one_side_runs_one_expert():
    timestep = torch.tensor([900.0, 800.0])

    out = _forward(_dual_expert(out_channels=4, boundary=500.0), timestep)

    assert torch.equal(out[:, 0, 0, 0, 0], torch.tensor([1900.0, 1800.0]))
