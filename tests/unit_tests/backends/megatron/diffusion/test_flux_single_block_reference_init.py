# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for Flux._init_single_block_linears_as_reference (single_block_reference_init).

The reference SingleStreamBlock applies xavier_uniform_ to its fused linear1 [3H + F, H] and
linear2 [H, H + F]; Primus holds those as four bf16 pieces. Every piece must follow U(-b, b)
with the fused bound b: right bound, right spread and no bias. The bias test is the one that
matters: Tensor.uniform_ drawn directly into a bf16 tensor has a mean of about -0.002 b.
"""

import math
from types import SimpleNamespace

import torch

from primus.backends.megatron.core.models.diffusion.flux.model import Flux

H, F = 1024, 4096  # the Flux 3072 / 12288 ratio, small enough for a CPU test


def _single_block(dtype):
    def lin(out_f, in_f):
        return SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(out_f, in_f, dtype=dtype)))

    return SimpleNamespace(
        self_attention=SimpleNamespace(linear_qkv=lin(3 * H, H), linear_proj=lin(H, H)),
        mlp=SimpleNamespace(linear_fc1=lin(F, H), linear_fc2=lin(H, F)),
    )


def _init(dtype, n_single=8):
    torch.manual_seed(0)
    blocks = [_single_block(dtype) for _ in range(n_single)]
    fake = SimpleNamespace(
        config=SimpleNamespace(num_joint_layers=1),
        transformer=SimpleNamespace(layers=[None] + blocks),
    )
    Flux._init_single_block_linears_as_reference(fake)
    return blocks


def _fused(blocks):
    linear1 = torch.cat(
        [torch.cat([b.self_attention.linear_qkv.weight, b.mlp.linear_fc1.weight]).float().flatten() for b in blocks]
    )
    linear2 = torch.cat(
        [torch.cat([b.self_attention.linear_proj.weight, b.mlp.linear_fc2.weight], 1).float().flatten() for b in blocks]
    )
    return linear1, linear2


def test_fused_xavier_bounds_spread_and_no_bias():
    blocks = _init(torch.bfloat16)
    b1 = math.sqrt(6.0 / (H + 3 * H + F))  # linear1: fan_in H, fan_out 3H + F
    b2 = math.sqrt(6.0 / (H + F + H))  # linear2: fan_in H + F, fan_out H
    for w, b in zip(_fused(blocks), (b1, b2)):
        std = b / math.sqrt(3.0)
        # bf16 rounding of the fp32 draw can land one ulp beyond b
        assert w.abs().max().item() <= b * (1 + 2**-8)
        assert abs(w.std().item() / std - 1) < 2e-3
        # unbiased: within 4 standard errors (a bf16-native draw sits 11-22 out here)
        assert abs(w.mean().item()) < 4 * std / math.sqrt(w.numel())


def test_pieces_keep_their_dtype():
    for blk in _init(torch.bfloat16):
        for w in (
            blk.self_attention.linear_qkv.weight,
            blk.self_attention.linear_proj.weight,
            blk.mlp.linear_fc1.weight,
            blk.mlp.linear_fc2.weight,
        ):
            assert w.dtype == torch.bfloat16
