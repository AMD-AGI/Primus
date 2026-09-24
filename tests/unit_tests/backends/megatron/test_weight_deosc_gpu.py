# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""GPU bitwise checks for the batched MXFP4 de-osc QDQ.

Grouped expert weights go through one native 3D Primus-Turbo quantize instead
of one 2D quantize per expert, and DP shard boundary fragments are padded to
64-row tiles for that 3D kernel. Both must reproduce the per-expert 2D result
bit for bit at every scale rounding mode, or the snap targets would move.
"""

import pytest
import torch

from tests.unit_tests.backends.megatron.conftest import requires_mxfp4
from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.optimizer import weight_deosc  # noqa: E402  isort:skip

_deps_ok, _deps_reason = weight_deosc.deosc_dependencies_available()
pytestmark = [
    requires_mxfp4,
    pytest.mark.skipif(not _deps_ok, reason=_deps_reason or "MXFP4 QDQ unavailable"),
]

BLOCK = weight_deosc.MXFP4_SCALING_BLOCK_SIZE


@pytest.fixture(params=[0, 2], ids=["round0", "round2"])
def scale_rounding_mode(request, monkeypatch):
    monkeypatch.setattr(weight_deosc, "_forward_scale_rounding_mode", lambda: request.param)
    return request.param


def _weights(shape, seed=0):
    """BF16 weights whose 32x32 blocks span several binades, plus zero blocks."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32)
    rows, cols = shape[-2:]
    block_scale = torch.exp2(
        torch.randint(-12, 4, (rows // BLOCK, cols // BLOCK), generator=gen, device="cuda").float()
    )
    w = w * block_scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    w.view(-1, rows, cols)[:, :BLOCK, :BLOCK] = 0
    return w.to(torch.bfloat16)


def _per_expert_qdq(w3d):
    return torch.stack([weight_deosc.qdq_mxfp4(w3d[g]) for g in range(w3d.shape[0])], dim=0)


def _reference_local_shard(shard, shape, start, end, model_dtype):
    """The per-matrix loop the batched shard QDQ replaced, on the 2D path."""
    local_model = shard.reshape(-1).to(dtype=model_dtype)
    q_local = torch.empty_like(local_model)
    rows, cols = shape[-2:]
    matrix_numel = rows * cols
    for matrix_idx in range(start // matrix_numel, (end - 1) // matrix_numel + 1):
        matrix_base = matrix_idx * matrix_numel
        local_begin = max(start, matrix_base)
        local_end = min(end, matrix_base + matrix_numel)
        begin_in_matrix = local_begin - matrix_base
        end_in_matrix = local_end - matrix_base
        tile_row_begin = (begin_in_matrix // cols // BLOCK) * BLOCK
        tile_row_end = ((end_in_matrix - 1) // cols // BLOCK + 1) * BLOCK
        tile = torch.zeros((tile_row_end - tile_row_begin, cols), device=shard.device, dtype=model_dtype)
        tile_begin = begin_in_matrix - tile_row_begin * cols
        tile_end = end_in_matrix - tile_row_begin * cols
        tile.reshape(-1)[tile_begin:tile_end].copy_(local_model[local_begin - start : local_end - start])
        q_tile = weight_deosc.qdq_mxfp4(tile)
        q_local[local_begin - start : local_end - start].copy_(q_tile.reshape(-1)[tile_begin:tile_end])
    return q_local


@pytest.mark.parametrize(
    "shape",
    [(4, 128, 256), (2, 2880, 2880), (2, 5760, 2880)],
    ids=["small", "gptoss_fc2", "gptoss_fc1"],
)
def test_batched_3d_qdq_matches_per_expert_2d(shape, scale_rounding_mode):
    w = _weights(shape)
    batched = weight_deosc.qdq_mxfp4(w)
    assert batched.dtype == w.dtype and batched.shape == w.shape
    assert torch.equal(batched, _per_expert_qdq(w))


_SHAPE = (4, 192, 256)
_MATRIX = _SHAPE[1] * _SHAPE[2]


@pytest.mark.parametrize(
    "start,end",
    [
        (0, 4 * _MATRIX),
        (_MATRIX, 3 * _MATRIX),
        (40 * 256 + 3, 3 * _MATRIX + 70 * 256 + 5),
        (40 * 256 + 2, 3 * _MATRIX + 70 * 256 + 6),
        (_MATRIX + 33 * 256 + 17, 2 * _MATRIX + 100 * 256 + 9),
        (2 * _MATRIX + 5, 2 * _MATRIX + 31 * 256 + 200),
        (_MATRIX - 256 * 7 - 11, _MATRIX + 256 * 2 + 1),
    ],
    ids=[
        "all_experts",
        "aligned_experts",
        "both_ends_split_odd_offset",
        "both_ends_split_even_unaligned",
        "adjacent_split",
        "one_tile_row",
        "straddle",
    ],
)
def test_local_shard_qdq_matches_per_matrix_loop(start, end, scale_rounding_mode):
    full = _weights(_SHAPE, seed=1).float()
    shard = full.reshape(-1)[start:end].clone()
    q_local, model = weight_deosc.qdq_mxfp4_local_shard(
        shard, _SHAPE, start, end, torch.bfloat16, return_model=True
    )
    assert torch.equal(model, shard.to(torch.bfloat16))
    assert torch.equal(q_local, _reference_local_shard(shard, _SHAPE, start, end, torch.bfloat16))
