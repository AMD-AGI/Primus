###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

from primus.backends.megatron.patches.te_patches import (
    l2norm_large_tensor_patches as patch,
)


def _fake_l2norm(calls):
    def l2norm(chunk_size, noop_flag, tensor_lists, per_tensor):
        calls.append((tensor_lists, per_tensor))
        flat = torch.cat([t.reshape(-1) for t in tensor_lists[0]])
        return torch.linalg.vector_norm(flat).reshape(1), None

    return l2norm


def test_large_tensor_is_split_into_views(monkeypatch):
    monkeypatch.setattr(patch, "_MAX_NUMEL", 4)
    calls = []
    l2norm = patch.split_large_tensors(_fake_l2norm(calls))
    small = torch.ones(3)
    big = torch.arange(10, dtype=torch.float32).reshape(2, 5)

    norm, _ = l2norm(2048, None, [[small, big]], False)

    (tensors,), per_tensor = calls[0]
    assert [t.numel() for t in tensors] == [3, 4, 4, 2]
    assert tensors[0] is small
    assert [t.data_ptr() for t in tensors[1:]] == [big.data_ptr() + 4 * i * 4 for i in range(3)]
    assert per_tensor is False
    expected = torch.linalg.vector_norm(torch.cat([small, big.reshape(-1)]))
    assert torch.allclose(norm, expected.reshape(1))


def test_small_tensors_and_per_tensor_pass_through(monkeypatch):
    monkeypatch.setattr(patch, "_MAX_NUMEL", 4)
    calls = []
    l2norm = patch.split_large_tensors(_fake_l2norm(calls))
    small = [torch.ones(4), torch.ones(2)]
    big = [torch.ones(3), torch.ones(10)]

    l2norm(2048, None, [small], False)
    l2norm(2048, None, [big], True)

    assert calls[0][0][0] is small
    assert calls[1][0][0] is big
