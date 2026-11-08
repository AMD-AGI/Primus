###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Over-limit tensors reach multi-tensor kernels as views no larger than the limit."""

import pytest
import torch

from primus.backends.megatron.patches.te_patches import (
    multi_tensor_large_tensor_patches as mod,
)


class _RecordingOp:
    def __init__(self):
        self.calls = []

    def __call__(self, chunk_size, noop_flag, tensor_lists, *args):
        self.calls.append(tensor_lists)
        # Behave like multi_tensor_l2norm(per_tensor=False): total norm over all tensors.
        flat = torch.cat([t.reshape(-1).float() for t in tensor_lists[0]])
        return flat.norm(), None


@pytest.fixture(autouse=True)
def small_limit(monkeypatch):
    monkeypatch.setattr(mod, "_MAX_NUMEL", 10)


def test_small_tensors_pass_through_unchanged():
    op = _RecordingOp()
    safe = mod.make_large_tensor_safe(op, is_l2norm=True)
    a, b = torch.ones(10), torch.ones(3)
    safe(2048, None, [[a, b]], False)
    assert op.calls[0][0][0] is a and op.calls[0][0][1] is b


def test_large_tensor_is_split_and_norm_unchanged():
    op = _RecordingOp()
    safe = mod.make_large_tensor_safe(op, is_l2norm=True)
    big, small = torch.arange(25, dtype=torch.float32), torch.ones(4)
    norm, _ = safe(2048, None, [[big, small]], False)
    assert [t.numel() for t in op.calls[0][0]] == [10, 10, 5, 4]
    assert all(t.data_ptr() >= big.data_ptr() for t in op.calls[0][0][:3])
    assert torch.allclose(norm, torch.cat([big, small]).norm())


def test_scale_lists_split_in_step_and_write_through():
    def scale(chunk_size, noop_flag, tensor_lists, s):
        for src, dst in zip(*tensor_lists):
            assert src.numel() == dst.numel() <= 10
            dst.copy_(src * s)

    safe = mod.make_large_tensor_safe(scale)
    src, dst = torch.arange(23, dtype=torch.float32), torch.zeros(23)
    safe(2048, None, [[src], [dst]], 2.0)
    assert torch.equal(dst, src * 2)


def test_per_tensor_norm_of_large_tensor_raises():
    safe = mod.make_large_tensor_safe(_RecordingOp(), is_l2norm=True)
    with pytest.raises(RuntimeError, match="per-tensor"):
        safe(2048, None, [[torch.ones(11)]], True)


def test_wrapping_is_idempotent():
    op = _RecordingOp()
    safe = mod.make_large_tensor_safe(op)
    assert mod.make_large_tensor_safe(safe) is safe
    assert mod.make_large_tensor_safe(None) is None
