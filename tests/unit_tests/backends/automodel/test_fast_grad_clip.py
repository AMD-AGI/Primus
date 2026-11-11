###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for ``training/grad_clip.py``.

WHAT THESE ARE DEFENDING:
  The fused gradient clip replaces ``torch.nn.utils.clip_grads_with_norm_``; it has to scale the
  gradients bit for bit as torch does, clipped or not.
"""

import pytest

torch = pytest.importorskip("torch")


class TestFastGradClip:
    """The fused clip against torch's own ``clip_grads_with_norm_``, on BF16 gradients."""

    @staticmethod
    def _params(seed):
        gen = torch.Generator().manual_seed(seed)
        params = [torch.nn.Parameter(torch.zeros(n, dtype=torch.bfloat16)) for n in (7, 1000, 3, 4096)]
        for p in params:
            p.grad = (torch.randn(p.shape, generator=gen) * 3).to(torch.bfloat16)
        return params

    @pytest.mark.parametrize("max_norm", [0.5, 1e6])  # clipped, and a coefficient clamped to 1
    def test_bitwise_equal_to_torch(self, max_norm):
        from torch.nn.utils import clip_grad

        from primus.backends.automodel.training import grad_clip

        # torch's own implementation, whatever ``torch.nn.utils.clip_grads_with_norm_`` is bound to now.
        torch_clip = getattr(clip_grad, "clip_grads_with_norm_", None) or clip_grad._clip_grads_with_norm_

        ours, ref = self._params(0), self._params(0)
        total_norm = torch.linalg.vector_norm(torch.stack([p.grad.double().norm() for p in ref]))
        grad_clip.clip_grads_with_norm_(ours, max_norm, total_norm)
        torch_clip(ref, max_norm, total_norm, foreach=False)
        for a, b in zip(ours, ref):
            assert torch.equal(a.grad, b.grad)

    def test_no_gradients_is_a_no_op(self):
        from primus.backends.automodel.training import grad_clip

        p = torch.nn.Parameter(torch.ones(3))
        grad_clip.clip_grads_with_norm_([p], 1.0, torch.tensor(2.0, dtype=torch.float64))
        assert p.grad is None
