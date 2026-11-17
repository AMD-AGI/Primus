###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for ``training/grad_clip.py``.

WHAT THESE ARE DEFENDING:
  The fused gradient clip replaces ``torch.nn.utils.clip_grads_with_norm_``; it has to scale the
  gradients bit for bit as torch does, clipped or not, including FSDP2's DTensor gradients, whose
  local shards it scales directly.
"""

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def fake_four_ranks():
    """An in-process fake process group of four ranks: DTensors without communication."""
    import torch.distributed as dist

    fake_pg = pytest.importorskip("torch.testing._internal.distributed.fake_pg")
    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    dist.init_process_group("fake", store=fake_pg.FakeStore(), rank=0, world_size=4)
    yield
    dist.destroy_process_group()


DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")),
]
# Seed 0 alone happens to give the same result with the coefficient in FP32 and in BF16.
SEEDS = [0, 1, 2]


def _torch_clip():
    """torch's own implementation, whatever ``torch.nn.utils.clip_grads_with_norm_`` is bound to now."""
    from torch.nn.utils import clip_grad

    return getattr(clip_grad, "clip_grads_with_norm_", None) or clip_grad._clip_grads_with_norm_


class TestFastGradClip:
    """The fused clip against torch's own ``clip_grads_with_norm_``, on BF16 gradients."""

    @staticmethod
    def _params(seed, device):
        gen = torch.Generator().manual_seed(seed)
        params = []
        for n in (7, 1000, 3, 4096):
            p = torch.nn.Parameter(torch.zeros(n, dtype=torch.bfloat16, device=device))
            p.grad = (torch.randn(n, generator=gen) * 3).to(device=device, dtype=torch.bfloat16)
            params.append(p)
        return params

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("max_norm", [0.5, 1e6])  # clipped, and a coefficient clamped to 1
    def test_bitwise_equal_to_torch(self, device, seed, max_norm):
        from primus.backends.automodel.training import grad_clip

        ours, ref = self._params(seed, device), self._params(seed, device)
        total_norm = torch.linalg.vector_norm(torch.stack([p.grad.double().norm() for p in ref]))
        grad_clip.clip_grads_with_norm_(ours, max_norm, total_norm)
        _torch_clip()(ref, max_norm, total_norm, foreach=False)
        for a, b in zip(ours, ref):
            assert torch.equal(a.grad, b.grad)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("placement", ["shard", "replicate", "partial"])
    def test_dtensor_local_shards_bitwise_equal_to_torch(self, fake_four_ranks, device, seed, placement):
        """FSDP2's gradients are DTensors: each local shard is scaled as torch scales a plain tensor."""
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

        from primus.backends.automodel.training import grad_clip

        mesh = init_device_mesh(device, (4,))
        place = {"shard": Shard(0), "replicate": Replicate(), "partial": Partial()}[placement]

        ref = self._params(seed, device)
        ours = []
        for p in ref:
            param = torch.nn.Parameter(
                DTensor.from_local(torch.zeros_like(p), mesh, [place], run_check=False)
            )
            param.grad = DTensor.from_local(p.grad.clone(), mesh, [place], run_check=False)
            ours.append(param)
        total_norm = torch.linalg.vector_norm(torch.stack([p.grad.double().norm() for p in ref]))
        grad_clip.clip_grads_with_norm_(ours, 0.5, total_norm)
        _torch_clip()(ref, 0.5, total_norm, foreach=False)
        for a, b in zip(ours, ref):
            assert torch.equal(a.grad.to_local(), b.grad)

    def test_no_gradients_is_a_no_op(self):
        from primus.backends.automodel.training import grad_clip

        p = torch.nn.Parameter(torch.ones(3))
        grad_clip.clip_grads_with_norm_([p], 1.0, torch.tensor(2.0, dtype=torch.float64))
        assert p.grad is None
