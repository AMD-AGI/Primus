###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.patches.turbo import flydsl_grouped_gemm_patches as patch


@pytest.mark.parametrize(
    ("cap", "num_local_experts", "gfx1250", "expected"),
    [
        (64, 128, True, 128),
        (64, 160, True, 160),
        (64, 161, True, None),
        (64, 64, True, None),
        (64, 32, True, None),
        (64, 128, False, None),
        (128, 128, True, None),
    ],
)
def test_raised_max_groups(cap, num_local_experts, gfx1250, expected):
    assert patch.raised_max_groups(cap, num_local_experts, gfx1250) == expected


def test_local_experts():
    assert patch.local_experts(SimpleNamespace(num_experts=128, expert_model_parallel_size=4)) == 32
    assert patch.local_experts(SimpleNamespace(num_experts=128, expert_model_parallel_size=1)) == 128
    assert patch.local_experts(SimpleNamespace(num_experts=None, expert_model_parallel_size=1)) == 0


def test_flydsl_grouped_gemm_over_64_experts(monkeypatch):
    turbo = pytest.importorskip("primus_turbo.pytorch")
    from primus_turbo.pytorch.core.utils import is_gfx1250
    from primus_turbo.pytorch.kernels.grouped_gemm import grouped_gemm_impl

    if not torch.cuda.is_available() or not is_gfx1250():
        pytest.skip("needs gfx1250")
    G, rows, K, N = 128, 64, 1024, 1024
    backend = grouped_gemm_impl.GroupedGEMMVariableKFlyDSLBackend
    monkeypatch.setattr(backend, "MAX_G", patch.raised_max_groups(backend.MAX_G, G, True))
    monkeypatch.setenv("PRIMUS_TURBO_GROUPED_GEMM_BACKEND", "bf16:flydsl")

    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(G * rows, K, device="cuda", dtype=torch.bfloat16, generator=gen)
    w = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16, generator=gen) / K**0.5
    dy = torch.randn(G * rows, N, device="cuda", dtype=torch.bfloat16, generator=gen)
    group_lens = torch.full((G,), rows, dtype=torch.int64, device="cuda")
    xg, wg = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
    turbo.ops.grouped_gemm(xg, wg, group_lens, trans_b=True).backward(dy)

    xs, dys = x.view(G, rows, K).float(), dy.view(G, rows, N).float()
    ref_dw = torch.bmm(dys.transpose(1, 2), xs)
    ref_dx = torch.bmm(dys, w.float()).view(G * rows, K)
    torch.testing.assert_close(wg.grad.float(), ref_dw, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(xg.grad.float(), ref_dx, rtol=2e-2, atol=2e-2)
