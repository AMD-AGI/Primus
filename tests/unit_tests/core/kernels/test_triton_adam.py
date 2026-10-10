###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the backend-neutral Triton Adam kernels.

The single-tensor ``torch.optim.AdamW`` / ``Adam`` for-loop implementation is
the reference; the per-tensor and batched kernels must agree with it and with
each other, independent of grid size, chunk boundaries and table caching.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Triton Adam kernels require a GPU", allow_module_level=True)

from primus.core.kernels import triton_adam as ta

HP = dict(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1)


def _reference(p, g, steps, adam_w_mode):
    cls = torch.optim.AdamW if adam_w_mode else torch.optim.Adam
    ref = torch.nn.Parameter(p.clone())
    opt = cls(
        [ref],
        lr=HP["lr"],
        betas=(HP["beta1"], HP["beta2"]),
        eps=HP["eps"],
        weight_decay=HP["weight_decay"],
        foreach=False,
    )
    for _ in range(steps):
        ref.grad = g.clone()
        opt.step()
    return ref.detach(), opt.state[ref]["exp_avg"], opt.state[ref]["exp_avg_sq"]


def _fresh(p):
    return p.clone(), torch.zeros_like(p), torch.zeros_like(p)


@pytest.mark.parametrize("adam_w_mode", [True, False])
@pytest.mark.parametrize("batched", [False, True])
def test_matches_torch_reference(adam_w_mode, batched):
    gen = torch.Generator(device="cuda").manual_seed(0)
    p0 = torch.randn(70_001, device="cuda", generator=gen)
    g = torch.randn(70_001, device="cuda", generator=gen)
    p, m, v = _fresh(p0)
    for step in range(1, 4):
        if batched:
            ta.triton_multi_tensor_adam_step_([(p, g, m, v)], step=step, adam_w_mode=adam_w_mode, **HP)
        else:
            ta.triton_adam_step_(p, g, m, v, step=step, adam_w_mode=adam_w_mode, **HP)
    ref_p, ref_m, ref_v = _reference(p0, g, 3, adam_w_mode)
    torch.testing.assert_close(p, ref_p, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(m, ref_m, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(v, ref_v, rtol=1e-5, atol=1e-7)


def test_grid_size_does_not_change_result():
    base = torch.randn(3_000_000, device="cuda")
    grad = torch.randn_like(base)
    outs = []
    for grid in (1, 7, 320, 0):
        p, m, v = _fresh(base)
        ta.triton_adam_step_(p, grad, m, v, step=1, grid_size=grid, **HP)
        outs.append((p, m, v))
    for got in outs[1:]:
        for a, b in zip(got, outs[0]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_batched_matches_per_tensor_across_chunk_boundaries():
    chunk = ta._MULTI_TENSOR_CHUNK
    sizes = [1, 2047, 2048, chunk - 1, chunk, chunk + 1, 3 * chunk + 5, 4096]
    gen = torch.Generator(device="cuda").manual_seed(0)
    base = [torch.randn(n, device="cuda", generator=gen) for n in sizes]
    grads = [torch.randn(n, device="cuda", generator=gen) for n in sizes]
    batched = [(b.clone(), g, torch.zeros_like(b), torch.zeros_like(b)) for b, g in zip(base, grads)]
    ta.triton_multi_tensor_adam_step_(batched, step=3, **HP)
    for got, b, g in zip(batched, base, grads):
        ref = (b.clone(), g, torch.zeros_like(b), torch.zeros_like(b))
        ta.triton_adam_step_(*ref, step=3, **HP)
        for i in (0, 2, 3):
            torch.testing.assert_close(got[i], ref[i])


def test_batched_cache_rebuilds_when_addresses_change():
    cache = {}
    first = [(torch.randn(100, device="cuda"),) + tuple(torch.zeros(100, device="cuda") for _ in range(3))]
    ta.triton_multi_tensor_adam_step_(first, cache=cache, step=1, **HP)
    tables = cache["tables"]
    ta.triton_multi_tensor_adam_step_(first, cache=cache, step=2, **HP)
    assert cache["tables"] is tables

    second = [
        (torch.randn(300, device="cuda"), torch.randn(300, device="cuda"))
        + tuple(torch.zeros(300, device="cuda") for _ in range(2))
    ]
    ref = tuple(t.clone() for t in second[0])
    ta.triton_multi_tensor_adam_step_(second, cache=cache, step=1, **HP)
    assert cache["tables"] is not tables
    ta.triton_adam_step_(*ref, step=1, **HP)
    torch.testing.assert_close(second[0][0], ref[0])


def test_bf16_params_with_fp32_states():
    p0 = torch.randn(5000, device="cuda").to(torch.bfloat16)
    g = torch.randn(5000, device="cuda").to(torch.bfloat16)
    p, m, v = p0.clone(), torch.zeros(5000, device="cuda"), torch.zeros(5000, device="cuda")
    ta.triton_multi_tensor_adam_step_([(p, g, m, v)], step=1, **HP)
    ref_p, _, _ = _reference(p0.float(), g.float(), 1, True)
    assert p.dtype == torch.bfloat16
    torch.testing.assert_close(p.float(), ref_p, rtol=1.6e-2, atol=1e-2)
