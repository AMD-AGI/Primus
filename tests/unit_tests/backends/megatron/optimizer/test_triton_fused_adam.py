# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for TritonFusedAdam.

TE ``FusedAdam`` is the reference: the Triton path must produce the same
params / exp_avg / exp_avg_sq over several steps, keep TE's state layout
(``step`` in the param group, fp32 states), fall back to TE for unsupported
configurations, and be picked up by Megatron's optimizer factory through the
patch.
"""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("TritonFusedAdam requires a GPU", allow_module_level=True)

from transformer_engine.pytorch.optimizers import FusedAdam

from primus.backends.megatron.core.optimizer import triton_fused_adam as tfa
from primus.backends.megatron.core.optimizer.triton_fused_adam import (
    TritonFusedAdam,
    triton_adam_step_,
    triton_multi_tensor_adam_step_,
)

# Llama-like mix: odd sizes and a norm vector take the batched path,
# (2048, 4096) is above the batching threshold and gets its own launch.
SHAPES = [(4096,), (1000, 37), (3, 5), (2048, 4096), (1,)]
HPARAMS = dict(lr=7.2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


def _make_params(dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, device="cuda", generator=gen).to(dtype)) for s in SHAPES]


def _set_grads(params_a, params_b, step):
    gen = torch.Generator(device="cuda").manual_seed(1000 + step)
    for pa, pb in zip(params_a, params_b):
        g = torch.randn(pa.shape, device="cuda", generator=gen).to(pa.dtype)
        pa.grad = g.clone()
        pb.grad = g.clone()


def _run_pair(dtype, steps=5, **opt_kwargs):
    ref_params = _make_params(dtype)
    tri_params = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]
    ref = FusedAdam(ref_params, **HPARAMS, **opt_kwargs)
    tri = TritonFusedAdam(tri_params, **HPARAMS, **opt_kwargs)
    for step in range(steps):
        _set_grads(ref_params, tri_params, step)
        ref.step()
        tri.step()
    torch.cuda.synchronize()
    return ref, ref_params, tri, tri_params


@pytest.mark.parametrize("adam_w_mode", [True, False])
@pytest.mark.parametrize("bias_correction", [True, False])
def test_fp32_matches_te(adam_w_mode, bias_correction):
    ref, ref_params, tri, tri_params = _run_pair(
        torch.float32, adam_w_mode=adam_w_mode, bias_correction=bias_correction
    )
    for pr, pt in zip(ref_params, tri_params):
        torch.testing.assert_close(pt, pr, rtol=1e-5, atol=1e-6)
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(tri.state[pt][name], ref.state[pr][name], rtol=1e-5, atol=1e-7)
    assert tri.param_groups[0]["step"] == ref.param_groups[0]["step"] == 5


def test_bf16_params_match_te():
    _, ref_params, _, tri_params = _run_pair(torch.bfloat16)
    for pr, pt in zip(ref_params, tri_params):
        assert pt.dtype == torch.bfloat16
        torch.testing.assert_close(pt.float(), pr.float(), rtol=1.6e-2, atol=1e-2)


def test_grid_size_does_not_change_result():
    n = 3_000_000
    base = torch.randn(n, device="cuda")
    grad = torch.randn(n, device="cuda")
    outs = []
    for grid in (1, 7, 320, 0):
        p, m, v = base.clone(), torch.zeros_like(base), torch.zeros_like(base)
        triton_adam_step_(
            p,
            grad,
            m,
            v,
            lr=1e-3,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.1,
            step=1,
            grid_size=grid,
        )
        outs.append((p, m, v))
    for p, m, v in outs[1:]:
        torch.testing.assert_close(p, outs[0][0], rtol=0, atol=0)
        torch.testing.assert_close(m, outs[0][1], rtol=0, atol=0)
        torch.testing.assert_close(v, outs[0][2], rtol=0, atol=0)


def test_multi_tensor_matches_per_tensor_across_chunk_boundaries():
    chunk = tfa._MULTI_TENSOR_CHUNK
    sizes = [1, 2047, 2048, chunk - 1, chunk, chunk + 1, 3 * chunk + 5, 4096]
    gen = torch.Generator(device="cuda").manual_seed(0)
    base = [torch.randn(n, device="cuda", generator=gen) for n in sizes]
    grads = [torch.randn(n, device="cuda", generator=gen) for n in sizes]
    hp = dict(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1, step=3)
    batched = [(b.clone(), g, torch.zeros_like(b), torch.zeros_like(b)) for b, g in zip(base, grads)]
    triton_multi_tensor_adam_step_(batched, **hp)
    for (p, m, v), b, g in zip(((t[0], t[2], t[3]) for t in batched), base, grads):
        ref = (b.clone(), g, torch.zeros_like(b), torch.zeros_like(b))
        triton_adam_step_(*ref, **hp)
        torch.testing.assert_close(p, ref[0])
        torch.testing.assert_close(m, ref[2])
        torch.testing.assert_close(v, ref[3])


def test_multi_tensor_cache_rebuilds_when_addresses_change():
    hp = dict(lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1, step=1)
    cache = {}
    first = [(torch.randn(100, device="cuda"),) + tuple(torch.zeros(100, device="cuda") for _ in range(3))]
    triton_multi_tensor_adam_step_(first, cache=cache, **hp)
    tables = cache["tables"]
    triton_multi_tensor_adam_step_(first, cache=cache, **hp)
    assert cache["tables"] is tables

    second = [
        (torch.randn(300, device="cuda"), torch.randn(300, device="cuda"))
        + tuple(torch.zeros(300, device="cuda") for _ in range(2))
    ]
    ref = tuple(t.clone() for t in second[0])
    triton_multi_tensor_adam_step_(second, cache=cache, **hp)
    assert cache["tables"] is not tables
    triton_adam_step_(*ref, **hp)
    torch.testing.assert_close(second[0][0], ref[0])


def test_many_small_tensors_use_one_launch_and_match_te(monkeypatch):
    calls = []
    monkeypatch.setattr(tfa, "triton_adam_step_", lambda *a, **k: calls.append(1))
    gen = torch.Generator(device="cuda").manual_seed(0)
    shapes = [(64, 128)] * 200 + [(4096,)] * 50
    ref_params = [torch.nn.Parameter(torch.randn(s, device="cuda", generator=gen)) for s in shapes]
    tri_params = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]
    ref = FusedAdam(ref_params, **HPARAMS)
    tri = TritonFusedAdam(tri_params, **HPARAMS)
    for step in range(3):
        _set_grads(ref_params, tri_params, step)
        ref.step()
        tri.step()
    assert calls == []
    for pr, pt in zip(ref_params, tri_params):
        torch.testing.assert_close(pt, pr, rtol=1e-5, atol=1e-6)


def test_misaligned_views_fall_back_to_per_tensor_launch():
    storage = torch.randn(10_001, device="cuda")
    ref_p = torch.nn.Parameter(storage[1:].clone())
    tri_p = storage[1:]
    assert tri_p.data_ptr() % tfa._MULTI_TENSOR_ALIGN_BYTES != 0
    ref = FusedAdam([ref_p], **HPARAMS)
    tri = TritonFusedAdam([tri_p], **HPARAMS)
    grad = torch.randn_like(ref_p)
    ref_p.grad, tri_p.grad = grad.clone(), grad.clone()
    ref.step()
    tri.step()
    torch.testing.assert_close(tri_p, ref_p, rtol=1e-5, atol=1e-6)


def test_params_without_grad_are_skipped():
    params = _make_params(torch.float32)
    before = [p.detach().clone() for p in params]
    opt = TritonFusedAdam(params, **HPARAMS)
    params[0].grad = torch.randn_like(params[0])
    opt.step()
    assert not torch.equal(params[0], before[0])
    for p, b in zip(params[1:], before[1:]):
        assert torch.equal(p, b)


def test_state_dict_roundtrip():
    _, _, tri, tri_params = _run_pair(torch.float32, steps=2)
    sd = tri.state_dict()
    fresh = TritonFusedAdam([torch.nn.Parameter(p.detach().clone()) for p in tri_params], **HPARAMS)
    fresh.load_state_dict(sd)
    assert fresh.param_groups[0]["step"] == 2
    for p_old, p_new in zip(tri_params, fresh.param_groups[0]["params"]):
        torch.testing.assert_close(fresh.state[p_new]["exp_avg"], tri.state[p_old]["exp_avg"])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(master_weights=True),
        dict(exp_avg_dtype=torch.bfloat16, exp_avg_sq_dtype=torch.bfloat16),
    ],
)
def test_unsupported_configs_fall_back_to_te(kwargs, monkeypatch):
    calls = []
    orig = FusedAdam.step

    def spy(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(FusedAdam, "step", spy)
    params = _make_params(torch.bfloat16)
    opt = TritonFusedAdam(params, **HPARAMS, **kwargs)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    assert calls == [1]


def test_patch_routes_megatron_factory_to_triton():
    import megatron.core.optimizer as optimizer_module

    from primus.backends.megatron.patches import triton_fused_adam_patches as patch_mod
    from primus.backends.megatron.patches._patch_guard import _SENTINEL_ATTR

    ctx = SimpleNamespace(
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(use_triton_fused_adam=True))}
    )
    orig_adam = optimizer_module.Adam
    try:
        patch_mod.patch_triton_fused_adam(ctx)
        assert optimizer_module.Adam is TritonFusedAdam
        opt = optimizer_module.Adam(_make_params(torch.float32), lr=1e-3, adam_w_mode=True)
        assert isinstance(opt, FusedAdam)
        assert opt.grid_size == 0
    finally:
        optimizer_module.Adam = orig_adam
        getattr(optimizer_module, _SENTINEL_ATTR, set()).discard(patch_mod._PATCH_KEY)
