###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU tests: the Triton param-remainder Adam is bit for bit TE's ROCm multi_tensor_adam_param_remainder,
and its gradient scale bit for bit an in-place ``grad *= scale`` before it."""

import pytest
import torch

from primus.core.utils.gpu_arch import is_gfx1250

pytestmark = pytest.mark.skipif(
    not is_gfx1250(),
    reason="needs a gfx1250 GPU: the param-remainder kernel is validated against TE there only",
)
te_mta = pytest.importorskip("transformer_engine.pytorch.optimizers.multi_tensor_apply")
import transformer_engine_torch as tex  # noqa: E402  (importable once transformer_engine.pytorch is)

from primus.core.kernels.triton_adam import (  # noqa: E402
    _MULTI_TENSOR_CHUNK,
    triton_multi_tensor_adam_param_remainder_step_,
)

SIZES = [1, 17, 70001, 3 * _MULTI_TENSOR_CHUNK + 5]


def _state(n, g_dtype, mv_dtype, gen):
    master = torch.randn(n, device="cuda", generator=gen) * 0.02
    master[: max(1, n // 64)] *= 50.0
    bits = master.view(torch.int32)
    rem = (bits & 0xFFFF).to(torch.int16)
    p16 = (bits >> 16).to(torch.int16)
    p16 = torch.where(rem < 0, p16 + 1, p16)
    scale = torch.exp(torch.randn(n, device="cuda", generator=gen) * 2)
    g = (torch.randn(n, device="cuda", generator=gen) * scale * 1e-4).to(g_dtype)
    g[: min(n, 3)] = 0
    m = (torch.randn(n, device="cuda", generator=gen) * 1e-4).to(mv_dtype)
    v = (torch.rand(n, device="cuda", generator=gen) * 1e-7).to(mv_dtype)
    return [g, p16.view(torch.bfloat16), m, v, rem]


def _bits(t):
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _te(states, lr, beta1, beta2, eps, step, mode, weight_decay):
    flag = torch.zeros(1, dtype=torch.int, device="cuda")
    lists = [list(x) for x in zip(*states)]
    te_mta.multi_tensor_applier(
        tex.multi_tensor_adam_param_remainder, flag, lists, lr, beta1, beta2, eps, step, mode, 1, weight_decay
    )


def _assert_same(ref, got):
    for a, b in zip(ref, got):
        for x, y in zip(a, b):
            assert torch.equal(_bits(x), _bits(y))


@pytest.mark.parametrize(
    "g_dtype,mv_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("adam_w_mode", [True, False])
@pytest.mark.parametrize("step", [1, 7, 1000])
def test_matches_te(g_dtype, mv_dtype, adam_w_mode, step):
    gen = torch.Generator(device="cuda").manual_seed(step)
    hp = dict(lr=3e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1)
    states = [_state(n, g_dtype, mv_dtype, gen) for n in SIZES]
    ref = [[t.clone() for t in s] for s in states]
    _te(ref, hp["lr"], hp["beta1"], hp["beta2"], hp["eps"], step, int(adam_w_mode), hp["weight_decay"])
    triton_multi_tensor_adam_param_remainder_step_(
        [tuple(s) for s in states], step=step, adam_w_mode=adam_w_mode, **hp
    )
    _assert_same(ref, states)


def test_misaligned_views_match_te():
    gen = torch.Generator(device="cuda").manual_seed(3)
    base = [_state(70001 + 1, torch.bfloat16, torch.bfloat16, gen) for _ in range(2)]
    states = [[t[1:] for t in s] for s in base]  # 2-byte-aligned starts
    ref = [[t.clone() for t in s] for s in states]
    _te(ref, 1e-3, 0.9, 0.95, 1e-8, 5, 1, 0.1)
    triton_multi_tensor_adam_param_remainder_step_(
        [tuple(s) for s in states], lr=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1, step=5
    )
    _assert_same(ref, states)


@pytest.mark.parametrize("g_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("coeff", [1.0, 0.3141, 0.0])
def test_grad_scale_matches_scaling_first(g_dtype, coeff):
    gen = torch.Generator(device="cuda").manual_seed(11)
    states = [_state(n, g_dtype, torch.bfloat16, gen) for n in SIZES]
    scale = torch.tensor([coeff], dtype=torch.float32, device="cuda")
    ref = [[t.clone() for t in s] for s in states]
    for s in ref:
        s[0].copy_((s[0].float() * scale).to(g_dtype))
    _te(ref, 3e-4, 0.9, 0.95, 1e-8, 2, 1, 0.1)
    triton_multi_tensor_adam_param_remainder_step_(
        [tuple(s) for s in states],
        lr=3e-4,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        step=2,
        grad_scale=scale,
    )
    # the gradient itself is left unscaled
    _assert_same([s[1:] for s in ref], [s[1:] for s in states])


def test_cached_tables_follow_new_addresses():
    gen = torch.Generator(device="cuda").manual_seed(5)
    cache = {}
    for _ in range(2):
        states = [_state(n, torch.bfloat16, torch.bfloat16, gen) for n in SIZES[:3]]
        ref = [[t.clone() for t in s] for s in states]
        _te(ref, 3e-4, 0.9, 0.95, 1e-8, 3, 1, 0.1)
        triton_multi_tensor_adam_param_remainder_step_(
            [tuple(s) for s in states],
            lr=3e-4,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.1,
            step=3,
            cache=cache,
        )
        _assert_same(ref, states)
