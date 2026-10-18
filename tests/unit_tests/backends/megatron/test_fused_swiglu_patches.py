###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused SwiGLU patch: plumbing on CPU (is_cuda stand-ins), numerics on a GPU."""

import importlib

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("triton")

import primus.backends.megatron.patches.fused_swiglu_patches as SWP  # noqa: E402
from primus.backends.megatron.core.fusions import triton_swiglu as S  # noqa: E402
from primus.backends.megatron.patches._rebind import rebind_everywhere  # noqa: E402
from tests.unit_tests.backends.megatron.conftest import (  # noqa: E402
    cuda_like,
    requires_gpu,
)


@pytest.fixture
def restore_targets(restore_patched):
    saved = []
    for modname, name, _ in SWP._TARGETS:
        mod = importlib.import_module(modname)
        restore_patched(mod)
        saved.append((mod, name, getattr(mod, name)))
    yield
    for mod, name, orig in saved:
        rebind_everywhere(mod, name, orig)


def test_patch_rebinds_importers_once_and_cpu_calls_fall_back(restore_targets):
    import megatron.core.fusions.fused_bias_swiglu as mfus
    import megatron.core.transformer.mlp as mlp

    orig = mfus.bias_swiglu_impl
    rebound = SWP._patch_sites()
    assert "megatron.core.transformer.mlp.bias_swiglu_impl" in rebound
    assert mlp.bias_swiglu_impl is mfus.bias_swiglu_impl and mfus.bias_swiglu_impl.__wrapped__ is orig
    assert SWP._patch_sites() == []  # idempotent: no double wrapping
    assert mfus.bias_swiglu_impl.__wrapped__ is orig

    x = torch.randn(6, 8)
    out = mfus.bias_swiglu_impl(x, None)  # CPU fp32 -> original Megatron function
    torch.testing.assert_close(out, torch.nn.functional.silu(x[:, :4]) * x[:, 4:])


def test_drop_ins_fall_back_outside_the_contract(monkeypatch):
    monkeypatch.setattr(S, "fused_swiglu", lambda x, weights=None: "fused")
    warned = []
    monkeypatch.setattr(S, "warn_fallback_once", lambda key, why: warned.append(why))
    calls = []

    def orig(*a):
        calls.append(len(a))
        return "orig"

    cpu = torch.randn(4, 8, dtype=torch.bfloat16)
    x = cuda_like(cpu)
    w = cuda_like(torch.rand(4, 1))
    b = x[:, 0]
    plain = S.make_megatron_plain(orig, "key")
    weighted = S.make_megatron_weighted(orig, "key")
    primus_weighted = S.make_primus_weighted(orig, "key")
    assert plain.__wrapped__ is orig
    # control: in-contract calls take the fused path
    assert plain(x, None) == weighted(x, None, w) == primus_weighted(x, None, w) == "fused"
    assert plain(cuda_like(torch.randn(3, 2, 8, dtype=torch.bfloat16)), None) == "fused"  # 3-D [s, b, 2H]
    assert weighted(x, None, cuda_like(torch.rand(4))) == "fused"  # 1-D weights
    assert calls == [] and warned == []
    # one contract violation per call
    assert plain(cpu, None) == "orig"  # CPU input
    assert plain(x, b) == "orig"  # bias
    assert plain(x, None, True) == "orig"  # fp8 store
    assert plain(x, None, False, True) == "orig"  # cpu offload
    assert plain(cuda_like(torch.randn(4, 8)), None) == "orig"  # fp32 input
    assert plain(cuda_like(torch.randn(4, 7, dtype=torch.bfloat16)), None) == "orig"  # odd last dim
    assert plain(cuda_like(torch.randn(2, 2, 2, 8, dtype=torch.bfloat16)), None) == "orig"  # 4-D
    assert weighted(x, b, w) == "orig"  # bias
    assert weighted(x, None, w, True) == "orig"  # fp8 store
    assert weighted(x, None, cuda_like(torch.rand(3, 1))) == "orig"  # weights row count
    assert weighted(x, None, cuda_like(torch.rand(4, 2))) == "orig"  # weights numel
    assert weighted(x, None, cuda_like(torch.rand(2, 2))) == "orig"  # weights shape
    assert weighted(x, None, None) == "orig"  # no weights
    assert primus_weighted(x, b, w) == "orig"  # bias
    assert primus_weighted(x, None, w, True) == "orig"  # fp8 store
    assert primus_weighted(x, None, w, False, 7.0) == "orig"  # clamp
    # the full original signature is forwarded, and every fallback names its reason
    assert calls == [4] * 7 + [4] * 6 + [5] * 3
    assert len(warned) == len(calls) and {"bias", "fp8_input_store", "clamp_value"} <= set(warned)


def _swiglu_reference(y, w=None):
    y = y.float()
    h = y.shape[-1] // 2
    out = torch.nn.functional.silu(y[..., :h]) * y[..., h:]
    return out if w is None else out * w.float().reshape(*out.shape[:-1], 1)


# Tolerances: bf16 output and input grads (one rounding each), fp32 probs grad summed over the row.


@requires_gpu
@pytest.mark.parametrize("M,H", [(7, 64), (33, 2048 + 64)])
@pytest.mark.parametrize("weighted", [False, True])
def test_fused_matches_reference(M, H, weighted):
    torch.manual_seed(0)
    y = torch.randn(M, 2 * H, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(M, 1, device="cuda", dtype=torch.float32) if weighted else None
    g = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)

    y_ref = y.detach().float().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True) if weighted else None
    ref = _swiglu_reference(y_ref, w_ref)
    ref.backward(g.float())

    y_f = y.detach().clone().requires_grad_(True)
    w_f = w.detach().clone().requires_grad_(True) if weighted else None
    out = S.fused_swiglu(y_f, w_f)
    out.backward(g)
    torch.testing.assert_close(out.float(), ref.detach(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(y_f.grad.float(), y_ref.grad, atol=3e-2, rtol=2e-2)
    if weighted:
        assert w_f.grad.dtype == torch.float32 and w_f.grad.shape == w.shape
        torch.testing.assert_close(w_f.grad, w_ref.grad, atol=1e-2, rtol=1e-3)


@requires_gpu
@pytest.mark.parametrize("weighted", [False, True])
def test_drop_ins_3d_match_reference(weighted):
    def orig(*a):
        raise AssertionError("original SwiGLU taken")

    torch.manual_seed(0)
    S_, B, H = 5, 3, 96
    y = torch.randn(S_, B, 2 * H, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(S_ * B, 1, device="cuda", dtype=torch.float32) if weighted else None
    g = torch.randn(S_, B, H, device="cuda", dtype=torch.bfloat16)
    y_ref = y.detach().float().requires_grad_(True)
    ref = _swiglu_reference(y_ref, w)
    ref.backward(g.float())

    y_f = y.detach().clone().requires_grad_(True)
    if weighted:
        out = S.make_megatron_weighted(orig, "key")(y_f, None, w)
    else:
        out = S.make_megatron_plain(orig, "key")(y_f, None)
    assert out.shape == (S_, B, H)
    out.backward(g)
    torch.testing.assert_close(out.float(), ref.detach(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(y_f.grad.float(), y_ref.grad, atol=3e-2, rtol=2e-2)
