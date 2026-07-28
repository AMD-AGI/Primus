###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Small-kernel-fusion 2026-07-03 — fused Triton HyperConnection ``collapse``.

Pins :class:`HCCollapseFn` against the eager
``(pre.unsqueeze(-1) * x).sum(-2)`` reference AND the actual
:meth:`HyperMixer.collapse` call site, FWD + BWD, across dtypes and K.

GPU-only; CPU runs are skipped at collection time.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("fused HC collapse Triton kernel requires CUDA / HIP", allow_module_level=True)

pytest.importorskip("triton", reason="Triton not installed")

from primus.backends.megatron.core.transformer.hyper_connection import (  # noqa: E402
    HyperMixer,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.hc_collapse import (  # noqa: E402
    HCCollapseFn,
    eager_hc_collapse,
    hc_collapse_triton,
    is_triton_kernel_supported,
    is_triton_path_enabled,
)


@contextmanager
def _env(key, value):
    prev = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _tol(dtype):
    return {
        torch.float32: (1e-5, 1e-5),
        torch.float16: (2e-3, 2e-3),
        torch.bfloat16: (1e-2, 1e-2),
    }[dtype]


def _mk(shape, dtype, seed, requires_grad=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(shape, dtype=dtype, device="cuda", generator=gen)
    if requires_grad:
        x.requires_grad_(True)
    return x


def _assert_at_least_as_accurate(got, x, pre, dtype):
    """For low-precision dtypes the eager path rounds each ``pre*x`` product to
    the input dtype before summing, while the kernel accumulates in fp32. Rather
    than pin the kernel to the *less* accurate eager path, pin FWD parity in
    fp32 and, for fp16/bf16, assert the kernel matches an fp32 gold within the
    output dtype's rounding (i.e. the kernel is at least as accurate as eager).
    """
    if dtype == torch.float32:
        ref = eager_hc_collapse(x, pre)
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
        return
    gold = (pre.float().unsqueeze(-1) * x.float()).sum(dim=-2)
    atol, rtol = _tol(dtype)
    # allow output-rounding slack on top of the relative tolerance
    torch.testing.assert_close(got.float(), gold, atol=3 * atol, rtol=rtol)


class TestFwd:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("K", [1, 2, 4, 8])
    @pytest.mark.parametrize("D", [64, 512, 4096])
    def test_parity(self, dtype, K, D):
        x = _mk((2, 130, K, D), dtype, seed=1)
        pre = _mk((2, 130, K), dtype, seed=2)
        got = HCCollapseFn.apply(x, pre)
        _assert_at_least_as_accurate(got, x, pre, dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_call_site_parity(self, dtype):
        x = _mk((3, 40, 4, 4096), dtype, seed=3)
        pre = _mk((3, 40, 4), dtype, seed=4)
        got = HyperMixer.collapse(x, pre)
        _assert_at_least_as_accurate(got, x, pre, dtype)


class TestBwd:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("K", [2, 4])
    def test_bwd_parity(self, dtype, K):
        D = 512
        xb = _mk((2, 64, K, D), dtype, seed=10)
        pb = _mk((2, 64, K), dtype, seed=11)
        xt = xb.detach().clone().requires_grad_(True)
        pt = pb.detach().clone().requires_grad_(True)
        xe = xb.detach().clone().requires_grad_(True)
        pe = pb.detach().clone().requires_grad_(True)

        out_t = HCCollapseFn.apply(xt, pt)
        out_e = eager_hc_collapse(xe, pe)
        g = torch.randn_like(out_t)
        out_t.backward(g)
        out_e.backward(g.detach().clone())

        if dtype == torch.float32:
            torch.testing.assert_close(xt.grad, xe.grad, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(pt.grad, pe.grad, atol=1e-4, rtol=1e-4)
            return
        # Low precision: compare against fp32 gold (kernel accumulates in fp32,
        # so it is at least as accurate as the bf16 eager reference).
        xg = xb.detach().float().requires_grad_(True)
        pg = pb.detach().float().requires_grad_(True)
        out_g = eager_hc_collapse(xg, pg)
        out_g.backward(g.float())
        atol, rtol = _tol(dtype)
        torch.testing.assert_close(xt.grad.float(), xg.grad, atol=3 * atol, rtol=rtol)
        torch.testing.assert_close(pt.grad.float(), pg.grad, atol=5e-2, rtol=5e-2)


class TestDispatch:
    def test_env_flag(self):
        x = _mk((2, 32, 4, 512), torch.bfloat16, seed=20)
        pre = _mk((2, 32, 4), torch.bfloat16, seed=21)
        with _env("PRIMUS_HC_COLLAPSE_TRITON", "1"):
            assert is_triton_path_enabled()
            on = hc_collapse_triton(x, pre)
        with _env("PRIMUS_HC_COLLAPSE_TRITON", "0"):
            assert not is_triton_path_enabled()
            off = hc_collapse_triton(x, pre)
        # triton (fp32 accum) vs eager (bf16 accum): both valid bf16
        # approximations of the same op, so allow K-way bf16 rounding slack.
        torch.testing.assert_close(on, off, atol=6e-2, rtol=6e-2)

    def test_support_predicate(self):
        x = _mk((2, 4, 512), torch.bfloat16, seed=22)
        pre = _mk((2, 4), torch.bfloat16, seed=23)
        assert is_triton_kernel_supported(x, pre)
        assert not is_triton_kernel_supported(x.cpu(), pre.cpu())
