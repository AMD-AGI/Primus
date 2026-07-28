###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Small-kernel-fusion 2026-07-03 — fused Triton RMSNorm FWD/BWD parity.

Pins :class:`FusedRMSNormFn` (Triton kernel in
``primus...v4_attention_kernels._triton_common.rmsnorm``) against the eager
RMSNorm reference AND the actual model call sites it replaces:

* ``_per_head_rms_norm``               (parameter-less, out=in_dtype)
* ``LocalRMSNorm``                     (weighted + weight grad, mid-cast)
* ``HyperMixer._packed_logits`` RMS    (parameter-less, out=fp32)
* ``HyperHead`` RMS                    (parameter-less, out=fp32)

GPU-only; CPU runs are skipped at collection time.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("fused RMSNorm Triton kernel requires CUDA / HIP", allow_module_level=True)

pytest.importorskip("triton", reason="Triton not installed")

# Import the model package first so the deepseek_v4_attention <-> block cyclic
# import resolves cleanly before we pull `_per_head_rms_norm` off the leaf.
import primus.backends.megatron.core.models.deepseek_v4  # noqa: E402,F401
from primus.backends.megatron.core.transformer.deepseek_v4_attention import (  # noqa: E402
    _per_head_rms_norm,
)
from primus.backends.megatron.core.transformer.local_rmsnorm import (  # noqa: E402
    LocalRMSNorm,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.rmsnorm import (  # noqa: E402
    FusedRMSNormFn,
    eager_rms_norm,
    fused_rms_norm,
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


# ---------------------------------------------------------------------------
# FWD parity vs eager reference — the four site contracts.
# ---------------------------------------------------------------------------


class TestFwdParity:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("D", [128, 512, 4096, 16384])
    def test_no_weight_in_dtype(self, dtype, D):
        """per-head RMS contract: no weight, out=in_dtype."""
        x = _mk((64, D), dtype, seed=1)
        out_t = FusedRMSNormFn.apply(x, None, 1e-6, False, dtype)
        out_e = eager_rms_norm(x, None, eps=1e-6, mid_cast=False, out_dtype=dtype)
        atol, rtol = _tol(dtype)
        torch.testing.assert_close(out_t, out_e, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("D", [512, 4096])
    def test_weighted_mid_cast(self, dtype, D):
        """LocalRMSNorm contract: weight, mid-cast, out=promote(in, weight)."""
        x = _mk((128, D), dtype, seed=2)
        w = _mk((D,), torch.float32, seed=3)
        out_dtype = torch.promote_types(dtype, torch.float32)
        out_t = FusedRMSNormFn.apply(x, w, 1e-6, True, out_dtype)
        out_e = eager_rms_norm(x, w, eps=1e-6, mid_cast=True, out_dtype=out_dtype)
        atol, rtol = _tol(dtype)
        torch.testing.assert_close(out_t, out_e, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_no_weight_fp32_out(self, dtype):
        """HyperMixer / HyperHead RMS contract: no weight, out=fp32."""
        x = _mk((256, 16384), dtype, seed=4)
        out_t = FusedRMSNormFn.apply(x, None, 1e-6, False, torch.float32)
        out_e = eager_rms_norm(x, None, eps=1e-6, mid_cast=False, out_dtype=torch.float32)
        atol, rtol = _tol(dtype)
        torch.testing.assert_close(out_t, out_e, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Parity vs the ACTUAL call sites (the code paths this kernel replaces).
# ---------------------------------------------------------------------------


class TestCallSiteParity:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_per_head_rms_norm(self, dtype):
        # [B, S, H, head_dim] as in _apply_q.
        x = _mk((1, 128, 8, 512), dtype, seed=10)
        ref = _per_head_rms_norm(x, eps=1e-6)
        got = fused_rms_norm(x, None, eps=1e-6, mid_cast=False, out_dtype=dtype)
        atol, rtol = _tol(dtype)
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_local_rmsnorm_forward(self, dtype):
        norm = LocalRMSNorm(512, eps=1e-6).cuda()
        with torch.no_grad():
            norm.weight.normal_()
        x = _mk((16, 64, 512), dtype, seed=11)
        ref = norm(x)
        out_dtype = torch.promote_types(dtype, norm.weight.dtype)
        got = fused_rms_norm(x, norm.weight, eps=1e-6, mid_cast=True, out_dtype=out_dtype)
        atol, rtol = _tol(dtype)
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# BWD parity vs eager autograd.
# ---------------------------------------------------------------------------


class TestBwdParity:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("D", [512, 4096])
    def test_no_weight_bwd(self, dtype, D):
        xb = _mk((128, D), dtype, seed=20)
        xt = xb.detach().clone().requires_grad_(True)
        xe = xb.detach().clone().requires_grad_(True)
        out_t = FusedRMSNormFn.apply(xt, None, 1e-6, False, dtype)
        out_e = eager_rms_norm(xe, None, eps=1e-6, mid_cast=False, out_dtype=dtype)
        g = torch.randn_like(out_t)
        out_t.backward(g)
        out_e.backward(g.detach().clone())
        atol, rtol = _tol(dtype)
        torch.testing.assert_close(xt.grad, xe.grad, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_weighted_bwd_x_and_w(self, dtype):
        D = 512
        xb = _mk((256, D), dtype, seed=21)
        wb = _mk((D,), torch.float32, seed=22)
        out_dtype = torch.promote_types(dtype, torch.float32)

        xt = xb.detach().clone().requires_grad_(True)
        wt = wb.detach().clone().requires_grad_(True)
        xe = xb.detach().clone().requires_grad_(True)
        we = wb.detach().clone().requires_grad_(True)

        out_t = FusedRMSNormFn.apply(xt, wt, 1e-6, True, out_dtype)
        out_e = eager_rms_norm(xe, we, eps=1e-6, mid_cast=True, out_dtype=out_dtype)
        g = torch.randn_like(out_t)
        out_t.backward(g)
        out_e.backward(g.detach().clone())

        atol, rtol = _tol(dtype)
        torch.testing.assert_close(xt.grad, xe.grad, atol=atol, rtol=rtol)
        # Weight grad is a cross-row reduction; the eager reference rounds the
        # upstream grad to bf16 per row while the kernel accumulates in fp32, so
        # for bf16 the per-element diff is dominated by reduction noise where the
        # true sum has cancellation. Compare against an fp32 "gold" reduction to
        # confirm the kernel is at least as accurate as eager.
        if dtype == torch.float32:
            torch.testing.assert_close(wt.grad, we.grad, atol=1e-4, rtol=1e-4)
        else:
            gold = (
                g.float() * (xb.float() * torch.rsqrt(xb.float().pow(2).mean(-1, keepdim=True) + 1e-6))
            ).sum(0)
            err_kernel = (wt.grad.float() - gold).abs().max()
            err_eager = (we.grad.float() - gold).abs().max()
            assert err_kernel <= err_eager + 1e-3, (err_kernel, err_eager)


# ---------------------------------------------------------------------------
# Dispatch / edge cases.
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_env_flag(self):
        x = _mk((32, 512), torch.bfloat16, seed=30)
        with _env("PRIMUS_RMSNORM_TRITON", "1"):
            assert is_triton_path_enabled()
            on = fused_rms_norm(x, None, eps=1e-6, out_dtype=torch.bfloat16)
        with _env("PRIMUS_RMSNORM_TRITON", "0"):
            assert not is_triton_path_enabled()
            off = fused_rms_norm(x, None, eps=1e-6, out_dtype=torch.bfloat16)
        torch.testing.assert_close(on, off, atol=1e-2, rtol=1e-2)

    def test_support_predicate(self):
        good = _mk((4, 512), torch.bfloat16, seed=31)
        assert is_triton_kernel_supported(good, None)
        assert not is_triton_kernel_supported(good.cpu(), None)

    def test_cpu_falls_back(self):
        x = torch.randn(4, 512, dtype=torch.float32)
        # CPU input: dispatcher must not raise, returns eager result.
        out = fused_rms_norm(x, None, eps=1e-6, out_dtype=torch.float32)
        ref = eager_rms_norm(x, None, eps=1e-6, mid_cast=False, out_dtype=torch.float32)
        torch.testing.assert_close(out, ref)
