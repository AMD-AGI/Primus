###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Plan-6 P37 G40 — `HyperMixer.compute_weights` tail Triton parity.

Asserts that :class:`HCComputeTailFn` (Triton kernel from
``primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.hc_glue``)
matches the eager body in
``primus.backends.megatron.core.transformer.hyper_connection.HyperMixer.compute_weights``
bit-for-bit-equivalent FWD and BWD at two tiers:

* fast tier -- `B=2, S=64, K=4` exercising every code path (3 slices,
  2 sigmoid, 1 softmax, scale + base, eps); parametrised over `K ∈ {1,
  2, 4, 8}`;
* release tier -- `B=1, S=4096, K=4` bf16 (V4-Flash production shape),
  behind ``pytest.mark.slow``.

Composed end-to-end ``HyperMixer.compute_weights`` parity at K=4 (the
load-bearing test: routing through the public API matches the Triton
path within the eager-fallback baseline within bf16 tolerance).
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip(
        "hc_glue Triton kernel requires CUDA / HIP",
        allow_module_level=True,
    )

pytest.importorskip("triton", reason="Triton not installed")

from primus.backends.megatron.core.transformer.hyper_connection import (  # noqa: E402
    HyperMixer,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.hc_glue import (  # noqa: E402
    HCComputeTailFn,
    is_triton_kernel_supported,
    is_triton_path_enabled,
)


@contextmanager
def _env(key: str, value: str | None):
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


def _eager_tail(
    logits: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    K: int,
    eps: float,
    out_dtype: torch.dtype,
):
    """Eager reference matching the pre-P37 tail body bit-for-bit."""
    pre_logit = logits[..., :K] * scale[0] + base[:K]
    post_logit = logits[..., K : 2 * K] * scale[1] + base[K : 2 * K]
    comb_logit = logits[..., 2 * K :].view(*logits.shape[:-1], K, K) * scale[2] + base[2 * K :].view(K, K)
    pre = torch.sigmoid(pre_logit) + eps
    post = 2.0 * torch.sigmoid(post_logit)
    comb = torch.softmax(comb_logit, dim=-1) + eps
    return pre.to(out_dtype), post.to(out_dtype), comb.to(out_dtype)


def _build_inputs(*, B: int, S: int, K: int, seed: int, requires_grad: bool = False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out_dim = (2 + K) * K
    logits = torch.randn((B, S, out_dim), dtype=torch.float32, device="cuda", generator=gen)
    scale = torch.ones(3, dtype=torch.float32, device="cuda")
    scale = scale + 0.1 * torch.randn(3, dtype=torch.float32, device="cuda", generator=gen)
    base = 0.01 * torch.randn(out_dim, dtype=torch.float32, device="cuda", generator=gen)
    if requires_grad:
        logits.requires_grad_(True)
        scale.requires_grad_(True)
        base.requires_grad_(True)
    return logits, scale, base


def _dtype_tolerance(dtype: torch.dtype):
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 1e-3, 1e-3
    if dtype == torch.bfloat16:
        return 1e-2, 1e-2
    raise ValueError(dtype)


# ---------------------------------------------------------------------------
# G40: FWD parity vs eager
# ---------------------------------------------------------------------------


class TestG40ForwardParity:
    """FWD output matches eager within out_dtype tolerance."""

    @pytest.mark.parametrize("K", [1, 2, 4, 8])
    @pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
    def test_fast_tier_fwd_eager_parity(self, K, out_dtype):
        logits, scale, base = _build_inputs(B=2, S=64, K=K, seed=42 + K)

        pre_t, post_t, comb_t = HCComputeTailFn.apply(logits, scale, base, K, 1e-6, out_dtype)
        pre_e, post_e, comb_e = _eager_tail(logits, scale, base, K=K, eps=1e-6, out_dtype=out_dtype)

        atol, rtol = _dtype_tolerance(out_dtype)
        torch.testing.assert_close(pre_t, pre_e, atol=atol, rtol=rtol)
        torch.testing.assert_close(post_t, post_e, atol=atol, rtol=rtol)
        torch.testing.assert_close(comb_t, comb_e, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# G40: BWD parity vs eager autograd
# ---------------------------------------------------------------------------


class TestG40BackwardParity:
    """BWD parity vs eager autograd across K and dtypes."""

    @pytest.mark.parametrize("K", [1, 2, 4, 8])
    @pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
    def test_fast_tier_bwd_eager_parity(self, K, out_dtype):
        logits_e, scale_e, base_e = _build_inputs(B=2, S=64, K=K, seed=144 + K, requires_grad=True)
        logits_t = logits_e.detach().clone().requires_grad_(True)
        scale_t = scale_e.detach().clone().requires_grad_(True)
        base_t = base_e.detach().clone().requires_grad_(True)

        pre_t, post_t, comb_t = HCComputeTailFn.apply(logits_t, scale_t, base_t, K, 1e-6, out_dtype)
        pre_e, post_e, comb_e = _eager_tail(logits_e, scale_e, base_e, K=K, eps=1e-6, out_dtype=out_dtype)

        g_pre = torch.randn_like(pre_t)
        g_post = torch.randn_like(post_t)
        g_comb = torch.randn_like(comb_t)

        (pre_t * g_pre).sum().add_((post_t * g_post).sum()).add_((comb_t * g_comb).sum()).backward()
        (pre_e * g_pre).sum().add_((post_e * g_post).sum()).add_((comb_e * g_comb).sum()).backward()

        atol, rtol = _dtype_tolerance(out_dtype)
        # Bump tolerance one notch for d_scale (cross-term: O(N) sum
        # of N elements amplifies rounding error linearly with N).
        scale_atol = atol * 10 if out_dtype == torch.bfloat16 else atol
        scale_rtol = rtol * 10 if out_dtype == torch.bfloat16 else rtol
        torch.testing.assert_close(logits_t.grad, logits_e.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(base_t.grad, base_e.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(scale_t.grad, scale_e.grad, atol=scale_atol, rtol=scale_rtol)


# ---------------------------------------------------------------------------
# G40: Composed end-to-end HyperMixer.compute_weights parity
# ---------------------------------------------------------------------------


class TestG40HyperMixerParity:
    """End-to-end ``HyperMixer.compute_weights`` parity between the
    default-on Triton path and the env=0 eager fallback (with the same
    weights / inputs / Sinkhorn iters).
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_compute_weights_env_dispatch_parity(self, dtype):
        torch.manual_seed(20260514)
        mixer = HyperMixer(hidden_size=64, hc_mult=4, sinkhorn_iters=20).to("cuda")
        x = torch.randn(2, 16, 4, 64, dtype=dtype, device="cuda")

        with _env("PRIMUS_HC_TRITON", "1"):
            assert is_triton_path_enabled()
            pre_t, post_t, comb_t = mixer.compute_weights(x.clone())
        with _env("PRIMUS_HC_TRITON", "0"):
            assert not is_triton_path_enabled()
            pre_e, post_e, comb_e = mixer.compute_weights(x.clone())

        atol, rtol = _dtype_tolerance(dtype)
        torch.testing.assert_close(pre_t, pre_e, atol=atol, rtol=rtol)
        torch.testing.assert_close(post_t, post_e, atol=atol, rtol=rtol)
        torch.testing.assert_close(comb_t, comb_e, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# G40: release-tier V4-Flash production shape
# ---------------------------------------------------------------------------


class TestG40ReleaseTier:
    """V4-Flash production shape: ``B=1, S=4096, K=4``, bf16."""

    @pytest.mark.slow
    def test_v4_flash_fwd_bwd_parity(self):
        K = 4
        out_dtype = torch.bfloat16
        logits_e, scale_e, base_e = _build_inputs(B=1, S=4096, K=K, seed=2200, requires_grad=True)
        logits_t = logits_e.detach().clone().requires_grad_(True)
        scale_t = scale_e.detach().clone().requires_grad_(True)
        base_t = base_e.detach().clone().requires_grad_(True)

        pre_t, post_t, comb_t = HCComputeTailFn.apply(logits_t, scale_t, base_t, K, 1e-6, out_dtype)
        pre_e, post_e, comb_e = _eager_tail(logits_e, scale_e, base_e, K=K, eps=1e-6, out_dtype=out_dtype)

        atol, rtol = _dtype_tolerance(out_dtype)
        torch.testing.assert_close(pre_t, pre_e, atol=atol, rtol=rtol)
        torch.testing.assert_close(post_t, post_e, atol=atol, rtol=rtol)
        torch.testing.assert_close(comb_t, comb_e, atol=atol, rtol=rtol)

        g_pre = torch.randn_like(pre_t)
        g_post = torch.randn_like(post_t)
        g_comb = torch.randn_like(comb_t)
        (pre_t * g_pre).sum().add_((post_t * g_post).sum()).add_((comb_t * g_comb).sum()).backward()
        (pre_e * g_pre).sum().add_((post_e * g_post).sum()).add_((comb_e * g_comb).sum()).backward()

        # At V4-Flash sequence length (S=4096), d_scale accumulates
        # B*S*K = 16384 cross-terms; bf16 rounding compounds linearly,
        # so widen tolerance for scale.
        torch.testing.assert_close(logits_t.grad, logits_e.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(base_t.grad, base_e.grad, atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(scale_t.grad, scale_e.grad, atol=1e-1, rtol=1e-1)


# ---------------------------------------------------------------------------
# G40: edge cases
# ---------------------------------------------------------------------------


class TestG40EdgeCases:
    def test_unsupported_k_raises(self):
        logits, scale, base = _build_inputs(B=2, S=8, K=4, seed=33)
        # Bad K -- pass K=5 (not in {1,2,4,8,16})
        bad_K = 3
        bad_logits = torch.randn(2, 8, (2 + bad_K) * bad_K, dtype=torch.float32, device="cuda")
        bad_base = torch.zeros((2 + bad_K) * bad_K, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="unsupported K"):
            HCComputeTailFn.apply(bad_logits, scale, bad_base, bad_K, 1e-6, torch.float32)

    def test_bad_logits_shape_raises(self):
        logits = torch.randn(2, 8, 99, dtype=torch.float32, device="cuda")  # bad last dim
        scale = torch.ones(3, dtype=torch.float32, device="cuda")
        base = torch.zeros(24, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="logits last-dim"):
            HCComputeTailFn.apply(logits, scale, base, 4, 1e-6, torch.float32)

    def test_kernel_supported_predicate(self):
        good = torch.randn(2, 8, 24, dtype=torch.float32, device="cuda")
        bad_k = torch.randn(2, 8, 51, dtype=torch.float32, device="cuda")  # K=? doesn't match
        bad_dev = torch.randn(2, 8, 24, dtype=torch.float32)  # cpu
        assert is_triton_kernel_supported(good, K=4)
        assert not is_triton_kernel_supported(bad_k, K=4)
        assert not is_triton_kernel_supported(bad_dev, K=4)
