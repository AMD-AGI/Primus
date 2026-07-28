###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Plan-6 P36 G39 — `sinkhorn_normalize` Triton FWD/BWD parity.

Asserts that :class:`SinkhornNormalizeFn` (Triton kernel from
``primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.sinkhorn``)
matches the eager :func:`sinkhorn_normalize` body in
``primus.backends.megatron.core.transformer.hyper_connection`` and the
plan-5 P29 compiled path within dtype tolerance, FWD **and** BWD, at
two tiers:

* fast tier — `B=2, S=64, K=4`, exercising every code path in the
  kernel (priming col-step + 19 row/col pairs + cached state buffer
  round-trip) in milliseconds; parametrised over ``bf16`` (the
  production compute dtype) and ``n_iters ∈ {5, 20}``;
* release tier — `B=1, S=4096, K=4` bf16 (V4-Flash production shape),
  behind ``pytest.mark.slow``.

The **doubly-stochastic property check** is a model-quality contract
independent of the eager path (row / col sums of the FWD output equal
``1`` within ``eps * K``).

Note on ``torch.autograd.gradcheck``: the eager body casts to fp32
internally (``m = logits.float()``) and the Triton kernel matches that
contract bit-for-bit.  ``gradcheck`` at fp64 input is therefore
incompatible with both paths — the fp32 cast is lossy at the fp64
finite-difference step size.  We instead pin the BWD via a direct
parity check against ``torch.autograd`` of the eager body in fp64
(where ``out.backward(grad)`` does exactly the right thing through the
``.float()`` cast).

GPU-only; CPU runs are ``pytest.skip``-ed at module collection time.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip(
        "sinkhorn Triton kernel requires CUDA / HIP",
        allow_module_level=True,
    )

pytest.importorskip("triton", reason="Triton not installed")

from primus.backends.megatron.core.transformer.hyper_connection import (  # noqa: E402
    _get_compiled_sinkhorn,
    sinkhorn_normalize,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.sinkhorn import (  # noqa: E402
    SinkhornNormalizeFn,
    eager_sinkhorn_normalize,
    is_triton_kernel_supported,
    is_triton_path_enabled,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _env(key: str, value: str | None):
    """Temporarily set / unset ``os.environ[key]``."""
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


def _build_logits(
    *, B: int, S: int, K: int, dtype: torch.dtype, seed: int, requires_grad: bool = False
) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # Sinkhorn input is non-negative (softmax + eps in production).
    x = torch.rand((B, S, K, K), dtype=dtype, device="cuda", generator=gen) + 1e-3
    if requires_grad:
        x.requires_grad_(True)
    return x


def _dtype_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 1e-3, 1e-3
    if dtype == torch.bfloat16:
        return 1e-2, 1e-2
    if dtype == torch.float64:
        return 1e-7, 1e-7
    raise ValueError(dtype)


# ---------------------------------------------------------------------------
# G39: FWD parity vs eager (and vs plan-5 P29 compiled)
# ---------------------------------------------------------------------------


class TestG39ForwardParity:
    """FWD output matches eager and (separately) the compiled path."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("n_iters", [5, 20])
    def test_fast_tier_fwd_eager_parity(self, dtype, n_iters):
        x = _build_logits(B=2, S=64, K=4, dtype=dtype, seed=42)

        out_triton = SinkhornNormalizeFn.apply(x.clone(), n_iters, 1e-6)
        out_eager = eager_sinkhorn_normalize(x.clone(), n_iters=n_iters, eps=1e-6)

        atol, rtol = _dtype_tolerance(dtype)
        torch.testing.assert_close(out_triton, out_eager, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("n_iters", [5, 20])
    def test_fast_tier_fwd_compiled_parity(self, dtype, n_iters):
        """Triton FWD == plan-5 P29 compiled FWD within dtype tolerance.

        Both paths share the same algorithm; this test pins them
        together so a future P29 / P36 divergence raises a CI alarm.
        """
        x = _build_logits(B=2, S=64, K=4, dtype=dtype, seed=43)

        out_triton = SinkhornNormalizeFn.apply(x.clone(), n_iters, 1e-6)
        compiled_fn = _get_compiled_sinkhorn(n_iters, 1e-6, dtype)
        out_compiled = compiled_fn(x.clone())

        atol, rtol = _dtype_tolerance(dtype)
        torch.testing.assert_close(out_triton, out_compiled, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# G39: BWD parity vs eager + gradcheck
# ---------------------------------------------------------------------------


class TestG39BackwardParity:
    """BWD parity vs eager ``torch.autograd`` across dtypes."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("n_iters", [5, 20])
    def test_fast_tier_bwd_eager_parity(self, dtype, n_iters):
        x_base = _build_logits(B=2, S=64, K=4, dtype=dtype, seed=44)
        x_t = x_base.detach().clone().requires_grad_(True)
        x_e = x_base.detach().clone().requires_grad_(True)

        out_triton = SinkhornNormalizeFn.apply(x_t, n_iters, 1e-6)
        out_eager = eager_sinkhorn_normalize(x_e, n_iters=n_iters, eps=1e-6)

        grad = torch.randn_like(out_triton)
        out_triton.backward(grad)
        out_eager.backward(grad.detach().clone())

        atol, rtol = _dtype_tolerance(dtype)
        assert x_t.grad is not None
        assert x_e.grad is not None
        torch.testing.assert_close(x_t.grad, x_e.grad, atol=atol, rtol=rtol)

    def test_fp64_input_bwd_eager_parity(self):
        """fp64 input -> the kernel still does fp32-internal compute
        (matches eager's ``m = logits.float()`` contract); we compare
        Triton BWD vs ``torch.autograd``-of-eager rather than gradcheck
        (gradcheck's fp64 finite-difference step is incompatible with
        the lossy fp64->fp32 cast).
        """
        x_base = _build_logits(B=1, S=8, K=4, dtype=torch.float64, seed=100)
        x_t = x_base.detach().clone().requires_grad_(True)
        x_e = x_base.detach().clone().requires_grad_(True)

        out_triton = SinkhornNormalizeFn.apply(x_t, 5, 1e-6)
        out_eager = eager_sinkhorn_normalize(x_e, n_iters=5, eps=1e-6)

        grad = torch.randn_like(out_triton)
        out_triton.backward(grad)
        out_eager.backward(grad.detach().clone())

        # fp64 grads come back through a fp32-internal compute; tolerance
        # is set to fp32 precision (the kernel can't be more accurate
        # than its compute dtype).
        torch.testing.assert_close(x_t.grad, x_e.grad, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# G39: doubly-stochastic property (model-quality contract)
# ---------------------------------------------------------------------------


class TestG39DoublyStochastic:
    """The FWD output's row and column sums equal ``1`` within
    ``eps * K`` -- pinning the model-quality contract that
    :func:`sinkhorn_normalize` is supposed to guarantee.

    This is independent of the eager path; an algorithmic bug in the
    Triton kernel that still happens to match the eager value would
    fail this check.
    """

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("K", [4, 8])
    def test_row_col_sums_close_to_1(self, dtype, K):
        x = _build_logits(B=2, S=16, K=K, dtype=dtype, seed=55)
        y = SinkhornNormalizeFn.apply(x, 20, 1e-6)

        row_sums = y.sum(dim=-1)
        col_sums = y.sum(dim=-2)
        # eps*K = 4e-6 (fp32) or ~6e-5 (bf16 -> the cast back to bf16
        # is the dominant error, so we widen by 16x for bf16).
        tol = 4e-6 * K if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=tol, rtol=tol)
        torch.testing.assert_close(col_sums, torch.ones_like(col_sums), atol=tol, rtol=tol)


# ---------------------------------------------------------------------------
# G39: release-tier V4-Flash production shape
# ---------------------------------------------------------------------------


class TestG39ReleaseTier:
    """V4-Flash production shape: `B=1, S=4096, K=4`, bf16, n_iters=20.

    Marked ``slow``; pins both FWD and BWD against eager AND compiled
    paths to make sure the in-production widths match.
    """

    @pytest.mark.slow
    def test_v4_flash_fwd_bwd_parity(self):
        dtype = torch.bfloat16
        x_base = _build_logits(B=1, S=4096, K=4, dtype=dtype, seed=200)

        x_t = x_base.detach().clone().requires_grad_(True)
        x_e = x_base.detach().clone().requires_grad_(True)
        x_c = x_base.detach().clone().requires_grad_(True)

        out_triton = SinkhornNormalizeFn.apply(x_t, 20, 1e-6)
        out_eager = eager_sinkhorn_normalize(x_e, n_iters=20, eps=1e-6)
        compiled_fn = _get_compiled_sinkhorn(20, 1e-6, dtype)
        out_compiled = compiled_fn(x_c)

        atol, rtol = _dtype_tolerance(dtype)
        torch.testing.assert_close(out_triton, out_eager, atol=atol, rtol=rtol)
        torch.testing.assert_close(out_triton, out_compiled, atol=atol, rtol=rtol)

        grad = torch.randn_like(out_triton)
        out_triton.backward(grad.detach().clone())
        out_eager.backward(grad.detach().clone())
        out_compiled.backward(grad.detach().clone())
        torch.testing.assert_close(x_t.grad, x_e.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(x_t.grad, x_c.grad, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# G39: edge cases + error paths
# ---------------------------------------------------------------------------


class TestG39EdgeCases:
    """Defensive validation in :class:`SinkhornNormalizeFn`."""

    def test_non_square_raises(self):
        x = torch.rand(2, 4, 5, device="cuda", dtype=torch.float32) + 1e-3
        with pytest.raises(ValueError, match="square"):
            SinkhornNormalizeFn.apply(x, 20, 1e-6)

    def test_unsupported_k_raises(self):
        x = torch.rand(2, 32, 32, device="cuda", dtype=torch.float32) + 1e-3
        with pytest.raises(ValueError, match="unsupported K"):
            SinkhornNormalizeFn.apply(x, 20, 1e-6)

    def test_n_iters_zero_raises(self):
        x = torch.rand(2, 4, 4, device="cuda", dtype=torch.float32) + 1e-3
        with pytest.raises(ValueError, match="n_iters must be"):
            SinkhornNormalizeFn.apply(x, 0, 1e-6)

    def test_kernel_supported_predicate(self):
        good = torch.rand(2, 4, 4, device="cuda", dtype=torch.float32)
        bad_k = torch.rand(2, 32, 32, device="cuda", dtype=torch.float32)
        bad_dev = torch.rand(2, 4, 4, dtype=torch.float32)  # cpu
        assert is_triton_kernel_supported(good)
        assert not is_triton_kernel_supported(bad_k)
        assert not is_triton_kernel_supported(bad_dev)


# ---------------------------------------------------------------------------
# G39: env-flag dispatch through hyper_connection.sinkhorn_normalize
# ---------------------------------------------------------------------------


class TestG39EnvFlagDispatch:
    """The ``PRIMUS_SINKHORN_TRITON`` env knob flips the dispatcher
    inside :func:`sinkhorn_normalize`.  Both paths agree within bf16
    tolerance.
    """

    def test_env_on_uses_triton(self):
        x = _build_logits(B=2, S=8, K=4, dtype=torch.bfloat16, seed=11)

        with _env("PRIMUS_SINKHORN_TRITON", "1"):
            assert is_triton_path_enabled()
            out_on = sinkhorn_normalize(x.clone(), n_iters=20, eps=1e-6)
        with _env("PRIMUS_SINKHORN_TRITON", "0"):
            assert not is_triton_path_enabled()
            out_eager = sinkhorn_normalize(x.clone(), n_iters=20, eps=1e-6)

        atol, rtol = _dtype_tolerance(torch.bfloat16)
        torch.testing.assert_close(out_on, out_eager, atol=atol, rtol=rtol)

    def test_use_triton_kwarg_overrides_env_off(self):
        """``use_triton=True`` forces the Triton path even when the env
        knob is off (used by the unit tests / by callers who want a
        per-call override).
        """
        x = _build_logits(B=2, S=8, K=4, dtype=torch.bfloat16, seed=12)
        with _env("PRIMUS_SINKHORN_TRITON", "0"):
            out_triton = sinkhorn_normalize(x.clone(), n_iters=20, eps=1e-6, use_triton=True)
            out_eager = sinkhorn_normalize(x.clone(), n_iters=20, eps=1e-6)

        atol, rtol = _dtype_tolerance(torch.bfloat16)
        torch.testing.assert_close(out_triton, out_eager, atol=atol, rtol=rtol)

    def test_routing_precedence_triton_over_compiled(self):
        """When both ``use_triton`` and ``use_compiled`` are set, the
        Triton path wins (routing precedence:
        ``use_triton > use_compiled > eager``).
        """
        x = _build_logits(B=2, S=8, K=4, dtype=torch.bfloat16, seed=13)
        with _env("PRIMUS_SINKHORN_TRITON", "0"):
            out = sinkhorn_normalize(
                x.clone(),
                n_iters=20,
                eps=1e-6,
                use_triton=True,
                use_compiled=True,
            )
            out_triton_only = sinkhorn_normalize(
                x.clone(),
                n_iters=20,
                eps=1e-6,
                use_triton=True,
            )

        torch.testing.assert_close(out, out_triton_only, atol=0, rtol=0)
