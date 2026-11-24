###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# GPU cases ported from ROCm/maxtext release/v26.10 (tests/unit/turbo_grouped_gemm_test.py),
# Copyright 2023-2026 Google LLC, licensed under the Apache License, Version 2.0.
###############################################################################

"""Tests for the Primus-Turbo grouped GEMM wrapper used by MaxText MoE sparse matmul."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

from primus.backends.maxtext.kernels import turbo_grouped_gemm  # noqa: E402

try:
    import primus_turbo.jax  # noqa: F401

    HAS_PRIMUS_TURBO = True
except ImportError:
    HAS_PRIMUS_TURBO = False


def _gcn_arch() -> str:
    try:
        return str(getattr(jax.devices()[0], "compute_capability", ""))
    except RuntimeError:
        return ""


# FlyDSL kernels are gfx950-only; on other GPUs they abort the process instead of raising.
requires_gfx950 = pytest.mark.skipif("gfx950" not in _gcn_arch(), reason="FlyDSL grouped GEMM targets gfx950")


def _backend_params(backends):
    return [pytest.param(b, marks=requires_gfx950) if b == "flydsl" else b for b in backends]


def _reference(lhs, rhs, group_sizes):
    out = jax.lax.ragged_dot(lhs.astype(jnp.float32), rhs.astype(jnp.float32), group_sizes)
    rows = jnp.arange(lhs.shape[0])[:, None]
    return jnp.where(rows < jnp.sum(group_sizes), out, 0.0)


def _rel_err(x, ref):
    x, ref = np.asarray(x, np.float32), np.asarray(ref, np.float32)
    return np.linalg.norm(x - ref) / max(np.linalg.norm(ref), 1e-30)


@pytest.mark.parametrize("group_offset", [0, 1, 2, 3, 4])
def test_window_apply_matches_masked_ragged_dot(group_offset):
    """Ring-of-experts + ragged sort: only the local experts' rows are computed, at their global rows."""
    group_sizes = jnp.array([3, 0, 5, 2, 4, 1], jnp.int32)
    m, k, n, e_local = 18, 8, 4, 2
    lhs = jax.random.normal(jax.random.key(0), (m, k), jnp.float32)
    full = jax.random.normal(jax.random.key(1), (group_sizes.shape[0], k, n), jnp.float32)
    local = full[group_offset : group_offset + e_local]

    run = lambda l, g: jax.lax.ragged_dot(l, local, g)  # noqa: E731
    out = jax.jit(lambda x, off: turbo_grouped_gemm.window_apply(x, e_local, group_sizes, off, run))(
        lhs, group_offset
    )
    masked = full.at[:group_offset].set(0).at[group_offset + e_local :].set(0)
    np.testing.assert_allclose(out, jax.lax.ragged_dot(lhs, masked, group_sizes), atol=1e-5)


def test_rejects_unknown_backend():
    x = jnp.zeros((16, 8), jnp.bfloat16)
    w = jnp.zeros((2, 8, 8), jnp.bfloat16)
    with pytest.raises(ValueError, match="Unknown turbo grouped GEMM backend"):
        turbo_grouped_gemm.gmm(x, w, jnp.array([8, 8], jnp.int32), backend="nope")


@pytest.mark.skipif(not HAS_PRIMUS_TURBO, reason="primus_turbo is not installed")
class TestTurboGroupedGemm:
    """Forward and gradients of turbo_grouped_gemm.gmm against a float32 ragged_dot reference."""

    def _check(self, backend, group_sizes, m, k, n):
        group_sizes = jnp.asarray(group_sizes, jnp.int32)
        num_groups, valid = group_sizes.shape[0], int(jnp.sum(group_sizes))
        k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
        lhs = jax.random.normal(k1, (m, k), jnp.bfloat16)
        rhs = (jax.random.normal(k2, (num_groups, k, n), jnp.float32) / np.sqrt(k)).astype(jnp.bfloat16)
        cotangent = jax.random.normal(k3, (m, n), jnp.float32)

        def loss(fn, x, w):
            return jnp.sum(fn(x, w).astype(jnp.float32) * cotangent)

        turbo = lambda x, w: turbo_grouped_gemm.gmm(x, w, group_sizes, backend=backend)  # noqa: E731
        ref = lambda x, w: _reference(x, w, group_sizes)  # noqa: E731
        out = jax.jit(turbo)(lhs, rhs)
        d_lhs, d_rhs = jax.jit(jax.grad(lambda x, w: loss(turbo, x, w), argnums=(0, 1)))(lhs, rhs)
        ref_out = ref(lhs, rhs)
        ref_d_lhs, ref_d_rhs = jax.grad(lambda x, w: loss(ref, x, w), argnums=(0, 1))(lhs, rhs)

        assert out.dtype == jnp.bfloat16
        assert _rel_err(out, ref_out) < 1e-2
        assert _rel_err(d_lhs, ref_d_lhs) < 1e-2
        assert _rel_err(d_rhs, ref_d_rhs) < 1e-2
        # Rows past the last group are padding and must come back as exact zeros.
        assert float(jnp.max(jnp.abs(out[valid:].astype(jnp.float32)), initial=0.0)) == 0.0
        assert float(jnp.max(jnp.abs(d_lhs[valid:].astype(jnp.float32)), initial=0.0)) == 0.0

    @pytest.mark.parametrize("backend", _backend_params(["ck", "hipblaslt", "flydsl"]))
    def test_balanced(self, backend):
        self._check(backend, [256] * 8, m=2048, k=256, n=512)

    @pytest.mark.parametrize("backend", _backend_params(["ck", "hipblaslt", "flydsl"]))
    def test_empty_groups_and_padding(self, backend):
        self._check(backend, [0, 300, 17, 0, 512, 1, 200, 0], m=1536, k=256, n=384)

    @pytest.mark.parametrize("backend", _backend_params(["hipblaslt", "flydsl"]))
    def test_single_group(self, backend):
        self._check(backend, [1000], m=1024, k=512, n=256)

    @requires_gfx950
    def test_flydsl_skewed_many_groups(self):
        sizes = [0] * 31 + [3000] + [1] * 32
        self._check("flydsl", sizes, m=3200, k=192, n=640)

    def test_rejects_float32(self):
        x = jnp.zeros((16, 8), jnp.float32)
        w = jnp.zeros((2, 8, 8), jnp.float32)
        with pytest.raises(ValueError):
            turbo_grouped_gemm.gmm(x, w, jnp.array([8, 8], jnp.int32))
