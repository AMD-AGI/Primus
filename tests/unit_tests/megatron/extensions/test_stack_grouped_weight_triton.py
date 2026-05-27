###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Plan-6 P34 G37 — `stack_grouped_weight` Triton FWD/BWD equivalence to eager.

Asserts that :class:`StackGroupedWeightFn` (Triton kernel from
``primus.backends.megatron.core.extensions._triton.stack_grouped_weight``)
produces ``[E, N, K]`` layout-equivalent results to the eager
``torch.stack(weights, dim=0).transpose(1, 2).contiguous()`` chain, FWD
**and** BWD, at two tiers:

* fast tier — ``E=4, K=8, N=8, fp32`` shapes exercising every code path
  in the kernel (per-expert pointer dispatch, tile-level transpose,
  per-axis masking for non-multiple-of-BLOCK shapes) in milliseconds;
* release tier — ``E=32, K=4096, N=2048, bf16`` shapes mirroring the
  V4-Flash EP=8 proxy widths, behind ``pytest.mark.slow``.

The operation is a pure layout transform — no fp arithmetic — so both
FWD and BWD must be **bit-equal** at every dtype (``atol=0``).  This is
stricter than the typical "elementwise within rtol" attention parity
check and is the right bar because no rounding is involved.

GPU-only; CPU runs are ``pytest.skip``-ed at module collection time.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import List

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip(
        "stack_grouped_weight Triton kernel requires CUDA / HIP",
        allow_module_level=True,
    )

pytest.importorskip("triton", reason="Triton not installed")

from primus.backends.megatron.core.extensions._triton.stack_grouped_weight import (  # noqa: E402
    StackGroupedWeightFn,
    eager_stack_grouped_weight,
    is_triton_path_enabled,
    stack_grouped_weight,
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


def _random_weights(
    *, E: int, K: int, N: int, dtype: torch.dtype, requires_grad: bool = False, seed: int = 0
) -> List[torch.Tensor]:
    """Build ``E`` random ``[K, N]`` tensors on CUDA with the given dtype."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.randn(K, N, dtype=dtype, device="cuda", generator=gen, requires_grad=requires_grad)
        for _ in range(E)
    ]


# ---------------------------------------------------------------------------
# G37 — FWD bit-equal
# ---------------------------------------------------------------------------


class TestG37ForwardBitEqual:
    """Forward output must be bit-equal to ``torch.stack + transpose +
    contiguous`` (the operation is a pure layout transform; no rounding).
    """

    @pytest.mark.parametrize(
        "E, K, N, dtype",
        [
            (4, 8, 8, torch.float32),
            (1, 7, 13, torch.float32),  # prime dims to stress masking
            (3, 9, 9, torch.float16),
            (3, 9, 9, torch.bfloat16),
            (8, 32, 17, torch.bfloat16),  # uneven N (last tile partial)
        ],
    )
    def test_fast_tier_bit_equal(self, E, K, N, dtype):
        weights = _random_weights(E=E, K=K, N=N, dtype=dtype, seed=42 + E + K + N)
        out_triton = StackGroupedWeightFn.apply(*weights)
        out_eager = eager_stack_grouped_weight(weights)

        assert out_triton.shape == out_eager.shape == (E, N, K)
        assert out_triton.dtype is dtype
        assert out_triton.is_contiguous()
        assert torch.equal(out_triton, out_eager), (
            "stack_grouped_weight FWD must be bit-equal to eager "
            f"(diff_max={(out_triton.float() - out_eager.float()).abs().max().item()})"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("mode", ["fc1", "fc2"])
    def test_release_tier_bit_equal_v4_flash_widths(self, mode):
        """V4-Flash EP=8 widths — the actual shapes hit in the EP8 proxy."""
        if mode == "fc1":
            E, K, N = 32, 4096, 4096  # fc1: out=2*ffn=4096, in=hidden=4096
        else:
            E, K, N = 32, 4096, 2048  # fc2: out=hidden=4096, in=ffn=2048
        weights = _random_weights(E=E, K=K, N=N, dtype=torch.bfloat16, seed=0)
        out_triton = StackGroupedWeightFn.apply(*weights)
        out_eager = eager_stack_grouped_weight(weights)
        assert torch.equal(out_triton, out_eager)


# ---------------------------------------------------------------------------
# G37 — BWD bit-equal + gradcheck
# ---------------------------------------------------------------------------


class TestG37BackwardBitEqual:
    """BWD must scatter gradient back to each ``weight{i}.grad`` such that
    the result equals the eager VJP (``torch.stack + transpose + contiguous``
    has an exact bijection VJP — no rounding).
    """

    @pytest.mark.parametrize(
        "E, K, N, dtype",
        [
            (4, 8, 8, torch.float32),
            (3, 9, 9, torch.bfloat16),
            (8, 32, 17, torch.bfloat16),
        ],
    )
    def test_grad_bit_equal_vs_eager(self, E, K, N, dtype):
        # Build two parallel sets of weights so each forward owns its own
        # autograd graph; copy the values so the grads are computed against
        # the same input distribution.
        weights_eager = _random_weights(E=E, K=K, N=N, dtype=dtype, requires_grad=True, seed=11 + E + K + N)
        weights_triton = [w.detach().clone().requires_grad_(True) for w in weights_eager]

        out_eager = eager_stack_grouped_weight(weights_eager)
        out_triton = StackGroupedWeightFn.apply(*weights_triton)

        # Use a deterministic random upstream gradient.
        gen = torch.Generator(device="cuda").manual_seed(7 + E + K + N)
        dout = torch.randn(E, N, K, dtype=dtype, device="cuda", generator=gen)

        out_eager.backward(dout)
        out_triton.backward(dout)

        for i, (we, wt) in enumerate(zip(weights_eager, weights_triton)):
            assert we.grad is not None
            assert wt.grad is not None
            assert torch.equal(we.grad, wt.grad), (
                f"weight{i}.grad differs between eager and Triton paths; "
                f"diff_max={(we.grad.float() - wt.grad.float()).abs().max().item()}"
            )

    def test_gradcheck_fast_tier(self):
        """`torch.autograd.gradcheck` at the fast tier (fp64 for numerical
        precision; gradcheck requires fp64 to keep the central-difference
        epsilon meaningful).
        """
        E, K, N = 3, 5, 7
        weights = _random_weights(E=E, K=K, N=N, dtype=torch.float64, requires_grad=True, seed=99)
        torch.autograd.gradcheck(
            StackGroupedWeightFn.apply,
            weights,
            eps=1e-6,
            atol=1e-7,
            rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# G37 — non-contiguous / shape-mismatch error paths
# ---------------------------------------------------------------------------


class TestG37ErrorPaths:
    """Plan-6 P34 design notes call out three defensive preconditions:

    * shape mismatch across experts;
    * dtype / device mismatch;
    * non-contiguous input (Megatron allocator always returns contiguous,
      so non-contiguous indicates a bug upstream).

    Each should raise a clear error rather than silently miscompiling.
    """

    def test_shape_mismatch_across_experts(self):
        weights = _random_weights(E=2, K=8, N=8, dtype=torch.float32)
        weights.append(torch.randn(8, 9, dtype=torch.float32, device="cuda"))
        with pytest.raises(ValueError, match="differs from weight0.shape"):
            StackGroupedWeightFn.apply(*weights)

    def test_dtype_mismatch_across_experts(self):
        weights = _random_weights(E=2, K=8, N=8, dtype=torch.float32)
        weights.append(torch.randn(8, 8, dtype=torch.float16, device="cuda"))
        with pytest.raises(TypeError, match="differs from weight0.dtype"):
            StackGroupedWeightFn.apply(*weights)

    def test_non_contiguous_input(self):
        # Slice a larger tensor to produce a non-contiguous [K, N] view.
        big = torch.randn(8, 16, dtype=torch.float32, device="cuda")
        non_contig = big[:, ::2]  # [8, 8] but stride-2 along the last axis
        assert not non_contig.is_contiguous()
        contig = torch.randn(8, 8, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="must be contiguous"):
            StackGroupedWeightFn.apply(non_contig, contig)

    def test_empty_weights_list(self):
        with pytest.raises(ValueError, match="at least one weight"):
            StackGroupedWeightFn.apply()


# ---------------------------------------------------------------------------
# Env-flag dispatch
# ---------------------------------------------------------------------------


class TestEnvFlagDispatch:
    """``PRIMUS_STACK_GROUPED_WEIGHT_TRITON`` controls the dispatch between
    the Triton autograd Function and the eager chain.  Default-on; ``"0"``
    means eager; any other value (including unset) means Triton.
    """

    def test_default_on(self):
        with _env("PRIMUS_STACK_GROUPED_WEIGHT_TRITON", None):
            assert is_triton_path_enabled() is True

    def test_explicit_one_is_on(self):
        with _env("PRIMUS_STACK_GROUPED_WEIGHT_TRITON", "1"):
            assert is_triton_path_enabled() is True

    def test_explicit_zero_is_off(self):
        with _env("PRIMUS_STACK_GROUPED_WEIGHT_TRITON", "0"):
            assert is_triton_path_enabled() is False

    def test_dispatch_off_routes_through_eager(self):
        weights = _random_weights(E=3, K=8, N=8, dtype=torch.float32)
        with _env("PRIMUS_STACK_GROUPED_WEIGHT_TRITON", "0"):
            out_off = stack_grouped_weight(weights)
        with _env("PRIMUS_STACK_GROUPED_WEIGHT_TRITON", "1"):
            out_on = stack_grouped_weight(weights)
        # Both routes are layout transforms; outputs must be bit-equal.
        assert torch.equal(out_off, out_on)


# ---------------------------------------------------------------------------
# Sanity: per-expert pointer bijection
# ---------------------------------------------------------------------------


class TestPerExpertBijection:
    """The kernel writes ``out[e, n, k] = weights[e][k, n]``.  Build a
    distinctive sentinel pattern per expert (so a misaligned pointer
    would be caught loudly) and verify the output reflects the bijection.
    """

    def test_distinct_per_expert_pattern(self):
        E, K, N = 4, 5, 6
        weights: List[torch.Tensor] = []
        for e in range(E):
            # Each expert's weight[k, n] = e * 1000 + k * 100 + n.  Distinct
            # for every (e, k, n); any cross-expert pointer aliasing would
            # show up as a wrong sentinel in the output.
            w = (
                torch.arange(K * N, dtype=torch.float32, device="cuda").view(K, N).clone()
                + e * 1000
                + torch.arange(K, dtype=torch.float32, device="cuda").view(K, 1) * 99
            )
            weights.append(w)

        out = StackGroupedWeightFn.apply(*weights)
        for e in range(E):
            for k in range(K):
                for n in range(N):
                    assert out[e, n, k].item() == weights[e][k, n].item(), (
                        f"Bijection broken at (e={e}, n={n}, k={k}): "
                        f"out={out[e, n, k].item()} vs "
                        f"weight={weights[e][k, n].item()}"
                    )
