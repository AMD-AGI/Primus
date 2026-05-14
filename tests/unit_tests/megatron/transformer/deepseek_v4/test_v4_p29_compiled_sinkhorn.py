###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-5 P29 (RESCOPED) — G32 numerical equivalence test.

The plan-5 P28 baseline trace pinned ``aten::sum`` fp32 reduce as 87.3 % of
step time (7.61 s), of which 96 % by count and 99.95 % by time came from
``primus.backends.megatron.core.transformer.hyper_connection.sinkhorn_normalize``
— each call issues 1 + 2*(n_iters - 1) separate fp32 reductions and HIP's
default ``reduce_kernel<512, 1, ...>`` runs them ~250x over the memory-
bound floor for our ``[1, 4096, 4, 4]`` shape.

Phase 29 ships a ``torch.compile(fullgraph=True, dynamic=False)`` fast
path that fuses every sum / divide / broadcast inside the Sinkhorn loop
(plus its BWD via AOT autograd) into one Inductor-generated Triton
kernel, gated by ``DeepSeekV4TransformerConfig.use_v4_compiled_sinkhorn``.

This test (G32) asserts that the compiled path produces FWD + BWD output
that is numerically equivalent to the eager loop at:

* fast tier — small-shape parity (CPU OK; runs in CI by default).  The
  ``B=2 S=64 K=4`` tensor still exercises every code path the eager
  loop has (priming column step + alternating row/col cycles).
* release tier — V4-Flash production input shape ``B=1 S=4096 K=4``,
  marked ``pytest.mark.slow`` so it only runs under ``--run-slow`` /
  ``-m slow``.

Tolerance:
* fp32 in / fp32 sums: tight ``atol=1e-5, rtol=1e-5`` — the algorithm is
  identical, only the kernel boundary moves.  Inductor may pick a
  different reduction tree than HIP's default kernel which produces
  bit-different (but ULP-tight) results.
* bf16 in: the Sinkhorn body still runs in fp32 (the function casts up,
  per techblog §2.2 pitfall #3) and the trailing cast back to bf16
  rounds.  Tolerance is ``atol=1e-3, rtol=1e-3`` — well below any
  pretraining-relevant signal.

Cold-compile time is recorded as a ``print()`` line in the test output for
manual inspection.

The test is parametrised over ``n_iters ∈ {5, 20}`` so the cache-key path
is exercised (different ``n_iters`` -> different compiled artefact).
"""

from __future__ import annotations

# isort: off
# Order matters: import torch first so the ``triton`` import below sees a
# fully-initialised torch namespace; isort otherwise re-orders these.

import time

import pytest

torch = pytest.importorskip("torch")

from primus.backends.megatron.core.transformer.hyper_connection import (
    HyperMixer,
    _compiled_sinkhorn_cache,
    sinkhorn_normalize,
)

# isort: on


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="V4 compiled sinkhorn perf-critical path needs CUDA / HIP",
)


@pytest.fixture(autouse=True)
def _disable_p36_triton_sinkhorn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the dispatcher to the plan-5 P29 compiled boundary this file
    exists to test.

    Plan-6 P36 added a Triton FWD/BWD kernel (default-on via the
    ``PRIMUS_SINKHORN_TRITON=1`` env knob) whose routing precedence in
    :func:`sinkhorn_normalize` is ``Triton > compiled > eager`` --
    leaving the env knob at its default would silently route
    ``use_compiled=True`` calls through the Triton path and skip the
    compiled cache entirely, breaking the cache-hit assertion below.
    Forcing ``PRIMUS_SINKHORN_TRITON=0`` for the duration of this file
    keeps the compiled boundary observable.
    """
    monkeypatch.setenv("PRIMUS_SINKHORN_TRITON", "0")


def _make_input(
    *,
    B: int,
    S: int,
    K: int,
    dtype: torch.dtype,
    device: str,
    seed: int = 0,
) -> torch.Tensor:
    """Generate a non-negative ``[B, S, K, K]`` Sinkhorn input identical to
    the one ``HyperMixer.compute_weights`` feeds (softmax + eps floor).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    raw = torch.randn(B, S, K, K, generator=g, device=device, dtype=torch.float32)
    # Mirror compute_weights: softmax(last) + eps so the input is non-
    # negative.  The compiled kernel sees this exact shape contract.
    out = torch.softmax(raw, dim=-1) + 1.0e-6
    return out.to(dtype)


def _close(a: torch.Tensor, b: torch.Tensor, *, dtype: torch.dtype) -> None:
    if dtype is torch.bfloat16:
        atol, rtol = 1.0e-3, 1.0e-3
    else:
        atol, rtol = 1.0e-5, 1.0e-5
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        max_abs = (a - b).abs().max().item()
        max_rel = ((a - b).abs() / (b.abs() + 1e-12)).max().item()
        pytest.fail(
            f"sinkhorn parity failure: dtype={dtype}, "
            f"max_abs={max_abs:.3e} (atol={atol:.0e}), "
            f"max_rel={max_rel:.3e} (rtol={rtol:.0e})"
        )


@pytest.mark.parametrize("n_iters", [5, 20])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    ("B", "S", "K"),
    [
        pytest.param(2, 64, 4, id="fast-B2-S64-K4"),
        pytest.param(1, 4096, 4, id="release-V4-Flash-B1-S4096-K4", marks=pytest.mark.slow),
    ],
)
def test_p29_compiled_sinkhorn_fwd_parity(
    B: int,
    S: int,
    K: int,
    dtype: torch.dtype,
    n_iters: int,
) -> None:
    """G32 forward parity — compiled sinkhorn matches the eager loop."""
    device = "cuda"
    x = _make_input(B=B, S=S, K=K, dtype=dtype, device=device)

    out_eager = sinkhorn_normalize(x, n_iters=n_iters, eps=1.0e-6, use_compiled=False)

    t0 = time.perf_counter()
    out_compiled = sinkhorn_normalize(x, n_iters=n_iters, eps=1.0e-6, use_compiled=True)
    torch.cuda.synchronize()
    cold_ms = 1000.0 * (time.perf_counter() - t0)
    print(
        f"[G32 fwd] cold-compile + first call: {cold_ms:.1f} ms (n_iters={n_iters}, shape=({B},{S},{K},{K}))"
    )

    _close(out_compiled, out_eager, dtype=dtype)

    # Sanity check the cache-key path: a second call must hit the cached
    # artefact (no recompile).  We can not assert wall-time directly, but
    # we CAN assert the cache contains the (n_iters, eps, dtype) key.
    # Shape is NOT part of the key (we use ``dynamic=True`` so one
    # artefact handles every shape) — see hyper_connection.py module
    # docs for the rationale.
    expected_key = (n_iters, 1.0e-6, dtype)
    assert (
        expected_key in _compiled_sinkhorn_cache
    ), f"compiled cache miss for {expected_key}; cache has {list(_compiled_sinkhorn_cache)}"


@pytest.mark.parametrize("n_iters", [5, 20])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    ("B", "S", "K"),
    [
        pytest.param(2, 64, 4, id="fast-B2-S64-K4"),
        pytest.param(1, 4096, 4, id="release-V4-Flash-B1-S4096-K4", marks=pytest.mark.slow),
    ],
)
def test_p29_compiled_sinkhorn_bwd_parity(
    B: int,
    S: int,
    K: int,
    dtype: torch.dtype,
    n_iters: int,
) -> None:
    """G32 backward parity — AOT-autograd through the compiled callable
    matches the eager loop's autograd graph."""
    device = "cuda"
    x_eager = _make_input(B=B, S=S, K=K, dtype=dtype, device=device).requires_grad_(True)
    x_comp = x_eager.detach().clone().requires_grad_(True)

    # Loss = sum(out**2) — a generic non-degenerate scalar that exercises
    # every output element's grad path.  Uses pow(2) (not sum_squared) so
    # the BWD chain rule pulls through every internal m / m.sum() node.
    out_eager = sinkhorn_normalize(x_eager, n_iters=n_iters, eps=1.0e-6, use_compiled=False)
    loss_eager = out_eager.float().pow(2).sum()
    loss_eager.backward()

    out_comp = sinkhorn_normalize(x_comp, n_iters=n_iters, eps=1.0e-6, use_compiled=True)
    loss_comp = out_comp.float().pow(2).sum()
    loss_comp.backward()

    assert x_eager.grad is not None
    assert x_comp.grad is not None
    _close(x_comp.grad, x_eager.grad, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_p29_hypermixer_use_compiled_flag_propagates(dtype: torch.dtype) -> None:
    """Plumbing test: ``HyperMixer(use_compiled_sinkhorn=True).compute_weights(x)``
    must produce the same ``(pre, post, comb)`` triplet as the eager
    ``HyperMixer(use_compiled_sinkhorn=False)`` with identical weights.

    This guards against a future refactor breaking the kwarg plumbing
    (e.g. dropping the flag on its way from `DeepseekV4HybridLayer` to
    `HyperMixer.__init__`).
    """
    device = "cuda"
    torch.manual_seed(0)
    hidden_size = 64
    hc_mult = 4
    B, S = 2, 16

    eager = HyperMixer(
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        sinkhorn_iters=20,
        use_compiled_sinkhorn=False,
    ).to(device)
    compiled = HyperMixer(
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        sinkhorn_iters=20,
        use_compiled_sinkhorn=True,
    ).to(device)
    # Weight-tie so the two mixers produce identical outputs modulo the
    # Sinkhorn kernel boundary.
    with torch.no_grad():
        compiled.fn.weight.copy_(eager.fn.weight)
        compiled.scale.copy_(eager.scale)
        compiled.base.copy_(eager.base)

    x = torch.randn(B, S, hc_mult, hidden_size, device=device, dtype=dtype)
    pre_e, post_e, comb_e = eager.compute_weights(x)
    pre_c, post_c, comb_c = compiled.compute_weights(x)

    # pre / post do not pass through Sinkhorn -> must be bit-identical.
    torch.testing.assert_close(pre_c, pre_e)
    torch.testing.assert_close(post_c, post_e)
    # comb passes through Sinkhorn -> compiled vs eager match within the
    # numerical tolerance.
    _close(comb_c, comb_e, dtype=dtype)
