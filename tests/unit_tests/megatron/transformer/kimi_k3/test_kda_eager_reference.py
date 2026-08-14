###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Numerical tests for the eager KDA reference kernels.

Coverage:

* chunked form vs. an **independently written** literal ``O(T)`` loop
  (:func:`kda_reference_impls.naive_kda_loop`) — the cheapest way to
  catch a chunking bug;
* chunked form vs. this package's own sequential reference, across chunk
  sizes, so a chunk-size-dependent bug cannot hide;
* causality — the output at ``t`` must not move when inputs after ``t``
  change;
* ``initial_state`` / ``output_final_state`` round-trip: running two
  halves with a carried state must equal one long run;
* numerical behaviour at long sequence length and in bf16 vs fp32;
* gradient correctness via ``torch.autograd.gradcheck`` in float64;
* the gate and L2-norm transforms against their published formulas.
"""

from __future__ import annotations

import pytest
import torch

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
    eager_chunk_kda,
    eager_recurrent_kda,
    kda_gate,
    kda_l2norm,
)
from tests.unit_tests.megatron.transformer.kimi_k3.kda_reference_impls import (
    assert_close_scaled,
    hf_kda_gate,
    hf_l2norm,
    naive_kda_loop,
)

# fp32 accumulation over a few hundred sequential steps, compared against a
# float64 oracle or a differently-ordered fp32 computation. Stated as
# max|delta| / max|reference|.
FP32_TOL = 1e-5


def _random_kda_inputs(
    batch,
    seq_len,
    num_heads,
    k_dim,
    v_dim,
    *,
    device,
    dtype=torch.float32,
    seed=0,
    lower_bound=-5.0,
):
    """Inputs in the ranges the real module produces.

    ``g`` is a *per-channel* bounded log-decay in ``(lower_bound, 0)`` and
    ``beta`` is a sigmoid, exactly as ``KimiDeltaAttention`` supplies them.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    q = rnd(batch, seq_len, num_heads, k_dim).to(device=device, dtype=dtype)
    k = rnd(batch, seq_len, num_heads, k_dim).to(device=device, dtype=dtype)
    v = rnd(batch, seq_len, num_heads, v_dim).to(device=device, dtype=dtype)
    g = (lower_bound * torch.sigmoid(rnd(batch, seq_len, num_heads, k_dim))).to(device)
    beta = torch.sigmoid(rnd(batch, seq_len, num_heads)).to(device)
    return q, k, v, g, beta


# ---------------------------------------------------------------------------
# Chunked vs. the literal recurrence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch,seq_len,num_heads,k_dim,v_dim",
    [
        (2, 32, 2, 16, 16),  # shorter than one chunk -> all padding
        (1, 64, 2, 16, 24),  # exactly one chunk, K != V
        (2, 130, 3, 24, 16),  # ragged, several chunks
    ],
    ids=["sub_chunk", "one_chunk", "ragged"],
)
def test_chunked_matches_independent_naive_loop(batch, seq_len, num_heads, k_dim, v_dim, kda_device):
    """The chunked form must equal a literal float64 triple loop."""
    q, k, v, g, beta = _random_kda_inputs(batch, seq_len, num_heads, k_dim, v_dim, device=kda_device, seed=11)
    chunk_out, chunk_state = eager_chunk_kda(
        q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True, chunk_size=64
    )
    naive_out, naive_state = naive_kda_loop(kda_l2norm(q), kda_l2norm(k), v, g, beta)

    tag = f"chunked vs naive {batch}x{seq_len}x{num_heads}x{k_dim}x{v_dim}"
    assert_close_scaled(chunk_out.double(), naive_out, FP32_TOL, f"{tag} out")
    assert_close_scaled(chunk_state.double(), naive_state, FP32_TOL, f"{tag} state")


def test_sequential_matches_independent_naive_loop(kda_device):
    """Pin the sequential reference itself, so it can serve as an oracle."""
    q, k, v, g, beta = _random_kda_inputs(2, 96, 3, 32, 32, device=kda_device, seed=12)
    seq_out, seq_state = eager_recurrent_kda(
        q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    naive_out, naive_state = naive_kda_loop(kda_l2norm(q), kda_l2norm(k), v, g, beta)
    assert_close_scaled(seq_out.double(), naive_out, FP32_TOL, "sequential vs naive out")
    assert_close_scaled(seq_state.double(), naive_state, FP32_TOL, "sequential vs naive state")


@pytest.mark.parametrize("chunk_size", [16, 32, 64, 128])
def test_chunked_is_chunk_size_invariant(chunk_size, kda_device):
    """The result must not depend on the chunk tiling."""
    q, k, v, g, beta = _random_kda_inputs(2, 192, 2, 32, 32, device=kda_device, seed=13)
    chunk_out, _ = eager_chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True, chunk_size=chunk_size)
    seq_out, _ = eager_recurrent_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    assert_close_scaled(chunk_out, seq_out, FP32_TOL, f"chunk_size={chunk_size} vs sequential")


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [eager_chunk_kda, eager_recurrent_kda], ids=["chunked", "sequential"])
def test_output_at_t_is_independent_of_the_future(fn, kda_device):
    """Perturb every input strictly after ``cut``; outputs up to ``cut`` must not move."""
    batch, seq_len, num_heads, k_dim = 2, 128, 3, 32
    cut = 77
    q, k, v, g, beta = _random_kda_inputs(batch, seq_len, num_heads, k_dim, k_dim, device=kda_device, seed=14)
    base, _ = fn(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)

    gen = torch.Generator(device="cpu").manual_seed(15)
    q2, k2, v2, g2, beta2 = (x.clone() for x in (q, k, v, g, beta))
    tail = slice(cut + 1, seq_len)
    n_tail = seq_len - cut - 1
    q2[:, tail] = torch.randn(batch, n_tail, num_heads, k_dim, generator=gen).to(q)
    k2[:, tail] = torch.randn(batch, n_tail, num_heads, k_dim, generator=gen).to(k)
    v2[:, tail] = torch.randn(batch, n_tail, num_heads, k_dim, generator=gen).to(v)
    g2[:, tail] = (-5.0 * torch.sigmoid(torch.randn(batch, n_tail, num_heads, k_dim, generator=gen))).to(g)
    beta2[:, tail] = torch.sigmoid(torch.randn(batch, n_tail, num_heads, generator=gen)).to(beta)

    perturbed, _ = fn(q2, k2, v2, g2, beta2, use_qk_l2norm_in_kernel=True)

    prefix_err = (base[:, : cut + 1] - perturbed[:, : cut + 1]).abs().max().item()
    suffix_err = (base[:, cut + 1 :] - perturbed[:, cut + 1 :]).abs().max().item()
    print(f"[causality {fn.__name__}] prefix max|d|={prefix_err:.3e}  suffix max|d|={suffix_err:.3e}")
    torch.testing.assert_close(base[:, : cut + 1], perturbed[:, : cut + 1], rtol=0, atol=0)
    # sanity: the perturbation must actually have done something
    assert suffix_err > 1e-3, "the future-perturbation had no effect; the test would be vacuous"


# ---------------------------------------------------------------------------
# State carry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [eager_chunk_kda, eager_recurrent_kda], ids=["chunked", "sequential"])
def test_split_run_with_carried_state_equals_one_long_run(fn, kda_device):
    """``initial_state`` / ``output_final_state`` must compose."""
    batch, seq_len, num_heads, k_dim = 2, 128, 2, 32
    split = 64
    q, k, v, g, beta = _random_kda_inputs(batch, seq_len, num_heads, k_dim, k_dim, device=kda_device, seed=16)

    full, full_state = fn(q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
    first, mid_state = fn(
        q[:, :split],
        k[:, :split],
        v[:, :split],
        g[:, :split],
        beta[:, :split],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    second, end_state = fn(
        q[:, split:],
        k[:, split:],
        v[:, split:],
        g[:, split:],
        beta[:, split:],
        initial_state=mid_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    joined = torch.cat([first, second], dim=1)
    assert_close_scaled(joined, full, FP32_TOL, f"state carry {fn.__name__} out")
    assert_close_scaled(end_state, full_state, FP32_TOL, f"state carry {fn.__name__} state")


# ---------------------------------------------------------------------------
# Stability: long sequences and bf16
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [512, 2048])
def test_long_sequence_stays_finite_and_matches_the_sequential_form(seq_len, kda_device):
    """The chunked form must not drift or overflow over many chunks.

    This is where an implementation that divides by the cumulative
    retention ``Γ`` instead of taking differences of cumulative log-decays
    blows up: with ``g >= -5`` and ``C = 64``, ``1 / Γ`` reaches
    ``exp(320)``, well past the fp32 range.
    """
    q, k, v, g, beta = _random_kda_inputs(1, seq_len, 2, 32, 32, device=kda_device, seed=17)
    chunk_out, chunk_state = eager_chunk_kda(
        q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    assert torch.isfinite(chunk_out).all(), "chunked output has non-finite entries"
    assert torch.isfinite(chunk_state).all(), "carried state has non-finite entries"

    seq_out, _ = eager_recurrent_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    assert_close_scaled(chunk_out, seq_out, FP32_TOL, f"T={seq_len} chunked vs sequential")


def test_bf16_inputs_track_the_fp32_result(kda_device):
    """bf16 in / bf16 out, with the internal compute in fp32.

    bf16 has ~3 decimal digits, so the tolerance is loose by construction;
    what matters is that the error is *bounded by the input quantisation*
    and not by an unstable algorithm.
    """
    if kda_device != "cuda":
        pytest.skip("bf16 comparison is only meaningful on the accelerator")
    q, k, v, g, beta = _random_kda_inputs(2, 256, 2, 32, 32, device=kda_device, seed=18)
    ref, _ = eager_chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)

    q16, k16, v16 = (x.bfloat16() for x in (q, k, v))
    got, _ = eager_chunk_kda(q16, k16, v16, g, beta, use_qk_l2norm_in_kernel=True)
    assert got.dtype == torch.bfloat16, "output dtype must follow v.dtype"
    assert torch.isfinite(got).all()

    # baseline: what the input rounding alone costs, measured through the
    # sequential reference on the same rounded inputs.
    seq16, _ = eager_recurrent_kda(q16, k16, v16, g, beta, use_qk_l2norm_in_kernel=True)
    scale = ref.abs().max().item()
    quant = (seq16.float() - ref).abs().max().item()
    err = (got.float() - ref).abs().max().item()
    print(f"[bf16] |out|max={scale:.3e}  chunked err={err:.3e}  input-quantisation err={quant:.3e}")
    assert err < 5e-2 * scale, f"bf16 chunked error {err:.3e} is large relative to |out|max {scale:.3e}"
    assert err < max(4.0 * quant, 1e-3), (
        f"bf16 chunked error {err:.3e} exceeds what input quantisation alone explains "
        f"({quant:.3e}); the chunked algorithm is losing accuracy of its own."
    )


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [eager_chunk_kda, eager_recurrent_kda], ids=["chunked", "sequential"])
def test_gradcheck_float64(fn, kda_device):
    """``torch.autograd.gradcheck`` on a tiny instance in float64."""
    batch, seq_len, num_heads, k_dim, v_dim, chunk = 1, 12, 2, 4, 3, 4
    gen = torch.Generator(device="cpu").manual_seed(19)

    def leaf(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float64).to(kda_device).requires_grad_(True)

    q = leaf(batch, seq_len, num_heads, k_dim)
    k = leaf(batch, seq_len, num_heads, k_dim)
    v = leaf(batch, seq_len, num_heads, v_dim)
    g_raw = leaf(batch, seq_len, num_heads, k_dim)
    beta_raw = leaf(batch, seq_len, num_heads)

    kwargs = {"use_qk_l2norm_in_kernel": True}
    if fn is eager_chunk_kda:
        kwargs["chunk_size"] = chunk

    def f(q_, k_, v_, g_, b_):
        # feed the gate/beta through their real activations so the tested
        # graph is the one training actually builds
        out, _ = fn(q_, k_, v_, -5.0 * torch.sigmoid(g_), torch.sigmoid(b_), **kwargs)
        return out

    assert torch.autograd.gradcheck(f, (q, k, v, g_raw, beta_raw), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_backward_populates_every_input_grad(kda_device):
    """A plain ``sum().backward()`` must reach all five inputs."""
    q, k, v, g, beta = _random_kda_inputs(2, 96, 2, 16, 16, device=kda_device, seed=20)
    q, k, v, g, beta = (x.clone().requires_grad_(True) for x in (q, k, v, g, beta))
    out, _ = eager_chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    out.float().sum().backward()
    for name, t in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        assert t.grad is not None, f"{name}.grad was not populated"
        assert torch.isfinite(t.grad).all(), f"{name}.grad has non-finite entries"
        assert t.grad.abs().max() > 0, f"{name}.grad is all zeros"


# ---------------------------------------------------------------------------
# The input transforms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lower_bound", [None, -5.0, -2.0])
def test_gate_matches_the_published_formula(lower_bound, kda_device):
    """:func:`kda_gate` vs an independent transcription of ``fla``'s naive gate."""
    num_heads, head_dim = 4, 16
    gen = torch.Generator(device="cpu").manual_seed(21)
    z = torch.randn(2, 32, num_heads, head_dim, generator=gen).to(kda_device)
    A_log = torch.randn(num_heads, generator=gen).to(kda_device)
    dt_bias = torch.randn(num_heads * head_dim, generator=gen).to(kda_device)

    got = kda_gate(z, A_log, dt_bias, lower_bound)
    want = hf_kda_gate(z, A_log, dt_bias, lower_bound)
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    assert (got <= 0).all(), "log-decay must be non-positive"
    if lower_bound is not None:
        assert (got > lower_bound).all(), f"bounded gate must stay above {lower_bound}"


def test_l2norm_matches_the_published_formula(kda_device):
    gen = torch.Generator(device="cpu").manual_seed(22)
    x = torch.randn(2, 16, 3, 8, generator=gen).to(kda_device)
    torch.testing.assert_close(kda_l2norm(x), hf_l2norm(x), rtol=1e-6, atol=1e-6)
    # a normalised row has unit norm up to the epsilon
    norms = kda_l2norm(x).pow(2).sum(-1)
    assert (norms - 1.0).abs().max() < 1e-4


# ---------------------------------------------------------------------------
# Contract violations
# ---------------------------------------------------------------------------


def test_shape_validation(kda_device):
    q, k, v, g, beta = _random_kda_inputs(2, 32, 2, 16, 16, device=kda_device, seed=23)
    with pytest.raises(ValueError, match="q, k, g must share a shape"):
        eager_chunk_kda(q, k[:, :, :, :8], v, g, beta)
    with pytest.raises(ValueError, match="beta must be"):
        eager_chunk_kda(q, k, v, g, beta[:, :, :1])
