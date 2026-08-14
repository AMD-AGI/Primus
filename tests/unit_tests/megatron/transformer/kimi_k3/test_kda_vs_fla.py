###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Cross-checks of the eager KDA reference against ``flash-linear-attention``.

``fla`` is the upstream home of the KDA kernels the HF model calls, so
agreeing with it is strong evidence that the eager reference implements
*the* KDA and not merely a self-consistent variant. Every test here skips
cleanly when ``fla`` is absent.

Two of these tests also pin down version hazards that the adapter in
:mod:`...kda_kernels._fla` exists to neutralise:

* :func:`test_fla_chunk_kda_does_not_activate_beta_itself` records
  whether the installed ``chunk_kda`` honours the
  ``use_beta_sigmoid_in_kernel=True`` that HF's
  ``modeling_kimi_linear.py:622`` passes. No released signature declares
  it, so it lands in ``**kwargs`` and is dropped.
* :func:`test_fla_naive_recurrent_kda_matches_our_sequential_reference`
  compares against ``fla``'s own eager reference, which is the least
  ambiguous statement of the recurrence available upstream.
"""

from __future__ import annotations

import pytest
import torch

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
    eager_chunk_kda,
    eager_recurrent_kda,
    kda_gate,
    kda_l2norm,
    resolve_kda_backend,
)
from tests.unit_tests.megatron.transformer.kimi_k3.kda_reference_impls import (
    hf_causal_conv1d_silu,
    hf_rms_norm_gated,
)

pytest.importorskip("fla", reason="flash-linear-attention is not installed")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fla's kernels are Triton and require an accelerator"
)


def _inputs(batch, seq_len, num_heads, k_dim, v_dim, *, dtype=torch.float32, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).cuda()

    q = rnd(batch, seq_len, num_heads, k_dim).to(dtype)
    k = rnd(batch, seq_len, num_heads, k_dim).to(dtype)
    v = rnd(batch, seq_len, num_heads, v_dim).to(dtype)
    g = -5.0 * torch.sigmoid(rnd(batch, seq_len, num_heads, k_dim))
    beta = torch.sigmoid(rnd(batch, seq_len, num_heads))
    return q, k, v, g, beta


# ---------------------------------------------------------------------------
# fla's own eager reference
# ---------------------------------------------------------------------------


@requires_cuda
def test_fla_naive_recurrent_kda_matches_our_sequential_reference():
    """``fla.ops.kda.naive.naive_recurrent_kda`` vs :func:`eager_recurrent_kda`."""
    naive = pytest.importorskip("fla.ops.kda.naive", reason="fla build has no naive KDA reference")
    q, k, v, g, beta = _inputs(2, 128, 3, 32, 32, seed=31)
    ours, our_state = eager_recurrent_kda(q, k, v, g, beta, output_final_state=True)
    theirs, their_state = naive.naive_recurrent_kda(q, k, v, g, beta, output_final_state=True)
    err = (ours - theirs).abs().max().item()
    print(f"[fla naive_recurrent_kda] max|dout|={err:.3e}")
    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(our_state, their_state, rtol=1e-5, atol=1e-6)


@requires_cuda
def test_fla_naive_chunk_kda_matches_our_chunked_reference():
    """``fla.ops.kda.naive.naive_chunk_kda`` vs :func:`eager_chunk_kda`."""
    naive = pytest.importorskip("fla.ops.kda.naive", reason="fla build has no naive KDA reference")
    if not hasattr(naive, "naive_chunk_kda"):
        pytest.skip("fla build has no naive_chunk_kda")
    q, k, v, g, beta = _inputs(2, 128, 3, 32, 32, seed=32)  # T divisible by 64
    ours, _ = eager_chunk_kda(q, k, v, g, beta, chunk_size=64)
    theirs, _ = naive.naive_chunk_kda(q, k, v, g, beta, chunk_size=64)
    err = (ours - theirs).abs().max().item()
    print(f"[fla naive_chunk_kda] max|dout|={err:.3e}")
    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# fla's fused Triton chunk kernel
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_fla_chunk_kda_matches_the_eager_reference(dtype):
    """The dispatched ``fla`` backend must agree with the eager reference."""
    fla_chunk_kda = resolve_kda_backend("fla")
    q, k, v, g, beta = _inputs(2, 256, 4, 64, 64, dtype=dtype, seed=33)

    ours, our_state = eager_chunk_kda(q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
    theirs, their_state = fla_chunk_kda(
        q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    scale = ours.float().abs().max().item()
    err = (ours.float() - theirs.float()).abs().max().item()
    state_err = (our_state - their_state.float()).abs().max().item()
    print(f"[fla chunk_kda {dtype}] |out|max={scale:.3e}  max|dout|={err:.3e}  max|dstate|={state_err:.3e}")
    tol = 2e-5 if dtype == torch.float32 else 8e-3
    assert err < tol * max(scale, 1.0), f"fla chunk_kda disagrees by {err:.3e} (tol {tol:.1e} rel)"


@requires_cuda
def test_fla_chunk_kda_does_not_activate_beta_itself():
    """Record the ``use_beta_sigmoid_in_kernel`` hazard.

    HF's ``modeling_kimi_linear.py:603,622`` hands ``chunk_kda`` a raw
    ``b_proj(x)`` together with ``use_beta_sigmoid_in_kernel=True``. If
    the installed ``fla`` honoured that flag, passing a raw ``beta``
    would match passing ``sigmoid(beta)`` to the eager reference. It does
    not — so the flag is inert and the *caller* owns the sigmoid, which
    is why :mod:`...kda_kernels._fla` takes an already-activated ``beta``
    and refuses to forward the flag.
    """
    from fla.ops.kda import chunk_kda

    q, k, v, g, beta_raw = _inputs(2, 128, 2, 32, 32, seed=34)
    beta_raw = torch.randn_like(beta_raw)  # unbounded, as b_proj(x) is

    out_raw, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta_raw,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    eager_raw, _ = eager_chunk_kda(q, k, v, g, beta_raw, use_qk_l2norm_in_kernel=True)
    eager_sig, _ = eager_chunk_kda(q, k, v, g, torch.sigmoid(beta_raw), use_qk_l2norm_in_kernel=True)

    err_raw = (out_raw.float() - eager_raw).abs().max().item()
    err_sig = (out_raw.float() - eager_sig).abs().max().item()
    print(
        "[use_beta_sigmoid_in_kernel] fla-vs-eager(raw beta)="
        f"{err_raw:.3e}  fla-vs-eager(sigmoid beta)={err_sig:.3e}"
    )
    assert err_raw < err_sig, (
        "The installed fla appears to honour use_beta_sigmoid_in_kernel after all "
        f"(raw err {err_raw:.3e} >= sigmoid err {err_sig:.3e}). If so, the _fla adapter's "
        "beta contract must be revisited."
    )
    assert err_raw < 2e-5 * max(eager_raw.abs().max().item(), 1.0)


@requires_cuda
def test_fla_chunk_kda_in_kernel_gate_matches_kda_gate():
    """``use_gate_in_kernel=True`` must equal applying :func:`kda_gate` outside."""
    from fla.ops.kda import chunk_kda

    num_heads, k_dim = 4, 32
    gen = torch.Generator(device="cpu").manual_seed(35)
    z = torch.randn(2, 128, num_heads, k_dim, generator=gen).cuda()
    A_log = torch.randn(num_heads, generator=gen).cuda()
    dt_bias = torch.randn(num_heads * k_dim, generator=gen).cuda()
    q, k, v, _, beta = _inputs(2, 128, num_heads, k_dim, k_dim, seed=36)

    in_kernel, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=z,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    outside, _ = eager_chunk_kda(
        q, k, v, kda_gate(z, A_log, dt_bias, -5.0), beta, use_qk_l2norm_in_kernel=True
    )
    err = (in_kernel.float() - outside).abs().max().item()
    print(f"[in-kernel gate] max|dout|={err:.3e}")
    assert err < 2e-5 * max(outside.abs().max().item(), 1.0)


# ---------------------------------------------------------------------------
# The surrounding fla modules the HF model uses
# ---------------------------------------------------------------------------


@requires_cuda
def test_l2norm_matches_fla():
    from fla.modules.l2norm import l2norm

    x = torch.randn(2, 16, 3, 32).cuda()
    torch.testing.assert_close(kda_l2norm(x), l2norm(x.contiguous()), rtol=1e-5, atol=1e-6)


@requires_cuda
def test_short_convolution_matches_fla():
    """Our ``padding=k-1`` + truncate conv must equal ``fla``'s ``ShortConvolution``."""
    from fla.modules import ShortConvolution

    channels, kernel_size = 64, 4
    conv = ShortConvolution(hidden_size=channels, kernel_size=kernel_size, activation="silu").cuda()
    x = torch.randn(2, 96, channels).cuda()
    theirs, _ = conv(x)
    ours = hf_causal_conv1d_silu(x, conv.weight, conv.bias)
    err = (ours - theirs).abs().max().item()
    print(f"[ShortConvolution] max|d|={err:.3e}")
    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-6)


@requires_cuda
def test_gated_rms_norm_matches_fla():
    """:class:`KimiGatedRMSNorm` must equal ``FusedRMSNormGated(activation='sigmoid')``."""
    from fla.modules import FusedRMSNormGated

    from primus.backends.megatron.core.transformer.kimi_k3 import KimiGatedRMSNorm

    head_dim, eps = 64, 1e-5
    theirs = FusedRMSNormGated(head_dim, eps=eps, activation="sigmoid").cuda()
    with torch.no_grad():
        theirs.weight.normal_(mean=1.0, std=0.1)

    ours = KimiGatedRMSNorm(hidden_size=head_dim, eps=eps).cuda()
    with torch.no_grad():
        ours.weight.copy_(theirs.weight)

    x = torch.randn(2, 32, 4, head_dim).cuda()
    gate = torch.randn(2, 32, 4, head_dim).cuda()
    got = ours(x, gate)
    want = theirs(x.reshape(-1, head_dim), gate.reshape(-1, head_dim)).reshape(x.shape)
    # and the independent transcription in the test helpers
    also = hf_rms_norm_gated(x, gate, theirs.weight, eps)
    print(f"[FusedRMSNormGated] max|d module|={(got - want).abs().max().item():.3e}")
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(also, want, rtol=1e-5, atol=1e-6)
