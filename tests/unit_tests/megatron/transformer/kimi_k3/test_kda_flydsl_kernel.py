###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native FlyDSL KDA backend (WP9): parity against the eager reference.

Hardware- and dependency-gated: everything here needs the ``flydsl`` pip
package and a gfx950 (CDNA4) device, so the module skips wholesale
elsewhere. The eager reference these compare against is *not* gated — it
is covered by ``test_kda_eager_reference.py``.

Tolerance budget follows the DeepSeek-V4 convention
(``deepseek_v4/v4_attention_test_utils.py:43-69``): bf16 forward
``atol = rtol = 2e-2``, bf16 backward ``5e-2``, both against the **fp32**
eager reference. fp32-against-fp32 is held to a much tighter bound,
because there the only difference is summation order.

The tests are layered so a failure localises itself:

* :func:`test_kda_flydsl_scores_kernel_matches_dense_oracle` checks the
  ``@flyc.kernel`` alone against an ``O(C²K)`` fp64 evaluation of the
  definition — no blocking, no reference-row trickery.
* :func:`test_kda_flydsl_scores_kernel_survives_a_saturated_gate` pins the
  reason ``SUB_BLOCK`` is 16 rather than 32 or 64.
* :func:`test_kda_flydsl_scores_bwd_kernel_matches_dense_oracle` and
  :func:`..._survives_a_saturated_gate` do the same two things for the *adjoint*
  kernel, whose reference rows are not the forward's and whose overflow bound
  therefore had to be re-derived, and
  :func:`..._drops_the_masked_gradient` is the negative control for the mask it
  applies to the incoming gradient.
* the remaining tests exercise the assembled backend end to end.
"""

from __future__ import annotations

import pytest
import torch

flydsl = pytest.importorskip("flydsl", reason="the FlyDSL KDA backend needs the flydsl package")

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (  # noqa: E402
    eager_chunk_kda,
    resolve_kda_backend,
)

# bf16 vs the fp32 reference; fp32 vs fp32 only differs by summation order
BF16_FWD_TOL = dict(atol=2e-2, rtol=2e-2)
BF16_BWD_TOL = dict(atol=5e-2, rtol=5e-2)
FP32_TOL = dict(atol=1e-5, rtol=1e-5)

GATE_LOWER_BOUND = -5.0  # Kimi K3's kda_gate_lower_bound


def _on_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    return arch.startswith("gfx950")


pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="the FlyDSL KDA kernel is built for gfx950 (CDNA4)"
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def _make_inputs(batch, seq_len, num_heads, k_dim, v_dim, dtype, device="cuda", seed=0):
    """Inputs shaped like a real KDA layer's.

    ``q``/``k`` are L2-normalised: feeding raw ``randn`` makes the WY/UT
    triangular inverse lose conditioning and every KDA implementation blows
    up, so an un-normalised fixture makes a *correct* kernel look broken
    (``DECISIONS.md``, "q/k must be L2-normalised before any KDA kernel").
    ``g`` is the bounded gate ``lower_bound * sigmoid(.)``, and ``beta`` is
    already sigmoid-activated — the caller owns that (``DESIGN.md`` §3.4.4b).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    shape_qk = (batch, seq_len, num_heads, k_dim)
    q = _l2(torch.randn(*shape_qk, generator=gen, device=device, dtype=torch.float32))
    k = _l2(torch.randn(*shape_qk, generator=gen, device=device, dtype=torch.float32))
    v = torch.randn(batch, seq_len, num_heads, v_dim, generator=gen, device=device, dtype=torch.float32)
    z = torch.randn(*shape_qk, generator=gen, device=device, dtype=torch.float32) * 3.0
    g = GATE_LOWER_BOUND * torch.sigmoid(z)
    beta = torch.sigmoid(
        torch.randn(batch, seq_len, num_heads, generator=gen, device=device, dtype=torch.float32)
    )
    return tuple(x.to(dtype) for x in (q, k, v, g, beta))


def _dense_oracle(q, k, cg):
    """``out[r,c] = Σ_d left[r,d]·exp(cg[r,d] − cg[c,d])·right[c,d]`` in fp64.

    A literal transcription of the definition: one exponential per
    ``(r, c, d)``, no sub-blocking and no choice of reference row, so it
    shares no code and no idea with the thing under test.
    """
    num_chunks, chunk_size, _ = q.shape
    q64, k64, cg64 = q.double(), k.double(), cg.double()
    a_qk = torch.zeros(num_chunks, chunk_size, chunk_size, dtype=torch.float64, device=q.device)
    a_kk = torch.zeros_like(a_qk)
    for r in range(chunk_size):
        for c in range(r + 1):
            decay = (cg64[:, r] - cg64[:, c]).exp()
            a_qk[:, r, c] = (q64[:, r] * decay * k64[:, c]).sum(-1)
            if c < r:
                a_kk[:, r, c] = (k64[:, r] * decay * k64[:, c]).sum(-1)
    return a_qk, a_kk


def _make_chunk_inputs(num_chunks, chunk_size, k_dim, device="cuda", seed=0, saturated=False):
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (num_chunks, chunk_size, k_dim)
    q = _l2(torch.randn(*shape, generator=gen, device=device, dtype=torch.float32))
    k = _l2(torch.randn(*shape, generator=gen, device=device, dtype=torch.float32))
    if saturated:
        g = torch.full(shape, GATE_LOWER_BOUND, device=device, dtype=torch.float32)
    else:
        z = torch.randn(*shape, generator=gen, device=device, dtype=torch.float32) * 3.0
        g = GATE_LOWER_BOUND * torch.sigmoid(z)
    return q, k, g.cumsum(dim=-2).contiguous()


# ---------------------------------------------------------------------------
# the kernel in isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k_dim", [32, 64, 128])
def test_kda_flydsl_scores_kernel_matches_dense_oracle(k_dim):
    """The ``@flyc.kernel`` reproduces the definition to fp32 precision."""
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
        decay_scores,
    )

    q, k, cg = _make_chunk_inputs(4, 64, k_dim)
    a_qk, a_kk = decay_scores(q, k, cg)
    ref_qk, ref_kk = _dense_oracle(q, k, cg)

    assert torch.isfinite(a_qk).all() and torch.isfinite(a_kk).all()
    assert (a_qk.double() - ref_qk).abs().max() < 1e-5
    assert (a_kk.double() - ref_kk).abs().max() < 1e-5


def test_kda_flydsl_scores_kernel_respects_the_two_masks():
    """``Aqk`` keeps its diagonal; ``Akk`` does not.

    ``o_t`` reads the POST-update state ``S_t``, which is the whole reason
    the intra-chunk attention matrix retains its diagonal while the
    delta-correction matrix is strictly lower.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
        decay_scores,
    )

    q, k, cg = _make_chunk_inputs(3, 64, 64)
    a_qk, a_kk = decay_scores(q, k, cg)

    assert a_qk.triu(diagonal=1).abs().max() == 0.0
    assert a_kk.triu(diagonal=0).abs().max() == 0.0
    assert a_qk.diagonal(dim1=-2, dim2=-1).abs().max() > 0.0


def test_kda_flydsl_scores_kernel_survives_a_saturated_gate():
    """Every step at the ``-5`` bound — the case that kills the published form.

    With ``g = -5`` throughout and ``C = 64``, the two-matmul form's
    ``1/Γ`` factor is ``exp(320)``, i.e. ``inf`` in fp32. The kernel
    references the cumulative decay per 16-row sub-block instead, which
    caps every exponent it evaluates at ``exp(75)``.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
        decay_scores,
    )

    q, k, cg = _make_chunk_inputs(2, 64, 128, saturated=True)
    assert float((-cg).max()) > 300.0, "fixture does not actually saturate the gate"
    assert torch.isinf((-cg).exp()).any(), "the naive 1/Gamma factor should overflow here"

    a_qk, a_kk = decay_scores(q, k, cg)
    ref_qk, ref_kk = _dense_oracle(q, k, cg)

    assert torch.isfinite(a_qk).all() and torch.isfinite(a_kk).all()
    assert (a_qk.double() - ref_qk).abs().max() < 1e-5
    assert (a_kk.double() - ref_kk).abs().max() < 1e-5


def test_kda_flydsl_scores_kernel_agrees_with_its_torch_twin():
    """The kernel and the blocked torch implementation of its adjoint agree.

    ``decay_scores_torch`` is what supplies the backward, so a divergence
    here would silently make the gradient inconsistent with the forward.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
        decay_scores,
        decay_scores_torch,
    )

    q, k, cg = _make_chunk_inputs(4, 64, 128)
    a_kernel = decay_scores(q, k, cg)
    a_torch = decay_scores_torch(q, k, cg)
    for got, want in zip(a_kernel, a_torch):
        assert (got - want).abs().max() < 1e-5


# ---------------------------------------------------------------------------
# the score kernel's native adjoint
# ---------------------------------------------------------------------------


def _dense_bwd_oracle(q, k, cg, d_aqk, d_akk):
    """The adjoint of ``_dense_oracle``, in fp64, from the definition.

    One exponential per ``(r, c)`` and one explicit accumulation per term, so
    it shares no blocking, no reference row and no factorisation with the
    kernel. The masks are applied by the loop bounds, exactly as the forward
    oracle applies them.
    """
    _, chunk_size, _ = q.shape
    q64, k64, cg64 = q.double(), k.double(), cg.double()
    dq = torch.zeros_like(q64)
    dk = torch.zeros_like(q64)
    dcg = torch.zeros_like(q64)
    for r in range(chunk_size):
        for c in range(r + 1):
            decay = (cg64[:, r] - cg64[:, c]).exp()
            gq = d_aqk[:, r, c].double().unsqueeze(-1)
            term = gq * decay * k64[:, c]
            dq[:, r] += term
            dk[:, c] += gq * decay * q64[:, r]
            wq = q64[:, r] * term
            dcg[:, r] += wq
            dcg[:, c] -= wq
            if c < r:
                gk = d_akk[:, r, c].double().unsqueeze(-1)
                dk[:, r] += gk * decay * k64[:, c]
                dk[:, c] += gk * decay * k64[:, r]
                wk = k64[:, r] * gk * decay * k64[:, c]
                dcg[:, r] += wk
                dcg[:, c] -= wk
    return dq, dk, dcg


def _make_score_grads(num_chunks, chunk_size, device="cuda", seed=17):
    """Upstream gradients as autograd hands them over: **dense**.

    ``Aqk`` feeds ``o = Aqk @ T + Rq``, whose adjoint ``do @ Tᵀ`` is a full
    matrix, so the kernel is the thing that has to drop the two triangles.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (num_chunks, chunk_size, chunk_size)
    return (
        torch.randn(*shape, generator=gen, device=device, dtype=torch.float32),
        torch.randn(*shape, generator=gen, device=device, dtype=torch.float32),
    )


@pytest.mark.parametrize("k_dim", [32, 64, 128])
def test_kda_flydsl_scores_bwd_kernel_matches_dense_oracle(k_dim):
    """The adjoint kernel reproduces the definition's adjoint to fp32 precision."""
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
        decay_scores_bwd,
    )

    q, k, cg = _make_chunk_inputs(3, 64, k_dim)
    d_aqk, d_akk = _make_score_grads(3, 64)
    got = decay_scores_bwd(q, k, cg, d_aqk, d_akk)
    want = _dense_bwd_oracle(q, k, cg, d_aqk, d_akk)

    for name, a, b in zip(("dq", "dk", "dcg"), got, want):
        assert torch.isfinite(a).all(), name
        assert (a.double() - b).abs().max() < 1e-5, name


def test_kda_flydsl_scores_bwd_kernel_survives_a_saturated_gate():
    """``g = -5`` everywhere — and the guard has to be re-derived, not inherited.

    The forward references the *first* row of each row-block off the diagonal.
    The adjoint's ``Σ_r`` direction owns a **column** block, whose columns sit
    *after* that row, so reusing the forward's choice would put
    ``exp(cg[first] − cg[c])`` at ``exp((SUB_BLOCK−1)·5) = exp(75)``; it
    references the block's last row instead, which keeps both decay factors in
    ``(0, 1]``. See ``kda_decay_scores_bwd_kernel``'s module docstring.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
        decay_scores_bwd,
    )

    q, k, cg = _make_chunk_inputs(2, 64, 128, saturated=True)
    d_aqk, d_akk = _make_score_grads(2, 64)
    assert float((-cg).max()) > 300.0, "fixture does not actually saturate the gate"
    assert torch.isinf((-cg).exp()).any(), "the naive 1/Gamma factor should overflow here"

    got = decay_scores_bwd(q, k, cg, d_aqk, d_akk)
    want = _dense_bwd_oracle(q, k, cg, d_aqk, d_akk)
    for name, a, b in zip(("dq", "dk", "dcg"), got, want):
        assert torch.isfinite(a).all(), name
        assert (a.double() - b).abs().max() < 1e-5, name


def test_kda_flydsl_scores_bwd_twin_is_the_adjoint_of_the_forward_twin():
    """``decay_scores_bwd_torch`` against autograd through ``decay_scores_torch``.

    The kernel is checked against this twin, and the twin is the fallback for
    unsupported geometries, so it needs its own independent check: that it is
    the adjoint of the *blocked forward* rather than merely self-consistent.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
        decay_scores_bwd_torch,
        decay_scores_torch,
    )

    q, k, cg = _make_chunk_inputs(3, 64, 64)
    d_aqk, d_akk = _make_score_grads(3, 64)
    got = decay_scores_bwd_torch(q, k, cg, d_aqk, d_akk)

    leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, cg)]
    aqk, akk = decay_scores_torch(*leaves)
    want = torch.autograd.grad((aqk, akk), leaves, (d_aqk, d_akk), allow_unused=True)

    for name, a, b in zip(("dq", "dk", "dcg"), got, want):
        rel = (a - b).abs().max() / b.abs().max().clamp_min(1e-30)
        assert rel < 1e-5, f"{name}: {rel:.3e}"


def test_kda_flydsl_scores_bwd_kernel_agrees_with_its_torch_twin():
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
        _decay_scores_bwd_flydsl,
        decay_scores_bwd_torch,
    )

    q, k, cg = _make_chunk_inputs(4, 64, 128)
    d_aqk, d_akk = _make_score_grads(4, 64)
    got = _decay_scores_bwd_flydsl(q, k, cg, d_aqk, d_akk)
    want = decay_scores_bwd_torch(q, k, cg, d_aqk, d_akk)
    for name, a, b in zip(("dq", "dk", "dcg"), got, want):
        rel = (a - b).abs().max() / b.abs().max().clamp_min(1e-30)
        assert rel < 1e-5, f"{name}: {rel:.3e}"


def test_kda_flydsl_scores_bwd_kernel_drops_the_masked_gradient():
    """The forward writes exact zeros above its diagonals, so nothing above them
    may reach an input gradient.

    A negative control: perturbing the upstream gradient *only* in the discarded
    triangles must leave all three gradients bit-identical. Without the in-kernel
    mask the diagonal sub-blocks would happily contract those entries — and they
    are the entries whose decay factor reaches ``exp(75)``.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
        _decay_scores_bwd_flydsl,
    )

    q, k, cg = _make_chunk_inputs(2, 64, 64)
    d_aqk, d_akk = _make_score_grads(2, 64)
    base = _decay_scores_bwd_flydsl(q, k, cg, d_aqk, d_akk)

    noise_qk = torch.triu(torch.randn_like(d_aqk) * 1e3, diagonal=1)
    noise_kk = torch.triu(torch.randn_like(d_akk) * 1e3, diagonal=0)
    assert noise_qk.abs().max() > 0.0 and noise_kk.abs().max() > 0.0
    perturbed = _decay_scores_bwd_flydsl(q, k, cg, d_aqk + noise_qk, d_akk + noise_kk)

    for name, a, b in zip(("dq", "dk", "dcg"), base, perturbed):
        assert (a - b).abs().max() == 0.0, name


def test_kda_flydsl_rejects_an_unsupported_head_dim():
    """An unusable geometry must name the fallbacks, not fail inside a compile."""
    backend = resolve_kda_backend("flydsl")
    q, k, v, g, beta = _make_inputs(1, 64, 2, 96, 96, torch.float32)
    with pytest.raises(ValueError, match="eager | eager_recurrent | fla"):
        backend(q, k, v, g, beta)


# ---------------------------------------------------------------------------
# the fused state-sweep kernel in isolation
# ---------------------------------------------------------------------------


def _make_sweep_operands(nbh, nc, chunk, k_dim, v_dim, *, emit_rq, has_e, op_dtype, seed=0):
    """Operands scaled the way the real assembly produces them.

    Raw ``randn`` gives every contraction a gain of ``sqrt(K)``, so the
    recurrence expands by ~64x per chunk and *any* difference in summation order
    is amplified out of all proportion — which says nothing about the kernel. The
    real operands come from L2-normalised ``q``/``k`` and a decay in ``(0, 1]``,
    i.e. the map is contractive; these are scaled to match.
    """
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(seed)
    nb = nbh * nc

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, device=dev, dtype=torch.float32)

    return dict(
        amat=(rnd(nb, 2 * chunk if emit_rq else chunk, k_dim) * k_dim**-0.5).to(op_dtype),
        yc=rnd(nb, chunk, v_dim),
        xt=(rnd(nb, k_dim, chunk) * chunk**-0.5).to(op_dtype),
        dec=torch.rand(nb, k_dim, generator=gen, device=dev),
        s0=rnd(nbh, k_dim, v_dim),
        e_in=rnd(nb, k_dim, v_dim) if has_e else None,
        num_chunks=nc,
        emit_rq=emit_rq,
    )


# (mode, operand dtype, tolerance). The MFMA path rounds its operands to bf16 and
# the twin rounds at the same four places, so the residual is fp32 summation
# order — except that rounding is a *step* function, so a 1e-7 difference before
# the rounding can become a full bf16 ulp after it, and the recurrence carries it
# forward. 5e-3 is that effect, not an inconsistency; the fp32 path pins the
# arithmetic itself at 1e-5.
_SWEEP_MODES = [("valu", torch.float32, 1e-5), ("mfma", torch.bfloat16, 5e-3)]


@pytest.mark.parametrize("mode,op_dtype,tol", _SWEEP_MODES)
@pytest.mark.parametrize(
    "direction",
    [
        # the forward configuration: Rq emitted alongside T, no additive term
        dict(sgn_t=-1.0, sgn_x=1.0, reverse=False, emit_rq=True, has_e=False),
        # and the backward one, which is the same kernel run the other way
        dict(sgn_t=1.0, sgn_x=-1.0, reverse=True, emit_rq=False, has_e=True),
    ],
    ids=["forward", "reverse"],
)
def test_kda_flydsl_state_sweep_kernel_matches_its_torch_twin(mode, op_dtype, tol, direction):
    """The fused sweep and :func:`state_sweep_torch` compute the same recurrence.

    The twin is also the fallback for unsupported geometries, so a divergence
    here would mean two different backends depending on the head dim.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.sweep import (
        _run_sweep,
        state_sweep_torch,
    )

    direction = dict(direction)
    ops = _make_sweep_operands(
        3, 4, 64, 128, 128, op_dtype=op_dtype, emit_rq=direction.pop("emit_rq"),
        has_e=direction.pop("has_e"),
    )
    kw = {**ops, **direction}
    got = _run_sweep(True, mode, op_dtype, emit_states=True, **kw)
    want = state_sweep_torch(op_dtype=op_dtype, **kw)
    for name, a, b in zip(("rq", "t", "states", "s_final"), got, want):
        if a is None:
            assert b is None or name == "rq"
            continue
        assert torch.isfinite(a).all(), name
        rel = (a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)
        assert rel < tol, f"{name}: {rel:.3e} >= {tol:.0e}"


def test_kda_flydsl_state_sweep_adjoint_matches_autograd():
    """The hand-written sweep adjoint against autograd through the same recurrence.

    A custom kernel is opaque to autograd, so :class:`_FusedSweep` carries the
    analytic adjoint of its four lines. This differentiates a plain
    autograd-visible transcription of those same four lines and requires the two
    gradients to agree — which is the only check that the adjoint is not merely
    self-consistent.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.sweep import (
        fused_chunk_sweep,
    )

    nbh, nc, chunk, k_dim, v_dim = 2, 4, 64, 128, 128
    nb = nbh * nc
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(3)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, device=dev, dtype=torch.float32)

    leaves = dict(
        qw=rnd(nb, 2 * chunk, k_dim) * k_dim**-0.5,
        u=rnd(nb, chunk, v_dim),
        aqk=torch.tril(rnd(nb, chunk, chunk)) * chunk**-0.5,
        kg=rnd(nb, chunk, k_dim) * k_dim**-0.5,
        dec=torch.rand(nb, k_dim, generator=gen, device=dev),
        s0=rnd(nbh, k_dim, v_dim),
    )
    do = rnd(nb, chunk, v_dim)
    dsf = rnd(nbh, k_dim, v_dim)

    def run(fn):
        args = {k: t.detach().clone().requires_grad_(True) for k, t in leaves.items()}
        o, sf = fn(**args)
        torch.autograd.backward((o, sf), (do, dsf))
        return {k: t.grad for k, t in args.items()}

    def autograd_reference(qw, u, aqk, kg, dec, s0):
        """The four lines of the recurrence, differentiable, no custom kernel."""
        qwv = qw.view(nbh, nc, 2 * chunk, k_dim)
        uv = u.view(nbh, nc, chunk, v_dim)
        aqkv = aqk.view(nbh, nc, chunk, chunk)
        kgv = kg.view(nbh, nc, chunk, k_dim)
        decv = dec.view(nbh, nc, k_dim)
        state, outs = s0, []
        for n in range(nc):
            read = qwv[:, n] @ state
            tv = uv[:, n] - read[:, chunk:]
            outs.append(aqkv[:, n] @ tv + read[:, :chunk])
            state = decv[:, n].unsqueeze(-1) * state + kgv[:, n].transpose(-1, -2) @ tv
        return torch.stack(outs, dim=1).reshape(nb, chunk, v_dim), state

    got = run(lambda **kw: fused_chunk_sweep(num_chunks=nc, op_dtype=torch.float32, **kw))
    want = run(autograd_reference)
    for name in leaves:
        a, b = got[name], want[name]
        assert a is not None, name
        rel = (a - b).abs().max() / b.abs().max().clamp_min(1e-30)
        assert rel < 1e-5, f"d{name}: {rel:.3e}"


def test_kda_flydsl_state_sweep_names_why_a_geometry_is_unsupported():
    """The sweep is stricter than the score kernel, and falls back rather than fails.

    ``head_dim=32`` clears the score kernel but leaves fewer than one 16-row MFMA
    tile per wave, so the assembly has to use the torch twin. The reason has to be
    reported, not swallowed, or a silent fallback becomes a silent slowdown.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.sweep import (
        supports_sweep,
    )

    assert supports_sweep(64, 128, 128) is None
    for reason in (supports_sweep(64, 32, 32), supports_sweep(32, 128, 128)):
        assert reason is not None and reason.strip()


# ---------------------------------------------------------------------------
# the assembled backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch,seq_len,num_heads,k_dim,v_dim",
    [
        (2, 128, 3, 64, 64),
        (1, 256, 4, 128, 128),  # production head geometry
        (2, 100, 2, 64, 64),  # T not a multiple of the chunk size
    ],
)
def test_kda_flydsl_forward_matches_eager_reference_fp32(batch, seq_len, num_heads, k_dim, v_dim):
    backend = resolve_kda_backend("flydsl")
    inputs = _make_inputs(batch, seq_len, num_heads, k_dim, v_dim, torch.float32)

    o_ref, s_ref = eager_chunk_kda(*inputs, output_final_state=True)
    o_fly, s_fly = backend(*inputs, output_final_state=True)

    assert torch.isfinite(o_fly).all()
    torch.testing.assert_close(o_fly, o_ref, **FP32_TOL)
    torch.testing.assert_close(s_fly, s_ref, **FP32_TOL)


def test_kda_flydsl_forward_matches_eager_reference_bf16():
    """bf16 inputs against the **fp32** reference, at the DSv4 tolerance."""
    backend = resolve_kda_backend("flydsl")
    inputs = _make_inputs(1, 256, 4, 128, 128, torch.bfloat16)
    ref_inputs = tuple(x.float() for x in inputs)

    o_ref, _ = eager_chunk_kda(*ref_inputs)
    o_fly, _ = backend(*inputs)

    assert o_fly.dtype == torch.bfloat16
    torch.testing.assert_close(o_fly.float(), o_ref, **BF16_FWD_TOL)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kda_flydsl_backward_matches_eager_reference(dtype):
    """Gradients for all five inputs, against the fp32 reference."""
    backend = resolve_kda_backend("flydsl")
    base = _make_inputs(1, 192, 3, 64, 64, dtype)
    fly_in = [x.detach().clone().requires_grad_(True) for x in base]
    ref_in = [x.detach().float().requires_grad_(True) for x in base]

    o_fly, _ = backend(*fly_in)
    o_ref, _ = eager_chunk_kda(*ref_in)

    gen = torch.Generator(device=o_ref.device).manual_seed(11)
    upstream = torch.randn(o_ref.shape, generator=gen, device=o_ref.device, dtype=torch.float32)
    o_fly.backward(upstream.to(o_fly.dtype))
    o_ref.backward(upstream)

    tol = FP32_TOL if dtype == torch.float32 else BF16_BWD_TOL
    for name, got, want in zip(("q", "k", "v", "g", "beta"), fly_in, ref_in):
        assert got.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(got.grad).all(), f"non-finite gradient for {name}"
        torch.testing.assert_close(got.grad.float(), want.grad, msg=lambda m, n=name: f"{n}: {m}", **tol)


def test_kda_flydsl_honours_a_non_default_chunk_size():
    """``chunk_size = 32`` must work: the kernel only needs a multiple of 16.

    ``fla`` cannot honour this (it fixes its tiling internally and the adapter
    raises), so it is a capability the FlyDSL backend has and the production
    backend does not.
    """
    backend = resolve_kda_backend("flydsl")
    inputs = _make_inputs(2, 128, 3, 64, 64, torch.float32)
    o_ref, s_ref = eager_chunk_kda(*inputs, output_final_state=True, chunk_size=32)
    o_fly, s_fly = backend(*inputs, output_final_state=True, chunk_size=32)
    torch.testing.assert_close(o_fly, o_ref, **FP32_TOL)
    torch.testing.assert_close(s_fly, s_ref, **FP32_TOL)


def test_kda_flydsl_honours_an_initial_state():
    """A carried state must be consumed, differentiated and advanced."""
    backend = resolve_kda_backend("flydsl")
    q, k, v, g, beta = _make_inputs(2, 128, 3, 64, 64, torch.float32)
    h0 = torch.randn(2, 3, 64, 64, device=q.device, dtype=torch.float32)

    o_ref, s_ref = eager_chunk_kda(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    o_fly, s_fly = backend(q, k, v, g, beta, initial_state=h0, output_final_state=True)
    torch.testing.assert_close(o_fly, o_ref, **FP32_TOL)
    torch.testing.assert_close(s_fly, s_ref, **FP32_TOL)

    h0_fly = h0.detach().clone().requires_grad_(True)
    o, _ = backend(q, k, v, g, beta, initial_state=h0_fly)
    o.sum().backward()
    assert h0_fly.grad is not None and torch.isfinite(h0_fly.grad).all()


def test_kda_flydsl_applies_l2norm_when_asked():
    """``use_qk_l2norm_in_kernel`` must match the reference's own handling."""
    backend = resolve_kda_backend("flydsl")
    gen = torch.Generator(device="cuda").manual_seed(3)
    q = torch.randn(1, 128, 2, 64, generator=gen, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 128, 2, 64, generator=gen, device="cuda", dtype=torch.float32)
    v = torch.randn(1, 128, 2, 64, generator=gen, device="cuda", dtype=torch.float32)
    g = GATE_LOWER_BOUND * torch.sigmoid(
        torch.randn(1, 128, 2, 64, generator=gen, device="cuda", dtype=torch.float32)
    )
    beta = torch.sigmoid(torch.randn(1, 128, 2, generator=gen, device="cuda", dtype=torch.float32))

    o_ref, _ = eager_chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    o_fly, _ = backend(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    torch.testing.assert_close(o_fly, o_ref, **FP32_TOL)


def test_kda_flydsl_collapses_to_the_recurrent_reference():
    """Against the ``O(T)`` literal recurrence, not just the chunked form.

    The chunked algorithm and the sequential one share no structure, so
    agreeing with the recurrence is independent evidence that the chunking,
    the UT transform and the state sweep are all right.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
        eager_recurrent_kda,
    )

    backend = resolve_kda_backend("flydsl")
    inputs = _make_inputs(2, 128, 3, 64, 64, torch.float32)
    o_rec, s_rec = eager_recurrent_kda(*inputs, output_final_state=True)
    o_fly, s_fly = backend(*inputs, output_final_state=True)
    torch.testing.assert_close(o_fly, o_rec, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(s_fly, s_rec, atol=1e-4, rtol=1e-4)


def test_kda_flydsl_agrees_with_the_fla_backend():
    """Drop-in interchangeability with the Triton backend it is measured against."""
    pytest.importorskip("fla.ops.kda", reason="needs flash-linear-attention")
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
        load_fla_kda_backend,
    )

    fla_backend = load_fla_kda_backend()
    fly_backend = resolve_kda_backend("flydsl")
    inputs = _make_inputs(1, 256, 4, 128, 128, torch.bfloat16)

    o_fla, _ = fla_backend(*inputs)
    o_fly, _ = fly_backend(*inputs)
    torch.testing.assert_close(o_fly.float(), o_fla.float(), **BF16_FWD_TOL)


def test_kda_flydsl_ut_inverse_matches_forward_substitution():
    """Neumann doubling must equal the reference's ``C``-step substitution.

    :func:`...ops.ut_inverse` replaces the eager reference's serial loop; a
    difference here would be a silent divergence in the WY representation.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
        ut_inverse,
    )

    chunk_size = 64
    gen = torch.Generator(device="cuda").manual_seed(5)
    low = torch.randn(6, chunk_size, chunk_size, generator=gen, device="cuda", dtype=torch.float32)
    low = torch.tril(low, diagonal=-1) * 0.3

    got = ut_inverse(low)

    # the eager reference's own recurrence (reference.py:372-377)
    attn = low.clone()
    for r in range(1, chunk_size):
        row = attn[..., r, :r].clone()
        sub = attn[..., :r, :r].clone()
        attn[..., r, :r] = row + (row.unsqueeze(-1) * sub).sum(-2)
    want = attn + torch.eye(chunk_size, device=low.device, dtype=low.dtype)

    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_kda_flydsl_ut_inverse_gradient_is_correct():
    """The analytic adjoint ``dL = tril(Pᵀ dP Pᵀ, −1)`` against autograd."""
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
        _ut_inverse_doubling,
        ut_inverse,
    )

    gen = torch.Generator(device="cuda").manual_seed(9)
    base = torch.tril(
        torch.randn(3, 32, 32, generator=gen, device="cuda", dtype=torch.float64), diagonal=-1
    ) * 0.3
    analytic = base.detach().clone().requires_grad_(True)
    numeric = base.detach().clone().requires_grad_(True)
    upstream = torch.randn(3, 32, 32, generator=gen, device="cuda", dtype=torch.float64)

    ut_inverse(analytic).backward(upstream)
    _ut_inverse_doubling(numeric).backward(upstream)

    torch.testing.assert_close(analytic.grad, torch.tril(numeric.grad, diagonal=-1), atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# the UT-inverse kernel (WP9 pass 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_kda_flydsl_ut_inverse_kernel_matches_a_fp64_oracle(chunk_size):
    """The kernel against ``(I − L)^{-1}`` evaluated in fp64.

    Not against the doubling it replaces: the doubling is itself the less
    accurate of the two (``5C³`` FMAs against ``C³/3``), so it is the wrong
    thing to hold the kernel to. This is the definition.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
        _ut_inverse_flydsl,
    )

    gen = torch.Generator(device="cuda").manual_seed(21)
    low = torch.tril(
        torch.randn(9, chunk_size, chunk_size, generator=gen, device="cuda", dtype=torch.float32),
        diagonal=-1,
    ) * 0.3

    got = _ut_inverse_flydsl(low)

    ref = torch.eye(chunk_size, dtype=torch.float64, device="cuda").expand_as(low.double()).clone()
    l64 = low.double()
    for r in range(1, chunk_size):
        ref[:, r] = ref[:, r] + (l64[:, r, :r].unsqueeze(-1) * ref[:, :r]).sum(-2)

    torch.testing.assert_close(got.double(), ref, atol=1e-5, rtol=1e-5)


def test_kda_flydsl_ut_inverse_kernel_writes_an_exact_unit_triangle():
    """``P`` is unit lower triangular by construction, so say so exactly.

    The kernel allocates its output with ``empty`` and writes every element,
    including the ones it knows are ``1`` and ``0``. If it ever stopped writing
    the strict upper triangle, uninitialised memory would leak into ``ut`` and
    from there into ``W`` and ``U`` — a bug that random inputs would hide most
    of the time.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
        _ut_inverse_flydsl,
    )

    gen = torch.Generator(device="cuda").manual_seed(22)
    low = torch.tril(
        torch.randn(5, 64, 64, generator=gen, device="cuda", dtype=torch.float32), diagonal=-1
    )

    got = _ut_inverse_flydsl(low)

    assert torch.equal(got.triu(1), torch.zeros_like(got.triu(1)))
    assert torch.equal(
        got.diagonal(dim1=-2, dim2=-1), torch.ones_like(got.diagonal(dim1=-2, dim2=-1))
    )


def test_kda_flydsl_ut_inverse_falls_back_off_its_supported_set():
    """fp64 and unbuilt widths must take the doubling, not raise or truncate.

    ``test_kda_flydsl_ut_inverse_gradient_is_correct`` runs in fp64 precisely
    because the adjoint needs the precision, and the kernel is fp32-only; the
    selection predicate is what keeps that test on the torch path.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
        _use_ut_kernel,
    )

    low64 = torch.zeros(2, 64, 64, device="cuda", dtype=torch.float32)
    assert _use_ut_kernel(low64)
    assert not _use_ut_kernel(low64.double())
    assert not _use_ut_kernel(low64.cpu())
    assert not _use_ut_kernel(torch.zeros(2, 48, 48, device="cuda"))


def test_kda_flydsl_runs_correctly_on_a_non_default_stream():
    """The kernels must go to torch's *current* stream, not to the null stream.

    ``fx.Stream(None)`` resolves to ``c_void_p(0)``. On this image torch's
    default current stream is also ``0x0``, so the two coincide until a caller
    uses a stream of its own — and ``torch.cuda.Stream()`` is **non-blocking**,
    so it does not serialise against the legacy default stream. Without
    :func:`..._stream.with_current_stream` the kernels would run unordered
    against the torch ops around them.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
        load_flydsl_kda_backend,
    )

    backend = load_flydsl_kda_backend()
    inputs = _make_inputs(1, 128, 2, 64, 64, torch.float32)
    with torch.no_grad():
        want, _ = backend(*inputs)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        # something big enough that an unordered kernel would read the wrong
        # thing rather than accidentally winning the race
        scratch = torch.randn(4096, 4096, device="cuda")
        scratch @ scratch
        with torch.no_grad():
            got, _ = backend(*inputs)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)


def test_kda_flydsl_forward_replays_correctly_from_a_cuda_graph():
    """Capture must actually record the FlyDSL kernels, not silently skip them.

    ``torch.cuda.graph`` captures on a side stream, so a kernel launched on the
    null stream is *not* recorded: capture completes, and the replay reproduces
    whatever was in the output buffer. That is not a hypothetical — it is what
    this path did until pass 5, at max relative error **1.0** on every shape,
    and it is why pass 4's "dispatch-free ceiling" numbers were withdrawn.

    Forward only and under ``no_grad``: hipBLASLt refuses stream capture, and on
    this image that failure wedges the process rather than raising, so the
    backward is deliberately not captured here.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
        load_flydsl_kda_backend,
    )

    backend = load_flydsl_kda_backend()
    inputs = _make_inputs(1, 256, 2, 64, 64, torch.float32)
    with torch.no_grad():
        want, _ = backend(*inputs)
    want = want.clone()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            with torch.no_grad():
                backend(*inputs)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.no_grad():
            static_out, _ = backend(*inputs)
    static_out.zero_()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(static_out, want, atol=1e-5, rtol=1e-5)


def test_kda_flydsl_state_sweep_block_v_choice_does_not_change_the_answer():
    """``block_v`` is a pure tuning knob, so every legal value must agree.

    Pass 4 moved the production choice from 64 to 16 on a 1.5x speed
    measurement. That is only a safe thing to have done if the width cannot
    change the result — one workgroup owns a disjoint slice of ``V`` and the
    recurrence is independent along ``V``.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.kda_state_sweep_kernel import (  # noqa: E501
        build_kda_state_sweep,
        supports_sweep_geometry,
    )
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.sweep import (
        _pick_block_v,
    )

    nbh, nc, C, K, V = 2, 3, 64, 64, 64
    nb = nbh * nc
    assert _pick_block_v(V) == 16, "the tuned choice is the one shipping"

    gen = torch.Generator(device="cuda").manual_seed(23)

    def rnd(*shape, dtype=torch.float32):
        return (
            torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32) * 0.1
        ).to(dtype)

    amat, yc = rnd(nb, 2 * C, K, dtype=torch.bfloat16), rnd(nb, C, V)
    xt = rnd(nb, K, C, dtype=torch.bfloat16)
    dec = torch.rand(nb, K, generator=gen, device="cuda")
    s0 = rnd(nbh, K, V)
    dummy = torch.empty(1, device="cuda")

    ref = None
    for block_v in (16, 32, 64):
        if supports_sweep_geometry(C, K, V, block_v) is not None:
            continue
        launch = build_kda_state_sweep(
            chunk_size=C, k_dim=K, v_dim=V, block_v=block_v, mode="mfma",
            emit_rq=True, emit_states=False, has_e=False,
            sgn_t=-1.0, sgn_x=1.0, reverse=False,
        )
        rq = torch.empty(nb, C, V, device="cuda")
        t_all = torch.empty(nb, C, V, device="cuda")
        sf = torch.empty(nbh, K, V, device="cuda")
        launch(
            amat.reshape(-1), yc.reshape(-1), xt.reshape(-1), dec.reshape(-1), dummy,
            s0.reshape(-1), rq.reshape(-1), t_all.reshape(-1), dummy, sf.reshape(-1),
            int(nbh), int(nc),
        )
        if ref is None:
            ref = (rq.clone(), t_all.clone(), sf.clone())
        else:
            for got, want in zip((rq, t_all, sf), ref):
                torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)
    assert ref is not None, "no legal block_v for this geometry"
