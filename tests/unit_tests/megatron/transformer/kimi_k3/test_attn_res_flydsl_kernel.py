###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The fused FlyDSL attention-residual mixer, against the eager oracle.

Structure mirrors ``test_kda_flydsl_kernel.py``: parity across shapes and dtypes
in both directions, geometry refusals, backend dispatch, and — the part that
makes the rest mean anything — **bug injection**.

Why bug injection is here
-------------------------
A softmax over a handful of similar candidates is forgiving: a wrong
implementation still produces something of the right magnitude, pointing roughly
the right way. ``test_attention_residual.py`` already guards the *eager* path
with three negative controls for exactly that reason. The kernel needs the same
treatment, and for a compiled kernel the only honest way to do it is to break the
**emitted MLIR** rather than the torch glue around it — otherwise the test proves
that a Python wrapper is wired up, not that the kernel computes the right thing.

So the FlyDSL builders take a test-only ``inject`` argument
(``FWD_INJECTIONS`` / ``BWD_INJECTIONS``, each a plausible mistake), and
:func:`test_injected_forward_defect_is_caught` /
:func:`test_injected_backward_defect_is_caught` require the *same* assertion that
passes on the correct kernel to reject every one of them. If any injection
slipped through, the corresponding parity test would be worthless.

The one that matters most is ``mix_normalised``: mixing the RMS-normalised
candidates instead of the raw ones. It is the single most likely way to get this
module wrong (``modeling_kimi_linear.py:1083`` builds ``k`` for the scores,
``:1087`` mixes ``v_float``), and it rescales the residual stream to unit RMS
while leaving every other property intact.
"""

from __future__ import annotations

import os

import pytest
import torch  # noqa: F401  # must precede any transformer_engine import

for _var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
    os.environ.pop(_var, None)

flydsl = pytest.importorskip(
    "flydsl", reason="the FlyDSL attention-residual backend needs the flydsl package"
)

from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels import (  # noqa: E402
    ATTN_RES_BACKENDS,
    eager_attn_res_mix,
    fused_score_weight,
    resolve_attn_res_backend,
)

# fp32 vs fp32: the two differ only in summation order and in the rsqrt, which
# the kernel refines to ~1 ULP on purpose. Same band as the KDA kernel's.
FP32_TOL = dict(atol=1e-5, rtol=1e-5)
# bf16 in, fp32 accumulate, bf16 out. One bf16 ULP near 1.0 is 7.8e-3, so this
# is the same "a few ULP" budget the KDA kernel is held to.
BF16_TOL = dict(atol=2e-2, rtol=2e-2)
# The two *scorer* gradients are reductions over every one of the N*hidden
# terms, in a different order from autograd's, so their rounding grows with N
# while the per-element gradients' does not. MEASURED at N = 256, hidden = 256,
# num_blocks in {1, 3}: d_norm_weight max_abs 3.8e-6, d_proj_weight max_abs
# 1.2e-4 -- and both at max_rel ~6e-5, i.e. the absolute number is large only
# because d_proj_weight itself is (it sums 256 x 256 terms). The band is set from
# that measurement with ~4x of headroom, and the numbers are reported in
# flydsl_modules/FINDINGS.md rather than hidden behind it. For scale, the
# injected backward defects move these by 5.7e0 and 2.2e2.
FP32_SUM_TOL = dict(atol=5e-4, rtol=1e-4)


def _on_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    return arch.startswith("gfx950")


pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="the FlyDSL attention-residual kernel is built for gfx950 (CDNA4)"
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _make_inputs(num_tokens, num_blocks, hidden, dtype, device="cuda", seed=0, scale=1.0):
    """Inputs shaped like a real mixer call.

    ``norm_weight`` is ones-plus-noise rather than ``randn``: a fresh mixer
    initialises it to exactly ones (``attention_residual.py:reset_parameters``)
    and it is an RMSNorm *gain*, so a sign-flipping draw would not be
    representative. ``proj_weight`` follows ``init_method_std = 0.02``.
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)

    prefix_sum = rnd(num_tokens, hidden) * scale
    block_residual = rnd(num_tokens, num_blocks, hidden) * scale
    norm_weight = 1.0 + 0.1 * rnd(hidden)
    proj_weight = 0.02 * rnd(1, hidden)
    return tuple(
        x.to(dtype) for x in (prefix_sum, block_residual, norm_weight, proj_weight)
    )


def _grads(backend, inputs, grad_out, eps):
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    out = backend(*leaves, eps)
    out.backward(grad_out)
    return out.detach(), [t.grad for t in leaves]


# ---------------------------------------------------------------------------
# forward parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens,num_blocks,hidden",
    [
        (8, 1, 64),  # smallest legal: one wave, one checkpoint
        (64, 2, 64),
        (64, 3, 256),  # 256 == BLOCK exactly, one element per thread
        (128, 8, 128),  # the 93-layer release's candidate count
        (37, 3, 512),  # token count not a multiple of anything
        (4096, 3, 2048),  # the scaled config's per-microbatch mixer call
    ],
)
def test_forward_matches_eager_fp32(num_tokens, num_blocks, hidden):
    backend = resolve_attn_res_backend("flydsl")
    inputs = _make_inputs(num_tokens, num_blocks, hidden, torch.float32)
    torch.testing.assert_close(
        backend(*inputs, 1e-5), eager_attn_res_mix(*inputs, 1e-5), **FP32_TOL
    )


@pytest.mark.parametrize("num_blocks", [1, 3, 8])
def test_forward_matches_eager_bf16(num_blocks):
    """bf16 inputs against the bf16 eager path, which up-casts internally too."""
    backend = resolve_attn_res_backend("flydsl")
    inputs = _make_inputs(512, num_blocks, 256, torch.bfloat16)
    torch.testing.assert_close(
        backend(*inputs, 1e-5), eager_attn_res_mix(*inputs, 1e-5), **BF16_TOL
    )


def test_forward_survives_a_tiny_residual_stream():
    """``eps`` is the only thing keeping the rsqrt finite when a candidate is 0.

    Not a hypothetical: ``block_residual`` starts as literal zeros at layer 0 of
    a fresh model, and the mixer is called on it.
    """
    backend = resolve_attn_res_backend("flydsl")
    prefix_sum, block_residual, norm_weight, proj_weight = _make_inputs(
        64, 2, 128, torch.float32
    )
    block_residual = torch.zeros_like(block_residual)
    inputs = (prefix_sum, block_residual, norm_weight, proj_weight)
    got = backend(*inputs, 1e-5)
    assert torch.isfinite(got).all(), "a zero candidate must not produce inf/nan"
    torch.testing.assert_close(got, eager_attn_res_mix(*inputs, 1e-5), **FP32_TOL)


def test_forward_survives_a_large_score_spread():
    """The softmax max subtraction, at a spread that would overflow without it."""
    backend = resolve_attn_res_backend("flydsl")
    inputs = _make_inputs(256, 3, 256, torch.float32, scale=50.0)
    got = backend(*inputs, 1e-5)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, eager_attn_res_mix(*inputs, 1e-5), **FP32_TOL)


def test_forward_preserves_the_leading_shape():
    """Megatron is sequence-first, so the mixer sees ``[s, b, ...]``, not ``[n, ...]``."""
    backend = resolve_attn_res_backend("flydsl")
    seq, batch, num_blocks, hidden = 32, 4, 2, 128
    prefix_sum, block_residual, norm_weight, proj_weight = _make_inputs(
        seq * batch, num_blocks, hidden, torch.float32
    )
    ps3 = prefix_sum.view(seq, batch, hidden)
    br4 = block_residual.view(seq, batch, num_blocks, hidden)
    got = backend(ps3, br4, norm_weight, proj_weight, 1e-5)
    assert got.shape == (seq, batch, hidden)
    torch.testing.assert_close(
        got.reshape(seq * batch, hidden),
        eager_attn_res_mix(prefix_sum, block_residual, norm_weight, proj_weight, 1e-5),
        **FP32_TOL,
    )


# ---------------------------------------------------------------------------
# backward parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,tol", [(torch.float32, FP32_TOL), (torch.bfloat16, BF16_TOL)])
@pytest.mark.parametrize("num_blocks", [1, 3])
def test_backward_matches_eager(dtype, tol, num_blocks):
    """All four input gradients, against autograd through the eager reference."""
    backend = resolve_attn_res_backend("flydsl")
    inputs = _make_inputs(256, num_blocks, 256, dtype)
    grad_out = torch.randn(256, 256, device="cuda", dtype=dtype)

    out_fly, grads_fly = _grads(backend, inputs, grad_out, 1e-5)
    out_ref, grads_ref = _grads(eager_attn_res_mix, inputs, grad_out, 1e-5)

    torch.testing.assert_close(out_fly, out_ref, **tol)
    names = ("d_prefix_sum", "d_block_residual", "d_norm_weight", "d_proj_weight")
    for name, gf, gr in zip(names, grads_fly, grads_ref):
        assert gf is not None, f"{name} did not reach its leaf"
        assert torch.isfinite(gf).all(), f"{name} is not finite"
        band = tol
        if dtype == torch.float32 and name in ("d_norm_weight", "d_proj_weight"):
            band = FP32_SUM_TOL
        torch.testing.assert_close(gf, gr, msg=lambda m, n=name: f"{n}: {m}", **band)


def test_backward_reaches_both_scorer_factors_separately():
    """``norm_weight`` and ``proj_weight`` are separate leaves, not one fused vector.

    The kernel only ever sees the fused ``[hidden]`` product, so the split is
    done by plain autograd on the torch side; this pins that both halves get a
    gradient and that they are the analytically correct pair rather than equal.
    """
    backend = resolve_attn_res_backend("flydsl")
    prefix_sum, block_residual, norm_weight, proj_weight = _make_inputs(
        128, 2, 128, torch.float32
    )
    grad_out = torch.randn(128, 128, device="cuda")
    _, (_, _, d_norm, d_proj) = _grads(
        backend, (prefix_sum, block_residual, norm_weight, proj_weight), grad_out, 1e-5
    )
    # d_norm = dW * proj and d_proj = dW * norm for the same dW, so
    # d_norm * norm == d_proj.squeeze(0) * proj.squeeze(0) elementwise.
    torch.testing.assert_close(
        d_norm * norm_weight, d_proj.squeeze(0) * proj_weight.squeeze(0), **FP32_TOL
    )
    assert d_norm.abs().max() > 0
    assert d_proj.abs().max() > 0


def test_forward_and_backward_agree_on_the_saved_softmax():
    """The backward recomputes ``p`` from the forward's saved ``r`` and ``dot``.

    If those two disagreed the gradient would be subtly wrong everywhere and no
    single-tensor comparison would localise it, so this checks the property
    directly: gradients through the kernel must match a *finite-difference*
    estimate, which shares no code with either implementation.
    """
    backend = resolve_attn_res_backend("flydsl")
    inputs = _make_inputs(16, 2, 64, torch.float32)
    prefix_sum, block_residual, norm_weight, proj_weight = inputs
    grad_out = torch.randn(16, 64, device="cuda")

    _, (d_ps, _, _, _) = _grads(backend, inputs, grad_out, 1e-5)

    h = 1e-3
    idx = (3, 17)
    bumped = prefix_sum.clone()
    bumped[idx] += h
    up = (backend(bumped, block_residual, norm_weight, proj_weight, 1e-5) * grad_out).sum()
    bumped[idx] -= 2 * h
    dn = (backend(bumped, block_residual, norm_weight, proj_weight, 1e-5) * grad_out).sum()
    fd = float((up - dn) / (2 * h))
    assert abs(fd - float(d_ps[idx])) < 2e-2 * max(1.0, abs(fd)), (
        f"analytic d_prefix_sum {float(d_ps[idx])} disagrees with the central "
        f"difference {fd}"
    )


# ---------------------------------------------------------------------------
# bug injection -- the tests above are only worth what these prove
# ---------------------------------------------------------------------------


def _injection_names():
    from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._flydsl_v1 import (
        BWD_INJECTIONS,
        FWD_INJECTIONS,
        FWD_NEUTRAL_VARIANTS,
    )

    return FWD_INJECTIONS, BWD_INJECTIONS, FWD_NEUTRAL_VARIANTS


@pytest.fixture
def clean_kernels():
    """Guarantee the injected kernel does not leak into any other test."""
    from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._flydsl_v1 import (
        inject_defect,
    )

    yield inject_defect
    inject_defect()


def _fixture_for(defect):
    """The inputs at which ``defect`` is observable.

    Two of the four defects are algebraically harmless at a generic input and
    only bite in a specific regime, so a single shared fixture would let them
    through — which is itself worth knowing, and is why each gets its own:

    ``no_softmax_max``
        needs a large score spread. Two measured details. First, *scaling the
        inputs does not work*: the score is ``<v/rms(v), w>``, which is invariant
        to the scale of ``v``, so the spread has to come from ``proj_weight``.
        Second, the spread has to be **well** past the overflow point, not just
        past it: at ``|score| ~ 360`` the injected kernel does produce ``nan``,
        but the *clean* kernel also drifts outside the fp32 band there (measured
        slack ``+7.8e-6``), because the softmax is then almost one-hot and the
        surviving runner-up term is relatively imprecise. At ``|score| ~ 2400``
        the softmax is exactly one-hot, the clean kernel is comfortably inside
        the band (slack ``-9.0e-6``) and the injected one is ``nan``.
    ``drop_eps``
        needs a candidate whose norm is (near-)zero, so that ``eps`` is the only
        thing keeping the rsqrt finite. ``block_residual`` really is all zeros at
        layer 0 of a fresh model, so this is the production case, not a contrived
        one.
    """
    prefix_sum, block_residual, norm_weight, proj_weight = _make_inputs(
        256, 3, 256, torch.float32, seed=17
    )
    if defect == "no_softmax_max":
        # |score| reaches ~2400, and exp(2400) overflows fp32 (max ~ exp(88.7))
        # without the max subtraction. See the docstring for why 300 is not
        # enough even though it already overflows.
        proj_weight = proj_weight * 2000.0
    elif defect == "drop_eps":
        block_residual = torch.zeros_like(block_residual)
    return prefix_sum, block_residual, norm_weight, proj_weight


@pytest.mark.parametrize("defect", _injection_names()[0])
def test_injected_forward_defect_is_caught(defect, clean_kernels):
    """Every named forward defect must fail the very assertion that passes clean."""
    inputs = _fixture_for(defect)
    ref = eager_attn_res_mix(*inputs, 1e-5)
    assert torch.isfinite(ref).all(), "the eager oracle itself must be finite here"

    backend = resolve_attn_res_backend("flydsl")
    torch.testing.assert_close(backend(*inputs, 1e-5), ref, **FP32_TOL)  # sanity

    clean_kernels(fwd=defect)
    got = backend(*inputs, 1e-5)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, ref, **FP32_TOL)


@pytest.mark.parametrize("defect", _injection_names()[1])
def test_injected_backward_defect_is_caught(defect, clean_kernels):
    """Every named backward defect must fail the gradient assertions."""
    inputs = _make_inputs(256, 3, 256, torch.float32)
    grad_out = torch.randn(256, 256, device="cuda")
    _, grads_ref = _grads(eager_attn_res_mix, inputs, grad_out, 1e-5)

    backend = resolve_attn_res_backend("flydsl")
    _, grads_clean = _grads(backend, inputs, grad_out, 1e-5)
    bands = (FP32_TOL, FP32_TOL, FP32_SUM_TOL, FP32_SUM_TOL)
    for gc, gr, band in zip(grads_clean, grads_ref, bands):  # sanity
        torch.testing.assert_close(gc, gr, **band)

    clean_kernels(bwd=defect)
    _, grads_bad = _grads(backend, inputs, grad_out, 1e-5)
    caught = []
    names = ("d_prefix_sum", "d_block_residual", "d_norm_weight", "d_proj_weight")
    for name, gb, gr, band in zip(names, grads_bad, grads_ref, bands):
        try:
            torch.testing.assert_close(gb, gr, **band)
        except AssertionError:
            caught.append(name)
    assert caught, f"backward defect {defect!r} passed every gradient comparison"


@pytest.mark.parametrize("variant", _injection_names()[2])
def test_neutral_variant_is_inside_the_band_and_that_is_the_point(variant, clean_kernels):
    """Two build-time variants are *measured* to be harmless. Pin the measurement.

    These are the two things one would expect to be defects and that are not:
    dropping the Newton refinement of ``rsqrt``, and reordering the candidates.
    Asserting them keeps the claim honest — if a future change made either of
    them matter, this test fails and the docstring stops being true.

    ``stream_first`` is *mathematically* an exact no-op — softmax is
    permutation-equivariant and the output sums over every candidate — but it is
    measured **not bit-identical**, because permuting the candidates permutes the
    order of the FMA chain that accumulates them. Measured max abs difference
    against the unpermuted kernel is one fp32 ULP territory, well inside the
    band. That distinction is the useful one: candidate *order* cannot be
    validated at this level at all, only one level up where a checkpoint slot has
    to line up with the layer that wrote it.
    """
    inputs = _make_inputs(256, 3, 256, torch.float32, seed=23)
    ref = eager_attn_res_mix(*inputs, 1e-5)
    backend = resolve_attn_res_backend("flydsl")
    clean = backend(*inputs, 1e-5)

    clean_kernels(fwd=variant)
    got = backend(*inputs, 1e-5)

    torch.testing.assert_close(got, ref, **FP32_TOL)
    if variant == "stream_first":
        torch.testing.assert_close(got, clean, **FP32_TOL)


def test_an_unknown_injection_name_raises():
    """A typo in a test must not silently mean "no defect injected"."""
    from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._flydsl_v1.attn_res_mixer_kernel import (
        build_attn_res_mixer_fwd,
    )

    with pytest.raises(ValueError, match="unknown injection"):
        build_attn_res_mixer_fwd(
            hidden=64, num_blocks=1, elem_dtype="f32", eps=1e-5, inject="mix_normalized"
        )


# ---------------------------------------------------------------------------
# geometry refusals and dispatch
# ---------------------------------------------------------------------------


def test_rejects_an_unsupported_hidden_size():
    """An unusable geometry must name the fallback, not fail inside a compile."""
    backend = resolve_attn_res_backend("flydsl")
    inputs = _make_inputs(8, 1, 100, torch.float32)  # 100 % 64 != 0
    with pytest.raises(ValueError, match="attn_res_backend: eager"):
        backend(*inputs, 1e-5)


def test_rejects_an_unsupported_dtype():
    backend = resolve_attn_res_backend("flydsl")
    inputs = _make_inputs(8, 1, 64, torch.float16)
    with pytest.raises(ValueError, match="attn_res_backend: eager"):
        backend(*inputs, 1e-5)


def test_rejects_a_cpu_tensor():
    backend = resolve_attn_res_backend("flydsl")
    inputs = tuple(t.cpu() for t in _make_inputs(8, 1, 64, torch.float32))
    with pytest.raises(ValueError, match="attn_res_backend: eager"):
        backend(*inputs, 1e-5)


def test_zero_checkpoints_routes_to_eager_and_returns_the_stream():
    """``num_blocks == 0`` is a softmax over one candidate, i.e. the identity."""
    backend = resolve_attn_res_backend("flydsl")
    prefix_sum, _, norm_weight, proj_weight = _make_inputs(16, 1, 64, torch.float32)
    empty = prefix_sum.new_zeros(16, 0, 64)
    got = backend(prefix_sum, empty, norm_weight, proj_weight, 1e-5)
    torch.testing.assert_close(got, prefix_sum, **FP32_TOL)


def test_mismatched_hidden_is_rejected():
    backend = resolve_attn_res_backend("flydsl")
    prefix_sum, _, norm_weight, proj_weight = _make_inputs(8, 1, 64, torch.float32)
    wrong = torch.zeros(8, 1, 128, device="cuda")
    with pytest.raises(ValueError, match="block_residual hidden"):
        backend(prefix_sum, wrong, norm_weight, proj_weight, 1e-5)


def test_backend_registry_and_unknown_name():
    assert set(ATTN_RES_BACKENDS) == {"eager", "flydsl"}
    assert resolve_attn_res_backend("eager") is eager_attn_res_mix
    with pytest.raises(ValueError, match="Unknown attention-residual backend"):
        resolve_attn_res_backend("triton_v99")


def test_module_selects_the_kernel_from_the_config():
    """``attn_res_backend: flydsl`` must reach the mixer, resolved at construction."""
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
        KimiK3TransformerConfig,
    )
    from primus.backends.megatron.core.transformer.kimi_k3.attention_residual import (
        AttentionResidualMixer,
    )

    def _config(backend):
        return KimiK3TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=8,
            kv_channels=128,
            layernorm_epsilon=1e-5,
            params_dtype=torch.float32,
            activation_func=torch.nn.functional.silu,
            init_method_std=0.02,
            attn_res_backend=backend,
        )

    eager_mixer = AttentionResidualMixer(config=_config("eager")).cuda()
    fly_mixer = AttentionResidualMixer(config=_config("flydsl")).cuda()
    assert eager_mixer.backend_name == "eager"
    assert eager_mixer.attn_res_backend is eager_attn_res_mix
    assert fly_mixer.backend_name == "flydsl"
    assert fly_mixer.attn_res_backend is not eager_attn_res_mix

    # and the two agree, with the same parameters
    with torch.no_grad():
        fly_mixer.norm_weight.copy_(eager_mixer.norm_weight)
        fly_mixer.proj_weight.copy_(eager_mixer.proj_weight)
    prefix_sum = torch.randn(64, 128, device="cuda")
    block_residual = torch.randn(64, 2, 128, device="cuda")
    torch.testing.assert_close(
        fly_mixer(prefix_sum, block_residual),
        eager_mixer(prefix_sum, block_residual),
        **FP32_TOL,
    )

    with pytest.raises(ValueError, match="attn_res_backend must be one of"):
        AttentionResidualMixer(config=_config("triton_v99"))


def test_score_weight_factorisation_is_shared_by_both_backends():
    """One copy of ``norm_weight ⊙ proj_weight``, used by eager and by the kernel."""
    norm_weight = torch.randn(64, device="cuda")
    proj_weight = torch.randn(1, 64, device="cuda")
    torch.testing.assert_close(
        fused_score_weight(norm_weight, proj_weight),
        norm_weight.float() * proj_weight.squeeze(0).float(),
    )


def test_importing_attn_res_kernels_does_not_import_flydsl():
    """The dispatcher must stay importable on a build or GPU without flydsl.

    Checked in a **subprocess**, because the assertion is about ``sys.modules``
    and this process has already imported flydsl. ``PYTHONPATH`` is inherited
    from ``sys.path`` so the child finds the in-tree Megatron the same way the
    parent did.
    """
    import subprocess
    import sys

    code = (
        "import sys, torch; "
        "import primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels as m; "
        "assert 'flydsl' not in sys.modules, sorted(k for k in sys.modules if 'flydsl' in k); "
        "print('ok')"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "ok" in out.stdout
