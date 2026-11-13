###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the attention-residual mixer backends in ``kimi_k3/attn_res_kernels``.

``eager`` is the numerical oracle. ``eager_bcast`` must match it on CPU up to
fp32 rounding; ``flydsl`` (gfx950 / gfx1250 only) must match it in values and
in all four gradients, including when one compiled kernel is reused for a
different token count (the launcher caches per hidden size and block count,
not per token count). On gfx1250 bf16 runs the vector kernels when hidden is a
multiple of 1024 and the scalar kernels otherwise; both are checked, and each
test-only defect the builders can inject has to fail the same parity check.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# kimi_k3/__init__.py imports megatron.core (kimi_delta_attention.py)
pytest.importorskip(
    "megatron.core.transformer.module",
    reason="megatron.core is not importable in this environment",
)

from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels import (  # noqa: E402
    ATTN_RES_BACKENDS,
    resolve_attn_res_backend,
)

EPS = 1e-6


def _inputs(tokens, hidden, num_blocks, dtype, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ps = torch.randn(tokens, hidden, generator=g).to(device=device, dtype=dtype)
    br = torch.randn(tokens, num_blocks, hidden, generator=g).to(device=device, dtype=dtype)
    nw = (1.0 + 0.1 * torch.randn(hidden, generator=g)).to(device)
    pw = (0.05 * torch.randn(1, hidden, generator=g)).to(device)
    dout = torch.randn(tokens, hidden, generator=g).to(device=device, dtype=dtype)
    return ps, br, nw, pw, dout


def _run(backend, ps, br, nw, pw, dout):
    fn = resolve_attn_res_backend(backend)
    ps, br, nw, pw = (t.detach().clone().requires_grad_() for t in (ps, br, nw, pw))
    out = fn(ps, br, nw, pw, EPS)
    out.backward(dout)
    return [t.detach().float() for t in (out, ps.grad, br.grad, nw.grad, pw.grad)]


def _rel_errors(ref, got):
    """Max-normalised error of each of the five results; NaN counts as infinite."""
    errs = []
    for a, b in zip(ref, got):
        assert a.shape == b.shape, f"shape {tuple(b.shape)} != {tuple(a.shape)}"
        err = 0.0 if a.numel() == 0 else ((a - b).abs().max() / a.abs().max().clamp_min(1e-30)).item()
        errs.append(float("inf") if err != err else err)
    return errs


def _assert_close(ref, got, rel):
    names = ("out", "d_prefix_sum", "d_block_residual", "d_norm_weight", "d_proj_weight")
    for name, err in zip(names, _rel_errors(ref, got)):
        assert err < rel, f"{name}: relative error {err:.2e} >= {rel:.0e}"


def test_backend_names():
    assert ATTN_RES_BACKENDS == ("eager", "eager_bcast", "flydsl")
    with pytest.raises(ValueError):
        resolve_attn_res_backend("cuda")


@pytest.mark.parametrize("num_blocks", [0, 1, 3])
@pytest.mark.parametrize("dtype, rel", [(torch.float64, 1e-12), (torch.float32, 1e-5)])
def test_eager_bcast_matches_eager(num_blocks, dtype, rel):
    args = _inputs(6, 16, num_blocks, dtype, "cpu", seed=num_blocks)
    _assert_close(_run("eager", *args), _run("eager_bcast", *args), rel)


def _flydsl_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("no ROCm / CUDA device")
    try:
        resolve_attn_res_backend("flydsl")
    except ImportError as e:
        pytest.skip(f"flydsl backend unavailable: {e}")


@pytest.mark.parametrize(
    "tokens, hidden, num_blocks, dtype, rel",
    [
        (512, 1024, 3, torch.float32, 1e-4),  # fp32: the scalar kernels, whatever the hidden size
        (256, 768, 2, torch.float32, 1e-4),
        (256, 768, 2, torch.bfloat16, 2e-2),  # bf16, hidden % 1024 != 0: the scalar kernels on gfx1250 too
        (8192, 7168, 1, torch.bfloat16, 2e-2),  # Kimi K3: seq 4096 x micro-batch 2, hidden 7168
        (8192, 7168, 3, torch.bfloat16, 2e-2),
    ],
)
def test_flydsl_matches_eager(tokens, hidden, num_blocks, dtype, rel):
    _flydsl_or_skip()
    args = _inputs(tokens, hidden, num_blocks, dtype, "cuda", seed=tokens + num_blocks)
    _assert_close(_run("eager", *args), _run("flydsl", *args), rel)


@pytest.mark.parametrize("dtype, rel", [(torch.float32, 1e-4), (torch.bfloat16, 2e-2)])
def test_flydsl_reuses_kernel_across_token_counts(dtype, rel):
    _flydsl_or_skip()
    for tokens in (1024, 3072, 1024):
        args = _inputs(tokens, 2048, 2, dtype, "cuda", seed=tokens)
        _assert_close(_run("eager", *args), _run("flydsl", *args), rel)


def test_vector_kernel_gate(monkeypatch):
    pytest.importorskip("flydsl")
    from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._flydsl_v1.attn_res_mixer_kernel import (
        use_vector_kernel,
        vector_kernel_knobs,
    )

    assert use_vector_kernel(7168, "bf16", "gfx1250")
    assert use_vector_kernel(2048, "bf16", "gfx1250:sramecc+:xnack-")
    assert not use_vector_kernel(768, "bf16", "gfx1250")  # hidden % 1024 != 0
    assert not use_vector_kernel(7168, "f32", "gfx1250")
    assert not use_vector_kernel(7168, "bf16", "gfx950")
    assert not use_vector_kernel(7168, "bf16", "gfx1250", allowed=False)

    monkeypatch.delenv("K3P_ATTNRES_WPE", raising=False)
    for off in ("0", "false", "OFF", " no "):
        monkeypatch.setenv("K3P_ATTNRES_VEC", off)
        assert vector_kernel_knobs() == (False, None)
    for on in ("1", "", "true"):
        monkeypatch.setenv("K3P_ATTNRES_VEC", on)
        assert vector_kernel_knobs() == (True, None)
    monkeypatch.setenv("K3P_ATTNRES_WPE", "3")
    assert vector_kernel_knobs() == (True, 3)
    for bad in ("0", "two", "-1"):
        monkeypatch.setenv("K3P_ATTNRES_WPE", bad)
        with pytest.raises(ValueError):
            vector_kernel_knobs()


@pytest.mark.parametrize("off", ["0", "off"])
def test_flydsl_scalar_fallback_matches_eager(monkeypatch, off):
    """``K3P_ATTNRES_VEC`` turned off inside the process builds the scalar kernels (it is in the cache key)."""
    _flydsl_or_skip()
    monkeypatch.setenv("K3P_ATTNRES_VEC", off)
    args = _inputs(1024, 2048, 2, torch.bfloat16, "cuda", seed=5)
    _assert_close(_run("eager", *args), _run("flydsl", *args), 2e-2)


def test_flydsl_misaligned_views_match_eager():
    """Inputs and the output gradient as views 2 bytes past a 16-byte boundary (the vector kernels load 16 B)."""
    _flydsl_or_skip()
    ps, br, nw, pw, dout = _inputs(512, 2048, 2, torch.bfloat16, "cuda", seed=9)
    ref = _run("eager", ps, br, nw, pw, dout)

    def shifted(t, grad=False):
        buf = torch.cat((t.new_zeros(1), t.reshape(-1))).requires_grad_(grad)
        return buf, buf[1:].view(t.shape)

    (ps_buf, ps_v), (br_buf, br_v), (_, dout_v) = shifted(ps, True), shifted(br, True), shifted(dout)
    assert ps_v.data_ptr() % 16 and br_v.data_ptr() % 16 and dout_v.data_ptr() % 16
    nw, pw = (t.detach().clone().requires_grad_() for t in (nw, pw))
    out = resolve_attn_res_backend("flydsl")(ps_v, br_v, nw, pw, EPS)
    out.backward(dout_v)
    got = [out.detach(), ps_buf.grad[1:].view(ps.shape), br_buf.grad[1:].view(br.shape), nw.grad, pw.grad]
    _assert_close(ref, [t.float() for t in got], 2e-2)


#: (hidden, dtype, rel): on gfx1250 the bf16 row builds the vector kernels and the fp32 row the scalar ones
_KERNEL_PATHS = [(2048, torch.bfloat16, 2e-2), (768, torch.float32, 1e-4)]
_FWD_DEFECTS = ("mix_normalised", "score_unnormalised", "no_softmax_max", "drop_eps")
_BWD_DEFECTS = ("no_softmax_jacobian", "drop_rms_term", "drop_score_term", "wrong_r_power")


def _defect_inputs(defect, tokens, hidden, dtype):
    """Inputs on which ``defect`` is visible at the parity tolerance, two checkpoints and the stream.

    Each candidate leans on the scorer direction by ``gamma``, so the RMS term of the backward is as
    large as the scorer term and the scores stay of order one (the softmax is not one-hot). With
    independent random candidates the RMS term is ~1/sqrt(hidden) of the gradient, under bf16 rounding.
    The scale of 4 keeps the scores and the true gradients as they are and puts ``r`` near 1/4, so
    ``r``, ``r^2`` and ``r^3`` are far apart.
    """
    g = torch.Generator(device="cpu").manual_seed(11)
    w_dir = torch.randn(hidden, generator=g)
    w_dir = w_dir / w_dir.norm()
    gamma = torch.tensor([0.5, 1.0, 1.5]).view(1, 3, 1)
    v = 4.0 * (torch.randn(tokens, 3, hidden, generator=g) + gamma * w_dir * hidden**0.5)
    nw = torch.ones(hidden)
    pw = (w_dir / hidden**0.5).unsqueeze(0)
    if defect == "no_softmax_max":
        pw = pw * 1000.0  # scores of ~1e3: exp overflows unless the max is subtracted
    if defect == "drop_eps":
        v[:, 0] = 0.0  # a zero-norm checkpoint, as block_residual is at layer 0 of a fresh model
    dout = torch.randn(tokens, hidden, generator=g)
    ps, br = v[:, 2].contiguous(), v[:, :2].contiguous()
    return ps.to("cuda", dtype), br.to("cuda", dtype), nw.to("cuda"), pw.to("cuda"), dout.to("cuda", dtype)


@pytest.mark.parametrize("hidden, dtype, rel", _KERNEL_PATHS)
@pytest.mark.parametrize(
    "direction, defect", [("fwd", d) for d in _FWD_DEFECTS] + [("bwd", d) for d in _BWD_DEFECTS]
)
def test_flydsl_parity_catches_injected_defects(direction, defect, hidden, dtype, rel):
    """The negative control: the correct kernel passes on these inputs and the defective one does not."""
    _flydsl_or_skip()
    from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._flydsl_v1 import (
        BWD_INJECTIONS,
        FWD_INJECTIONS,
        inject_defect,
    )

    assert (FWD_INJECTIONS, BWD_INJECTIONS) == (_FWD_DEFECTS, _BWD_DEFECTS)
    args = _defect_inputs(defect, 512, hidden, dtype)
    ref = _run("eager", *args)
    # a forward defect is judged on the output, a backward one on the four gradients
    judged = slice(0, 1) if direction == "fwd" else slice(1, 5)
    good = _rel_errors(ref, _run("flydsl", *args))[judged]
    assert max(good) < rel, f"the correct kernel fails on the {defect!r} inputs: {good}"
    inject_defect(**{direction: defect})
    try:
        bad = _rel_errors(ref, _run("flydsl", *args))[judged]
    finally:
        inject_defect()
    assert max(bad) >= rel, f"{direction} defect {defect!r} passes the parity check: {bad}"


@pytest.mark.parametrize("hidden, dtype, rel", _KERNEL_PATHS)
@pytest.mark.parametrize("variant", ["no_newton", "stream_first"])
def test_flydsl_neutral_variants_stay_in_band(variant, hidden, dtype, rel):
    """The forward variants documented as harmless keep the output inside the parity band."""
    _flydsl_or_skip()
    from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._flydsl_v1 import (
        FWD_NEUTRAL_VARIANTS,
        inject_defect,
    )

    assert variant in FWD_NEUTRAL_VARIANTS
    args = _defect_inputs("", 512, hidden, dtype)
    ref = _run("eager", *args)
    inject_defect(fwd=variant)
    try:
        err = _rel_errors(ref, _run("flydsl", *args))[0]
    finally:
        inject_defect()
    assert err < rel, f"variant {variant!r}: output error {err:.2e} >= {rel:.0e}"
