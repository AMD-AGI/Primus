###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Muse Spark fused per-expert post norm + routing prob (``post_expert_norm.py``).

CPU: the eager entry point matches the norm module path, and the Triton kernels match
eager autograd under the Triton interpreter (``TRITON_INTERPRET=1``, set by this file
before Triton is imported in a subprocess). GPU: ``test_fused_kernel_matches_eager_on_the_gpu``
runs the compiled kernels forward and backward (skipped without a GPU).
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

MOD = (
    Path(__file__).resolve().parents[5]
    / "primus/backends/megatron/core/transformer/muse_spark/post_expert_norm.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("muse_spark_post_expert_norm", MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eager(y, w, p, eps):
    # independent of the module: scale, round, normalize in fp32, round, prob, round
    s = (y.double() * (1 + w.double())).to(y.dtype).double()
    n = (s / torch.sqrt((s * s).mean(-1, keepdim=True) + eps)).to(y.dtype).double()
    return (n * p.double()[:, None]).to(y.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reference_matches_an_independent_formula(dtype):
    m = _load()
    g = torch.Generator().manual_seed(0)
    y = (torch.randn(9, 40, generator=g) * 5).to(dtype)
    w = (torch.randn(40, generator=g) * 0.3).to(dtype)
    p = torch.rand(9, generator=g)
    got = m.post_expert_norm_probs_reference(y, w, p, 1e-8)
    want = _eager(y, w, p, 1e-8)
    assert got.dtype == dtype
    torch.testing.assert_close(
        got.float(), want.float(), rtol=1e-2 if dtype != torch.float32 else 1e-5, atol=0
    )


def test_cpu_entry_point_is_the_eager_path_with_grads():
    m = _load()
    y = torch.randn(5, 16, requires_grad=True)
    w = torch.zeros(16, requires_grad=True)
    p = torch.rand(5, requires_grad=True)
    assert not m.fused_rms_norm_available(y)
    out = m.post_expert_norm_probs(y, w, p, 1e-8)
    out.sum().backward()
    assert y.grad is not None and w.grad is not None and p.grad is not None
    # unit weight scale (w = 0): each row is unit RMS times its prob
    rms = out.detach().pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, p.detach(), rtol=1e-5, atol=1e-6)


class _ReportsCuda(torch.Tensor):
    """A CPU tensor that says it is on the GPU, so only the width decides availability."""

    @property
    def is_cuda(self):
        return True


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_width_bound_decides_availability(dtype):
    # each clause of the availability check is tested alone (the others hold)
    m = _load()
    if m.triton is None:
        pytest.skip("triton is not installed")
    at = torch.empty(1, m.MAX_FUSED_HIDDEN, dtype=dtype).as_subclass(_ReportsCuda)
    wide = torch.empty(1, m.MAX_FUSED_HIDDEN + 1, dtype=dtype).as_subclass(_ReportsCuda)
    assert at.is_cuda and wide.is_cuda
    assert m.fused_rms_norm_available(at)
    assert not m.fused_rms_norm_available(wide)
    assert m.MAX_FUSED_HIDDEN == 16384


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_wide_rows_on_the_gpu_use_the_eager_path():
    m = _load()
    if m.triton is None:
        pytest.skip("triton is not installed")
    assert m.fused_rms_norm_available(torch.empty(1, m.MAX_FUSED_HIDDEN, device="cuda"))
    h = m.MAX_FUSED_HIDDEN + 1
    y = torch.randn(3, h, device="cuda")
    assert not m.fused_rms_norm_available(y)
    w = torch.randn(h, device="cuda") * 0.1
    p = torch.rand(3, device="cuda")
    torch.testing.assert_close(
        m.post_expert_norm_probs(y, w, p, 1e-8), m.post_expert_norm_probs_reference(y, w, p, 1e-8)
    )


_INTERP = r"""
import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location("m", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
worst = 0.0
for dt, tol in ((torch.float32, 1e-5), (torch.float16, 2e-3)):
    for t, h in ((0, 96), (1, 64), (7, 96), (33, 130)):
        g = torch.Generator().manual_seed(t * h)
        y = (torch.randn(t, h, generator=g) * 3).to(dt).requires_grad_()
        w = (torch.randn(h, generator=g) * 0.3).to(dt).requires_grad_()
        p = torch.rand(t, generator=g).requires_grad_()
        go = torch.randn(t, h, generator=g).to(dt)
        ref = m.post_expert_norm_probs_reference(y, w, p, 1e-8)
        rg = torch.autograd.grad(ref, (y, w, p), go)
        out = m._PostExpertNormProbs.apply(y, w, p, 1e-8)
        kg = torch.autograd.grad(out, (y, w, p), go)
        if t == 0:  # an expert with no routed token
            assert out.shape == (0, h) and kg[0].shape == (0, h) and kg[2].shape == (0,)
            assert kg[1].shape == (h,) and not kg[1].any(), kg[1]
            continue
        for a, b in zip((out, *kg), (ref, *rg)):
            e = float((a.float() - b.float()).abs().max() / b.float().abs().max())
            assert e <= tol, (dt, t, h, e)
        # probs=None (the norm alone, HAS_P=False) is bit-exact to unit probs, both passes
        ones = torch.ones(t, requires_grad=True)
        o1 = m._PostExpertNormProbs.apply(y, w, ones, 1e-8)
        g1 = torch.autograd.grad(o1, (y, w), go)
        o0 = m._PostExpertNormProbs.apply(y, w, None, 1e-8)
        g0 = torch.autograd.grad(o0, (y, w), go)
        assert torch.equal(o0, o1) and all(torch.equal(a, b) for a, b in zip(g0, g1)), (dt, t, h)
        assert torch.equal(m.post_expert_norm_probs_reference(y, w, None, 1e-8),
                           m.post_expert_norm_probs_reference(y, w, torch.ones(t), 1e-8))
print("ok")
"""


def test_triton_kernels_match_eager_autograd_in_the_interpreter():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1")
    r = subprocess.run([sys.executable, "-c", _INTERP, str(MOD)], env=env, capture_output=True, text=True)
    reason = _interpreter_limit(r.stderr) if r.returncode else None
    if reason:
        pytest.skip(
            f"this Triton build's interpreter cannot run the kernels ({reason}); the GPU test covers them"
        )
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def _interpreter_limit(stderr):
    """The interpreter limit named in ``stderr``, else None (a real failure).

    Some Triton builds (e.g. 3.6) have no interpreter, or one that cannot take a runtime
    scalar as a ``range()`` bound (the backward kernel's ``for k in range(0,
    tiles_per_prog)``): an ``InterpreterError`` with "only 0-dimensional arrays can be
    converted to Python scalars". Those are limits of the interpreter, not of the kernels.
    Any other error (a wrong value, an assert, an exception in the kernels' Python) fails."""
    if "TRITON_INTERPRET" in stderr and "InterpreterError" not in stderr and "AssertionError" not in stderr:
        return "no interpreter"
    if (
        "InterpreterError" in stderr
        and "only 0-dimensional arrays can be converted to Python scalars" in stderr
    ):
        return "a runtime scalar as a range() bound"
    return None


@pytest.mark.parametrize(
    "stderr, limit",
    [
        (
            "triton.runtime.errors.InterpreterError: TypeError('only 0-dimensional arrays can be "
            "converted to Python scalars')",
            "a runtime scalar as a range() bound",
        ),
        ("RuntimeError: TRITON_INTERPRET is not supported", "no interpreter"),
        ("AssertionError: (torch.float32, 7, 33, 0.5)", None),  # a wrong value
        ("triton.runtime.errors.InterpreterError: IndexError('out of range')", None),  # a kernel bug
        ("ValueError: bad shape", None),
    ],
)
def test_only_a_known_interpreter_limit_skips(stderr, limit):
    assert _interpreter_limit(stderr) == limit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("t, h", [(128, 1000), (2048, 4096), (1, 4096), (37, 1000), (0, 1000)])
@pytest.mark.parametrize("probs_2d", [False, True], ids=["probs_T", "probs_T1"])
def test_fused_kernel_matches_eager_on_the_gpu(dtype, tol, t, h, probs_2d):
    # the compiled kernels, forward and backward (H = 1000: not a power of two, the column
    # mask; 4096: the released width; T = 0: an expert with no routed token)
    m = _load()
    if m.triton is None:
        pytest.skip("triton is not installed")
    g = torch.Generator(device="cuda").manual_seed(t * h)
    scale = 10.0 ** (torch.rand(t, 1, generator=g, device="cuda") * 4 - 2)  # row RMS 1e-2..1e2
    y = (torch.randn(t, h, generator=g, device="cuda") * scale).to(dtype).requires_grad_()
    w = (torch.randn(h, generator=g, device="cuda") * 0.3).to(dtype).requires_grad_()
    p_shape = (t, 1) if probs_2d else (t,)  # the [T] and [T, 1] probs contract
    p = torch.rand(p_shape, generator=g, device="cuda").requires_grad_()
    go = torch.randn(t, h, generator=g, device="cuda").to(dtype)
    assert m.fused_rms_norm_available(y)
    ref = m.post_expert_norm_probs_reference(y, w, p, 1e-8)
    ref_grads = torch.autograd.grad(ref, (y, w, p), go)
    out = m.post_expert_norm_probs(y, w, p, 1e-8)
    assert out.dtype == dtype and out.shape == (t, h)
    grads = torch.autograd.grad(out, (y, w, p), go)
    if t == 0:
        assert grads[0].shape == (0, h) and grads[2].shape == p_shape
        assert grads[1].shape == (h,) and torch.isfinite(grads[1]).all() and not grads[1].any()
        return
    assert grads[2].shape == p_shape  # dp keeps the probs' own shape
    for name, a, b in zip(("out", "dy", "dw", "dp"), (out, *grads), (ref, *ref_grads)):
        assert a.dtype == b.dtype, name
        err = float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30))
        assert err <= tol, (name, dtype, t, h, err)


def _spy_fused_kernel(monkeypatch, norms):
    """Record each call of the Triton entry point the norms use (it still runs): a regression
    that makes the module fall back to the eager formula fails the test."""
    from primus.backends.megatron.core.transformer.muse_spark import post_expert_norm

    launched = []
    real_apply = post_expert_norm._PostExpertNormProbs.apply

    def apply(y, weight, probs, eps):
        launched.append(tuple(y.shape))
        return real_apply(y, weight, probs, eps)

    monkeypatch.setattr(post_expert_norm._PostExpertNormProbs, "apply", staticmethod(apply))
    return launched


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
def test_the_gated_residual_post_norm_uses_the_fused_kernel_on_the_gpu(monkeypatch, dtype, tol):
    # MuseSparkInputScaledRMSNorm on [S, B, H]: the fused kernel (prob 1) equals the eager norm,
    # and the fused module did run the Triton kernel (a spy calls through to it)
    from types import SimpleNamespace

    from primus.backends.megatron.core.transformer.muse_spark import (
        norms,
        post_expert_norm,
    )
    from primus.backends.megatron.core.transformer.muse_spark.norms import (
        MuseSparkInputScaledRMSNorm,
    )

    if post_expert_norm.triton is None:
        pytest.skip("Triton is not available: the fused kernel cannot run")
    launched = _spy_fused_kernel(monkeypatch, norms)

    h = 1024
    fused = MuseSparkInputScaledRMSNorm(SimpleNamespace(hidden_size=h), h, eps=1e-6).cuda().to(dtype)
    eager = (
        MuseSparkInputScaledRMSNorm(
            SimpleNamespace(hidden_size=h, muse_spark_fused_post_expert_norm=False), h, eps=1e-6
        )
        .cuda()
        .to(dtype)
    )
    assert fused.fused and not eager.fused
    with torch.no_grad():
        fused.weight.normal_(0, 0.3)
        eager.weight.copy_(fused.weight)
    x = (torch.randn(7, 3, h, device="cuda") * 4).to(dtype)
    xf, xe = x.clone().requires_grad_(), x.clone().requires_grad_()
    of = fused(xf)
    assert launched == [(21, h)]  # the fused module ran the Triton kernel on the 7*3 rows
    oe = eager(xe)
    assert launched == [(21, h)]  # the eager module did not
    assert of.shape == x.shape and of.dtype == dtype
    torch.testing.assert_close(of.float(), oe.float(), rtol=tol, atol=tol)
    g = torch.randn_like(of)
    of.backward(g)
    oe.backward(g)
    torch.testing.assert_close(xf.grad.float(), xe.grad.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(fused.weight.grad.float(), eager.weight.grad.float(), rtol=tol, atol=tol * h)


def test_the_gated_residual_post_norm_is_eager_on_the_cpu():
    from types import SimpleNamespace

    from primus.backends.megatron.core.transformer.muse_spark.norms import (
        MuseSparkInputScaledRMSNorm,
    )

    norm = MuseSparkInputScaledRMSNorm(SimpleNamespace(hidden_size=16), 16, eps=1e-6)
    with torch.no_grad():
        norm.weight.normal_(0, 0.3)
    x = torch.randn(4, 2, 16, dtype=torch.bfloat16)
    s = (x.float() * (1 + norm.weight.float())).to(x.dtype).float()
    want = (s * torch.pow(s.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)).to(x.dtype)
    assert torch.equal(norm(x), want)


@pytest.mark.parametrize("fused", [True, False])
def test_the_weightless_norms_route_to_the_shared_rmsnorm(monkeypatch, fused):
    """The attention post_norm (weightless MuseSparkRMSNorm) calls Primus's shared RMSNorm
    (fused_rms_norm, no weight, out_dtype = x.dtype) when the config allows it, and the eager
    reference (eager_rms_norm) otherwise; the weighted Q / K norm passes its weight."""
    from types import SimpleNamespace

    from primus.backends.megatron.core.transformer.muse_spark import (
        centered_norm,
        norms,
    )

    calls = []

    def shared(x, weight, *, eps, mid_cast, out_dtype):
        calls.append((tuple(x.shape), weight is None, eps, mid_cast, out_dtype))
        return norms.eager_rms_norm(x, weight, eps=eps, mid_cast=mid_cast, out_dtype=out_dtype)

    monkeypatch.setattr(norms, "fused_rms_norm", shared)
    # the weighted norm reaches it through centered_norm.weighted_rms_norm
    monkeypatch.setattr(centered_norm, "fused_rms_norm", shared)
    # the attention flag alone decides; the MoE flag is the opposite and is not read here
    cfg = SimpleNamespace(
        hidden_size=8, muse_spark_fused_attn_norm=fused, muse_spark_fused_post_expert_norm=not fused
    )
    post = norms.MuseSparkRMSNorm(cfg, 8, eps=1e-6, with_scale=False)
    x = torch.randn(2, 3, 8)
    out = post(x)
    want = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)
    torch.testing.assert_close(out, want)
    assert calls == ([((2, 3, 8), True, 1e-6, False, torch.float32)] if fused else [])
    calls.clear()
    norms.MuseSparkRMSNorm(cfg, 8, eps=1e-6)(x)
    assert calls == [((2, 3, 8), False, 1e-6, False, torch.float32)]


@pytest.mark.parametrize("fused", [True, False])
def test_the_input_scaled_norm_routes_to_the_fused_kernel(monkeypatch, fused):
    """MuseSparkInputScaledRMSNorm (the gated residual's post-MoE norm) calls the kernel with
    its own weight and no probs when the kernel is available and the MoE flag allows it (a
    recorder stands in for the kernel on the CPU); with the flag off it runs the eager formula."""
    from types import SimpleNamespace

    from primus.backends.megatron.core.transformer.muse_spark import norms
    from primus.backends.megatron.core.transformer.muse_spark.post_expert_norm import (
        post_expert_norm_probs_reference,
    )

    calls = []

    def kernel(rows, weight, probs, eps):
        calls.append((tuple(rows.shape), weight, probs, eps))
        return post_expert_norm_probs_reference(rows, weight, probs, eps)

    monkeypatch.setattr(norms, "fused_rms_norm_available", lambda rows: True)
    monkeypatch.setattr(norms, "post_expert_norm_probs", kernel)
    # the MoE flag alone decides; the attention flag is the opposite and is not read here
    cfg = SimpleNamespace(
        hidden_size=8, muse_spark_fused_post_expert_norm=fused, muse_spark_fused_attn_norm=not fused
    )
    norm = norms.MuseSparkInputScaledRMSNorm(cfg, 8, eps=1e-8)
    with torch.no_grad():
        norm.weight.normal_(0, 0.1)
    x = torch.randn(2, 3, 8)
    out = norm(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out, post_expert_norm_probs_reference(x, norm.weight, None, 1e-8))
    if fused:
        assert len(calls) == 1
        shape, weight, probs, eps = calls[0]
        assert shape == (6, 8) and weight is norm.weight and probs is None and eps == 1e-8
    else:
        assert calls == []


def test_the_centered_norm_routes_to_the_shared_fused_rms_norm(monkeypatch):
    """MuseSparkCenteredRMSNorm calls Primus's shared fused_rms_norm with the fp32 weight
    1 + w, no mid cast and the input dtype (a recorder stands in for it)."""
    from primus.backends.megatron.core.transformer.muse_spark import (
        centered_norm,
        norms,
    )

    calls = []
    real = centered_norm.fused_rms_norm

    def recorder(x, weight, **kw):
        calls.append((tuple(x.shape), weight.dtype, kw))
        return real(x, weight, **kw)

    monkeypatch.setattr(centered_norm, "fused_rms_norm", recorder)
    norm = norms.MuseSparkCenteredRMSNorm(None, 8, eps=1e-5)
    with torch.no_grad():
        norm.weight.normal_(0, 0.1)
    x = torch.randn(2, 3, 8).to(torch.bfloat16)
    out = norm(x)
    assert torch.equal(out, centered_norm.centered_rms_norm_reference(x, norm.weight, 1e-5))
    assert calls == [((2, 3, 8), torch.float32, dict(eps=1e-5, mid_cast=False, out_dtype=torch.bfloat16))]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-5), (torch.bfloat16, 1e-2)])
def test_the_fused_weightless_norm_matches_eager_on_the_gpu(monkeypatch, dtype, tol):
    from primus.backends.megatron.core.transformer.muse_spark import norms
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        rmsnorm,
    )

    launched = []
    real_apply = rmsnorm.FusedRMSNormFn.apply

    def apply(x, *a):
        launched.append(tuple(x.reshape(-1, x.shape[-1]).shape))
        return real_apply(x, *a)

    monkeypatch.setattr(rmsnorm.FusedRMSNormFn, "apply", staticmethod(apply))
    monkeypatch.delenv("PRIMUS_RMSNORM_TRITON", raising=False)
    x = (torch.randn(5, 4, 128, device="cuda") * 3).to(dtype).requires_grad_()
    xe = x.detach().clone().requires_grad_()
    out = norms.weightless_rms(x, 1e-6, True)
    assert launched == [(20, 128)]  # the fused path ran the Triton kernel
    want = norms.weightless_rms(xe, 1e-6, False)
    assert launched == [(20, 128)]  # the eager path did not
    torch.testing.assert_close(out.float(), want.float(), rtol=tol, atol=tol)
    g = torch.randn_like(out)
    out.backward(g)
    want.backward(g)
    torch.testing.assert_close(x.grad.float(), xe.grad.float(), rtol=tol * 10, atol=tol * 10)


@pytest.mark.parametrize(
    "text, fused",
    [("false", False), ("0", False), ("off", False), ("true", True), (None, True), ("", True), ("  ", True)],
)
def test_the_fused_norm_flags_are_parsed_strictly(text, fused):
    """A string "false" turns the kernels off (raw truthiness would keep them on); an unset
    value (None, YAML ``~``, a blank env value) keeps the default (on). The attention and MoE
    norms read their own flag."""
    from types import SimpleNamespace

    from primus.backends.megatron.core.transformer.muse_spark import norms

    attn = norms.MuseSparkRMSNorm(
        SimpleNamespace(hidden_size=8, muse_spark_fused_attn_norm=text), 8, with_scale=False
    )
    moe = norms.MuseSparkInputScaledRMSNorm(
        SimpleNamespace(hidden_size=8, muse_spark_fused_post_expert_norm=text), 8
    )
    assert attn.fused is fused and moe.fused is fused
    # unset: both default on
    assert norms.MuseSparkRMSNorm(SimpleNamespace(hidden_size=8), 8, with_scale=False).fused is True


@pytest.mark.parametrize(
    "key",
    ["muse_spark_fused_attn_norm", "muse_spark_fused_post_expert_norm", "muse_spark_fused_centered_norm"],
)
def test_a_fused_norm_flag_typo_raises(key):
    from types import SimpleNamespace

    from primus.backends.megatron.core.transformer.muse_spark import norms

    cfg = SimpleNamespace(hidden_size=8, **{key: "ture"})
    cls = {
        "muse_spark_fused_attn_norm": norms.MuseSparkRMSNorm,
        "muse_spark_fused_post_expert_norm": norms.MuseSparkInputScaledRMSNorm,
        "muse_spark_fused_centered_norm": norms.MuseSparkCenteredRMSNorm,
    }[key]
    with pytest.raises(ValueError, match=key):
        cls(cfg, 8)


@pytest.mark.parametrize("flag, want_calls", [(None, 1), ("true", 1), ("false", 0), (False, 0)])
def test_the_centered_norm_flag_turns_the_kernel_off(monkeypatch, flag, want_calls):
    from types import SimpleNamespace

    from primus.backends.megatron.core.transformer.muse_spark import (
        centered_norm,
        norms,
    )

    calls = []
    real = centered_norm.fused_rms_norm

    def recorder(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(centered_norm, "fused_rms_norm", recorder)
    cfg = SimpleNamespace(hidden_size=8)
    if flag is not None:
        cfg.muse_spark_fused_centered_norm = flag
    norms.MuseSparkCenteredRMSNorm(cfg, 8)(torch.randn(3, 8))
    assert len(calls) == want_calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_kernel_skips_dw_for_a_constant_weight(monkeypatch):
    """No dw for a weight that needs no grad: the backward launches the kernel with
    HAS_DW=False and returns None for dw; dy equals the dy of a run that does compute dw."""
    h = 512
    y = (torch.randn(37, h, device="cuda") * 3).to(torch.bfloat16)
    w = torch.zeros(h, device="cuda")
    w_grad = w.clone().requires_grad_()
    ya, yb = y.clone().requires_grad_(), y.clone().requires_grad_()
    g = torch.randn_like(y)
    mod = _load()
    if mod.triton is None:
        pytest.skip("needs Triton")

    launches, returned = [], []
    real_kernel = mod._bwd_kernel

    class _Spy:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(kwargs["HAS_DW"])
                return real_kernel[grid](*args, **kwargs)

            return launch

    real_backward = mod._PostExpertNormProbs.backward

    def spy_backward(ctx, grad):
        out = real_backward(ctx, grad)
        returned.append(out[1])
        return out

    monkeypatch.setattr(mod, "_bwd_kernel", _Spy())
    monkeypatch.setattr(mod._PostExpertNormProbs, "backward", staticmethod(spy_backward))
    mod.post_expert_norm_probs(ya, w, None, 1e-6).backward(g)
    mod.post_expert_norm_probs(yb, w_grad, None, 1e-6).backward(g)
    assert launches == [False, True]
    assert returned[0] is None and returned[1] is not None
    assert torch.equal(ya.grad, yb.grad) and w_grad.grad is not None


def test_reference_takes_probs_as_t_or_t_by_1():
    # the eager path and the kernel share one contract: one prob per row, [T] or [T, 1]
    mod = _load()
    g = torch.Generator().manual_seed(0)
    y = torch.randn(5, 8, generator=g)
    w = torch.randn(8, generator=g) * 0.1
    p = torch.rand(5, generator=g)
    flat = mod.post_expert_norm_probs_reference(y, w, p, 1e-8)
    col = mod.post_expert_norm_probs_reference(y, w, p[:, None], 1e-8)
    assert flat.shape == col.shape == (5, 8)
    torch.testing.assert_close(col, flat)
    torch.testing.assert_close(mod.post_expert_norm_probs(y, w, p[:, None], 1e-8), flat)
