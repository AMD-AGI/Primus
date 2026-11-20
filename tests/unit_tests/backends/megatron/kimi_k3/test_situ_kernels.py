###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the situ activation backends in ``kimi_k3/situ_kernels``.

``eager`` (``situ_pre_mul_fused``, fp32 inside) is the reference. ``flydsl`` (gfx1250
only) must match it in the output and the input gradient, and, with the router
probability folded in, in the probability gradient too: for the plain kernels with
and without the 4-vectors-per-lane layout, odd row counts, inputs over 2 GiB (split
into row chunks) and views 2 bytes past a 16-byte boundary. Two real defects, the
halves of ``[gate | up]`` swapped and a wrong ``beta``, have to fail the same check.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

# kimi_k3/__init__.py imports megatron.core (kimi_delta_attention.py)
pytest.importorskip(
    "megatron.core.transformer.module",
    reason="megatron.core is not importable in this environment",
)

from primus.backends.megatron.core.transformer.kimi_k3.situ_activation import (  # noqa: E402
    SituActivation,
    situ_pre_mul_fused,
)
from primus.backends.megatron.core.transformer.kimi_k3.situ_kernels import (  # noqa: E402
    SITU_BACKENDS,
    resolve_situ_backend,
)

BETA, LINEAR_BETA = 4.0, 25.0  # Kimi K3's config.json


def _rel_err(a, b):
    """Max-normalised error; NaN counts as infinite."""
    a, b = a.float(), b.float()
    err = ((a - b).abs().max() / a.abs().max().clamp_min(1e-30)).item()
    return float("inf") if err != err else err


def test_backend_names():
    assert SITU_BACKENDS == ("eager", "flydsl")
    with pytest.raises(ValueError):
        resolve_situ_backend("cuda")
    with pytest.raises(ValueError):
        SituActivation(backend="cuda")
    # checked before the kernels load, so on any device and at build time rather than on the first forward
    with pytest.raises(ValueError, match="situ_backend: eager"):
        SituActivation(backend="flydsl", beta=BETA, linear_beta=None)
    assert SituActivation().backend_name == "eager"
    cfg = SimpleNamespace(
        situ_backend="eager", activation_situ_beta=BETA, activation_situ_linear_beta=LINEAR_BETA
    )
    act = SituActivation(config=cfg)
    assert (act.backend_name, act.beta, act.linear_beta, act.fuses_probs) == (
        "eager",
        BETA,
        LINEAR_BETA,
        False,
    )
    assert repr(act) == f"SituActivation(beta={BETA}, linear_beta={LINEAR_BETA}, backend=eager)"


def test_flydsl_probe_needs_the_flydsl_package(monkeypatch):
    """Without the package the backend fails when it is resolved (at build), naming the eager fallback."""
    monkeypatch.setitem(sys.modules, "flydsl", None)
    monkeypatch.setitem(sys.modules, "flydsl.compiler", None)
    with pytest.raises(ImportError, match="situ_backend: eager") as err:
        resolve_situ_backend("flydsl")
    assert "flydsl" in str(err.value.__cause__)


def _flydsl_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("no ROCm / CUDA device")
    try:
        resolve_situ_backend("flydsl")
    except ImportError as e:
        pytest.skip(f"flydsl backend unavailable: {e}")


def _x(rows, inter, seed, scale=3.0):
    """``[rows, 2 * inter]`` bf16 with values in both soft-clamp regions (|x| up to ~4 beta)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (scale * torch.randn(rows, 2 * inter, generator=g)).to("cuda", torch.bfloat16)


def _plain(fn, x, dy, **kw):
    x = x.detach().clone().requires_grad_()
    y = fn(x, beta=kw.get("beta", BETA), linear_beta=LINEAR_BETA)
    y.backward(dy)
    return y.detach(), x.grad


def _eager_with_probs(x, p, dy):
    """The routed experts' unfused path, in fp32: ``situ(x) * probs``."""
    x, p = x.detach().float().requires_grad_(), p.detach().clone().requires_grad_()
    y = situ_pre_mul_fused(x, beta=BETA, linear_beta=LINEAR_BETA) * p
    y.backward(dy.float())
    return y.detach(), x.grad, p.grad


@pytest.mark.parametrize(
    "rows, inter",
    [
        (4096, 3072),  # the routed experts' width: 384 vectors a row, the 4-vectors-per-lane kernels
        (1001, 3072),  # an odd row count
        (512, 24),  # 3 vectors a row: the one-vector kernels
        (2048, 33792),  # the dense layer's width (fc1 out 2 x 33792)
    ],
)
def test_flydsl_situ_matches_eager(rows, inter):
    _flydsl_or_skip()
    x = _x(rows, inter, seed=rows + inter)
    dy = _x(rows, inter // 2, seed=7)[:, :inter].contiguous()
    ref_y, ref_dx = _plain(situ_pre_mul_fused, x, dy)
    act = SituActivation(backend="flydsl", beta=BETA, linear_beta=LINEAR_BETA)
    got_y, got_dx = _plain(lambda t, **kw: act(t), x, dy)
    assert _rel_err(ref_y, got_y) < 1e-2
    assert _rel_err(ref_dx, got_dx) < 1e-2


@pytest.mark.parametrize("rows", [1000, 65536, 229376])  # 229376 x 6144 bf16 is over 2 GiB: row chunks
def test_flydsl_situ_with_probs_matches_eager(rows):
    _flydsl_or_skip()
    inter = 3072
    x = _x(rows, inter, seed=rows)
    g = torch.Generator(device="cpu").manual_seed(rows + 1)
    p = torch.rand(rows, 1, generator=g).cuda()
    dy = torch.randn(rows, inter, generator=g).to("cuda", torch.bfloat16)
    ref_y, ref_dx, ref_dp = _eager_with_probs(x, p, dy)
    act = SituActivation(backend="flydsl", beta=BETA, linear_beta=LINEAR_BETA)
    assert act.fuses_probs
    xg, pg = x.detach().clone().requires_grad_(), p.detach().clone().requires_grad_()
    y = act.forward_with_probs(xg, pg)
    y.backward(dy)
    assert y.dtype == torch.bfloat16
    for name, a, b in (("out", ref_y, y.detach()), ("dx", ref_dx, xg.grad), ("dprobs", ref_dp, pg.grad)):
        assert _rel_err(a, b) < 1e-2, f"{name}: {_rel_err(a, b):.2e}"


@pytest.mark.parametrize(
    "inter, dtype",
    [
        (3076, torch.bfloat16),  # 2 x 3076 is not a multiple of 16: neither FlyDSL kernel takes it
        (3072, torch.float32),  # the FlyDSL kernels take bf16 only
    ],
)
def test_flydsl_forward_with_probs_falls_back_to_eager(inter, dtype):
    """Inputs the FlyDSL kernels do not take go through the eager reference, gradients included."""
    _flydsl_or_skip()
    rows = 1000
    x = _x(rows, inter, seed=inter).to(dtype)
    g = torch.Generator(device="cpu").manual_seed(5)
    p = torch.rand(rows, 1, generator=g).cuda()
    dy = torch.randn(rows, inter, generator=g).to("cuda", dtype)
    ref_y, ref_dx, ref_dp = _eager_with_probs(x, p, dy)
    act = SituActivation(backend="flydsl", beta=BETA, linear_beta=LINEAR_BETA)
    xg, pg = x.detach().clone().requires_grad_(), p.detach().clone().requires_grad_()
    y = act.forward_with_probs(xg, pg)
    y.backward(dy)
    assert y.dtype == dtype
    for name, a, b in (("out", ref_y, y.detach()), ("dx", ref_dx, xg.grad), ("dprobs", ref_dp, pg.grad)):
        assert _rel_err(a, b) < 1e-2, f"{name}: {_rel_err(a, b):.2e}"


def test_flydsl_situ_misaligned_views_match_eager():
    """``x`` and ``dy`` as views 2 bytes past a 16-byte boundary (the kernels move 16 B per access)."""
    _flydsl_or_skip()
    x, dy = _x(1024, 3072, seed=3), _x(1024, 1536, seed=4)
    ref_y, ref_dx = _plain(situ_pre_mul_fused, x, dy)

    def shifted(t, grad=False):
        buf = torch.cat((t.new_zeros(1), t.reshape(-1))).requires_grad_(grad)
        return buf, buf[1:].view(t.shape)

    (x_buf, x_v), (_, dy_v) = shifted(x, True), shifted(dy)
    assert x_v.data_ptr() % 16 and dy_v.data_ptr() % 16
    act = SituActivation(backend="flydsl", beta=BETA, linear_beta=LINEAR_BETA)
    y = act(x_v)
    y.backward(dy_v)
    assert _rel_err(ref_y, y.detach()) < 1e-2
    assert _rel_err(ref_dx, x_buf.grad[1:].view(x.shape)) < 1e-2


def test_flydsl_parity_catches_real_defects():
    """The negative control: the halves swapped, or beta 2 instead of 4, fail the check the kernel passes."""
    _flydsl_or_skip()
    from primus.backends.megatron.core.transformer.kimi_k3.situ_kernels._flydsl_gfx1250 import (
        flydsl_situ_pre_mul_fused,
    )

    x, dy = _x(2048, 3072, seed=11), _x(2048, 1536, seed=12)
    ref_y, ref_dx = _plain(situ_pre_mul_fused, x, dy)
    good_y, good_dx = _plain(flydsl_situ_pre_mul_fused, x, dy)
    assert max(_rel_err(ref_y, good_y), _rel_err(ref_dx, good_dx)) < 1e-2
    swapped = torch.cat(x.chunk(2, dim=-1)[::-1], dim=-1)
    assert _rel_err(ref_y, _plain(flydsl_situ_pre_mul_fused, swapped, dy)[0]) >= 1e-2
    assert _rel_err(ref_y, _plain(flydsl_situ_pre_mul_fused, x, dy, beta=BETA / 2)[0]) >= 1e-2


def test_flydsl_rejects_unsupported_inputs():
    _flydsl_or_skip()
    from primus.backends.megatron.core.transformer.kimi_k3.situ_kernels._flydsl_gfx1250 import (
        flydsl_situ_pre_mul_fused,
        flydsl_situ_with_probs,
    )

    x = _x(64, 3072, seed=1)
    with pytest.raises(ValueError, match="situ_backend: eager"):
        flydsl_situ_pre_mul_fused(x.float(), beta=BETA, linear_beta=LINEAR_BETA)
    with pytest.raises(ValueError, match="situ_backend: eager"):
        flydsl_situ_pre_mul_fused(x, beta=BETA, linear_beta=None)
    with pytest.raises(ValueError, match="situ_backend: eager"):
        flydsl_situ_with_probs(x, torch.rand(32, 1, device="cuda"), beta=BETA, linear_beta=LINEAR_BETA)
    with pytest.raises(ValueError, match="probs is on cpu"):
        flydsl_situ_with_probs(x, torch.rand(64, 1), beta=BETA, linear_beta=LINEAR_BETA)
    with pytest.raises(ValueError, match="not one value for each"):
        flydsl_situ_with_probs(x, torch.rand(32, 2, device="cuda"), beta=BETA, linear_beta=LINEAR_BETA)
