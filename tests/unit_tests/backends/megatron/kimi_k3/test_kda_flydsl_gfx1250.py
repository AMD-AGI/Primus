###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the gfx1250 ``flydsl`` KDA backend in ``kda_kernels/_flydsl_gfx1250``.

The eager ops are the reference: the chunk kernel against ``eager_chunk_kda`` (and, where fla
is installed, against fla's ``chunk_kda`` with ``safe_gate`` / ``disable_recompute``, the same
algorithm), the L2 norm, gate, gated RMSNorm, chunk-local cumsum and short conv against their
eager forms, in the output and every input gradient. Real defects (a negated ``beta``, a halved
gate bound, a dropped SiLU) have to fail the same checks. Inputs the kernels do not take raise
with the ``kda_backend: fla`` hint, and the backend runs without importing fla. Off gfx1250 only
the option resolution is checked.
"""

from __future__ import annotations

import math
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

# kimi_k3/__init__.py imports megatron.core (kimi_delta_attention.py)
pytest.importorskip(
    "megatron.core.transformer.module",
    reason="megatron.core is not importable in this environment",
)

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (  # noqa: E402
    FLYDSL_KDA_ARCHS,
    eager_chunk_kda,
    kda_gate,
    kda_l2norm,
    load_flydsl_kda_backend,
    load_flydsl_kda_ops,
)
from primus.backends.megatron.core.transformer.kimi_k3.kimi_delta_attention import (  # noqa: E402
    KimiGatedRMSNorm,
    _resolve_kda_ops,
)

LOWER_BOUND = -5.0
D = 128


def _rel_err(a, b):
    """Norm-relative error; NaN counts as infinite."""
    a, b = a.double(), b.double()
    err = ((a - b).norm() / b.norm().clamp_min(1e-30)).item()
    return math.inf if math.isnan(err) else err


def _gfx1250_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("no ROCm / CUDA device")
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    if not arch.startswith("gfx1250"):
        pytest.skip(f"device arch is {arch!r}, not gfx1250")
    try:
        return load_flydsl_kda_backend(), load_flydsl_kda_ops()
    except ImportError as e:
        pytest.skip(f"flydsl backend unavailable: {e}")


def _rand(*shape, seed, scale=1.0, dtype=torch.bfloat16):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (scale * torch.randn(*shape, generator=g)).to("cuda", dtype)


def _grads(fn, inputs, dy):
    leaves = [t.detach().clone().requires_grad_() for t in inputs]
    y = fn(*leaves)
    y.backward(dy)
    return [y.detach()] + [t.grad for t in leaves]


def _max_err(ref, got):
    return max(_rel_err(g, r) for r, g in zip(ref, got))


def _kda_inputs(B, T, H, seed):
    a_log = torch.log(torch.rand(H, generator=torch.Generator().manual_seed(seed)) * 15 + 1).cuda()
    return [
        kda_l2norm(_rand(B, T, H, D, seed=seed + 1)),
        kda_l2norm(_rand(B, T, H, D, seed=seed + 2)),
        _rand(B, T, H, D, seed=seed + 3),
        kda_gate(_rand(B, T, H, D, seed=seed + 4, scale=2.0), a_log, None, LOWER_BOUND),
        torch.sigmoid(_rand(B, T, H, seed=seed + 5, dtype=torch.float32)),
    ]


def test_archs():
    assert FLYDSL_KDA_ARCHS == ("gfx950", "gfx1250")


def test_other_backends_keep_their_ops():
    cfg = SimpleNamespace(kda_backend="fla", kda_fla_fused_ops=False, kda_fla_safe_gate=True)
    assert _resolve_kda_ops(cfg, "fla", LOWER_BOUND) == (kda_l2norm, kda_gate, None, {"safe_gate": True})


def test_resolution_on_gfx1250():
    chunk, ops = _gfx1250_or_skip()
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
        _flydsl_gfx1250,
    )

    assert chunk is _flydsl_gfx1250.flydsl_chunk_kda
    l2norm, gate, norm, conv = ops
    cfg = SimpleNamespace(
        kda_backend="flydsl", kda_fla_fused_ops=True, kda_fla_safe_gate=True, kda_fla_disable_recompute=True
    )
    assert _resolve_kda_ops(cfg, "flydsl", LOWER_BOUND) == (l2norm, gate, conv, {})
    for bound in (None, -6.0):
        with pytest.raises(ValueError, match="safe_gate algorithm"):
            _resolve_kda_ops(cfg, "flydsl", bound)
    assert (
        KimiGatedRMSNorm(
            D, config=SimpleNamespace(kda_backend="flydsl", params_dtype=torch.float32)
        )._fused_forward
        is norm
    )


@pytest.mark.parametrize("B, T, H", [(1, 512, 4), (2, 1024, 3)])
def test_chunk_matches_eager(B, T, H):
    chunk, _ = _gfx1250_or_skip()
    inputs, do = _kda_inputs(B, T, H, seed=T + H), _rand(B, T, H, D, seed=99)
    ref = _grads(lambda *t: eager_chunk_kda(*[x.float() for x in t])[0], inputs, do.float())
    got = _grads(lambda *t: chunk(*t)[0], inputs, do)
    assert got[0].dtype == torch.bfloat16 and all(torch.isfinite(t).all() for t in got)
    assert _max_err(ref, got) < 2e-2, [_rel_err(g, r) for r, g in zip(ref, got)]
    negated = _grads(lambda q, k, v, g, b: chunk(q, k, v, g, -b)[0], inputs, do)
    assert _rel_err(negated[0], ref[0]) >= 2e-2


def test_chunk_matches_fla_safe_gate():
    chunk, _ = _gfx1250_or_skip()
    fla_mod = pytest.importorskip("primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._fla")
    inputs, do = _kda_inputs(2, 2048, 8, seed=5), _rand(2, 2048, 8, D, seed=6)
    ref = _grads(lambda *t: fla_mod.fla_chunk_kda(*t, safe_gate=True, disable_recompute=True)[0], inputs, do)
    got = _grads(lambda *t: chunk(*t)[0], inputs, do)
    assert _max_err(ref, got) < 1e-3, [_rel_err(g, r) for r, g in zip(ref, got)]


def test_chunk_rejects_unsupported_inputs():
    chunk, _ = _gfx1250_or_skip()
    q, k, v, g, beta = _kda_inputs(1, 128, 2, seed=1)
    bad = [
        dict(initial_state=torch.zeros(1, 2, D, D, device="cuda")),
        dict(output_final_state=True),
        dict(use_qk_l2norm_in_kernel=True),
        dict(chunk_size=32),
        dict(cu_seqlens=torch.tensor([0, 128], device="cuda")),
    ]
    for kw in bad:
        with pytest.raises(ValueError, match="kda_backend: fla"):
            chunk(q, k, v, g, beta, **kw)
    with pytest.raises(ValueError, match="kda_backend: fla"):
        chunk(q[:, :100], k[:, :100], v[:, :100], g[:, :100], beta[:, :100])
    with pytest.raises(ValueError, match="kda_backend: fla"):
        chunk(q.half(), k, v, g, beta)
    with pytest.raises(ValueError, match="kda_backend: fla"):
        chunk(q[..., :64], k[..., :64], v[..., :64], g[..., :64], beta)


def test_l2norm_matches_eager():
    _, (l2norm, _, _, _) = _gfx1250_or_skip()
    x, dy = _rand(2, 256, 6, D, seed=1, scale=3.0), _rand(2, 256, 6, D, seed=2)
    assert _max_err(_grads(kda_l2norm, [x], dy), _grads(l2norm, [x], dy)) < 1e-3


def test_gate_matches_eager_and_catches_a_wrong_bound():
    _, (_, gate, _, _) = _gfx1250_or_skip()
    z, dy = _rand(2, 256, 6, D, seed=3, scale=2.0), _rand(2, 256, 6, D, seed=4, dtype=torch.float32)
    a_log = torch.log(torch.rand(6, generator=torch.Generator().manual_seed(5)) * 15 + 1).cuda()
    dt = _rand(6 * D, seed=6, scale=0.1, dtype=torch.float32)
    ref = _grads(lambda z, a, b: kda_gate(z, a, b, LOWER_BOUND), [z, a_log, dt], dy)
    got = _grads(lambda z, a, b: gate(z, a, b, LOWER_BOUND), [z, a_log, dt], dy)
    assert got[0].dtype == torch.float32
    assert _max_err(ref, got) < 1e-3, [_rel_err(g, r) for r, g in zip(ref, got)]
    wrong = _grads(lambda z, a, b: gate(z, a, b, LOWER_BOUND / 2), [z, a_log, dt], dy)
    assert _rel_err(wrong[0], ref[0]) >= 1e-2
    with pytest.raises(ValueError, match="bounded gate"):
        gate(z, a_log, dt, None)


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_gated_rmsnorm_matches_eager(weight_dtype):
    _gfx1250_or_skip()
    eager = KimiGatedRMSNorm(D, eps=1e-5, params_dtype=weight_dtype, device="cuda")
    fly = KimiGatedRMSNorm(
        D, eps=1e-5, config=SimpleNamespace(kda_backend="flydsl", params_dtype=weight_dtype), device="cuda"
    )
    with torch.no_grad():
        w = 1 + _rand(D, seed=7, scale=0.1, dtype=torch.float32)
        eager.weight.copy_(w)
        fly.weight.copy_(w)
    x, gate, dy = (
        _rand(2, 256, 6, D, seed=8),
        _rand(2, 256, 6, D, seed=9, scale=2.0),
        _rand(2, 256, 6, D, seed=10),
    )

    def run(norm, flip=1):
        norm.weight.grad = None
        y, dx, dgate = _grads(lambda x, g: norm(x, flip * g), [x, gate], dy)
        return [y, dx, dgate, norm.weight.grad]

    ref, got = run(eager), run(fly)
    assert got[0].dtype == torch.bfloat16 and got[3].dtype == weight_dtype
    assert _max_err(ref, got) < 1e-2, [_rel_err(g, r) for r, g in zip(ref, got)]
    assert _rel_err(run(fly, flip=-1)[0], ref[0]) >= 1e-2


@pytest.mark.parametrize("reverse", [False, True])
def test_chunk_local_cumsum(reverse):
    _gfx1250_or_skip()
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_gfx1250.cumsum_kernel import (
        chunk_local_cumsum,
    )

    x = -torch.rand(2, 512, 3, D, device="cuda") * 5
    c = x.view(2, 8, 64, 3, D)
    ref = (c.flip(2).cumsum(2).flip(2) if reverse else c.cumsum(2)).view_as(x) * (
        1.0 if reverse else 1 / math.log(2)
    )
    got = chunk_local_cumsum(x, scale=1.0 if reverse else 1 / math.log(2), reverse=reverse)
    assert _rel_err(got, ref) < 1e-6


def test_conv_matches_torch_and_catches_a_dropped_silu():
    _, (_, _, _, conv) = _gfx1250_or_skip()
    B, T, C, W = 2, 512, 384, 4
    x, dy = _rand(B, T, C, seed=11), _rand(B, T, C, seed=12)
    w = _rand(C, W, seed=13, scale=0.5, dtype=torch.float32)

    def torch_conv(x, w, act=True):
        y = F.conv1d(F.pad(x.float().transpose(1, 2), (W - 1, 0)), w.unsqueeze(1), groups=C).transpose(1, 2)
        return (F.silu(y) if act else y).to(x.dtype)

    ref = _grads(torch_conv, [x, w], dy)
    got = _grads(conv, [x, w], dy)
    assert _max_err(ref, got) < 1e-2, [_rel_err(g, r) for r, g in zip(ref, got)]
    assert _rel_err(got[0], _grads(lambda x, w: torch_conv(x, w, act=False), [x, w], dy)[0]) >= 1e-2


def test_backend_runs_without_fla():
    _gfx1250_or_skip()
    code = """
import sys, torch
from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import load_flydsl_kda_backend, load_flydsl_kda_ops
chunk = load_flydsl_kda_backend()
l2norm, gate, norm, conv = load_flydsl_kda_ops()
B, T, H, D = 1, 128, 2, 128
x = lambda *s: torch.randn(*s, device="cuda").bfloat16().requires_grad_()
q, k, v, z, gt = (x(B, T, H, D) for _ in range(5))
g = gate(z, torch.zeros(H, device="cuda"), None, -5.0)
o, _ = chunk(l2norm(q), l2norm(k), v, g, torch.rand(B, T, H, device="cuda"))
norm(o, gt, torch.ones(D, device="cuda"), 1e-5).sum().backward()
c = conv(x(B, T, 256), torch.randn(256, 4, device="cuda"))
c.sum().backward()
torch.cuda.synchronize()
assert not [m for m in sys.modules if m == "fla" or m.startswith("fla.")], "fla was imported"
print("ok")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-2000:]
