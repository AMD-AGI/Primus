###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for ``kda_fla_fused_ops``, ``kda_fla_safe_gate`` and ``kda_fla_disable_recompute``.

The eager ops in ``kda_kernels/_eager`` (``kda_l2norm``, ``kda_gate``) and the fp32
``KimiGatedRMSNorm`` are the reference: fla's fused kernels must match them in the output
and every input gradient. fla's ``chunk_kda`` with ``safe_gate`` / ``disable_recompute``
must match its defaults on inputs shaped like Kimi K3's (bounded gate, L2-normalised q / k).
A wrong gate bound and a negated output gate have to fail the same checks. Without a GPU,
the option resolution (defaults, errors, the fields being ignored by other backends) is still checked.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

# kimi_k3/__init__.py imports megatron.core (kimi_delta_attention.py)
pytest.importorskip(
    "megatron.core.transformer.module",
    reason="megatron.core is not importable in this environment",
)

try:
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
        KimiK3TransformerConfig,
    )
except RuntimeError as error:
    # models.kimi_k3 imports megatron.training, whose import chain needs a GPU (as in test_model_registry.py)
    if torch.cuda.is_available():
        raise
    pytest.skip(f"importing models.kimi_k3 needs a GPU: {error}", allow_module_level=True)
from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (  # noqa: E402
    kda_gate,
    kda_l2norm,
)
from primus.backends.megatron.core.transformer.kimi_k3.kimi_delta_attention import (  # noqa: E402
    KimiGatedRMSNorm,
    _resolve_kda_ops,
)

LOWER_BOUND = -5.0  # Kimi K3's kda_gate_lower_bound
B, T, H, D = 2, 1024, 8, 128


def _rel_err(a, b):
    """Max-normalised error; NaN counts as infinite."""
    a, b = a.float(), b.float()
    err = ((a - b).abs().max() / a.abs().max().clamp_min(1e-30)).item()
    return float("inf") if err != err else err


def _opts(**kw):
    return SimpleNamespace(
        **{"kda_fla_fused_ops": False, "kda_fla_safe_gate": False, "kda_fla_disable_recompute": False, **kw}
    )


def test_config_fields_default_off():
    defaults = {f.name: f.default for f in dataclasses.fields(KimiK3TransformerConfig)}
    for name in ("kda_fla_fused_ops", "kda_fla_safe_gate", "kda_fla_disable_recompute"):
        assert defaults[name] is False


def test_resolution_without_options_keeps_the_eager_ops():
    assert _resolve_kda_ops(_opts(), "eager", LOWER_BOUND) == (kda_l2norm, kda_gate, None, {})
    assert _resolve_kda_ops(SimpleNamespace(), "fla", LOWER_BOUND) == (kda_l2norm, kda_gate, None, {})


def test_resolution_chunk_options():
    both = _opts(kda_fla_safe_gate=True, kda_fla_disable_recompute=True)
    assert _resolve_kda_ops(both, "fla", LOWER_BOUND)[3] == {"safe_gate": True, "disable_recompute": True}
    for backend in ("eager", "eager_recurrent"):  # fla-only arguments: the other backends ignore them
        assert _resolve_kda_ops(both, backend, None)[3] == {}
    for bound in (None, -6.0, 0.0):
        with pytest.raises(ValueError, match="kda_fla_safe_gate needs"):
            _resolve_kda_ops(_opts(kda_fla_safe_gate=True), "fla", bound)
    assert _resolve_kda_ops(_opts(kda_fla_disable_recompute=True), "fla", None)[3] == {
        "disable_recompute": True
    }


def _fla_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("no ROCm / CUDA device")
    try:
        from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
            load_fla_fused_ops,
        )

        return load_fla_fused_ops()
    except ImportError as e:
        pytest.skip(f"fla unavailable: {e}")


def _rand(*shape, seed, scale=1.0, dtype=torch.bfloat16):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (scale * torch.randn(*shape, generator=g)).to("cuda", dtype)


def _grads(fn, inputs, dy):
    leaves = [t.detach().clone().requires_grad_() for t in inputs]
    y = fn(*leaves)
    y.backward(dy)
    return y.detach(), [t.grad for t in leaves]


def _check(ref, got, tol=1e-2):
    (ry, rg), (gy, gg) = ref, got
    errs = [_rel_err(ry, gy)] + [_rel_err(a, b) for a, b in zip(rg, gg)]
    assert max(errs) < tol, errs
    return errs


def test_fused_resolution_returns_the_fla_kernels():
    l2norm, gate, _ = _fla_or_skip()
    assert _resolve_kda_ops(_opts(kda_fla_fused_ops=True), "eager", LOWER_BOUND) == (l2norm, gate, None, {})


def test_fused_l2norm_matches_eager():
    l2norm, _, _ = _fla_or_skip()
    x, dy = _rand(B, T, H, D, seed=1, scale=3.0), _rand(B, T, H, D, seed=2)
    ref = _grads(kda_l2norm, [x], dy)
    got = _grads(l2norm, [x], dy)
    assert got[0].dtype == torch.bfloat16
    _check(ref, got)


def test_fused_gate_matches_eager_and_catches_a_wrong_bound():
    _, gate, _ = _fla_or_skip()
    z, dy = _rand(B, T, H, D, seed=3, scale=2.0), _rand(B, T, H, D, seed=4, dtype=torch.float32)
    a_log = torch.log(torch.rand(H, generator=torch.Generator().manual_seed(5)) * 15 + 1).cuda()
    dt_bias = _rand(H * D, seed=6, scale=0.1, dtype=torch.float32)
    ref = _grads(lambda z, a, b: kda_gate(z, a, b, LOWER_BOUND), [z, a_log, dt_bias], dy)
    got = _grads(lambda z, a, b: gate(z, a, b, LOWER_BOUND), [z, a_log, dt_bias], dy)
    assert got[0].dtype == torch.float32
    assert got[0].min() >= LOWER_BOUND and got[0].max() <= 0  # the fp32 sigmoid underflows to 0
    _check(ref, got)
    wrong = _grads(lambda z, a, b: gate(z, a, b, LOWER_BOUND / 2), [z, a_log, dt_bias], dy)
    assert _rel_err(ref[0], wrong[0]) >= 1e-2


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_fused_gated_rmsnorm_matches_eager_and_catches_a_negated_gate(weight_dtype):
    _fla_or_skip()
    eager = KimiGatedRMSNorm(D, eps=1e-5, params_dtype=weight_dtype, device="cuda")
    cfg = SimpleNamespace(kda_fla_fused_ops=True, params_dtype=weight_dtype)
    fused = KimiGatedRMSNorm(D, eps=1e-5, config=cfg, device="cuda")
    assert eager._fused_forward is None and fused._fused_forward is not None
    with torch.no_grad():
        w = 1 + _rand(D, seed=7, scale=0.1, dtype=torch.float32)
        eager.weight.copy_(w)
        fused.weight.copy_(w)
    x, gate, dy = _rand(B, T, H, D, seed=8), _rand(B, T, H, D, seed=9, scale=2.0), _rand(B, T, H, D, seed=10)

    def run(norm, flip=1):
        norm.weight.grad = None
        y, (dx, dgate) = _grads(lambda x, g: norm(x, flip * g), [x, gate], dy)
        return y, [dx, dgate, norm.weight.grad]

    ref, got = run(eager), run(fused)
    assert got[0].dtype == torch.bfloat16
    _check(ref, got)
    assert _rel_err(ref[0], run(fused, flip=-1)[0]) >= 1e-2


def _kda_inputs(seed):
    q = kda_l2norm(_rand(B, T, H, D, seed=seed))
    k = kda_l2norm(_rand(B, T, H, D, seed=seed + 1))
    v = _rand(B, T, H, D, seed=seed + 2)
    a_log = torch.log(torch.rand(H, generator=torch.Generator().manual_seed(seed + 3)) * 15 + 1).cuda()
    g = kda_gate(_rand(B, T, H, D, seed=seed + 4, scale=2.0), a_log, None, LOWER_BOUND)
    beta = torch.sigmoid(_rand(B, T, H, seed=seed + 5, dtype=torch.float32))
    return [q, k, v, g, beta]


@pytest.mark.parametrize(
    "options",
    [{"safe_gate": True}, {"disable_recompute": True}, {"safe_gate": True, "disable_recompute": True}],
)
def test_chunk_options_match_the_defaults(options):
    _fla_or_skip()
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._fla import (
        fla_chunk_kda,
    )

    inputs, dy = _kda_inputs(seed=20), _rand(B, T, H, D, seed=30)

    def run(**kw):
        return _grads(lambda *t: fla_chunk_kda(*t, chunk_size=64, **kw)[0], inputs, dy)

    _check(run(), run(**options))
