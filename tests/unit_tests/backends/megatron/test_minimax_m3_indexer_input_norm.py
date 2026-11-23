###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_fused_indexer_input_norm``: the MSA indexer's input norm as one Triton launch must give
the bits of ``rms_norm(x.float(), w.float() [+ 1], eps).to(bf16)`` (ATen's vectorised kernel) at
MiniMax-M3.1-Flash's shape, at MBS 2, longer sequences and other widths, and keep the eager ops
wherever it does not apply or its self-check fails."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("triton")
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture
def mod(monkeypatch):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.transformer.minimax_m3 import indexer_input_norm

    monkeypatch.setattr(bitwise_check, "_results", {})
    return indexer_input_norm


def _attention_cls():
    from primus.backends.megatron.core.transformer.minimax_m3 import attention

    return next(c for c in vars(attention).values() if isinstance(c, type) and "_indexer_input" in vars(c))


def _fake(eps, zero_centered, switch=False, weight=None):
    """What ``_indexer_input`` / ``_indexer_input_eager`` read from ``self``."""
    cls = _attention_cls()
    fake = SimpleNamespace(
        config=SimpleNamespace(
            msa_fused_indexer_input_norm=switch,
            layernorm_zero_centered_gamma=zero_centered,
            layernorm_epsilon=eps,
        ),
        _fused_input_norm_weight=lambda: weight,
    )
    fake._indexer_input_eager = lambda x, w: cls._indexer_input_eager(fake, x, w)
    return fake


def _original(eps, zero_centered):
    """The original ops: the real ``_indexer_input_eager`` method."""
    return _fake(eps, zero_centered)._indexer_input_eager


def _rows(shape, kind, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(shape, generator=gen, device="cuda")
    if kind == "big":
        x *= 300.0
    elif kind == "tiny":
        x *= 1e-3
    elif kind == "residual":  # a few outlier channels, like a residual stream
        x = x * torch.rand(shape[-1], generator=gen, device="cuda") * 20.0
    elif kind == "zeros":
        x = x * (torch.rand(shape, generator=gen, device="cuda") > 0.5)
        x[0] = 0.0
    return x.to(torch.bfloat16)


def _weight(n, zero_centered, seed=1):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, generator=gen, device="cuda") * (0.05 if zero_centered else 1.0)
    return (w + (0.0 if zero_centered else 1.0)).to(torch.bfloat16)


def test_switch_defaults_to_off():
    from primus.backends.megatron.core.models.minimax_m3.minimax_m3_transformer_config import (
        MSATransformerConfig,
    )

    assert MSATransformerConfig.__dataclass_fields__["msa_fused_indexer_input_norm"].default is False


@needs_gpu
@pytest.mark.parametrize(
    "shape",
    [(4096, 1, 6144), (4096, 2, 6144), (8192, 1, 6144), (16384, 1, 6144), (512, 3, 7168), (512, 1, 8192), (512, 1, 1028), (64, 1, 520)],
)
@pytest.mark.parametrize("zero_centered", [True, False])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_kernel_matches_aten_bitwise(mod, require_bitwise, shape, zero_centered, eps):
    """The kernel itself (``_launch``), not the dispatcher, against the eager ops."""
    w = _weight(shape[-1], zero_centered)
    assert mod.supported(_rows(shape, "randn"), w)
    original = _original(eps, zero_centered)
    require_bitwise(
        mod.self_check(_rows(shape, "randn"), w, eps, zero_centered, original), "msa_indexer_input_norm"
    )
    for kind in ("randn", "big", "tiny", "residual", "zeros"):
        x = _rows(shape, kind)
        want = original(x, w)
        got = mod._launch(x, w, eps, zero_centered)
        assert got.dtype == want.dtype and got.shape == want.shape
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), kind


@needs_gpu
def test_kernel_is_repeatable(mod, require_bitwise):
    x, w = _rows((4096, 1, 6144), "residual"), _weight(6144, True)
    original = _original(1e-6, True)
    require_bitwise(mod.self_check(x, w, 1e-6, True, original), "msa_indexer_input_norm")
    want = original(x, w)
    assert all(torch.equal(mod._launch(x, w, 1e-6, True), want) for _ in range(200))


@needs_gpu
def test_each_width_eps_and_gamma_kind_is_checked(mod, require_bitwise):
    for n, eps, zc in [(6144, 1e-6, True), (6144, 1e-5, True), (6144, 1e-6, False), (2048, 1e-6, True)]:
        mod.indexer_input_norm(_rows((64, 1, n), "randn"), _weight(n, zc), eps, zc, _original(eps, zc))
    from primus.backends.megatron.core.fusions import bitwise_check

    assert set(bitwise_check._results) == {
        ("msa_indexer_input_norm", (6144, 1e-6, True)),
        ("msa_indexer_input_norm", (6144, 1e-5, True)),
        ("msa_indexer_input_norm", (6144, 1e-6, False)),
        ("msa_indexer_input_norm", (2048, 1e-6, True)),
    }


@needs_gpu
def test_unsupported_inputs_keep_the_eager_ops(mod, monkeypatch):
    """Eligibility is decided before any work: no kernel build, no self-check; the patched method
    then returns what the original ops return."""

    def boom(*a, **k):
        raise AssertionError("the kernel must not run")

    monkeypatch.setattr(mod, "_launch", boom)
    monkeypatch.setattr(mod, "self_check", boom)
    w = _weight(6144, True)
    cases = [
        (_rows((8, 1, 6144), "randn").float(), w.float()),  # FP32 residual stream
        (_rows((8, 1, 6146), "randn"), _weight(6146, True)),  # width not a multiple of 4
        (_rows((8, 1, 512), "randn"), _weight(512, True)),  # ATen uses a smaller block
        (_rows((8, 1, 8196), "randn"), _weight(8196, True)),  # wider than the kernel is built for
        (_rows((8, 1, 12288), "randn")[..., ::2], w),  # not contiguous
        (_rows((0, 1, 6144), "randn"), w),  # empty
    ]
    cls = _attention_cls()
    for x, wt in cases:
        assert not mod.supported(x, wt)
        assert mod.indexer_input_norm(x, wt, 1e-6, True, _original(1e-6, True)) is None
        got = cls._indexer_input(_fake(1e-6, True, switch=True, weight=wt), x)
        assert torch.equal(got, _original(1e-6, True)(x, wt))


def test_cpu_input_keeps_the_eager_ops(mod):
    x, w = torch.randn(4, 1, 2048).to(torch.bfloat16), torch.randn(2048).to(torch.bfloat16)
    assert not mod.supported(x, w)
    got = _attention_cls()._indexer_input(_fake(1e-6, True, switch=True, weight=w), x)
    assert torch.equal(got, _original(1e-6, True)(x, w))


def test_width_bound_is_the_largest_tested_width(mod):
    assert mod._MAX_WIDTH == 8192


@needs_gpu
def test_failed_self_check_keeps_the_eager_ops(mod, monkeypatch):
    """A kernel that is one BF16 ulp off in one element loses its check; eager from then on."""
    from primus.backends.megatron.core.fusions import bitwise_check

    real = mod._launch

    def off_by_one(x, w, eps, zc):
        y = real(x, w, eps, zc)
        y.view(-1).view(torch.int16)[3] ^= 1
        return y

    monkeypatch.setattr(mod, "_launch", off_by_one)
    x, w = _rows((64, 1, 6144), "randn"), _weight(6144, True)
    assert mod.indexer_input_norm(x, w, 1e-6, True, _original(1e-6, True)) is None
    assert bitwise_check._results == {("msa_indexer_input_norm", (6144, 1e-6, True)): False}
    got = _attention_cls()._indexer_input(_fake(1e-6, True, switch=True, weight=w), x)
    assert torch.equal(got, _original(1e-6, True)(x, w))


@needs_gpu
def test_decided_check_builds_no_probe(mod, monkeypatch, require_bitwise):
    x, w = _rows((64, 1, 6144), "randn"), _weight(6144, True)
    original = _original(1e-6, True)
    require_bitwise(mod.self_check(x, w, 1e-6, True, original), "msa_indexer_input_norm")

    def boom(*a, **k):
        raise AssertionError("probe built after the check was decided")

    monkeypatch.setattr(mod, "_probe", boom)
    assert torch.equal(mod.indexer_input_norm(x, w, 1e-6, True, original), original(x, w))


@needs_gpu
def test_probe_moves_the_weight_off_its_init_value_and_has_edge_rows(mod):
    x, w = mod._probe((1, 1, 6144), torch.zeros(6144, device="cuda", dtype=torch.bfloat16))
    assert w.dtype == torch.bfloat16 and w.count_nonzero() > 6000
    assert x.dtype == torch.bfloat16 and x.shape == (2096, 6144)
    # the per-row scalars (sum / N, + eps, rsqrt) are checked on >= 2048 ordinary rows
    assert x[:2048].isfinite().all() and (x[:2048].float().abs().sum(dim=1) > 0).all()
    assert (x[2080] == 0).all() and x[2081].isinf().any()


@needs_gpu
@pytest.mark.parametrize("zero_centered", [True, False])
@pytest.mark.parametrize("switch", [True, False])
def test_patched_indexer_input_matches_the_original(mod, require_bitwise, switch, zero_centered):
    """``MiniMaxM3Attention._indexer_input`` with the switch on against the same method with it off."""
    cls = _attention_cls()
    weight = torch.nn.Parameter(_weight(6144, zero_centered))
    hidden = _rows((4096, 2, 6144), "residual").requires_grad_()
    want = cls._indexer_input(_fake(1e-6, zero_centered, False, weight), hidden)
    got = cls._indexer_input(_fake(1e-6, zero_centered, switch, weight), hidden)
    if switch:
        require_bitwise(
            mod.self_check(hidden.detach(), weight.detach(), 1e-6, zero_centered, _original(1e-6, zero_centered)),
            "msa_indexer_input_norm",
        )
    assert not got.requires_grad and got.dtype == want.dtype
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
