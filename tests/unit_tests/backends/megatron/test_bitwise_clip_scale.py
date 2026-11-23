###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The Triton gradient-clip scale against TE's ``multi_tensor_scale_tensor``, bit for bit."""

import pytest
import torch

pytest.importorskip("triton")
te_optimizers = pytest.importorskip("transformer_engine.pytorch.optimizers")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton and TE kernels need a GPU")


def _te_scale(tensors, coeff):
    """Megatron's call: TE's applier around TE's kernel, in place."""
    noop = torch.zeros(1, dtype=torch.int, device="cuda")
    te_optimizers.multi_tensor_applier(
        te_optimizers.multi_tensor_scale_tensor, noop, [tensors, tensors], coeff
    )


def _coeff(value):
    return torch.tensor([value], dtype=torch.float32, device="cuda")


def _same_bits(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.fixture
def clean_checks():
    from primus.backends.megatron.core.fusions import bitwise_check

    saved = dict(bitwise_check._results)
    bitwise_check._results.clear()
    yield bitwise_check
    bitwise_check._results.clear()
    bitwise_check._results.update(saved)


def test_self_check_passes(require_bitwise, clean_checks):
    from primus.backends.megatron.core.fusions import clip_scale

    require_bitwise(clip_scale.checked(_te_scale, _coeff(0.2718)), "bitwise_clip_scale")


@pytest.mark.parametrize("coeff", [1.0, 0.159744, 0.5, 1.0 / 3.0, 0.99999994, 1.0e-3, 3.0e-39, 0.0])
def test_kernel_matches_te_on_every_bf16_value(require_bitwise, coeff):
    # the kernel itself, not the patched dispatcher (which runs TE after a failed check)
    from primus.backends.megatron.core.fusions import clip_scale

    bits = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16)
    lengths = [65536, 65535, 4097, 4096, 4095, 1000, 3, 1]
    got = [bits[:n].clone() for n in lengths]
    want = [bits[:n].clone() for n in lengths]
    clip_scale.scale_tensors(got, _coeff(coeff))
    _te_scale(want, _coeff(coeff))
    # NaN payloads included
    require_bitwise(all(_same_bits(a, b) for a, b in zip(got, want)), f"bitwise_clip_scale coeff={coeff}")


@pytest.mark.parametrize("dist", ["normal", "tiny", "huge"])
def test_kernel_matches_te_on_a_gradient_list(require_bitwise, dist):
    from primus.backends.megatron.core.fusions import clip_scale

    gen = torch.Generator(device="cuda").manual_seed(7)
    scale = {"normal": 1.0, "tiny": 1.0e-38, "huge": 1.0e38}[dist]
    shapes = [(6144, 3072), (5, 7, 11), (1,), (128, 1000, 33), (65536 * 3 + 5,)]
    grads = [(torch.randn(s, device="cuda", generator=gen) * scale).bfloat16() for s in shapes]
    got = [g.clone() for g in grads]
    want = [g.clone() for g in grads]
    c = _coeff(0.0437)
    clip_scale.scale_tensors(got, c)
    _te_scale(want, c)
    require_bitwise(all(_same_bits(a, b) for a, b in zip(got, want)), f"bitwise_clip_scale {dist}")
    for a, g in zip(got, grads):
        assert a.shape == g.shape and a.dtype == torch.bfloat16


@pytest.mark.parametrize("limit", [4096, 5000, 65536 * 3 + 5, 1_000_000])
def test_large_tensors_take_several_launches(require_bitwise, monkeypatch, limit):
    """A tensor above ``MAX_ELEMS`` is scaled slice by slice (every launch has an int32 element
    count, the kernel variant the self-check ran); here with the limit lowered."""
    from primus.backends.megatron.core.fusions import clip_scale

    assert clip_scale.MAX_ELEMS < 2**31
    launches = []
    kernel = clip_scale._scale

    class Counting:
        def __getitem__(self, grid):
            launches.append(grid)
            return kernel[grid]

    monkeypatch.setattr(clip_scale, "MAX_ELEMS", limit)
    monkeypatch.setattr(clip_scale, "_scale", Counting())
    gen = torch.Generator(device="cuda").manual_seed(13)
    shapes = [(65536 * 3 + 5,), (700, 300), (1,), (limit,), (limit + 1,)]
    grads = [torch.randn(s, device="cuda", generator=gen).bfloat16() for s in shapes]
    got = [g.clone() for g in grads]
    want = [g.clone() for g in grads]
    c = _coeff(0.0437)
    clip_scale.scale_tensors(got, c)
    _te_scale(want, c)
    require_bitwise(
        all(_same_bits(a, b) for a, b in zip(got, want)), f"bitwise_clip_scale in slices of {limit}"
    )
    assert len(launches) == sum(-(-g.numel() // limit) for g in grads)


def test_kernel_is_stable_over_repeats(require_bitwise):
    from primus.backends.megatron.core.fusions import clip_scale

    gen = torch.Generator(device="cuda").manual_seed(11)
    grad = torch.randn(3_000_001, device="cuda", generator=gen).bfloat16()
    c = _coeff(0.3141)
    want = grad.clone()
    _te_scale([want], c)
    ok = True
    for _ in range(200):
        got = grad.clone()
        clip_scale.scale_tensors([got], c)
        ok = ok and _same_bits(got, want)
    require_bitwise(ok, "bitwise_clip_scale over 200 repeats")


def test_eligibility():
    from primus.backends.megatron.core.fusions.clip_scale import eligible

    g = torch.zeros(64, 8, device="cuda", dtype=torch.bfloat16)
    c = _coeff(0.5)
    assert eligible([[g], [g]], c)
    assert eligible([[g, g[:3]], [g, g[:3]]], c)
    assert not eligible([[g], [g.clone()]], c)  # not in place
    assert not eligible([[g.float()], [g.float()]], c)  # FP32 gradients stay on TE
    assert not eligible([[g[:, ::2]], [g[:, ::2]]], c)  # strided
    assert not eligible([[g], [g]], 0.5)  # host coefficient: Megatron's other branch
    assert not eligible([[g], [g]], c.double())
    assert not eligible([[g], [g]], c.cpu())
    assert not eligible([[], []], c)
    assert not eligible([[g]], c)


def _patched(calls):
    from primus.backends.megatron.patches.te_patches.bitwise_clip_scale_patches import (
        make_bitwise_scale,
    )

    def original(chunk_size, noop_flag, tensor_lists, coeff):
        calls.append(len(tensor_lists[0]))
        return te_optimizers.multi_tensor_scale_tensor(chunk_size, noop_flag, tensor_lists, coeff)

    return make_bitwise_scale(original, te_optimizers.multi_tensor_applier)


def test_patched_scale_matches_te_and_leaves_other_calls_alone(require_bitwise, clean_checks):
    from primus.backends.megatron.core.fusions import clip_scale

    calls = []
    scale = _patched(calls)
    gen = torch.Generator(device="cuda").manual_seed(3)
    grads = [torch.randn(n, device="cuda", generator=gen).bfloat16() for n in (70001, 13, 4096)]
    got = [g.clone() for g in grads]
    want = [g.clone() for g in grads]
    c = _coeff(0.61)
    noop = torch.zeros(1, dtype=torch.int, device="cuda")
    te_optimizers.multi_tensor_applier(scale, noop, [got, got], c)
    _te_scale(want, c)
    assert all(_same_bits(a, b) for a, b in zip(got, want))  # either path gives TE's bits
    require_bitwise(clean_checks._results.get(("bitwise_clip_scale", None), False), "bitwise_clip_scale")
    # the check ran TE on probes only; the real list did not go through TE
    assert 3 not in calls
    # FP32 gradients: TE's call, unchanged
    calls.clear()
    g32 = torch.randn(1000, device="cuda", generator=gen)
    want32 = g32 * c
    te_optimizers.multi_tensor_applier(scale, noop, [[g32], [g32]], c)
    assert calls == [1] and torch.equal(g32, want32)
    assert clip_scale.eligible([[got[0]], [got[0]]], c)


def test_patched_scale_keeps_te_when_the_self_check_fails(monkeypatch, clean_checks):
    from primus.backends.megatron.core.fusions import clip_scale

    def broken(tensors, coeff):
        for t in tensors:
            t.zero_()

    calls = []
    scale = _patched(calls)
    monkeypatch.setattr(clip_scale, "scale_tensors", broken)
    g = torch.randn(5000, device="cuda").bfloat16()
    want = g.clone()
    c = _coeff(0.25)
    _te_scale([want], c)
    noop = torch.zeros(1, dtype=torch.int, device="cuda")
    te_optimizers.multi_tensor_applier(scale, noop, [[g], [g]], c)
    assert _same_bits(g, want)
    assert clean_checks._results == {("bitwise_clip_scale", None): False}
    assert calls[-1] == 1  # the real list went through TE


def test_switch_defaults_to_off():
    import os

    import yaml

    root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    with open(os.path.join(root, "primus/configs/modules/megatron/primus_megatron_module.yaml")) as f:
        assert yaml.safe_load(f)["bitwise_clip_scale"] is False
