###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark weighted RMSNorms under Megatron's deterministic mode.

The shared fused RMSNorm adds the weight gradient with ``tl.atomic_add``; under
``torch.use_deterministic_algorithms`` ``weighted_rms_norm`` keeps the weight product in
torch instead. Both paths give the same forward and gradients, and the deterministic one
is bit-reproducible on GPU.
"""

import pytest
import torch

from primus.backends.megatron.core.transformer.muse_spark.centered_norm import (
    centered_rms_norm,
    centered_rms_norm_reference,
    weighted_rms_norm,
)


@pytest.fixture
def deterministic():
    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was)


def _run(x, w, eps=1e-5):
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    out = weighted_rms_norm(x, 1.0 + w.float(), eps)
    out.float().pow(2).sum().backward()
    return out.detach(), x.grad, w.grad


def _inputs(device, dtype=torch.bfloat16):
    g = torch.Generator(device="cpu").manual_seed(0)
    x = (torch.randn(64, 256, generator=g) * 3).to(device=device, dtype=dtype)
    w = (torch.randn(256, generator=g) * 0.1).to(device=device)
    return x, w


def test_deterministic_path_matches_default_path(deterministic):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, w = _inputs(device)
    det = _run(x, w)
    torch.use_deterministic_algorithms(False)
    ref = _run(x, w)
    assert det[0].dtype == x.dtype
    torch.testing.assert_close(det[0].float(), ref[0].float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(det[1].float(), ref[1].float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(det[2].float(), ref[2].float(), atol=1e-2, rtol=1e-3)


def test_deterministic_centered_norm_matches_reference(deterministic):
    x, w = _inputs("cpu", torch.float32)
    torch.testing.assert_close(centered_rms_norm(x, w, 1e-5), centered_rms_norm_reference(x, w, 1e-5))


@pytest.mark.parametrize("det", [True, False])
def test_deterministic_mode_runs_the_shared_kernel_without_its_weight(monkeypatch, det):
    # the shared kernel's weight gradient is an atomic sum: in deterministic mode it must be
    # called with no weight (the product is formed in torch); otherwise once, with the weight
    from primus.backends.megatron.core.transformer.muse_spark import centered_norm

    calls = []
    real = centered_norm.fused_rms_norm

    def spy(x, weight=None, **kw):
        calls.append((weight is None, kw["out_dtype"]))
        return real(x, weight, **kw)

    monkeypatch.setattr(centered_norm, "fused_rms_norm", spy)
    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(det, warn_only=True)
    try:
        x, w = _inputs("cpu")
        out = weighted_rms_norm(x, 1.0 + w.float(), 1e-5)
    finally:
        torch.use_deterministic_algorithms(was)
    assert out.dtype == x.dtype
    assert calls == ([(True, torch.float32)] if det else [(False, x.dtype)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_deterministic_path_is_bit_reproducible_on_gpu(deterministic):
    x, w = _inputs("cuda")
    x = x.repeat(64, 1)  # many rows: the atomic path would reorder the weight-gradient sum
    first = _run(x, w)
    for _ in range(3):
        again = _run(x, w)
        for a, b in zip(first, again):
            assert torch.equal(a, b)
