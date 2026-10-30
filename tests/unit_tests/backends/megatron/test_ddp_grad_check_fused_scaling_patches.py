###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU tests: the fused gradient check scales every bucket exactly as start_grad_sync would."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from primus.core.patches import PatchContext

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
pgb = pytest.importorskip("megatron.core.distributed.param_and_grad_buffer")

# Load just this patch so the test does not import unrelated GPU patch modules.
_PATCH_PATH = (
    Path(__file__).resolve().parents[4]
    / "primus/backends/megatron/patches/ddp_grad_check_fused_scaling_patches.py"
)
_spec = importlib.util.spec_from_file_location("ddp_grad_check_fused_scaling_under_test", _PATCH_PATH)
patch_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patch_mod)

GROUP = pgb._ParamAndGradBucketGroup
UPSTREAM_CHECK = GROUP.check_grads


@pytest.fixture
def patched():
    patch_mod.patch_fused_grad_check_scaling(PatchContext(backend="megatron", phase="before_train"))
    yield
    GROUP.check_grads = UPSTREAM_CHECK


def _group(sizes, dtype, factor, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    buckets = [
        SimpleNamespace(
            grad_data=(torch.randn(n, device="cuda", generator=g) * 3).to(dtype),
            gradient_scaling_factor=factor,
        )
        for n in sizes
    ]
    return SimpleNamespace(buckets=buckets)


def _start_grad_sync(group, check_for_nan=True):
    """The two steps of start_grad_sync that the patch touches, in its order."""
    GROUP.check_grads(group, check_for_nan, False)
    with torch.no_grad():
        for bucket in group.buckets:
            if bucket.gradient_scaling_factor != 1.0:
                bucket.grad_data *= bucket.gradient_scaling_factor


def _bits(t):
    return t.view(
        {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[t.dtype]
    )


def test_upstream_layout_matches():
    assert patch_mod._upstream_layout()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("factor", [0.25, 1.0 / 3.0, 1.0])
def test_scale_and_norm(dtype, factor):
    x = (torch.randn(3 * patch_mod._BLOCK + 129, device="cuda") * 3).to(dtype)
    want = x.clone()
    want_norm = want.norm(p=2)
    want *= factor
    got_norm = patch_mod.scale_and_norm(x, factor)
    assert torch.equal(_bits(x), _bits(want))
    assert got_norm.dtype == want_norm.dtype
    torch.testing.assert_close(got_norm.float(), want_norm.float(), rtol=1e-2, atol=0)


def test_empty_bucket():
    x = torch.empty(0, dtype=torch.bfloat16, device="cuda")
    assert patch_mod.scale_and_norm(x, 0.25).item() == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_steps_match_upstream(patched, dtype):
    sizes = [5000, 2 * patch_mod._BLOCK, 0, 77]
    ref, got = _group(sizes, dtype, 0.25), _group(sizes, dtype, 0.25)
    for step in range(3):
        for r, b in zip(ref.buckets, got.buckets):
            r.grad_data += step  # the next step's gradients, the same on both sides
            b.grad_data += step
        GROUP.check_grads = UPSTREAM_CHECK
        _start_grad_sync(ref)
        patch_mod.patch_fused_grad_check_scaling(PatchContext(backend="megatron", phase="before_train"))
        _start_grad_sync(got)
        for r, b in zip(ref.buckets, got.buckets):
            assert torch.equal(_bits(b.grad_data), _bits(r.grad_data)), f"step {step}"
            assert getattr(b, patch_mod._TRUE_FACTOR) == 0.25


def _upstream_error(group):
    GROUP.check_grads = UPSTREAM_CHECK
    with pytest.raises(RuntimeError) as err:
        _start_grad_sync(group)
    return str(err.value).split(": ", 1)[1]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_raises_upstream_error(bad):
    sizes = [300, 400, 500]
    ref, got = _group(sizes, torch.bfloat16, 0.25), _group(sizes, torch.bfloat16, 0.25)
    ref.buckets[1].grad_data[17] = bad
    got.buckets[1].grad_data[17] = bad
    want = _upstream_error(ref)
    patch_mod.patch_fused_grad_check_scaling(PatchContext(backend="megatron", phase="before_train"))
    try:
        with pytest.raises(RuntimeError) as err:
            _start_grad_sync(got)
    finally:
        GROUP.check_grads = UPSTREAM_CHECK
    assert str(err.value).split(": ", 1)[1] == want
    assert ("NaN" if bad != bad else "Inf") in want


def test_unfused_call_restores_the_factors(patched):
    group = _group([1000, 2000], torch.bfloat16, 0.25)
    want = [b.grad_data.clone() * 0.25 for b in group.buckets]
    _start_grad_sync(group)  # fused: factors parked at 1
    assert all(b.gradient_scaling_factor == 1.0 for b in group.buckets)
    for b, w in zip(group.buckets, want):
        assert torch.equal(_bits(b.grad_data), _bits(w))
        b.grad_data.copy_(w / 0.25)
    _start_grad_sync(group, check_for_nan=False)  # upstream: the true factors come back
    assert all(b.gradient_scaling_factor == 0.25 for b in group.buckets)
    for b, w in zip(group.buckets, want):
        assert torch.equal(_bits(b.grad_data), _bits(w))
