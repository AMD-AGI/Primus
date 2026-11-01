###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU tests: clipping by contiguous runs scales the gradients exactly as TE's multi-tensor kernel."""

import gc
import importlib.util
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from primus.core.patches import PatchContext

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
clip_grads = pytest.importorskip("megatron.core.optimizer.clip_grads")
if clip_grads.multi_tensor_scale_tensor_impl is None:
    pytest.skip("needs Transformer Engine's multi_tensor_scale_tensor", allow_module_level=True)

from primus.backends.megatron.core.extensions._triton import grad_scale  # noqa: E402
from primus.backends.megatron.patches import _patch_guard  # noqa: E402
from primus.backends.megatron.patches._rebind import rebind_everywhere  # noqa: E402

_PATCH_PATH = (
    Path(__file__).resolve().parents[4]
    / "primus/backends/megatron/patches/clip_grad_coalesced_scale_patches.py"
)
_spec = importlib.util.spec_from_file_location("clip_grad_coalesced_scale_under_test", _PATCH_PATH)
patch_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patch_mod)

UPSTREAM = clip_grads.clip_grad_by_total_norm_fp32
BLOCK = grad_scale._BLOCK
# (start, length) slices of one buffer: adjacent ones, a gap, and pieces smaller and larger than a block
SLICES = [
    (0, 1000),
    (1000, 3 * BLOCK + 7),
    (3 * BLOCK + 1007, 64),
    (3 * BLOCK + 1200, BLOCK - 3),
    (5 * BLOCK, 129),
]
BUF = 6 * BLOCK


@pytest.fixture(autouse=True)
def _no_plans():
    patch_mod._PLANS.clear()
    yield
    patch_mod._PLANS.clear()


def _views(dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    buf = (torch.randn(BUF, device="cuda", generator=g) * 4).to(dtype)
    return buf, [buf[s : s + n] for s, n in SLICES]


def _bits(t):
    return t.view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("coeff", [0.8374, 1.0 / 3.0, 1e-3])
def test_runs_match_te_kernel(dtype, coeff):
    ref_buf, ref = _views(dtype)
    got_buf, got = _views(dtype)
    scale = torch.tensor([coeff], dtype=torch.float32, device="cuda")
    dummy = torch.zeros(1, dtype=torch.int, device="cuda")
    clip_grads.multi_tensor_applier(clip_grads.multi_tensor_scale_tensor_impl, dummy, [ref, ref], scale)
    runs = patch_mod.contiguous_runs(got)
    assert len(runs) == 3  # [0, 3B+1007) [3B+1007, 3B+1071) is adjacent too; then the gap, then two more
    for run in runs:
        grad_scale.scale_by_device_scalar(run, scale)
    assert torch.equal(_bits(got_buf), _bits(ref_buf))


def test_overlap_and_non_contiguous_are_refused():
    buf, _ = _views(torch.bfloat16)
    assert patch_mod.contiguous_runs([buf[0:100], buf[50:150]]) is None
    assert patch_mod.contiguous_runs([buf[0:100:2]]) is None


def test_separate_tensors_are_left_to_upstream():
    """Nothing to merge, e.g. fp32 main grads copied out of a bf16 buffer every step."""
    assert patch_mod.contiguous_runs([torch.zeros(100, device="cuda") for _ in range(3)]) is None


def test_the_plan_cache_keeps_no_gradient_alive():
    buf, views = _views(torch.bfloat16)
    assert len(patch_mod.contiguous_runs(views)) == 3
    assert all(isinstance(i, int) and isinstance(n, int) for p in patch_mod._PLANS.values() for i, n in p)
    alive = weakref.ref(buf)
    del buf, views
    gc.collect()
    assert alive() is None


def test_a_plan_that_no_longer_fits_is_remade():
    a, b = torch.zeros(1000, device="cuda"), torch.zeros(1000, device="cuda")
    key = tuple((g.data_ptr(), g.numel(), g.dtype, g.device) for g in (a, b))
    patch_mod._PLANS[key] = ((0, 2000),)  # as if a and b were adjacent views of one buffer
    assert patch_mod.contiguous_runs([a, b]) is None
    assert patch_mod._PLANS[key] == ()


def _params(decoupled, separate=False):
    dtype = torch.bfloat16 if decoupled else torch.float32
    buf, views = _views(dtype, seed=1)
    if separate:
        views = [v.clone() for v in views]
    ps = []
    for v in views:
        p = SimpleNamespace(grad=None)
        if decoupled:
            p.decoupled_grad = v
        else:
            p.grad = v
        ps.append(p)
    return views, ps


@pytest.mark.parametrize("decoupled", [True, False])
@pytest.mark.parametrize("separate", [False, True])
def test_clip_matches_upstream(decoupled, separate):
    total_norm = torch.tensor([1.37], dtype=torch.float32, device="cuda")
    ref_grads, ref_params = _params(decoupled, separate)
    UPSTREAM(ref_params, 1.0, total_norm, use_decoupled_grad=decoupled)
    got_grads, got_params = _params(decoupled, separate)
    patch_mod._make_clip(UPSTREAM)(got_params, 1.0, total_norm, use_decoupled_grad=decoupled)
    for r, g in zip(ref_grads, got_grads):
        assert torch.equal(_bits(g), _bits(r))


def test_patch_rebinds_the_optimizer_modules():
    from megatron.core.optimizer import optimizer as mcore_optimizer

    try:
        patch_mod.patch_clip_grad_coalesced_scale(PatchContext(backend="megatron", phase="before_train"))
        patch_mod.patch_clip_grad_coalesced_scale(PatchContext(backend="megatron", phase="before_train"))
        patched = clip_grads.clip_grad_by_total_norm_fp32
        assert patched.__wrapped__ is UPSTREAM  # wrapped once
        assert mcore_optimizer.clip_grad_by_total_norm_fp32 is patched
    finally:
        rebind_everywhere(clip_grads, "clip_grad_by_total_norm_fp32", UPSTREAM)
        getattr(clip_grads, _patch_guard._SENTINEL_ATTR, set()).discard(patch_mod._PATCH_KEY)
    assert mcore_optimizer.clip_grad_by_total_norm_fp32 is UPSTREAM
