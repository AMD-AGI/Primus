###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU tests: clipping by contiguous runs scales the gradients exactly as TE's multi-tensor kernel."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from primus.core.patches import PatchContext

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
clip_grads = pytest.importorskip("megatron.core.optimizer.clip_grads")
if clip_grads.multi_tensor_scale_tensor_impl is None:
    pytest.skip("needs Transformer Engine's multi_tensor_scale_tensor", allow_module_level=True)

_PATCH_PATH = (
    Path(__file__).resolve().parents[4]
    / "primus/backends/megatron/patches/clip_grad_coalesced_scale_patches.py"
)
_spec = importlib.util.spec_from_file_location("clip_grad_coalesced_scale_under_test", _PATCH_PATH)
patch_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patch_mod)

UPSTREAM = clip_grads.clip_grad_by_total_norm_fp32
BLOCK = patch_mod._BLOCK
# (start, length) slices of one buffer: adjacent ones, a gap, and pieces smaller and larger than a block
SLICES = [
    (0, 1000),
    (1000, 3 * BLOCK + 7),
    (3 * BLOCK + 1007, 64),
    (3 * BLOCK + 1200, BLOCK - 3),
    (5 * BLOCK, 129),
]
BUF = 6 * BLOCK


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
        patch_mod.scale_by_device_scalar(run, scale)
    assert torch.equal(_bits(got_buf), _bits(ref_buf))


def test_overlap_and_non_contiguous_are_refused():
    buf, _ = _views(torch.bfloat16)
    assert patch_mod.contiguous_runs([buf[0:100], buf[50:150]]) is None
    assert patch_mod.contiguous_runs([buf[0:100:2]]) is None


@pytest.mark.parametrize("decoupled", [True, False])
def test_clip_matches_upstream(decoupled):
    dtype = torch.bfloat16 if decoupled else torch.float32

    def params():
        buf, views = _views(dtype, seed=1)
        ps = []
        for v in views:
            p = SimpleNamespace(grad=None)
            if decoupled:
                p.decoupled_grad = v
            else:
                p.grad = v
            ps.append(p)
        return buf, ps

    total_norm = torch.tensor([1.37], dtype=torch.float32, device="cuda")
    ref_buf, ref_params = params()
    UPSTREAM(ref_params, 1.0, total_norm, use_decoupled_grad=decoupled)
    got_buf, got_params = params()
    patch_mod._make_clip(UPSTREAM)(got_params, 1.0, total_norm, use_decoupled_grad=decoupled)
    assert torch.equal(_bits(got_buf), _bits(ref_buf))


def test_patch_rebinds_the_optimizer_module():
    from megatron.core.optimizer import optimizer as mcore_optimizer

    try:
        patch_mod.patch_clip_grad_coalesced_scale(PatchContext(backend="megatron", phase="before_train"))
        assert getattr(mcore_optimizer.clip_grad_by_total_norm_fp32, "_primus_coalesced", False)
        assert getattr(clip_grads.clip_grad_by_total_norm_fp32, "_primus_coalesced", False)
    finally:
        clip_grads.clip_grad_by_total_norm_fp32 = UPSTREAM
        mcore_optimizer.clip_grad_by_total_norm_fp32 = UPSTREAM
