###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The opt-in Primus fusion patches are registered, off by default and gated by their own flag only.

For each flag: the registered patch (as run_patches sees it) does not apply when the flag is absent or false and
applies when it is true, and primus_megatron_module.yaml declares the flag with the default false. Runs on CPU.
"""

import importlib
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("megatron")
pytest.importorskip("triton")
yaml = pytest.importorskip("yaml")

import primus  # noqa: E402
from primus.core.patches.context import PatchContext  # noqa: E402
from primus.core.patches.patch_registry import PatchRegistry  # noqa: E402

_PATCH_MODULES = {
    "megatron.triton.pinned_configs": "primus.backends.megatron.patches.triton_pinned_config_patches",
    "megatron.loss.fused_lm_cross_entropy": "primus.backends.megatron.patches.triton_lm_cross_entropy_patches",
    "megatron.activation.fused_swiglu": "primus.backends.megatron.patches.triton_swiglu_patches",
    "megatron.moe.triton_router_gemm": "primus.backends.megatron.patches.triton_router_gemm_patches",
    "megatron.optimizer.fused_adamw_clip": "primus.backends.megatron.patches.fused_adamw_clip_patches",
    "megatron.ddp.global_grad_nan_check": "primus.backends.megatron.patches.fused_adamw_clip_patches",
}


def _patch(patch_id):
    mod = importlib.import_module(_PATCH_MODULES[patch_id])
    if PatchRegistry.get(patch_id) is None:  # some suites clear the global registry
        importlib.reload(mod)
    patch = PatchRegistry.get(patch_id)
    assert patch is not None, patch_id
    return patch


def _ctx(**args):
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**args))},
    )


def _module_yaml():
    path = os.path.join(
        os.path.dirname(primus.__file__), "configs", "modules", "megatron", "primus_megatron_module.yaml"
    )
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize(
    "patch_id,arg,extra",
    [
        (
            "megatron.triton.pinned_configs",
            "use_primus_pinned_triton_configs",
            dict(multi_latent_attention=True, apply_rope_fusion=True, moe_permute_fusion=True),
        ),
        ("megatron.loss.fused_lm_cross_entropy", "use_primus_fused_lm_cross_entropy", {}),
        ("megatron.activation.fused_swiglu", "use_primus_fused_swiglu", {}),
        ("megatron.moe.triton_router_gemm", "use_primus_triton_router_gemm", {}),
        ("megatron.optimizer.fused_adamw_clip", "use_primus_fused_adamw_clip", {}),
        ("megatron.ddp.global_grad_nan_check", "use_primus_global_grad_nan_check", {}),
    ],
)
def test_opt_in_patch_is_off_by_default(patch_id, arg, extra):
    patch = _patch(patch_id)
    assert patch.backend == "megatron" and patch.phase == "before_train"
    assert not patch.applies_to(_ctx(**extra))
    assert not patch.applies_to(_ctx(**extra, **{arg: False}))
    assert patch.applies_to(_ctx(**extra, **{arg: True}))
    module_yaml = _module_yaml()
    assert arg in module_yaml and module_yaml[arg] is False
