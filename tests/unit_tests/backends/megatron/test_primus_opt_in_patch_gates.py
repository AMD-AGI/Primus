###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The opt-in Primus fusion patches are registered, off by default (also in the module yaml) and gated by
their own flag only; their shared log helpers."""

import importlib
import os

import pytest

pytest.importorskip("megatron")
pytest.importorskip("triton")
yaml = pytest.importorskip("yaml")

import primus  # noqa: E402
from primus.backends.megatron.core.fusions import _notices  # noqa: E402
from tests.unit_tests.backends.megatron.conftest import (  # noqa: E402
    patch_ctx,
    registered_patch,
)

_PATCHES = "primus.backends.megatron.patches."


def _module_yaml():
    path = os.path.join(
        os.path.dirname(primus.__file__), "configs", "modules", "megatron", "primus_megatron_module.yaml"
    )
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize(
    "patch_id,module,arg,extra",
    [
        (
            "megatron.triton.pinned_configs",
            "pinned_triton_configs_patches",
            "use_primus_pinned_triton_configs",
            dict(multi_latent_attention=True, apply_rope_fusion=True, moe_permute_fusion=True),
        ),
        (
            "megatron.loss.fused_lm_cross_entropy",
            "fused_lm_cross_entropy_patches",
            "use_primus_fused_lm_cross_entropy",
            {},
        ),
        ("megatron.activation.fused_swiglu", "fused_swiglu_patches", "use_primus_fused_swiglu", {}),
        (
            "megatron.moe.triton_router_gemm",
            "triton_router_gemm_patches",
            "use_primus_triton_router_gemm",
            {},
        ),
        (
            "megatron.optimizer.triton_adamw_step",
            "triton_adamw_step_patches",
            "use_primus_triton_adamw_step",
            {},
        ),
        (
            "megatron.ddp.global_grad_nan_check",
            "triton_adamw_step_patches",
            "use_primus_global_grad_nan_check",
            {},
        ),
    ],
)
def test_opt_in_patch_is_off_by_default(patch_id, module, arg, extra):
    patch = registered_patch(patch_id, importlib.import_module(_PATCHES + module))
    assert patch.backend == "megatron" and patch.phase == "before_train"
    assert not patch.applies_to(patch_ctx(**extra))
    assert not patch.applies_to(patch_ctx(**extra, **{arg: False}))
    assert patch.applies_to(patch_ctx(**extra, **{arg: True}))
    module_yaml = _module_yaml()
    assert arg in module_yaml and module_yaml[arg] is False


def test_fallback_is_logged_once_per_reason(monkeypatch):
    logged = []
    monkeypatch.setattr(_notices, "warning_rank_0", logged.append)
    _notices.warn_fallback_once.cache_clear()
    for reason in ("a", "b", "a"):
        _notices.warn_fallback_once("test.key", reason)
    assert logged == ["[Patch:test.key] original path: a", "[Patch:test.key] original path: b"]


@pytest.mark.parametrize("arch,warns", [("gfx1250", False), ("gfx950", True), ("", False)])
def test_gfx1250_notice(monkeypatch, arch, warns):
    logged = []
    monkeypatch.setattr(_notices, "warning_rank_0", logged.append)
    monkeypatch.setattr(_notices, "_gpu_arch", lambda: arch)
    _notices.warn_if_not_gfx1250("test.key", "use_test_switch")
    assert bool(logged) == warns
