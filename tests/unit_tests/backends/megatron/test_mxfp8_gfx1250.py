###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP8 on gfx1250 through Primus-Turbo, the parts decided from the args alone: the argument check, the
precision-aware optimizer patch, and that neither does anything on the shipped defaults. Runs without a GPU."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from megatron.core.optimizer.optimizer_config import OptimizerConfig

import primus.backends.megatron.patches.moe_patches.permute_fusion_patches  # noqa: F401  registers its patch
import primus.core.utils.gpu_arch as gpu_arch
from primus.backends.megatron.patches import optimizer_patches
from primus.backends.megatron.patches.args import rocm_arg_validation
from primus.core.config.yaml_loader import parse_yaml
from primus.core.patches import PatchContext
from primus.core.patches.patch_registry import PatchRegistry

_PAO_PATCH = "megatron.optimizer.precision_aware_turbo_mxfp8"
_TURBO_MXFP8 = dict(
    fp8="e4m3",
    fp8_recipe="mxfp8",
    enable_primus_turbo=True,
    use_turbo_gemm=True,
    use_turbo_grouped_gemm=True,
    use_precision_aware_optimizer=True,
    fp8_param_gather=False,
)
_NO_TURBO_GEMM = dict(use_turbo_gemm=False, use_turbo_grouped_gemm=False)


def _args(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{**_TURBO_MXFP8, **overrides})


def _ctx(params) -> PatchContext:
    return PatchContext(
        backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=params)}
    )


_NO_GEMM = "quantizes only in Primus-Turbo's GEMMs"
_OLD_TURBO = "needs a Primus-Turbo with the gfx1250 MXFP8 GEMMs"


@pytest.mark.parametrize(
    "overrides, gfx1250, turbo_has_mxfp8, refused",
    [
        ({}, True, True, None),
        (dict(use_turbo_gemm=False), True, True, None),  # the grouped experts alone
        (_NO_TURBO_GEMM, True, True, _NO_GEMM),  # nothing would quantize: TE's autocast stays off there
        (dict(enable_primus_turbo=False), True, True, _NO_GEMM),
        ({}, True, False, _OLD_TURBO),  # its MXFP8 GEMM would fall back to a backend that is wrong there
        (_NO_TURBO_GEMM, False, False, None),  # TE's MXFP8 elsewhere
        (dict(fp8=None, **_NO_TURBO_GEMM), True, False, None),
        (dict(fp8_recipe="tensorwise", **_NO_TURBO_GEMM), True, False, None),
    ],
)
def test_mxfp8_on_gfx1250_needs_turbo_gemms(monkeypatch, overrides, gfx1250, turbo_has_mxfp8, refused):
    monkeypatch.setattr(gpu_arch, "is_gfx1250", lambda: gfx1250)
    turbo_ops = ModuleType("primus_turbo.pytorch.ops")  # what the check imports once the args pass
    if turbo_has_mxfp8:
        turbo_ops.mx_wgrad_accum_supported = lambda grouped, main_grad: True
    monkeypatch.setitem(sys.modules, turbo_ops.__name__, turbo_ops)
    if refused:
        with pytest.raises(ValueError, match=refused):
            rocm_arg_validation.validate_mxfp8_on_gfx1250(_args(**overrides))
    else:
        rocm_arg_validation.validate_mxfp8_on_gfx1250(_args(**overrides))


@pytest.mark.parametrize(
    "overrides, gfx1250, want",
    [
        ({}, True, True),
        (dict(use_turbo_gemm=False), True, True),
        ({}, False, False),  # TE's MXFP8 recipe runs elsewhere: Megatron's behaviour stays
        (_NO_TURBO_GEMM, True, False),
        (dict(enable_primus_turbo=False), True, False),
        (dict(fp8_param_gather=True), True, False),  # FP8 primary weights need MCore's FP32 masters
        (dict(fp8_recipe="tensorwise"), True, False),
        (dict(use_precision_aware_optimizer=False), True, False),
    ],
)
def test_precision_aware_patch_condition(monkeypatch, overrides, gfx1250, want):
    monkeypatch.setattr(gpu_arch, "is_gfx1250", lambda: gfx1250)
    assert bool(PatchRegistry.get(_PAO_PATCH).condition(_ctx(_args(**overrides)))) is want


def test_precision_aware_patch_keeps_te_master_weights(monkeypatch):
    """Megatron gives up TE's master weights for every FP8 recipe but delayed scaling; the patch keeps them for
    mxfp8 and leaves the other recipes alone."""
    monkeypatch.setattr(
        OptimizerConfig, "__post_init__", OptimizerConfig.__post_init__
    )  # restored afterwards

    def te_master_weights(recipe):
        return OptimizerConfig(
            optimizer="adam",
            use_distributed_optimizer=True,
            use_precision_aware_optimizer=True,
            main_params_dtype=torch.float32,
            exp_avg_dtype=torch.bfloat16,
            exp_avg_sq_dtype=torch.bfloat16,
            fp8_recipe=recipe,
        ).use_precision_aware_optimizer_no_fp8_or_ds_fp8

    assert not te_master_weights("mxfp8")
    optimizer_patches.patch_precision_aware_for_turbo_mxfp8(SimpleNamespace())
    assert te_master_weights("mxfp8") and te_master_weights(None)
    assert not te_master_weights("blockwise")


@pytest.mark.parametrize("enable_primus_turbo", [False, True])
def test_inert_by_default(monkeypatch, enable_primus_turbo):
    """On the shipped defaults, even on gfx1250, nothing applies and nothing imports Primus-Turbo or
    TransformerEngine (unimportable here): the patch conditions, as the patch runner evaluates them for every
    Megatron job, and the argument check."""
    for name in (
        "primus_turbo",
        "transformer_engine",
        "primus.backends.megatron.core.extensions.primus_turbo",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(gpu_arch, "is_gfx1250", lambda: True)
    configs = Path(__file__).resolve().parents[4] / "primus/configs"
    params = {}
    for name in (
        "models/megatron/language_model.yaml",
        "modules/megatron/trainer_base.yaml",
        "modules/megatron/primus_megatron_module.yaml",
        "modules/megatron/primus_turbo.yaml",
    ):
        params.update(parse_yaml(str(configs / name)))
    params = SimpleNamespace(**{**params, "enable_primus_turbo": enable_primus_turbo})
    for patch_id in (_PAO_PATCH, "megatron.moe.permute_fusion"):
        assert not PatchRegistry.get(patch_id).applies_to(_ctx(params))
    rocm_arg_validation.validate_mxfp8_on_gfx1250(params)
