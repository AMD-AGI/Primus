###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Runtime Megatron-LM patches for MLPerf Llama2-70B LoRA MXFP4.

Important: never delegate to Primus-Turbo ``fp4_utils.get_fp4_recipe`` here —
that function can return a ``(recipe, reason)`` tuple, which breaks TE when it
expects a recipe object with ``.mxfp4()`` / ``.mxfp8()`` methods.
"""

from __future__ import annotations

import enum
import functools
import os
from typing import Any

from primus.backends.megatron_bridge.patches.mlperf_llama2_70b.conditions import is_llama2_70b_mlperf
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCHED_ATTR = "_primus_mlperf_llama2_70b_megatron_patched"
_mxfp4_phase = True


def _mark_patched(obj: Any) -> None:
    setattr(obj, _PATCHED_ATTR, True)


def _already_patched(obj: Any) -> bool:
    return bool(getattr(obj, _PATCHED_ATTR, False))


def is_mxfp4_phase() -> bool:
    return _mxfp4_phase


def set_mxfp4_phase(active: bool) -> None:
    global _mxfp4_phase
    _mxfp4_phase = active


def _upstream_get_fp4_recipe_handles_mxfp4() -> bool:
    """True only when Megatron's get_fp4_recipe already builds MXFP4BlockScaling."""
    try:
        import inspect

        from megatron.core import fp4_utils

        source = inspect.getsource(fp4_utils.get_fp4_recipe)
        return "MXFP4BlockScaling" in source or "Fp4Recipe.mxfp4" in source
    except Exception:
        return False


def _build_mxfp4_get_fp4_recipe(orig_get_fp4_recipe):
    import transformer_engine.common.recipe
    from megatron.core.enums import Fp4Recipe
    from megatron.core.fp8_utils import _get_custom_recipe
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import is_te_min_version

    @functools.wraps(orig_get_fp4_recipe)
    def _mlperf_get_fp4_recipe(config: TransformerConfig):
        if config.fp4_recipe == Fp4Recipe.nvfp4:
            if not is_te_min_version("2.7.0.dev0"):
                raise ValueError("NVFP4BlockScaling requires TransformerEngine >= 2.7.0.dev0.")
            fp4_recipe = transformer_engine.common.recipe.NVFP4BlockScaling()
        elif config.fp4_recipe == Fp4Recipe.mxfp4:
            if not is_te_min_version("2.8.0"):
                raise ValueError("MXFP4BlockScaling requires TransformerEngine >= 2.8.0.")
            fp4_recipe = transformer_engine.common.recipe.MXFP4BlockScaling()
            fp4_recipe.use_hadamard = os.environ.get("NVTE_MXFP4_USE_HADAMARD", "0") == "1"
        elif config.fp4_recipe == Fp4Recipe.custom:
            fp4_recipe = _get_custom_recipe(config.fp4_quantizer_factory)
        else:
            raise ValueError(
                f"Unsupported FP4 recipe: {config.fp4_recipe}. Supported: nvfp4, mxfp4, custom."
            )
        return fp4_recipe

    return _mlperf_get_fp4_recipe


@register_patch(
    "mlperf_llama2_70b.megatron.fp4_mxfp4",
    backend="megatron",
    phase="before_train",
    condition=is_llama2_70b_mlperf,
    description="MXFP4 recipe + phase tracking for MLPerf Llama2 (single recipe object)",
)
def patch_fp4_mxfp4(ctx: PatchContext) -> None:
    from megatron.core import enums, fp4_utils

    recipe_already_handles_mxfp4 = _upstream_get_fp4_recipe_handles_mxfp4()

    if not hasattr(enums.Fp4Recipe, "mxfp4"):
        class _MlperfFp4Recipe(str, enum.Enum):
            nvfp4 = "nvfp4"
            mxfp4 = "mxfp4"
            custom = "custom"

        enums.Fp4Recipe = _MlperfFp4Recipe
        log_rank_0("[Patch:mlperf_llama2_70b.megatron.fp4_mxfp4] Added Fp4Recipe.mxfp4")

    fp4_utils.is_mxfp4_phase = is_mxfp4_phase
    fp4_utils.set_mxfp4_phase = set_mxfp4_phase
    fp4_utils._mxfp4_phase = _mxfp4_phase

    if recipe_already_handles_mxfp4:
        log_rank_0(
            "[Patch:mlperf_llama2_70b.megatron.fp4_mxfp4] "
            "Upstream get_fp4_recipe already supports mxfp4; phase tracking only"
        )
        return

    if _already_patched(fp4_utils.get_fp4_recipe):
        log_rank_0(
            "[Patch:mlperf_llama2_70b.megatron.fp4_mxfp4] "
            "get_fp4_recipe already patched for MLPerf mxfp4"
        )
        return

    fp4_utils.get_fp4_recipe = _build_mxfp4_get_fp4_recipe(fp4_utils.get_fp4_recipe)
    _mark_patched(fp4_utils.get_fp4_recipe)

    log_rank_0("[Patch:mlperf_llama2_70b.megatron.fp4_mxfp4] MXFP4 get_fp4_recipe patch applied")


@register_patch(
    "mlperf_llama2_70b.megatron.te_swiglu",
    backend="megatron",
    phase="before_train",
    condition=is_llama2_70b_mlperf,
    description="Optional TE SwiGLU path when USE_TE_SWIGLU=1",
)
def patch_te_swiglu(ctx: PatchContext) -> None:
    if os.getenv("USE_TE_SWIGLU", "0") != "1":
        log_rank_0("[Patch:mlperf_llama2_70b.megatron.te_swiglu] USE_TE_SWIGLU!=1; skipping")
        return

    import transformer_engine.common.recipe
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from megatron.core.fusions import fused_bias_swiglu as swiglu_mod

    if _already_patched(swiglu_mod.SwiGLUFunction.forward):
        return

    orig_forward = swiglu_mod.SwiGLUFunction.forward

    @staticmethod
    @swiglu_mod.nvtx_decorator()
    def _te_swiglu_forward(ctx, input_tensor, fp8_input_store, cpu_offload_input):
        import torch

        ctx.fp8_input_store = fp8_input_store
        ctx.ori_input_dtype = input_tensor.dtype
        input_for_backward = (
            input_tensor.to(torch.float8_e4m3fn) if fp8_input_store else input_tensor
        )
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
        ctx.save_for_backward(input_for_backward)

        swiglu_op = te.ops.SwiGLU()
        if fp8_input_store:
            recipe = transformer_engine.common.recipe.DelayedScaling(
                fp8_format=transformer_engine.common.recipe.Format.E4M3,
                amax_history_len=8,
                amax_compute_algo="max",
                margin=2,
                interval=1,
            )
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                return swiglu_op(input_tensor)
        return swiglu_op(input_tensor)

    @staticmethod
    @swiglu_mod.nvtx_decorator()
    def _te_swiglu_backward(ctx, grad_output):
        import torch

        input_tensor = ctx.saved_tensors[0]
        input_tensor = (
            input_tensor.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input_tensor
        )
        return tex.dswiglu(grad_output, input_tensor, None), None, None

    swiglu_mod.SwiGLUFunction.forward = _te_swiglu_forward
    swiglu_mod.SwiGLUFunction.backward = _te_swiglu_backward
    _mark_patched(_te_swiglu_forward)
    log_rank_0("[Patch:mlperf_llama2_70b.megatron.te_swiglu] TE SwiGLU patches applied")
