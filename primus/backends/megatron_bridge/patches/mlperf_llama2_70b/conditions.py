###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.patches import PatchContext, get_args

_MLPERF_MODEL_TOKEN = "llama2_70b_lora_mxfp4"
_MLPERF_FLAVOR_TOKEN = "llama2_70b_lora_mxfp4_config"
_MLPERF_CONFIG_TOKEN = "llama2_70b_lora_mlperf_posttrain"


def is_llama2_70b_mlperf(ctx: PatchContext) -> bool:
    """Return True when the active run targets MLPerf Llama2-70B LoRA."""
    model_name = str(ctx.model_name or "")
    if _MLPERF_MODEL_TOKEN in model_name:
        return True

    try:
        args = get_args(ctx)
    except AssertionError:
        args = None

    if args is not None:
        model = str(getattr(args, "model", "") or "")
        if _MLPERF_MODEL_TOKEN in model:
            return True

        recipe = str(getattr(args, "recipe", "") or "")
        flavor = str(getattr(args, "flavor", "") or "")
        if _MLPERF_FLAVOR_TOKEN in flavor and (
            "llama2_custom" in recipe or "recipes.mlperf_llama2_70b" in recipe
        ):
            return True

    primus_config = ctx.extra.get("primus_config")
    if primus_config is not None:
        config_file = str(getattr(primus_config, "config_file", "") or "")
        if _MLPERF_CONFIG_TOKEN in config_file:
            return True

    return False
