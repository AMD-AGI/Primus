###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_kv_rank_0, warning_rank_0


@register_patch(
    "megatron.args.moe_layer_freq",
    backend="megatron",
    phase="build_args",
    description="Normalize moe_layer_freq when provided as a string expression",
)
def patch_moe_layer_freq(ctx: PatchContext):
    """
    Normalize moe_layer_freq so Megatron does not see string expressions.

    Example:
        moe_layer_freq: "([0]*1+[1]*26)"
        -> args.moe_layer_freq becomes the evaluated list.
    """
    args = ctx.extra.get("backend_args", {})
    if not args or not hasattr(args, "moe_layer_freq"):
        return

    value = getattr(args, "moe_layer_freq", None)
    if isinstance(value, str):
        try:
            # Evaluate simple Python expressions used in configs, e.g. "([0]*1+[1]*26)".
            normalized = eval(value, {}, {})
        except Exception:
            warning_rank_0(
                f"[Patch:megatron.args.moe_layer_freq][WARN] "
                f"Failed to eval moe_layer_freq='{value}', keep as-is"
            )
            return

        args.moe_layer_freq = normalized
        log_kv_rank_0(
            "[Patch:megatron.args.moe_layer_freq] -moe_layer_freq",
            f"{args.moe_layer_freq}",
        )
