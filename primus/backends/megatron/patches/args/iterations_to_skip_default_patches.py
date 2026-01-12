###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_kv_rank_0


@register_patch(
    "megatron.args.iterations_to_skip_default",
    backend="megatron",
    phase="build_args",
    description="Ensure iterations_to_skip has a list default instead of None",
)
def patch_iterations_to_skip_default(ctx: PatchContext):
    """
    Align iterations_to_skip behavior with trainer defaults:

        if args.iterations_to_skip is None:
            args.iterations_to_skip = []
    """
    args = ctx.extra.get("backend_args", {})
    if not args:
        return

    if getattr(args, "iterations_to_skip", None) is None:
        args.iterations_to_skip = []
        log_kv_rank_0(
            "[Patch:megatron.args.iterations_to_skip_default] -iterations_to_skip",
            f"{args.iterations_to_skip}",
        )
