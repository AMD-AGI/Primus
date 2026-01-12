###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.args.checkpoint_path",
    backend="megatron",
    phase="build_args",
    description="Set checkpoint save path based on experiment root",
)
def patch_checkpoint_path(ctx: PatchContext):
    """
    Configure checkpoint save path.

    Sets args.save to <exp_root>/checkpoints and warns if user
    provided a different path.
    """
    args = ctx.extra.get("backend_args", {})
    primus_config = ctx.extra.get("primus_config", {})

    if args and primus_config.exp_root_path:
        ckpt_path = os.path.abspath(os.path.join(primus_config.exp_root_path, "checkpoints"))

        if hasattr(args, "save") and args.save is not None and args.save != ckpt_path:
            log_rank_0(
                f"[Patch:megatron.args.checkpoint_path][WARN] "
                f"args.save is deprecated; overriding to: {ckpt_path}"
            )

        args.save = ckpt_path
        log_rank_0(f"[Patch:megatron.args.checkpoint_path] save → {ckpt_path}")
