###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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

    Behavior:
      - args.save is None  (YAML explicitly set `save: null`, or unset):
            user has opted out of saving — leave args.save = None so that
            Megatron's `checkpoint_and_decide_exit` (training.py:2340) skips
            save_checkpoint_and_time(...). This is what allows runs configured
            with `save: null` + `save_interval: <any>` to avoid hitting the
            checkpoint save path entirely.
      - args.save is non-None:
            user has opted in to saving — route the destination to
            <exp_root>/checkpoints, warning if it differs from what the user
            specified.
    """
    args = ctx.extra.get("backend_args", {})
    primus_config = ctx.extra.get("primus_config", {})

    if not args or not primus_config.exp_root_path:
        return

    # Respect explicit opt-out via `save: null` in YAML.
    if not hasattr(args, "save") or args.save is None:
        log_rank_0(
            "[Patch:megatron.args.checkpoint_path] "
            "args.save is None (opt-out); skipping save-path override."
        )
        return

    ckpt_path = os.path.abspath(os.path.join(primus_config.exp_root_path, "checkpoints"))

    if args.save != ckpt_path:
        log_rank_0(
            f"[Patch:megatron.args.checkpoint_path][WARN] "
            f"args.save is deprecated; overriding to: {ckpt_path}"
        )

    args.save = ckpt_path
    log_rank_0(f"[Patch:megatron.args.checkpoint_path] save → {ckpt_path}")
