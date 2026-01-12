###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.args.tensorboard_path",
    backend="megatron",
    phase="build_args",
    description="Set TensorBoard directory based on experiment root",
)
def patch_tensorboard_path(ctx: PatchContext):
    """
    Configure TensorBoard directory.

    Sets args.tensorboard_dir to <exp_root>/tensorboard if TensorBoard
    is enabled, otherwise sets it to None.
    """

    args = ctx.extra.get("backend_args", {})
    primus_config = ctx.extra.get("primus_config", {})
    module_config = ctx.extra.get("module_config", {})

    disable_tensorboard = getattr(module_config.params, "disable_tensorboard", False)

    if args and getattr(args, "profile", False):
        disable_tensorboard = False
        log_rank_0("[Patch:megatron.args.profile_tensorboard] Enabled TensorBoard (profile=True)")

    exp_root_path = primus_config.exp_root_path
    if args and exp_root_path:
        if not disable_tensorboard:
            tb_path = os.path.abspath(os.path.join(exp_root_path, "tensorboard"))
            args.tensorboard_dir = tb_path
            log_rank_0(f"[Patch:megatron.args.tensorboard_path] tensorboard_dir → {tb_path}")
        else:
            log_rank_0("[Patch:megatron.args.tensorboard_path] TensorBoard disabled")
