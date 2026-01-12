###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_kv_rank_0


@register_patch(
    "megatron.args.data_path_split",
    backend="megatron",
    phase="build_args",
    description="Split space-separated data paths into lists",
)
def patch_data_path_split(ctx: PatchContext):
    """
    Preprocess data paths.

    Converts space-separated data path strings into lists:
    "data1 data2 data3" -> ["data1", "data2", "data3"]

    Applies to:
    - data_path
    - train_data_path
    - valid_data_path
    - test_data_path
    """
    args = ctx.extra.get("backend_args", {})
    if not args:
        return

    data_path = getattr(args, "data_path", None)
    train_data_path = getattr(args, "train_data_path", None)
    valid_data_path = getattr(args, "valid_data_path", None)
    test_data_path = getattr(args, "test_data_path", None)

    if data_path is not None:
        args.data_path = data_path.split(" ")
        log_kv_rank_0("[Patch:megatron.args.data_path_split]   -data_path", f"{args.data_path}")
    if train_data_path is not None:
        args.train_data_path = train_data_path.split(" ")
        log_kv_rank_0(
            "[Patch:megatron.args.data_path_split]   -train_data_path",
            f"{args.train_data_path}",
        )
    if valid_data_path is not None:
        args.valid_data_path = valid_data_path.split(" ")
        log_kv_rank_0(
            "[Patch:megatron.args.data_path_split]   -valid_data_path",
            f"{args.valid_data_path}",
        )
    if test_data_path is not None:
        args.test_data_path = test_data_path.split(" ")
        log_kv_rank_0(
            "[Patch:megatron.args.data_path_split]   -test_data_path",
            f"{args.test_data_path}",
        )
