###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_kv_rank_0, log_rank_0


@register_patch(
    "megatron.args.mock_data",
    backend="megatron",
    phase="build_args",
    description="Disable data paths when using mock data",
)
def patch_mock_data(ctx: PatchContext):
    """
    Handle mock data mode.

    When mock_data=True, sets all data paths to None to prevent
    Megatron from trying to load actual data files.
    """
    args = ctx.extra.get("backend_args", {})
    module_config = ctx.extra.get("module_config", {})

    if not args or not module_config:
        return

    mock_data = getattr(module_config.params, "mock_data", False)
    if mock_data:
        args.data_path = None
        args.train_data_path = None
        args.valid_data_path = None
        args.test_data_path = None
        log_rank_0("[Patch:megatron.args.mock_data] Mock data enabled; all data paths set to None")

    log_kv_rank_0(f"[Patch:megatron.args.mock_data] -mock_data", f"{mock_data}")
    log_kv_rank_0(f"[Patch:megatron.args.mock_data]   -data_path", f"{args.data_path}")
    log_kv_rank_0(f"[Patch:megatron.args.mock_data]   -train_data_path", f"{args.train_data_path}")
    log_kv_rank_0(f"[Patch:megatron.args.mock_data]   -valid_data_path", f"{args.valid_data_path}")
    log_kv_rank_0(f"[Patch:megatron.args.mock_data]   -test_data_path", f"{args.test_data_path}")
