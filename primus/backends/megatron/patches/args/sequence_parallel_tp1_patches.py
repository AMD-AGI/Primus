###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_kv_rank_0, log_rank_0


@register_patch(
    "megatron.args.sequence_parallel_tp1",
    backend="megatron",
    phase="build_args",
    description="Disable sequence_parallel when tensor_model_parallel_size == 1",
)
def patch_sequence_parallel_tp1(ctx: PatchContext):
    """
    Align sequence_parallel behavior with trainer defaults:

        if args.tensor_model_parallel_size == 1:
            args.sequence_parallel = False
    """
    args = ctx.extra.get("backend_args", {})
    if not args:
        return

    tp_size = getattr(args, "tensor_model_parallel_size", None)
    if tp_size == 1:
        # Only log when we actually change the flag.
        if getattr(args, "sequence_parallel", None):
            log_rank_0(
                "[Patch:megatron.args.sequence_parallel_tp1] "
                "sequence_parallel=True is incompatible with tp_size=1; forcing to False."
            )
        args.sequence_parallel = False
        log_kv_rank_0(
            "[Patch:megatron.args.sequence_parallel_tp1] -sequence_parallel",
            f"{args.sequence_parallel}",
        )
