###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron FSDP Patches

This module contains patches that modify Megatron's FSDP integration to use
Primus-specific implementations when requested via module_config.
"""


from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0


@register_patch(
    "megatron.fsdp.torch_fsdp2",
    backend="megatron",
    phase="before_train",
    description=(
        "Replace Megatron's TorchFullyShardedDataParallel with Primus implementation "
        "when use_torch_fsdp2 is enabled."
    ),
)
def patch_torch_fsdp(ctx: PatchContext):
    """
    Patch Megatron to use Primus's TorchFullyShardedDataParallel wrapper.

    Behavior (moved from MegatronTrainer.patch_torch_fsdp):
        - If module_config.use_torch_fsdp2 is True:
            * Patch megatron.core.distributed.torch_fully_sharded_data_parallel.
            * Patch megatron.training.training.torch_FSDP reference.
    """
    args = ctx.extra.get("backend_args", {})
    if not args:
        warning_rank_0("[Patch:megatron.fsdp.torch_fsdp2] backend_args is not set.")
        return

    if not getattr(args, "use_torch_fsdp2", False):
        log_rank_0("[Patch:megatron.fsdp.torch_fsdp2] use_torch_fsdp2 is not enabled.")
        return

    log_rank_0("[Patch:megatron.fsdp.torch_fsdp2] Patching torch_FSDP2 with Primus implementation...")

    try:
        # Import custom FSDP wrapper
        # Patch Megatron's internal reference to FSDP2 class
        import megatron.core.distributed.torch_fully_sharded_data_parallel as torch_fsdp_module

        from primus.backends.megatron.core.distributed.torch_fully_sharded_data_parallel import (
            PrimusTorchFullyShardedDataParallel,
        )

        torch_fsdp_module.TorchTorchFullyShardedDataParallel = PrimusTorchFullyShardedDataParallel

        # Patch training code reference
        from megatron.training import training

        training.torch_FSDP = PrimusTorchFullyShardedDataParallel

        log_rank_0("[Patch:megatron.fsdp.torch_fsdp2] torch_FSDP2 patch applied successfully.")

    except ImportError as e:
        warning_rank_0(
            f"[Patch:megatron.fsdp.torch_fsdp2] Failed to patch torch_FSDP2: missing dependencies: {e}"
        )
    except Exception as e:
        warning_rank_0(
            f"[Patch:megatron.fsdp.torch_fsdp2] Unexpected error occurred during FSDP patching: {e}"
        )
