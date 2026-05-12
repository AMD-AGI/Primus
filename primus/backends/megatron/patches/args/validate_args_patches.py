###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Patches for Megatron's ``validate_args`` function.

Registration order matters: the base wrapper is registered first so that
``_primus_original_validate_args`` exists before any source-mod patch reads it.
"""

import inspect

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0

# ---------------------------------------------------------------------------
# Base wrapper — always active, must be registered first
# ---------------------------------------------------------------------------


@register_patch(
    "megatron.validate_args",
    backend="megatron",
    phase="before_train",
    description="Wrap validate_args with ROCm-specific validation",
)
def patch_validate_args(ctx: PatchContext):
    """Store the original ``validate_args`` and install a wrapper that appends
    ROCm-specific checks.  Subsequent patches in this file may replace
    ``_primus_original_validate_args`` with a source-modified version."""
    import megatron.training.arguments as megatron_args
    import megatron.training.initialize as megatron_init

    from primus.modules.trainer.megatron.utils import validate_args_on_rocm

    megatron_args._primus_original_validate_args = megatron_args.validate_args

    def patched_validate_args(*args, **kwargs):
        megatron_args._primus_original_validate_args(*args, **kwargs)
        validated_args = args[0] if args else kwargs.get("args", None)
        validate_args_on_rocm(validated_args)
        return validated_args

    megatron_args.validate_args = patched_validate_args
    megatron_init.validate_args = patched_validate_args
    log_rank_0("[Patch:megatron.validate_args] Wrapped with ROCm validation")


# ---------------------------------------------------------------------------
# Source-level modifications — conditional, registered after the base wrapper
# ---------------------------------------------------------------------------


def _patch_validate_args_source(ori_code: str, new_code: str) -> None:
    """Replace a code fragment in the original ``validate_args`` source and
    store the recompiled function back."""
    import megatron.training.arguments as megatron_args

    original = megatron_args._primus_original_validate_args
    source = inspect.getsource(original)
    modified_source = source.replace(ori_code, new_code)
    namespace = {}
    exec(modified_source, original.__globals__, namespace)  # noqa: S102
    megatron_args._primus_original_validate_args = namespace[original.__name__]


@register_patch(
    "megatron.validate_args.decoder_pipeline_manual_split",
    backend="megatron",
    phase="before_train",
    description="Add decoder_pipeline_manual_split_list guard to validate_args pipeline split check",
    condition=lambda ctx: getattr(get_args(ctx), "decoder_pipeline_manual_split_list", None) is not None,
)
def patch_validate_args_pipeline_split(ctx: PatchContext):
    """
    When ``decoder_pipeline_manual_split_list`` is set, the original Megatron
    validation condition must also check that the list is ``None`` before
    falling into the default pipeline-split path.
    """
    ori_code = (
        "if args.decoder_first_pipeline_num_layers is None "
        "and args.decoder_last_pipeline_num_layers is None:"
    )
    new_code = "if args.decoder_pipeline_manual_split_list is None and " + ori_code.split("if ")[-1]
    _patch_validate_args_source(ori_code, new_code)
    log_rank_0(
        "[Patch:megatron.validate_args.decoder_pipeline_manual_split] "
        "Patched validate_args for decoder_pipeline_manual_split_list"
    )


@register_patch(
    "megatron.validate_args.fp4_te_version",
    backend="megatron",
    phase="before_train",
    description="Suppress TE version check for FP4 in validate_args on ROCm",
    condition=lambda ctx: getattr(get_args(ctx), "fp4", False),
)
def patch_validate_args_fp4(ctx: PatchContext):
    """
    ROCm TE does not yet report the version that Megatron requires for FP4.
    Suppress the ``ValueError`` so validation can continue; FP4 compatibility
    is verified separately by Primus.
    """
    ori_code = (
        "raise ValueError("
        '"--fp4-format requires Transformer Engine >= 2.7.0.dev0 '
        'for NVFP4BlockScaling support.")'
    )
    _patch_validate_args_source(ori_code, "pass")
    log_rank_0(
        "[Patch:megatron.validate_args.fp4_te_version] "
        "Patched validate_args to suppress FP4 TE version check"
    )
