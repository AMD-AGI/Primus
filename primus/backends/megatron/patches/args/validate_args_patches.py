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
from primus.core.utils.module_utils import log_rank_0

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

    from primus.backends.megatron.patches.args.rocm_arg_validation import (
        validate_args_on_rocm,
    )

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


def _te_cross_entropy_has_fp32_grad() -> bool:
    """Whether TE's fused cross entropy keeps the input gradient in FP32 (TE PR #3193).

    The fix renames ``cross_entropy_backward``'s first parameter from ``_input`` to
    ``grad_input``; ROCm TE builds report versions that do not track when it landed.
    """
    try:
        from transformer_engine.pytorch.triton import cross_entropy as te_triton_ce
    except ImportError:
        return False
    return next(iter(inspect.signature(te_triton_ce.cross_entropy_backward).parameters), None) == "grad_input"


@register_patch(
    "megatron.validate_args.te_cross_entropy_version",
    backend="megatron",
    phase="before_train",
    description="Keep TE cross-entropy fusion on Megatron-LM main below TE 2.19, as on the pinned Megatron",
    condition=lambda ctx: getattr(get_args(ctx), "cross_entropy_loss_fusion", False)
    and getattr(get_args(ctx), "cross_entropy_fusion_impl", None) == "te",
)
def patch_validate_args_te_cross_entropy(ctx: PatchContext):
    """
    Megatron-core 0.17-0.19 reject ``cross_entropy_fusion_impl='te'`` outright, and
    Megatron-LM main rejects it below TE 2.19, because older TE kept the input gradient in
    BF16 (TE PR #3193). ROCm TE versions do not track the fix (IFU-release_v2.18 has it;
    2.15 / 2.17.1 do not). Drop the check either way: the native fusion materialises FP32
    logits and runs out of memory on Llama-3.1-70B (TP1, seq 8192), and the kernel without
    the fix is what the pinned Megatron already runs. Older Megatron has no such check.
    """
    import megatron.training.arguments as megatron_args

    # validate_args can only be source-patched once, so pick the form that is present.
    source = inspect.getsource(megatron_args._primus_original_validate_args)
    for ori_code, new_code in (
        ('and not is_te_min_version("2.19.0")', "and False"),
        ("args.cross_entropy_loss_fusion and args.cross_entropy_fusion_impl == 'te'\n", "False\n"),
    ):
        if ori_code in source:
            _patch_validate_args_source(ori_code, new_code)
            break
    if _te_cross_entropy_has_fp32_grad():
        log_rank_0(
            "[Patch:megatron.validate_args.te_cross_entropy_version] "
            "TE fused cross entropy has the FP32-gradient fix; patched validate_args version check"
        )
    else:
        log_rank_0(
            "[Patch:megatron.validate_args.te_cross_entropy_version] WARNING: "
            "TE fused cross entropy lacks the FP32-gradient fix (TE PR #3193); keeping it as on "
            "the pinned Megatron. Use ROCm TE >= 2.18 for FP32 input gradients."
        )
