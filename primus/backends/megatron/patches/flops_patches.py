###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
FLOPs Calculation Patches

Patches for FLOPs estimation and performance profiling in Megatron.
"""

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.distributed_logging import log_rank_0


# ============================================================================
# FLOPs Calculator Patches
# ============================================================================


@register_patch(
    "megatron.flops.use_primus_estimator",
    backend="megatron",
    phase="before_train",
    description="Replace Megatron FLOPs calculator with Primus implementation",
)
def use_primus_flops_estimator(ctx: PatchContext):
    """
    Replace Megatron's FLOPs calculator with Primus implementation.

    Primus's FLOPs estimator supports:
        - Standard Transformer architectures
        - Hybrid architectures (Attention + Mamba + MLP)
        - MoE (Mixture of Experts) models
        - GQA (Grouped Query Attention)
        - SwiGLU activation

    The Primus estimator provides more accurate FLOPs calculation for
    hybrid models that combine different layer types.

    Benefits:
        - Accurate FLOPs reporting for hybrid architectures
        - Better performance profiling
        - Support for custom layer combinations
    """
    try:
        import megatron.training.training as megatron_training

        import primus.core.utils.flops_estimator as primus_flops

        # Replace Megatron's FLOPs calculator with Primus implementation
        megatron_training.num_floating_point_operations = primus_flops.num_floating_point_operations

        log_rank_0("[Patch:megatron.flops.use_primus_estimator] Replaced FLOPs calculator with Primus implementation")

    except ImportError as e:
        log_rank_0(f"[Patch:megatron.flops.use_primus_estimator][SKIP] Failed to import modules: {e}")
    except AttributeError as e:
        log_rank_0(
            f"[Patch:megatron.flops.use_primus_estimator][WARN] "
            f"Megatron version may not have num_floating_point_operations: {e}"
        )

