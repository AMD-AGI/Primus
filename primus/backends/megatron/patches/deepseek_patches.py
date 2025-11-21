###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek Model-Specific Patches

Handles DeepSeek model quirks and optimizations.
"""

from primus.core.patches import PatchContext, register_patch

# ============================================================================
# DeepSeek V2 Patches
# ============================================================================


@register_patch(
    "megatron.deepseek_v2.fix_hang_overlap_param_gather",
    backend="megatron",
    phase="before_train",
    description="Workaround DeepSeek V2 hang with overlap_param_gather_with_optimizer_step",
    condition=lambda ctx: ctx.model_name and "deepseek_v2" in ctx.model_name.lower(),
)
def _fix_deepseek_v2_hang(ctx: PatchContext):
    """
    Fix DeepSeek V2 training hang issue.

    Issue: overlap_param_gather_with_optimizer_step causes hang with DeepSeek V2
    Fix: Force disable this option
    """
    args = ctx.extra.get("args")
    if args is None:
        return

    if hasattr(args, "overlap_param_gather_with_optimizer_step"):
        old_value = getattr(args, "overlap_param_gather_with_optimizer_step")
        if old_value:
            print(
                "[Patch] DeepSeek V2: Disabling overlap_param_gather_with_optimizer_step "
                "to prevent training hang"
            )
            setattr(args, "overlap_param_gather_with_optimizer_step", False)


# ============================================================================
# DeepSeek V3 Patches
# ============================================================================


@register_patch(
    "megatron.deepseek_v3.moe_load_balance",
    backend="megatron",
    phase="before_train",
    description="Fix MoE load balancing for DeepSeek V3 with many experts",
    condition=lambda ctx: ctx.model_name and "deepseek_v3" in ctx.model_name.lower(),
)
def _fix_deepseek_v3_moe_load_balance(ctx: PatchContext):
    """
    Fix MoE load balancing for DeepSeek V3.

    Issue: Default aux loss weight causes training hang with >256 experts
    Fix: Reduce aux loss weight
    """
    args = ctx.extra.get("args")
    if args is None:
        return

    # Check if using MoE
    num_experts = getattr(args, "num_experts", 0)
    if num_experts > 1:
        if hasattr(args, "moe_aux_loss_coeff"):
            old_value = getattr(args, "moe_aux_loss_coeff", None)
            if old_value is None or old_value > 0.001:
                print(
                    f"[Patch] DeepSeek V3 MoE: Setting moe_aux_loss_coeff=0.001 "
                    f"(was {old_value}) for {num_experts} experts"
                )
                setattr(args, "moe_aux_loss_coeff", 0.001)


@register_patch(
    "megatron.deepseek_v3.mla_attention_stability",
    backend="megatron",
    phase="before_train",
    description="Improve numerical stability for DeepSeek V3 MLA attention",
    condition=lambda ctx: ctx.model_name and "deepseek_v3" in ctx.model_name.lower(),
)
def _fix_deepseek_v3_mla_attention(ctx: PatchContext):
    """
    Improve numerical stability for DeepSeek V3 MLA (Multi-head Latent Attention).

    Issue: MLA attention has numerical instability with bf16
    Fix: Enable gradient checkpointing for MLA layers
    """
    args = ctx.extra.get("args")
    if args is None:
        return

    if hasattr(args, "recompute_granularity"):
        # Enable selective recomputation for attention layers
        if not getattr(args, "recompute_granularity", None):
            print("[Patch] DeepSeek V3 MLA: Enabling selective recomputation for stability")
            setattr(args, "recompute_granularity", "selective")


# ============================================================================
# DeepSeek Tokenizer Patches
# ============================================================================


@register_patch(
    "megatron.deepseek.tokenizer_compat",
    backend="megatron",
    phase="after_build_args",
    description="DeepSeek tokenizer compatibility fix",
    condition=lambda ctx: ctx.model_name and "deepseek" in ctx.model_name.lower(),
)
def _fix_deepseek_tokenizer(ctx: PatchContext):
    """
    Fix tokenizer compatibility for DeepSeek models.

    Issue: DeepSeek uses custom tokenizer incompatible with default HF tokenizer
    Fix: Adjust tokenizer settings
    """
    args = ctx.extra.get("args")
    if args is None:
        return

    if hasattr(args, "tokenizer_type"):
        tokenizer_type = getattr(args, "tokenizer_type", None)
        if tokenizer_type == "GPT2BPETokenizer":
            # DeepSeek-specific tokenizer settings
            if hasattr(args, "vocab_extra_ids"):
                if getattr(args, "vocab_extra_ids", 0) == 0:
                    print("[Patch] DeepSeek: Setting vocab_extra_ids for custom tokenizer")
                    setattr(args, "vocab_extra_ids", 100)
