###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek Model-Specific Patches

Handles DeepSeek model quirks and optimizations.
"""

from primus.core.patches import (
    FunctionPatch,
    PatchContext,
    PatchPriority,
    PatchRegistry,
)


# Example: DeepSeek V3 MLA attention fix
def patched_deepseek_mla_attention(original_func, *args, **kwargs):
    """
    Fix for DeepSeek V3 MLA (Multi-head Latent Attention) implementation.

    Issue: Original implementation has numerical instability with bf16
    Fix: Add gradient scaling and numerical stability improvements
    """
    import torch

    # Enable gradient checkpointing for MLA layers
    if hasattr(torch, "utils") and hasattr(torch.utils, "checkpoint"):
        kwargs["use_checkpoint"] = True

    # Call original with modifications
    result = original_func(*args, **kwargs)

    # Apply gradient scaling if needed
    if torch.is_autocast_enabled():
        result = result * 0.5  # Scale down to prevent overflow

    return result


PatchRegistry.register(
    FunctionPatch(
        name="deepseek_v3_mla_attention_fix",
        description="Fix numerical instability in DeepSeek V3 MLA attention",
        target_module="megatron.core.models.gpt.gpt_layer_specs",
        target_function="get_gpt_layer_with_transformer_engine_spec",
        patch_function=patched_deepseek_mla_attention,
        wrap=True,
        framework="megatron",
        models=["deepseek_v3", "deepseek_v3_671B"],
        priority=PatchPriority.HIGH,
    )
)


# Example: DeepSeek MoE load balancing fix
class DeepSeekMoEPatch:
    """Custom patch for DeepSeek MoE load balancing."""

    @staticmethod
    def check_condition(context: PatchContext) -> bool:
        """Only apply if using MoE."""
        if context.config:
            return context.config.get("num_experts", 0) > 1
        return False

    @staticmethod
    def patched_moe_load_balance(original_func, *args, **kwargs):
        """
        Fix load balancing for DeepSeek MoE.

        Issue: Default load balancing causes training hang with >256 experts
        Fix: Use auxiliary loss with reduced weight
        """
        # Reduce auxiliary loss weight for large expert count
        if "aux_loss_weight" not in kwargs:
            kwargs["aux_loss_weight"] = 0.001  # Reduced from default 0.01

        return original_func(*args, **kwargs)


# Register with custom condition
patch = FunctionPatch(
    name="deepseek_moe_load_balance_fix",
    description="Fix load balancing for DeepSeek MoE with many experts",
    target_module="megatron.core.transformer.moe.moe_layer",
    target_function="MoELayer",
    patch_function=DeepSeekMoEPatch.patched_moe_load_balance,
    wrap=True,
    framework="megatron",
    models=["deepseek_v3", "deepseek_v3_671B"],
    priority=PatchPriority.NORMAL,
)
patch.check_condition = DeepSeekMoEPatch.check_condition
PatchRegistry.register(patch)


# Example: DeepSeek tokenizer compatibility
def patched_deepseek_tokenizer_init(original_func, *args, **kwargs):
    """
    Fix tokenizer initialization for DeepSeek models.

    Issue: DeepSeek uses custom tokenizer that's incompatible with HF
    Fix: Add compatibility layer
    """
    # Add DeepSeek-specific tokenizer args
    if "add_bos_token" not in kwargs:
        kwargs["add_bos_token"] = False
    if "add_eos_token" not in kwargs:
        kwargs["add_eos_token"] = False

    return original_func(*args, **kwargs)


PatchRegistry.register(
    FunctionPatch(
        name="deepseek_tokenizer_compat",
        description="DeepSeek tokenizer compatibility fix",
        target_module="megatron.training.tokenizer.tokenizer",
        target_function="build_tokenizer",
        patch_function=patched_deepseek_tokenizer_init,
        wrap=True,
        framework="megatron",
        models=["deepseek_v2", "deepseek_v3", "deepseek_v3_671B"],
        priority=PatchPriority.NORMAL,
    )
)
