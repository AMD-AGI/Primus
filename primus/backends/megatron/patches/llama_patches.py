###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Llama Model-Specific Patches

Handles Llama 2/3 model quirks and optimizations.
"""

from primus.core.patches import FunctionPatch, PatchPriority, PatchRegistry


# Example: Llama 3 RoPE scaling fix
def patched_llama3_rope_scaling(original_func, *args, **kwargs):
    """
    Fix RoPE scaling for Llama 3 long context.

    Issue: Default RoPE doesn't handle >8K context properly
    Fix: Apply YaRN scaling for long contexts
    """
    # Check if context length > 8192
    if "max_position_embeddings" in kwargs:
        max_pos = kwargs["max_position_embeddings"]
        if max_pos > 8192:
            # Apply YaRN scaling
            kwargs["rope_scaling"] = {
                "type": "yarn",
                "factor": max_pos / 8192,
                "original_max_position_embeddings": 8192,
            }

    return original_func(*args, **kwargs)


PatchRegistry.register(
    FunctionPatch(
        name="llama3_rope_scaling_fix",
        description="Fix RoPE scaling for Llama 3 long context (>8K)",
        target_module="megatron.core.models.gpt.gpt_model",
        target_function="GPTModel",
        patch_function=patched_llama3_rope_scaling,
        wrap=True,
        framework="megatron",
        models=["llama3_8B", "llama3_70B", "llama3_405B"],
        priority=PatchPriority.HIGH,
    )
)


# Example: Llama 2 GQA (Grouped Query Attention) fix
def patched_llama2_gqa(original_func, *args, **kwargs):
    """
    Fix GQA implementation for Llama 2.

    Issue: GQA with TP > 1 causes incorrect attention outputs
    Fix: Adjust KV head distribution across TP ranks
    """

    # Get tensor parallel size
    try:
        from megatron.core import parallel_state

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
    except Exception:
        tp_size = 1

    # Adjust GQA for tensor parallelism
    if tp_size > 1 and "num_key_value_heads" in kwargs:
        num_kv_heads = kwargs["num_key_value_heads"]
        kwargs.get("num_attention_heads", num_kv_heads)

        # Ensure KV heads are evenly divisible by TP size
        if num_kv_heads % tp_size != 0:
            # Round up to nearest multiple
            num_kv_heads = ((num_kv_heads + tp_size - 1) // tp_size) * tp_size
            kwargs["num_key_value_heads"] = num_kv_heads
            print(
                f"[Primus:Patch] Adjusted KV heads for GQA: {kwargs['num_key_value_heads']} "
                f"(TP={tp_size})"
            )

    return original_func(*args, **kwargs)


PatchRegistry.register(
    FunctionPatch(
        name="llama2_gqa_tp_fix",
        description="Fix GQA with tensor parallelism for Llama 2",
        target_module="megatron.core.transformer.attention",
        target_function="Attention",
        patch_function=patched_llama2_gqa,
        wrap=True,
        framework="megatron",
        models=["llama2_7B", "llama2_13B", "llama2_70B"],
        priority=PatchPriority.HIGH,
    )
)


# Example: Llama flash attention compatibility
def patched_llama_flash_attn(original_func, *args, **kwargs):
    """
    Enable flash attention for Llama models.

    Issue: Flash attention not auto-enabled for Llama
    Fix: Force enable if available
    """
    try:
        pass

        kwargs["use_flash_attn"] = True
    except ImportError:
        pass

    return original_func(*args, **kwargs)


PatchRegistry.register(
    FunctionPatch(
        name="llama_flash_attn_enable",
        description="Auto-enable flash attention for Llama models",
        target_module="megatron.core.transformer.attention",
        target_function="Attention",
        patch_function=patched_llama_flash_attn,
        wrap=True,
        framework="megatron",
        models=["llama2_7B", "llama2_13B", "llama2_70B", "llama3_8B", "llama3_70B", "llama3_405B"],
        priority=PatchPriority.NORMAL,
    )
)
