# ###############################################################################
# # Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# #
# # See LICENSE for license information.
# ###############################################################################

# """
# Llama Model-Specific Patches

# Handles Llama 2/3 model quirks and optimizations.
# """

# from primus.core.patches import PatchContext, register_patch

# # ============================================================================
# # Llama 3 Patches
# # ============================================================================


# @register_patch(
#     "megatron.llama3.rope_scaling_long_context",
#     backend="megatron",
#     phase="after_build_args",
#     description="Fix RoPE scaling for Llama 3 long context (>8K)",
#     condition=lambda ctx: ctx.model_name and "llama3" in ctx.model_name.lower(),
# )
# def _fix_llama3_rope_scaling(ctx: PatchContext):
#     """
#     Fix RoPE scaling for Llama 3 with long context.

#     Issue: Default RoPE doesn't handle >8K context properly
#     Fix: Apply YaRN scaling for long contexts
#     """
#     args = ctx.extra.get("args")
#     if args is None:
#         return

#     max_pos = getattr(args, "max_position_embeddings", 8192)
#     if max_pos > 8192:
#         if hasattr(args, "rope_scaling_type"):
#             if not getattr(args, "rope_scaling_type", None):
#                 print(f"[Patch] Llama 3: Enabling YaRN RoPE scaling for {max_pos} context length")
#                 setattr(args, "rope_scaling_type", "yarn")
#                 setattr(args, "rope_scaling_factor", max_pos / 8192)


# # ============================================================================
# # Llama 2 Patches
# # ============================================================================


# @register_patch(
#     "megatron.llama2.gqa_tensor_parallel_fix",
#     backend="megatron",
#     phase="after_build_args",
#     description="Fix GQA with tensor parallelism for Llama 2",
#     condition=lambda ctx: ctx.model_name and "llama2" in ctx.model_name.lower(),
# )
# def _fix_llama2_gqa_tp(ctx: PatchContext):
#     """
#     Fix GQA (Grouped Query Attention) with tensor parallelism for Llama 2.

#     Issue: GQA with TP > 1 causes incorrect attention outputs
#     Fix: Adjust KV head distribution across TP ranks
#     """
#     args = ctx.extra.get("args")
#     if args is None:
#         return

#     # Get tensor parallel size
#     try:
#         from megatron.core import parallel_state

#         tp_size = parallel_state.get_tensor_model_parallel_world_size()
#     except Exception:
#         tp_size = 1

#     if tp_size > 1:
#         num_kv_heads = getattr(args, "num_key_value_heads", None)
#         if num_kv_heads and num_kv_heads % tp_size != 0:
#             # Round up to nearest multiple
#             new_kv_heads = ((num_kv_heads + tp_size - 1) // tp_size) * tp_size
#             print(
#                 f"[Patch] Llama 2 GQA: Adjusting KV heads from {num_kv_heads} to "
#                 f"{new_kv_heads} for TP={tp_size}"
#             )
#             setattr(args, "num_key_value_heads", new_kv_heads)


# # ============================================================================
# # General Llama Patches
# # ============================================================================


# @register_patch(
#     "megatron.llama.enable_flash_attention",
#     backend="megatron",
#     phase="after_build_args",
#     description="Auto-enable flash attention for Llama models",
#     condition=lambda ctx: ctx.model_name and "llama" in ctx.model_name.lower(),
# )
# def _enable_llama_flash_attention(ctx: PatchContext):
#     """
#     Auto-enable flash attention for Llama models if available.

#     Benefit: Significant speedup and memory reduction
#     """
#     args = ctx.extra.get("args")
#     if args is None:
#         return

#     # Check if flash attention is available
#     try:
#         import flash_attn  # noqa: F401

#         flash_available = True
#     except ImportError:
#         flash_available = False

#     if flash_available and hasattr(args, "use_flash_attn"):
#         if not getattr(args, "use_flash_attn", False):
#             print("[Patch] Llama: Enabling flash attention")
#             setattr(args, "use_flash_attn", True)
