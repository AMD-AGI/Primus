###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Tokenizer Builder Patches

Override Megatron's build_tokenizer to use Primus version which properly
handles custom tokenizer types (Llama2Tokenizer, Llama3Tokenizer, etc.)
with HuggingFace Hub ID support.

Background:
-----------
Megatron's official _Llama2Tokenizer only supports local SentencePiece files,
while Primus extends it to support HuggingFace Hub IDs (e.g., meta-llama/Llama-2-7b-hf).

Without this patch, the new architecture (PrimusRuntime) would call Megatron's
official build_tokenizer, causing failures when using custom tokenizer types
with Hub IDs.

This patch ensures both legacy and new architectures use the same tokenizer
building logic.
"""

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.tokenizer.build_tokenizer_override",
    backend="megatron",
    phase="setup",
    description="Override Megatron's build_tokenizer to support Primus custom tokenizer types with HuggingFace Hub IDs",
)
def patch_build_tokenizer_override(ctx: PatchContext):
    """
    Monkey-patch Megatron's build_tokenizer with Primus version.

    This ensures that custom tokenizer types (Llama2Tokenizer, Llama3Tokenizer,
    DeepSeekV2Tokenizer, etc.) are properly handled:

    - All custom types use _HuggingFaceTokenizer internally
    - Support for HuggingFace Hub IDs (e.g., meta-llama/Llama-2-7b-hf)
    - Consistent behavior between legacy and new architectures

    Without this patch:
    -------------------
    - tokenizer_type: Llama2Tokenizer
      tokenizer_model: meta-llama/Llama-2-7b-hf
      → Calls Megatron's _Llama2Tokenizer
      → Expects local file path
      → ❌ FileNotFoundError

    With this patch:
    ----------------
    - tokenizer_type: Llama2Tokenizer
      tokenizer_model: meta-llama/Llama-2-7b-hf
      → Calls Primus build_tokenizer
      → Maps to _HuggingFaceTokenizer
      → Supports Hub ID
      → ✅ Success
    """
    try:
        import megatron.training.global_vars as megatron_global_vars
        import pretrain_gpt
    except ImportError as e:
        log_rank_0(
            f"[Patch:megatron.tokenizer.build_tokenizer_override] "
            f"Skip patch (Megatron not available): {e}"
        )
        return

    # Import Primus build_tokenizer
    from primus.backends.megatron.training.tokenizer.tokenizer import (
        build_tokenizer as primus_build_tokenizer,
    )

    # Save original for reference (optional)
    if not hasattr(megatron_global_vars, "_original_build_tokenizer"):
        megatron_global_vars._original_build_tokenizer = megatron_global_vars.build_tokenizer
    if not hasattr(pretrain_gpt, "_original_build_tokenizer"):
        pretrain_gpt._original_build_tokenizer = pretrain_gpt.build_tokenizer

    # Replace Megatron's build_tokenizer with Primus version
    megatron_global_vars.build_tokenizer = primus_build_tokenizer
    pretrain_gpt.build_tokenizer = primus_build_tokenizer

    log_rank_0(
        "[Patch:megatron.tokenizer.build_tokenizer_override] "
        "✓ Replaced Megatron build_tokenizer with Primus version"
    )
