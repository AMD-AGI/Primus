###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Extra Megatron tokenizers."""

import math
from collections import OrderedDict

from megatron.training.arguments import (
    _add_tokenizer_args as megatron_add_tokenizer_args,
)

# Handle both old (megatron.training.tokenizer) and new (megatron.core.tokenizers)
# Megatron-LM module structures.
try:
    from megatron.training.tokenizer import build_tokenizer as megatron_build_tokenizer
    from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer
except ImportError:
    from megatron.core.tokenizers.utils.build_tokenizer import (
        build_tokenizer as megatron_build_tokenizer,
    )
    from megatron.core.tokenizers.text.libraries.huggingface_tokenizer import (
        HuggingFaceTokenizer as _HuggingFaceTokenizer,
    )

from primus.modules.module_utils import log_rank_0, warning_rank_0

CUSTOM_TOKENIZER_TYPES = {
    "DeepSeekV2Tokenizer",
    "DeepSeekV3Tokenizer",
    "Llama2Tokenizer",
    "Llama3Tokenizer",
    "MixtralTokenizer",
    "Llama4Tokenizer",
    "KimiK2Tokenizer",
    "HuggingFaceTokenizer",
}


def _ensure_unique_identifiers(tokenizer, args):
    """Attach unique_identifiers for MCore dataset cache hashing compatibility."""
    if hasattr(tokenizer, "unique_identifiers"):
        return tokenizer

    unique_identifiers = OrderedDict()
    unique_identifiers["class"] = f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    unique_identifiers["tokenizer_path"] = getattr(args, "tokenizer_model", None)
    unique_identifiers["tokenizer_type"] = getattr(args, "tokenizer_type", None)

    for arg_name in (
        "vocab_file",
        "merge_file",
        "tokenizer_hf_no_use_fast",
        "tokenizer_hf_no_include_special_tokens",
        "trust_remote_code",
    ):
        if hasattr(args, arg_name):
            unique_identifiers[arg_name] = str(getattr(args, arg_name))

    tokenizer.unique_identifiers = unique_identifiers
    return tokenizer


def _build_custom_hf_tokenizer(args, **kwargs):
    """Build custom tokenizer aliases through HF-compatible Megatron path."""
    tokenizer = None
    original_tokenizer_type = args.tokenizer_type
    fallback_reason = None

    # Prefer the upstream builder for MCore compatibility (e.g. unique_identifiers).
    builder = None if megatron_build_tokenizer is build_tokenizer else megatron_build_tokenizer
    if builder is not None:
        args.tokenizer_type = "HuggingFaceTokenizer"
        try:
            tokenizer = builder(args, **kwargs)
        except Exception as e:
            tokenizer = None
            fallback_reason = (
                "upstream build_tokenizer failed with "
                f"{type(e).__name__}: {e}"
            )
        finally:
            args.tokenizer_type = original_tokenizer_type

    # Fallback to direct HF tokenizer construction for legacy/custom flows.
    if tokenizer is None:
        if fallback_reason is None:
            fallback_reason = "upstream build_tokenizer unavailable (recursive patch path)"
        warning_rank_0(
            "[TokenizerCompat] Falling back to direct HuggingFace tokenizer construction "
            f"for tokenizer_type={original_tokenizer_type}, tokenizer_model={args.tokenizer_model}. "
            f"Reason: {fallback_reason}"
        )
        tokenizer = _HuggingFaceTokenizer(args.tokenizer_model, trust_remote_code=True)

    return _ensure_unique_identifiers(tokenizer, args)


def _add_tokenizer_args(parser):
    parser = megatron_add_tokenizer_args(parser)
    tokenizer_arg = next(action for action in parser._actions if action.dest == "tokenizer_type")
    custom_choices = [t for t in CUSTOM_TOKENIZER_TYPES]
    tokenizer_arg.choices = list(set(tokenizer_arg.choices).union(custom_choices))
    return parser


def build_tokenizer(args, **kwargs):
    """Initialize tokenizer."""

    log_rank_0(f"-building {args.tokenizer_type} tokenizer...")

    # Select and instantiate the tokenizer.
    if args.tokenizer_type in CUSTOM_TOKENIZER_TYPES:
        tokenizer = _build_custom_hf_tokenizer(args, **kwargs)
    else:
        return megatron_build_tokenizer(args, **kwargs)

    # Add vocab size (if not already set from a checkpoint).
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args, logging_enabled=True):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    after = int(math.ceil(after / multiple) * multiple)
    if args.rank == 0 and logging_enabled:
        print(
            " > padded vocab (size: {}) with {} dummy tokens "
            "(new size: {})".format(orig_vocab_size, after - orig_vocab_size, after),
            flush=True,
        )
    return after
